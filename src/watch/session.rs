//! Session watcher - monitors Claude session transcripts for changes.
//!
//! Uses the notify crate to watch `~/.claude/projects/<project-id>/` for
//! new or modified `.jsonl` transcript files.

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};

use crate::session::{discover_sessions_in_dir, get_project_dir, SessionInfo};

/// Events emitted by the session watcher
#[derive(Debug, Clone)]
pub enum SessionEvent {
    /// A new session transcript was created
    Created(SessionInfo),
    /// An existing session transcript was modified
    Modified(SessionInfo),
}

/// Error type for session watcher
#[derive(Debug)]
pub enum WatcherError {
    /// No Claude project directory found for the given path
    NoProjectDir(PathBuf),
    /// Failed to create file watcher
    NotifyError(notify::Error),
    /// Failed to discover sessions
    DiscoveryError(String),
}

impl std::fmt::Display for WatcherError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WatcherError::NoProjectDir(p) => {
                write!(f, "No Claude project directory found for {:?}", p)
            }
            WatcherError::NotifyError(e) => write!(f, "File watcher error: {}", e),
            WatcherError::DiscoveryError(e) => write!(f, "Session discovery error: {}", e),
        }
    }
}

impl std::error::Error for WatcherError {}

impl From<notify::Error> for WatcherError {
    fn from(e: notify::Error) -> Self {
        WatcherError::NotifyError(e)
    }
}

/// Watches Claude session transcripts for changes
pub struct SessionWatcher {
    _watcher: RecommendedWatcher,
    project_dir: PathBuf,
    event_rx: Receiver<SessionEvent>,
}

impl SessionWatcher {
    /// Create a new session watcher for the given project path.
    ///
    /// On creation, scans existing sessions and emits Created events for all of them,
    /// then watches for new changes.
    pub fn new(project_path: &Path) -> Result<Self, WatcherError> {
        let project_dir = get_project_dir(project_path)
            .ok_or_else(|| WatcherError::NoProjectDir(project_path.to_path_buf()))?;

        let (event_tx, event_rx) = mpsc::channel();

        // Emit events for all existing sessions first
        Self::emit_existing_sessions(&project_dir, &event_tx)?;

        // Set up file watcher
        let watcher = Self::create_watcher(project_dir.clone(), event_tx)?;

        Ok(Self {
            _watcher: watcher,
            project_dir,
            event_rx,
        })
    }

    /// Emit Created events for all existing sessions
    fn emit_existing_sessions(
        project_dir: &Path,
        event_tx: &Sender<SessionEvent>,
    ) -> Result<(), WatcherError> {
        let sessions =
            discover_sessions_in_dir(project_dir).map_err(WatcherError::DiscoveryError)?;

        for session in sessions {
            // Ignore send errors - receiver may have been dropped
            let _ = event_tx.send(SessionEvent::Created(session));
        }

        Ok(())
    }

    /// Create the file watcher
    fn create_watcher(
        project_dir: PathBuf,
        event_tx: Sender<SessionEvent>,
    ) -> Result<RecommendedWatcher, WatcherError> {
        let dir_for_closure = project_dir.clone();

        let mut watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
            if let Ok(event) = res {
                Self::handle_notify_event(&dir_for_closure, &event, &event_tx);
            }
        })?;

        watcher.watch(&project_dir, RecursiveMode::NonRecursive)?;

        Ok(watcher)
    }

    /// Handle a notify event, converting to SessionEvent if applicable
    fn handle_notify_event(
        project_dir: &Path,
        event: &Event,
        event_tx: &Sender<SessionEvent>,
    ) {
        // Only process Create and Modify events
        let is_create = matches!(event.kind, EventKind::Create(_));
        let is_modify = matches!(event.kind, EventKind::Modify(_));

        if !is_create && !is_modify {
            return;
        }

        // Process each path in the event
        for path in &event.paths {
            // Filter for .jsonl files only
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }

            // Build SessionInfo from the file
            if let Some(session_info) = Self::path_to_session_info(path, project_dir) {
                let session_event = if is_create {
                    SessionEvent::Created(session_info)
                } else {
                    SessionEvent::Modified(session_info)
                };

                // Ignore send errors - receiver may have been dropped
                let _ = event_tx.send(session_event);
            }
        }
    }

    /// Convert a file path to SessionInfo
    fn path_to_session_info(path: &Path, _project_dir: &Path) -> Option<SessionInfo> {
        let session_id = path.file_stem()?.to_str()?.to_string();
        let metadata = std::fs::metadata(path).ok()?;
        let modified = metadata.modified().ok()?;
        let size_bytes = metadata.len();

        // Convert SystemTime to DateTime<Utc>
        let duration = modified.duration_since(std::time::UNIX_EPOCH).ok()?;
        let modified_at =
            chrono::DateTime::from_timestamp(duration.as_secs() as i64, duration.subsec_nanos())?;

        Some(SessionInfo {
            session_id,
            transcript_path: path.to_path_buf(),
            modified_at,
            size_bytes,
        })
    }

    /// Get the project directory being watched
    pub fn project_dir(&self) -> &Path {
        &self.project_dir
    }

    /// Get the next session event (blocking)
    pub fn recv(&self) -> Option<SessionEvent> {
        self.event_rx.recv().ok()
    }

    /// Try to get a session event without blocking
    pub fn try_recv(&self) -> Option<SessionEvent> {
        self.event_rx.try_recv().ok()
    }

    /// Get an iterator over session events
    pub fn iter(&self) -> impl Iterator<Item = SessionEvent> + '_ {
        std::iter::from_fn(|| self.recv())
    }
}
