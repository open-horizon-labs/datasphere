//! All-projects watcher - monitors all Claude projects for session changes.
//!
//! AIDEV-NOTE: Watches ~/.claude/projects/ recursively for *.jsonl changes.
//! Emits events with project_id parsed from path. Used by daemon mode.

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};

use crate::session::{claude_projects_dir, SessionInfo};

/// Event from all-projects watcher with project context
#[derive(Debug, Clone)]
pub struct ProjectSessionEvent {
    /// Project ID (directory name under ~/.claude/projects/)
    pub project_id: String,
    /// Session info
    pub session: SessionInfo,
    /// Whether this is a new session (vs modified)
    pub is_new: bool,
}

/// Error type for all-projects watcher
#[derive(Debug)]
pub enum AllProjectsWatcherError {
    /// Could not determine Claude projects directory
    NoProjectsDir,
    /// Failed to create file watcher
    NotifyError(notify::Error),
}

impl std::fmt::Display for AllProjectsWatcherError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AllProjectsWatcherError::NoProjectsDir => {
                write!(f, "Could not determine Claude projects directory")
            }
            AllProjectsWatcherError::NotifyError(e) => write!(f, "File watcher error: {}", e),
        }
    }
}

impl std::error::Error for AllProjectsWatcherError {}

impl From<notify::Error> for AllProjectsWatcherError {
    fn from(e: notify::Error) -> Self {
        AllProjectsWatcherError::NotifyError(e)
    }
}

/// Watches all Claude projects for session changes
pub struct AllProjectsWatcher {
    _watcher: RecommendedWatcher,
    projects_dir: PathBuf,
    event_rx: Receiver<ProjectSessionEvent>,
}

impl AllProjectsWatcher {
    /// Create a new watcher for all Claude projects.
    ///
    /// Does NOT emit initial events for existing sessions - the caller should scan
    /// existing sessions separately (e.g., using `list_all_projects()` + `discover_sessions_in_dir()`).
    pub fn new() -> Result<Self, AllProjectsWatcherError> {
        let projects_dir =
            claude_projects_dir().ok_or(AllProjectsWatcherError::NoProjectsDir)?;

        if !projects_dir.exists() {
            std::fs::create_dir_all(&projects_dir).ok();
        }

        let (event_tx, event_rx) = mpsc::channel();

        // Set up file watcher (recursive to catch all project subdirectories)
        let watcher = Self::create_watcher(projects_dir.clone(), event_tx)?;

        Ok(Self {
            _watcher: watcher,
            projects_dir,
            event_rx,
        })
    }

    /// Create the file watcher
    fn create_watcher(
        projects_dir: PathBuf,
        event_tx: Sender<ProjectSessionEvent>,
    ) -> Result<RecommendedWatcher, AllProjectsWatcherError> {
        let dir_for_closure = projects_dir.clone();

        let mut watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
            if let Ok(event) = res {
                Self::handle_notify_event(&dir_for_closure, &event, &event_tx);
            }
        })?;

        watcher.watch(&projects_dir, RecursiveMode::Recursive)?;

        Ok(watcher)
    }

    /// Handle a notify event, converting to ProjectSessionEvent if applicable
    fn handle_notify_event(
        projects_dir: &Path,
        event: &Event,
        event_tx: &Sender<ProjectSessionEvent>,
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

            // Parse project_id and session info from path
            if let Some((project_id, session_info)) =
                Self::parse_path(path, projects_dir)
            {
                // Skip agent sessions
                if session_info.session_id.starts_with("agent-") {
                    continue;
                }

                let session_event = ProjectSessionEvent {
                    project_id,
                    session: session_info,
                    is_new: is_create,
                };

                // Ignore send errors - receiver may have been dropped
                let _ = event_tx.send(session_event);
            }
        }
    }

    /// Parse path to extract project_id and SessionInfo
    /// Path format: ~/.claude/projects/<project-id>/<session-id>.jsonl
    fn parse_path(path: &Path, projects_dir: &Path) -> Option<(String, SessionInfo)> {
        // Get relative path from projects_dir
        let rel_path = path.strip_prefix(projects_dir).ok()?;

        // First component is project_id
        let mut components = rel_path.components();
        let project_id = components.next()?.as_os_str().to_str()?.to_string();

        // Session ID is filename without extension
        let session_id = path.file_stem()?.to_str()?.to_string();

        // Get metadata
        let metadata = std::fs::metadata(path).ok()?;
        let modified = metadata.modified().ok()?;
        let size_bytes = metadata.len();

        // Convert SystemTime to DateTime<Utc>
        let duration = modified.duration_since(std::time::UNIX_EPOCH).ok()?;
        let modified_at =
            chrono::DateTime::from_timestamp(duration.as_secs() as i64, duration.subsec_nanos())?;

        let session_info = SessionInfo {
            session_id,
            transcript_path: path.to_path_buf(),
            modified_at,
            size_bytes,
        };

        Some((project_id, session_info))
    }

    /// Get the projects directory being watched
    pub fn projects_dir(&self) -> &Path {
        &self.projects_dir
    }

    /// Get the next event (blocking)
    pub fn recv(&self) -> Option<ProjectSessionEvent> {
        self.event_rx.recv().ok()
    }

    /// Try to get an event without blocking
    pub fn try_recv(&self) -> Option<ProjectSessionEvent> {
        self.event_rx.try_recv().ok()
    }

    /// Get an iterator over events
    pub fn iter(&self) -> impl Iterator<Item = ProjectSessionEvent> + '_ {
        std::iter::from_fn(|| self.recv())
    }
}
