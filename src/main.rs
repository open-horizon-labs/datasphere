use chrono::{Local, Utc};
use clap::{Parser, Subcommand};
use datasphere::{
    chunk_text, chunk_transcript, discover_sessions, discover_sessions_in_dir, distill_chunk,
    embed, list_all_projects, read_transcript, resolve_anthropic_model, AllProjectsWatcher,
    BatchQueue, EmbedError, Job, JobStatus, LlmError, Node, Processed, Queue, SessionInfo,
    SourceType, Store,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::Mutex;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

#[derive(Parser)]
#[command(name = "ds")]
#[command(about = "Datasphere - distills knowledge from Claude Code sessions")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show database statistics
    Stats {
        /// Custom database path (defaults to ~/.datasphere/db)
        #[arg(long)]
        db: Option<PathBuf>,
    },

    /// Show stored nodes
    Show {
        /// Maximum number of nodes to show
        #[arg(short, long, default_value = "10")]
        limit: usize,

        /// Custom database path (defaults to ~/.datasphere/db)
        #[arg(long)]
        db: Option<PathBuf>,
    },

    /// Scan and distill transcripts (one-shot)
    Scan {
        /// Maximum number of transcripts to process (newest first)
        #[arg(short, long)]
        limit: Option<usize>,

        /// Project path to scan (defaults to current directory)
        #[arg(short, long)]
        project: Option<PathBuf>,
    },

    /// Start the daemon (watches all projects)
    Start {
        /// Run in foreground instead of daemonizing
        #[arg(short, long)]
        foreground: bool,
    },

    /// Stop the running daemon
    Stop,

    /// Show daemon status
    Status,

    /// Show or manage the job queue
    Queue {
        #[command(subcommand)]
        action: Option<QueueAction>,
    },

    /// Add a text file to the knowledge graph (no LLM distillation)
    Add {
        /// Path to the file to add
        file: PathBuf,
    },

    /// Search the knowledge graph for relevant nodes
    Query {
        /// Search query text
        query: String,

        /// Maximum number of results
        #[arg(short, long, default_value = "5")]
        limit: usize,

        /// Output format (text or json)
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Filter by namespace (omit to search all namespaces)
        #[arg(short, long)]
        namespace: Option<String>,

        /// Custom database path (defaults to ~/.datasphere/db)
        #[arg(long)]
        db: Option<PathBuf>,
    },

    /// Delete everything (database + queue) and start fresh
    Reset,

    /// Add a note directly to the knowledge graph
    Note {
        /// The note content
        content: String,

        /// Source tag (default: "note")
        #[arg(short, long, default_value = "note")]
        source: String,

        /// Namespace for the note (default: "personal")
        #[arg(short, long, default_value = "personal")]
        namespace: String,
    },

    /// Find nodes similar to a given node
    Related {
        /// Node ID (UUID) to find related nodes for
        node_id: String,

        /// Maximum number of results
        #[arg(short, long, default_value = "5")]
        limit: usize,

        /// Output format (text or json)
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Custom database path (defaults to ~/.datasphere/db)
        #[arg(long)]
        db: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum QueueAction {
    /// Show queue counts (default)
    Status,
    /// List pending jobs
    Pending,
    /// List failed jobs
    Failed,
    /// Clear completed jobs
    Clear,
    /// Delete entire queue (all jobs, all statuses)
    Nuke,
    /// Retry failed jobs (requeue for processing)
    Retry {
        /// Specific job source_id to retry (retries all if omitted)
        source_id: Option<String>,
    },
}

/// Get the default database path
fn default_db_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".datasphere")
        .join("db")
}

/// Resolve database path from optional override or default
fn resolve_db_path(db_override: Option<PathBuf>) -> PathBuf {
    db_override.unwrap_or_else(default_db_path)
}

/// Get the daemon PID file path
fn daemon_pid_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".datasphere")
        .join("daemon.pid")
}

/// Get the daemon log file path
fn daemon_log_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".datasphere")
        .join("daemon.log")
}

/// Load blacklisted project IDs from ~/.datasphere/blacklist
/// Returns empty set if file doesn't exist.
/// File format: one project ID per line, # for comments.
fn load_blacklist() -> HashSet<String> {
    let path = dirs::home_dir()
        .map(|h| h.join(".datasphere").join("blacklist"))
        .unwrap_or_default();

    std::fs::read_to_string(&path)
        .map(|content| {
            content
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(|l| l.to_string())
                .collect()
        })
        .unwrap_or_default()
}

/// Number of old log files to keep
const LOG_ROTATION_KEEP: usize = 7;

/// Rotate log files: daemon.log -> daemon.log.1 -> daemon.log.2 -> ...
fn rotate_logs() {
    let base = daemon_log_path();

    // Delete oldest log if it exists
    let oldest = base.with_extension(format!("log.{}", LOG_ROTATION_KEEP));
    let _ = std::fs::remove_file(&oldest);

    // Shift existing logs: .log.N -> .log.N+1
    for i in (1..LOG_ROTATION_KEEP).rev() {
        let from = base.with_extension(format!("log.{}", i));
        let to = base.with_extension(format!("log.{}", i + 1));
        let _ = std::fs::rename(&from, &to);
    }

    // Rename current log to .log.1
    let _ = std::fs::rename(&base, base.with_extension("log.1"));
}

/// Daemon logger with daily rotation
struct DaemonLog {
    file: std::fs::File,
    current_date: chrono::NaiveDate,
}

impl DaemonLog {
    fn new() -> std::io::Result<Self> {
        let log_path = daemon_log_path();

        // Ensure directory exists
        if let Some(parent) = log_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Open log file in append mode
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;

        Ok(Self {
            file,
            current_date: Local::now().date_naive(),
        })
    }

    fn log(&mut self, msg: &str) {
        use std::io::Write;

        // Check if we need to rotate (new day)
        let today = Local::now().date_naive();
        if today != self.current_date {
            // Flush current file
            let _ = self.file.flush();

            // Rotate logs
            rotate_logs();

            // Open new log file
            match std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(daemon_log_path())
            {
                Ok(new_file) => {
                    self.file = new_file;
                    let _ = writeln!(self.file, "[{}] [ROTATE] New log file for {}", Local::now().format("%H:%M:%S"), today);
                }
                Err(_) => {
                    // Continue writing to old (now renamed) file handle
                }
            }
            // Update date regardless to prevent retry loop
            self.current_date = today;
        }

        // Add timestamp to all log entries
        let _ = writeln!(self.file, "[{}] {}", Local::now().format("%H:%M:%S"), msg);
        let _ = self.file.flush();
    }
}

/// Global logger for daemon mode (None = use stdout)
static DAEMON_LOGGER: std::sync::Mutex<Option<DaemonLog>> = std::sync::Mutex::new(None);

/// Log a message - writes to daemon log file if in daemon mode, otherwise stdout
macro_rules! dlog {
    ($($arg:tt)*) => {{
        let msg = format!($($arg)*);
        if let Ok(mut guard) = DAEMON_LOGGER.lock() {
            if let Some(ref mut logger) = *guard {
                logger.log(&msg);
            } else {
                println!("{}", msg);
            }
        } else {
            println!("{}", msg);
        }
    }};
}

/// Check if daemon is running, returns PID if running
fn is_daemon_running() -> Option<u32> {
    let pid_path = daemon_pid_path();
    if !pid_path.exists() {
        return None;
    }

    let pid_str = std::fs::read_to_string(&pid_path).ok()?;
    let pid: u32 = pid_str.trim().parse().ok()?;

    // Check if process is actually running
    #[cfg(unix)]
    {
        // kill(pid, 0) checks if process exists without sending a signal
        let result = unsafe { libc::kill(pid as i32, 0) };
        if result == 0 {
            return Some(pid);
        }
    }

    // PID file exists but process is dead - clean up
    let _ = std::fs::remove_file(&pid_path);
    None
}

/// Format bytes as human-readable size
fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

/// Calculate total size of a directory recursively
fn dir_size(path: &PathBuf) -> u64 {
    if !path.exists() {
        return 0;
    }

    walkdir::WalkDir::new(path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| e.metadata().ok())
        .map(|m| m.len())
        .sum()
}

/// Hamming distance threshold for considering a session "changed"
/// AIDEV-NOTE: 10 bits out of 64 (~15%) means meaningful content change
const SIMHASH_CHANGE_THRESHOLD: u32 = 10;

/// Error type for session processing
#[derive(Debug)]
enum ProcessError {
    /// Rate limit hit - caller should back off
    RateLimit(String),
    /// Request was cancelled (e.g., daemon shutdown)
    Cancelled,
    /// Other errors
    Other(String),
}

impl std::fmt::Display for ProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessError::RateLimit(msg) => write!(f, "Rate limit: {}", msg),
            ProcessError::Cancelled => write!(f, "Request cancelled"),
            ProcessError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for ProcessError {}

impl From<LlmError> for ProcessError {
    fn from(e: LlmError) -> Self {
        match e {
            LlmError::RateLimit(msg) => ProcessError::RateLimit(msg),
            LlmError::Cancelled => ProcessError::Cancelled,
            LlmError::Other(msg) => ProcessError::Other(msg),
        }
    }
}

impl From<String> for ProcessError {
    fn from(e: String) -> Self {
        ProcessError::Other(e)
    }
}

impl From<lancedb::Error> for ProcessError {
    fn from(e: lancedb::Error) -> Self {
        ProcessError::Other(e.to_string())
    }
}

impl From<EmbedError> for ProcessError {
    fn from(e: EmbedError) -> Self {
        match e {
            EmbedError::Cancelled => ProcessError::Cancelled,
            _ => ProcessError::Other(e.to_string()),
        }
    }
}

impl From<datasphere::transcript::TranscriptError> for ProcessError {
    fn from(e: datasphere::transcript::TranscriptError) -> Self {
        ProcessError::Other(e.to_string())
    }
}

/// Quick check if a transcript has any meaningful content (messages/summaries)
/// Returns false for empty or content-free transcripts to avoid queueing them
fn transcript_has_content(path: &std::path::Path) -> bool {
    match read_transcript(path) {
        Ok(entries) => entries.iter().any(|e| e.is_message() || e.is_summary()),
        Err(_) => false,
    }
}

/// Result type for process_session
enum ProcessResult {
    /// Session was processed immediately, created N nodes
    Processed(usize),
    /// Session was skipped (empty or unchanged)
    Skipped,
    /// Session was queued for batch processing
    QueuedForBatch,
}

/// Process a single session transcript with incremental chunk processing
/// AIDEV-NOTE: Compares per-chunk simhashes to only re-distill changed chunks.
/// Returns ProcessResult indicating what happened
async fn process_session(
    store: &Store,
    session: &SessionInfo,
    cancel_token: &CancellationToken,
    _batch_queue: Option<Arc<Mutex<BatchQueue>>>,
) -> Result<ProcessResult, ProcessError> {
    dlog!("  Reading transcript...");

    // Parse transcript
    let entries = read_transcript(&session.transcript_path)?;
    dlog!("  Parsed {} entries", entries.len());

    if entries.is_empty() {
        dlog!("  Empty transcript, skipping");
        return Ok(ProcessResult::Skipped);
    }

    // Get all messages for context
    let messages: Vec<_> = entries
        .iter()
        .filter(|e| e.is_message() || e.is_summary())
        .collect();

    if messages.is_empty() {
        dlog!("  No messages found, skipping");
        return Ok(ProcessResult::Skipped);
    }

    // Count message types
    let user_count = messages.iter().filter(|e| e.is_user()).count();
    let assistant_count = messages.iter().filter(|e| e.is_assistant()).count();
    let summary_count = messages.iter().filter(|e| e.is_summary()).count();

    dlog!(
        "  Collected {} items ({} user, {} assistant, {} summaries)",
        messages.len(),
        user_count,
        assistant_count,
        summary_count
    );

    // Format context for LLM
    let context = datasphere::format_context(&messages);
    if context.trim().is_empty() {
        dlog!("  Empty context after formatting, skipping");
        return Ok(ProcessResult::Skipped);
    }

    dlog!("  Context size: {} chars", context.len());

    // Compute overall SimHash of context for quick change detection
    let current_simhash = simhash::simhash(&context) as i64;

    // Check if session was previously processed
    let existing_processed = store.get_processed(&session.session_id).await?;

    if let Some(ref existing) = existing_processed {
        let hamming = simhash::hamming_distance(existing.simhash as u64, current_simhash as u64);

        if hamming <= SIMHASH_CHANGE_THRESHOLD {
            dlog!(
                "  Unchanged (simhash distance: {} bits, threshold: {})",
                hamming, SIMHASH_CHANGE_THRESHOLD
            );
            return Ok(ProcessResult::Skipped);
        }

        dlog!(
            "  Session changed (simhash distance: {} bits), checking chunks...",
            hamming
        );
    }

    // Chunk the transcript and compute per-chunk simhashes
    let chunks = chunk_transcript(&context);
    let chunk_simhashes: Vec<i64> = chunks
        .iter()
        .map(|chunk| simhash::simhash(chunk) as i64)
        .collect();

    dlog!("  Split into {} chunk(s)", chunks.len());

    // Get existing nodes for this session (for incremental comparison)
    let existing_nodes = store.get_nodes_by_source(&session.session_id).await?;

    // Build map of existing chunk simhashes -> node IDs
    // AIDEV-NOTE: Chunks are matched by simhash, not by index (boundaries may shift)
    let mut existing_by_simhash: HashMap<i64, Vec<Uuid>> = HashMap::new();
    for node in &existing_nodes {
        if let Some(chunk_simhash) = node.chunk_simhash {
            existing_by_simhash
                .entry(chunk_simhash)
                .or_default()
                .push(node.id);
        }
    }

    // Determine which chunks need distillation (new or changed)
    let mut chunks_to_distill: Vec<(usize, &str, i64)> = Vec::new();
    let mut kept_node_ids: HashSet<Uuid> = HashSet::new();
    let mut new_chunk_simhashes: HashSet<i64> = HashSet::new();

    for (i, (chunk, &chunk_simhash)) in chunks.iter().zip(chunk_simhashes.iter()).enumerate() {
        new_chunk_simhashes.insert(chunk_simhash);

        if let Some(node_ids) = existing_by_simhash.get(&chunk_simhash) {
            // Chunk unchanged - keep existing node(s)
            // AIDEV-NOTE: Take first matching node (in case of duplicates)
            if let Some(&node_id) = node_ids.first() {
                kept_node_ids.insert(node_id);
                dlog!("    Chunk {} unchanged (simhash {:016x})", i + 1, chunk_simhash as u64);
            }
        } else {
            // New or changed chunk - needs distillation
            chunks_to_distill.push((i, chunk, chunk_simhash));
        }
    }

    // Delete orphaned nodes (chunks that no longer exist)
    let mut deleted_count = 0;
    for node in &existing_nodes {
        if !kept_node_ids.contains(&node.id) {
            store.delete_node(node.id).await?;
            deleted_count += 1;
        }
    }
    if deleted_count > 0 {
        dlog!("  Deleted {} orphaned node(s)", deleted_count);
    }

    // Delete old processed record if it exists (will re-create with updated info)
    if existing_processed.is_some() {
        store.delete_processed(&session.session_id).await?;
    }

    if chunks_to_distill.is_empty() {
        dlog!("  All chunks unchanged, keeping {} existing node(s)", kept_node_ids.len());

        // Re-record as processed with current simhash
        let node_ids: Vec<String> = kept_node_ids.iter().map(|id| id.to_string()).collect();
        let record = Processed {
            source_id: session.session_id.clone(),
            source_type: "session".to_string(),
            simhash: current_simhash,
            processed_at: Utc::now(),
            node_count: node_ids.len() as i32,
            node_ids,
        };
        store.insert_processed(&record).await?;

        return Ok(ProcessResult::Processed(0)); // 0 new nodes created
    }

    dlog!(
        "  Distilling {} changed chunk(s), keeping {} unchanged...",
        chunks_to_distill.len(),
        kept_node_ids.len()
    );

    // Distill changed chunks
    let distill_start = Instant::now();
    let mut new_node_ids: Vec<String> = Vec::new();
    let total_to_distill = chunks_to_distill.len();

    for (processed, (chunk_idx, chunk, chunk_simhash)) in chunks_to_distill.into_iter().enumerate() {
        // Distill the chunk
        match distill_chunk(chunk, cancel_token).await {
            Ok(Some(insight)) => {
                // Embed and store
                let embed_start = Instant::now();
                let embedding = embed(&insight.content, cancel_token).await?;
                dlog!(
                    "    Chunk {} distilled + embedded in {:.1}s ({}/{})",
                    chunk_idx + 1,
                    embed_start.elapsed().as_secs_f32(),
                    processed + 1,
                    total_to_distill
                );

                let node = Node::with_chunk(
                    insight.content,
                    session.session_id.clone(),
                    SourceType::Session,
                    embedding,
                    insight.confidence,
                    "personal".to_string(),
                    chunk_simhash,
                    chunk_idx as i32,
                );
                let node_id = node.id.to_string();
                store.insert_node(&node).await?;
                new_node_ids.push(node_id);
            }
            Ok(None) => {
                dlog!("    Chunk {} distillation too short, skipping", chunk_idx + 1);
            }
            Err(LlmError::RateLimit(msg)) => {
                return Err(ProcessError::RateLimit(msg));
            }
            Err(LlmError::Cancelled) => {
                return Err(ProcessError::Cancelled);
            }
            Err(e) => {
                dlog!("    Chunk {} distillation failed: {}", chunk_idx + 1, e);
                // Continue with other chunks
            }
        }
    }

    dlog!("  Distillation completed in {:.1}s", distill_start.elapsed().as_secs_f32());

    // Combine kept and new node IDs
    let all_node_ids: Vec<String> = kept_node_ids
        .iter()
        .map(|id| id.to_string())
        .chain(new_node_ids.iter().cloned())
        .collect();

    // Record as processed
    let record = Processed {
        source_id: session.session_id.clone(),
        source_type: "session".to_string(),
        simhash: current_simhash,
        processed_at: Utc::now(),
        node_count: all_node_ids.len() as i32,
        node_ids: all_node_ids,
    };
    store.insert_processed(&record).await?;

    dlog!(
        "  Done! {} new node(s), {} kept, {} total",
        new_node_ids.len(),
        kept_node_ids.len(),
        record.node_count
    );
    Ok(ProcessResult::Processed(new_node_ids.len()))
}

/// Process completed batch results
/// AIDEV-NOTE: Batch results contain distilled text that needs to be embedded and stored
/// For chunked sessions, results are aggregated into multiple nodes per session
async fn process_batch_results(
    store: &Store,
    batch_queue: &Arc<Mutex<BatchQueue>>,
    cancel_token: &CancellationToken,
) -> Result<usize, Box<dyn std::error::Error>> {
    use std::collections::HashMap;

    // Poll for completed batches
    // AIDEV-NOTE: Using tokio::sync::Mutex to safely hold lock across await
    let completed = {
        let mut bq = batch_queue.lock().await;
        bq.poll_pending().await.map_err(|e| format!("Poll error: {}", e))?
    };

    if completed.is_empty() {
        return Ok(0);
    }

    let mut total_nodes = 0;

    for (batch, results) in completed {
        dlog!("[BATCH] Processing completed batch {} ({} results)", &batch.batch_id[..8.min(batch.batch_id.len())], results.len());

        // Get request metadata for this batch
        let request_meta = {
            let bq = batch_queue.lock().await;
            bq.get_request_meta(&batch)
        };

        // Group results by session_id for chunk aggregation
        // AIDEV-NOTE: Multiple chunks from same session become multiple nodes, one Processed record
        let mut session_results: HashMap<String, Vec<(String, i64, Option<usize>)>> = HashMap::new();

        for result in results {
            // Get metadata for this result
            let meta = request_meta.get(&result.custom_id);
            let (session_id, simhash, chunk_index) = match meta {
                Some(m) => (m.session_id.clone(), m.simhash, m.chunk_index),
                None => {
                    // Fallback: parse custom_id
                    let sid = result.custom_id.split(':').next().unwrap_or(&result.custom_id).to_string();
                    (sid, 0, None)
                }
            };

            match result.result_type.as_str() {
                "succeeded" => {
                    if let Some(text) = result.text {
                        let trimmed = text.trim();
                        if trimmed.len() >= 50 {
                            // Embed the distilled content
                            let embedding = embed(trimmed, cancel_token).await?;

                            // Create and store node with chunk metadata
                            let node = Node::with_chunk(
                                trimmed.to_string(),
                                session_id.clone(),
                                SourceType::Session,
                                embedding,
                                1.0,
                                "personal".to_string(),
                                simhash,
                                chunk_index.unwrap_or(0) as i32,
                            );
                            let node_id = node.id.to_string();
                            store.insert_node(&node).await?;

                            // Track this node for the session
                            session_results
                                .entry(session_id.clone())
                                .or_default()
                                .push((node_id, simhash, chunk_index));

                            let chunk_info = chunk_index.map(|i| format!(" chunk {}", i)).unwrap_or_default();
                            dlog!("    {}{} -> created node", &session_id[..8.min(session_id.len())], chunk_info);
                            total_nodes += 1;
                        } else {
                            dlog!("    {} -> distillation too short ({} chars)", &session_id[..8.min(session_id.len())], trimmed.len());
                        }
                    }
                }
                "errored" => {
                    let err_msg = result.error.as_deref().unwrap_or("unknown error");
                    dlog!("    {} -> error: {}", &session_id[..8.min(session_id.len())], err_msg);
                }
                other => {
                    dlog!("    {} -> {}", &session_id[..8.min(session_id.len())], other);
                }
            }

            // Log cost if available
            if let Some(usage) = result.usage {
                dlog!("    [Cost] {} in / {} out", usage.input_tokens, usage.output_tokens);
            }
        }

        // Create Processed records for each session (aggregating all chunks)
        for (session_id, nodes_info) in session_results {
            if nodes_info.is_empty() {
                continue;
            }

            // Use simhash from first chunk (all chunks have same session simhash)
            let simhash = nodes_info[0].1;
            let node_ids: Vec<String> = nodes_info.into_iter().map(|(id, _, _)| id).collect();
            let node_count = node_ids.len() as i32;

            let record = Processed {
                source_id: session_id.clone(),
                source_type: "session".to_string(),
                simhash,
                processed_at: Utc::now(),
                node_count,
                node_ids,
            };
            store.insert_processed(&record).await?;
            dlog!("    {} -> recorded {} node(s)", &session_id[..8.min(session_id.len())], node_count);
        }
    }

    Ok(total_nodes)
}

/// Process a single text file (no LLM distillation, direct embedding)
/// Returns number of nodes created
async fn process_file(
    store: &Store,
    file_path: &PathBuf,
    cancel_token: &CancellationToken,
) -> Result<usize, Box<dyn std::error::Error>> {
    // Canonicalize path to avoid duplicates
    let canonical_path = file_path.canonicalize()
        .map_err(|e| format!("Failed to canonicalize path: {}", e))?;
    let source_id = canonical_path.to_string_lossy().to_string();

    dlog!("Processing file: {}", canonical_path.display());

    // Read file content
    let content = std::fs::read_to_string(&canonical_path)
        .map_err(|e| format!("Failed to read file: {}", e))?;

    if content.trim().is_empty() {
        dlog!("  Empty file, skipping");
        return Ok(0);
    }

    dlog!("  File size: {} chars", content.len());

    // Compute SimHash
    let current_simhash = simhash::simhash(&content) as i64;

    // Check if already processed
    if let Some(existing) = store.get_processed(&source_id).await? {
        let hamming = simhash::hamming_distance(existing.simhash as u64, current_simhash as u64);

        if hamming <= SIMHASH_CHANGE_THRESHOLD {
            dlog!(
                "  Unchanged (simhash distance: {} bits, threshold: {})",
                hamming, SIMHASH_CHANGE_THRESHOLD
            );
            return Ok(0);
        }

        dlog!(
            "  File changed (simhash distance: {} bits), re-embedding...",
            hamming
        );

        // Delete old nodes, then processed record
        let node_ids: Vec<Uuid> = existing.node_ids
            .iter()
            .filter_map(|id| id.parse::<Uuid>().ok())
            .collect();
        if !node_ids.is_empty() {
            store.delete_nodes(&node_ids).await?;
            dlog!("  Deleted {} old node(s)", node_ids.len());
        }
        store.delete_processed(&source_id).await?;
    }

    // Chunk content if needed
    let chunks = chunk_text(&content);
    dlog!("  Chunks: {}", chunks.len());

    // Embed each chunk and create nodes
    let mut node_ids = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let embed_start = Instant::now();
        let embedding = embed(chunk, cancel_token).await?;
        dlog!("  Embedded {}/{} in {:.1}s", i + 1, chunks.len(), embed_start.elapsed().as_secs_f32());

        let node = Node::new(
            chunk.clone(),
            source_id.clone(),
            SourceType::File,
            embedding,
            1.0, // Full confidence for raw file content
        );
        let node_id = node.id.to_string();
        store.insert_node(&node).await?;

        node_ids.push(node_id);
    }

    // Record as processed
    let record = Processed {
        source_id: source_id.clone(),
        source_type: "file".to_string(),
        simhash: current_simhash,
        processed_at: Utc::now(),
        node_count: node_ids.len() as i32,
        node_ids,
    };
    store.insert_processed(&record).await?;

    dlog!("  Done! Created {} node(s)", record.node_count);
    Ok(record.node_count as usize)
}

/// Process a note (direct text input, no LLM distillation)
/// Returns the node IDs created
async fn process_note(
    store: &Store,
    content: &str,
    source_tag: &str,
    namespace: &str,
    cancel_token: &CancellationToken,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let timestamp = Utc::now();
    let source_id = format!("{}:{}", source_tag, timestamp.to_rfc3339());

    println!("Adding note...");
    println!("  Source: {}", source_id);
    println!("  Namespace: {}", namespace);
    println!("  Content: {} chars", content.len());

    if content.trim().is_empty() {
        return Err("Note content is empty".into());
    }

    // Chunk content if needed
    let chunks = chunk_text(content);
    println!("  Chunks: {}", chunks.len());

    // Embed each chunk and create nodes
    let mut node_ids = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let embed_start = Instant::now();
        let embedding = embed(chunk, cancel_token).await?;
        println!(
            "  Embedded {}/{} in {:.1}s",
            i + 1,
            chunks.len(),
            embed_start.elapsed().as_secs_f32()
        );

        let node = Node::with_namespace(
            chunk.clone(),
            source_id.clone(),
            SourceType::File, // Notes use File type (no session association)
            embedding,
            1.0, // Full confidence for direct input
            namespace.to_string(),
        );
        let node_id = node.id.to_string();
        store.insert_node(&node).await?;
        node_ids.push(node_id);
    }

    println!("  Created {} node(s)", node_ids.len());
    Ok(node_ids)
}

/// Run scan command - one-shot distillation
async fn run_scan(
    project: Option<PathBuf>,
    limit: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let project_path = project.unwrap_or_else(|| {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    });

    println!("ds scan");
    println!("===========");
    println!("Project: {}", project_path.display());

    // Discover sessions
    println!("\nDiscovering sessions...");
    let sessions = match discover_sessions(&project_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to discover sessions: {}", e);
            return Ok(());
        }
    };

    if sessions.is_empty() {
        println!("No sessions found for this project.");
        return Ok(());
    }

    // Apply limit
    let sessions: Vec<SessionInfo> = match limit {
        Some(n) => sessions.into_iter().take(n).collect(),
        None => sessions,
    };

    println!("Found {} session(s) to process", sessions.len());
    if let Some(n) = limit {
        println!("(limited to {} newest)", n);
    }

    // Open store
    let db_path = default_db_path();
    println!("\nDatabase: {}", db_path.display());
    let store = Store::open(db_path.to_str().unwrap()).await?;

    // Create a cancel token for one-shot scan (won't be cancelled)
    let cancel_token = CancellationToken::new();

    // Process each session
    let mut total_nodes = 0;
    let mut skipped = 0;
    let mut failed = 0;

    for (i, session) in sessions.iter().enumerate() {
        println!(
            "\n[{}/{}] Session: {} ({})",
            i + 1,
            sessions.len(),
            &session.session_id[..8],
            format_size(session.size_bytes)
        );

        match process_session(&store, session, &cancel_token, None).await {
            Ok(ProcessResult::Processed(nodes)) => {
                total_nodes += nodes;
            }
            Ok(ProcessResult::Skipped) => {
                skipped += 1;
            }
            Ok(ProcessResult::QueuedForBatch) => {
                // Scan mode doesn't use batching, this shouldn't happen
                eprintln!("  Unexpectedly queued for batch");
            }
            Err(e) => {
                eprintln!("  Error: {}", e);
                failed += 1;
            }
        }
    }

    // Summary
    println!("\n-----------");
    println!("Scan complete!");
    println!("  Processed: {} sessions", sessions.len() - skipped - failed);
    println!("  Skipped:   {} (already processed)", skipped);
    println!("  Failed:    {}", failed);
    println!("  Nodes:     {} created", total_nodes);

    Ok(())
}

/// Delay between processing jobs (rate limiting)
const JOB_DELAY_MS: u64 = 500;

/// Rate limit backoff configuration
/// Exponential backoff: 1min → 2min → 4min → 8min → 16min → 32min → 60min (capped)
const RATE_LIMIT_INITIAL_BACKOFF_SECS: u64 = 60;
const RATE_LIMIT_MAX_BACKOFF_SECS: u64 = 3600; // 1 hour cap

/// Start daemon in background
fn run_start_daemon() -> Result<(), Box<dyn std::error::Error>> {
    // Check if already running
    if let Some(pid) = is_daemon_running() {
        println!("Daemon already running (PID {})", pid);
        return Ok(());
    }

    // Get path to current executable
    let exe = std::env::current_exe()?;
    let log_path = daemon_log_path();

    // Ensure .datasphere directory exists
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    // Spawn daemon process (it manages its own log file with rotation)
    let child = std::process::Command::new(&exe)
        .arg("start")
        .arg("--foreground")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .stdin(std::process::Stdio::null())
        .spawn()?;

    // Write PID file
    let pid = child.id();
    std::fs::write(daemon_pid_path(), pid.to_string())?;

    println!("Daemon started (PID {})", pid);
    println!("Log: {}", log_path.display());

    Ok(())
}

/// Stop running daemon
fn run_stop() -> Result<(), Box<dyn std::error::Error>> {
    match is_daemon_running() {
        Some(pid) => {
            #[cfg(unix)]
            {
                // Send SIGTERM
                let result = unsafe { libc::kill(pid as i32, libc::SIGTERM) };
                if result == 0 {
                    println!("Stopping daemon (PID {})...", pid);

                    // Poll for graceful shutdown (up to 30 seconds)
                    let mut stopped = false;
                    for _ in 0..60 {
                        std::thread::sleep(std::time::Duration::from_millis(500));
                        if is_daemon_running().is_none() {
                            stopped = true;
                            break;
                        }
                    }

                    if !stopped {
                        println!("Daemon still running after 30s, sending SIGKILL...");
                        unsafe { libc::kill(pid as i32, libc::SIGKILL) };
                        std::thread::sleep(std::time::Duration::from_millis(100));
                    }

                    // Clean up PID file
                    let _ = std::fs::remove_file(daemon_pid_path());
                    println!("Daemon stopped");
                } else {
                    eprintln!("Failed to stop daemon: {}", std::io::Error::last_os_error());
                }
            }

            #[cfg(not(unix))]
            {
                eprintln!("Stop not supported on this platform");
            }
        }
        None => {
            println!("Daemon not running");
        }
    }
    Ok(())
}

/// Show daemon status
fn run_status() -> Result<(), Box<dyn std::error::Error>> {
    match is_daemon_running() {
        Some(pid) => {
            println!("Daemon running (PID {})", pid);
            println!("Log: {}", daemon_log_path().display());
        }
        None => {
            println!("Daemon not running");
        }
    }
    Ok(())
}

/// Spawns a watchdog thread that detects system sleep by monitoring time jumps.
/// Returns a receiver that fires when system wakes from sleep.
///
/// The watchdog runs in a std::thread (not tokio) so it's immune to tokio timer
/// issues that can occur after macOS sleep. It uses blocking_send to bridge
/// into the async world.
fn spawn_sleep_watchdog() -> tokio::sync::mpsc::Receiver<()> {
    let (tx, rx) = tokio::sync::mpsc::channel(1);

    std::thread::spawn(move || {
        let mut last = std::time::Instant::now();
        loop {
            std::thread::sleep(std::time::Duration::from_secs(1));
            let elapsed = last.elapsed();
            // If a 1-second sleep took >5 seconds, system likely slept
            if elapsed > std::time::Duration::from_secs(5) {
                eprintln!(
                    "[WATCHDOG] Detected ~{}s time jump (system sleep?)",
                    elapsed.as_secs()
                );
                // blocking_send works from std::thread into tokio channel
                let _ = tx.blocking_send(());
            }
            last = std::time::Instant::now();
        }
    });

    rx
}

/// Run start command - daemon mode watching all projects (foreground)
async fn run_start_foreground() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize global logger for daemon mode
    {
        let mut guard = DAEMON_LOGGER.lock().unwrap();
        *guard = Some(DaemonLog::new()?);
    }

    dlog!("ds start");
    dlog!("============");

    // Open store
    let db_path = default_db_path();
    dlog!("Database: {}", db_path.display());
    let store = Store::open(db_path.to_str().unwrap()).await?;

    // Open queue
    let queue = Queue::open_default().map_err(|e| format!("Failed to open queue: {}", e))?;

    // Resume any pending/processing jobs from previous run
    let (pending, processing, _, _) = queue.counts().unwrap_or((0, 0, 0, 0));
    if pending > 0 || processing > 0 {
        dlog!("Resuming {} pending, {} processing jobs from previous run", pending, processing);
    }

    // Initialize batch queue for large sessions (50% cost savings)
    // AIDEV-NOTE: Sessions >12K tokens are queued for batch API processing
    // Uses resolve_anthropic_model() for consistent model IDs with real-time processing
    let model_env = std::env::var("DATASPHERE_MODEL").unwrap_or_else(|_| "opus".to_string());
    let batch_model = resolve_anthropic_model(&model_env).to_string();
    let batch_state_path = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".datasphere")
        .join("batch_queue.jsonl");
    let batch_queue = Arc::new(Mutex::new(BatchQueue::new(
        batch_model,
        batch_state_path,
    )));

    // Load any pending batches from previous run
    {
        let mut bq = batch_queue.lock().await;
        if let Err(e) = bq.load_state() {
            dlog!("Warning: failed to load batch state: {}", e);
        }
        let pending_batches = bq.pending_batch_count();
        let queued_requests = bq.queued_request_count();
        if pending_batches > 0 || queued_requests > 0 {
            dlog!("Batch queue: {} pending batch(es), {} queued request(s)", pending_batches, queued_requests);
        }
    }

    // Scan existing sessions and queue unprocessed ones
    dlog!("\nScanning existing sessions...");
    let projects = list_all_projects().unwrap_or_default();
    let blacklist = load_blacklist();
    if !blacklist.is_empty() {
        dlog!("Blacklist: {} project(s)", blacklist.len());
    }
    let mut queued_initial = 0;

    for project in &projects {
        // Skip blacklisted projects
        if blacklist.contains(&project.project_id) {
            dlog!("[SKIP] Blacklisted: {}", project.project_id);
            continue;
        }
        if let Ok(sessions) = discover_sessions_in_dir(&project.project_dir) {
            for session in sessions {
                // Check if already processed
                if store.get_processed(&session.session_id).await?.is_some() {
                    continue;
                }

                // Skip empty transcripts
                if !transcript_has_content(&session.transcript_path) {
                    continue;
                }

                let job = Job {
                    source_id: session.session_id.clone(),
                    source_type: "session".to_string(),
                    project_id: project.project_id.clone(),
                    transcript_path: session.transcript_path.to_string_lossy().to_string(),
                    queued_at: Utc::now(),
                    status: JobStatus::Pending,
                    error: None,
                };

                if queue.add(job).is_ok() {
                    queued_initial += 1;
                }
            }
        }
    }

    if queued_initial > 0 {
        dlog!("Queued {} unprocessed session(s)", queued_initial);
    } else {
        dlog!("All sessions already processed");
    }

    // Create all-projects watcher (mutable so we can recreate on wake)
    dlog!("\nStarting all-projects watcher...");
    let mut watcher = match AllProjectsWatcher::new() {
        Ok(w) => w,
        Err(e) => {
            dlog!("Failed to create watcher: {}", e);
            return Ok(());
        }
    };
    dlog!("Watching: {}", watcher.projects_dir().display());

    // Stats
    let mut queued_count = 0;
    let mut processed_count = 0;
    let mut total_nodes = 0;
    let mut batched_count = 0;

    // Batch polling interval tracking
    let mut last_batch_poll = Instant::now();
    const BATCH_POLL_INTERVAL: Duration = Duration::from_secs(60); // Poll batches every minute

    // Rate limit backoff state (local to single-threaded daemon loop)
    let mut rate_limit_backoff_secs: u64 = 0;
    let mut rate_limit_until: Option<Instant> = None;

    // Spawn sleep watchdog to detect system sleep/wake
    let mut wake_rx = spawn_sleep_watchdog();

    // Create cancellation token for graceful shutdown of in-flight requests
    let cancel_token = CancellationToken::new();

    dlog!("\nDaemon running (Ctrl+C to stop)...\n");

    // Set up signal handling for graceful shutdown
    #[cfg(unix)]
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    let mut shutdown = false;

    // Main loop: poll for watcher events, process queue
    while !shutdown {
        // Check for shutdown signals
        #[cfg(unix)]
        {
            use tokio::time::timeout;
            // Non-blocking check for SIGTERM
            if let Ok(Some(())) = timeout(Duration::from_millis(0), sigterm.recv()).await {
                dlog!("\n[SHUTDOWN] Received SIGTERM, cancelling in-flight requests...");
                cancel_token.cancel();
                shutdown = true;
                continue;
            }
        }
        // Check for Ctrl+C (cross-platform)
        {
            use tokio::time::timeout;
            if let Ok(Ok(())) = timeout(Duration::from_millis(0), tokio::signal::ctrl_c()).await {
                dlog!("\n[SHUTDOWN] Received Ctrl+C, cancelling in-flight requests...");
                cancel_token.cancel();
                shutdown = true;
                continue;
            }
        }
        // Drain all pending watcher events into queue
        while let Some(event) = watcher.try_recv() {
            // Skip blacklisted projects
            if blacklist.contains(&event.project_id) {
                continue;
            }

            // Skip empty transcripts
            if !transcript_has_content(&event.session.transcript_path) {
                continue;
            }

            let job = Job {
                source_id: event.session.session_id.clone(),
                source_type: "session".to_string(),
                project_id: event.project_id.clone(),
                transcript_path: event.session.transcript_path.to_string_lossy().to_string(),
                queued_at: Utc::now(),
                status: JobStatus::Pending,
                error: None,
            };

            match queue.add(job) {
                Ok(true) => {
                    queued_count += 1;
                    dlog!(
                        "[QUEUE] {} ({}) from {}",
                        &event.session.session_id[..8.min(event.session.session_id.len())],
                        if event.is_new { "new" } else { "modified" },
                        &event.project_id
                    );
                }
                Ok(false) => {} // Duplicate, already queued
                Err(e) => dlog!("Failed to queue job: {}", e),
            }
        }

        // Check if we're in rate limit backoff
        if let Some(until) = rate_limit_until {
            let remaining = until.saturating_duration_since(Instant::now());
            if !remaining.is_zero() {
                // Sleep for remaining time, capped at 1s for signal responsiveness
                // Use select! so we can wake up early if system resumes from sleep
                tokio::select! {
                    _ = tokio::time::sleep(remaining.min(Duration::from_secs(1))) => {},
                    _ = wake_rx.recv() => {
                        dlog!("[WAKE] System resumed from sleep, reinitializing watcher...");
                        watcher = AllProjectsWatcher::new()?;
                    }
                }
                continue;
            } else {
                // Backoff period ended
                dlog!("[RATE_LIMIT] Backoff period ended, resuming processing");
                rate_limit_until = None;
            }
        }

        // Process one job from queue
        if let Ok(Some(job)) = queue.pop_pending() {
            dlog!(
                "[PROCESS] {} from {}",
                &job.source_id[..8.min(job.source_id.len())],
                &job.project_id
            );

            // Build SessionInfo from job
            let session = SessionInfo {
                session_id: job.source_id.clone(),
                transcript_path: PathBuf::from(&job.transcript_path),
                modified_at: job.queued_at, // Use queued time as proxy
                size_bytes: std::fs::metadata(&job.transcript_path)
                    .map(|m| m.len())
                    .unwrap_or(0),
            };

            match process_session(&store, &session, &cancel_token, Some(Arc::clone(&batch_queue))).await {
                Ok(ProcessResult::Processed(nodes)) => {
                    // Success - reset backoff on successful processing
                    if rate_limit_backoff_secs > 0 {
                        dlog!("[RATE_LIMIT] Success after backoff, resetting backoff state");
                        rate_limit_backoff_secs = 0;
                    }

                    if let Err(e) = queue.mark_done(&job.source_id) {
                        dlog!("  Failed to mark done: {}", e);
                    }
                    processed_count += 1;
                    total_nodes += nodes;
                }
                Ok(ProcessResult::Skipped) => {
                    // Already processed or empty
                    if let Err(e) = queue.mark_done(&job.source_id) {
                        dlog!("  Failed to mark done: {}", e);
                    }
                }
                Ok(ProcessResult::QueuedForBatch) => {
                    // Large session queued for batch processing
                    if let Err(e) = queue.mark_done(&job.source_id) {
                        dlog!("  Failed to mark done: {}", e);
                    }
                    batched_count += 1;
                }
                Err(ProcessError::Cancelled) => {
                    // Request cancelled (shutdown in progress)
                    // Return job to pending state so it will be retried on next start
                    dlog!("[CANCELLED] Job interrupted, returning to pending");
                    if let Err(e) = queue.mark_pending(&job.source_id) {
                        dlog!("  Failed to return job to pending: {}", e);
                    }
                }
                Err(ProcessError::RateLimit(msg)) => {
                    // Rate limit hit - apply exponential backoff
                    // Calculate next backoff duration
                    rate_limit_backoff_secs = if rate_limit_backoff_secs == 0 {
                        RATE_LIMIT_INITIAL_BACKOFF_SECS
                    } else {
                        (rate_limit_backoff_secs * 2).min(RATE_LIMIT_MAX_BACKOFF_SECS)
                    };

                    let backoff_duration = Duration::from_secs(rate_limit_backoff_secs);
                    rate_limit_until = Some(Instant::now() + backoff_duration);

                    dlog!(
                        "[RATE_LIMIT] Hit rate limit: {}. Backing off for {} seconds",
                        msg.lines().next().unwrap_or(&msg),
                        rate_limit_backoff_secs
                    );

                    // Return job to pending state so it will be retried
                    if let Err(e) = queue.mark_pending(&job.source_id) {
                        dlog!("  Failed to return job to pending: {}", e);
                    }
                }
                Err(e) => {
                    dlog!("  Error: {}", e);
                    if let Err(e2) = queue.mark_failed(&job.source_id, &e.to_string()) {
                        dlog!("  Failed to mark failed: {}", e2);
                    }
                }
            }

            // Rate limit between jobs (only if not in backoff)
            // Use select! so we can wake up early if system resumes from sleep
            if rate_limit_until.is_none() {
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_millis(JOB_DELAY_MS)) => {},
                    _ = wake_rx.recv() => {
                        dlog!("[WAKE] System resumed from sleep, reinitializing watcher...");
                        watcher = AllProjectsWatcher::new()?;
                    }
                }
            }
        } else {
            // No pending jobs, sleep briefly before checking again
            // Use select! so we can wake up early if system resumes from sleep
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_millis(100)) => {},
                _ = wake_rx.recv() => {
                    dlog!("[WAKE] System resumed from sleep, reinitializing watcher...");
                    watcher = AllProjectsWatcher::new()?;
                }
            }
        }

        // Batch processing: check for submission and poll for results
        // AIDEV-NOTE: Only do this periodically to avoid excessive API calls
        if last_batch_poll.elapsed() >= BATCH_POLL_INTERVAL {
            last_batch_poll = Instant::now();

            // Check if we should submit a batch
            let should_submit = {
                let bq = batch_queue.lock().await;
                bq.should_submit()
            };

            if should_submit {
                dlog!("[BATCH] Submitting queued requests...");
                let submit_result = {
                    let mut bq = batch_queue.lock().await;
                    bq.submit().await
                };
                match submit_result {
                    Ok(batch_id) => {
                        dlog!("[BATCH] Submitted batch: {}", &batch_id[..8.min(batch_id.len())]);
                    }
                    Err(e) => {
                        dlog!("[BATCH] Submit error: {}", e);
                    }
                }
            }

            // Poll for completed batches and process results
            match process_batch_results(&store, &batch_queue, &cancel_token).await {
                Ok(nodes) if nodes > 0 => {
                    total_nodes += nodes;
                    dlog!("[BATCH] Created {} node(s) from batch results", nodes);
                }
                Ok(_) => {}
                Err(e) => {
                    dlog!("[BATCH] Error processing results: {}", e);
                }
            }
        }
    }

    // Clean up PID file on graceful shutdown
    let _ = std::fs::remove_file(daemon_pid_path());

    // Save batch state before shutdown
    {
        let bq = batch_queue.lock().await;
        if let Err(e) = bq.save_state() {
            dlog!("Warning: failed to save batch state: {}", e);
        }
    }

    dlog!("\nDaemon stopped.");
    dlog!("  Queued:    {} sessions", queued_count);
    dlog!("  Processed: {} sessions", processed_count);
    dlog!("  Batched:   {} sessions", batched_count);
    dlog!("  Nodes:     {} created", total_nodes);
    Ok(())
}

/// Run queue command - show/manage job queue
fn run_queue(action: Option<QueueAction>) -> Result<(), Box<dyn std::error::Error>> {
    let queue = Queue::open_default().map_err(|e| format!("Failed to open queue: {}", e))?;

    match action.unwrap_or(QueueAction::Status) {
        QueueAction::Status => {
            let (pending, processing, done, failed) = queue.counts()?;
            println!("ds queue");
            println!("============");
            println!("Pending:    {}", pending);
            println!("Processing: {}", processing);
            println!("Done:       {}", done);
            println!("Failed:     {}", failed);
        }

        QueueAction::Pending => {
            let jobs = queue.list_pending()?;
            if jobs.is_empty() {
                println!("No pending jobs.");
            } else {
                println!("Pending jobs ({}):", jobs.len());
                for job in jobs {
                    println!(
                        "  {} ({})",
                        &job.source_id[..8.min(job.source_id.len())],
                        &job.project_id
                    );
                }
            }
        }

        QueueAction::Failed => {
            let jobs = queue.list_failed()?;
            if jobs.is_empty() {
                println!("No failed jobs.");
            } else {
                println!("Failed jobs ({}):", jobs.len());
                for job in jobs {
                    println!(
                        "  {} ({}) - {}",
                        &job.source_id[..8.min(job.source_id.len())],
                        &job.project_id,
                        job.error.as_deref().unwrap_or("unknown error")
                    );
                }
            }
        }

        QueueAction::Clear => {
            let cleared = queue.clear_done()?;
            println!("Cleared {} completed jobs.", cleared);
        }

        QueueAction::Nuke => {
            let nuked = queue.nuke()?;
            println!("Nuked {} jobs.", nuked);
        }

        QueueAction::Retry { source_id } => {
            match source_id {
                Some(id) => {
                    if queue.retry_one(&id)? {
                        println!("Requeued job: {}", &id[..8.min(id.len())]);
                    } else {
                        println!("Job not found or not in failed state: {}", &id[..8.min(id.len())]);
                    }
                }
                None => {
                    let count = queue.retry_all()?;
                    if count > 0 {
                        println!("Requeued {} failed job(s).", count);
                    } else {
                        println!("No failed jobs to retry.");
                    }
                }
            }
        }
    }

    Ok(())
}

/// Run query command - search knowledge graph
async fn run_query(
    query: &str,
    limit: usize,
    format: &str,
    namespace_filter: Option<&str>,
    db_override: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let db_path = resolve_db_path(db_override);

    if !db_path.exists() {
        eprintln!("Database not found at: {}", db_path.display());
        return Ok(());
    }

    let store = Store::open(db_path.to_str().unwrap()).await?;

    // Create a cancel token for one-shot query (won't be cancelled)
    let cancel_token = CancellationToken::new();

    // Embed the query
    let embedding = embed(query, &cancel_token).await?;

    // Search for similar nodes (fetch more if filtering)
    let fetch_limit = if namespace_filter.is_some() { limit * 3 } else { limit };
    let results = store.search_similar_with_scores(&embedding, fetch_limit).await?;

    // Filter by namespace if specified
    let results: Vec<_> = if let Some(ns) = namespace_filter {
        results
            .into_iter()
            .filter(|(node, _)| node.namespace == ns)
            .take(limit)
            .collect()
    } else {
        results.into_iter().take(limit).collect()
    };

    if results.is_empty() {
        if format == "json" {
            println!("[]");
        } else {
            println!("No relevant results found.");
        }
        return Ok(());
    }

    if format == "json" {
        // JSON output for MCP consumption
        let json_results: Vec<serde_json::Value> = results
            .iter()
            .map(|(node, score)| {
                serde_json::json!({
                    "id": node.id.to_string(),
                    "content": node.content,
                    "source": node.source,
                    "namespace": node.namespace,
                    "similarity": score,
                    "timestamp": node.timestamp.to_rfc3339(),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&json_results)?);
    } else {
        // Human-readable text output
        for (i, (node, score)) in results.iter().enumerate() {
            println!("─── Result {} (similarity: {:.2}) ───", i + 1, score);
            println!("Source:    {}", node.source);
            println!("Namespace: {}", node.namespace);
            println!("Time:      {}", node.timestamp.format("%Y-%m-%d %H:%M"));
            println!("{}", node.content);
            println!();
        }
    }

    Ok(())
}

/// Run related command - find nodes similar to a given node
async fn run_related(
    node_id: &str,
    limit: usize,
    format: &str,
    db_override: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let db_path = resolve_db_path(db_override);

    if !db_path.exists() {
        eprintln!("Database not found at: {}", db_path.display());
        return Ok(());
    }

    // Parse node ID as UUID
    let uuid = node_id.parse::<Uuid>().map_err(|_| {
        format!("Invalid node ID: {}. Expected a UUID.", node_id)
    })?;

    let store = Store::open(db_path.to_str().unwrap()).await?;

    // Get the source node
    let node = store.get_node(uuid).await?.ok_or_else(|| {
        format!("Node not found: {}", node_id)
    })?;

    // Search for similar nodes (request one extra to filter out self)
    let results = store.search_similar_with_scores(&node.embedding, limit + 1).await?;

    // Filter out the source node itself
    let results: Vec<_> = results
        .into_iter()
        .filter(|(n, _)| n.id != uuid)
        .take(limit)
        .collect();

    if results.is_empty() {
        if format == "json" {
            println!("[]");
        } else {
            println!("No related nodes found.");
        }
        return Ok(());
    }

    if format == "json" {
        let json_results: Vec<serde_json::Value> = results
            .iter()
            .map(|(n, score)| {
                serde_json::json!({
                    "id": n.id.to_string(),
                    "content": n.content,
                    "source": n.source,
                    "similarity": score,
                    "timestamp": n.timestamp.to_rfc3339(),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&json_results)?);
    } else {
        println!("Nodes related to {}:", &node_id[..8.min(node_id.len())]);
        println!();
        for (i, (n, score)) in results.iter().enumerate() {
            println!("─── {} (similarity: {:.2}) ───", i + 1, score);
            println!("ID:     {}", n.id);
            println!("Source: {}", n.source);
            println!("Time:   {}", n.timestamp.format("%Y-%m-%d %H:%M"));
            println!("{}", n.content);
            println!();
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Stats { db } => {
            let db_path = resolve_db_path(db);

            if !db_path.exists() {
                println!("Database not found at: {}", db_path.display());
                println!("Run 'ds scan' to create the database.");
                return Ok(());
            }

            let store = Store::open(db_path.to_str().unwrap()).await?;

            let nodes = store.count_nodes().await?;
            let processed = store.count_processed().await?;
            let db_size = dir_size(&db_path);

            println!("ds stats");
            println!("============");
            println!("Database:   {}", db_path.display());
            println!("Size:       {}", format_size(db_size));
            println!();
            println!("Nodes:      {}", nodes);
            println!("Processed:  {} transcripts", processed);
        }

        Commands::Show { limit, db } => {
            let db_path = resolve_db_path(db);

            if !db_path.exists() {
                println!("Database not found at: {}", db_path.display());
                return Ok(());
            }

            let store = Store::open(db_path.to_str().unwrap()).await?;
            let nodes = store.list_nodes(limit).await?;

            if nodes.is_empty() {
                println!("No nodes stored yet.");
                return Ok(());
            }

            for (i, node) in nodes.iter().enumerate() {
                println!("─── Node {} ───", i + 1);
                println!("ID:        {}", node.id);
                println!("Source:    {}", node.source);
                println!("Namespace: {}", node.namespace);
                println!("Time:      {}", node.timestamp.format("%Y-%m-%d %H:%M"));
                println!("Content:\n{}", node.content);
                println!();
            }
        }

        Commands::Scan { limit, project } => {
            run_scan(project, limit).await?;
        }

        Commands::Start { foreground } => {
            if foreground {
                run_start_foreground().await?;
            } else {
                run_start_daemon()?;
            }
        }

        Commands::Stop => {
            run_stop()?;
        }

        Commands::Status => {
            run_status()?;
        }

        Commands::Queue { action } => {
            run_queue(action)?;
        }

        Commands::Add { file } => {
            if !file.exists() {
                eprintln!("File not found: {}", file.display());
                return Ok(());
            }

            let db_path = default_db_path();
            let store = Store::open(db_path.to_str().unwrap()).await?;
            let cancel_token = CancellationToken::new();

            println!("ds add");
            println!("==========");

            match process_file(&store, &file, &cancel_token).await {
                Ok(nodes) => {
                    if nodes > 0 {
                        println!("\nCreated {} node(s)", nodes);
                    }
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                }
            }
        }

        Commands::Note { content, source, namespace } => {
            let db_path = default_db_path();
            let store = Store::open(db_path.to_str().unwrap()).await?;
            let cancel_token = CancellationToken::new();

            println!("ds note");
            println!("==========");

            match process_note(&store, &content, &source, &namespace, &cancel_token).await {
                Ok(node_ids) => {
                    for id in &node_ids {
                        println!("  ID: {}", id);
                    }
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                }
            }
        }

        Commands::Query { query, limit, format, namespace, db } => {
            run_query(&query, limit, &format, namespace.as_deref(), db).await?;
        }

        Commands::Related { node_id, limit, format, db } => {
            run_related(&node_id, limit, &format, db).await?;
        }

        Commands::Reset => {
            println!("ds reset");
            println!("============");

            // Delete database
            let db_path = default_db_path();
            if db_path.exists() {
                std::fs::remove_dir_all(&db_path)
                    .map_err(|e| format!("Failed to delete database: {}", e))?;
                println!("Deleted database: {}", db_path.display());
            } else {
                println!("Database not found (already clean)");
            }

            // Nuke queue
            let queue = Queue::open_default()?;
            let nuked = queue.nuke()?;
            if nuked > 0 {
                println!("Nuked {} queued jobs", nuked);
            } else {
                println!("Queue was empty");
            }

            println!("\nReset complete. Run 'ds start' to begin fresh.");
        }
    }

    Ok(())
}
