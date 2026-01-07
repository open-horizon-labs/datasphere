use chrono::Utc;
use clap::{Parser, Subcommand};
use engram::{
    chunk_text, discover_sessions, discover_sessions_in_dir, embed, extract_knowledge,
    list_all_projects, read_transcript, AllProjectsWatcher, Job, JobStatus, Node, Processed,
    Queue, SessionInfo, SourceType, Store,
};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use uuid::Uuid;

#[derive(Parser)]
#[command(name = "engram")]
#[command(about = "Background daemon that distills knowledge from local sources")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Show database statistics
    Stats,

    /// Show stored nodes
    Show {
        /// Maximum number of nodes to show
        #[arg(short, long, default_value = "10")]
        limit: usize,
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
    Start,

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
    },
}

#[derive(Subcommand)]
enum QueueAction {
    /// Show queue counts (default)
    Status,
    /// List pending jobs
    Pending,
    /// Clear completed jobs
    Clear,
}

/// Get the default database path
fn default_db_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".engram")
        .join("db")
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

/// Process a single session transcript
/// Returns (nodes_created, skipped) tuple
async fn process_session(
    store: &Store,
    session: &SessionInfo,
) -> Result<(usize, bool), Box<dyn std::error::Error>> {
    println!("  Reading transcript...");

    // Parse transcript
    let entries = read_transcript(&session.transcript_path)?;
    println!("  Parsed {} entries", entries.len());

    if entries.is_empty() {
        println!("  Empty transcript, skipping");
        return Ok((0, true));
    }

    // Get all messages for context
    let messages: Vec<_> = entries
        .iter()
        .filter(|e| e.is_message() || e.is_summary())
        .collect();

    if messages.is_empty() {
        println!("  No messages found, skipping");
        return Ok((0, true));
    }

    // Count message types
    let user_count = messages.iter().filter(|e| e.is_user()).count();
    let assistant_count = messages.iter().filter(|e| e.is_assistant()).count();
    let summary_count = messages.iter().filter(|e| e.is_summary()).count();

    println!(
        "  Collected {} items ({} user, {} assistant, {} summaries)",
        messages.len(),
        user_count,
        assistant_count,
        summary_count
    );

    // Format context for LLM
    let context = engram::format_context(&messages);
    if context.trim().is_empty() {
        println!("  Empty context after formatting, skipping");
        return Ok((0, true));
    }

    println!("  Context size: {} chars", context.len());

    // Compute SimHash of context
    let current_simhash = simhash::simhash(&context) as i64;

    // Check if already processed
    if let Some(existing) = store.get_processed(&session.session_id).await? {
        let hamming = simhash::hamming_distance(existing.simhash as u64, current_simhash as u64);

        if hamming <= SIMHASH_CHANGE_THRESHOLD {
            println!(
                "  Unchanged (simhash distance: {} bits, threshold: {})",
                hamming, SIMHASH_CHANGE_THRESHOLD
            );
            return Ok((0, true));
        }

        println!(
            "  Session changed (simhash distance: {} bits), re-distilling...",
            hamming
        );

        // Delete old nodes, then processed record
        let node_ids: Vec<Uuid> = existing.node_ids
            .iter()
            .filter_map(|id| id.parse::<Uuid>().ok())
            .collect();
        if !node_ids.is_empty() {
            store.delete_nodes(&node_ids).await?;
            println!("  Deleted {} old node(s)", node_ids.len());
        }
        store.delete_processed(&session.session_id).await?;
    }

    // Distill knowledge via LLM
    println!("  Distilling via LLM...");
    let distill_start = Instant::now();
    let extraction = match extract_knowledge(&context).await {
        Ok(result) => result,
        Err(e) => {
            eprintln!("  LLM extraction failed: {}", e);
            return Err(e.into());
        }
    };
    let distill_elapsed = distill_start.elapsed();

    // Log chunking info if used
    if extraction.chunks_used > 1 {
        println!("  Distilled {} chunks in {:.1}s", extraction.chunks_used, distill_elapsed.as_secs_f32());
    } else {
        println!("  Distilled in {:.1}s", distill_elapsed.as_secs_f32());
    }

    if extraction.insights.is_empty() {
        println!("  No substantive knowledge found");
        // Still record as processed
        let record = Processed {
            source_id: session.session_id.clone(),
            source_type: "session".to_string(),
            simhash: current_simhash,
            processed_at: Utc::now(),
            node_count: 0,
            node_ids: Vec::new(),
        };
        store.insert_processed(&record).await?;
        return Ok((0, false));
    }

    println!("  Extracted {} insight(s)", extraction.insights.len());

    // Embed and store each insight as a separate node
    let total_insights = extraction.insights.len();
    let mut node_ids = Vec::new();
    for (i, insight) in extraction.insights.into_iter().enumerate() {
        let embed_start = Instant::now();
        let embedding = embed(&insight.content).await?;
        println!("  Embedded {}/{} in {:.1}s", i + 1, total_insights, embed_start.elapsed().as_secs_f32());

        let node = insight.into_node(
            session.session_id.clone(),
            SourceType::Session,
            embedding,
        );
        let node_id = node.id.to_string();

        store.insert_node(&node).await?;

        node_ids.push(node_id);
    }

    // Record as processed
    let record = Processed {
        source_id: session.session_id.clone(),
        source_type: "session".to_string(),
        simhash: current_simhash,
        processed_at: Utc::now(),
        node_count: node_ids.len() as i32,
        node_ids,
    };
    store.insert_processed(&record).await?;

    println!("  Done! Created {} node(s)", record.node_count);
    Ok((record.node_count as usize, false))
}

/// Process a single text file (no LLM distillation, direct embedding)
/// Returns number of nodes created
async fn process_file(
    store: &Store,
    file_path: &PathBuf,
) -> Result<usize, Box<dyn std::error::Error>> {
    // Canonicalize path to avoid duplicates
    let canonical_path = file_path.canonicalize()
        .map_err(|e| format!("Failed to canonicalize path: {}", e))?;
    let source_id = canonical_path.to_string_lossy().to_string();

    println!("Processing file: {}", canonical_path.display());

    // Read file content
    let content = std::fs::read_to_string(&canonical_path)
        .map_err(|e| format!("Failed to read file: {}", e))?;

    if content.trim().is_empty() {
        println!("  Empty file, skipping");
        return Ok(0);
    }

    println!("  File size: {} chars", content.len());

    // Compute SimHash
    let current_simhash = simhash::simhash(&content) as i64;

    // Check if already processed
    if let Some(existing) = store.get_processed(&source_id).await? {
        let hamming = simhash::hamming_distance(existing.simhash as u64, current_simhash as u64);

        if hamming <= SIMHASH_CHANGE_THRESHOLD {
            println!(
                "  Unchanged (simhash distance: {} bits, threshold: {})",
                hamming, SIMHASH_CHANGE_THRESHOLD
            );
            return Ok(0);
        }

        println!(
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
            println!("  Deleted {} old node(s)", node_ids.len());
        }
        store.delete_processed(&source_id).await?;
    }

    // Chunk content if needed
    let chunks = chunk_text(&content);
    println!("  Chunks: {}", chunks.len());

    // Embed each chunk and create nodes
    let mut node_ids = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
        let embed_start = Instant::now();
        let embedding = embed(chunk).await?;
        println!("  Embedded {}/{} in {:.1}s", i + 1, chunks.len(), embed_start.elapsed().as_secs_f32());

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

    println!("  Done! Created {} node(s)", record.node_count);
    Ok(record.node_count as usize)
}

/// Run scan command - one-shot distillation
async fn run_scan(
    project: Option<PathBuf>,
    limit: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let project_path = project.unwrap_or_else(|| {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    });

    println!("engram scan");
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

        match process_session(&store, session).await {
            Ok((nodes, was_skipped)) => {
                total_nodes += nodes;
                if was_skipped {
                    skipped += 1;
                }
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

/// Run start command - daemon mode watching all projects
async fn run_start() -> Result<(), Box<dyn std::error::Error>> {
    println!("engram start");
    println!("============");

    // Open store
    let db_path = default_db_path();
    println!("Database: {}", db_path.display());
    let store = Store::open(db_path.to_str().unwrap()).await?;

    // Open queue
    let queue = Queue::open_default().map_err(|e| format!("Failed to open queue: {}", e))?;

    // Resume any pending/processing jobs from previous run
    let (pending, processing, _, _) = queue.counts().unwrap_or((0, 0, 0, 0));
    if pending > 0 || processing > 0 {
        println!("Resuming {} pending, {} processing jobs from previous run", pending, processing);
    }

    // Scan existing sessions and queue unprocessed ones
    println!("\nScanning existing sessions...");
    let projects = list_all_projects().unwrap_or_default();
    let mut queued_initial = 0;

    for project in &projects {
        if let Ok(sessions) = discover_sessions_in_dir(&project.project_dir) {
            for session in sessions {
                // Check if already processed
                if store.get_processed(&session.session_id).await?.is_some() {
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
        println!("Queued {} unprocessed session(s)", queued_initial);
    } else {
        println!("All sessions already processed");
    }

    // Create all-projects watcher
    println!("\nStarting all-projects watcher...");
    let watcher = match AllProjectsWatcher::new() {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Failed to create watcher: {}", e);
            return Ok(());
        }
    };
    println!("Watching: {}", watcher.projects_dir().display());

    // Stats (prefixed with _ since loop is infinite, never printed)
    let mut _queued_count = 0;
    let mut _processed_count = 0;
    let mut _total_nodes = 0;

    println!("\nDaemon running (Ctrl+C to stop)...\n");

    // Main loop: poll for watcher events, process queue
    loop {
        // Drain all pending watcher events into queue
        while let Some(event) = watcher.try_recv() {
            let job = Job {
                source_id: event.session.session_id.clone(),
                source_type: "session".to_string(),
                project_id: event.project_id.clone(),
                transcript_path: event.session.transcript_path.to_string_lossy().to_string(),
                queued_at: Utc::now(),
                status: JobStatus::Pending,
                error: None,
            };

            if let Err(e) = queue.add(job) {
                eprintln!("Failed to queue job: {}", e);
            } else {
                _queued_count += 1;
                println!(
                    "[QUEUE] {} ({}) from {}",
                    &event.session.session_id[..8.min(event.session.session_id.len())],
                    if event.is_new { "new" } else { "modified" },
                    &event.project_id[..20.min(event.project_id.len())]
                );
            }
        }

        // Process one job from queue
        if let Ok(Some(job)) = queue.pop_pending() {
            println!(
                "[PROCESS] {} from {}",
                &job.source_id[..8.min(job.source_id.len())],
                &job.project_id[..20.min(job.project_id.len())]
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

            match process_session(&store, &session).await {
                Ok((nodes, was_skipped)) => {
                    if let Err(e) = queue.mark_done(&job.source_id) {
                        eprintln!("  Failed to mark done: {}", e);
                    }
                    if !was_skipped {
                        _processed_count += 1;
                        _total_nodes += nodes;
                    }
                }
                Err(e) => {
                    eprintln!("  Error: {}", e);
                    if let Err(e2) = queue.mark_failed(&job.source_id, &e.to_string()) {
                        eprintln!("  Failed to mark failed: {}", e2);
                    }
                }
            }

            // Rate limit
            tokio::time::sleep(Duration::from_millis(JOB_DELAY_MS)).await;
        } else {
            // No pending jobs, sleep briefly before checking again
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    // Note: This is unreachable but kept for completeness
    #[allow(unreachable_code)]
    {
        println!("\nDaemon stopped.");
        println!("  Queued:    {} sessions", _queued_count);
        println!("  Processed: {} sessions", _processed_count);
        println!("  Nodes:     {} created", _total_nodes);
        Ok(())
    }
}

/// Run queue command - show/manage job queue
fn run_queue(action: Option<QueueAction>) -> Result<(), Box<dyn std::error::Error>> {
    let queue = Queue::open_default().map_err(|e| format!("Failed to open queue: {}", e))?;

    match action.unwrap_or(QueueAction::Status) {
        QueueAction::Status => {
            let (pending, processing, done, failed) = queue.counts()?;
            println!("engram queue");
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
                        &job.project_id[..30.min(job.project_id.len())]
                    );
                }
            }
        }

        QueueAction::Clear => {
            let cleared = queue.clear_done()?;
            println!("Cleared {} completed jobs.", cleared);
        }
    }

    Ok(())
}

/// Run query command - search knowledge graph
async fn run_query(
    query: &str,
    limit: usize,
    format: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let db_path = default_db_path();

    if !db_path.exists() {
        eprintln!("Database not found at: {}", db_path.display());
        return Ok(());
    }

    let store = Store::open(db_path.to_str().unwrap()).await?;

    // Embed the query
    let embedding = embed(query).await?;

    // Search for similar nodes
    let results = store.search_similar_with_scores(&embedding, limit).await?;

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
            println!("Source: {}", node.source);
            println!("Time:   {}", node.timestamp.format("%Y-%m-%d %H:%M"));
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
) -> Result<(), Box<dyn std::error::Error>> {
    let db_path = default_db_path();

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
        Commands::Stats => {
            let db_path = default_db_path();

            if !db_path.exists() {
                println!("Database not found at: {}", db_path.display());
                println!("Run engram scan to create the database.");
                return Ok(());
            }

            let store = Store::open(db_path.to_str().unwrap()).await?;

            let nodes = store.count_nodes().await?;
            let processed = store.count_processed().await?;
            let db_size = dir_size(&db_path);

            println!("engram stats");
            println!("============");
            println!("Database:   {}", db_path.display());
            println!("Size:       {}", format_size(db_size));
            println!();
            println!("Nodes:      {}", nodes);
            println!("Processed:  {} transcripts", processed);
        }

        Commands::Show { limit } => {
            let db_path = default_db_path();

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
                println!("ID:      {}", node.id);
                println!("Source:  {}", node.source);
                println!("Time:    {}", node.timestamp.format("%Y-%m-%d %H:%M"));
                println!("Content:\n{}", node.content);
                println!();
            }
        }

        Commands::Scan { limit, project } => {
            run_scan(project, limit).await?;
        }

        Commands::Start => {
            run_start().await?;
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

            println!("engram add");
            println!("==========");

            match process_file(&store, &file).await {
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

        Commands::Query { query, limit, format } => {
            run_query(&query, limit, &format).await?;
        }

        Commands::Related { node_id, limit, format } => {
            run_related(&node_id, limit, &format).await?;
        }
    }

    Ok(())
}
