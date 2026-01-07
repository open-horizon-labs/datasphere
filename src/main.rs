use chrono::Utc;
use clap::{Parser, Subcommand};
use engram::{
    chunk_text, discover_sessions, embed, extract_knowledge, read_transcript, Edge, Node, Processed,
    SessionEvent, SessionInfo, SessionWatcher, SourceType, Store,
};
use std::path::PathBuf;
use uuid::Uuid;

#[derive(Parser)]
#[command(name = "engram")]
#[command(about = "Background daemon that distills and links knowledge from local sources")]
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

    /// Start the daemon (continuous watching)
    Start {
        /// Maximum number of transcripts to process on startup
        #[arg(short, long)]
        limit: Option<usize>,

        /// Project path to watch (defaults to current directory)
        #[arg(short, long)]
        project: Option<PathBuf>,
    },

    /// Add a text file to the knowledge graph (no LLM distillation)
    Add {
        /// Path to the file to add
        file: PathBuf,
    },
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

/// Similarity threshold for creating edges between nodes
/// AIDEV-NOTE: 0.6 is relatively permissive - "for serendipity" per spec
const SIMILARITY_THRESHOLD: f32 = 0.6;

/// Maximum nodes to search when linking
const MAX_SIMILAR_SEARCH: usize = 20;

/// Link a newly created node to similar existing nodes
/// Returns number of edges created
async fn link_node(
    store: &Store,
    node: &Node,
) -> Result<usize, Box<dyn std::error::Error>> {
    let similar = store
        .search_similar_with_scores(&node.embedding, MAX_SIMILAR_SEARCH)
        .await?;

    let mut edges_created = 0;
    for (other, similarity) in similar {
        // Skip self-links
        if other.id == node.id {
            continue;
        }
        // Skip if below threshold
        if similarity < SIMILARITY_THRESHOLD {
            continue;
        }

        let edge = Edge::new(node.id, other.id, similarity);
        store.insert_edge(&edge).await?;
        edges_created += 1;
    }

    Ok(edges_created)
}

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

        // Delete old nodes and their edges, then processed record
        let node_ids: Vec<Uuid> = existing.node_ids
            .iter()
            .filter_map(|id| id.parse::<Uuid>().ok())
            .collect();
        if !node_ids.is_empty() {
            // Delete edges first (referential integrity)
            for node_id in &node_ids {
                store.delete_edges_for_node(*node_id).await?;
            }
            store.delete_nodes(&node_ids).await?;
            println!("  Deleted {} old node(s) and their edges", node_ids.len());
        }
        store.delete_processed(&session.session_id).await?;
    }

    // Distill knowledge via LLM
    println!("  Distilling via LLM...");
    let extraction = match extract_knowledge(&context).await {
        Ok(result) => result,
        Err(e) => {
            eprintln!("  LLM extraction failed: {}", e);
            return Err(e.into());
        }
    };

    // Log chunking info if used
    if extraction.chunks_used > 1 {
        println!("  Used {} chunks for distillation", extraction.chunks_used);
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
    let mut total_edges = 0;
    for (i, insight) in extraction.insights.into_iter().enumerate() {
        println!("  Embedding insight {}/{}...", i + 1, total_insights);
        let embedding = embed(&insight.content).await?;

        let node = insight.into_node(
            session.session_id.clone(),
            SourceType::Session,
            embedding,
        );
        let node_id = node.id.to_string();

        println!("  Storing node {}...", &node_id[..8]);
        store.insert_node(&node).await?;

        // Link to similar existing nodes
        let edges = link_node(store, &node).await?;
        if edges > 0 {
            println!("  Linked to {} similar node(s)", edges);
        }
        total_edges += edges;

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

    println!("  Done! Created {} node(s), {} edge(s)", record.node_count, total_edges);
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

        // Delete old nodes and their edges, then processed record
        let node_ids: Vec<Uuid> = existing.node_ids
            .iter()
            .filter_map(|id| id.parse::<Uuid>().ok())
            .collect();
        if !node_ids.is_empty() {
            // Delete edges first (referential integrity)
            for node_id in &node_ids {
                store.delete_edges_for_node(*node_id).await?;
            }
            store.delete_nodes(&node_ids).await?;
            println!("  Deleted {} old node(s) and their edges", node_ids.len());
        }
        store.delete_processed(&source_id).await?;
    }

    // Chunk content if needed
    let chunks = chunk_text(&content);
    println!("  Chunks: {}", chunks.len());

    // Embed each chunk and create nodes
    let mut node_ids = Vec::new();
    let mut total_edges = 0;
    for (i, chunk) in chunks.iter().enumerate() {
        println!("  Embedding chunk {}/{}...", i + 1, chunks.len());
        let embedding = embed(chunk).await?;

        let node = Node::new(
            chunk.clone(),
            source_id.clone(),
            SourceType::File,
            embedding,
            1.0, // Full confidence for raw file content
        );
        let node_id = node.id.to_string();
        store.insert_node(&node).await?;

        // Link to similar existing nodes
        let edges = link_node(store, &node).await?;
        if edges > 0 {
            println!("  Linked to {} similar node(s)", edges);
        }
        total_edges += edges;

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

    println!("  Done! Created {} node(s), {} edge(s)", record.node_count, total_edges);
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

/// Run start command - daemon mode
async fn run_start(
    project: Option<PathBuf>,
    limit: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let project_path = project.unwrap_or_else(|| {
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    });

    println!("engram start");
    println!("============");
    println!("Project: {}", project_path.display());

    // Open store
    let db_path = default_db_path();
    println!("Database: {}", db_path.display());
    let store = Store::open(db_path.to_str().unwrap()).await?;

    // Create watcher - this emits Created events for all existing sessions
    println!("\nStarting session watcher...");
    let watcher = match SessionWatcher::new(&project_path) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Failed to create watcher: {}", e);
            return Ok(());
        }
    };
    println!("Watching: {}", watcher.project_dir().display());

    // Process events
    let mut processed_count = 0;
    let mut total_nodes = 0;

    println!("\nProcessing sessions (Ctrl+C to stop)...\n");

    for event in watcher.iter() {
        let (session, event_type) = match event {
            SessionEvent::Created(s) => (s, "new"),
            SessionEvent::Modified(s) => (s, "modified"),
        };

        // Apply limit for initial scan (Created events come first)
        if event_type == "new" {
            if let Some(max) = limit {
                if processed_count >= max {
                    println!(
                        "Reached limit of {} sessions, skipping remaining initial sessions",
                        max
                    );
                    // Drain remaining Created events without processing
                    while let Some(SessionEvent::Created(_)) = watcher.try_recv() {}
                    println!("Now watching for new changes...\n");
                    continue;
                }
            }
        }

        println!(
            "[{}] Session {} ({}) - {}",
            if event_type == "new" { "NEW" } else { "MOD" },
            &session.session_id[..8.min(session.session_id.len())],
            format_size(session.size_bytes),
            event_type
        );

        match process_session(&store, &session).await {
            Ok((nodes, was_skipped)) => {
                if !was_skipped {
                    processed_count += 1;
                    total_nodes += nodes;
                }
            }
            Err(e) => {
                eprintln!("  Error: {}", e);
            }
        }
    }

    println!("\nDaemon stopped.");
    println!("  Processed: {} sessions", processed_count);
    println!("  Nodes:     {} created", total_nodes);

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
            let edges = store.count_edges().await?;
            let processed = store.count_processed().await?;
            let db_size = dir_size(&db_path);

            println!("engram stats");
            println!("============");
            println!("Database:   {}", db_path.display());
            println!("Size:       {}", format_size(db_size));
            println!();
            println!("Nodes:      {}", nodes);
            println!("Edges:      {}", edges);
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

        Commands::Start { limit, project } => {
            run_start(project, limit).await?;
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
    }

    Ok(())
}
