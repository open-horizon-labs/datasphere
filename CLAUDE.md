# CLAUDE.md

Engram — background daemon that distills and links knowledge from Claude Code sessions.

## Build & Test

```bash
cargo build              # Development build
cargo build --release    # Release build
cargo test               # Run tests
```

## Architecture

```
[All Claude projects] ──▶ AllProjectsWatcher ──▶ Queue ──▶ Processor
     (~/.claude/                                              │
      projects/)                                              ▼
                                                    ┌─────────────────┐
                                                    │ For each job:   │
                                                    │ 1. Read JSONL   │
                                                    │ 2. Distill      │
                                                    │ 3. Embed        │
                                                    │ 4. Link         │
                                                    │ 5. Store        │
                                                    └─────────────────┘
```

### Components

| Component | Purpose |
|-----------|---------|
| AllProjectsWatcher | Watch `~/.claude/projects/` recursively for `.jsonl` changes |
| Queue | Persistent JSONL queue at `~/.engram/queue.jsonl` |
| Distiller | Extract insights via Claude (chunked, no synthesis) |
| Embedder | OpenAI `text-embedding-3-small` (1536 dims) |
| Linker | Create edges where cosine similarity > 0.6 |
| Store | LanceDB (nodes, edges, processed tables) |

## Module Structure

```
src/
├── main.rs           # CLI: scan, start, query, queue, stats, show, add
├── lib.rs            # Public exports
├── core/
│   ├── node.rs       # Node type (EMBEDDING_DIM = 1536)
│   └── edge.rs       # Edge type (source, target, weight)
├── store/
│   ├── lance.rs      # LanceDB wrapper + vector search
│   └── schema.rs     # Arrow schemas
├── distill.rs        # LLM knowledge extraction (chunked)
├── embed.rs          # OpenAI embeddings + text chunking
├── llm.rs            # Claude CLI wrapper
├── queue.rs          # Persistent job queue
├── session.rs        # Session discovery + project listing
├── transcript/
│   ├── reader.rs     # JSONL parsing, context formatting
│   └── types.rs      # TranscriptEntry types
└── watch/
    ├── session.rs    # Single-project watcher
    └── all_projects.rs  # All-projects watcher for daemon
```

## Storage

```
~/.engram/
├── db/               # LanceDB database
│   ├── nodes.lance/
│   ├── edges.lance/
│   └── processed.lance/
└── queue.jsonl       # Job queue (append-only)
```

## Key Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `EMBEDDING_DIM` | 1536 | text-embedding-3-small dimensions |
| `SIMILARITY_THRESHOLD` | 0.6 | Minimum similarity to create edge |
| `MAX_SIMILAR_SEARCH` | 20 | Nodes to check when linking |
| `SIMHASH_CHANGE_THRESHOLD` | 10 | Bits changed to trigger re-process |
| `JOB_DELAY_MS` | 500 | Rate limiting between jobs |

## CLI Commands

```bash
engram scan              # One-shot scan current project
engram start             # Daemon: watch all projects
engram query "text"      # Search knowledge graph
engram query -f json "x" # JSON output for MCP
engram queue             # Show queue counts
engram queue pending     # List pending jobs
engram queue clear       # Remove completed jobs
engram stats             # Database statistics
engram show              # Display nodes
engram add <file>        # Add file (no LLM, direct embed)
```

## MCP Server

`mcp/` contains Node.js MCP server that shells out to `engram query`:

```bash
cd mcp && npm install
claude mcp add engram -s user -- node /path/to/mcp/index.js
```

Exposes single tool: `engram_query(query, limit?)`.

## Design Principles

1. **Background first**: Daemon watches all projects, processes incrementally
2. **Explicit queue**: All jobs are queued for introspection and rate limiting
3. **Local**: All data stays on machine
4. **No synthesis**: Each chunk becomes its own node (avoids recency bias)
5. **Semantic linking**: Nodes connected by embedding similarity
