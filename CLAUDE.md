# CLAUDE.md

Engram — background daemon that distills and links knowledge from local sources.

## Build & Test

```bash
cargo build              # Development build
cargo build --release    # Release build
cargo test               # Run tests
```

## Architecture

```
[Session transcripts] ──┐
[File changes] ─────────┼──▶ Distill ──▶ Link ──▶ LanceDB
[Git commits] ──────────┘
```

### Components

| Component | Purpose |
|-----------|---------|
| Watcher | Monitor sources (sessions, files, git) for changes |
| Distiller | Extract structured nodes via LLM |
| Linker | Create edges via embedding similarity + explicit refs |
| Store | LanceDB persistence (nodes + edges tables) |

## Module Structure

```
src/
├── main.rs           # CLI entry point
├── lib.rs            # Core library exports
├── core/
│   ├── mod.rs
│   ├── node.rs       # Node types (EMBEDDING_DIM = 1536)
│   └── edge.rs       # Edge types
├── store/
│   ├── mod.rs
│   ├── lance.rs      # LanceDB wrapper
│   └── schema.rs     # Arrow schemas
├── distill.rs        # LLM knowledge extraction
├── embed.rs          # OpenAI embeddings (text-embedding-3-small)
├── llm.rs            # Claude CLI wrapper
├── session.rs        # Session discovery
├── transcript/
│   ├── mod.rs
│   ├── reader.rs     # JSONL parsing, context formatting
│   └── types.rs      # Transcript entry types
└── watch/
    ├── mod.rs
    └── session.rs    # File watcher for sessions
```

## Storage

```
~/.engram/
└── db/               # LanceDB database
    ├── nodes.lance/
    ├── edges.lance/
    └── processed.lance/
```

## Design Principles

1. **Background first**: Runs as daemon, not CLI tool
2. **Incremental**: Only process changed sources
3. **Local**: All data stays on machine
4. **Composable**: MCP server for external queries (later)
