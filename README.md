# Engram

Background daemon that distills and links knowledge from Claude Code sessions into a queryable knowledge graph.

## What It Does

Engram watches your Claude Code sessions, extracts insights via LLM distillation, embeds them for semantic search, and links related concepts together. Think of it as long-term memory for your AI coding sessions.

```
┌─────────────────┐     ┌──────────┐     ┌─────────┐     ┌─────────┐
│ Session JSONL   │────▶│ Distill  │────▶│  Embed  │────▶│ LanceDB │
│ (~/.claude/     │     │ (Claude) │     │ (OpenAI)│     │         │
│   projects/)    │     └──────────┘     └─────────┘     └─────────┘
└─────────────────┘                            │              │
                                               │   ┌──────────┘
                                               ▼   ▼
                                         ┌───────────┐
                                         │   Link    │
                                         │(similarity│
                                         │  > 0.6)   │
                                         └───────────┘
```

## Installation

```bash
# Build and install CLI
cargo install --path .

# Verify
engram --help
```

## Quick Start

```bash
# One-shot scan of current project's sessions
engram scan

# Start daemon (watches ALL projects continuously)
engram start

# Query the knowledge graph
engram query "how to chunk large texts"

# Check database stats
engram stats
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `engram scan` | One-shot distillation of sessions (current project) |
| `engram start` | Daemon mode - watches all projects, queues jobs |
| `engram query <text>` | Semantic search of knowledge graph |
| `engram queue` | Show job queue status |
| `engram queue pending` | List pending jobs |
| `engram queue clear` | Clear completed jobs |
| `engram stats` | Show database statistics |
| `engram show` | Display stored nodes |
| `engram add <file>` | Add a text file to the graph (no LLM distillation) |

## MCP Server

The `mcp/` directory contains a minimal MCP server for Claude Code integration:

```bash
# Install dependencies
cd mcp && npm install

# Add to Claude Code
claude mcp add engram -s user -- node /path/to/engram/mcp/index.js
```

This exposes `engram_query` tool that Claude can use to search your knowledge graph during conversations.

## How It Works

### 1. Watching
The daemon watches `~/.claude/projects/` for changes to session transcript files (`.jsonl`). Events are queued and processed one at a time with rate limiting.

### 2. Distillation
Each session is chunked and sent to Claude for knowledge extraction. The LLM identifies:
- Key insights and learnings
- Patterns and approaches
- Decisions and their rationale

### 3. Embedding
Extracted insights are embedded using OpenAI's `text-embedding-3-small` (1536 dimensions) for semantic search.

### 4. Linking
Nodes are linked to similar existing nodes when cosine similarity > 0.6. This creates a graph where related concepts cluster together.

### 5. Change Detection
SimHash is used to detect meaningful changes. Sessions are only re-processed when content changes significantly (Hamming distance > 10 bits).

## Storage

```
~/.engram/
├── db/                    # LanceDB database
│   ├── nodes.lance/       # Knowledge nodes with embeddings
│   ├── edges.lance/       # Similarity links between nodes
│   └── processed.lance/   # Processing records (deduplication)
└── queue.jsonl            # Persistent job queue
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Required for embeddings |
| `ANTHROPIC_API_KEY` | Required for distillation (or use `claude` CLI auth) |

## Configuration

Currently configuration is via code constants:
- Similarity threshold: 0.6
- Max similar search: 20 nodes
- Job processing delay: 500ms
- SimHash change threshold: 10 bits

## Development

```bash
cargo build              # Dev build
cargo test               # Run tests
cargo build --release    # Release build
```

## Architecture

See [CLAUDE.md](./CLAUDE.md) for detailed module structure and design principles.
