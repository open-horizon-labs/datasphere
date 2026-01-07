# Datasphere

> Named after the AI knowledge network in Dan Simmons' *Hyperion Cantos* — a vast repository where all information exists and can be accessed.

Background daemon that distills knowledge from Claude Code sessions into a queryable knowledge graph.

## What It Does

Datasphere watches your Claude Code sessions, extracts insights via LLM distillation, and embeds them for semantic search. Think of it as long-term memory for your AI coding sessions.

```
┌─────────────────┐     ┌──────────┐     ┌─────────┐     ┌─────────┐
│ Session JSONL   │────▶│ Distill  │────▶│  Embed  │────▶│ LanceDB │
│ (~/.claude/     │     │  (LLM)   │     │         │     │ (nodes) │
│   projects/)    │     └──────────┘     └─────────┘     └─────────┘
└─────────────────┘
```

## Installation

```bash
# Build and install CLI
cargo install --path .

# Verify
ds --help
```

## Quick Start

```bash
# One-shot scan of current project's sessions
ds scan

# Start daemon (watches ALL projects continuously)
ds start

# Query the knowledge graph
ds query "how to chunk large texts"

# Check database stats
ds stats
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `ds scan` | One-shot distillation of sessions (current project) |
| `ds start` | Daemon mode - watches all projects, queues jobs |
| `ds query <text>` | Semantic search of knowledge graph |
| `ds queue` | Show job queue status |
| `ds queue pending` | List pending jobs |
| `ds queue clear` | Clear completed jobs |
| `ds queue nuke` | Delete all jobs |
| `ds stats` | Show database statistics |
| `ds show` | Display stored nodes |
| `ds add <file>` | Add a text file to the graph (no LLM distillation) |
| `ds related <id>` | Find nodes similar to a given node |
| `ds reset` | Delete database and queue (fresh start) |

## MCP Server

The `mcp/` directory contains a minimal MCP server for Claude Code integration:

```bash
# Install dependencies
cd mcp && npm install

# Add to Claude Code
claude mcp add datasphere -s user -- node /path/to/datasphere/mcp/index.js
```

This exposes `datasphere_query` and `datasphere_related` tools that Claude can use to search your knowledge graph during conversations.

## How It Works

### 1. Watching
The daemon watches `~/.claude/projects/` for changes to session transcript files (`.jsonl`). Events are queued and processed one at a time with rate limiting.

### 2. Distillation
Each session is chunked and sent to an LLM for knowledge extraction. The LLM identifies:
- Key insights and learnings
- Patterns and approaches
- Decisions and their rationale

### 3. Embedding
Extracted insights are embedded (1536 dimensions) for semantic search.

### 4. Change Detection
SimHash is used to detect meaningful changes. Sessions are only re-processed when content changes significantly (Hamming distance > 10 bits).

## Storage

```
~/.datasphere/
├── db/                    # LanceDB database
│   ├── nodes.lance/       # Knowledge nodes with embeddings
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
