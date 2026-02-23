# Claude Enhances Long Context with LanceDB

A LanceDB-powered MCP server that gives [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) semantic code search capabilities, overcoming context window limitations on large codebases.

## The Problem

Claude Code's context window is finite. On large repositories, reading every file to find relevant code is slow and expensive. Developers need a way to let Claude search by *meaning* — not just filenames or exact strings — so it can pinpoint the right code without exhausting its context.

## The Solution

This project provides an MCP (Model Context Protocol) server backed by [LanceDB](https://lancedb.com/) that:

1. **Indexes** your codebase into semantic chunks using Tree-sitter syntax parsing
2. **Embeds** those chunks using sentence-transformers (`all-MiniLM-L6-v2`, 22M params, 384 dims)
3. **Searches** via hybrid vector + full-text search with Reciprocal Rank Fusion reranking
4. **Returns** compact results (file path, line range, node type, symbol name, truncated snippet) so Claude can decide what to read in full

## Architecture

```
File Discovery → Tree-sitter Parsing → LanceDB Embedding → Hybrid Search
  (respects        (syntax-aware          (auto-embed via      (vector + FTS
   .gitignore)      chunking)              registry)            with RRF)
```

### Pipeline Details

**File Discovery** (`indexer.py`): Walks the repository respecting `.gitignore` rules via `pathspec`. Skips sensitive files (`.env`, keys, certs), large files (>1 MB), and standard build/cache directories (`node_modules`, `__pycache__`, `.venv`, `target`, `dist`, etc.). Supports path-traversal protection for user-provided paths.

**Chunking** (`chunker.py`): For supported languages, Tree-sitter parses source files into an AST and extracts top-level entities (functions, classes, interfaces, structs, enums, traits, impl blocks). Class bodies are decomposed into individual method chunks for granular search. Module-level code not covered by extracted entities is captured separately. Oversized chunks (>2000 chars / ~500 tokens) are split by lines with part numbering. For non-parsed languages, a line-based fallback (50 lines with 5-line overlap) is used.

**Embedding**: LanceDB's embedding registry auto-embeds the `text` field using `sentence-transformers/all-MiniLM-L6-v2`. This model was benchmarked against `all-mpnet-base-v2` (110M) and `BAAI/bge-base-en-v1.5` (109M) and selected for best signal-to-noise discrimination on code search queries and fastest throughput (~2300 sentences/sec on CPU).

**Search** (`server.py`): Supports three modes — `hybrid` (default, vector + FTS with RRF reranking), `vector` (pure semantic), and `fts` (pure keyword). Metadata filters (language, file path prefix, node type) are applied as prefilters. Results include file path, line range, node type, symbol name, relevance score, and a truncated snippet (200 chars).

**Change Detection** (`indexer.py`): SHA-256 content hashing per file. On re-index, unchanged files are skipped. Changed files have their old chunks deleted before new chunks are inserted.

### MCP Tools

The server exposes 4 tools over MCP stdio transport:

| Tool | Purpose |
|------|---------|
| `search_code` | Hybrid vector+FTS search with language, file path, and node_type filters |
| `index_files` | Full or incremental indexing with SHA-256 content-hash change detection |
| `index_status` | Check index health — chunk/file counts, languages, node types, vector/FTS index status |
| `remove_files` | Remove deleted files from the index to keep it consistent |

### Server Lifecycle

On startup, the FastMCP server connects to LanceDB, loads the embedding function from the registry, and opens or creates the `code_chunks` table. On shutdown, the table is compacted via `table.optimize()`. Logging goes to stderr (required for stdio MCP servers).

## Supported Languages

**Tree-sitter syntax-aware chunking** (extracts functions, classes, methods, interfaces, structs, enums, traits, impl blocks):
- Python, JavaScript, TypeScript, TSX, Rust, Go
- Java, C, C++, Ruby, C# *(optional, install with `--extra all-languages`)*

**Line-based fallback chunking** (50 lines, 5-line overlap):
- Markdown, YAML, TOML, JSON, HTML, CSS, SCSS, Shell, SQL, GraphQL, Protobuf, Terraform, Dockerfiles

**Always skipped**:
- Sensitive files: `.env`, `*.pem`, `*.key`, `*.crt`, SSH keys
- Large files: >1 MB
- Build directories: `node_modules`, `__pycache__`, `.venv`, `target`, `build`, `dist`, `vendor`, etc.

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- Or: Docker (no Python/uv required)

### Install (Native)

```bash
cd lancedb-mcp-server
uv sync

# Optional: add Java, C/C++, Ruby, C# grammars
uv sync --extra all-languages
```

### Configure Claude Code

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "lancedb-code": {
      "command": "uv",
      "args": ["--directory", "./lancedb-mcp-server", "run", "server.py"]
    }
  }
}
```

### Usage

Once configured, Claude Code automatically has access to the search tools. A typical workflow:

1. **Index your repo**: Claude calls `index_files` to build the search index
2. **Search by meaning**: Claude calls `search_code` with natural language queries
3. **Incremental updates**: After edits, Claude calls `index_files` with changed paths

### Docker

Build and run without installing Python, uv, or any native dependencies on your host.

**Image details:**
- Multi-stage build: builder (Python 3.12 + uv + all deps) → runtime (Python 3.12-slim + git + uv)
- `UV_TORCH_BACKEND=cpu` avoids CUDA libraries, keeping the image at ~2.3 GB
- All tree-sitter grammars included (including optional languages)
- Embedding model pre-downloaded and baked into the image via `scripts/prefetch_model.py`
- LanceDB index persists across container restarts via a named volume

```bash
# Build the image
cd lancedb-mcp-server && docker compose build

# Test that stdio transport responds to MCP initialize
echo '{"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1.0"}},"id":1}' \
  | docker run -i --rm \
      -v "$(pwd)/..:/workspace:ro" \
      -v lancedb-data:/data/lancedb \
      lancedb-code-mcp:latest
```

Configure Claude Code to use the Docker image in your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "lancedb-code": {
      "command": "bash",
      "args": [
        "-c",
        "docker run -i --rm -v \"$(pwd):/workspace:ro\" -v lancedb-data:/data/lancedb lancedb-code-mcp:latest"
      ]
    }
  }
}
```

**Volume mounts:**

| Mount | Container Path | Purpose |
|-------|---------------|---------|
| Source repo | `/workspace` (read-only bind) | Codebase to index |
| LanceDB data | `/data/lancedb` (named volume) | Persistent search index |

> **Note:** Docker on macOS runs in a Linux VM and cannot access Apple Silicon GPUs (MPS/Metal). For the default model (`all-MiniLM-L6-v2`, 22M params), CPU is faster than MPS due to the model's small size, so this is not a limitation in practice.

### Run the Integration Test

```bash
cd lancedb-mcp-server
uv run python test_integration.py
```

The integration test exercises the full pipeline in 7 steps: file discovery, Tree-sitter chunking, LanceDB indexing, FTS index creation, vector/FTS/hybrid search, incremental re-indexing (verifies unchanged files are skipped), and table statistics. The test database is automatically cleaned up.

## Configuration

All settings are via environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LANCEDB_PATH` | `./.lancedb` | Database directory (relative to repo root if not absolute) |
| `LANCEDB_REPO_ROOT` | `.` (cwd) | Repository root for file scanning |
| `LANCEDB_EMBEDDING_PROVIDER` | `sentence-transformers` | LanceDB embedding registry key |
| `LANCEDB_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model name (22M params, 384 dims) |
| `LANCEDB_TABLE_NAME` | `code_chunks` | LanceDB table name |

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `lancedb` >=0.21.2 | Vector database with auto-embedding and hybrid search |
| `mcp[cli]` >=1.6.0 | Model Context Protocol server framework (FastMCP) |
| `sentence-transformers` >=4.0.0 | Embedding model (`all-MiniLM-L6-v2`) |
| `tree-sitter` >=0.23.0 | Syntax-aware code parsing |
| `pathspec` >=0.12.0 | `.gitignore`-compatible file matching |

## Agent Team (Claude Agent SDK)

A team of specialized agents built with the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) that provides a programmatic interface to the LanceDB code search tools.

### Architecture

```
User Query
    │
    ▼
Orchestrator (sonnet)
    ├── Task → Indexer Agent (haiku)      — index_files, index_status, remove_files
    ├── Task → Searcher Agent (sonnet)    — search_code, Read, Grep, Glob
    ├── Task → Reviewer Agent (opus)      — search_code, Read, Grep, Glob
    └── Task → Q&A Agent (sonnet)         — search_code, index_status, Read, Grep, Glob
```

The orchestrator routes user queries to the appropriate specialist agent via the `Task` tool. Each agent has focused prompts and access to specific MCP tools from the lancedb-code server.

| Agent | Role | Model |
|-------|------|-------|
| Orchestrator | Routes queries to specialists | Sonnet |
| Indexer | Full/incremental indexing, status, cleanup | Haiku |
| Searcher | Semantic code search with filters | Sonnet |
| Reviewer | Code quality and security review | Opus |
| Q&A | Codebase explanations and architecture | Sonnet |

### Install

```bash
cd agents && uv sync
```

### Usage

```bash
# Requires ANTHROPIC_API_KEY
cd agents

# Index the repository
uv run python orchestrator.py "Index the repository"

# Search for code
uv run python orchestrator.py "Find all error handling code"

# Code review
uv run python orchestrator.py "Review server.py for security issues"

# Codebase Q&A
uv run python orchestrator.py "How does the chunking pipeline work?"
```

### Run Validation Tests

```bash
cd agents && uv run python test_agents.py
```

## Project Structure

```
├── lancedb-mcp-server/
│   ├── server.py              # FastMCP entry point, 4 tools, lifespan management
│   ├── chunker.py             # Tree-sitter syntax-aware chunking + line-based fallback
│   ├── indexer.py             # File discovery, content-hash change detection, LanceDB ingestion
│   ├── config.py              # Environment-based configuration and constants
│   ├── test_integration.py    # End-to-end integration test (7-step pipeline)
│   ├── pyproject.toml         # Dependencies and build config
│   ├── uv.lock                # Locked dependency versions
│   ├── Dockerfile             # Multi-stage Docker build (CPU-only torch, ~2.3 GB image)
│   ├── docker-compose.yml     # Build orchestration + volume definitions
│   ├── .dockerignore          # Excludes .venv, __pycache__, .lancedb from build context
│   └── scripts/
│       └── prefetch_model.py  # Pre-download embedding model at build time
├── agents/
│   ├── orchestrator.py        # Entry point: query routing to specialist agents
│   ├── config.py              # Shared config: models, paths, MCP server, prompt loader
│   ├── test_agents.py         # Validation tests (imports, config, agent definitions)
│   ├── pyproject.toml         # Dependencies (claude-agent-sdk)
│   ├── agents/                # Agent definitions package
│   │   ├── __init__.py        # Exports ALL_AGENTS dict
│   │   ├── indexer.py         # Indexer agent definition
│   │   ├── searcher.py        # Searcher agent definition
│   │   ├── reviewer.py        # Reviewer agent definition
│   │   └── qa.py              # Q&A agent definition
│   └── prompts/               # System prompts (Markdown, editable without code changes)
│       ├── orchestrator.md
│       ├── indexer.md
│       ├── searcher.md
│       ├── reviewer.md
│       └── qa.md
├── Docs/
│   └── Integrating-LanceDB-with-Claude-Code-CLI.md  # Architecture guide (36 citations)
├── CLAUDE.md                  # Claude Code project guidance (gitignored)
└── LICENSE                    # Apache 2.0
```

## Documentation

See [`Docs/Integrating-LanceDB-with-Claude-Code-CLI.md`](Docs/Integrating-LanceDB-with-Claude-Code-CLI.md) for the full architectural guide covering embedding strategies, chunking approaches, storage backends, and search optimization — with 36 citations.

## License

Apache 2.0
