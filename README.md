# Claude Enhances Long Context with LanceDB

A LanceDB-powered MCP server that gives [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) semantic code search capabilities, overcoming context window limitations on large codebases.

## The Problem

Claude Code's context window is finite. On large repositories, reading every file to find relevant code is slow and expensive. Developers need a way to let Claude search by *meaning* — not just filenames or exact strings — so it can pinpoint the right code without exhausting its context.

## The Solution

This project provides an MCP (Model Context Protocol) server backed by [LanceDB](https://lancedb.com/) that:

1. **Indexes** your codebase into semantic chunks using Tree-sitter syntax parsing
2. **Embeds** those chunks using sentence-transformers (`all-MiniLM-L6-v2`)
3. **Searches** via hybrid vector + full-text search with Reciprocal Rank Fusion reranking
4. **Returns** compact results (file path, line range, snippet) so Claude can decide what to read in full

## Architecture

```
File Discovery → Tree-sitter Parsing → LanceDB Embedding → Hybrid Search
  (respects        (syntax-aware          (auto-embed via      (vector + FTS
   .gitignore)      chunking)              registry)            with RRF)
```

The server exposes 4 tools over MCP stdio transport:

| Tool | Purpose |
|------|---------|
| `search_code` | Semantic search with language/path/node_type filters |
| `index_files` | Full or incremental indexing with content-hash change detection |
| `index_status` | Check index health, chunk/file counts, languages |
| `remove_files` | Remove deleted files from the index |

## Supported Languages

**Tree-sitter syntax-aware chunking** (functions, classes, methods, etc.):
- Python, JavaScript, TypeScript, TSX, Rust, Go
- Java, C, C++, Ruby, C# *(optional, install with `--extra all-languages`)*

**Line-based fallback chunking**:
- Markdown, YAML, TOML, JSON, HTML, CSS, SCSS, Shell, SQL, GraphQL, Protobuf, Terraform, Dockerfiles

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Install

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

### Run the Integration Test

```bash
cd lancedb-mcp-server
uv run python test_integration.py
```

This exercises the full pipeline: file discovery, Tree-sitter chunking, LanceDB indexing, vector/FTS/hybrid search, and incremental re-indexing.

## Configuration

All settings are via environment variables:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LANCEDB_PATH` | `./.lancedb` | Database directory |
| `LANCEDB_REPO_ROOT` | `.` (cwd) | Repository root for file scanning |
| `LANCEDB_EMBEDDING_PROVIDER` | `sentence-transformers` | LanceDB registry key |
| `LANCEDB_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model name |
| `LANCEDB_TABLE_NAME` | `code_chunks` | LanceDB table name |

## Project Structure

```
├── lancedb-mcp-server/
│   ├── server.py              # FastMCP entry point, 4 tools
│   ├── chunker.py             # Tree-sitter + line-based chunking
│   ├── indexer.py             # File discovery, hashing, ingestion
│   ├── config.py              # Environment-based configuration
│   ├── test_integration.py    # End-to-end integration test
│   └── pyproject.toml         # Dependencies and build config
├── Docs/
│   └── Integrating-LanceDB-with-Claude-Code-CLI.md  # Architecture guide (36 citations)
└── LICENSE                    # Apache 2.0
```

## Documentation

See [`Docs/Integrating-LanceDB-with-Claude-Code-CLI.md`](Docs/Integrating-LanceDB-with-Claude-Code-CLI.md) for the full architectural guide covering embedding strategies, chunking approaches, storage backends, and search optimization — with 36 citations.

## License

Apache 2.0
