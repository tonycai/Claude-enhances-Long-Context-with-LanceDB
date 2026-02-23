"""LanceDB MCP server for semantic code search in Claude Code CLI.

Exposes four tools over stdio transport:
  - search_code:   hybrid vector + full-text search with metadata filters
  - index_files:   index or re-index source files into LanceDB
  - index_status:  check index health and stats
  - remove_files:  remove deleted files from the index
"""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import lancedb as ldb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import RRFReranker
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

from config import VALID_NODE_TYPES, Config
from indexer import IndexResult
from indexer import index_files as do_index_files
from indexer import remove_files as do_remove_files

# ---------------------------------------------------------------------------
# Logging — must use stderr for stdio MCP servers.
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("lancedb-code-mcp")


# ---------------------------------------------------------------------------
# Application context & lifespan
# ---------------------------------------------------------------------------

config = Config()


def _build_schema(embedding_func):
    """Dynamically build a LanceModel schema with the configured embedding function."""
    ndims = embedding_func.ndims()

    class CodeChunk(LanceModel):
        text: str = embedding_func.SourceField()
        vector: Vector(ndims) = embedding_func.VectorField()
        chunk_id: str
        file_path: str
        start_line: int
        end_line: int
        language: str
        node_type: str
        symbol_name: str
        content_hash: str

    return CodeChunk


@dataclass
class AppContext:
    db: ldb.DBConnection
    table: ldb.table.Table
    config: Config
    schema: type


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Initialize the LanceDB connection and table on startup."""
    logger.info("Starting LanceDB MCP server")
    logger.info("  DB path: %s", config.db_full_path)
    logger.info("  Repo root: %s", config.repo_root_path)
    logger.info("  Embedding: %s / %s", config.embedding_provider, config.embedding_model)

    # Connect to LanceDB.
    db = ldb.connect(config.db_full_path)

    # Load embedding function from registry.
    registry = get_registry()
    embedding_func = registry.get(config.embedding_provider).create(
        name=config.embedding_model
    )
    schema = _build_schema(embedding_func)

    # Open or create table.
    table_names = db.list_tables()
    if config.table_name in table_names:
        table = db.open_table(config.table_name)
        logger.info("Opened existing table '%s'", config.table_name)
    else:
        table = db.create_table(config.table_name, schema=schema)
        logger.info("Created new table '%s'", config.table_name)

    try:
        yield AppContext(db=db, table=table, config=config, schema=schema)
    finally:
        # Compact and clean up on shutdown.
        try:
            table.optimize()
            logger.info("Table optimized on shutdown")
        except Exception as exc:
            logger.warning("optimize() failed on shutdown: %s", exc)


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "lancedb-code",
    instructions=(
        "Semantic code search server backed by LanceDB. "
        "Use search_code to find relevant code by meaning. "
        "Use index_files to build or update the search index. "
        "Use index_status to check if the index is current."
    ),
    lifespan=app_lifespan,
)


# ---------------------------------------------------------------------------
# Tool: search_code
# ---------------------------------------------------------------------------


def _sanitize_str(value: str) -> str:
    """Escape single quotes for safe use in LanceDB SQL filter expressions."""
    return value.replace("'", "''")


def _format_results(results: list[dict], query: str) -> str:
    """Format search results into a compact, Claude-friendly string."""
    if not results:
        return f'No results found for "{query}".'

    lines: list[str] = [f'Found {len(results)} results for "{query}":\n']
    for i, r in enumerate(results, 1):
        fp = r.get("file_path", "?")
        sl = r.get("start_line", "?")
        el = r.get("end_line", "?")
        ntype = r.get("node_type", "block")
        sym = r.get("symbol_name", "")
        lang = r.get("language", "")
        text = r.get("text", "")
        dist = r.get("_distance")
        score = r.get("_relevance_score") or r.get("_score") or r.get("score")

        # Truncate snippet for compact output.
        snippet = text[:200].rstrip()
        if len(text) > 200:
            snippet += "..."

        score_str = ""
        if score is not None:
            score_str = f" | score: {score:.3f}"
        elif dist is not None:
            score_str = f" | dist: {dist:.3f}"

        sym_str = f" {sym}" if sym else ""
        lines.append(
            f"{i}. {fp}:{sl}-{el} [{ntype}]{sym_str} ({lang}){score_str}\n"
            f"   {snippet}\n"
        )
    return "\n".join(lines)


@mcp.tool()
def search_code(
    query: str,
    limit: int = 10,
    language: str | None = None,
    file_path_pattern: str | None = None,
    node_type: str | None = None,
    query_type: str = "hybrid",
    ctx: Context[ServerSession, AppContext] = None,
) -> str:
    """Search the codebase semantically. Returns ranked code snippets with file locations.

    Use this to find relevant code by meaning rather than exact string matching.
    Supports hybrid (vector + keyword), pure vector, or pure full-text search.

    Args:
        query: Natural language description of what you're looking for.
        limit: Maximum number of results to return (default 10).
        language: Filter by programming language (e.g. "python", "typescript", "rust").
        file_path_pattern: Filter by file path prefix (e.g. "src/auth/").
        node_type: Filter by code structure type: function, class, method, module, block.
        query_type: Search mode — "hybrid" (default), "vector", or "fts".
    """
    app: AppContext = ctx.request_context.lifespan_context

    # Build filter expression.
    filters: list[str] = []
    if language:
        filters.append(f"language = '{_sanitize_str(language)}'")
    if file_path_pattern:
        filters.append(f"file_path LIKE '{_sanitize_str(file_path_pattern)}%'")
    if node_type:
        if node_type not in VALID_NODE_TYPES:
            return (
                f"Invalid node_type '{node_type}'. "
                f"Valid types: {', '.join(sorted(VALID_NODE_TYPES))}"
            )
        filters.append(f"node_type = '{_sanitize_str(node_type)}'")

    where_clause = " AND ".join(filters) if filters else None

    # Validate query_type.
    if query_type not in ("hybrid", "vector", "fts"):
        query_type = "hybrid"

    try:
        search = app.table.search(query, query_type=query_type)

        if where_clause:
            search = search.where(where_clause, prefilter=True)

        if query_type == "hybrid":
            search = search.rerank(reranker=RRFReranker())

        results = search.limit(limit).to_list()
    except Exception as exc:
        logger.error(
            "Search failed for query=%r, query_type=%s, where=%r: %s",
            query, query_type, where_clause, exc,
        )
        return f"Search error: {exc}"

    return _format_results(results, query)


# ---------------------------------------------------------------------------
# Tool: index_files
# ---------------------------------------------------------------------------


@mcp.tool()
def index_files(
    paths: list[str] | None = None,
    force: bool = False,
    ctx: Context[ServerSession, AppContext] = None,
) -> str:
    """Index source files into the search database.

    Call with no arguments to index the entire repository. Pass specific file
    paths for incremental updates after editing files.

    Args:
        paths: File paths to index. None means scan the full repository.
        force: Re-index files even if unchanged (default False).
    """
    app: AppContext = ctx.request_context.lifespan_context

    try:
        result: IndexResult = do_index_files(
            table=app.table, config=app.config, paths=paths, force=force
        )
    except Exception as exc:
        logger.error(
            "Indexing failed (paths=%r, force=%s): %s",
            paths, force, exc,
        )
        return f"Indexing error: {exc}"

    # Rebuild FTS index after ingestion.
    try:
        app.table.create_fts_index("text", replace=True)
    except Exception as exc:
        logger.warning("FTS index rebuild failed (non-fatal): %s", exc)

    return (
        f"Indexing complete in {result.duration_ms}ms:\n"
        f"  Files scanned: {result.files_scanned}\n"
        f"  Files indexed: {result.files_indexed}\n"
        f"  Files skipped (unchanged): {result.files_skipped}\n"
        f"  Chunks created: {result.chunks_created}"
    )


# ---------------------------------------------------------------------------
# Tool: index_status
# ---------------------------------------------------------------------------


@mcp.tool()
def index_status(ctx: Context[ServerSession, AppContext] = None) -> str:
    """Check the current state of the code search index.

    Returns the number of indexed files, chunks, languages, and whether the
    index is healthy. Use this to determine if the index needs rebuilding.
    """
    app: AppContext = ctx.request_context.lifespan_context

    try:
        arrow_table = app.table.to_arrow()
    except Exception as exc:
        logger.debug("to_arrow() failed (treating as empty index): %s", exc)
        return "Index is empty. Run index_files to build it."

    if arrow_table.num_rows == 0:
        return "Index is empty. Run index_files to build it."

    total_chunks = arrow_table.num_rows
    total_files = len(set(arrow_table.column("file_path").to_pylist()))

    lang_counts: dict[str, int] = {}
    if "language" in arrow_table.column_names:
        from collections import Counter
        lang_counts = dict(Counter(arrow_table.column("language").to_pylist()))

    node_counts: dict[str, int] = {}
    if "node_type" in arrow_table.column_names:
        from collections import Counter
        node_counts = dict(Counter(arrow_table.column("node_type").to_pylist()))

    lang_summary = ", ".join(f"{k}: {v}" for k, v in sorted(lang_counts.items()))
    node_summary = ", ".join(f"{k}: {v}" for k, v in sorted(node_counts.items()))

    # Check index health.
    indices = []
    try:
        indices = app.table.list_indices()
    except Exception as exc:
        logger.debug("list_indices() failed (non-fatal): %s", exc)
    has_vector_idx = any("vector" in str(idx) for idx in indices)
    has_fts_idx = any("fts" in str(idx).lower() or "text" in str(idx) for idx in indices)

    return (
        f"Index status:\n"
        f"  Total chunks: {total_chunks}\n"
        f"  Total files: {total_files}\n"
        f"  Languages: {lang_summary}\n"
        f"  Node types: {node_summary}\n"
        f"  Vector index: {'yes' if has_vector_idx else 'no (brute-force search)'}\n"
        f"  FTS index: {'yes' if has_fts_idx else 'no'}"
    )


# ---------------------------------------------------------------------------
# Tool: remove_files
# ---------------------------------------------------------------------------


@mcp.tool()
def remove_files(
    paths: list[str],
    ctx: Context[ServerSession, AppContext] = None,
) -> str:
    """Remove files from the search index.

    Use after deleting files to keep the index consistent.

    Args:
        paths: File paths to remove from the index.
    """
    app: AppContext = ctx.request_context.lifespan_context

    removed = do_remove_files(app.table, paths, app.config)
    return f"Removed {removed} file(s) from the index."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
