"""LanceDB MCP server for semantic code search in Claude Code CLI.

Exposes seven tools over stdio transport:
  - search_code:     hybrid vector + full-text search with metadata filters
  - index_files:     index or re-index source files into LanceDB
  - index_status:    check index health and stats
  - remove_files:    remove deleted files from the index
  - switch_project:  switch to (or create) a project context
  - list_projects:   list all registered projects
  - remove_project:  unregister a project (optionally drop its table)
"""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import lancedb as ldb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import RRFReranker
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

from config import VALID_NODE_TYPES, Config
from errors import ProjectError
from indexer import IndexResult
from indexer import index_files as do_index_files
from indexer import remove_files as do_remove_files
from projects import (
    DEFAULT_PROJECT_NAME,
    DEFAULT_TABLE_NAME,
    ProjectState,
    check_repo_root,
    create_project,
    load_registry,
    save_registry,
    validate_project_name,
)

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
    config: Config
    schema: type
    projects: dict[str, ProjectState]       # name → state
    active_project: str | None
    tables: dict[str, ldb.table.Table] = field(default_factory=dict)  # lazy cache


def _open_or_create_table(
    db: ldb.DBConnection, table_name: str, schema: type,
) -> ldb.table.Table:
    """Open an existing LanceDB table or create a new one.

    Uses open-first strategy because ``db.list_tables()`` can miss tables
    that exist on disk (observed with persistent Docker volumes across runs).
    """
    try:
        table = db.open_table(table_name)
        logger.info("Opened existing table '%s'", table_name)
    except Exception:
        table = db.create_table(table_name, schema=schema)
        logger.info("Created new table '%s'", table_name)
    return table


def _resolve_table(
    app: AppContext, project: str | None = None,
) -> tuple[ldb.table.Table, ProjectState]:
    """Resolve the LanceDB table and project state for the given project.

    If *project* is ``None``, the active project is used.  Raises
    ``ProjectError`` if no project can be determined or if the requested
    project is not registered.
    """
    name = project or app.active_project
    if not name:
        raise ProjectError(
            "No active project. Use switch_project to select one.",
        )

    proj = app.projects.get(name)
    if proj is None:
        registered = ", ".join(sorted(app.projects)) or "(none)"
        raise ProjectError(
            f"Project '{name}' not found. Registered projects: {registered}",
            context={"project": name},
        )

    # Lazy-open the table.
    if name not in app.tables:
        app.tables[name] = _open_or_create_table(
            app.db, proj.table_name, app.schema,
        )

    return app.tables[name], proj


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Initialize the LanceDB connection, load project registry, yield context."""
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

    # Load project registry (or bootstrap from legacy state).
    projects = load_registry(config.db_full_path)

    if not projects:
        # Legacy fallback: if the old default table exists, adopt it.
        table_names = db.list_tables()
        if DEFAULT_TABLE_NAME in table_names:
            logger.info(
                "No project registry found; adopting existing '%s' table "
                "as project '%s'",
                DEFAULT_TABLE_NAME, DEFAULT_PROJECT_NAME,
            )
            from datetime import datetime, timezone
            projects[DEFAULT_PROJECT_NAME] = ProjectState(
                name=DEFAULT_PROJECT_NAME,
                repo_root=str(config.repo_root_path),
                table_name=DEFAULT_TABLE_NAME,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            save_registry(config.db_full_path, projects)
        else:
            # No existing table — create the default project.
            projects[DEFAULT_PROJECT_NAME] = ProjectState(
                name=DEFAULT_PROJECT_NAME,
                repo_root=str(config.repo_root_path),
                table_name=DEFAULT_TABLE_NAME,
                created_at=__import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
            )
            save_registry(config.db_full_path, projects)

    # Determine active project.
    if config.active_project:
        try:
            validate_project_name(config.active_project)
            active = config.active_project
            if active not in projects:
                logger.warning(
                    "LANCEDB_ACTIVE_PROJECT='%s' is not in the registry yet; "
                    "it will become active once created via switch_project",
                    active,
                )
        except ProjectError:
            logger.warning(
                "LANCEDB_ACTIVE_PROJECT='%s' is not a valid project name; "
                "falling back to default selection",
                config.active_project,
            )
            active = (
                DEFAULT_PROJECT_NAME if DEFAULT_PROJECT_NAME in projects
                else (next(iter(projects)) if projects else None)
            )
    elif DEFAULT_PROJECT_NAME in projects:
        active = DEFAULT_PROJECT_NAME
    else:
        active = next(iter(projects)) if projects else None

    logger.info(
        "Loaded %d project(s); active: %s",
        len(projects), active,
    )

    app = AppContext(
        db=db,
        config=config,
        schema=schema,
        projects=projects,
        active_project=active,
    )

    try:
        yield app
    finally:
        # Compact and clean up all open tables on shutdown.
        for tname, tbl in app.tables.items():
            try:
                tbl.optimize()
                logger.info("Table '%s' optimized on shutdown", tname)
            except Exception as exc:
                logger.warning("optimize() failed for '%s' on shutdown: %s", tname, exc)


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "lancedb-code",
    instructions=(
        "Semantic code search server backed by LanceDB. "
        "Typical workflow: (1) call index_files to build/update the index, "
        "(2) call search_code with a natural language query to find relevant code, "
        "(3) call index_files with specific paths after editing files. "
        "For multi-repo setups: use switch_project to create/switch between "
        "project contexts, each with its own isolated index and repo root. "
        "Use list_projects to see all registered projects and index_status "
        "to check if an index needs rebuilding."
    ),
    lifespan=app_lifespan,
)


# ---------------------------------------------------------------------------
# Tool: switch_project
# ---------------------------------------------------------------------------


@mcp.tool()
def switch_project(
    project: str,
    repo_root: str | None = None,
    ctx: Context[ServerSession, AppContext] = None,
) -> str:
    """Switch to an existing project or create a new one.

    Each project has its own isolated search index and repo root directory.
    Use this to work across multiple repositories without running separate servers.

    To create a new project, both ``project`` and ``repo_root`` are required.
    To switch to an existing project, only ``project`` is needed.
    Optionally pass ``repo_root`` when switching to update the repo path.

    Examples:
        - Create: switch_project("backend", "/home/user/backend-repo")
        - Switch: switch_project("backend")
        - Update root: switch_project("backend", "/new/path/to/backend")

    Args:
        project: Project name (start with a letter, 1-63 chars, letters/digits/underscore/hyphen).
        repo_root: Absolute path to the repository root. Required when creating a new project.
    """
    app: AppContext = ctx.request_context.lifespan_context

    try:
        if project in app.projects:
            # Existing project — switch to it.
            if repo_root is not None:
                resolved = str(Path(repo_root).resolve())
                app.projects[project].repo_root = resolved
                save_registry(app.config.db_full_path, app.projects)
            app.active_project = project
            proj = app.projects[project]
            msg = (
                f"Switched to project '{project}'.\n"
                f"  repo_root: {proj.repo_root}\n"
                f"  table: {proj.table_name}"
            )
            warning = check_repo_root(proj.repo_root)
            if warning:
                msg += f"\n  {warning}"
            return msg

        # New project — repo_root is required.
        if repo_root is None:
            return (
                f"Project '{project}' does not exist. "
                f"Provide repo_root to create it."
            )

        proj = create_project(project, repo_root)
        app.projects[project] = proj

        # Eagerly open/create the table so it's ready.
        app.tables[project] = _open_or_create_table(
            app.db, proj.table_name, app.schema,
        )

        save_registry(app.config.db_full_path, app.projects)
        app.active_project = project

        msg = (
            f"Created and switched to project '{project}'.\n"
            f"  repo_root: {proj.repo_root}\n"
            f"  table: {proj.table_name}"
        )
        warning = check_repo_root(proj.repo_root)
        if warning:
            msg += f"\n  {warning}"
        return msg
    except ProjectError as exc:
        return f"Project error: {exc}"


# ---------------------------------------------------------------------------
# Tool: list_projects
# ---------------------------------------------------------------------------


@mcp.tool()
def list_projects(
    ctx: Context[ServerSession, AppContext] = None,
) -> str:
    """List all registered projects with their details.

    Returns each project's name, repo root path, LanceDB table name, and
    creation timestamp. The currently active project is marked with an
    asterisk (*). Use switch_project to change the active project or
    create a new one.
    """
    app: AppContext = ctx.request_context.lifespan_context

    if not app.projects:
        return "No projects registered. Use switch_project to create one."

    lines: list[str] = [f"Registered projects ({len(app.projects)}):\n"]
    for name in sorted(app.projects):
        proj = app.projects[name]
        marker = " *" if name == app.active_project else ""
        lines.append(
            f"  {name}{marker}\n"
            f"    repo_root:  {proj.repo_root}\n"
            f"    table:      {proj.table_name}\n"
            f"    created_at: {proj.created_at}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: remove_project
# ---------------------------------------------------------------------------


@mcp.tool()
def remove_project(
    project: str,
    drop_table: bool = False,
    ctx: Context[ServerSession, AppContext] = None,
) -> str:
    """Remove a project from the registry and optionally delete its index data.

    By default, only the registry entry is removed and the LanceDB table is
    kept on disk. Set ``drop_table=True`` to permanently delete the table and
    all indexed chunks. If the removed project was active, the next available
    project becomes active automatically.

    Args:
        project: Name of the project to remove (must be a registered project).
        drop_table: If True, also drop the LanceDB table and all its indexed data.
    """
    app: AppContext = ctx.request_context.lifespan_context

    if project not in app.projects:
        registered = ", ".join(sorted(app.projects)) or "(none)"
        return f"Project '{project}' not found. Registered: {registered}"

    proj = app.projects.pop(project)

    # Close cached table handle.
    app.tables.pop(project, None)

    if drop_table:
        try:
            app.db.drop_table(proj.table_name)
            logger.info("Dropped table '%s' for project '%s'", proj.table_name, project)
        except Exception as exc:
            logger.warning(
                "Failed to drop table '%s' for project '%s': %s",
                proj.table_name, project, exc,
            )

    save_registry(app.config.db_full_path, app.projects)

    # Update active project.
    if app.active_project == project:
        if app.projects:
            app.active_project = (
                DEFAULT_PROJECT_NAME
                if DEFAULT_PROJECT_NAME in app.projects
                else next(iter(app.projects))
            )
        else:
            app.active_project = None

    drop_msg = " Table dropped." if drop_table else ""
    active_msg = (
        f" Active project: {app.active_project}"
        if app.active_project
        else " No active project."
    )
    return f"Removed project '{project}'.{drop_msg}{active_msg}"


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
    project: str | None = None,
    ctx: Context[ServerSession, AppContext] = None,
) -> str:
    """Search the codebase semantically. Returns ranked code snippets with file locations.

    Use this to find relevant code by meaning rather than exact string matching.
    Supports hybrid (vector + keyword), pure vector, or pure full-text search.
    Results include file path, line range, node type, symbol name, and a truncated
    snippet so you can decide what to read in full.

    Run index_files first if the index is empty or stale.

    Args:
        query: Natural language description of what you're looking for.
        limit: Maximum number of results to return (default 10).
        language: Filter by programming language (e.g. "python", "typescript", "rust").
        file_path_pattern: Filter by file path prefix (e.g. "src/auth/").
        node_type: Filter by code structure type: function, class, method, module, block.
        query_type: Search mode — "hybrid" (default, best quality), "vector" (semantic only), or "fts" (keyword only).
        project: Project to search in. Defaults to the active project.
    """
    app: AppContext = ctx.request_context.lifespan_context

    try:
        table, _proj = _resolve_table(app, project)
    except ProjectError as exc:
        return f"Project error: {exc}"

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
        search = table.search(query, query_type=query_type)

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
    project: str | None = None,
    ctx: Context[ServerSession, AppContext] = None,
) -> str:
    """Index source files into the search database.

    Call with no arguments to index the entire repository. Pass specific file
    paths for incremental updates after editing files. Unchanged files are
    automatically skipped via content-hash detection unless ``force=True``.

    Supports Python, JavaScript, TypeScript, Rust, Go, and more via Tree-sitter
    syntax-aware chunking. Other text formats use line-based fallback chunking.

    Args:
        paths: File paths to index. None means scan the full repository.
        force: Re-index files even if unchanged (default False).
        project: Project to index into. Defaults to the active project.
    """
    app: AppContext = ctx.request_context.lifespan_context

    try:
        table, proj = _resolve_table(app, project)
    except ProjectError as exc:
        return f"Project error: {exc}"

    try:
        result: IndexResult = do_index_files(
            table=table,
            config=app.config,
            paths=paths,
            force=force,
            repo_root=Path(proj.repo_root),
        )
    except Exception as exc:
        logger.error(
            "Indexing failed (paths=%r, force=%s): %s",
            paths, force, exc,
        )
        return f"Indexing error: {exc}"

    # Rebuild FTS index after ingestion.
    try:
        table.create_fts_index("text", replace=True)
    except Exception as exc:
        logger.warning("FTS index rebuild failed (non-fatal): %s", exc)

    return (
        f"Indexing complete for project '{proj.name}' in {result.duration_ms}ms:\n"
        f"  Files scanned: {result.files_scanned}\n"
        f"  Files indexed: {result.files_indexed}\n"
        f"  Files skipped (unchanged): {result.files_skipped}\n"
        f"  Chunks created: {result.chunks_created}"
    )


# ---------------------------------------------------------------------------
# Tool: index_status
# ---------------------------------------------------------------------------


@mcp.tool()
def index_status(
    project: str | None = None,
    ctx: Context[ServerSession, AppContext] = None,
) -> str:
    """Check the current state of the code search index.

    Returns the number of indexed files, chunks, languages, node type
    breakdown, and whether vector/FTS indices are present. Use this to
    determine if the index needs rebuilding or if index_files should be run.

    Args:
        project: Project to check. Defaults to the active project.
    """
    app: AppContext = ctx.request_context.lifespan_context

    try:
        table, proj = _resolve_table(app, project)
    except ProjectError as exc:
        return f"Project error: {exc}"

    try:
        arrow_table = table.to_arrow()
    except Exception as exc:
        logger.debug("to_arrow() failed (treating as empty index): %s", exc)
        return f"Index for project '{proj.name}' is empty. Run index_files to build it."

    if arrow_table.num_rows == 0:
        return f"Index for project '{proj.name}' is empty. Run index_files to build it."

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
        indices = table.list_indices()
    except Exception as exc:
        logger.debug("list_indices() failed (non-fatal): %s", exc)
    has_vector_idx = any("vector" in str(idx) for idx in indices)
    has_fts_idx = any("fts" in str(idx).lower() or "text" in str(idx) for idx in indices)

    return (
        f"Index status for project '{proj.name}':\n"
        f"  Repo root: {proj.repo_root}\n"
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
    project: str | None = None,
    ctx: Context[ServerSession, AppContext] = None,
) -> str:
    """Remove specific files from the search index.

    Use after deleting or renaming files to keep the index consistent.
    Removes all indexed chunks for the specified file paths. Paths can
    be absolute or relative to the project's repo root.

    Args:
        paths: File paths to remove from the index.
        project: Project to remove files from. Defaults to the active project.
    """
    app: AppContext = ctx.request_context.lifespan_context

    try:
        table, proj = _resolve_table(app, project)
    except ProjectError as exc:
        return f"Project error: {exc}"

    removed = do_remove_files(
        table, paths, app.config, repo_root=Path(proj.repo_root),
    )
    return f"Removed {removed} file(s) from the index (project '{proj.name}')."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
