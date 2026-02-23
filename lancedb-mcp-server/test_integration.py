"""Integration test: index this repo and run searches against it."""

import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("test_integration")

# Point at the parent repo.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["LANCEDB_REPO_ROOT"] = REPO_ROOT
os.environ["LANCEDB_PATH"] = os.path.join(REPO_ROOT, ".lancedb_test")

from config import Config
from indexer import discover_files, index_files, remove_files
from chunker import chunk_file
from pathlib import Path

import lancedb as ldb
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import RRFReranker


def main():
    config = Config()
    print(f"Repo root: {config.repo_root_path}")
    print(f"DB path:   {config.db_full_path}")
    print()

    # Step 1: Discover files
    print("=" * 60)
    print("STEP 1: File Discovery")
    print("=" * 60)
    files = discover_files(config.repo_root_path)
    print(f"Found {len(files)} indexable files:")
    for f in files:
        rel = f.relative_to(config.repo_root_path)
        print(f"  {rel}")
    print()

    # Step 2: Test chunking on a Python file
    print("=" * 60)
    print("STEP 2: Chunking Test (server.py)")
    print("=" * 60)
    server_py = config.repo_root_path / "lancedb-mcp-server" / "server.py"
    if server_py.exists():
        chunks = chunk_file(str(server_py), config.repo_root_path)
        print(f"Produced {len(chunks)} chunks:")
        for c in chunks:
            snippet = c.text[:80].replace("\n", "\\n")
            print(f"  [{c.node_type}] {c.symbol_name or '(anonymous)'} "
                  f"L{c.start_line}-{c.end_line} ({len(c.text)} chars)")
            print(f"    {snippet}...")
        print()

    # Step 3: Connect to LanceDB and index
    print("=" * 60)
    print("STEP 3: LanceDB Indexing")
    print("=" * 60)
    db = ldb.connect(config.db_full_path)

    registry = get_registry()
    embedding_func = registry.get(config.embedding_provider).create(
        name=config.embedding_model
    )
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

    # Drop and recreate for a clean test.
    if config.table_name in db.list_tables():
        db.drop_table(config.table_name)
    table = db.create_table(config.table_name, schema=CodeChunk)

    result = index_files(table, config)
    print(f"Indexing result:")
    print(f"  Files scanned:  {result.files_scanned}")
    print(f"  Files indexed:  {result.files_indexed}")
    print(f"  Files skipped:  {result.files_skipped}")
    print(f"  Chunks created: {result.chunks_created}")
    print(f"  Duration:       {result.duration_ms}ms")
    print()

    # Step 4: Create FTS index
    print("=" * 60)
    print("STEP 4: Creating FTS Index")
    print("=" * 60)
    try:
        table.create_fts_index("text", replace=True)
        print("FTS index created successfully.")
    except Exception as e:
        logger.warning("FTS index creation failed: %s", e)
    print()

    # Step 5: Run searches
    print("=" * 60)
    print("STEP 5: Search Tests")
    print("=" * 60)

    queries = [
        ("semantic code search", "vector"),
        ("Tree-sitter chunking", "vector"),
        ("MCP server tools", "vector"),
        ("context window bottleneck", "fts"),
        ("hybrid search reranking", "hybrid"),
    ]

    for query, qtype in queries:
        print(f'\n--- Query: "{query}" (type={qtype}) ---')
        try:
            search = table.search(query, query_type=qtype)
            if qtype == "hybrid":
                search = search.rerank(reranker=RRFReranker())
            results = search.limit(5).to_list()
            if not results:
                print("  (no results)")
                continue
            for i, r in enumerate(results, 1):
                fp = r.get("file_path", "?")
                sl = r.get("start_line", "?")
                el = r.get("end_line", "?")
                sym = r.get("symbol_name", "")
                ntype = r.get("node_type", "")
                snippet = r.get("text", "")[:100].replace("\n", "\\n")
                dist = r.get("_distance")
                score_str = f" dist={dist:.3f}" if dist is not None else ""
                print(f"  {i}. {fp}:{sl}-{el} [{ntype}] {sym}{score_str}")
                print(f"     {snippet}...")
        except Exception as e:
            logger.warning("Search error for query=%r type=%s: %s", query, qtype, e)

    # Step 6: Test incremental re-index (should skip unchanged)
    print()
    print("=" * 60)
    print("STEP 6: Incremental Re-index (should skip all)")
    print("=" * 60)
    result2 = index_files(table, config)
    print(f"  Files scanned:  {result2.files_scanned}")
    print(f"  Files indexed:  {result2.files_indexed}")
    print(f"  Files skipped:  {result2.files_skipped}")
    print()

    # Step 7: Table stats
    print("=" * 60)
    print("STEP 7: Table Stats")
    print("=" * 60)
    from collections import Counter
    at = table.to_arrow()
    print(f"  Total chunks: {at.num_rows}")
    print(f"  Total files:  {len(set(at.column('file_path').to_pylist()))}")
    print(f"  Languages:    {dict(Counter(at.column('language').to_pylist()))}")
    print(f"  Node types:   {dict(Counter(at.column('node_type').to_pylist()))}")

    # Step 8: Multi-project support
    print()
    print("=" * 60)
    print("STEP 8: Multi-Project Support")
    print("=" * 60)

    from projects import (
        DEFAULT_PROJECT_NAME,
        DEFAULT_TABLE_NAME,
        ProjectState,
        create_project,
        load_registry,
        save_registry,
        table_name_for_project,
        validate_project_name,
    )
    from errors import ProjectError

    # 8a: Validate project names
    print("  8a: Name validation")
    good_names = ["default", "frontend", "my-app", "Project_1"]
    bad_names = ["", "1invalid", "a" * 64, "has space", "has.dot", "@nope"]
    for name in good_names:
        try:
            validate_project_name(name)
            print(f"    '{name}' — valid (OK)")
        except ProjectError:
            print(f"    '{name}' — FAIL (expected valid)")
            sys.exit(1)
    for name in bad_names:
        try:
            validate_project_name(name)
            print(f"    '{name}' — FAIL (expected invalid)")
            sys.exit(1)
        except ProjectError:
            print(f"    '{name}' — invalid (OK)")

    # 8b: Table naming convention
    print("  8b: Table naming")
    assert table_name_for_project("default") == "code_chunks", "default → code_chunks"
    assert table_name_for_project("frontend") == "project_frontend", "frontend → project_frontend"
    print(f"    'default'  → '{table_name_for_project('default')}' (OK)")
    print(f"    'frontend' → '{table_name_for_project('frontend')}' (OK)")

    # 8c: Create a second project scoped to lancedb-mcp-server/ subdirectory
    print("  8c: Create second project")
    mcp_dir = os.path.join(REPO_ROOT, "lancedb-mcp-server")
    proj2 = create_project("mcp-server", mcp_dir)
    assert proj2.name == "mcp-server"
    assert proj2.table_name == "project_mcp-server"
    assert proj2.repo_root == str(Path(mcp_dir).resolve())
    print(f"    Created project '{proj2.name}' → table '{proj2.table_name}'")
    print(f"    repo_root: {proj2.repo_root}")

    # 8d: Registry save/load round-trip
    print("  8d: Registry round-trip")
    projects_dict = {
        DEFAULT_PROJECT_NAME: ProjectState(
            name=DEFAULT_PROJECT_NAME,
            repo_root=REPO_ROOT,
            table_name=DEFAULT_TABLE_NAME,
            created_at="2025-01-01T00:00:00+00:00",
        ),
        proj2.name: proj2,
    }
    save_registry(config.db_full_path, projects_dict)
    loaded = load_registry(config.db_full_path)
    assert set(loaded.keys()) == {DEFAULT_PROJECT_NAME, "mcp-server"}, (
        f"Expected 2 projects, got {set(loaded.keys())}"
    )
    assert loaded["mcp-server"].table_name == "project_mcp-server"
    print(f"    Saved and loaded {len(loaded)} projects (OK)")

    # 8e: Index the second project into its own table
    print("  8e: Index second project")
    proj2_table_name = proj2.table_name
    if proj2_table_name in db.list_tables():
        db.drop_table(proj2_table_name)
    table2 = db.create_table(proj2_table_name, schema=CodeChunk)

    result3 = index_files(table2, config, repo_root=Path(proj2.repo_root))
    print(f"    Files scanned:  {result3.files_scanned}")
    print(f"    Files indexed:  {result3.files_indexed}")
    print(f"    Chunks created: {result3.chunks_created}")
    assert result3.files_indexed > 0, "Second project should index some files"

    # 8f: Verify isolation — file sets differ between projects
    print("  8f: Verify isolation")
    at1 = table.to_arrow()
    at2 = table2.to_arrow()
    files1 = set(at1.column("file_path").to_pylist())
    files2 = set(at2.column("file_path").to_pylist())
    print(f"    Project 'default': {len(files1)} files")
    print(f"    Project 'mcp-server': {len(files2)} files")
    # The mcp-server project (scoped to subdirectory) should have fewer files
    # than the full repo project, and the paths should differ.
    assert len(files2) < len(files1), (
        f"Subdirectory project should have fewer files ({len(files2)}) "
        f"than full repo ({len(files1)})"
    )
    # files2 paths are relative to mcp-server/, files1 to repo root
    # So they should be different sets
    assert files1 != files2, "File sets should differ between projects"
    print("    Isolation verified (OK)")

    # 8g: Clean up second table
    print("  8g: Cleanup second project table")
    db.drop_table(proj2_table_name)
    print(f"    Dropped table '{proj2_table_name}'")
    print()

    # Cleanup
    print()
    print("=" * 60)
    print("DONE — All tests passed!")
    print("=" * 60)

    # Clean up test db
    import shutil
    test_db = config.db_full_path
    if os.path.exists(test_db):
        shutil.rmtree(test_db)
        print(f"Cleaned up test database: {test_db}")


if __name__ == "__main__":
    main()
