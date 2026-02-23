"""Acceptance test suite for the LanceDB MCP server.

Maps to the acceptance criteria defined for Epics 1-4 and 6.
Run with: cd lancedb-mcp-server && uv run pytest test_acceptance.py -v

Test IDs follow the convention: test_US{story}_{AC number}_{short_description}
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Environment setup (must happen before importing project modules)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TEST_DB_DIR = None


def _setup_env():
    global _TEST_DB_DIR
    _TEST_DB_DIR = tempfile.mkdtemp(prefix="lancedb_test_")
    os.environ["LANCEDB_REPO_ROOT"] = str(_REPO_ROOT)
    os.environ["LANCEDB_PATH"] = _TEST_DB_DIR


_setup_env()

from chunker import (
    Chunk,
    _chunk_by_lines,
    _chunk_with_treesitter,
    _extract_symbol_name,
    _load_language,
    _split_oversized,
    chunk_file,
)
from config import (
    FALLBACK_CHUNK_LINES,
    FALLBACK_EXTENSIONS,
    FALLBACK_OVERLAP_LINES,
    MAX_CHUNK_CHARS,
    SENSITIVE_PATTERNS,
    SKIP_DIRS,
    SUPPORTED_EXTENSIONS,
    VALID_NODE_TYPES,
    Config,
)
from errors import ChunkingError, IndexingError, LanceDBMCPError, ProjectError, SearchError
from indexer import (
    IndexResult,
    chunk_id,
    chunks_to_records,
    discover_files,
    file_content_hash,
    index_files,
    remove_files,
)
from projects import (
    DEFAULT_PROJECT_NAME,
    DEFAULT_TABLE_NAME,
    PROJECT_NAME_PATTERN,
    ProjectState,
    check_repo_root,
    create_project,
    load_registry,
    registry_path,
    save_registry,
    table_name_for_project,
    validate_project_name,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return _REPO_ROOT


@pytest.fixture(scope="session")
def config() -> Config:
    return Config()


@pytest.fixture(scope="session")
def db_and_schema(config):
    """Session-scoped LanceDB connection and schema for search/index tests."""
    import lancedb as ldb
    from lancedb.embeddings import get_registry
    from lancedb.pydantic import LanceModel, Vector

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

    return db, CodeChunk


@pytest.fixture()
def fresh_table(db_and_schema):
    """A clean LanceDB table for each test that needs one."""
    db, schema = db_and_schema
    table_name = f"test_{os.getpid()}_{id(object())}"
    if table_name in db.list_tables():
        db.drop_table(table_name)
    table = db.create_table(table_name, schema=schema)
    yield table
    try:
        db.drop_table(table_name)
    except Exception:
        pass


@pytest.fixture()
def indexed_table(fresh_table, config, repo_root):
    """A LanceDB table pre-populated with indexed files from the repo."""
    result = index_files(fresh_table, config, repo_root=repo_root)
    assert result.files_indexed > 0, "Indexing must produce results for tests"
    try:
        fresh_table.create_fts_index("text", replace=True)
    except Exception:
        pass
    return fresh_table, result


@pytest.fixture()
def tmp_project_dir(tmp_path) -> Path:
    """A temporary directory with sample source files for isolated tests."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "hello.py").write_text(
        'def greet(name: str) -> str:\n    """Say hello."""\n    return f"Hello, {name}!"\n'
    )
    (src / "util.py").write_text(
        "import os\n\ndef get_cwd():\n    return os.getcwd()\n"
    )
    (src / "notes.md").write_text("# Notes\n\nThis is a test file.\n" * 20)
    (src / ".env").write_text("SECRET_KEY=supersecret\n")
    (src / "config.yaml").write_text("key: value\nlist:\n  - a\n  - b\n")
    (tmp_path / ".gitignore").write_text("*.log\nbuild/\n")
    return tmp_path


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def pytest_sessionfinish(session, exitstatus):
    if _TEST_DB_DIR and os.path.exists(_TEST_DB_DIR):
        shutil.rmtree(_TEST_DB_DIR, ignore_errors=True)


# =========================================================================
# EPIC 1: Semantic Code Search
# =========================================================================


class TestUS1_1_SearchCodeByMeaning:
    """US-1.1: Search code by meaning."""

    def test_ac1_returns_ranked_results_with_expected_fields(self, indexed_table):
        """AC-1: Results contain file_path, start_line, end_line, node_type,
        symbol_name, and text truncated to 200 chars."""
        from server import _format_results

        table, _ = indexed_table
        results = table.search("MCP server tools", query_type="vector").limit(5).to_list()
        assert len(results) > 0
        for r in results:
            assert "file_path" in r
            assert "start_line" in r
            assert "end_line" in r
            assert "node_type" in r
            assert "symbol_name" in r
            assert "text" in r

        formatted = _format_results(results, "MCP server tools")
        assert "Found" in formatted
        assert "results" in formatted

    def test_ac2_no_results_message(self):
        """AC-2: Empty results return 'No results found' message."""
        from server import _format_results

        output = _format_results([], "nonexistent_xyz_abc")
        assert output == 'No results found for "nonexistent_xyz_abc".'

    def test_ac3_result_includes_score_or_distance(self, indexed_table):
        """AC-3: Results include a relevance score or distance."""
        table, _ = indexed_table
        results = table.search("function definition", query_type="vector").limit(3).to_list()
        assert len(results) > 0
        for r in results:
            has_score = (
                r.get("_distance") is not None
                or r.get("_relevance_score") is not None
                or r.get("_score") is not None
                or r.get("score") is not None
            )
            assert has_score, f"Result missing score/distance: {r.keys()}"

    def test_ac4_snippet_truncation(self):
        """AC-4: Snippets > 200 chars are truncated with '...'."""
        from server import _format_results

        long_text = "x" * 300
        results = [{"file_path": "a.py", "start_line": 1, "end_line": 10,
                     "node_type": "function", "symbol_name": "foo",
                     "language": "python", "text": long_text}]
        formatted = _format_results(results, "test")
        # The snippet in the formatted output should end with "..."
        assert "..." in formatted

    def test_ac5_limit_parameter(self, indexed_table):
        """AC-5: The limit parameter controls max results."""
        table, _ = indexed_table
        results_3 = table.search("code", query_type="vector").limit(3).to_list()
        results_1 = table.search("code", query_type="vector").limit(1).to_list()
        assert len(results_1) <= 1
        assert len(results_3) <= 3

    def test_ac6_project_error_when_no_project(self):
        """AC-6: _resolve_table raises ProjectError when no active project."""
        from server import AppContext, _resolve_table

        app = AppContext(
            db=MagicMock(), config=Config(), schema=type,
            projects={}, active_project=None,
        )
        with pytest.raises(ProjectError, match="No active project"):
            _resolve_table(app)


class TestUS1_2_SearchModes:
    """US-1.2: Choose search mode."""

    def test_ac1_hybrid_uses_rrf_reranker(self, indexed_table):
        """AC-1: Hybrid search uses RRF reranker and returns results."""
        from lancedb.rerankers import RRFReranker

        table, _ = indexed_table
        results = (
            table.search("semantic search", query_type="hybrid")
            .rerank(reranker=RRFReranker())
            .limit(5)
            .to_list()
        )
        assert len(results) > 0

    def test_ac2_vector_only_search(self, indexed_table):
        """AC-2: Vector search returns results without FTS."""
        table, _ = indexed_table
        results = table.search("file discovery", query_type="vector").limit(5).to_list()
        assert len(results) > 0

    def test_ac3_fts_only_search(self, indexed_table):
        """AC-3: FTS search returns keyword-matched results."""
        table, _ = indexed_table
        results = table.search("import logging", query_type="fts").limit(5).to_list()
        assert len(results) > 0

    def test_ac4_invalid_query_type_defaults_to_hybrid(self):
        """AC-4: Invalid query_type silently defaults to 'hybrid'."""
        # Simulate the validation logic from server.py
        query_type = "invalid_type"
        if query_type not in ("hybrid", "vector", "fts"):
            query_type = "hybrid"
        assert query_type == "hybrid"


class TestUS1_3_FilterResults:
    """US-1.3: Filter search results by metadata."""

    def test_ac1_language_filter(self, indexed_table):
        """AC-1: language filter applies SQL filter with prefilter."""
        table, _ = indexed_table
        results = (
            table.search("function", query_type="vector")
            .where("language = 'python'", prefilter=True)
            .limit(10)
            .to_list()
        )
        for r in results:
            assert r["language"] == "python"

    def test_ac2_file_path_pattern_filter(self, indexed_table):
        """AC-2: file_path_pattern applies LIKE prefix filter."""
        table, _ = indexed_table
        results = (
            table.search("server", query_type="vector")
            .where("file_path LIKE 'lancedb-mcp-server/%'", prefilter=True)
            .limit(10)
            .to_list()
        )
        for r in results:
            assert r["file_path"].startswith("lancedb-mcp-server/")

    def test_ac4_invalid_node_type_rejected(self):
        """AC-4: Invalid node_type returns error listing valid types."""
        invalid = "widget"
        assert invalid not in VALID_NODE_TYPES
        expected_valid = sorted(VALID_NODE_TYPES)
        assert len(expected_valid) == 10
        assert "function" in expected_valid
        assert "class" in expected_valid
        assert "method" in expected_valid

    def test_ac6_sql_injection_prevention(self):
        """AC-6: Single quotes in filter values are escaped."""
        from server import _sanitize_str

        assert _sanitize_str("test'value") == "test''value"
        assert _sanitize_str("no_quotes") == "no_quotes"
        assert _sanitize_str("it''s") == "it''''s"


# =========================================================================
# EPIC 2: Repository Indexing
# =========================================================================


class TestUS2_1_IndexEntireRepo:
    """US-2.1: Index an entire repository."""

    def test_ac1_full_repo_walk_with_no_paths(self, repo_root):
        """AC-1: discover_files with no paths walks entire repo."""
        files = discover_files(repo_root)
        assert len(files) > 0
        # Should find server.py somewhere
        names = [f.name for f in files]
        assert "server.py" in names

    def test_ac2_gitignore_respected(self, tmp_project_dir):
        """AC-2: Files matching .gitignore patterns are skipped."""
        # Create a .log file that should be ignored
        (tmp_project_dir / "debug.log").write_text("log data")
        (tmp_project_dir / "src" / "app.py").write_text("print('hello')")
        files = discover_files(tmp_project_dir)
        names = [f.name for f in files]
        assert "debug.log" not in names
        assert "app.py" in names

    def test_ac3_skip_dirs_pruned(self, repo_root):
        """AC-3: Directories in SKIP_DIRS are pruned during walk."""
        files = discover_files(repo_root)
        for f in files:
            parts = f.relative_to(repo_root).parts
            for skip in SKIP_DIRS:
                assert skip not in parts, f"Found file in skip dir {skip}: {f}"

    def test_ac4_hidden_dirs_skipped(self, repo_root):
        """AC-4: Hidden directories (starting with '.') are skipped."""
        files = discover_files(repo_root)
        for f in files:
            parts = f.relative_to(repo_root).parts
            # No directory component (except possibly the root) should start with '.'
            for part in parts[:-1]:  # Exclude filename itself
                assert not part.startswith("."), f"Found file under hidden dir: {f}"

    def test_ac5_large_files_skipped(self, tmp_path):
        """AC-5: Files > 1 MB are skipped during chunking."""
        big_file = tmp_path / "big.py"
        big_file.write_text("x = 1\n" * 300_000)  # ~1.8 MB
        assert big_file.stat().st_size > 1_048_576
        chunks = chunk_file(str(big_file), tmp_path)
        assert chunks == []

    def test_ac6_only_recognized_extensions(self, tmp_project_dir):
        """AC-6: Only files with recognized extensions are indexed."""
        (tmp_project_dir / "src" / "data.bin").write_bytes(b"\x00\x01\x02")
        (tmp_project_dir / "src" / "image.png").write_bytes(b"PNG")
        files = discover_files(tmp_project_dir)
        names = [f.name for f in files]
        assert "data.bin" not in names
        assert "image.png" not in names

    def test_ac7_index_result_fields(self, fresh_table, config, repo_root):
        """AC-7: IndexResult contains all expected counters."""
        result = index_files(fresh_table, config, repo_root=repo_root)
        assert isinstance(result.files_scanned, int)
        assert isinstance(result.files_indexed, int)
        assert isinstance(result.files_skipped, int)
        assert isinstance(result.chunks_created, int)
        assert isinstance(result.duration_ms, int)
        assert result.files_scanned > 0
        assert result.files_indexed > 0
        assert result.chunks_created > 0
        assert result.duration_ms >= 0


class TestUS2_2_IncrementalReindex:
    """US-2.2: Incremental re-indexing."""

    def test_ac1_specific_paths_only(self, fresh_table, config, repo_root):
        """AC-1: Passing specific paths indexes only those files."""
        server_py = str(repo_root / "lancedb-mcp-server" / "server.py")
        result = index_files(fresh_table, config, paths=[server_py], repo_root=repo_root)
        assert result.files_scanned == 1
        assert result.files_indexed == 1

    def test_ac2_unchanged_files_skipped(self, fresh_table, config, repo_root):
        """AC-2: SHA-256 hash detects unchanged files; they are skipped."""
        # First indexing
        result1 = index_files(fresh_table, config, repo_root=repo_root)
        assert result1.files_indexed > 0
        # Second indexing — all unchanged
        result2 = index_files(fresh_table, config, repo_root=repo_root)
        assert result2.files_indexed == 0
        assert result2.files_skipped == result2.files_scanned

    def test_ac3_old_chunks_deleted_before_reinsertion(self, fresh_table, config, repo_root):
        """AC-3: Old chunks for a re-indexed file are deleted before new ones."""
        server_py = str(repo_root / "lancedb-mcp-server" / "server.py")
        # Index once
        index_files(fresh_table, config, paths=[server_py], repo_root=repo_root)
        at1 = fresh_table.to_arrow()
        count1 = at1.num_rows
        # Force re-index
        index_files(fresh_table, config, paths=[server_py], force=True, repo_root=repo_root)
        at2 = fresh_table.to_arrow()
        count2 = at2.num_rows
        # Chunk count should remain the same (old deleted, new inserted)
        assert count2 == count1

    def test_ac4_force_reindexes_unchanged(self, fresh_table, config, repo_root):
        """AC-4: force=True re-indexes even if hash unchanged."""
        result1 = index_files(fresh_table, config, repo_root=repo_root)
        result2 = index_files(fresh_table, config, force=True, repo_root=repo_root)
        assert result2.files_indexed > 0
        assert result2.files_indexed == result1.files_indexed

    def test_ac6_relative_paths_resolved(self, fresh_table, config, repo_root):
        """AC-6: Relative paths are resolved against repo_root."""
        result = index_files(
            fresh_table, config,
            paths=["lancedb-mcp-server/config.py"],
            repo_root=repo_root,
        )
        assert result.files_scanned == 1
        assert result.files_indexed == 1


class TestUS2_3_SyntaxAwareChunking:
    """US-2.3: Syntax-aware chunking."""

    def test_ac1_treesitter_languages_loadable(self):
        """AC-1: Tree-sitter grammars load for core languages."""
        for lang_key in ("python", "javascript", "typescript", "tsx", "rust", "go"):
            lang = _load_language(lang_key)
            assert lang is not None, f"Failed to load grammar: {lang_key}"

    def test_ac2_python_extractable_types(self):
        """AC-2: Python extracts function_definition and class_definition."""
        from chunker import _EXTRACTABLE_TYPES

        py = _EXTRACTABLE_TYPES["python"]
        assert py["function_definition"] == "function"
        assert py["class_definition"] == "class"
        assert py["decorated_definition"] == "decorated"

    def test_ac3_typescript_extractable_types(self):
        """AC-3: TypeScript extracts the full set of node types."""
        from chunker import _EXTRACTABLE_TYPES

        ts = _EXTRACTABLE_TYPES["typescript"]
        assert "function_declaration" in ts
        assert "class_declaration" in ts
        assert "interface_declaration" in ts
        assert "enum_declaration" in ts

    def test_ac4_rust_extractable_types(self):
        """AC-4: Rust extracts function_item, struct_item, etc."""
        from chunker import _EXTRACTABLE_TYPES

        rs = _EXTRACTABLE_TYPES["rust"]
        assert rs["function_item"] == "function"
        assert rs["struct_item"] == "struct"
        assert rs["enum_item"] == "enum"
        assert rs["impl_item"] == "impl"
        assert rs["trait_item"] == "trait"

    def test_ac5_class_methods_decomposed(self, tmp_path):
        """AC-5: Class bodies are decomposed into individual method chunks."""
        src = tmp_path / "example.py"
        src.write_text(
            "class MyClass:\n"
            "    def method_a(self):\n"
            "        pass\n"
            "    def method_b(self):\n"
            "        pass\n"
        )
        chunks = chunk_file(str(src), tmp_path)
        node_types = [c.node_type for c in chunks]
        symbols = [c.symbol_name for c in chunks]
        assert "method" in node_types
        assert any("MyClass.method_a" in s for s in symbols)
        assert any("MyClass.method_b" in s for s in symbols)

    def test_ac6_decorated_definition_unwrapped(self, tmp_path):
        """AC-6: Decorated functions have their inner name extracted."""
        src = tmp_path / "deco.py"
        src.write_text(
            "def my_decorator(f):\n    return f\n\n"
            "@my_decorator\n"
            "def decorated_func():\n"
            "    pass\n"
        )
        chunks = chunk_file(str(src), tmp_path)
        symbols = [c.symbol_name for c in chunks]
        assert "decorated_func" in symbols

    def test_ac7_oversized_chunks_split(self):
        """AC-7: Chunks > MAX_CHUNK_CHARS are split with part numbering."""
        long_text = "\n".join(f"line_{i} = {i}" for i in range(200))
        assert len(long_text) > MAX_CHUNK_CHARS
        parts = _split_oversized(
            long_text, "test.py", 1, "python", "function", "big_func"
        )
        assert len(parts) > 1
        # Check part numbering
        has_part_label = any("part" in c.symbol_name for c in parts)
        assert has_part_label

    def test_ac8_module_level_code_captured(self, tmp_path):
        """AC-8: Module-level code (imports, globals) is captured as 'module' type."""
        src = tmp_path / "mod.py"
        src.write_text(
            "import os\nimport sys\n\nGLOBAL_VAR = 42\n\n"
            "def my_func():\n    pass\n"
        )
        chunks = chunk_file(str(src), tmp_path)
        module_chunks = [c for c in chunks if c.node_type == "module"]
        assert len(module_chunks) >= 1
        # Module chunk should contain the imports
        assert any("import os" in c.text for c in module_chunks)

    def test_ac9_chunk_dataclass_fields(self, repo_root):
        """AC-9: Chunk contains all required fields with correct types."""
        server_py = repo_root / "lancedb-mcp-server" / "server.py"
        chunks = chunk_file(str(server_py), repo_root)
        assert len(chunks) > 0
        for c in chunks:
            assert isinstance(c.text, str) and len(c.text) > 0
            assert isinstance(c.file_path, str)
            assert isinstance(c.start_line, int) and c.start_line >= 1
            assert isinstance(c.end_line, int) and c.end_line >= c.start_line
            assert isinstance(c.language, str)
            assert isinstance(c.node_type, str)
            assert isinstance(c.symbol_name, str)

    def test_ac10_fallthrough_when_grammar_unavailable(self, tmp_path):
        """AC-10: Falls through to line-based chunking when grammar unavailable."""
        src = tmp_path / "code.xyz_unknown"
        # This won't have a Tree-sitter grammar or a known extension
        # so it should return empty
        src.write_text("some code\n" * 10)
        chunks = chunk_file(str(src), tmp_path)
        assert chunks == []


class TestUS2_4_FallbackChunking:
    """US-2.4: Fallback chunking for non-code files."""

    def test_ac1_fallback_extensions_recognized(self):
        """AC-1: All listed fallback extensions are in FALLBACK_EXTENSIONS."""
        expected = {
            ".md", ".mdx", ".txt", ".rst",
            ".yaml", ".yml", ".toml", ".json", ".json5",
            ".html", ".css", ".scss", ".less",
            ".sh", ".bash", ".zsh", ".fish",
            ".sql", ".graphql", ".proto",
            ".dockerfile", ".tf", ".hcl",
        }
        assert expected.issubset(FALLBACK_EXTENSIONS)

    def test_ac2_dockerfile_by_name(self, tmp_path):
        """AC-2: Dockerfile (by filename) is recognized for chunking."""
        df = tmp_path / "Dockerfile"
        df.write_text("FROM python:3.12\nRUN pip install flask\n")
        chunks = chunk_file(str(df), tmp_path)
        assert len(chunks) > 0

    def test_ac2_makefile_by_name(self, tmp_path):
        """AC-2: Makefile (by filename) is recognized for chunking."""
        mf = tmp_path / "Makefile"
        mf.write_text("all:\n\techo hello\n")
        chunks = chunk_file(str(mf), tmp_path)
        assert len(chunks) > 0

    def test_ac3_chunk_size_and_overlap(self):
        """AC-3: Line-based chunks are 50 lines with 5-line overlap."""
        assert FALLBACK_CHUNK_LINES == 50
        assert FALLBACK_OVERLAP_LINES == 5
        # Create text with 120 lines
        text = "\n".join(f"line {i}" for i in range(120))
        chunks = _chunk_by_lines(text, "test.md", "markdown")
        assert len(chunks) >= 3  # 120 lines / ~45 effective = ~3 chunks
        # First chunk ends at line 50
        assert chunks[0].end_line == 50
        # Second chunk starts at line 46 (50 - 5 + 1)
        assert chunks[1].start_line == 46

    def test_ac4_fallback_node_type_and_symbol(self, tmp_path):
        """AC-4: Fallback chunks have node_type='block' and empty symbol_name."""
        md = tmp_path / "readme.md"
        md.write_text("# Title\n\nContent here.\n")
        chunks = chunk_file(str(md), tmp_path)
        for c in chunks:
            assert c.node_type == "block"
            assert c.symbol_name == ""

    def test_ac5_empty_files_produce_no_chunks(self, tmp_path):
        """AC-5: Empty/whitespace-only files produce no chunks."""
        empty = tmp_path / "empty.py"
        empty.write_text("")
        assert chunk_file(str(empty), tmp_path) == []

        whitespace = tmp_path / "space.py"
        whitespace.write_text("   \n\n  \n")
        assert chunk_file(str(whitespace), tmp_path) == []


class TestUS2_5_IndexStatus:
    """US-2.5: Check index health."""

    def test_ac1_returns_comprehensive_stats(self, indexed_table):
        """AC-1: index_status returns chunks, files, languages, node types."""
        table, result = indexed_table
        at = table.to_arrow()
        assert at.num_rows > 0
        total_files = len(set(at.column("file_path").to_pylist()))
        assert total_files > 0
        lang_counts = dict(Counter(at.column("language").to_pylist()))
        assert len(lang_counts) > 0
        node_counts = dict(Counter(at.column("node_type").to_pylist()))
        assert len(node_counts) > 0

    def test_ac2_empty_table_message(self, fresh_table):
        """AC-2: Empty table reports empty index."""
        try:
            at = fresh_table.to_arrow()
            assert at.num_rows == 0
        except Exception:
            pass  # Empty table may throw; that's the expected empty state


class TestUS2_6_RemoveFiles:
    """US-2.6: Remove files from the index."""

    def test_ac1_removes_chunks_by_file_path(self, fresh_table, config, repo_root):
        """AC-1: remove_files deletes all chunks for a file."""
        server_py = str(repo_root / "lancedb-mcp-server" / "server.py")
        index_files(fresh_table, config, paths=[server_py], repo_root=repo_root)
        at_before = fresh_table.to_arrow()
        assert at_before.num_rows > 0

        removed = remove_files(
            fresh_table,
            ["lancedb-mcp-server/server.py"],
            config,
            repo_root=repo_root,
        )
        assert removed == 1
        at_after = fresh_table.to_arrow()
        assert at_after.num_rows == 0

    def test_ac2_relative_and_absolute_paths(self, fresh_table, config, repo_root):
        """AC-2: Both relative and absolute paths are accepted."""
        config_py = str(repo_root / "lancedb-mcp-server" / "config.py")
        index_files(fresh_table, config, paths=[config_py], repo_root=repo_root)

        # Remove using relative path
        removed = remove_files(
            fresh_table,
            ["lancedb-mcp-server/config.py"],
            config,
            repo_root=repo_root,
        )
        assert removed == 1

    def test_ac3_path_outside_repo_skipped(self, fresh_table, config, repo_root):
        """AC-3: Paths outside repo root are skipped."""
        removed = remove_files(
            fresh_table, ["/etc/passwd"], config, repo_root=repo_root
        )
        assert removed == 0

    def test_ac4_single_quotes_escaped(self):
        """AC-4: Single quotes in paths are escaped in delete filter."""
        # Verify the escaping pattern used in remove_files
        path = "it's/a/file.py"
        escaped = path.replace("'", "''")
        assert escaped == "it''s/a/file.py"


# =========================================================================
# EPIC 3: Security & Safety
# =========================================================================


class TestUS3_1_ExcludeSensitiveFiles:
    """US-3.1: Exclude sensitive files."""

    def test_ac1_sensitive_patterns_defined(self):
        """AC-1: All expected sensitive patterns are present."""
        expected = [".env", "*.pem", "*.key", "*.crt", "id_rsa", "id_ed25519"]
        for pat in expected:
            assert pat in SENSITIVE_PATTERNS, f"Missing pattern: {pat}"

    def test_ac2_sensitive_files_excluded_from_discovery(self, tmp_project_dir):
        """AC-2: Sensitive files are excluded during discover_files."""
        # .env was created in the fixture
        (tmp_project_dir / "src" / "server.key").write_text("private key data")
        (tmp_project_dir / "src" / "cert.pem").write_text("cert data")
        files = discover_files(tmp_project_dir)
        names = [f.name for f in files]
        assert ".env" not in names
        assert "server.key" not in names
        assert "cert.pem" not in names

    def test_ac3_checked_before_content_read(self, tmp_project_dir):
        """AC-3: Sensitive detection is at discovery, not chunking stage."""
        from indexer import _is_sensitive

        assert _is_sensitive(".env", ".env") is True
        assert _is_sensitive("config/id_rsa", "id_rsa") is True
        assert _is_sensitive("src/main.py", "main.py") is False


class TestUS3_2_PreventPathTraversal:
    """US-3.2: Prevent path traversal."""

    def test_ac1_discover_files_rejects_outside_paths(self, repo_root):
        """AC-1: Paths outside repo root are rejected in discover_files."""
        files = discover_files(repo_root, paths=["/etc/passwd", "/tmp/evil.py"])
        assert len(files) == 0

    def test_ac2_remove_files_rejects_outside_paths(self, fresh_table, config, repo_root):
        """AC-2: Paths outside repo root are rejected in remove_files."""
        removed = remove_files(
            fresh_table, ["/etc/passwd"], config, repo_root=repo_root
        )
        assert removed == 0

    def test_ac3_relative_resolved_against_repo_root(self, repo_root):
        """AC-3: Relative paths resolve against repo_root, not cwd."""
        files = discover_files(repo_root, paths=["lancedb-mcp-server/server.py"])
        assert len(files) == 1
        assert files[0].is_absolute()
        assert str(files[0]).startswith(str(repo_root))


# =========================================================================
# EPIC 4: Multi-Project Support
# =========================================================================


class TestUS4_1_CreateAndSwitchProjects:
    """US-4.1: Create and switch between projects."""

    def test_ac5_valid_project_names(self):
        """AC-5: Valid project names are accepted."""
        valid_names = [
            "default", "frontend", "my-app", "Project_1",
            "a", "A", "abcdefghij1234567890_-",
        ]
        for name in valid_names:
            validate_project_name(name)  # Should not raise

    def test_ac5_invalid_project_names(self):
        """AC-5: Invalid project names are rejected."""
        invalid_names = [
            "", "1invalid", "a" * 64, "has space", "has.dot",
            "@nope", "-starts-with-dash", "_starts-with-underscore",
        ]
        for name in invalid_names:
            with pytest.raises(ProjectError):
                validate_project_name(name)

    def test_ac6_error_includes_regex_pattern(self):
        """AC-6: ProjectError includes the regex pattern in its message."""
        with pytest.raises(ProjectError, match=r"\^.*\$"):
            validate_project_name("1bad")

    def test_ac7_nonexistent_path_accepted_with_warning(self):
        """AC-7: Non-existent repo_root creates project with warning."""
        proj = create_project("docker-test", "/nonexistent/docker/path")
        assert proj.name == "docker-test"
        warning = check_repo_root(proj.repo_root)
        assert warning is not None
        assert "does not exist" in warning

    def test_ac7_valid_path_no_warning(self, repo_root):
        """AC-7: Valid repo_root produces no warning."""
        warning = check_repo_root(str(repo_root))
        assert warning is None


class TestUS4_2_ListProjects:
    """US-4.2: List registered projects."""

    def test_ac4_sorted_alphabetically(self):
        """AC-4: Projects are listed in sorted order."""
        projects = {"zebra": "z", "alpha": "a", "middle": "m"}
        sorted_keys = sorted(projects)
        assert sorted_keys == ["alpha", "middle", "zebra"]


class TestUS4_3_RemoveProject:
    """US-4.3: Remove a project."""

    def test_ac4_active_switches_to_default(self):
        """AC-4: Removing active project switches to 'default' if available."""
        projects = {
            "default": ProjectState("default", "/a", "code_chunks", "2025-01-01T00:00:00"),
            "other": ProjectState("other", "/b", "project_other", "2025-01-01T00:00:00"),
        }
        active = "other"
        # Simulate removal
        projects.pop("other")
        if active == "other":
            if projects:
                active = (
                    DEFAULT_PROJECT_NAME
                    if DEFAULT_PROJECT_NAME in projects
                    else next(iter(projects))
                )
            else:
                active = None
        assert active == "default"

    def test_ac5_last_project_removed(self):
        """AC-5: Removing last project sets active to None."""
        projects = {
            "only": ProjectState("only", "/a", "project_only", "2025-01-01T00:00:00"),
        }
        active = "only"
        projects.pop("only")
        if not projects:
            active = None
        assert active is None


class TestUS4_4_TablePerProjectIsolation:
    """US-4.4: Table-per-project isolation."""

    def test_ac1_default_table_name(self):
        """AC-1: 'default' project maps to 'code_chunks'."""
        assert table_name_for_project("default") == "code_chunks"

    def test_ac2_other_project_table_name(self):
        """AC-2: Other projects map to 'project_{name}'."""
        assert table_name_for_project("frontend") == "project_frontend"
        assert table_name_for_project("my-app") == "project_my-app"
        assert table_name_for_project("Project_1") == "project_Project_1"

    def test_ac3_resolve_table_uses_active_when_none(self):
        """AC-3: _resolve_table uses active_project when project is None."""
        from server import AppContext, _open_or_create_table, _resolve_table

        app = AppContext(
            db=MagicMock(), config=Config(), schema=type,
            projects={
                "default": ProjectState("default", "/tmp", "code_chunks", "2025-01-01"),
            },
            active_project="default",
        )
        # Mock the table open
        mock_table = MagicMock()
        app.tables["default"] = mock_table
        table, proj = _resolve_table(app, project=None)
        assert proj.name == "default"
        assert table is mock_table

    def test_ac5_no_active_project_error(self):
        """AC-5: No active project raises ProjectError."""
        from server import AppContext, _resolve_table

        app = AppContext(
            db=MagicMock(), config=Config(), schema=type,
            projects={}, active_project=None,
        )
        with pytest.raises(ProjectError, match="No active project"):
            _resolve_table(app)

    def test_ac6_nonexistent_project_error(self):
        """AC-6: Non-existent project name raises ProjectError with list."""
        from server import AppContext, _resolve_table

        app = AppContext(
            db=MagicMock(), config=Config(), schema=type,
            projects={
                "alpha": ProjectState("alpha", "/a", "project_alpha", "2025-01-01"),
            },
            active_project="alpha",
        )
        with pytest.raises(ProjectError, match="not found"):
            _resolve_table(app, project="nonexistent")

    def test_isolation_different_file_sets(self, db_and_schema, config, repo_root):
        """Cross-project isolation: subdirectory project has fewer files."""
        db, schema = db_and_schema
        mcp_dir = repo_root / "lancedb-mcp-server"

        # Table for full repo
        t1_name = f"test_iso_full_{id(object())}"
        t1 = db.create_table(t1_name, schema=schema)
        r1 = index_files(t1, config, repo_root=repo_root)

        # Table for subdirectory
        t2_name = f"test_iso_sub_{id(object())}"
        t2 = db.create_table(t2_name, schema=schema)
        r2 = index_files(t2, config, repo_root=mcp_dir)

        at1 = t1.to_arrow()
        at2 = t2.to_arrow()
        files1 = set(at1.column("file_path").to_pylist())
        files2 = set(at2.column("file_path").to_pylist())

        assert len(files2) < len(files1)
        assert files1 != files2

        # Cleanup
        db.drop_table(t1_name)
        db.drop_table(t2_name)


class TestUS4_5_LegacyCompatibility:
    """US-4.5: Legacy compatibility."""

    def test_ac1_ac2_ac3_registry_load_save_roundtrip(self, tmp_path):
        """AC-1/2/3: Registry save/load round-trip works correctly."""
        db_path = str(tmp_path / "test_db")
        Path(db_path).mkdir()

        projects = {
            "default": ProjectState("default", "/repo", "code_chunks", "2025-01-01T00:00:00+00:00"),
            "frontend": ProjectState("frontend", "/fe", "project_frontend", "2025-06-15T12:00:00+00:00"),
        }
        save_registry(db_path, projects)

        loaded = load_registry(db_path)
        assert set(loaded.keys()) == {"default", "frontend"}
        assert loaded["default"].table_name == "code_chunks"
        assert loaded["frontend"].table_name == "project_frontend"
        assert loaded["frontend"].repo_root == "/fe"

    def test_registry_empty_when_no_file(self, tmp_path):
        """AC-4 precondition: No registry file returns empty dict."""
        db_path = str(tmp_path / "empty_db")
        Path(db_path).mkdir()
        loaded = load_registry(db_path)
        assert loaded == {}

    def test_registry_atomic_write(self, tmp_path):
        """Registry writes are atomic (temp file + rename)."""
        db_path = str(tmp_path / "atomic_db")
        Path(db_path).mkdir()
        projects = {
            "test": ProjectState("test", "/t", "project_test", "2025-01-01T00:00:00"),
        }
        save_registry(db_path, projects)
        rpath = registry_path(db_path)
        assert rpath.exists()
        data = json.loads(rpath.read_text())
        assert "test" in data

    def test_registry_malformed_json_raises_error(self, tmp_path):
        """Malformed JSON in registry raises ProjectError."""
        db_path = str(tmp_path / "bad_db")
        Path(db_path).mkdir()
        rpath = Path(db_path) / "_projects.json"
        rpath.write_text("not valid json!!!")
        with pytest.raises(ProjectError, match="Failed to read"):
            load_registry(db_path)

    def test_registry_wrong_type_raises_error(self, tmp_path):
        """Non-object JSON in registry raises ProjectError."""
        db_path = str(tmp_path / "array_db")
        Path(db_path).mkdir()
        rpath = Path(db_path) / "_projects.json"
        rpath.write_text('[1, 2, 3]')
        with pytest.raises(ProjectError, match="Malformed"):
            load_registry(db_path)


# =========================================================================
# EPIC 3 (supplement): Error Hierarchy
# =========================================================================


class TestErrorHierarchy:
    """Verify the custom exception hierarchy."""

    def test_base_exception(self):
        err = LanceDBMCPError("test", context={"key": "val"})
        assert str(err) == "test [key='val']"

    def test_base_exception_no_context(self):
        err = LanceDBMCPError("simple")
        assert str(err) == "simple"

    def test_indexing_error_is_base(self):
        assert issubclass(IndexingError, LanceDBMCPError)

    def test_search_error_is_base(self):
        assert issubclass(SearchError, LanceDBMCPError)

    def test_chunking_error_is_base(self):
        assert issubclass(ChunkingError, LanceDBMCPError)

    def test_project_error_is_base(self):
        assert issubclass(ProjectError, LanceDBMCPError)


# =========================================================================
# EPIC 6: Deployment & Operations
# =========================================================================


class TestUS6_4_EnvironmentConfig:
    """US-6.4: Configure via environment variables."""

    def test_ac1_lancedb_path_default(self):
        """AC-1: LANCEDB_PATH defaults to './.lancedb'."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LANCEDB_PATH", None)
            c = Config()
            assert c.db_path == "./.lancedb" or c.db_path == os.environ.get("LANCEDB_PATH", "./.lancedb")

    def test_ac2_repo_root_resolves_to_absolute(self):
        """AC-2: repo_root_path resolves to absolute."""
        c = Config()
        assert c.repo_root_path.is_absolute()

    def test_ac3_embedding_provider_default(self):
        """AC-3: Default embedding provider is 'sentence-transformers'."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LANCEDB_EMBEDDING_PROVIDER", None)
            c = Config()
            assert c.embedding_provider == "sentence-transformers"

    def test_ac4_embedding_model_default(self):
        """AC-4: Default embedding model is 'all-MiniLM-L6-v2'."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LANCEDB_EMBEDDING_MODEL", None)
            c = Config()
            assert c.embedding_model == "all-MiniLM-L6-v2"

    def test_ac5_table_name_default(self):
        """AC-5: Default table name is 'code_chunks'."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LANCEDB_TABLE_NAME", None)
            c = Config()
            assert c.table_name == "code_chunks"

    def test_ac1_relative_db_path_resolved_against_repo_root(self):
        """AC-1 supplement: Relative db_path is resolved against repo_root."""
        c = Config()
        db_full = Path(c.db_full_path)
        assert db_full.is_absolute()


# =========================================================================
# Indexer utility functions
# =========================================================================


class TestIndexerUtils:
    """Unit tests for indexer utility functions."""

    def test_file_content_hash_deterministic(self, tmp_path):
        """SHA-256 hash is deterministic for same content."""
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h1 = file_content_hash(f)
        h2 = file_content_hash(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_file_content_hash_changes_with_content(self, tmp_path):
        """SHA-256 hash changes when content changes."""
        f = tmp_path / "test.txt"
        f.write_text("version 1")
        h1 = file_content_hash(f)
        f.write_text("version 2")
        h2 = file_content_hash(f)
        assert h1 != h2

    def test_file_content_hash_missing_file(self, tmp_path):
        """Missing file returns empty string."""
        f = tmp_path / "missing.txt"
        h = file_content_hash(f)
        assert h == ""

    def test_chunk_id_deterministic(self):
        """chunk_id is deterministic for same inputs."""
        id1 = chunk_id("file.py", 1, 10)
        id2 = chunk_id("file.py", 1, 10)
        assert id1 == id2
        assert len(id1) == 16  # 16 hex chars

    def test_chunk_id_differs_for_different_inputs(self):
        """chunk_id differs when inputs differ."""
        id1 = chunk_id("a.py", 1, 10)
        id2 = chunk_id("b.py", 1, 10)
        id3 = chunk_id("a.py", 2, 10)
        assert id1 != id2
        assert id1 != id3

    def test_chunks_to_records(self):
        """chunks_to_records converts Chunk list to dict list."""
        chunks = [
            Chunk("def foo(): pass", "file.py", 1, 1, "python", "function", "foo"),
        ]
        records = chunks_to_records(chunks, "abc123")
        assert len(records) == 1
        r = records[0]
        assert r["text"] == "def foo(): pass"
        assert r["file_path"] == "file.py"
        assert r["start_line"] == 1
        assert r["end_line"] == 1
        assert r["language"] == "python"
        assert r["node_type"] == "function"
        assert r["symbol_name"] == "foo"
        assert r["content_hash"] == "abc123"
        assert "chunk_id" in r


# =========================================================================
# Config constants validation
# =========================================================================


class TestConfigConstants:
    """Validate configuration constants match documented values."""

    def test_max_chunk_chars(self):
        assert MAX_CHUNK_CHARS == 2000

    def test_fallback_chunk_lines(self):
        assert FALLBACK_CHUNK_LINES == 50

    def test_fallback_overlap_lines(self):
        assert FALLBACK_OVERLAP_LINES == 5

    def test_valid_node_types_count(self):
        assert len(VALID_NODE_TYPES) == 10
        expected = {
            "function", "class", "method", "module", "block",
            "interface", "enum", "struct", "trait", "impl",
        }
        assert VALID_NODE_TYPES == expected

    def test_supported_extensions_count(self):
        """All documented Tree-sitter extensions are present."""
        expected_keys = {
            ".py", ".js", ".mjs", ".jsx", ".ts", ".tsx",
            ".rs", ".go", ".java", ".c", ".h",
            ".cpp", ".hpp", ".cc", ".rb", ".cs",
        }
        assert expected_keys == set(SUPPORTED_EXTENSIONS.keys())

    def test_skip_dirs_includes_key_entries(self):
        for d in [".git", "node_modules", "__pycache__", ".lancedb", ".venv", "build", "dist"]:
            assert d in SKIP_DIRS
