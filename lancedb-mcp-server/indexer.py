"""File discovery, change detection, and LanceDB ingestion pipeline."""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import lancedb
import pathspec

from chunker import Chunk, chunk_file
from config import (
    FALLBACK_EXTENSIONS,
    SENSITIVE_PATTERNS,
    SKIP_DIRS,
    SUPPORTED_EXTENSIONS,
    Config,
)

logger = logging.getLogger(__name__)


@dataclass
class IndexResult:
    """Summary of an indexing operation."""

    files_scanned: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    chunks_created: int = 0
    duration_ms: int = 0


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def _load_gitignore(repo_root: Path) -> pathspec.PathSpec | None:
    """Load .gitignore patterns from the repo root."""
    gitignore = repo_root / ".gitignore"
    if not gitignore.exists():
        return None
    try:
        patterns = gitignore.read_text(encoding="utf-8").splitlines()
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)
    except (OSError, IOError):
        return None


def _is_sensitive(rel_path: str, name: str) -> bool:
    """Check if a file matches any sensitive pattern."""
    for pattern in SENSITIVE_PATTERNS:
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(rel_path, pattern):
            return True
    return False


def _indexable_extension(path: Path) -> bool:
    """Check if a file has an indexable extension."""
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_EXTENSIONS or suffix in FALLBACK_EXTENSIONS:
        return True
    if path.name.lower() in ("dockerfile", "makefile", "cmakelists.txt"):
        return True
    return False


def discover_files(repo_root: Path, paths: list[str] | None = None) -> list[Path]:
    """Discover indexable files in the repository.

    If ``paths`` is provided, only those specific files are returned (after
    validation).  Otherwise the full repo is walked.
    """
    repo_root = repo_root.resolve()

    if paths is not None:
        result: list[Path] = []
        for p in paths:
            abs_p = Path(p)
            if not abs_p.is_absolute():
                abs_p = repo_root / p
            abs_p = abs_p.resolve()
            # Path-traversal check.
            if not str(abs_p).startswith(str(repo_root)):
                logger.warning("Skipping path outside repo root: %s", p)
                continue
            if abs_p.is_file() and _indexable_extension(abs_p):
                result.append(abs_p)
        return result

    gitignore_spec = _load_gitignore(repo_root)
    result = []

    for dirpath, dirnames, filenames in repo_root.walk():
        # Prune skipped directories in-place.
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        ]

        for fname in filenames:
            file_path = dirpath / fname
            rel = str(file_path.relative_to(repo_root))

            if gitignore_spec and gitignore_spec.match_file(rel):
                continue
            if _is_sensitive(rel, fname):
                continue
            if not _indexable_extension(file_path):
                continue

            result.append(file_path)

    return sorted(result)


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------


def file_content_hash(path: Path) -> str:
    """Return SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    try:
        h.update(path.read_bytes())
    except (OSError, IOError):
        return ""
    return h.hexdigest()


def chunk_id(file_path: str, start_line: int, end_line: int) -> str:
    """Deterministic chunk identifier."""
    raw = f"{file_path}:{start_line}:{end_line}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def chunks_to_records(chunks: list[Chunk], content_hash: str) -> list[dict]:
    """Convert Chunk dataclasses to dicts ready for LanceDB insertion."""
    records: list[dict] = []
    for c in chunks:
        records.append({
            "text": c.text,
            "chunk_id": chunk_id(c.file_path, c.start_line, c.end_line),
            "file_path": c.file_path,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "language": c.language,
            "node_type": c.node_type,
            "symbol_name": c.symbol_name,
            "content_hash": content_hash,
        })
    return records


def index_files(
    table: lancedb.table.Table,
    config: Config,
    paths: list[str] | None = None,
    force: bool = False,
) -> IndexResult:
    """Index files into LanceDB.

    Parameters
    ----------
    table:
        The LanceDB table to write into.
    config:
        Server configuration.
    paths:
        Specific file paths to index. ``None`` means the full repo.
    force:
        If True, re-index files even if their content hash is unchanged.
    """
    start = time.monotonic()
    repo_root = config.repo_root_path
    files = discover_files(repo_root, paths)
    result = IndexResult(files_scanned=len(files))

    # Build a set of existing content hashes per file for change detection.
    existing_hashes: dict[str, str] = {}
    if not force:
        try:
            arrow_table = table.to_arrow()
            if arrow_table.num_rows > 0 and "file_path" in arrow_table.column_names:
                fp_col = arrow_table.column("file_path").to_pylist()
                ch_col = arrow_table.column("content_hash").to_pylist()
                # Keep last hash per file path.
                for fp, ch in zip(fp_col, ch_col):
                    existing_hashes[fp] = ch
        except Exception:
            # Table may be empty or not yet created.
            pass

    all_records: list[dict] = []
    files_to_delete: list[str] = []

    for fpath in files:
        rel = str(fpath.relative_to(repo_root))
        fhash = file_content_hash(fpath)

        if not force and existing_hashes.get(rel) == fhash:
            result.files_skipped += 1
            continue

        chunks = chunk_file(str(fpath), repo_root)
        if not chunks:
            result.files_skipped += 1
            continue

        records = chunks_to_records(chunks, fhash)
        all_records.extend(records)
        files_to_delete.append(rel)
        result.files_indexed += 1
        result.chunks_created += len(records)

    # Delete old chunks for files being re-indexed.
    if files_to_delete:
        for fp in files_to_delete:
            try:
                escaped = fp.replace("'", "''")
                table.delete(f"file_path = '{escaped}'")
            except Exception as exc:
                logger.warning("Failed to delete old chunks for %s: %s", fp, exc)

    # Insert new chunks.
    if all_records:
        try:
            table.add(all_records)
        except Exception as exc:
            logger.error("Failed to add records to LanceDB: %s", exc)
            raise

    result.duration_ms = int((time.monotonic() - start) * 1000)
    return result


def remove_files(table: lancedb.table.Table, paths: list[str], config: Config) -> int:
    """Remove all chunks for the given file paths. Returns count of files removed."""
    removed = 0
    repo_root = config.repo_root_path
    for p in paths:
        abs_p = Path(p)
        if not abs_p.is_absolute():
            abs_p = repo_root / p
        abs_p = abs_p.resolve()
        if not str(abs_p).startswith(str(repo_root)):
            logger.warning("Skipping path outside repo root: %s", p)
            continue
        rel = str(abs_p.relative_to(repo_root))
        try:
            escaped = rel.replace("'", "''")
            table.delete(f"file_path = '{escaped}'")
            removed += 1
        except Exception as exc:
            logger.warning("Failed to remove %s: %s", rel, exc)
    return removed
