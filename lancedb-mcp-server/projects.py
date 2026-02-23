"""Multi-project registry management for the LanceDB MCP server."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from errors import ProjectError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,62}$")
REGISTRY_FILENAME = "_projects.json"
DEFAULT_PROJECT_NAME = "default"
DEFAULT_TABLE_NAME = "code_chunks"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ProjectState:
    """Persistent state for a registered project."""

    name: str           # validated project name
    repo_root: str      # absolute path to repo root
    table_name: str     # LanceDB table name
    created_at: str     # ISO 8601 UTC


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_project_name(name: str) -> None:
    """Raise ``ProjectError`` if *name* is not a valid project identifier.

    Rules:
    - Must start with a letter.
    - May contain letters, digits, underscores, and hyphens.
    - Length 1-63 characters.
    """
    if not PROJECT_NAME_PATTERN.match(name):
        raise ProjectError(
            f"Invalid project name '{name}'. Must match "
            f"/{PROJECT_NAME_PATTERN.pattern}/ (start with a letter, "
            f"1-63 chars, letters/digits/underscore/hyphen).",
            context={"name": name},
        )


def table_name_for_project(name: str) -> str:
    """Return the LanceDB table name for a given project name.

    ``"default"`` maps to the legacy ``"code_chunks"`` table; all other
    projects use ``"project_{name}"``.
    """
    if name == DEFAULT_PROJECT_NAME:
        return DEFAULT_TABLE_NAME
    return f"project_{name}"


# ---------------------------------------------------------------------------
# Registry persistence
# ---------------------------------------------------------------------------


def registry_path(db_path: str) -> Path:
    """Return the path to ``_projects.json`` inside the DB directory."""
    return Path(db_path) / REGISTRY_FILENAME


def load_registry(db_path: str) -> dict[str, ProjectState]:
    """Load the project registry from disk.

    Returns an empty dict if the file does not exist.  Raises
    ``ProjectError`` if the file exists but is malformed.
    """
    rpath = registry_path(db_path)
    if not rpath.exists():
        return {}

    try:
        raw = json.loads(rpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectError(
            f"Failed to read project registry at {rpath}: {exc}",
            context={"path": str(rpath)},
        ) from exc

    if not isinstance(raw, dict):
        raise ProjectError(
            f"Malformed project registry (expected object, got {type(raw).__name__})",
            context={"path": str(rpath)},
        )

    projects: dict[str, ProjectState] = {}
    for name, data in raw.items():
        try:
            projects[name] = ProjectState(
                name=data["name"],
                repo_root=data["repo_root"],
                table_name=data["table_name"],
                created_at=data["created_at"],
            )
        except (KeyError, TypeError) as exc:
            logger.warning("Skipping malformed project entry '%s': %s", name, exc)

    return projects


def save_registry(db_path: str, projects: dict[str, ProjectState]) -> None:
    """Persist the project registry to disk (atomic write)."""
    rpath = registry_path(db_path)
    rpath.parent.mkdir(parents=True, exist_ok=True)

    payload = {name: asdict(state) for name, state in projects.items()}
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    # Atomic write: write to a temp file in the same directory, then rename.
    fd, tmp = tempfile.mkstemp(dir=str(rpath.parent), suffix=".tmp")
    try:
        os.write(fd, data.encode("utf-8"))
        os.close(fd)
        os.replace(tmp, str(rpath))
    except Exception:
        os.close(fd) if not os.get_inheritable(fd) else None  # noqa: E501
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# Project creation
# ---------------------------------------------------------------------------


def create_project(name: str, repo_root: str) -> ProjectState:
    """Validate inputs and return a new ``ProjectState``.

    The caller is responsible for persisting the state to the registry.
    """
    validate_project_name(name)

    resolved = str(Path(repo_root).resolve())
    if not Path(resolved).is_dir():
        raise ProjectError(
            f"repo_root does not exist or is not a directory: {repo_root}",
            context={"repo_root": repo_root},
        )

    return ProjectState(
        name=name,
        repo_root=resolved,
        table_name=table_name_for_project(name),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
