"""Shared configuration for the LanceDB agent team."""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

AGENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENTS_DIR.parent
MCP_SERVER_DIR = PROJECT_ROOT / "lancedb-mcp-server"
PROMPTS_DIR = AGENTS_DIR / "prompts"

# ---------------------------------------------------------------------------
# Model IDs
# ---------------------------------------------------------------------------

MODEL_ORCHESTRATOR = "claude-sonnet-4-6"
MODEL_INDEXER = "claude-haiku-4-5"
MODEL_SEARCHER = "claude-sonnet-4-6"
MODEL_REVIEWER = "claude-opus-4-6"
MODEL_QA = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# MCP server configuration (stdio transport)
# ---------------------------------------------------------------------------

MCP_SERVERS = {
    "lancedb-code": {
        "type": "stdio",
        "command": "uv",
        "args": ["--directory", str(MCP_SERVER_DIR), "run", "server.py"],
    }
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_prompt(name: str) -> str:
    """Load a prompt file from the prompts/ directory."""
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()
