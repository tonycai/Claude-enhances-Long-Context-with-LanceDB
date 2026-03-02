"""Environment-based configuration for the LanceDB MCP server."""

import os
from dataclasses import dataclass, field
from pathlib import Path


# Extensions recognized for syntax-aware chunking via Tree-sitter.
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".rb": "ruby",
    ".cs": "csharp",
}

# Extensions that get line-based fallback chunking.
FALLBACK_EXTENSIONS: set[str] = {
    ".md", ".mdx", ".txt", ".rst",
    ".yaml", ".yml", ".toml", ".json", ".json5",
    ".html", ".css", ".scss", ".less",
    ".sh", ".bash", ".zsh", ".fish",
    ".sql", ".graphql", ".proto",
    ".dockerfile", ".tf", ".hcl",
}

# File patterns to never index (security-sensitive or binary).
SENSITIVE_PATTERNS: list[str] = [
    ".env", ".env.*",
    "*.pem", "*.key", "*.crt", "*.p12", "*.pfx", "*.jks",
    "*.keystore", "*.truststore",
    "id_rsa", "id_ed25519", "id_ecdsa",
    ".ssh/*", ".gnupg/*",
]

# Directories to always skip during file discovery.
SKIP_DIRS: set[str] = {
    ".git", ".hg", ".svn",
    "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "target", "build", "dist", "out", ".next", ".nuxt",
    ".lancedb",
    ".venv", "venv", "env",
    ".tox", ".nox",
    "vendor", "third_party",
}

# Valid node types for search filtering.
VALID_NODE_TYPES: set[str] = {
    "function", "class", "method", "module", "block", "interface", "enum",
    "struct", "trait", "impl",
}

MAX_CHUNK_CHARS = 2000  # ~500 tokens at ~4 chars/token
FALLBACK_CHUNK_LINES = 50
FALLBACK_OVERLAP_LINES = 5


@dataclass
class Config:
    """Server configuration loaded from environment variables."""

    db_path: str = field(
        default_factory=lambda: os.environ.get("LANCEDB_PATH", "./.lancedb")
    )
    repo_root: str = field(
        default_factory=lambda: os.environ.get("LANCEDB_REPO_ROOT", ".")
    )
    embedding_provider: str = field(
        default_factory=lambda: os.environ.get(
            "LANCEDB_EMBEDDING_PROVIDER", "sentence-transformers"
        )
    )
    embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "LANCEDB_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
    )
    table_name: str = field(
        default_factory=lambda: os.environ.get("LANCEDB_TABLE_NAME", "code_chunks")
    )
    active_project: str | None = field(
        default_factory=lambda: os.environ.get("LANCEDB_ACTIVE_PROJECT")
    )

    @property
    def repo_root_path(self) -> Path:
        return Path(self.repo_root).resolve()

    @property
    def db_full_path(self) -> str:
        p = Path(self.db_path)
        if not p.is_absolute():
            p = self.repo_root_path / p
        return str(p.resolve())
