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
    ".swift": "swift",
}

# Extensions that get line-based fallback chunking.
FALLBACK_EXTENSIONS: set[str] = {
    ".md", ".mdx", ".txt", ".rst",
    ".yaml", ".yml", ".toml", ".json", ".json5",
    ".html", ".css", ".scss", ".less",
    ".sh", ".bash", ".zsh", ".fish",
    ".sql", ".graphql", ".proto",
    ".dockerfile", ".tf", ".hcl",
    # Apple localization files (often hold Chinese / Japanese UI text).
    ".strings", ".xcstrings",
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
    # Xcode / Swift build output and dependency checkouts.
    "DerivedData", "Pods", "Carthage", "xcuserdata",
}

# Valid node types for search filtering.
VALID_NODE_TYPES: set[str] = {
    "function", "class", "method", "module", "block", "interface", "enum",
    "struct", "trait", "impl",
}

MAX_CHUNK_CHARS = 2000  # ~500 tokens at ~4 chars/token
FALLBACK_CHUNK_LINES = 50
FALLBACK_OVERLAP_LINES = 5


def _default_chunk_token_budget() -> int:
    """Token budget for line-based chunks (0 = no limit).

    Chinese and Japanese text uses roughly one token per character, so a
    50-line Markdown chunk can be far longer than a model's input limit and
    the tail would be cut off before embedding. E5 models read at most 512
    tokens, so keep their chunks under that. Override with
    LANCEDB_CHUNK_TOKEN_BUDGET.
    """
    raw = os.environ.get("LANCEDB_CHUNK_TOKEN_BUDGET")
    if raw is not None and raw.strip():
        try:
            return max(0, int(raw))
        except ValueError:
            return 0
    model = os.environ.get("LANCEDB_EMBEDDING_MODEL", "").lower()
    if "e5" in model.split("/")[-1]:
        return 480
    return 0


CHUNK_TOKEN_BUDGET = _default_chunk_token_budget()


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
    # Query / passage prefixes for models that expect them (E5 family).
    # Unset means "pick the model's default"; set to "" to disable.
    query_prefix_override: str | None = field(
        default_factory=lambda: os.environ.get("LANCEDB_QUERY_PREFIX")
    )
    passage_prefix_override: str | None = field(
        default_factory=lambda: os.environ.get("LANCEDB_PASSAGE_PREFIX")
    )
    # Full-text search tokenizer. "simple" splits on spaces and punctuation,
    # which works for English but not for Chinese or Japanese (no spaces
    # between words). "ngram" indexes 2- and 3-character pieces, so keyword
    # search also works for CJK text.
    fts_tokenizer: str = field(
        default_factory=lambda: os.environ.get("LANCEDB_FTS_TOKENIZER", "simple")
    )

    @property
    def query_prefix(self) -> str:
        if self.query_prefix_override is not None:
            return self.query_prefix_override
        from embeddings import default_prefixes

        return default_prefixes(self.embedding_model)[0]

    @property
    def passage_prefix(self) -> str:
        if self.passage_prefix_override is not None:
            return self.passage_prefix_override
        from embeddings import default_prefixes

        return default_prefixes(self.embedding_model)[1]

    def fts_index_kwargs(self) -> dict:
        """Keyword arguments for ``table.create_fts_index``."""
        if self.fts_tokenizer.lower() == "ngram":
            return {
                "base_tokenizer": "ngram",
                "ngram_min_length": 2,
                "ngram_max_length": 3,
                "lower_case": True,
                "ascii_folding": True,
                "stem": False,
                "remove_stop_words": False,
            }
        return {}

    @property
    def repo_root_path(self) -> Path:
        return Path(self.repo_root).resolve()

    @property
    def db_full_path(self) -> str:
        p = Path(self.db_path)
        if not p.is_absolute():
            p = self.repo_root_path / p
        return str(p.resolve())
