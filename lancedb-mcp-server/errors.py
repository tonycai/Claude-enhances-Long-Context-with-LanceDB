"""Custom exception hierarchy for the LanceDB MCP server."""


class LanceDBMCPError(Exception):
    """Base exception. Carries an optional context dict for structured metadata."""

    def __init__(self, message: str, *, context: dict | None = None):
        self.context = context or {}
        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        if self.context:
            ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{base} [{ctx}]"
        return base


class IndexingError(LanceDBMCPError):
    """Raised when file indexing or LanceDB ingestion fails."""


class SearchError(LanceDBMCPError):
    """Raised when a search query fails."""


class ChunkingError(LanceDBMCPError):
    """Raised when source file chunking fails."""


class ProjectError(LanceDBMCPError):
    """Raised when a project operation fails (invalid name, not found, etc.)."""
