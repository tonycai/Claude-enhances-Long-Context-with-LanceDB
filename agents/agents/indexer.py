"""Indexer agent — manages full and incremental repository indexing."""

from claude_agent_sdk import AgentDefinition

from config import load_prompt

INDEXER_AGENT = AgentDefinition(
    description="Manages the LanceDB code search index: full/incremental indexing, status checks, and file removal.",
    prompt=load_prompt("indexer.md"),
    model="haiku",
    tools=[
        "mcp__lancedb-code__index_files",
        "mcp__lancedb-code__index_status",
        "mcp__lancedb-code__remove_files",
    ],
)
