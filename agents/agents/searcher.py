"""Searcher agent — semantic code search via LanceDB hybrid search."""

from claude_agent_sdk import AgentDefinition

from config import load_prompt

SEARCHER_AGENT = AgentDefinition(
    description="Searches the codebase semantically using LanceDB hybrid vector+FTS search with metadata filters.",
    prompt=load_prompt("searcher.md"),
    model="sonnet",
    tools=[
        "mcp__lancedb-code__search_code",
        "mcp__lancedb-code__index_status",
        "Read",
        "Grep",
        "Glob",
    ],
)
