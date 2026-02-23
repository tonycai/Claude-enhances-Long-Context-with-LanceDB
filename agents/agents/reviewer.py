"""Reviewer agent — code review via semantic search and file reading."""

from claude_agent_sdk import AgentDefinition

from config import load_prompt

REVIEWER_AGENT = AgentDefinition(
    description="Conducts code review by searching for relevant code, reading source files, and analyzing quality, security, and maintainability.",
    prompt=load_prompt("reviewer.md"),
    tools=[
        "mcp__lancedb-code__search_code",
        "Read",
        "Grep",
        "Glob",
    ],
)
