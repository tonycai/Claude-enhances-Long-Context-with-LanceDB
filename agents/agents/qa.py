"""Q&A agent — answers codebase questions using semantic search."""

from claude_agent_sdk import AgentDefinition

from config import load_prompt

QA_AGENT = AgentDefinition(
    description="Answers questions about the codebase by searching semantically, reading source files, and explaining implementation details.",
    prompt=load_prompt("qa.md"),
    model="sonnet",
    tools=[
        "mcp__lancedb-code__search_code",
        "mcp__lancedb-code__index_status",
        "Read",
        "Grep",
        "Glob",
    ],
)
