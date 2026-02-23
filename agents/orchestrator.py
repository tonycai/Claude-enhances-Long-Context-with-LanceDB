"""Orchestrator entry point for the LanceDB agent team.

Usage:
    uv run python orchestrator.py "Index the repository"
    uv run python orchestrator.py "Find all authentication-related code"
    uv run python orchestrator.py "Review the server module for issues"
    uv run python orchestrator.py "How does the chunking pipeline work?"
"""

from __future__ import annotations

import asyncio
import sys

from claude_agent_sdk import ClaudeAgentOptions, query

from agents import ALL_AGENTS
from config import MCP_SERVERS, MODEL_ORCHESTRATOR, PROJECT_ROOT, load_prompt


async def run(prompt: str) -> None:
    """Run the orchestrator with a user query, streaming output to stdout."""
    async for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            system_prompt=load_prompt("orchestrator.md"),
            model=MODEL_ORCHESTRATOR,
            cwd=str(PROJECT_ROOT),
            allowed_tools=["Task", "Read", "Glob", "Grep"],
            mcp_servers=MCP_SERVERS,
            agents=ALL_AGENTS,
            permission_mode="acceptEdits",
            max_turns=30,
        ),
    ):
        if message.type == "result":
            print(message.result)
        elif message.type == "assistant":
            for block in getattr(message, "content", []):
                if hasattr(block, "text"):
                    print(block.text, end="", flush=True)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python orchestrator.py <query>")
        print()
        print("Examples:")
        print('  uv run python orchestrator.py "Index the repository"')
        print('  uv run python orchestrator.py "Find all error handling code"')
        print('  uv run python orchestrator.py "Review server.py for issues"')
        print('  uv run python orchestrator.py "How does search work?"')
        sys.exit(1)

    user_query = " ".join(sys.argv[1:])
    asyncio.run(run(user_query))


if __name__ == "__main__":
    main()
