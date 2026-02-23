"""Orchestrator entry point for the LanceDB agent team.

Usage:
    uv run python orchestrator.py [--debug] "Index the repository"
    uv run python orchestrator.py "Find all authentication-related code"
    uv run python orchestrator.py "Review the server module for issues"
    uv run python orchestrator.py "How does the chunking pipeline work?"
"""

from __future__ import annotations

import asyncio
import logging
import sys

from claude_agent_sdk import CLINotFoundError, ClaudeAgentOptions, ProcessError, query

from agents import ALL_AGENTS
from config import MCP_SERVERS, MODEL_ORCHESTRATOR, PROJECT_ROOT, load_prompt, setup_logging

logger = logging.getLogger(__name__)


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
            if message.is_error:
                print(f"Error: {message.result}", file=sys.stderr)
            else:
                print(message.result)
            if message.total_cost_usd is not None:
                logger.debug("Total cost: $%.4f", message.total_cost_usd)
        elif message.type == "assistant":
            for block in getattr(message, "content", []):
                if hasattr(block, "text"):
                    print(block.text, end="", flush=True)
        else:
            logger.debug("Unhandled message type: %s", message.type)


def main() -> None:
    # Parse --debug flag before the query.
    args = sys.argv[1:]
    debug = False
    if "--debug" in args:
        debug = True
        args.remove("--debug")

    setup_logging(debug=debug)

    if not args:
        print("Usage: uv run python orchestrator.py [--debug] <query>")
        print()
        print("Examples:")
        print('  uv run python orchestrator.py "Index the repository"')
        print('  uv run python orchestrator.py "Find all error handling code"')
        print('  uv run python orchestrator.py "Review server.py for issues"')
        print('  uv run python orchestrator.py "How does search work?"')
        print()
        print("Options:")
        print("  --debug    Enable debug logging to stderr")
        sys.exit(1)

    user_query = " ".join(args)
    try:
        asyncio.run(run(user_query))
    except CLINotFoundError:
        print("Error: Claude Code CLI not found.", file=sys.stderr)
        print("Install: npm install -g @anthropic-ai/claude-code", file=sys.stderr)
        sys.exit(1)
    except ProcessError as e:
        print(f"Error: Agent process failed: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
