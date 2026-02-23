"""Validation tests for the LanceDB agent team.

Checks imports, configuration, prompt loading, and agent definitions
without requiring an API key or running the MCP server.

Usage:
    cd agents && uv run python test_agents.py
"""

from __future__ import annotations

import sys
import traceback

# Track results.
passed = 0
failed = 0


def check(name: str, fn):
    """Run a check function and report pass/fail."""
    global passed, failed
    try:
        fn()
        print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc(limit=2)
        failed += 1


# -----------------------------------------------------------------------
# 1. Config imports and paths
# -----------------------------------------------------------------------

print("\n--- Config ---")


def test_config_imports():
    from config import (
        AGENTS_DIR,
        MCP_SERVER_DIR,
        MCP_SERVERS,
        MODEL_INDEXER,
        MODEL_ORCHESTRATOR,
        MODEL_QA,
        MODEL_REVIEWER,
        MODEL_SEARCHER,
        PROJECT_ROOT,
        PROMPTS_DIR,
        load_prompt,
    )

    assert AGENTS_DIR.exists(), f"AGENTS_DIR does not exist: {AGENTS_DIR}"
    assert PROJECT_ROOT.exists(), f"PROJECT_ROOT does not exist: {PROJECT_ROOT}"
    assert MCP_SERVER_DIR.exists(), f"MCP_SERVER_DIR does not exist: {MCP_SERVER_DIR}"
    assert PROMPTS_DIR.exists(), f"PROMPTS_DIR does not exist: {PROMPTS_DIR}"
    assert "lancedb-code" in MCP_SERVERS


check("config imports and paths", test_config_imports)


# -----------------------------------------------------------------------
# 2. Prompt loading
# -----------------------------------------------------------------------

print("\n--- Prompts ---")

EXPECTED_PROMPTS = [
    "orchestrator.md",
    "indexer.md",
    "searcher.md",
    "reviewer.md",
    "qa.md",
]


def test_prompts_exist():
    from config import PROMPTS_DIR

    for name in EXPECTED_PROMPTS:
        path = PROMPTS_DIR / name
        assert path.exists(), f"Missing prompt: {path}"
        content = path.read_text()
        assert len(content) > 50, f"Prompt too short ({len(content)} chars): {name}"


check("all prompt files exist and have content", test_prompts_exist)


def test_load_prompt():
    from config import load_prompt

    for name in EXPECTED_PROMPTS:
        text = load_prompt(name)
        assert isinstance(text, str)
        assert len(text) > 50


check("load_prompt returns non-empty strings", test_load_prompt)


def test_load_prompt_missing():
    from config import load_prompt

    try:
        load_prompt("nonexistent.md")
        raise AssertionError("Should have raised FileNotFoundError")
    except FileNotFoundError:
        pass


check("load_prompt raises FileNotFoundError for missing files", test_load_prompt_missing)


# -----------------------------------------------------------------------
# 3. Agent SDK imports
# -----------------------------------------------------------------------

print("\n--- Agent SDK ---")


def test_sdk_imports():
    from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions, query

    assert callable(query)
    assert AgentDefinition is not None
    assert ClaudeAgentOptions is not None


check("claude_agent_sdk core imports", test_sdk_imports)


# -----------------------------------------------------------------------
# 4. Agent definitions
# -----------------------------------------------------------------------

print("\n--- Agent Definitions ---")


def test_agent_definitions():
    from claude_agent_sdk import AgentDefinition

    from agents import ALL_AGENTS

    assert isinstance(ALL_AGENTS, dict), "ALL_AGENTS should be a dict"
    assert set(ALL_AGENTS.keys()) == {"indexer", "searcher", "reviewer", "qa"}

    for name, agent in ALL_AGENTS.items():
        assert isinstance(agent, AgentDefinition), f"{name} is not an AgentDefinition"
        assert agent.description, f"{name} has empty description"
        assert agent.prompt, f"{name} has empty prompt"
        assert agent.tools, f"{name} has empty tools list"


check("ALL_AGENTS dict with 4 valid AgentDefinitions", test_agent_definitions)


def test_indexer_tools():
    from agents import INDEXER_AGENT

    assert "mcp__lancedb-code__index_files" in INDEXER_AGENT.tools
    assert "mcp__lancedb-code__index_status" in INDEXER_AGENT.tools
    assert "mcp__lancedb-code__remove_files" in INDEXER_AGENT.tools


check("indexer has correct MCP tools", test_indexer_tools)


def test_searcher_tools():
    from agents import SEARCHER_AGENT

    assert "mcp__lancedb-code__search_code" in SEARCHER_AGENT.tools
    assert "Read" in SEARCHER_AGENT.tools


check("searcher has search_code and Read tools", test_searcher_tools)


def test_reviewer_tools():
    from agents import REVIEWER_AGENT

    assert "mcp__lancedb-code__search_code" in REVIEWER_AGENT.tools
    assert "Read" in REVIEWER_AGENT.tools
    assert "Grep" in REVIEWER_AGENT.tools


check("reviewer has search_code, Read, and Grep tools", test_reviewer_tools)


def test_qa_tools():
    from agents import QA_AGENT

    assert "mcp__lancedb-code__search_code" in QA_AGENT.tools
    assert "mcp__lancedb-code__index_status" in QA_AGENT.tools
    assert "Read" in QA_AGENT.tools


check("qa has search_code, index_status, and Read tools", test_qa_tools)


# -----------------------------------------------------------------------
# 5. Orchestrator module
# -----------------------------------------------------------------------

print("\n--- Orchestrator ---")


def test_orchestrator_imports():
    import orchestrator

    assert hasattr(orchestrator, "run"), "orchestrator missing run()"
    assert hasattr(orchestrator, "main"), "orchestrator missing main()"
    assert callable(orchestrator.run)
    assert callable(orchestrator.main)


check("orchestrator module exports run() and main()", test_orchestrator_imports)


# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------

print(f"\n{'=' * 40}")
print(f"Results: {passed} passed, {failed} failed")
print(f"{'=' * 40}")

if failed > 0:
    sys.exit(1)
