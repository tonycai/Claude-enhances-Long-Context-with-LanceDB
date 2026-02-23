"""Acceptance test suite for the Claude Agent Team (Epic 5).

Maps to the acceptance criteria defined for Epic 5: Agent Team Orchestration.
Run with: cd agents && uv run pytest test_acceptance.py -v

Test IDs follow the convention: test_US{story}_{AC number}_{short_description}
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def agents_dir() -> Path:
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def project_root(agents_dir) -> Path:
    return agents_dir.parent


# =========================================================================
# EPIC 5: Agent Team Orchestration
# =========================================================================


class TestUS5_1_RouteQueriesToSpecialists:
    """US-5.1: Route queries to specialist agents."""

    def test_ac7_orchestrator_model(self):
        """AC-7: Orchestrator uses claude-sonnet-4-6."""
        from config import MODEL_ORCHESTRATOR

        assert MODEL_ORCHESTRATOR == "claude-sonnet-4-6"

    def test_ac8_orchestrator_allowed_tools(self):
        """AC-8: Orchestrator allowed tools are Task, Read, Glob, Grep."""
        # Verify the orchestrator.py defines these tools
        import orchestrator
        import inspect

        source = inspect.getsource(orchestrator.run)
        assert "Task" in source
        assert "Read" in source
        assert "Glob" in source
        assert "Grep" in source

    def test_all_agents_registered(self):
        """All 4 specialist agents are registered in ALL_AGENTS."""
        from agents import ALL_AGENTS

        assert set(ALL_AGENTS.keys()) == {"indexer", "searcher", "reviewer", "qa"}

    def test_all_agents_are_agent_definitions(self):
        """Each agent is a valid AgentDefinition."""
        from claude_agent_sdk import AgentDefinition

        from agents import ALL_AGENTS

        for name, agent in ALL_AGENTS.items():
            assert isinstance(agent, AgentDefinition), f"{name} not AgentDefinition"

    def test_orchestrator_prompt_contains_routing_rules(self):
        """The orchestrator prompt defines routing rules for all 4 agents."""
        from config import load_prompt

        prompt = load_prompt("orchestrator.md")
        assert "indexer" in prompt.lower()
        assert "searcher" in prompt.lower()
        assert "reviewer" in prompt.lower()
        assert "qa" in prompt.lower()
        assert "routing" in prompt.lower() or "route" in prompt.lower()

    def test_orchestrator_prompt_sequential_rule(self):
        """AC-5: The orchestrator prompt instructs sequential multi-step execution."""
        from config import load_prompt

        prompt = load_prompt("orchestrator.md")
        assert "sequential" in prompt.lower()

    def test_orchestrator_prompt_fallback_rule(self):
        """AC-6: The orchestrator defaults to qa for questions, searcher for code."""
        from config import load_prompt

        prompt = load_prompt("orchestrator.md")
        assert "qa" in prompt
        assert "searcher" in prompt


class TestUS5_2_IndexViaAgent:
    """US-5.2: Index via agent."""

    def test_ac1_indexer_model_haiku(self):
        """AC-1: Indexer agent uses 'haiku' model."""
        from agents import INDEXER_AGENT

        assert INDEXER_AGENT.model == "haiku"

    def test_ac2_indexer_has_exactly_3_tools(self):
        """AC-2: Indexer has index_files, index_status, remove_files."""
        from agents import INDEXER_AGENT

        expected_tools = {
            "mcp__lancedb-code__index_files",
            "mcp__lancedb-code__index_status",
            "mcp__lancedb-code__remove_files",
        }
        assert set(INDEXER_AGENT.tools) == expected_tools

    def test_ac3_indexer_prompt_full_scan(self):
        """AC-3: Indexer prompt instructs full-repo indexing with no arguments."""
        from config import load_prompt

        prompt = load_prompt("indexer.md")
        assert "no arg" in prompt.lower() or "no arguments" in prompt.lower()

    def test_ac4_indexer_prompt_specific_paths(self):
        """AC-4: Indexer prompt instructs passing paths for specific files."""
        from config import load_prompt

        prompt = load_prompt("indexer.md")
        assert "path" in prompt.lower()

    def test_ac5_indexer_prompt_reports_results(self):
        """AC-5: Indexer prompt instructs reporting results clearly."""
        from config import load_prompt

        prompt = load_prompt("indexer.md")
        assert "report" in prompt.lower()

    def test_indexer_has_nonempty_description(self):
        """Indexer agent has a non-empty description."""
        from agents import INDEXER_AGENT

        assert len(INDEXER_AGENT.description) > 20

    def test_indexer_has_nonempty_prompt(self):
        """Indexer agent has a non-empty prompt."""
        from agents import INDEXER_AGENT

        assert len(INDEXER_AGENT.prompt) > 50


class TestUS5_3_SearchViaAgent:
    """US-5.3: Search via agent."""

    def test_ac1_searcher_model_sonnet(self):
        """AC-1: Searcher agent uses 'sonnet' model."""
        from agents import SEARCHER_AGENT

        assert SEARCHER_AGENT.model == "sonnet"

    def test_ac2_searcher_tools(self):
        """AC-2: Searcher has search_code, index_status, Read, Grep, Glob."""
        from agents import SEARCHER_AGENT

        expected = {
            "mcp__lancedb-code__search_code",
            "mcp__lancedb-code__index_status",
            "Read",
            "Grep",
            "Glob",
        }
        assert set(SEARCHER_AGENT.tools) == expected

    def test_ac3_searcher_prompt_translates_queries(self):
        """AC-3: Searcher prompt instructs translating natural language to search calls."""
        from config import load_prompt

        prompt = load_prompt("searcher.md")
        assert "translate" in prompt.lower() or "natural language" in prompt.lower()

    def test_ac4_searcher_prompt_default_hybrid(self):
        """AC-4: Searcher prompt defaults to hybrid search."""
        from config import load_prompt

        prompt = load_prompt("searcher.md")
        assert "hybrid" in prompt.lower()

    def test_ac5_searcher_prompt_uses_read_for_full_context(self):
        """AC-5: Searcher prompt instructs using Read for full file context."""
        from config import load_prompt

        prompt = load_prompt("searcher.md")
        assert "Read" in prompt

    def test_ac6_searcher_prompt_presents_file_paths(self):
        """AC-6: Searcher presents results with file paths and line ranges."""
        from config import load_prompt

        prompt = load_prompt("searcher.md")
        assert "file path" in prompt.lower() or "file paths" in prompt.lower()
        assert "line" in prompt.lower()


class TestUS5_4_ReviewViaAgent:
    """US-5.4: Review code via agent."""

    def test_ac1_reviewer_model_opus(self):
        """AC-1: Reviewer agent uses 'opus' model."""
        from agents import REVIEWER_AGENT

        assert REVIEWER_AGENT.model == "opus"

    def test_ac2_reviewer_tools(self):
        """AC-2: Reviewer has search_code, Read, Grep, Glob."""
        from agents import REVIEWER_AGENT

        expected = {
            "mcp__lancedb-code__search_code",
            "Read",
            "Grep",
            "Glob",
        }
        assert set(REVIEWER_AGENT.tools) == expected

    def test_ac3_reviewer_prompt_covers_5_dimensions(self):
        """AC-3: Reviewer prompt covers security, quality, error handling,
        performance, maintainability."""
        from config import load_prompt

        prompt = load_prompt("reviewer.md").lower()
        assert "security" in prompt
        assert "quality" in prompt
        assert "error handling" in prompt
        assert "performance" in prompt
        assert "maintainability" in prompt

    def test_ac4_reviewer_prompt_severity_levels(self):
        """AC-4: Reviewer organizes findings by severity."""
        from config import load_prompt

        prompt = load_prompt("reviewer.md").lower()
        assert "critical" in prompt
        assert "warning" in prompt
        assert "suggestion" in prompt

    def test_ac5_reviewer_prompt_finding_format(self):
        """AC-5: Each finding includes file, severity, issue, recommendation."""
        from config import load_prompt

        prompt = load_prompt("reviewer.md")
        assert "File" in prompt or "file" in prompt
        assert "Severity" in prompt or "severity" in prompt
        assert "Issue" in prompt or "issue" in prompt
        assert "Recommendation" in prompt or "recommendation" in prompt

    def test_ac6_reviewer_prompt_grep_for_smells(self):
        """AC-6: Reviewer uses Grep for TODO/FIXME and code smells."""
        from config import load_prompt

        prompt = load_prompt("reviewer.md")
        assert "TODO" in prompt or "FIXME" in prompt
        assert "Grep" in prompt


class TestUS5_5_QAViaAgent:
    """US-5.5: Ask codebase questions via agent."""

    def test_ac1_qa_model_sonnet(self):
        """AC-1: Q&A agent uses 'sonnet' model."""
        from agents import QA_AGENT

        assert QA_AGENT.model == "sonnet"

    def test_ac2_qa_tools(self):
        """AC-2: Q&A has search_code, index_status, Read, Grep, Glob."""
        from agents import QA_AGENT

        expected = {
            "mcp__lancedb-code__search_code",
            "mcp__lancedb-code__index_status",
            "Read",
            "Grep",
            "Glob",
        }
        assert set(QA_AGENT.tools) == expected

    def test_ac3_qa_prompt_no_speculation(self):
        """AC-3: Q&A prompt instructs no speculation."""
        from config import load_prompt

        prompt = load_prompt("qa.md").lower()
        assert "speculate" in prompt or "not speculate" in prompt

    def test_ac4_qa_prompt_response_format(self):
        """AC-4: Q&A response format includes summary, references, explanation."""
        from config import load_prompt

        prompt = load_prompt("qa.md").lower()
        assert "summary" in prompt
        assert "reference" in prompt or "file path" in prompt
        assert "explain" in prompt or "explanation" in prompt

    def test_ac5_qa_prompt_architecture_workflow(self):
        """AC-5: Q&A checks index_status for architecture questions."""
        from config import load_prompt

        prompt = load_prompt("qa.md")
        assert "index_status" in prompt or "architecture" in prompt.lower()


class TestUS5_6_EditablePrompts:
    """US-5.6: Editable prompts without code changes."""

    def test_ac1_all_5_prompts_exist(self, agents_dir):
        """AC-1: All 5 prompt files exist in agents/prompts/."""
        prompts_dir = agents_dir / "prompts"
        expected = ["orchestrator.md", "indexer.md", "searcher.md", "reviewer.md", "qa.md"]
        for name in expected:
            path = prompts_dir / name
            assert path.exists(), f"Missing prompt: {path}"

    def test_ac1_all_prompts_are_markdown(self, agents_dir):
        """AC-1: All prompt files have .md extension."""
        prompts_dir = agents_dir / "prompts"
        for f in prompts_dir.iterdir():
            if f.is_file() and not f.name.startswith("."):
                assert f.suffix == ".md", f"Non-markdown prompt: {f}"

    def test_ac2_load_prompt_strips_whitespace(self):
        """AC-2: load_prompt returns content stripped of leading/trailing whitespace."""
        from config import load_prompt

        for name in ["orchestrator.md", "indexer.md", "searcher.md", "reviewer.md", "qa.md"]:
            text = load_prompt(name)
            assert text == text.strip(), f"Prompt {name} not stripped"

    def test_ac3_missing_prompt_raises_error(self):
        """AC-3: Missing prompt file raises FileNotFoundError."""
        from config import load_prompt

        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_prompt.md")

    def test_ac3_error_includes_full_path(self):
        """AC-3: FileNotFoundError message includes full path."""
        from config import PROMPTS_DIR, load_prompt

        try:
            load_prompt("missing.md")
            pytest.fail("Should have raised FileNotFoundError")
        except FileNotFoundError as e:
            assert str(PROMPTS_DIR / "missing.md") in str(e)

    def test_all_prompts_have_substantial_content(self):
        """All prompts have meaningful content (>50 chars)."""
        from config import load_prompt

        for name in ["orchestrator.md", "indexer.md", "searcher.md", "reviewer.md", "qa.md"]:
            text = load_prompt(name)
            assert len(text) > 50, f"Prompt {name} too short: {len(text)} chars"


# =========================================================================
# Agent Config
# =========================================================================


class TestAgentConfig:
    """Configuration for the agent team."""

    def test_project_root_exists(self, project_root):
        """PROJECT_ROOT points to an existing directory."""
        from config import PROJECT_ROOT

        assert PROJECT_ROOT.exists()
        assert PROJECT_ROOT.is_dir()

    def test_agents_dir_exists(self, agents_dir):
        """AGENTS_DIR points to an existing directory."""
        from config import AGENTS_DIR

        assert AGENTS_DIR.exists()
        assert AGENTS_DIR.is_dir()

    def test_mcp_server_dir_exists(self):
        """MCP_SERVER_DIR points to an existing directory."""
        from config import MCP_SERVER_DIR

        assert MCP_SERVER_DIR.exists()

    def test_prompts_dir_exists(self):
        """PROMPTS_DIR points to an existing directory."""
        from config import PROMPTS_DIR

        assert PROMPTS_DIR.exists()

    def test_mcp_servers_has_lancedb_code(self):
        """MCP_SERVERS includes 'lancedb-code' with stdio transport."""
        from config import MCP_SERVERS

        assert "lancedb-code" in MCP_SERVERS
        server = MCP_SERVERS["lancedb-code"]
        assert server["type"] == "stdio"
        assert "command" in server
        assert "args" in server
        assert isinstance(server["args"], list)

    def test_env_var_override_mcp_command(self):
        """LANCEDB_MCP_COMMAND env var overrides MCP_SERVERS."""
        import config

        os.environ["LANCEDB_MCP_COMMAND"] = "docker run -i --rm lancedb:test"
        try:
            importlib.reload(config)
            server = config.MCP_SERVERS["lancedb-code"]
            assert server["command"] == "bash"
            assert server["args"] == ["-c", "docker run -i --rm lancedb:test"]
        finally:
            del os.environ["LANCEDB_MCP_COMMAND"]
            importlib.reload(config)

    def test_model_assignments(self):
        """Model IDs are correctly assigned per agent role."""
        from config import MODEL_INDEXER, MODEL_ORCHESTRATOR, MODEL_QA, MODEL_REVIEWER, MODEL_SEARCHER

        assert MODEL_ORCHESTRATOR == "claude-sonnet-4-6"
        assert MODEL_INDEXER == "claude-haiku-4-5"
        assert MODEL_SEARCHER == "claude-sonnet-4-6"
        assert MODEL_REVIEWER == "claude-opus-4-6"
        assert MODEL_QA == "claude-sonnet-4-6"

    def test_setup_logging_default(self):
        """setup_logging with default (debug=False) targets WARNING level."""
        import logging

        from config import setup_logging

        # Clear existing handlers so basicConfig takes effect.
        root = logging.getLogger()
        root.handlers.clear()
        setup_logging(debug=False)
        assert root.level == logging.WARNING

    def test_setup_logging_debug(self):
        """setup_logging with debug=True targets DEBUG level."""
        import logging

        from config import setup_logging

        # Clear existing handlers so basicConfig takes effect.
        root = logging.getLogger()
        root.handlers.clear()
        setup_logging(debug=True)
        assert root.level == logging.DEBUG


# =========================================================================
# Orchestrator module
# =========================================================================


class TestOrchestrator:
    """Orchestrator entry point validation."""

    def test_exports_run_and_main(self):
        """orchestrator module exports run() and main()."""
        import orchestrator

        assert hasattr(orchestrator, "run")
        assert hasattr(orchestrator, "main")
        assert callable(orchestrator.run)
        assert callable(orchestrator.main)

    def test_run_is_async(self):
        """orchestrator.run is an async function."""
        import asyncio
        import inspect

        import orchestrator

        assert inspect.iscoroutinefunction(orchestrator.run)

    def test_main_handles_no_args(self, capsys):
        """main() with no args prints usage and exits."""
        import orchestrator

        original_argv = sys.argv
        try:
            sys.argv = ["orchestrator.py"]
            with pytest.raises(SystemExit) as exc_info:
                orchestrator.main()
            assert exc_info.value.code == 1
        finally:
            sys.argv = original_argv
