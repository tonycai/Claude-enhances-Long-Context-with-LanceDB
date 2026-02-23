"""Agent definitions for the LanceDB code search team."""

from agents.indexer import INDEXER_AGENT
from agents.qa import QA_AGENT
from agents.reviewer import REVIEWER_AGENT
from agents.searcher import SEARCHER_AGENT

ALL_AGENTS = {
    "indexer": INDEXER_AGENT,
    "searcher": SEARCHER_AGENT,
    "reviewer": REVIEWER_AGENT,
    "qa": QA_AGENT,
}

__all__ = [
    "INDEXER_AGENT",
    "SEARCHER_AGENT",
    "REVIEWER_AGENT",
    "QA_AGENT",
    "ALL_AGENTS",
]
