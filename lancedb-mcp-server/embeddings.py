"""Embedding function setup, including multilingual models that need prefixes.

Some retrieval models expect different text for queries and for documents.
The E5 family (for example ``intfloat/multilingual-e5-base``) is trained with
``"query: "`` in front of search queries and ``"passage: "`` in front of
indexed text; leaving the prefixes out lowers retrieval quality.

LanceDB's built-in ``sentence-transformers`` function embeds queries and
documents the same way, so this module registers a small subclass that adds
the prefixes. The prefixes are saved in the table metadata together with the
model name, so a table always embeds queries with the settings it was built
with.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from lancedb.embeddings import get_registry, register
from lancedb.embeddings.sentence_transformers import SentenceTransformerEmbeddings

logger = logging.getLogger(__name__)

PREFIXED_ST_ALIAS = "sentence-transformers-prefixed"


def default_prefixes(model_name: str) -> tuple[str, str]:
    """Return the (query, passage) prefixes a model expects by default."""
    name = model_name.lower()
    if "e5" in name.split("/")[-1]:
        return "query: ", "passage: "
    return "", ""


@register(PREFIXED_ST_ALIAS)
class PrefixedSentenceTransformerEmbeddings(SentenceTransformerEmbeddings):
    """sentence-transformers embeddings with optional query/passage prefixes."""

    query_prefix: str = ""
    passage_prefix: str = ""

    def compute_query_embeddings(self, query: str, *args, **kwargs):
        texts = self.sanitize_input(query)
        return self.generate_embeddings([self.query_prefix + t for t in texts])

    def compute_source_embeddings(self, texts, *args, **kwargs):
        texts = self.sanitize_input(texts)
        return self.generate_embeddings([self.passage_prefix + t for t in texts])


def build_embedding_function(config) -> Any:
    """Create the embedding function described by *config*."""
    registry = get_registry()
    if config.embedding_provider == "sentence-transformers":
        return registry.get(PREFIXED_ST_ALIAS).create(
            name=config.embedding_model,
            query_prefix=config.query_prefix,
            passage_prefix=config.passage_prefix,
            # Models are pre-downloaded into the image; never run code
            # shipped inside a model repository.
            trust_remote_code=False,
        )
    return registry.get(config.embedding_provider).create(
        name=config.embedding_model
    )


def embedding_signature_of_function(func: Any) -> tuple[str, str, str]:
    """(model name, query prefix, passage prefix) for an embedding function."""
    return (
        str(getattr(func, "name", "")),
        str(getattr(func, "query_prefix", "") or ""),
        str(getattr(func, "passage_prefix", "") or ""),
    )


def embedding_signature_of_table(table: Any) -> tuple[str, str, str] | None:
    """Read (model name, query prefix, passage prefix) from a table's metadata.

    Returns ``None`` if the table has no embedding function recorded or the
    metadata cannot be read. Reads the raw JSON so that the model is not
    loaded just to compare settings.
    """
    try:
        metadata = table.schema.metadata or {}
        raw = metadata.get(b"embedding_functions") or metadata.get(
            "embedding_functions"
        )
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        entries = json.loads(raw)
        if not entries:
            return None
        model = entries[0].get("model", {}) or {}
        return (
            str(model.get("name", "")),
            str(model.get("query_prefix", "") or ""),
            str(model.get("passage_prefix", "") or ""),
        )
    except Exception as exc:  # noqa: BLE001 - metadata format is best effort
        logger.debug("Could not read embedding metadata: %s", exc)
        return None
