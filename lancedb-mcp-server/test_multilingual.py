"""Offline tests for multilingual search and Swift support.

No model download is needed: the embedding model is replaced by a small
hash-based stand-in, so these tests check the plumbing (chunking, prefixes,
table metadata, CJK full-text search), not embedding quality.

Run with: cd lancedb-mcp-server && uv run pytest test_multilingual.py -v
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import lancedb as ldb
import numpy as np
import pytest

import chunker
from chunker import _chunk_by_lines, chunk_file, estimate_tokens
from config import FALLBACK_CHUNK_LINES, FALLBACK_OVERLAP_LINES, Config
from embeddings import (
    PrefixedSentenceTransformerEmbeddings,
    build_embedding_function,
    default_prefixes,
    embedding_signature_of_function,
    embedding_signature_of_table,
)
from indexer import discover_files

DIMS = 32


def _hash_vector(text: str) -> list[float]:
    v = np.zeros(DIMS, dtype=np.float32)
    t = text.lower()
    for n in (2, 3):
        for i in range(max(1, len(t) - n + 1)):
            h = int(hashlib.md5(t[i:i + n].encode("utf-8")).hexdigest(), 16)
            v[h % DIMS] += 1.0 if (h >> 8) & 1 else -1.0
    norm = np.linalg.norm(v)
    return (v / norm if norm else v).tolist()


@pytest.fixture()
def fake_model(monkeypatch):
    """Replace the sentence-transformers model with a hash stand-in."""
    seen: list[str] = []

    def generate(self, texts):
        texts = [str(t) for t in texts]
        seen.extend(texts)
        return [_hash_vector(t) for t in texts]

    monkeypatch.setattr(PrefixedSentenceTransformerEmbeddings, "generate_embeddings", generate)
    return seen


def _config(model: str, **overrides) -> Config:
    cfg = Config()
    cfg.embedding_provider = "sentence-transformers"
    cfg.embedding_model = model
    cfg.query_prefix_override = None
    cfg.passage_prefix_override = None
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


# ---------------------------------------------------------------------------
# Swift chunking
# ---------------------------------------------------------------------------

SWIFT_SOURCE = '''import SwiftUI
import Foundation

/// 字幕面板：只在内存里保存，不写入磁盘
struct CaptionsView: View {
    let lines: [String]

    // 主界面
    var body: some View {
        VStack {
            Text("hello")
        }
    }

    init(lines: [String]) {
        self.lines = lines
    }
}

/// 面接の練習モード
protocol PracticeEngine {
    func nextQuestion() -> String
}

extension CaptionsView {
    /// 清空字幕
    func clear() {
        print("clear")
    }

    subscript(i: Int) -> String { lines[i] }
}

enum Mode: String {
    case captions, practice

    func label() -> String { rawValue }
}

final class Meter {
    deinit {}
}

// 计算延迟
func latency(ms: Int) -> Double {
    return Double(ms) / 1000.0
}

let defaultMode = Mode.captions
'''


@pytest.fixture()
def swift_chunks(tmp_path):
    path = tmp_path / "Captions.swift"
    path.write_text(SWIFT_SOURCE, encoding="utf-8")
    return chunk_file(str(path), tmp_path)


def _by_symbol(chunks):
    return {c.symbol_name: c for c in chunks}


def test_swift_types_and_members(swift_chunks):
    by = _by_symbol(swift_chunks)
    assert all(c.language == "swift" for c in swift_chunks)

    assert by["CaptionsView"].node_type in ("struct", "impl")
    assert by["PracticeEngine"].node_type == "interface"
    assert by["Mode"].node_type == "enum"
    assert by["Meter"].node_type == "class"
    assert by["latency"].node_type == "function"

    # SwiftUI body is a computed property: indexed on its own.
    assert by["CaptionsView.body"].node_type == "method"
    assert "VStack" in by["CaptionsView.body"].text
    assert by["CaptionsView.init"].node_type == "method"
    assert by["CaptionsView.clear"].node_type == "method"
    assert by["CaptionsView.subscript"].node_type == "method"
    assert by["Mode.label"].node_type == "method"
    assert by["Meter.deinit"].node_type == "method"
    assert by["PracticeEngine.nextQuestion"].node_type == "method"

    # Single-line stored properties stay inside the type chunk.
    assert "CaptionsView.lines" not in by


def test_swift_extension_is_impl(swift_chunks):
    impls = [c for c in swift_chunks if c.node_type == "impl"]
    assert len(impls) == 1
    assert impls[0].symbol_name == "CaptionsView"
    assert "func clear" in impls[0].text


def test_swift_doc_comments_kept_with_declaration(swift_chunks):
    by = _by_symbol(swift_chunks)
    struct = next(c for c in swift_chunks if c.node_type == "struct")
    assert struct.text.startswith("/// 字幕面板")
    assert struct.start_line == 4
    assert by["CaptionsView.body"].text.startswith("// 主界面")
    assert by["CaptionsView.clear"].text.startswith("/// 清空字幕")
    assert by["PracticeEngine"].text.startswith("/// 面接の練習モード")
    assert by["latency"].text.startswith("// 计算延迟")


def test_swift_module_level_code_captured(swift_chunks):
    module = [c for c in swift_chunks if c.node_type == "module"]
    assert len(module) == 1
    assert "import SwiftUI" in module[0].text
    assert "let defaultMode" in module[0].text
    # Doc comments moved into their declarations, not duplicated here.
    assert "字幕面板" not in module[0].text


def test_swift_and_localization_files_discovered(tmp_path):
    (tmp_path / "App.swift").write_text("struct A {}\n")
    (tmp_path / "Localizable.xcstrings").write_text('{"strings": {}}\n')
    (tmp_path / ".env").write_text("SECRET=1\n")
    derived = tmp_path / "DerivedData"
    derived.mkdir()
    (derived / "Gen.swift").write_text("struct G {}\n")

    found = {p.relative_to(tmp_path).as_posix() for p in discover_files(tmp_path)}
    assert found == {"App.swift", "Localizable.xcstrings"}


# ---------------------------------------------------------------------------
# Line-based chunks and the token budget
# ---------------------------------------------------------------------------


def test_estimate_tokens_counts_cjk_per_character():
    assert estimate_tokens("abcd" * 10) == 10
    assert estimate_tokens("字幕不写入磁盘") == 7


def test_token_budget_limits_cjk_chunks(monkeypatch):
    monkeypatch.setattr(chunker, "CHUNK_TOKEN_BUDGET", 480)
    lines = [f"第{i}行：真实面试只显示字幕和翻译，不调用任何模型，字幕只在内存里。" for i in range(120)]
    chunks = _chunk_by_lines("\n".join(lines), "docs/prd.md", "md")

    assert len(chunks) > 1
    for c in chunks:
        assert estimate_tokens(c.text) <= 480 or c.start_line == c.end_line
    covered = set()
    for c in chunks:
        covered.update(range(c.start_line, c.end_line + 1))
    assert covered == set(range(1, len(lines) + 1))
    starts = [c.start_line for c in chunks]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)


def test_no_budget_keeps_original_windows(monkeypatch):
    monkeypatch.setattr(chunker, "CHUNK_TOKEN_BUDGET", 0)
    text = "\n".join(f"line {i}" for i in range(130))
    chunks = _chunk_by_lines(text, "notes.txt", "txt")
    step = FALLBACK_CHUNK_LINES - FALLBACK_OVERLAP_LINES
    assert [c.start_line for c in chunks] == [1, 1 + step, 1 + 2 * step]
    assert chunks[0].end_line == FALLBACK_CHUNK_LINES


# ---------------------------------------------------------------------------
# Embedding prefixes and table metadata
# ---------------------------------------------------------------------------


def test_default_prefixes():
    assert default_prefixes("intfloat/multilingual-e5-base") == ("query: ", "passage: ")
    assert default_prefixes("intfloat/multilingual-e5-small") == ("query: ", "passage: ")
    assert default_prefixes("BAAI/bge-m3") == ("", "")
    assert default_prefixes("all-MiniLM-L6-v2") == ("", "")


def test_prefix_overrides():
    cfg = _config("intfloat/multilingual-e5-base", query_prefix_override="", passage_prefix_override="")
    assert (cfg.query_prefix, cfg.passage_prefix) == ("", "")


def _table(tmp_path, cfg, name="t"):
    from server import _build_schema, _open_or_create_table

    func = build_embedding_function(cfg)
    db = ldb.connect(str(tmp_path / "db"))
    table = _open_or_create_table(db, name, _build_schema(func), embedding_signature_of_function(func))
    return db, table, func


def _row(text, i):
    return {
        "text": text, "chunk_id": f"c{i}", "file_path": f"f{i}.md",
        "start_line": 1, "end_line": 1, "language": "md",
        "node_type": "block", "symbol_name": "", "content_hash": "h",
    }


def test_e5_prefixes_applied_and_saved(tmp_path, fake_model):
    cfg = _config("intfloat/multilingual-e5-base")
    _db, table, _func = _table(tmp_path, cfg)
    fake_model.clear()

    table.add([_row("字幕只在内存里", 0), _row("面接の練習", 1)])
    assert fake_model == ["passage: 字幕只在内存里", "passage: 面接の練習"]

    fake_model.clear()
    table.search("字幕", query_type="vector").limit(1).to_list()
    assert fake_model == ["query: 字幕"]

    assert embedding_signature_of_table(table) == (
        "intfloat/multilingual-e5-base", "query: ", "passage: ",
    )


def test_model_change_recreates_table(tmp_path, fake_model):
    from server import _build_schema, _open_or_create_table

    db, table, _ = _table(tmp_path, _config("all-MiniLM-L6-v2"))
    table.add([_row("hello", 0)])
    assert table.count_rows() == 1

    # Same model: data kept.
    func = build_embedding_function(_config("all-MiniLM-L6-v2"))
    same = _open_or_create_table(db, "t", _build_schema(func), embedding_signature_of_function(func))
    assert same.count_rows() == 1

    # Different model: old vectors are useless, table starts empty.
    func = build_embedding_function(_config("intfloat/multilingual-e5-base"))
    new = _open_or_create_table(db, "t", _build_schema(func), embedding_signature_of_function(func))
    assert new.count_rows() == 0
    assert embedding_signature_of_table(new)[0] == "intfloat/multilingual-e5-base"


def test_legacy_table_kept_with_default_model(tmp_path, fake_model, monkeypatch):
    """A table built by the old server (plain sentence-transformers, MiniLM)
    is kept when the server still uses MiniLM."""
    from lancedb.embeddings import get_registry
    from lancedb.embeddings.sentence_transformers import SentenceTransformerEmbeddings
    from server import _build_schema, _open_or_create_table

    monkeypatch.setattr(
        SentenceTransformerEmbeddings, "generate_embeddings",
        lambda self, texts: [_hash_vector(str(t)) for t in texts],
    )
    legacy = get_registry().get("sentence-transformers").create(name="all-MiniLM-L6-v2")
    db = ldb.connect(str(tmp_path / "db"))
    old = db.create_table("legacy", schema=_build_schema(legacy))
    old.add([_row("hello", 0)])
    assert embedding_signature_of_table(old) == ("all-MiniLM-L6-v2", "", "")

    func = build_embedding_function(_config("all-MiniLM-L6-v2"))
    table = _open_or_create_table(db, "legacy", _build_schema(func), embedding_signature_of_function(func))
    assert table.count_rows() == 1


# ---------------------------------------------------------------------------
# Full-text search for Chinese and Japanese
# ---------------------------------------------------------------------------

DOCS = [
    "真实面试时字幕绝不写入磁盘，崩溃恢复只恢复模式和连接。",
    "面接の練習モードでは、回答の候補を三つ表示します。",
    "The client sends a keepalive message every 10 seconds.",
]


def _fts_table(tmp_path, tokenizer):
    cfg = _config("all-MiniLM-L6-v2", fts_tokenizer=tokenizer)
    _db, table, _ = _table(tmp_path, cfg, name=f"fts_{tokenizer}")
    table.add([_row(t, i) for i, t in enumerate(DOCS)])
    table.create_fts_index("text", replace=True, **cfg.fts_index_kwargs())
    return table


@pytest.mark.parametrize("query,expected", [
    ("字幕", 0),
    ("写入磁盘", 0),
    ("面接", 1),
    ("練習モード", 1),
    ("keepalive", 2),
])
def test_ngram_fts_finds_cjk_and_english(tmp_path, fake_model, query, expected):
    table = _fts_table(tmp_path, "ngram")
    results = table.search(query, query_type="fts").limit(3).to_list()
    assert results, f"no FTS results for {query!r}"
    assert results[0]["text"] == DOCS[expected]


def test_simple_tokenizer_misses_cjk_words(tmp_path, fake_model):
    """Why the image uses n-grams: whitespace tokens can't match a CJK word."""
    table = _fts_table(tmp_path, "simple")
    assert table.search("字幕", query_type="fts").limit(3).to_list() == []


def test_hybrid_search_with_ngram_index(tmp_path, fake_model):
    from lancedb.rerankers import RRFReranker

    table = _fts_table(tmp_path, "ngram")
    results = (
        table.search("字幕", query_type="hybrid")
        .rerank(reranker=RRFReranker())
        .limit(3)
        .to_list()
    )
    assert results[0]["text"] == DOCS[0]
