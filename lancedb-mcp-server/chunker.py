"""Syntax-aware code chunking using Tree-sitter with line-based fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Language, Node, Parser

from config import (
    CHUNK_TOKEN_BUDGET,
    FALLBACK_CHUNK_LINES,
    FALLBACK_EXTENSIONS,
    FALLBACK_OVERLAP_LINES,
    MAX_CHUNK_CHARS,
    SUPPORTED_EXTENSIONS,
)

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single chunk of source code with its metadata."""

    text: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    node_type: str
    symbol_name: str


# ---------------------------------------------------------------------------
# Tree-sitter language loading
# ---------------------------------------------------------------------------

_LANGUAGES: dict[str, Language] = {}


def _load_language(lang_key: str) -> Language | None:
    """Lazily load a Tree-sitter language grammar. Returns None if unavailable."""
    if lang_key in _LANGUAGES:
        return _LANGUAGES[lang_key]

    import_map: dict[str, tuple[str, str]] = {
        "python": ("tree_sitter_python", "language"),
        "javascript": ("tree_sitter_javascript", "language"),
        "typescript": ("tree_sitter_typescript", "language_typescript"),
        "tsx": ("tree_sitter_typescript", "language_tsx"),
        "rust": ("tree_sitter_rust", "language"),
        "go": ("tree_sitter_go", "language"),
        "java": ("tree_sitter_java", "language"),
        "c": ("tree_sitter_c", "language"),
        "cpp": ("tree_sitter_cpp", "language"),
        "ruby": ("tree_sitter_ruby", "language"),
        "csharp": ("tree_sitter_c_sharp", "language"),
        "swift": ("tree_sitter_swift", "language"),
    }

    spec = import_map.get(lang_key)
    if spec is None:
        return None

    module_name, func_name = spec
    try:
        import importlib

        mod = importlib.import_module(module_name)
        lang_func = getattr(mod, func_name)
        lang = Language(lang_func())
        _LANGUAGES[lang_key] = lang
        return lang
    except Exception as exc:
        logger.debug("Tree-sitter grammar not available for %s: %s", lang_key, exc)
        return None


# ---------------------------------------------------------------------------
# Node-type mapping per language
# ---------------------------------------------------------------------------

# Tree-sitter node types that represent top-level extractable entities.
_EXTRACTABLE_TYPES: dict[str, dict[str, str]] = {
    "python": {
        "function_definition": "function",
        "class_definition": "class",
        "decorated_definition": "decorated",
    },
    "javascript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "export_statement": "export",
        "lexical_declaration": "block",
        "expression_statement": "block",
    },
    "typescript": {
        "function_declaration": "function",
        "class_declaration": "class",
        "export_statement": "export",
        "interface_declaration": "interface",
        "type_alias_declaration": "block",
        "enum_declaration": "enum",
        "lexical_declaration": "block",
    },
    "tsx": {
        "function_declaration": "function",
        "class_declaration": "class",
        "export_statement": "export",
        "interface_declaration": "interface",
        "type_alias_declaration": "block",
        "enum_declaration": "enum",
        "lexical_declaration": "block",
    },
    "rust": {
        "function_item": "function",
        "struct_item": "struct",
        "enum_item": "enum",
        "impl_item": "impl",
        "trait_item": "trait",
        "mod_item": "module",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_declaration": "block",
    },
    "java": {
        "class_declaration": "class",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
        "method_declaration": "method",
    },
    "c": {
        "function_definition": "function",
        "struct_specifier": "struct",
        "enum_specifier": "enum",
    },
    "cpp": {
        "function_definition": "function",
        "class_specifier": "class",
        "struct_specifier": "struct",
        "namespace_definition": "module",
    },
    "ruby": {
        "method": "function",
        "class": "class",
        "module": "module",
    },
    "csharp": {
        "class_declaration": "class",
        "interface_declaration": "interface",
        "method_declaration": "method",
        "enum_declaration": "enum",
        "struct_declaration": "struct",
        "namespace_declaration": "module",
    },
}


def _extract_symbol_name(node: Node) -> str:
    """Extract the name identifier from a Tree-sitter node."""
    # Handle decorated definitions (Python) — unwrap to the inner definition.
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return _extract_symbol_name(child)
        return ""

    # Handle export statements (JS/TS) — unwrap to the inner declaration.
    if node.type == "export_statement":
        for child in node.children:
            name = _extract_symbol_name(child)
            if name:
                return name
        return ""

    # Standard: look for a child named "name".
    name_node = node.child_by_field_name("name")
    if name_node:
        return name_node.text.decode("utf-8", errors="replace")

    return ""


def _node_to_chunk_type(node: Node, lang_key: str) -> str | None:
    """Map a Tree-sitter node type to our chunk node_type. Returns None if not extractable."""
    types = _EXTRACTABLE_TYPES.get(lang_key, {})
    return types.get(node.type)


# ---------------------------------------------------------------------------
# Chunking logic
# ---------------------------------------------------------------------------


def _split_oversized(text: str, file_path: str, start_line: int, language: str,
                     node_type: str, symbol_name: str) -> list[Chunk]:
    """Split a chunk that exceeds MAX_CHUNK_CHARS into smaller pieces by lines."""
    lines = text.split("\n")
    chunks: list[Chunk] = []
    i = 0
    part = 0
    while i < len(lines):
        end = min(i + FALLBACK_CHUNK_LINES, len(lines))
        chunk_text = "\n".join(lines[i:end])
        if len(chunk_text) > MAX_CHUNK_CHARS and end - i > 1:
            # Even the reduced window is too big; take what fits.
            end = i + max(1, (end - i) // 2)
            chunk_text = "\n".join(lines[i:end])
        part += 1
        suffix = f" (part {part})" if part > 1 or end < len(lines) else ""
        chunks.append(Chunk(
            text=chunk_text,
            file_path=file_path,
            start_line=start_line + i,
            end_line=start_line + end - 1,
            language=language,
            node_type=node_type,
            symbol_name=f"{symbol_name}{suffix}",
        ))
        i = end
    return chunks


def _chunk_with_treesitter(source: bytes, file_path: str, lang_key: str,
                           lang: Language) -> list[Chunk]:
    """Parse source with Tree-sitter and extract structural chunks."""
    if lang_key == "swift":
        return _chunk_swift(source, file_path, lang)

    parser = Parser(lang)
    tree = parser.parse(source)
    root = tree.root_node

    extractable = _EXTRACTABLE_TYPES.get(lang_key, {})
    if not extractable:
        return []

    chunks: list[Chunk] = []
    covered_ranges: list[tuple[int, int]] = []  # (start_byte, end_byte)

    for child in root.children:
        if child.type not in extractable:
            continue

        chunk_type = extractable[child.type]
        symbol = _extract_symbol_name(child)
        text = child.text.decode("utf-8", errors="replace")
        start_line = child.start_point[0] + 1  # 1-indexed
        end_line = child.end_point[0] + 1

        # For classes, also extract methods as separate chunks.
        if chunk_type == "class" and lang_key in ("python", "javascript", "typescript",
                                                   "tsx", "java", "csharp", "ruby"):
            _extract_class_methods(child, file_path, lang_key, symbol, chunks)

        if len(text) > MAX_CHUNK_CHARS:
            chunks.extend(_split_oversized(
                text, file_path, start_line, lang_key, chunk_type, symbol
            ))
        else:
            chunks.append(Chunk(
                text=text,
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                language=lang_key,
                node_type=chunk_type,
                symbol_name=symbol,
            ))

        covered_ranges.append((child.start_byte, child.end_byte))

    # Capture module-level code not covered by any extracted entity.
    module_lines = _extract_uncovered_text(source, covered_ranges)
    if module_lines.strip():
        line_count = source[:covered_ranges[0][0]].count(b"\n") if covered_ranges else 0
        chunks.append(Chunk(
            text=module_lines if len(module_lines) <= MAX_CHUNK_CHARS
            else module_lines[:MAX_CHUNK_CHARS],
            file_path=file_path,
            start_line=1,
            end_line=1 + module_lines.count("\n"),
            language=lang_key,
            node_type="module",
            symbol_name="",
        ))

    return chunks


def _extract_class_methods(class_node: Node, file_path: str, lang_key: str,
                           class_name: str, chunks: list[Chunk]) -> None:
    """Extract individual methods from a class node as separate chunks."""
    method_types = {
        "python": ("function_definition",),
        "javascript": ("method_definition",),
        "typescript": ("method_definition", "public_field_definition"),
        "tsx": ("method_definition", "public_field_definition"),
        "java": ("method_declaration", "constructor_declaration"),
        "csharp": ("method_declaration", "constructor_declaration"),
        "ruby": ("method",),
    }
    target_types = method_types.get(lang_key, ())

    body = class_node.child_by_field_name("body")
    container = body if body else class_node

    for child in container.children:
        actual = child
        # Handle Python decorated methods.
        if child.type == "decorated_definition":
            for sub in child.children:
                if sub.type in target_types:
                    actual = child  # Keep the decorator wrapper.
                    break

        if actual.type in target_types or child.type in target_types:
            node = actual if actual.type in target_types else child
            text = child.text.decode("utf-8", errors="replace")
            symbol = _extract_symbol_name(node)
            start_line = child.start_point[0] + 1
            end_line = child.end_point[0] + 1
            qualified = f"{class_name}.{symbol}" if symbol else class_name

            if len(text) > MAX_CHUNK_CHARS:
                chunks.extend(_split_oversized(
                    text, file_path, start_line, lang_key, "method", qualified
                ))
            else:
                chunks.append(Chunk(
                    text=text,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    language=lang_key,
                    node_type="method",
                    symbol_name=qualified,
                ))


# ---------------------------------------------------------------------------
# Swift
# ---------------------------------------------------------------------------

# class_declaration covers class, struct, enum, actor and extension; the
# declaration_kind field says which one.
_SWIFT_KIND_TO_TYPE: dict[str, str] = {
    "class": "class",
    "struct": "struct",
    "enum": "enum",
    "actor": "class",
    "extension": "impl",
}
_SWIFT_COMMENT_TYPES = {"comment", "multiline_comment"}
_SWIFT_MEMBER_TYPES = {
    "function_declaration",
    "init_declaration",
    "deinit_declaration",
    "subscript_declaration",
    "protocol_function_declaration",
}


def _swift_type_kind(node: Node) -> str | None:
    """Chunk node_type for a Swift type declaration, or None."""
    if node.type == "protocol_declaration":
        return "interface"
    if node.type != "class_declaration":
        return None
    kind = node.child_by_field_name("declaration_kind")
    kind_text = kind.text.decode("utf-8", errors="replace") if kind else "class"
    return _SWIFT_KIND_TO_TYPE.get(kind_text, "class")


def _swift_symbol(node: Node) -> str:
    if node.type == "init_declaration":
        return "init"
    if node.type == "deinit_declaration":
        return "deinit"
    if node.type == "subscript_declaration":
        return "subscript"
    return _extract_symbol_name(node)


def _swift_member_wanted(node: Node) -> bool:
    """Methods, initializers and multi-line properties (e.g. SwiftUI ``body``)."""
    if node.type in _SWIFT_MEMBER_TYPES:
        return True
    if node.type == "property_declaration":
        computed = any(c.type == "computed_property" for c in node.children)
        return computed or (node.end_point[0] - node.start_point[0]) >= 2
    return False


def _leading_comment_start(siblings: list[Node], idx: int) -> Node:
    """First node of the comment block directly above ``siblings[idx]``.

    Doc comments often explain intent in the author's own language, so they
    are kept with the declaration they describe.
    """
    first = siblings[idx]
    j = idx - 1
    while (
        j >= 0
        and siblings[j].type in _SWIFT_COMMENT_TYPES
        and siblings[j].end_point[0] >= first.start_point[0] - 1
    ):
        first = siblings[j]
        j -= 1
    return first


def _swift_chunks_for(text: str, file_path: str, start_line: int, end_line: int,
                      node_type: str, symbol: str) -> list[Chunk]:
    if len(text) > MAX_CHUNK_CHARS:
        return _split_oversized(text, file_path, start_line, "swift", node_type, symbol)
    return [Chunk(
        text=text,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        language="swift",
        node_type=node_type,
        symbol_name=symbol,
    )]


def _swift_body(node: Node) -> Node | None:
    body = node.child_by_field_name("body")
    if body is not None:
        return body
    for child in node.children:
        if child.type.endswith("_body"):
            return child
    return None


def _extract_swift_members(type_node: Node, source: bytes, file_path: str,
                           type_name: str, chunks: list[Chunk]) -> None:
    """Add chunks for the members of a Swift type, recursing into nested types."""
    body = _swift_body(type_node)
    if body is None:
        return
    members = list(body.children)
    for i, member in enumerate(members):
        nested_kind = _swift_type_kind(member)
        if nested_kind is None and not _swift_member_wanted(member):
            continue
        first = _leading_comment_start(members, i)
        text = source[first.start_byte:member.end_byte].decode("utf-8", errors="replace")
        symbol = _swift_symbol(member)
        qualified = f"{type_name}.{symbol}" if symbol else type_name
        if nested_kind is not None:
            _extract_swift_members(member, source, file_path, qualified, chunks)
        chunks.extend(_swift_chunks_for(
            text, file_path, first.start_point[0] + 1, member.end_point[0] + 1,
            nested_kind or "method", qualified,
        ))


def _chunk_swift(source: bytes, file_path: str, lang: Language) -> list[Chunk]:
    """Chunk a Swift file: types, their members, and top-level functions."""
    tree = Parser(lang).parse(source)
    siblings = list(tree.root_node.children)

    chunks: list[Chunk] = []
    covered: list[tuple[int, int]] = []

    for i, node in enumerate(siblings):
        kind = _swift_type_kind(node)
        if kind is None and node.type == "function_declaration":
            kind = "function"
        if kind is None:
            continue

        first = _leading_comment_start(siblings, i)
        text = source[first.start_byte:node.end_byte].decode("utf-8", errors="replace")
        symbol = _swift_symbol(node)
        if kind != "function":
            _extract_swift_members(node, source, file_path, symbol, chunks)
        chunks.extend(_swift_chunks_for(
            text, file_path, first.start_point[0] + 1, node.end_point[0] + 1,
            kind, symbol,
        ))
        covered.append((first.start_byte, node.end_byte))

    module_text = _extract_uncovered_text(source, covered)
    if module_text.strip():
        chunks.append(Chunk(
            text=module_text if len(module_text) <= MAX_CHUNK_CHARS
            else module_text[:MAX_CHUNK_CHARS],
            file_path=file_path,
            start_line=1,
            end_line=1 + module_text.count("\n"),
            language="swift",
            node_type="module",
            symbol_name="",
        ))

    return chunks


def _extract_uncovered_text(source: bytes, covered: list[tuple[int, int]]) -> str:
    """Extract source text not covered by any extracted entity."""
    if not covered:
        return source.decode("utf-8", errors="replace")

    covered_sorted = sorted(covered, key=lambda r: r[0])
    parts: list[bytes] = []

    # Before first entity.
    if covered_sorted[0][0] > 0:
        parts.append(source[: covered_sorted[0][0]])

    # Gaps between entities.
    for i in range(len(covered_sorted) - 1):
        gap_start = covered_sorted[i][1]
        gap_end = covered_sorted[i + 1][0]
        if gap_end > gap_start:
            gap = source[gap_start:gap_end]
            if gap.strip():
                parts.append(gap)

    # After last entity.
    if covered_sorted[-1][1] < len(source):
        tail = source[covered_sorted[-1][1]:]
        if tail.strip():
            parts.append(tail)

    return b"\n".join(parts).decode("utf-8", errors="replace")


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 ASCII chars per token, 1 token per CJK char."""
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return (len(text) - ascii_chars) + (ascii_chars + 3) // 4


def _window_end(lines: list[str], start: int) -> int:
    """End (exclusive) of the next line window, honouring the token budget."""
    hard_end = min(start + FALLBACK_CHUNK_LINES, len(lines))
    if CHUNK_TOKEN_BUDGET <= 0:
        return hard_end
    end = start + 1  # always take at least one line
    used = estimate_tokens(lines[start])
    while end < hard_end:
        cost = estimate_tokens(lines[end]) + 1
        if used + cost > CHUNK_TOKEN_BUDGET:
            break
        used += cost
        end += 1
    return end


def _chunk_by_lines(text: str, file_path: str, language: str) -> list[Chunk]:
    """Fall back to line-based chunking with overlap.

    Windows hold up to FALLBACK_CHUNK_LINES lines, and stop earlier when
    CHUNK_TOKEN_BUDGET is set and the next line would exceed it.
    """
    lines = text.split("\n")
    chunks: list[Chunk] = []
    i = 0
    while i < len(lines):
        end = _window_end(lines, i)
        chunk_text = "\n".join(lines[i:end])
        chunks.append(Chunk(
            text=chunk_text,
            file_path=file_path,
            start_line=i + 1,
            end_line=end,
            language=language,
            node_type="block",
            symbol_name="",
        ))
        if end >= len(lines):
            break
        i = end - min(FALLBACK_OVERLAP_LINES, (end - i) // 2)
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_file(file_path: str, repo_root: Path) -> list[Chunk]:
    """Chunk a source file into semantic pieces.

    Uses Tree-sitter for supported languages, line-based fallback otherwise.
    Returns an empty list if the file is not indexable.
    """
    abs_path = Path(file_path)
    if not abs_path.is_absolute():
        abs_path = repo_root / file_path

    suffix = abs_path.suffix.lower()
    # Handle Dockerfile specially.
    if abs_path.name.lower() in ("dockerfile", "makefile", "cmakelists.txt"):
        suffix = "." + abs_path.name.lower()

    lang_key = SUPPORTED_EXTENSIONS.get(suffix)
    is_fallback = suffix in FALLBACK_EXTENSIONS or suffix in (".dockerfile", ".makefile")

    if lang_key is None and not is_fallback:
        return []

    try:
        source_bytes = abs_path.read_bytes()
    except (OSError, IOError) as exc:
        logger.warning("Cannot read %s: %s", file_path, exc)
        return []

    # Skip very large files (> 1 MB).
    if len(source_bytes) > 1_048_576:
        logger.info("Skipping large file (%d bytes): %s", len(source_bytes), file_path)
        return []

    source_text = source_bytes.decode("utf-8", errors="replace")
    rel_path = str(abs_path.relative_to(repo_root))

    if lang_key:
        lang = _load_language(lang_key)
        if lang:
            chunks = _chunk_with_treesitter(source_bytes, rel_path, lang_key, lang)
            if chunks:
                return chunks
            # If Tree-sitter produced nothing (e.g. empty file), fall through.

    # Fallback: line-based chunking.
    effective_lang = lang_key or suffix.lstrip(".")
    if source_text.strip():
        return _chunk_by_lines(source_text, rel_path, effective_lang)

    return []
