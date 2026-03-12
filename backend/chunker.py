"""
chunker.py — splits markdown files into smart chunks based on headings.

Recursive fallback chain (stops as soon as a level fits):
  H2 too big → split by H3
    H3 too big → split by H4
      H4 too big → split by paragraph
        Paragraph too big → split by sentences
          Truly unsplittable → hard truncate to MISTRAL_MAX_TOKENS + warn

Key invariant: code fences (```...```) are never split by heading/paragraph/
sentence logic — they are extracted as placeholders before splitting and
restored afterward. However, if a code block is so large that it alone
exceeds MISTRAL_MAX_TOKENS, it is hard-truncated like any other text so the
Mistral API call never fails.

Merging: any chunk below MIN_TOKENS is merged with its neighbour.
"""

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

MIN_TOKENS = 50
MAX_TOKENS = 800
MISTRAL_MAX_TOKENS = 8192   # hard API limit — no chunk may exceed this
MAX_HEADING_DEPTH = 4       # recurse H2 → H3 → H4, then fall back to paragraphs

# Characters per token (conservative estimate — code/markdown can be ~2-3 chars/token)
_CHARS_PER_TOKEN = 3

# Placeholder used to protect code blocks during text splitting
_CODE_PLACEHOLDER = "\x00CODE{}\x00"
_CODE_PLACEHOLDER_RE = re.compile(r"\x00CODE(\d+)\x00")


@dataclass
class Chunk:
    id: str
    source_file: str
    heading: str
    body: str


# ─── Token estimation ────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    return len(text) // _CHARS_PER_TOKEN


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Hard-truncate text to at most max_tokens (character-based estimate)."""
    return text[: max_tokens * _CHARS_PER_TOKEN]


# ─── Code block protection ───────────────────────────────────────────────────

def _extract_code_blocks(text: str) -> tuple[str, list[str]]:
    """
    Replace every fenced code block (``` ... ```) with a placeholder token.
    Returns (sanitised_text, [code_block_0, code_block_1, ...]).
    The original text can be restored with _restore_code_blocks().
    """
    code_blocks: list[str] = []

    def replacer(match: re.Match) -> str:
        idx = len(code_blocks)
        code_blocks.append(match.group(0))
        return _CODE_PLACEHOLDER.format(idx)

    sanitised = re.sub(
        r"```[\w]*\n.*?```",
        replacer,
        text,
        flags=re.DOTALL,
    )
    return sanitised, code_blocks


def _restore_code_blocks(text: str, code_blocks: list[str]) -> str:
    """Substitute placeholders back with their original code blocks."""
    def replacer(match: re.Match) -> str:
        return code_blocks[int(match.group(1))]
    return _CODE_PLACEHOLDER_RE.sub(replacer, text)


def _contains_placeholder(text: str) -> bool:
    return bool(_CODE_PLACEHOLDER_RE.search(text))


# ─── Low-level splitting helpers ─────────────────────────────────────────────

def _split_by_heading(text: str, level: int) -> list[tuple[str, str]]:
    """
    Split text by markdown headings of exactly `level` hashes.
    Never splits inside a code block (placeholders contain no # characters).
    Returns [(heading_text, body), ...], or [] if no headings at this level.
    """
    hashes = "#" * level
    pattern = rf"^{hashes} (.+)$"
    parts = re.split(pattern, text, flags=re.MULTILINE)

    if len(parts) == 1:
        return []

    results = []
    if parts[0].strip():
        results.append(("(intro)", parts[0].strip()))

    it = iter(parts[1:])
    for heading, body in zip(it, it):
        results.append((heading.strip(), body.strip()))

    return results


def _split_by_paragraph(text: str) -> list[tuple[str, str]]:
    """
    Split on blank lines. Placeholders land in their own paragraph,
    so code blocks are never fused with adjacent prose mid-split.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [("(para)", p) for p in paras]


def _split_by_sentences(text: str, window: int = 10) -> list[tuple[str, str]]:
    """
    Split prose into sentence windows. Paragraphs that consist solely of a
    placeholder are kept atomic (not sentence-split).
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    results: list[tuple[str, str]] = []

    for para in paragraphs:
        if _contains_placeholder(para):
            results.append(("(code)", para))
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para.strip())
        for i in range(0, len(sentences), window):
            results.append(("(sentences)", " ".join(sentences[i : i + window])))

    return results if results else [("(sentences)", text)]


def _estimate_tokens_with_code(text: str, code_blocks: list[str]) -> int:
    """
    Token estimate that accounts for actual code block sizes,
    not the tiny placeholder strings that replace them during splitting.
    """
    restored = _restore_code_blocks(text, code_blocks)
    return _estimate_tokens(restored)


# ─── Recursive splitter ───────────────────────────────────────────────────────

def _recursive_split(
    heading_path: str,
    text: str,
    level: int,
    code_blocks: list[str],
    source_hint: str = "",
) -> list[tuple[str, str]]:
    """
    Recursively split `text` (code blocks replaced by placeholders) until every
    piece fits within MAX_TOKENS, using real code block sizes for size checks.

    If a piece cannot be split further and still exceeds MISTRAL_MAX_TOKENS,
    it is hard-truncated so the Mistral API never receives an oversized input.
    """
    token_count = _estimate_tokens_with_code(text, code_blocks)

    # ── Already fits within soft limit ───────────────────────────────────────
    if token_count <= MAX_TOKENS:
        return [(heading_path, text)]

    # ── Try next heading level ────────────────────────────────────────────────
    if level <= MAX_HEADING_DEPTH:
        sub_sections = _split_by_heading(text, level)
        if sub_sections:
            results = []
            for sub_heading, sub_body in sub_sections:
                child_path = (
                    f"{heading_path} > {sub_heading}"
                    if sub_heading != "(intro)"
                    else heading_path
                )
                results.extend(
                    _recursive_split(child_path, sub_body, level + 1, code_blocks, source_hint)
                )
            return _merge_short(results, code_blocks)

    # ── Fall back to paragraphs ───────────────────────────────────────────────
    para_sections = _split_by_paragraph(text)
    if len(para_sections) > 1:
        results = []
        for _, para_body in para_sections:
            results.extend(
                _recursive_split(heading_path, para_body, level + 1, code_blocks, source_hint)
            )
        return _merge_short(results, code_blocks)

    # ── Content with code blocks still oversized ─────────────────────────────
    # Restore, split in half by line count, re-process each half independently.
    if _CODE_PLACEHOLDER_RE.search(text.strip()):
        restored = _restore_code_blocks(text, code_blocks)
        lines = restored.splitlines()
        if len(lines) >= 2:
            mid = len(lines) // 2
            halves = [
                (heading_path,                 "\n".join(lines[:mid]).strip()),
                (f"{heading_path} (cont.)",    "\n".join(lines[mid:]).strip()),
            ]
            results = []
            for half_heading, half_text in halves:
                half_sanitised, half_codes = _extract_code_blocks(half_text)
                for h, b in _recursive_split(half_heading, half_sanitised, level + 1, half_codes, source_hint):
                    results.append((h, _restore_code_blocks(b, half_codes)))
            return results
        # Single line — truncate as absolute last resort
        warnings.warn(
            f"Single-line chunk (~{token_count} tokens) in {source_hint!r} "
            f"under {heading_path!r} cannot be split. Truncating.",
            stacklevel=2,
        )
        return [(heading_path, _truncate_to_tokens(restored, MISTRAL_MAX_TOKENS))]

    # ── Fall back to sentence windows (prose only) ────────────────────────────
    sentence_sections = _split_by_sentences(text)
    if len(sentence_sections) > 1:
        return [(heading_path, body) for _, body in sentence_sections]

    # ── Truly unsplittable prose — hard truncate ──────────────────────────────
    if token_count > MISTRAL_MAX_TOKENS:
        warnings.warn(
            f"Chunk (~{token_count} tokens) in {source_hint!r} under heading "
            f"{heading_path!r} cannot be split further and exceeds Mistral's "
            f"limit of {MISTRAL_MAX_TOKENS} tokens. Truncating.",
            stacklevel=2,
        )
        restored = _restore_code_blocks(text, code_blocks)
        return [(heading_path, _truncate_to_tokens(restored, MISTRAL_MAX_TOKENS))]

    return [(heading_path, text)]


def _merge_short(sections: list[tuple[str, str]], code_blocks: list[str]) -> list[tuple[str, str]]:
    """
    Merge consecutive chunks below MIN_TOKENS into their neighbour.
    Uses real code block sizes for size checks.
    """
    if not sections:
        return sections

    merged: list[tuple[str, str]] = []
    pending_heading, pending_body = sections[0]

    for heading, body in sections[1:]:
        if _estimate_tokens_with_code(pending_body, code_blocks) < MIN_TOKENS:
            pending_body = pending_body + "\n\n" + body
            if pending_heading in ("(intro)", "(para)", "(sentences)", "(code)"):
                pending_heading = heading
        else:
            merged.append((pending_heading, pending_body))
            pending_heading, pending_body = heading, body

    if _estimate_tokens_with_code(pending_body, code_blocks) < MIN_TOKENS and merged:
        last_h, last_b = merged[-1]
        merged[-1] = (last_h, last_b + "\n\n" + pending_body)
    else:
        merged.append((pending_heading, pending_body))

    return merged


# ─── Public API ───────────────────────────────────────────────────────────────

def chunk_file(filepath: str | Path) -> list[Chunk]:
    """
    Main entry point. Takes a path to a .md file, returns a list of Chunk objects.
    Code blocks are protected throughout and restored in each final chunk.
    """
    path = Path(filepath)
    raw_text = path.read_text(encoding="utf-8")
    filename = path.stem
    chunks: list[Chunk] = []

    sanitised, code_blocks = _extract_code_blocks(raw_text)

    h2_sections = _split_by_heading(sanitised, level=2)
    if not h2_sections:
        title = _get_h1_title(sanitised) or filename
        h2_sections = [(title, sanitised)]

    for h2_heading, h2_body in h2_sections:
        flat = _recursive_split(h2_heading, h2_body, level=3, code_blocks=code_blocks, source_hint=filename)
        flat = _merge_short(flat, code_blocks)

        for heading, body in flat:
            restored = _restore_code_blocks(body, code_blocks).strip()
            if not restored:
                continue
            chunks.append(Chunk(
                id=f"{filename}__{len(chunks)}",
                source_file=str(path),
                heading=heading,
                body=restored,
            ))

    return chunks


def _get_h1_title(text: str) -> str | None:
    match = re.search(r"^# (.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None