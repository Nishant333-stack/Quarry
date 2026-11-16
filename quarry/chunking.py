"""Structure-aware chunking.

Splits on the document's own structure (headings, paragraphs, tables, code
fences) before falling back to sentences, and prefixes every chunk with its
heading breadcrumb so a chunk still says what it is about once it is alone in
a prompt.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    heading_path: str
    text: str            # breadcrumb + body: exactly what gets embedded and cited
    token_count: int
    content_hash: str


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token); good enough for budgeting."""
    return max(1, (len(text) + 3) // 4)


def _sections(text: str) -> list[tuple[str, list[str]]]:
    """Return [(heading_path, [block, ...])] where a block is an atomic unit."""
    stack: list[tuple[int, str]] = []
    sections: list[tuple[str, list[str]]] = []
    blocks: list[str] = []
    buf: list[str] = []
    in_fence = False

    def flush_block():
        if buf:
            blocks.append("\n".join(buf).strip())
            buf.clear()

    def flush_section():
        nonlocal blocks
        flush_block()
        if blocks:
            sections.append((" > ".join(h for _, h in stack), blocks))
        blocks = []

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            buf.append(line)
            if not in_fence:
                flush_block()
            continue
        if in_fence:
            buf.append(line)
            continue
        m = _HEADING.match(line)
        if m:
            flush_section()
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, m.group(2)))
        elif not line.strip():
            flush_block()
        else:
            buf.append(line)
    flush_section()
    return sections


def _split_oversized(block: str, max_tokens: int) -> list[str]:
    if approx_tokens(block) <= max_tokens:
        return [block]
    if block.lstrip().startswith(("|", "```")):
        units, sep = block.splitlines(), "\n"      # tables / code: split on rows
    else:
        units, sep = _SENTENCE.split(block), " "
    out, cur = [], ""
    for u in units:
        cand = f"{cur}{sep}{u}" if cur else u
        if cur and approx_tokens(cand) > max_tokens:
            out.append(cur)
            cur = u
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out


def _tail(text: str, overlap_tokens: int) -> str:
    """Trailing sentences of ``text`` worth at most ``overlap_tokens``."""
    if overlap_tokens <= 0 or text.lstrip().startswith(("|", "```")):
        return ""
    keep: list[str] = []
    for s in reversed(_SENTENCE.split(text)):
        if approx_tokens(" ".join([s, *keep])) > overlap_tokens:
            break
        keep.insert(0, s)
    return " ".join(keep)


def chunk_document(title: str, text: str, max_tokens: int = 320, overlap_tokens: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []

    def emit(path: str, body: str):
        crumb = path if path.startswith(title) else " > ".join(p for p in (title, path) if p)
        full = f"{crumb}\n{body}".strip()
        chunks.append(
            Chunk(
                ordinal=len(chunks),
                heading_path=path,
                text=full,
                token_count=approx_tokens(full),
                content_hash=hashlib.sha256(full.encode()).hexdigest(),
            )
        )

    for path, blocks in _sections(text):
        pieces = [p for b in blocks for p in _split_oversized(b, max_tokens)]
        cur = ""
        for piece in pieces:
            cand = f"{cur}\n\n{piece}" if cur else piece
            if cur and approx_tokens(cand) > max_tokens:
                emit(path, cur)
                carry = _tail(cur, overlap_tokens)
                cur = f"{carry}\n\n{piece}" if carry else piece
            else:
                cur = cand
        if cur:
            emit(path, cur)
    return chunks
