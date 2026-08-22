"""Heading-aware text chunking.

Chunks keep the markdown heading path they came from
(e.g. "Install > Windows").

Prefixing that path gives the retriever and model useful context
that a plain character split throws away.
"""

import re
from dataclasses import dataclass

from config import CHUNK_OVERLAP, CHUNK_SIZE

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SEPARATORS = ("\n\n", "\n", ". ", " ")


@dataclass(frozen=True)
class Chunk:
    text: str
    heading: str
    index: int

    @property
    def with_heading(self) -> str:
        return f"[{self.heading}]\n{self.text}" if self.heading else self.text


def _iter_sections(text: str):
    """Yield (heading_path, body) pairs from markdown-ish text."""
    stack: list[str] = []
    buffer: list[str] = []

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if not match:
            buffer.append(line)
            continue
        body = "\n".join(buffer).strip()
        if body:
            yield " > ".join(stack), body
        buffer = []
        level = len(match.group(1))
        stack = stack[: level - 1]
        stack.append(match.group(2).strip())

    body = "\n".join(buffer).strip()

    if body:
        yield " > ".join(stack), body


def _get_overlap(previous: str, overlap: int) -> str:
    """Return a context overlap ending at a sensible text boundary."""
    if overlap <= 0 or not previous:
        return ""
    if len(previous) <= overlap:
        return previous
    tail = previous[-overlap:]
    sentence_breaks = [
        tail.rfind(". "),
        tail.rfind("? "),
        tail.rfind("! "),
    ]
    best_break = max(sentence_breaks)
    if best_break >= 0:
        return tail[best_break + 2 :].strip()
    space = tail.find(" ")
    if space >= 0:
        return tail[space + 1 :].strip()
    return tail.strip()


def _apply_overlap(pieces: list[str], overlap: int) -> list[str]:
    """Add semantic overlap between adjacent pieces."""
    if overlap <= 0 or len(pieces) < 2:
        return pieces
    overlapped = [pieces[0]]
    for previous, piece in zip(pieces, pieces[1:]):
        tail = _get_overlap(previous, overlap)
        if tail:
            overlapped.append(f"{tail}\n{piece}")
        else:
            overlapped.append(piece)

    return overlapped


def _split_block(text: str, size: int, overlap: int) -> list[str]:
    """Split text on progressively finer separators."""
    if len(text) <= size:
        return [text]
    for separator in _SEPARATORS:
        parts = text.split(separator)
        if len(parts) == 1:
            continue
        pieces: list[str] = []
        current = ""
        for part in parts:
            candidate = f"{current}{separator}{part}" if current else part
            if len(candidate) <= size:
                current = candidate
                continue
            if current:
                pieces.append(current)
            if len(part) <= size:
                current = part
            else:
                pieces.extend(_split_block(part, size, 0))
                current = ""
        if current:
            pieces.append(current)
        if pieces:
            return _apply_overlap(pieces, overlap)

    step = max(size - overlap, 1)

    pieces = [text[i : i + size] for i in range(0, len(text), step)]
    return _apply_overlap(pieces, overlap)


def chunk_text(
    text: str,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split page text into retrieval chunks while preserving heading context."""
    chunks: list[Chunk] = []
    for heading, body in _iter_sections(text):
        pieces = _split_block(body, size, overlap)
        for piece in pieces:
            piece = piece.strip()
            if piece:
                chunks.append(
                    Chunk(
                        text=piece,
                        heading=heading,
                        index=len(chunks),
                    )
                )

    return chunks
