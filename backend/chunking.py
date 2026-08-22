"""Heading-aware text chunking.

Chunks keep the markdown heading path they came from (e.g. "Install > Windows").
Prefixing that path gives the retriever and the model useful context that a
plain character split throws away.
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
    """Yield ``(heading_path, body)`` pairs from markdown-ish text."""
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


def _split_block(text: str, size: int, overlap: int) -> list[str]:
    """Split text on the coarsest separator that keeps pieces under ``size``."""
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
            # A single oversized part still needs splitting by a finer separator.
            current = part if len(part) <= size else ""
            if not current:
                pieces.extend(_split_block(part, size, overlap))
        if current:
            pieces.append(current)

        if pieces:
            return _apply_overlap(pieces, overlap)

    return [text[i : i + size] for i in range(0, len(text), max(size - overlap, 1))]


def _apply_overlap(pieces: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or len(pieces) < 2:
        return pieces
    overlapped = [pieces[0]]
    for previous, piece in zip(pieces, pieces[1:]):
        tail = previous[-overlap:]
        overlapped.append(f"{tail}\n{piece}" if tail else piece)
    return overlapped


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Split page text into retrieval chunks, preserving heading context."""
    chunks: list[Chunk] = []
    for heading, body in _iter_sections(text):
        for piece in _split_block(body, size, overlap):
            piece = piece.strip()
            if piece:
                chunks.append(Chunk(text=piece, heading=heading, index=len(chunks)))
    return chunks
