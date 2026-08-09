"""Heading-aware markdown chunking for embedding.

bge-small has a 512-token sequence limit; chunks target ~1300 chars
(≈350 tokens) with a hard cap, and each chunk carries its section heading
for context. Adjacent chunks overlap by one paragraph.
"""
from __future__ import annotations

import re

TARGET_CHARS = 1300
MAX_CHARS = 1800

_HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)


def _sections(md: str) -> list[tuple[str, str]]:
    """Split into (heading, body) sections; heading may be ''."""
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in (md or "").splitlines():
        if re.match(r"^#{1,6}\s", line):
            sections.append((line.strip(), []))
        else:
            sections[-1][1].append(line)
    return [(h, "\n".join(body).strip()) for h, body in sections
            if (h or "".join(body).strip())]


def _hard_split(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]


def chunk_markdown(md: str, *, target: int = TARGET_CHARS,
                   maximum: int = MAX_CHARS) -> list[str]:
    chunks: list[str] = []
    for heading, body in _sections(md):
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        if heading and not paras:
            paras = [""]
        current: list[str] = []
        size = 0
        fresh = False  # does current hold anything beyond the carried overlap?
        for para in paras:
            pieces = _hard_split(para, target) if len(para) > maximum else [para]
            for piece in pieces:
                if size + len(piece) > target and current:
                    chunks.append(_render(heading, current))
                    # one-paragraph overlap, but never a huge one — a big tail
                    # would blow the next chunk past the max size
                    tail = current[-1]
                    current = [tail] if len(tail) <= 400 else []
                    size = sum(len(p) for p in current)
                    fresh = False
                current.append(piece)
                size += len(piece)
                fresh = True
        if fresh and (heading or any(c.strip() for c in current)):
            chunks.append(_render(heading, current))
    return [c for c in chunks if c.strip()]


def _render(heading: str, paras: list[str]) -> str:
    body = "\n\n".join(p for p in paras if p.strip())
    return f"{heading}\n\n{body}".strip() if heading else body
