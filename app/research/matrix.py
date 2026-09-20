"""Comparison matrix: a finished run's findings as a table.

Runs after the fact over stored findings — no searching, no fetching, the
same shape as re-synthesis. The note stage already extracts a claim, a
verbatim supporting quote and a confidence score from every source, which is
exactly what a table cell needs; synthesis then flattens all of it into
prose. This recovers the structure that was already there.
"""
from __future__ import annotations

import logging

from app.llm import prompts
from app.llm.client import LLM, est_tokens
from app.models import MatrixOut
from app.research.notes import Finding

log = logging.getLogger(__name__)

# Est tokens of source notes fed to the matrix call. Lower than synthesis:
# the output is structured JSON over every entity/dimension pair, so the
# model needs headroom to finish the object rather than truncate mid-cell.
_INPUT_BUDGET = 18_000
_PER_SOURCE_CHARS = 1_800


def _block(f: Finding) -> str:
    body = (f.notes_md or f.summary or "").strip()
    if len(body) > _PER_SOURCE_CHARS:
        body = body[:_PER_SOURCE_CHARS].rstrip() + " …"
    return f"{f.citation_line()} relevance {f.relevance}/10\n{body}\n"


def fit_to_budget(findings: list[Finding]) -> tuple[list[Finding], int]:
    """Highest-relevance findings that fit the input budget, plus the count dropped.

    A 46-source run does not fit in one call. Dropping the weakest sources is
    honest and reported; silently truncating the prompt mid-source is not.
    """
    ranked = sorted(findings, key=lambda f: (-f.relevance, f.idx))
    kept: list[Finding] = []
    size = 0
    for f in ranked:
        t = est_tokens(_block(f))
        if kept and size + t > _INPUT_BUDGET:
            continue
        kept.append(f)
        size += t
    kept.sort(key=lambda f: f.idx)          # citation order, not score order
    return kept, len(findings) - len(kept)


async def build(llm: LLM, *, query: str,
                findings: list[Finding]) -> tuple[MatrixOut, int]:
    used, dropped = fit_to_budget(findings)
    prompt = prompts.MATRIX.format(
        query=query, notes_block="\n".join(_block(f) for f in used))
    out = await llm.chat_json(
        "matrix", [{"role": "user", "content": prompt}],
        MatrixOut, max_tokens=4000, temperature=0.2,
    )
    return out, dropped


def _cell(text: str) -> str:
    """A pipe or newline inside a value would break the table row."""
    return " ".join((text or "").split()).replace("|", "\\|")


_MAX_CITES = 3


def _cites(sources: list[int]) -> str:
    """At most three markers per cell, then a count.

    A live run put twelve citations in one cell. Twelve is not more credible
    than three, it is just unreadable — and the table is the artifact people
    scan, so density matters more here than in prose.
    """
    ids = sorted({n for n in sources if n > 0})
    if not ids:
        return ""
    shown = "".join(f"[{n}]" for n in ids[:_MAX_CITES])
    extra = len(ids) - _MAX_CITES
    return f"{shown} +{extra}" if extra > 0 else shown


def _key(s: str) -> str:
    return " ".join(str(s or "").split()).lower()


def render_matrix_md(out: MatrixOut, *, title: str, dropped: int = 0) -> str:
    """Dimensions as rows, entities as columns.

    That way round because a comparison usually has few entities and many
    axes, and a table is easier to read tall than wide.
    """
    # The model names the axes once and the cells again; a stray space or a
    # capital between the two turned a filled cell into "—". Match on a
    # normalized key and keep the axis strings for display.
    lookup = {(_key(c.entity), _key(c.dimension)): c for c in out.cells}
    entities = [e for e in out.entities if e]
    dimensions = [d for d in out.dimensions if d]
    filled = 0

    lines = [f"# {title} — comparison", ""]
    lines.append("| | " + " | ".join(_cell(e) for e in entities) + " |")
    lines.append("| --- " * (len(entities) + 1) + "|")

    conflicted = False
    for dim in dimensions:
        row = [f"**{_cell(dim)}**"]
        for ent in entities:
            c = lookup.get((_key(ent), _key(dim)))
            if c is None or not c.value.strip():
                row.append("—")
                continue
            filled += 1
            value = _cell(c.value)
            if c.conflict:
                conflicted = True
                value += " †"
            row.append(f"{value} {_cites(c.sources)}".strip())
        lines.append("| " + " | ".join(row) + " |")

    total = len(entities) * len(dimensions)
    footnotes = [f"_{filled} of {total} cells filled from the sources; "
                 f"— marks a cell the sources shown here did not answer._"]
    if conflicted:
        footnotes.append("_† sources disagree on this cell._")
    if dropped:
        footnotes.append(f"_Built from the highest-scoring sources; {dropped} "
                         f"lower-scoring source(s) did not fit the model's "
                         f"context._")
    lines += ["", "  \n".join(footnotes)]

    if out.caveats_md.strip():
        lines += ["", "## Disagreements and gaps", "", out.caveats_md.strip()]
    return "\n".join(lines) + "\n"
