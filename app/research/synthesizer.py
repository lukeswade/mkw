"""Final synthesis: overview.md (map-reduce when notes exceed budget),
follow-up suggestions, and the sources bibliography."""
from __future__ import annotations

import logging

from app.llm import prompts
from app.llm.client import LLM, est_tokens
from app.llm.json_utils import LLMJsonError
from app.models import FollowUpsOut, RECENCY_LABELS
from app.research.notes import Finding

log = logging.getLogger(__name__)

_SINGLE_CALL_BUDGET = 36_000   # est tokens of notes for a one-shot synthesis
_BATCH_BUDGET = 30_000         # est tokens per map batch


def _note_block(f: Finding) -> str:
    return f"{f.citation_line()}\n    {f.url}\n{f.notes_md}\n"


async def synthesize(llm: LLM, *, query: str, title: str, brief: str,
                     recency_desc: str, today: str, state_md: str,
                     findings: list[Finding]) -> str:
    blocks = [_note_block(f) for f in findings]

    if est_tokens("".join(blocks)) > _SINGLE_CALL_BUDGET:
        blocks = await _map_digest(llm, query, blocks)

    prompt = prompts.SYNTH.format(
        query=query, title=title, brief=brief, recency_desc=recency_desc,
        today=today, state_md=state_md or "(none)",
        notes_block="\n".join(blocks),
    )
    return await llm.chat(
        "synth", [{"role": "user", "content": prompt}],
        max_tokens=8000, temperature=0.4,
    )


async def _map_digest(llm: LLM, query: str, blocks: list[str]) -> list[str]:
    """Compress note blocks into per-batch digests, preserving [n] citations."""
    batches: list[list[str]] = [[]]
    size = 0
    for b in blocks:
        t = est_tokens(b)
        if size + t > _BATCH_BUDGET and batches[-1]:
            batches.append([])
            size = 0
        batches[-1].append(b)
        size += t
    digests = []
    for batch in batches:
        prompt = prompts.SYNTH_PARTIAL.format(query=query,
                                              notes_block="\n".join(batch))
        digests.append(await llm.chat(
            "synth", [{"role": "user", "content": prompt}],
            max_tokens=4000, temperature=0.3,
        ))
    return digests


async def follow_ups(llm: LLM, *, query: str, overview: str) -> FollowUpsOut:
    prompt = prompts.FOLLOWUPS.format(query=query, overview=overview[:24_000])
    try:
        return await llm.chat_json(
            "followups", [{"role": "user", "content": prompt}],
            FollowUpsOut, max_tokens=1500, temperature=0.5,
        )
    except LLMJsonError as e:
        log.warning("follow-ups skipped: %s", e)
        return FollowUpsOut(items=[])


def render_sources_md(findings: list[Finding]) -> str:
    lines = ["# Sources", ""]
    for f in findings:
        date = f.published or "undated"
        lines.append(
            f'{f.idx}. <a id="src-{f.idx}"></a>**{f.title}** — {f.domain}, '
            f"{date}, relevance {f.relevance}/10  \n   <{f.url}>"
        )
    if not findings:
        lines.append("_No sources were kept._")
    return "\n".join(lines) + "\n"


def render_further_md(items) -> str:
    lines = ["# Further research", ""]
    if not items:
        lines.append("_No follow-up suggestions._")
    for i, item in enumerate(items, 1):
        label = RECENCY_LABELS.get(item.recency, item.recency)
        lines.append(f"{i}. **{item.query}**  \n"
                     f"   {item.rationale}  \n"
                     f"   _suggested: depth {item.depth}, {label.lower()}_")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
