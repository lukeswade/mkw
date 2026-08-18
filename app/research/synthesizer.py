"""Final synthesis: overview.md (map-reduce when notes exceed budget),
follow-up suggestions, and the sources bibliography."""
from __future__ import annotations

import logging

from app.llm import prompts
from app.llm.client import LLM, est_tokens
from app.llm.json_utils import LLMJsonError
from app.models import FollowUpsOut, RECENCY_LABELS
from app.research.notes import Finding, render_facts

log = logging.getLogger(__name__)

_SINGLE_CALL_BUDGET = 36_000   # est tokens of notes for a one-shot synthesis
_BATCH_BUDGET = 30_000         # est tokens per map batch
_PREVIOUS_OVERVIEW_CHARS = 9_000  # ~3k tokens of the parent overview


def looks_like_document(text: str) -> bool:
    """True when the output is a markdown document, not leaked reasoning.

    A thinking-mode model without a reasoning parser streams its planning
    monologue as content ("We need answer user's request…") and can burn the
    whole token budget without ever writing the document. The tell is simple:
    a real overview starts with a markdown heading almost immediately.
    """
    for line in text.strip().splitlines()[:3]:
        if line.lstrip().startswith("#"):
            return True
    return False


def _note_block(f: Finding) -> str:
    block = f"{f.citation_line()}\n    {f.url}\n{f.notes_md}\n"
    # Verbatim evidence is the point of extracting quotes — synthesis has to
    # see them or the claims it writes can't be grounded in the source wording.
    evidence = render_facts(f.key_facts, indent="  ", quotes=True, limit=6)
    if evidence:
        block += f"  Extracted facts and verbatim evidence:\n{evidence}\n"
    return block


async def synthesize(llm: LLM, *, query: str, title: str, brief: str,
                     recency_desc: str, today: str, state_md: str,
                     findings: list[Finding], bus=None, run_id: str = "",
                     previous_overview: str = "") -> str:
    blocks = [_note_block(f) for f in findings]

    if est_tokens("".join(blocks)) > _SINGLE_CALL_BUDGET:
        blocks = await _map_digest(llm, query, blocks)

    prompt = prompts.SYNTH.format(
        query=query, title=title, brief=brief, recency_desc=recency_desc,
        today=today, state_md=state_md or "(none)",
        notes_block="\n".join(blocks),
    )
    if previous_overview:
        # Evergreen refreshes and follow-ups lead with what changed — nobody
        # wants to re-read a 90%-identical overview to find the new part.
        clipped = previous_overview[:_PREVIOUS_OVERVIEW_CHARS]
        prompt += prompts.SYNTH_DELTA_BLOCK.format(previous_overview=clipped)
    messages = [{"role": "user", "content": prompt}]

    # Synthesis is the longest single call in a run and the one the user is
    # actually waiting on, so stream it into the progress pane rather than
    # sitting behind a spinner. A stream failure falls back to a normal call —
    # the document matters more than the animation.
    # A 100+ source deep run deserves a longer report than a 6-source one.
    max_out = 8000 if len(findings) <= 30 else 12000

    text = None
    if bus is not None and run_id:
        try:
            text = await llm.chat_stream("synth", messages, bus, run_id,
                                         max_tokens=max_out, temperature=0.4)
        except Exception:
            log.warning("streaming synthesis failed, retrying unstreamed",
                        exc_info=True)
    if text is None:
        text = await llm.chat("synth", messages, max_tokens=max_out,
                              temperature=0.4)

    if not looks_like_document(text):
        # Leaked reasoning monologue instead of a document. One stern retry;
        # publishing the monologue as an overview wastes the whole run.
        log.warning("synthesis output is not a document, retrying once")
        if bus is not None and run_id:
            bus.publish(run_id, "log",
                        message=("synthesis produced reasoning text instead "
                                 "of the document — retrying once"))
        stern = (prompt + "\n\nIMPORTANT: Output ONLY the final markdown "
                 "document itself, beginning immediately with the '# ' title "
                 "line. No planning, no reasoning, no commentary.")
        retry = await llm.chat("synth", [{"role": "user", "content": stern}],
                               max_tokens=max_out, temperature=0.4)
        if looks_like_document(retry):
            return retry
    return text


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
