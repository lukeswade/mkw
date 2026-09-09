"""Final synthesis: overview.md (map-reduce when notes exceed budget),
follow-up suggestions, and the sources bibliography."""
from __future__ import annotations

import logging
import re

from app.llm import prompts
from app.llm.client import LLM, est_tokens
from app.llm.json_utils import LLMJsonError
from app.models import FollowUpsOut, RECENCY_LABELS, source_rank
from app.research.notes import Finding, render_facts

log = logging.getLogger(__name__)

# Est tokens of notes above which synthesis is map-reduced into digests.
# Kept high on purpose: one big call is SAFER than several small ones. A
# 9k-token single-call synthesis was verified clean, while the digest path
# adds calls that share a long common prefix — which is what made a poisoned
# KV-cache block repeat its damage across every digest of a run. Digesting is
# for genuinely enormous runs, not a safety measure.
_SINGLE_CALL_BUDGET = 28_000
_BATCH_BUDGET = 20_000         # est tokens per map batch
_DIGEST_BASE_TOKENS = 4000     # room for a digest covering one part
_DIGEST_PER_PART_TOKENS = 600  # ...plus this for each extra part it carries
_DIGEST_MAX_TOKENS = 8000
_STRONG_UNCITED = 7            # relevance at which "read but unused" is news
_PREVIOUS_OVERVIEW_CHARS = 9_000  # ~3k tokens of the parent overview


def looks_degenerate(text: str) -> bool:
    """True for repetition-collapse output: 8000 tokens of "!!!!!!".

    Local models at long prompt lengths can fall into a loop that runs until
    the token cap. The tell is character diversity: real prose over hundreds
    of characters uses dozens of distinct ones, a loop uses a handful.
    """
    sample = (text or "").strip()
    if len(sample) < 200:
        return False
    sample = sample[:4000]
    distinct = len(set(sample))
    top_share = max(sample.count(c) for c in set(sample)) / len(sample)
    return distinct <= 12 or top_share > 0.5


def looks_like_document(text: str) -> bool:
    """True when the output is a markdown document, not leaked reasoning.

    A thinking-mode model without a reasoning parser streams its planning
    monologue as content ("We need answer user's request…") and can burn the
    whole token budget without ever writing the document. The tell is simple:
    a real overview starts with a markdown heading almost immediately.
    """
    if looks_degenerate(text):
        return False
    for line in text.strip().splitlines()[:3]:
        if line.lstrip().startswith("#"):
            return True
    return False


def _note_block(f: Finding) -> str:
    # The kind of source is stated inline: a governing body's spec and a
    # listicle used to reach synthesis as peers, so a run could quote a drill
    # blog over the official manual sitting beside it at 8/10.
    kind = f" [{f.source_type}]" if f.source_type else ""
    if f.source_type == "standard" and f.publisher:
        kind = f" [{f.source_type}: {f.publisher}]"
    block = f"{f.citation_line()}{kind}\n    {f.url}\n{f.notes_md}\n"
    # Verbatim evidence is the point of extracting quotes — synthesis has to
    # see them or the claims it writes can't be grounded in the source wording.
    evidence = render_facts(f.key_facts, indent="  ", quotes=True, limit=6)
    if evidence:
        block += f"  Extracted facts and verbatim evidence:\n{evidence}\n"
    return block


def group_by_facet(findings: list[Finding],
                   facet_of: dict[str, str] | None
                   ) -> list[tuple[str, list[Finding]]]:
    """Findings grouped by the part of the question they were fetched for.

    Arrival order used to decide which notes shared a digest batch, so a part
    with few sources could sit in a batch of twenty about something else and
    be compressed out of existence. 2026-09-09, a U8 coaching run: six
    mixed-ability sources — one of them the official coaching manual, kept at
    8/10 — were read, digested away, and then named an open question by the
    synthesis that had just dropped them. Grouping keeps a part whole, and
    strongest-first means a squeeze drops the weakest source, not a random
    one. Rank leads that ordering and relevance breaks its ties: an official
    standard at 7/10 outranks a roundup at 9/10, because relevance measures
    how much a page says, not whether to believe it."""
    groups: dict[str, list[Finding]] = {}
    for f in findings:
        groups.setdefault((facet_of or {}).get(f.query, ""), []).append(f)
    return [(facet, sorted(fs, key=lambda f: (source_rank(f.source_type),
                                              -f.relevance)))
            for facet, fs in groups.items()]


def _pack(groups: list[tuple[str, list[str]]]
          ) -> list[tuple[list[str], list[str]]]:
    """Batches of (part names, note blocks) under the per-batch budget.

    A part is never scattered across batches — only split when it exceeds the
    budget by itself — so the digest prompt can name what a batch must cover.
    """
    batches: list[tuple[list[str], list[str]]] = []
    parts: list[str] = []
    blocks: list[str] = []
    size = 0
    for facet, group in groups:
        need = sum(est_tokens(b) for b in group)
        if need > _BATCH_BUDGET:
            if blocks:
                batches.append((parts, blocks))
                parts, blocks, size = [], [], 0
            chunk: list[str] = []
            used = 0
            for b in group:
                t = est_tokens(b)
                if used + t > _BATCH_BUDGET and chunk:
                    batches.append(([facet], chunk))
                    chunk, used = [], 0
                chunk.append(b)
                used += t
            if chunk:
                batches.append(([facet], chunk))
            continue
        if size + need > _BATCH_BUDGET and blocks:
            batches.append((parts, blocks))
            parts, blocks, size = [], [], 0
        parts.append(facet)
        blocks.extend(group)
        size += need
    if blocks:
        batches.append((parts, blocks))
    return batches


def cited_ids(overview: str) -> set[int]:
    """The [n] citation ids a finished document actually uses."""
    return {int(n) for n in re.findall(r"\[(\d+)\]", overview)}


def funnel_losses(overview: str, findings: list[Finding],
                  facet_of: dict[str, str] | None
                  ) -> tuple[list[str], list[Finding]]:
    """What the run read and the overview then failed to use.

    Two different failures. A part whose every source went uncited is the
    map-reduce dropping a whole topic. A high-relevance source going uncited
    is a weaker signal but still worth naming. The coverage check upstream
    runs on what was *searched*, so neither of these could be seen before."""
    cited = cited_ids(overview)
    dropped = [facet for facet, fs in group_by_facet(findings, facet_of)
               if facet and not any(f.idx in cited for f in fs)]
    strong = [f for f in findings
              if f.idx not in cited and f.relevance >= _STRONG_UNCITED]
    return dropped, strong


async def synthesize(llm: LLM, *, query: str, title: str, brief: str,
                     recency_desc: str, today: str, state_md: str,
                     findings: list[Finding], bus=None, run_id: str = "",
                     previous_overview: str = "",
                     uncovered_facets: list[str] | None = None,
                     facet_of: dict[str, str] | None = None,
                     premises: list[str] | None = None,
                     placeholder_on_failure: bool = True) -> str:
    groups = [(facet, [_note_block(f) for f in fs])
              for facet, fs in group_by_facet(findings, facet_of)]
    blocks = [b for _, group in groups for b in group]

    if est_tokens("".join(blocks)) > _SINGLE_CALL_BUDGET:
        blocks = await _map_digest(llm, query, groups)

    prompt = prompts.SYNTH.format(
        query=query, title=title, brief=brief, recency_desc=recency_desc,
        today=today, state_md=state_md or "(none)",
        notes_block="\n".join(blocks),
    )
    if premises:
        # The question asserted something checkable. Saying whether it holds
        # comes before answering, because a wrong premise changes the answer.
        prompt += prompts.SYNTH_PREMISE_BLOCK.format(
            premises="\n".join(f"- {p}" for p in premises))
    if uncovered_facets:
        # Synthesis is given the original question, so left alone it writes a
        # confident section for every ask, researched or not.
        prompt += prompts.SYNTH_COVERAGE_BLOCK.format(
            uncovered="\n".join(f"- {f}" for f in uncovered_facets))
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
        # Both attempts failed. Publishing the output anyway is how a run
        # ends up showing 8000 exclamation marks where its overview should
        # be — the research itself is intact, so say so and point at the
        # one-click rebuild instead. Re-synthesis passes
        # placeholder_on_failure=False: overwriting a run's existing overview
        # with a placeholder would destroy something usable.
        if not placeholder_on_failure:
            return retry
        log.error("synthesis unusable twice; writing a placeholder overview")
        if bus is not None and run_id:
            bus.publish(run_id, "log",
                        message=("synthesis failed twice — sources are saved; "
                                 "use Re-synthesize to rebuild the overview"))
        return (f"# {title}\n\n"
                f"> **Synthesis failed.** The research completed and all "
                f"{len(findings)} sources below are saved with their notes and "
                f"evidence, but the model did not return a usable document "
                f"after two attempts — most often a repetition loop on an "
                f"over-long prompt.\n>\n"
                f"> Press **Re-synthesize** on this run to rebuild the "
                f"overview from the stored sources without re-searching. "
                f"If it fails again, a smaller model prompt helps: lower the "
                f"depth, or set a repetition penalty on your inference "
                f"server.\n")
    return text


async def _map_digest(llm: LLM, query: str,
                      groups: list[tuple[str, list[str]]]) -> list[str]:
    """Compress note blocks into per-batch digests, preserving [n] citations.

    Each batch names the parts of the question it carries, and gets output
    room in proportion to how many it carries, so a thinly-sourced part is not
    squeezed out by a populous one sharing its batch."""
    digests = []
    for parts, batch in _pack(groups):
        named = [p for p in parts if p]
        part_block = ""
        if named:
            part_block = (
                "\nThese notes were gathered for the following parts of the "
                "question. Cover EVERY one of them — a part carried by a "
                "single source still gets its specific detail and its [n]:\n"
                + "\n".join(f"- {p}" for p in named) + "\n")
        prompt = prompts.SYNTH_PARTIAL.format(query=query,
                                              part_block=part_block,
                                              notes_block="\n".join(batch))
        digest = await llm.chat(
            "synth", [{"role": "user", "content": prompt}],
            max_tokens=min(_DIGEST_MAX_TOKENS,
                           _DIGEST_BASE_TOKENS
                           + _DIGEST_PER_PART_TOKENS * max(0, len(named) - 1)),
            temperature=0.3,
        )
        if looks_degenerate(digest):
            # A collapsed digest is worse than no digest: it silently poisons
            # the synthesis that consumes it, and the failure then looks like
            # synthesis being at fault. Fall back to this batch's raw notes.
            log.warning("digest collapsed; using its raw notes instead")
            digests.extend(batch)
        else:
            digests.append(digest)
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
