"""Round-end gap analysis: rewrite the state doc, decide saturation, aim next round."""
from __future__ import annotations

import logging

from app.llm import prompts
from app.llm.client import LLM, est_tokens
from app.llm.json_utils import LLMJsonError
from app.models import GapOut
from app.research.notes import Finding, render_facts

log = logging.getLogger(__name__)

_STATE_MAX_TOKENS = 2000  # hard truncate guard on top of the prompt's word cap


def render_round_findings(findings: list[Finding]) -> str:
    if not findings:
        return "(no new relevant sources this round)"
    blocks = []
    for f in findings:
        facts = render_facts(f.key_facts, indent="  ", quotes=False, limit=6)
        blocks.append(f"{f.citation_line()}\n  {f.summary}\n{facts}".rstrip())
    return "\n".join(blocks)


def _truncate_state(state_md: str) -> str:
    if est_tokens(state_md) <= _STATE_MAX_TOKENS:
        return state_md
    return state_md[: _STATE_MAX_TOKENS * 3] + "\n\n[state truncated]"


def _fresh(queries: list[str], searched: list[str], breadth: int) -> list[str]:
    seen = {s.lower().strip() for s in searched}
    return [q for q in queries if q.lower().strip() not in seen][:breadth]


def _keep_fresh(out: GapOut, searched: list[str], breadth: int) -> None:
    """Drop already-searched queries AND their aligned tags, in place.

    Filtering next_queries alone shifted next_query_facets and
    next_query_scopes a slot left for every query dropped — and dropping is
    the normal case here, since the prompt forbids repeats and this filter
    exists precisely to catch the ones it proposes anyway. The result was a
    round crediting its sources to the wrong part of the question and
    searching them in the wrong scope (2026-09-09).
    """
    seen = {s.lower().strip() for s in searched}
    keep = [i for i, q in enumerate(out.next_queries)
            if q.lower().strip() not in seen][:breadth]

    def picked(xs: list[str]) -> list[str]:
        return [xs[i] if i < len(xs) else "" for i in keep]

    out.next_query_facets = picked(out.next_query_facets)
    out.next_query_scopes = picked(out.next_query_scopes)
    out.next_queries = [out.next_queries[i] for i in keep]


async def _retry_for_queries(llm: LLM, prompt: str, searched: list[str],
                             breadth: int, previous: GapOut) -> GapOut:
    """One stern re-ask when gap analysis proposes nothing while not saturated.

    Mirrors the synthesis retry: a single bad structured answer should not be
    allowed to end work that has depth remaining.
    """
    stern = (prompt + "\n\nIMPORTANT: your previous answer left next_queries "
             "empty while also reporting that the research is NOT saturated. "
             "Those two cannot both be true. Every query you propose now must "
             "be genuinely NEW — not one of the already-searched queries "
             "above, and not a reworded version of one. Attack a facet of the "
             "brief that has not been searched at all: a different component "
             "or subsystem, a different failure mode, or a different KIND of "
             "source (hands-on forum thread, video walkthrough, manufacturer "
             "documentation, specification table). If you genuinely cannot "
             "name one, set saturated to true instead.")
    try:
        out = await llm.chat_json(
            "gap", [{"role": "user", "content": stern}],
            GapOut, max_tokens=3000, temperature=0.6,
        )
    except LLMJsonError as e:
        log.warning("gap retry for queries failed: %s", e)
        return previous
    out.state_md = _truncate_state(out.state_md) or previous.state_md
    _keep_fresh(out, searched, breadth)
    if out.next_queries:
        log.info("gap retry recovered %d queries", len(out.next_queries))
    return out


async def analyze(llm: LLM, *, query: str, brief: str, recency_desc: str,
                  round_no: int, depth: int, breadth: int, state_md: str,
                  new_findings: list[Finding], searched: list[str],
                  authority: str = "", variant: str = "default",
                  coverage: str = "") -> GapOut:
    authority_block = (prompts.AUTHORITY_BLOCK.format(authority_sites=authority)
                       if authority else "")
    coverage_block = (prompts.COVERAGE_BLOCK.format(coverage=coverage)
                      if coverage else "")
    # "anchored" is under A/B test (GAP_VARIANT); the stern re-ask below
    # builds on the same prompt, so it inherits the variant.
    template = prompts.GAP_ANCHORED if variant == "anchored" else prompts.GAP
    prompt = template.format(
        round=round_no, depth=depth, query=query, brief=brief,
        recency_desc=recency_desc, state_md=state_md or "(empty)",
        round_findings=render_round_findings(new_findings),
        searched="\n".join(f"- {q}" for q in searched) or "(none)",
        breadth=breadth, authority_block=authority_block,
        coverage_block=coverage_block,
    )
    try:
        out = await llm.chat_json(
            "gap", [{"role": "user", "content": prompt}],
            GapOut, max_tokens=3000, temperature=0.3,
        )
        out.state_md = _truncate_state(out.state_md) or state_md
        _keep_fresh(out, searched, breadth)
        if not out.next_queries and not out.saturated:
            # The model reports gaps remain but named no way to attack them --
            # usually because everything it proposed was a repeat and the
            # filter above emptied the list. The pipeline stops the whole run
            # on an empty list, so this ends a depth-10 run at round 3 on a
            # single malformed verdict. Ask once more, explicitly.
            out = await _retry_for_queries(llm, prompt, searched, breadth, out)
        return out
    except LLMJsonError as e:
        # degrade: keep old state, propose nothing (pipeline treats as a dry signal)
        log.warning("gap analysis degraded (round %d): %s", round_no, e)
        return GapOut(state_md=state_md, saturated=False, next_queries=[])
