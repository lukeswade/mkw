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


async def analyze(llm: LLM, *, query: str, brief: str, recency_desc: str,
                  round_no: int, depth: int, breadth: int, state_md: str,
                  new_findings: list[Finding], searched: list[str],
                  authority: str = "") -> GapOut:
    authority_block = (prompts.AUTHORITY_BLOCK.format(authority_sites=authority)
                       if authority else "")
    prompt = prompts.GAP.format(
        round=round_no, depth=depth, query=query, brief=brief,
        recency_desc=recency_desc, state_md=state_md or "(empty)",
        round_findings=render_round_findings(new_findings),
        searched="\n".join(f"- {q}" for q in searched) or "(none)",
        breadth=breadth, authority_block=authority_block,
    )
    try:
        out = await llm.chat_json(
            "gap", [{"role": "user", "content": prompt}],
            GapOut, max_tokens=3000, temperature=0.3,
        )
        out.state_md = _truncate_state(out.state_md) or state_md
        out.next_queries = [q for q in out.next_queries
                            if q.lower() not in {s.lower() for s in searched}][:breadth]
        return out
    except LLMJsonError as e:
        # degrade: keep old state, propose nothing (pipeline treats as a dry signal)
        log.warning("gap analysis degraded (round %d): %s", round_no, e)
        return GapOut(state_md=state_md, saturated=False, next_queries=[])
