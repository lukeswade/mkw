"""Query unpacking: research question → title, brief, initial sub-queries."""
from __future__ import annotations

import logging

from app.llm import prompts
from app.llm.client import LLM
from app.llm.json_utils import LLMJsonError
from app.models import PlannerOut

log = logging.getLogger(__name__)


async def plan(llm: LLM, *, query: str, recency_desc: str, today: str,
               breadth: int, prior: str = "", authority: str = "") -> PlannerOut:
    prior_block = prompts.PRIOR_BLOCK.format(prior=prior) if prior else ""
    authority_block = (prompts.AUTHORITY_BLOCK.format(authority_sites=authority)
                       if authority else "")
    prompt = prompts.PLANNER.format(
        query=query, recency_desc=recency_desc, today=today,
        breadth=breadth, prior_block=prior_block,
        authority_block=authority_block,
    )
    try:
        out = await llm.chat_json(
            "planner", [{"role": "user", "content": prompt}],
            PlannerOut, max_tokens=1500, temperature=0.5,
        )
        out.subqueries = out.subqueries[:breadth]
        return out
    except LLMJsonError as e:
        # degrade: research the raw query directly
        log.warning("planner degraded to raw query: %s", e)
        return PlannerOut(title=query[:120], brief=query, subqueries=[query])
