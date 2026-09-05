"""A search query for a part of the question nothing has answered.

The pipeline used to build these from the facet's own words, in two forms,
and both were bad. Plain, "call prep" returned a Russian-English dictionary,
an internet calling service and a nonprofit called Prep for Prep. Anchored,
"Workato AIRO call prep" returned the vendor's sign-in page and its API
marketing. Nine such queries across one run produced no sources at all.

The model is good at this and proved it in the same run: gap analysis, asked
nothing about facets, wrote "Build AI agent for call prep CRM automation" of
its own accord. So it is asked directly. One call per round, and any failure
falls back to the mechanical form — a search stage must never stop a run.
"""
from __future__ import annotations

import logging

from app.llm import prompts
from app.llm.client import LLM
from app.models import FacetQueriesOut
from app.research import facets as facet_plan

log = logging.getLogger(__name__)


async def write(llm: LLM, *, query: str, brief: str, facets: list[str],
                searched: list[str]) -> dict[str, tuple[str, str]]:
    """facet -> (query, scope) for the facets the model answered. Missing
    facets are the caller's problem, not an error: it has a fallback."""
    if not facets:
        return {}
    prompt = prompts.FACET_QUERIES.format(
        query=query, brief=brief,
        facets="\n".join(f"- {f}" for f in facets),
        searched="\n".join(f"- {q}" for q in searched[-40:]) or "(none)",
    )
    try:
        out = await llm.chat_json(
            "facet_queries", [{"role": "user", "content": prompt}],
            FacetQueriesOut, max_tokens=800, temperature=0.3)
    except Exception as e:  # noqa: BLE001 — never let this stop a round
        log.debug("facet query stage degraded to the mechanical form: %s", e)
        return {}
    known = {facet_plan.normalize(f): f for f in facets}
    seen = {q.strip().lower() for q in searched}
    written: dict[str, tuple[str, str]] = {}
    for i, name in enumerate(out.facets):
        if i >= len(out.queries):
            break
        key = facet_plan.normalize(name)
        facet = known.get(key) or known.get(facet_plan.normalize(
            facet_plan._closest(key, list(known))))
        text = out.queries[i].strip()
        if not facet or not text or text.lower() in seen:
            continue
        scope = out.scopes[i].strip().lower() if i < len(out.scopes) else "web"
        written[facet] = (text, scope or "web")
    return written
