"""Facet coverage: a long question's separate asks each get research.

The failure these guard against (2026-09-04): an eleven-deliverable prompt
produced 27 queries, 26 naming one vendor; four deliverables were never
searched, and the report had a confident section for each anyway.
"""
from __future__ import annotations

from collections import Counter

import httpx
import respx

from app.db import Repo, connect
from app.models import GapOut, PlannerOut
from app.research import facets as f
from app.research.orchestrator import Orchestrator
from app.research.progress import ProgressBus
from app.research.searcher import generalize_query, query_terms
from tests.fake_llm import FakeLLM
from tests.test_pipeline_e2e import SX, article, make_cfg, sx_payload, sx_result
from app.models import RunParams


# ---- the facet table ------------------------------------------------------------

def test_facets_are_normalized_deduplicated_and_capped():
    assert f.clean_facets(["  Cost of Ownership ", "COST OF OWNERSHIP", "safety"]) \
        == ["cost of ownership", "safety"]
    assert len(f.clean_facets([f"facet {i}" for i in range(30)])) == f.MAX_FACETS
    assert f.clean_facets([None, "", "  "]) == []


def test_a_renamed_facet_tag_still_counts_toward_its_facet():
    """Models rename their own facets between stages; dropping the tag would
    silently zero a facet's coverage and re-search it forever."""
    facets = ["commercial licensing terms", "ui embedding"]
    got = f.align(facets, ["q1", "q2", "q3"],
                  ["Commercial Licensing Terms", "embedding", "nonsense xyz"])
    assert got == {"q1": "commercial licensing terms", "q2": "ui embedding"}
    assert f.align(facets, ["q1"], []) == {}          # untagged is not an error


def test_no_facet_takes_more_than_a_third_of_a_round_while_another_has_none():
    facets = ["cost", "safety", "installation"]
    proposed = [f"cost q{i}" for i in range(6)]
    facet_of = {q: "cost" for q in proposed}
    chosen, starved, spare = f.allocate(proposed, facet_of, Counter(), facets, 6)
    assert chosen == ["cost q0", "cost q1"]           # ceil(6/3)
    assert starved == ["safety", "installation"]      # need a query of their own
    assert spare == proposed[2:]                      # displaced, not lost


def test_slots_go_to_the_thinnest_facet_first():
    facets = ["cost", "safety", "installation"]
    proposed = ["c1", "s1", "i1", "c2"]
    facet_of = {"c1": "cost", "c2": "cost", "s1": "safety", "i1": "installation"}
    kept = Counter({"cost": 9, "safety": 1})
    chosen, starved, _ = f.allocate(proposed, facet_of, kept, facets, 3)
    assert chosen == ["i1", "s1", "c1"]               # 0 sources, then 1, then 9
    assert starved == []


def test_untagged_queries_still_run_but_a_starved_facet_keeps_its_slot():
    """A planner that names facets and forgets to tag its queries must not
    lose the round: untagged queries belong to no facet, so they cannot
    breach the cap. One slot is still held for the facet with nothing."""
    chosen, starved, spare = f.allocate(["a", "b", "c"], {}, Counter(), ["cost"], 3)
    assert chosen == ["a", "b"] and starved == ["cost"] and spare == ["c"]
    chosen, starved, spare = f.allocate(["a", "b"], {}, Counter(), [], 5)
    assert chosen == ["a", "b"] and starved == [] and spare == []


def test_a_top_up_query_names_the_subject_and_drops_filler():
    assert f.facet_query("how it compares with competitors", "Workato") \
        == "Workato compares competitors"
    assert f.facet_query("call prep", "") == "call prep"
    assert f.subject_terms("Workato AIRO Feasibility", ["Agent Studio", "x"]) == "Agent Studio"
    assert f.subject_terms("Workato AIRO Feasibility Study", []) == "Workato AIRO"


def test_uncovered_names_what_produced_nothing():
    assert f.uncovered(["cost", "safety"], Counter({"cost": 3})) == ["safety"]
    assert "cost: 3" in f.coverage_lines(["cost", "safety"], Counter({"cost": 3}))
    assert "safety: 0" in f.coverage_lines(["cost", "safety"], Counter({"cost": 3}))


# ---- the question's own list ---------------------------------------------------

# The shape a person actually writes a brief in: a numbered block and a
# colon-led block of plain lines, CRLF from a browser textarea.
_BRIEF = (
    "I'm evaluating a platform for my team.\r\n\r\n"
    "Primary use cases:\r\n"
    "1. Call Prep\r\n"
    "2. Completed Call Briefs\r\n"
    "3. Organization health checks\r\n\r\n"
    "Your report should include, but is not limited to:\r\n\r\n"
    "recommendations, tips and tricks\r\n"
    "caveats and known functionality gaps\r\n"
    "how it compares with competing platforms\r\n"
    "anything and everything else\r\n"
)


def test_the_questions_own_list_is_read_from_the_question():
    """The planner folded four numbered use cases into one facet and nothing
    downstream could recover them. A list the question already wrote is not a
    judgement call, so it is taken from the text."""
    assert f.enumerated(_BRIEF) == [
        "call prep",
        "completed call briefs",
        "organization health checks",
        "recommendations tips tricks",
        "caveats known functionality gaps",
        "compares competing platforms",
    ]                                    # "anything and everything else" names no research


def test_prose_after_a_colon_is_not_a_list():
    assert f.enumerated("One question: does it support custom objects?") == []
    # two lines is a sentence that wrapped; three is a list
    assert f.enumerated("Cover this:\nthe cost\nthe risk") == []
    assert f.enumerated("Cover this:\nthe cost\nthe risk\nthe timeline") == [
        "cost", "risk", "timeline"]
    assert f.enumerated("") == [] and f.enumerated("no lists here at all") == []


def test_a_very_long_list_item_is_cut_on_a_word_boundary():
    item = "- the feasibility of creating a standard base agent that " \
           "evolves and adapts and specializes based on customer tech stack " \
           "and data structure and operational processes\n- second item here"
    name = f.enumerated(item)[0]
    assert len(name) <= 110 and not name.endswith(" ") and name.split()[-1].isalpha()


def test_merge_puts_the_questions_own_asks_first_and_drops_the_planners_repeats():
    got = f.merge(["call prep agent", "pricing"], ["call prep", "completed call briefs"])
    assert got == ["call prep", "completed call briefs", "pricing"]
    assert f.merge(["only what the model named"], []) == ["only what the model named"]


# ---- credit only for a source about the facet ------------------------------------

# Verbatim from the run that exposed this: the query was "Workato AIRO call
# prep" and this is the page it kept, whose own summary says it lacks the
# use cases. It credited the call-prep facet, so nothing was reported missing.
_OVERVIEW = ("AIRO — This document provides a high-level overview of Workato AIRO's "
             "capabilities, including its multi-agent architecture, blueprint planning "
             "and MCP server integration. It lacks specific technical details on "
             "multi-tenant agent customization, Insightly-specific use cases, or "
             "competitor comparisons.")


def test_a_facet_is_credited_only_for_a_source_about_it():
    assert not f.about("call prep", _OVERVIEW)
    assert f.about("call prep", "How to run AI call prep before a sales call")
    # plurals and punctuation must not decide it
    assert f.about("organization/contact health checks summaries overviews",
                   "Account health check scoring for every contact organization")
    # one-word facets need only that word; an unnamed facet credits anything
    assert f.about("pricing", "Workato pricing tiers explained") and not f.about("pricing", _OVERVIEW)
    assert f.about("", _OVERVIEW)


def test_an_ask_the_question_listed_is_searched_plainly_before_the_vendor():
    """Five vendor-anchored use-case queries kept five vendor overview pages
    and nothing about the use cases."""
    assert f.top_up_queries("call prep", "Workato AIRO", from_question=True) \
        == ["call prep", "Workato AIRO call prep"]
    # a facet the planner invented describes the subject, so it keeps the anchor
    assert f.top_up_queries("dynamic agent personalization", "Workato AIRO", from_question=False) \
        == ["Workato AIRO dynamic agent personalization", "dynamic agent personalization"]
    # no subject to anchor to: one query, not a duplicate pair
    assert f.top_up_queries("call prep", "", from_question=True) == ["call prep"]


# ---- the model writes the query for an unanswered ask -----------------------------

def test_two_names_for_one_ask_merge():
    """The question asked about "competing platforms"; the planner called the
    same thing "competitor landscape comparison". Both survived, and one was
    reported unanswered while the other held ten sources."""
    asked = ["workato airo compares competing platforms tools amazon agentcore competitors"]
    assert f.merge(["competitor landscape comparison"], asked) == asked
    # genuinely different asks still stand apart
    assert len(f.merge(["oem pricing model"], ["insightly ui embedding"])) == 2


async def test_the_model_writes_the_query_and_the_mechanical_form_is_the_fallback():
    from app.research import facet_queries
    llm = FakeLLM({"facet_queries": [{
        "facets": ["Call Prep", "unknown facet"],
        "queries": ["AI agent call prep CRM sales", "ignored"],
        "scopes": ["web+video"]}]})
    got = await facet_queries.write(llm, query="q", brief="b",
                                    facets=["call prep", "completed call briefs"],
                                    searched=["something else"])
    assert got == {"call prep": ("AI agent call prep CRM sales", "web+video")}

    # a query already searched is not offered again
    llm2 = FakeLLM({"facet_queries": [{"facets": ["call prep"],
                                       "queries": ["AI agent call prep CRM sales"]}]})
    assert await facet_queries.write(llm2, query="q", brief="b", facets=["call prep"],
                                     searched=["ai agent call prep crm sales"]) == {}

    # the stage never stops a round: an unscripted model degrades to nothing,
    # and the caller falls back to the mechanical form
    assert await facet_queries.write(FakeLLM({}), query="q", brief="b",
                                     facets=["call prep"], searched=[]) == {}
    assert await facet_queries.write(FakeLLM({}), query="q", brief="b",
                                     facets=[], searched=[]) == {}


# ---- the zero-result retry ------------------------------------------------------

def test_a_query_that_matched_nothing_is_thinned_of_its_invented_names():
    known = query_terms("Does Workato AIRO support multi-tenant agents for Insightly CRM?")
    # Genie and JSON are the planner's own invention: the question never used them
    assert generalize_query("Workato AIRO Genie structured output JSON schema", known) \
        == "Workato AIRO structured output schema"
    # every name came from the question: keep the subject, drop the rest
    assert generalize_query("Workato AIRO Insightly CRM integration case study", known) \
        == "Workato integration case study"
    # nothing to thin — the caller accepts the empty result rather than re-searching
    assert generalize_query("fly rod reel seat repair", known) == ""
    assert generalize_query("Workato pricing", known) == ""


# ---- end to end -----------------------------------------------------------------

def _script(gap_rounds):
    return {
        "triage": [{"drop": []}],
        "planner": [{
            "title": "Home Battery Buying Guide",
            "brief": "Cost, safety certification and installation of home batteries.",
            "facets": ["cost", "safety certification", "installation"],
            "subqueries": ["battery cost per kwh", "battery cost trend",
                           "battery cost comparison", "battery cost forecast"],
            "query_facets": ["cost", "cost", "cost", "cost"],
            "keywords": ["battery"],
        }],
        "notes": [{"relevance": 8, "summary": "Useful.", "notes_md": "Notes.",
                   "key_facts": [], "published_date": None}],
        "gap": gap_rounds,
        "synth": ["# Home Battery Buying Guide\n\n## TL;DR\n\n- Cost is falling [1].\n"],
        "followups": [{"items": []}],
    }


@respx.mock
async def test_a_long_question_spends_its_slots_on_the_parts_with_no_sources(data_dir):
    """Round 1's queries all attack one facet. Two of them run; the slots the
    cap frees go to the facets nothing has answered, and the facets that stay
    unanswered are named in the report instead of being written up blind."""
    cfg = make_cfg(data_dir)
    seen: list[str] = []

    def handler(req):
        q = req.url.params.get("q", "")
        seen.append(q)
        if "cost" in q:                      # only the cost facet has anything to find
            return httpx.Response(200, json=sx_payload(
                [sx_result(f"https://cost{len(seen)}.example.com/p",
                           f"Battery cost analysis {len(seen)}")]))
        if "safety" in q:
            # the failure this guards: a facet's own query turns up a page
            # about something else, which used to credit the facet
            return httpx.Response(200, json=sx_payload(
                [sx_result("https://offtopic.example.com/p", "Battery cost analysis extra")]))
        return httpx.Response(200, json=sx_payload([]))

    respx.get(f"{SX}/search").mock(side_effect=handler)
    respx.get(url__regex=r"https://cost\d+\.example\.com/p").mock(
        side_effect=lambda req: httpx.Response(
            200, html=article(f"Battery cost analysis from {req.url.host}")))
    respx.get("https://offtopic.example.com/p").mock(
        return_value=httpx.Response(200, html=article("Battery cost analysis extra")))

    seen_prompts: dict[str, list[str]] = {}

    class Capture(FakeLLM):
        async def chat_json(self, kind, messages, schema, **kw):
            seen_prompts.setdefault(kind, []).append(messages[0]["content"])
            return await super().chat_json(kind, messages, schema, **kw)

    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: Capture(_script([
                            {"state_md": "s", "saturated": False,
                             "next_queries": ["battery cost warranty"],
                             "next_query_facets": ["cost"]},
                            {"state_md": "s", "saturated": True, "next_queries": []}])))
    run_id = orch.enqueue(RunParams(query="home battery cost, safety certification "
                                          "and installation", depth=4,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)

    round1 = seen[:4]
    assert sum(1 for q in round1 if "cost" in q) == 2, round1     # capped at a third
    assert any("safety certification" in q for q in seen), seen   # starved facets searched
    assert any("installation" in q for q in seen), seen

    overview = (cfg.research_dir / run_id / "overview.md").read_text()
    assert "## Not researched" in overview
    # the safety query DID keep a source; it was about cost, so the facet is
    # still reported unanswered rather than quietly counted as covered
    assert "safety certification" in overview and "installation" in overview
    events = (cfg.research_dir / run_id / "events.jsonl").read_text()
    assert "no sources yet for" in events
    assert "about something else" in events

    # Both judging stages are told which part of the question a page was
    # fetched for, so a page about the general practice is not thrown away
    # for failing to mention the product the brief centres on.
    triage_prompts = "\n".join(seen_prompts.get("triage", []))
    assert "— for the part: cost" in triage_prompts
    assert "Do NOT drop such a candidate" in triage_prompts
    notes_prompts = "\n".join(seen_prompts.get("notes", []))
    assert "PART OF THE QUESTION THIS SOURCE WAS FETCHED FOR: cost" in notes_prompts
    assert "FETCHED FOR: safety certification" in notes_prompts   # the off-topic page still carried its part
