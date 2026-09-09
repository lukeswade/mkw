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


# ---- credit by the note-taker's answer, vetoed on zero shared words -------------

def test_a_part_is_credited_by_the_note_takers_answer_with_a_one_word_floor():
    """"health checks" was reported unanswered while three customer-health-
    SCORING guides sat in the findings: the reader's word and the field's word
    differ, and the two-word rule could not bridge it. The note-taker, asked
    how much the page contributes to that part, can."""
    facet = "organization/contact health checks summaries overviews"
    scoring = "Customer health scores in the age of AI. Modern customer health scoring methodologies."
    assert f.credits(facet, scoring, part_relevance=5, threshold=4)
    assert not f.credits(facet, scoring, part_relevance=2, threshold=4)
    # the same note-taker scored a vendor launch press release 6 under the
    # risk part; zero shared words is the veto that keeps that out
    launch = "Workato Launches AIRO for Enterprise Multi-Agent Automation. Announces global availability."
    assert not f.credits("customer dispute/conflict/risk identification mitigation", launch, part_relevance=6, threshold=4)
    # no answer from the note-taker: the two-word rule stands
    assert not f.credits(facet, scoring, part_relevance=None, threshold=4)
    assert f.credits("call prep", "AI call prep for sales teams", part_relevance=None, threshold=4)


def test_a_short_facet_sharing_one_distinctive_word_merges_but_not_on_the_questions_furniture():
    asked = ["recommended agents team build as essentially template then distributed per-customer basis"]
    assert f.merge(["template specialization"], asked) == asked          # one distinctive word: same ask
    # "insightly" is in three facets here, so sharing it proves nothing
    asked2 = ["insightly ui embedding", "insightly api access", "insightly data model"]
    assert len(f.merge(["insightly pricing"], asked2)) == 4


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
                   "key_facts": [], "published_date": None, "part_relevance": 8}],
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


# ---- checking what the question assumes ------------------------------------
# The U8 run built five sections on "the fields we play on are way too small"
# without ever checking it against US Soccer's published 4v4 dimensions. If the
# fields are standard, the advice changes completely.

def test_a_premise_without_a_query_cannot_be_checked_so_it_is_dropped():
    from app.models import PlannerOut
    p = PlannerOut(title="t", subqueries=["a"],
                   premises=["fields are too small", "cleats are wrong"],
                   premise_queries=["us soccer 4v4 field dimensions"])
    assert p.premises == ["fields are too small"]
    assert p.premise_queries == ["us soccer 4v4 field dimensions"]


def test_premises_are_capped_at_two():
    from app.models import PlannerOut
    p = PlannerOut(title="t", subqueries=["a"],
                   premises=["a", "b", "c", "d"],
                   premise_queries=["qa", "qb", "qc", "qd"])
    assert len(p.premises) == 2 and len(p.premise_queries) == 2


def test_a_question_with_no_checkable_premise_spends_nothing():
    from app.models import PlannerOut
    p = PlannerOut(title="t", subqueries=["a"])
    assert p.premises == [] and p.premise_queries == []


@respx.mock
async def test_the_premise_is_searched_and_answered_before_the_question(data_dir):
    """The premise query runs in round one on top of the facet allocation, its
    source is credited to the premise for grouping, and synthesis is told to
    open with the verdict."""
    cfg = make_cfg(data_dir)
    seen: list[str] = []

    def handler(req):
        q = req.url.params.get("q", "")
        seen.append(q)
        host = "standard" if "dimensions" in q else "drills"
        return httpx.Response(200, json=sx_payload(
            [sx_result(f"https://{host}.example.com/p", f"{host} page")]))

    respx.get(f"{SX}/search").mock(side_effect=handler)
    for host in ("standard", "drills"):
        respx.get(f"https://{host}.example.com/p").mock(
            return_value=httpx.Response(200, html=article(f"All about {host}")))

    sc = _script([{"state_md": "s", "saturated": True, "next_queries": []}])
    sc["planner"] = [{
        "title": "Coaching U8", "brief": "How to coach a U8 team.",
        "facets": ["coaching drills"],
        "subqueries": ["u8 coaching drills"],
        "query_facets": ["coaching drills"],
        "keywords": ["u8"],
        "premises": ["the fields we play on are way too small"],
        "premise_queries": ["us youth soccer 4v4 u8 field dimensions"],
    }]
    seen_prompts: list[str] = []

    class Capture(FakeLLM):
        async def chat(self, kind, messages, **kw):
            if kind == "synth":
                seen_prompts.append(messages[0]["content"])
            return await super().chat(kind, messages, **kw)

    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: Capture(sc))
    run_id = orch.enqueue(RunParams(query="how do I coach my u8 team, the "
                                          "fields are way too small",
                                    depth=2, recency="all", origin="cli"))
    await orch.execute_now(run_id)

    assert any("dimensions" in q for q in seen), seen
    synth = "\n".join(seen_prompts)
    assert "Checking what the question assumes" in synth
    assert "the fields we play on are way too small" in synth
    events = (cfg.research_dir / run_id / "events.jsonl").read_text()
    assert "the question assumes" in events


# ---- the three regressions the first premise run exposed --------------------

def test_a_premise_query_credits_the_part_of_the_question_it_answers():
    """The run that shipped premise checking closed by claiming it had not
    researched field dimensions, under a section citing six sources about
    them: the premise query's sources were credited to the premise label, so
    the planner's own facet finished on zero."""
    fs = ["field dimension standards", "age-appropriate coaching methods"]
    assert f.facet_for_query(
        "US Youth Soccer U8 4v4 field dimensions guidelines", fs
    ) == "field dimension standards"


def test_one_shared_word_is_not_enough_to_claim_a_part_was_covered():
    """Crediting a part that was never really searched is worse than
    reporting it uncovered, so the bar is two shared stems."""
    fs = ["skill development drills", "field dimension standards"]
    assert f.facet_for_query("soccer drills for kids", fs) == ""
    assert f.facet_for_query("how to boil an egg", fs) == ""


def test_a_premise_with_no_matching_part_keeps_its_own_label():
    assert f.facet_for_query("us soccer field dimensions", []) == ""


def test_a_standard_must_name_the_body_that_published_it():
    from app.models import NotesOut
    real = NotesOut(relevance=9, source_type="standard", publisher="US Youth Soccer")
    assert real.source_type == "standard"
    # a retailer restating a governing body's table reports a rule, it does
    # not publish one — ten such pages were ranked as standards in one run
    reported = NotesOut(relevance=8, source_type="standard")
    assert reported.source_type == "aggregator"
    assert NotesOut(relevance=8, source_type="standard", publisher="   ").source_type == "aggregator"


def test_the_premise_cap_drops_retailers_and_keeps_the_rule_maker():
    """A flat top-N would have lost the answer: US Soccer's own document
    ranked tenth of ten, behind the equipment shops."""
    from app.research.pipeline import cap_premise_results
    from app.research.searcher import SearchResult

    def r(url, via="premise q"):
        return SearchResult(url=url, title="t", snippet="s", engine="e",
                            published=None, score=1.0, via_query=via)

    results = ([r(f"https://shop{i}.com/dimensions") for i in range(9)]
               + [r("https://usyouthsoccer.org/pdi")]
               + [r("https://other.com/p", via="a facet query")])
    out = cap_premise_results(results, ["premise q"], cap=2)
    urls = [x.url for x in out]
    assert "https://usyouthsoccer.org/pdi" in urls      # the rule-maker survives
    assert "https://other.com/p" in urls                # other queries untouched
    assert len([u for u in urls if "shop" in u]) == 1    # cap 2 = org + one shop
    assert len(urls) == 3


def test_the_premise_cap_is_a_no_op_below_the_cap_and_without_premises():
    from app.research.pipeline import cap_premise_results
    from app.research.searcher import SearchResult
    rs = [SearchResult(url=f"https://a{i}.com/p", title="t", snippet="s",
                       engine="e", published=None, score=1.0, via_query="pq")
          for i in range(3)]
    assert cap_premise_results(rs, ["pq"], cap=4) == rs
    assert cap_premise_results(rs, []) == rs
    assert cap_premise_results([], ["pq"]) == []


def test_an_untagged_gap_query_is_credited_by_what_it_is_about():
    """The gap stage often returns next_queries with no next_query_facets, and
    an untagged query used to count toward nothing: six sources answering
    'small field tactics' were credited nowhere, the run called that facet
    unresearched under a section built from them, and round two re-attacked a
    facet that was already covered (run 20260909_063444)."""
    facets = ["small field tactics", "skill development drills",
              "team organization strategies", "age appropriate coaching"]
    got = f.align(facets, ["u8 soccer practice plan 60 minutes small field"], [])
    assert got == {"u8 soccer practice plan 60 minutes small field":
                   "small field tactics"}


def test_the_fallback_stays_quiet_when_a_query_matches_nothing():
    """An untagged query still counts toward nothing rather than being forced
    onto the nearest facet — over-crediting hides a real gap."""
    facets = ["small field tactics", "skill development drills"]
    for q in ("preventing swarming 4v4 u8 soccer",
              "how to coach mixed skill u8 soccer team",
              "managing talented player u8 soccer team"):
        assert f.align(facets, [q], []) == {}, q


def test_an_explicit_tag_still_beats_the_fallback():
    """The model's own tag is evidence; the fallback is only for its absence."""
    facets = ["small field tactics", "age appropriate coaching"]
    q = "u8 soccer practice plan 60 minutes small field"
    assert f.align(facets, [q], ["age appropriate coaching"]) == {q: "age appropriate coaching"}
    assert f.align(facets, ["anything"], ["tactics"]) == {"anything": "small field tactics"}


# ---- index-aligned model arrays cannot drift --------------------------------
# Two independent reviewers found the same defect: every model that emits a
# query list plus tag lists aligned BY INDEX cleaned each field on its own
# criteria, so one blank query shifted every later tag a slot left. A round
# then credited its sources to the wrong part of the question and searched
# them in the wrong scope.

def test_a_blank_query_takes_its_tags_with_it():
    from app.models import GapOut
    g = GapOut(next_queries=["", "field dimensions u8", "coaching drills u8"],
               next_query_facets=["dropped", "field dimension standards",
                                  "skill development drills"],
               next_query_scopes=["web", "web", "video"])
    assert g.next_queries == ["field dimensions u8", "coaching drills u8"]
    assert g.next_query_facets == ["field dimension standards",
                                   "skill development drills"]
    assert g.next_query_scopes == ["web", "video"]


def test_a_short_tag_list_is_padded_not_shifted():
    from app.models import PlannerOut
    p = PlannerOut(title="t", subqueries=["a q", "b q", "c q"],
                   query_facets=["alpha"], query_scopes=["web"])
    assert p.subqueries == ["a q", "b q", "c q"]
    assert p.query_facets == ["alpha", "", ""]
    assert p.query_scopes == ["web", "", ""]


def test_an_omitted_tag_list_stays_omitted():
    from app.models import PlannerOut
    p = PlannerOut(title="t", subqueries=["a q"])
    assert p.query_facets == [] and p.query_scopes == []


def test_a_part_with_no_query_written_for_it_is_dropped_whole():
    """facet_queries.write() iterates facets and indexes queries, so a facet
    with no query would silently take the NEXT facet's query."""
    from app.models import FacetQueriesOut
    o = FacetQueriesOut(facets=["cost", "safety", "install"],
                        queries=["cost q", "", "install q"],
                        scopes=["web", "web", "video"])
    assert o.facets == ["cost", "install"]
    assert o.queries == ["cost q", "install q"]
    assert o.scopes == ["web", "video"]


def test_gap_drops_an_already_searched_query_with_its_tags():
    from app.models import GapOut
    from app.research.gap import _keep_fresh
    g = GapOut(next_queries=["old q", "new q", "other q"],
               next_query_facets=["stale", "field dimension standards", "drills"],
               next_query_scopes=["web", "video", "web"])
    _keep_fresh(g, ["OLD Q"], breadth=8)          # case-insensitive match
    assert g.next_queries == ["new q", "other q"]
    assert g.next_query_facets == ["field dimension standards", "drills"]
    assert g.next_query_scopes == ["video", "web"]


def test_gap_breadth_truncation_also_keeps_the_pairing():
    from app.models import GapOut
    from app.research.gap import _keep_fresh
    g = GapOut(next_queries=["a", "b", "c"],
               next_query_facets=["fa", "fb", "fc"],
               next_query_scopes=["web", "video", "news"])
    _keep_fresh(g, [], breadth=2)
    assert g.next_queries == ["a", "b"]
    assert g.next_query_facets == ["fa", "fb"]
    assert g.next_query_scopes == ["web", "video"]
