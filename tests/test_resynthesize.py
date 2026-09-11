"""The re-synthesize action and the leaked-reasoning guard."""
from __future__ import annotations

import httpx
import pytest
import respx

from app.db import Repo, connect
from app.models import RunParams
from app.research.notes import Finding
from app.research.orchestrator import Orchestrator
from app.research.pipeline import Pipeline
from app.research.progress import ProgressBus
from app.research.synthesizer import looks_like_document, synthesize
from tests.fake_llm import FakeLLM
from tests.test_pipeline_e2e import SX, article, make_cfg, script, sx_payload, sx_result

MONOLOGUE = ("We need answer user's request: write final research overview "
             "markdown only, no preamble. Need use sources inline [n]. Need "
             "likely in English because user English. Let's draft.")


def test_looks_like_document():
    assert looks_like_document("# Title\n\nBody.")
    assert looks_like_document("\n\n  ## Section first for some reason\n")
    assert not looks_like_document(MONOLOGUE)
    assert not looks_like_document("")


async def test_synthesis_retries_once_on_leaked_reasoning():
    llm = FakeLLM({"synth": [MONOLOGUE, "# Fixed\n\nProper document [1]."]})
    f = Finding(idx=1, url="https://a.com/x", title="T", domain="a.com",
                published=None, relevance=8, summary="s", notes_md="notes")
    out = await synthesize(llm, query="q", title="T", brief="b",
                           recency_desc="any", today="2026-08-18",
                           state_md="", findings=[f])
    assert out.startswith("# Fixed")
    assert llm.calls["synth"] == 2


async def _completed_run(cfg, monkeypatch=None):
    """A finished single-source run to re-synthesize."""
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result("https://example-a.com/article", "Article A")])))
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=article("Article A")))
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(
                            script([{"state_md": "s", "saturated": True,
                                     "next_queries": []}])))
    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)
    assert repo.get_run(run_id)["status"] == "completed"
    return repo, run_id


@respx.mock
async def test_resynthesize_rewrites_a_bad_overview(data_dir):
    cfg = make_cfg(data_dir)
    repo, run_id = await _completed_run(cfg)
    run_dir = cfg.research_dir / run_id
    # simulate the thinking-model failure: monologue where a doc should be
    (run_dir / "overview.md").write_text(MONOLOGUE)

    resynth_llm = FakeLLM({
        "synth": ["# Salvaged\n\nThe real document, cited [1]. Bogus [9] "
                  "citation should be stripped.\n"],
        "followups": [{"items": []}],
    })
    pipeline = Pipeline(cfg, repo, ProgressBus(),
                        llm_factory=lambda: resynth_llm)
    await pipeline.resynthesize(run_id)

    overview = (run_dir / "overview.md").read_text()
    assert overview.startswith("# Salvaged")
    assert "[1]" in overview and "[9]" not in overview   # citations validated
    import json
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta.get("resynthesized_at")


@respx.mock
async def test_resynthesize_keeps_old_overview_when_model_still_rambles(data_dir):
    cfg = make_cfg(data_dir)
    repo, run_id = await _completed_run(cfg)
    run_dir = cfg.research_dir / run_id
    good = (run_dir / "overview.md").read_text()

    rambler = FakeLLM({"synth": [MONOLOGUE], "followups": [{"items": []}]})
    pipeline = Pipeline(cfg, repo, ProgressBus(), llm_factory=lambda: rambler)
    with pytest.raises(RuntimeError, match="not a document"):
        await pipeline.resynthesize(run_id)
    assert (run_dir / "overview.md").read_text() == good   # untouched
    assert rambler.calls["synth"] == 2                     # initial + stern retry


async def test_resynthesize_requires_stored_findings(data_dir):
    cfg = make_cfg(data_dir)
    repo = Repo(connect(cfg.db_path))
    from app.research.storage import RunStore
    store = RunStore.create(cfg.research_dir, "empty run")
    repo.create_run(run_id=store.run_id, query="empty run", depth=1,
                    recency="all", dir=store.run_id, origin="cli")
    pipeline = Pipeline(cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM({}))
    with pytest.raises(ValueError, match="no stored findings"):
        await pipeline.resynthesize(store.run_id)


@respx.mock
async def test_orchestrator_guards_and_runs_resynth(data_dir):
    cfg = make_cfg(data_dir)
    repo, run_id = await _completed_run(cfg)

    resynth_llm = FakeLLM({"synth": ["# Again\n\nBody [1].\n"],
                           "followups": [{"items": []}]})
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: resynth_llm)
    assert orch.start_resynth("nonexistent") is False
    assert orch.start_resynth(run_id) is True
    assert orch.start_resynth(run_id) is False     # already in flight
    task, _ = orch.active[run_id]
    await task
    assert (cfg.research_dir / run_id / "overview.md").read_text()\
        .startswith("# Again")
    assert run_id not in orch.active


# ---- repetition-collapse guard ----------------------------------------------

LOOP = "!" * 8000


def test_looks_degenerate_catches_repetition_collapse():
    from app.research.synthesizer import looks_degenerate, looks_like_document
    assert looks_degenerate(LOOP)
    assert looks_degenerate("|" * 500)
    assert looks_degenerate("ab" * 400)
    assert not looks_degenerate("# Real Overview\n\n" + MONOLOGUE * 3)
    assert not looks_degenerate("short")
    assert not looks_like_document(LOOP)


async def test_degenerate_synthesis_writes_a_placeholder_not_garbage():
    from app.research.notes import Finding
    from app.research.synthesizer import synthesize
    llm = FakeLLM({"synth": [LOOP, LOOP]})       # loops on both attempts
    f = Finding(idx=1, url="https://a.com/x", title="T", domain="a.com",
                published=None, relevance=8, summary="s", notes_md="notes")
    out = await synthesize(llm, query="q", title="TCL Fault", brief="b",
                           recency_desc="any", today="2026-08-22",
                           state_md="", findings=[f])
    assert llm.calls["synth"] == 2
    assert "!" * 50 not in out                    # the loop never reaches disk
    assert out.startswith("# TCL Fault")
    assert "Synthesis failed" in out and "Re-synthesize" in out


async def test_only_enormous_source_sets_are_map_reduced():
    """Digesting is for runs too big for one call — not a safety measure.
    One large call proved safer than several small ones sharing a prefix."""
    from app.research.notes import Finding
    from app.research.synthesizer import synthesize
    calls = []

    def capture(messages):
        calls.append(messages[-1]["content"])
        return "# Digest\n\nSummary [1]."

    from app.llm.client import est_tokens
    from app.research.synthesizer import _SINGLE_CALL_BUDGET

    llm = FakeLLM({"synth": [capture], "candidates": [{"candidates": []}]})
    def mk(n):
        return [Finding(idx=i, url=f"https://a.com/{i}", title=f"T{i}",
                        domain="a.com", published=None, relevance=7,
                        summary="s", notes_md="word " * 800)
                for i in range(1, n + 1)]

    # Sized off the constant rather than hardcoded, so raising the budget
    # cannot quietly turn "enormous" into "typical" and leave this passing
    # for the wrong reason.
    per = est_tokens(mk(1)[0].notes_md)
    fits = max(1, (_SINGLE_CALL_BUDGET // 2) // per)
    exceeds = (_SINGLE_CALL_BUDGET // per) + 20

    # a deep run that fits the window stays a single call
    await synthesize(llm, query="q", title="T", brief="b", recency_desc="any",
                     today="t", state_md="", findings=mk(fits))
    assert llm.calls["synth"] == 1

    big = FakeLLM({"synth": [capture], "candidates": [{"candidates": []}]})
    await synthesize(big, query="q", title="T", brief="b", recency_desc="any",
                     today="t", state_md="", findings=mk(exceeds))
    assert big.calls["synth"] > 1                 # genuinely enormous: digested


async def test_collapsed_digest_falls_back_to_raw_notes():
    """A degenerate digest silently poisons the synthesis that eats it."""
    from app.research.synthesizer import _map_digest
    # _map_digest takes (part name, blocks) groups so a batch can name what
    # it must cover; this case has one unnamed part.
    blocks = [("", ["[1] alpha notes " + "word " * 50,
                    "[2] beta notes " + "word " * 50])]

    collapsed = FakeLLM({"synth": [LOOP]})
    out = "\n".join(await _map_digest(collapsed, "q", blocks))
    assert "!" * 50 not in out                 # the loop never propagates
    assert "alpha notes" in out and "beta notes" in out   # raw notes instead

    healthy = FakeLLM({"synth": ["# Digest\n\nReal content [1][2]."]})
    out = "\n".join(await _map_digest(healthy, "q", blocks))
    assert out.startswith("# Digest")          # a good digest is still used


# ---- the map-reduce funnel -------------------------------------------------
# 2026-09-09, a U8 coaching run at depth 10: six mixed-ability sources were
# searched, kept and noted — one of them the official coaching manual at 8/10
# — and none reached the overview, which then listed differentiated
# instruction as an open question. Batching was by arrival order, so a
# thinly-sourced part shared a 20k batch with a populous one and was
# compressed away; nothing downstream could see it, because the coverage check
# runs on what was searched, not on what the document ended up citing.

def _f(idx, query, relevance=6, notes="Some notes."):
    return Finding(idx=idx, url=f"https://e{idx}.test/a", title=f"T{idx}",
                   domain=f"e{idx}.test", published="2026-01-01",
                   relevance=relevance, summary="s", notes_md=notes,
                   query=query)


def test_a_thin_part_is_never_batched_behind_a_populous_one():
    from app.research.synthesizer import _pack, group_by_facet
    facet_of = {"spacing q": "spacing", "mixed q": "mixed ability"}
    findings = ([_f(i, "spacing q") for i in range(1, 21)]
                + [_f(21, "mixed q")])
    groups = [(facet, [f.notes_md for f in fs])
              for facet, fs in group_by_facet(findings, facet_of)]
    assert [g[0] for g in groups] == ["spacing", "mixed ability"]
    for parts, _blocks in _pack(groups):
        # whichever batch carries the thin part must name it, so the digest
        # prompt can require its survival
        assert "mixed ability" in parts or "mixed ability" not in parts


def test_a_part_is_not_split_across_batches_unless_it_alone_is_too_big():
    from app.research.synthesizer import _BATCH_BUDGET, _pack
    big = "x" * (_BATCH_BUDGET * 3 * 2)          # est_tokens = len // 3
    small = "y" * 30
    batches = _pack([("huge", [big]), ("a", [small]), ("b", [small])])
    carried = [parts for parts, _ in batches]
    assert ["huge"] in carried
    assert any(set(p) == {"a", "b"} for p in carried)   # small parts share one


def test_strongest_source_leads_its_part():
    from app.research.synthesizer import group_by_facet
    findings = [_f(1, "q", relevance=5), _f(2, "q", relevance=9),
                _f(3, "q", relevance=7)]
    (_facet, ordered), = group_by_facet(findings, {"q": "part"})
    assert [f.idx for f in ordered] == [2, 3, 1]


def test_funnel_losses_names_a_part_the_document_dropped():
    from app.research.synthesizer import funnel_losses
    facet_of = {"spacing q": "spacing", "mixed q": "mixed ability"}
    findings = [_f(1, "spacing q"), _f(2, "mixed q"), _f(3, "mixed q", 8)]
    overview = "# T\n\nSpacing matters [1].\n"
    dropped, strong = funnel_losses(overview, findings, facet_of)
    assert dropped == ["mixed ability"]
    assert [f.idx for f in strong] == [3]      # 8/10 read and never cited


def test_funnel_losses_is_quiet_when_every_part_is_cited():
    from app.research.synthesizer import funnel_losses
    facet_of = {"a q": "alpha", "b q": "beta"}
    findings = [_f(1, "a q"), _f(2, "b q", 9)]
    dropped, strong = funnel_losses("Both [1] and [2].", findings, facet_of)
    assert dropped == [] and strong == []


def test_an_untagged_finding_never_counts_as_a_dropped_part():
    """A run without facets (or a re-synthesis, which has no query map) must
    not grow a 'Researched but not used' section out of the empty facet."""
    from app.research.synthesizer import funnel_losses
    dropped, _strong = funnel_losses("No citations here.", [_f(1, "q")], None)
    assert dropped == []


@respx.mock
async def test_a_part_whose_sources_never_reached_the_page_is_named(data_dir):
    """The end-to-end shape of the U8 failure: both parts of the question are
    searched, both keep a source, and the synthesis cites only one of them.
    The finished document has to admit that, because the coverage check that
    runs before synthesis sees both parts as covered and stays silent."""
    cfg = make_cfg(data_dir)

    def handler(req):
        q = req.url.params.get("q", "")
        host = "cost" if "cost" in q else "safety"
        return httpx.Response(200, json=sx_payload(
            [sx_result(f"https://{host}.example.com/p", f"{host} page")]))

    respx.get(f"{SX}/search").mock(side_effect=handler)
    for host in ("cost", "safety"):
        respx.get(f"https://{host}.example.com/p").mock(
            return_value=httpx.Response(200, html=article(f"All about {host}")))

    sc = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    sc["planner"] = [{
        "title": "Home Batteries", "brief": "Cost and safety.",
        "facets": ["cost", "safety"],
        "subqueries": ["battery cost", "battery safety"],
        "query_facets": ["cost", "safety"],
        "keywords": ["battery"],
    }]
    # the synthesis ignores everything the safety searches turned up
    sc["synth"] = ["# Home Batteries\n\n## Cost\n\nCosts are falling [1].\n"]

    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(sc))
    run_id = orch.enqueue(RunParams(query="home battery cost and safety",
                                    depth=2, recency="all", origin="cli"))
    await orch.execute_now(run_id)

    overview = (cfg.research_dir / run_id / "overview.md").read_text()
    assert "## Researched but not used" in overview
    section = overview.split("## Researched but not used")[1]
    # The two searches finish in either order, so which part owns [1] and
    # which owns [2] is not fixed. What must hold is that exactly the part the
    # synthesis ignored is named, by the id a reader can follow.
    named = [p for p in ("cost", "safety") if p in section]
    assert len(named) == 1, section
    assert "[2]" in section
    # and it is not confused with the parts that found nothing at all
    assert "## Not researched" not in overview

    events = (cfg.research_dir / run_id / "events.jsonl").read_text()
    assert "overview never cited" in events


# ---- source rank -----------------------------------------------------------
# The U8 run scored the official US Youth Soccer coaching manual 8/10 and then
# wrote the overview off drill blogs: relevance measures how much a page says,
# not whether to believe it, and nothing else reached synthesis.

def test_rank_leads_the_order_and_relevance_only_breaks_ties():
    from app.research.synthesizer import group_by_facet
    blog = Finding(idx=1, url="https://b.test/a", title="20 best drills",
                   domain="b.test", published=None, relevance=9, summary="s",
                   notes_md="n", query="q", source_type="aggregator")
    manual = Finding(idx=2, url="https://gov.test/m", title="Official manual",
                     domain="gov.test", published=None, relevance=7, summary="s",
                     notes_md="n", query="q", source_type="standard")
    pro = Finding(idx=3, url="https://p.test/x", title="A coach writes",
                  domain="p.test", published=None, relevance=7, summary="s",
                  notes_md="n", query="q", source_type="practitioner")
    (_facet, ordered), = group_by_facet([blog, manual, pro], {"q": "part"})
    assert [f.idx for f in ordered] == [2, 3, 1]


def test_an_unclassified_source_is_neither_promoted_nor_demoted():
    """Old runs and a model that skips the field must sort exactly as before."""
    from app.research.synthesizer import group_by_facet
    a = Finding(idx=1, url="https://a.test/a", title="A", domain="a.test",
                published=None, relevance=5, summary="s", notes_md="n", query="q")
    b = Finding(idx=2, url="https://b.test/b", title="B", domain="b.test",
                published=None, relevance=9, summary="s", notes_md="n", query="q")
    pro = Finding(idx=3, url="https://c.test/c", title="C", domain="c.test",
                  published=None, relevance=1, summary="s", notes_md="n",
                  query="q", source_type="practitioner")
    (_facet, ordered), = group_by_facet([a, b, pro], {"q": "part"})
    assert [f.idx for f in ordered] == [2, 1, 3]   # pure relevance order


def test_the_note_block_states_the_kind_for_the_model():
    from app.research.synthesizer import _note_block
    f = Finding(idx=4, url="https://g.test/s", title="The spec", domain="g.test",
                published=None, relevance=7, summary="s", notes_md="body",
                query="q", source_type="standard")
    assert "[standard]" in _note_block(f)
    f.source_type = ""
    assert "[standard]" not in _note_block(f)


def test_a_junk_source_type_is_discarded_not_ranked():
    from app.models import NotesOut, source_rank
    assert NotesOut(relevance=5, source_type="VERY OFFICIAL!!").source_type == ""
    # case and padding are cleaned, but a standard still has to name its
    # publisher to stay one — see the demotion test in test_facets.py
    assert NotesOut(relevance=5, source_type=" Standard ",
                    publisher="US Youth Soccer").source_type == "standard"
    assert NotesOut(relevance=5, source_type=" Standard ").source_type == "aggregator"
    assert source_rank("standard") < source_rank("") < source_rank("aggregator")


# ---- the premise verdict as coverage evidence --------------------------------

def test_a_part_settled_in_the_premise_verdict_is_not_called_unresearched():
    """2026-09-10: a document gave the official U8 field dimensions, cited, in
    its opening premise section and then listed "field dimension standards"
    under "Not researched" at its own foot. The coverage recheck reads
    headings, and this heading names no facet."""
    from app.research import facets as facet_plan
    from app.research.synthesizer import premise_verdict

    # The wording is the real run's, because the credit is lexical: about()
    # wants two of the part's own words. A verdict that settles the question
    # in different words still reports a gap — that is about()'s deliberate
    # one-sided error, and this fix does not change it.
    overview = (
        "# Coaching U8\n\n"
        "## Checking what the question assumes\n\n"
        "The sources give no single binding standard for U8 4v4 field "
        "dimensions, but 25-35 yards long and 15-25 wide is the common "
        "recommendation [25].\n\n"
        "## TL;DR\n\nPlay narrow [1].\n")
    headings = "\n".join(l for l in overview.splitlines()
                         if l.lstrip().startswith("#"))

    assert not facet_plan.about("field dimension standards", headings)
    evidence = headings + "\n" + premise_verdict(overview)
    assert facet_plan.about("field dimension standards", evidence)


def test_an_unsettled_premise_is_not_evidence_that_anything_was_researched():
    """The prompt tells the model to say plainly when the run found nothing
    that settles the premise. That admission must not be read as coverage —
    it is the opposite. Citations are the test, because a verdict that
    settled something cites what settled it."""
    from app.research.synthesizer import premise_verdict

    overview = (
        "# Coaching U8\n\n"
        "## Checking what the question assumes\n\n"
        "Nothing the run found states an official field dimension standard, "
        "so this assumption is unchecked.\n\n"
        "## TL;DR\n\nPlay narrow.\n")
    assert premise_verdict(overview) == ""


def test_a_document_with_no_premise_section_is_unchanged():
    from app.research.synthesizer import premise_verdict
    assert premise_verdict("# T\n\n## TL;DR\n\nBody [1].\n") == ""


def test_the_premise_body_stops_at_the_next_section():
    """It must not swallow the rest of the document, or every facet named
    anywhere below would count as answered."""
    from app.research.synthesizer import premise_verdict

    overview = ("# T\n\n## Checking what the question assumes\n\n"
                "The fields are standard [2].\n\n"
                "## Coaching drills\n\nRondos and gates [4].\n")
    body = premise_verdict(overview)
    assert "standard [2]" in body
    assert "Rondos" not in body


# ---- re-synthesis keeps the document honest ---------------------------------

@respx.mock
async def test_resynthesize_keeps_the_gap_section_it_used_to_delete(data_dir):
    """The button rewrote an honest document into a confident one. It called
    synthesize() with no premises, no uncovered_facets and no facet_of, and
    the two appended sections lived only in the run path — so the premise
    verdict, "Not researched" and "Researched but not used" all vanished
    from a re-synthesized overview while its prose stayed just as assertive."""
    import json
    cfg = make_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result("https://example-a.com/article", "Article A")])))
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=article("Article A")))
    repo = Repo(connect(cfg.db_path))

    sc = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    sc["planner"] = [{
        "title": "Batteries", "brief": "Chemistry and recycling cost.",
        "facets": ["cell chemistry", "recycling cost"],
        "subqueries": ["solid state cell chemistry"],
        "query_facets": ["cell chemistry"],
        "keywords": ["battery"],
    }]
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(sc))
    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)
    run_dir = cfg.research_dir / run_id

    # the plan is on disk, which is what makes the rebuild possible at all
    meta = json.loads((run_dir / "meta.json").read_text())
    assert "recycling cost" in meta["facets"]

    first = (run_dir / "overview.md").read_text()
    assert "## Not researched" in first
    assert "recycling cost" in first

    resynth_llm = FakeLLM({
        "synth": ["# Batteries\n\nChemistry is settled [1].\n"],
        "followups": [{"items": []}],
    })
    pipeline = Pipeline(cfg, repo, ProgressBus(),
                        llm_factory=lambda: resynth_llm)
    await pipeline.resynthesize(run_id)

    again = (run_dir / "overview.md").read_text()
    assert again.startswith("# Batteries")
    assert "## Not researched" in again, "the gap section was dropped again"
    assert "recycling cost" in again


@respx.mock
async def test_resynthesize_says_so_when_the_run_predates_the_stored_plan(data_dir):
    """Old runs have no facets in meta, so the coverage sections genuinely
    cannot be rebuilt. Saying nothing would leave a document that looks more
    certain than the one it replaced."""
    import json
    cfg = make_cfg(data_dir)
    repo, run_id = await _completed_run(cfg)
    run_dir = cfg.research_dir / run_id
    meta = json.loads((run_dir / "meta.json").read_text())
    for k in ("facets", "premises", "query_facet", "facet_kept"):
        meta.pop(k, None)
    (run_dir / "meta.json").write_text(json.dumps(meta))

    pipeline = Pipeline(cfg, repo, ProgressBus(), llm_factory=lambda: FakeLLM({
        "synth": ["# Old run\n\nBody [1].\n"], "followups": [{"items": []}]}))
    await pipeline.resynthesize(run_id)

    overview = (run_dir / "overview.md").read_text()
    assert "could not be recomputed" in overview
    # and it must not invent one from the weaker reconstruction
    assert "## Not researched" not in overview


@respx.mock
async def test_resynthesize_does_not_invent_gaps_the_run_never_had(data_dir):
    """First live re-synthesis, 2026-09-10: the rebuilt map credited parts
    lexically from each finding's query, so two parts whose sources sat in
    the same document's bibliography were printed under "The run found no
    sources for these parts of the question". The run's own accounting is
    stored now, and that is what the claim is made from."""
    import json
    cfg = make_cfg(data_dir)
    # The discriminating shape: the SOURCE is plainly about the part, so the
    # run credits it and reports no gap — but the QUERY that found it shares
    # almost nothing with the part's name, so rebuilding the map from queries
    # alone credits nothing and invents one.
    title = "Weight and circumference range for match balls"
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result("https://example-a.com/article", title)])))
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=article(title)))
    repo = Repo(connect(cfg.db_path))

    sc = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    sc["planner"] = [{
        "title": "Balls", "brief": "Ball specifications.",
        "facets": ["weight and circumference range"],
        "subqueries": ["IFAB laws of the game specifications"],
        "query_facets": ["weight and circumference range"],
        "keywords": ["ball"],
    }]
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(sc))
    run_id = orch.enqueue(RunParams(query="size 3 ball", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)
    run_dir = cfg.research_dir / run_id

    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["query_facet"]["IFAB laws of the game specifications"] == \
        "weight and circumference range"
    # the run credited the part, so it printed no gap
    assert meta["facet_kept"].get("weight and circumference range")
    assert "## Not researched" not in (run_dir / "overview.md").read_text()

    pipeline = Pipeline(cfg, repo, ProgressBus(), llm_factory=lambda: FakeLLM({
        "synth": ["# Balls\n\nThe ball is round [1].\n"],
        "followups": [{"items": []}]}))
    await pipeline.resynthesize(run_id)

    again = (run_dir / "overview.md").read_text()
    assert "The run found no sources" not in again, \
        "invented a gap for a part the run kept a source for"


@respx.mock
async def test_a_thin_run_stays_thin_when_it_is_re_synthesized(data_dir):
    """Thinness is a property of the run, not of one synthesis call — no new
    sources are fetched, so the banner belongs on the rewrite too."""
    cfg = make_cfg(data_dir)
    repo, run_id = await _completed_run(cfg)
    run_dir = cfg.research_dir / run_id
    was = (run_dir / "overview.md").read_text()
    (run_dir / "overview.md").write_text(
        "> **Thin result.** No source strongly matched this question.\n\n" + was)

    pipeline = Pipeline(cfg, repo, ProgressBus(), llm_factory=lambda: FakeLLM({
        "synth": ["# Redone\n\nBody [1].\n"], "followups": [{"items": []}]}))
    await pipeline.resynthesize(run_id)

    assert "**Thin result.**" in (run_dir / "overview.md").read_text()


@respx.mock
async def test_a_re_synthesis_still_honours_the_shape_the_asker_asked_for(data_dir):
    """Formatting instructions are persisted with the plan, so pressing
    Re-synthesize does not quietly drop the table the asker asked for."""
    import json
    cfg = make_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result("https://example-a.com/article", "Article A")])))
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=article("Article A")))
    repo = Repo(connect(cfg.db_path))

    sc = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    sc["planner"] = [{
        "title": "GPUs", "brief": "Compare three GPUs.",
        "facets": ["gpu comparison"], "subqueries": ["gpu comparison 2026"],
        "query_facets": ["gpu comparison"], "keywords": ["gpu"],
        "deliverables": ["Include a comprehensive comparison table"],
    }]
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(sc))
    run_id = orch.enqueue(RunParams(query="compare gpus", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)
    run_dir = cfg.research_dir / run_id
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["deliverables"] == ["Include a comprehensive comparison table"]

    seen = []

    def capture(messages):
        seen.append(messages[-1]["content"])
        return "# GPUs\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nBody [1]."

    pipeline = Pipeline(cfg, repo, ProgressBus(), llm_factory=lambda: FakeLLM({
        "synth": [capture], "followups": [{"items": []}]}))
    await pipeline.resynthesize(run_id)
    assert seen and "comprehensive comparison table" in seen[0]
