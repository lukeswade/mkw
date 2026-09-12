"""Candidate triage, full-page notes input, and index-page link chasing."""
from __future__ import annotations

import json
import httpx
import respx

from app.db import Repo, connect
from app.llm.json_utils import LLMJsonError
from app.models import RunParams
from app.research.notes import take_notes
from app.research.orchestrator import Orchestrator
from app.research.progress import ProgressBus
from tests.fake_llm import FakeLLM
from tests.test_pipeline_e2e import SX, article, make_cfg, script, sx_payload, sx_result


def _five_candidates():
    return [sx_result(f"https://example-{c}.com/article", f"Article {c.upper()}")
            for c in "abcde"]


def _run(cfg, llm_script):
    repo = Repo(connect(cfg.db_path))
    llm = FakeLLM(llm_script)
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: llm)
    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=1,
                                    recency="all", origin="cli"))
    return repo, llm, orch, run_id


# ---- triage --------------------------------------------------------------------

@respx.mock
async def test_triage_drops_candidates_before_any_fetch(data_dir):
    cfg = make_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(
        200, json=sx_payload(_five_candidates())))
    # routes exist ONLY for the survivors — fetching a dropped one explodes
    for c in ("a", "c", "e"):
        respx.get(f"https://example-{c}.com/article").mock(
            return_value=httpx.Response(200, html=article(f"Article {c.upper()}")))

    s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    s["triage"] = [{"drop": [1, 3]}]      # within the half-a-round drop cap
    repo, llm, orch, run_id = _run(make_cfg(data_dir), s)
    await orch.execute_now(run_id)

    cfg = orch.cfg_loader()
    findings = repo.findings_for_run(run_id)
    assert {f["domain"] for f in findings} == {"example-a.com", "example-c.com",
                                               "example-e.com"}
    events = [json.loads(l) for l in (cfg.research_dir / run_id / "events.jsonl").read_text().splitlines() if l.strip()]
    # one event per triage round carries the drops; no line per dropped page
    tri = [e for e in events if e["type"] == "triage"]
    assert len(tri) == 1 and tri[0]["dropped"] == 2 and tri[0]["considered"] == 5
    assert tri[0]["to_read"] == 3 and len(tri[0]["dropped_urls"]) == 2
    assert not any(e["type"] == "source_skipped" and str(e.get("reason", "")).startswith("dropped") for e in events)
    assert llm.calls["notes"] == 3


@respx.mock
async def test_triage_failure_degrades_to_keeping_everything(data_dir):
    cfg = make_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(
        200, json=sx_payload(_five_candidates())))
    for c in "abcde":
        respx.get(f"https://example-{c}.com/article").mock(
            return_value=httpx.Response(200, html=article(f"Article {c.upper()}")))

    s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    s["triage"] = [LLMJsonError("model emitted garbage")]
    repo, llm, orch, run_id = _run(cfg, s)
    await orch.execute_now(run_id)
    assert len(repo.findings_for_run(run_id)) == 5


@respx.mock
async def test_condemning_everything_keeps_the_floor(data_dir):
    """A round Bing Videos filled with 60 junk videos was condemned whole by
    triage — and the verdict was discarded as 'broken', so all 60 were
    fetched. Condemning everything is handled like any over-cull: the
    best-ranked floor stays, the rest go."""
    cfg = make_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(
        200, json=sx_payload(_five_candidates())))
    for c in "abcde":
        respx.get(f"https://example-{c}.com/article").mock(
            return_value=httpx.Response(200, html=article(f"Article {c.upper()}")))

    s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    s["triage"] = [{"drop": [0, 1, 2, 3, 4]}]   # drop-all
    repo, llm, orch, run_id = _run(cfg, s)
    await orch.execute_now(run_id)
    findings = repo.findings_for_run(run_id)
    assert len(findings) == 3                   # triage_floor(5): the best-ranked three
    assert {f["domain"] for f in findings} == {"example-a.com", "example-b.com", "example-c.com"}


# ---- notes input: whole page when it fits ---------------------------------------

def _notes_payload():
    return {"relevance": 8, "summary": "s", "notes_md": "n",
            "key_facts": [], "published_date": None}


async def test_notes_reads_the_whole_page_when_it_fits():
    captured: list[str] = []

    def capture(messages):
        captured.append(messages[-1]["content"])
        return _notes_payload()

    llm = FakeLLM({"notes": [capture]})
    text = ("Torque specification discussion up front. "
            + "Unremarkable filler sentence. " * 200
            + "Closing detail: the flimflam value is 15.")
    await take_notes(llm, brief="b", recency_desc="any", today="t",
                     url="u", title="t", detected_date=None, text=text,
                     keywords=["torque"])
    # the old keyword-excerpt path would have cut everything far from
    # "torque"; the whole document must reach the model now
    assert "flimflam value is 15" in captured[0]


async def test_notes_excerpts_only_oversized_documents():
    captured: list[str] = []

    def capture(messages):
        captured.append(messages[-1]["content"])
        return _notes_payload()

    llm = FakeLLM({"notes": [capture]})
    text = ("padding sentence with nothing in it. " * 1500
            + " The torque spec is 15 ft-lb on the 2UZ. "
            + "more padding after the fact. " * 500)
    assert len(text) > 40_000
    await take_notes(llm, brief="b", recency_desc="any", today="t",
                     url="u", title="t", detected_date=None, text=text,
                     keywords=["torque"])
    prompt = captured[0]
    assert "torque spec is 15 ft-lb" in prompt      # keyword window survives
    assert len(prompt) < len(text)                   # but not the whole doc


# ---- index/stub pages chase their own children -----------------------------------

@respx.mock
async def test_thin_index_page_chases_its_child_links(data_dir):
    cfg = make_cfg(data_dir)
    cfg.reference_chasing = True   # index descent rides the chasing switch
    index_html = (
        "<html><head><title>Spark Plug</title></head><body><main><article>"
        "<h1>Spark Plug</h1>"
        "<p>Service and repair section for the spark plug system. Choose a "
        "subsection below to read the full procedure and specifications for "
        "this engine. Sections are grouped by inspection type and task.</p>"
        '<a href="/gx470/spark-plug/replacement-procedure">Spark plug '
        "replacement procedure</a> "
        '<a href="/gx470/spark-plug/specifications">Spark plug torque '
        "specifications</a>"
        "</article></main></body></html>")
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result("https://manuals.example.com/gx470/spark-plug/", "Spark Plug")])))
    respx.get("https://manuals.example.com/gx470/spark-plug/").mock(
        return_value=httpx.Response(200, html=index_html))
    respx.get("https://manuals.example.com/gx470/spark-plug/replacement-procedure").mock(
        return_value=httpx.Response(200, html=article("Replacement Procedure")))
    respx.get("https://manuals.example.com/gx470/spark-plug/specifications").mock(
        return_value=httpx.Response(200, html=article("Torque Specifications")))

    s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    s["planner"] = [{"title": "T", "brief": "GX470 spark plug replacement.",
                     "subqueries": ["gx470 spark plug"]}]
    s["notes"] = [
        {"relevance": 2, "summary": "Just a section index.", "notes_md": "n",
         "key_facts": [], "published_date": None},          # the stub
        {"relevance": 8, "summary": "The procedure.", "notes_md": "n",
         "key_facts": [], "published_date": None},          # its children
    ]
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(s))
    run_id = orch.enqueue(RunParams(query="gx470 spark plug replacement",
                                    depth=1, recency="all", origin="cli"))
    await orch.execute_now(run_id)

    findings = repo.findings_for_run(run_id)
    assert len(findings) == 2                     # both children, not the stub
    assert all("manuals.example.com" == f["domain"] for f in findings)
    run_dir = cfg.research_dir / run_id
    finding_md = "".join(p.read_text()
                         for p in (run_dir / "findings").glob("*.md"))
    assert "linked from index page on manuals.example.com" in finding_md
    events = run_dir.joinpath("events.jsonl").read_text()
    assert "relevance 2/10" in events             # the stub was still scored


# ---- triage must not discard platform siblings or authority sites -----------------

def test_triage_prompt_protects_sibling_models():
    """A "different model" instruction made triage drop charm.li factory
    manuals for the Tundra/Sequoia — same 2UZ-FE engine as the GX470 in the
    question, and the best sources in the run."""
    from app.llm import prompts
    assert "Do NOT drop a candidate merely because its title names a " \
           "different product" in prompts.TRIAGE.replace("\n", " ")
    assert "same engine, chipset, platform or codebase" in \
        prompts.TRIAGE.replace("\n", " ")


@respx.mock
async def test_authority_site_candidates_survive_triage(data_dir):
    """Curating a site as authoritative outranks a title-level guess.

    Needs more than three candidates or triage never runs at all — the
    earlier version of this test had two, so it exercised nothing and passed
    only because one of its URLs was deliberately left unmocked and the
    failing fetch dropped that source for it.
    """
    cfg = make_cfg(data_dir)
    cfg.authority_sites = "charm.li — factory service manuals for cars"
    fsm = ("https://charm.li/Toyota/2006/Tundra%20V8-4.7L%20(2UZ-FE)/"
           "Spark%20Plug/")
    junk = [f"https://junk{i}.example.com/ad" for i in range(4)]
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result(fsm, "2UZ-FE spark plugs — factory manual")]
        + [sx_result(u, "Buy cheap online") for u in junk])))
    respx.get(fsm).mock(return_value=httpx.Response(
        200, html=article("Tundra 2UZ-FE Spark Plug")))
    for i, u in enumerate(junk):
        respx.get(u).mock(return_value=httpx.Response(
            200, html=article(f"Cheap Plugs Ad {i}")))

    s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    # index 0 is the FSM page: it shares the most words with the query, and
    # pick() orders by that. Triage condemns it along with one junk page.
    s["triage"] = [{"drop": [0, 1]}]
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(s))
    run_id = orch.enqueue(RunParams(query="2UZ-FE spark plugs factory manual",
                                    depth=1, recency="all", origin="cli"))
    await orch.execute_now(run_id)

    domains = {f["domain"] for f in repo.findings_for_run(run_id)}
    assert "charm.li" in domains          # condemned, but curated as authority
    assert len(domains) == 4              # the other condemned page stayed out
    events = (cfg.research_dir / run_id / "events.jsonl").read_text()
    assert '"type": "triage"' in events  # triage genuinely ran


def test_authority_domains_parses_the_settings_blob(data_dir):
    from app.research.pipeline import Pipeline
    from app.research.progress import ProgressBus as PB
    cfg = make_cfg(data_dir)
    cfg.authority_sites = ("charm.li — factory service manuals\n"
                           "https://www.nist.gov/ — standards\n"
                           "not-a-domain line\n")
    p = Pipeline(cfg, Repo(connect(cfg.db_path)), PB())
    assert p._authority_domains() == frozenset({"charm.li", "nist.gov"})


# ---- triage may veto at most half a round ---------------------------------------

@respx.mock
async def test_triage_cannot_cull_more_than_half_a_round(data_dir):
    """A live depth-10 run had triage drop 21, 26 and 23 of 30 candidates in
    consecutive rounds, discarding pages that answered the round's own queries
    by name. Three prompt revisions failed to restrain it, so the ceiling is
    structural: the best-ranked of the condemned are reprieved."""
    cfg = make_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(
        200, json=sx_payload(_five_candidates())))
    for c in "abcde":
        respx.get(f"https://example-{c}.com/article").mock(
            return_value=httpx.Response(200, html=article(f"Article {c.upper()}")))

    s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    s["triage"] = [{"drop": [0, 1, 2, 3]}]     # 4 of 5 — over the cap of 2
    repo, llm, orch, run_id = _run(cfg, s)
    await orch.execute_now(run_id)

    findings = repo.findings_for_run(run_id)
    assert len(findings) == 3                  # 5 candidates, at most 2 dropped
    # the reprieve goes to the best-ranked condemned candidates (lowest index),
    # never to whichever ones happened to sort last
    domains = {f["domain"] for f in findings}
    assert "example-a.com" in domains and "example-b.com" in domains


# ---- gap analysis must not end a run on an empty query list ---------------------

async def test_gap_retries_when_it_proposes_nothing_but_is_not_saturated():
    """A live depth-10 run ended at round 3 of 10 because gap analysis
    returned saturated=false with next_queries=[] — every query it proposed
    was a reword of one already searched, and the repeat filter emptied the
    list. The pipeline stops the run outright on an empty list."""
    from app.research.gap import analyze
    llm = FakeLLM({"gap": [
        # round's first answer: gaps remain, but every query is a repeat
        {"state_md": "s", "saturated": False,
         "next_queries": ["gx470 spark plug", "GX470 Spark Plug"]},
        # the stern re-ask finds a genuinely new angle
        {"state_md": "s", "saturated": False,
         "next_queries": ["2UZ-FE ignition coil removal torque"]},
    ]})
    out = await analyze(llm, query="q", brief="b", recency_desc="any",
                        round_no=3, depth=10, breadth=4, state_md="s",
                        new_findings=[], searched=["gx470 spark plug"])
    assert out.next_queries == ["2UZ-FE ignition coil removal torque"]
    assert llm.calls["gap"] == 2


async def test_gap_accepts_an_honest_saturation_verdict():
    """The retry must not badger a model that legitimately says it is done."""
    from app.research.gap import analyze
    llm = FakeLLM({"gap": [{"state_md": "s", "saturated": True,
                            "next_queries": []}]})
    out = await analyze(llm, query="q", brief="b", recency_desc="any",
                        round_no=3, depth=10, breadth=4, state_md="s",
                        new_findings=[], searched=["gx470 spark plug"])
    assert out.saturated is True and out.next_queries == []
    assert llm.calls["gap"] == 1        # no pointless second call


@respx.mock
async def test_use_prior_false_skips_the_prior_knowledge_step(data_dir):
    """'Diagnose this fresh': a run can be told to ignore every earlier run,
    so a previous wrong conclusion cannot anchor the new one."""
    cfg = make_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(
        200, json=sx_payload([sx_result("https://example-a.com/article", "A")])))
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=article("Article A")))

    class SpyRag:
        def __init__(self):
            self.consulted = 0
        async def prior_knowledge(self, query, exclude_run=None):
            self.consulted += 1
            return "AN EARLIER RUN CONCLUDED THE PANEL IS DEAD", []
        async def index_run(self, *a, **k):
            return None
        async def link_related(self, *a, **k):
            return None

    async def run_with(use_prior: bool):
        rag = SpyRag()
        seen: list[str] = []

        def capture_planner(messages):
            seen.append(messages[-1]["content"])
            return {"title": "T", "brief": "Diagnose the TV.",
                    "subqueries": ["tcl roku tv no power"], "keywords": []}

        s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
        s["planner"] = [capture_planner]
        repo = Repo(connect(cfg.db_path))
        orch = Orchestrator(lambda: cfg, repo, ProgressBus(), rag=rag,
                            llm_factory=lambda: FakeLLM(s))
        run_id = orch.enqueue(RunParams(query="tcl roku tv will not turn on",
                                        depth=1, recency="all", origin="cli",
                                        use_prior=use_prior))
        assert bool(repo.get_run(run_id)["use_prior"]) is use_prior
        await orch.execute_now(run_id)
        return rag.consulted, "".join(seen)

    consulted, prompt = await run_with(True)
    assert consulted == 1
    assert "PANEL IS DEAD" in prompt        # earlier findings reach the planner

    consulted, prompt = await run_with(False)
    assert consulted == 0                   # never even asked
    assert "PANEL IS DEAD" not in prompt


def test_unchecked_use_prior_box_actually_turns_it_off(data_dir, monkeypatch):
    """An unchecked HTML checkbox is omitted from the POST body entirely. A
    route default of "on" therefore makes the box impossible to clear — the
    first version of this shipped that way and silently ran with memory on."""
    from fastapi.testclient import TestClient
    from app.config import load_settings
    from app.web.server import create_app
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    app = create_app(enable_worker=False, enable_bot=False)
    cfg = load_settings(str(data_dir))
    base = {"query": "tcl roku tv will not turn on", "depth": "1",
            "recency": "all", "categories": "general"}
    with TestClient(app) as client:
        on_id = client.post("/runs", data={**base, "use_prior": "on"},
                            follow_redirects=False
                            ).headers["location"].rsplit("/", 1)[-1]
        # box unticked: the field simply is not sent at all
        off_id = client.post("/runs", data=base, follow_redirects=False
                             ).headers["location"].rsplit("/", 1)[-1]
    repo = Repo(connect(cfg.db_path))
    assert bool(repo.get_run(on_id)["use_prior"]) is True
    assert bool(repo.get_run(off_id)["use_prior"]) is False


def test_run_page_shows_what_it_searched(data_dir, monkeypatch):
    """A finished run gave no indication of which categories it used, nor
    whether it ran with or without earlier research."""
    from fastapi.testclient import TestClient
    from app.config import load_settings
    from app.web.server import create_app
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    app = create_app(enable_worker=False, enable_bot=False)
    cfg = load_settings(str(data_dir))
    from app.research.storage import RunStore
    repo = Repo(connect(cfg.db_path))
    store = RunStore.create(cfg.research_dir, "categorised run")
    repo.create_run(run_id=store.run_id, query="q", depth=3, recency="all",
                    dir=store.run_id, origin="web", status="completed",
                    categories="general,videos", use_prior=False)
    # a queued/running run renders a different template than a finished one
    running = RunStore.create(cfg.research_dir, "in-flight run")
    repo.create_run(run_id=running.run_id, query="q", depth=3, recency="all",
                    dir=running.run_id, origin="web", status="running",
                    categories="general,videos", use_prior=False)
    with TestClient(app) as client:
        done = client.get(f"/runs/{store.run_id}").text
        live = client.get(f"/runs/{running.run_id}").text
    for body in (done, live):
        assert "general · videos" in body
        assert "no prior research" in body


def _cand(url, title="", snippet="", engine="bing"):
    from app.research.searcher import SearchResult
    return SearchResult(url=url, title=title, snippet=snippet, engine=engine,
                        published=None, score=1.0, via_query="q")


def test_triage_cannot_condemn_a_domain_that_already_produced_a_source():
    """15 pages across two runs — including the most authoritative guide for
    one question — were dropped on a title while their domain already had a
    kept source. Generic hosts are exempt: reddit proves nothing about reddit."""
    from app.research.pipeline import spare_productive
    cands = [_cand("https://kindlemodding.org/jailbreaking/WinterBreak/"),
             _cand("https://www.reddit.com/r/kindle/comments/abc/x/"),
             _cand("https://www.capitalone.com/credit-cards/cabelas/")]
    spared = spare_productive({0, 1, 2}, cands, {"kindlemodding.org", "reddit.com"})
    assert spared == {0}


def test_engine_filler_that_shares_no_word_with_the_round_is_not_fetched():
    from app.research.pipeline import filter_by_vocabulary
    from app.research.dedupe import vocabulary
    vocab = vocabulary("fly rod handle rotating reel seat repair",
                       "JB Weld vs epoxy for fly rod handle repair")
    pool = [_cand("https://www.rodbuilding.org/read.php?2,1", "Loose reel seat", "epoxy the seat"),
            _cand("https://www.google.com/travel/flights", "Find Cheap Flights", "Book a flight"),
            _cand("https://finance.yahoo.com/quote/FIX/", "FIX stock price", "Comfort Systems USA"),
            _cand("https://charm.li/x", "1998 Sierra", "wiring", engine="google cse")]
    kept, dropped = filter_by_vocabulary(pool, vocab, exempt=frozenset({"charm.li"}))
    assert [c.url for c in dropped] == ["https://www.google.com/travel/flights",
                                        "https://finance.yahoo.com/quote/FIX/"]
    assert len(kept) == 2                                       # the match and the exempt authority site


def test_filler_is_dropped_only_when_real_matches_can_fill_the_round():
    """It must never take a slot from a real match, and it is fetched only
    when there is nothing better to fetch — a starved round keeps it."""
    from app.research.pipeline import filter_by_vocabulary
    from app.research.dedupe import vocabulary
    vocab = vocabulary("fly rod reel seat repair")
    real = [_cand(f"https://forum{i}.com/t", "reel seat repair", "epoxy") for i in range(3)]
    junk = [_cand("https://www.google.com/travel/flights", "Cheap Flights", "book")]
    kept, dropped = filter_by_vocabulary(real + junk, vocab, limit=3)      # 3 matches fill a round of 3
    assert dropped == junk
    kept, dropped = filter_by_vocabulary(real + junk, vocab, limit=10)     # starved: keep, rank later
    assert dropped == [] and len(kept) == 4


def test_the_adaptive_cap_starts_at_two_and_is_earned():
    from app.research.pipeline import adaptive_cap
    assert adaptive_cap(2, 0) == 2          # round one, or a source that kept nothing
    assert adaptive_cap(2, 2) == 4          # kept two of two -> four next round
    assert adaptive_cap(2, 9) == 6          # ceiling


def test_round_one_credit_comes_only_from_related_runs_and_skips_platforms():
    """Yield is topic-bound, so credit is computed over runs judged related to
    this question; 3+ kept earns two slots, exactly 2 earns one, and generic
    platforms earn nothing — a kept reddit thread says nothing about reddit."""
    from app.research.pipeline import seed_from_related, adaptive_cap
    seed = seed_from_related({"classicflyrodforum.com": 5, "www.paflyfish.com": 2,
                              "reddit.com": 8, "youtube.com": 6, "one-off.net": 1})
    assert seed == {"classicflyrodforum.com": 2, "paflyfish.com": 1}
    assert adaptive_cap(2, 0 + seed["classicflyrodforum.com"]) == 4      # starts round one at four


async def test_the_note_taker_is_told_when_it_is_reading_a_demonstration_video():
    from app.models import NotesOut
    from app.research.notes import take_notes

    class Capture:
        def __init__(self): self.prompt = ""
        async def chat_json(self, kind, messages, schema, **kw):
            self.prompt = messages[0]["content"]
            return NotesOut(relevance=7, summary="s", notes_md="n", key_facts=[])

    for kind, expected in (("video", True), ("", False)):
        llm = Capture()
        await take_notes(llm, brief="b", recency_desc="all time", today="2026-09-03", url="u",
                         title="t", detected_date=None, text="some text", source_kind=kind)
        assert ("SOURCE TYPE: video" in llm.prompt) is expected, kind


def test_at_most_one_site_query_per_round_unless_the_site_is_an_authority():
    from app.research.pipeline import limit_site_queries
    out = limit_site_queries(["site:deskthority.net DIY trackball ZMK build",
                              "site:geekhack.org custom trackball nRF52840",
                              "site:rodbuilding.org reel seat repair",
                              "DIY trackball ZMK firmware guide"],
                             authority=frozenset({"rodbuilding.org"}))
    assert out == [("site:deskthority.net DIY trackball ZMK build", "site:deskthority.net DIY trackball ZMK build"),
                   ("custom trackball nRF52840", "site:geekhack.org custom trackball nRF52840"),   # opened to the web
                   ("site:rodbuilding.org reel seat repair", "site:rodbuilding.org reel seat repair"),  # authority keeps its scope
                   ("DIY trackball ZMK firmware guide", "DIY trackball ZMK firmware guide")]


def test_triage_keeps_a_floor_not_half_the_round():
    """Pages the half-round cap forced back in were kept 6% of the time (5 of
    85) against 55% for the rest. The floor protects a small round from one
    bad verdict; a junk-heavy large round can lose most of itself."""
    from app.research.pipeline import triage_floor
    assert triage_floor(5) == 3          # small round: at most 2 dropped (unchanged behaviour)
    assert triage_floor(36) == 4
    assert triage_floor(44) == 5         # was 22 kept; now 5
    assert triage_floor(60) == 6


def test_quotes_the_source_does_not_contain_are_removed_and_counted():
    from app.models import Fact, NotesOut
    from app.research.notes import verify_quotes
    text = ("The reel seat is bonded to the blank with a slow-cure epoxy; when that bond fails the "
            "seat spins freely. Inject fresh epoxy through a small hole and rotate the seat to spread it — "
            "“Rod Bond” or Flex Coat both work well here.")
    # raw dicts: the model's own validator normalises facts on input
    notes = NotesOut(relevance=7, summary="s", notes_md="n", key_facts=[
        {"claim": "a", "evidence_quote": "the seat spins freely"},                                   # verbatim
        {"claim": "b", "evidence_quote": '"Rod Bond" or Flex Coat both work well here'},               # curly vs straight quotes
        {"claim": "c", "evidence_quote": "Inject fresh epoxy through a small hole and rotate the seat to spread it around"},  # one extra word: tolerated
        {"claim": "d", "evidence_quote": "Use JB Weld and heat the seat with a torch to remove it"},  # not in the source
        {"claim": "e", "evidence_quote": "epoxy"},                                                    # too short to judge
    ])
    assert len(notes.key_facts) == 5
    assert verify_quotes(notes, text) == (0, 1)
    assert [f.evidence_quote is not None for f in notes.key_facts] == [True, True, True, False, True]


def test_a_paraphrased_quote_is_repaired_to_the_sources_own_sentence():
    from app.models import NotesOut
    from app.research.notes import verify_quotes
    text = ("Most builders inject a slow-cure rod bond epoxy through a small hole drilled in the reel seat. "
            "Rotating the seat while the epoxy is wet spreads it around the arbor. Let it cure overnight before use.")
    notes = NotesOut(relevance=7, summary="s", notes_md="n", key_facts=[
        {"claim": "a", "evidence_quote": "builders inject slow-cure rod bond epoxy through a hole drilled in the reel seat"},  # near miss
        {"claim": "b", "evidence_quote": "The seat should be heated with a torch and pulled off with pliers"},               # not in the source
    ])
    assert verify_quotes(notes, text) == (1, 1)
    assert notes.key_facts[0].evidence_quote.startswith("Most builders inject a slow-cure rod bond epoxy")
    assert notes.key_facts[1].evidence_quote is None


def test_authority_sites_start_at_the_ceiling_not_uncapped():
    from app.research.pipeline import authority_domains_from, _SOURCE_CAP_MAX, adaptive_cap
    auth = authority_domains_from("charm.li — service manuals\nrodbuilding.org — rod building forum\n\nhttps://www.badcaps.net/ — board repair")
    assert auth == frozenset({"charm.li", "rodbuilding.org", "badcaps.net"})
    assert _SOURCE_CAP_MAX == 6 and adaptive_cap(2, 0) == 2      # an authority site gets max(6, earned) in pick()


@respx.mock
async def test_a_round_triage_guts_is_refilled_from_what_the_share_cap_held_back(data_dir):
    """Two engines each offer 40 results; the share cap takes a third of the
    round from each. Triage then condemns nearly all of them. The freed slots
    go to what the cap held back, through a second triage, instead of the
    round running on three candidates."""
    cfg = make_cfg(data_dir)
    def res(eng, i):
        r = sx_result(f"https://{eng}{i}.com/p", f"Solid state battery report {eng} {i}")
        r["engine"] = eng
        return r
    pool = [res("alpha", i) for i in range(40)] + [res("beta", i) for i in range(40)]
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(pool)))
    # distinct bodies: identical pages fold into one under the duplicate-content check
    respx.get(url__regex=r"https://(alpha|beta)\d+\.com/p").mock(
        side_effect=lambda req: httpx.Response(200, html=article(
            f"Solid state battery report from {req.url.host}: cathode data {req.url.host}")))

    s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    s["triage"] = [{"drop": list(range(1, 60))},   # first look: condemn all but the first
                   {"drop": []}]                    # the refill: all worth reading
    repo, llm, orch, run_id = _run(cfg, s)
    await orch.execute_now(run_id)

    assert llm.calls["triage"] == 2
    events = (cfg.research_dir / run_id / "events.jsonl").read_text()
    assert "round refilled" in events
    from app.research.pipeline import triage_floor
    survivors_alone = triage_floor(16)             # what the round would have read without the refill
    assert len(repo.findings_for_run(run_id)) > survivors_alone


def test_triage_spares_roundups_for_the_note_taker_to_judge():
    """Measured 2026-09-11: 9 of 33 random triage drops would have been kept,
    all at >= 6, six of them roundups; a reworded prompt changed nothing."""
    from types import SimpleNamespace as NS
    from app.research.pipeline import spare_roundups, _ROUNDUP_SPARES
    c = lambda t, u="https://x.com/p": NS(title=t, url=u)
    cands = [c("13 Best Free & Open Source Knowledge Base Software (2026)"),
             c("AI Agent Memory Systems Compared: RAG vs Local SQLite"),
             c("Homebrew Formulae: homebrew-cask"),
             c("Top Vector Databases 2026 — pgvector vs Pinecone WHO WINS?", "https://www.youtube.com/watch?v=x"),
             c("Best AI Agent Memory Frameworks in 2026: Compared and Ranked"),
             c("Logseq alternatives for agent workflows"),
             c("Which knowledge base is right for you?"),
             c("The Complete Guide to Obsidian Automation")]
    drop = set(range(len(cands)))
    spared = spare_roundups(drop, cands)
    assert 2 not in spared and 7 not in spared          # not roundups
    assert 3 not in spared                              # a video, whatever its title
    assert len(spared) == _ROUNDUP_SPARES               # bounded per round
    assert spared == {0, 1, 4, 5}                       # best-ranked first
    assert spare_roundups({2, 7}, cands) == set()


def test_a_standard_hosted_or_discussed_by_a_third_party_is_an_aggregator():
    from app.models import NotesOut
    from app.research.notes import demote_third_party_standard as d
    std = lambda: NotesOut(relevance=9, source_type="standard", publisher="NousResearch")
    assert d(std(), "https://github.com/NousResearch/hermes-agent/issues/844").source_type == "aggregator"
    assert d(std(), "https://glama.ai/mcp/servers/jeanibarz/knowledge-base-mcp-server").source_type == "aggregator"
    assert d(std(), "https://discourse.joplinapp.org/t/joplin-mcp-server/12").source_type == "aggregator"
    kept = d(std(), "https://github.com/NousResearch/hermes-agent")
    assert kept.source_type == "standard" and kept.publisher == "NousResearch"
    assert d(std(), "https://joplinapp.org/help/about/changelog/desktop/").source_type == "standard"
    assert d(std(), "https://pypi.org/project/joplin-mcp/").source_type == "standard"
    other = d(NotesOut(relevance=5, source_type="practitioner"), "https://glama.ai/x")
    assert other.source_type == "practitioner"


def test_the_borderline_recheck_is_on_by_default():
    from app.config import Settings
    assert Settings().notes_recheck == "on"
