"""Candidate triage, full-page notes input, and index-page link chasing."""
from __future__ import annotations

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
    events = (cfg.research_dir / run_id / "events.jsonl").read_text()
    assert "dropped at triage" in events
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
async def test_condemning_everything_is_ignored(data_dir):
    cfg = make_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(
        200, json=sx_payload(_five_candidates())))
    for c in "abcde":
        respx.get(f"https://example-{c}.com/article").mock(
            return_value=httpx.Response(200, html=article(f"Article {c.upper()}")))

    s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    s["triage"] = [{"drop": [0, 1, 2, 3, 4]}]   # drop-all = broken verdict
    repo, llm, orch, run_id = _run(cfg, s)
    await orch.execute_now(run_id)
    assert len(repo.findings_for_run(run_id)) == 5


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
    assert "dropped at triage" in events  # triage genuinely ran


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


# ---- fat indexes and multi-level descent ----------------------------------------

def test_looks_like_index_distinguishes_directories_from_leaves():
    """The three tiers of charm.li's GX470 manual, which no length threshold
    alone can separate: a 125k-char table of contents, an 872-char shell
    naming two children, and a 977-char leaf holding the factory spec."""
    from app.research.pipeline import looks_like_index
    base = "https://charm.li/Lexus/2008/GX/Repair/"

    toc_links = [(f"{base}Section%20{i}/", f"Section {i} of the manual")
                 for i in range(200)]
    toc_text = " ".join(a for _u, a in toc_links)
    assert looks_like_index(toc_text, toc_links, base)

    shell_links = [(base + "Spark%20Plug/Specifications/", "Specifications"),
                   (base + "Spark%20Plug/Application/", "Application and ID")]
    shell_text = "Spark Plug " + "banner boilerplate about LEMON manuals " * 20
    assert len(shell_text) > 600          # too long for the stub rule
    assert looks_like_index(shell_text, shell_links, base + "Spark%20Plug/")

    leaf = base + "Spark%20Plug/Specifications/"
    leaf_links = [(base, "Repair and Diagnosis"), ("https://charm.li/", "Home")]
    leaf_text = ("Spark plug electrode gap 1.0 to 1.1 mm (0.039 to 0.043 in.) "
                 "Maximum electrode gap 1.3 mm " + "boilerplate " * 60)
    assert not looks_like_index(leaf_text, leaf_links, leaf)   # points only up


@respx.mock
async def test_index_chase_descends_through_multiple_levels(data_dir):
    """A service manual buries its content three levels down. One hop reaches
    a subsection index and stops; the leaf is only reachable by descending.

    Curated as an authority site on purpose: a fat directory is only followed
    there. Letting any link-dense 0/10 page start a descent walked three hops
    into a spark-plug manufacturer's company-philosophy pages.
    """
    cfg = make_cfg(data_dir)
    cfg.authority_sites = "manuals.example.com — factory service manuals"
    M = "https://manuals.example.com"

    def index(title, children):
        # Varied prose on purpose: trafilatura collapses a repeated identical
        # sentence, which would leave too little text to extract at all.
        prose = " ".join(
            f"Subsection {i} covers spark plug service task number {i} for "
            f"this engine family, including inspection and torque values."
            for i in range(8))
        links = "".join(f"<a href='{h}'>{t}</a> " for h, t in children)
        return (f"<html><head><title>{title}</title></head><body><main>"
                f"<article><h1>{title}</h1><p>{prose}</p>"
                f"{links}</article></main></body></html>")

    respx.get(f"{SX}/search").mock(return_value=httpx.Response(
        200, json=sx_payload([sx_result(f"{M}/gx470/repair/", "Repair")])))
    respx.get(f"{M}/gx470/repair/").mock(return_value=httpx.Response(
        200, html=index("Repair", [("/gx470/repair/spark-plug/", "Spark Plug")])))
    respx.get(f"{M}/gx470/repair/spark-plug/").mock(return_value=httpx.Response(
        200, html=index("Spark Plug",
                        [("/gx470/repair/spark-plug/specifications/",
                          "Spark plug torque specifications")])))
    respx.get(f"{M}/gx470/repair/spark-plug/specifications/").mock(
        return_value=httpx.Response(200, html=article("Torque Specifications")))

    s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    s["planner"] = [{"title": "T", "brief": "GX470 spark plug torque specs.",
                     "subqueries": ["gx470 spark plug torque"]}]
    s["notes"] = [
        {"relevance": 1, "summary": "Directory.", "notes_md": "", 
         "key_facts": [], "published_date": None},           # level 1 index
        {"relevance": 1, "summary": "Directory.", "notes_md": "",
         "key_facts": [], "published_date": None},           # level 2 index
        {"relevance": 9, "summary": "The factory spec.", "notes_md": "n",
         "key_facts": [], "published_date": None},           # level 3 leaf
    ]
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(s))
    run_id = orch.enqueue(RunParams(query="gx470 spark plug torque specs",
                                    depth=1, recency="all", origin="cli"))
    await orch.execute_now(run_id)

    findings = repo.findings_for_run(run_id)
    assert [f["url"] for f in findings] == \
        [f"{M}/gx470/repair/spark-plug/specifications/"]
    assert findings[0]["relevance"] == 9


@respx.mock
async def test_link_dense_page_off_authority_does_not_start_a_descent(data_dir):
    """The regression that killed a live run: a spark-plug manufacturer's
    homepage is link-dense and scores 0/10, so it read as an index and the
    chase walked three hops into /info/philosophy/ and /info/vision/, burning
    twelve fetches and twelve notes calls on company boilerplate. Fat
    directories are only followed on curated authority domains."""
    cfg = make_cfg(data_dir)
    cfg.authority_sites = "charm.li — factory service manuals"
    corp = "https://plugmaker.example.com"
    nav = "".join(f"<a href='/info/{p}/'>{p.title()} of the company</a> "
                  for p in ("philosophy", "vision", "gallery", "history",
                            "careers", "investors", "brands", "contact"))
    home = ("<html><head><title>Plugmaker</title></head><body><main><article>"
            "<h1>Plugmaker</h1><p>" +
            " ".join(f"Corporate statement number {i} about spark plug "
                     f"manufacturing excellence worldwide." for i in range(6))
            + f"</p>{nav}</article></main></body></html>")

    respx.get(f"{SX}/search").mock(return_value=httpx.Response(
        200, json=sx_payload([sx_result(f"{corp}/", "Spark plugs")])))
    respx.get(f"{corp}/").mock(return_value=httpx.Response(200, html=home))
    # no routes for /info/* — descending into them would fail this test

    s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    s["planner"] = [{"title": "T", "brief": "GX470 spark plug replacement.",
                     "subqueries": ["gx470 spark plug"]}]
    s["notes"] = [{"relevance": 0, "summary": "Corporate homepage.",
                   "notes_md": "", "key_facts": [], "published_date": None}]
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(s))
    run_id = orch.enqueue(RunParams(query="gx470 spark plug replacement",
                                    depth=1, recency="all", origin="cli"))
    await orch.execute_now(run_id)

    events = (cfg.research_dir / run_id / "events.jsonl").read_text()
    assert "hop 1" not in events              # never descended
    assert not repo.findings_for_run(run_id)
