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
    for c in ("a", "c"):
        respx.get(f"https://example-{c}.com/article").mock(
            return_value=httpx.Response(200, html=article(f"Article {c.upper()}")))

    s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    s["triage"] = [{"drop": [1, 3, 4]}]
    repo, llm, orch, run_id = _run(make_cfg(data_dir), s)
    await orch.execute_now(run_id)

    cfg = orch.cfg_loader()
    findings = repo.findings_for_run(run_id)
    assert {f["domain"] for f in findings} == {"example-a.com", "example-c.com"}
    events = (cfg.research_dir / run_id / "events.jsonl").read_text()
    assert "dropped at triage" in events
    assert llm.calls["notes"] == 2


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
    """Curating a site as authoritative outranks a title-level guess."""
    cfg = make_cfg(data_dir)
    cfg.authority_sites = "charm.li — factory service manuals for cars"
    fsm = ("https://charm.li/Toyota/2006/Tundra%20V8-4.7L%20(2UZ-FE)/"
           "Spark%20Plug/")
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload([
        sx_result(fsm, "Spark Plug — Tundra 2UZ-FE"),
        sx_result("https://junk.example.com/ad", "Buy spark plugs cheap"),
    ])))
    respx.get(fsm).mock(return_value=httpx.Response(
        200, html=article("Tundra 2UZ-FE Spark Plug")))
    # NOTE: no route for junk.example.com — fetching it would fail the test

    s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    s["triage"] = [{"drop": [0, 1]}]      # triage condemns the FSM page too
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(s))
    run_id = orch.enqueue(RunParams(query="gx470 2UZ-FE spark plugs", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)

    findings = repo.findings_for_run(run_id)
    assert [f["domain"] for f in findings] == ["charm.li"]


def test_authority_domains_parses_the_settings_blob(data_dir):
    from app.research.pipeline import Pipeline
    from app.research.progress import ProgressBus as PB
    cfg = make_cfg(data_dir)
    cfg.authority_sites = ("charm.li — factory service manuals\n"
                           "https://www.nist.gov/ — standards\n"
                           "not-a-domain line\n")
    p = Pipeline(cfg, Repo(connect(cfg.db_path)), PB())
    assert p._authority_domains() == frozenset({"charm.li", "nist.gov"})
