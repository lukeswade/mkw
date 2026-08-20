"""Returning a blank page after reading nine documents is its own failure.

When nothing clears the relevance bar the run should hand back the best
partial matches, clearly labelled — not an empty overview that reads like
"this topic has no sources".
"""
import httpx
import respx

from app.db import Repo, connect
from app.models import RunParams
from app.research.orchestrator import Orchestrator
from app.research.progress import ProgressBus
from app.research.storage import RunStore
from tests.fake_llm import FakeLLM
from tests.test_pipeline_e2e import article, make_cfg, sx_payload, sx_result, SX


def _script(relevance: int) -> dict:
    return {
        "triage": [{"drop": []}],
        "planner": [{"title": "Thin Topic", "brief": "Investigate.",
                     "subqueries": ["q1"]}],
        "notes": [{"relevance": relevance, "summary": "Only tangential.",
                   "notes_md": "Loosely related background.",
                   "key_facts": [{"claim": "A partial fact.",
                                  "evidence_quote": None, "confidence": 4}],
                   "published_date": None}],
        "gap": [{"state_md": "s", "saturated": True, "next_queries": []}],
        "synth": ["# Thin Topic\n\nBest available reading [1].\n"],
        "followups": [{"items": []}],
    }


def _mock_web():
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload([
        sx_result("https://example-a.com/article", "A"),
        sx_result("https://example-b.org/report", "B"),
    ])))
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=article("A")))
    respx.get("https://example-b.org/report").mock(
        return_value=httpx.Response(200, html=article("B")))


async def _run(cfg, relevance):
    _mock_web()
    repo = Repo(connect(cfg.db_path))
    bus = ProgressBus()
    llm = FakeLLM(_script(relevance))
    orch = Orchestrator(lambda: cfg, repo, bus, llm_factory=lambda: llm)
    run_id = orch.enqueue(RunParams(query="a thin topic", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)
    return repo, run_id, RunStore(cfg.research_dir / run_id)


@respx.mock
async def test_weak_sources_are_promoted_when_nothing_clears_the_bar(data_dir):
    cfg = make_cfg(data_dir)                      # default threshold is 4
    repo, run_id, store = await _run(cfg, relevance=3)

    findings = repo.findings_for_run(run_id)
    assert findings, "a run that read two documents must not return nothing"
    assert all(f["relevance"] == 3 for f in findings)
    overview = store.overview_path.read_text()
    assert "Thin result" in overview               # labelled, not passed off
    assert "partial matches" in overview
    assert "no strong matches" in repo.get_run(run_id)["stop_reason"]
    for f in findings:
        assert (store.dir / f["path"]).exists()


@respx.mock
async def test_worthless_sources_are_still_dropped(data_dir):
    """Below the floor there is nothing worth promoting."""
    cfg = make_cfg(data_dir)
    repo, run_id, store = await _run(cfg, relevance=0)
    assert repo.findings_for_run(run_id) == []
    assert "No relevant sources" in store.overview_path.read_text()


@respx.mock
async def test_strong_sources_are_not_labelled_thin(data_dir):
    cfg = make_cfg(data_dir)
    repo, run_id, store = await _run(cfg, relevance=8)
    assert len(repo.findings_for_run(run_id)) == 2
    assert "Thin result" not in store.overview_path.read_text()
