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
