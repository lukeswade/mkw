"""The run queue must outlive a run that cannot even be started.

Orchestrator._loop built its Pipeline before entering any try block, so one
exception there escaped the worker task and killed it. Nothing supervises
that task, so the failure was completely silent: the run stayed on "queued",
every run submitted afterwards also stayed on "queued", and only a restart
(where recover() re-enqueues them) ever cleared it.
"""
from __future__ import annotations

import asyncio

import httpx
import respx

from app.db import Repo, connect
from app.models import RunParams
from app.research.orchestrator import Orchestrator
from app.research.progress import ProgressBus
from app.research.storage import RunStore
from tests.fake_llm import FakeLLM
from tests.test_pipeline_e2e import SX, make_cfg, script, sx_payload

# Depth 0 is the cheapest complete run: one search, one chat call, no fetching.
QUICK = dict(script([]), chat=["# Answer\n\nA cited answer [1]."])


async def settled(repo, run_id: str, timeout: float = 5.0) -> str:
    """Wait for a run to leave queued/running; return its final status."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        status = repo.get_run(run_id)["status"]
        if status not in ("queued", "running"):
            return status
        await asyncio.sleep(0.02)
    return repo.get_run(run_id)["status"]


def orchestrator(cfg, repo, bus, cfg_loader=None) -> Orchestrator:
    respx.get(f"{SX}/search").mock(
        return_value=httpx.Response(200, json=sx_payload([])))
    return Orchestrator(cfg_loader or (lambda: cfg), repo, bus,
                        llm_factory=lambda: FakeLLM(QUICK))


@respx.mock
async def test_a_run_that_cannot_start_does_not_kill_the_worker(data_dir):
    cfg = make_cfg(data_dir)
    repo = Repo(connect(cfg.db_path))
    broken = {"yes": False}

    def cfg_loader():
        if broken["yes"]:
            broken["yes"] = False        # fail exactly once
            raise RuntimeError("settings.json is unreadable")
        return cfg

    orch = orchestrator(cfg, repo, ProgressBus(), cfg_loader)
    doomed = orch.enqueue(RunParams(query="first", depth=0, recency="all",
                                    origin="cli"))
    broken["yes"] = True                 # trips when the worker starts it
    orch.start()
    try:
        assert await settled(repo, doomed) == "failed"
        assert not orch._worker.done()   # the worker survived it

        # and the queue keeps draining — this is the part that used to hang
        later = orch.enqueue(RunParams(query="second", depth=0, recency="all",
                                       origin="cli"))
        assert await settled(repo, later) == "completed"
    finally:
        await orch.stop()


@respx.mock
async def test_a_failed_start_is_reported_not_left_queued(data_dir):
    """A run stuck on "queued" with no event is indistinguishable from one
    the worker simply has not reached yet, so say what happened."""
    cfg = make_cfg(data_dir)
    repo = Repo(connect(cfg.db_path))
    broken = {"yes": False}

    def cfg_loader():
        if broken["yes"]:
            broken["yes"] = False
            raise RuntimeError("settings.json is unreadable")
        return cfg

    orch = orchestrator(cfg, repo, ProgressBus(), cfg_loader)
    run_id = orch.enqueue(RunParams(query="doomed", depth=0, recency="all",
                                    origin="cli"))
    broken["yes"] = True
    orch.start()
    try:
        assert await settled(repo, run_id) == "failed"
        row = repo.get_run(run_id)
        assert "could not be started" in (row["error"] or "")

        events = RunStore(cfg.research_dir / row["dir"]).read_events()
        assert [e["type"] for e in events][-2:] == ["error", "done"]
        assert events[-1]["status"] == "failed"
    finally:
        await orch.stop()
