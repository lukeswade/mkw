"""A model server that is down stops the run instead of being skipped around.

take_notes turned every failed notes call into a skipped source, which is
right for one bad page and wrong when the server itself is down: out of
memory, unreachable, or refusing the key, every source fails the same way,
and a run would read its way through a round of "unusable notes output"
before failing at synthesis. The claim judge did the same and finished with
a report of "unverifiable" verdicts. ServerUnavailable now stops both.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from app.config import Settings
from app.db import Repo, connect
from app.llm.client import LLM, LLMError, PromptExceedsFreeMemory, ServerOutOfMemory
from app.models import RunParams
from app.research import verify
from app.research.notes import take_notes
from app.research.orchestrator import Orchestrator
from app.research.pipeline import _gather_or_cancel
from app.research.progress import ProgressBus
from tests.fake_llm import FakeLLM
from tests.test_pipeline_e2e import SX, article, make_cfg, script, sx_payload, sx_result

_OOM = ServerOutOfMemory("LLM call 'notes' refused: the server is out of memory. "
                         "oMLX prefill memory guard rejected this prompt")
_CLAIM = "EmbeddingGemma has 308 million parameters."
_EVIDENCE = [verify.Evidence(n=1, label="a page", url="https://example.com/eg",
                             text="With just 308M parameters, EmbeddingGemma ...")]


@pytest.mark.asyncio
async def test_notes_let_a_down_server_stop_the_run(data_dir):
    llm = LLM(Settings(data_dir=str(data_dir), llm_provider="openai",
                       llm_api_key="sk-test", llm_model="m"))
    llm.chat_json = AsyncMock(side_effect=_OOM)
    with pytest.raises(ServerOutOfMemory):
        await take_notes(llm, brief="b", recency_desc="all time",
                         today="2026-10-07", url="http://x/y", title="t",
                         detected_date=None, text="a page worth reading",
                         keywords=None)


@pytest.mark.asyncio
async def test_a_claim_check_fails_rather_than_report_every_claim_unverifiable():
    down = SimpleNamespace(chat_json=AsyncMock(side_effect=_OOM))
    with pytest.raises(ServerOutOfMemory):
        await verify.judge(down, _CLAIM, _EVIDENCE)
    # a call that fails for its own reasons still costs just that claim
    slow = SimpleNamespace(chat_json=AsyncMock(
        side_effect=LLMError("exceeded the 600s ceiling")))
    out = await verify.judge(slow, _CLAIM, _EVIDENCE)
    assert out.verdict == "unverifiable"


@pytest.mark.asyncio
async def test_the_first_failure_in_a_round_cancels_the_other_sources():
    cancelled: list[int] = []

    async def refused():
        await asyncio.sleep(0)
        raise _OOM

    async def reading(i: int):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(i)
            raise

    with pytest.raises(ServerOutOfMemory):
        await _gather_or_cancel([refused(), reading(1), reading(2)])
    assert sorted(cancelled) == [1, 2]


@pytest.mark.asyncio
async def test_a_round_that_succeeds_runs_every_source():
    done: list[int] = []

    async def read(i: int):
        await asyncio.sleep(0)
        done.append(i)

    await _gather_or_cancel(read(i) for i in range(4))
    assert sorted(done) == [0, 1, 2, 3]


@respx.mock
async def test_a_run_whose_server_runs_out_of_memory_fails_with_the_reason(data_dir):
    cfg = make_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result(f"https://example-{c}.com/article", f"Article {c.upper()}")
         for c in "abcde"])))
    for c in "abcde":
        respx.get(f"https://example-{c}.com/article").mock(
            return_value=httpx.Response(200, html=article(f"Article {c.upper()}")))
    s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    s["notes"] = [_OOM]

    repo = Repo(connect(cfg.db_path))
    llm = FakeLLM(s)
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(), llm_factory=lambda: llm)
    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)

    row = repo.get_run(run_id)
    assert row["status"] == "failed"
    assert "out of memory" in row["error"]
    assert llm.calls["synth"] == 0
    events = [json.loads(line) for line in
              (cfg.research_dir / run_id / "events.jsonl").read_text().splitlines()
              if line.strip()]
    assert not any(e["type"] == "source_skipped"
                   and e.get("reason") == "unusable notes output" for e in events)


@respx.mock
async def test_a_page_too_big_for_the_free_memory_is_skipped_and_the_run_completes(data_dir):
    """2026-10-07 afternoon: 0.66 GB of headroom refused one long page. Stopping
    the run there threw away every page that fit; skipping it is right."""
    cfg = make_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result(f"https://example-{c}.com/article", f"Article {c.upper()}")
         for c in "abcde"])))
    for c in "abcde":
        respx.get(f"https://example-{c}.com/article").mock(
            return_value=httpx.Response(200, html=article(f"Article {c.upper()}")))
    s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    notes_ok = s["notes"][0]

    def notes(messages):
        if "example-c.com" in messages[-1]["content"]:
            return PromptExceedsFreeMemory(
                "LLM call 'notes' refused: not enough free memory for a prompt this long.")
        return notes_ok
    s["notes"] = [notes]

    repo = Repo(connect(cfg.db_path))
    llm = FakeLLM(s)
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(), llm_factory=lambda: llm)
    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)

    assert repo.get_run(run_id)["status"] == "completed"
    assert {f["domain"] for f in repo.findings_for_run(run_id)} == {
        "example-a.com", "example-b.com", "example-d.com", "example-e.com"}
