"""Giving the model back: the app asks a local server to unload its model
once it has stopped using it (llm/idle.py).

The model is pinned on the server, so nothing else ever unloads it; it held
about 20 GB for hours after each run until it was unloaded by hand.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app.config import Settings
from app.llm import idle
from app.llm.client import LLM
from app.web.server import create_app

URL = "http://model.test/v1/models/Qwen3.6-35B-A3B-oQ4e-mtp/unload"


def cfg(**kw) -> Settings:
    base = dict(llm_provider="local", llm_model="Jundot/Qwen3.6-35B-A3B-oQ4e-mtp",
                llm_base_url="http://model.test/v1", llm_api_key="k",
                model_idle_unload_seconds=120)
    return Settings(**{**base, **kw})


@pytest.fixture(autouse=True)
def fresh_activity(monkeypatch):
    monkeypatch.setattr(idle, "activity", idle.Activity())


def used(ago: float, now: float = 1000.0) -> float:
    """The model was last used `ago` seconds before `now`."""
    idle.activity.used = True
    idle.activity.last = now - ago
    return now


# --- knowing when the model is in use ------------------------------------------

def test_a_call_is_counted_while_it_runs_and_stamped_when_it_ends():
    with idle.calling():
        assert idle.activity.in_flight == 1 and idle.activity.used
        assert not idle.due(0)                       # in flight is never idle
    assert idle.activity.in_flight == 0 and idle.activity.last > 0
    with pytest.raises(RuntimeError):
        with idle.calling():
            raise RuntimeError("server refused")
    assert idle.activity.in_flight == 0              # a failed call ends too


def test_due_only_after_the_idle_time_and_only_if_this_process_used_the_model():
    assert not idle.due(120, now=10_000.0)           # never used here: not ours to unload
    now = used(ago=119)
    assert not idle.due(120, now)
    assert idle.due(120, used(ago=120))


async def _reply(**_kw):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"),
                                                    finish_reason="stop")],
                           usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))


async def test_the_clients_calls_are_what_is_counted():
    llm = LLM(cfg())
    seen = []

    async def create(**kw):
        seen.append(idle.activity.in_flight)
        return await _reply()
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    await llm.chat("probe", [{"role": "user", "content": "hi"}])
    assert seen == [1] and idle.activity.in_flight == 0 and idle.activity.used


async def test_a_streamed_call_is_counted_for_as_long_as_it_streams():
    # synthesis is streamed and is the longest call of a run
    llm = LLM(cfg())
    seen = []

    async def chunks():
        for word in ("# T", " body"):
            seen.append(idle.activity.in_flight)
            yield SimpleNamespace(usage=None, choices=[SimpleNamespace(
                delta=SimpleNamespace(content=word))])

    async def create(**kw):
        return chunks()
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    bus = SimpleNamespace(publish=lambda *a, **k: None)
    assert await llm.chat_stream("synth", [{"role": "user", "content": "hi"}], bus, "r1") == "# T body"
    assert seen == [1, 1] and idle.activity.in_flight == 0


# --- asking the server ---------------------------------------------------------

@respx.mock
async def test_unload_names_the_model_as_the_server_does_and_reads_its_answer():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json={"status": "ok"}))
    assert await idle.unload(cfg()) == "unloaded"
    assert route.calls[0].request.headers["authorization"] == "Bearer k"
    respx.post(URL).mock(return_value=httpx.Response(400))
    assert await idle.unload(cfg()) == "not loaded"
    respx.post(URL).mock(return_value=httpx.Response(404))
    assert await idle.unload(cfg()) == "unsupported"
    respx.post(URL).mock(side_effect=httpx.ConnectError("down"))
    assert await idle.unload(cfg()) == "unreachable"


@respx.mock
async def test_a_separate_notes_model_is_unloaded_too():
    main = respx.post(URL).mock(return_value=httpx.Response(400))
    fast = respx.post("http://model.test/v1/models/small-model/unload").mock(
        return_value=httpx.Response(200))
    assert await idle.unload(cfg(fast_model="small-model")) == "unloaded"
    assert main.called and fast.called


# --- deciding ------------------------------------------------------------------

@respx.mock
async def test_the_model_is_unloaded_once_after_the_idle_time():
    route = respx.post(URL).mock(return_value=httpx.Response(200))
    now = used(ago=121)
    assert await idle.tick(cfg(), busy=False, now=now) == "unloaded"
    assert await idle.tick(cfg(), busy=False, now=now + 60) is None     # nothing of ours is loaded
    assert route.call_count == 1


@respx.mock
async def test_nothing_is_unloaded_when_off_busy_early_or_not_a_local_server():
    route = respx.post(URL).mock(return_value=httpx.Response(200))
    now = used(ago=500)
    assert await idle.tick(cfg(model_idle_unload_seconds=0), busy=False, now=now) is None
    assert await idle.tick(cfg(), busy=True, now=now) is None           # a run is active or queued
    assert await idle.tick(cfg(llm_provider="anthropic"), busy=False, now=now) is None
    assert await idle.tick(cfg(), busy=False, now=used(ago=30)) is None
    assert not route.called


@respx.mock
async def test_a_server_without_the_route_is_asked_once_not_every_fifteen_seconds():
    route = respx.post(URL).mock(return_value=httpx.Response(404))
    now = used(ago=500)
    assert await idle.tick(cfg(), busy=False, now=now) == "unsupported"
    assert await idle.tick(cfg(), busy=False, now=now + 15) is None
    assert route.call_count == 1


async def test_the_watcher_rereads_settings_and_survives_a_bad_check(monkeypatch):
    calls = []

    async def tick(c, busy):
        calls.append((c, busy))
        if len(calls) == 1:
            raise RuntimeError("boom")
    monkeypatch.setattr(idle, "tick", tick)
    task = asyncio.create_task(idle.watch(lambda: f"cfg{len(calls)}", lambda: len(calls) % 2 == 1,
                                          every=0.01))
    for _ in range(200):
        if len(calls) >= 3:
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls[:3] == [("cfg0", False), ("cfg1", True), ("cfg2", False)]


# --- in the app ----------------------------------------------------------------

def test_the_app_runs_the_watcher_beside_its_worker(data_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    started = []

    async def watch(cfg_loader, is_busy, every=15.0):
        started.append(is_busy())
        await asyncio.Event().wait()
    monkeypatch.setattr(idle, "watch", watch)
    with TestClient(create_app(enable_worker=True, enable_bot=False)) as client:
        assert client.get("/health").json()["status"] == "ok"
    assert started == [False]                        # started, and nothing was running


def test_the_setting_is_saved_from_the_form_within_bounds(data_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    from app.config import load_settings
    with TestClient(create_app(enable_worker=False, enable_bot=False)) as client:
        page = client.get("/settings").text
        assert 'name="model_idle_unload_seconds"' in page
        client.post("/settings", data={"model_idle_unload_seconds": "120"}, follow_redirects=False)
        assert load_settings(str(data_dir)).model_idle_unload_seconds == 120
        client.post("/settings", data={"model_idle_unload_seconds": "-5"}, follow_redirects=False)
        assert load_settings(str(data_dir)).model_idle_unload_seconds == 0
