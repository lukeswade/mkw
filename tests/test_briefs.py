"""Named briefs: a saved reading list plus a standing interest."""
from __future__ import annotations

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app.config import load_settings, save_settings
from app.db import Repo, connect
from app.models import RunParams
from app.refresh_worker import run_due_briefs
from app.research.orchestrator import Orchestrator
from app.research.progress import ProgressBus
from app.web.server import create_app
from tests.fake_llm import FakeLLM
from tests.test_feeds import RSS
from tests.test_pipeline_e2e import SX, article, make_cfg, script, sx_payload

PAGE = ('<html><head><link rel="alternate" type="application/rss+xml" '
        'href="/feed.xml"></head><body>x</body></html>')


@pytest.fixture
def web(data_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    app = create_app(enable_worker=False, enable_bot=False)
    cfg = load_settings(str(data_dir))
    return app, cfg, Repo(connect(cfg.db_path))


# ---- managing them ----------------------------------------------------------

def test_creating_and_listing_a_brief(web):
    app, _cfg, repo = web
    with TestClient(app) as c:
        c.post("/briefs", data={"name": "Local LLM", "topic": "inference",
                                "recency": "week", "depth": "4"},
               follow_redirects=False)
        body = c.get("/briefs").text
    assert "Local LLM" in body and "inference" in body
    assert [b["name"] for b in repo.list_briefs()] == ["Local LLM"]


def test_a_brief_needs_a_name(web):
    app, _cfg, repo = web
    with TestClient(app) as c:
        r = c.post("/briefs", data={"name": "   "}, follow_redirects=False)
    assert r.status_code == 303 and "error" in r.headers["location"]
    assert repo.list_briefs() == []


@respx.mock
def test_adding_a_source_by_address_attaches_it_to_that_brief(web):
    app, _cfg, repo = web
    respx.get("https://site.test/").mock(return_value=httpx.Response(200, html=PAGE))
    respx.get("https://site.test/feed.xml").mock(
        return_value=httpx.Response(200, content=RSS))
    bid = repo.create_brief(name="Local LLM")
    other = repo.create_brief(name="Automotive")
    with TestClient(app) as c:
        r = c.post(f"/briefs/{bid}/feeds", data={"site": "site.test"})
    assert "Local LLM Weekly" in r.text
    assert "https://site.test/feed.xml" in repo.get_brief(bid)["feeds"]
    # and only to that one — the point of naming them
    assert not repo.get_brief(other)["feeds"].strip()


def test_editing_replaces_feeds_and_settings(web):
    app, _cfg, repo = web
    bid = repo.create_brief(name="Old", feeds="https://a/f\n")
    with TestClient(app) as c:
        c.post(f"/briefs/{bid}/edit",
               data={"name": "New", "topic": "t", "feeds": "https://b/f\n",
                     "recency": "month", "depth": "6"},
               follow_redirects=False)
    b = repo.get_brief(bid)
    assert (b["name"], b["topic"], b["recency"], b["depth"]) == \
        ("New", "t", "month", 6)
    assert "https://b/f" in b["feeds"] and "https://a/f" not in b["feeds"]


def test_running_a_brief_with_no_feeds_is_refused(web):
    app, _cfg, repo = web
    bid = repo.create_brief(name="Empty")
    with TestClient(app) as c:
        r = c.post(f"/briefs/{bid}/run", follow_redirects=False)
    assert "no+feeds" in r.headers["location"]


# ---- what a run actually reads ----------------------------------------------

@respx.mock
async def test_a_named_brief_uses_its_own_feeds_and_topic(data_dir):
    """The whole point: two briefs must not read each other's sources."""
    cfg = make_cfg(data_dir)
    cfg.feeds = "https://global.test/feed.xml"          # the old global list
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload([])))
    respx.get("https://site.test/feed.xml").mock(
        return_value=httpx.Response(200, content=RSS))
    respx.get("https://example.com/mlx-030").mock(
        return_value=httpx.Response(200, html=article("MLX 0.30")))
    respx.get("https://example.com/old").mock(
        return_value=httpx.Response(200, html=article("Older")))
    global_feed = respx.get("https://global.test/feed.xml").mock(
        return_value=httpx.Response(200, content=RSS))

    repo = Repo(connect(cfg.db_path))
    bid = repo.create_brief(name="Named", feeds="https://site.test/feed.xml\n",
                            topic="only MLX things")

    seen: list[str] = []

    class Recorder(FakeLLM):
        async def chat_json(self, kind, messages, schema, **kw):
            if schema.__name__ == "BriefFilterOut":
                seen.append(messages[-1]["content"])
                return schema.model_validate({"keep": [0]})
            return await super().chat_json(kind, messages, schema, **kw)

    s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: Recorder(s))
    rid = orch.enqueue(RunParams(query="Brief: Named", depth=4, recency="all",
                                 origin="cli", kind="brief", brief_id=bid))
    await orch.execute_now(rid)

    assert repo.get_run(rid)["status"] == "completed"
    assert not global_feed.called                  # the global list stayed out
    assert seen and "only MLX things" in seen[0]   # the brief's own interest


@respx.mock
async def test_without_a_brief_id_the_global_list_still_works(data_dir):
    """Back-compat: the Settings feeds behave as one unnamed brief."""
    cfg = make_cfg(data_dir)
    cfg.feeds = "https://global.test/feed.xml"
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload([])))
    g = respx.get("https://global.test/feed.xml").mock(
        return_value=httpx.Response(200, content=RSS))
    respx.get("https://example.com/mlx-030").mock(
        return_value=httpx.Response(200, html=article("MLX")))
    respx.get("https://example.com/old").mock(
        return_value=httpx.Response(200, html=article("Old")))
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(script(
                            [{"state_md": "s", "saturated": True,
                              "next_queries": []}])))
    rid = orch.enqueue(RunParams(query="Brief", depth=4, recency="all",
                                 origin="cli", kind="brief"))
    await orch.execute_now(rid)
    assert g.called
    assert repo.get_run(rid)["status"] == "completed"


# ---- scheduling -------------------------------------------------------------

class _Orch:
    def __init__(self):
        self.queued: list[RunParams] = []

    def enqueue(self, params):
        self.queued.append(params)
        return f"run-{len(self.queued)}"


async def test_only_daily_briefs_with_feeds_are_scheduled(data_dir):
    cfg = make_cfg(data_dir)
    repo = Repo(connect(cfg.db_path))
    on = repo.create_brief(name="Daily", feeds="https://a/f\n")
    repo.update_brief(on, daily=1)
    repo.create_brief(name="Manual", feeds="https://b/f\n")       # daily off
    empty = repo.create_brief(name="Daily but empty")
    repo.update_brief(empty, daily=1)

    orch = _Orch()
    assert await run_due_briefs(orch, repo) == 1
    assert orch.queued[0].brief_id == on
    assert repo.get_brief(on)["last_run_at"] is not None


async def test_a_scheduled_brief_does_not_re_fire_immediately(data_dir):
    """last_run_at is stamped on enqueue, so a failing brief cannot retry
    every fifteen minutes for a day."""
    cfg = make_cfg(data_dir)
    repo = Repo(connect(cfg.db_path))
    bid = repo.create_brief(name="Daily", feeds="https://a/f\n")
    repo.update_brief(bid, daily=1)
    orch = _Orch()
    assert await run_due_briefs(orch, repo) == 1
    assert await run_due_briefs(orch, repo) == 0


# ---- the legacy global list lives on the Briefs page now --------------------

def test_the_new_tab_no_longer_starts_briefs(web):
    """Briefs outgrew the research form; they have their own page."""
    app, _cfg, _repo = web
    with TestClient(app) as c:
        home = c.get("/").text
    assert 'value="brief"' not in home
    assert "Brief from feeds" not in home


def test_global_feeds_appear_on_the_briefs_page_and_can_run(web):
    app, cfg, _repo = web
    save_settings(cfg.settings_path, {"feeds": "https://a.test/feed\n"})
    with TestClient(app) as c:
        assert "Settings feeds" in c.get("/briefs").text
        r = c.post("/briefs/global/run", follow_redirects=False)
    # "global" must not be parsed as a brief id — those routes are declared
    # after these on purpose
    assert r.status_code == 303 and "/runs/" in r.headers["location"]


def test_adopting_global_feeds_moves_them_into_a_named_brief(web):
    app, cfg, repo = web
    save_settings(cfg.settings_path,
                  {"feeds": "https://a.test/feed\nhttps://b.test/f\n"})
    with TestClient(app) as c:
        c.post("/briefs/global/adopt", follow_redirects=False)
        body = c.get("/briefs").text
    briefs = repo.list_briefs()
    assert [b["name"] for b in briefs] == ["My feeds"]
    assert "a.test" in briefs[0]["feeds"] and "b.test" in briefs[0]["feeds"]
    # and the global list is emptied, so the same feeds are not read twice
    assert not load_settings(str(cfg.data_path)).feeds.strip()
    assert "Settings feeds" not in body


def test_adopting_with_no_global_feeds_is_refused(web):
    app, _cfg, repo = web
    with TestClient(app) as c:
        r = c.post("/briefs/global/adopt", follow_redirects=False)
    assert "error" in r.headers["location"]
    assert repo.list_briefs() == []
