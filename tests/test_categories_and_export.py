"""Per-run search categories and the self-contained HTML export."""
from __future__ import annotations

import httpx
import respx

from app.db import Repo, connect
from app.models import RunParams
from app.research.orchestrator import Orchestrator
from app.research.progress import ProgressBus
from tests.fake_llm import FakeLLM
from tests.test_next_level import _script
from tests.test_pipeline_e2e import SX, article, make_cfg, sx_payload, sx_result


# ---- per-run categories -----------------------------------------------------------

@respx.mock
async def test_run_categories_override_the_global_setting(data_dir):
    cfg = make_cfg(data_dir)
    assert cfg.search_categories == "general,science"
    seen_categories: list[str] = []

    def handler(req):
        seen_categories.append(req.url.params.get("categories", ""))
        return httpx.Response(200, json=sx_payload(
            [sx_result("https://example-a.com/article", "Article A")]))

    respx.get(f"{SX}/search").mock(side_effect=handler)
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=article("Article A")))

    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(_script()))
    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=1,
                                    recency="all", origin="cli",
                                    categories="science,news"))
    await orch.execute_now(run_id)
    assert seen_categories and all(c == "science,news" for c in seen_categories)
    assert repo.get_run(run_id)["categories"] == "science,news"


@respx.mock
async def test_empty_categories_fall_back_to_global(data_dir):
    cfg = make_cfg(data_dir)
    seen: list[str] = []

    def handler(req):
        seen.append(req.url.params.get("categories", ""))
        return httpx.Response(200, json=sx_payload(
            [sx_result("https://example-a.com/article", "Article A")]))

    respx.get(f"{SX}/search").mock(side_effect=handler)
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=article("Article A")))

    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(_script()))
    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)
    assert seen and all(c == "general,science" for c in seen)


# ---- web form + retry inheritance -------------------------------------------------

def test_form_categories_reach_the_run_row(data_dir, monkeypatch):
    from fastapi.testclient import TestClient
    from tests.test_web import make_app
    app, _cfg = make_app(data_dir, monkeypatch)
    with TestClient(app) as client:
        repo = app.state.repo
        r = client.post("/runs", data={
            "query": "per-run category test", "depth": 1, "recency": "all",
            "categories": ["science", "news"]}, follow_redirects=False)
        assert r.status_code == 303
        run_id = r.headers["location"].rsplit("/", 1)[-1]
        assert repo.get_run(run_id)["categories"] == "science,news"

        r = client.post(f"/runs/{run_id}/retry", follow_redirects=False)
        retry_id = r.headers["location"].rsplit("/", 1)[-1]
        assert repo.get_run(retry_id)["categories"] == "science,news"


# ---- self-contained HTML export ----------------------------------------------------

@respx.mock
async def test_html_export_is_a_complete_standalone_page(data_dir, monkeypatch):
    from fastapi.testclient import TestClient
    from tests.test_web import make_app

    cfg = make_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result("https://example-a.com/article", "Article A")])))
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=article("Article A")))
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(_script(
                            synth="# T\n\nAn overview claim [1].\n")))
    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)

    app, _cfg = make_app(data_dir, monkeypatch)
    with TestClient(app) as client:
        r = client.get(f"/runs/{run_id}/export.html")
    assert r.status_code == 200
    page = r.text
    assert page.startswith("<!doctype html>")
    assert "An overview claim" in page
    assert "<details>" in page                      # collapsible source notes
    assert "https://example-a.com/article" in page  # bibliography link
    # self-contained: no references to the app's own static assets or hosts
    assert "/static/" not in page
    assert 'src="http' not in page
    assert "attachment" in r.headers["content-disposition"]


@respx.mock
async def test_interactive_export_is_a_one_file_mini_app(data_dir, monkeypatch):
    from fastapi.testclient import TestClient
    from tests.test_web import make_app

    cfg = make_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result("https://example-a.com/article", "Article A")])))
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=article("Article A")))
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(_script(
                            synth="# T\n\nAn overview claim [1].\n")))
    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)

    app, _cfg = make_app(data_dir, monkeypatch)
    with TestClient(app) as client:
        r = client.get(f"/runs/{run_id}/export-interactive.html")
    assert r.status_code == 200
    page = r.text
    assert page.startswith("<!doctype html>")
    assert 'data-tab="sources"' in page          # tab bar
    assert 'id="search-box"' in page             # client-side search
    assert 'href="#src-1"' in page               # citation jump target exists
    assert 'id="src-1"' in page
    assert "cited ×1" in page                    # back-reference count
    assert "— planning —" in page or "planning" in page   # log tab content
    assert "/static/" not in page                # self-contained
    assert 'src="http' not in page
    assert "interactive.html" in r.headers["content-disposition"]
