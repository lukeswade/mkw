"""The next-level batch: citation chasing, delta synthesis, page-2 backfill,
domain blocklist, and the stripped in-app readme."""
from __future__ import annotations

import httpx
import respx
from fastapi.testclient import TestClient

from app.db import Repo, connect, utcnow
from app.models import RunParams
from app.research.extractor import extract_links
from app.research.fetcher import Fetched
from app.research.orchestrator import Orchestrator
from app.research.pipeline import select_references
from app.research.progress import ProgressBus
from app.research.storage import RunStore
from app.web.routes_readme import _strip_images
from tests.fake_llm import FakeLLM
from tests.test_pipeline_e2e import SX, article, make_cfg, sx_payload, sx_result
from tests.test_web import make_app


# ---- link extraction --------------------------------------------------------

def _page(html: str, url: str = "https://source.com/post") -> Fetched:
    return Fetched(url=url, final_url=url, content_type="text/html",
                   body=html.encode())


def test_extract_links_resolves_and_filters():
    links = extract_links(_page(
        '<html><body>'
        '<a href="https://cited.org/paper">A cited paper</a>'
        '<a href="/relative/page">Relative link</a>'
        '<a href="mailto:x@y.z">mail</a>'
        '<a href="javascript:void(0)">js</a>'
        '</body></html>'))
    urls = dict(links)
    assert urls["https://cited.org/paper"] == "A cited paper"
    assert "https://source.com/relative/page" in urls  # resolved absolute
    assert all(u.startswith("http") for u in urls)


def test_extract_links_skips_pdf():
    pdf = Fetched(url="u", final_url="u", content_type="application/pdf",
                  body=b"%PDF")
    assert extract_links(pdf) == []


# ---- reference selection ----------------------------------------------------

def test_select_references_wants_citations_not_chrome():
    links = [
        ("https://source.com/about", "About us"),                # same domain
        ("https://twitter.com/share?u=x", "battery tweet"),      # social
        ("https://cited.org/solid-state-battery-yield", "solid state battery yield study"),
        ("https://ads.example.com/promo", "subscribe now"),      # zero overlap
    ]
    refs = select_references(links, source_url="https://source.com/post",
                             context="solid state battery manufacturing",
                             seen=set())
    assert [u for u, _a in refs] == ["https://cited.org/solid-state-battery-yield"]


def test_select_references_respects_seen_and_cap():
    links = [(f"https://cited.org/battery-report-{i}", f"battery report {i}")
             for i in range(10)]
    seen = {"https://cited.org/battery-report-0"}
    refs = select_references(links, source_url="https://elsewhere.com/x",
                             context="battery report", seen=seen, per_source=3)
    assert len(refs) == 3
    assert all(u != "https://cited.org/battery-report-0" for u, _ in refs)


# ---- e2e: citation chasing ----------------------------------------------------

def _script(synth="# T\n\nBody [1].\n"):
    return {
        # keep-all triage: candidate selection is under test elsewhere
        "triage": [{"drop": []}],
        "planner": [{"title": "T", "brief": "Investigate solid state batteries.",
                     "subqueries": ["q1"]}],
        "notes": [{"relevance": 8, "summary": "Useful.", "notes_md": "Notes.",
                   "key_facts": [], "published_date": None}],
        "gap": [{"state_md": "s", "saturated": True, "next_queries": []}],
        "synth": [synth],
        "followups": [{"items": []}],
    }


@respx.mock
async def test_citation_chasing_fetches_referenced_pages(data_dir):
    cfg = make_cfg(data_dir)
    linked = article("Article A").replace(
        "</article>",
        '<a href="https://ref-site.com/battery-manufacturing-deep-dive">'
        "solid state battery manufacturing deep dive</a></article>")
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result("https://example-a.com/article", "Article A")])))
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=linked))
    respx.get("https://ref-site.com/battery-manufacturing-deep-dive").mock(
        return_value=httpx.Response(200, html=article("Deep Dive")))

    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(_script()))
    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)

    findings = repo.findings_for_run(run_id)
    assert {f["domain"] for f in findings} == {"example-a.com", "ref-site.com"}
    store = RunStore(cfg.research_dir / run_id)
    chased = next(f for f in findings if f["domain"] == "ref-site.com")
    finding_md = (store.dir / chased["path"]).read_text()
    assert "cited by [1]" in finding_md  # provenance is visible


@respx.mock
async def test_chasing_can_be_disabled(data_dir):
    cfg = make_cfg(data_dir)
    cfg.reference_chasing = False
    linked = article("Article A").replace(
        "</article>",
        '<a href="https://ref-site.com/battery-manufacturing-deep-dive">'
        "solid state battery manufacturing deep dive</a></article>")
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result("https://example-a.com/article", "Article A")])))
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=linked))
    # NOTE: no route for ref-site.com — a fetch there would fail this test

    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(_script()))
    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)
    assert len(repo.findings_for_run(run_id)) == 1


# ---- e2e: blocked domains -------------------------------------------------------

@respx.mock
async def test_blocked_domains_are_never_fetched(data_dir):
    cfg = make_cfg(data_dir)
    cfg.blocked_domains = "example-b.org, pinterest.com"
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload([
        sx_result("https://example-a.com/article", "Article A"),
        sx_result("https://example-b.org/report", "Report B"),
    ])))
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=article("Article A")))
    # NOTE: no route for example-b.org — fetching it would fail the test

    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(_script()))
    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)
    assert {f["domain"] for f in repo.findings_for_run(run_id)} == {"example-a.com"}


# ---- e2e: page-2 backfill --------------------------------------------------------

@respx.mock
async def test_starved_round_pulls_page_two(data_dir):
    cfg = make_cfg(data_dir)
    pages_requested = []

    def handler(req):
        q = req.url.params["q"]
        page = req.url.params.get("pageno", "1")
        pages_requested.append((q, page))
        if page == "1":
            return httpx.Response(200, json=sx_payload(
                [sx_result("https://example-a.com/article", "Article A")]))
        return httpx.Response(200, json=sx_payload(
            [sx_result("https://example-e.com/followup", "Follow-up E")]
            if q == "q1" else []))

    respx.get(f"{SX}/search").mock(side_effect=handler)
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=article("Article A")))
    respx.get("https://example-e.com/followup").mock(
        return_value=httpx.Response(200, html=article("Follow-up E")))

    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(_script()))
    # breadth at depth 1 is 3; page 1 yields a single candidate → starved
    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)

    assert ("q1", "2") in pages_requested
    assert {f["domain"] for f in repo.findings_for_run(run_id)} == \
        {"example-a.com", "example-e.com"}


# ---- e2e: delta synthesis for parented runs ------------------------------------

@respx.mock
async def test_followup_run_synthesis_leads_with_whats_new(data_dir):
    cfg = make_cfg(data_dir)
    repo = Repo(connect(cfg.db_path))

    parent_store = RunStore.create(cfg.research_dir, "parent research")
    parent_id = parent_store.run_id
    repo.create_run(run_id=parent_id, query="parent research", depth=1,
                    recency="all", dir=parent_id, origin="web")
    repo.update_run(parent_id, status="completed", finished_at=utcnow())
    parent_store.write_overview(
        "# Parent\n\nEarlier finding: yields were 60% zanzibar-fact.\n")

    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result("https://example-a.com/article", "Article A")])))
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=article("Article A")))

    captured: list[str] = []

    def synth_capture(messages):
        captured.append(messages[-1]["content"])
        return "# T\n\n## What's new since the last look\n\n- newer [1]\n"

    script = _script()
    script["synth"] = [synth_capture]
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(script))
    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=1,
                                    recency="all", origin="web",
                                    parent_run_id=parent_id))
    await orch.execute_now(run_id)

    assert captured, "synthesis never ran"
    prompt = captured[0]
    assert "What's new since the last look" in prompt
    assert "zanzibar-fact" in prompt            # the parent overview was shown
    assert repo.get_run(run_id)["status"] == "completed"


@respx.mock
async def test_run_without_parent_gets_no_delta_block(data_dir):
    cfg = make_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result("https://example-a.com/article", "Article A")])))
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=article("Article A")))

    captured: list[str] = []

    def synth_capture(messages):
        captured.append(messages[-1]["content"])
        return "# T\n\nBody [1].\n"

    script = _script()
    script["synth"] = [synth_capture]
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(script))
    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=1,
                                    recency="all", origin="web"))
    await orch.execute_now(run_id)
    assert captured and "What's new since the last look" not in captured[0]


# ---- readme image stripping -----------------------------------------------------

def test_strip_images_removes_badges_and_screenshots():
    md = ("# Title\n\n"
          "[![CI](https://github.com/x/y/badge.svg)](https://github.com/x/y)\n"
          "[![Licence](https://img.shields.io/badge/l-MIT-blue.svg)](LICENSE)\n\n"
          "Intro text stays.\n\n"
          "![screenshot](docs/screenshots/home.png)\n\n"
          "More prose stays too.\n")
    out = _strip_images(md)
    assert "Intro text stays." in out
    assert "More prose stays too." in out
    assert "![" not in out
    assert "img.shields.io" not in out
    assert "docs/screenshots" not in out


def test_readme_page_serves_no_images(data_dir, monkeypatch):
    app, _ = make_app(data_dir, monkeypatch)
    with TestClient(app) as client:
        r = client.get("/readme")
    assert r.status_code == 200
    assert "Deep Research" in r.text
    prose = r.text.split('class="prose"', 1)[1]
    assert "<img" not in prose
    # the screenshot gallery must go with its images, or its captions narrate
    # pictures that aren't there
    assert "What it looks like" not in prose
    # spot-check the content tracks the current feature set
    assert "Citation chasing" in prose
    assert "Export PDF" in prose
    assert "general,science" in prose
