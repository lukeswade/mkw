"""Full pipeline run over a fake internet: respx-mocked SearXNG + pages,
scripted FakeLLM. Covers dedupe, recency filtering, SSRF guard, PDF handling,
fetch failure tolerance, citation validation, artifacts, and cancellation.
"""
from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timedelta

import httpx
import pytest
import respx

from app.config import Settings
from app.db import Repo, connect
from app.models import RunParams
from app.research.orchestrator import Orchestrator
from app.research.progress import ProgressBus
from app.research.storage import RunStore
from tests.fake_llm import FakeLLM

SX = "http://searxng.test"

PARA = ("This is a substantive paragraph about solid state batteries with "
        "enough words to satisfy the extractor minimum length requirements. "
        "It discusses energy density figures, manufacturing yield problems, "
        "and the specific companies working on sulfide electrolytes today. ")


def article(title: str) -> str:
    # Body text is salted with the title so distinct fixture pages read as
    # distinct content to the pipeline's near-duplicate collapse — the same
    # way distinct real pages carry distinct prose. Serving the SAME title
    # from two URLs therefore models a scraped clone.
    salt = re.sub(r"[^a-z0-9]+", "", title.lower()) or "topic"
    body = f"<p>{PARA}</p>" + "".join(
        f"<p>Finding {i} for {salt}: measurement series {salt}-{i} recorded "
        f"result {i} under condition {salt}{i}, with commentary specific to "
        f"the {salt} experiment, stage {i} of the {salt} write-up.</p>"
        for i in range(6))
    return (f"<!DOCTYPE html><html><head><title>{title}</title></head>"
            f"<body><main><article><h1>{title}</h1>{body}</article></main>"
            f"</body></html>")


def pdf_bytes() -> bytes:
    import pymupdf
    doc = pymupdf.open()
    page = doc.new_page()
    text = "PDF research paper on electrolyte tooling. " + " ".join(
        f"Bench row {i}: pdf-only figure {i} with annotation {i}."
        for i in range(30))
    for i, line in enumerate([text[j:j + 80] for j in range(0, len(text), 80)]):
        page.insert_text((50, 60 + 14 * i), line)
    out = doc.tobytes()
    doc.close()
    return out


def sx_payload(results):
    return {"results": results, "unresponsive_engines": []}


def sx_result(url, title, published=None):
    return {"url": url, "title": title, "content": "snippet text here",
            "engine": "test", "publishedDate": published}


def make_cfg(data_dir) -> Settings:
    s = Settings(data_dir=str(data_dir), searxng_url=SX, respect_robots=False)
    s.ensure_dirs()
    return s


def script(gap_rounds: list[dict]) -> dict:
    return {
        # keep-all triage: candidate selection is under test elsewhere
        "triage": [{"drop": []}],
        "planner": [{"title": "Test Research", "brief": "Investigate X thoroughly.",
                     "subqueries": ["q1", "q2"]}],
        "notes": [{"relevance": 8, "summary": "Useful source about X.",
                   "notes_md": "Detailed flibbertigibbet notes about X.",
                   "key_facts": [
                       {"claim": "Fact one.", "evidence_quote": "quoted one.",
                        "confidence": 9},
                       {"claim": "Fact two.", "evidence_quote": None,
                        "confidence": 6},
                   ],
                   "published_date": None}],
        "gap": gap_rounds,
        "synth": ["# Test Research\n\n## TL;DR\n\n- Key point [1][2]\n\n"
                  "## Detail\n\nSolid claim [1]. Another [3]. Bogus citation [99].\n\n"
                  "## Open questions\n\n- More?\n"],
        "followups": [{"items": [{"query": "Follow-up question about X",
                                  "rationale": "Gap remains.", "depth": 3,
                                  "recency": "6months"}]}],
    }


def mock_internet(old_date: str):
    respx.get(f"{SX}/search").mock(side_effect=lambda req: httpx.Response(
        200, json={
            "q1": sx_payload([
                sx_result("https://example-a.com/article", "Article A"),
                sx_result("https://example-a.com/article?utm_source=x", "Dupe of A"),
                sx_result("https://example-b.org/report", "Report B"),
                sx_result("https://example-old.com/ancient", "Too Old", old_date),
                sx_result("http://192.168.1.7/internal", "Private"),
            ]),
            "q2": sx_payload([
                sx_result("https://example-c.net/paper.pdf", "PDF Paper"),
                sx_result("https://example-d.io/flaky", "Flaky Site"),
                sx_result("https://example-b.org/report", "Report B again"),
            ]),
            "q3": sx_payload([
                sx_result("https://example-e.com/followup", "Follow-up E"),
            ]),
        }[req.url.params["q"]]))

    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=article("Article A")))
    respx.get("https://example-b.org/report").mock(
        return_value=httpx.Response(200, html=article("Report B")))
    respx.get("https://example-c.net/paper.pdf").mock(
        return_value=httpx.Response(200, content=pdf_bytes(),
                                    headers={"content-type": "application/pdf"}))
    respx.get("https://example-d.io/flaky").mock(
        side_effect=httpx.ConnectTimeout("boom"))
    respx.get("https://example-e.com/followup").mock(
        return_value=httpx.Response(200, html=article("Follow-up E")))
    # NOTE: no routes for example-old.com (pre-filtered by date) or
    # 192.168.1.7 (SSRF guard) — a request to either would fail the test.


@respx.mock
async def test_full_run(data_dir):
    cfg = make_cfg(data_dir)
    old = (datetime.now() - timedelta(days=700)).isoformat()
    mock_internet(old)

    llm = FakeLLM(script(gap_rounds=[
        {"state_md": "State after round 1.", "saturated": False,
         "next_queries": ["q3"]},
        {"state_md": "Final state.", "saturated": True, "next_queries": []},
    ]))
    repo = Repo(connect(cfg.db_path))
    bus = ProgressBus()
    orch = Orchestrator(lambda: cfg, repo, bus, llm_factory=lambda: llm)

    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=4,
                                    recency="6months", origin="cli"))
    await orch.execute_now(run_id)

    row = repo.get_run(run_id)
    assert row["status"] == "completed"
    assert row["stop_reason"].startswith("saturated")
    assert row["title"] == "Test Research"

    store = RunStore(cfg.research_dir / row["dir"])
    overview = store.overview_path.read_text()
    assert "[1]" in overview and "[99]" not in overview  # bogus citation stripped

    findings = repo.findings_for_run(run_id)
    assert len(findings) == 4  # A, B, PDF (r1) + E (r2)
    domains = {f["domain"] for f in findings}
    assert domains == {"example-a.com", "example-b.org", "example-c.net",
                       "example-e.com"}
    for f in findings:
        assert (store.dir / f["path"]).exists()

    sources = store.sources_path.read_text()
    for f in findings:
        assert f["url"] in sources

    assert (store.dir / "rounds/round-01.md").exists()
    assert (store.dir / "rounds/round-02.md").exists()

    meta = store.read_meta()
    assert meta["status"] == "completed"
    assert len(meta["followups"]) == 1
    assert meta["followups"][0]["recency"] == "6months"

    further = store.further_path.read_text()
    assert "Follow-up question about X" in further

    stats = json.loads(row["stats_json"])
    assert stats["rounds"] == 2
    assert stats["sources_kept"] == 4
    assert stats["sources_skipped"] == 2  # SSRF-blocked + connect timeout

    events = store.read_events()
    assert events[-1]["type"] == "done"
    assert events[-1]["status"] == "completed"
    kinds = {e["type"] for e in events}
    assert {"plan", "round_start", "finding", "gap", "source_skipped"} <= kinds

    # keyword index populated
    hits = repo.fts_search("flibbertigibbet")
    assert hits and hits[0]["run_id"] == run_id



@respx.mock
async def test_cancellation_keeps_round1_findings(data_dir):
    cfg = make_cfg(data_dir)

    async def search_handler(req):
        q = req.url.params["q"]
        if q == "q3":  # round 2 hangs so cancellation lands deterministically
            await asyncio.sleep(30)
            return httpx.Response(200, json=sx_payload([]))
        return httpx.Response(200, json={
            "q1": sx_payload([sx_result("https://example-a.com/article", "A")]),
            "q2": sx_payload([sx_result("https://example-b.org/report", "B")]),
        }[q])

    respx.get(f"{SX}/search").mock(side_effect=search_handler)
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=article("Article A")))
    respx.get("https://example-b.org/report").mock(
        return_value=httpx.Response(200, html=article("Report B")))

    llm = FakeLLM(script(gap_rounds=[
        {"state_md": "s1", "saturated": False, "next_queries": ["q3"]},
    ]))
    repo = Repo(connect(cfg.db_path))
    bus = ProgressBus()
    orch = Orchestrator(lambda: cfg, repo, bus, llm_factory=lambda: llm)

    run_id = orch.enqueue(RunParams(query="cancel me", depth=4,
                                    recency="all", origin="cli"))
    row = repo.get_run(run_id)
    store = RunStore(cfg.research_dir / row["dir"])
    _replay, queue = bus.subscribe(run_id, store)

    task = asyncio.create_task(orch.execute_now(run_id))
    while True:
        e = await asyncio.wait_for(queue.get(), 15)
        if e["type"] == "round_start" and e.get("round") == 2:
            break
    await asyncio.sleep(0.1)  # let round-2 search get in flight
    assert orch.cancel(run_id)
    with pytest.raises(asyncio.CancelledError):
        await task

    row = repo.get_run(run_id)
    assert row["status"] == "cancelled"
    findings = repo.findings_for_run(run_id)
    assert len(findings) == 2  # round-1 work survives
    for f in findings:
        assert (store.dir / f["path"]).exists()
    events = store.read_events()
    assert events[-1]["type"] == "done"
    assert events[-1]["status"] == "cancelled"
