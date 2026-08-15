"""Knowledge-layer integration: real local embeddings + Chroma on tmp dirs.
Loads the baked bge-small model once (class-level cache)."""
from __future__ import annotations

import json

import pytest

pytest.importorskip("sentence_transformers", reason="vector layer deps not installed")
pytest.importorskip("chromadb", reason="vector layer deps not installed")

from app.config import Settings
from app.db import Repo, connect
from app.rag.service import RagService, _parse_finding_md
from app.research.storage import RunStore
from tests.fake_llm import FakeLLM

BATTERY_TEXT = (
    "Solid-state batteries replace the liquid electrolyte with a solid "
    "sulfide or oxide electrolyte. Manufacturing yield remains the core "
    "blocker: sulfide electrolytes react with moisture, forcing dry-room "
    "production. QuantumScape and Toyota lead on oxide and sulfide routes "
    "respectively, targeting energy density above 400 Wh/kg."
)


def seed_run(cfg, repo, query: str, title: str, body: str) -> str:
    store = RunStore.create(cfg.research_dir, query)
    run_id = store.run_id
    repo.create_run(run_id=run_id, query=query, depth=2, recency="all",
                    dir=run_id, origin="cli", status="queued")
    repo.update_run(run_id, status="completed", title=title)
    store.write_meta({"run_id": run_id, "query": query, "depth": 2,
                      "recency": "all", "origin": "cli", "title": title,
                      "status": "completed"})
    store.write_overview(f"# {title}\n\n## TL;DR\n\n{body}\n\n## Detail\n\n{body}")
    store.write_finding(1, "Battery Source", (
        f"# [1] Battery Source\n\n- **URL:** https://example.com/{run_id}\n"
        f"- **Domain:** example.com\n- **Published:** 2026-05-01\n"
        f"- **Relevance:** 8/10\n- **Found via:** {query}\n\n"
        f"**Summary:** About batteries.\n\n## Notes\n\n{body}\n"))
    repo.add_finding(run_id=run_id, idx=1, url=f"https://example.com/{run_id}",
                     title="Battery Source", domain="example.com",
                     published_date="2026-05-01", relevance=8,
                     path="findings/001_battery-source.md", summary="About batteries.")
    return run_id


@pytest.fixture
def cfg(data_dir):
    c = Settings(data_dir=str(data_dir))
    c.ensure_dirs()
    return c


async def test_full_knowledge_cycle(cfg):
    repo = Repo(connect(cfg.db_path))
    svc = RagService(cfg, llm_factory=lambda: FakeLLM(
        {"ask": ["Sulfide electrolytes have moisture problems "
                 "[run: Solid State Batteries]."]}))

    run_a = seed_run(cfg, repo, "solid state battery manufacturing",
                     "Solid State Batteries", BATTERY_TEXT)
    n = await svc.index_run(repo, run_a)
    assert n >= 2  # overview + finding chunks

    hits = await svc.semantic_search("sulfide electrolyte manufacturing yield")
    assert hits and hits[0]["run_id"] == run_a
    assert hits[0]["score"] > 0.5

    # an unrelated query scores clearly lower than an on-topic one
    off_topic = await svc.semantic_search("medieval French poetry")
    assert not off_topic or off_topic[0]["score"] < hits[0]["score"]

    # second, similar run links back to the first
    run_b = seed_run(cfg, repo, "solid state battery production yields",
                     "Battery Production Yields", BATTERY_TEXT)
    await svc.index_run(repo, run_b)
    links = repo.links_for_run(run_b)
    assert any(l["kind"] == "similar" and
               {l["src_run_id"], l["dst_run_id"]} == {run_a, run_b}
               for l in links)

    # prior knowledge feeds the planner
    block, related = await svc.prior_knowledge("battery electrolyte yields")
    assert "sulfide" in block.lower()
    assert related and related[0][1] > 0.5

    # ask with citations
    result = await svc.ask("what blocks solid state battery manufacturing?", repo)
    assert "sulfide" in result["answer"].lower()
    assert result["sources"] and result["sources"][0]["run_id"] in (run_a, run_b)

    # honest no-answer path
    svc_empty = RagService(Settings(data_dir=str(cfg.data_path / "empty")),
                           llm_factory=lambda: FakeLLM({}))
    svc_empty.cfg.ensure_dirs()
    empty = await svc_empty.ask("anything", Repo(connect(svc_empty.cfg.db_path)))
    assert "doesn't cover" in empty["answer"]


async def test_reindex_restores_db_from_disk(cfg):
    repo = Repo(connect(cfg.db_path))
    svc = RagService(cfg, llm_factory=lambda: FakeLLM({}))
    run_id = seed_run(cfg, repo, "battery reindex test", "Reindex Test",
                      BATTERY_TEXT)
    await svc.index_run(repo, run_id)

    # simulate database loss (cascade wipes findings)
    repo.conn.execute("DELETE FROM runs")
    repo.conn.execute("DELETE FROM fts")
    repo.conn.commit()
    assert repo.get_run(run_id) is None

    n = await svc.reindex_all(repo)
    assert n == 1
    row = repo.get_run(run_id)
    assert row is not None and row["title"] == "Reindex Test"
    findings = repo.findings_for_run(run_id)
    assert len(findings) == 1
    assert findings[0]["url"].startswith("https://example.com/")
    assert findings[0]["relevance"] == 8
    assert repo.fts_search("sulfide")
    hits = await svc.semantic_search("battery manufacturing")
    assert hits and hits[0]["run_id"] == run_id


def test_parse_finding_md():
    parsed = _parse_finding_md(
        "# [7] Some Title\n\n- **URL:** https://x.y/z\n- **Domain:** x.y\n"
        "- **Published:** unknown\n- **Relevance:** 6/10\n"
        "- **Found via:** q\n\n**Summary:** The summary.\n")
    assert parsed == {"idx": 7, "title": "Some Title", "url": "https://x.y/z",
                      "domain": "x.y", "published_date": None, "relevance": 6.0,
                      "summary": "The summary."}
    assert _parse_finding_md("no header here") is None


async def test_similar_hint_partial(cfg, monkeypatch):
    """Typing a query you've already researched should surface the run."""
    from fastapi.testclient import TestClient
    from tests.test_web import make_app

    repo = Repo(connect(cfg.db_path))
    svc = RagService(cfg, llm_factory=lambda: FakeLLM({}))
    run_id = seed_run(cfg, repo, "solid state battery manufacturing",
                      "Solid State Batteries", BATTERY_TEXT)
    await svc.index_run(repo, run_id)

    monkeypatch.setenv("DATA_DIR", str(cfg.data_path))
    app = __import__("app.web.server", fromlist=["create_app"]).create_app(
        enable_worker=False, enable_bot=False)
    with TestClient(app) as client:
        hit = client.get("/partials/similar", params={
            "query": "sulfide electrolyte manufacturing yields for batteries"})
        assert "Solid State Batteries" in hit.text

        miss = client.get("/partials/similar", params={
            "query": "medieval French poetry and its rhyme schemes"})
        assert "Solid State Batteries" not in miss.text

        short = client.get("/partials/similar", params={"query": "hi"})
        assert "similar-hint" not in short.text
