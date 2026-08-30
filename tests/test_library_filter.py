"""Library: filter by result type, and blend both search engines."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import load_settings
from app.db import Repo, connect
from app.research.storage import RunStore
from app.web.routes_library import _rank_points
from app.web.server import create_app


@pytest.fixture
def lib(data_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    app = create_app(enable_worker=False, enable_bot=False)
    cfg = load_settings(str(data_dir))
    repo = Repo(connect(cfg.db_path))

    def seed(kind="research", *, title, matrix=False, body=""):
        store = RunStore.create(cfg.research_dir, title)
        repo.create_run(run_id=store.run_id, query=title, depth=3,
                        recency="all", dir=store.run_id, origin="web",
                        status="completed", kind=kind)
        repo.update_run(store.run_id, title=title)
        if matrix:
            store.write_matrix(f"# {title} — comparison\n")
        if body:
            repo.fts_add(store.run_id, "overview", title, body)
        return store.run_id

    return app, cfg, repo, seed


# ---- the kind filter --------------------------------------------------------

def test_browsing_filters_by_result_type(lib):
    app, _cfg, _repo, seed = lib
    seed("research", title="Plain research")
    seed("research", title="Compared research", matrix=True)
    seed("brief", title="A brief")
    seed("verify", title="A claim check")

    with TestClient(app) as c:
        def titles(kind=""):
            body = c.get("/library", params={"kind": kind}).text
            return {t for t in ("Plain research", "Compared research",
                                "A brief", "A claim check") if t in body}

        assert titles() == {"Plain research", "Compared research",
                            "A brief", "A claim check"}
        assert titles("research") == {"Plain research", "Compared research"}
        assert titles("brief") == {"A brief"}
        assert titles("verify") == {"A claim check"}
        # the one filter that is not a kind: research narrowed to those with
        # a comparison table on disk
        assert titles("research+matrix") == {"Compared research"}


def test_an_unknown_filter_value_falls_back_to_all(lib):
    app, _cfg, _repo, seed = lib
    seed("brief", title="A brief")
    with TestClient(app) as c:
        body = c.get("/library", params={"kind": "../etc/passwd"}).text
    assert "A brief" in body


def test_searching_respects_the_filter_too(lib):
    app, _cfg, _repo, seed = lib
    seed("research", title="Widget analysis", body="widgets everywhere")
    seed("brief", title="Widget weekly", body="widgets everywhere")
    with TestClient(app) as c:
        both = c.get("/library", params={"q": "widgets"}).text
        only = c.get("/library", params={"q": "widgets", "kind": "brief"}).text
    assert "Widget analysis" in both and "Widget weekly" in both
    assert "Widget weekly" in only and "Widget analysis" not in only


# ---- blended search ---------------------------------------------------------

def test_rank_fusion_prefers_agreement_over_a_lone_first_place():
    """Blending raw scores would be meaningless — bm25 and cosine are not the
    same quantity — so only rank position counts."""
    lone_first = _rank_points(0)
    second_twice = _rank_points(1) + _rank_points(1)
    assert second_twice > lone_first
    assert _rank_points(0) > _rank_points(1) > _rank_points(9)


def test_both_engines_run_on_every_search(lib, monkeypatch):
    app, _cfg, _repo, seed = lib
    keyword_only = seed("research", title="Exact phrase match",
                        body="tokamak containment")
    semantic_only = seed("research", title="Paraphrase match", body="unrelated")

    class FakeRag:
        def __init__(self):
            self.asked = []

        async def semantic_search(self, q, limit=20):
            self.asked.append(q)
            return [{"run_id": semantic_only, "text": "fusion reactor vessel",
                     "score": 0.8, "kind": "overview", "title": "Paraphrase match"}]

    rag = FakeRag()
    with TestClient(app) as c:
        app.state.rag = rag
        body = c.get("/library", params={"q": "tokamak containment"}).text

    assert rag.asked == ["tokamak containment"]      # semantic ran unasked
    assert "Exact phrase match" in body              # keyword hit survived
    assert "Paraphrase match" in body                # semantic hit too
    assert "keyword only" not in body


def test_a_hit_found_by_both_engines_is_labelled(lib):
    app, _cfg, _repo, seed = lib
    shared = seed("research", title="Found twice", body="tokamak containment")

    class FakeRag:
        async def semantic_search(self, q, limit=20):
            return [{"run_id": shared, "text": "same run, other engine",
                     "score": 0.9, "kind": "overview", "title": "Found twice"}]

    with TestClient(app) as c:
        app.state.rag = FakeRag()
        body = c.get("/library", params={"q": "tokamak"}).text
    assert "keyword + meaning" in body
    assert body.count("Found twice") >= 1            # merged, not duplicated
    assert body.count('class="hit-card') == 1


def test_a_broken_vector_index_does_not_break_search(lib):
    """Keyword results must survive the semantic half failing."""
    app, _cfg, _repo, seed = lib
    seed("research", title="Still findable", body="tokamak containment")

    class BrokenRag:
        async def semantic_search(self, *a, **k):
            raise RuntimeError("chroma is down")

    with TestClient(app) as c:
        app.state.rag = BrokenRag()
        body = c.get("/library", params={"q": "tokamak"}).text
    assert "Still findable" in body


def test_no_vector_index_says_so_rather_than_failing_quietly(lib):
    app, _cfg, _repo, seed = lib
    seed("research", title="Still findable", body="tokamak containment")
    with TestClient(app) as c:
        app.state.rag = None
        body = c.get("/library", params={"q": "tokamak"}).text
    assert "Still findable" in body
    assert "keyword only" in body


# ---- the two lists are one template -----------------------------------------

def test_the_new_tab_and_library_render_runs_identically(lib):
    """They were separate copies of the same markup, so they drifted — the
    Library grew kind badges and the New tab did not."""
    app, _cfg, _repo, seed = lib
    seed("brief", title="A brief run")
    seed("research", title="Compared run", matrix=True)
    with TestClient(app) as c:
        home = c.get("/partials/recent-runs").text
        library = c.get("/library").text
    for body in (home, library):
        assert 'class="run-flags"' in body          # status stacked over kind
        assert 'class="kind kind-brief"' in body
        assert 'class="kind kind-matrix"' in body
    # the heading belongs to the New tab only
    assert "Research runs" in home
    assert "Research runs" not in library
