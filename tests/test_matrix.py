"""The comparison-matrix action: a post-hoc pass over stored findings."""
from __future__ import annotations

import re

import httpx
import pytest
import respx

from app.db import Repo, connect
from app.models import MatrixOut
from app.research.matrix import fit_to_budget, render_matrix_md
from app.research.notes import Finding
from app.research.orchestrator import Orchestrator
from app.research.progress import ProgressBus
from app.research.pipeline import Pipeline
from app.models import RunParams
from tests.fake_llm import FakeLLM
from tests.test_pipeline_e2e import SX, article, make_cfg, script, sx_payload, sx_result


def _out(**kw) -> dict:
    base = {
        "applicable": True, "reason": "",
        "entities": ["MLX", "llama.cpp"],
        "dimensions": ["Decode speed", "Model formats"],
        "cells": [
            {"entity": "MLX", "dimension": "Decode speed",
             "value": "41 tok/s", "sources": [1], "conflict": False},
            {"entity": "llama.cpp", "dimension": "Decode speed",
             "value": "28 tok/s", "sources": [1], "conflict": True},
            {"entity": "MLX", "dimension": "Model formats",
             "value": "MLX only", "sources": [1], "conflict": False},
        ],
        "caveats_md": "Speeds vary with quantisation.",
    }
    base.update(kw)
    return base


# ---- rendering ------------------------------------------------------------------

def test_render_puts_dimensions_in_rows_and_marks_gaps():
    md = render_matrix_md(MatrixOut.model_validate(_out()), title="Runtimes")
    lines = md.splitlines()
    assert lines[0] == "# Runtimes — comparison"
    assert lines[2] == "| | MLX | llama.cpp |"
    # the unfilled llama.cpp/Model formats cell shows a gap, not a guess
    formats = next(l for l in lines if l.startswith("| **Model formats**"))
    assert formats.endswith("| — |")
    assert "3 of 4 cells filled" in md


def test_render_flags_conflicts_and_keeps_citations():
    md = render_matrix_md(MatrixOut.model_validate(_out()), title="T")
    assert "28 tok/s † [1]" in md
    assert "† sources disagree" in md
    assert "## Disagreements and gaps" in md


def test_render_escapes_pipes_that_would_break_the_table():
    out = MatrixOut.model_validate(_out(cells=[
        {"entity": "MLX", "dimension": "Decode speed",
         "value": "fast | sometimes\nslow", "sources": [], "conflict": False}]))
    md = render_matrix_md(out, title="T")
    row = next(l for l in md.splitlines() if l.startswith("| **Decode speed**"))
    # 3 columns (dimension + 2 entities) => 4 unescaped pipes. An unescaped
    # pipe inside a value would silently add a column and shift the row.
    assert len(re.findall(r"(?<!\\)\|", row)) == 4
    assert "\\|" in row                            # the value's pipe was escaped
    assert "\n" not in row                          # and its newline flattened


def test_render_reports_dropped_sources_rather_than_hiding_them():
    md = render_matrix_md(MatrixOut.model_validate(_out()), title="T", dropped=7)
    assert "7 lower-scoring source(s) did not fit" in md


# ---- budget ---------------------------------------------------------------------

def test_fit_to_budget_keeps_the_strongest_and_restores_citation_order():
    many = [Finding(idx=i, url=f"https://a.com/{i}", title=f"T{i}",
                    domain="a.com", published=None, relevance=(i % 10),
                    summary="s", notes_md="word " * 2000)
            for i in range(1, 61)]
    kept, dropped = fit_to_budget(many)
    assert dropped > 0 and kept                      # 60 huge sources cannot fit
    assert len(kept) + dropped == len(many)
    assert [f.idx for f in kept] == sorted(f.idx for f in kept)   # citation order
    assert min(f.relevance for f in kept) >= 5       # weakest were the ones cut


def test_fit_to_budget_always_keeps_at_least_one():
    huge = [Finding(idx=1, url="https://a.com/1", title="T", domain="a.com",
                    published=None, relevance=9, summary="s",
                    notes_md="word " * 200_000)]
    kept, dropped = fit_to_budget(huge)
    assert len(kept) == 1 and dropped == 0


# ---- the action end to end ------------------------------------------------------

async def _completed_run(cfg, matrix_payload):
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result("https://example-a.com/article", "Article A")])))
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=article("Article A")))
    s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    s["matrix"] = [matrix_payload]
    llm = FakeLLM(s)
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(), llm_factory=lambda: llm)
    run_id = orch.enqueue(RunParams(query="MLX vs llama.cpp", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)
    assert repo.get_run(run_id)["status"] == "completed"
    return repo, orch, run_id, llm


@respx.mock
async def test_matrix_action_writes_a_table_without_re_searching(data_dir):
    cfg = make_cfg(data_dir)
    repo, orch, run_id, llm = await _completed_run(cfg, _out())
    searches_before = llm.calls["notes"]

    pipeline = Pipeline(cfg, repo, ProgressBus(), llm_factory=lambda: llm)
    await pipeline.build_matrix(run_id)

    md = (cfg.research_dir / run_id / "matrix.md").read_text()
    assert md.startswith("# ")
    assert "| | MLX | llama.cpp |" in md
    assert llm.calls["notes"] == searches_before      # nothing was re-read
    assert llm.calls["matrix"] == 1
    import json
    meta = json.loads((cfg.research_dir / run_id / "meta.json").read_text())
    assert meta.get("matrix_built_at")


@respx.mock
async def test_matrix_refuses_when_the_run_is_not_a_comparison(data_dir):
    """Most runs are not comparisons. Saying so beats inventing two columns."""
    cfg = make_cfg(data_dir)
    repo, orch, run_id, llm = await _completed_run(cfg, _out(
        applicable=False, reason="This is a step-by-step repair guide.",
        entities=[], dimensions=[], cells=[]))

    pipeline = Pipeline(cfg, repo, ProgressBus(), llm_factory=lambda: llm)
    with pytest.raises(RuntimeError, match="repair guide"):
        await pipeline.build_matrix(run_id)
    assert not (cfg.research_dir / run_id / "matrix.md").exists()


@respx.mock
async def test_matrix_refuses_a_single_entity_even_if_marked_applicable(data_dir):
    """A one-column table is not a comparison, whatever the model claims."""
    cfg = make_cfg(data_dir)
    repo, orch, run_id, llm = await _completed_run(
        cfg, _out(entities=["MLX"], cells=[]))
    pipeline = Pipeline(cfg, repo, ProgressBus(), llm_factory=lambda: llm)
    with pytest.raises(RuntimeError):
        await pipeline.build_matrix(run_id)
    assert not (cfg.research_dir / run_id / "matrix.md").exists()


@respx.mock
async def test_matrix_strips_citations_beyond_the_bibliography(data_dir):
    cfg = make_cfg(data_dir)
    repo, orch, run_id, llm = await _completed_run(cfg, _out(cells=[
        {"entity": "MLX", "dimension": "Decode speed",
         "value": "41 tok/s", "sources": [1, 99], "conflict": False}]))
    pipeline = Pipeline(cfg, repo, ProgressBus(), llm_factory=lambda: llm)
    await pipeline.build_matrix(run_id)
    md = (cfg.research_dir / run_id / "matrix.md").read_text()
    assert "[1]" in md and "[99]" not in md


@respx.mock
async def test_orchestrator_guards_the_matrix_action(data_dir):
    cfg = make_cfg(data_dir)
    repo, orch, run_id, llm = await _completed_run(cfg, _out())
    assert orch.start_matrix("nonexistent") is False
    assert orch.start_matrix(run_id) is True
    assert orch.start_matrix(run_id) is False        # already in flight
    task, _ = orch.active[run_id]
    await task
    assert (cfg.research_dir / run_id / "matrix.md").exists()
    assert run_id not in orch.active


def test_cells_cap_their_citations():
    """A live run put twelve markers in one cell — unreadable, and no more
    credible than three."""
    out = MatrixOut.model_validate(_out(cells=[
        {"entity": "MLX", "dimension": "Decode speed", "value": "fast",
         "sources": [2, 3, 11, 15, 20, 24, 28], "conflict": False}]))
    row = next(l for l in render_matrix_md(out, title="T").splitlines()
               if l.startswith("| **Decode speed**"))
    assert "[2][3][11] +4" in row
    assert "[15]" not in row


def test_dropped_footnote_reads_cleanly():
    md = render_matrix_md(MatrixOut.model_validate(_out()), title="T", dropped=18)
    assert "Built from the highest-scoring sources; 18 lower-scoring" in md


# ---- knowing when it finished ---------------------------------------------------

def _client(data_dir, monkeypatch):
    from fastapi.testclient import TestClient
    from app.config import load_settings
    from app.web.server import create_app
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    app = create_app(enable_worker=False, enable_bot=False)
    return TestClient(app), load_settings(str(data_dir))


def _seed(cfg, kind="research", status="completed"):
    from app.research.storage import RunStore
    repo = Repo(connect(cfg.db_path))
    store = RunStore.create(cfg.research_dir, f"a {kind} run")
    repo.create_run(run_id=store.run_id, query="q", depth=3, recency="all",
                    dir=store.run_id, origin="web", status=status, kind=kind)
    return repo, store


def test_matrix_status_reloads_the_page_once_the_table_exists(data_dir, monkeypatch):
    """'Refresh in a minute' is not an answer — the button reports itself."""
    client, cfg = _client(data_dir, monkeypatch)
    _repo, store = _seed(cfg)
    with client:
        still = client.get(f"/runs/{store.run_id}/matrix-status")
        assert still.status_code == 200
        assert "HX-Refresh" not in still.headers          # nothing to show yet
        assert "does not weigh two or more" in still.text  # job not running

        store.write_matrix("# T — comparison\n")
        done = client.get(f"/runs/{store.run_id}/matrix-status")
        assert done.headers.get("HX-Refresh") == "true"
        assert "Comparison ready" in done.text


def test_a_non_comparison_says_so_instead_of_polling_forever(data_dir, monkeypatch):
    """The build can legitimately produce nothing; that must end the poll."""
    client, cfg = _client(data_dir, monkeypatch)
    _repo, store = _seed(cfg)
    with client:
        r = client.get(f"/runs/{store.run_id}/matrix-status")
    assert "hx-trigger" not in r.text                     # polling stopped
    assert "Log tab" in r.text                            # and says where to look


def test_library_badges_name_the_run_kind(data_dir, monkeypatch):
    client, cfg = _client(data_dir, monkeypatch)
    for kind in ("research", "brief", "verify"):
        _seed(cfg, kind=kind)
    _repo, with_matrix = _seed(cfg)
    with_matrix.write_matrix("# T — comparison\n")
    with client:
        body = client.get("/library").text
    assert 'class="kind kind-research"' in body
    assert 'class="kind kind-brief"' in body
    assert 'class="kind kind-verify"' in body and "claim check" in body
    assert 'class="kind kind-matrix"' in body             # artifact, not a kind
