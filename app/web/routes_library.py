"""Library: browse all runs, filter by kind, search keyword + semantic together.

Search used to make you choose keyword or semantic up front, which is a
choice about retrieval mechanics rather than about what you want. Both run on
every search now and the two rankings are fused, so an exact phrase and a
paraphrase both land.
"""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request

from app.web.markdown import highlight_snippet

log = logging.getLogger(__name__)
router = APIRouter()

# What the kind dropdown offers: (value, label). "research+matrix" is not a
# run kind — it narrows research runs to the ones carrying a comparison table.
KIND_FILTERS = (
    ("", "All types"),
    ("research", "Research"),
    ("research+matrix", "Research with a comparison"),
    ("brief", "Briefs"),
    ("verify", "Claim checks"),
)
_VALID_KINDS = {v for v, _l in KIND_FILTERS}

# Reciprocal rank fusion. Blending raw scores would be wrong — sqlite's bm25
# and cosine similarity are not the same quantity or even the same direction —
# so only the ORDER each engine produced is used. k dampens the top-rank
# advantage so a strong second opinion can outrank a lone first place.
_RRF_K = 60
_SEARCH_LIMIT = 30


def _rank_points(position: int) -> float:
    return 1.0 / (_RRF_K + position)


def _kind_of(row) -> str:
    try:
        return row["kind"] or "research"
    except (KeyError, IndexError):
        return "research"


def _matches(row, kind_filter: str, has_matrix: bool) -> bool:
    if not kind_filter:
        return True
    if kind_filter == "research+matrix":
        return _kind_of(row) == "research" and has_matrix
    return _kind_of(row) == kind_filter


@router.get("/library")
async def library(request: Request, q: str = "", kind: str = ""):
    repo = request.app.state.repo
    rag = request.app.state.rag
    templates = request.app.state.templates
    research_dir = request.app.state.cfg_loader().research_dir

    kind = kind if kind in _VALID_KINDS else ""

    def has_matrix(row) -> bool:
        return (research_dir / row["dir"] / "matrix.md").exists()

    ctx: dict = {"nav": "library", "q": q, "kind": kind,
                 "kind_filters": KIND_FILTERS,
                 "rag_available": rag is not None,
                 "results": None, "runs": None}

    if not q.strip():
        rows = [r for r in repo.list_runs(limit=400)
                if _matches(r, kind, has_matrix(r))][:200]
        ctx["runs"] = [{"row": r,
                        "stats": json.loads(r["stats_json"]) if r["stats_json"] else {},
                        # A matrix is an artifact on a research run, not a
                        # fourth kind, so it is a separate marker.
                        "has_matrix": has_matrix(r)}
                       for r in rows]
        return templates.TemplateResponse(request, "library.html", ctx)

    # ---- both engines, then fuse ------------------------------------------
    merged: dict[str, dict] = {}

    def add(run_id: str, points: float, **fields) -> None:
        entry = merged.setdefault(run_id, {"points": 0.0, "engines": set()})
        entry["points"] += points
        # First writer wins on display fields: whichever engine ranked it
        # higher got here first, and its snippet is the better one to show.
        for key, value in fields.items():
            entry.setdefault(key, value)

    for i, hit in enumerate(repo.fts_search(q, limit=_SEARCH_LIMIT)):
        add(hit["run_id"], _rank_points(i), kind=hit["kind"],
            title=hit["title"], snippet=highlight_snippet(hit["snip"]))
        merged[hit["run_id"]]["engines"].add("keyword")

    if rag is not None:
        try:
            for i, hit in enumerate(await rag.semantic_search(q, limit=_SEARCH_LIMIT)):
                add(hit["run_id"], _rank_points(i), kind=hit.get("kind", ""),
                    title=hit.get("title", ""), snippet=hit.get("text", ""),
                    score=hit.get("score"))
                merged[hit["run_id"]]["engines"].add("semantic")
        except Exception:
            # A vector-index problem must not take keyword search down with it.
            log.exception("semantic half of the search failed; keyword only")

    results = []
    for run_id, entry in sorted(merged.items(), key=lambda kv: -kv[1]["points"]):
        row = repo.get_run(run_id)
        if row is None or not _matches(row, kind, has_matrix(row)):
            continue
        results.append({"run": row, "snippet": entry.get("snippet", ""),
                        "kind": entry.get("kind", ""),
                        "title": entry.get("title", ""),
                        "score": entry.get("score"),
                        "both": len(entry["engines"]) > 1,
                        "has_matrix": has_matrix(row)})
    ctx["results"] = results
    return templates.TemplateResponse(request, "library.html", ctx)
