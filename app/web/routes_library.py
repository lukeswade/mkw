"""Library: browse all runs, keyword search (FTS5), semantic search (Chroma)."""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Request

from app.web.markdown import highlight_snippet

log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/library")
async def library(request: Request, q: str = "", mode: str = "keyword"):
    repo = request.app.state.repo
    rag = request.app.state.rag
    templates = request.app.state.templates

    ctx: dict = {"nav": "library", "q": q, "mode": mode,
                 "rag_available": rag is not None, "results": None, "runs": None}

    if not q.strip():
        rows = repo.list_runs(limit=200)
        ctx["runs"] = [{"row": r,
                        "stats": json.loads(r["stats_json"]) if r["stats_json"] else {}}
                       for r in rows]
    elif mode == "semantic" and rag is not None:
        hits = await rag.semantic_search(q, limit=20)
        results = []
        for h in hits:
            row = repo.get_run(h["run_id"])
            if row is not None:
                results.append({"run": row, "snippet": h["text"],
                                "score": h["score"], "kind": h["kind"],
                                "title": h.get("title") or row["title"]})
        ctx["results"] = results
    else:
        ctx["mode"] = "keyword"
        results = []
        for hit in repo.fts_search(q, limit=30):
            row = repo.get_run(hit["run_id"])
            if row is not None:
                results.append({"run": row,
                                "snippet": highlight_snippet(hit["snip"]),
                                "kind": hit["kind"], "title": hit["title"]})
        ctx["results"] = results

    return templates.TemplateResponse(request, "library.html", ctx)
