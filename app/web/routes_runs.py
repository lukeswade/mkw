"""Run lifecycle routes: create, view (live + terminal), SSE, cancel, files."""
from __future__ import annotations

import asyncio
import shutil
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from pydantic import ValidationError

from app.models import RunParams
from app.research.progress import format_event
from app.research.storage import SERVABLE_RE, RunStore
from app.web.markdown import render, render_overview

log = logging.getLogger(__name__)
router = APIRouter()

ACTIVE_STATUSES = ("queued", "running")


def _tpl(request: Request):
    return request.app.state.templates


def _store(request: Request, row) -> RunStore:
    cfg = request.app.state.cfg_loader()
    return RunStore(cfg.research_dir / row["dir"])


def _row_or_404(request: Request, run_id: str):
    row = request.app.state.repo.get_run(run_id)
    if row is None:
        raise HTTPException(404, "run not found")
    return row


def _runs_context(request: Request, limit: int = 20) -> dict:
    repo = request.app.state.repo
    rows = repo.list_runs(limit=limit)
    runs = []
    for r in rows:
        stats = json.loads(r["stats_json"]) if r["stats_json"] else {}
        runs.append({"row": r, "stats": stats})
    return {"runs": runs}


@router.get("/")
async def index(request: Request):
    ctx = _runs_context(request)
    ctx["nav"] = "home"
    return _tpl(request).TemplateResponse(request, "index.html", ctx)


@router.get("/partials/recent-runs")
async def recent_runs_partial(request: Request):
    return _tpl(request).TemplateResponse(
        request, "partials/runs_list.html", _runs_context(request))


@router.post("/runs")
async def create_run(request: Request, query: str = Form(...),
                     depth: int = Form(3), recency: str = Form("all"),
                     parent_run_id: str = Form("")):
    try:
        params = RunParams(query=query, depth=depth, recency=recency,
                           origin="web", parent_run_id=parent_run_id or None)
    except ValidationError as e:
        ctx = _runs_context(request)
        ctx["nav"] = "home"
        ctx["error"] = "; ".join(err["msg"] for err in e.errors())
        ctx["prefill"] = {"query": query, "depth": depth, "recency": recency}
        return _tpl(request).TemplateResponse(request, "index.html", ctx,
                                              status_code=422)
    run_id = request.app.state.orch.enqueue(params)
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


@router.get("/runs/{run_id}")
async def run_page(request: Request, run_id: str):
    row = _row_or_404(request, run_id)
    repo = request.app.state.repo
    store = _store(request, row)
    meta = store.read_meta()

    ctx: dict = {
        "nav": "home",
        "row": row,
        "run_id": run_id,
        "meta": meta,
        "stats": json.loads(row["stats_json"]) if row["stats_json"] else None,
        "parent": repo.get_run(row["parent_run_id"]) if row["parent_run_id"] else None,
    }

    if row["status"] in ACTIVE_STATUSES:
        ctx["queue_position"] = request.app.state.orch.queue_position(run_id)
        return _tpl(request).TemplateResponse(request, "run_active.html", ctx)

    findings = repo.findings_for_run(run_id)
    overview_md = (store.overview_path.read_text(encoding="utf-8")
                   if store.overview_path.exists() else "")
    finding_cards = []
    for f in findings:
        body = ""
        p = store.dir / f["path"]
        if p.is_file():
            body = render(p.read_text(encoding="utf-8"))
        finding_cards.append({"row": f, "html": body})

    log_lines = [line for e in store.read_events()
                 if (line := format_event(e))]
    links = repo.links_for_run(run_id)
    related = []
    for l in links:
        if l["kind"] == "followup" and l["src_run_id"] == run_id:
            related.append(("follow-up", l["dst_run_id"], l["dst_title"]))
        elif l["kind"] == "followup":
            related.append(("follows up on", l["src_run_id"], l["src_title"]))
        elif l["src_run_id"] == run_id:
            related.append(("related", l["dst_run_id"], l["dst_title"]))
        else:
            related.append(("related", l["src_run_id"], l["src_title"]))

    ctx.update({
        "overview_html": render_overview(overview_md, len(findings)),
        "findings": findings,
        "finding_cards": finding_cards,
        "followups": meta.get("followups", []),
        "log_text": "\n".join(log_lines),
        "related": related,
        "files": [n for n in ("overview.md", "further-research.md", "sources.md",
                              "meta.json", "events.jsonl")
                  if (store.dir / n).exists()],
    })
    return _tpl(request).TemplateResponse(request, "run.html", ctx)


@router.get("/runs/{run_id}/events")
async def run_events(request: Request, run_id: str):
    row = _row_or_404(request, run_id)
    bus = request.app.state.bus
    store = _store(request, row)
    try:
        after = int(request.headers.get("last-event-id", "0") or 0)
    except ValueError:
        after = 0
    replay, queue = bus.subscribe(run_id, store, after_seq=after)

    def sse(e: dict) -> str:
        return f"id: {e.get('seq', 0)}\ndata: {json.dumps(e)}\n\n"

    async def gen():
        try:
            for e in replay:
                yield sse(e)
            if queue is None:
                return  # terminal run: replay (ending in 'done') is everything
            while True:
                try:
                    e = await asyncio.wait_for(queue.get(), 15.0)
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    if await request.is_disconnected():
                        return
                    continue
                yield sse(e)
                if e.get("type") == "done":
                    return
        finally:
            if queue is not None:
                bus.unsubscribe(run_id, queue)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@router.post("/runs/{run_id}/cancel")
async def cancel_run(request: Request, run_id: str):
    _row_or_404(request, run_id)
    request.app.state.orch.cancel(run_id)
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


@router.post("/runs/{run_id}/evergreen")
async def toggle_evergreen(request: Request, run_id: str):
    repo = request.app.state.repo
    row = repo.get_run(run_id)
    if not row:
        return Response("Not found", status_code=404)
    # Toggle evergreen boolean
    new_status = not bool(row["evergreen"])
    repo.update_run(run_id, evergreen=new_status)
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


@router.post("/runs/{run_id}/retry")
async def retry_run(request: Request, run_id: str):
    row = _row_or_404(request, run_id)
    params = RunParams(query=row["query"], depth=row["depth"],
                       recency=row["recency"], origin="web",
                       parent_run_id=run_id)
    new_id = request.app.state.orch.enqueue(params)
    return RedirectResponse(f"/runs/{new_id}", status_code=303)


@router.get("/runs/{run_id}/file/{name:path}")
async def run_file(request: Request, run_id: str, name: str,
                   download: bool = False):
    row = _row_or_404(request, run_id)
    if not SERVABLE_RE.match(name):
        raise HTTPException(404)
    cfg = request.app.state.cfg_loader()
    run_dir = (cfg.research_dir / row["dir"]).resolve()
    target = (run_dir / name).resolve()
    if not target.is_relative_to(run_dir) or not target.is_file():
        raise HTTPException(404)
    if name.endswith(".json"):
        media = "application/json"
    elif name.endswith(".jsonl"):
        media = "application/x-ndjson"
    else:
        media = "text/markdown; charset=utf-8"
    headers = {}
    if download:
        headers["Content-Disposition"] = (
            f'attachment; filename="{run_id}_{Path(name).name}"')
    return FileResponse(target, media_type=media, headers=headers)

@router.delete("/runs/{run_id}")
async def delete_run(request: Request, run_id: str):
    row = _row_or_404(request, run_id)
    cfg = request.app.state.cfg_loader()
    run_dir = (cfg.research_dir / row["dir"]).resolve()
    
    # Delete from DB
    request.app.state.repo.delete_run(run_id)
    
    # Delete files
    if run_dir.exists() and run_dir.is_dir():
        shutil.rmtree(run_dir)
        
    return Response(status_code=200, headers={"HX-Redirect": "/library"})
