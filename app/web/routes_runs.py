"""Run lifecycle routes: create, view (live + terminal), SSE, cancel, files."""
from __future__ import annotations

import asyncio
import shutil
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import (FileResponse, HTMLResponse, RedirectResponse,
                               Response, StreamingResponse)
from pydantic import ValidationError

from app.models import RunParams
from app.research.estimate import estimate_run
from app.research.progress import format_event
from app.research.storage import SERVABLE_RE, RunStore
from app.web.export import (PdfExportError, build_run_html, interactive_html,
                            render_pdf, standalone_html)
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


def _initiator(request: Request) -> str:
    """Who pressed the button.

    Behind the Cloudflare tunnel, Access injects the authenticated user's
    email (and strips any client attempt to spoof it). A request without the
    header came over the LAN, where the label is configurable — a household
    box knows who its local user is better than we do.
    """
    email = request.headers.get("cf-access-authenticated-user-email", "").strip()
    if email:
        return email.split("@", 1)[0][:120]
    return request.app.state.cfg_loader().lan_user_label


def _related_links(repo, run_id: str,
                   exclude: str | None = None) -> list[tuple[str, str, str]]:
    """One entry per related run. The parent is excluded (the header already
    says "follows up on"), and a run linked both as follow-up and as similar
    appears once, with the more specific label winning."""
    related: dict[str, tuple[str, str, str]] = {}
    for l in repo.links_for_run(run_id):
        if l["kind"] == "followup" and l["src_run_id"] == run_id:
            entry = ("follow-up", l["dst_run_id"], l["dst_title"])
        elif l["kind"] == "followup":
            entry = ("follows up on", l["src_run_id"], l["src_title"])
        elif l["src_run_id"] == run_id:
            entry = ("related", l["dst_run_id"], l["dst_title"])
        else:
            entry = ("related", l["src_run_id"], l["src_title"])
        rid = entry[1]
        if rid == exclude or rid == run_id:
            continue
        if rid not in related or related[rid][0] == "related":
            related[rid] = entry
    return list(related.values())


def _run_header_context(request: Request, run_id: str) -> dict:
    """Context for partials/run_header.html — shared by the run page and the
    evergreen toggle, which swaps the header in place."""
    repo = request.app.state.repo
    row = repo.get_run(run_id)
    return {
        "row": row,
        "run_id": run_id,
        "parent": repo.get_run(row["parent_run_id"]) if row["parent_run_id"] else None,
        "related": _related_links(repo, run_id, exclude=row["parent_run_id"]),
    }


def _finding_cards(store: RunStore, findings) -> list[dict]:
    cards = []
    for f in findings:
        body = ""
        p = store.dir / f["path"]
        if p.is_file():
            body = render(p.read_text(encoding="utf-8"))
        cards.append({"row": f, "html": body})
    return cards


def _runs_context(request: Request, limit: int = 20) -> dict:
    repo = request.app.state.repo
    rows = repo.list_runs(limit=limit)
    runs = []
    for r in rows:
        stats = json.loads(r["stats_json"]) if r["stats_json"] else {}
        runs.append({"row": r, "stats": stats})
    return {"runs": runs}


def _index_context(request: Request, depth: int = 3) -> dict:
    cfg = request.app.state.cfg_loader()
    ctx = _runs_context(request)
    ctx.update({
        "nav": "home",
        "llm_configured": cfg.llm_is_configured,
        "provider_label": cfg.provider.label,
        "estimate": estimate_run(request.app.state.repo, depth=depth),
    })
    return ctx


@router.get("/")
async def index(request: Request):
    return _tpl(request).TemplateResponse(request, "index.html",
                                          _index_context(request))


@router.get("/partials/similar")
async def similar_partial(request: Request, query: str = ""):
    """As-you-type hint: you may have already researched this.

    A depth-5 run is half an hour of GPU time — worth one embedding lookup
    to mention the answer might already be in the library.
    """
    rag = request.app.state.rag
    repo = request.app.state.repo
    matches: list[dict] = []
    if rag is not None and len(query.strip()) >= 12:
        try:
            best: dict[str, float] = {}
            for hit in await rag.semantic_search(query.strip(), limit=8):
                rid = hit["run_id"]
                best[rid] = max(best.get(rid, 0.0), hit["score"])
            for rid, score in sorted(best.items(), key=lambda t: -t[1])[:2]:
                row = repo.get_run(rid)
                if row is not None and row["status"] == "completed" \
                        and score >= 0.62:
                    matches.append({"run": row, "score": score})
        except Exception:
            log.debug("similar lookup failed", exc_info=True)
    return _tpl(request).TemplateResponse(
        request, "partials/similar_hint.html", {"matches": matches})


@router.get("/partials/estimate")
async def estimate_partial(request: Request, depth: int = 3):
    """Live pre-flight estimate as the depth slider moves."""
    depth = max(0, min(10, depth))
    return _tpl(request).TemplateResponse(
        request, "partials/estimate.html",
        {"estimate": estimate_run(request.app.state.repo, depth=depth)})


@router.get("/partials/recent-runs")
async def recent_runs_partial(request: Request):
    return _tpl(request).TemplateResponse(
        request, "partials/runs_list.html", _runs_context(request))


@router.post("/runs")
async def create_run(request: Request, query: str = Form(...),
                     depth: int = Form(3), recency: str = Form("all"),
                     parent_run_id: str = Form("")):
    form = await request.form()
    categories = ",".join(str(c).strip() for c in form.getlist("categories")
                          if str(c).strip())[:200]
    try:
        params = RunParams(query=query, depth=depth, recency=recency,
                           origin="web", parent_run_id=parent_run_id or None,
                           created_by=_initiator(request),
                           categories=categories)
    except ValidationError as e:
        ctx = _index_context(request, depth=depth if 0 <= depth <= 10 else 3)
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
    finding_cards = _finding_cards(store, findings)

    log_lines = [line for e in store.read_events()
                 if (line := format_event(e))]
    related = _related_links(repo, run_id, exclude=row["parent_run_id"])

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
async def toggle_evergreen(request: Request, run_id: str, view: str = ""):
    repo = request.app.state.repo
    row = _row_or_404(request, run_id)
    repo.update_run(run_id, evergreen=not bool(row["evergreen"]))
    if view == "star":
        # list views swap just the star button in place
        return _tpl(request).TemplateResponse(
            request, "partials/evergreen_star.html",
            {"r": repo.get_run(run_id)})
    if request.headers.get("hx-request"):
        return _tpl(request).TemplateResponse(
            request, "partials/run_header.html", _run_header_context(request, run_id))
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


@router.post("/runs/{run_id}/retry")
async def retry_run(request: Request, run_id: str):
    row = _row_or_404(request, run_id)
    try:
        inherited = (row["categories"] or "")
    except (KeyError, IndexError):
        inherited = ""
    params = RunParams(query=row["query"], depth=row["depth"],
                       recency=row["recency"], origin="web",
                       parent_run_id=run_id, created_by=_initiator(request),
                       categories=inherited)
    new_id = request.app.state.orch.enqueue(params)
    return RedirectResponse(f"/runs/{new_id}", status_code=303)


@router.post("/runs/{run_id}/resynthesize")
async def resynthesize_run(request: Request, run_id: str):
    """Regenerate the overview from stored findings — no re-searching.

    For when the research succeeded but the final synthesis call didn't
    (leaked reasoning text, truncation, model failure)."""
    _row_or_404(request, run_id)
    started = request.app.state.orch.start_resynth(run_id)
    if started:
        return HTMLResponse(
            '<span class="resynth-note">Re-synthesizing from the stored '
            'sources — refresh this page in a few minutes.</span>')
    return HTMLResponse(
        '<span class="resynth-note">Could not start: the run is busy or '
        'has no stored findings.</span>')


@router.get("/runs/{run_id}/export.pdf")
async def export_pdf(request: Request, run_id: str):
    """The whole run — question, overview, bibliography, source notes — as
    one PDF, generated with the already-present pymupdf. Nothing to install,
    works offline, survives outside the tool."""
    row = _row_or_404(request, run_id)
    repo = request.app.state.repo
    store = _store(request, row)
    findings = repo.findings_for_run(run_id)
    overview_md = (store.overview_path.read_text(encoding="utf-8")
                   if store.overview_path.exists() else "")
    finished = (row["finished_at"] or row["created_at"] or "")[:10]
    meta_line = (f"depth {row['depth']} · {row['recency']} · "
                 f"{len(findings)} sources · {row['status']} {finished} · "
                 f"generated by Deep Research")
    html = build_run_html(
        title=row["title"] or row["query"], query=row["query"],
        meta_line=meta_line,
        overview_html=render_overview(overview_md, len(findings)),
        findings=findings, cards=_finding_cards(store, findings))
    try:
        pdf = render_pdf(html)
    except PdfExportError as e:
        raise HTTPException(500, f"PDF export failed: {e}")
    return Response(pdf, media_type="application/pdf", headers={
        "Content-Disposition": f'attachment; filename="{run_id}.pdf"'})


@router.get("/runs/{run_id}/export.html")
async def export_html(request: Request, run_id: str):
    """The whole run as one self-contained web page — overview, bibliography,
    collapsible source notes, no external assets. Opens from a file://
    double-click, survives outside the tool, shares over anything."""
    row = _row_or_404(request, run_id)
    repo = request.app.state.repo
    store = _store(request, row)
    findings = repo.findings_for_run(run_id)
    overview_md = (store.overview_path.read_text(encoding="utf-8")
                   if store.overview_path.exists() else "")
    finished = (row["finished_at"] or row["created_at"] or "")[:10]
    meta_line = (f"depth {row['depth']} · {row['recency']} · "
                 f"{len(findings)} sources · {row['status']} {finished} · "
                 f"generated by Deep Research")
    page = standalone_html(
        title=row["title"] or row["query"], query=row["query"],
        meta_line=meta_line,
        overview_html=render_overview(overview_md, len(findings)),
        findings=findings, cards=_finding_cards(store, findings))
    # octet-stream, not text/html: Cloudflare (and other RUM-injecting
    # proxies) rewrite text/html responses in transit and add a beacon
    # <script> to the file, which breaks the zero-external-assets promise
    # and phones home when the saved file is opened offline.
    return Response(page, media_type="application/octet-stream", headers={
        "Content-Disposition": f'attachment; filename="{run_id}.html"'})


@router.get("/runs/{run_id}/export-interactive.html")
async def export_interactive(request: Request, run_id: str):
    """The run as a portable mini-app in one file: tabs, client-side search,
    live citation jumps, theme toggle — still zero external assets."""
    row = _row_or_404(request, run_id)
    repo = request.app.state.repo
    store = _store(request, row)
    findings = repo.findings_for_run(run_id)
    overview_md = (store.overview_path.read_text(encoding="utf-8")
                   if store.overview_path.exists() else "")
    finished = (row["finished_at"] or row["created_at"] or "")[:10]
    meta_line = (f"depth {row['depth']} · {row['recency']} · "
                 f"{len(findings)} sources · {row['status']} {finished} · "
                 f"generated by Deep Research")
    log_lines: list[str] = []
    events_path = store.dir / "events.jsonl"
    if events_path.is_file():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            try:
                text = format_event(json.loads(line))
            except ValueError:
                continue
            if text:
                log_lines.append(text)
    page = interactive_html(
        title=row["title"] or row["query"], query=row["query"],
        meta_line=meta_line,
        overview_html=render_overview(overview_md, len(findings)),
        findings=findings, cards=_finding_cards(store, findings),
        log_lines=log_lines)
    # octet-stream so proxies can't inject a beacon script (see export_html)
    return Response(page, media_type="application/octet-stream", headers={
        "Content-Disposition":
            f'attachment; filename="{run_id}-interactive.html"'})


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
    """Remove a run from every store it touches.

    Order matters: stop the pipeline first, or it keeps writing into a
    directory we are about to remove.
    """
    row = _row_or_404(request, run_id)
    cfg = request.app.state.cfg_loader()
    orch = request.app.state.orch

    if row["status"] in ("queued", "running"):
        orch.cancel(run_id)
        for _ in range(50):  # give the task ~5s to unwind before we delete
            await asyncio.sleep(0.1)
            if run_id not in orch.active:
                break

    research_root = cfg.research_dir.resolve()
    run_dir = (research_root / row["dir"]).resolve()

    # Vectors are a separate store with no foreign key to cascade from; if this
    # is skipped the run keeps surfacing in Ask and semantic search.
    rag = request.app.state.rag
    if rag is not None:
        try:
            rag.index.delete_run(run_id)
        except Exception:
            log.exception("could not drop vectors for %s", run_id)

    # DB row; findings/run_links cascade, fts is cleaned inside delete_run.
    request.app.state.repo.delete_run(run_id)
    request.app.state.bus.detach(run_id)

    if run_dir.is_dir() and run_dir.is_relative_to(research_root) \
            and run_dir != research_root:
        shutil.rmtree(run_dir, ignore_errors=True)
    else:
        log.error("refusing to delete %s — outside %s", run_dir, research_root)

    return Response(status_code=200, headers={"HX-Redirect": "/library"})
