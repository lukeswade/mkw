"""Named briefs: a saved reading list plus a standing interest.

Feeds used to be one global list and the topic was typed fresh each run, so
you could follow exactly one subject. A brief bundles the two and can be
scheduled on its own, which is what lets several live side by side.
"""
from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from markupsafe import escape

from app.models import BRIEF_DEFAULT_QUERY, RECENCY_CHOICES, RunParams
from app.research.feed_discovery import discover
from app.research.feeds import parse_feed_list

log = logging.getLogger(__name__)
router = APIRouter()


def _tpl(request: Request):
    return request.app.state.templates


def _brief_rows(request: Request) -> list[dict]:
    repo = request.app.state.repo
    out = []
    for b in repo.list_briefs():
        out.append({"row": b, "feed_count": len(parse_feed_list(b["feeds"]))})
    return out


@router.get("/briefs")
async def briefs_page(request: Request, error: str = ""):
    cfg = request.app.state.cfg_loader()
    return _tpl(request).TemplateResponse(request, "briefs.html", {
        "nav": "briefs",
        "briefs": _brief_rows(request),
        "recency_choices": RECENCY_CHOICES,
        "global_feeds": len(parse_feed_list(getattr(cfg, "feeds", ""))),
        "error": error,
    })


@router.post("/briefs")
async def create_brief(request: Request, name: str = Form(""),
                       topic: str = Form(""), recency: str = Form("week"),
                       depth: int = Form(4)):
    name = name.strip()
    if not name:
        return RedirectResponse("/briefs?error=Give+the+brief+a+name",
                                status_code=303)
    request.app.state.repo.create_brief(
        name=name[:80], topic=topic.strip()[:400],
        recency=recency if recency in RECENCY_CHOICES else "week",
        depth=max(0, min(10, depth)))
    return RedirectResponse("/briefs", status_code=303)


# Declared before the {brief_id} routes on purpose: FastAPI matches in
# declaration order, and "global" would otherwise be parsed as an id.
@router.post("/briefs/global/run")
async def run_global_brief(request: Request):
    """Run the pre-named global feed list, which has no brief row of its own."""
    cfg = request.app.state.cfg_loader()
    if not parse_feed_list(getattr(cfg, "feeds", "")):
        return RedirectResponse("/briefs?error=No+feeds+in+Settings",
                                status_code=303)
    run_id = request.app.state.orch.enqueue(RunParams(
        query=BRIEF_DEFAULT_QUERY, depth=4, recency="week", origin="web",
        kind="brief", categories="general"))
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


@router.post("/briefs/global/adopt")
async def adopt_global_feeds(request: Request):
    """Move the global list into a real brief, so it can carry an interest
    and a schedule like every other one."""
    from app.config import save_settings

    cfg = request.app.state.cfg_loader()
    feeds = getattr(cfg, "feeds", "")
    if not parse_feed_list(feeds):
        return RedirectResponse("/briefs?error=No+feeds+in+Settings",
                                status_code=303)
    request.app.state.repo.create_brief(name="My feeds", feeds=feeds,
                                        recency="week", depth=4)
    save_settings(cfg.settings_path, {"feeds": ""})
    return RedirectResponse("/briefs", status_code=303)


@router.post("/briefs/{brief_id}/feeds")
async def add_feed_to_brief(request: Request, brief_id: int,
                            site: str = Form("")):
    """Resolve a site address to its feed and attach it to this brief."""
    repo = request.app.state.repo
    cfg = request.app.state.cfg_loader()
    brief = repo.get_brief(brief_id)
    if brief is None:
        return HTMLResponse('<span class="hint">That brief is gone.</span>')
    site = site.strip()
    if not site:
        return HTMLResponse('<span class="hint">Enter a site address.</span>')

    async with httpx.AsyncClient(headers={"User-Agent": cfg.user_agent},
                                 timeout=20.0) as client:
        found = await discover(client, site)
    if found is None:
        return HTMLResponse(
            f'<span class="hint">No feed found at <code>{escape(site)}</code>.'
            f'</span>')
    if found.url in parse_feed_list(brief["feeds"]):
        return HTMLResponse(
            f'<span class="hint">Already in this brief: '
            f'<strong>{escape(found.title)}</strong>.</span>')

    blob = (brief["feeds"].rstrip() + "\n") if brief["feeds"].strip() else ""
    repo.update_brief(brief_id, feeds=f"{blob}{found.url}\n")
    return HTMLResponse(
        f'<span class="hint">Added <strong>{escape(found.title)}</strong> '
        f'({found.entries} recent items). Reload to see the list.</span>')


@router.post("/briefs/{brief_id}/edit")
async def edit_brief(request: Request, brief_id: int, name: str = Form(""),
                     topic: str = Form(""), feeds: str = Form(""),
                     recency: str = Form("week"), depth: int = Form(4)):
    repo = request.app.state.repo
    if repo.get_brief(brief_id) is None:
        return RedirectResponse("/briefs", status_code=303)
    repo.update_brief(
        brief_id, name=(name.strip() or "Untitled brief")[:80],
        topic=topic.strip()[:400], feeds=feeds,
        recency=recency if recency in RECENCY_CHOICES else "week",
        depth=max(0, min(10, depth)))
    return RedirectResponse("/briefs", status_code=303)


@router.post("/briefs/{brief_id}/daily")
async def toggle_daily(request: Request, brief_id: int):
    repo = request.app.state.repo
    brief = repo.get_brief(brief_id)
    if brief is None:
        return RedirectResponse("/briefs", status_code=303)
    repo.update_brief(brief_id, daily=0 if brief["daily"] else 1)
    return RedirectResponse("/briefs", status_code=303)


@router.post("/briefs/{brief_id}/run")
async def run_brief(request: Request, brief_id: int):
    repo = request.app.state.repo
    brief = repo.get_brief(brief_id)
    if brief is None:
        return RedirectResponse("/briefs", status_code=303)
    if not parse_feed_list(brief["feeds"]):
        return RedirectResponse(
            "/briefs?error=That+brief+has+no+feeds+yet", status_code=303)
    run_id = request.app.state.orch.enqueue(RunParams(
        query=f"Brief: {brief['name']}", depth=brief["depth"],
        recency=brief["recency"], origin="web", kind="brief",
        brief_id=brief_id,
        categories="general"))          # unused by feeds, but the field is required
    repo.update_brief(brief_id, last_run_at=_now())
    return RedirectResponse(f"/runs/{run_id}", status_code=303)


@router.post("/briefs/{brief_id}/delete")
async def delete_brief(request: Request, brief_id: int):
    request.app.state.repo.delete_brief(brief_id)
    return RedirectResponse("/briefs", status_code=303)


def _now() -> str:
    from app.db import utcnow
    return utcnow()
