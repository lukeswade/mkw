"""Settings page: masked secrets, provider switch, connectivity test buttons."""
from __future__ import annotations

import asyncio
import logging

import httpx
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from markupsafe import escape

from app.config import SECRET_FIELDS, load_settings, mask_secret, save_settings
from app.llm.client import LLM, LLMError
from app.research.searcher import Searcher, SearxngError

log = logging.getLogger(__name__)
router = APIRouter()

_TEXT_FIELDS = ("llm_provider", "llm_base_url", "llm_model", "fast_model",
                "telegram_allowed_user_ids", "searxng_url", "lan_user_label",
                "authority_sites", "feeds", "browser_solver_url",
                "embedding_model", "embedding_base_url")
_SECRET_FORM_FIELDS = ("llm_api_key", "telegram_bot_token", "web_password")


@router.get("/settings")
async def settings_page(request: Request, saved: int = 0):
    cfg = request.app.state.cfg_loader()
    masked = {f: mask_secret(getattr(cfg, f)) for f in SECRET_FIELDS}
    return request.app.state.templates.TemplateResponse(
        request, "settings.html",
        {"nav": "settings", "cfg": cfg, "masked": masked, "saved": saved})


@router.post("/settings")
async def settings_save(request: Request):
    form = await request.form()
    updates: dict = {}
    for f in _TEXT_FIELDS:
        if f in form:
            updates[f] = str(form[f]).strip()
    if "llm_concurrency" in form:
        try:
            updates["llm_concurrency"] = max(1, min(16, int(str(form["llm_concurrency"]))))
        except ValueError:
            pass
    if "results_per_query" in form:
        try:
            updates["results_per_query"] = max(1, min(20, int(str(form["results_per_query"]))))
        except ValueError:
            pass
    updates["respect_robots"] = form.get("respect_robots") == "on"
    updates["reference_chasing"] = form.get("reference_chasing") == "on"
    updates["browser_impersonation"] = form.get("browser_impersonation") == "on"
    if "blocked_domains" in form:
        updates["blocked_domains"] = str(form["blocked_domains"]).strip()
    for f in _SECRET_FORM_FIELDS:
        value = str(form.get(f, "")).strip()
        if value:  # blank = leave unchanged
            updates[f] = value
        if form.get(f"{f}_clear") == "on":
            updates[f] = ""

    cfg = request.app.state.cfg_loader()
    save_settings(cfg.settings_path, updates)
    return RedirectResponse("/settings?saved=1", status_code=303)


@router.post("/settings/test-llm")
async def test_llm(request: Request):
    cfg = load_settings()
    try:
        llm = LLM(cfg)
        reply = await asyncio.wait_for(
            llm.chat("test", [{"role": "user", "content":
                               "Reply with exactly: OK"}], max_tokens=8),
            timeout=45)
        msg = (f"✓ {cfg.llm_provider} responded ({llm.model}): "
               f"{reply.strip()[:60] or '(empty)'}")
        ok = True
    except (LLMError, asyncio.TimeoutError, Exception) as e:
        msg, ok = f"✗ {e}", False
    return request.app.state.templates.TemplateResponse(
        request, "partials/test_result.html", {"ok": ok, "msg": msg})


@router.post("/settings/test-searxng")
async def test_searxng(request: Request):
    cfg = load_settings()
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            results = await Searcher(cfg.searxng_url, client).search("test", "all")
        msg, ok = f"✓ SearXNG at {cfg.searxng_url} returned {len(results)} results", True
    except SearxngError as e:
        msg, ok = f"✗ {e}", False
    except Exception as e:
        msg, ok = f"✗ cannot reach SearXNG at {cfg.searxng_url}: {e}", False
    return request.app.state.templates.TemplateResponse(
        request, "partials/test_result.html", {"ok": ok, "msg": msg})


@router.post("/settings/add-feed")
async def add_feed(request: Request, site: str = Form("")):
    """Turn whatever the user typed into a feed and append it.

    People know a site's address, not where it keeps its feed, so asking for
    the feed URL directly is asking them to go and find it first.
    """
    from app.research.feed_discovery import discover
    from app.research.feeds import parse_feed_list

    cfg = load_settings()
    site = site.strip()
    if not site:
        return HTMLResponse('<span class="hint">Enter a site address.</span>')

    async with httpx.AsyncClient(
            headers={"User-Agent": cfg.user_agent}, timeout=20.0) as client:
        found = await discover(client, site)

    if found is None:
        return HTMLResponse(
            f'<span class="hint">No feed found at '
            f'<code>{escape(site)}</code>. Some sites do not publish one; '
            f'if you know the feed URL, paste it in the box above.</span>')

    existing = parse_feed_list(cfg.feeds)
    if found.url in existing:
        return HTMLResponse(
            f'<span class="hint">Already subscribed to '
            f'<strong>{escape(found.title)}</strong>.</span>')

    blob = (cfg.feeds.rstrip() + "\n" if cfg.feeds.strip() else "")
    save_settings(cfg.settings_path, {"feeds": f"{blob}{found.url}\n"})
    return HTMLResponse(
        f'<span class="hint">Added <strong>{escape(found.title)}</strong> '
        f'({found.entries} recent items) — <code>{escape(found.url)}</code>. '
        f'Reload to see it in the list.</span>')


@router.post("/settings/follow-source")
async def follow_source(request: Request, domain: str = Form("")):
    """Subscribe to a source that already proved useful in a run.

    A feed list built by remembering sites is a chore; one built from the
    domains your own research kept citing builds itself.
    """
    from app.research.feed_discovery import discover
    from app.research.feeds import parse_feed_list

    cfg = load_settings()
    domain = domain.strip().lower()
    if not domain:
        return HTMLResponse('<span class="hint">No source to follow.</span>')

    async with httpx.AsyncClient(
            headers={"User-Agent": cfg.user_agent}, timeout=20.0) as client:
        found = await discover(client, domain)

    if found is None:
        return HTMLResponse(
            f'<span class="hint">{escape(domain)} does not publish a feed '
            f'this could find.</span>')
    if found.url in parse_feed_list(cfg.feeds):
        return HTMLResponse(
            f'<span class="hint">Already following '
            f'<strong>{escape(found.title)}</strong>.</span>')

    blob = (cfg.feeds.rstrip() + "\n" if cfg.feeds.strip() else "")
    save_settings(cfg.settings_path, {"feeds": f"{blob}{found.url}\n"})
    return HTMLResponse(
        f'<span class="hint">Following <strong>{escape(found.title)}</strong> '
        f'— it will appear in your next brief.</span>')
