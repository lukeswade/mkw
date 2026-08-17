"""Settings page: masked secrets, provider switch, connectivity test buttons."""
from __future__ import annotations

import asyncio
import logging

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from app.config import SECRET_FIELDS, load_settings, mask_secret, save_settings
from app.llm.client import LLM, LLMError
from app.research.searcher import Searcher, SearxngError

log = logging.getLogger(__name__)
router = APIRouter()

_TEXT_FIELDS = ("llm_provider", "llm_base_url", "llm_model", "fast_model",
                "telegram_allowed_user_ids", "searxng_url", "lan_user_label",
                "authority_sites", "browser_solver_url")
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
