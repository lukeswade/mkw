"""Telegram bot lifecycle, run inside FastAPI's event loop.

Manual start/stop (never run_polling — it owns the loop and installs signal
handlers). Any failure here degrades to web-only mode instead of crashing
the container.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging

from telegram import Update
from telegram.ext import Application

from app.telegram.handlers import register_handlers

log = logging.getLogger(__name__)


async def start_bot(cfg_loader, orch, repo, bus) -> Application | None:
    cfg = cfg_loader()
    token = cfg.telegram_bot_token.strip()
    if not token:
        log.info("no TELEGRAM_BOT_TOKEN — Telegram disabled, web-only mode")
        return None
    try:
        app = (Application.builder().token(token)
               .connect_timeout(10).read_timeout(15).build())
        app.bot_data.update({
            "orch": orch, "repo": repo, "bus": bus, "cfg_loader": cfg_loader,
            "allowed_ids": cfg.allowed_telegram_ids,
            "watchers": set(),
        })
        register_handlers(app)
        await app.initialize()
        await app.start()
        await app.updater.start_polling(
            drop_pending_updates=True, allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        log.warning("Telegram bot failed to start (%s) — web-only mode. "
                    "409 Conflict means another instance is polling this token.",
                    e)
        return None
    if not cfg.allowed_telegram_ids:
        log.warning("Telegram bot is up but TELEGRAM_ALLOWED_USER_IDS is empty "
                    "— nobody can use it. Message the bot /id to find yours.")
    log.info("Telegram bot polling as configured")
    return app


async def stop_bot(app: Application | None) -> None:
    if app is None:
        return
    for task in list(app.bot_data.get("watchers", ())):
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    with contextlib.suppress(Exception):
        if app.updater and app.updater.running:
            await app.updater.stop()
        await app.stop()
        await app.shutdown()
