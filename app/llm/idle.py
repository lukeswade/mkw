"""Give the model back when the app has finished with it.

A local model server keeps a pinned model in memory until it is told to
unload it: 18-26 GB of a 48 GB machine, held for hours after a ten-minute
run, until someone opens the server's app and unloads it by hand. With
`model_idle_unload_seconds` set, the app asks the server to unload the model
that long after its last model call, when no run is active or queued.

It acts only after this process has used the model, and once per spell of
use: it never unloads a model somebody else loaded, and a server without
the route is asked once and left alone. A run started from the CLI is
another process and is not seen here; the worst that costs is one reload.
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass

import httpx

from app.llm import providers

log = logging.getLogger(__name__)

_CHECK_EVERY = 15.0


@dataclass
class Activity:
    in_flight: int = 0       # model calls being answered right now
    last: float = 0.0        # monotonic time the latest one ended
    used: bool = False       # this process has called the model since the last unload


activity = Activity()


@contextmanager
def calling():
    """Wraps one request to the model server."""
    activity.in_flight += 1
    activity.used = True
    try:
        yield
    finally:
        activity.in_flight -= 1
        activity.last = time.monotonic()


def due(seconds: int, now: float | None = None) -> bool:
    """The model was used here, nothing is in flight, and it has sat idle
    for `seconds`."""
    if not activity.used or activity.in_flight > 0:
        return False
    return (time.monotonic() if now is None else now) - activity.last >= seconds


async def unload(cfg) -> str:
    """Ask the server to unload the app's model(s); what it said, in a word.

    The server knows a model by its own name, without the "owner/" a model
    setting may carry. 400 is the server saying it was not loaded."""
    names = list(dict.fromkeys(
        n.rsplit("/", 1)[-1] for n in (cfg.resolved_model, cfg.fast_model) if n))
    base = cfg.resolved_base_url.rstrip("/")
    headers = {"Authorization": f"Bearer {cfg.resolved_api_key}"}
    outcomes = []
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            for name in names:
                r = await client.post(f"{base}/models/{name}/unload", headers=headers)
                outcomes.append("unloaded" if r.status_code in (200, 202) else
                                "not loaded" if r.status_code == 400 else "unsupported")
    except httpx.HTTPError:
        return "unreachable"
    for word in ("unloaded", "not loaded"):
        if word in outcomes:
            return word
    return "unsupported"


async def tick(cfg, busy: bool, now: float | None = None) -> str | None:
    """One look: unload if it is time. Returns what happened, or None when
    there was nothing to do."""
    seconds = cfg.model_idle_unload_seconds
    if seconds <= 0 or busy or not providers.is_local(cfg.llm_provider):
        return None
    if not due(seconds, now):
        return None
    outcome = await unload(cfg)
    # one attempt per spell of use, whatever the answer: a server that has no
    # such route, or is down, is not asked again every fifteen seconds
    activity.used = False
    if outcome == "unloaded":
        log.info("model unloaded after %ss idle", seconds)
    elif outcome != "not loaded":
        log.info("model not unloaded after %ss idle: server %s", seconds, outcome)
    return outcome


async def watch(cfg_loader, is_busy, every: float = _CHECK_EVERY) -> None:
    """Runs for the life of the app. Settings are re-read each time, so the
    switch takes effect without a restart."""
    while True:
        await asyncio.sleep(every)
        try:
            await tick(cfg_loader(), is_busy())
        except Exception:
            log.exception("idle unload check failed")
