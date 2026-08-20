"""Background refresh for runs marked evergreen.

A run flagged evergreen is re-researched periodically against a recent time
window, producing a child run linked to it. The flag stays on the original —
that way a failed refresh can't break the chain, and the toggle stays where
the user put it.
"""
from __future__ import annotations

import asyncio
import logging

from app.models import RunParams

log = logging.getLogger(__name__)

REFRESH_INTERVAL_HOURS = 24
# The due-check is one indexed query, so polling often is cheap and means a
# restart doesn't skip a cycle (the old worker slept 24h before its first look).
CHECK_EVERY_SECONDS = 900


def _categories_of(row) -> str:
    try:
        return row["categories"] or ""
    except (KeyError, IndexError):
        return ""


async def refresh_due_runs(orchestrator, repo) -> int:
    """Enqueue a refresh for every evergreen run that is due. Returns the count."""
    due = repo.evergreen_due(REFRESH_INTERVAL_HOURS)
    for row in due:
        params = RunParams(
            query=row["query"],
            depth=row["depth"],
            recency="month",           # refreshes look for what's new
            parent_run_id=row["id"],
            origin=row["origin"] or "web",
            origin_chat_id=row["origin_chat_id"],
            created_by=row["created_by"] or "",
            categories=_categories_of(row),
        )
        new_id = orchestrator.enqueue(params)
        log.info("evergreen refresh %s queued for %s", new_id, row["id"])
    return len(due)


async def refresh_loop(orchestrator, repo) -> None:
    log.info("evergreen refresh worker started (every %ds, interval %dh)",
             CHECK_EVERY_SECONDS, REFRESH_INTERVAL_HOURS)
    while True:
        try:
            await refresh_due_runs(orchestrator, repo)
        except asyncio.CancelledError:
            log.info("evergreen refresh worker stopping")
            raise
        except Exception:
            # A bad row or a transient DB error must not kill the loop.
            log.exception("evergreen refresh check failed")
        try:
            await asyncio.sleep(CHECK_EVERY_SECONDS)
        except asyncio.CancelledError:
            log.info("evergreen refresh worker stopping")
            raise
