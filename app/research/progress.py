"""Per-run progress pub/sub with crash-safe persistence.

Every event is appended to the run's events.jsonl *and* fanned out to live
subscriber queues. publish() is fully synchronous (no await between the file
append and the queue puts), so subscribe()'s attach-then-replay sequence can
never miss or duplicate an event within the single event loop.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from pathlib import Path

from app.research.storage import RunStore

TERMINAL_EVENTS = {"done"}


class ProgressBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._stores: dict[str, RunStore] = {}
        self._seq: dict[str, int] = {}

    def attach(self, store: RunStore) -> None:
        """Register an active run. Seq continues from any existing event log."""
        run_id = store.run_id
        self._stores[run_id] = store
        existing = store.read_events()
        self._seq[run_id] = existing[-1]["seq"] if existing else 0

    def detach(self, run_id: str) -> None:
        self._stores.pop(run_id, None)
        self._seq.pop(run_id, None)
        self._subs.pop(run_id, None)

    def is_active(self, run_id: str) -> bool:
        return run_id in self._stores

    def publish(self, run_id: str, type_: str, **fields) -> dict:
        store = self._stores.get(run_id)
        if store is None:
            return {}
        self._seq[run_id] += 1
        event = {"seq": self._seq[run_id], "ts": round(time.time(), 2),
                 "type": type_, **fields}
        store.append_event(event)
        for q in list(self._subs.get(run_id, [])):
            q.put_nowait(event)
        return event

    def subscribe(self, run_id: str, store: RunStore,
                  after_seq: int = 0) -> tuple[list[dict], asyncio.Queue | None]:
        """Return (replay, live_queue). live_queue is None for inactive runs."""
        if not self.is_active(run_id):
            return store.read_events(after_seq), None
        q: asyncio.Queue = asyncio.Queue()
        self._subs[run_id].append(q)  # attach FIRST, then read — no gap
        replay = store.read_events(after_seq)
        return replay, q

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        subs = self._subs.get(run_id)
        if subs and q in subs:
            subs.remove(q)
