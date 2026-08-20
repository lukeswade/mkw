"""Run queue: one research run at a time, cancellation, startup recovery.

In-process by design (single uvicorn worker). Settings are re-loaded for each
run so changes saved on the Settings page apply from the next run onward.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

from app.config import Settings
from app.db import Repo, utcnow
from app.models import RunParams
from app.research.pipeline import Pipeline
from app.research.progress import ProgressBus
from app.research.storage import RunStore

log = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, cfg_loader: Callable[[], Settings], repo: Repo,
                 bus: ProgressBus, rag=None, llm_factory=None):
        self.cfg_loader = cfg_loader
        self.repo = repo
        self.bus = bus
        self.rag = rag
        self.llm_factory = llm_factory
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.active: dict[str, tuple[asyncio.Task, Pipeline]] = {}
        self._worker: asyncio.Task | None = None
        self._shutting_down = False

    # ---- submission --------------------------------------------------------
    def enqueue(self, params: RunParams) -> str:
        cfg = self.cfg_loader()
        cfg.ensure_dirs()
        store = RunStore.create(cfg.research_dir, params.query)
        run_id = store.run_id
        store.write_meta({
            "run_id": run_id,
            "query": params.query,
            "depth": params.depth,
            "recency": params.recency,
            "origin": params.origin,
            "created_by": params.created_by,
            "categories": params.categories,
            "parent_run_id": params.parent_run_id,
            "status": "queued",
            "created_at": utcnow(),
        })
        self.repo.create_run(
            run_id=run_id, query=params.query, depth=params.depth,
            recency=params.recency, dir=run_id, origin=params.origin,
            parent_run_id=params.parent_run_id,
            origin_chat_id=params.origin_chat_id, evergreen=params.evergreen,
            created_by=params.created_by, categories=params.categories)
        if params.parent_run_id and self.repo.get_run(params.parent_run_id):
            self.repo.add_run_link(params.parent_run_id, run_id, "followup", None)
        self.bus.attach(store)
        self.bus.publish(run_id, "status", status="queued",
                         position=self.queue.qsize() + 1)
        self.queue.put_nowait(run_id)
        return run_id

    def queue_position(self, run_id: str) -> int | None:
        """1-based position among queued runs (None if not queued)."""
        queued = [r["id"] for r in self.repo.runs_with_status("queued")]
        try:
            return queued.index(run_id) + 1
        except ValueError:
            return None

    # ---- lifecycle ---------------------------------------------------------------
    def start(self) -> None:
        self._worker = asyncio.create_task(self._loop(), name="research-worker")

    async def stop(self) -> None:
        self._shutting_down = True
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
        self._worker = None

    def recover(self) -> None:
        """Mark orphaned running→interrupted; re-enqueue queued runs."""
        queued = self.repo.recover_on_startup()
        for row in self.repo.runs_with_status("interrupted"):
            store = self._store_for(row)
            if store and store.read_meta().get("status") in ("running", "queued"):
                store.update_meta(status="interrupted",
                                  stop_reason="process restart")
        for run_id in queued:
            row = self.repo.get_run(run_id)
            store = self._store_for(row)
            if store is None:
                self.repo.update_run(run_id, status="failed",
                                     error="run directory missing")
                continue
            self.bus.attach(store)
            self.queue.put_nowait(run_id)
            log.info("re-enqueued run %s after restart", run_id)

    def _store_for(self, row) -> RunStore | None:
        if row is None:
            return None
        d = self.cfg_loader().research_dir / row["dir"]
        return RunStore(d) if d.is_dir() else None

    # ---- worker -----------------------------------------------------------------
    async def _loop(self) -> None:
        while True:
            run_id = await self.queue.get()
            row = self.repo.get_run(run_id)
            if row is None or row["status"] != "queued":
                continue  # cancelled while queued, or gone
            pipeline = Pipeline(self.cfg_loader(), self.repo, self.bus,
                                rag=self.rag, llm_factory=self.llm_factory)
            task = asyncio.create_task(pipeline.execute(run_id))
            self.active[run_id] = (task, pipeline)
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.cancelled() and not self._shutting_down:
                    continue  # user cancelled this run → next queue item
                # worker itself is shutting down: stop the child, mark interrupted
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                self.repo.update_run(run_id, status="interrupted",
                                     stop_reason="shutdown during run",
                                     finished_at=utcnow())
                raise
            except Exception:
                log.exception("pipeline for %s escaped its error handling", run_id)
            finally:
                self.active.pop(run_id, None)

    # ---- cancellation ----------------------------------------------------------
    def cancel(self, run_id: str) -> bool:
        entry = self.active.get(run_id)
        if entry:
            task, pipeline = entry
            pipeline.cancel_requested = True  # checked between documents
            task.cancel()
            return True
        row = self.repo.get_run(run_id)
        if row and row["status"] == "queued":
            self.repo.update_run(run_id, status="cancelled",
                                 stop_reason="cancelled while queued",
                                 finished_at=utcnow())
            store = self._store_for(row)
            if store:
                store.update_meta(status="cancelled")
                self.bus.publish(run_id, "status", status="cancelled")
                self.bus.publish(run_id, "done", status="cancelled")
                self.bus.detach(run_id)
            return True
        return False

    def start_resynth(self, run_id: str) -> bool:
        """Regenerate a finished run's overview in the background.

        Refused while the run is queued/running or already being worked on.
        """
        row = self.repo.get_run(run_id)
        if (row is None or row["status"] in ("queued", "running")
                or run_id in self.active):
            return False
        store = self._store_for(row)
        if store is None:
            return False
        pipeline = Pipeline(self.cfg_loader(), self.repo, self.bus,
                            rag=self.rag, llm_factory=self.llm_factory)
        self.bus.attach(store)

        async def _job() -> None:
            try:
                await pipeline.resynthesize(run_id)
            except Exception as e:
                log.exception("re-synthesis failed for %s", run_id)
                self.bus.publish(run_id, "log",
                                 message=f"re-synthesis failed: {e}")
            finally:
                self.active.pop(run_id, None)
                self.bus.detach(run_id)

        task = asyncio.create_task(_job(), name=f"resynth-{run_id}")
        self.active[run_id] = (task, pipeline)
        return True

    async def execute_now(self, run_id: str) -> None:
        """Run synchronously (CLI path, no worker loop)."""
        pipeline = Pipeline(self.cfg_loader(), self.repo, self.bus,
                            rag=self.rag, llm_factory=self.llm_factory)
        self.active[run_id] = (asyncio.current_task(), pipeline)  # type: ignore[arg-type]
        try:
            await pipeline.execute(run_id)
        finally:
            self.active.pop(run_id, None)
