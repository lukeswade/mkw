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
            "use_prior": params.use_prior,
            "kind": params.kind,
            "brief_id": params.brief_id,
            "parent_run_id": params.parent_run_id,
            "status": "queued",
            "created_at": utcnow(),
        })
        if params.document:
            store.write_document(params.document)
        self.repo.create_run(
            run_id=run_id, query=params.query, depth=params.depth,
            recency=params.recency, dir=run_id, origin=params.origin,
            parent_run_id=params.parent_run_id,
            origin_chat_id=params.origin_chat_id, evergreen=params.evergreen,
            created_by=params.created_by, categories=params.categories,
            use_prior=params.use_prior, kind=params.kind,
            brief_id=params.brief_id)
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
        self._backfill_has_matrix()
        self._backfill_outcomes()

    def _backfill_outcomes(self) -> None:
        """Seed candidate_outcomes from runs that predate the table, once."""
        import json
        import re
        from urllib.parse import urlsplit
        research_dir = self.cfg_loader().research_dir
        done = 0
        for row in self.repo.list_runs(limit=10_000):
            if row["status"] not in ("completed", "cancelled", "interrupted", "failed"):
                continue
            if self.repo.has_outcomes(row["id"]):
                continue
            path = research_dir / row["dir"] / "events.jsonl"
            if not path.is_file():
                continue
            n = 0
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("type") == "finding":
                    # findings carry the domain but not the url; the domain is what matters
                    self.repo.record_outcome(run_id=row["id"], url="", domain=e.get("domain", ""),
                                             engine=e.get("engine", ""), outcome="kept",
                                             relevance=e.get("relevance"))
                    n += 1
                elif e.get("type") == "source_skipped":
                    reason = e.get("reason", "")
                    if reason.startswith(("dropped", "duplicate")):
                        continue
                    m = re.search(r"relevance (\d+)/10", reason)
                    host = urlsplit(e.get("url", "")).netloc.lower().removeprefix("www.")
                    if not host:
                        continue
                    self.repo.record_outcome(run_id=row["id"], url=e.get("url", ""), domain=host,
                                             engine=e.get("engine", ""),
                                             outcome="rejected" if m else "fail",
                                             relevance=int(m.group(1)) if m else None)
                    n += 1
            if n:
                done += 1
        if done:
            log.info("candidate outcomes backfilled for %d run(s)", done)

    def _backfill_has_matrix(self) -> None:
        """Settle has_matrix for rows that predate the column, once.

        One stat per unknown row at boot, instead of one per row per page
        load for the rest of the install's life.
        """
        research_dir = self.cfg_loader().research_dir
        unknown = self.repo.runs_with_unknown_matrix()
        for row in unknown:
            present = RunStore(research_dir / row["dir"]).has_comparison()
            self.repo.update_run(row["id"], has_matrix=1 if present else 0)
        if unknown:
            log.info("has_matrix settled for %d run(s)", len(unknown))

    def _store_for(self, row) -> RunStore | None:
        if row is None:
            return None
        d = self.cfg_loader().research_dir / row["dir"]
        return RunStore(d) if d.is_dir() else None

    # ---- worker -----------------------------------------------------------------
    async def _loop(self) -> None:
        while True:
            run_id = await self.queue.get()
            try:
                row = self.repo.get_run(run_id)
                if row is None or row["status"] != "queued":
                    continue  # cancelled while queued, or gone
                pipeline = Pipeline(self.cfg_loader(), self.repo, self.bus,
                                    rag=self.rag, llm_factory=self.llm_factory)
                task = asyncio.create_task(pipeline.execute(run_id))
                self.active[run_id] = (task, pipeline)
            except asyncio.CancelledError:
                raise
            except Exception:
                # This is the ONLY worker. Anything raised while starting a run
                # used to escape the loop and kill it silently, after which
                # every run ever queued sat in 'queued' for ever with nothing
                # to notice — an unreadable /data/settings.json reaching
                # cfg_loader() is enough to do it. Fail this run, keep serving.
                log.exception("could not start run %s", run_id)
                try:
                    self.repo.update_run(run_id, status="failed",
                                         error="the run could not be started",
                                         stop_reason="failed to start",
                                         finished_at=utcnow())
                    self.bus.publish(run_id, "done", status="failed")
                except Exception:
                    log.exception("could not even mark %s failed", run_id)
                continue
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
        # A 'running' row with nothing in self.active is a run whose owner is
        # gone: a CLI run killed from outside, or a crash between the status
        # write and the task starting. Only a process restart used to
        # reconcile it, so the row stayed 'running' for ever and every guard
        # that asks "is a run active?" answered yes (hit for real 2026-09-09).
        stale = bool(row) and row["status"] == "running" and run_id not in self.active
        if row and (row["status"] == "queued" or stale):
            self.repo.update_run(
                run_id,
                status="interrupted" if stale else "cancelled",
                stop_reason=("its worker is gone" if stale
                             else "cancelled while queued"),
                finished_at=utcnow())
            final = "interrupted" if stale else "cancelled"
            store = self._store_for(row)
            if store:
                store.update_meta(status=final)
            self.bus.publish(run_id, "status", status=final)
            self.bus.publish(run_id, "done", status=final)
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


    def start_matrix(self, run_id: str) -> bool:
        """Build a finished run's comparison table in the background.

        Same guard as re-synthesis: refused while the run is queued, running,
        or already being worked on.
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
                await pipeline.build_matrix(run_id)
            except Exception as e:
                log.exception("matrix build failed for %s", run_id)
                self.bus.publish(run_id, "log", message=f"matrix failed: {e}")
            finally:
                self.active.pop(run_id, None)
                self.bus.detach(run_id)

        task = asyncio.create_task(_job(), name=f"matrix-{run_id}")
        self.active[run_id] = (task, pipeline)
        return True
