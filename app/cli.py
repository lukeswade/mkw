"""Command-line interface.

    python -m app.cli run "your research question" --depth 3 --recency month
    python -m app.cli runs
    python -m app.cli ask "question over accumulated research"   (knowledge layer)
    python -m app.cli reindex                                     (knowledge layer)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from app.config import load_settings
from app.db import Repo, connect
from app.models import RECENCY_CHOICES, RunParams
from app.research.orchestrator import Orchestrator
from app.research.progress import ProgressBus, format_event
from app.research.storage import RunStore


async def _print_events(replay: list[dict], queue: asyncio.Queue | None) -> None:
    for e in replay:
        line = format_event(e)
        if line:
            print(line, flush=True)
    if queue is None:
        return
    while True:
        e = await queue.get()
        line = format_event(e)
        if line:
            print(line, flush=True)
        if e.get("type") in ("done",):
            return


def _build_rag(cfg):
    """Knowledge layer is optional at CLI import time (heavy deps)."""
    try:
        from app.rag.service import RagService
    except ImportError:
        return None
    return RagService(cfg)


async def _cmd_run(args) -> int:
    cfg = load_settings()
    cfg.ensure_dirs()
    repo = Repo(connect(cfg.db_path))
    bus = ProgressBus()
    orch = Orchestrator(load_settings, repo, bus, rag=_build_rag(cfg))
    params = RunParams(query=args.query, depth=args.depth,
                       recency=args.recency, origin="cli", created_by="CLI")
    run_id = orch.enqueue(params)
    row = repo.get_run(run_id)
    store = RunStore(cfg.research_dir / row["dir"])
    replay, queue = bus.subscribe(run_id, store)
    printer = asyncio.create_task(_print_events(replay, queue))
    try:
        await orch.execute_now(run_id)
    except asyncio.CancelledError:
        pass
    try:
        await asyncio.wait_for(printer, timeout=5)
    except asyncio.TimeoutError:
        printer.cancel()

    row = repo.get_run(run_id)
    print(f"\nrun directory: {store.dir}")
    if row["stats_json"]:
        stats = json.loads(row["stats_json"])
        llm_stats = stats.get("llm", {})
        cost = llm_stats.get("est_cost_usd")
        print(f"rounds: {stats.get('rounds')} · sources kept: "
              f"{stats.get('sources_kept')} · skipped: {stats.get('sources_skipped')}"
              f" · LLM calls: {llm_stats.get('calls')}"
              + (f" · est cost: ${cost}" if cost is not None else ""))
    return 0 if row["status"] == "completed" else 1


async def _cmd_runs(_args) -> int:
    cfg = load_settings()
    cfg.ensure_dirs()
    repo = Repo(connect(cfg.db_path))
    for r in repo.list_runs(limit=30):
        print(f"{r['id']}  [{r['status']}]  depth={r['depth']} "
              f"recency={r['recency']}  {r['title'] or r['query']}")
    return 0


async def _cmd_ask(args) -> int:
    cfg = load_settings()
    cfg.ensure_dirs()
    rag = _build_rag(cfg)
    if rag is None:
        print("knowledge layer unavailable (rag deps not installed)")
        return 1
    repo = Repo(connect(cfg.db_path))
    answer = await rag.ask(args.question, repo)
    print(answer)
    return 0


async def _cmd_resynth(args) -> int:
    from app.research.pipeline import Pipeline
    cfg = load_settings()
    cfg.ensure_dirs()
    repo = Repo(connect(cfg.db_path))
    row = repo.get_run(args.run_id)
    if row is None:
        print(f"run {args.run_id} not found")
        return 1
    bus = ProgressBus()
    store = RunStore(cfg.research_dir / row["dir"])
    bus.attach(store)
    pipeline = Pipeline(cfg, repo, bus, rag=_build_rag(cfg))
    replay, queue = bus.subscribe(args.run_id, store)
    printer = asyncio.create_task(_print_events(replay, queue))
    try:
        await pipeline.resynthesize(args.run_id)
    except Exception as e:
        print(f"re-synthesis failed: {e}")
        return 1
    finally:
        printer.cancel()
    print(f"overview rewritten: {store.overview_path}")
    return 0


async def _cmd_reindex(_args) -> int:
    cfg = load_settings()
    cfg.ensure_dirs()
    rag = _build_rag(cfg)
    if rag is None:
        print("knowledge layer unavailable (rag deps not installed)")
        return 1
    repo = Repo(connect(cfg.db_path))
    n = await rag.reindex_all(repo)
    print(f"reindexed {n} run(s) from {cfg.research_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m app.cli",
                                description="Local deep-research agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="start a research run")
    run_p.add_argument("query")
    run_p.add_argument("--depth", "-d", type=int, default=3,
                       choices=range(0, 11), metavar="0-10")
    run_p.add_argument("--recency", "-r", default="all", choices=RECENCY_CHOICES)
    run_p.set_defaults(fn=_cmd_run)

    runs_p = sub.add_parser("runs", help="list research runs")
    runs_p.set_defaults(fn=_cmd_runs)

    ask_p = sub.add_parser("ask", help="ask a question over accumulated research")
    ask_p.add_argument("question")
    ask_p.set_defaults(fn=_cmd_ask)

    re_p = sub.add_parser("reindex", help="rebuild search indexes from disk")
    re_p.set_defaults(fn=_cmd_reindex)

    rs_p = sub.add_parser(
        "resynth",
        help="rewrite a run's overview from its stored findings (no re-search)")
    rs_p.add_argument("run_id")
    rs_p.set_defaults(fn=_cmd_resynth)

    args = p.parse_args(argv)
    return asyncio.run(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
