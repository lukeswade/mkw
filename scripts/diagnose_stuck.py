#!/usr/bin/env python3
"""Why is this run not moving? — read-only triage for a stalled run.

Answers the one question the UI cannot: a run sitting on "queued" or
"running" is either genuinely working (a long LLM call publishes nothing
until it returns), waiting behind another run, or wedged. This reads the
same three things the app writes — the runs table, each run's events.jsonl,
and app.log — and says which.

    python3 scripts/diagnose_stuck.py                # ./data
    python3 scripts/diagnose_stuck.py --data-dir /srv/deep-research/data

Run it on the HOST, not in the container: .dockerignore excludes scripts/
and neither compose file bind-mounts it, so this file does not exist inside
the app image. It does not need to — compose bind-mounts ./data:/data, so
the host sees the same database and event logs the container is writing.

Stdlib only (works on a stock macOS python3), opens the database read-only:
safe to run against a live instance mid-run.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Longer than this with nothing written to events.jsonl is worth explaining.
QUIET_SECONDS = 180
EVENTS_SHOWN = 12

# What the last recorded event implies about where the run actually is. The
# pipeline publishes nothing during an LLM call, so a long silence after one
# of these is expected work, not a hang.
PHASE_MEANING = {
    "planning": ("the planner LLM call", True),
    "quick answer": ("the depth-0 answer LLM call", True),
    "gap analysis": ("the gap-analysis LLM call", True),
    "synthesis": ("the synthesis LLM call (streams to the browser, but its "
                  "chunks are deliberately never written to events.jsonl, so "
                  "this file stays silent for the whole call)", True),
    "indexing": ("embedding the run into the vector index — one un-timed "
                 "call that encodes every chunk of the run at once", True),
}


def parse_ts(value: str | None) -> float | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def ago(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def read_events(path: Path) -> list[dict]:
    events: list[dict] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # torn final line after a crash
    except OSError:
        pass
    return events


def describe(event: dict) -> str:
    typ = event.get("type", "?")
    if typ == "phase":
        return f"phase: {event.get('phase')}"
    if typ == "status":
        return f"status: {event.get('status')}"
    if typ == "round_start":
        return (f"round {event.get('round')}/{event.get('depth')} started "
                f"({len(event.get('queries', []))} queries)")
    if typ == "searched":
        return (f"searched: {event.get('results')} results → "
                f"{event.get('candidates')} candidates")
    if typ == "finding":
        return f"kept [{event.get('idx')}] {event.get('title', '')[:60]}"
    if typ == "source_skipped":
        return f"skipped ({event.get('reason')}) {event.get('url', '')[:60]}"
    if typ == "gap":
        return (f"gap: saturated={event.get('saturated')}, "
                f"{len(event.get('next_queries', []))} next queries")
    if typ == "log":
        return f"log: {event.get('message', '')[:100]}"
    if typ == "error":
        return f"ERROR: {event.get('message', '')[:120]}"
    if typ == "done":
        return f"done: {event.get('status')} ({event.get('stop_reason', '')})"
    return typ


# Events the pipeline emits while working through a round. One of these after
# a "phase" marker means the run has moved past that phase.
PROGRESS_TYPES = {"round_start", "searched", "finding", "source_skipped", "gap"}


def current_phase(events: list[dict]) -> str:
    """The phase the run is still inside, or "" if it has moved past it."""
    for e in reversed(events):
        typ = e.get("type")
        if typ in PROGRESS_TYPES:
            return ""
        if typ == "phase":
            return str(e.get("phase") or "")
    return ""


def explain(row: sqlite3.Row, events: list[dict], quiet_for: float | None,
            others_running: int, settings: dict) -> list[str]:
    """The verdict lines for one stalled run."""
    status = row["status"]
    out: list[str] = []

    if status == "queued":
        if others_running:
            out.append(
                "WAITING — the orchestrator runs one research run at a time, "
                "and another run is still going. This run starts when that "
                "one finishes. Diagnose the running one below.")
        else:
            out.append(
                "STUCK — nothing is running, yet this run was never picked "
                "up. The background worker task ('research-worker') is gone: "
                "Orchestrator._loop builds its Pipeline outside its own "
                "try/except, so one exception there kills the worker "
                "silently and every later run queues forever.")
            out.append(
                "  Fix now: restart the app. Orchestrator.recover() "
                "re-enqueues queued runs on boot, so this run resumes.")
            out.append(
                "  Confirm: look for 'Task exception was never retrieved' "
                "or a traceback naming orchestrator.py in app.log / "
                "`docker compose logs app`.")
        return out

    # status == "running"
    if quiet_for is None:
        out.append(
            "STUCK — marked running but no events were ever written. It died "
            "between the status update and the first step, or the run "
            "directory is not writable.")
        return out

    phase = current_phase(events)
    meaning, expected = PHASE_MEANING.get(phase, ("", False))

    if quiet_for < QUIET_SECONDS:
        out.append(f"MOVING — last event {ago(quiet_for)} ago. Not stuck.")
        return out

    if expected:
        out.append(f"BUSY (probably) — silent for {ago(quiet_for)} inside "
                   f"{meaning}.")
        out.append("  No event is published until that call returns, so "
                   "silence here is normal on a local model; the run is only "
                   "wedged if it outlasts llm_timeout × 3 plus backoff "
                   "(see the settings printed above).")
        if phase == "indexing":
            out.append("  Indexing is the one step with no timeout, but in "
                       "the Docker image bge-small is baked in and "
                       "HF_HUB_OFFLINE=1, so it cannot stall on a download — "
                       "it is CPU-encoding every chunk of the run. Expect "
                       "seconds to a couple of minutes, not hours.")
    else:
        tail = describe(events[-1]) if events else "(none)"
        out.append(f"SILENT for {ago(quiet_for)} — last event was: {tail}")
        out.append("  This is the per-document notes stage: a fetch plus one "
                   "LLM call per source, llm_concurrency at a time, and each "
                   "document publishes only when it finishes.")
        timeout = int(settings.get("llm_timeout", 180) or 180)
        worst = timeout * 3 + 10  # three attempts plus 2s/8s backoff
        out.append(f"  Worst case for one document at llm_timeout={timeout}s "
                   f"is ~{worst // 60}m {worst % 60}s (3 attempts + backoff) "
                   f"before it gives up — and when it does give up, "
                   f"take_notes re-raises LLMError, which fails the whole "
                   f"run rather than skipping that one source.")
        out.append("  If llm_concurrency > 1 and the inference server does "
                   "not batch (llama.cpp single slot, Ollama with "
                   "OLLAMA_NUM_PARALLEL=1, LM Studio), the concurrent calls "
                   "queue server-side, so each one's wall time multiplies "
                   "and timeouts become likely. Try llm_concurrency=1.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="./data",
                    help="DATA_DIR of the instance (default: ./data)")
    ap.add_argument("--log-lines", type=int, default=25,
                    help="warning/error lines to show from app.log")
    args = ap.parse_args()

    data = Path(args.data_dir)
    db_path = data / "app.sqlite3"
    if not db_path.is_file():
        print(f"no database at {db_path} — wrong --data-dir?", file=sys.stderr)
        return 2

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    settings = {}
    try:
        settings = json.loads((data / "settings.json").read_text())
    except (OSError, json.JSONDecodeError):
        pass
    relevant = ("llm_provider", "llm_model", "fast_model", "llm_concurrency",
                "llm_timeout", "embedding_model", "embedding_base_url",
                "browser_solver_url")
    shown = {k: settings[k] for k in relevant if k in settings}
    print("settings.json (run-affecting keys):",
          json.dumps(shown) if shown else "(defaults only)")
    print()

    rows = conn.execute(
        "SELECT * FROM runs WHERE status IN ('queued','running') "
        "ORDER BY created_at").fetchall()
    if not rows:
        print("No run is queued or running. Five most recent:\n")
        recent = conn.execute(
            "SELECT id, status, stop_reason, error, query "
            "FROM runs ORDER BY created_at DESC LIMIT 5").fetchall()
        for r in recent:
            detail = r["error"] or r["stop_reason"] or ""
            print(f"  {r['id']}  {r['status']:<11} {r['query'][:50]}"
                  f"{'  — ' + detail[:60] if detail else ''}")
        if recent and recent[0]["status"] == "interrupted":
            print("\n  The newest run was INTERRUPTED — the process died "
                  "mid-run and recover() marked it on the next boot.")
            print("  With `restart: unless-stopped`, Docker restarts the app "
                  "silently, so this looks like a run that simply stopped. "
                  "Check `docker compose ps` (uptime) and `docker inspect "
                  "--format '{{.State.OOMKilled}} {{.RestartCount}}' "
                  "$(docker compose ps -q app)`.")
        return 0

    running = sum(1 for r in rows if r["status"] == "running")
    print(f"{len(rows)} run(s) queued or running "
          f"({running} running, {len(rows) - running} queued)\n")

    now = time.time()
    for row in rows:
        events = read_events(data / "research_data" / row["dir"] / "events.jsonl")
        last_ts = events[-1].get("ts") if events else None
        quiet_for = (now - last_ts) if last_ts else None
        started = parse_ts(row["started_at"] or row["created_at"])

        print("=" * 72)
        print(f"{row['id']}  [{row['status']}]  depth {row['depth']}  "
              f"{row['origin']}")
        print(f"  query: {row['query'][:100]}")
        silence = f"{ago(quiet_for)} ago" if quiet_for is not None else "never"
        print(f"  age:   {ago(now - started if started else None)}"
              f"   events: {len(events)}"
              f"   last event: {silence}")
        print()
        for line in explain(row, events, quiet_for, running, settings):
            print(f"  {line}")

        if events:
            print(f"\n  last {min(EVENTS_SHOWN, len(events))} events:")
            for e in events[-EVENTS_SHOWN:]:
                stamp = time.strftime("%H:%M:%S", time.localtime(e.get("ts", 0)))
                print(f"    [{stamp}] {describe(e)}")
        print()

    log = data / "app.log"
    if log.is_file():
        bad = [ln.rstrip() for ln in log.read_text(errors="replace").splitlines()
               if " WARNING " in ln or " ERROR " in ln]
        if bad:
            print("=" * 72)
            print(f"last {min(args.log_lines, len(bad))} warning/error lines "
                  f"in app.log:")
            for ln in bad[-args.log_lines:]:
                print(f"  {ln[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
