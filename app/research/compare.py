"""Two runs of one question, side by side.

Every improvement this week was settled by a paired run and scored with a
shell script. This is that script as a module, used by the /compare page and
by `app.cli ab`.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


def _ts(row, key):
    return datetime.fromisoformat(row[key]) if row[key] else None


def summarize(repo, research_dir: Path, run_id: str) -> dict:
    row = repo.get_run(run_id)
    if row is None:
        raise KeyError(run_id)
    events_path = research_dir / row["dir"] / "events.jsonl"
    ev = []
    if events_path.is_file():
        for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                ev.append(json.loads(line))
            except ValueError:
                continue
    stats = json.loads(row["stats_json"]) if row["stats_json"] else {}
    started, finished = _ts(row, "started_at"), _ts(row, "finished_at")
    minutes = round((finished - started).total_seconds() / 60, 1) if started and finished else None
    findings = [e for e in ev if e["type"] == "finding"]
    post = [e for e in ev if e["type"] == "source_skipped"
            and not e.get("reason", "").startswith(("dropped", "duplicate"))]
    rel = [int(m.group(1)) for e in post if (m := re.search(r"relevance (\d+)/10", e["reason"]))]
    logs = [e["message"] for e in ev if e["type"] == "log"]
    filler = sum(int(m.group(1)) for msg in logs if (m := re.match(r"(\d+) result\(s\) shared no words", msg)))
    rounds = [e for e in ev if e["type"] == "round_start"]
    llm = stats.get("llm", {})
    return {
        "id": run_id, "title": row["title"] or row["query"], "status": row["status"],
        "created_at": row["created_at"], "depth": row["depth"], "minutes": minutes,
        "rounds": len(rounds), "searches": stats.get("searches"),
        "results seen": sum(e.get("results", 0) for e in ev if e["type"] == "searched"),
        "candidates": sum(e.get("candidates", 0) for e in ev if e["type"] == "searched"),
        "filler not fetched": filler,
        # one "triage" event per round since 2026-09-11; older runs carry a
        # source_skipped per dropped page instead
        "dropped at triage": (sum(e.get("dropped", 0) for e in ev if e["type"] == "triage")
                              + sum(1 for e in ev if e["type"] == "source_skipped" and e.get("reason", "").startswith("dropped"))),
        "read, scored 0-1": sum(1 for r in rel if r <= 1),
        "read, scored 2-3": sum(1 for r in rel if 2 <= r <= 3),
        "kept": len(findings),
        "kept 7+": sum(1 for f in findings if f.get("relevance", 0) >= 7),
        "kept, avg relevance": round(sum(f.get("relevance", 0) for f in findings) / len(findings), 2) if findings else None,
        "distinct kept domains": len({f.get("domain") for f in findings}),
        "engines refused": len(stats.get("blocked_engines") or {}),
        "llm calls": llm.get("calls"), "prompt tokens": llm.get("prompt_tokens"),
        "est cost usd": llm.get("est_cost_usd"),
        "quotes removed": stats.get("quotes_unverified", 0),
        "stop reason": next((e.get("stop_reason") for e in ev if e["type"] == "done"), None),
        "_queries": [(e["round"], q, (e.get("scopes") or [""] * len(e["queries"]))[i] if i < len(e.get("scopes") or []) else "")
                     for e in rounds for i, q in enumerate(e["queries"])],
        "_kept": [(f.get("relevance"), f.get("title"), f.get("domain")) for f in findings],
        "_by_engine": dict(Counter(e.get("engine", "?") for e in findings + post).most_common(8)),
    }


COMPARE_KEYS = ["minutes", "rounds", "searches", "results seen", "candidates", "filler not fetched",
                "dropped at triage", "read, scored 0-1", "read, scored 2-3", "kept", "kept 7+",
                "kept, avg relevance", "distinct kept domains", "engines refused", "llm calls",
                "prompt tokens", "est cost usd", "quotes removed", "stop reason"]

# Which direction is better, for the arrow in the table. None = neutral.
HIGHER_IS_BETTER = {"kept": True, "kept 7+": True, "kept, avg relevance": True, "distinct kept domains": True,
                    "read, scored 0-1": False, "read, scored 2-3": False, "engines refused": False,
                    "minutes": False, "llm calls": False, "prompt tokens": False, "est cost usd": False}


def side_by_side(a: dict, b: dict) -> list[dict]:
    rows = []
    for k in COMPARE_KEYS:
        va, vb = a.get(k), b.get(k)
        better = None
        if k in HIGHER_IS_BETTER and isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va != vb:
            better = "b" if (vb > va) == HIGHER_IS_BETTER[k] else "a"
        rows.append({"key": k, "a": va, "b": vb, "better": better})
    return rows


def render_text(a: dict, b: dict, label_a: str = "A", label_b: str = "B") -> str:
    rows = side_by_side(a, b)
    w = max(len(r["key"]) for r in rows)
    out = [f"{'':{w}}  {label_a:>22}  {label_b:>22}"]
    for r in rows:
        mark = "  <" if r["better"] == "a" else ("  >" if r["better"] == "b" else "")
        out.append(f"{r['key']:{w}}  {str(r['a']):>22}  {str(r['b']):>22}{mark}")
    return "\n".join(out)
