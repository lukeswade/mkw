"""Pre-flight estimate for a run, before you commit to it.

Depth is an abstract dial — "7" tells you nothing about whether you are about
to wait two minutes or forty. The caps in pipeline.py bound the work, and your
own completed runs say how long a source actually takes on your hardware and
model, so the estimate calibrates itself as you use the app.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from app.research.pipeline import (breadth_for_depth, candidates_per_round,
                                   max_docs_for_depth,
                                   max_llm_calls_for_depth, rounds_for_depth)

# Fallbacks until this install has completed runs to learn from. Seconds per
# kept source and per LLM call, measured against a mid-size cloud model.
DEFAULT_SECONDS_PER_SOURCE = 12.0
MIN_SAMPLES = 2
# Typically a bit over half of fetched candidates survive relevance filtering,
# robots, dead links and the recency window.
KEEP_RATE = 0.55


@dataclass
class Estimate:
    depth: int
    queries: int
    sources_low: int
    sources_high: int
    llm_calls: int
    seconds: float
    cost_usd: float | None = None
    calibrated: bool = False
    samples: int = 0

    @property
    def duration_label(self) -> str:
        m = self.seconds / 60
        if self.depth == 0:
            return "seconds"
        if m < 1.5:
            return "about a minute"
        if m < 60:
            return f"about {round(m)} min"
        hours = m / 60
        return f"about {hours:.1f} h"

    @property
    def sources_label(self) -> str:
        if self.depth == 0:
            return "no web search"
        return f"{self.sources_low}–{self.sources_high} sources"


def _history(repo) -> tuple[float | None, float | None, int]:
    """(seconds per source, cost per source, sample count) from real runs."""
    rows = repo.conn.execute(
        "SELECT stats_json, started_at, finished_at FROM runs"
        " WHERE status = 'completed' AND stats_json IS NOT NULL"
        "   AND started_at IS NOT NULL AND finished_at IS NOT NULL"
        " ORDER BY created_at DESC LIMIT 20").fetchall()
    from datetime import datetime

    secs: list[float] = []
    costs: list[float] = []
    for r in rows:
        try:
            stats = json.loads(r["stats_json"])
            kept = int(stats.get("sources_kept") or 0)
            if kept < 1:
                continue
            began = datetime.fromisoformat(r["started_at"])
            ended = datetime.fromisoformat(r["finished_at"])
            elapsed = (ended - began).total_seconds()
            if elapsed <= 0:
                continue
            secs.append(elapsed / kept)
            cost = (stats.get("llm") or {}).get("est_cost_usd")
            if cost:
                costs.append(float(cost) / kept)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue

    def median(xs: list[float]) -> float | None:
        if not xs:
            return None
        xs = sorted(xs)
        mid = len(xs) // 2
        return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2

    return median(secs), median(costs), len(secs)


def estimate_run(repo, depth: int) -> Estimate:
    depth = max(0, min(10, depth))
    if depth == 0:
        return Estimate(depth=0, queries=0, sources_low=0, sources_high=0,
                        llm_calls=1, seconds=20.0)

    breadth = breadth_for_depth(depth)
    rounds = rounds_for_depth(depth)
    queries = breadth * rounds
    # The caps are ceilings; a typical run saturates before reaching them.
    cap = max_docs_for_depth(depth)
    fetched = min(cap, candidates_per_round(breadth) * rounds)
    likely = int(fetched * KEEP_RATE)
    sources_low = max(2, int(likely * 0.5))
    sources_high = max(sources_low + 1, min(cap, likely))
    llm_calls = min(max_llm_calls_for_depth(depth),
                    3 + rounds * 2 + sources_high)

    per_source, cost_per_source, samples = _history(repo)
    calibrated = per_source is not None and samples >= MIN_SAMPLES
    rate = per_source if calibrated else DEFAULT_SECONDS_PER_SOURCE
    midpoint = (sources_low + sources_high) / 2
    seconds = midpoint * rate

    cost = None
    if calibrated and cost_per_source:
        cost = round(midpoint * cost_per_source, 3)

    return Estimate(depth=depth, queries=queries, sources_low=sources_low,
                    sources_high=sources_high, llm_calls=llm_calls,
                    seconds=seconds, cost_usd=cost, calibrated=calibrated,
                    samples=samples)
