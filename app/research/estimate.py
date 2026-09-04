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
# Fallback keep rate. It was the whole estimate once, and it was wrong by a
# wide margin: measured across this install's real runs the fraction of
# fetched candidates that survive ranged from 2% to 58%, and at depth 10 the
# estimate promised 23-46 sources against a median actual of 14. Now used
# only until the install has its own history to measure from.
KEEP_RATE = 0.55
HISTORY_RUNS = 40
# The pipeline changes; a fortnight ago a depth-5 run took 84 minutes and
# this week it takes 22. When the last two weeks hold enough runs, they are
# the history; older runs only fill in when they do not.
HISTORY_DAYS = 14


@dataclass
class Estimate:
    depth: int
    queries: int
    sources_low: int
    sources_high: int
    llm_calls: int
    seconds: float
    cost_usd: float | None = None
    calibrated: bool = False          # timing came from real runs
    samples: int = 0                  # runs that supplied timing
    keep_rate: float = KEEP_RATE      # measured when there is history
    depth_samples: int = 0            # runs at THIS depth that set the band

    @property
    def depth_calibrated(self) -> bool:
        """The source band is this depth's own history, not a formula."""
        return self.depth_samples >= MIN_SAMPLES

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
        if self.sources_low == self.sources_high:
            return f"~{self.sources_low} sources"
        return f"{self.sources_low}–{self.sources_high} sources"


@dataclass
class _History:
    secs_per_source: float | None
    cost_per_source: float | None
    samples: int                      # runs with usable timing
    keep_rate: float | None           # median kept/(kept+skipped), or None
    kept_at_depth: list[int] = field(default_factory=list)


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2


def _percentile(xs: list[int], p: float) -> int:
    """Nearest-rank on a sorted list; small samples want no interpolation."""
    idx = int(round(p / 100 * (len(xs) - 1)))
    return xs[max(0, min(len(xs) - 1, idx))]


def _history(repo, depth: int) -> _History:
    """What this install's own completed research runs say.

    Three things are learned: seconds and cost per kept source (for the
    time and money lines), the fraction of fetched candidates that actually
    survived (replacing the KEEP_RATE guess), and how many sources runs AT
    THIS DEPTH really kept — which is the honest predictor for the band, since
    real outcomes vary far more by topic than any formula captures.
    """
    rows = repo.conn.execute(
        "SELECT depth, stats_json, started_at, finished_at FROM runs"
        " WHERE status = 'completed' AND stats_json IS NOT NULL"
        "   AND kind = 'research' AND depth > 0"
        " ORDER BY created_at DESC LIMIT ?", (HISTORY_RUNS,)).fetchall()
    from datetime import datetime, timedelta, timezone
    cutoff = (datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)).isoformat()

    def _recent(r) -> bool:
        return bool(r["started_at"]) and r["started_at"] >= cutoff

    recent = [r for r in rows if _recent(r)]
    if sum(1 for r in recent if r["finished_at"]) >= MIN_SAMPLES:
        rows = recent

    secs: list[float] = []
    costs: list[float] = []
    rates: list[float] = []
    at_depth: list[int] = []
    for r in rows:
        try:
            stats = json.loads(r["stats_json"])
            kept = int(stats.get("sources_kept") or 0)
            if kept < 1:
                continue
            skipped = stats.get("sources_skipped")
            if skipped is not None:
                rates.append(kept / (kept + int(skipped)))
            if r["depth"] == depth:
                at_depth.append(kept)
            if not r["started_at"] or not r["finished_at"]:
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

    return _History(
        secs_per_source=_median(secs),
        cost_per_source=_median(costs),
        samples=len(secs),
        keep_rate=_median(rates) if len(rates) >= MIN_SAMPLES else None,
        kept_at_depth=at_depth,
    )


def estimate_run(repo, depth: int) -> Estimate:
    depth = max(0, min(10, depth))
    if depth == 0:
        return Estimate(depth=0, queries=0, sources_low=0, sources_high=0,
                        llm_calls=1, seconds=20.0)

    breadth = breadth_for_depth(depth)
    rounds = rounds_for_depth(depth)
    queries = breadth * rounds
    cap = max_docs_for_depth(depth)
    hist = _history(repo, depth)
    keep_rate = hist.keep_rate if hist.keep_rate is not None else KEEP_RATE

    if len(hist.kept_at_depth) >= MIN_SAMPLES:
        # The middle half of what runs at this depth actually kept. A wide
        # band is the truth here, not a failure of the estimate.
        xs = sorted(hist.kept_at_depth)
        sources_low = max(1, min(cap, _percentile(xs, 25)))
        sources_high = max(sources_low, min(cap, _percentile(xs, 75)))
    else:
        # No history at this depth: the formula, with the measured keep rate
        # once there is one. The caps are ceilings a typical run never
        # reaches, so the band sits well below them.
        fetched = min(cap, candidates_per_round(breadth) * rounds)
        likely = int(fetched * keep_rate)
        sources_low = max(2, int(likely * 0.5))
        sources_high = max(sources_low + 1, min(cap, likely))
    llm_calls = min(max_llm_calls_for_depth(depth),
                    3 + rounds * 2 + sources_high)

    calibrated = hist.secs_per_source is not None and hist.samples >= MIN_SAMPLES
    rate = hist.secs_per_source if calibrated else DEFAULT_SECONDS_PER_SOURCE
    midpoint = (sources_low + sources_high) / 2
    seconds = midpoint * rate

    cost = None
    if calibrated and hist.cost_per_source:
        cost = round(midpoint * hist.cost_per_source, 3)

    return Estimate(depth=depth, queries=queries, sources_low=sources_low,
                    sources_high=sources_high, llm_calls=llm_calls,
                    seconds=seconds, cost_usd=cost, calibrated=calibrated,
                    samples=hist.samples, keep_rate=keep_rate,
                    depth_samples=len(hist.kept_at_depth))
