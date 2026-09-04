"""Engines a network-level block has taken out sit on the app's bench.

SearXNG suspends a refusing engine for 180 s and retries. Against a block
tied to the address — Google's "unusual traffic from your network", a
CAPTCHA wall, an access denial — every retry refreshes the block, and the
engine is re-poked once a search for days. Google CSE, the second-best
engine on this install, was lost that way on 2026-09-04.

The bench notices the pattern (the same engine refused in every search for
half an hour or more), leaves it out of the app's queries for six hours,
then lets it back in as its own probe: an answer clears it; another refusal
doubles the sit-out, up to two days. Timeouts and HTTP errors never qualify:
those are the engine having a bad minute, and SearXNG's 180 s handles them.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

BENCH_AFTER_REFUSALS = 10
BENCH_AFTER_MINUTES = 30
BENCH_BASE_HOURS = 6
BENCH_MAX_HOURS = 48
BENCH_REASON = re.compile(r"too many requests|captcha|access denied|unusual traffic", re.I)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _parse(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


class EngineBench:
    def __init__(self, repo, now=None):
        self.repo = repo
        self._now = now or (lambda: datetime.now(timezone.utc))

    def excluded(self) -> frozenset[str]:
        """Engines to leave out of the next query."""
        now = self._now()
        out = set()
        for row in self.repo.engine_bench_all():
            until = _parse(row["benched_until"])
            if until and until > now:
                out.add(row["engine"])
        return frozenset(out)

    def observe(self, refused: dict[str, str], answered: set[str]) -> list[str]:
        """Record one search: which engines refused (name → reason) and which
        returned results. Returns log-worthy events."""
        now = self._now()
        events: list[str] = []
        rows = {r["engine"]: dict(r) for r in self.repo.engine_bench_all()}
        for engine in answered:
            row = rows.get(engine)
            if row and (row["refusals"] or row["benched_until"]):
                if row["benched_until"]:
                    events.append(f"{engine} answers again and leaves the bench")
                self.repo.engine_bench_upsert(engine, refusals=0, first_refused=None,
                                              last_reason=None, benched_until=None, strikes=0)
        for engine, reason in refused.items():
            if not BENCH_REASON.search(reason or ""):
                continue
            row = rows.get(engine) or {"refusals": 0, "first_refused": None,
                                       "benched_until": None, "strikes": 0}
            until = _parse(row["benched_until"])
            if until and until > now:
                continue                      # still benched; nothing new
            if until:                         # the probe after a sit-out failed
                strikes = int(row["strikes"]) + 1
                hours = min(BENCH_BASE_HOURS * 2 ** (strikes - 1), BENCH_MAX_HOURS)
                self.repo.engine_bench_upsert(
                    engine, refusals=int(row["refusals"]) + 1, first_refused=row["first_refused"],
                    last_reason=reason, benched_until=_iso(now + timedelta(hours=hours)), strikes=strikes)
                events.append(f"{engine} still refuses ({reason}): benched {hours}h more")
                continue
            refusals = int(row["refusals"]) + 1
            first = _parse(row["first_refused"]) or now
            streak_min = int((now - first).total_seconds() // 60)
            if refusals >= BENCH_AFTER_REFUSALS and streak_min >= BENCH_AFTER_MINUTES:
                self.repo.engine_bench_upsert(
                    engine, refusals=refusals, first_refused=_iso(first), last_reason=reason,
                    benched_until=_iso(now + timedelta(hours=BENCH_BASE_HOURS)), strikes=1)
                events.append(f"{engine} has refused every search for {streak_min} min "
                              f"({reason}): benched {BENCH_BASE_HOURS}h, then tried once")
            else:
                self.repo.engine_bench_upsert(
                    engine, refusals=refusals, first_refused=_iso(first), last_reason=reason,
                    benched_until=None, strikes=int(row["strikes"]))
        return events
