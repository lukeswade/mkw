"""A vendor's billing cycle, anchored on a day of the month.

Brave's Search API bills on a monthly cycle starting the day the plan began
(here the 31st), not on the calendar month. The current cycle started on the
most recent occurrence of that day, clamped to the month's length.
"""
from __future__ import annotations

import calendar
from datetime import datetime, timezone


def cycle_start(anchor_day: int, now: datetime | None = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    day = max(1, min(31, int(anchor_day or 1)))
    y, m = now.year, now.month
    start_day = min(day, calendar.monthrange(y, m)[1])
    start = now.replace(day=start_day, hour=0, minute=0, second=0, microsecond=0)
    if start > now:
        y, m = (y, m - 1) if m > 1 else (y - 1, 12)
        start = now.replace(year=y, month=m, day=min(day, calendar.monthrange(y, m)[1]),
                            hour=0, minute=0, second=0, microsecond=0)
    return start
