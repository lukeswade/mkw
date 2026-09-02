"""Display-time formatting.

Everything is stored in UTC. People read clocks in their own zone: run lists,
the static run log and the CLI all printed UTC (the container's clock), while
the live log in the browser printed the viewer's local time — the same run
showed two different clocks. One configured zone, applied on every
server-rendered timestamp.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_TZ = "America/Chicago"
_cache: tuple[float, str] = (0.0, DEFAULT_TZ)


def zone(name: str | None) -> ZoneInfo:
    try:
        return ZoneInfo(name or DEFAULT_TZ)
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        return ZoneInfo(DEFAULT_TZ)


def current() -> str:
    """The configured zone name, re-read from settings at most once a minute
    so a change on the Settings page shows up without a restart."""
    global _cache
    at, name = _cache
    if time.monotonic() - at > 60:
        try:
            from app.config import load_settings
            name = getattr(load_settings(), "display_timezone", "") or DEFAULT_TZ
        except Exception:
            name = name or DEFAULT_TZ
        _cache = (time.monotonic(), name)
    return name


def local_time(ts: float, tz: str | None = None, fmt: str = "%H:%M:%S") -> str:
    return datetime.fromtimestamp(ts or 0, timezone.utc).astimezone(zone(tz or current())).strftime(fmt)


def local_iso(iso: str | None, tz: str | None = None, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """An ISO-8601 UTC stamp (with offset, or naive meaning UTC) -> local."""
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except ValueError:
        return str(iso)[:16].replace("T", " ")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(zone(tz or current())).strftime(fmt)
