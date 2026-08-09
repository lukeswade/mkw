"""SearXNG JSON API client with recency mapping.

SearXNG's time_range only supports day/week/month/year, so the seven UI
recency options map to the nearest engine filter plus a post-filter cutoff
(applied here on engine-reported dates, and again after extraction on the
document's own date).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx

log = logging.getLogger(__name__)

# recency option → SearXNG time_range param (None = omit)
RECENCY_TO_TIME_RANGE: dict[str, str | None] = {
    "week": "week",
    "month": "month",
    "3months": "year",
    "6months": "year",
    "1year": "year",
    "3years": None,
    "all": None,
}

# recency option → post-filter cutoff in days. Engines don't reliably honor
# time_range (verified empirically — a 2009 page came back under "month"),
# so every window gets a deterministic date check on top; undated results
# are still kept and tagged.
RECENCY_CUTOFF_DAYS: dict[str, int | None] = {
    "week": 8,
    "month": 32,
    "3months": 93,
    "6months": 186,
    "1year": 370,
    "3years": 1100,
    "all": None,
}


def cutoff_for(recency: str, now: datetime | None = None) -> datetime | None:
    days = RECENCY_CUTOFF_DAYS.get(recency)
    if days is None:
        return None
    return (now or datetime.now()) - timedelta(days=days)


def categories_for(recency: str) -> str:
    # freshness-focused runs benefit from the news category
    return "general,news" if recency in ("week", "month") else "general"


def parse_published(value: str | None) -> datetime | None:
    """Engine publishedDate → naive local-ish datetime (best effort)."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.replace(tzinfo=None)


class SearxngError(Exception):
    pass


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    engine: str
    published: datetime | None
    score: float
    via_query: str = ""  # the sub-query that surfaced this result


class Searcher:
    def __init__(self, base_url: str, client: httpx.AsyncClient):
        self.base_url = base_url.rstrip("/")
        self.client = client

    async def search(self, query: str, recency: str, *, pageno: int = 1) -> list[SearchResult]:
        params = {
            "q": query,
            "format": "json",
            "language": "en",
            "safesearch": 0,
            "pageno": pageno,
            "categories": categories_for(recency),
        }
        time_range = RECENCY_TO_TIME_RANGE.get(recency)
        if time_range:
            params["time_range"] = time_range

        resp = await self.client.get(
            f"{self.base_url}/search", params=params,
            headers={"Accept": "application/json"},
        )
        if resp.status_code == 403:
            raise SearxngError(
                "SearXNG returned 403 for format=json — the instance must "
                "enable the JSON API: add 'json' under search.formats in "
                "searxng/settings.yml, then restart the searxng container."
            )
        resp.raise_for_status()
        data = resp.json()

        unresponsive = data.get("unresponsive_engines") or []
        if unresponsive:
            log.info("searxng unresponsive engines for %r: %s", query, unresponsive)

        cutoff = cutoff_for(recency)
        out: list[SearchResult] = []
        for item in data.get("results", []):
            url = item.get("url")
            if not url or not str(url).startswith(("http://", "https://")):
                continue
            published = parse_published(item.get("publishedDate"))
            # pre-fetch date filter: drop only when a date is present AND outside
            if cutoff and published and published < cutoff:
                continue
            out.append(SearchResult(
                url=str(url),
                title=(item.get("title") or "").strip() or str(url),
                snippet=(item.get("content") or "").strip(),
                engine=item.get("engine") or "",
                published=published,
                score=float(item.get("score") or 0.0),
            ))
        return out
