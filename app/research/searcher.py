"""SearXNG JSON API client with recency mapping.

SearXNG's time_range only supports day/week/month/year, so the seven UI
recency options map to the nearest engine filter plus a post-filter cutoff
(applied here on engine-reported dates, and again after extraction on the
document's own date).
"""
from __future__ import annotations

import asyncio
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


# The stock `general` category is four gate-happy engines (Google, Brave,
# DuckDuckGo, Startpage). Including science and it reaches Crossref, OpenAlex,
# Semantic Scholar, arXiv, Stack Overflow and GitHub — better research sources
# that also do not CAPTCHA, so a run still works when the big engines throttle.
DEFAULT_CATEGORIES = "general,science,it"


def categories_for(recency: str, base: str = DEFAULT_CATEGORIES) -> str:
    # freshness-focused runs additionally benefit from the news category
    return f"{base},news" if recency in ("week", "month") else base


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
    def __init__(self, base_url: str, client: httpx.AsyncClient,
                 categories: str = DEFAULT_CATEGORIES,
                 max_concurrent: int = 2, timeout: float = 45.0):
        self.base_url = base_url.rstrip("/")
        self.client = client
        self.categories = categories or DEFAULT_CATEGORIES
        # Searches need a longer budget than page fetches: SearXNG fans one
        # query out to a dozen-plus engines and waits for the slow ones. The
        # shared 15s client timeout was killing multi-category queries.
        self.timeout = timeout
        # A round fires every sub-query at once, and each SearXNG query fans
        # out to a dozen engines. Firing five of those simultaneously is what
        # trips the rate limits in the first place.
        self._sem = asyncio.Semaphore(max(1, max_concurrent))
        # Observability for the "search itself is broken" case, which otherwise
        # looks identical to "this topic has no sources".
        self.searches = 0
        self.empty_searches = 0
        self.blocked_engines: dict[str, str] = {}

    @property
    def degraded(self) -> bool:
        """Every search came back empty and engines were reporting blocks."""
        return (self.searches > 0 and self.empty_searches == self.searches
                and bool(self.blocked_engines))

    async def search(self, query: str, recency: str, *, pageno: int = 1) -> list[SearchResult]:
        params = {
            "q": query,
            "format": "json",
            "language": "en",
            "safesearch": 0,
            "pageno": pageno,
            "categories": categories_for(recency, self.categories),
        }
        time_range = RECENCY_TO_TIME_RANGE.get(recency)
        if time_range:
            params["time_range"] = time_range

        async with self._sem:
            resp = await self.client.get(
                f"{self.base_url}/search", params=params,
                headers={"Accept": "application/json"},
                timeout=self.timeout,
            )
        if resp.status_code == 403:
            raise SearxngError(
                "SearXNG returned 403 for format=json — the instance must "
                "enable the JSON API: add 'json' under search.formats in "
                "searxng/settings.yml, then restart the searxng container."
            )
        resp.raise_for_status()
        data = resp.json()

        self.searches += 1
        unresponsive = data.get("unresponsive_engines") or []
        for entry in unresponsive:
            if isinstance(entry, (list, tuple)) and entry:
                self.blocked_engines[str(entry[0])] = str(entry[-1])
        if unresponsive:
            log.info("searxng unresponsive engines for %r: %s", query, unresponsive)
        if not data.get("results"):
            self.empty_searches += 1

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
