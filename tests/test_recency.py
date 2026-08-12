from datetime import datetime, timedelta

import httpx
import pytest
import respx

from app.research.searcher import (
    RECENCY_CUTOFF_DAYS,
    RECENCY_TO_TIME_RANGE,
    Searcher,
    SearxngError,
    categories_for,
    cutoff_for,
    parse_published,
)

BASE = "http://sx.test"


def test_recency_tables_cover_all_options():
    options = {"week", "month", "3months", "6months", "1year", "3years", "all"}
    assert set(RECENCY_TO_TIME_RANGE) == options
    assert set(RECENCY_CUTOFF_DAYS) == options
    assert RECENCY_TO_TIME_RANGE["week"] == "week"
    assert RECENCY_TO_TIME_RANGE["3months"] == "year"
    assert RECENCY_TO_TIME_RANGE["3years"] is None
    assert RECENCY_CUTOFF_DAYS["6months"] == 186
    # engines don't reliably honor time_range → every window has a cutoff
    assert all(RECENCY_CUTOFF_DAYS[r] for r in options - {"all"})


def test_cutoff_for():
    now = datetime(2026, 8, 9)
    assert cutoff_for("all", now) is None
    assert cutoff_for("week", now) == now - timedelta(days=8)
    assert cutoff_for("3months", now) == now - timedelta(days=93)


def test_categories_for():
    # science/it reach engines that don't CAPTCHA and are better research
    # sources; news is added only where freshness is the point
    assert categories_for("week").endswith(",news")
    assert "science" in categories_for("1year")
    assert categories_for("1year") == "general,science"


def test_parse_published():
    assert parse_published("2026-01-02T10:00:00Z") == datetime(2026, 1, 2, 10)
    assert parse_published("2026-01-02T10:00:00+02:00").hour == 10
    assert parse_published("not a date") is None
    assert parse_published(None) is None


@respx.mock
async def test_search_prefilters_dated_results():
    old = (datetime.now() - timedelta(days=400)).isoformat()
    respx.get(f"{BASE}/search").mock(return_value=httpx.Response(200, json={
        "results": [
            {"url": "https://a.com/new", "title": "New", "content": "c",
             "engine": "g", "publishedDate": datetime.now().isoformat()},
            {"url": "https://a.com/old", "title": "Old", "content": "c",
             "engine": "g", "publishedDate": old},
            {"url": "https://a.com/undated", "title": "Undated", "content": "c",
             "engine": "g", "publishedDate": None},
            {"url": "ftp://a.com/bad-scheme", "title": "Bad", "content": "c",
             "engine": "g"},
        ],
        "unresponsive_engines": [["brave", "too many requests"]],
    }))
    async with httpx.AsyncClient() as client:
        results = await Searcher(BASE, client).search("q", "6months")
    urls = [r.url for r in results]
    assert "https://a.com/new" in urls
    assert "https://a.com/undated" in urls      # undated kept, tagged later
    assert "https://a.com/old" not in urls      # dated outside window dropped
    assert all(u.startswith("http") for u in urls)


@respx.mock
async def test_search_time_range_param():
    route = respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(200, json={"results": []}))
    async with httpx.AsyncClient() as client:
        await Searcher(BASE, client).search("q", "month")
        assert route.calls.last.request.url.params["time_range"] == "month"
        await Searcher(BASE, client).search("q", "all")
        assert "time_range" not in route.calls.last.request.url.params


@respx.mock
async def test_search_403_gives_settings_hint():
    respx.get(f"{BASE}/search").mock(return_value=httpx.Response(403))
    async with httpx.AsyncClient() as client:
        with pytest.raises(SearxngError, match="formats"):
            await Searcher(BASE, client).search("q", "all")
