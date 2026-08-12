"""A blocked search engine and an empty topic look identical in the data —
telling the user "no sources exist" when the truth is "search was down" is the
worst failure mode this app has, because it reads as a finished answer."""
import httpx
import pytest
import respx

from app.research.searcher import DEFAULT_CATEGORIES, Searcher, categories_for

BASE = "http://sx.test"


def _payload(results, unresponsive=None):
    return {"results": results, "unresponsive_engines": unresponsive or []}


def test_categories_include_non_gated_sources():
    """general alone is four engines that all rate-limit."""
    cats = categories_for("all")
    assert "science" in cats
    assert categories_for("week").endswith(",news")


@respx.mock
async def test_degraded_when_every_search_is_empty_and_engines_blocked():
    respx.get(f"{BASE}/search").mock(return_value=httpx.Response(200, json=_payload(
        [], [["brave", "too many requests"], ["duckduckgo", "CAPTCHA"]])))
    async with httpx.AsyncClient() as client:
        s = Searcher(BASE, client)
        await s.search("q1", "all")
        await s.search("q2", "all")
    assert s.degraded
    assert s.blocked_engines == {"brave": "too many requests",
                                 "duckduckgo": "CAPTCHA"}


@respx.mock
async def test_not_degraded_when_a_topic_is_simply_empty():
    """Empty results with healthy engines is a real answer, not a failure."""
    respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(200, json=_payload([])))
    async with httpx.AsyncClient() as client:
        s = Searcher(BASE, client)
        await s.search("obscure", "all")
    assert not s.degraded


@respx.mock
async def test_not_degraded_when_some_searches_succeed():
    """Partial engine failure is normal and must not be reported as an outage."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        results = [] if calls["n"] == 1 else [
            {"url": "https://example.com/a", "title": "A", "content": "c",
             "engine": "bing"}]
        return httpx.Response(200, json=_payload(
            results, [["brave", "too many requests"]]))

    respx.get(f"{BASE}/search").mock(side_effect=handler)
    async with httpx.AsyncClient() as client:
        s = Searcher(BASE, client)
        await s.search("q1", "all")
        await s.search("q2", "all")
    assert not s.degraded
    assert s.empty_searches == 1 and s.searches == 2


@respx.mock
async def test_searches_are_throttled():
    """Firing every sub-query at once across a dozen engines is what trips the
    rate limits in the first place."""
    import asyncio
    live = {"now": 0, "peak": 0}

    async def handler(request):
        live["now"] += 1
        live["peak"] = max(live["peak"], live["now"])
        await asyncio.sleep(0.05)
        live["now"] -= 1
        return httpx.Response(200, json=_payload([]))

    respx.get(f"{BASE}/search").mock(side_effect=handler)
    async with httpx.AsyncClient() as client:
        s = Searcher(BASE, client, max_concurrent=2)
        await asyncio.gather(*(s.search(f"q{i}", "all") for i in range(6)))
    assert live["peak"] <= 2


@respx.mock
async def test_custom_categories_are_sent():
    route = respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(200, json=_payload([])))
    async with httpx.AsyncClient() as client:
        await Searcher(BASE, client, categories="general").search("q", "all")
    assert route.calls.last.request.url.params["categories"] == "general"


def test_engine_tiers():
    from app.research.searcher import engine_tier
    for general in ("bing", "google cse", "DuckDuckGo", "mojeek", "wikipedia"):
        assert engine_tier(general) == 0, general
    # specialist engines are backfill, not the main course
    for specialist in ("crossref", "openalex", "semantic scholar", "arxiv",
                       "mdn", "docker hub", "google scholar", ""):
        assert engine_tier(specialist) == 1, specialist


def test_it_category_is_excluded_by_default():
    """MDN matched 'enclosure' and 'node' as literal API names and flooded a
    real run with SVGSVGElement.checkEnclosure and AudioNode.channelCountMode."""
    assert "it" not in DEFAULT_CATEGORIES.split(",")
    assert DEFAULT_CATEGORIES.split(",") == ["general", "science"]


def test_general_web_outranks_specialist_but_specialist_survives():
    """Sorting must be stable so every sub-query still contributes."""
    from app.research.searcher import SearchResult, engine_tier

    def r(engine, url):
        return SearchResult(url=url, title=url, snippet="", engine=engine,
                            published=None, score=1.0)

    merged = [r("crossref", "https://doi.org/1"), r("bing", "https://a.com/1"),
              r("openalex", "https://doi.org/2"), r("mojeek", "https://b.com/1")]
    merged.sort(key=lambda x: engine_tier(x.engine))
    assert [x.engine for x in merged] == ["bing", "mojeek", "crossref", "openalex"]
