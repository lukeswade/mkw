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


# ---- small-index engines get a short twin of every long query ---------------

def test_shorten_query_keeps_the_first_content_words():
    from app.research.searcher import shorten_query
    assert shorten_query("how do I fix a loose reel seat on a fly rod") == "fix loose reel"
    assert shorten_query("2UZ-FE valve cover torque spec") == "2UZ-FE valve cover"
    assert shorten_query("QMK vs VIA") == "QMK VIA"            # short stays short
    assert shorten_query("the the the") == "the the the"        # never returns nothing


@respx.mock
async def test_a_long_query_is_also_sent_short_to_the_small_indexes():
    """Measured: boardreader 0->8, searchmysite 1->10, wiby 0->12 results going
    from seven words to three. The planner writes five to eight."""
    seen = []

    def handler(request):
        seen.append(dict(request.url.params))
        if "engines" in request.url.params:
            return httpx.Response(200, json=_payload([
                {"url": "https://forum.test/t/1", "title": "Thread", "content": "c",
                 "engine": "boardreader"}]))
        return httpx.Response(200, json=_payload([
            {"url": "https://big.test/a", "title": "A", "content": "c",
             "engine": "bing"}]))
    respx.get(f"{BASE}/search").mock(side_effect=handler)
    async with httpx.AsyncClient() as client:
        out = await Searcher(BASE, client).search(
            "how to remove a stuck reel seat from a fly rod blank", "all")

    assert len(seen) == 2
    twin = [p for p in seen if "engines" in p][0]
    assert twin["engines"] == "boardreader,searchmysite,wiby"
    assert len(twin["q"].split()) <= 3
    assert "categories" not in twin, "engines= replaces the category selection"
    assert {r.url for r in out} == {"https://big.test/a", "https://forum.test/t/1"}


@respx.mock
async def test_short_queries_site_queries_and_page_two_get_no_twin():
    route = respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(200, json=_payload([])))
    async with httpx.AsyncClient() as client:
        s = Searcher(BASE, client)
        await s.search("fly rod repair", "all")                       # 3 words
        await s.search("site:charm.li GX470 spark plug gap", "all")   # site:
        await s.search("how to remove a stuck reel seat", "all", pageno=2)
    assert all("engines" not in c.request.url.params for c in route.calls)


@respx.mock
async def test_the_twin_can_be_switched_off_and_its_failure_is_not_fatal():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if "engines" in request.url.params:
            return httpx.Response(500)
        return httpx.Response(200, json=_payload([
            {"url": "https://big.test/a", "title": "A", "content": "c", "engine": "bing"}]))
    respx.get(f"{BASE}/search").mock(side_effect=handler)
    async with httpx.AsyncClient() as client:
        out = await Searcher(BASE, client).search(
            "how to remove a stuck reel seat from a fly rod", "all")
        assert [r.url for r in out] == ["https://big.test/a"]   # main results stand
        calls["n"] = 0
        await Searcher(BASE, client, small_index_engines=frozenset()).search(
            "how to remove a stuck reel seat from a fly rod", "all")
        assert calls["n"] == 1


def test_the_paid_brave_engine_ranks_with_the_general_web_not_as_backfill():
    """braveapi was absent from the general tier: 20 organic results per query
    sorted behind bing's first-word junk, zero candidates from it in any run
    while every search spent a paid request on it."""
    from app.research.searcher import engine_tier, engine_preferred
    assert engine_tier("braveapi") == 0 and engine_preferred("braveapi")
    assert engine_tier("bing") == 0 and not engine_preferred("bing")
    assert engine_tier("crossref") == 1


def test_promoted_video_engines_share_the_first_turn_with_keyed_engines():
    """On a videos run, braveapi and google cse alone filled a 28-slot round
    before the promoted youtube engine placed one candidate."""
    from app.research.searcher import engine_order, VIDEO_ENGINES
    assert engine_order("braveapi") == (0, 0)
    assert engine_order("bing") == (0, 1)
    assert engine_order("youtube") == (1, 1)                        # plain run: backfill
    assert engine_order("youtube", VIDEO_ENGINES) == (0, 0)        # videos run: first turn
    assert engine_order("crossref") == (1, 1)
