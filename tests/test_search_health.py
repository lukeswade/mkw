"""A blocked search engine and an empty topic look identical in the data —
telling the user "no sources exist" when the truth is "search was down" is the
worst failure mode this app has, because it reads as a finished answer."""
import httpx
import pytest
import respx

from app.research.pipeline import round_refusals
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
    assert twin["engines"] == "searchmysite,wiby"          # boardreader retired: parsing error on every search
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
    assert engine_order("braveapi")[:2] == (0, 0)
    assert engine_order("bing")[:2] == (0, 1)
    assert engine_order("youtube")[:2] == (1, 1)                        # plain run: backfill
    assert engine_order("youtube", VIDEO_ENGINES)[:2] == (0, 0)        # videos run: first turn
    assert engine_order("crossref")[:2] == (1, 1)


def test_a_scope_narrows_a_query_to_categories_the_run_selected_never_wider():
    """A balloon question hit PubMed, arXiv, Ask Ubuntu and Stack Overflow on
    every round. A query's scope narrows its categories; the run's own
    selection is the ceiling; nothing usable means today's behaviour."""
    from app.research.searcher import categories_for_scope as c
    allowed = "general,science,q&a,videos,social media"
    assert c("web", allowed) == "general"
    assert c("web+video", allowed) == "general,videos"
    assert c("video+social", allowed) == "videos,social media"
    assert c("code", allowed) is None                      # `it` not selected: fall back, don't widen
    assert c("academic", "general") is None
    # no usable scope = a web query, plus video when the run selected it — not every category
    assert c("", allowed) == "general,videos" and c("nonsense", allowed) == "general,videos"
    assert c("", "general,science") == "general"
    assert c("", "science,it") is None                     # general not selected: the run's list stands
    assert c("WEB + Video", allowed) == "general,videos"    # tolerant of case and spacing


async def test_the_searcher_sends_the_narrowed_categories_for_that_query_only():
    import httpx
    from app.research.searcher import Searcher
    seen = []
    async def handler(req):
        seen.append(dict(req.url.params)); return httpx.Response(200, json={"results": [], "unresponsive_engines": []})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    s = Searcher("http://sx", client, categories="general,science,videos", small_index_engines=frozenset())
    await s.search("fly rod grip", "all", categories="videos")
    await s.search("fly rod grip 2", "all")
    assert seen[0]["categories"] == "videos" and seen[1]["categories"] == "general,science,videos"


def test_planner_and_gap_scopes_are_optional_and_tolerant():
    """Tolerance must not be bought with a shift.

    This test used to assert that a null scope was DROPPED, which is exactly
    the defect: the scopes are aligned to subqueries by index, so dropping one
    moves every later scope onto the wrong query. A null now becomes "", which
    downstream reads the same as unspecified, and the pairing survives.
    """
    from app.models import PlannerOut, GapOut
    p = PlannerOut(title="t", brief="b", subqueries=["a", "b"])                      # omitted
    assert p.query_scopes == []
    p = PlannerOut(title="t", brief="b", subqueries=["a", "b"], query_scopes=["Web+Video", None])
    assert p.query_scopes == ["web+video", ""]
    assert dict(zip(p.subqueries, p.query_scopes)) == {"a": "web+video", "b": ""}
    # the case the old assertion was blind to: the null comes FIRST, and
    # dropping it would have handed query "a" the scope meant for "b"
    p = PlannerOut(title="t", brief="b", subqueries=["a", "b"], query_scopes=[None, "Web+Video"])
    assert dict(zip(p.subqueries, p.query_scopes)) == {"a": "", "b": "web+video"}
    g = GapOut(next_queries=["x"], next_query_scopes=["code"])
    assert g.next_query_scopes == ["code"]


def test_engines_are_drawn_in_order_of_what_their_results_have_been_worth():
    from app.research.searcher import engine_order
    yields = {"braveapi": 0.61, "bing": 0.12, "google cse": 0.40}
    order = sorted(["bing", "braveapi", "google cse", "mojeek"], key=lambda e: engine_order(e, yields=yields))
    assert order[:2] == ["braveapi", "google cse"]               # keyed engines first turn, then by yield
    assert order[2:] == ["bing", "mojeek"]                       # a measured 12% still beats no record at all
    assert engine_order("newengine", yields=yields)[2] == 1      # unproven: after every engine with a record
    assert engine_order("bing", yields=yields)[2] == 0
    # the failure this guards: a fresh video engine must not outrank a proven one on a videos run
    from app.research.searcher import VIDEO_ENGINES
    v = {"duckduckgo videos": 0.446, "youtube": 0.526}
    got = sorted(["bing videos", "duckduckgo videos", "youtube"], key=lambda e: engine_order(e, VIDEO_ENGINES, v))
    assert got == ["youtube", "duckduckgo videos", "bing videos"]


def test_no_engine_fills_more_than_its_share_of_a_round():
    """Bing Videos, 60 results a query and serving junk, took all 60 slots of
    a round while DuckDuckGo Videos and YouTube held the answer."""
    from collections import Counter
    from app.research.dedupe import rank_diverse
    from app.research.pipeline import engine_share
    from app.research.searcher import SearchResult
    mk = lambda i, eng: SearchResult(url=f"https://site{i}.com/p", title="t", snippet="",
                                     engine=eng, published=None, score=0.0)
    pool = [mk(i, "bing videos") for i in range(30)] + [mk(100 + i, "duckduckgo videos") for i in range(5)]
    skips = Counter()
    chosen = rank_diverse(pool, set(), per_domain=2, limit=12, per_engine=4, engine_skips=skips)
    assert Counter(c.engine for c in chosen) == {"bing videos": 4, "duckduckgo videos": 4}   # the same share for all
    assert skips == {"bing videos": 26, "duckduckgo videos": 1}
    held = []
    rank_diverse(pool, set(), per_domain=2, limit=12, per_engine=4, held_back=held)
    assert len(held) == 27 and all(h.engine in ("bing videos", "duckduckgo videos") for h in held)
    # results with no attribution are not one source and are never capped as one
    assert len(rank_diverse([mk(i, "") for i in range(10)], set(), per_domain=2, limit=8, per_engine=2)) == 8
    assert engine_share(60) == 20 and engine_share(28) == 10 and engine_share(2) == 1


async def test_a_long_query_also_reaches_youtube_shortened_when_videos_are_in_scope():
    """A balloon question never surfaced 'Balloon Rock Hand Horns': video
    search matches short keyword strings, and the queries read like web
    queries. With videos in scope, the YouTube engine gets the short twin."""
    import httpx
    from app.research.searcher import Searcher
    seen = []
    async def handler(req):
        seen.append(dict(req.url.params)); return httpx.Response(200, json={"results": [], "unresponsive_engines": []})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    s = Searcher("http://sx", client, categories="general,videos", small_index_engines=frozenset())
    await s.search("how to twist a rock on hand sign from a 260 balloon animal tutorial", "all")
    twins = [p for p in seen if "engines" in p]
    assert len(twins) == 1 and twins[0]["engines"] == "youtube"
    assert len(twins[0]["q"].split()) < 8                                     # shortened
    seen.clear()
    s2 = Searcher("http://sx", client, categories="general,science", small_index_engines=frozenset())
    await s2.search("how to twist a rock on hand sign from a 260 balloon animal tutorial", "all")
    assert not [p for p in seen if "engines" in p]                             # videos not in scope: no twin


async def test_only_requests_that_carry_general_count_as_brave_requests():
    """'searches' over-read the Brave dashboard by two thirds: it counted the
    twins sent to named engines and the queries scoped away from general."""
    import httpx
    from app.research.searcher import Searcher
    async def handler(req): return httpx.Response(200, json={"results": [], "unresponsive_engines": []})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    s = Searcher("http://sx", client, categories="general,science,videos", small_index_engines=frozenset({"wiby"}))
    await s.search("how to twist a longhorn balloon animal from a 260 balloon step by step", "all")   # general + a twin
    await s.search("longhorn balloon tutorial", "all", categories="videos")                             # scoped: not Brave
    await s.search("reel seat", "all", categories="general")                                            # general only
    assert s.searches == 4 and s.brave_requests == 2


def test_a_read_source_line_ends_with_its_score_kept_or_not():
    """Asked for on 2026-09-03: the score at the end of every completed
    source record, after the title. Kept lines had it; rejected reads still
    led with it."""
    from app.research.progress import format_event
    read = {"type": "source_skipped", "ts": 0, "url": "https://x.com/v",
            "reason": "relevance 0/10", "title": "The right way to tie a fishing hook"}
    line = format_event(read)
    assert line.endswith('"The right way to tie a fishing hook" · 0/10') and "(relevance" not in line
    unread = dict(read, reason="no caption transcript")
    assert format_event(unread).endswith('(no caption transcript)  "The right way to tie a fishing hook"')


def test_a_refill_is_ordered_by_what_this_run_has_kept_not_by_keyed_first():
    """Brave took a video round's whole refill (69 candidates, 1 kept) because
    keyed engines take the first turn everywhere. In a refill the promoted
    engines go first, then this run's own kept-per-read, then the install's."""
    from app.research.searcher import refill_order, VIDEO_ENGINES
    read = {"braveapi": 20, "youtube": 10, "duckduckgo videos": 4}
    kept = {"braveapi": 1, "youtube": 6, "duckduckgo videos": 1}
    inst = {"braveapi": 0.49, "duckduckgo videos": 0.45, "youtube": 0.2, "bing": 0.3}
    key = lambda e: refill_order(e, VIDEO_ENGINES, read, kept, inst)
    order = sorted(["braveapi", "youtube", "duckduckgo videos", "bing"], key=key)
    assert order == ["youtube", "duckduckgo videos", "braveapi", "bing"]   # promoted first; then run yield
    # nothing read yet this run: promoted engines still first, then the install's record
    fresh = sorted(["braveapi", "duckduckgo videos", "bing"], key=lambda e: refill_order(e, VIDEO_ENGINES, {}, {}, inst))
    assert fresh == ["duckduckgo videos", "braveapi", "bing"]


async def test_a_benched_engine_is_left_out_by_naming_the_others():
    import httpx
    from app.research.searcher import Searcher
    seen = []
    async def handler(req):
        if req.url.path.endswith("/config"):
            return httpx.Response(200, json={"engines": [
                {"name": "braveapi", "categories": ["general", "web"], "enabled": True},
                {"name": "google cse", "categories": ["general", "web"], "enabled": True},
                {"name": "mwmbl", "categories": ["general"], "enabled": True},
                {"name": "arxiv", "categories": ["science"], "enabled": True},
                {"name": "qwant", "categories": ["general"], "enabled": False}]})
        seen.append(dict(req.url.params))
        return httpx.Response(200, json={"results": [], "unresponsive_engines": []})
    class Bench:
        def __init__(self, ex): self.ex = frozenset(ex)
        def excluded(self): return self.ex
        def observe(self, refused, answered): return []
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    s = Searcher("http://sx", client, categories="general,science", small_index_engines=frozenset(), bench=Bench({"google cse"}))
    await s.search("fly rod grip", "all")
    assert "categories" not in seen[0] and seen[0]["engines"] == "arxiv,braveapi,mwmbl"
    s = Searcher("http://sx", client, categories="general,science", small_index_engines=frozenset(), bench=Bench(set()))
    await s.search("fly rod grip", "all")
    assert seen[1]["categories"] == "general,science" and "engines" not in seen[1]
    s = Searcher("http://sx", client, categories="science", small_index_engines=frozenset(), bench=Bench({"google cse"}))
    await s.search("fly rod grip", "all")
    assert seen[2]["categories"] == "science"          # benched engine not in the asked categories: unchanged


def test_a_persistently_blocked_engine_is_benched_then_probed_with_backoff(data_dir):
    from datetime import datetime, timedelta, timezone
    from app.db import Repo, connect
    from app.research.bench import EngineBench, BENCH_AFTER_REFUSALS
    from tests.test_pipeline_e2e import make_cfg
    cfg = make_cfg(data_dir); repo = Repo(connect(cfg.db_path))
    t = [datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)]
    bench = EngineBench(repo, now=lambda: t[0])
    refused = {"google cse": "Suspended: too many requests", "crossref": "timeout"}
    for _ in range(BENCH_AFTER_REFUSALS):           # ten refusals over 27 minutes
        bench.observe(refused, {"braveapi"}); t[0] += timedelta(minutes=3)
    assert bench.excluded() == frozenset()          # not yet half an hour: SearXNG's own retry still applies
    events = bench.observe(refused, {"braveapi"})   # 30 min in
    assert bench.excluded() == frozenset({"google cse"})   # crossref timed out; that never benches
    assert any("benched 6h" in e for e in events)
    t[0] += timedelta(hours=6, minutes=1)
    assert bench.excluded() == frozenset()          # the probe is allowed
    events = bench.observe({"google cse": "too many requests"}, set())
    assert bench.excluded() == frozenset({"google cse"}) and any("12h more" in e for e in events)
    t[0] += timedelta(hours=12, minutes=1)
    events = bench.observe({}, {"google cse", "braveapi"})
    assert bench.excluded() == frozenset() and any("leaves the bench" in e for e in events)
    row = {r["engine"]: dict(r) for r in repo.engine_bench_all()}["google cse"]
    assert row["strikes"] == 0 and row["refusals"] == 0


@respx.mock
async def test_a_refusal_is_reported_in_its_own_round_only():
    """A GitHub 503 in round one was re-announced as "refused this round" in
    rounds two and three (2026-09-05): blocked_engines lives for the run."""
    route = respx.get(f"{BASE}/search")
    route.side_effect = [
        httpx.Response(200, json=_payload([], [["github code", "HTTP error"]])),
        httpx.Response(200, json=_payload([{"url": "https://a.example/x", "title": "x",
                                            "content": "c", "engines": ["brave"]}])),
    ]
    async with httpx.AsyncClient() as client:
        s = Searcher(BASE, client)
        before_round_1 = s.searches
        await s.search("q1", "all")
        assert round_refusals(s, before_round_1) == {"github code": "HTTP error"}
        before_round_2 = s.searches
        await s.search("q2", "all")
    assert s.blocked_engines == {"github code": "HTTP error"}   # run-wide memory kept
    assert round_refusals(s, before_round_2) == {}               # but not re-announced


def test_a_searcher_without_timing_reports_everything():
    class Feeds:
        blocked_engines = {"https://dead.example/feed": "ConnectError"}
    assert round_refusals(Feeds(), 5) == Feeds.blocked_engines


# ---- a dated search must not silently shed engines --------------------------

def _config(engines):
    """SearXNG /config shape: name, enabled, categories, time_range_support."""
    return {"engines": [
        {"name": n, "enabled": True, "categories": ["general"],
         "time_range_support": tr} for n, tr in engines]}


def _capture():
    """Record every outbound /search request and answer each with one result."""
    seen = []

    def handler(req):
        seen.append(dict(req.url.params))
        return httpx.Response(200, json=_payload(
            [{"url": f"https://e.test/{len(seen)}", "title": "t", "content": "c",
              "engine": "x"}]))
    return seen, handler


@respx.mock
async def test_a_dated_search_asks_the_undated_engines_too():
    """SearXNG drops every engine without time_range_support from a search
    carrying a time range. On this install that was five of seven general
    engines, so every run with a recency other than "all time" ran on
    braveapi and google cse alone — and the indie indexes that exist to
    answer when the majors CAPTCHA were silently absent."""
    respx.get(f"{BASE}/config").mock(return_value=httpx.Response(200, json=_config(
        [("google cse", True), ("braveapi", True),
         ("bing", False), ("marginalia", False)])))
    seen, handler = _capture()
    respx.get(f"{BASE}/search").mock(side_effect=handler)

    async with httpx.AsyncClient() as client:
        s = Searcher(BASE, client)
        out = await s._query("voice cloning", "1year")

    assert len(seen) == 2, "expected one dated request and one undated"
    dated = [r for r in seen if "time_range" in r]
    undated = [r for r in seen if "time_range" not in r]
    assert len(dated) == 1 and len(undated) == 1
    assert set(dated[0]["engines"].split(",")) == {"braveapi", "google cse"}
    assert set(undated[0]["engines"].split(",")) == {"bing", "marginalia"}
    # still ONE logical search, and both halves' results come back
    assert s.searches == 1
    assert len(out) == 2


@respx.mock
async def test_an_undated_search_is_a_single_request_as_before():
    respx.get(f"{BASE}/config").mock(return_value=httpx.Response(200, json=_config(
        [("google cse", True), ("bing", False)])))
    seen, handler = _capture()
    respx.get(f"{BASE}/search").mock(side_effect=handler)
    async with httpx.AsyncClient() as client:
        s = Searcher(BASE, client)
        await s._query("voice cloning", "all")
    assert len(seen) == 1
    assert "time_range" not in seen[0]
    assert "categories" in seen[0]        # unchanged: search by category


@respx.mock
async def test_no_split_when_every_engine_can_date_filter():
    respx.get(f"{BASE}/config").mock(return_value=httpx.Response(200, json=_config(
        [("google cse", True), ("braveapi", True)])))
    seen, handler = _capture()
    respx.get(f"{BASE}/search").mock(side_effect=handler)
    async with httpx.AsyncClient() as client:
        s = Searcher(BASE, client)
        await s._query("voice cloning", "1year")
    assert len(seen) == 1 and seen[0]["time_range"] == "year"


@respx.mock
async def test_when_nothing_can_date_filter_the_range_is_dropped_not_the_engines():
    """Sending the range to a pool where no engine supports it returns
    nothing at all — which is how this was failing. Our own cutoff still
    drops results dated outside the window."""
    respx.get(f"{BASE}/config").mock(return_value=httpx.Response(200, json=_config(
        [("bing", False), ("marginalia", False)])))
    seen, handler = _capture()
    respx.get(f"{BASE}/search").mock(side_effect=handler)
    async with httpx.AsyncClient() as client:
        s = Searcher(BASE, client)
        out = await s._query("voice cloning", "1year")
    assert len(seen) == 1
    assert "time_range" not in seen[0]
    assert out


@respx.mock
async def test_a_split_search_that_answers_on_one_half_is_not_an_empty_search():
    """empty_searches drives the degraded banner. A split search is still
    one search, so a half that answers must not be averaged away — nor may
    the quiet half count as its own failure."""
    respx.get(f"{BASE}/config").mock(return_value=httpx.Response(200, json=_config(
        [("google cse", True), ("bing", False)])))

    def handler(req):
        if "time_range" in dict(req.url.params):
            return httpx.Response(200, json=_payload([]))        # dated half: nothing
        return httpx.Response(200, json=_payload(
            [{"url": "https://e.test/1", "title": "t", "content": "c", "engine": "bing"}]))
    respx.get(f"{BASE}/search").mock(side_effect=handler)

    async with httpx.AsyncClient() as client:
        s = Searcher(BASE, client)
        out = await s._query("voice cloning", "1year")
    assert s.searches == 1
    assert s.empty_searches == 0
    assert len(out) == 1
