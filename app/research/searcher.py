"""SearXNG JSON API client with recency mapping.

SearXNG's time_range only supports day/week/month/year, so the seven UI
recency options map to the nearest engine filter plus a post-filter cutoff
(applied here on engine-reported dates, and again after extraction on the
document's own date).
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import urlsplit

import httpx

log = logging.getLogger(__name__)

# recency option → SearXNG time_range param (None = omit)
_SITE_RE = re.compile(r"\bsite:([A-Za-z0-9.-]+)")


def _site_scope(query: str) -> str | None:
    m = _SITE_RE.search(query)
    return m.group(1).lower().strip(".").removeprefix("www.") if m else None


def _in_site(url: str, site: str) -> bool:
    host = urlsplit(url).netloc.lower().split(":")[0].removeprefix("www.")
    return host == site or host.endswith("." + site)


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
# DuckDuckGo, Startpage); when they all throttle at once a run finds nothing.
# Adding `science` reaches Crossref, OpenAlex, Semantic Scholar and arXiv,
# which do not CAPTCHA and are good sources — but only as backfill, see
# engine_tier: a practical question should not be answered out of a journal.
# `it` is deliberately excluded: MDN and Docker Hub match generic words like
# "node" and "enclosure" and flood the candidate pool with noise.
DEFAULT_CATEGORIES = "general,science"
# What the New page and Settings offer. q&a is stackoverflow + askubuntu +
# superuser, measured working. Categories SearXNG advertises but has no
# enabled engine for (books, blogs, apps, shopping, movies) are absent on
# purpose: they are dead switches.
CATEGORY_OPTIONS = ("general", "science", "it", "q&a", "news", "videos",
                    "social media", "files")


def split_categories(value: str) -> list[str]:
    """'general, science' -> ['general', 'science']; order kept, blanks dropped."""
    return [c.strip() for c in (value or "").split(",") if c.strip()]


def category_options(configured: str) -> list[str]:
    """The standard list, plus anything the config names that is not in it,
    so a hand-typed category survives a round trip through the checkboxes."""
    extra = [c for c in split_categories(configured) if c not in CATEGORY_OPTIONS]
    return list(CATEGORY_OPTIONS) + extra

# General-web engines. Everything else (academic, code, Q&A) is backfill that
# only gets picked once these have had their turn.
_GENERAL_WEB_ENGINES = frozenset({
    "bing", "google", "google cse", "duckduckgo", "brave", "startpage",
    "mojeek", "qwant", "yahoo", "wikipedia", "wikidata", "presearch",
    "marginalia", "mullvad leta", "mwmbl",
    # The API engine that a BRAVE_API_KEY turns on. It was missing here, so
    # its 20 organic results per query sorted behind bing's first-word junk
    # as "backfill": zero candidates from it in any run since engine logging
    # began, while every search spent a paid request on it.
    "braveapi",
})
# Keyed engines return real organic results for the query as written; the
# scrapers' degraded, cookie-less sessions do not. Within the general tier
# these take their turn first.
_PREFERRED_ENGINES = frozenset({"braveapi", "marginalia", "google cse"})


def _named(params: dict, engines: list[str]) -> dict:
    """The same request, narrowed to named engines. `engines` replaces the
    category selection, as SearXNG itself does."""
    out = {k: v for k, v in params.items() if k != "categories"}
    out["engines"] = ",".join(engines)
    return out


def engine_preferred(engine: str) -> bool:
    return (engine or "").strip().lower() in _PREFERRED_ENGINES


# An engine with no record yet (fewer reads than Repo.engine_yields counts)
# takes its turn after every engine that has one. It used to sort as if half
# its results were kept, which on 2026-09-04 put Bing Videos (4 reads) ahead
# of DuckDuckGo Videos (65 reads, 45% kept) on the round Bing was serving
# junk — 420 junk results took all 60 slots. A default never beats a
# measurement.


def engine_order(engine: str, promote: frozenset[str] = frozenset(),
                 yields: dict[str, float] | None = None) -> tuple[int, int, int, float]:
    """Sort key: (tier, turn, unproven, -yield). Keyed engines AND the
    engines the run promoted (video engines on a videos run) take the first
    turn within their tier — without that, two keyed engines' 39 results
    filled a 28-slot round before the promoted YouTube engine got a single
    candidate in. Within a turn, engines with a record come before engines
    without one, and among those with a record the ones whose results have
    turned into kept sources more often (the install's own last fortnight)
    come first."""
    e = (engine or "").strip().lower()
    first = engine_preferred(e) or e in promote
    y = (yields or {}).get(e)
    return (engine_tier(e, promote), 0 if first else 1,
            0 if y is not None else 1, -round(y or 0.0, 3))


def refill_order(engine: str, promote: frozenset[str],
                 run_read: dict[str, int], run_kept: dict[str, int],
                 yields: dict[str, float] | None = None) -> tuple[int, int, int, float]:
    """Sort key for a round's refill (what the engine share cap held back).
    The engines the run asked for (promoted) go first; keyed engines get no
    first turn here — Brave took a video round's whole refill and kept 1 of
    69. Then by what each engine has kept in THIS run, engines with reads
    this run ahead of those without, which fall back to the install's
    record."""
    e = (engine or "").strip().lower()
    read = int(run_read.get(e, 0))
    y = (run_kept.get(e, 0) / read) if read else (yields or {}).get(e)
    return (engine_tier(e, promote), 0 if e in promote else 1,
            0 if read else 1, -round(y or 0.0, 3))


# Video engines. Normally tier 1 (a video is a worse answer than a page for
# most questions), but when a run explicitly selects the videos category the
# user asked for video — relegating it below every web result then makes the
# category useless, which is exactly what happened on a how-to run.
VIDEO_ENGINES = frozenset({
    "youtube", "bing videos", "duckduckgo videos", "google videos",
    "dailymotion", "vimeo", "peertube", "sepiasearch", "invidious",
    "rumble", "odysee",
})


def engine_tier(engine: str, promote: frozenset[str] = frozenset()) -> int:
    """0 = leads the ranking, 1 = backfill. Lower sorts first."""
    e = (engine or "").strip().lower()
    if e in promote:
        return 0
    return 0 if e in _GENERAL_WEB_ENGINES else 1


# What a query is about -> which SearXNG category can hold the answer. Every
# sub-query used to fan out to every engine in the run's categories: a balloon
# question hit PubMed, arXiv, Ask Ubuntu and Stack Overflow on every round,
# which is slower, adds junk to the pool, burns triage tokens on it, and
# earns the shared address rate-limit strikes for nothing.
SCOPE_CATEGORIES = {
    "web": "general", "video": "videos", "code": "it", "academic": "science",
    "qa": "q&a", "news": "news", "social": "social media", "files": "files",
}


def categories_for_scope(scope: str, allowed: str) -> str | None:
    """SearXNG categories for a query's scope, narrowed to what the run
    selected — a scope may narrow the run's categories, never widen them.
    A query with no usable scope is a web query: general, plus videos when
    the run selected video. It used to mean "every category the run
    selected", and a balloon question's unscoped queries hit Crossref,
    OpenAlex and PubMed until all three suspended it (2026-09-04). None —
    "use the run's categories" — only when the run did not select general."""
    wanted = [SCOPE_CATEGORIES[t] for t in
              re.split(r"[+,/ ]+", (scope or "").strip().lower()) if t in SCOPE_CATEGORIES]
    allowed_set = set(split_categories(allowed))
    if not wanted:
        if "general" not in allowed_set:
            return None
        return "general,videos" if "videos" in allowed_set else "general"
    kept = [c for c in dict.fromkeys(wanted) if not allowed_set or c in allowed_set]
    return ",".join(kept) if kept else None


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


# Engines that crawl their own small index and match closer to AND than to a
# sentence. Measured 2026-08-31 on one real sub-query at four lengths:
#                  7w   5w   3w   2w
#   boardreader     0    0    8    8
#   searchmysite    1    4   10   10
#   wiby            0    1   12   12
# The planner writes 5-8 word queries, so these contributed almost nothing.
# A long query is therefore also sent to them alone, shortened to its first
# few content words. None of them rate-limits, so the extra request costs
# nothing against the address the majors are already judging.
SMALL_INDEX_ENGINES = frozenset({"searchmysite", "wiby"})   # boardreader retired 2026-09-04: parsing error on every search
_SHORT_WORDS = 3
_SHORT_TRIGGER = 5          # queries this long or longer get a short twin
_QUERY_STOPWORDS = frozenset(
    "the a an of for to in on at by with and or vs how do does did is are was "
    "what which why when where who i my your best top guide diy".split())


def query_terms(text: str) -> frozenset[str]:
    """Lowercase word tokens of the user's own question — what tells a
    speculative proper noun from one they actually asked about."""
    return frozenset(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _entity_positions(tokens: list[str]) -> list[int]:
    """Indices of tokens that look like a name rather than a word:
    ALL-CAPS acronyms and Capitalised words."""
    out = []
    for i, tok in enumerate(tokens):
        core = tok.strip("\"'()[],.:;")
        if len(core) < 2 or not any(c.isalpha() for c in core):
            continue
        if core.isupper() or (core[:1].isupper() and any(c.islower() for c in core[1:])):
            out.append(i)
    return out


def generalize_query(query: str, known: frozenset[str] = frozenset()) -> str:
    """A query that returned nothing, with its rare names thinned out.

    Naming three or four rare entities at once matches no page: 25 of one
    depth-10 run's 54 searches came back empty, every one of them a brand
    string the planner had assembled ("Workato AIRO Genie structured
    output"). Drop the names the user's own question never used; if the
    question used them all, keep the subject and drop the rest. Empty string
    when there is nothing to thin — the caller then accepts the empty result.
    """
    tokens = query.split()
    entities = _entity_positions(tokens)
    if len(entities) < 2:
        return ""
    drop = {i for i in entities
            if tokens[i].strip("\"'()[],.:;").lower() not in known}
    if not drop:
        drop = set(entities[1:])
    kept = [t for i, t in enumerate(tokens) if i not in drop]
    if len(kept) < 2 or len(kept) == len(tokens):
        return ""
    return " ".join(kept)


def shorten_query(query: str, words: int = _SHORT_WORDS) -> str:
    """The first few content words of a query — what a small index can match."""
    tokens = [w for w in query.split() if not w.lower().startswith("site:")]
    content = [w for w in tokens
               if w.lower().strip("?.,!:;\"'()") not in _QUERY_STOPWORDS]
    return " ".join((content or tokens)[:words])


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    engine: str
    published: datetime | None
    score: float
    via_query: str = ""  # the sub-query that surfaced this result
    author: str = ""     # channel / account, when the engine reports one


class Searcher:
    def __init__(self, base_url: str, client: httpx.AsyncClient,
                 categories: str = DEFAULT_CATEGORIES,
                 max_concurrent: int = 2, timeout: float = 45.0,
                 small_index_engines: frozenset[str] = SMALL_INDEX_ENGINES,
                 bench=None, known_terms: frozenset[str] = frozenset()):
        self.base_url = base_url.rstrip("/")
        self.client = client
        self.categories = categories or DEFAULT_CATEGORIES
        self.small_index_engines = small_index_engines
        # research/bench.py: engines a network block has taken out. Optional;
        # without one every query goes by category exactly as before.
        self.bench = bench
        # The user's own question, tokenised: see generalize_query.
        self.known_terms = known_terms
        self.generalized = 0
        self.bench_events: list[str] = []
        self._engine_map_cache: dict[str, set[str]] | None = None
        self._tr_support_cache: frozenset[str] | None = None
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
        # Requests that reached Brave's API: one per SearXNG request carrying
        # the general category. Twins sent to named engines and queries
        # scoped to videos/it/science do not count — "searches" did, and
        # over-read the Brave dashboard by two thirds.
        self.brave_requests = 0
        self.blocked_engines: dict[str, str] = {}
        # The search (1-based count) at which each engine last refused.
        # blocked_engines lives for the whole run; a round reports only
        # the refusals that happened during its own searches.
        self.blocked_at: dict[str, int] = {}

    @property
    def degraded(self) -> bool:
        """Every search came back empty and engines were reporting blocks."""
        return (self.searches > 0 and self.empty_searches == self.searches
                and bool(self.blocked_engines))

    async def _query(self, query: str, recency: str, *, pageno: int = 1,
                     engines: str | None = None,
                     categories: str | None = None) -> list[SearchResult]:
        """One SearXNG request, parsed. `engines` narrows to named engines and
        then replaces the category selection, as SearXNG itself does."""
        params: dict = {
            "q": query,
            "format": "json",
            "language": "en",
            "safesearch": 0,
            "pageno": pageno,
        }
        if engines:
            params["engines"] = engines
        else:
            cats = categories_for(recency, categories or self.categories)
            params["categories"] = cats
            if "general" in split_categories(cats):
                self.brave_requests += 1
            # A benched engine is left out by naming every other engine in
            # the categories — the only way the API excludes one. Nothing
            # benched, or no engine map: the categories go as before.
            excluded = self.bench.excluded() if self.bench is not None else frozenset()
            if excluded:
                emap = await self._engine_map()
                wanted = set().union(*(emap.get(c, set()) for c in split_categories(cats))) if emap else set()
                if wanted & excluded and wanted - excluded:
                    params["engines"] = ",".join(sorted(wanted - excluded))
                    del params["categories"]
        time_range = RECENCY_TO_TIME_RANGE.get(recency)

        # One logical search, however many requests it takes.
        self.searches += 1

        if time_range:
            plan = await self._time_range_plan(params, time_range)
        else:
            plan = [params]

        out: list[SearchResult] = []
        seen: set[str] = set()
        answered = False
        for req in plan:
            got, had_results = await self._send(req, query, recency)
            answered = answered or had_results
            for r in got:
                if r.url not in seen:
                    seen.add(r.url)
                    out.append(r)
        if not answered:
            self.empty_searches += 1
        return out

    async def _time_range_plan(self, params: dict, time_range: str) -> list[dict]:
        """Split one dated search into the engines that can date-filter and
        the engines that cannot.

        SearXNG drops any engine without `time_range_support` from a search
        that carries a time range. On this install that is five of the seven
        general engines — bing, marginalia, mwmbl, searchmysite and wiby —
        so every run with a recency other than "all time" was querying
        braveapi and google cse alone, and the indie indexes that exist
        precisely to answer when the majors CAPTCHA were silently absent.
        2026-09-11, a depth-10 run: 53 searches on two engines, 31 of them
        empty.

        Asking the rest without the parameter costs one extra request and
        loses nothing: `cutoff_for(recency)` already drops a result whose
        date is outside the window, so the dating is done either way — by
        the engine where it can be, by us where it cannot.
        """
        emap = await self._engine_map()
        supports = await self._time_range_support()
        # An empty map means /config was unreachable — guessing a split would
        # be worse than today's behaviour. An empty `supports` with a good map
        # is a real answer ("none of them can"), not a missing one, so only
        # the map is allowed to veto.
        if not emap:
            return [dict(params, time_range=time_range)]

        if params.get("engines"):
            wanted = {e.strip() for e in params["engines"].split(",") if e.strip()}
        else:
            cats = split_categories(params.get("categories", ""))
            wanted = set().union(*(emap.get(c, set()) for c in cats)) if cats else set()
        if not wanted:
            return [dict(params, time_range=time_range)]

        dated = sorted(wanted & supports)
        undated = sorted(wanted - supports)
        if not dated:
            # Nothing can filter; sending the range would return nothing at
            # all, which is how this was failing.
            return [_named(params, undated)]
        if not undated:
            return [dict(params, time_range=time_range)]
        return [dict(_named(params, dated), time_range=time_range),
                _named(params, undated)]

    async def _send(self, params: dict, query: str,
                    recency: str) -> tuple[list[SearchResult], bool]:
        """One SearXNG request, parsed. Returns its results and whether it
        answered at all — the caller decides what counts as an empty search,
        because a split search is still one search."""
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

        unresponsive = data.get("unresponsive_engines") or []
        for entry in unresponsive:
            if isinstance(entry, (list, tuple)) and entry:
                self.blocked_engines[str(entry[0])] = str(entry[-1])
                self.blocked_at[str(entry[0])] = self.searches
        if unresponsive:
            log.info("searxng unresponsive engines for %r: %s", query, unresponsive)
        if self.bench is not None:
            try:
                refused = {str(e[0]): str(e[-1]) for e in unresponsive
                           if isinstance(e, (list, tuple)) and e}
                answered = {str(eng) for item in data.get("results", [])
                            for eng in (item.get("engines") or [])}
                self.bench_events.extend(self.bench.observe(refused, answered))
            except Exception as ex:  # noqa: BLE001 — bookkeeping never stops a search
                log.debug("bench bookkeeping failed: %s", ex)

        cutoff = cutoff_for(recency)
        site = _site_scope(query)
        out: list[SearchResult] = []
        for item in data.get("results", []):
            url = item.get("url")
            if not url or not str(url).startswith(("http://", "https://")):
                continue
            # A site:-scoped query is a promise to the pipeline. Some engines
            # honor the operator; others (bing, notoriously) quietly drop it
            # and return keyword matches from anywhere, which would flood the
            # candidate pool with junk. Enforce the scope here.
            if site and not _in_site(str(url), site):
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
                author=(item.get("author") or "").strip(),
            ))
        return out, bool(data.get("results"))

    async def _time_range_support(self) -> frozenset[str]:
        """Engines SearXNG will keep in a search that carries a time range."""
        if self._tr_support_cache is None:
            await self._engine_map()          # fills both caches
        return self._tr_support_cache or frozenset()

    async def _engine_map(self) -> dict[str, set[str]]:
        """category → enabled engine names, from SearXNG's /config, once per
        searcher. Empty when unavailable, which means "search by category"."""
        if self._engine_map_cache is None:
            try:
                resp = await self.client.get(f"{self.base_url}/config", timeout=self.timeout)
                resp.raise_for_status()
                emap: dict[str, set[str]] = {}
                dated: set[str] = set()
                for e in resp.json().get("engines", []):
                    if e.get("enabled"):
                        for cat in e.get("categories") or []:
                            emap.setdefault(str(cat), set()).add(str(e["name"]))
                        if e.get("time_range_support"):
                            dated.add(str(e["name"]))
                self._engine_map_cache = emap
                self._tr_support_cache = frozenset(dated)
            except Exception as ex:  # noqa: BLE001
                log.debug("engine map unavailable: %s", ex)
                self._engine_map_cache = {}
                self._tr_support_cache = frozenset()
        return self._engine_map_cache

    def drain_bench_events(self) -> list[str]:
        events, self.bench_events = self.bench_events, []
        return events

    async def search(self, query: str, recency: str, *, pageno: int = 1,
                     categories: str | None = None) -> list[SearchResult]:
        """`categories` narrows this one query (see categories_for_scope)."""
        out = await self._query(query, recency, pageno=pageno, categories=categories)

        site = _site_scope(query)
        # Site-restricted indexes are thin: a long specific query against one
        # usually matches nothing, while a short one finds the pages. Planners
        # keep writing long ones despite prompt guidance, so enforce the
        # shortening here: one retry with the first five words.
        if site and not out and pageno == 1:
            words = [w for w in query.split()
                     if not w.lower().startswith("site:")]
            if len(words) > 5:
                short = f"site:{site} " + " ".join(words[:5])
                log.info("site-scoped query found nothing, retrying "
                         "shorter: %r", short)
                return await self.search(short, recency, pageno=pageno,
                                         categories=categories)

        # A query that matched nothing anywhere is usually over-specified
        # with rare names rather than wrong about the topic. One retry with
        # them thinned out turns a wasted search into evidence.
        if not out and pageno == 1 and not site:
            general = generalize_query(query, self.known_terms)
            if general:
                log.info("no results for %r — retrying generalized: %r",
                         query, general)
                self.generalized += 1
                out = await self._query(general, recency, pageno=pageno,
                                        categories=categories)

        # The short twin for small-index engines (see SMALL_INDEX_ENGINES).
        # Page 1 only, never for site: queries, and only when shortening
        # actually changes something. Video search matches short keyword
        # strings the same way: a balloon question never surfaced "Balloon
        # Rock Hand Horns" because its queries read like web queries, so
        # when videos are in this query's scope the YouTube engine gets the
        # shortened twin too.
        if pageno == 1 and not site and len(query.split()) >= _SHORT_TRIGGER:
            short = shorten_query(query)
            twin_engines = set(self.small_index_engines)
            if "videos" in split_categories(categories or self.categories):
                twin_engines.add("youtube")
            if short and short.lower() != query.lower() and twin_engines:
                try:
                    twin = await self._query(
                        short, recency, pageno=1,
                        engines=",".join(sorted(twin_engines)))
                except Exception as e:  # noqa: BLE001 — the main results stand
                    log.debug("short twin %r failed: %s", short, e)
                    twin = []
                have = {r.url for r in out}
                for r in twin:
                    if r.url not in have:
                        have.add(r.url)
                        out.append(r)
        return out
