"""Feeds as a search source: parsing, hardening, and the searcher surface."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import respx

from app.research.feeds import (FeedSearcher, parse_feed, parse_feed_list)

RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>Local LLM Weekly</title>
  <item>
    <title>MLX 0.30 ships paged attention</title>
    <link>https://example.com/mlx-030</link>
    <description>A long-awaited change lands.</description>
    <pubDate>Wed, 20 Aug 2026 09:00:00 GMT</pubDate>
  </item>
  <item>
    <title>Older post</title>
    <link>https://example.com/old</link>
    <pubDate>Mon, 01 Jan 2024 09:00:00 GMT</pubDate>
  </item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>llama.cpp releases</title>
  <entry>
    <title>b4321</title>
    <link rel="alternate" href="https://example.org/b4321"/>
    <summary>Metal backend fixes.</summary>
    <updated>2026-08-21T12:00:00Z</updated>
  </entry>
</feed>"""


def test_parses_rss():
    title, entries = parse_feed(RSS, "https://example.com/feed.xml")
    assert title == "Local LLM Weekly"
    assert [e.url for e in entries] == ["https://example.com/mlx-030",
                                        "https://example.com/old"]
    assert entries[0].published.year == 2026
    assert entries[0].summary == "A long-awaited change lands."


def test_parses_atom_and_prefers_alternate_links():
    title, entries = parse_feed(ATOM, "https://example.org/atom")
    assert title == "llama.cpp releases"
    assert entries[0].url == "https://example.org/b4321"
    assert entries[0].published.month == 8


def test_relative_entry_links_resolve_against_the_feed():
    rss = RSS.replace(b"https://example.com/mlx-030", b"/mlx-030")
    _t, entries = parse_feed(rss, "https://example.com/blog/feed.xml")
    assert entries[0].url == "https://example.com/mlx-030"


def test_malformed_feed_yields_nothing_rather_than_raising():
    assert parse_feed(b"not xml at all", "https://x.com/f") == ("", [])
    assert parse_feed(b"", "https://x.com/f") == ("", [])


def test_entity_expansion_cannot_blow_up_the_parser():
    """Feeds are third-party XML; a billion-laughs payload must not expand."""
    bomb = (b'<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">'
            b'<!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">'
            b'<!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">'
            b']><rss version="2.0"><channel><title>&lol3;</title></channel></rss>')
    title, entries = parse_feed(bomb, "https://evil.example/f")
    assert "lollol" not in title           # entities were not resolved
    assert entries == []


def test_feed_list_parsing_ignores_comments_and_junk():
    blob = ("https://a.com/feed  # weekly\n"
            "\n"
            "# just a comment\n"
            "not-a-url\n"
            "https://b.com/atom\n"
            "https://a.com/feed\n")          # duplicate
    assert parse_feed_list(blob) == ["https://a.com/feed", "https://b.com/atom"]


# ---- the searcher surface -------------------------------------------------------

@respx.mock
async def test_searcher_returns_entries_newest_first_and_honours_recency():
    respx.get("https://example.com/feed.xml").mock(
        return_value=httpx.Response(200, content=RSS))
    respx.get("https://example.org/atom").mock(
        return_value=httpx.Response(200, content=ATOM))
    async with httpx.AsyncClient() as c:
        s = FeedSearcher(["https://example.com/feed.xml",
                          "https://example.org/atom"], c)
        # "all" keeps everything, including the 2024 item
        every = await s.search("ignored", "all")
        assert len(every) == 3
        assert [r.url for r in every][0] == "https://example.org/b4321"  # newest

    async with httpx.AsyncClient() as c2:
        s2 = FeedSearcher(["https://example.com/feed.xml"], c2)
        recent = await s2.search("ignored", "week")
        assert all(r.url != "https://example.com/old" for r in recent)


@respx.mock
async def test_searcher_ignores_the_query_and_has_no_page_two():
    respx.get("https://example.com/feed.xml").mock(
        return_value=httpx.Response(200, content=RSS))
    async with httpx.AsyncClient() as c:
        s = FeedSearcher(["https://example.com/feed.xml"], c)
        a = await s.search("spark plugs", "all")
        b = await s.search("something entirely different", "all")
        assert [r.url for r in a] == [r.url for r in b]   # query is irrelevant
        assert await s.search("x", "all", pageno=2) == []


@respx.mock
async def test_one_dead_feed_does_not_sink_the_brief():
    respx.get("https://example.com/feed.xml").mock(
        return_value=httpx.Response(200, content=RSS))
    respx.get("https://dead.example/feed").mock(
        return_value=httpx.Response(503))
    async with httpx.AsyncClient() as c:
        s = FeedSearcher(["https://example.com/feed.xml",
                          "https://dead.example/feed"], c)
        results = await s.search("", "all")
        assert len(results) == 2                      # the live feed still ran
        assert "https://dead.example/feed" in s.blocked_engines
        assert s.degraded is False                    # not every feed failed


@respx.mock
async def test_all_feeds_failing_reports_degraded():
    respx.get("https://dead.example/feed").mock(return_value=httpx.Response(503))
    async with httpx.AsyncClient() as c:
        s = FeedSearcher(["https://dead.example/feed"], c)
        assert await s.search("", "all") == []
        assert s.degraded is True


# ---- a brief as a run -----------------------------------------------------------

from app.db import Repo, connect                                      # noqa: E402
from app.models import RunParams                                      # noqa: E402
from app.research.orchestrator import Orchestrator                    # noqa: E402
from app.research.progress import ProgressBus                         # noqa: E402
from tests.fake_llm import FakeLLM                                    # noqa: E402
from tests.test_pipeline_e2e import (SX, article, make_cfg, script,   # noqa: E402
                                     sx_payload, sx_result)


def _brief_cfg(data_dir):
    cfg = make_cfg(data_dir)
    cfg.feeds = "https://example.com/feed.xml"
    return cfg


@respx.mock
async def test_a_brief_reads_feeds_and_never_touches_the_search_engine(data_dir):
    cfg = _brief_cfg(data_dir)
    sx = respx.get(f"{SX}/search").mock(
        return_value=httpx.Response(200, json=sx_payload([])))
    respx.get("https://example.com/feed.xml").mock(
        return_value=httpx.Response(200, content=RSS))
    respx.get("https://example.com/mlx-030").mock(
        return_value=httpx.Response(200, html=article("MLX 0.30")))
    respx.get("https://example.com/old").mock(
        return_value=httpx.Response(200, html=article("Older post")))

    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(script(
                            [{"state_md": "s", "saturated": True,
                              "next_queries": []}])))
    run_id = orch.enqueue(RunParams(query="Brief", depth=4, recency="all",
                                    origin="cli", kind="brief"))
    await orch.execute_now(run_id)

    assert repo.get_run(run_id)["status"] == "completed"
    assert not sx.called                       # the search engine was never asked
    domains = {f["domain"] for f in repo.findings_for_run(run_id)}
    assert domains == {"example.com"}


@respx.mock
async def test_a_brief_skips_items_an_earlier_brief_already_reported(data_dir):
    """Yesterday's items are still inside today's window."""
    cfg = _brief_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload([])))
    respx.get("https://example.com/feed.xml").mock(
        return_value=httpx.Response(200, content=RSS))
    respx.get("https://example.com/mlx-030").mock(
        return_value=httpx.Response(200, html=article("MLX 0.30")))
    respx.get("https://example.com/old").mock(
        return_value=httpx.Response(200, html=article("Older post")))

    repo = Repo(connect(cfg.db_path))

    async def one_brief() -> int:
        orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                            llm_factory=lambda: FakeLLM(script(
                                [{"state_md": "s", "saturated": True,
                                  "next_queries": []}])))
        rid = orch.enqueue(RunParams(query="Brief", depth=4, recency="all",
                                     origin="cli", kind="brief"))
        await orch.execute_now(rid)
        return len(repo.findings_for_run(rid))

    assert await one_brief() == 2      # both items are new the first time
    assert await one_brief() == 0      # and none of them are new the second


@respx.mock
async def test_suppression_survives_url_canonicalisation(data_dir):
    """Stored finding urls keep a trailing slash; canonicalize strips it. The
    suppression set was raw, so it never matched and briefs repeated."""
    cfg = _brief_cfg(data_dir)
    slashed = RSS.replace(b"https://example.com/mlx-030",
                          b"https://example.com/mlx-030/")
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload([])))
    respx.get("https://example.com/feed.xml").mock(
        return_value=httpx.Response(200, content=slashed))
    respx.get("https://example.com/mlx-030/").mock(
        return_value=httpx.Response(200, html=article("MLX 0.30")))
    respx.get("https://example.com/old").mock(
        return_value=httpx.Response(200, html=article("Older post")))

    repo = Repo(connect(cfg.db_path))

    async def one_brief() -> set[str]:
        orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                            llm_factory=lambda: FakeLLM(script(
                                [{"state_md": "s", "saturated": True,
                                  "next_queries": []}])))
        rid = orch.enqueue(RunParams(query="Brief", depth=4, recency="all",
                                     origin="cli", kind="brief"))
        await orch.execute_now(rid)
        return {f["url"] for f in repo.findings_for_run(rid)}

    first = await one_brief()
    assert any(u.rstrip("/").endswith("mlx-030") for u in first)
    assert await one_brief() == set()      # nothing repeats, slash or no slash


@respx.mock
async def test_a_brief_with_no_feeds_configured_fails_loudly(data_dir):
    cfg = make_cfg(data_dir)
    cfg.feeds = ""
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload([])))
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM({}))
    run_id = orch.enqueue(RunParams(query="Brief", depth=4, recency="all",
                                    origin="cli", kind="brief"))
    await orch.execute_now(run_id)
    row = repo.get_run(run_id)
    assert row["status"] == "failed"
    assert "no feeds configured" in (row["error"] or "")


@respx.mock
async def test_a_research_run_is_unaffected_by_the_brief_path(data_dir):
    cfg = _brief_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result("https://example-a.com/article", "Article A")])))
    respx.get("https://example-a.com/article").mock(
        return_value=httpx.Response(200, html=article("Article A")))
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(script(
                            [{"state_md": "s", "saturated": True,
                              "next_queries": []}])))
    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)
    assert repo.get_run(run_id)["kind"] == "research"
    assert {f["domain"] for f in repo.findings_for_run(run_id)} == {"example-a.com"}


def test_a_brief_can_be_started_with_no_question(data_dir, monkeypatch):
    """The New form's textarea is optional for a brief, so the route must be
    too — it was left required, which 422'd instead of starting the run."""
    from fastapi.testclient import TestClient
    from app.config import load_settings
    from app.web.server import create_app
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("FEEDS", "https://example.com/feed.xml")
    app = create_app(enable_worker=False, enable_bot=False)
    cfg = load_settings(str(data_dir))
    with TestClient(app) as client:
        r = client.post("/runs", data={"depth": "4", "recency": "week",
                                       "kind": "brief"},
                        follow_redirects=False)
        assert r.status_code == 303, r.text
        run_id = r.headers["location"].rsplit("/", 1)[-1]
    row = Repo(connect(cfg.db_path)).get_run(run_id)
    assert row["kind"] == "brief"
    assert row["query"]                       # given a title-worthy default


def test_a_research_run_still_requires_a_question(data_dir, monkeypatch):
    from fastapi.testclient import TestClient
    from app.web.server import create_app
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    app = create_app(enable_worker=False, enable_bot=False)
    with TestClient(app) as client:
        r = client.post("/runs", data={"depth": "3", "recency": "all"},
                        follow_redirects=False)
        # the form is re-rendered with the message, under a 422
        assert r.status_code == 422
        assert "String should have at least 3 characters" in r.text


def test_a_brief_caps_per_feed_not_per_domain(data_dir):
    """Three GitHub release feeds share one domain. Capping by domain let two
    items through from all three combined, which defeats a curated list."""
    from app.research.dedupe import rank_diverse
    from app.research.searcher import SearchResult

    def r(url, feed):
        return SearchResult(url=url, title=url, snippet="", engine="feed",
                            published=None, score=1.0, via_query=feed)

    pool = [r(f"https://github.com/{repo}/releases/tag/v{n}", f"{repo} releases")
            for repo in ("mlx", "llama.cpp", "ollama") for n in range(1, 4)]

    by_domain = rank_diverse(pool, set(), per_domain=2, limit=30)
    assert len(by_domain) == 2                       # the bug: one domain, two items

    by_feed = rank_diverse(pool, set(), per_domain=6, limit=30,
                           group=lambda x: x.via_query)
    assert len(by_feed) == 9                         # all three feeds represented
    assert len({x.via_query for x in by_feed}) == 3


def test_grouping_still_defaults_to_domain_for_web_search():
    from app.research.dedupe import rank_diverse
    from app.research.searcher import SearchResult
    pool = [SearchResult(url=f"https://spam.com/{n}", title="t", snippet="",
                         engine="google", published=None, score=1.0)
            for n in range(6)]
    assert len(rank_diverse(pool, set(), per_domain=2, limit=30)) == 2


# ---- topic filter + brief-specific notes ----------------------------------------

class _FilterLLM:
    """Records the prompts it sees so a test can assert which rubric ran."""
    def __init__(self, keep):
        self.keep = keep
        self.prompts: list[str] = []

    async def chat_json(self, kind, messages, schema, **_kw):
        self.prompts.append(messages[-1]["content"])
        return schema.model_validate({"keep": self.keep})


async def test_topic_filter_narrows_entries_to_the_stated_interest():
    from app.research.feeds import FeedEntry
    entries = [
        FeedEntry("https://a/1", "MLX 0.30 ships paged attention", "", None, "f"),
        FeedEntry("https://a/2", "EVE Online moves to Python 3", "", None, "f"),
    ]
    llm = _FilterLLM(keep=[0])
    s = FeedSearcher([], httpx.AsyncClient(), topic="local LLM inference", llm=llm)
    kept = await s._by_topic(entries)
    assert [e.url for e in kept] == ["https://a/1"]
    assert s.filtered_out == 1
    assert "local LLM inference" in llm.prompts[0]


async def test_no_topic_means_no_filtering_call():
    from app.research.feeds import FeedEntry
    entries = [FeedEntry("https://a/1", "anything", "", None, "f")]
    llm = _FilterLLM(keep=[])
    s = FeedSearcher([], httpx.AsyncClient(), topic="", llm=llm)
    assert await s._by_topic(entries) == entries
    assert llm.prompts == []                    # the model was never asked


async def test_a_broken_filter_keeps_everything_rather_than_emptying_the_brief():
    from app.research.feeds import FeedEntry
    entries = [FeedEntry("https://a/1", "t", "", None, "f"),
               FeedEntry("https://a/2", "t2", "", None, "f")]

    class Boom:
        async def chat_json(self, *a, **k):
            raise RuntimeError("model down")

    s = FeedSearcher([], httpx.AsyncClient(), topic="x", llm=Boom())
    assert await s._by_topic(entries) == entries

    # and a filter that matches nothing is treated as a failed filter, not as
    # "the reader wants an empty brief"
    s2 = FeedSearcher([], httpx.AsyncClient(), topic="x", llm=_FilterLLM(keep=[]))
    assert await s2._by_topic(entries) == entries


@respx.mock
async def test_a_brief_scores_news_value_not_question_answering(data_dir):
    """A terse changelog is high value to a brief and low value to research."""
    cfg = _brief_cfg(data_dir)
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload([])))
    respx.get("https://example.com/feed.xml").mock(
        return_value=httpx.Response(200, content=RSS))
    for slug in ("mlx-030", "old"):
        respx.get(f"https://example.com/{slug}").mock(
            return_value=httpx.Response(200, html=article(slug)))

    seen: list[str] = []

    def capture_notes(messages):
        seen.append(messages[-1]["content"])
        return {"relevance": 8, "summary": "s", "notes_md": "n",
                "key_facts": [], "published_date": None}

    s = script([{"state_md": "s", "saturated": True, "next_queries": []}])
    s["notes"] = [capture_notes]
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(s))
    rid = orch.enqueue(RunParams(query="Brief", depth=4, recency="all",
                                 origin="cli", kind="brief"))
    await orch.execute_now(rid)

    assert seen, "no notes calls were made"
    joined = "".join(seen)
    assert "CHANGE THE READER SHOULD KNOW ABOUT" in joined   # the brief rubric
    assert "research brief" not in joined.lower()            # not the research one
