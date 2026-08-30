"""Finding a site's feed from its address, and following a source from a run."""
from __future__ import annotations

import httpx
import respx
from fastapi.testclient import TestClient

from app.config import load_settings
from app.research.feed_discovery import (declared_feeds, discover, github_atom,
                                         normalise_input)
from app.web.server import create_app

RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>The Feed</title>
  <item><title>One</title><link>https://site.test/one</link></item>
</channel></rss>"""

PAGE = """<html><head>
  <link rel="stylesheet" href="/x.css">
  <link rel="alternate" type="application/rss+xml" href="/atom/everything/">
</head><body>hi</body></html>"""


# ---- what people actually type ---------------------------------------------

def test_bare_hosts_and_paths_are_accepted():
    assert normalise_input("site.test") == "https://site.test"
    assert normalise_input("  HTTPS://Site.Test/a/b  ") == "https://site.test/a/b"
    assert normalise_input("site.test/blog#frag") == "https://site.test/blog"
    assert normalise_input("") == ""


def test_owner_repo_shorthand_expands_to_github():
    """Advertised in the UI and initially not implemented: without this,
    "ollama/ollama" became https://ollama/ollama, a host that does not exist."""
    assert normalise_input("ollama/ollama") == "https://github.com/ollama/ollama"
    assert normalise_input("ggml-org/llama.cpp") == \
        "https://github.com/ggml-org/llama.cpp"
    # a real hostname with a path is not shorthand — the dot gives it away
    assert normalise_input("site.test/feed") == "https://site.test/feed"


def test_github_urls_map_to_their_releases_atom():
    assert github_atom("https://github.com/ml-explore/mlx") == \
        "https://github.com/ml-explore/mlx/releases.atom"
    assert github_atom("https://github.com/o/r.git") == \
        "https://github.com/o/r/releases.atom"
    assert github_atom("https://github.com/onlyowner") is None
    assert github_atom("https://site.test/o/r") is None


def test_declared_feeds_ignores_non_feed_links():
    found = declared_feeds(PAGE, "https://site.test/page")
    assert found == ["https://site.test/atom/everything/"]


# ---- discovery order --------------------------------------------------------

@respx.mock
async def test_a_declared_feed_beats_guessing():
    respx.get("https://site.test/").mock(return_value=httpx.Response(200, html=PAGE))
    respx.get("https://site.test/atom/everything/").mock(
        return_value=httpx.Response(200, content=RSS))
    async with httpx.AsyncClient() as c:
        got = await discover(c, "site.test")
    assert got.how == "declared"
    assert got.url == "https://site.test/atom/everything/"
    assert got.title == "The Feed" and got.entries == 1


@respx.mock
async def test_conventional_paths_are_tried_when_nothing_is_declared():
    bare = "<html><head></head><body>no feed link</body></html>"
    respx.get("https://site.test/").mock(return_value=httpx.Response(200, html=bare))
    respx.get("https://site.test/feed").mock(return_value=httpx.Response(404))
    respx.get("https://site.test/feed.xml").mock(
        return_value=httpx.Response(200, content=RSS))
    async with httpx.AsyncClient() as c:
        got = await discover(c, "site.test")
    assert got.how == "conventional"
    assert got.url == "https://site.test/feed.xml"


@respx.mock
async def test_a_candidate_that_parses_to_nothing_is_not_a_feed():
    """A 200 that is not a feed — a soft-404 HTML page — must not count."""
    bare = "<html><head></head><body>nope</body></html>"
    respx.get("https://site.test/").mock(return_value=httpx.Response(200, html=bare))
    for path in ("/feed", "/feed.xml", "/rss.xml", "/atom.xml", "/index.xml",
                 "/feed/", "/rss", "/blog/feed.xml", "/feeds/all.atom.xml"):
        respx.get(f"https://site.test{path}").mock(
            return_value=httpx.Response(200, html="<html>still not a feed</html>"))
    async with httpx.AsyncClient() as c:
        assert await discover(c, "site.test") is None


@respx.mock
async def test_an_unreachable_site_returns_nothing_rather_than_raising():
    respx.get("https://site.test/").mock(side_effect=httpx.ConnectError("down"))
    async with httpx.AsyncClient() as c:
        assert await discover(c, "site.test") is None


# ---- the two endpoints ------------------------------------------------------

def _app(data_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    return create_app(enable_worker=False, enable_bot=False), load_settings(str(data_dir))


@respx.mock
def test_adding_a_feed_by_site_address_saves_it(data_dir, monkeypatch):
    app, cfg = _app(data_dir, monkeypatch)
    respx.get("https://site.test/").mock(return_value=httpx.Response(200, html=PAGE))
    respx.get("https://site.test/atom/everything/").mock(
        return_value=httpx.Response(200, content=RSS))
    with TestClient(app) as c:
        r = c.post("/settings/add-feed", data={"site": "site.test"})
    assert "The Feed" in r.text
    assert "https://site.test/atom/everything/" in load_settings(str(data_dir)).feeds


@respx.mock
def test_adding_the_same_feed_twice_does_not_duplicate_it(data_dir, monkeypatch):
    app, _cfg = _app(data_dir, monkeypatch)
    respx.get("https://site.test/").mock(return_value=httpx.Response(200, html=PAGE))
    respx.get("https://site.test/atom/everything/").mock(
        return_value=httpx.Response(200, content=RSS))
    with TestClient(app) as c:
        c.post("/settings/add-feed", data={"site": "site.test"})
        again = c.post("/settings/add-feed", data={"site": "site.test"})
    assert "Already subscribed" in again.text
    feeds = load_settings(str(data_dir)).feeds
    assert feeds.count("https://site.test/atom/everything/") == 1


@respx.mock
def test_a_site_with_no_feed_says_so_and_saves_nothing(data_dir, monkeypatch):
    app, _cfg = _app(data_dir, monkeypatch)
    respx.get("https://site.test/").mock(
        return_value=httpx.Response(200, html="<html><body>x</body></html>"))
    # every conventional path 404s
    respx.route(method="GET", host="site.test").mock(
        return_value=httpx.Response(404))
    with TestClient(app) as c:
        r = c.post("/settings/add-feed", data={"site": "site.test"})
    assert "No feed found" in r.text
    assert not load_settings(str(data_dir)).feeds.strip()


@respx.mock
def test_following_a_source_from_a_finding(data_dir, monkeypatch):
    """The other end of the problem: the list grows from research you did."""
    app, _cfg = _app(data_dir, monkeypatch)
    respx.get("https://site.test/").mock(return_value=httpx.Response(200, html=PAGE))
    respx.get("https://site.test/atom/everything/").mock(
        return_value=httpx.Response(200, content=RSS))
    with TestClient(app) as c:
        r = c.post("/settings/follow-source", data={"domain": "site.test"})
    assert "Following" in r.text and "The Feed" in r.text
    assert "https://site.test/atom/everything/" in load_settings(str(data_dir)).feeds
