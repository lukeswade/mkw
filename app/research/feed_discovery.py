"""Find a site's feed from its address.

Nobody knows where a site keeps its feed; everybody knows the site. Given
"simonwillison.net" or a link to one article on it, this returns the feed URL
to subscribe to — from the page's own <link rel="alternate"> declaration
first, then from the handful of conventional paths, then from the shape of
the host itself (a GitHub repo publishes releases.atom).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from lxml import html as lxml_html

from app.research.feeds import parse_feed

log = logging.getLogger(__name__)

_TIMEOUT = 15.0
_MAX_BYTES = 3 * 1024 * 1024
_FEED_TYPES = ("application/rss+xml", "application/atom+xml",
               "application/feed+json", "text/xml", "application/xml")
# Tried in order when a page declares nothing. Cheap, and covers most of the
# static-site generators and blog engines in the wild.
_CONVENTIONAL = ("/feed", "/feed.xml", "/rss.xml", "/atom.xml", "/index.xml",
                 "/feed/", "/rss", "/blog/feed.xml", "/feeds/all.atom.xml")


@dataclass
class Discovered:
    url: str
    title: str
    entries: int
    how: str          # "declared", "conventional", or "github"


def normalise_input(raw: str) -> str:
    """Accept what a person actually types: a bare host, or any page on it."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    if not re.match(r"^https?://", raw, re.I):
        # "owner/repo" with no dot in the first segment is GitHub shorthand,
        # not a hostname — "https://ollama/ollama" resolves to nothing.
        if re.fullmatch(r"[\w.-]+/[\w.-]+/?", raw) and "." not in raw.split("/")[0]:
            raw = "https://github.com/" + raw.strip("/")
        else:
            raw = "https://" + raw.lstrip("/")
    parts = urlsplit(raw)
    if not parts.netloc:
        return ""
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(),
                       parts.path, parts.query, ""))


def github_atom(url: str) -> str | None:
    """github.com/owner/repo → its releases feed.

    A repo's release notes are the single most useful feed for a stack you
    actually run, and the atom URL is pure boilerplate nobody should type.
    """
    parts = urlsplit(url)
    if parts.netloc.removeprefix("www.") != "github.com":
        return None
    segments = [p for p in parts.path.split("/") if p]
    if len(segments) < 2:
        return None
    owner, repo = segments[0], segments[1].removesuffix(".git")
    return f"https://github.com/{owner}/{repo}/releases.atom"


def declared_feeds(html_text: str, base_url: str) -> list[str]:
    """<link rel="alternate" type="application/rss+xml" href="…">"""
    try:
        doc = lxml_html.fromstring(html_text)
    except Exception:
        return []
    out: list[str] = []
    for link in doc.xpath("//link[@rel]"):
        rels = (link.get("rel") or "").lower().split()
        if "alternate" not in rels:
            continue
        if (link.get("type") or "").lower() not in _FEED_TYPES:
            continue
        href = (link.get("href") or "").strip()
        if href:
            absolute = urljoin(base_url, href)
            if absolute not in out:
                out.append(absolute)
    return out


async def _try_feed(client: httpx.AsyncClient, url: str,
                    how: str) -> Discovered | None:
    """A candidate only counts if it parses and actually has entries."""
    try:
        r = await client.get(url, timeout=_TIMEOUT, follow_redirects=True)
        if r.status_code >= 400:
            return None
        title, entries = parse_feed(r.content[:_MAX_BYTES], str(r.url))
    except Exception as e:
        log.debug("feed candidate failed %s: %s", url, e)
        return None
    if not entries:
        return None
    return Discovered(url=str(r.url), title=title or str(r.url),
                      entries=len(entries), how=how)


async def discover(client: httpx.AsyncClient, raw: str) -> Discovered | None:
    """The feed for whatever the user typed, or None with nothing found."""
    url = normalise_input(raw)
    if not url:
        return None

    if (atom := github_atom(url)) is not None:
        if found := await _try_feed(client, atom, "github"):
            return found

    # The address itself might already be a feed — people paste those too.
    if direct := await _try_feed(client, url, "declared"):
        return direct

    try:
        page = await client.get(url, timeout=_TIMEOUT, follow_redirects=True)
        page.raise_for_status()
        body = page.text[:_MAX_BYTES]
        base = str(page.url)
    except Exception as e:
        log.info("could not read %s: %s", url, e)
        return None

    for candidate in declared_feeds(body, base):
        if found := await _try_feed(client, candidate, "declared"):
            return found

    root = urlunsplit((*urlsplit(base)[:2], "", "", ""))
    for path in _CONVENTIONAL:
        if found := await _try_feed(client, urljoin(root, path), "conventional"):
            return found
    return None
