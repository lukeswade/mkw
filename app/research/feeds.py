"""RSS/Atom feeds as a search source.

A brief has no question to search — it has a reading list. FeedSearcher
satisfies the same small surface the pipeline asks of Searcher (search,
categories, blocked_engines, degraded), so everything downstream — fetching,
extraction, note-taking, deduplication, synthesis, export — runs unchanged.

Feeds are third-party XML, so the parser is hardened: entities are not
resolved and the network is never touched during parse. lxml comes in with
trafilatura, so this adds no dependency.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin

import httpx
from lxml import etree

from app.llm import prompts
from app.models import BriefFilterOut
from app.research.searcher import SearchResult, cutoff_for

log = logging.getLogger(__name__)

MAX_FEEDS = 40
MAX_ENTRIES_PER_FEED = 25
# Candidates one feed may contribute to a round. The web-search
# cap of two per domain collapsed three GitHub release feeds into
# two items total, because they share a domain.
PER_FEED_PER_ROUND = 6
_FEED_TIMEOUT = 20.0
# A feed that parsed fine and simply had nothing new. Reported, never a failure.
NO_ENTRIES = "no entries"
_MAX_FEED_BYTES = 8 * 1024 * 1024

# Entry text is a teaser, not the article: it exists to help triage decide
# whether the page is worth fetching, and the fetch is what gets read.
_SNIPPET_CHARS = 400

_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "dc": "http://purl.org/dc/elements/1.1/",
    "content": "http://purl.org/rss/1.0/modules/content/",
}


@dataclass
class FeedEntry:
    url: str
    title: str
    summary: str
    published: datetime | None
    feed_title: str


def _text(node) -> str:
    if node is None:
        return ""
    return " ".join("".join(node.itertext()).split())


def _parse_date(raw: str) -> datetime | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    try:                                   # RFC 822, the RSS convention
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        try:                               # ISO 8601, the Atom convention
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt is None:
        return None
    # The codebase stores naive datetimes and compares them against
    # cutoff_for(), which is built from a local datetime.now(). Feeds carry
    # real offsets, so convert into local time before dropping the tzinfo
    # rather than discarding the offset and shifting the entry by hours.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone().replace(tzinfo=None)


def _hardened_parser() -> etree.XMLParser:
    """Feeds are attacker-controllable XML.

    resolve_entities=False stops billion-laughs expansion, no_network=True
    stops an external DTD turning a parse into an outbound request.
    """
    return etree.XMLParser(resolve_entities=False, no_network=True,
                           huge_tree=False, recover=True)


def parse_feed(body: bytes, feed_url: str) -> tuple[str, list[FeedEntry]]:
    """(feed title, entries) from RSS 2.0 or Atom. Malformed feeds yield []."""
    try:
        root = etree.fromstring(body, parser=_hardened_parser())
    except (etree.XMLSyntaxError, ValueError):
        log.debug("unparseable feed: %s", feed_url)
        return "", []
    if root is None:
        return "", []

    channel = root.find("channel")
    if channel is not None:                                    # RSS
        feed_title = _text(channel.find("title")) or feed_url
        items, link_of, date_of = channel.findall("item"), _rss_link, _rss_date
    else:                                                      # Atom
        feed_title = _text(root.find("atom:title", _NS)) or feed_url
        items, link_of, date_of = (root.findall("atom:entry", _NS),
                                   _atom_link, _atom_date)

    entries: list[FeedEntry] = []
    for item in items[:MAX_ENTRIES_PER_FEED]:
        url = link_of(item)
        if not url:
            continue
        title = (_text(item.find("title"))
                 or _text(item.find("atom:title", _NS)) or url)
        summary = (_text(item.find("description"))
                   or _text(item.find("atom:summary", _NS))
                   or _text(item.find("atom:content", _NS)))
        entries.append(FeedEntry(
            url=urljoin(feed_url, url), title=title[:300],
            summary=summary[:_SNIPPET_CHARS], published=date_of(item),
            feed_title=feed_title[:120]))
    return feed_title, entries


def _rss_link(item) -> str:
    link = _text(item.find("link"))
    if link:
        return link
    guid = item.find("guid")
    raw = _text(guid)
    return raw if raw.startswith("http") else ""


def _rss_date(item) -> datetime | None:
    for tag in ("pubDate", "{http://purl.org/dc/elements/1.1/}date"):
        if (dt := _parse_date(_text(item.find(tag)))) is not None:
            return dt
    return None


def _atom_link(item) -> str:
    for link in item.findall("atom:link", _NS):
        rel = link.get("rel") or "alternate"
        if rel == "alternate" and link.get("href"):
            return link.get("href")
    first = item.find("atom:link", _NS)
    return (first.get("href") or "") if first is not None else ""


def _atom_date(item) -> datetime | None:
    for tag in ("atom:published", "atom:updated"):
        if (dt := _parse_date(_text(item.find(tag, _NS)))) is not None:
            return dt
    return None


def parse_feed_list(blob: str) -> list[str]:
    """One URL per line; blank lines and # comments ignored."""
    out = []
    for line in (blob or "").splitlines():
        url = line.split("#", 1)[0].strip()
        if url.startswith(("http://", "https://")) and url not in out:
            out.append(url)
    return out[:MAX_FEEDS]


class FeedSearcher:
    """Stands in for Searcher when a run is a brief.

    The query is ignored on purpose: a brief is defined by its feed list, not
    by a question. Everything downstream is unchanged.
    """

    def __init__(self, feed_urls: list[str], client: httpx.AsyncClient,
                 max_concurrent: int = 4, topic: str = "", llm=None):
        self.feed_urls = feed_urls
        self.client = client
        # A brief over unfiltered feeds reports whatever the authors wrote
        # that week. `topic` narrows it to what the reader follows; without
        # one, everything the feeds published is fair game.
        self.topic = (topic or "").strip()
        self.llm = llm
        self.filtered_out = 0
        self._sem = asyncio.Semaphore(max(1, max_concurrent))
        # The pipeline reads these off whatever searcher it was handed.
        self.categories = ""
        self.blocked_engines: dict[str, str] = {}
        self.searches = 0
        self.empty_searches = 0
        self._entries: list[SearchResult] | None = None

    @property
    def degraded(self) -> bool:
        """Every feed FAILED — distinct from 'the feeds had nothing new'.

        NO_ENTRIES went into the same dict as a fetch error, so a brief whose
        feeds were merely quiet came out as "This run found nothing because
        the search engines were unavailable", telling the reader to go and fix
        searxng/settings.yml over a week with no news in it (2026-09-09). The
        reason strings are still reported; they just no longer count as
        failures here.
        """
        if not self.feed_urls:
            return False
        failed = [u for u, why in self.blocked_engines.items() if why != NO_ENTRIES]
        return len(failed) == len(self.feed_urls)

    async def _one(self, url: str) -> list[FeedEntry]:
        async with self._sem:
            try:
                r = await self.client.get(url, timeout=_FEED_TIMEOUT,
                                          follow_redirects=True)
                r.raise_for_status()
                body = r.content[:_MAX_FEED_BYTES]
            except Exception as e:
                self.blocked_engines[url] = type(e).__name__
                log.warning("feed unreachable %s: %s", url, e)
                return []
        _title, entries = parse_feed(body, url)
        if not entries:
            self.blocked_engines.setdefault(url, NO_ENTRIES)
        return entries

    async def _by_topic(self, entries: list[FeedEntry]) -> list[FeedEntry]:
        """Narrow the week's items to the reader's stated interest.

        Titles only, one call: this decides what is worth opening, not what
        is worth keeping — the notes stage still scores everything fetched.
        A failure here keeps everything, because a broken filter must not
        silently empty the brief.
        """
        if not self.topic or self.llm is None or not entries:
            return entries
        listing = "\n".join(
            f"{i}. {e.title} — {e.feed_title}" for i, e in enumerate(entries))
        try:
            out = await self.llm.chat_json(
                "triage",
                [{"role": "user", "content": prompts.BRIEF_FILTER.format(
                    topic=self.topic, items=listing)}],
                BriefFilterOut, max_tokens=1200, temperature=0.0)
        except Exception as e:
            log.warning("brief topic filter failed, keeping everything: %s", e)
            return entries
        keep = {i for i in out.keep if 0 <= i < len(entries)}
        if not keep:
            log.warning("topic filter matched nothing; keeping all %d entries",
                        len(entries))
            return entries
        self.filtered_out = len(entries) - len(keep)
        return [e for i, e in enumerate(entries) if i in keep]

    async def search(self, query: str, recency: str, *,
                     pageno: int = 1) -> list[SearchResult]:
        # A feed has no page 2, and the pipeline's starved-round backfill will
        # ask for one. Returning [] is the honest answer.
        if pageno != 1:
            return []
        self.searches += 1
        if self._entries is None:
            gathered = await asyncio.gather(
                *(self._one(u) for u in self.feed_urls))
            cutoff = cutoff_for(recency)
            seen: set[str] = set()
            results: list[SearchResult] = []
            kept_entries: list[FeedEntry] = []
            for entry in sorted(
                    (e for batch in gathered for e in batch),
                    key=lambda e: e.published or datetime.min, reverse=True):
                if entry.url in seen:
                    continue
                if cutoff and entry.published and entry.published < cutoff:
                    continue
                seen.add(entry.url)
                kept_entries.append(entry)
            kept_entries = await self._by_topic(kept_entries)
            for entry in kept_entries:
                results.append(SearchResult(
                    url=entry.url, title=entry.title, snippet=entry.summary,
                    engine="feed", published=entry.published, score=1.0,
                    via_query=entry.feed_title or "feed"))
            self._entries = results
        if not self._entries:
            self.empty_searches += 1
        return list(self._entries)
