"""Reddit threads → post + comment text via the public .json API.

www.reddit.com serves a JavaScript shell with nothing to extract, and even
old.reddit.com HTML extracts as sidebar boilerplate more often than not. But
every thread URL answers with clean JSON when `.json` is appended — the post,
and the comment tree where the actual first-hand experience lives.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

from app.research.extractor import (MAX_TEXT_CHARS, MIN_TEXT_CHARS, Extracted)
from app.research.fetcher import Fetcher, SkipReason

log = logging.getLogger(__name__)

_REDDIT_HOSTS = frozenset({
    "reddit.com", "www.reddit.com", "old.reddit.com", "new.reddit.com",
    "m.reddit.com", "np.reddit.com",
})
_MAX_COMMENTS = 80


def is_thread(url: str) -> bool:
    parts = urlsplit(url)
    return (parts.netloc.lower() in _REDDIT_HOSTS
            and "/comments/" in parts.path)


def _json_url(url: str) -> str:
    parts = urlsplit(url)
    path = parts.path.rstrip("/") + ".json"
    return urlunsplit((parts.scheme or "https", parts.netloc, path,
                       "limit=100", ""))


def _walk(children: list, depth: int, out: list[str]) -> None:
    for child in children:
        if len(out) >= _MAX_COMMENTS:
            return
        if child.get("kind") != "t1":
            continue
        data = child.get("data") or {}
        body = (data.get("body") or "").strip()
        if body and body not in ("[deleted]", "[removed]"):
            score = data.get("score", 0)
            out.append(f"{'  ' * depth}[{score:+d}] {body}")
        replies = data.get("replies")
        if isinstance(replies, dict):
            _walk((replies.get("data") or {}).get("children") or [],
                  depth + 1, out)


def _text_of(nodes) -> str:
    return " ".join(nodes[0].text_content().split()) if nodes else ""


def _thread_from_html(fetched) -> tuple[str, str, list[str]]:
    """(title, selftext, comment_lines) parsed from an old.reddit thread page.

    The fallback for when reddit has revoked this IP's anonymous .json
    access (it blocks the API for days while serving HTML normally).
    old.reddit markup has been stable for a decade."""
    import lxml.html
    doc = lxml.html.fromstring(fetched.body.decode("utf-8", errors="replace"))
    title = _text_of(doc.xpath('//div[@id="siteTable"]'
                               '//a[contains(@class,"title")]'))
    selftext = _text_of(doc.xpath('//div[@id="siteTable"]'
                                  '//div[contains(@class,"usertext-body")]'))
    lines: list[str] = []
    comment_xp = ('.//div[contains(concat(" ",normalize-space(@class)," "),'
                  '" comment ")]')
    area = doc.xpath('//div[contains(@class,"commentarea")]')
    for el in (area[0].xpath(comment_xp) if area else []):
        if len(lines) >= _MAX_COMMENTS:
            break
        body = _text_of(el.xpath('(.//div[contains(@class,"usertext-body")])[1]'))
        if not body or body in ("[deleted]", "[removed]"):
            continue
        score = _text_of(el.xpath('(.//span[contains(@class,"score")])[1]'))
        depth = max(0, len(el.xpath(
            'ancestor::div[contains(concat(" ",normalize-space(@class)," "),'
            '" comment ")]')))
        prefix = f"[{score}] " if score else ""
        lines.append(f"{'  ' * depth}{prefix}{body}")
    return title, selftext, lines


async def _html_fallback(fetcher: Fetcher, url: str,
                         api_err: Exception) -> tuple[Extracted, str]:
    page = await fetcher.fetch(url)  # HTML; raises SkipReason itself
    title, selftext, comments = _thread_from_html(page)
    parts = [f"# {title}"] if title else []
    if selftext:
        parts.append(selftext)
    if comments:
        parts.append("## Comments\n\n" + "\n\n".join(comments))
    text = "\n\n".join(parts)
    if len(text) < MIN_TEXT_CHARS:
        raise SkipReason(f"reddit thread unreadable (api: {api_err})") \
            from api_err
    log.info("reddit .json unavailable (%s); read %s via old.reddit HTML",
             api_err, url)
    doc = Extracted(
        text=text[:MAX_TEXT_CHARS],
        title=f"{title} — reddit thread" if title else None,
        date=None)
    return doc, url


async def thread(fetcher: Fetcher, url: str) -> tuple[Extracted, str]:
    """(document, canonical_thread_url) for a reddit thread.

    Reads the .json API when this IP is allowed to; falls back to parsing
    the old.reddit HTML page (which reddit keeps serving even while the
    anonymous API is blocked). Raises SkipReason when both fail.
    """
    try:
        fetched = await fetcher.fetch(_json_url(url),
                                      extra_types=("application/json",))
        listings = json.loads(fetched.body)
        post = listings[0]["data"]["children"][0]["data"]
        comment_children = listings[1]["data"]["children"]
    except SkipReason as api_err:
        return await _html_fallback(fetcher, url, api_err)
    except (ValueError, LookupError, TypeError) as e:
        # a block page served as 200 parses as garbage — same fallback
        return await _html_fallback(fetcher, url,
                                    SkipReason("unparseable api response"))

    title = (post.get("title") or "").strip()
    subreddit = post.get("subreddit") or ""
    selftext = (post.get("selftext") or "").strip()
    parts = [f"# {title}"]
    if selftext and selftext not in ("[deleted]", "[removed]"):
        parts.append(selftext)
    comments: list[str] = []
    _walk(comment_children, 0, comments)
    if comments:
        parts.append("## Comments\n\n" + "\n\n".join(comments))
    text = "\n\n".join(parts)
    if len(text) < MIN_TEXT_CHARS:
        raise SkipReason("reddit thread has no text")

    date = None
    created = post.get("created_utc")
    if created:
        try:
            date = datetime.fromtimestamp(
                float(created), tz=timezone.utc).date().isoformat()
        except (ValueError, OSError, OverflowError):
            date = None
    permalink = post.get("permalink")
    canonical = (f"https://www.reddit.com{permalink}" if permalink else url)
    doc = Extracted(
        text=text[:MAX_TEXT_CHARS],
        title=f"{title} — r/{subreddit} (reddit thread)" if title else None,
        date=date)
    return doc, canonical
