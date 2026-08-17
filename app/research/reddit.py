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


async def thread(fetcher: Fetcher, url: str) -> tuple[Extracted, str]:
    """(document, canonical_thread_url) for a reddit thread.

    Raises SkipReason on fetch failure or an unreadable/empty thread.
    """
    fetched = await fetcher.fetch(_json_url(url),
                                  extra_types=("application/json",))
    try:
        listings = json.loads(fetched.body)
        post = listings[0]["data"]["children"][0]["data"]
        comment_children = listings[1]["data"]["children"]
    except (ValueError, LookupError, TypeError) as e:
        raise SkipReason("unreadable reddit thread") from e

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
