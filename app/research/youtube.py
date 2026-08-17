"""YouTube videos → caption transcripts.

How-to knowledge increasingly lives in video, and a watch page is a
JavaScript shell with nothing to extract. But nearly every video carries a
caption track — authored or auto-generated. The caption URLs embedded in the
public watch page are dead ends (they 200 with an empty body unless the
request carries the web player's proof-of-origin token), so the track list is
requested from the InnerTube player API as the Android app instead, whose
caption URLs are not token-gated. Two small requests replace a 1.3 MB page
fetch and turn a dead candidate into a readable document.
"""
from __future__ import annotations

import asyncio
import logging
import re
import weakref
from urllib.parse import parse_qsl, urlsplit

import httpx

from app.research.extractor import (MAX_TEXT_CHARS, MIN_TEXT_CHARS, Extracted,
                                    _clean_date)

log = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YT_HOSTS = ("youtube.com", "music.youtube.com", "youtube-nocookie.com")

_PLAYER_API = "https://www.youtube.com/youtubei/v1/player"
_ANDROID_CONTEXT = {"client": {"clientName": "ANDROID",
                               "clientVersion": "20.10.38",
                               "androidSdkVersion": 30, "hl": "en"}}
_ANDROID_UA = "com.google.android.youtube/20.10.38 (Linux; U; Android 11) gzip"

# One semaphore per event loop (a module-level primitive binds to whichever
# loop touches it first, which breaks under test loops).
_politeness_sems: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = None  # type: ignore[assignment]


def _politeness() -> asyncio.Semaphore:
    global _politeness_sems
    if _politeness_sems is None:
        _politeness_sems = weakref.WeakKeyDictionary()
    loop = asyncio.get_running_loop()
    sem = _politeness_sems.get(loop)
    if sem is None:
        sem = _politeness_sems[loop] = asyncio.Semaphore(2)
    return sem


def video_id(url: str) -> str | None:
    """The 11-char video id, or None if `url` isn't a single-video page."""
    parts = urlsplit(url)
    host = parts.netloc.lower().removeprefix("www.").removeprefix("m.")
    vid = ""
    if host == "youtu.be":
        vid = parts.path.lstrip("/").split("/")[0]
    elif host in _YT_HOSTS:
        segments = parts.path.split("/")
        if parts.path == "/watch":
            vid = dict(parse_qsl(parts.query)).get("v", "")
        elif len(segments) > 2 and segments[1] in ("shorts", "embed", "live", "v"):
            vid = segments[2]
    return vid if _ID_RE.match(vid) else None


def _pick_track(tracks: list[dict]) -> dict:
    """Prefer authored English, then auto-generated English, then anything."""
    def rank(t: dict) -> tuple[bool, bool]:
        lang = (t.get("languageCode") or "").lower()
        return (not lang.startswith("en"), t.get("kind") == "asr")
    return min(tracks, key=rank)


def _caption_text(body: bytes) -> str:
    """Caption payload → plain text. The endpoint answers JSON (json3) or one
    of two XML shapes depending on what the track URL already pins."""
    head = body.lstrip()[:1]
    lines: list[str] = []
    if head == b"{":
        import json
        try:
            data = json.loads(body)
        except ValueError:
            return ""
        for event in data.get("events", []):
            text = "".join(s.get("utf8", "") for s in event.get("segs") or [])
            text = " ".join(text.split())
            if text:
                lines.append(text)
    elif head == b"<":
        try:
            import lxml.etree
            root = lxml.etree.fromstring(body)
        except Exception:
            return ""
        for node in root.iter("text", "p"):  # format1 / srv3
            text = " ".join("".join(node.itertext()).split())
            if text:
                lines.append(text)
    return "\n".join(lines)


async def transcript(client: httpx.AsyncClient, vid: str) -> Extracted | None:
    """Caption transcript plus title/date for one video id, or None."""
    headers = {"User-Agent": _ANDROID_UA}
    async with _politeness():
        try:
            resp = await client.post(
                _PLAYER_API, json={"context": _ANDROID_CONTEXT, "videoId": vid},
                headers=headers)
            if resp.status_code != 200:
                return None
            pr = resp.json()
        except (httpx.HTTPError, ValueError):
            log.debug("innertube player call failed for %s", vid, exc_info=True)
            return None
        tracks = (pr.get("captions", {})
                    .get("playerCaptionsTracklistRenderer", {})
                    .get("captionTracks") or [])
        if not tracks:
            return None
        base = _pick_track(tracks).get("baseUrl") or ""
        if base.startswith("/"):
            base = "https://www.youtube.com" + base
        if not base.startswith("http"):
            return None
        sep = "&" if "?" in base else "?"
        try:
            cap = await client.get(f"{base}{sep}fmt=json3", headers=headers,
                                   follow_redirects=True)
            if cap.status_code != 200:
                return None
        except httpx.HTTPError:
            log.debug("caption fetch failed for %s", vid, exc_info=True)
            return None
    text = _caption_text(cap.content)
    if len(text) < MIN_TEXT_CHARS:
        return None
    details = pr.get("videoDetails", {})
    micro = pr.get("microformat", {}).get("playerMicroformatRenderer", {})
    title = (details.get("title") or "").strip() or None
    author = (details.get("author") or "").strip()
    if title and author:
        title = f"{title} — {author} (video transcript)"
    return Extracted(
        text=text[:MAX_TEXT_CHARS], title=title,
        date=_clean_date(micro.get("publishDate") or micro.get("uploadDate")))
