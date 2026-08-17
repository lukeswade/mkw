"""Polite async page fetching.

- SSRF guard: literal-IP and DNS-resolved targets must not be private,
  loopback, link-local, or reserved (config escape hatch: allow_private_fetch).
  DNS failure does NOT block — the request itself will fail naturally, and
  this keeps offline tests (mocked transports) working.
- robots.txt honored only when respect_robots is enabled (off by
  default; a fetch failure still means allow).
- Per-domain concurrency of 2 plus a minimum interval between hits.
- Manual redirect following so every hop is SSRF-checked.
- Streaming reads with a size cap.
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
import time
import urllib.robotparser
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import httpx

from app.config import Settings
from app.research.dedupe import domain_of

log = logging.getLogger(__name__)

MAX_BYTES = 3_000_000
MAX_REDIRECTS = 5
ALLOWED_TYPES = ("text/html", "application/xhtml", "text/plain", "text/xml",
                 "application/xml", "application/pdf")
DOMAIN_MIN_INTERVAL = 1.0  # seconds between hits on the same domain
# Slower lanes for hosts with strict unauthenticated rate limits (reddit
# allows roughly ten requests a minute before answering 403/429).
_DOMAIN_INTERVALS = {"reddit.com": 7.0}

# Domains that never yield text to an anonymous client — login walls or pure
# JS shells. Skipping up front saves the fetch and reports an honest reason
# instead of a mystery http 400 or "no extractable text".
_LOGIN_WALLED = frozenset({
    "instagram.com", "facebook.com", "m.facebook.com", "twitter.com", "x.com",
    "tiktok.com", "linkedin.com", "threads.net", "pinterest.com",
})

# Same content, server-rendered: www.reddit.com serves a JavaScript shell with
# nothing to extract, old.reddit.com serves the thread as plain HTML.
_HOST_REWRITES = {
    "reddit.com": "old.reddit.com",
    "www.reddit.com": "old.reddit.com",
    "m.reddit.com": "old.reddit.com",
    "new.reddit.com": "old.reddit.com",
}


def rewrite_host(url: str) -> str:
    parts = urlsplit(url)
    target = _HOST_REWRITES.get(parts.netloc.lower())
    if target is None:
        return url
    return urlunsplit((parts.scheme, target, parts.path, parts.query,
                       parts.fragment))


@dataclass
class Fetched:
    url: str          # requested URL
    final_url: str    # after redirects
    content_type: str
    body: bytes


class SkipReason(Exception):
    """Fetch skipped for a stated reason (not an error)."""


class Fetcher:
    def __init__(self, cfg: Settings, client: httpx.AsyncClient):
        self.cfg = cfg
        self.client = client
        self._sem = asyncio.Semaphore(cfg.fetch_concurrency)
        self._domain_sems: dict[str, asyncio.Semaphore] = {}
        self._domain_last: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    # ---- SSRF guard ---------------------------------------------------------
    async def _host_allowed(self, url: str) -> bool:
        if self.cfg.allow_private_fetch:
            return True
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            return False
        host = (parts.hostname or "").strip("[]")
        if not host:
            return False
        try:
            addrs = [ipaddress.ip_address(host)]
        except ValueError:
            try:
                infos = await asyncio.get_running_loop().getaddrinfo(
                    host, None, type=socket.SOCK_STREAM)
                addrs = [ipaddress.ip_address(info[4][0]) for info in infos]
            except (socket.gaierror, OSError, ValueError):
                return True  # unresolvable → let the request fail on its own
        return not any(
            a.is_private or a.is_loopback or a.is_link_local or a.is_reserved
            or a.is_multicast for a in addrs
        )

    # ---- robots -------------------------------------------------------------
    async def _robots_allows(self, url: str) -> bool:
        if not self.cfg.respect_robots:
            return True
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            parser = None
            try:
                resp = await self.client.get(f"{origin}/robots.txt", timeout=6)
                if resp.status_code == 200 and len(resp.content) < 500_000:
                    parser = urllib.robotparser.RobotFileParser()
                    parser.parse(resp.text.splitlines())
            except httpx.HTTPError:
                parser = None  # unreachable robots → allow
            self._robots[origin] = parser
        parser = self._robots[origin]
        return parser is None or parser.can_fetch(self.cfg.user_agent, url)

    # ---- politeness -----------------------------------------------------------
    def _domain_sem(self, domain: str) -> asyncio.Semaphore:
        if domain not in self._domain_sems:
            self._domain_sems[domain] = asyncio.Semaphore(2)
        return self._domain_sems[domain]

    async def _polite_get(self, url: str,
                          extra_types: tuple[str, ...] = ()) -> httpx.Response | Fetched | None:
        """One SSRF-checked, politeness-throttled GET without redirects."""
        allowed = ALLOWED_TYPES + extra_types
        domain = domain_of(url)
        interval = next((v for k, v in _DOMAIN_INTERVALS.items()
                         if domain == k or domain.endswith("." + k)),
                        DOMAIN_MIN_INTERVAL)
        async with self._domain_sem(domain):
            wait = self._domain_last.get(domain, 0.0) + interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                async with self.client.stream("GET", url, follow_redirects=False) as resp:
                    if resp.status_code in (301, 302, 303, 307, 308):
                        return resp  # caller follows
                    if resp.status_code != 200:
                        raise SkipReason(f"http {resp.status_code}")
                    ctype = (resp.headers.get("content-type") or "text/html").split(";")[0].strip().lower()
                    if not any(ctype.startswith(t) for t in allowed):
                        raise SkipReason(f"content-type {ctype}")
                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in resp.aiter_bytes():
                        chunks.append(chunk)
                        size += len(chunk)
                        if size > MAX_BYTES:
                            if ctype == "application/pdf":
                                raise SkipReason("pdf too large")
                            break  # keep the head of oversized HTML
                    body = b"".join(chunks)[:MAX_BYTES]
                    return Fetched(url=url, final_url=str(resp.url),
                                   content_type=ctype, body=body)
            finally:
                self._domain_last[domain] = time.monotonic()

    # ---- public -----------------------------------------------------------------
    async def fetch(self, url: str,
                    extra_types: tuple[str, ...] = ()) -> Fetched:
        """Fetch one URL. Raises SkipReason with a human-readable cause."""
        async with self._sem:
            current = url
            for _hop in range(MAX_REDIRECTS + 1):
                current = rewrite_host(current)  # also on every redirect hop
                if domain_of(current) in _LOGIN_WALLED:
                    raise SkipReason("login-walled site")
                if not await self._host_allowed(current):
                    raise SkipReason("blocked address (SSRF guard)")
                if not await self._robots_allows(current):
                    raise SkipReason("disallowed by robots.txt")
                try:
                    result = await self._polite_get(current, extra_types)
                except httpx.HTTPError as e:
                    raise SkipReason(f"fetch failed: {type(e).__name__}") from e
                if isinstance(result, Fetched):
                    return result
                # redirect
                location = result.headers.get("location")
                if not location:
                    raise SkipReason("redirect without location")
                current = str(httpx.URL(current).join(location))
            raise SkipReason("too many redirects")
