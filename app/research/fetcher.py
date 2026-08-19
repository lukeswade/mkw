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
import re
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
    via: str = "http" # transport that produced it: http | impersonated | browser


class SkipReason(Exception):
    """Fetch skipped for a stated reason (not an error)."""


# Refusals that usually mean a bot wall rather than a real answer: worth
# retrying with a stronger disguise. 202 is the challenge-interstitial some
# forums serve; 520-526 are Cloudflare's own error band.
_CHALLENGE_CODES = frozenset({202, 401, 403, 405, 406, 429, 503})
_STATUS_RE = re.compile(r"http (\d+)")


def _challenge_status(reason: str) -> bool:
    m = _STATUS_RE.search(reason)
    if not m:
        return False
    code = int(m.group(1))
    return code in _CHALLENGE_CODES or 520 <= code <= 526


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
        """Fetch one URL. Raises SkipReason with a human-readable cause.

        When the plain client is refused with a bot-wall status, the fetch
        escalates: once with a real Chrome TLS fingerprint (curl_cffi), then —
        if a solver is configured — through a real headless browser that can
        pass JavaScript challenges.
        """
        async with self._sem:
            try:
                return await self._hops(url, extra_types, self._polite_get)
            except SkipReason as err:
                for attempt in self._escalations(extra_types):
                    if not _challenge_status(str(err)):
                        break
                    try:
                        return await attempt(url, extra_types)
                    except SkipReason as e:
                        err = e
                raise err

    def _escalations(self, extra_types: tuple[str, ...]) -> list:
        out = []
        if getattr(self.cfg, "browser_impersonation", True):
            out.append(lambda u, et: self._hops(u, et, self._curl_get))
        # The solver renders pages in a browser, so it only makes sense for
        # HTML — an API fetch (reddit .json) would come back wrapped in markup.
        if getattr(self.cfg, "browser_solver_url", "") and not extra_types:
            out.append(self._solver_get)
        return out

    async def _hops(self, url: str, extra_types: tuple[str, ...],
                    getter) -> Fetched:
        """The redirect-following loop, with every hop guard-checked."""
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
                result = await getter(current, extra_types)
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

    # ---- escalation transports ---------------------------------------------------
    async def _curl_get(self, url: str,
                        extra_types: tuple[str, ...] = ()):
        """One GET presenting a real Chrome TLS fingerprint.

        Most CDN bot walls (Cloudflare, Akamai) reject on the TLS handshake —
        python clients have a recognizable one no User-Agent can hide. libcurl
        built to impersonate Chrome recovers those pages without a browser.
        """
        try:
            from curl_cffi.requests import AsyncSession
        except ImportError as e:
            raise SkipReason("impersonation unavailable") from e
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
                async with AsyncSession(impersonate="chrome",
                                        timeout=25) as session:
                    resp = await session.get(
                        url, allow_redirects=False,
                        headers={"Accept-Language": "en-US,en;q=0.9"})
            except Exception as e:  # curl_cffi has its own error hierarchy
                raise SkipReason(f"fetch failed: {type(e).__name__}") from e
            finally:
                self._domain_last[domain] = time.monotonic()
        if resp.status_code in (301, 302, 303, 307, 308):
            return resp  # caller follows, with guards
        if resp.status_code != 200:
            raise SkipReason(f"http {resp.status_code} (impersonated)")
        ctype = (resp.headers.get("content-type") or "text/html").split(";")[0].strip().lower()
        if not any(ctype.startswith(t) for t in allowed):
            raise SkipReason(f"content-type {ctype}")
        return Fetched(url=url, final_url=url, content_type=ctype,
                       body=resp.content[:MAX_BYTES], via="impersonated")

    async def render(self, url: str) -> Fetched:
        """Render one page in the browser solver, regardless of status.

        The JS-shell fallback: a page that answered 200 but extracted to
        nothing usually builds its content client-side — one real render
        recovers it. Raises SkipReason (no solver configured, or it failed)."""
        if not getattr(self.cfg, "browser_solver_url", ""):
            raise SkipReason("no browser solver configured")
        return await self._solver_get(url)

    async def _solver_get(self, url: str,
                          extra_types: tuple[str, ...] = ()) -> Fetched:
        """Last resort: a FlareSolverr sidecar drives a real headless browser
        through the page's JavaScript challenge and hands back the HTML."""
        if not await self._host_allowed(url):
            raise SkipReason("blocked address (SSRF guard)")
        base = self.cfg.browser_solver_url.rstrip("/")
        try:
            resp = await self.client.post(
                f"{base}/v1",
                json={"cmd": "request.get", "url": url, "maxTimeout": 45000},
                timeout=60)
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            raise SkipReason(f"browser solver unreachable: {type(e).__name__}") from e
        solution = data.get("solution") or {}
        status = int(solution.get("status") or 0)
        if data.get("status") != "ok" or status >= 400 or not solution.get("response"):
            detail = str(status or data.get("message") or "no response")
            detail = detail.splitlines()[0][:120]  # not a stacktrace dump
            raise SkipReason(f"browser solver failed ({detail})")
        log.info("browser solver recovered %s", url)
        return Fetched(url=url, final_url=solution.get("url") or url,
                       content_type="text/html",
                       body=solution["response"].encode()[:MAX_BYTES],
                       via="browser")
