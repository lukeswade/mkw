"""URL canonicalization, cross-round dedupe, and domain-diversity ranking."""
from __future__ import annotations

from collections import Counter
from typing import Iterable, TypeVar
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.research.searcher import SearchResult

_TRACKING_KEYS = {"fbclid", "gclid", "msclkid", "igshid", "mc_cid", "mc_eid",
                  "ref", "ref_src", "source", "cmpid"}

T = TypeVar("T")


def canonicalize(url: str) -> str:
    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()
    for default in (":80", ":443"):
        if netloc.endswith(default):
            netloc = netloc.rsplit(":", 1)[0]
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    query = urlencode([
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_KEYS
    ])
    return urlunsplit((scheme, netloc, path, query, ""))  # fragment dropped


def domain_of(url: str) -> str:
    host = urlsplit(url).netloc.lower().rsplit("@", 1)[-1].split(":")[0]
    return host.removeprefix("www.")


def interleave(lists: Iterable[list[T]]) -> list[T]:
    """Round-robin merge so every sub-query contributes to the top ranks."""
    out: list[T] = []
    lists = [list(l) for l in lists]
    i = 0
    while any(lists):
        for l in lists:
            if i < len(l):
                out.append(l[i])
        i += 1
        lists = [l for l in lists if i < len(l)]
    return out


def rank_diverse(results: list[SearchResult], seen: set[str], *,
                 per_domain: int = 2, limit: int = 12) -> list[SearchResult]:
    """Pick fetch candidates: drop seen/duplicate URLs, cap per-domain count."""
    out: list[SearchResult] = []
    taken: set[str] = set()
    domain_counts: Counter[str] = Counter()
    for r in results:
        cu = canonicalize(r.url)
        if cu in seen or cu in taken:
            continue
        d = domain_of(cu)
        if domain_counts[d] >= per_domain:
            continue
        taken.add(cu)
        domain_counts[d] += 1
        out.append(r)
        if len(out) >= limit:
            break
    return out
