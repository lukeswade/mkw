"""URL canonicalization, cross-round dedupe, and domain-diversity ranking."""
from __future__ import annotations

import re
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


_STOPWORDS = frozenset(
    "the and for with from that this what which how are was were been being "
    "have has had can could should would will may might must about into over "
    "under between best top guide 2024 2025 2026 2027".split())


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(t) >= 3 and t not in _STOPWORDS}


def lexical_overlap(query: str, text: str) -> float:
    """Fraction of the query's content words present in `text`.

    Deliberately crude — this ranks which candidates are worth a fetch and a
    full LLM notes call, it does not judge them. Its job is to push 'GitHub
    Desktop download' below 'SX1262 vs SX1276 comparison' for a LoRa query,
    since every junk candidate that gets through costs ~30-60s of local-model
    time before scoring 0/10.
    """
    query_tokens = _content_tokens(query)
    if not query_tokens:
        return 0.0
    return len(query_tokens & _content_tokens(text)) / len(query_tokens)


def text_fingerprint(text: str, k: int = 8) -> frozenset[int]:
    """Hashed word k-shingles of `text`, for near-duplicate detection.

    Python's salted hash() is fine here: fingerprints are only ever compared
    within one process (one run), never stored.
    """
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    if len(words) < k:
        return frozenset({hash(" ".join(words))} if words else ())
    return frozenset(hash(" ".join(words[i:i + k]))
                     for i in range(len(words) - k + 1))


def similarity(a: frozenset[int], b: frozenset[int]) -> float:
    """Containment of the smaller fingerprint in the larger.

    Containment rather than Jaccard so a scraped clone that pads the stolen
    article with extra junk still registers as a duplicate.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


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
