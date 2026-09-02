"""URL canonicalization, cross-round dedupe, and domain-diversity ranking."""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, TypeVar
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.research.searcher import SearchResult

_TRACKING_KEYS = {"fbclid", "gclid", "msclkid", "igshid", "mc_cid", "mc_eid",
                  "ref", "ref_src", "source", "cmpid",
                  # Bing stamps a per-request msockid on every outbound link,
                  # so the same page arrived as a new URL every round: one
                  # Capital One page was triaged seven times and fetched once
                  # in a single run. The others are affiliate/analytics tags
                  # of the same kind (Amazon, Bilibili, Google, Marketo).
                  "msockid", "ascsubtag", "spm_id_from", "trackid", "srsltid",
                  "yclid", "gbraid", "wbraid", "mkt_tok", "igsh", "_ga"}
# Params that are tracking only on particular hosts: `tag` is Amazon's
# affiliate id but a real facet on many blogs and forums.
_HOST_TRACKING_KEYS = {"amazon.": {"tag"}}

T = TypeVar("T")


def _fold_host(host: str) -> str:
    """www. and mobile hosts serve the same page under a different name.

    domain_of already stripped www., so diversity capping saw one domain
    while canonicalize saw two URLs — the same page fetched and read twice,
    at the cost of a notes call each. en.m.wikipedia.org turned up alongside
    en.wikipedia.org in a live run, and mobile hosts additionally bought a
    second slot against the per-domain cap.
    """
    host = host.removeprefix("www.").removeprefix("m.")
    return host.replace(".m.", ".", 1)


def canonicalize(url: str) -> str:
    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()
    for default in (":80", ":443"):
        if netloc.endswith(default):
            netloc = netloc.rsplit(":", 1)[0]
    netloc = _fold_host(netloc)
    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    host_keys: set[str] = set()
    for marker, keys in _HOST_TRACKING_KEYS.items():
        if marker in netloc:
            host_keys |= keys
    query = urlencode([
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_KEYS
        and k.lower() not in host_keys
    ])
    return urlunsplit((scheme, netloc, path, query, ""))  # fragment dropped


def domain_of(url: str) -> str:
    host = urlsplit(url).netloc.lower().rsplit("@", 1)[-1].split(":")[0]
    return _fold_host(host)


# ---- pages nothing can read -------------------------------------------------
# The pipeline has exactly three acquisition paths: a YouTube caption
# transcript, a reddit .json thread, and fetch-plus-extract. A URL none of
# them can serve is not a weak candidate, it is a guaranteed zero — and it
# still costs a fetch (up to a 45s browser-solver attempt) before anything
# discovers that. One depth-8 run spent 20 of 52 candidates here.
#
# YouTube is deliberately absent: it has a transcript path, so it is only
# dropped later, and only when a video genuinely has no captions.
_UNREADABLE_HOSTS = frozenset({
    # login-walled: the fetch returns an interstitial, never the content
    "facebook.com", "instagram.com", "tiktok.com", "threads.net",
    "x.com", "twitter.com", "linkedin.com",
    # video hosts with no transcript path of their own
    "dailymotion.com", "vimeo.com", "twitch.tv", "rumble.com",
    "bitchute.com", "odysee.com", "sepiasearch.org", "bilibili.com",
    # shells that never carry the article: msn.com syndicates other sites'
    # stories inside a JavaScript app (4 of 4 extractions failed in one run,
    # the originals were already in the results); scribd is a login-walled
    # viewer; tumblr share widgets and the BSI link resolver are redirect
    # pages with no text of their own.
    "msn.com", "scribd.com", "tumblr.com", "linkresolver.bsigroup.com",
})
# A subreddit or user page is an index, not a thread: the JSON path only
# serves /comments/ URLs, and the generic fetch of /r/koreader/ cost a 45s
# browser render before extraction found nothing to read.
_REDDIT_SUFFIX = ("reddit.com",)
# PeerTube is a federation of hundreds of interchangeable mirrors — blocking
# them by name is whack-a-mole, but they all share a URL shape, and one query
# returned nine mirrors of a single video.
_PEERTUBE_PATH = re.compile(r"^/(?:videos/watch|w)/[0-9a-f]{8}-[0-9a-f-]{20,}",
                            re.IGNORECASE)

# Noise every user would otherwise discover the hard way. Merged with the
# user's own blocked_domains rather than replacing it.
DEFAULT_BLOCKED = frozenset({
    "dictionary.com", "thesaurus.com", "merriam-webster.com",
    "thefreedictionary.com", "dictionary.cambridge.org", "vocabulary.com",
    "wordnik.com", "definitions.net",
    "pinterest.com", "quora.com",
    # Storefronts. Across three depth-10 runs these produced 0 kept sources
    # and 45 candidates that were triaged, fetched or read before scoring
    # 0/10 — product listings, cart pages, and a co-branded credit card
    # that Bing returned for "Cabela's" seven times.
    "amazon.com", "ebay.com", "walmart.com", "homedepot.com", "lowes.com",
    "basspro.com", "cabelas.com", "capitalone.com", "aliexpress.com",
    "etsy.com", "bestbuy.com", "target.com", "acehardware.com", "shop.app",
})


def is_blocked(url: str, blocked: frozenset[str]) -> bool:
    """Subdomain-aware: kdp.amazon.com is amazon.com for this purpose."""
    return _under(domain_of(url), blocked)


def is_unreadable(url: str) -> bool:
    """True when no acquisition path could ever return text for this URL."""
    host = domain_of(url)
    if host in _UNREADABLE_HOSTS:
        return True
    if any(host.endswith("." + h) for h in _UNREADABLE_HOSTS):
        return True
    path = urlsplit(url).path or ""
    if _under(host, frozenset(_REDDIT_SUFFIX)) and "/comments/" not in path:
        return True
    return bool(_PEERTUBE_PATH.match(path))


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


def _under(host: str, domains: frozenset[str]) -> bool:
    return any(host == d or host.endswith("." + d) for d in domains)


def rank_diverse(results: list[SearchResult], seen: set[str], *,
                 per_domain: int = 2, limit: int = 12,
                 group=None,
                 uncapped: frozenset[str] = frozenset()) -> list[SearchResult]:
    """Pick fetch candidates: drop seen/duplicate URLs, cap per-source count.

    `group` decides what counts as one source, defaulting to the domain. A
    brief overrides it with the feed, because three GitHub release feeds all
    live on github.com — capping by domain let two entries through from all
    three combined, which is the opposite of what a curated reading list
    should do.

    `uncapped` domains ignore per_domain. The cap exists to stop one SEO farm
    flooding a round; a site the user has curated as authoritative is the
    opposite of that, and a forum holding five good threads on the question
    was yielding two per round. Authority sites already bypass triage for the
    same reason: curated judgment outranks a heuristic.
    """
    key = group or (lambda r: domain_of(canonicalize(r.url)))
    out: list[SearchResult] = []
    taken: set[str] = set()
    domain_counts: Counter[str] = Counter()
    for r in results:
        cu = canonicalize(r.url)
        if cu in seen or cu in taken:
            continue
        d = key(r)
        if domain_counts[d] >= per_domain and not _under(str(d), uncapped):
            continue
        taken.add(cu)
        domain_counts[d] += 1
        out.append(r)
        if len(out) >= limit:
            break
    return out


# ---- what a candidate must share with the question ----------------------------
def stem_token(t: str) -> str:
    """Crude suffix strip so 'kindles' meets 'kindle' and 'jailbreaking'
    meets 'jailbreak'. Not a stemmer; just enough to stop plurals and
    participles from reading as different words."""
    for suffix in ("ing", "ed"):
        if len(t) > 5 and t.endswith(suffix):
            return t[:-len(suffix)]
    if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]                      # kindles -> kindle, glues -> glue
    return t


def vocabulary(*texts: str) -> frozenset[str]:
    """Stemmed content tokens of every text given."""
    return frozenset(stem_token(t) for text in texts for t in _content_tokens(text))


def shares_vocabulary(text: str, vocab: frozenset[str]) -> bool:
    return any(stem_token(t) in vocab for t in _content_tokens(text))


# Root and index pages: a homepage, a forum index, a blog landing page, a
# downloads page. Eight of one run's wasted notes calls were these; but three
# root pages in another run were kept (a project's own site), so they are
# ranked last, not dropped — picked only when a round has room to spare.
_INDEX_PATH = re.compile(
    r"^/(?:forums?|blogs?|news|downloads?|community|threads|home|"
    r"index\.(?:php|html?))?$", re.IGNORECASE)


def looks_like_index(url: str) -> bool:
    parts = urlsplit(url)
    if parts.query:
        return False
    return bool(_INDEX_PATH.match((parts.path or "/").rstrip("/") or "/"))
