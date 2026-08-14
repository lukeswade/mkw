from datetime import datetime

from app.research.dedupe import canonicalize, domain_of, interleave, rank_diverse
from app.research.searcher import SearchResult


def _r(url, score=1.0):
    return SearchResult(url=url, title=url, snippet="", engine="e",
                        published=None, score=score)


def test_canonicalize():
    assert canonicalize("HTTPS://Example.com/Path/") == "https://example.com/Path"
    assert canonicalize("https://example.com:443/a") == "https://example.com/a"
    assert (canonicalize("https://example.com/a?utm_source=x&id=2&fbclid=z")
            == "https://example.com/a?id=2")
    assert canonicalize("https://example.com/a#section") == "https://example.com/a"
    assert canonicalize("https://example.com") == "https://example.com/"


def test_domain_of():
    assert domain_of("https://www.example.com:8443/x") == "example.com"
    assert domain_of("http://sub.example.org/") == "sub.example.org"


def test_interleave():
    assert interleave([[1, 2, 3], [4], [5, 6]]) == [1, 4, 5, 2, 6, 3]


def test_rank_diverse_dedupes_and_caps_domains():
    results = [
        _r("https://a.com/1"),
        _r("https://a.com/1?utm_source=feed"),   # dupe after canonicalization
        _r("https://a.com/2"),
        _r("https://a.com/3"),                   # over per-domain cap
        _r("https://b.com/1"),
        _r("https://seen.com/old"),
    ]
    seen = {canonicalize("https://seen.com/old")}
    picked = rank_diverse(results, seen, per_domain=2, limit=10)
    urls = [r.url for r in picked]
    assert urls == ["https://a.com/1", "https://a.com/2", "https://b.com/1"]


def test_rank_diverse_limit():
    results = [_r(f"https://d{i}.com/x") for i in range(10)]
    assert len(rank_diverse(results, set(), per_domain=2, limit=4)) == 4


def test_lexical_overlap_ranks_filler_below_real_matches():
    """'GitHub Desktop download' burned a full local-model notes call before
    scoring 0/10 for a LoRa query. Overlap ranking keeps that from repeating."""
    from app.research.dedupe import lexical_overlap

    q = "SX1276 LoRa transceiver power consumption deep sleep"
    on_topic = lexical_overlap(q, "SX1262 vs SX1276 LoRa module comparison "
                                  "and power consumption guide")
    filler = lexical_overlap(q, "GitHub Desktop | Download for macOS")
    assert on_topic > 0.4
    assert filler == 0.0
    assert on_topic > filler


def test_lexical_overlap_ignores_stopwords_and_short_tokens():
    from app.research.dedupe import lexical_overlap

    # 'the', 'best', 'for', years — none of these should create false matches
    assert lexical_overlap("the best guide for 2026", "Best 2026 guide") == 0.0
    assert lexical_overlap("", "anything") == 0.0
    assert lexical_overlap("solar charging", "") == 0.0


def test_lexical_overlap_is_case_insensitive():
    from app.research.dedupe import lexical_overlap

    assert lexical_overlap("ESP32 LoRa", "esp32 lora field report") == 1.0
