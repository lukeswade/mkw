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


def test_www_and_mobile_hosts_are_the_same_page():
    """domain_of folded www. but canonicalize did not, so the same page was
    fetched and read twice; en.m.wikipedia.org also bought a second slot
    against the per-domain diversity cap."""
    from app.research.dedupe import canonicalize, domain_of
    assert canonicalize("https://example.com/a") == canonicalize("https://www.example.com/a")
    assert canonicalize("https://example.com/a") == canonicalize("https://m.example.com/a")
    assert canonicalize("https://en.wikipedia.org/wiki/X") == \
        canonicalize("https://en.m.wikipedia.org/wiki/X")
    assert domain_of("https://en.m.wikipedia.org/wiki/X") == "en.wikipedia.org"
    # a host that merely starts with m is not a mobile host
    assert canonicalize("https://maps.example.com/a") != canonicalize("https://example.com/a")


def test_authority_sites_are_not_capped_per_domain():
    """The 2-per-domain cap stops an SEO farm flooding a round. A curated
    authority site is the opposite of that, and a forum with five good
    threads was yielding two. Same rule that already exempts them from
    triage: curated judgment outranks a heuristic."""
    from app.research.dedupe import rank_diverse
    from app.research.searcher import SearchResult

    def r(url):
        return SearchResult(url=url, title=url, snippet="", engine="bing",
                            published=None, score=1.0)
    pool = ([r(f"https://charm.li/manual/{i}") for i in range(5)]
            + [r(f"https://docs.charm.li/page/{i}") for i in range(3)]   # subdomain
            + [r(f"https://seo-farm.com/{i}") for i in range(5)])
    picked = rank_diverse(pool, set(), per_domain=2, limit=50,
                          uncapped=frozenset({"charm.li"}))
    hosts = [p.url.split("/")[2] for p in picked]
    assert hosts.count("charm.li") == 5
    assert hosts.count("docs.charm.li") == 3, "a subdomain of an authority counts"
    assert hosts.count("seo-farm.com") == 2, "everyone else is still capped"
    # and with nothing uncapped, the old behaviour holds exactly
    plain = rank_diverse(pool, set(), per_domain=2, limit=50)
    assert [p.url.split("/")[2] for p in plain].count("charm.li") == 2


def test_tracking_parameters_do_not_make_a_new_page():
    """Bing stamps a per-request msockid on every result link: one Capital One
    page was triaged seven times and fetched once in a single run."""
    a = canonicalize("https://www.capitalone.com/credit-cards/cabelas/?msockid=3f29d314d1d16c602a9bc4dcd02e6d04")
    b = canonicalize("https://www.capitalone.com/credit-cards/cabelas/?msockid=3212d88e1e3e6b1a2f8fcd131fc36a3b")
    assert a == b == "https://capitalone.com/credit-cards/cabelas"
    assert canonicalize("https://www.amazon.com/s?k=spey+rod&tag=vs-pl-x&ascsubtag=v1-c4") == "https://amazon.com/s?k=spey+rod"
    # `tag` is only tracking on Amazon; elsewhere it is a real facet
    assert canonicalize("https://blog.example.org/posts?tag=epoxy") == "https://blog.example.org/posts?tag=epoxy"
    assert canonicalize("https://www.bilibili.com/video/BV1aN/?spm_id_from=333.788&trackid=web_relat&uid=42") == "https://bilibili.com/video/BV1aN?uid=42"


def test_storefronts_are_blocked_by_default_including_subdomains():
    from app.research.dedupe import DEFAULT_BLOCKED, is_blocked
    for d in ("amazon.com", "ebay.com", "basspro.com", "capitalone.com"):
        assert d in DEFAULT_BLOCKED
    assert is_blocked("https://kdp.amazon.com/", DEFAULT_BLOCKED)          # subdomain
    assert is_blocked("https://www.ebay.com/p/1218654443", DEFAULT_BLOCKED)
    assert not is_blocked("https://www.rodbuilding.org/read.php?2,106022", DEFAULT_BLOCKED)


def test_shells_and_indexes_nothing_can_read():
    from app.research.dedupe import is_unreadable
    for url in ("https://www.bilibili.com/video/BV1b4sRzSEZz/",
                "https://www.msn.com/en-us/technology/software/i-installed-koreader",
                "https://www.scribd.com/document/935226823/Renogy-40a",
                "https://www.tumblr.com/widgets/share/tool?posttype=link",
                "https://www.reddit.com/r/koreader/",                      # subreddit index, not a thread
                "https://old.reddit.com/user/someone/"):
        assert is_unreadable(url), url
    for url in ("https://www.reddit.com/r/kindle/comments/6r6xtn/updating_paperwhite/",
                "https://kindlemodding.org/jailbreaking/WinterBreak/"):
        assert not is_unreadable(url), url


def test_vocabulary_overlap_is_stemmed_and_spans_question_brief_and_queries():
    from app.research.dedupe import shares_vocabulary, vocabulary
    vocab = vocabulary("Jailbreak Kindle Paperwhite 3 for BookOrbit",
                       "must establish the exploit for firmware 5.8.2.1",
                       "how to install KOReader on Kindle Paperwhite 3")
    assert shares_vocabulary("Jailbroken Kindles can now do more", vocab)      # kindles -> kindle
    assert shares_vocabulary("Installing KOReader", vocab)                     # installing -> install
    for junk in ("https://www.capitalone.com/credit-cards/cabelas/",
                 "Find Cheap Flights Worldwide - Google Flights",
                 "Ubuntu 22.04 LTS download https://releases.ubuntu.com/jammy/"):
        assert not shares_vocabulary(junk, vocab), junk


def test_root_and_index_pages_are_recognised_not_thread_pages():
    from app.research.dedupe import looks_like_index
    for url in ("https://www.montanaangler.com", "https://www.flyfishing.co.uk/",
                "https://www.salmonfishingforum.com/forums/", "https://fishingmagic.com/forums/",
                "https://ubuntu.com/download", "https://www.suffix.be/blog/"):
        assert looks_like_index(url), url
    for url in ("https://koreader.rocks/user_guide/", "https://www.rodbuilding.org/read.php?2,106022",
                "https://kindlemodding.org/jailbreaking/WinterBreak/"):
        assert not looks_like_index(url), url


def _yt(vid, author):
    return SearchResult(url=f"https://www.youtube.com/watch?v={vid}", title=vid, snippet="",
                        engine="youtube", published=None, score=1.0, author=author)


def test_one_source_is_a_channel_a_repo_a_subreddit_not_a_host():
    """All of YouTube counted as one domain: a question that asked for videos
    got two per round while the engine had twenty on point."""
    from app.research.dedupe import source_key
    assert source_key(_yt("a1", "Stretch Clendennen")) == "youtube:stretch clendennen"
    assert source_key(_yt("a2", "")) == "youtube:https://youtube.com/watch?v=a2"
    assert source_key(_r("https://github.com/bastardkb/charybdis/blob/main/README.md")) == "github.com:bastardkb/charybdis"
    assert source_key(_r("https://github.com/bastardkb")) == "github.com"
    assert source_key(_r("https://www.reddit.com/r/Trackballs/comments/abc/x/")) == "reddit:r/trackballs"
    assert source_key(_r("https://medium.com/@someone/post-1")) == "medium:@someone"
    assert source_key(_r("https://rodbuilding.org/read.php?2,1")) == "rodbuilding.org"


def test_video_results_from_different_channels_all_get_through():
    results = [_yt("v1", "A"), _yt("v2", "A"), _yt("v3", "A"),      # three from one channel
               _yt("v4", "B"), _yt("v5", "C"), _yt("v6", "C")]
    picked = rank_diverse(results, set(), per_domain=2, limit=10)
    assert [r.url[-2:] for r in picked] == ["v1", "v2", "v4", "v5", "v6"]     # channel A capped at 2, not YouTube at 2


def test_a_videos_run_lifts_the_cap_on_video_hosts_entirely():
    from app.research.dedupe import VIDEO_HOSTS
    results = [_yt(f"v{i}", "A") for i in range(5)] + [_r("https://x.com/1"), _r("https://x.com/2"), _r("https://x.com/3")]
    picked = rank_diverse(results, set(), per_domain=2, limit=10, uncapped=VIDEO_HOSTS)
    assert sum("youtube" in r.url for r in picked) == 5 and sum("x.com" in r.url for r in picked) == 2


def test_repos_and_subreddits_are_capped_per_repo_and_per_sub():
    results = [_r("https://github.com/o/r1/blob/a"), _r("https://github.com/o/r1/blob/b"), _r("https://github.com/o/r1/blob/c"),
               _r("https://github.com/o/r2"), _r("https://www.reddit.com/r/A/comments/1/x"), _r("https://www.reddit.com/r/B/comments/2/y")]
    picked = rank_diverse(results, set(), per_domain=2, limit=10)
    assert len(picked) == 5          # r1 capped at 2; r2, r/A, r/B each their own source
