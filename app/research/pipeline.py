"""The per-run research state machine.

    plan → [search → fetch → extract → notes]×rounds → gap → … → synthesize
         → follow-ups → index

Everything durable is written incrementally: findings as they are accepted,
a round log after every round, events after every step — a crash loses only
in-flight work, never what's already on disk.
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlsplit

import httpx

from app.config import Settings
from app.db import Repo, utcnow, row_get
from app.llm import prompts
from app.llm.client import LLM
from app.llm.json_utils import LLMJsonError
from app.models import BRIEF_DEFAULT_QUERY, VERIFY_CLAIM_CAP, RECENCY_LABELS, TriageOut
from app.research import gap as gap_stage
from app.research import planner as planner_stage
from app.research import synthesizer
from app.research import feeds, matrix, reddit, verify, youtube
from app.research.dedupe import (DEFAULT_BLOCKED, VIDEO_HOSTS, canonicalize, domain_of, interleave, is_blocked, is_unreadable, lexical_overlap, looks_like_index, rank_diverse, shares_vocabulary, similarity, source_key, text_fingerprint, vocabulary)
from app.research.extractor import extract, looks_bot_walled, extract_links
from app.research.fetcher import Fetcher, SkipReason
from app.research.notes import (RELEVANCE_KEEP, Finding, finding_markdown,
                                take_notes)
from app.research.progress import ProgressBus
from app.research.searcher import (VIDEO_ENGINES, Searcher, SearchResult,
                                   SearxngError, categories_for_scope, cutoff_for,
                                   engine_order, engine_tier)
from app.research.storage import RunStore, validate_citations

log = logging.getLogger(__name__)


# ---- depth semantics ---------------------------------------------------------
# The UI depth 0-10 is a half-step scale: each step is worth half a research
# "unit" (one unit ≈ one full search round with its budgets). Depth 2 is one
# unit, 6 is three, 10 is five — twice the granularity where runs actually
# live, with genuine quick-look settings at 1 and 3. Depth 0 stays quick chat.

def effort_for_depth(depth: int) -> float:
    return depth / 2

def rounds_for_depth(depth: int) -> int:
    return max(1, math.ceil(effort_for_depth(depth)))

def breadth_for_depth(depth: int) -> int:
    return min(2 + math.ceil(effort_for_depth(depth)), 10)

def max_docs_for_depth(depth: int) -> int:
    # Slightly superlinear: the top of the scale is "deep research" (the
    # Perplexity/OpenAI-DR benchmark reads sources by the dozens-to-hundreds).
    # depth 2 → 13, 6 → 45, 8 → 64, 10 → 85; floor of 8 so even a quick
    # half-unit look can cite a handful of sources.
    effort = effort_for_depth(depth)
    return max(8, round(effort * (12 + effort)))

def candidates_per_round(breadth: int) -> int:
    # breadth*3 starved runs whose topics live on hard-to-search sites; the
    # wider net costs only fetches for candidates the ranker put below the
    # old cut line — the notes-call budget is still governed by triage and
    # relevance.
    #
    # Doubled again (was *4+2) because triage is ONE call per round over the
    # whole list, so a wider pool costs a longer prompt rather than more LLM
    # calls. 2.5x was tempting and rejected: at depth 8 it would put ~33 notes
    # calls in a round against a 153-call ceiling, so the last round would die
    # on "LLM call cap reached" — a widening that silently shortens the run.
    return breadth * 8 + 4

def max_llm_calls_for_depth(depth: int) -> int:
    # A ceiling against runaways, not a target: every analyzed document is
    # one notes call, so the budget must comfortably exceed the source cap.
    # 3x, not 2x, since candidates_per_round doubled: more documents are now
    # read per document kept, and this must stay a backstop rather than
    # become the thing that ends the run.
    return 25 + 3 * max_docs_for_depth(depth)

def saturation_patience(depth: int) -> int:
    """Consecutive 'saturated' verdicts needed before a run stops early.

    Models declare "saturated" cheaply; believing the first verdict made
    deep runs behave like shallow ones. Deep runs demand a second opinion."""
    return 1 if depth <= 6 else 2


# Below the keep threshold but not worthless — promoted only if the run would
# otherwise return nothing at all.
_WEAK_FLOOR = 2
_WEAK_MAX = 4

# Citation chasing: at most this many cited references are fetched per round.
_REFS_PER_ROUND = 4
# When gap analysis proposes nothing but the run is not saturated and rounds
# remain, the most productive queries so far are re-run on the next result
# page rather than ending the run. Pages past this are engine filler.
_FALLBACK_MAX_PAGE = 3
# Relevance for a page a claim check fetched and read but no verdict cited.
# Not zero: zero reads as "judged worthless" when it means "did not settle it".
_CONSULTED = 3

# Extracted text whose smaller fingerprint is ≥ this contained in an earlier
# document's is the same content: a scraped SEO clone or a syndicated copy.
# Genuinely distinct articles on one topic land far lower (~0.1-0.3).
_DUP_CONTAINMENT = 0.7

# A low-scoring page with less text than this is usually a section/index
# shell (service-manual directories are the canonical case) — the content
# lives one level down, so its own child links are worth following.
_STUB_CHARS = 600

# Link targets that are never worth a fetch: social shares and video, which
# either have no extractable text or are pure engagement chrome.
_REF_SKIP_DOMAINS = frozenset({
    "twitter.com", "x.com", "facebook.com", "linkedin.com", "instagram.com",
    "reddit.com", "youtube.com", "youtu.be", "pinterest.com", "t.me",
    "tiktok.com", "medium.com/m",
    # Tip jars. A kept blog post links to its own donation page with an
    # on-topic anchor ("support my Kindle work"), so the overlap score let it
    # through: both arms of one A/B read buymeacoffee.com/4dcube in full.
    "buymeacoffee.com", "ko-fi.com", "patreon.com", "paypal.me",
    "paypal.com", "liberapay.com", "github.com/sponsors", "opencollective.com",
})
# Site boilerplate a page links to from every footer: never a citation.
_REF_SKIP_PATH = re.compile(
    r"/(?:terms|tos|privacy|legal|cookies?|imprint|impressum|donate|sponsors?|"
    r"support-us|about|contact|login|signin|signup|register|subscribe|newsletter)"
    r"(?:[/.?#-]|$)", re.IGNORECASE)


def select_references(links: list[tuple[str, str]], *, source_url: str,
                      context: str, seen: set[str],
                      per_source: int = 3,
                      same_domain_ok: bool = False) -> list[tuple[str, str]]:
    """Rank a page's outbound links by how much they smell like citations.

    Same-domain links are navigation, not references; a link only qualifies
    if its anchor text or URL path shares content words with the research
    context — that is what separates 'further reading' from footer chrome.
    (`same_domain_ok` flips that rule for index/section pages, where the
    same-domain children ARE the content.)
    """
    source_domain = domain_of(source_url)
    scored: list[tuple[float, str, str]] = []
    picked_urls: set[str] = set()
    for url, anchor in links:
        domain = domain_of(url)
        if (domain == source_domain and not same_domain_ok) \
                or domain in _REF_SKIP_DOMAINS:
            continue
        if _REF_SKIP_PATH.search(urlsplit(url).path or ""):
            continue
        canonical = canonicalize(url)
        if canonical in seen or canonical in picked_urls:
            continue
        path_words = re.sub(r"[/_\-.]", " ", urlsplit(url).path)
        score = lexical_overlap(context, f"{anchor} {path_words}")
        if score <= 0:
            continue
        picked_urls.add(canonical)
        scored.append((score, url, anchor))
    scored.sort(key=lambda t: -t[0])
    return [(url, anchor) for _s, url, anchor in scored[:per_source]]


def _browser_headers(cfg) -> dict[str, str]:
    """CDNs fingerprint on more than the UA string; a bare request reads as a bot."""
    return {
        "User-Agent": cfg.user_agent,
        "Accept": ("text/html,application/xhtml+xml,application/xml;"
                   "q=0.9,*/*;q=0.8"),
        "Accept-Language": "en-US,en;q=0.9",
    }


@dataclass
class _RunState:
    findings: list[Finding] = field(default_factory=list)
    weak: list[tuple[int, dict]] = field(default_factory=list)
    seen_urls: set[str] = field(default_factory=set)
    # How many candidates one source may contribute per round, and what
    # counts as a source. Web search caps two per domain; a brief caps per
    # feed, since the feeds were chosen deliberately.
    per_source: int = 2
    group_by: object = None
    # Briefs score "is this a change I should know about", not "does this
    # answer the question" — a terse changelog is high value, not low.
    notes_template: str | None = None
    fingerprints: list[tuple[frozenset[int], str]] = field(default_factory=list)
    # Canonical URLs triage condemned but a rule put back (the half-round cap,
    # authority sites, productive domains). Their outcomes are the evidence for
    # whether those rules help or hurt.
    spared_urls: set[str] = field(default_factory=set)
    # Domains that have never produced a kept source for this install over
    # many reads and several runs (Repo.dead_domains). Ranked last, never
    # dropped: blocking is the reader's call, offered on the run page.
    dead_domains: frozenset = frozenset()
    # Kept sources per diversity key (channel, repo, subreddit, domain) — what
    # a source has earned toward a larger share of later rounds.
    kept_by_source: Counter = field(default_factory=Counter)
    # Credit carried in from related earlier runs (see seed_from_related).
    seed_by_source: Counter = field(default_factory=Counter)
    # query text -> scope ("web+video", ...) from the planner or gap stage.
    query_scope: dict[str, str] = field(default_factory=dict)
    searched: list[str] = field(default_factory=list)
    state_md: str = ""
    rounds_done: int = 0
    skipped: int = 0
    # Candidates killed before a fetch was spent on them, and every engine
    # that refused at any point in the run — both are reported at the end so
    # a thin run can say why it was thin.
    pre_dropped: int = 0
    blocked_engines: dict = field(default_factory=dict)
    # _finalize reports how thin the run was, and needs the depth to know
    # what "thin" means at this setting.
    depth: int = 0


# Domains so broad that "this domain already gave us a source" says nothing
# about the next page on it. Everywhere else, a domain that produced a kept
# source is a domain worth reading again.
# The per-source cap a source can earn. Round one gives every source two
# slots; a source that keeps what it is given gets more in later rounds.
# Measured over 28 runs: when a source went two for two in a round, its
# later-round picks were kept 58% of the time against a ~20% base rate —
# while 58% of sources that merely reached the cap had missed with both, so
# the cap must be earned, not raised for everyone.
_SOURCE_CAP_MAX = 6


def adaptive_cap(base: int, kept_so_far: int, ceiling: int = _SOURCE_CAP_MAX) -> int:
    return min(ceiling, base + kept_so_far)


def seed_from_related(kept_by_domain: dict[str, int]) -> dict[str, int]:
    """Round-one credit per source from what related earlier runs kept.

    Yield is topic-bound — a forum that keeps 85% on fly-rod questions keeps
    nothing on a Kindle question — so this is computed only over runs the
    knowledge layer judged related to THIS question. Three or more kept
    sources across them earn two extra slots (a start of four), exactly two
    earn one. Generic platforms are excluded: a kept reddit thread says
    nothing about the next subreddit.
    """
    seed: dict[str, int] = {}
    for domain, n in kept_by_domain.items():
        d = (domain or "").lower().removeprefix("www.")
        if not d or any(d == g or d.endswith("." + g) for g in _GENERIC_DOMAINS):
            continue
        if n >= 3:
            seed[d] = 2
        elif n == 2:
            seed[d] = 1
    return seed


_SITE_TOKEN = re.compile(r"\bsite:\S+\s*", re.IGNORECASE)


def limit_site_queries(queries: list[str], authority: frozenset[str]
                       ) -> list[tuple[str, str]]:
    """(query to search, query as written). At most one site:-scoped query per
    round keeps its scope unless the site is an authority; the rest are
    opened to the whole web by dropping the operator.

    Only Google CSE honours site: — the other engines drop it and return
    keyword matches from anywhere (enforced away by the searcher) — so a
    site: query is a bet on one engine. One round spent half its breadth on
    two of them and Google CSE refused the next round.
    """
    out: list[tuple[str, str]] = []
    scoped_used = False
    seen: set[str] = set()
    for q in queries:
        m = _SITE_TOKEN.search(q)
        new_q = q
        if m:
            site = m.group(0).split(":", 1)[1].strip().lower().removeprefix("www.")
            is_authority = any(site == a or site.endswith("." + a) for a in authority)
            if not is_authority:
                if scoped_used:
                    new_q = " ".join(_SITE_TOKEN.sub(" ", q).split()) or q
                else:
                    scoped_used = True
        key = new_q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append((new_q, q))
    return out


def triage_floor(n: int) -> int:
    """How many candidates a round keeps no matter what triage says.

    The cap used to be half the round. Measured on nine runs with the
    spared marker: pages the cap forced back in were kept 6% of the time
    (5 of 85) against 55% for the rest, and 47% scored 0-1 against 10%.
    When triage wants to drop most of a round, it is right; the rule-based
    reprieves (authority sites, productive domains) carry the protection
    against its false negatives. What remains is a floor: the best-ranked
    tenth of a round, never fewer than three, so a round cannot be emptied
    by one bad verdict.
    """
    return max(3, -(-n // 10))


def _under_any(key: str, domains: frozenset[str]) -> bool:
    host = key.split(":", 1)[0] if ":" in key else key
    return any(host == d or host.endswith("." + d) for d in domains)


_GENERIC_DOMAINS = frozenset({
    "wikipedia.org", "reddit.com", "youtube.com", "youtu.be", "github.com",
    "amazon.com", "medium.com", "stackexchange.com", "stackoverflow.com",
    "quora.com", "facebook.com", "x.com", "twitter.com",
})


def spare_productive(drop: set[int], candidates: list,
                     kept_domains: set[str]) -> set[int]:
    """Indices triage condemned on a domain that already yielded a kept
    source this run — deterministic memory outranking a title-level guess.

    Two depth-10 runs lost 15 such pages, including the single most
    authoritative guide for one question, because the model judged a title.
    Generic hosts are exempt: a kept reddit thread says nothing about the
    next reddit thread.
    """
    productive = {d for d in kept_domains
                  if not any(d == g or d.endswith("." + g) for g in _GENERIC_DOMAINS)}
    if not productive:
        return set()
    return {i for i in drop
            if any(domain_of(candidates[i].url) == d
                   or domain_of(candidates[i].url).endswith("." + d)
                   for d in productive)}


def filter_by_vocabulary(pool: list, vocab: frozenset[str],
                         exempt: frozenset[str] = frozenset(),
                         limit: int | None = None) -> tuple[list, list]:
    """Split candidates into (kept, dropped): dropped share not one content
    word — title, snippet or URL — with the question, the brief, or any query
    of the round. Search engines return these for a single common word in the
    query ("fly" -> flights, "fix" -> a stock ticker); each one cost a fetch
    and a full notes call before scoring 0/10. Measured on two depth-10 runs:
    at least 25 of 77 such calls, and 0 kept sources in one run, 1 upper
    bound in the other (a page titled "Glues" whose snippet the real check
    would have seen). Authority sites are exempt, as everywhere.

    With `limit`, filler is dropped only when the matching results alone can
    fill the round: it must never take a slot from a real match, and it is
    fetched only when there is nothing better to fetch. A live round has
    hundreds of results against a limit under a hundred, so the drop applies;
    a starved round keeps its filler, ranked last.
    """
    kept, dropped = [], []
    for r in pool:
        host = domain_of(r.url)
        text = f"{r.title} {r.snippet} {r.url}"
        if (any(host == a or host.endswith("." + a) for a in exempt)
                or shares_vocabulary(text, vocab)):
            kept.append(r)
        else:
            dropped.append(r)
    if limit is not None and len(kept) < limit:
        return kept + dropped, []
    return kept, dropped


class Pipeline:
    def __init__(self, cfg: Settings, repo: Repo, bus: ProgressBus, rag=None,
                 llm_factory=None):
        self.cfg = cfg
        self.repo = repo
        self.bus = bus
        self.rag = rag  # knowledge-layer hooks (M3); None → skipped
        self.llm_factory = llm_factory or (lambda: LLM(cfg))
        self.cancel_requested = False

    async def _triage(self, run_id: str, llm, query: str, brief: str,
                      queries: list[str], candidates: list,
                      state: "_RunState") -> list:
        """Drop candidates whose title/url/snippet already condemns them.

        Asked as a drop-list on purpose: an under-delivering model (lazy,
        truncated) then keeps extra junk — which relevance scoring catches —
        instead of silently discarding good candidates. An OVER-delivering
        model is the real hazard, so its verdict is capped at half a round."""
        lines = []
        for i, c in enumerate(candidates):
            snippet = " ".join((c.snippet or "").split())[:200]
            lines.append(f"{i}. {c.title[:120]} — {c.url[:150]} — {snippet}"
                         f" — via: {c.via_query[:80]}")
        prompt = prompts.TRIAGE.format(
            query=query, brief=brief,
            queries="\n".join(f"- {q}" for q in queries) or "- (none)",
            candidates="\n".join(lines))
        try:
            out = await llm.chat_json(
                "triage", [{"role": "user", "content": prompt}],
                TriageOut, max_tokens=400, temperature=0.0)
        except LLMJsonError as e:
            log.warning("triage degraded to keep-all: %s", e)
            return candidates
        drop = {i for i in out.drop if 0 <= i < len(candidates)}
        condemned = set(drop)
        authority = self._authority_domains()
        if authority:
            spared = {i for i in drop
                      if any(domain_of(candidates[i].url) == a
                             or domain_of(candidates[i].url).endswith("." + a)
                             for a in authority)}
            if spared:
                log.info("triage spared %d authority-site candidate(s)",
                         len(spared))
                drop -= spared
        productive = spare_productive(
            drop, candidates, {f.domain for f in state.findings})
        if productive:
            log.info("triage spared %d candidate(s) on domains that already "
                     "produced kept sources", len(productive))
            drop -= productive
        if len(drop) == len(candidates):
            # condemning everything is a broken verdict, not a judgment —
            # there is no signal in it to salvage, so ignore it wholesale
            return candidates
        # A floor, not a half-round cap: see triage_floor. The reprieve goes
        # to the best-ranked of the condemned — pick() sorted candidates
        # best-first — so a round can never be emptied by one bad verdict,
        # while a verdict that condemns most of a junk-heavy round stands.
        cap = len(candidates) - triage_floor(len(candidates))
        if len(drop) > cap:
            spared = set(sorted(drop)[:len(drop) - cap])
            log.info("triage over-culled (%d of %d); sparing %d best-ranked",
                     len(drop), len(candidates), len(spared))
            drop -= spared
        for i in condemned - drop:
            state.spared_urls.add(canonicalize(candidates[i].url))
        if drop:
            state.skipped += len(drop)
            for i, c in enumerate(candidates):
                if i in drop:
                    self.bus.publish(run_id, "source_skipped", url=c.url,
                                     reason="dropped at triage",
                                     title=(c.title or "")[:120],
                                     engine=c.engine or "")
            self.bus.publish(run_id, "log",
                             message=(f"triage dropped {len(drop)} of "
                                      f"{len(candidates)} candidates before "
                                      f"fetching"))
        return [c for i, c in enumerate(candidates) if i not in drop]

    def _authority_domains(self) -> frozenset[str]:
        """Domains from the curated authority list (first token of each line).

        Curating a site as authoritative is a standing judgment that outranks
        a title-level guess: triage dropped charm.li factory-manual pages
        because their URLs named a sibling model, losing the best sources in
        the run. Authority candidates therefore bypass pre-fetch filtering
        entirely — they still face full relevance scoring after being read."""
        out = set()
        for line in (getattr(self.cfg, "authority_sites", "") or "").splitlines():
            token = line.strip().split()[0].strip("-—:,") if line.strip() else ""
            token = token.lower().removeprefix("http://").removeprefix("https://")
            token = token.split("/")[0].removeprefix("www.")
            if "." in token:
                out.add(token)
        return frozenset(out)

    def _blocked_domains(self) -> frozenset[str]:
        raw = getattr(self.cfg, "blocked_domains", "") or ""
        own = frozenset(
            d.strip().lower().removeprefix("www.")
            for d in raw.replace(";", ",").split(",") if d.strip())
        # Union, not override: dictionary and thesaurus pages are noise for
        # every research query, and every user should not have to learn that
        # from a wasted run.
        return own | DEFAULT_BLOCKED

    # ---- entry point -----------------------------------------------------------
    async def execute(self, run_id: str) -> None:
        row = self.repo.get_run(run_id)
        if row is None:
            log.error("run %s not in DB", run_id)
            return
        store = RunStore(self.cfg.research_dir / row["dir"])
        if not self.bus.is_active(run_id):
            self.bus.attach(store)

        self.repo.update_run(run_id, status="running", started_at=utcnow())
        store.update_meta(status="running", started_at=utcnow())
        self.bus.publish(run_id, "status", status="running")

        try:
            await self._run(run_id, row, store)
        except asyncio.CancelledError:
            # sync-only cleanup — never await inside a CancelledError handler
            self.repo.update_run(run_id, status="cancelled",
                                 stop_reason="cancelled by user",
                                 finished_at=utcnow())
            store.update_meta(status="cancelled")
            self.bus.publish(run_id, "status", status="cancelled")
            self.bus.publish(run_id, "done", status="cancelled")
            raise
        except Exception as e:
            log.exception("run %s failed", run_id)
            self.repo.update_run(run_id, status="failed", error=str(e)[:2000],
                                 finished_at=utcnow())
            store.update_meta(status="failed", error=str(e)[:2000])
            self.bus.publish(run_id, "error", message=str(e)[:500])
            self.bus.publish(run_id, "done", status="failed")
        finally:
            self.bus.detach(run_id)

    # ---- main flow ----------------------------------------------------------------
    async def _run(self, run_id: str, row, store: RunStore) -> None:
        cfg = self.cfg
        if row_get(row, "kind", "research") == "verify":
            # Verification has no rounds and no gap analysis: it is a fan-out
            # over claims, not a search that deepens. It stays a run so it
            # inherits the library, exports, Ask and the progress stream.
            await self._verify_run(run_id, row, store)
            return
        query, depth, recency = row["query"], row["depth"], row["recency"]
        breadth = breadth_for_depth(depth)
        rounds = rounds_for_depth(depth)
        if row_get(row, "kind", "research") == "brief":
            # There is no second round: the feeds were read, and gap analysis
            # would only invent web searches a brief never asked for.
            rounds = 1
        recency_desc = prompts.RECENCY_DESC[recency]
        today = datetime.now().date().isoformat()
        llm = self.llm_factory()
        state = _RunState(depth=depth)

        # Browser-shaped headers to match the browser UA: CDNs fingerprint on
        # more than the UA string, and a bare request still reads as a bot.
        headers = _browser_headers(cfg)
        timeout = httpx.Timeout(15.0, connect=10.0)
        limits = httpx.Limits(max_connections=cfg.fetch_concurrency * 2)
        async with httpx.AsyncClient(headers=headers, timeout=timeout,
                                     limits=limits) as http:
            run_categories = (row_get(row, "categories", "") or "").strip()
            kind = row_get(row, "kind", "research")
            if kind == "brief":
                # A brief has a reading list, not a question. FeedSearcher
                # satisfies the same surface, so nothing downstream changes.
                # A named brief carries its own reading list and standing
                # interest; without one we fall back to the global setting,
                # which behaves as a single unnamed brief.
                saved = None
                if brief_id := row_get(row, "brief_id", None):
                    saved = self.repo.get_brief(int(brief_id))
                feed_blob = saved["feeds"] if saved else getattr(cfg, "feeds", "")
                feed_urls = feeds.parse_feed_list(feed_blob)
                if not feed_urls:
                    raise ValueError(
                        "no feeds configured — add them to this brief"
                        if saved else
                        "no feeds configured — add them in Settings")
                # For a brief the query is not a search — it is the
                # reader's standing interest, used to narrow the week's
                # items. A brief started with no question keeps everything.
                if saved and (saved["topic"] or "").strip():
                    topic = saved["topic"].strip()
                else:
                    topic = "" if row["query"].strip() == BRIEF_DEFAULT_QUERY \
                        else row["query"]
                searcher = feeds.FeedSearcher(feed_urls, http, topic=topic,
                                              llm=llm)
                # Yesterday's items are still inside today's window, so a
                # brief that does not remember what it already reported
                # repeats itself on day two.
                state.per_source = feeds.PER_FEED_PER_ROUND
                state.notes_template = prompts.NOTES_BRIEF
                state.group_by = lambda r: r.via_query or domain_of(r.url)
                # seen_urls holds CANONICAL urls — rank_diverse canonicalizes
                # each candidate before the membership test. Stored finding
                # urls keep their trailing slash, which canonicalize strips,
                # so raw urls here silently never matched and the brief
                # repeated itself.
                already = {canonicalize(u)
                           for u in self.repo.recent_finding_urls("brief")}
                state.seen_urls |= already
                self.bus.publish(
                    run_id, "log",
                    message=(f"reading {len(feed_urls)} feed(s)"
                             + (f", filtered to “{topic}”" if topic else "")
                             + f"; skipping {len(already)} item(s) already "
                               f"briefed"))
            else:
                searcher = Searcher(
                    cfg.searxng_url, http,
                    categories=run_categories or cfg.search_categories,
                    max_concurrent=cfg.search_concurrency)
            fetcher = Fetcher(cfg, http)

            # 1. prior knowledge from earlier runs (knowledge layer, optional)
            prior = ""
            related: list = []
            if self.rag is not None and bool(row_get(row, "use_prior", 1)):
                prior, related = await self.rag.prior_knowledge(query, exclude_run=run_id)
                for other_id, score in related:
                    self.repo.add_run_link(run_id, other_id, "similar", score)
                if related:
                    self.bus.publish(run_id, "log",
                                     message=f"building on {len(related)} related earlier run(s)")
            elif self.rag is not None and depth > 0 and kind == "research":
                # Memory is off for the planner, but sourcing may still learn
                # which sites paid off on this topic: that feeds the cap, not
                # the content.
                try:
                    related = await self.rag.related_runs(query, exclude_run=run_id)
                except Exception as e:  # the knowledge layer is optional
                    log.debug("related-run lookup failed: %s", e)
                    related = []
            if related and depth > 0 and kind == "research":
                seed = seed_from_related(self.repo.kept_domains_for_runs(
                    [rid for rid, _s in related]))
                if seed:
                    state.seed_by_source.update(seed)
                    self.bus.publish(
                        run_id, "log",
                        message=("earlier research on this topic kept sources from "
                                 + ", ".join(sorted(seed, key=lambda d: -seed[d]))
                                 + " — they start with a larger share"))

            if depth == 0:
                # Depth 0 is an instant answer in the style of a search
                # engine's AI overview: one search, snippet-grounded cited
                # summary, no page fetching. Falls back to plain chat when
                # search has nothing.
                self.bus.publish(run_id, "phase", phase="quick answer")
                self.repo.update_run(run_id, title=query[:100])
                store.update_meta(title=query[:100])
                results = []
                try:
                    results = (await searcher.search(query, recency))[:8]
                except Exception as e:
                    log.warning("depth-0 search failed, answering from the "
                                "model alone: %s", e)
                if results:
                    self.bus.publish(
                        run_id, "log",
                        message=f"grounding on {len(results)} search results")
                    snippets = "\n".join(
                        f"[{i}] {r.title}\n    {r.url}\n    {r.snippet}"
                        for i, r in enumerate(results, 1))
                    prior_block = (prompts.PRIOR_BLOCK.format(prior=prior)
                                   if prior else "")
                    messages = [{"role": "user",
                                 "content": prompts.QUICK_ANSWER.format(
                                     query=query, today=today,
                                     recency_desc=recency_desc,
                                     snippets=snippets,
                                     prior_block=prior_block)}]
                else:
                    messages = [
                        {"role": "system", "content": "You are a helpful AI answering a direct query."},
                        {"role": "user", "content": f"Query: {query}\n\nContext (if any):\n{prior}\n\nPlease answer the query based on the context and your knowledge."}
                    ]
                try:
                    final_text = await llm.chat_stream("chat", messages,
                                                       self.bus, run_id,
                                                       max_tokens=2048)
                except Exception:
                    log.warning("streaming quick answer failed, retrying "
                                "unstreamed", exc_info=True)
                    final_text = await llm.chat("chat", messages,
                                                max_tokens=2048)
                if results:
                    final_text, _ = validate_citations(final_text,
                                                       len(results))
                    final_text += "\n\n## Sources\n" + "\n".join(
                        f"{i}. [{r.title}]({r.url})"
                        for i, r in enumerate(results, 1))
                store.write_overview(final_text)
                self.repo.fts_add(run_id, "overview", query[:100], final_text)
                
                # properly finalize the run
                stats = {"rounds": 0, "urls_considered": 0, "sources_kept": 0, "sources_skipped": 0, "llm": llm.usage_summary()}
                self.repo.set_stats(run_id, stats)
                self.repo.update_run(run_id, status="completed", stop_reason="chat completed", finished_at=utcnow())
                store.update_meta(status="completed", stop_reason="chat completed", finished_at=utcnow(), stats=stats)
                self.bus.publish(run_id, "done", status="completed", stop_reason="chat completed", sources=0)
                return

            # 2. plan
            self.bus.publish(run_id, "phase", phase="planning")
            the_plan = await planner_stage.plan(
                llm, query=query, recency_desc=recency_desc, today=today,
                breadth=breadth, prior=prior,
                authority=getattr(self.cfg, "authority_sites", ""),
                variant=getattr(self.cfg, "planner_variant", "default"))
            self.repo.update_run(run_id, title=the_plan.title)
            store.update_meta(title=the_plan.title, brief=the_plan.brief)
            self.bus.publish(run_id, "plan", title=the_plan.title,
                             brief=the_plan.brief, subqueries=the_plan.subqueries)

            # 3. research rounds
            try:
                state.dead_domains = frozenset(d["domain"] for d in self.repo.dead_domains())
            except Exception as e:  # never let bookkeeping stop a run
                log.debug("dead-domain lookup failed: %s", e)
            state.query_scope.update(zip(the_plan.subqueries, the_plan.query_scopes))
            queries = self._apply_site_limit(run_id, state, the_plan.subqueries)
            current_keywords = the_plan.keywords
            dry_rounds = 0
            saturated_streak = 0
            pageno = 1
            stop_reason = "depth limit reached"
            for round_no in range(1, rounds + 1):
                self._check_cancel()
                state.rounds_done = round_no
                self.bus.publish(run_id, "round_start", round=round_no,
                                 depth=rounds, queries=queries,
                                 scopes=[state.query_scope.get(q, "") for q in queries])

                kept = await self._round(run_id, store, state, searcher, fetcher,
                                         llm, query, the_plan.brief,
                                         recency_desc, today, recency, queries,
                                         breadth, current_keywords,
                                         pageno=pageno)
                state.searched.extend(queries)

                if len(state.findings) >= max_docs_for_depth(depth):
                    stop_reason = "source cap reached"
                    break
                if llm.total_calls >= max_llm_calls_for_depth(depth):
                    stop_reason = "LLM call cap reached"
                    break

                self._check_cancel()
                self.bus.publish(run_id, "phase", phase="gap analysis",
                                 round=round_no)
                gap = await gap_stage.analyze(
                    llm, query=query, brief=the_plan.brief,
                    recency_desc=recency_desc, round_no=round_no, depth=rounds,
                    breadth=breadth, state_md=state.state_md,
                    new_findings=kept, searched=state.searched,
                    authority=getattr(self.cfg, "authority_sites", ""),
                    variant=getattr(self.cfg, "gap_variant", "default"))
                state.state_md = gap.state_md
                store.write_round(round_no, self._round_md(
                    round_no, queries, kept, gap.saturated, state))
                self.bus.publish(run_id, "gap", saturated=gap.saturated,
                                 next_queries=gap.next_queries)

                dry_rounds = dry_rounds + 1 if len(kept) < 2 else 0
                saturated_streak = saturated_streak + 1 if gap.saturated else 0
                if (saturated_streak >= saturation_patience(depth)
                        and round_no >= min(2, rounds)):
                    stop_reason = "saturated — no material gaps left"
                    break
                if dry_rounds >= 2:
                    stop_reason = "two consecutive dry rounds"
                    break
                if round_no == rounds:
                    break
                if gap.next_queries:
                    state.query_scope.update(zip(gap.next_queries, gap.next_query_scopes))
                    queries = self._apply_site_limit(run_id, state, gap.next_queries)
                    current_keywords = gap.keywords
                    pageno = 1
                    continue
                # The model named no new queries while reporting gaps remain.
                # Four of twelve depth-10 runs ended here at round 3 or 4 with
                # 4-40 sources of a possible 85 — starved, not saturated. The
                # stern re-ask in gap.analyze had already failed. Rather than
                # discard the remaining depth, go one page deeper on the
                # queries that have actually produced sources; seen URLs are
                # skipped, so page 2 costs only what is new. Dry rounds and
                # the saturation streak still end the run if this finds
                # nothing, and pages past _FALLBACK_MAX_PAGE are not worth it.
                fallback = self._productive_queries(state, breadth)
                if pageno >= _FALLBACK_MAX_PAGE or not fallback:
                    stop_reason = "no further queries proposed"
                    break
                pageno += 1
                queries = fallback
                self.bus.publish(
                    run_id, "log",
                    message=(f"gap analysis proposed nothing with "
                             f"{rounds - round_no} round(s) left — re-searching "
                             f"the {len(fallback)} most productive quer"
                             f"{'y' if len(fallback) == 1 else 'ies'} on "
                             f"page {pageno}"))

            # 4. synthesis
            self._check_cancel()
            await self._finalize(run_id, store, state, llm, query, the_plan,
                                 recency, recency_desc, today, stop_reason,
                                 searcher=searcher, fetcher=fetcher,
                                 previous_overview=self._parent_overview(row))

    def _record_outcome(self, run_id: str, c, outcome: str,
                        relevance: int | None = None) -> None:
        """One row per page read: what the install learns about a domain over
        many runs. Bookkeeping must never stop a run."""
        try:
            self.repo.record_outcome(run_id=run_id, url=canonicalize(c.url),
                                     domain=domain_of(c.url), engine=c.engine or "",
                                     outcome=outcome, relevance=relevance)
        except Exception as e:
            log.debug("outcome not recorded for %s: %s", c.url, e)

    def _apply_site_limit(self, run_id: str, state: "_RunState",
                          queries: list[str]) -> list[str]:
        pairs = limit_site_queries(queries, self._authority_domains())
        widened = [(new_q, old_q) for new_q, old_q in pairs if new_q != old_q]
        for new_q, old_q in widened:
            state.query_scope[new_q] = state.query_scope.get(old_q, "")
        if widened:
            self.bus.publish(
                run_id, "log",
                message=(f"{len(widened)} site:-scoped quer{'y' if len(widened) == 1 else 'ies'} "
                         f"beyond the first opened to the whole web (only Google CSE "
                         f"honours site:)"))
        return [new_q for new_q, _old in pairs]

    @staticmethod
    def _productive_queries(state: "_RunState", breadth: int) -> list[str]:
        """The searched queries that produced the most kept sources.

        Only queries that were actually searched qualify — a reference-chased
        page records where it was linked from, which is not searchable.
        Falls back to the planner's opening queries when nothing was kept.
        """
        searched = set(state.searched)
        counts: dict[str, int] = {}
        for f in state.findings:
            if f.query in searched:
                counts[f.query] = counts.get(f.query, 0) + 1
        ranked = sorted(counts, key=lambda q: -counts[q])[:breadth]
        return ranked or list(dict.fromkeys(state.searched))[:breadth]

    # ---- one search round ------------------------------------------------------------
    async def _round(self, run_id, store, state, searcher, fetcher, llm,
                     query, brief, recency_desc, today, recency, queries,
                     breadth, keywords, pageno: int = 1) -> list[Finding]:
        run_categories = getattr(searcher, "categories", None)

        def cats(q: str) -> str | None:
            # A query's scope narrows which categories it hits; the run's own
            # selection stays the ceiling. Web research only — a brief's
            # reading list is chosen, not searched, and its FeedSearcher has
            # no categories. Off (QUERY_SCOPES=off) = the old fan-out.
            if state.group_by is not None or not run_categories:
                return None
            if str(getattr(self.cfg, "query_scopes", "on")).lower() in ("off", "0", "false", "no"):
                return None
            return categories_for_scope(state.query_scope.get(q, ""), run_categories)

        def scoped(q: str, **kw):
            c = cats(q)
            return searcher.search(q, recency, **kw, **({"categories": c} if c else {}))

        results_lists = await asyncio.gather(
            *(scoped(q, pageno=pageno) for q in queries), return_exceptions=True)
        narrowed = {q: c for q in queries if (c := cats(q)) and c != run_categories}
        if narrowed:
            self.bus.publish(run_id, "log", message=(
                f"{len(narrowed)} of {len(queries)} queries searched a narrower scope: "
                + ", ".join(f"{q[:40]!r} → {c}" for q, c in list(narrowed.items())[:4])
                + (" …" if len(narrowed) > 4 else "")))
        merged_lists, errors, pairs = [], [], []
        for q, res in zip(queries, results_lists):
            if isinstance(res, BaseException):
                errors.append(res)
                self.bus.publish(run_id, "log",
                                 # str() on a timeout is empty, which produced
                                 # log lines that named no cause at all
                                 message=(f"search failed for {q!r}: "
                                          f"{type(res).__name__}"
                                          f"{f' — {res}' if str(res) else ''}"))
            else:
                for r in res:
                    r.via_query = q
                merged_lists.append(res)
                pairs.append((q, res))
        if errors and not merged_lists:
            raise errors[0] if isinstance(errors[0], SearxngError) else RuntimeError(
                f"all searches failed: {errors[0]}")

        # A run that selected the videos category wants video ranked with the
        # web results, not behind all of them.
        promote = (VIDEO_ENGINES if "video" in (searcher.categories or "")
                   else frozenset())

        def pick(pool: list, limit: int) -> list:
            # Stable sort keeps round-robin order inside each tier, so every
            # sub-query still contributes. Ordering: a practical web page
            # outranks a journal abstract; within a tier, results whose
            # title/snippet share words with their sub-query outrank engine
            # filler — every filler candidate that slips through costs a fetch
            # plus a full notes call before it scores 0/10.
            blocked = self._blocked_domains()
            before = len(pool)
            pool = [r for r in pool if not is_blocked(r.url, blocked)
                    and not is_unreadable(r.url)]
            state.pre_dropped += before - len(pool)
            # Web research only (a brief's reading list is chosen, not
            # searched, and its interest text may be empty): a candidate that
            # shares no word with the question, the brief, or any of this
            # round's queries is engine filler, not a weak lead.
            if state.group_by is None and len(vocab) >= 3:
                pool, filler = filter_by_vocabulary(
                    pool, vocab, exempt=self._authority_domains(), limit=limit)
                if filler:
                    state.pre_dropped += len(filler)
                    filler_dropped.extend(filler)
            pool.sort(key=lambda r: (
                not shares_vocabulary(f"{r.title} {r.snippet} {r.url}", vocab)
                if state.group_by is None else False,   # surviving filler last of all
                domain_of(r.url) in state.dead_domains,  # then never-productive domains
                *engine_order(r.engine, promote),   # tier, then keyed/promoted first
                looks_like_index(r.url),      # roots and indexes last in tier
                -lexical_overlap(r.via_query, f"{r.title} {r.snippet}")))
            # A question that selected videos wants the videos: the host cap
            # does not apply to video hosts then (channel grouping still does
            # not — every video may stand on its own).
            uncapped = self._authority_domains() | (VIDEO_HOSTS if promote else frozenset())
            # Web research only: a brief's per-feed share is fixed by design.
            cap_for = (None if state.group_by is not None else
                       (lambda key: adaptive_cap(
                           state.per_source,
                           state.kept_by_source[key] + state.seed_by_source[key])))
            chosen = rank_diverse(pool, state.seen_urls,
                                  per_domain=state.per_source,
                                  limit=limit, group=state.group_by,
                                  uncapped=uncapped, cap_for=cap_for)
            if cap_for is not None:
                taken = Counter(source_key(c) for c in chosen)
                earned = {k: n for k, n in taken.items()
                          if n > state.per_source and not _under_any(k, uncapped)}
                if earned:
                    # A share can be earned in this run or carried in from
                    # related earlier research; the log says which.
                    def _label(k: str, n: int) -> str:
                        name = k.split(":", 1)[-1] or k
                        tag = " (earlier research)" if state.seed_by_source[k] and not state.kept_by_source[k] else ""
                        return f"{name} ×{n}{tag}"
                    self.bus.publish(
                        run_id, "log",
                        message=("larger share this round: " + ", ".join(
                            _label(k, n) for k, n in sorted(earned.items(), key=lambda kv: -kv[1]))))
            for c in chosen:
                state.seen_urls.add(canonicalize(c.url))
            return chosen

        vocab = vocabulary(query, brief, *queries)
        filler_dropped: list = []
        merged = interleave(merged_lists)
        candidates = pick(merged, candidates_per_round(breadth))

        # Starved round: most results were duplicates or already seen. Pull
        # page 2 from the most productive queries before giving up — cheaper
        # than a dry round, which burns one of the run's two dry-round lives.
        if len(candidates) < breadth and pairs:
            extra = []
            for q, _res in sorted(pairs, key=lambda pr: -len(pr[1]))[:2]:
                try:
                    more = await scoped(q, pageno=pageno + 1)
                    for r in more:
                        r.via_query = q
                    extra.extend(more)
                except Exception as e:
                    log.debug("page-2 backfill failed for %r: %s", q, e)
            if extra:
                backfill = pick(extra, candidates_per_round(breadth) - len(candidates))
                if backfill:
                    self.bus.publish(run_id, "log",
                                     message=(f"round was starved — pulled "
                                              f"{len(backfill)} more candidates "
                                              f"from page 2"))
                    candidates.extend(backfill)

        # Triage: one fast-model look at titles/urls/snippets before anything
        # is fetched. A doomed candidate that slips through costs a fetch (up
        # to a 45s browser-solver attempt) plus minutes of local-model notes
        # time before scoring 0/10 — this call costs seconds and drops most
        # of them. Degrades to keeping everything.
        if len(candidates) > 3:
            candidates = await self._triage(run_id, llm, query, brief,
                                            queries, candidates, state)

        total_results = sum(len(l) for l in merged_lists)
        self.bus.publish(run_id, "searched", results=total_results,
                         candidates=len(candidates))
        if filler_dropped:
            self.bus.publish(
                run_id, "log",
                message=(f"{len(filler_dropped)} result(s) shared no words with "
                         f"the question or this round's queries and were not "
                         f"fetched"))
        # Without this there is no way to tell a filter that narrowed the
        # feeds from one that silently did nothing.
        if set_aside := getattr(searcher, "filtered_out", 0):
            self.bus.publish(
                run_id, "log",
                message=f"topic filter set aside {set_aside} off-topic item(s)")
        # Partial engine failure used to be invisible: `degraded` needs EVERY
        # search to come back empty, so four of five web engines could sit on
        # a CAPTCHA all run and nothing said a word — the results just quietly
        # got worse. Report whatever refused, whether or not anything landed.
        if searcher.blocked_engines:
            state.blocked_engines.update(searcher.blocked_engines)
            blocked = ", ".join(f"{k} ({v})" for k, v in
                                sorted(searcher.blocked_engines.items()))
            self.bus.publish(
                run_id, "log",
                message=(f"no results — every engine refused: {blocked}"
                         if total_results == 0
                         else f"{len(searcher.blocked_engines)} search engine(s) "
                              f"refused this round: {blocked}"))
        if not candidates:
            return []

        cutoff = cutoff_for(recency)
        kept: list[Finding] = []
        references: list[SearchResult] = []

        async def process(c, harvest_refs: bool = True) -> None:
            if self.cancel_requested:
                return
            # Three acquisition paths: video → caption transcript, reddit →
            # the thread's .json API, everything else → fetch + extract.
            fetched = None  # set only on the generic path; gates link harvest
            source_kind = ""
            try:
                if vid := youtube.video_id(c.url):
                    final_url = c.url
                    source_kind = "video"
                    doc = await youtube.transcript(fetcher.client, vid)
                    if doc is None:
                        raise SkipReason("no caption transcript")
                elif reddit.is_thread(c.url):
                    doc, final_url = await reddit.thread(fetcher, c.url)
                else:
                    fetched = await fetcher.fetch(c.url)
                    final_url = fetched.final_url
                    doc = extract(fetched)
                    if doc is None and fetched.via != "browser" \
                            and fetched.content_type.startswith("text/html") \
                            and getattr(self.cfg, "browser_solver_url", ""):
                        # A 200 that extracts to nothing is usually a JS
                        # shell — the content is built client-side. One real
                        # render in the solver recovers those pages.
                        try:
                            rendered = await fetcher.render(c.url)
                            doc = extract(rendered)
                            if doc is not None:
                                fetched = rendered
                                final_url = rendered.final_url
                        except SkipReason:
                            pass
                    if doc is None:
                        raise SkipReason(
                            "blocked by a bot wall"
                            if fetched is not None and looks_bot_walled(fetched)
                            else "no extractable text")
            except SkipReason as e:
                state.skipped += 1
                self.bus.publish(run_id, "source_skipped", url=c.url, reason=str(e),
                                 title=(c.title or "")[:120], engine=c.engine or "",
                                 spared=canonicalize(c.url) in state.spared_urls)
                self._record_outcome(run_id, c, "fail")
                return
            # A redirect target is the page we actually read. Only the URL we
            # asked for was remembered, so the same page could come back later
            # under its own name and be read again.
            if final_url and final_url != c.url:
                state.seen_urls.add(canonicalize(final_url))
            # Near-duplicate collapse: scraped SEO clones and syndicated
            # copies read as on-topic, so left alone they burn a notes call
            # each and can be "kept" several times as separate sources.
            fp = text_fingerprint(doc.text)
            dup = next((dom for other, dom in state.fingerprints
                        if similarity(fp, other) >= _DUP_CONTAINMENT), None)
            if dup is not None:
                state.skipped += 1
                self.bus.publish(run_id, "source_skipped", url=c.url,
                                 reason=f"duplicate of {dup} content",
                                 title=(c.title or "")[:120], engine=c.engine or "",
                                 spared=canonicalize(c.url) in state.spared_urls)
                return
            state.fingerprints.append((fp, domain_of(final_url)))
            detected_date = doc.date or (c.published.date().isoformat()
                                         if c.published else None)
            if cutoff and detected_date:
                try:
                    if datetime.fromisoformat(detected_date) < cutoff:
                        state.skipped += 1
                        self.bus.publish(run_id, "source_skipped", url=c.url,
                                         reason=f"outside recency window ({detected_date})",
                                 title=(c.title or "")[:120], engine=c.engine or "",
                                 spared=canonicalize(c.url) in state.spared_urls)
                        return
                except ValueError:
                    pass
            title = doc.title or c.title
            notes = await take_notes(
                llm, brief=brief, recency_desc=recency_desc, today=today,
                url=final_url, title=title,
                detected_date=detected_date, text=doc.text, keywords=keywords,
                template=state.notes_template, source_kind=source_kind)
            if notes is None:
                state.skipped += 1
                self.bus.publish(run_id, "source_skipped", url=c.url,
                                 reason="unusable notes output",
                                 title=(c.title or "")[:120], engine=c.engine or "",
                                 spared=canonicalize(c.url) in state.spared_urls)
                return
            if notes.relevance < getattr(self.cfg, "relevance_threshold",
                                         RELEVANCE_KEEP):
                state.skipped += 1
                # The notes call already ran, so this analysis is paid for.
                # Hold on to anything with a pulse: if the whole run ends up
                # empty, a thin answer beats a blank page — and returning
                # nothing when nine documents were read is its own failure.
                if notes.relevance >= _WEAK_FLOOR:
                    state.weak.append((notes.relevance, {
                        "url": final_url, "title": title,
                        "domain": domain_of(final_url),
                        "published": notes.published_date or detected_date,
                        "relevance": notes.relevance, "summary": notes.summary,
                        "notes_md": notes.notes_md,
                        "key_facts": [f.model_dump() for f in notes.key_facts],
                        "query": c.via_query,
                    }))
                # A thin low-scorer is often an index shell over the real
                # content (FSM section pages): follow its best child links.
                if (harvest_refs and fetched is not None
                        and self.cfg.reference_chasing
                        and len(doc.text) < _STUB_CHARS):
                    for ref_url, anchor in select_references(
                            extract_links(fetched), source_url=final_url,
                            context=f"{query} {brief} {c.via_query}",
                            seen=state.seen_urls, same_domain_ok=True):
                        references.append(SearchResult(
                            url=ref_url, title=anchor or ref_url,
                            snippet=anchor, engine="reference",
                            published=None, score=0.0,
                            via_query=(f"linked from index page on "
                                       f"{domain_of(final_url)}")))
                self.bus.publish(run_id, "source_skipped", url=c.url,
                                 reason=f"relevance {notes.relevance}/10",
                                 title=(c.title or "")[:120], engine=c.engine or "",
                                 spared=canonicalize(c.url) in state.spared_urls)
                self._record_outcome(run_id, c, "rejected", notes.relevance)
                return
            # idx assignment + append happen with no await in between → atomic
            idx = len(state.findings) + 1
            finding = Finding(
                idx=idx, url=final_url, title=title,
                domain=domain_of(final_url),
                published=notes.published_date or detected_date,
                relevance=notes.relevance, summary=notes.summary,
                notes_md=notes.notes_md, key_facts=[f.model_dump() for f in notes.key_facts],
                query=c.via_query,
            )
            state.findings.append(finding)
            kept.append(finding)
            state.kept_by_source[source_key(c)] += 1
            self._record_outcome(run_id, c, "kept", notes.relevance)
            finding.path = store.write_finding(idx, title, finding_markdown(finding))
            self.repo.add_finding(
                run_id=run_id, idx=idx, url=finding.url, title=finding.title,
                domain=finding.domain, published_date=finding.published,
                relevance=finding.relevance, path=finding.path,
                summary=finding.summary)
            self.bus.publish(run_id, "finding", idx=idx, title=finding.title,
                             domain=finding.domain, relevance=finding.relevance,
                             engine=c.engine or "",
                             spared=canonicalize(c.url) in state.spared_urls)

            # Citation chasing: the references a good source links to are
            # often better than anything a search engine returns, and
            # unreachable through one. Harvested here, fetched in a second
            # wave below (which does not harvest again — one hop per round).
            if harvest_refs and fetched is not None and self.cfg.reference_chasing:
                for url, anchor in select_references(
                        extract_links(fetched), source_url=final_url,
                        context=f"{query} {brief} {c.via_query}",
                        seen=state.seen_urls):
                    references.append(SearchResult(
                        url=url, title=anchor or url, snippet=anchor,
                        engine="reference", published=None, score=0.0,
                        via_query=f"cited by [{idx}] {finding.domain}"))

        await asyncio.gather(*(process(c) for c in candidates))

        if references and not self.cancel_requested:
            chase = pick(references, _REFS_PER_ROUND)
            if chase:
                self.bus.publish(
                    run_id, "log",
                    message=(f"chasing {len(chase)} reference(s) cited by "
                             f"kept sources"))
                await asyncio.gather(
                    *(process(c, harvest_refs=False) for c in chase))
        return kept

    # ---- finalization ---------------------------------------------------------------
    def _parent_overview(self, row) -> str:
        """The parent run's overview, for delta-focused synthesis."""
        parent_id = row["parent_run_id"]
        if not parent_id:
            return ""
        parent = self.repo.get_run(parent_id)
        if parent is None:
            return ""
        path = self.cfg.research_dir / parent["dir"] / "overview.md"
        try:
            return path.read_text(encoding="utf-8") if path.is_file() else ""
        except OSError:
            return ""

    async def _finalize(self, run_id, store, state, llm, query, the_plan,
                        recency, recency_desc, today, stop_reason,
                        searcher=None, fetcher=None,
                        previous_overview: str = "") -> None:
        thin = False
        if not state.findings and state.weak:
            thin = True
            for score, data in sorted(state.weak, key=lambda w: -w[0])[:_WEAK_MAX]:
                idx = len(state.findings) + 1
                f = Finding(idx=idx, **data)
                state.findings.append(f)
                f.path = store.write_finding(idx, f.title, finding_markdown(f))
                self.repo.add_finding(
                    run_id=run_id, idx=idx, url=f.url, title=f.title,
                    domain=f.domain, published_date=f.published,
                    relevance=f.relevance, path=f.path, summary=f.summary)
            state.skipped -= len(state.findings)
            self.bus.publish(
                run_id, "log",
                message=(f"nothing cleared the relevance bar; keeping the "
                         f"{len(state.findings)} best partial matches so the "
                         f"run returns something rather than nothing"))
        findings = state.findings
        store.write_sources(synthesizer.render_sources_md(findings))

        if findings:
            self.bus.publish(run_id, "phase", phase="synthesis",
                             sources=len(findings))
            overview = await synthesizer.synthesize(
                llm, query=query, title=the_plan.title, brief=the_plan.brief,
                recency_desc=recency_desc, today=today,
                state_md=state.state_md, findings=findings,
                bus=self.bus, run_id=run_id,
                previous_overview=previous_overview)
            if thin:
                overview = (
                    "> **Thin result.** No source strongly matched this "
                    "question, so the overview below is built from the best "
                    "partial matches available. Treat it as a starting point: "
                    "a narrower question, a broader recency window, or a retry "
                    "once search engines recover will usually do better.\n\n"
                    + overview)
                stop_reason = f"{stop_reason} (no strong matches)"
            overview, removed = validate_citations(overview, len(findings))
            if removed:
                self.bus.publish(run_id, "log",
                                 message=f"stripped invalid citations: {sorted(removed)}")
            fu = await synthesizer.follow_ups(llm, query=query, overview=overview)
        elif searcher is not None and searcher.degraded:
            # Every search came back empty *and* engines were reporting blocks.
            # Saying "no sources exist" here would be a lie about the topic.
            blocked = "\n".join(f"- **{k}** — {v}" for k, v in
                                 sorted(searcher.blocked_engines.items()))
            stop_reason = "search engines unavailable"
            overview = (
                f"# {the_plan.title}\n\n"
                f"**This run found nothing because the search engines were "
                f"unavailable, not because the topic has no sources.**\n\n"
                f"Every engine SearXNG queried refused the request:\n\n"
                f"{blocked}\n\n"
                f"This is usually temporary rate-limiting from too many "
                f"searches in a short window. Wait a few minutes and use "
                f"*Retry with same parameters*. If it persists, check the "
                f"engine mix in `searxng/settings.yml` — engines like Crossref, "
                f"OpenAlex and Stack Overflow do not rate-limit the way "
                f"Google and DuckDuckGo do.\n")
            fu = synthesizer.FollowUpsOut(items=[])
        else:
            overview = (f"# {the_plan.title}\n\nNo relevant sources were found "
                        f"for this query within the selected recency window "
                        f"({RECENCY_LABELS[recency].lower()}). Try a broader "
                        f"window or a rephrased query.\n")
            fu = synthesizer.FollowUpsOut(items=[])

        store.write_overview(overview)
        store.write_further(synthesizer.render_further_md(fu.items))
        followups_json = [f.model_dump() for f in fu.items]

        # vector index + cross-run similarity links (optional knowledge layer)
        if self.rag is not None and findings:
            self.bus.publish(run_id, "phase", phase="indexing")
            try:
                await self.rag.index_run(self.repo, run_id)
            except Exception:
                log.exception("indexing failed for %s (run still completes)", run_id)

        # library keyword index
        self.repo.fts_delete_run(run_id)
        self.repo.fts_add(run_id, "overview", the_plan.title, overview)
        for f in findings:
            self.repo.fts_add(run_id, "finding", f.title,
                              f"{f.summary}\n{f.notes_md}")

        stats = {
            "rounds": state.rounds_done,
            "searches": getattr(searcher, "searches", 0),
            "empty_searches": getattr(searcher, "empty_searches", 0),
            "blocked_engines": dict(getattr(searcher, "blocked_engines", {})),
            "urls_considered": len(state.seen_urls),
            "sources_kept": len(findings),
            "sources_skipped": state.skipped,
            "pre_dropped": state.pre_dropped,
            # Which rungs of the fetch ladder a run actually needed.
            "impersonated": getattr(fetcher, "impersonated", 0),
            "browser_solved": getattr(fetcher, "solved", 0),
            "pow_solved": getattr(fetcher, "pow_solved", 0),
            # What a healthy run of this depth would have kept, so the page
            # can tell a thin run from a normal one without re-deriving it.
            "sources_expected": max_docs_for_depth(state.depth),
            "llm": llm.usage_summary(),
        }
        self.repo.set_stats(run_id, stats)
        self.repo.update_run(run_id, status="completed", stop_reason=stop_reason,
                             finished_at=utcnow())
        store.update_meta(status="completed", stop_reason=stop_reason,
                          finished_at=utcnow(), stats=stats,
                          followups=followups_json)
        self.bus.publish(run_id, "done", status="completed",
                         stop_reason=stop_reason, sources=len(findings))

    # ---- re-synthesis ------------------------------------------------------------
    async def resynthesize(self, run_id: str) -> None:
        """Regenerate overview + follow-ups from a run's stored findings.

        No searching, no fetching, no note-taking — this exists for when the
        research succeeded but the final synthesis call didn't (a thinking
        model leaked its monologue, a truncation, a crash). Minutes instead
        of re-running everything.
        """
        row = self.repo.get_run(run_id)
        if row is None:
            raise ValueError(f"run {run_id} not found")
        store = RunStore(self.cfg.research_dir / row["dir"])
        meta = store.read_meta()
        findings = self._stored_findings(run_id, store)
        if not findings:
            raise ValueError("run has no stored findings to synthesize from")

        query = row["query"]
        title = meta.get("title") or row["title"] or query[:120]
        brief = meta.get("brief") or query
        recency_desc = prompts.RECENCY_DESC[row["recency"]]
        today = datetime.now().date().isoformat()
        llm = self.llm_factory()

        self.bus.publish(run_id, "phase", phase="synthesis",
                         sources=len(findings))
        self.bus.publish(run_id, "log",
                         message="re-synthesizing overview from stored findings")
        overview = await synthesizer.synthesize(
            llm, query=query, title=title, brief=brief,
            recency_desc=recency_desc, today=today, state_md="",
            findings=findings, bus=self.bus, run_id=run_id,
            previous_overview=self._parent_overview(row),
            # never replace a run's existing overview with a placeholder
            placeholder_on_failure=False)
        if not synthesizer.looks_like_document(overview):
            self.bus.publish(run_id, "log",
                             message=("re-synthesis still produced reasoning "
                                      "text, not a document — keeping the "
                                      "existing overview"))
            raise RuntimeError("synthesis output is not a document")
        overview, removed = validate_citations(overview, len(findings))
        if removed:
            self.bus.publish(run_id, "log",
                             message=f"stripped invalid citations: {sorted(removed)}")
        fu = await synthesizer.follow_ups(llm, query=query, overview=overview)

        store.write_overview(overview)
        store.write_further(synthesizer.render_further_md(fu.items))
        store.update_meta(followups=[f.model_dump() for f in fu.items],
                          resynthesized_at=utcnow())
        self.repo.fts_delete_run(run_id)
        self.repo.fts_add(run_id, "overview", title, overview)
        for f in findings:
            self.repo.fts_add(run_id, "finding", f.title, f.notes_md)
        if self.rag is not None:
            try:
                await self.rag.index_run(self.repo, run_id)
            except Exception:
                log.exception("re-indexing after resynthesis failed for %s",
                              run_id)
        self.bus.publish(run_id, "log", message="overview re-synthesized")
        self.bus.publish(run_id, "resynthesized", sources=len(findings))

    async def _verify_run(self, run_id: str, row, store: RunStore) -> None:
        """Check a pasted document's claims against the library, then the web."""
        cfg = self.cfg
        document = store.read_document()
        if not document.strip():
            raise ValueError("nothing to verify — the document was empty")

        llm = self.llm_factory()
        self.bus.publish(run_id, "phase", phase="extracting claims")
        claims, clipped = await verify.extract_claims(llm, document)
        if not claims:
            raise ValueError("no checkable claims were found in the document")

        checked, capped = verify.select_claims(claims, VERIFY_CLAIM_CAP)
        uncheckable = [c for c in claims if not c.checkable]
        self.bus.publish(
            run_id, "log",
            message=(f"{len(claims)} claim(s) found; checking {len(checked)}"
                     + (f", capped {len(capped)}" if capped else "")
                     + (f", {len(uncheckable)} not checkable" if uncheckable else "")))

        headers = _browser_headers(cfg)
        timeout = httpx.Timeout(15.0, connect=10.0)
        results: list[verify.Checked] = []
        async with httpx.AsyncClient(headers=headers, timeout=timeout) as http:
            # The run's own categories, like a research run — this always
            # took the global default, so a claim check could not be pointed
            # at the engines its subject actually lives in.
            searcher = Searcher(
                cfg.searxng_url, http,
                categories=((row_get(row, "categories", "") or "").strip()
                            or cfg.search_categories),
                max_concurrent=cfg.search_concurrency)
            fetcher = Fetcher(cfg, http)
            for i, claim in enumerate(checked, 1):
                if self.cancel_requested:
                    break
                self.bus.publish(run_id, "phase", phase="checking claims",
                                 round=i)
                results.append(await self._check_one(
                    run_id, llm, searcher, fetcher, claim))

        title = row["title"] or row["query"][:100]
        report = verify.render_report(title, results, skipped=capped,
                                      uncheckable=uncheckable, clipped=clipped)
        store.write_overview(report)
        findings = self._record_evidence(run_id, store, results)
        store.write_sources(synthesizer.render_sources_md(findings))
        store.update_meta(
            claims_checked=len(results), claims_found=len(claims),
            verdicts={v: sum(1 for r in results if r.verdict.verdict == v)
                      for v in ("supported", "contested", "unsupported",
                                "unverifiable")})
        self.repo.fts_delete_run(run_id)
        self.repo.fts_add(run_id, "overview", title, report)
        for f in findings:
            self.repo.fts_add(run_id, "finding", f.title, f.notes_md)
        if self.rag is not None:
            try:
                await self.rag.index_run(self.repo, run_id)
            except Exception:
                log.exception("indexing the claim check failed for %s", run_id)
        self.repo.update_run(run_id, title=title, status="completed",
                             stop_reason=f"{len(results)} claim(s) checked",
                             finished_at=utcnow())
        store.update_meta(status="completed")
        self.bus.publish(run_id, "status", status="completed")
        self.bus.publish(run_id, "done", status="completed",
                         sources=len(results),
                         stop_reason=f"{len(results)} claim(s) checked")

    def _record_evidence(self, run_id: str, store: RunStore,
                         results: list) -> list[Finding]:
        """Register the pages fetched while checking claims as this run's sources.

        Without this a claim check kept nothing: no Sources tab, no
        bibliography in any export, and nothing added to the library, so the
        reading it did could never inform a later run. Library evidence is
        excluded — it already belongs to the run it came from, and recording
        it again would duplicate that research under a new id.

        Relevance reflects how the page was actually used: the confidence of
        the best verdict that cited it, or _CONSULTED for one that was fetched
        and read without being cited. Scoring the uncited ones 0 read as
        "judged worthless" when the truth is "did not settle it" — and it is
        the common case, because library passages are numbered first and a
        verdict often cites only those.
        """
        best: dict[str, dict] = {}
        for r in results:
            used = set(r.verdict.sources)
            for e in r.evidence:
                if not e.url.startswith(("http://", "https://")):
                    continue                      # a /runs/… library passage
                entry = best.setdefault(e.url, {"evidence": e, "claims": [],
                                                "score": _CONSULTED,
                                                "cited": False})
                entry["claims"].append(r.claim.text)
                if e.n in used:
                    entry["cited"] = True
                    entry["score"] = max(entry["score"], r.verdict.confidence)

        findings: list[Finding] = []
        for idx, (url, entry) in enumerate(best.items(), 1):
            e = entry["evidence"]
            label = e.label.split(" — ", 1)[-1] if " — " in e.label else e.label
            claims = entry["claims"]
            how = "Cited by" if entry["cited"] else "Consulted while checking"
            summary = (f"{how} {len(claims)} claim(s), "
                       f"starting with: {claims[0][:140]}")
            f = Finding(idx=idx, url=url, title=label[:200] or url,
                        domain=domain_of(url), published=None,
                        relevance=entry["score"], summary=summary,
                        notes_md=e.text[:6000],
                        query="claim verification")
            f.path = store.write_finding(idx, f.title, finding_markdown(f))
            self.repo.add_finding(
                run_id=run_id, idx=idx, url=f.url, title=f.title,
                domain=f.domain, published_date=None, relevance=f.relevance,
                path=f.path, summary=f.summary)
            findings.append(f)
        return findings

    async def _check_one(self, run_id: str, llm, searcher, fetcher,
                         claim) -> "verify.Checked":
        """Library first; the web only when the library cannot settle it."""
        evidence = await verify.library_evidence(self.rag, claim.text)
        verdict = await verify.judge(llm, claim.text, evidence)
        if evidence and verify.settled(verdict):
            self.bus.publish(run_id, "log",
                             message=(f"“{claim.text[:70]}” — "
                                      f"{verdict.verdict} from your library"))
            return verify.Checked(claim=claim, verdict=verdict,
                                  evidence=evidence, via="library")

        kept = verify.trim_for_fallthrough(evidence)
        web = await self._web_evidence(searcher, fetcher, llm, claim.text,
                                       start_n=len(kept) + 1)
        combined = verify.renumber(kept + web)
        verdict = await verify.judge(llm, claim.text, combined)
        self.bus.publish(run_id, "log",
                         message=(f"“{claim.text[:70]}” — {verdict.verdict} "
                                  f"from {len(web)} web source(s)"))
        return verify.Checked(claim=claim, verdict=verdict,
                              evidence=combined,
                              via="library+web" if evidence else "web")

    async def _web_evidence(self, searcher, fetcher, llm, claim: str,
                            *, start_n: int) -> list["verify.Evidence"]:
        try:
            hits = await searcher.search(claim, "all")
        except Exception as e:
            log.warning("claim search failed: %s", e)
            return []
        out: list[verify.Evidence] = []
        for r in hits[:verify._WEB_PAGES_PER_CLAIM * 2]:
            if len(out) >= verify._WEB_PAGES_PER_CLAIM:
                break
            try:
                fetched = await fetcher.fetch(r.url)
                doc = extract(fetched)
            except (SkipReason, Exception) as e:      # noqa: B014
                log.debug("evidence fetch skipped %s: %s", r.url, e)
                continue
            if doc is None or not doc.text.strip():
                continue
            out.append(verify.Evidence(
                n=start_n + len(out), label=f"{domain_of(r.url)} — {r.title}",
                url=r.url, text=doc.text))
        return out

    def _stored_findings(self, run_id: str, store: RunStore) -> list[Finding]:
        """Rebuild Findings from disk for the post-hoc actions.

        The finding .md file is the full record (summary, notes, quoted
        evidence) — feed it whole rather than re-deriving its parts.
        """
        out: list[Finding] = []
        for r in self.repo.findings_for_run(run_id):
            try:
                body = (store.dir / r["path"]).read_text(encoding="utf-8")
            except OSError:
                body = r["summary"] or ""
            out.append(Finding(
                idx=r["idx"], url=r["url"], title=r["title"],
                domain=r["domain"], published=r["published_date"],
                relevance=r["relevance"], summary=r["summary"] or "",
                notes_md=body, key_facts=[]))
        return out

    async def build_matrix(self, run_id: str) -> None:
        """Turn a finished run's findings into a comparison table.

        Post-hoc like resynthesize: reads stored findings, writes matrix.md,
        touches neither the search nor the fetch layer. Refuses politely when
        the research was not a comparison in the first place.
        """
        row = self.repo.get_run(run_id)
        if row is None:
            raise ValueError(f"run {run_id} not found")
        store = RunStore(self.cfg.research_dir / row["dir"])
        findings = self._stored_findings(run_id, store)
        if not findings:
            raise ValueError("run has no stored findings to compare")

        meta = store.read_meta()
        title = meta.get("title") or row["title"] or row["query"][:120]
        self.bus.publish(run_id, "log",
                         message="building comparison matrix from stored sources")
        out, dropped = await matrix.build(
            self.llm_factory(), query=row["query"], findings=findings)

        if not out.applicable or len(out.entities) < 2:
            reason = out.reason or ("this research does not compare two or "
                                    "more named things")
            self.bus.publish(run_id, "log", message=f"no matrix built — {reason}")
            raise RuntimeError(reason)

        md = matrix.render_matrix_md(out, title=title, dropped=dropped)
        md, removed = validate_citations(md, len(findings))
        if removed:
            self.bus.publish(run_id, "log",
                             message=f"stripped invalid citations: {sorted(removed)}")
        store.write_matrix(md)
        store.update_meta(matrix_built_at=utcnow())
        self.repo.update_run(run_id, has_matrix=1)
        self.bus.publish(
            run_id, "log",
            message=(f"matrix built: {len(out.entities)} × "
                     f"{len(out.dimensions)} from {len(findings) - dropped} sources"))
        self.bus.publish(run_id, "matrix", entities=len(out.entities),
                         dimensions=len(out.dimensions))

    # ---- helpers -----------------------------------------------------------------
    def _check_cancel(self) -> None:
        if self.cancel_requested:
            raise asyncio.CancelledError()

    @staticmethod
    def _round_md(round_no, queries, kept, saturated, state) -> str:
        lines = [f"# Round {round_no}", "", "## Queries", ""]
        lines += [f"- {q}" for q in queries]
        lines += ["", f"## Sources kept ({len(kept)})", ""]
        lines += [f"- {f.citation_line()}" for f in kept] or ["_none_"]
        lines += ["", f"_Saturated: {saturated}_", "",
                  "## Research state after this round", "", state.state_md or "_empty_"]
        return "\n".join(lines) + "\n"
