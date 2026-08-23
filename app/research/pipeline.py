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
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import unquote, urlsplit

import httpx

from app.config import Settings
from app.db import Repo, utcnow
from app.llm import prompts
from app.llm.client import LLM
from app.llm.json_utils import LLMJsonError
from app.models import RECENCY_LABELS, TriageOut
from app.research import gap as gap_stage
from app.research import planner as planner_stage
from app.research import synthesizer
from app.research import reddit, youtube
from app.research.dedupe import (canonicalize, domain_of, interleave,
                                 lexical_overlap, rank_diverse,
                                 similarity, text_fingerprint, _content_tokens)
from app.research.extractor import extract, extract_links
from app.research.fetcher import Fetcher, SkipReason
from app.research.notes import (RELEVANCE_KEEP, Finding, finding_markdown,
                                take_notes)
from app.research.progress import ProgressBus
from app.research.searcher import (VIDEO_ENGINES, Searcher, SearchResult,
                                   SearxngError, cutoff_for, engine_tier)
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
    return breadth * 4 + 2

def max_llm_calls_for_depth(depth: int) -> int:
    # A ceiling against runaways, not a target: every analyzed document is
    # one notes call, so the budget must comfortably exceed the source cap.
    return 25 + 2 * max_docs_for_depth(depth)

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

# Extracted text whose smaller fingerprint is ≥ this contained in an earlier
# document's is the same content: a scraped SEO clone or a syndicated copy.
# Genuinely distinct articles on one topic land far lower (~0.1-0.3).
_DUP_CONTAINMENT = 0.7

# A low-scoring page with less text than this is usually a section/index
# shell (service-manual directories are the canonical case) — the content
# lives one level down, so its own child links are worth following.
_STUB_CHARS = 600

# Index pages come in two shapes and only one of them is thin. charm.li's
# GX470 factory manual — the single most authoritative source for that query
# — is a 125,000-character table of contents across 4,761 links, which sails
# past any length threshold and then scores 0/10 because a wall of navigation
# text answers no question. The real content sits three levels below it.
_INDEX_ANCHOR_RATIO = 0.5     # anchor text as a share of visible text
_INDEX_MIN_LINKS = 25         # below this, a high ratio is just a short page
_INDEX_SHELL_CHARS = 1_500    # a small page that exists to list its children
_INDEX_LINK_LIMIT = 4_000     # fat directories bury the good link mid-document
_INDEX_CHILDREN = 4           # children followed per index page
_MAX_INDEX_HOPS = 3           # section → subsection → leaf, and stop
_INDEX_CHILD_MIN_MATCH = 0.5  # share of a child link that must be on-topic

# Link targets that are never worth a fetch: social shares and video, which
# either have no extractable text or are pure engagement chrome.
_REF_SKIP_DOMAINS = frozenset({
    "twitter.com", "x.com", "facebook.com", "linkedin.com", "instagram.com",
    "reddit.com", "youtube.com", "youtu.be", "pinterest.com", "t.me",
    "tiktok.com", "medium.com/m",
})


def child_links(source_url: str,
                links: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Same-domain links that descend deeper into this page's own path.

    This is what separates a directory from a leaf on a hierarchical
    documentation site: an index points down at its children, while the leaf
    at the bottom points only back up through its breadcrumbs.
    """
    base = urlsplit(source_url)
    # Descend from THIS page, not from its directory. Trimming to the last
    # slash made every sibling of a slashless URL look like a child, so
    # /manual/engine/spark-plug "contained" /manual/engine/oil-filter and any
    # leaf with one sibling link stopped reading as a leaf.
    prefix = base.path if base.path.endswith("/") else base.path + "/"
    out = []
    for url, anchor in links:
        part = urlsplit(url)
        if part.netloc != base.netloc:
            continue
        if part.path.startswith(prefix) and len(part.path) > len(prefix):
            out.append((url, anchor))
    return out


def looks_like_index(text: str, links: list[tuple[str, str]],
                     source_url: str) -> bool:
    """True when a page is a directory of links rather than content itself.

    Three tiers of charm.li's GX470 service manual make the case for each
    branch: "Repair and Diagnosis" is 125k characters of pure anchor text,
    "Spark Plug" is 872 characters that exist only to name two children, and
    "Spark Plug > Specifications" is 977 characters holding the actual
    factory electrode gap. Only the first two are indexes, and no length
    threshold alone can tell the second from the third — the tell is whether
    a page points at children at all.
    """
    body = " ".join((text or "").split())
    if not body:
        return False
    if len(body) < _STUB_CHARS:
        return True            # thin section shell — the original case
    if not child_links(source_url, links):
        return False           # a leaf, however long
    anchor_chars = sum(len(a) for _u, a in links)
    if len(links) >= _INDEX_MIN_LINKS \
            and anchor_chars / len(body) >= _INDEX_ANCHOR_RATIO:
        return True            # fat table of contents
    return len(body) < _INDEX_SHELL_CHARS


def _singularize(tokens: set[str]) -> set[str]:
    """Crude plural folding so link text matches the brief that describes it.

    Manuals title a page "Spark Plug" while a research brief asks about
    "spark plugs", and "Specifications" never matches "specification".
    Both sides get folded the same way, so exactness does not matter — only
    that the two agree.
    """
    return {t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss")
            else t for t in tokens}


def select_index_children(links: list[tuple[str, str]], *, source_url: str,
                          context: str, seen: set[str],
                          limit: int = _INDEX_CHILDREN) -> list[tuple[str, str]]:
    """Pick the children of an index page that are actually on-topic.

    Scored by how much of the LINK is relevant, not how much of the research
    context the link covers. lexical_overlap normalises by the context, which
    is a whole paragraph here, so every one of a directory's thousands of
    children scores near zero and the top-N is arbitrary. Normalising by the
    link instead separates them cleanly: "Spark Plug" is entirely on-topic,
    "ngk-2322-bue-surface-gap-spark-plug" mostly is not, "philosophy" is not
    at all.

    Only the last path segment is scored alongside the anchor — the rest of a
    breadcrumb path is shared with every sibling and cannot discriminate.
    """
    want = _singularize(_content_tokens(context))
    if not want:
        return []
    fresh, picked = [], set()
    for u, a in child_links(source_url, links):
        c = canonicalize(u)
        if c in seen or c in picked:
            continue
        picked.add(c)
        fresh.append((u, a))
    scored: list[tuple[float, str, str]] = []
    for url, anchor in fresh:
        tail = unquote(urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1])
        tokens = _singularize(
            _content_tokens(f"{anchor} {re.sub(r'[_\-.]', ' ', tail)}"))
        if not tokens:
            continue
        score = len(tokens & want) / len(tokens)
        if score < _INDEX_CHILD_MIN_MATCH:
            continue
        scored.append((score, url, anchor))
    scored.sort(key=lambda t: -t[0])
    return [(u, a) for _s, u, a in scored[:limit]]


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


@dataclass
class _RunState:
    findings: list[Finding] = field(default_factory=list)
    weak: list[tuple[int, dict]] = field(default_factory=list)
    seen_urls: set[str] = field(default_factory=set)
    fingerprints: list[tuple[frozenset[int], str]] = field(default_factory=list)
    searched: list[str] = field(default_factory=list)
    state_md: str = ""
    rounds_done: int = 0
    skipped: int = 0


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
        if len(drop) == len(candidates):
            # condemning everything is a broken verdict, not a judgment —
            # there is no signal in it to salvage, so ignore it wholesale
            return candidates
        # Triage is a cheap-junk filter, not a gatekeeper. Three separate
        # attempts to fix this in the prompt have failed: it keeps condemning
        # pages that answer the round's own queries by name (a GX470-specific
        # torque-spec page, NGK's own torque reference). So bound the damage
        # structurally — it may veto at most half a round, and the reprieve
        # goes to the best-ranked of the condemned. pick() already sorted
        # candidates best-first by engine tier then query overlap, so the
        # spared ones are those whose title/snippet most resemble the query
        # that found them: precisely the false negatives observed.
        cap = len(candidates) // 2
        if len(drop) > cap:
            spared = set(sorted(drop)[:len(drop) - cap])
            log.info("triage over-culled (%d of %d); sparing %d best-ranked",
                     len(drop), len(candidates), len(spared))
            drop -= spared
        if drop:
            state.skipped += len(drop)
            for i, c in enumerate(candidates):
                if i in drop:
                    self.bus.publish(run_id, "source_skipped", url=c.url,
                                     reason="dropped at triage")
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
        return frozenset(
            d.strip().lower().removeprefix("www.")
            for d in raw.replace(";", ",").split(",") if d.strip())

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
        query, depth, recency = row["query"], row["depth"], row["recency"]
        breadth = breadth_for_depth(depth)
        rounds = rounds_for_depth(depth)
        recency_desc = prompts.RECENCY_DESC[recency]
        today = datetime.now().date().isoformat()
        llm = self.llm_factory()
        state = _RunState()

        # Browser-shaped headers to match the browser UA: CDNs fingerprint on
        # more than the UA string, and a bare request still reads as a bot.
        headers = {
            "User-Agent": cfg.user_agent,
            "Accept": ("text/html,application/xhtml+xml,application/xml;"
                       "q=0.9,*/*;q=0.8"),
            "Accept-Language": "en-US,en;q=0.9",
        }
        timeout = httpx.Timeout(15.0, connect=10.0)
        limits = httpx.Limits(max_connections=cfg.fetch_concurrency * 2)
        async with httpx.AsyncClient(headers=headers, timeout=timeout,
                                     limits=limits) as http:
            run_categories = ""
            try:
                run_categories = (row["categories"] or "").strip()
            except (KeyError, IndexError):
                pass  # rows from before the migration
            searcher = Searcher(cfg.searxng_url, http,
                                categories=run_categories or cfg.search_categories,
                                max_concurrent=cfg.search_concurrency)
            fetcher = Fetcher(cfg, http)

            # 1. prior knowledge from earlier runs (knowledge layer, optional)
            prior = ""
            try:
                use_prior = bool(row["use_prior"])
            except (KeyError, IndexError):
                use_prior = True       # rows from before the migration
            if self.rag is not None and use_prior:
                prior, related = await self.rag.prior_knowledge(query, exclude_run=run_id)
                for other_id, score in related:
                    self.repo.add_run_link(run_id, other_id, "similar", score)
                if related:
                    self.bus.publish(run_id, "log",
                                     message=f"building on {len(related)} related earlier run(s)")

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
                authority=getattr(self.cfg, "authority_sites", ""))
            self.repo.update_run(run_id, title=the_plan.title)
            store.update_meta(title=the_plan.title, brief=the_plan.brief)
            self.bus.publish(run_id, "plan", title=the_plan.title,
                             brief=the_plan.brief, subqueries=the_plan.subqueries)

            # 3. research rounds
            queries = the_plan.subqueries
            current_keywords = the_plan.keywords
            dry_rounds = 0
            saturated_streak = 0
            stop_reason = "depth limit reached"
            for round_no in range(1, rounds + 1):
                self._check_cancel()
                state.rounds_done = round_no
                self.bus.publish(run_id, "round_start", round=round_no,
                                 depth=rounds, queries=queries)

                kept = await self._round(run_id, store, state, searcher, fetcher,
                                         llm, query, the_plan.brief,
                                         recency_desc, today, recency, queries,
                                         breadth, current_keywords)
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
                    authority=getattr(self.cfg, "authority_sites", ""))
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
                if not gap.next_queries:
                    stop_reason = "no further queries proposed"
                    break
                queries = gap.next_queries
                current_keywords = gap.keywords

            # 4. synthesis
            self._check_cancel()
            await self._finalize(run_id, store, state, llm, query, the_plan,
                                 recency, recency_desc, today, stop_reason,
                                 searcher=searcher,
                                 previous_overview=self._parent_overview(row))

    # ---- one search round ------------------------------------------------------------
    async def _round(self, run_id, store, state, searcher, fetcher, llm,
                     query, brief, recency_desc, today, recency, queries,
                     breadth, keywords) -> list[Finding]:
        results_lists = await asyncio.gather(
            *(searcher.search(q, recency) for q in queries),
            return_exceptions=True)
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
            pool = [r for r in pool
                    if domain_of(r.url) not in self._blocked_domains()]
            pool.sort(key=lambda r: (
                engine_tier(r.engine, promote),
                -lexical_overlap(r.via_query, f"{r.title} {r.snippet}")))
            chosen = rank_diverse(pool, state.seen_urls, per_domain=2,
                                  limit=limit)
            for c in chosen:
                state.seen_urls.add(canonicalize(c.url))
            return chosen

        merged = interleave(merged_lists)
        candidates = pick(merged, candidates_per_round(breadth))

        # Starved round: most results were duplicates or already seen. Pull
        # page 2 from the most productive queries before giving up — cheaper
        # than a dry round, which burns one of the run's two dry-round lives.
        if len(candidates) < breadth and pairs:
            extra = []
            for q, _res in sorted(pairs, key=lambda pr: -len(pr[1]))[:2]:
                try:
                    more = await searcher.search(q, recency, pageno=2)
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
        if total_results == 0 and searcher.blocked_engines:
            blocked = ", ".join(f"{k} ({v})" for k, v in
                                sorted(searcher.blocked_engines.items()))
            self.bus.publish(
                run_id, "log",
                message=f"no results — every engine refused: {blocked}")
        if not candidates:
            return []

        cutoff = cutoff_for(recency)
        kept: list[Finding] = []
        references: list[SearchResult] = []

        async def process(c, harvest_refs: bool = True, hop: int = 0) -> None:
            if self.cancel_requested:
                return
            # Three acquisition paths: video → caption transcript, reddit →
            # the thread's .json API, everything else → fetch + extract.
            fetched = None  # set only on the generic path; gates link harvest
            try:
                if vid := youtube.video_id(c.url):
                    final_url = c.url
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
                        raise SkipReason("no extractable text")
            except SkipReason as e:
                state.skipped += 1
                self.bus.publish(run_id, "source_skipped", url=c.url, reason=str(e))
                return
            # Near-duplicate collapse: scraped SEO clones and syndicated
            # copies read as on-topic, so left alone they burn a notes call
            # each and can be "kept" several times as separate sources.
            fp = text_fingerprint(doc.text)
            # Pages reached by descending an index share their site's chrome:
            # a manual leaf differs from its parent by a few hundred characters
            # of actual specification wrapped in the same banner, which reads
            # as a duplicate by containment. The clones this guard exists to
            # catch are syndicated copies, which are cross-domain by nature —
            # so on a descent, only judge against other domains.
            same_site_ok = hop > 0
            here = domain_of(final_url)
            dup = next((dom for other, dom in state.fingerprints
                        if similarity(fp, other) >= _DUP_CONTAINMENT
                        and not (same_site_ok and dom == here)), None)
            if dup is not None:
                state.skipped += 1
                self.bus.publish(run_id, "source_skipped", url=c.url,
                                 reason=f"duplicate of {dup} content")
                return
            state.fingerprints.append((fp, domain_of(final_url)))
            detected_date = doc.date or (c.published.date().isoformat()
                                         if c.published else None)
            if cutoff and detected_date:
                try:
                    if datetime.fromisoformat(detected_date) < cutoff:
                        state.skipped += 1
                        self.bus.publish(run_id, "source_skipped", url=c.url,
                                         reason=f"outside recency window ({detected_date})")
                        return
                except ValueError:
                    pass
            title = doc.title or c.title
            notes = await take_notes(
                llm, brief=brief, recency_desc=recency_desc, today=today,
                url=final_url, title=title,
                detected_date=detected_date, text=doc.text, keywords=keywords)
            if notes is None:
                state.skipped += 1
                self.bus.publish(run_id, "source_skipped", url=c.url,
                                 reason="unusable notes output")
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
                # A low-scoring index page is a directory over the real
                # content (FSM section pages): follow its best child links.
                # Gated on hop rather than harvest_refs — a manual's content
                # is several levels down, so the chase has to keep descending
                # while ordinary citation chasing stays at one hop.
                # Descend an index page toward the content it lists. A FAT
                # directory is only followed on a curated authority domain:
                # every corporate homepage on the web is link-dense and scores
                # 0/10, and letting those start a descent walked three hops
                # into NGK's company-philosophy pages and burned twelve notes
                # calls on them. Thin stubs are still followed anywhere, which
                # is the original behaviour.
                index_links = []
                if (fetched is not None and self.cfg.reference_chasing
                        and hop < _MAX_INDEX_HOPS):
                    deep_ok = domain_of(final_url) in self._authority_domains() \
                        or any(domain_of(final_url).endswith("." + a)
                               for a in self._authority_domains())
                    if deep_ok or len(doc.text) < _STUB_CHARS:
                        index_links = extract_links(fetched,
                                                    limit=_INDEX_LINK_LIMIT)
                        if not looks_like_index(doc.text, index_links,
                                                final_url):
                            index_links = []
                if index_links:
                    for ref_url, anchor in select_index_children(
                            index_links, source_url=final_url,
                            context=f"{query} {brief} {c.via_query}",
                            seen=state.seen_urls):
                        references.append(SearchResult(
                            url=ref_url, title=anchor or ref_url,
                            snippet=anchor, engine="reference",
                            published=None, score=0.0,
                            via_query=(f"linked from index page on "
                                       f"{domain_of(final_url)}")))
                self.bus.publish(run_id, "source_skipped", url=c.url,
                                 reason=f"relevance {notes.relevance}/10")
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
            finding.path = store.write_finding(idx, title, finding_markdown(finding))
            self.repo.add_finding(
                run_id=run_id, idx=idx, url=finding.url, title=finding.title,
                domain=finding.domain, published_date=finding.published,
                relevance=finding.relevance, path=finding.path,
                summary=finding.summary)
            self.bus.publish(run_id, "finding", idx=idx, title=finding.title,
                             domain=finding.domain, relevance=finding.relevance)

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

        # Each wave may surface a deeper index, so keep descending while
        # there is somewhere to go. Ordinary citation chasing still stops
        # after one hop (harvest_refs=False); only index pages re-arm this.
        hop = 0
        while references and not self.cancel_requested and hop < _MAX_INDEX_HOPS:
            chase = pick(references, _REFS_PER_ROUND)
            references = []
            if not chase:
                break
            hop += 1
            self.bus.publish(
                run_id, "log",
                message=(f"chasing {len(chase)} link(s) from kept sources "
                         f"and index pages (hop {hop})"))
            await asyncio.gather(
                *(process(c, harvest_refs=False, hop=hop) for c in chase))
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
                        searcher=None, previous_overview: str = "") -> None:
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
        findings: list[Finding] = []
        for r in self.repo.findings_for_run(run_id):
            # The finding .md file is the full record (summary, notes, quoted
            # evidence) — feed it whole rather than re-deriving its parts.
            try:
                body = (store.dir / r["path"]).read_text(encoding="utf-8")
            except OSError:
                body = r["summary"] or ""
            findings.append(Finding(
                idx=r["idx"], url=r["url"], title=r["title"],
                domain=r["domain"], published=r["published_date"],
                relevance=r["relevance"], summary=r["summary"] or "",
                notes_md=body, key_facts=[]))
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
