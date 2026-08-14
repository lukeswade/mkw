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
from dataclasses import dataclass, field
from datetime import datetime

import httpx

from app.config import Settings
from app.db import Repo, utcnow
from app.llm import prompts
from app.llm.client import LLM
from app.models import RECENCY_LABELS
from app.research import gap as gap_stage
from app.research import planner as planner_stage
from app.research import synthesizer
from app.research.dedupe import (canonicalize, domain_of, interleave,
                                 lexical_overlap, rank_diverse)
from app.research.extractor import extract
from app.research.fetcher import Fetcher, SkipReason
from app.research.notes import (RELEVANCE_KEEP, Finding, finding_markdown,
                                take_notes)
from app.research.progress import ProgressBus
from app.research.searcher import (Searcher, SearxngError, cutoff_for,
                                   engine_tier)
from app.research.storage import RunStore, validate_citations

log = logging.getLogger(__name__)


# ---- depth semantics ---------------------------------------------------------

def breadth_for_depth(depth: int) -> int:
    return min(2 + depth, 8)

def max_docs_for_depth(depth: int) -> int:
    return 12 * depth

def max_llm_calls_for_depth(depth: int) -> int:
    return 20 + 15 * depth


# Below the keep threshold but not worthless — promoted only if the run would
# otherwise return nothing at all.
_WEAK_FLOOR = 2
_WEAK_MAX = 4


@dataclass
class _RunState:
    findings: list[Finding] = field(default_factory=list)
    weak: list[tuple[int, dict]] = field(default_factory=list)
    seen_urls: set[str] = field(default_factory=set)
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
            searcher = Searcher(cfg.searxng_url, http,
                                categories=cfg.search_categories,
                                max_concurrent=cfg.search_concurrency)
            fetcher = Fetcher(cfg, http)

            # 1. prior knowledge from earlier runs (knowledge layer, optional)
            prior = ""
            if self.rag is not None:
                prior, related = await self.rag.prior_knowledge(query, exclude_run=run_id)
                for other_id, score in related:
                    self.repo.add_run_link(run_id, other_id, "similar", score)
                if related:
                    self.bus.publish(run_id, "log",
                                     message=f"building on {len(related)} related earlier run(s)")

            if depth == 0:
                self.bus.publish(run_id, "phase", phase="chatting")
                self.repo.update_run(run_id, title=query[:100])
                store.update_meta(title=query[:100])
                messages = [
                    {"role": "system", "content": "You are a helpful AI answering a direct query."},
                    {"role": "user", "content": f"Query: {query}\n\nContext (if any):\n{prior}\n\nPlease answer the query based on the context and your knowledge."}
                ]
                final_text = await llm.chat_stream("chat", messages, self.bus, run_id, max_tokens=2048)
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
                breadth=breadth, prior=prior)
            self.repo.update_run(run_id, title=the_plan.title)
            store.update_meta(title=the_plan.title, brief=the_plan.brief)
            self.bus.publish(run_id, "plan", title=the_plan.title,
                             brief=the_plan.brief, subqueries=the_plan.subqueries)

            # 3. research rounds
            queries = the_plan.subqueries
            current_keywords = the_plan.keywords
            dry_rounds = 0
            stop_reason = "depth limit reached"
            for round_no in range(1, depth + 1):
                self._check_cancel()
                state.rounds_done = round_no
                self.bus.publish(run_id, "round_start", round=round_no,
                                 depth=depth, queries=queries)

                kept = await self._round(run_id, store, state, searcher, fetcher,
                                         llm, the_plan.brief, recency_desc, today,
                                         recency, queries, breadth, current_keywords)
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
                    recency_desc=recency_desc, round_no=round_no, depth=depth,
                    breadth=breadth, state_md=state.state_md,
                    new_findings=kept, searched=state.searched)
                state.state_md = gap.state_md
                store.write_round(round_no, self._round_md(
                    round_no, queries, kept, gap.saturated, state))
                self.bus.publish(run_id, "gap", saturated=gap.saturated,
                                 next_queries=gap.next_queries)

                dry_rounds = dry_rounds + 1 if len(kept) < 2 else 0
                if gap.saturated and round_no >= min(2, depth):
                    stop_reason = "saturated — no material gaps left"
                    break
                if dry_rounds >= 2:
                    stop_reason = "two consecutive dry rounds"
                    break
                if round_no == depth:
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
                                 searcher=searcher)

    # ---- one search round ------------------------------------------------------------
    async def _round(self, run_id, store, state, searcher, fetcher, llm,
                     brief, recency_desc, today, recency, queries, breadth, keywords) -> list[Finding]:
        results_lists = await asyncio.gather(
            *(searcher.search(q, recency) for q in queries),
            return_exceptions=True)
        merged_lists, errors = [], []
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
        if errors and not merged_lists:
            raise errors[0] if isinstance(errors[0], SearxngError) else RuntimeError(
                f"all searches failed: {errors[0]}")

        merged = interleave(merged_lists)
        # Stable sort keeps the round-robin order inside each tier, so every
        # sub-query still contributes. Ordering: a practical web page outranks
        # a journal abstract, and within a tier, results whose title/snippet
        # actually share words with the sub-query outrank engine filler —
        # every filler candidate that slips through costs a fetch plus a full
        # notes call before it scores 0/10.
        merged.sort(key=lambda r: (
            engine_tier(r.engine),
            -lexical_overlap(r.via_query, f"{r.title} {r.snippet}")))
        candidates = rank_diverse(merged, state.seen_urls, per_domain=2,
                                  limit=breadth * 3)
        for c in candidates:
            state.seen_urls.add(canonicalize(c.url))
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

        async def process(c) -> None:
            if self.cancel_requested:
                return
            try:
                fetched = await fetcher.fetch(c.url)
            except SkipReason as e:
                state.skipped += 1
                self.bus.publish(run_id, "source_skipped", url=c.url, reason=str(e))
                return
            doc = extract(fetched)
            if doc is None:
                state.skipped += 1
                self.bus.publish(run_id, "source_skipped", url=c.url,
                                 reason="no extractable text")
                return
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
                url=fetched.final_url, title=title,
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
                        "url": fetched.final_url, "title": title,
                        "domain": domain_of(fetched.final_url),
                        "published": notes.published_date or detected_date,
                        "relevance": notes.relevance, "summary": notes.summary,
                        "notes_md": notes.notes_md,
                        "key_facts": [f.model_dump() for f in notes.key_facts],
                        "query": c.via_query,
                    }))
                self.bus.publish(run_id, "source_skipped", url=c.url,
                                 reason=f"relevance {notes.relevance}/10")
                return
            # idx assignment + append happen with no await in between → atomic
            idx = len(state.findings) + 1
            finding = Finding(
                idx=idx, url=fetched.final_url, title=title,
                domain=domain_of(fetched.final_url),
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

        await asyncio.gather(*(process(c) for c in candidates))
        return kept

    # ---- finalization ---------------------------------------------------------------
    async def _finalize(self, run_id, store, state, llm, query, the_plan,
                        recency, recency_desc, today, stop_reason,
                        searcher=None) -> None:
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
                bus=self.bus, run_id=run_id)
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
