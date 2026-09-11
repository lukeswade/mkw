"""Final synthesis: overview.md (map-reduce when notes exceed budget),
follow-up suggestions, and the sources bibliography."""
from __future__ import annotations

import logging
import re

from app.llm import prompts
from app.llm.client import LLM, LLMError, est_tokens
from app.llm.json_utils import LLMJsonError
from app.models import CandidatesOut, FollowUpsOut, RECENCY_LABELS, source_rank
from app.research.notes import Finding, render_facts

log = logging.getLogger(__name__)

# Est tokens of notes above which synthesis is map-reduced into digests.
# Kept high on purpose: one big call is SAFER than several small ones. A
# 9k-token single-call synthesis was verified clean, while the digest path
# adds calls that share a long common prefix — which is what made a poisoned
# KV-cache block repeat its damage across every digest of a run. Digesting is
# for genuinely enormous runs, not a safety measure.
# Raised from 28k on 2026-09-11 after measuring this install rather than
# guessing at it: the served model reports max_position_embeddings 262144, and
# a timed call put marginal prefill at 402 tok/s. est_tokens over-counts by
# about a third, so 100k here is ~75k real tokens — under a third of the
# window — and a measured A/B on a 66-source run showed single-call synthesis
# 23% faster than the four-batch digest (227s vs 297s) at identical citation
# coverage (30 vs 31 of 66). Above this the digest still runs, which is what
# keeps a 150-source run from spending six minutes in prefill.
_SINGLE_CALL_BUDGET = 100_000
_BATCH_BUDGET = 20_000         # est tokens per map batch
_DIGEST_BASE_TOKENS = 4000     # room for a digest covering one part
_DIGEST_PER_PART_TOKENS = 600  # ...plus this for each extra part it carries
_DIGEST_MAX_TOKENS = 8000
_STRONG_UNCITED = 7            # relevance at which "read but unused" is news
# Words of document per kept source, and the clamp around it. Measured over
# eleven completed runs on 2026-09-11: the documents that used their research
# sat at 47-60 words per source (23 sources -> 1077 words, 74% cited; 29 ->
# 1738, 83%), and the ones that wasted it sat at 15-21 (66 -> 1388, 36%; 89 ->
# 1304, 44%). Length was near-constant regardless of how much was gathered, so
# a deep run's extra sources had nowhere to go. The floor keeps a 6-source run
# from being padded to fill a quota; the ceiling keeps a 150-source run from
# trying to write a book.
_PREVIOUS_OVERVIEW_CHARS = 9_000  # ~3k tokens of the parent overview
# The candidate axis. Below the first number the document can cite every
# source anyway, so the extraction call would buy nothing; a candidate on a
# single source is a mention rather than something to assess; and the block
# only fires when there are at least two candidates, because one candidate is
# the subject of the question, not an axis of it.
_CANDIDATES_MIN_SOURCES = 8
_CANDIDATE_MIN_SOURCES = 2
_MAX_CANDIDATES = 12
# The reconciliation pass. Relevance at which a kept-but-uncited source is a
# loss rather than a judgment: on the 66-source reference run 7 of the 25
# uncited were relevance-4/5 listicles that SHOULD stay uncited, and forcing
# them in is padding under another name. Asymmetric on purpose — the cheap
# direction (leave a weak source out) is free, the expensive one (drop a
# strong one) gets a second call. The cap bounds that call's prefill.
_RECONCILE_MIN_RELEVANCE = 6
_RECONCILE_MAX_SOURCES = 24
_RECONCILE_MIN_WORD_RATIO = 0.95   # a revision that shrank the draft is refused
UNUSED_HEADING = "## Researched but not used"
# Horizontal whitespace only: `\s*` after the dash once swallowed a newline
# and read the following line as the reason.
_UNUSED_LINE = re.compile(
    r"^[ \t]*UNUSED:[ \t]*\[(\d+)\][ \t]*[—–-]+[ \t]*(.*?)[ \t]*$\n?", re.M)
_WORDS_PER_SOURCE = 50
_MIN_TARGET_WORDS = 900
_MAX_TARGET_WORDS = 5_000


def target_words(n_sources: int) -> int:
    """How long a document with this many sources should be."""
    return max(_MIN_TARGET_WORDS,
               min(_MAX_TARGET_WORDS, _WORDS_PER_SOURCE * max(0, n_sources)))


def looks_degenerate(text: str) -> bool:
    """True for repetition-collapse output: 8000 tokens of "!!!!!!".

    Local models at long prompt lengths can fall into a loop that runs until
    the token cap. The tell is character diversity: real prose over hundreds
    of characters uses dozens of distinct ones, a loop uses a handful.
    """
    sample = (text or "").strip()
    if len(sample) < 200:
        return False
    sample = sample[:4000]
    distinct = len(set(sample))
    top_share = max(sample.count(c) for c in set(sample)) / len(sample)
    return distinct <= 12 or top_share > 0.5


def looks_like_document(text: str) -> bool:
    """True when the output is a markdown document, not leaked reasoning.

    A thinking-mode model without a reasoning parser streams its planning
    monologue as content ("We need answer user's request…") and can burn the
    whole token budget without ever writing the document. The tell is simple:
    a real overview starts with a markdown heading almost immediately.
    """
    if looks_degenerate(text):
        return False
    for line in text.strip().splitlines()[:3]:
        if line.lstrip().startswith("#"):
            return True
    return False


def _note_block(f: Finding) -> str:
    # The kind of source is stated inline: a governing body's spec and a
    # listicle used to reach synthesis as peers, so a run could quote a drill
    # blog over the official manual sitting beside it at 8/10.
    kind = f" [{f.source_type}]" if f.source_type else ""
    if f.source_type == "standard" and f.publisher:
        kind = f" [{f.source_type}: {f.publisher}]"
    block = f"{f.citation_line()}{kind}\n    {f.url}\n{f.notes_md}\n"
    # Verbatim evidence is the point of extracting quotes — synthesis has to
    # see them or the claims it writes can't be grounded in the source wording.
    evidence = render_facts(f.key_facts, indent="  ", quotes=True, limit=6)
    if evidence:
        block += f"  Extracted facts and verbatim evidence:\n{evidence}\n"
    return block


def group_by_facet(findings: list[Finding],
                   facet_of: dict[str, str] | None
                   ) -> list[tuple[str, list[Finding]]]:
    """Findings grouped by the part of the question they were fetched for.

    Arrival order used to decide which notes shared a digest batch, so a part
    with few sources could sit in a batch of twenty about something else and
    be compressed out of existence. 2026-09-09, a U8 coaching run: six
    mixed-ability sources — one of them the official coaching manual, kept at
    8/10 — were read, digested away, and then named an open question by the
    synthesis that had just dropped them. Grouping keeps a part whole, and
    strongest-first means a squeeze drops the weakest source, not a random
    one. Rank leads that ordering and relevance breaks its ties: an official
    standard at 7/10 outranks a roundup at 9/10, because relevance measures
    how much a page says, not whether to believe it."""
    groups: dict[str, list[Finding]] = {}
    for f in findings:
        groups.setdefault((facet_of or {}).get(f.query, ""), []).append(f)
    return [(facet, sorted(fs, key=lambda f: (source_rank(f.source_type),
                                              -f.relevance)))
            for facet, fs in groups.items()]


def _pack(groups: list[tuple[str, list[str]]]
          ) -> list[tuple[list[str], list[str]]]:
    """Batches of (part names, note blocks) under the per-batch budget.

    A part is never scattered across batches — only split when it exceeds the
    budget by itself — so the digest prompt can name what a batch must cover.
    """
    batches: list[tuple[list[str], list[str]]] = []
    parts: list[str] = []
    blocks: list[str] = []
    size = 0
    for facet, group in groups:
        need = sum(est_tokens(b) for b in group)
        if need > _BATCH_BUDGET:
            if blocks:
                batches.append((parts, blocks))
                parts, blocks, size = [], [], 0
            chunk: list[str] = []
            used = 0
            for b in group:
                t = est_tokens(b)
                if used + t > _BATCH_BUDGET and chunk:
                    batches.append(([facet], chunk))
                    chunk, used = [], 0
                chunk.append(b)
                used += t
            if chunk:
                batches.append(([facet], chunk))
            continue
        if size + need > _BATCH_BUDGET and blocks:
            batches.append((parts, blocks))
            parts, blocks, size = [], [], 0
        parts.append(facet)
        blocks.extend(group)
        size += need
    if blocks:
        batches.append((parts, blocks))
    return batches


async def name_candidates(llm: LLM, *, query: str, findings: list[Finding]
                          ) -> list[tuple[str, list[int]]]:
    """The named things the question is choosing among, with their sources.

    Strongest-first, ids filtered to sources the run actually kept, and a
    candidate needs two sources to be listed. Any failure of the call means
    no candidate block — this is an enrichment of synthesis and must never
    cost the document itself.
    """
    if len(findings) < _CANDIDATES_MIN_SOURCES:
        return []
    known = {f.idx for f in findings}
    lines = [f"[{f.idx}] {f.title} — {f.domain}: "
             f"{' '.join((f.summary or '').split())[:200]}" for f in findings]
    prompt = prompts.CANDIDATES.format(query=query, sources="\n".join(lines))
    try:
        out = await llm.chat_json(
            "candidates", [{"role": "user", "content": prompt}],
            CandidatesOut, max_tokens=1500, temperature=0.2)
    except (LLMJsonError, LLMError) as e:
        log.warning("candidate extraction skipped: %s", e)
        return []
    picked: list[tuple[str, list[int]]] = []
    seen: set[str] = set()
    for c in out.candidates:
        ids = [i for i in c.sources if i in known]
        key = c.name.lower()
        if len(ids) < _CANDIDATE_MIN_SOURCES or key in seen:
            continue
        seen.add(key)
        picked.append((c.name, ids))
    picked.sort(key=lambda x: -len(x[1]))
    return picked[:_MAX_CANDIDATES]


def split_unused(text: str, allowed: set[int]) -> tuple[str, list[tuple[int, str]]]:
    """The document without its UNUSED lines, and those lines as (id, reason).

    Only ids the pass was actually asked about are kept — a model that
    accounts for a source it was never given is not saying anything true."""
    unused: list[tuple[int, str]] = []
    seen: set[int] = set()
    for m in _UNUSED_LINE.finditer(text):
        i = int(m.group(1))
        if i in allowed and i not in seen:
            seen.add(i)
            unused.append((i, m.group(2).strip()))
    return _UNUSED_LINE.sub("", text).rstrip() + "\n", unused


async def reconcile(llm: LLM, *, query: str, draft: str,
                    findings: list[Finding], max_out: int,
                    bus=None, run_id: str = "") -> str:
    """A second pass that places the strong sources the draft left uncited.

    Code decides what is in scope and code decides whether to accept the
    result: the revision replaces the draft only if it kept every citation
    the draft had and did not shrink. Anything else — a collapsed output, a
    rewrite that lost content, a call failure — leaves the draft as it was,
    so this pass can only add. Sources the model accounts for on UNUSED
    lines are appended under a heading with their stated reason, which is
    the part the reader could not see before: a source going uncited used
    to be indistinguishable from a source being forgotten.
    """
    had = cited_ids(draft)
    missing = sorted((f for f in findings
                      if f.idx not in had
                      and f.relevance >= _RECONCILE_MIN_RELEVANCE),
                     key=lambda f: (source_rank(f.source_type), -f.relevance))
    missing = missing[:_RECONCILE_MAX_SOURCES]
    if not missing:
        return draft
    if bus is not None and run_id:
        bus.publish(run_id, "log", message=(
            f"reconciling {len(missing)} kept source(s) at relevance "
            f"{_RECONCILE_MIN_RELEVANCE}+ the draft did not cite"))
    prompt = prompts.SYNTH_RECONCILE.format(
        query=query, n=len(missing), draft=draft.strip(),
        notes_block="\n".join(_note_block(f) for f in missing))
    try:
        text = await llm.chat("synth", [{"role": "user", "content": prompt}],
                              max_tokens=max_out, temperature=0.3)
    except LLMError as e:
        log.warning("reconciliation call failed, keeping draft: %s", e)
        return draft
    body, unused = split_unused(text, {f.idx for f in missing})
    # Measured 2026-09-11: the model placed all twelve sources it was given
    # AND listed eight of them as UNUSED with boilerplate reasons. A claim
    # about the document is checked against the document, not believed.
    now = cited_ids(body)
    unused = [(i, why) for i, why in unused if i not in now]
    reason = ""
    if not looks_like_document(body):
        reason = "output was not a document"
    elif not had <= now:
        reason = f"lost citations {sorted(had - now)}"
    elif len(body.split()) < _RECONCILE_MIN_WORD_RATIO * len(draft.split()):
        reason = f"shrank to {len(body.split())} from {len(draft.split())} words"
    if reason:
        log.warning("reconciliation rejected: %s", reason)
        if bus is not None and run_id:
            bus.publish(run_id, "log",
                        message=f"reconciliation rejected ({reason}); keeping the draft")
        return draft
    gained = sorted(now - had)
    if bus is not None and run_id:
        bus.publish(run_id, "log", message=(
            f"reconciliation cited {len(gained)} more source(s)"
            + (f", accounted for {len(unused)} as redundant" if unused else "")))
    if unused:
        by_idx = {f.idx: f for f in findings}
        unused = [(i, _checked_reason(why, i, now)) for i, why in unused]
        body = (body.rstrip() + f"\n\n{UNUSED_HEADING}\n\n"
                + "Read and kept, but not drawn on by the synthesis. Where "
                + "it named the cited source that covers the same ground, "
                + "that is shown:\n\n"
                + "\n".join(
                    f"- [{i}] {by_idx[i].title} — {why}" if why else
                    f"- [{i}] {by_idx[i].title}"
                    for i, why in unused) + "\n")
    return body


def cited_ids(overview: str) -> set[int]:
    """The [n] citation ids a finished document actually uses.

    Stops at the "Researched but not used" heading: the list under it names
    sources by the same [n] markers so the reader can jump to them, and
    counting those as citations once reported 58 of 66 cited for a document
    whose body cited 51. Everything below that heading is about what the
    document did NOT use."""
    cut = overview.find(UNUSED_HEADING)
    body = overview if cut < 0 else overview[:cut]
    return {int(n) for n in re.findall(r"\[(\d+)\]", body)}


def _checked_reason(why: str, idx: int, cited: set[int]) -> str:
    """A reason only if it names a cited source other than the one it is
    excusing. Measured 2026-09-11: "[25] duplicates the claim already covered
    by [25]" — the model excused a source by pointing at itself, and in
    another run pointed at nothing. A reason the reader cannot check is
    worse than none."""
    refs = {int(n) for n in re.findall(r"\[(\d+)\]", why)}
    if refs and idx not in refs and refs <= cited:
        return why
    return ""


PREMISE_HEADING_MARK = "question assumes"


def premise_verdict(overview: str) -> str:
    """The premise section's body, but only when it settled something.

    A part of the question can be answered inside the premise verdict
    rather than under a heading of its own. 2026-09-10: a run stated the
    official U8 field dimensions, with citations, in its first paragraph
    and then printed "field dimension standards" under `## Not researched`
    at the foot of the same document. The coverage recheck reads headings
    only, and this heading names no facet, so it could not see it.

    Returned only when the section cites a source. The prompt tells the
    model to say plainly when the run found nothing that settles the
    premise, and that admission is not evidence that anything was
    researched.
    """
    lines = overview.splitlines()
    start = next((i for i, l in enumerate(lines)
                  if l.lstrip().startswith("#")
                  and PREMISE_HEADING_MARK in l.lower()), None)
    if start is None:
        return ""
    body: list[str] = []
    for l in lines[start + 1:]:
        if l.lstrip().startswith("## "):
            break
        body.append(l)
    text = "\n".join(body)
    return text if cited_ids(text) else ""


def funnel_losses(overview: str, findings: list[Finding],
                  facet_of: dict[str, str] | None
                  ) -> tuple[list[str], list[Finding]]:
    """What the run read and the overview then failed to use.

    Two different failures. A part whose every source went uncited is the
    map-reduce dropping a whole topic. A high-relevance source going uncited
    is a weaker signal but still worth naming. The coverage check upstream
    runs on what was *searched*, so neither of these could be seen before."""
    cited = cited_ids(overview)
    dropped = [facet for facet, fs in group_by_facet(findings, facet_of)
               if facet and not any(f.idx in cited for f in fs)]
    strong = [f for f in findings
              if f.idx not in cited and f.relevance >= _STRONG_UNCITED]
    return dropped, strong


async def synthesize(llm: LLM, *, query: str, title: str, brief: str,
                     recency_desc: str, today: str, state_md: str,
                     findings: list[Finding], bus=None, run_id: str = "",
                     previous_overview: str = "",
                     uncovered_facets: list[str] | None = None,
                     facet_of: dict[str, str] | None = None,
                     premises: list[str] | None = None,
                     deliverables: list[str] | None = None,
                     placeholder_on_failure: bool = True) -> str:
    groups = [(facet, [_note_block(f) for f in fs])
              for facet, fs in group_by_facet(findings, facet_of)]
    blocks = [b for _, group in groups for b in group]

    if est_tokens("".join(blocks)) > _SINGLE_CALL_BUDGET:
        blocks = await _map_digest(llm, query, groups)

    prompt = prompts.SYNTH.format(
        query=query, title=title, brief=brief, recency_desc=recency_desc,
        today=today, state_md=state_md or "(none)",
        notes_block="\n".join(blocks),
    )
    # Length is set by how much was gathered, before any of the conditional
    # blocks: the deliverables block comes last and is allowed to overrule it
    # when the asker said something about length themselves.
    want = target_words(len(findings))
    prompt += prompts.SYNTH_LENGTH_BLOCK.format(n=len(findings), words=f"{want:,}")
    # A section per part of the question. Models follow structure far more
    # reliably than word counts — the length target alone moved a 66-source
    # run from 1,480 to 2,197 words against a 3,300 ask, and the parts with
    # two sources were the ones that vanished into a passing sentence.
    #
    # `uncovered_facets` is subtracted, not merely skipped. It is computed
    # from facet_kept, which is credit-gated, while these groups come from
    # query_facet, which is attribution — so a part can legitimately appear in
    # BOTH, and naming it here while SYNTH_COVERAGE_BLOCK forbids a section on
    # it would put two contradictory instructions in one prompt.
    gagged = set(uncovered_facets or ())
    named = [(f, len(fs)) for f, fs in groups if f and f not in gagged]
    if len(named) > 1:
        prompt += prompts.SYNTH_STRUCTURE_BLOCK.format(
            parts="\n".join(f"- {f} — {n} source{'s' if n != 1 else ''}"
                             for f, n in sorted(named, key=lambda x: -x[1])))
    # The second axis. The structure block above gives each PART of the
    # question a section; when the question is choosing among named things,
    # the parts are criteria and the candidates are a dimension the sections
    # do not span. 2026-09-11, measured on a 66-source evaluation: the 25
    # uncited sources clustered by product — every Joplin source, both
    # Logseq, three Obsidian — while the sections were the five criteria.
    # A candidate the document never names has no sentence its sources
    # could be cited in, however long the document is made.
    candidates = await name_candidates(llm, query=query, findings=findings)
    if len(candidates) >= 2:
        prompt += prompts.SYNTH_CANDIDATES_BLOCK.format(
            candidates="\n".join(
                f"- {name} — {len(ids)} source{'s' if len(ids) != 1 else ''}: "
                + " ".join(f"[{i}]" for i in ids)
                for name, ids in candidates))
        if bus is not None and run_id:
            bus.publish(run_id, "log", message=(
                f"{len(candidates)} candidates named across the sources: "
                + ", ".join(f"{n} ({len(ids)})" for n, ids in candidates)))
    if premises:
        # The question asserted something checkable. Saying whether it holds
        # comes before answering, because a wrong premise changes the answer.
        prompt += prompts.SYNTH_PREMISE_BLOCK.format(
            premises="\n".join(f"- {p}" for p in premises))
    if uncovered_facets:
        # Synthesis is given the original question, so left alone it writes a
        # confident section for every ask, researched or not.
        prompt += prompts.SYNTH_COVERAGE_BLOCK.format(
            uncovered="\n".join(f"- {f}" for f in uncovered_facets))
    if previous_overview:
        # Evergreen refreshes and follow-ups lead with what changed — nobody
        # wants to re-read a 90%-identical overview to find the new part.
        clipped = previous_overview[:_PREVIOUS_OVERVIEW_CHARS]
        prompt += prompts.SYNTH_DELTA_BLOCK.format(previous_overview=clipped)
    if deliverables:
        # What the asker said about the shape of the answer, and last of the
        # blocks deliberately: the form of the document is the one part of
        # this the asker specified in their own words, so it sits nearest the
        # model's turn and gets the final say over the generic instructions
        # above. 2026-09-11: a question that asked for a comprehensive
        # comparison table got a good document with no table in it.
        prompt += prompts.SYNTH_DELIVERABLES_BLOCK.format(
            deliverables="\n".join(f"- {d}" for d in deliverables))
    messages = [{"role": "user", "content": prompt}]

    # Synthesis is the longest single call in a run and the one the user is
    # actually waiting on, so stream it into the progress pane rather than
    # sitting behind a spinner. A stream failure falls back to a normal call —
    # the document matters more than the animation.
    # Derived from the target rather than set beside it, so the ceiling can
    # never be tighter than the length the prompt just asked for. ~2.5 tokens
    # per word leaves room for headings, citation markers and markdown.
    max_out = min(16_000, max(8_000, int(want * 2.5)))

    text = None
    if bus is not None and run_id:
        try:
            text = await llm.chat_stream("synth", messages, bus, run_id,
                                         max_tokens=max_out, temperature=0.4)
        except Exception:
            log.warning("streaming synthesis failed, retrying unstreamed",
                        exc_info=True)
    if text is None:
        text = await llm.chat("synth", messages, max_tokens=max_out,
                              temperature=0.4)

    if not looks_like_document(text):
        # Leaked reasoning monologue instead of a document. One stern retry;
        # publishing the monologue as an overview wastes the whole run.
        log.warning("synthesis output is not a document, retrying once")
        if bus is not None and run_id:
            bus.publish(run_id, "log",
                        message=("synthesis produced reasoning text instead "
                                 "of the document — retrying once"))
        stern = (prompt + "\n\nIMPORTANT: Output ONLY the final markdown "
                 "document itself, beginning immediately with the '# ' title "
                 "line. No planning, no reasoning, no commentary.")
        retry = await llm.chat("synth", [{"role": "user", "content": stern}],
                               max_tokens=max_out, temperature=0.4)
        if looks_like_document(retry):
            return await reconcile(llm, query=query, draft=retry,
                                   findings=findings, max_out=max_out,
                                   bus=bus, run_id=run_id)
        # Both attempts failed. Publishing the output anyway is how a run
        # ends up showing 8000 exclamation marks where its overview should
        # be — the research itself is intact, so say so and point at the
        # one-click rebuild instead. Re-synthesis passes
        # placeholder_on_failure=False: overwriting a run's existing overview
        # with a placeholder would destroy something usable.
        if not placeholder_on_failure:
            return retry
        log.error("synthesis unusable twice; writing a placeholder overview")
        if bus is not None and run_id:
            bus.publish(run_id, "log",
                        message=("synthesis failed twice — sources are saved; "
                                 "use Re-synthesize to rebuild the overview"))
        return (f"# {title}\n\n"
                f"> **Synthesis failed.** The research completed and all "
                f"{len(findings)} sources below are saved with their notes and "
                f"evidence, but the model did not return a usable document "
                f"after two attempts — most often a repetition loop on an "
                f"over-long prompt.\n>\n"
                f"> Press **Re-synthesize** on this run to rebuild the "
                f"overview from the stored sources without re-searching. "
                f"If it fails again, a smaller model prompt helps: lower the "
                f"depth, or set a repetition penalty on your inference "
                f"server.\n")
    # The draft is done; now the strong sources it left out get one more
    # chance to be placed, with the outcome measured and bounded in code.
    return await reconcile(llm, query=query, draft=text, findings=findings,
                           max_out=max_out, bus=bus, run_id=run_id)


async def _map_digest(llm: LLM, query: str,
                      groups: list[tuple[str, list[str]]]) -> list[str]:
    """Compress note blocks into per-batch digests, preserving [n] citations.

    Each batch names the parts of the question it carries, and gets output
    room in proportion to how many it carries, so a thinly-sourced part is not
    squeezed out by a populous one sharing its batch."""
    digests = []
    for parts, batch in _pack(groups):
        named = [p for p in parts if p]
        part_block = ""
        if named:
            part_block = (
                "\nThese notes were gathered for the following parts of the "
                "question. Cover EVERY one of them — a part carried by a "
                "single source still gets its specific detail and its [n]:\n"
                + "\n".join(f"- {p}" for p in named) + "\n")
        prompt = prompts.SYNTH_PARTIAL.format(query=query,
                                              part_block=part_block,
                                              notes_block="\n".join(batch))
        digest = await llm.chat(
            "synth", [{"role": "user", "content": prompt}],
            max_tokens=min(_DIGEST_MAX_TOKENS,
                           _DIGEST_BASE_TOKENS
                           + _DIGEST_PER_PART_TOKENS * max(0, len(named) - 1)),
            temperature=0.3,
        )
        if looks_degenerate(digest):
            # A collapsed digest is worse than no digest: it silently poisons
            # the synthesis that consumes it, and the failure then looks like
            # synthesis being at fault. Fall back to this batch's raw notes.
            log.warning("digest collapsed; using its raw notes instead")
            digests.extend(batch)
        else:
            digests.append(digest)
    return digests


async def follow_ups(llm: LLM, *, query: str, overview: str) -> FollowUpsOut:
    prompt = prompts.FOLLOWUPS.format(query=query, overview=overview[:24_000])
    try:
        return await llm.chat_json(
            "followups", [{"role": "user", "content": prompt}],
            FollowUpsOut, max_tokens=1500, temperature=0.5,
        )
    except LLMJsonError as e:
        log.warning("follow-ups skipped: %s", e)
        return FollowUpsOut(items=[])


def render_sources_md(findings: list[Finding]) -> str:
    lines = ["# Sources", ""]
    for f in findings:
        date = f.published or "undated"
        lines.append(
            f'{f.idx}. <a id="src-{f.idx}"></a>**{f.title}** — {f.domain}, '
            f"{date}, relevance {f.relevance}/10  \n   <{f.url}>"
        )
    if not findings:
        lines.append("_No sources were kept._")
    return "\n".join(lines) + "\n"


def render_further_md(items) -> str:
    lines = ["# Further research", ""]
    if not items:
        lines.append("_No follow-up suggestions._")
    for i, item in enumerate(items, 1):
        label = RECENCY_LABELS.get(item.recency, item.recency)
        lines.append(f"{i}. **{item.query}**  \n"
                     f"   {item.rationale}  \n"
                     f"   _suggested: depth {item.depth}, {label.lower()}_")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
