"""Claim verification: check a document's assertions, one at a time.

Library first, web second. Most of what gets pasted in touches something
already researched, and retrieval answers in seconds for free; only claims
the library cannot settle are worth a web sub-run. The order matters for
cost, not for correctness — a claim the library settles confidently is not
searched again.

Evidence is numbered per claim and the numbers are local to that claim's
table row, because a global bibliography across fifteen independent checks
would be unreadable and mostly unreferenced.
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from app.llm import prompts
from app.llm.client import LLM, est_tokens
from app.models import Claim, ClaimsOut, VerdictOut
from app.research.dedupe import _STOPWORDS, stem_token

log = logging.getLogger(__name__)

_DOCUMENT_BUDGET = 24_000        # est tokens of document fed to extraction
_LIBRARY_HITS = 6
# Retrieval is asked for more than it needs so the evidence can be spread
# across sources instead of taken as a top-N slice.
_LIBRARY_POOL = 24
_LIBRARY_PER_SOURCE = 2
# Source titles share a parent run, so a per-title cap alone still let one
# run fill every slot — a live check drew all six passages from one research
# run and every citation pointed back at it.
_LIBRARY_PER_RUN = 3
# When the library cannot settle a claim its passages are, by definition, not
# decisive, so they should not also crowd the low citation numbers the model
# reaches for first. Trimmed before the web evidence is appended.
_LIBRARY_KEEP_ON_FALLTHROUGH = 3
_LIBRARY_MIN_SCORE = 0.45
# A library verdict this confident is accepted without searching the web.
_LIBRARY_SETTLES_AT = 7
_WEB_PAGES_PER_CLAIM = 3
_EVIDENCE_CHARS = 1_200

_VERDICT_MARK = {
    "supported": "✓ supported",
    "contested": "± contested",
    "unsupported": "✗ unsupported",
    "unverifiable": "? unverifiable",
}


@dataclass
class Evidence:
    n: int
    label: str            # where it came from, shown to the reader
    url: str
    text: str


@dataclass
class Checked:
    claim: Claim
    verdict: VerdictOut
    evidence: list[Evidence] = field(default_factory=list)
    via: str = ""         # "library" or "web"


def clip_document(text: str) -> tuple[str, bool]:
    """Extraction sees the head of a long document, and says so."""
    if est_tokens(text) <= _DOCUMENT_BUDGET:
        return text, False
    return text[:_DOCUMENT_BUDGET * 3].rstrip(), True


async def extract_claims(llm: LLM, document: str) -> tuple[list[Claim], bool]:
    body, clipped = clip_document(document)
    out = await llm.chat_json(
        "verify", [{"role": "user",
                    "content": prompts.CLAIMS.format(document=body)}],
        ClaimsOut, max_tokens=3000, temperature=0.1)
    claims = [c for c in out.claims if c.text.strip()]
    return claims, clipped


def select_claims(claims: list[Claim], cap: int) -> tuple[list[Claim], list[Claim]]:
    """(checked, skipped) — most load-bearing first, order otherwise preserved.

    Opinions and predictions are separated out rather than dropped: they are
    not false, they are not checkable, and a reader should see that they were
    recognised rather than silently ignored.
    """
    checkable = [c for c in claims if c.checkable]
    ranked = sorted(enumerate(checkable), key=lambda p: (-p[1].importance, p[0]))
    keep_idx = {i for i, _c in ranked[:cap]}
    checked = [c for i, c in enumerate(checkable) if i in keep_idx]
    skipped = [c for i, c in enumerate(checkable) if i not in keep_idx]
    return checked, skipped


_TERM_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_YEAR_RE = re.compile(r"(19|20)\d\d")
_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _terms(text: str) -> set[str]:
    """Content words (stemmed) and distinctive numbers; years and short
    words say nothing about which part of a page a claim is about."""
    out = set()
    for t in _TERM_RE.findall(re.sub(r"(?<=\d),(?=\d)", "", text.lower())):
        if t.replace(".", "").isdigit():
            if len(t) >= 2 and not _YEAR_RE.fullmatch(t):
                out.add(t)
        elif len(t) >= 3 and t not in _STOPWORDS:
            out.add(stem_token(t))
    return out


def best_passage(claim: str, text: str, limit: int = _EVIDENCE_CHARS) -> str:
    """The `limit`-char window of `text` that has the most to do with `claim`.

    A web page was shown to the judge by its first 1,200 chars: masthead,
    byline, intro. Measured 2026-09-23 (scripts/eval/page_probe.py quotes):
    of 341 evidence quotes the note-taker had verified in their pages, 45%
    sat inside the first 1,200 chars; the median sat at char 1,383, a
    quarter past char 4,100. Windows start at sentence or line boundaries
    and are scored on the claim's words and numbers present, rarer ones
    weighted up, so no model is needed and nothing can fail. No overlap at
    all keeps the old behaviour: the head.
    """
    if len(text) <= limit:
        return text
    wanted = _terms(claim)
    if not wanted:
        return text[:limit]
    starts = {0} | {m.end() for m in _BOUNDARY_RE.finditer(text)}
    starts |= set(range(0, len(text), max(1, limit // 3)))
    starts = sorted(s for s in starts if s < len(text) - limit // 4)
    found = [(s, _terms(text[s:s + limit]) & wanted) for s in starts]
    df = Counter(t for _s, hit in found for t in hit)
    weight = {t: math.log(1 + len(found) / n) for t, n in df.items()}
    scored = [(sum(weight[t] for t in hit), s) for s, hit in found]
    best = max(score for score, _s in scored)
    if best == 0.0:
        return text[:limit]
    # Every window holding the best set of terms ties. The first puts the
    # matching text at the window's end, the last at its start; the middle
    # one centres it, so the sentence around it survives on both sides.
    tied = [s for score, s in scored if score == best]
    best_s = tied[len(tied) // 2]
    passage = text[best_s:best_s + limit].strip()
    return ("… " if best_s else "") + passage + (" …" if best_s + limit < len(text) else "")


def render_evidence(items: list[Evidence], claim: str = "") -> str:
    """Numbered evidence for the judge; with a claim, each long item is shown
    by its passage about the claim rather than by its first chars."""
    return "\n\n".join(
        f"[{e.n}] {e.label}\n"
        + (best_passage(claim, e.text) if claim else e.text[:_EVIDENCE_CHARS])
        for e in items
    ) or "(no evidence found)"


def verbatim_quote(quote: str, evidence: list[Evidence]) -> str:
    """The quote as the evidence actually has it: kept when verbatim,
    replaced with the evidence's own sentence when one clearly matches,
    dropped otherwise. The verdict table renders it inside quotation marks
    under real sources, so it must be real — the same rule notes follow."""
    from app.research.notes import _norm, _sentences, quote_is_verbatim, repair_quote
    if not quote:
        return ""
    joined = "\n".join(e.text for e in evidence)
    if quote_is_verbatim(quote, _norm(joined)):
        return quote
    return repair_quote(quote, _sentences(joined)) or ""


async def judge(llm: LLM, claim: str, evidence: list[Evidence]) -> VerdictOut:
    if not evidence:
        return VerdictOut(verdict="unverifiable", confidence=0,
                          reasoning="No evidence was found for this claim.")
    try:
        out = await llm.chat_json(
            "verify", [{"role": "user", "content": prompts.VERDICT.format(
                claim=claim, evidence=render_evidence(evidence, claim))}],
            VerdictOut, max_tokens=900, temperature=0.1)
        out.quote = verbatim_quote(out.quote, evidence)
        return out
    except Exception as e:
        log.warning("verdict failed for %r: %s", claim[:60], e)
        return VerdictOut(verdict="unverifiable", confidence=0,
                          reasoning=f"Adjudication failed: {e}")


async def library_evidence(rag, claim: str) -> list[Evidence]:
    """Passages from earlier runs that bear on the claim, spread across sources.

    Plain top-N retrieval handed all six slots to one document — the same
    source three times over — and produced a confident verdict on one side of
    a point the library actually disputes. Capping per source buys a view of
    the disagreement instead of the loudest match, the same reason a research
    round caps candidates per domain.
    """
    if rag is None:
        return []
    try:
        hits = await rag.semantic_search(claim, limit=_LIBRARY_POOL)
    except Exception as e:
        log.warning("library lookup failed: %s", e)
        return []
    out: list[Evidence] = []
    per_source: dict[str, int] = {}
    per_run: dict[str, int] = {}
    for h in hits:
        if len(out) >= _LIBRARY_HITS:
            break
        if h.get("score", 0) < _LIBRARY_MIN_SCORE:
            continue
        title = h.get("title") or h.get("run_id") or "earlier research"
        run_id = h.get("run_id") or ""
        if per_source.get(title, 0) >= _LIBRARY_PER_SOURCE:
            continue
        if run_id and per_run.get(run_id, 0) >= _LIBRARY_PER_RUN:
            continue
        per_source[title] = per_source.get(title, 0) + 1
        per_run[run_id] = per_run.get(run_id, 0) + 1
        out.append(Evidence(n=len(out) + 1,
                            label=f"your research — {title}",
                            url=f"/runs/{h.get('run_id', '')}",
                            text=h.get("text", "")))
    return out


def renumber(items: list[Evidence]) -> list[Evidence]:
    """Evidence numbers must be 1..n after any trim, or citations dangle."""
    for i, e in enumerate(items, 1):
        e.n = i
    return items


def trim_for_fallthrough(library: list[Evidence]) -> list[Evidence]:
    """Keep only the strongest library passages once the web is being consulted.

    The model reaches for the low numbers, and library evidence is numbered
    first. If the library could not settle the claim, letting it also occupy
    every low slot pushes the web evidence — the reason we are still here —
    to the back of the list.
    """
    return renumber(library[:_LIBRARY_KEEP_ON_FALLTHROUGH])


def settled(v: VerdictOut) -> bool:
    """Whether a library verdict is firm enough to skip the web.

    An "unsupported" NEVER settles from the library alone, however confident.
    The two errors are not symmetric: a wrong "supported" merely echoes
    research the reader already has, while a wrong "unsupported" tells them
    something true is false — and finding errors is the entire reason they
    pasted the document.

    This is not hypothetical. Asked whether llama.cpp has the faster prefill,
    retrieval returned six passages that were all one side of a genuinely
    disputed point (Ollama's migration note, which reports MLX prefill 57%
    faster) while the library's own comparison matrix records the opposite.
    The adjudicator called it unsupported at 10/10 and the shortcut stopped
    anything from checking.
    """
    if v.verdict == "unverifiable":
        return False
    if v.verdict == "unsupported":
        return False
    return v.confidence >= _LIBRARY_SETTLES_AT


# ---- rendering ------------------------------------------------------------------

def _cell(text: str) -> str:
    return " ".join((text or "").split()).replace("|", "\\|")


def render_report(title: str, results: list[Checked], *,
                  skipped: list[Claim], uncheckable: list[Claim],
                  clipped: bool) -> str:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.verdict.verdict] = counts.get(r.verdict.verdict, 0) + 1
    summary = ", ".join(f"{counts[k]} {_VERDICT_MARK[k].split(' ', 1)[1]}"
                        for k in ("supported", "contested", "unsupported",
                                  "unverifiable") if counts.get(k))

    # The run is already named "Claim check: …", so appending the suffix
    # produced "Claim check: X — claim check".
    heading = title if title.lower().startswith("claim check") \
        else f"{title} — claim check"
    lines = [f"# {heading}", ""]
    lines.append(f"**{len(results)} claim(s) checked**"
                 + (f" — {summary}." if summary else "."))
    lines.append("")
    lines.append("| Claim | Verdict | Why | Checked against |")
    lines.append("| --- | --- | --- | --- |")
    for r in results:
        v = r.verdict
        why = _cell(v.reasoning)
        if v.quote:
            # No HTML here. Markdown is rendered with html=False as a
            # stored-XSS guard (source text is lifted from fetched pages), so
            # a <br> renders as the literal characters, not a line break.
            why += f' — “{_cell(v.quote)[:220]}”'
        # Link each marker straight at its source. An internal /runs/… path
        # is not a valid autolink (no scheme), which is why library evidence
        # rendered as plain text while web evidence came out clickable.
        used = set(v.sources)
        cites = "".join(f"[[{e.n}]]({e.url})" for e in r.evidence
                        if e.n in used and e.url) or "—"
        lines.append(f"| {_cell(r.claim.text)} "
                     f"| {_VERDICT_MARK[v.verdict]} ({v.confidence}/10) "
                     f"| {why} | {r.via} {cites} |")

    notes = []
    if skipped:
        notes.append(f"_{len(skipped)} lower-importance claim(s) were not "
                     f"checked — the run caps how many web lookups it will "
                     f"do. They are listed below._")
    if uncheckable:
        notes.append(f"_{len(uncheckable)} statement(s) are opinions, "
                     f"predictions or recommendations rather than checkable "
                     f"facts, and were set aside._")
    if clipped:
        notes.append("_The document was longer than the extraction budget; "
                     "claims were taken from its opening section only._")
    if notes:
        lines += ["", "  \n".join(notes)]

    for heading, group in (("Not checked (capped)", skipped),
                           ("Not checkable (opinion or prediction)", uncheckable)):
        if group:
            lines += ["", f"## {heading}", ""]
            lines += [f"- {c.text}" for c in group]

    lines += ["", "## Evidence", ""]
    any_ev = False
    for r in results:
        if not r.evidence:
            continue
        any_ev = True
        lines.append(f"**{_cell(r.claim.text)[:120]}**")
        for e in r.evidence:
            label = _cell(e.label) or e.url
            lines.append(f"{e.n}. [{label}]({e.url})" if e.url
                         else f"{e.n}. {label}")
        lines.append("")
    if not any_ev:
        lines.append("_No evidence was gathered._")
    return "\n".join(lines).rstrip() + "\n"
