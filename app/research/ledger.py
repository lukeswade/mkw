"""One reconciled set of facts per named thing, before the report is written.

Synthesis reads every source's notes in one prompt, source by source. What
66 sources say about one company is spread over 66 blocks, and the document
that comes out can be right about it in one section and wrong in the next.
The 2026-11-15 finance report named two practices as acquisition targets
that source [32] says the platform it recommended had already bought, gave
that platform's backers three different ways, and put a figure measured on
one model and GPU under another model's name. Every fact needed was in the
notes.

Three steps, each bounded in code:
1. Names. Capitalised and numbered names are mined from the notes with the
   number of sources that use each; one call picks the ones the question
   turns on. It can only pick from the mined list, and a name needs two
   sources.
2. Evidence. Every key fact and notes line that mentions a name, with its
   source's number, title and date.
3. Reconcile. One call per name turns that evidence into a few lines, each
   with the sources it rests on. A line is kept only if it names the thing,
   its sources were among those shown, and every figure and numbered name in
   it is stated by them (the citation check's string pass).

The lines go into the synthesis prompt as facts the document must hold to in
every section and table. Any failure means no block: this is an enrichment
of synthesis and must never cost the document.
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from dataclasses import dataclass, field

from app.llm import prompts
from app.models import LedgerNamesOut, LedgerProfileOut
from app.research import citecheck
from app.research.dedupe import _STOPWORDS
from app.research.notes import Finding
from app.research.storage import CITATION_GROUP_RE

log = logging.getLogger(__name__)

_MIN_SOURCES = 8          # below this the notes fit in one head, the model's included
_NAME_MIN_SOURCES = 2
_MINED_SHOWN = 150        # mined names offered to the filtering call
_MAX_NAMES = 18
_PER_SOURCE = 3           # evidence lines one source may contribute to a name
_MAX_EVIDENCE = 30
_EVIDENCE_CHARS = 320
_MAX_LINES = 5
_LINE_CHARS = 360

# A name as the notes write it: capitalised or numbered words, with plain
# numbers allowed after the first ("RTX 3080 Ti", "Gemma 4 26B-A4B").
_CAP = r"(?:[A-Z][\w&'’.]*(?:-[\w.]+)*|[a-z]+\d[\w.]*(?:-[\w.]+)*|\d+[A-Za-z][\w.]*(?:-[\w.]+)*)"
_PHRASE = re.compile(rf"{_CAP}(?:[ ](?:{_CAP}|\d[\w.]*)){{0,4}}")
# Words that open a sentence capitalised and are not part of a name.
_OPENERS = _STOPWORDS | frozenset("""however while although though this these those it its
on at by as to from with when where if both each most many some several all any no not only also
additionally furthermore moreover thus therefore meanwhile overall according despite because since
after before during under over between among there here key note""".split())
_ARTICLES = frozenset({"the", "a", "an"})
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(\"“])")


@dataclass
class Entry:
    name: str
    aliases: list[str] = field(default_factory=list)
    sources: list[int] = field(default_factory=list)       # sources that mention it
    lines: list[tuple[str, list[int]]] = field(default_factory=list)


def _fold(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def mention_re(name: str) -> re.Pattern:
    """Matches the name however it is punctuated or spaced: "U.S. Urology
    Partners", "US Urology Partners" and "US-Urology Partners" are one name,
    whichever of them the entry is called. Whole words only, so "One GI" is
    not found in "none given"."""
    chars = re.findall(r"[A-Za-z0-9]", name)
    return re.compile(r"(?<![A-Za-z0-9])" + r"[\W_]*".join(map(re.escape, chars))
                      + r"(?![A-Za-z0-9])", re.I)


def _texts(f: Finding) -> list[str]:
    """What a source says, as separate statements: its facts, then its notes."""
    out = []
    for fact in sorted(f.key_facts, key=lambda k: -int(k.get("confidence") or 0)):
        claim = " ".join(str(fact.get("claim") or "").split())
        quote = " ".join(str(fact.get("evidence_quote") or "").split())
        if claim:
            out.append(claim + (f' ("{quote}")' if quote and quote.lower() not in claim.lower() else ""))
    for line in (f.notes_md or "").splitlines():
        line = line.strip().lstrip("-*•0123456789. ").strip()
        out += [s.strip() for s in _SENTENCE.split(line) if len(s.strip()) > 20]
    return out


def _clean(words: list[str]) -> str:
    while words and words[0].lower().strip(".,'’") in _ARTICLES:
        words = words[1:]
    return re.sub(r"['’]s$", "", " ".join(words).strip(".,;:'’-"))


def mine_names(findings: list[Finding]) -> list[tuple[str, int]]:
    """Names the notes use, with how many sources use each, most-used first.

    Deliberately generous: it also returns "Medicare" and "Florida". The
    filtering call drops those; this list is the limit of what it may keep.

    A capitalised word at the start of a statement may only be opening the
    sentence ("However Gastro Health grew"), so it is cut from a name unless
    the notes also write the name that way mid-sentence: "One GI" opens
    sentences too, and is a company."""
    found: list[tuple[int, list[str], bool]] = []
    for f in findings:
        for text in [f.title or ""] + _texts(f):
            for m in _PHRASE.finditer(CITATION_GROUP_RE.sub(" ", text)):
                words = m.group(0).split()
                # "Cardinal Health's MSO" is two things; the name ends at the 's
                cut = next((i for i, w in enumerate(words) if re.search(r"['’]s$", w)), None)
                found.append((f.idx, words if cut is None else words[:cut + 1], m.start() == 0))
    mid = {_fold(_clean(words)) for _, words, first in found if not first}
    seen: dict[str, set[int]] = {}
    forms: dict[str, Counter] = {}
    for idx, words, first in found:
        if first and _fold(_clean(words)) not in mid:
            while words and words[0].lower().strip(".,'’") in _OPENERS | _ARTICLES:
                words = words[1:]
        phrase = _clean(words)
        key = _fold(phrase)
        if len(key) < 3 or key.isdigit() or phrase.lower() in _OPENERS:
            continue
        seen.setdefault(key, set()).add(idx)
        forms.setdefault(key, Counter())[phrase] += 1
    ranked = sorted(seen, key=lambda k: (-len(seen[k]), k))
    return [(forms[k].most_common(1)[0][0], len(seen[k])) for k in ranked
            if len(seen[k]) >= _NAME_MIN_SOURCES]


def _mentions(names: list[str], findings: list[Finding]) -> list[int]:
    pats = [mention_re(n) for n in names]
    return [f.idx for f in findings
            if any(p.search(t) for p in pats for t in [f.title or ""] + _texts(f))]


async def pick_names(llm, *, query: str, findings: list[Finding],
                     also: list[str] | None = None) -> list[Entry]:
    """The named things the document must keep straight.

    The model only filters: it returns the mined names that are specific
    things, and code ranks them by how many sources mention each. Asked
    instead to choose its top 14, it chose a different 14 on each call and
    left out the company the question was about. `also` are names already
    chosen elsewhere (the comparison candidates), kept without asking. Every
    entry must be a mined name mentioned by two sources."""
    mined = mine_names(findings)
    if not mined:
        return []
    known = {_fold(n): n for n, _ in mined}
    listing = "\n".join(f"- {n} [{c}]" for n, c in mined[:_MINED_SHOWN])
    prompt = prompts.LEDGER_NAMES.format(query=query, names=listing)
    picked: list[tuple[str, list[str]]] = [(n, []) for n in (also or [])]
    try:
        out = await llm.chat_json("ledger_names", [{"role": "user", "content": prompt}],
                                  LedgerNamesOut, max_tokens=2500, temperature=0.0)
        for item in out.names:
            if _fold(item.name) in known:
                picked.append((known[_fold(item.name)],
                               [known[_fold(a)] for a in item.aliases if _fold(a) in known]))
    except Exception as e:
        log.warning("ledger: naming call failed (%s); using the candidates only", e)
    entries: list[Entry] = []
    taken: set[str] = set()
    for name, aliases in picked:
        keys = {_fold(n) for n in [name] + aliases}
        if keys & taken:
            continue
        ids = _mentions([name] + aliases, findings)
        if len(ids) < _NAME_MIN_SOURCES:
            continue
        taken |= keys
        entries.append(Entry(name, [a for a in aliases if _fold(a) != _fold(name)], ids))
    entries.sort(key=lambda e: -len(e.sources))
    return entries[:_MAX_NAMES]


def evidence(entry: Entry, findings: list[Finding]) -> list[tuple[int, str]]:
    """(source id, statement) for every statement that mentions the entry,
    strongest sources first, a few per source so one page cannot fill it."""
    pats = [mention_re(n) for n in [entry.name] + entry.aliases]
    out: list[tuple[int, str]] = []
    for f in sorted(findings, key=lambda f: (-f.relevance, f.idx)):
        own = 0
        in_title = any(p.search(f.title or "") for p in pats)
        seen: set[str] = set()
        for text in _texts(f):
            # a page titled for the thing is about it in every line: "Oct
            # 2025: Buyout of New Jersey Urology" names its buyer only there
            if not (in_title or any(p.search(text) for p in pats)):
                continue
            key = _fold(text)[:80]
            if key in seen:
                continue
            seen.add(key)
            out.append((f.idx, text[:_EVIDENCE_CHARS]))
            own += 1
            if own >= (_PER_SOURCE * 2 if in_title else _PER_SOURCE):
                break
        if len(out) >= _MAX_EVIDENCE:
            break
    return out[:_MAX_EVIDENCE]


def _checked_lines(entry: Entry, out: LedgerProfileOut, shown: set[int],
                   index: dict[int, citecheck._Index],
                   question: citecheck._Index) -> list[tuple[str, list[int]]]:
    pats = [mention_re(n) for n in [entry.name] + entry.aliases]
    kept: list[tuple[str, list[int]]] = []
    seen: set[str] = set()
    for line in out.lines:
        text = " ".join(CITATION_GROUP_RE.sub(" ", line.text).split()).strip(" -")
        ids = [n for n in dict.fromkeys(line.sources) if n in shown]
        if not text or len(text) > _LINE_CHARS or not ids:
            continue
        if not any(p.search(text) for p in pats):
            continue
        # every figure and numbered name must be stated by the sources the
        # line names; a line that fails is dropped, not repaired
        missing = [t for t in citecheck._absent(citecheck.claim_tokens(text),
                                                [index[n] for n in ids])
                   if not question.has(t)]
        if missing or _fold(text) in seen:
            continue
        seen.add(_fold(text))
        kept.append((text, ids))
        if len(kept) >= _MAX_LINES:
            break
    return kept


async def build(llm, *, query: str, today: str, findings: list[Finding],
                also: list[str] | None = None, bus=None, run_id: str = "") -> list[Entry]:
    """Entries with their reconciled lines; [] when there is nothing to say
    or anything failed."""
    if len(findings) < _MIN_SOURCES:
        return []
    try:
        return await _build(llm, query, today, findings, also, bus, run_id)
    except Exception:
        log.exception("ledger failed; synthesis goes ahead without it")
        return []


async def _build(llm, query, today, findings, also, bus, run_id) -> list[Entry]:
    entries = await pick_names(llm, query=query, findings=findings, also=also)
    if len(entries) < 2:
        return []
    by_idx = {f.idx: f for f in findings}
    index = {i: citecheck._index(citecheck._evidence_text(f)) for i, f in by_idx.items()}
    question = citecheck._index(query or "")

    async def one(entry: Entry) -> None:
        shown = evidence(entry, findings)
        if not shown:
            return
        listing = "\n".join(
            f"[{i}] {by_idx[i].title} ({by_idx[i].published or 'undated'}) — {text}"
            for i, text in shown)
        alias = (" (also written " + ", ".join(entry.aliases) + ")") if entry.aliases else ""
        prompt = prompts.LEDGER_PROFILE.format(
            query=(query or "")[:1500], today=today, name=entry.name, alias=alias,
            evidence=listing, limit=_MAX_LINES)
        try:
            out = await llm.chat_json("ledger", [{"role": "user", "content": prompt}],
                                      LedgerProfileOut, max_tokens=1200, temperature=0.1)
        except Exception as e:
            log.warning("ledger: no entry for %s (%s)", entry.name, e)
            return
        entry.lines = _checked_lines(entry, out, {i for i, _ in shown}, index, question)

    await asyncio.gather(*(one(e) for e in entries))
    done = [e for e in entries if e.lines]
    if bus is not None and run_id and done:
        bus.publish(run_id, "log", message=(
            f"facts reconciled across sources for {len(done)} named thing"
            f"{'' if len(done) == 1 else 's'}: " + ", ".join(e.name for e in done)))
    return done


def render(entries: list[Entry]) -> str:
    """The entries as the prompt block's body."""
    return "\n\n".join(
        e.name + "\n" + "\n".join(
            f"- {text} " + "".join(f"[{n}]" for n in ids) for text, ids in e.lines)
        for e in entries if e.lines)
