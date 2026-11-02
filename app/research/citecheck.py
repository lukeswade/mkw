"""Check each figure a finished report cites against the source it cites.

Synthesis writes one document from every source's notes, and figures drift
on the way: measured 2026-11-01 over five stored reports, about one cited
sentence with a figure in ten misstated the source it cites — a llama.cpp
build no source gave ("b9615+", twice in one report, its TL;DR included),
Whisper's WER as 3.2% where the source said 7.4%, a tap's depth reported as
the insert's in a repair procedure.

Two passes, and their order is the guard. A string pass, with no model call,
finds the sentences whose figures or numbered names appear nowhere in the
notes of the sources they cite. Only those go to the model, which may keep
the sentence, move the citation to the kept source that does state the
figure, or correct the sentence from the notes. Nothing it proposes is
applied unless code can confirm it: after a re-cite every figure in the
sentence must be stated by what it now cites, and a correction must pass a
stricter string check (a figure's unit must match too) and keep a citation.
A sentence the string pass clears is never sent, so the model cannot change
it.

The first dry run over eight stored reports applied 29 fixes, of which 8 did
harm, each traced to something sent that should not have been or accepted
that should not have been: a question in "Open questions" rewritten, "P99.9
remains less documented" turned into a claim about P99, the asker's own
engine dropped from a sentence, "E2E" read as a figure, a re-cite that left
"15 of 19 benchmarks" uncited, "12-15GB" accepted because a source said
"12B-15B parameters". Each is now excluded here, not argued with in the
prompt — the model ignored "never rewrite over naming alone" in the same run.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field

from app.llm import prompts
from app.models import CiteCheckOut
from app.research.notes import Finding, render_facts
from app.research.storage import CITATION_GROUP_RE, citation_ids
from app.research.synthesizer import UNUSED_HEADING

log = logging.getLogger(__name__)

# The most sentences sent to the model per document. Eight stored reports had
# 17-69 cited sentences with figures and the string pass flagged 3-19 of them,
# so this is a ceiling for outliers, not a working limit.
_MAX_CHECKS = 30
_EVIDENCE_CHARS = 6_000       # of each source's notes, shown to the model
_CONTEXT_CHARS = 1_500
_OTHERS_MAX = 3               # other kept sources offered for a re-cite
_REWRITE_RATIO = (0.4, 1.6)   # a correction is a sentence, not a new paragraph
_TAIL = (UNUSED_HEADING, "## Not researched")

# Words as the string pass sees them: "~24", "53.2%", "4GB", "Qwen3.6-35B-A3B",
# "b9615+", "3-12", "AVX2/FMA3". Trailing punctuation is stripped afterwards.
_WORD = re.compile(r"[~≈]?\w[\w.,%×+/–-]*")
_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_VERSION = re.compile(r"\d+(?:\.\d+){2,}")
_WITH_UNIT = re.compile(r"(\d+(?:\.\d+)?)([A-Za-z%×]{1,5}(?:/[A-Za-z]{1,3})?)")
_RANGE = re.compile(r"(\d+(?:\.\d+)?)[–-](\d+(?:\.\d+)?)([A-Za-z%×]{0,5})")
_ORDINAL = frozenset({"st", "nd", "rd", "th"})
_DIGITS = re.compile(r"\d+(?:\.\d+)?")
# A name this short ("E2E", "M4", "V3", "Q4", "P99") is an abbreviation or
# a bare tier more often than a claim, and one alone never flags a sentence.
_WEAK_NAME = 3
# Sentence ends: punctuation, space, then something that can start one.
_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z*_`\[(\"“])")
_ABBREV = re.compile(r"(?:\be\.g|\bi\.e|\bvs|\betc|\bapprox|\bca|\bno|\bfig)\.$", re.I)
_LIST_MARK = re.compile(r"^(\s*(?:[-*+]|\d{1,3}[.)])\s+)")
_BOLD_LABEL = re.compile(r"^(\*\*[^*\n]{1,80}\*\*:?\s*)")
_OPEN_QUESTIONS = re.compile(r"^##\s+open questions\b", re.I)
# A sentence about what is NOT known asserts no figure: "long-tail latency
# (P99.9) remains less documented" was rewritten into a claim about P99.
_ABSENCE = re.compile(
    r"\b(?:missing|unknown|unclear|uncertain|undocumented|unreported|unverified"
    r"|(?:less|poorly|under)[- ]documented|no (?:public )?(?:data|figures?|benchmarks?|numbers?)"
    r"|not (?:yet |fully |well |widely |publicly |independently )?"
    r"(?:documented|reported|published|benchmarked|detailed|measured|released"
    r"|confirmed|available|verified|tested))\b", re.I)


@dataclass(frozen=True)
class _Token:
    shown: str
    kind: str          # "num" | "ver" | "name"
    key: str           # "24", "53.2" | "0.7.3" | "qwen3635ba3b"
    unit: str = ""     # "gb", "%", "x" — a number's attached unit
    core: str = ""     # a name up to its last numbered part: Qwen3.8-Flash-Next → qwen38


@dataclass
class _Unit:
    start: int
    end: int
    text: str
    context: str
    ids: list[int]
    tokens: list[_Token]


@dataclass
class _Index:
    """What one source's evidence states, ready for presence checks."""
    numbers: set[str]
    versions: set[str]
    names: str         # the evidence folded to [a-z0-9]
    text: str          # lowercased, for unit-aware matching

    def has(self, t: _Token) -> bool:
        if t.kind == "num":
            return t.key in self.numbers
        if t.kind == "ver":
            return t.key in self.versions
        if t.key in self.names:
            return True
        # "Qwen 3.8 Flash" states Qwen3.8-Flash-Next by another name; a digit
        # must not follow, or "Qwen3" would be found inside "Qwen3.6".
        return bool(t.core) and re.search(re.escape(t.core) + r"(?!\d)", self.names) is not None

    def states(self, t: _Token) -> bool:
        """has(), and a number with a unit must appear with that unit: a
        correction once read "12-15GB" out of "12B-15B parameters"."""
        if not self.has(t) or t.kind != "num" or not t.unit:
            return self.has(t)
        n = re.escape(t.key) + r"(?:\.0+)?"
        other = r"\d+(?:\.\d+)?"
        u = re.escape(t.unit)
        pattern = (rf"(?<![\d.]){n}\s?(?:[-–]\s?{other}\s?)?{u}(?![a-z])"
                   rf"|(?<![\d.]){other}\s?[-–]\s?{n}\s?{u}(?![a-z])")
        return re.search(pattern, self.text) is not None


@dataclass
class CiteCheck:
    sentences: int = 0   # cited sentences carrying a figure or numbered name
    flagged: int = 0     # ... with one absent from the notes of what they cite
    recited: int = 0     # citation moved to the kept source stating it
    rewritten: int = 0   # sentence corrected from the notes
    kept: int = 0        # judged right as written
    rejected: int = 0    # a fix the string check could not confirm, not applied
    failed: int = 0      # the call failed; the sentence stands
    skipped: int = 0     # over the per-document ceiling
    changes: list[dict] = field(default_factory=list)

    def counts(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "changes"}

    def message(self) -> str:
        head = (f"citation check: {self.sentences} cited sentence"
                f"{'' if self.sentences == 1 else 's'} with figures")
        if not self.flagged:
            return head + (", all matched by their sources" if self.sentences else "")
        parts = [f"{n} {label}" for n, label in (
            (self.recited, "re-cited"), (self.rewritten, "corrected"),
            (self.kept, "judged right as written"),
            (self.rejected, "left as written (fix unconfirmed)"),
            (self.failed, "left as written (check failed)"),
            (self.skipped, "not checked (ceiling)")) if n]
        return (f"{head}, {self.flagged} not matched by their sources — "
                + ", ".join(parts))


def _fold(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _num_key(value: str) -> str:
    """One spelling per number: "1,200" → "1200", "24.0" → "24", "1.50" → "1.5"."""
    value = value.replace(",", "")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


def _unit(u: str) -> str:
    return u.lower().replace("×", "x")


def _core(name: str) -> str:
    parts = re.split(r"[-–_]", name)
    last = max((i for i, p in enumerate(parts) if any(c.isdigit() for c in p)), default=-1)
    core = _fold("".join(parts[:last + 1])) if last >= 0 else ""
    return core if core and core != _fold(name) else ""


def claim_tokens(text: str) -> list[_Token]:
    """The figures and numbered names a sentence asserts, citations excluded.

    Not years (2026), not a bare single digit (list counts, "3 options"), not
    ordinals, not names of three characters or fewer. A longer name with a
    digit in it is a claim: "b9615" or "Qwen3.8" absent from a source is how
    a figure attached to the wrong build or model shows."""
    out: list[_Token] = []
    seen: set[tuple[str, str]] = set()

    def add(t: _Token) -> None:
        if t.key and (t.kind, t.key) not in seen:
            seen.add((t.kind, t.key))
            out.append(t)

    def number(shown: str, value: str, unit: str = "", in_range: bool = False) -> None:
        value = value.replace(",", "")
        if "." not in value:
            if not unit and not in_range and len(value) == 1:
                return
            if not unit and 1900 <= int(value) <= 2099:
                return
        add(_Token(shown, "num", _num_key(value), _unit(unit)))

    def word(w: str) -> None:
        if not w or not any(c.isdigit() for c in w):
            return
        if _VERSION.fullmatch(w):
            add(_Token(w, "ver", w))
        elif m := _RANGE.fullmatch(w):
            number(w, m.group(1), m.group(3), in_range=True)
            number(w, m.group(2), m.group(3), in_range=True)
        elif _NUMBER.fullmatch(w):
            number(w, w)
        elif m := _WITH_UNIT.fullmatch(w):
            if m.group(2).lower() not in _ORDINAL:     # 1st, 2nd, 3rd ...
                number(w, m.group(1), m.group(2))
        elif "/" in w:                                 # AVX2/FMA3, RDNA3/4
            for part in w.split("/"):
                word(part.strip(".,;:–-"))
        elif any(c.isalpha() for c in w) and len(_fold(w)) > _WEAK_NAME:
            add(_Token(w, "name", _fold(w), core=_core(w)))

    for raw in _WORD.findall(CITATION_GROUP_RE.sub(" ", text)):
        word(raw.lstrip("~≈").rstrip(".,;:/–-"))
    return out


def _evidence_text(f: Finding) -> str:
    facts = render_facts(f.key_facts, quotes=True)
    return "\n".join(x for x in (f.title, f.summary, f.notes_md, facts) if x)


def _index(text: str) -> _Index:
    flat = text.replace(",", "")
    numbers = {_num_key(n) for n in _DIGITS.findall(flat)}
    # a rounding is the same figure: "~125GB" from "125.7GB", "18GB" from 18.4
    for n in [n for n in numbers if "." in n]:
        x = float(n)
        numbers |= {str(int(x)), str(round(x))}
    return _Index(numbers, set(_VERSION.findall(text)), _fold(text),
                  flat.lower().replace("×", "x"))


def _absent(tokens: list[_Token], sources: list[_Index]) -> list[_Token]:
    return [t for t in tokens if not any(s.has(t) for s in sources)]


def _units(md: str, valid: set[int]) -> list[_Unit]:
    """Prose sentences that cite something and assert a figure.

    Left alone: headings, tables, quotes, code, the "Open questions" section,
    questions, sentences about what is not known, and everything from the
    closing "Researched but not used" / "Not researched" sections."""
    end = len(md)
    for h in _TAIL:
        i = md.find("\n" + h)
        if i >= 0:
            end = min(end, i)
    units: list[_Unit] = []
    pos = 0
    fence = False
    open_questions = False
    for line in md[:end].splitlines(keepends=True):
        line_start, pos = pos, pos + len(line)
        stripped = line.strip()
        if stripped.startswith("```"):
            fence = not fence
            continue
        if stripped.startswith("## "):
            open_questions = bool(_OPEN_QUESTIONS.match(stripped))
        if fence or open_questions or not stripped or stripped[0] in "#|>":
            continue
        body = line.rstrip("\n")
        spans: list[list[int]] = []
        start = 0
        for m in _BREAK.finditer(body):
            spans.append([start, m.start()])
            start = m.end()
        spans.append([start, len(body)])
        merged: list[list[int]] = []
        for s in spans:  # "e.g. RDNA 3" is not two sentences
            if merged and _ABBREV.search(body[merged[-1][0]:merged[-1][1]]):
                merged[-1][1] = s[1]
            else:
                merged.append(s)
        for s, e in merged:
            seg = body[s:e]
            text = seg.strip()
            bare = CITATION_GROUP_RE.sub("", text).rstrip(" *_.")
            if bare.endswith("?") or _ABSENCE.search(text):
                continue
            ids = [n for n in dict.fromkeys(citation_ids(text)) if n in valid]
            if not ids:
                continue
            tokens = claim_tokens(text)
            if not tokens:
                continue
            a = line_start + s + (len(seg) - len(seg.lstrip()))
            units.append(_Unit(a, a + len(text), text,
                               body.strip()[:_CONTEXT_CHARS], ids, tokens))
    return units


def _block(f: Finding) -> str:
    return f"{f.citation_line()}\n{_evidence_text(f)[:_EVIDENCE_CHARS]}"


def _with_citations(text: str, ids: list[int]) -> str:
    """The sentence citing `ids` instead, where its last marker was."""
    marker = "".join(f"[{n}]" for n in ids)
    pieces = CITATION_GROUP_RE.split(text)[0::2]   # the text between markers
    if len(pieces) == 1:
        return text.rstrip() + " " + marker
    head = "".join(p.rstrip(" ") for p in pieces[:-1])
    sep = "" if not head or head[-1] in "( \n" else " "
    return head + sep + marker + pieces[-1]


def _keep_frame(original: str, revised: str) -> str:
    """A correction keeps the list marker and bold label the sentence had,
    and gains neither when it had none."""
    mark = _LIST_MARK.match(original)
    rest = original[mark.end():] if mark else original
    label = _BOLD_LABEL.match(rest)
    body = revised
    if own := _LIST_MARK.match(body):
        body = body[own.end():]
    if label and not _BOLD_LABEL.match(body):
        body = label.group(1) + body
    return (mark.group(1) if mark else "") + body


def _unsupported(tokens: list[_Token], ids: list[int], index: dict[int, _Index],
                 question: _Index, strict: bool = False) -> list[_Token]:
    sources = [index[n] for n in ids]
    return [t for t in tokens if not question.has(t)
            and not any((s.states(t) if strict else s.has(t)) for s in sources)]


def _decide(unit: _Unit, out: CiteCheckOut, allowed: set[int],
            index: dict[int, _Index], question: _Index) -> tuple[str, str | None]:
    if out.action == "keep":
        return "kept", None
    if out.action == "recite":
        ids = [n for n in dict.fromkeys(out.cite) if n in allowed]
        if not ids:
            return "rejected", None
        if set(ids) == set(unit.ids):
            return "kept", None
        # Every figure in the sentence must be stated by what it cites after
        # the move — the model's list alone once left "15 of 19 benchmarks"
        # to sources that never said it. Its list, else its list plus the
        # original citations, else nothing.
        for cand in (ids, ids + [n for n in unit.ids if n not in ids]):
            if not _unsupported(unit.tokens, cand, index, question):
                return "recited", _with_citations(unit.text, cand)
        return "rejected", None
    revised = " ".join(out.revised.split())
    if not revised or revised == unit.text:
        return "kept", None
    lo, hi = _REWRITE_RATIO
    if not lo <= len(revised) / max(1, len(unit.text)) <= hi:
        return "rejected", None
    ids = list(dict.fromkeys(citation_ids(revised)))
    if not ids or any(n not in allowed for n in ids):
        return "rejected", None
    if _unsupported(claim_tokens(revised), ids, index, question, strict=True):
        return "rejected", None
    return "rewritten", _keep_frame(unit.text, revised)


async def check(llm, overview: str, findings: list[Finding], *, query: str,
                bus=None, run_id: str = "") -> tuple[str, CiteCheck]:
    """The document with mismatched citations fixed, and what was done.

    Any failure leaves the document as it was: this is a correction pass
    over a finished overview and must never cost it."""
    summary = CiteCheck()
    try:
        return await _check(llm, overview, findings, query, summary, bus, run_id)
    except Exception:
        log.exception("citation check failed; keeping the overview as written")
        return overview, summary


async def _check(llm, overview, findings, query, summary, bus, run_id):
    by_idx = {f.idx: f for f in findings}
    index = {i: _index(_evidence_text(f)) for i, f in by_idx.items()}
    question = _index(query or "")
    units = _units(overview, set(by_idx))
    summary.sentences = len(units)
    # A figure from the asker's own setup ("on an RX 6700 XT") needs no
    # source, and a sentence whose only unmatched figures are that is never
    # sent: the model removed the asker's own engine from one when it was.
    flagged = []
    for u in units:
        miss = [t for t in _absent(u.tokens, [index[i] for i in u.ids])
                if not question.has(t)]
        if miss:
            flagged.append((u, miss))
    summary.flagged = len(flagged)
    if not flagged:
        if bus is not None and run_id and units:
            bus.publish(run_id, "log", message=summary.message())
        return overview, summary

    todo = flagged[:_MAX_CHECKS]
    summary.skipped = len(flagged) - len(todo)
    if bus is not None and run_id:
        bus.publish(run_id, "phase", phase="citation check")

    async def one(u: _Unit, miss: list[_Token]):
        others = sorted(
            (f for i, f in by_idx.items() if i not in u.ids
             and any(index[i].has(t) for t in miss)),
            key=lambda f: (-sum(index[f.idx].has(t) for t in miss), -f.relevance))
        others = others[:_OTHERS_MAX]
        prompt = prompts.CITE_CHECK.format(
            query=(query or "")[:2000], sentence=u.text, context=u.context,
            absent=", ".join(t.shown for t in miss),
            evidence="\n\n".join(_block(by_idx[i]) for i in u.ids),
            others=("\nOther kept sources that do state some of these "
                    "figures:\n\n" + "\n\n".join(_block(f) for f in others) + "\n")
            if others else "")
        try:
            out = await llm.chat_json(
                "citecheck", [{"role": "user", "content": prompt}], CiteCheckOut,
                max_tokens=1200, temperature=0.1)
        except Exception as e:
            log.warning("citation check call failed: %s", e)
            return u, "failed", None, ""
        allowed = set(u.ids) | {f.idx for f in others}
        outcome, new = _decide(u, out, allowed, index, question)
        return u, outcome, new, out.reason

    results = await asyncio.gather(*(one(u, m) for u, m in todo))
    edits = []
    for u, outcome, new, reason in results:
        setattr(summary, outcome, getattr(summary, outcome) + 1)
        if new is not None and new != u.text:
            edits.append((u.start, u.end, new))
            summary.changes.append({"outcome": outcome, "before": u.text,
                                    "after": new, "reason": reason})
    for start, end, new in sorted(edits, reverse=True):
        overview = overview[:start] + new + overview[end:]
    if bus is not None and run_id:
        bus.publish(run_id, "log", message=summary.message())
    return overview, summary
