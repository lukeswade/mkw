"""Score research reports against a frozen set of real runs.

    scripts/eval/report_eval.sh freeze             # once: copy each case's sources and report
    scripts/eval/report_eval.sh check              # every known error is detected in its baseline
    scripts/eval/report_eval.sh score              # the reports as the runs stored them
    scripts/eval/report_eval.sh resynth <label>    # rewrite every report with the code in the tree
    scripts/eval/report_eval.sh resynth <label> --set=fact_ledger=true   # ... with a setting changed
    scripts/eval/report_eval.sh score <label> [--judge]

Why: a change to how reports are written (prompts, synthesis structure, a
consistency pass) used to be judged by reading one report. This scores it on
the same eight questions every time, against errors already found in those
reports by hand and verified in their sources.

The set is data/eval/reports-2026-11-15/ (gitignored, like the retrieval set):
`cases.json`, and per case the plan (`meta.json`), every kept source as the run
stored it (`findings/`) and the report as it stood (`baseline.md`). An arm is
`arms/<label>/<case>.md`. A command that used the model asks the server to
unload it afterwards, unless a research run is using it; pass --keep-loaded
when another arm follows. Nothing is read from the database, nothing is
written outside the set, and `freeze` never overwrites a case.

Measured without a model, for every report:
  words          length of the body
  cited          kept sources the body cites, of those kept
  unmatched      cited sentences with a figure their sources' notes do not
                 state, of those with figures (citecheck's string pass)
  repeats        sentences restated in a later section (summaries excepted)
  lost parts     parts of the question whose kept sources all went uncited
  known errors   facts from cases.json the report gets wrong, by kind:
                 "substantive" (wrong subject, owner, status or comparison)
                 and "citation" (right figure, wrong source)
With --judge (needs the model server; about 7 minutes for the set), for
substantive facts and each report:
  contradicted   facts the report contradicts, judged one passage at a time;
                 counted only when the quoted sentence is in the report
  conflicts      pairs of statements that cannot both be true; counted only
                 when both quotes are in the report, both name the subject,
                 and the model confirms the pair when shown it alone

A known error is a regex screen, so it can miss an error worded a new way;
the judge is the check that reads, and on the baseline it agreed with 7 of
the 8 hand-verified substantive errors. The conflicts count finds few of the
conflicts there are (1 of at least 3 in the finance report): read it as
"at least". Both are reported; neither is edited.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.research import citecheck, synthesizer
from app.research.dedupe import _STOPWORDS, stem_token
from app.research.notes import Finding
from app.research.pipeline import _md_facts, _md_field, _md_section
from app.research.storage import citation_ids

SET = Path(os.environ.get("REPORT_EVAL_SET", "/data/eval/reports-2026-11-15"))
RUNS = Path(os.environ.get("REPORT_EVAL_RUNS", "/data/research_data"))
_TAIL = (synthesizer.UNUSED_HEADING, "## Not researched")
# Sections that restate the document on purpose; a repeat there is not waste.
_SUMMARY = re.compile(r"tl;?dr|summary|conclusion|takeaway|bottom line|at a glance|"
                      r"recommendation|open questions", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z*_`\[(\"“])")
_REPEAT_JACCARD = 0.5
_REPEAT_CONTAINED = 0.55
_REPEAT_MIN_WORDS = 8
_STRONG = 7


# --- reading a report ---------------------------------------------------------

def body_of(md: str) -> str:
    """The report without its closing "not used" / "not researched" lists."""
    end = len(md)
    for h in _TAIL:
        i = md.find("\n" + h)
        if i >= 0:
            end = min(end, i)
    return md[:end]


@dataclass
class Block:
    section: str      # the H2 it sits under
    headings: str     # "H2 > H3", the context a bullet is read in
    text: str


def blocks(md: str) -> list[Block]:
    """Paragraphs, bullets and table rows of the body, each with its headings.

    A bullet reading "Examples: Florida Urology Center, New Jersey Urology"
    says nothing wrong until it is read under "Add-on targets", so a block
    always carries the headings above it."""
    out: list[Block] = []
    h2 = h3 = label = ""
    items: list[tuple[int, str]] = []      # open list items: (indent, text)
    fence = False
    for line in body_of(md).splitlines():
        s = line.strip()
        if s.startswith("```"):
            fence = not fence
            continue
        if fence or not s or re.fullmatch(r"\|?[\s:|-]+\|?", s) or s == "---":
            continue
        if m := re.match(r"(#{1,6})\s+(.*)", s):
            level, title = len(m.group(1)), m.group(2).strip()
            if level == 2:
                h2, h3 = title, ""
            elif level >= 3:
                h3 = title
            label, items = "", []
            continue
        # A line that is only bold is a heading in all but markup:
        # "**Alternative: Qwen3-Coder 30B-A3B (MoE)**" names the subject of
        # the paragraph under it.
        if re.fullmatch(r"\*\*[^*\n]{3,140}\*\*:?", s):
            label, items = s.strip("*:"), []
            continue
        # A nested bullet is read under its parent: "Performance: ~24 tok/s"
        # is about whatever the bullet above it names.
        parents: list[str] = []
        if re.match(r"(?:[-*+]|\d{1,3}[.)])\s+", s):
            indent = len(line) - len(line.lstrip())
            items = [(i, t) for i, t in items if i < indent]
            parents = [t[:160] for _, t in items]
            items.append((indent, s))
        else:
            items = []
        out.append(Block(h2, " > ".join(h for h in (h2, h3, label, *parents) if h), s))
    return out


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


def _plain(text: str) -> str:
    """For comparing a quote with the report: markdown marks and spacing off."""
    return " ".join(re.sub(r"[*`]", "", text).split())


# --- known errors ---------------------------------------------------------------

def _cond_hits(cond: dict, haystack: str, cites: list[int]) -> bool:
    if not all(re.search(p, haystack, re.I) for p in cond.get("all", [])):
        return False
    if any(re.search(p, haystack, re.I) for p in cond.get("none", [])):
        return False
    need = cond.get("needs_cite")
    # "cites the wrong source" needs a citation to be wrong about: a
    # sentence with none is not a citation error.
    if need and (not cites or set(cites) & set(need)):
        return False
    return True


def cond_evidence(cond: dict, md: str) -> str:
    """The first block or sentence of the report this condition matches.

    A sentence-scoped condition may name its `context`: patterns the block
    around the sentence must match. "The recommended deployment is 4GB RAM
    [35]" is only about Linkwarden because of the bullet it closes."""
    for b in blocks(md):
        if cond.get("scope") == "sentence":
            around = b.headings + "\n" + b.text
            if not all(re.search(p, around, re.I) for p in cond.get("context", [])):
                continue
            for s in sentences(b.text):
                if _cond_hits(cond, s, citation_ids(s)):
                    return s
        elif _cond_hits(cond, b.headings + "\n" + b.text, citation_ids(b.text)):
            return b.text
    return ""


def fact_status(fact: dict, md: str) -> tuple[str, str]:
    """("present" | "clear" | "absent", evidence).

    present: every condition of the fact's violation is matched somewhere.
    absent:  the report never mentions the subject, so it cannot be wrong
             about it, or right."""
    hits = [cond_evidence(c, md) for c in fact["violation"]]
    if all(hits):
        return "present", hits[0]
    body = body_of(md)
    if not any(re.search(t, body, re.I) for t in fact.get("terms", [])):
        return "absent", ""
    return "clear", ""


# --- measures that need no model ---------------------------------------------

def _words(sentence: str) -> frozenset[str]:
    text = re.sub(r"\[\d[\d,; ]*\]", " ", sentence.lower())
    return frozenset(stem_token(w) for w in re.findall(r"[a-z0-9][a-z0-9.-]*[a-z0-9]|[a-z0-9]", text)
                     if len(w) > 2 and w not in _STOPWORDS)


def repeats(md: str, query: str = "") -> int:
    """Sentences that restate one from an earlier section.

    2026-11-15: a 3,800-word report gave its valuation multiples, an ASC
    premium and two state laws twice each. A restatement is rarely word for
    word, so two tests: half the words in common, or most of the shorter
    sentence's words plus a figure or numbered name in common ("10-14x" in
    both). Figures from the question do not count as that anchor, or every
    sentence naming the asker's GPU would pair up. Summaries are skipped;
    they restate by design."""
    question = citecheck._index(query or "")
    seen: list[tuple[str, frozenset[str], frozenset]] = []
    count = 0
    for b in blocks(md):
        if b.text.startswith("|") or _SUMMARY.search(b.section):
            continue
        for s in sentences(b.text):
            w = _words(s)
            if len(w) < _REPEAT_MIN_WORDS:
                continue
            anchors = frozenset((t.kind, t.key) for t in citecheck.claim_tokens(s)
                                if not question.has(t))
            for sec, prev, prev_anchors in seen:
                if sec == b.section:
                    continue
                shared = len(w & prev)
                if (shared / len(w | prev) >= _REPEAT_JACCARD
                        or (shared / min(len(w), len(prev)) >= _REPEAT_CONTAINED
                            and anchors & prev_anchors)):
                    count += 1
                    break
            seen.append((b.section, w, anchors))
    return count


def figure_stats(md: str, findings: list[Finding], query: str) -> tuple[int, int]:
    """(cited sentences with figures, those with one their sources do not state).

    citecheck's own string pass, read off its helpers so the eval and the
    live check cannot drift apart."""
    by_idx = {f.idx: f for f in findings}
    index = {i: citecheck._index(citecheck._evidence_text(f)) for i, f in by_idx.items()}
    question = citecheck._index(query or "")
    units = citecheck._units(md, set(by_idx))
    flagged = 0
    for u in units:
        miss = citecheck._absent(u.tokens, [index[i] for i in u.ids])
        flagged += any(not question.has(t) for t in miss)
    return len(units), flagged


def model_free(md: str, case: dict, meta: dict, findings: list[Finding]) -> dict:
    cited = synthesizer.cited_ids(md)
    facet_of = meta.get("query_facet") if isinstance(meta.get("query_facet"), dict) else {}
    lost, _strong = synthesizer.funnel_losses(md, findings, facet_of)
    figures, unmatched = figure_stats(md, findings, meta.get("query") or "")
    known = {}
    for fact in case.get("facts", []):
        status, evidence = fact_status(fact, md)
        known[fact["id"]] = {"kind": fact["kind"], "status": status, "evidence": evidence[:300]}
    return {
        "words": len(body_of(md).split()),
        "kept": len(findings),
        "cited": sum(1 for f in findings if f.idx in cited),
        "strong_uncited": sum(1 for f in findings if f.relevance >= _STRONG and f.idx not in cited),
        "figures": figures, "unmatched": unmatched,
        "repeats": repeats(md, meta.get("query") or ""),
        "lost_parts": list(lost),
        "known": known,
    }


# --- the judge ----------------------------------------------------------------

class FactVerdict(BaseModel):
    verdict: Literal["contradicted", "consistent", "not_addressed"] = "not_addressed"
    quote: str = Field(default="", max_length=600)
    why: str = Field(default="", max_length=300)

    @field_validator("verdict", mode="before")
    @classmethod
    def _known(cls, v):
        v = str(v or "").strip().lower().replace(" ", "_")
        return v if v in ("contradicted", "consistent", "not_addressed") else "not_addressed"

    @field_validator("quote", "why", mode="before")
    @classmethod
    def _clip(cls, v, info):
        return v[:600 if info.field_name == "quote" else 300] if isinstance(v, str) else v


class Pair(BaseModel):
    subject: str = Field(default="", max_length=120)
    a: str = Field(default="", max_length=600)
    b: str = Field(default="", max_length=600)
    why: str = Field(default="", max_length=300)

    @field_validator("subject", "a", "b", "why", mode="before")
    @classmethod
    def _clip(cls, v, info):
        n = {"subject": 120, "why": 300}.get(info.field_name, 600)
        return v[:n] if isinstance(v, str) else v


class PairVerdict(BaseModel):
    """Anything but a clear "conflict" is compatible: the costly mistake is
    counting a restatement as a contradiction."""
    verdict: Literal["conflict", "compatible"] = "compatible"
    why: str = Field(default="", max_length=300)

    @field_validator("verdict", mode="before")
    @classmethod
    def _known(cls, v):
        return "conflict" if str(v or "").strip().lower() == "conflict" else "compatible"

    @field_validator("why", mode="before")
    @classmethod
    def _clip(cls, v):
        return v[:300] if isinstance(v, str) else v


class Conflicts(BaseModel):
    pairs: list[Pair] = Field(default_factory=list, max_length=8)

    @field_validator("pairs", mode="before")
    @classmethod
    def _cap(cls, v):
        return v[:8] if isinstance(v, list) else v


FACT_PROMPT = """You are checking one passage of a research report against a \
fact that the report's own sources establish.

FACT:
{fact}

PASSAGE (from the section "{section}"):
{passage}

Does the passage state or clearly imply something that cannot be true if the \
fact is true? Produce a JSON object with exactly these keys:
- "verdict": "contradicted" if it does; "consistent" if it agrees with the \
fact or states it; "not_addressed" if it does not touch what the fact is \
about. A passage that merely leaves the fact out is not a contradiction.
- "quote": for "contradicted", the one sentence or table row that does so, \
copied exactly from the passage. Otherwise "".
- "why": one short sentence.

Respond with only the JSON object."""

CONFIRM_PROMPT = """Two statements from the same research report, both about \
{subject}:

A: {a}

B: {b}

Can both be true at the same time? Produce a JSON object with exactly these keys:
- "verdict": "conflict" if they cannot: they give the same attribute of \
{subject} two different values, owners, backers, dates, versions or statuses. \
"compatible" if they can: one repeats, summarises, rounds or adds detail to \
the other, or they are about different attributes or different things.
- "why": one short sentence.

Respond with only the JSON object."""

CONFLICT_PROMPT = """Below is a research report. Find statements in it that \
cannot both be true: the same named thing given two different values, owners, \
backers, dates, versions or statuses in two places.

These are NOT conflicts: two different things; a range and a figure inside it; \
a general figure and a specific case; a rounded and an exact figure; a place \
where the report itself says its sources disagree.

Produce a JSON object with one key, "pairs": an array of at most 8 objects, \
clearest first, each with
- "subject": the named thing both statements are about
- "a": the first statement, copied exactly (one sentence or one table row)
- "b": the second, copied exactly
- "why": one short sentence
An empty array if there are none.

REPORT:
{report}

Respond with only the JSON object."""


def passages_for(fact: dict, md: str, limit: int = 12) -> list[Block]:
    terms = fact.get("terms", [])
    return [b for b in blocks(md)
            if any(re.search(t, b.headings + "\n" + b.text, re.I) for t in terms)][:limit]


def _in_report(quote: str, md: str) -> bool:
    q = _plain(quote)
    return len(q) >= 12 and q in _plain(body_of(md))


_RANK = {"contradicted": 0, "unverified": 1, "failed": 2, "consistent": 3, "not_addressed": 4}


async def judge_fact(llm, fact: dict, md: str) -> dict:
    """Whether the report contradicts the fact anywhere.

    One call per passage. Shown every passage at once, the judge called a
    report "consistent" that doubled Qwen3.8's speed in its TL;DR and gave
    the figure to the right model six sections later: one right passage
    excused the wrong one. A quote that is not in the report downgrades
    "contradicted" to "unverified", which is not counted."""
    found = passages_for(fact, md)
    if not found:
        return {"verdict": "not_addressed", "quote": "", "why": "subject not mentioned"}

    async def one(b: Block) -> dict:
        prompt = FACT_PROMPT.format(fact=fact["fact"], section=b.headings or "top",
                                    passage=b.text)
        try:
            out = await llm.chat_json("eval_fact", [{"role": "user", "content": prompt}],
                                      FactVerdict, max_tokens=500, temperature=0.0)
        except Exception as e:
            return {"verdict": "failed", "quote": "", "why": str(e)[:200]}
        verdict = out.verdict
        if verdict == "contradicted" and not _in_report(out.quote, md):
            verdict = "unverified"
        return {"verdict": verdict, "quote": out.quote, "why": out.why}

    results = await asyncio.gather(*(one(b) for b in found))
    return min(results, key=lambda r: _RANK[r["verdict"]])


def verify_pairs(pairs: list[Pair], md: str) -> list[dict]:
    """The pairs a reader can check: both quotes are in the report, they are
    different statements, and the block each sits in names the subject."""
    found = blocks(md)

    def home(quote: str) -> str:
        q = _plain(quote)
        for b in found:
            if q in _plain(b.text):
                return _fold(b.headings + " " + b.text)
        return ""

    kept: list[dict] = []
    seen: set[frozenset[str]] = set()
    for p in pairs:
        a, b = _plain(p.a), _plain(p.b)
        subject = _fold(p.subject)
        if len(a) < 12 or len(b) < 12 or a == b or a in b or b in a or len(subject) < 3:
            continue
        ha, hb = home(p.a), home(p.b)
        if not ha or not hb or subject not in ha or subject not in hb:
            continue
        key = frozenset((a, b))
        if key in seen:
            continue
        seen.add(key)
        kept.append({"subject": p.subject, "a": p.a, "b": p.b, "why": p.why})
    return kept


def _fold(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


async def judge_conflicts(llm, md: str) -> dict:
    """Pairs of statements in the report that cannot both be true.

    Three gates, because the first alone is not usable: asked to find
    conflicts in a whole report, the model returned 7 checkable pairs across
    eight reports and 6 of them were one statement made twice. So the pairs
    it claims are checked in code (both quotes in the report, both naming the
    subject), and each survivor is then put to the model alone, two
    sentences at a time, where a restatement is easy to tell from a
    conflict. Only confirmed pairs are counted."""
    prompt = CONFLICT_PROMPT.format(report=body_of(md))
    try:
        out = await llm.chat_json("eval_conflicts", [{"role": "user", "content": prompt}],
                                  Conflicts, max_tokens=4000, temperature=0.0)
    except Exception as e:
        return {"failed": str(e)[:200], "claimed": 0, "checkable": 0, "pairs": []}
    checkable = verify_pairs(out.pairs, md)

    async def confirm(p: dict) -> bool:
        ask = CONFIRM_PROMPT.format(subject=p["subject"], a=p["a"], b=p["b"])
        try:
            v = await llm.chat_json("eval_confirm", [{"role": "user", "content": ask}],
                                    PairVerdict, max_tokens=300, temperature=0.0)
        except Exception:
            return False
        p["confirmed_why"] = v.why
        return v.verdict == "conflict"

    kept = await asyncio.gather(*(confirm(p) for p in checkable))
    return {"claimed": len(out.pairs), "checkable": len(checkable),
            "pairs": [p for p, ok in zip(checkable, kept) if ok]}


# --- the frozen set -------------------------------------------------------------

def load_cases(only: str | None = None) -> list[dict]:
    cases = json.loads((SET / "cases.json").read_text())["cases"]
    return [c for c in cases if not only or c["id"] == only]


def load_findings(case_dir: Path) -> list[Finding]:
    """A run's kept sources from its finding files, as re-synthesis reads them
    (pipeline._stored_findings), without the database."""
    out: list[Finding] = []
    for p in sorted((case_dir / "findings").glob("[0-9][0-9][0-9]_*.md")):
        body = p.read_text(encoding="utf-8", errors="replace")
        head = re.match(r"#\s*\[(\d+)\]\s*(.+)", body)
        if not head:
            continue
        rel = re.search(r"\*\*Relevance:\*\*\s*(\d+)", body)
        summary = re.search(r"^\*\*Summary:\*\*\s*(.+)$", body, re.M)
        published = _md_field(body, "Published")
        out.append(Finding(
            idx=int(head.group(1)), url=_md_field(body, "URL"),
            title=head.group(2).strip(), domain=_md_field(body, "Domain"),
            published=published if re.match(r"\d{4}", published or "") else None,
            relevance=int(rel.group(1)) if rel else 5,
            summary=summary.group(1).strip() if summary else "",
            notes_md=_md_section(body, "Notes") or body, key_facts=_md_facts(body),
            query=_md_field(body, "Found via"), source_type=_md_field(body, "Source type"),
            publisher=_md_field(body, "Publisher")))
    return out


def report_path(case_id: str, arm: str) -> Path:
    return SET / case_id / "baseline.md" if arm == "baseline" else SET / "arms" / arm / f"{case_id}.md"


def freeze() -> int:
    """Copy each case's plan, sources and report into the set, once."""
    bad = 0
    for case in load_cases():
        dest = SET / case["id"]
        if dest.exists():
            print(f"kept    {case['id']} (already frozen)")
            continue
        src = RUNS / case["run"]
        if not (src / "overview.md").exists():
            print(f"MISSING {case['id']}: {src}")
            bad += 1
            continue
        dest.mkdir(parents=True)
        shutil.copy(src / "meta.json", dest / "meta.json")
        shutil.copy(src / "overview.md", dest / "baseline.md")
        shutil.copytree(src / "findings", dest / "findings")
        print(f"froze   {case['id']}: {len(list((dest / 'findings').glob('*.md')))} sources, "
              f"{len((dest / 'baseline.md').read_text().split())} words")
    return bad


def check() -> int:
    """Every known error is detected in the report it was found in, and
    every source a fact names exists. A pattern that does not fire on its
    own baseline measures nothing."""
    bad = 0
    for case in load_cases():
        d = SET / case["id"]
        md = (d / "baseline.md").read_text()
        ids = {f.idx for f in load_findings(d)}
        for fact in case.get("facts", []):
            status, evidence = fact_status(fact, md)
            named = set(fact.get("sources", [])) | {n for c in fact["violation"]
                                                    for n in c.get("needs_cite", [])}
            problems = []
            if status != "present":
                problems.append(f"not detected in baseline ({status})")
            if named - ids:
                problems.append(f"names sources not in the run: {sorted(named - ids)}")
            bad += bool(problems)
            print(f"{'FAIL' if problems else 'ok  '} {case['id']} / {fact['id']}"
                  + (": " + "; ".join(problems) if problems else f"  «{evidence[:90]}»"))
    return bad


# --- scoring ------------------------------------------------------------------

async def score(arm: str, judge: bool, only: str | None) -> int:
    llm = None
    if judge:
        from app.config import load_settings
        from app.llm.client import LLM
        llm = LLM(load_settings())
    timing = {}
    arm_file = SET / "arms" / arm / "arm.json"
    if arm_file.exists():
        timing = json.loads(arm_file.read_text())
    rows = {}
    for case in load_cases(only):
        path = report_path(case["id"], arm)
        if not path.exists():
            print(f"no report for {case['id']} in arm {arm!r} ({path})", file=sys.stderr)
            continue
        d = SET / case["id"]
        md = path.read_text()
        meta = json.loads((d / "meta.json").read_text())
        m = model_free(md, case, meta, load_findings(d))
        m["secs"] = (timing.get(case["id"]) or {}).get("secs")
        if llm is not None:
            subst = [f for f in case.get("facts", []) if f["kind"] == "substantive"]
            verdicts = await asyncio.gather(*(judge_fact(llm, f, md) for f in subst))
            m["judged"] = {f["id"]: v for f, v in zip(subst, verdicts)}
            m["conflicts"] = await judge_conflicts(llm, md)
        rows[case["id"]] = m
    print_table(arm, rows, judge)
    record = {"at": time.strftime("%Y-%m-%d %H:%M:%S"), "arm": arm, "judge": judge,
              "rev": os.environ.get("REPORT_EVAL_REV", ""), "cases": rows,
              "totals": totals(rows)}
    with open(SET / "results.jsonl", "a") as fh:
        fh.write(json.dumps(record) + "\n")
    return 0


def totals(rows: dict) -> dict:
    t: Counter = Counter()
    for m in rows.values():
        for k in ("words", "kept", "cited", "figures", "unmatched", "repeats"):
            t[k] += m[k]
        t["lost_parts"] += len(m["lost_parts"])
        for k in m["known"].values():
            t[f"{k['kind']}_total"] += 1
            t[f"{k['kind']}_present"] += k["status"] == "present"
        for v in (m.get("judged") or {}).values():
            t["judged_total"] += 1
            t[f"judged_{v['verdict']}"] += 1
        if "conflicts" in m:
            t["conflicts"] += len(m["conflicts"]["pairs"])
            t["conflicts_claimed"] += m["conflicts"]["claimed"]
    return dict(t)


def print_table(arm: str, rows: dict, judge: bool) -> None:
    def known(m, kind):
        ks = [k for k in m["known"].values() if k["kind"] == kind]
        return f"{sum(k['status'] == 'present' for k in ks)}/{len(ks)}" if ks else "-"

    head = (f"{'case':26} {'words':>6} {'cited':>7} {'unmatched':>9} {'repeats':>7} "
            f"{'lost':>4} {'subst':>5} {'cite':>5}")
    if judge:
        head += f" {'contra':>6} {'conflicts':>9}"
    head += f" {'secs':>6}"
    print(f"\narm: {arm}\n{head}\n{'-' * len(head)}")
    for cid, m in rows.items():
        line = (f"{cid:26} {m['words']:>6} {m['cited']:>3}/{m['kept']:<3} "
                f"{m['unmatched']:>4}/{m['figures']:<4} {m['repeats']:>7} "
                f"{len(m['lost_parts']):>4} {known(m, 'substantive'):>5} {known(m, 'citation'):>5}")
        if judge:
            j = m.get("judged") or {}
            line += (f" {sum(v['verdict'] == 'contradicted' for v in j.values()):>3}/{len(j):<2}"
                     f" {len(m['conflicts']['pairs']):>9}")
        line += f" {m['secs'] if m['secs'] is not None else '':>6}"
        print(line)
    t = totals(rows)
    line = (f"{'TOTAL':26} {t.get('words', 0):>6} {t.get('cited', 0):>3}/{t.get('kept', 0):<3} "
            f"{t.get('unmatched', 0):>4}/{t.get('figures', 0):<4} {t.get('repeats', 0):>7} "
            f"{t.get('lost_parts', 0):>4} "
            f"{t.get('substantive_present', 0)}/{t.get('substantive_total', 0):<3} "
            f"{t.get('citation_present', 0)}/{t.get('citation_total', 0):<3}")
    if judge:
        line += (f" {t.get('judged_contradicted', 0):>3}/{t.get('judged_total', 0):<2}"
                 f" {t.get('conflicts', 0):>9}")
    print(f"{'-' * len(head)}\n{line}")
    present = [(cid, fid, k) for cid, m in rows.items() for fid, k in m["known"].items()
               if k["status"] == "present"]
    if present:
        print("\nknown errors present:")
        for cid, fid, k in present:
            print(f"  {cid} / {fid} [{k['kind']}]  «{k['evidence'][:110]}»")
    if judge:
        for cid, m in rows.items():
            for fid, v in (m.get("judged") or {}).items():
                if v["verdict"] not in ("consistent", "not_addressed"):
                    print(f"  judge {cid} / {fid}: {v['verdict']}  «{v['quote'][:110]}»")
            for p in m["conflicts"]["pairs"]:
                print(f"  conflict {cid} / {p['subject']}: «{p['a'][:80]}» vs «{p['b'][:80]}»")


# --- writing an arm -----------------------------------------------------------

class _Bus:
    def publish(self, *a, **k):
        return {}


async def resynth(label: str, only: str | None, overrides: dict | None = None) -> int:
    """Write every case's report again from its frozen sources with the code
    in the tree: synthesis, citation validation, the citation check if it is
    on, and the closing sections, as a run or a Re-synthesize would."""
    from app.config import load_settings
    from app.llm import prompts
    from app.llm.client import LLM
    from app.research import facets as facet_plan
    from app.research.pipeline import Pipeline
    from app.research.storage import validate_citations

    cfg = load_settings()
    if overrides:
        cfg = dataclasses.replace(cfg, **overrides)
    pipe = Pipeline(cfg, None, _Bus())
    out_dir = SET / "arms" / label
    out_dir.mkdir(parents=True, exist_ok=True)
    arm_file = out_dir / "arm.json"
    record = json.loads(arm_file.read_text()) if arm_file.exists() else {}
    for case in load_cases(only):
        target = out_dir / f"{case['id']}.md"
        if target.exists():
            print(f"kept    {case['id']} (already written in arm {label!r})")
            continue
        d = SET / case["id"]
        meta = json.loads((d / "meta.json").read_text())
        findings = load_findings(d)
        facets = [f for f in (meta.get("facets") or []) if isinstance(f, str)]
        facet_of = {str(k): str(v) for k, v in (meta.get("query_facet") or {}).items()}
        unanswered = facet_plan.uncovered(facets, Counter(meta.get("facet_kept") or {}))
        query = meta["query"]
        llm = LLM(cfg)
        t0 = time.monotonic()
        try:
            text = await synthesizer.synthesize(
                llm, query=query, title=meta.get("title") or query[:120],
                brief=meta.get("brief") or query,
                recency_desc=prompts.RECENCY_DESC[meta.get("recency") or "all"],
                # the day the run finished, so "as of" reads as it did then
                today=(meta.get("finished_at") or meta.get("created_at") or "")[:10],
                state_md="", findings=findings, previous_overview="",
                uncovered_facets=unanswered, facet_of=facet_of,
                premises=[p for p in (meta.get("premises") or []) if isinstance(p, str)],
                deliverables=[x for x in (meta.get("deliverables") or []) if isinstance(x, str)],
                placeholder_on_failure=False,
                fact_ledger=cfg.fact_ledger)
        except Exception as e:
            print(f"FAILED  {case['id']}: {type(e).__name__}: {str(e)[:200]}")
            record[case["id"]] = {"error": str(e)[:300]}
            arm_file.write_text(json.dumps(record, indent=1))
            continue
        if not synthesizer.looks_like_document(text):
            print(f"FAILED  {case['id']}: synthesis did not return a document")
            record[case["id"]] = {"error": "not a document"}
            arm_file.write_text(json.dumps(record, indent=1))
            continue
        text, _removed = validate_citations(text, len(findings))
        checked = None
        if cfg.citation_check:
            text, summary = await citecheck.check(llm, text, findings, query=query)
            checked = summary.counts()
        text, uncited = pipe._mark_honestly(case["id"], text, findings, facet_of, unanswered)
        secs = round(time.monotonic() - t0, 1)
        target.write_text(text)
        record[case["id"]] = {"secs": secs, "uncited": uncited, "citation_check": checked,
                              "llm": llm.usage_summary(),
                              "rev": os.environ.get("REPORT_EVAL_REV", ""),
                              "settings": overrides or {}}
        arm_file.write_text(json.dumps(record, indent=1))
        print(f"wrote   {case['id']}: {len(text.split())} words in {secs}s")
    return 0


def parse_overrides(argv: list[str]) -> dict:
    """--set name=value pairs, typed like the setting they replace, so an arm
    can switch a feature on without touching the saved settings."""
    from app.config import Settings
    defaults = Settings()
    out = {}
    for a in argv:
        if not a.startswith("--set="):
            continue
        key, _, raw = a[len("--set="):].partition("=")
        if not hasattr(defaults, key):
            raise SystemExit(f"unknown setting: {key}")
        like = getattr(defaults, key)
        if isinstance(like, bool):
            out[key] = raw.strip().lower() in {"1", "true", "yes", "on"}
        elif isinstance(like, int):
            out[key] = int(raw)
        else:
            out[key] = raw
    return out


def release_model(cfg, argv: list[str]) -> str:
    """Ask the model server to unload the model once the eval is done with it,
    and say what happened.

    The main model is pinned on the server, so it never unloads by itself and
    an eval would leave some 20 GB held until someone noticed. It is left
    alone when a research run is using it, when that cannot be told, or with
    --keep-loaded (several arms in a row). Never raises: a server without
    this route just keeps its model."""
    if "--keep-loaded" in argv:
        return "model left loaded (--keep-loaded)"
    import httpx
    health = os.environ.get("REPORT_EVAL_HEALTH", "http://host.docker.internal:8090/health")
    try:
        h = httpx.get(health, timeout=5).json()
    except Exception as e:
        return f"model left loaded: could not tell whether the app is idle ({type(e).__name__})"
    if h.get("active") or h.get("queued"):
        return "model left loaded: a research run is using it"
    name = (cfg.resolved_model or "").rsplit("/", 1)[-1]
    try:
        r = httpx.post(f"{cfg.resolved_base_url.rstrip('/')}/models/{name}/unload",
                       headers={"Authorization": f"Bearer {cfg.resolved_api_key}"}, timeout=60)
    except Exception as e:
        return f"model left loaded: the unload request failed ({type(e).__name__})"
    if r.status_code in (200, 202):
        return "model unloaded"
    if r.status_code == 400:
        return "model was not loaded"
    return f"model left loaded: the server answered {r.status_code}"


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    only = next((a.split("=", 1)[1] for a in argv if a.startswith("--only=")), None)
    cmd = args[0] if args else ""
    if cmd == "freeze":
        return freeze()
    if cmd == "check":
        return check()
    if cmd == "score" or (cmd == "resynth" and len(args) > 1):
        try:
            if cmd == "score":
                return asyncio.run(score(args[1] if len(args) > 1 else "baseline",
                                         "--judge" in argv, only))
            return asyncio.run(resynth(args[1], only, parse_overrides(argv)))
        finally:
            if cmd == "resynth" or "--judge" in argv:      # only these use the model
                from app.config import load_settings
                print(release_model(load_settings(), argv))
    print(__doc__.split("\n\n")[1], file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
