"""Per-document note-taking: relevance scoring + structured notes → Finding."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from app.llm import prompts
from app.llm.client import LLM
from app.llm.json_utils import LLMJsonError
from app.models import NotesOut

log = logging.getLogger(__name__)

# Document budget per notes call. Prompt processing is far faster than
# generation (locally and in every cloud), so feeding the whole page costs
# little and grades it on what it actually says — keyword excerpts kept
# rating pages from fragments. ~40k chars ≈ 13k est-tokens; only documents
# larger than that fall back to keyword excerpting / head+tail clipping.
_INPUT_CHARS = 40_000
_HEAD_CHARS = 34_000
_TAIL_CHARS = 6_000

# Sources at or above this score are kept. 4 = "real material on part of the
# brief" under the notes rubric — demanding briefs made the old bar of 5 throw
# away chip datasheets and comparison guides that answered a third of the
# question, leaving runs empty.
RELEVANCE_KEEP = 4


@dataclass
class Finding:
    idx: int
    url: str
    title: str
    domain: str
    published: str | None
    relevance: int
    summary: str
    notes_md: str
    key_facts: list[dict] = field(default_factory=list)
    path: str = ""
    query: str = ""

    def citation_line(self) -> str:
        date = self.published or "undated"
        return f"[{self.idx}] {self.title} — {self.domain} ({date})"


def render_facts(facts: list[dict], *, indent: str = "", quotes: bool = True,
                 limit: int | None = None) -> str:
    """Markdown bullets for extracted facts.

    Shared by the finding file, the gap prompt, and synthesis so a change to
    the Fact shape can't silently leave one consumer printing dict reprs.
    """
    lines: list[str] = []
    for fact in facts[:limit]:
        claim = str(fact.get("claim", "")).strip()
        if not claim:
            continue
        conf = fact.get("confidence")
        suffix = f" (confidence {conf}/10)" if conf is not None else ""
        lines.append(f"{indent}- {claim}{suffix}")
        quote = (fact.get("evidence_quote") or "").strip() if quotes else ""
        if quote:
            lines.append(f'{indent}  > "{quote}"')
    return "\n".join(lines)


def clip_text(text: str) -> str:
    if len(text) <= _HEAD_CHARS + _TAIL_CHARS:
        return text
    return (text[:_HEAD_CHARS] + "\n\n[... document truncated ...]\n\n"
            + text[-_TAIL_CHARS:])


HEADER_CHARS = 1000


def _keyword_pattern(keyword: str) -> re.Pattern | None:
    """Whole-word matcher for one keyword.

    Substring matching is wrong here: planner keywords are routinely short
    tokens like "AI", "EV" or "LFP", and a plain find() matches them inside
    said, chain, maintain, self… filling the excerpt budget with noise and
    crowding out real hits. Boundaries are only added where the keyword
    actually starts/ends with a word character, so phrases like "$/kWh" and
    "C&I" still match.
    """
    kw = str(keyword).strip().lower()
    if not kw:
        return None
    prefix = r"\b" if kw[0].isalnum() or kw[0] == "_" else ""
    suffix = r"\b" if kw[-1].isalnum() or kw[-1] == "_" else ""
    # collapse internal whitespace so "solid state" matches "solid\n state"
    body = r"\s+".join(re.escape(part) for part in kw.split())
    try:
        return re.compile(prefix + body + suffix, re.IGNORECASE)
    except re.error:
        return None


def select_excerpts(text: str, keywords: list[str], window: int = 1200, max_excerpts: int = 8, max_chars: int = 12000) -> str:
    if not text or not keywords:
        return ""

    half = max(1, window // 2)
    max_hits_per_keyword = 20
    min_truncated_excerpt = 200
    excerpt_joiner = "\n[…]\n"

    hits = []
    for k, kw in enumerate(keywords):
        pattern = _keyword_pattern(kw)
        if pattern is None:
            continue
        for count, m in enumerate(pattern.finditer(text)):
            if count >= max_hits_per_keyword:
                break
            hits.append({"pos": m.start(), "end": m.end(), "kw": k})

    if not hits:
        return ""

    # The top of a document carries the title, byline and publish date — the
    # notes prompt is asked for published_date, so excerpting keyword windows
    # alone quietly degrades date extraction. Always keep the header.
    hits.append({"pos": 0, "end": min(HEADER_CHARS, len(text)), "kw": -1})

    hits.sort(key=lambda h: h["pos"])
    
    ranges = []
    for h in hits:
        start = max(0, h["pos"] - half)
        end = min(len(text), h["end"] + half)
        
        if ranges and start <= ranges[-1]["end"]:
            if end > ranges[-1]["end"]:
                ranges[-1]["end"] = end
            ranges[-1]["kws"].add(h["kw"])
        else:
            ranges.append({"start": start, "end": end, "kws": {h["kw"]}, "order": len(ranges)})
            
    ranked = sorted(ranges, key=lambda r: (len(r["kws"]), -r["order"]), reverse=True)
    
    picked = []
    total = 0
    
    for r in ranked:
        if len(picked) >= max_excerpts:
            break
        length = r["end"] - r["start"]
        if total + length <= max_chars:
            picked.append({"start": r["start"], "end": r["end"]})
            total += length
        else:
            remaining = max_chars - total
            if remaining >= min_truncated_excerpt:
                picked.append({"start": r["start"], "end": r["start"] + remaining})
            break
            
    if not picked:
        return ""
        
    picked.sort(key=lambda r: r["start"])
    return excerpt_joiner.join(text[r["start"]:r["end"]].strip() for r in picked)


async def take_notes(llm: LLM, *, brief: str, recency_desc: str, today: str,
                     url: str, title: str, detected_date: str | None,
                     text: str, keywords: list[str] | None = None) -> NotesOut | None:
    """Returns None when the model output is unusable (doc gets skipped)."""
    if len(text) > _INPUT_CHARS:
        # Too big to feed whole: keyword-focused excerpts if we have
        # keywords, head+tail otherwise.
        filtered = (select_excerpts(text, keywords, window=2400,
                                    max_excerpts=24, max_chars=_INPUT_CHARS)
                    if keywords else "")
        text = filtered or clip_text(text)


    prompt = prompts.NOTES.format(
        brief=brief, recency_desc=recency_desc, today=today, url=url,
        title=title, detected_date=detected_date or "unknown",
        text=text,
    )
    try:
        return await llm.chat_json(
            "notes", [{"role": "user", "content": prompt}],
            # 350 words of notes plus up to 8 facts with verbatim quotes does
            # not fit in 1200 tokens; truncation there silently drops sources.
            NotesOut, max_tokens=2400, temperature=0.2,
        )
    except LLMJsonError as e:
        log.warning("notes skipped for %s: %s", url, e)
        return None


def finding_markdown(f: Finding) -> str:
    facts = render_facts(f.key_facts) or "_none extracted_"
    return f"""# [{f.idx}] {f.title}

- **URL:** {f.url}
- **Domain:** {f.domain}
- **Published:** {f.published or "unknown"}
- **Relevance:** {f.relevance}/10
- **Found via:** {f.query}

**Summary:** {f.summary}

## Notes

{f.notes_md}

## Key facts

{facts}
"""
