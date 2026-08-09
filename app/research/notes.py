"""Per-document note-taking: relevance scoring + structured notes → Finding."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.llm import prompts
from app.llm.client import LLM
from app.llm.json_utils import LLMJsonError
from app.models import NotesOut

log = logging.getLogger(__name__)

# ~6k est-tokens of document text per notes call
_HEAD_CHARS = 14_000
_TAIL_CHARS = 4_000

RELEVANCE_KEEP = 5


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


def clip_text(text: str) -> str:
    if len(text) <= _HEAD_CHARS + _TAIL_CHARS:
        return text
    return (text[:_HEAD_CHARS] + "\n\n[... document truncated ...]\n\n"
            + text[-_TAIL_CHARS:])


def select_excerpts(text: str, keywords: list[str], window: int = 1200, max_excerpts: int = 8, max_chars: int = 12000) -> str:
    if not text or not keywords:
        return ""

    hay = text.lower()
    half = max(1, window // 2)
    max_hits_per_keyword = 20
    min_truncated_excerpt = 200
    excerpt_joiner = "\n[…]\n"

    hits = []
    for k, kw in enumerate(keywords):
        kw_lower = str(kw).strip().lower()
        if not kw_lower:
            continue
        
        start_idx = 0
        count = 0
        while count < max_hits_per_keyword:
            i = hay.find(kw_lower, start_idx)
            if i == -1:
                break
            hits.append({"pos": i, "end": i + len(kw_lower), "kw": k})
            start_idx = i + len(kw_lower)
            count += 1
            
    if not hits:
        return ""

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
    if keywords:
        filtered = select_excerpts(text, keywords)
        if filtered:
            text = filtered
        else:
            text = clip_text(text)
    else:
        text = clip_text(text)
        
    prompt = prompts.NOTES.format(
        brief=brief, recency_desc=recency_desc, today=today, url=url,
        title=title, detected_date=detected_date or "unknown",
        text=text,
    )
    try:
        return await llm.chat_json(
            "notes", [{"role": "user", "content": prompt}],
            NotesOut, max_tokens=1200, temperature=0.2,
        )
    except LLMJsonError as e:
        log.warning("notes skipped for %s: %s", url, e)
        return None


def finding_markdown(f: Finding) -> str:
    facts_lines = []
    for fact in f.key_facts:
        claim = fact.get("claim", "")
        quote = fact.get("evidence_quote")
        conf = fact.get("confidence", 5)
        
        line = f"- **{claim}** (Confidence: {conf}/10)"
        if quote:
            line += f"\\n  > \"{quote}\""
        facts_lines.append(line)
        
    facts = "\\n".join(facts_lines) or "_none extracted_"
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
