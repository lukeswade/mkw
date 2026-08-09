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
    key_facts: list[str] = field(default_factory=list)
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


async def take_notes(llm: LLM, *, brief: str, recency_desc: str, today: str,
                     url: str, title: str, detected_date: str | None,
                     text: str) -> NotesOut | None:
    """Returns None when the model output is unusable (doc gets skipped)."""
    prompt = prompts.NOTES.format(
        brief=brief, recency_desc=recency_desc, today=today, url=url,
        title=title, detected_date=detected_date or "unknown",
        text=clip_text(text),
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
    facts = "\n".join(f"- {fact}" for fact in f.key_facts) or "_none extracted_"
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
