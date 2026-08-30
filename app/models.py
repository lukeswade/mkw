"""Pydantic models: run parameters and every structured LLM output.

LLM output models are deliberately forgiving (coercing validators instead of
hard Literals where a local model might improvise) — a parse failure costs a
repair round-trip, so we only fail on genuinely unusable output.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Recency = Literal["week", "month", "3months", "6months", "1year", "3years", "all"]
RECENCY_CHOICES: tuple[str, ...] = (
    "week", "month", "3months", "6months", "1year", "3years", "all",
)
RECENCY_LABELS: dict[str, str] = {
    "week": "Past week",
    "month": "Past month",
    "3months": "Past 3 months",
    "6months": "Past 6 months",
    "1year": "Past year",
    "3years": "Past 3 years",
    "all": "All time",
}

class RunParams(BaseModel):
    # 8000 chars comfortably holds a pasted multi-paragraph brief. The query
    # rides in the planner, triage, gap and synthesis prompts (notes get the
    # planner's distilled brief instead), so its cost is a few thousand
    # prefill tokens per run — nothing against a 64k context window.
    query: str = Field(min_length=3, max_length=8000)
    depth: int = Field(ge=0, le=10)
    recency: Recency = "all"
    origin: Literal["web", "telegram", "cli"] = "web"
    parent_run_id: str | None = None
    origin_chat_id: int | None = None
    evergreen: bool = False
    created_by: str = Field(default="", max_length=120)
    # SearXNG categories for this run, comma-separated. Empty = the global
    # SEARCH_CATEGORIES setting.
    categories: str = Field(default="", max_length=200)
    # False = ignore earlier runs entirely: no prior-knowledge block in the
    # planner and no "builds on" links. For re-diagnosing something from
    # scratch when previous conclusions might anchor the answer.
    use_prior: bool = True
    # "research" searches the web for an answer; "brief" ignores the query and
    # reads the configured feeds instead. Everything after search is identical.
    kind: Literal["research", "brief"] = "research"

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("query too short")
        return v


# ---- structured LLM outputs --------------------------------------------------

class PlannerOut(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    brief: str = ""
    subqueries: list[str] = Field(min_length=1, max_length=12)
    keywords: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("subqueries")
    @classmethod
    def _clean_queries(cls, v: list[str]) -> list[str]:
        cleaned = [q.strip() for q in v if q and q.strip()]
        if not cleaned:
            raise ValueError("no usable subqueries")
        return cleaned


class Fact(BaseModel):
    claim: str = Field(min_length=1, max_length=500)
    evidence_quote: str | None = None
    confidence: int = 5

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v):
        try:
            return max(0, min(10, int(float(v))))
        except (TypeError, ValueError):
            return 5

    @field_validator("evidence_quote", mode="before")
    @classmethod
    def _trim_quote(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        return v[:500] or None


class NotesOut(BaseModel):
    relevance: int = Field(ge=0, le=10)
    summary: str = ""
    notes_md: str = ""
    key_facts: list[Fact] = Field(default_factory=list, max_length=10)
    published_date: str | None = None

    @field_validator("key_facts", mode="before")
    @classmethod
    def _drop_unusable(cls, v):
        # One malformed fact must not cost us the whole document — same
        # tolerance FollowUpsOut already applies.
        if not isinstance(v, list):
            return []
        out = []
        for item in v:
            if isinstance(item, str) and item.strip():
                out.append({"claim": item.strip()})
            elif isinstance(item, dict) and str(item.get("claim", "")).strip():
                out.append(item)
        return out[:10]

    @field_validator("relevance", mode="before")
    @classmethod
    def _clamp_relevance(cls, v):
        try:
            return max(0, min(10, int(float(v))))
        except (TypeError, ValueError):
            return 0


class TriageOut(BaseModel):
    """Indices of search candidates NOT worth fetching.

    A drop-list, deliberately: when the model under-delivers (truncation,
    laziness) the failure mode is keeping extra junk — which relevance
    scoring catches — rather than silently discarding good candidates."""
    drop: list[int] = Field(default_factory=list, max_length=64)


class MatrixCell(BaseModel):
    entity: str = Field(default="", max_length=140)
    dimension: str = Field(default="", max_length=140)
    value: str = Field(default="", max_length=300)
    sources: list[int] = Field(default_factory=list, max_length=12)
    conflict: bool = False


class MatrixOut(BaseModel):
    """A comparison table recovered from a finished run's stored findings.

    `applicable` is the escape hatch: most runs are not comparisons, and the
    model is asked to say so rather than invent two things to put in columns.
    """
    applicable: bool = True
    reason: str = Field(default="", max_length=400)
    entities: list[str] = Field(default_factory=list, max_length=8)
    dimensions: list[str] = Field(default_factory=list, max_length=12)
    cells: list[MatrixCell] = Field(default_factory=list, max_length=160)
    caveats_md: str = ""


class GapOut(BaseModel):
    state_md: str = ""
    saturated: bool = False
    next_queries: list[str] = Field(default_factory=list, max_length=12)
    keywords: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("next_queries")
    @classmethod
    def _clean_queries(cls, v: list[str]) -> list[str]:
        return [q.strip() for q in v if q and q.strip()]


class FollowUp(BaseModel):
    query: str = Field(min_length=3)
    rationale: str = ""
    depth: int = 3
    recency: str = "6months"

    @field_validator("depth", mode="before")
    @classmethod
    def _clamp_depth(cls, v):
        try:
            return max(1, min(10, int(float(v))))
        except (TypeError, ValueError):
            return 3

    @field_validator("recency", mode="before")
    @classmethod
    def _coerce_recency(cls, v):
        return v if v in RECENCY_CHOICES else "6months"


class FollowUpsOut(BaseModel):
    items: list[FollowUp] = Field(default_factory=list, max_length=10)

    @field_validator("items", mode="before")
    @classmethod
    def _drop_unusable(cls, v):
        if not isinstance(v, list):
            return v
        return [i for i in v if isinstance(i, dict)
                and len(str(i.get("query", "")).strip()) >= 3][:10]
