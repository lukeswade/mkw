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

ENTITY_TYPES = (
    "person", "org", "technology", "concept", "place", "event", "product", "other",
)


class RunParams(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    depth: int = Field(ge=1, le=10)
    recency: Recency = "all"
    origin: Literal["web", "telegram", "cli"] = "web"
    parent_run_id: str | None = None
    origin_chat_id: int | None = None

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

    @field_validator("subqueries")
    @classmethod
    def _clean_queries(cls, v: list[str]) -> list[str]:
        cleaned = [q.strip() for q in v if q and q.strip()]
        if not cleaned:
            raise ValueError("no usable subqueries")
        return cleaned


class NotesOut(BaseModel):
    relevance: int = Field(ge=0, le=10)
    summary: str = ""
    notes_md: str = ""
    key_facts: list[str] = Field(default_factory=list, max_length=10)
    published_date: str | None = None

    @field_validator("relevance", mode="before")
    @classmethod
    def _clamp_relevance(cls, v):
        try:
            return max(0, min(10, int(float(v))))
        except (TypeError, ValueError):
            return 0


class GapOut(BaseModel):
    state_md: str = ""
    saturated: bool = False
    next_queries: list[str] = Field(default_factory=list, max_length=12)

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


class EntityItem(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    type: str = "other"
    salience: float = 0.5
    description: str = ""

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_type(cls, v):
        v = str(v or "").strip().lower()
        if v in {"organization", "organisation", "company"}:
            return "org"
        return v if v in ENTITY_TYPES else "other"

    @field_validator("salience", mode="before")
    @classmethod
    def _clamp_salience(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.5

    @field_validator("description")
    @classmethod
    def _trim_desc(cls, v: str) -> str:
        return v[:300]


class EntitiesOut(BaseModel):
    entities: list[EntityItem] = Field(default_factory=list, max_length=20)
