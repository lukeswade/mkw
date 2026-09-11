"""Pydantic models: run parameters and every structured LLM output.

LLM output models are deliberately forgiving (coercing validators instead of
hard Literals where a local model might improvise) — a parse failure costs a
repair round-trip, so we only fail on genuinely unusable output.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Recency = Literal["week", "month", "3months", "6months", "1year", "3years", "all"]

# A brief needs no question, but a run needs a title and the synthesis stage
# needs something to orient on. When the query IS this, there is no topic
# filter — the reader asked for everything their feeds published.
BRIEF_DEFAULT_QUERY = "Brief: what is new across my feeds"

# Verification is bounded: a long document can carry dozens of claims, and
# each unresolved one costs a web sub-run. The cap is reported in the output
# rather than applied silently.
VERIFY_CLAIM_CAP = 15

# What each run kind is called in the UI. "verify" is the internal name;
# "claim check" is what it does.
KIND_LABEL = {"research": "research", "brief": "brief", "verify": "claim check"}
KIND_HELP = {
    "research": "A question researched across the web over several rounds",
    "brief": "New items pulled from your configured feeds",
    "verify": "A pasted document's claims checked against your library and the web",
}
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
    kind: Literal["research", "brief", "verify"] = "research"
    # Which saved brief a brief run belongs to. None means the unnamed brief
    # built from the global FEEDS setting.
    brief_id: int | None = None
    # The text under verification. Held on disk beside the run rather than in
    # `query`, which rides in every prompt and is capped far below an article.
    document: str = Field(default="", max_length=120_000)

    @field_validator("query")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 3:
            raise ValueError("query too short")
        return v


# ---- structured LLM outputs --------------------------------------------------

def pair_parallel(data, fields: tuple[str, ...], require: tuple[str, ...]):
    """Filter index-aligned lists as tuples, before per-field validation.

    Three models emit a list of queries plus lists of facet tags and scopes
    aligned to it BY INDEX, and every field was cleaned on its own criteria:
    the query cleaner dropped blanks, the tag cleaner kept them. One blank
    query therefore shifted every later tag a slot left, so a round credited
    its sources to the wrong part of the question and searched them in the
    wrong scope. Two independent reviewers found this in the same place
    (2026-09-09).

    Dropping a whole tuple keeps the pairing; padding a short list with "" is
    safe because facets.align() falls back to matching the query text. The
    drift is simply not representable after this runs.
    """
    if not isinstance(data, dict):
        return data
    cols = {f: data[f] for f in fields
            if isinstance(data.get(f), list)}
    if not cols:
        return data
    n = max(len(v) for v in cols.values())

    def at(field, i):
        v = cols.get(field) or []
        return v[i] if i < len(v) else None

    keep = [i for i in range(n)
            if all(at(r, i) is not None and str(at(r, i)).strip()
                   for r in require)]
    out = dict(data)
    for f in cols:
        out[f] = [at(f, i) if at(f, i) is not None else "" for i in keep]
    return out


class PlannerOut(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    # The distinct things the question asks for. Data, not prose: the round
    # loop counts sources per facet and spends its slots on the thin ones
    # (research/facets.py). Optional — a model that omits them gets the old
    # single-thread behaviour.
    facets: list[str] = Field(default_factory=list, max_length=12)
    brief: str = ""
    subqueries: list[str] = Field(min_length=1, max_length=12)
    # One facet per subquery, aligned by index. Optional and tolerant.
    query_facets: list[str] = Field(default_factory=list, max_length=12)
    keywords: list[str] = Field(default_factory=list, max_length=20)
    # One scope per subquery ("web", "web+video", "code", ...), aligned by
    # index. Optional: a model that omits it gets the run's full categories.
    query_scopes: list[str] = Field(default_factory=list, max_length=12)
    # Assertions the question takes for granted that a PUBLISHED standard,
    # benchmark or consensus figure could settle. A U8 coach writing "the
    # fields we play on are way too small" states a fact that US Soccer has
    # numbers for; a run that never checks it builds five sections on an
    # assumption. Most questions carry none, and two is the ceiling — a run
    # that spends its breadth interrogating the premise stops answering the
    # question.
    premises: list[str] = Field(default_factory=list, max_length=2)
    premise_queries: list[str] = Field(default_factory=list, max_length=2)
    # Instructions about the SHAPE of the answer, not its subject: "include a
    # comparison table", "advantages and limitations for each", "keep it
    # short". 2026-09-11: a question asking for a comprehensive comparison
    # table got a good document with no table in it, because the brief — the
    # only thing that reaches synthesis — describes what to research and has
    # never carried what to produce. These travel on their own channel,
    # straight to synthesis, and are deliberately NOT facets: a run that
    # searches for "comparison table" wastes a query and then reports the
    # table as an unresearched part of the question.
    deliverables: list[str] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def _pair_premises(self):
        """A premise without a query cannot be checked, so it is not kept."""
        pairs = [(p, q) for p, q in zip(self.premises, self.premise_queries)
                 if p.strip() and q.strip()][:2]
        self.premises = [p.strip() for p, _ in pairs]
        self.premise_queries = [q.strip() for _, q in pairs]
        return self

    @model_validator(mode="before")
    @classmethod
    def _align_queries(cls, data):
        return pair_parallel(
            data, ("subqueries", "query_facets", "query_scopes"),
            ("subqueries",))

    @field_validator("subqueries")
    @classmethod
    def _clean_queries(cls, v: list[str]) -> list[str]:
        cleaned = [q.strip() for q in v if q and q.strip()]
        if not cleaned:
            raise ValueError("no usable subqueries")
        return cleaned

    @field_validator("query_scopes", mode="before")
    @classmethod
    def _clean_scopes(cls, v):
        return [str(x).strip().lower() for x in (v or []) if x is not None]

    @field_validator("premises", "premise_queries", mode="before")
    @classmethod
    def _clean_premises(cls, v):
        if not isinstance(v, list):
            return []
        return [str(x).strip() for x in v if str(x or "").strip()][:2]

    @field_validator("deliverables", mode="before")
    @classmethod
    def _clean_deliverables(cls, v):
        """Bound in code, not in the prompt. One runaway instruction must not
        be able to rewrite the synthesis prompt, so each is clamped to a
        sentence's worth and the list to four."""
        if not isinstance(v, list):
            return []
        return [str(x).strip()[:200] for x in v if str(x or "").strip()][:4]

    @field_validator("facets", "query_facets", mode="before")
    @classmethod
    def _clean_facets(cls, v):
        return [str(x).strip().lower() for x in (v or []) if x is not None][:12]


class FacetQueriesOut(BaseModel):
    """One search query per unanswered part of the question."""
    facets: list[str] = Field(default_factory=list, max_length=14)
    queries: list[str] = Field(default_factory=list, max_length=14)
    scopes: list[str] = Field(default_factory=list, max_length=14)

    @model_validator(mode="before")
    @classmethod
    def _align_queries(cls, data):
        return pair_parallel(data, ("facets", "queries", "scopes"),
                             ("facets", "queries"))

    @field_validator("facets", "queries", "scopes", mode="before")
    @classmethod
    def _clean(cls, v):
        # pair_parallel has already dropped the unusable tuples; this only
        # tidies what survived, so it must NOT change the list's length.
        return [str(x).strip() for x in (v or []) if x is not None][:14]


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


# Source ranks, best first. Synthesis follows the better rank when sources
# disagree, and a part's strongest source leads its digest. An unrecognised or
# missing value sorts with "practitioner" so an old run, or a model that skips
# the field, is neither promoted nor demoted.
SOURCE_TIERS: dict[str, int] = {
    "standard": 0,      # governing body, spec, law, manufacturer documentation
    "research": 1,      # peer-reviewed or formal study
    "practitioner": 2,  # a professional writing from direct experience
    "aggregator": 3,    # listicle, roundup, SEO content, forum thread
}
UNRANKED_TIER = SOURCE_TIERS["practitioner"]


def source_rank(source_type: str) -> int:
    return SOURCE_TIERS.get((source_type or "").strip().lower(), UNRANKED_TIER)


class NotesOut(BaseModel):
    relevance: int = Field(ge=0, le=10)
    summary: str = ""
    notes_md: str = ""
    key_facts: list[Fact] = Field(default_factory=list, max_length=10)
    published_date: str | None = None
    # How much the source contributes to the ONE part of the question it was
    # fetched for (research/facets.credits). Asked only when a part is named;
    # None otherwise, and the lexical rule decides.
    part_relevance: int | None = None
    # What KIND of source this is, so synthesis can tell a governing body from
    # a drill blog. Free text from a model is useless for ordering, so it is
    # clamped to the four ranks in SOURCE_TIERS; anything else becomes "",
    # which sorts neutrally rather than promoting an unparseable answer.
    source_type: str = ""
    # The organization whose OWN rules this page publishes. A retailer or a
    # "field dimensions guide" site restating a governing body's numbers is
    # reporting a standard, not publishing one, and the note-taker called ten
    # such pages "standard" in one run while the real US Soccer document
    # ranked among them and went uncited (2026-09-09). No publisher, no
    # standard: _demote_unsourced_standard enforces that in code.
    publisher: str = ""

    @field_validator("publisher", mode="before")
    @classmethod
    def _clean_publisher(cls, v):
        return str(v or "").strip()[:120]

    @model_validator(mode="after")
    def _demote_unsourced_standard(self):
        if self.source_type == "standard" and not self.publisher:
            self.source_type = "aggregator"
        return self

    @field_validator("source_type", mode="before")
    @classmethod
    def _clean_source_type(cls, v):
        t = str(v or "").strip().lower()
        return t if t in SOURCE_TIERS else ""

    @field_validator("part_relevance", mode="before")
    @classmethod
    def _clamp_part(cls, v):
        if v is None or v == "":
            return None
        try:
            return max(0, min(10, int(float(v))))
        except (TypeError, ValueError):
            return None

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


class Claim(BaseModel):
    text: str = Field(default="", max_length=600)
    # How much the document leans on this claim. Verification is capped, so
    # this decides what gets checked and what is reported as unchecked.
    importance: int = Field(default=5, ge=0, le=10)
    checkable: bool = True


class ClaimsOut(BaseModel):
    claims: list[Claim] = Field(default_factory=list, max_length=80)


class VerdictOut(BaseModel):
    verdict: Literal["supported", "contested", "unsupported",
                     "unverifiable"] = "unverifiable"
    confidence: int = Field(default=0, ge=0, le=10)
    reasoning: str = Field(default="", max_length=900)
    quote: str = Field(default="", max_length=400)
    sources: list[int] = Field(default_factory=list, max_length=10)


class BriefFilterOut(BaseModel):
    keep: list[int] = Field(default_factory=list, max_length=400)


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
    next_query_scopes: list[str] = Field(default_factory=list, max_length=12)
    # Which facet of the question each next query attacks (research/facets.py).
    next_query_facets: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="before")
    @classmethod
    def _align_queries(cls, data):
        return pair_parallel(
            data, ("next_queries", "next_query_facets", "next_query_scopes"),
            ("next_queries",))

    @field_validator("next_query_facets", "next_query_scopes", mode="before")
    @classmethod
    def _clean_facet_tags(cls, v):
        return [str(x).strip().lower() for x in (v or []) if x is not None][:12]

    @field_validator("next_queries")
    @classmethod
    def _clean_queries(cls, v: list[str]) -> list[str]:
        return [q.strip() for q in v if q and q.strip()]

    @field_validator("next_query_scopes", mode="before")
    @classmethod
    def _clean_scopes(cls, v):
        return [str(x).strip().lower() for x in (v or []) if x is not None]


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
