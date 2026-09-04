"""The question's facets: the distinct things one prompt asks for.

A long prompt asks for many things at once. The planner used to compress it
into a 2-4 sentence brief, and everything downstream ran off that brief —
triage, relevance scoring, gap analysis, and every later round's queries. A
facet the brief dropped could not be searched for, and a page about it would
have scored low if it had turned up anyway.

Measured on 2026-09-04: a prompt listing eleven deliverables produced 27
queries, 26 of them naming the same vendor. Four of the deliverables were
never searched at all, and the report still had a section for each, written
from the model's own prior knowledge over citations borrowed from unrelated
vendor documentation. The planner prompt already said to spread queries
across facets; wording lost, as it has here before.

Facets make the deliverables data rather than prose, so the round loop can
count what each has produced and spend its slots on what is thin.
"""
from __future__ import annotations

from collections import Counter

MAX_FACETS = 12
# Words that carry no search signal in a facet name.
_FILLER = frozenset(
    "the a an of for to in on at by and or with vs versus how what which any "
    "all other others etc anything everything else including but not limited "
    "it its this that these those they them is are be can could should would "
    "do does our we my your".split())


def normalize(name: str) -> str:
    """A facet's key: what the planner and gap stages must agree on."""
    return " ".join(str(name or "").strip().lower().replace("_", " ").split())[:60]


def clean_facets(names: list[str]) -> list[str]:
    """Normalized, de-duplicated, capped. Order is the planner's own."""
    out: list[str] = []
    for n in names or []:
        k = normalize(n)
        if k and k not in out:
            out.append(k)
    return out[:MAX_FACETS]


def _closest(tag: str, known: list[str]) -> str:
    """A tag naming no known facet, matched by shared words. Models rename
    their own facets between stages ("commercial terms" → "commercial"),
    and dropping those tags would silently zero a facet's coverage."""
    words = set(tag.split())
    best, score = "", 0
    for k in known:
        overlap = len(words & set(k.split()))
        if overlap > score:
            best, score = k, overlap
    return best if score else ""


def align(facets: list[str], queries: list[str], tags: list[str]) -> dict[str, str]:
    """query text → facet key. An untagged or unmatchable query is not an
    error; it simply counts toward no facet."""
    known = clean_facets(facets)
    tags = list(tags or [])
    out: dict[str, str] = {}
    for i, q in enumerate(queries or []):
        tag = normalize(tags[i]) if i < len(tags) else ""
        if not tag:
            continue
        out[q] = tag if tag in known else _closest(tag, known)
        if not out[q]:
            del out[q]
    return out


def per_facet_cap(breadth: int) -> int:
    """The most slots one facet may take while another facet has nothing:
    a third of the round, rounded up. Same rule as the per-engine share —
    a round spent entirely on the best-covered facet is the failure."""
    return max(1, -(-breadth // 3))


def allocate(proposed: list[str], facet_of: dict[str, str], kept: Counter,
             facets: list[str], breadth: int) -> tuple[list[str], list[str], list[str]]:
    """Choose this round's queries from what the planner or gap proposed.

    Returns (chosen, starved, leftover):
      chosen   — up to `breadth` queries, round-robin over facets ordered by
                 how few sources each has produced, capped per facet.
      starved  — facets with no sources and no query this round, in need of
                 one (the caller tops up; see facet_query).
      leftover — proposed queries the cap displaced, to fill any slot the
                 top-up does not use, so nothing is wasted.
    """
    keys = clean_facets(facets)
    proposed = list(proposed or [])
    if not keys or breadth <= 0:
        return proposed[:max(breadth, 0)], [], proposed[max(breadth, 0):]

    buckets: dict[str, list[str]] = {k: [] for k in keys}
    loose: list[str] = []
    for q in proposed:
        f = facet_of.get(q, "")
        (buckets[f] if f in buckets else loose).append(q)

    # Thinnest facet first; the planner's own order breaks ties.
    order = sorted(keys, key=lambda k: (kept.get(k, 0), keys.index(k)))
    cap = per_facet_cap(breadth) if len(keys) > 1 else breadth
    chosen: list[str] = []
    taken: Counter = Counter()
    progress = True
    while len(chosen) < breadth and progress:
        progress = False
        for k in order:
            if len(chosen) >= breadth:
                break
            if taken[k] >= cap or not buckets[k]:
                continue
            chosen.append(buckets[k].pop(0))
            taken[k] += 1
            progress = True

    starved = [k for k in order if not kept.get(k, 0) and not taken[k]]
    # Hold a slot for each starved facet so the caller can put a query on it,
    # then let the untagged queries have what is left. An untagged query
    # belongs to no facet, so running it cannot breach the cap — but a query
    # the cap displaced can, and those only return if nothing is starved.
    reserve = min(len(starved), max(0, breadth - len(chosen)))
    room = max(0, breadth - len(chosen) - reserve)
    chosen += loose[:room]
    displaced = [q for k in order for q in buckets[k]]
    return chosen, starved, loose[room:] + displaced


def facet_query(facet: str, subject: str = "") -> str:
    """A search query for a facet nothing has covered yet.

    Deliberately plain: an ordinary query on an unanswered part of the
    question beats a well-crafted one on the part that already has twenty
    sources. The subject is prepended because an unanchored query about a
    sub-topic returns pages about the sub-topic in general.
    """
    words = [w for w in normalize(facet).split() if w not in _FILLER]
    subj = " ".join(w for w in str(subject or "").split()[:3])
    parts = [w for w in ([subj] if subj else []) + words if w]
    return " ".join(parts)[:120].strip()


def subject_terms(title: str, keywords: list[str]) -> str:
    """The one or two words every top-up query should be anchored to."""
    for k in keywords or []:
        toks = str(k).split()
        if 1 <= len(toks) <= 2 and len(str(k)) > 2:
            return str(k)
    return " ".join(str(title or "").split()[:2])


def coverage_lines(facets: list[str], kept: Counter) -> str:
    """The table gap analysis is shown: what each facet has produced."""
    keys = clean_facets(facets)
    if not keys:
        return ""
    return "\n".join(f"- {k}: {kept.get(k, 0)}" for k in keys)


def uncovered(facets: list[str], kept: Counter) -> list[str]:
    """Facets the run produced no source for — what synthesis must not
    write a section about, and what the reader has to be told."""
    return [k for k in clean_facets(facets) if not kept.get(k, 0)]
