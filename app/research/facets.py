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

import re
from collections import Counter

# A brief that lists eleven asks gets eleven facets plus room for what the
# planner adds. Thin per facet at depth 10 (35 slots) and thinner below it —
# which is the honest signal: the run then says what it could not reach.
MAX_FACETS = 14
# Words that carry no search signal in a facet name.
_FILLER = frozenset(
    "the a an of for to in on at by and or with vs versus how what which any "
    "all other others etc anything everything else including but not limited "
    "it its this that these those they them is are be can could should would "
    "do does our we my your".split())


def normalize(name: str) -> str:
    """A facet's key: what the planner and gap stages must agree on."""
    text = " ".join(str(name or "").strip().lower().replace("_", " ").split())
    if len(text) <= 110:
        return text
    return text[:110].rsplit(" ", 1)[0] or text[:110]


# A question that numbers or lists its asks has already done the
# decomposition; reading it off the text is not a judgement call. The planner
# folded four numbered use cases into one facet called "agent template
# strategy" (2026-09-04), and nothing downstream could recover them — the
# machinery allocated perfectly across the facets it was given. So the list
# in the question is taken from the question, not from the model.
_MARKED_LINE = re.compile(r"(?m)^[ \t]*(?:[-*\u2022\u2013]|\d{1,2}[.)])[ \t]+(.{3,300}?)[ \t]*$")
_MAX_ITEM_CHARS = 300
_NAME_WORDS = 12


# Longest first. Enough that "comparison", "compares" and "comparing" meet:
# the question asked about "competing platforms" and the planner called the
# same thing "competitor landscape comparison", the two shared one stemmed
# word, so both survived the merge and one was reported as an unanswered ask
# while the other held ten sources.
_SUFFIXES = ("ations", "ation", "ments", "ment", "ities", "ity", "ison",
             "ings", "ing", "ies", "ers", "es", "ed", "er", "s")


def _stem(token: str) -> str:
    """Crude morphology: enough for two names of one ask to meet."""
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: len(token) - len(suffix)]
    return token


def content_words(text: str) -> frozenset[str]:
    """The words that carry a facet's meaning, split on punctuation so
    "organization/contact" counts as two, and stemmed so plurals meet."""
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return frozenset(_stem(t) for t in tokens if len(t) >= 2 and t not in _FILLER)


def about(facet: str, text: str) -> bool:
    """Is this source about the facet, or did the facet's query merely turn
    it up?

    Credit used to follow attribution alone, and five vendor-anchored
    use-case queries ("Workato AIRO call prep") each kept a generic vendor
    page — an overview, a keynote, a think-piece — none of them about the use
    case. Every facet then looked answered and the report said nothing about
    what it had missed. Two of the facet's own words must appear in the
    source, and the error is deliberately one-sided: an on-topic source that
    words it differently is reported as a gap, which is the safe way to be
    wrong.
    """
    wanted = content_words(facet)
    if not wanted:
        return True
    return len(wanted & content_words(text)) >= (2 if len(wanted) >= 2 else 1)


def credits(facet: str, text: str, part_relevance: int | None, threshold: int) -> bool:
    """Does a kept source count as answering the part it was fetched for?

    The lexical rule alone (two of the facet's words in the source) reported
    "health checks" unanswered while three customer-health-SCORING guides sat
    in the findings: the reader's word and the field's word differ. So when
    the note-taker was asked how much the page contributes to that part, its
    answer decides — gated by ONE shared word, because the same note-taker
    scored a vendor launch press release 6 under the risk part, and that page
    shares no word with "customer dispute/conflict/risk identification": the
    floor is what vetoes it. Without an answer, the two-word rule stands.
    """
    if part_relevance is None:
        return about(facet, text)
    wanted = content_words(facet)
    if not wanted:
        return part_relevance >= threshold
    return part_relevance >= threshold and bool(wanted & content_words(text))


def _facet_name(item: str) -> str:
    """A list item, reduced to the words that carry its meaning. Empty for a
    catch-all ("anything and everything else"): every word of it is filler,
    it names no research, and it would sit unanswered in every report."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9/&.'+-]*", str(item or ""))
    content = [w for w in words if w.lower() not in _FILLER]
    return normalize(" ".join(content[:_NAME_WORDS])) if content else ""


def enumerated(question: str) -> list[str]:
    """The asks the question itself lists.

    Two shapes, both of which a person writing a brief actually uses: lines
    marked with a bullet or a number, and the plain lines that follow a line
    ending in a colon. A colon block needs three lines before it counts, so
    ordinary prose after a colon is not mistaken for a list.
    """
    text = str(question or "").replace("\r\n", "\n").replace("\r", "\n")
    items = [m.group(1).strip() for m in _MARKED_LINE.finditer(text)]
    if len(items) < 2:
        items = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if not line.rstrip().endswith(":"):
            continue
        block: list[str] = []
        for nxt in lines[i + 1:]:
            stripped = nxt.strip()
            if not stripped:
                if block:
                    break
                continue
            if len(stripped) > _MAX_ITEM_CHARS or _MARKED_LINE.match(nxt):
                break
            block.append(stripped)
            if len(block) >= MAX_FACETS:
                break
        if len(block) >= 3:
            items += block
    out: list[str] = []
    for item in items:
        name = _facet_name(item)
        if name and name not in out:
            out.append(name)
    return out[:MAX_FACETS]


def _same_facet(a: str, b: str, common: frozenset[str] = frozenset()) -> bool:
    """Two names for one ask. Two shared content words is the test: 'call
    prep' and 'call prep agent' are the same facet, 'workato vs competitors'
    and 'workato oem pricing' are not. A short name that shares one word is
    the same ask too, when that word is distinctive for this question — the
    planner's "template specialization" against the reader's "recommended
    agents ... as a template ... distributed per customer" — but never on a
    word the whole facet list uses, like the product's own name."""
    aw, bw = content_words(a), content_words(b)
    if not aw or not bw:
        return a == b
    shared = aw & bw
    if len(shared) >= 2:
        return True
    return bool(shared - common) and min(len(aw), len(bw)) <= 2


def merge(model_facets: list[str], asked: list[str]) -> list[str]:
    """The question's own list first — it is literally what was asked — then
    whatever else the planner named that is not the same thing again."""
    asked_c, model_c = clean_facets(asked), clean_facets(model_facets)
    freq: Counter = Counter()
    for name in asked_c + model_c:
        freq.update(content_words(name))
    # A word in three or more facets is this question's furniture, not a
    # signal that two facets are one.
    common = frozenset(w for w, n in freq.items() if n >= 3)
    out = list(asked_c)
    for m in model_c:
        if not any(_same_facet(m, existing, common) for existing in out):
            out.append(m)
    return out[:MAX_FACETS]


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
    """query text → facet key.

    An untagged query used to count toward NO facet, which silently
    under-credited every round after the first: the gap stage often returns
    next_queries with no next_query_facets at all, and then nothing its
    sources answer is ever credited. 2026-09-09: six sources answering "small
    field tactics" were credited to nothing, the run reported that facet
    unresearched directly under a section built from them, and round two spent
    its slots re-attacking a facet that was already covered.

    So when the tag is missing or unmatchable, fall back to what the QUERY is
    about. That needs two shared stems (facet_for_query), which keeps it quiet
    when a query genuinely matches no facet — an untagged query still counts
    toward nothing rather than being forced somewhere."""
    known = clean_facets(facets)
    tags = list(tags or [])
    out: dict[str, str] = {}
    for i, q in enumerate(queries or []):
        tag = normalize(tags[i]) if i < len(tags) else ""
        hit = (tag if tag in known else _closest(tag, known)) if tag else ""
        if not hit:
            hit = facet_for_query(q, known)
        if hit:
            out[q] = hit
    return out


def facet_for_query(query: str, facets: list[str]) -> str:
    """The facet a free-text query is really about, by shared stems.

    A premise query and a facet name the same ground in different words: "US
    Youth Soccer U8 4v4 field dimensions guidelines" against the facet "field
    dimension standards". Without this the premise query's ten sources were
    credited to the premise label, the facet ended on zero, and the document
    closed by claiming it had not researched field dimensions — under a
    section that cited six sources about them (2026-09-09).

    Two shared stems, the same bar merge() uses between two facet names, and
    deliberately stricter than one: claiming a facet was covered when it was
    never really searched is a worse failure than reporting it uncovered.
    """
    words = content_words(query)
    if not words:
        return ""
    best, score = "", 1              # anything kept must beat 1, so >= 2
    for f in clean_facets(facets):
        overlap = len(words & content_words(f))
        if overlap > score:
            best, score = f, overlap
    return best


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
    words = [w for w in re.findall(r"[A-Za-z0-9]+", normalize(facet))
             if w.lower() not in _FILLER]
    subj = " ".join(w for w in str(subject or "").split()[:3])
    parts = [w for w in ([subj] if subj else []) + words if w]
    return " ".join(parts)[:120].strip()


def top_up_queries(facet: str, subject: str, from_question: bool) -> list[str]:
    """Queries to try for a facet with no sources, best first.

    A facet the question itself listed is a thing the reader wants
    understood, not a property of the product, and anchoring it to the
    vendor asks the wrong question: "Workato AIRO call prep" returned the
    vendor's overview page and nothing about preparing for a call. Those go
    out plain first and anchored only as the second attempt. A facet the
    planner invented describes the subject, so it keeps the anchor.
    """
    plain, anchored = facet_query(facet, ""), facet_query(facet, subject)
    order = [plain, anchored] if from_question else [anchored, plain]
    out: list[str] = []
    for q in order:
        if q and q not in out:
            out.append(q)
    return out


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
