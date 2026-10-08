"""Which parts of a long page survive a cut: keyword windows or a reranker?

For every fact in the eval set whose verified quote is still on its page, cut
the page the way notes do (to a share of its length under memory pressure, or
to the 56K budget when it is oversized) and count how often the quote
survives:
  keywords  notes._shrink: keyword windows, then the page's opening
  reranker  notes._rerank_cut: the pieces the reranker scores best against
            the run's brief, in page order
  head      the first N chars, for reference
Runs do not store the planner's keyword list, so keywords here are the most
common content words of the run's planned sub-queries: an approximation.

Runs in the dev image with the code under test mounted and the data read-only:
  docker run --rm -i -v $PWD/app:/srv/app/app:ro -v $PWD/data:/data:ro \\
    -e DATA_DIR=/data -w /srv/app mkw-app-dev:latest \\
    python - [eval dir] [rerank model] < scripts/eval/passage_pick.py
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

from app.config import load_settings
from app.rag.rerank import Reranker
from app.research.dedupe import _content_tokens
from app.research.notes import _INPUT_CHARS, _norm, _rerank_cut, _shrink

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/eval/2026-09-23")
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen3-Reranker-0.6B-mxfp8"
RESEARCH = Path("/data/research_data")
SHARES = (0.5, 0.25)
MIN_PAGE = 10_000            # shorter pages are rarely cut at all


def quote_for(q: dict) -> str | None:
    """The verbatim quote stored under the fact the query was written from."""
    run_dir = RESEARCH / q["run"] / "findings"
    for p in run_dir.glob("*.md"):
        md = p.read_text(encoding="utf-8")
        at = md.find(q["fact"])
        if at >= 0:
            m = re.search(r'^\s+> "?(.+?)"?\s*$', md[at:], re.M)
            return m.group(1) if m else None
    return None


def brief_and_keywords(run: str) -> tuple[str, list[str]]:
    d = RESEARCH / run
    meta = json.loads((d / "meta.json").read_text())
    subqueries: list[str] = []
    events = d / "events.jsonl"
    if events.is_file():
        for line in events.read_text().splitlines():
            if '"type": "plan"' in line:
                plan = json.loads(line)
                subqueries = list(plan.get("subqueries") or [])
                break
    counts = Counter(t for s in subqueries or [meta.get("query", "")]
                     for t in _content_tokens(s))
    return meta.get("brief") or meta.get("query", ""), [w for w, _ in counts.most_common(8)]


async def main() -> None:
    cfg = load_settings()
    base = (cfg.embedding_base_url or cfg.resolved_base_url).rstrip("/")
    key = cfg.embedding_api_key or cfg.resolved_api_key
    reranker = Reranker(base, MODEL, key)
    queries = json.loads((DATA / "queries.json").read_text())
    pages = {}
    for line in (DATA / "pages.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("text"):
            pages[r["url"]] = r["text"]

    cases, seen = [], set()
    for q in queries:
        text, quote = pages.get(q["url"]), quote_for(q)
        if not text or len(text) < MIN_PAGE or not quote or len(quote) < 20:
            continue
        key80 = _norm(quote)[:80]
        if key80 not in _norm(text) or (q["url"], key80) in seen:
            continue
        seen.add((q["url"], key80))
        brief, keywords = brief_and_keywords(q["run"])
        cases.append((q["url"], text, key80, brief, keywords))
    print(f"{len(cases)} verified quotes on pages of {MIN_PAGE:,}+ chars "
          f"({len({c[0] for c in cases})} pages)", flush=True)

    rows: dict[str, Counter] = {}
    secs: list[float] = []
    for url, text, key80, brief, keywords in cases:
        cuts = [(f"{int(s * 100)}% of the page", int(len(text) * s)) for s in SHARES]
        if len(text) > _INPUT_CHARS:
            cuts.append(("oversized, to 56K", _INPUT_CHARS))
        for label, budget in cuts:
            c = rows.setdefault(label, Counter())
            c["cases"] += 1
            c["head"] += key80 in _norm(text[:budget])
            c["keywords"] += key80 in _norm(_shrink(text, keywords, budget))
            t = time.monotonic()
            picked = await _rerank_cut(text, brief, budget, reranker)
            secs.append(time.monotonic() - t)
            if picked is None:
                c["rerank_failed"] += 1
            c["reranker"] += key80 in _norm(picked or "")
    print(f"\n{'cut':22} {'cases':>5} {'head':>6} {'keywords':>9} {'reranker':>9}")
    for label, c in rows.items():
        n = c["cases"]
        print(f"{label:22} {n:5} {c['head'] / n:6.0%} {c['keywords'] / n:9.0%} "
              f"{c['reranker'] / n:9.0%}" + (f"   ({c['rerank_failed']} rerank failures)"
                                             if c["rerank_failed"] else ""))
    secs.sort()
    if secs:
        print(f"\nrerank seconds per cut: median {secs[len(secs) // 2]:.1f}, "
              f"p90 {secs[int(len(secs) * 0.9)]:.1f}")


asyncio.run(main())
