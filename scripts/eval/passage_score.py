"""Score claim-check passage selection offline: does the judge see the evidence?

    PYTHONPATH=<code root> .venv/bin/python scripts/eval/passage_score.py [data dir]

For every eval claim (queries.json) whose source page was fetched
(pages.jsonl, from page_probe.py pages) and whose verified quote can be found
in that page, reports whether the quote is visible in what the judge is shown:
the page's first 1,200 chars (the old behaviour) versus verify.best_passage()
of whichever code is on PYTHONPATH. Host-side, no network, no model.
"""
import json
import re
import sys
from pathlib import Path

from app.research import verify
from app.research.notes import _norm

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "data/eval/2026-09-23")
RESEARCH = Path("data/research_data")
queries = json.loads((DATA / "queries.json").read_text())
pages = {}
for line in (DATA / "pages.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r.get("text"):
        pages[r["url"]] = r["text"]


def quote_for(q):
    """The verbatim quote stored under the fact the query was written from."""
    run_dir = RESEARCH / q["run"] / "findings"
    for p in run_dir.glob("*.md"):
        md = p.read_text(encoding="utf-8")
        at = md.find(q["fact"])
        if at >= 0:
            m = re.search(r'^\s+> "?(.+?)"?\s*$', md[at:], re.M)
            return m.group(1) if m else None
    return None


have = head = best = 0
best_fn = getattr(verify, "best_passage", None)
for q in queries:
    text, quote = pages.get(q["url"]), quote_for(q)
    if not text or not quote or len(quote) < 20:
        continue
    key = _norm(quote)[:80]
    if key not in _norm(text):
        continue  # the page changed since the run read it
    have += 1
    head += key in _norm(text[:verify._EVIDENCE_CHARS])
    if best_fn:
        best += key in _norm(best_fn(q["claim"], text))
print(f"claims whose verified quote is still on the fetched page: {have}")
print(f"  judge sees it with the first {verify._EVIDENCE_CHARS:,} chars: {head} ({head / have:.0%})")
if best_fn:
    print(f"  judge sees it with best_passage():          {best} ({best / have:.0%})")
else:
    print("  (this code has no verify.best_passage)")
