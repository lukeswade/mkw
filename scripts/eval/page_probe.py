"""Page-level probes, fetched exactly the way a research run reads a source.

    docker compose exec -T app python - <mode> [--data DIR] < scripts/eval/page_probe.py

Modes
  lengths  re-fetch what the 14 most recent runs kept; report extracted text
           length against the notes budget (56K chars) and the extractor's
           80K head cut. 2026-09-23: 19/401 over 56K (5%), 8 at the 80K cut.
  quotes   locate verified evidence quotes in their pages; report how many sit
           inside the 1,200 chars a claim-check judge is shown of each web page.
           2026-09-23: 152/341 (45%); median quote position 1,383 chars.
  pages    fetch the source page of every query in <data>/queries.json and
           write <data>/pages.jsonl, so passage selection can be scored
           offline against any version of the code (scripts/eval/passage_score.py).

Network only, no model calls; four fetches at a time.
"""
import asyncio, json, re, sqlite3, statistics as st, sys
from pathlib import Path

import httpx

from app.config import load_settings
from app.research import reddit, youtube
from app.research.extractor import extract
from app.research.fetcher import Fetcher, SkipReason
from app.research.notes import _norm
from app.research.pipeline import _browser_headers

WINDOW = 1_200   # verify._EVIDENCE_CHARS
NOTES_BUDGET, EXTRACT_CAP = 56_000, 80_000
MODE = sys.argv[1] if len(sys.argv) > 1 else "lengths"
DATA = Path(sys.argv[sys.argv.index("--data") + 1]) if "--data" in sys.argv else Path("/data/eval/2026-09-23")
cfg = load_settings()
con = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True, timeout=10)
con.row_factory = sqlite3.Row
_QUOTE = re.compile(r'^\s+> "?(.+?)"?\s*$', re.M)


def recent_runs(n=14):
    return [r["id"] for r in con.execute(
        "SELECT id FROM runs WHERE status='completed' AND depth >= 2 AND kind IS NOT 'verify' "
        "ORDER BY created_at DESC LIMIT ?", (n,))]


async def read(fetcher, url):
    if vid := youtube.video_id(url):
        return await youtube.transcript(fetcher.client, vid), "video"
    if reddit.is_thread(url):
        doc, _ = await reddit.thread(fetcher, url)
        return doc, "reddit"
    fetched = await fetcher.fetch(url)
    return extract(fetched), ("pdf" if fetched.content_type == "application/pdf" else "html")


async def fetch_all(urls, fn):
    out = []
    async with httpx.AsyncClient(headers=_browser_headers(cfg),
                                 timeout=httpx.Timeout(15.0, connect=10.0)) as http:
        fetcher, sem = Fetcher(cfg, http), asyncio.Semaphore(4)

        async def one(u):
            async with sem:
                try:
                    doc, kind = await read(fetcher, u)
                    out.extend(fn(u, doc, kind) if doc else [{"url": u, "error": "no text"}])
                except (SkipReason, Exception) as e:
                    out.append({"url": u, "error": str(e)[:80]})
        tasks = [asyncio.create_task(one(u)) for u in urls]
        done, pending = await asyncio.wait(tasks, timeout=900)
        for t in pending:
            t.cancel()
    return out


def lengths():
    urls, seen = [], set()
    for rid in recent_runs():
        for f in con.execute("SELECT url FROM findings WHERE run_id=? ORDER BY idx", (rid,)):
            if f["url"] and f["url"] not in seen:
                seen.add(f["url"]); urls.append(f["url"])
    rows = asyncio.run(fetch_all(urls, lambda u, d, k: [{"url": u, "kind": k, "chars": len(d.text)}]))
    ok = [r for r in rows if "chars" in r]
    over = [r for r in ok if r["chars"] > NOTES_BUDGET]
    capped = [r for r in ok if r["chars"] >= EXTRACT_CAP]
    print(f"{len(ok)}/{len(urls)} read | median {int(st.median(r['chars'] for r in ok)):,} chars | "
          f"over the notes budget {len(over)} ({len(over)/len(ok):.0%}) | at the 80K cut {len(capped)}")
    return rows


def quotes():
    items, seen = [], set()
    for rid in recent_runs():
        for f in con.execute("SELECT url, path FROM findings WHERE run_id=? ORDER BY idx", (rid,)):
            p = Path(cfg.research_dir) / rid / f["path"]
            if f["url"] in seen or youtube.video_id(f["url"]) or not p.is_file():
                continue
            qs = [q for q in _QUOTE.findall(p.read_text(encoding="utf-8")) if len(q) >= 30]
            if qs:
                seen.add(f["url"]); items.append((f["url"], qs[:3]))
    items = dict(items[:130])

    def locate(u, doc, kind):
        full, head = _norm(doc.text), len(_norm(doc.text[:WINDOW]))
        return [{"url": u, "pos": full.find(_norm(q)[:80]), "head": head} for q in items[u]]
    rows = asyncio.run(fetch_all(list(items), locate))
    found = [r for r in rows if r.get("pos", -1) >= 0]
    inside = sum(r["pos"] < r["head"] for r in found)
    print(f"{len(found)} quotes located | inside the judge's first {WINDOW:,} chars: {inside} "
          f"({inside/len(found):.0%}) | median position {int(st.median(r['pos'] for r in found)):,}")
    return rows


def pages():
    qs = json.loads((DATA / "queries.json").read_text())
    rows = asyncio.run(fetch_all(sorted({q["url"] for q in qs}),
                                 lambda u, d, k: [{"url": u, "kind": k, "text": d.text}]))
    with (DATA / "pages.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {sum('text' in r for r in rows)} pages ({sum('error' in r for r in rows)} failed) to {DATA / 'pages.jsonl'}")
    return []


result = {"lengths": lengths, "quotes": quotes, "pages": pages}[MODE]()
if MODE != "pages":
    (DATA / f"probe-{MODE}.json").write_text(json.dumps(result))
