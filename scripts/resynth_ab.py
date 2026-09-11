#!/usr/bin/env python
"""A/B one finished run's synthesis without re-searching anything.

Clones a completed run once per arm, re-synthesizes each clone under whatever
overrides that arm names, and leaves every arm as its own row in the Library
so the documents can be read side by side. The stored findings are reused, so
an arm costs a few minutes of local compute and no search quota.

Written for the 2026-09-11 synthesis-coverage work; see
docs/synthesis-coverage-2026-09-11.md for what it measured and why.

The image does not ship scripts/, so copy it in first:

    docker compose cp scripts/resynth_ab.py app:/data/resynth_ab.py
    docker compose exec -T -e PYTHONPATH=/srv/app -w /srv/app app \
        python /data/resynth_ab.py <run_id> [arm ...]

Arms are `name:key=value` pairs, applied to app.research.synthesizer for the
duration of that arm and restored afterwards. With no arms it runs one arm
named `current` against the deployed code, which is the baseline you usually
want:

    ... resynth_ab.py 20260911_173708_... \
        current \
        wide:_SINGLE_CALL_BUDGET=100000 \
        terse:_WORDS_PER_SOURCE=20
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import sys
import time

from app.config import load_settings
from app.db import Repo, connect
from app.llm.client import LLM
from app.research import synthesizer
from app.research.pipeline import Pipeline
from app.research.progress import ProgressBus
from app.research.storage import RunStore


def clone(cfg, repo, src_row, src_store, src_meta, label: str) -> str:
    """A completed copy of the source run, findings and plan included."""
    store = RunStore.create(cfg.research_dir, src_row["query"])
    rid = store.run_id
    shutil.rmtree(store.dir / "findings")
    shutil.copytree(src_store.dir / "findings", store.dir / "findings")
    if (src_store.dir / "sources.md").exists():
        shutil.copy(src_store.dir / "sources.md", store.dir / "sources.md")
    meta = dict(src_meta)
    meta.update(run_id=rid, status="completed", ab_arm=label)
    store.write_meta(meta)
    # parent_run_id stays None deliberately: a parent turns synthesis into a
    # "what changed since" delta document, which is a different test.
    repo.create_run(run_id=rid, query=src_row["query"], depth=src_row["depth"],
                    recency=src_row["recency"], dir=rid, origin="cli",
                    status="completed", categories=src_row["categories"] or "",
                    kind="research")
    # findings.run_id is a foreign key onto runs.id, so the row comes first.
    for f in repo.findings_for_run(src_row["id"]):
        repo.add_finding(run_id=rid, idx=f["idx"], url=f["url"], title=f["title"],
                         domain=f["domain"], published_date=f["published_date"],
                         relevance=f["relevance"], path=f["path"],
                         summary=f["summary"] or "")
    # The Library label differs from the document's own H1, which stays clean.
    repo.update_run(rid, title=f"[A/B {label}] {src_row['title']}")
    return rid


def parse_arm(spec: str) -> tuple[str, dict[str, int]]:
    name, _, rest = spec.partition(":")
    over: dict[str, int] = {}
    for pair in filter(None, rest.split(",")):
        k, _, v = pair.partition("=")
        over[k.strip()] = int(v)
    return name or "arm", over


async def run_arm(cfg, repo, src_row, src_store, src_meta,
                  name: str, over: dict[str, int]) -> dict:
    rid = clone(cfg, repo, src_row, src_store, src_meta, name)
    saved = {k: getattr(synthesizer, k) for k in over}
    for k, v in over.items():
        setattr(synthesizer, k, v)
    pipe = Pipeline(cfg, repo, ProgressBus(), llm_factory=lambda: LLM(cfg))
    t = time.time()
    try:
        await pipe.resynthesize(rid)
        err = ""
    except Exception as e:                      # noqa: BLE001 — reported, not raised
        err = f"{type(e).__name__}: {e}"
    finally:
        for k, v in saved.items():
            setattr(synthesizer, k, v)
    wall = time.time() - t
    store = RunStore(cfg.research_dir / rid)
    ov = store.overview_path.read_text() if store.overview_path.exists() else ""
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", ov)}
    n = len(repo.findings_for_run(rid)) or 1
    words = len(ov.split())
    return {
        "arm": name, "run_id": rid, "overrides": over or "(deployed code)",
        "sources": n, "cited": len(cited), "cite_rate": f"{100*len(cited)/n:.0f}%",
        "words": words, "words_per_cite": round(words / max(1, len(cited)), 1),
        "wall_s": round(wall, 1), "error": err,
        "sections": [l[3:].strip() for l in ov.splitlines() if l.startswith("## ")],
    }


async def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src_id, specs = sys.argv[1], sys.argv[2:] or ["current"]
    cfg = load_settings()
    repo = Repo(connect(cfg.db_path))
    src_row = repo.get_run(src_id)
    if src_row is None:
        sys.exit(f"run {src_id} not found")
    src_store = RunStore(cfg.research_dir / src_row["dir"])
    src_meta = src_store.read_meta()

    results = []
    for spec in specs:
        name, over = parse_arm(spec)
        r = await run_arm(cfg, repo, src_row, src_store, src_meta, name, over)
        results.append(r)
        print(json.dumps(r, indent=1), flush=True)

    print("\n=== SUMMARY ===")
    print(f"{'arm':<20}{'cited':>7}{'rate':>7}{'words':>8}{'w/cite':>8}{'wall':>8}")
    for r in results:
        print(f"{r['arm']:<20}{r['cited']:>7}{r['cite_rate']:>7}{r['words']:>8}"
              f"{r['words_per_cite']:>8}{r['wall_s']:>7}s"
              + (f"  ERR {r['error']}" if r["error"] else ""))


if __name__ == "__main__":
    asyncio.run(main())
