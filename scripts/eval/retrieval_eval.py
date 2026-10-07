"""Retrieval eval for the knowledge layer: Ask, claim checks, the similar-run
cutoffs, and what a different embedder or an added reranker would change.

Runs inside the app container, reading the library read-only and calling the
same OpenAI-compatible server the app uses:

    docker compose exec -T app python - [options] < scripts/eval/retrieval_eval.py

Modes
  --freeze             snapshot the library's chunks into <data>/corpus.jsonl and,
                       unless <data>/queries.json exists, write 160 queries from
                       stored key facts (the main model paraphrases each fact into
                       a question and a claim). Freeze once; every later eval runs
                       on the identical set, so numbers compare across models.
  (default)            evaluate an embedder, and optionally a reranker, on the set.

Options
  --data DIR           eval set directory (default /data/eval/2026-09-23)
  --embed-model NAME   server model id to embed with (default: the app's setting).
                       Evaluated directly: Settings, Chroma and the live index are
                       never touched, so this is safe while the app is serving.
  --query-prefix TEXT  override the query instruction. Default: the app's prefix
                       table; a Qwen3-Embedding model gets Qwen's Instruct format
                       when the table has none (the app's table has it since 6ba256f).
  --rerank-model NAME  rerank each query's top --rerank-pool chunks via /rerank
  --rerank-pool N      candidates handed to the reranker (default 50)
  --label NAME         results file stem (default derived from the models)

Labels. A query is answered when a shown chunk comes from a finding with the
fact's own URL ("strict"), or quotes every distinctive number in the fact
("lenient"; applies to facts with numbers). Both undercount: other sources and
overviews often carry the same fact in words. Compare models on the same label,
not against 100%.

Pauses while any research run is active or queued (polls the app's /health),
up to --max-pause seconds in total, so it can run overnight next to real use.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sqlite3
import sys
import time
from pathlib import Path

import httpx
import numpy as np

from app.config import load_settings
from app.rag.chunking import chunk_markdown
from app.rag.embeddings import RemoteEmbedder, make_embedder, prefixes_for

QWEN_INSTRUCT = ("Instruct: Given a research question or a claim, retrieve research "
                 "notes that answer or verify it\nQuery: ")
# What the app does today (app/rag/service.py, app/research/verify.py).
APP = {
    "question": dict(feature="Ask", pool=30, floor=0.35, per_run=3, per_title=0, limit=10),
    "claim": dict(feature="claim check", pool=24, floor=0.45, per_run=3, per_title=2, limit=6),
}
RUN_CUTOFFS = {"similar-run link": 0.55, "similar-run hint": 0.62}
N_QUERIES, SEED = 160, 7


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ---- politeness: never compete with a live research run --------------------
class Idle:
    def __init__(self, max_pause: float):
        self.left = max_pause
        self.warned = False

    def wait(self) -> None:
        while True:
            try:
                h = httpx.get("http://127.0.0.1:8090/health", timeout=5).json()
                busy = int(h.get("active", 0)) + int(h.get("queued", 0))
            except Exception as e:
                if not self.warned:
                    log(f"WARNING: /health unreachable ({e}); cannot tell whether a run is active")
                    self.warned = True
                busy = 0  # unreachable: carry on rather than hang
            if not busy:
                return
            if self.left <= 0:
                raise SystemExit("gave up: research runs kept the server busy past --max-pause")
            log(f"a research run is active; pausing ({int(self.left)}s of pause budget left)")
            time.sleep(30)
            self.left -= 30


# ---- the frozen set -----------------------------------------------------------
def build_corpus(cfg) -> list[dict]:
    con = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True, timeout=10)
    con.row_factory = sqlite3.Row
    out = []
    for d in sorted(Path(cfg.research_dir).iterdir()):
        if not (d / "meta.json").is_file():
            continue
        row = con.execute("SELECT * FROM runs WHERE id = ?", (d.name,)).fetchone()
        if row is None:
            continue
        ov = d / "overview.md"
        if ov.is_file():
            for c in chunk_markdown(ov.read_text(encoding="utf-8")):
                out.append({"run": d.name, "kind": "overview", "url": "",
                            "title": row["title"] or "", "query": row["query"] or "", "text": c})
        for f in con.execute("SELECT * FROM findings WHERE run_id = ? ORDER BY idx", (d.name,)):
            p = d / f["path"]
            if p.is_file():
                for c in chunk_markdown(p.read_text(encoding="utf-8")):
                    out.append({"run": d.name, "kind": "finding", "url": f["url"] or "",
                                "title": f["title"] or "", "query": "", "text": c})
    return out


_FACT_RE = re.compile(r"^- (.+?) \(confidence (\d+)/10\)\s*$", re.M)
_QA_PROMPT = """For each fact below, write two things a person might type WITHOUT having seen the source:
- "question": a natural question they would ask a research library, whose answer is this fact
- "claim": the fact restated as a claim they might paste in to be fact-checked
Paraphrase: do not copy the fact's distinctive wording or word order. Keep product and model names as people would type them.

Return JSON: {{"items": [{{"id": <id>, "question": "...", "claim": "..."}}, ...]}}

Facts:
{facts}"""


async def make_queries(cfg) -> list[dict]:
    from pydantic import BaseModel
    from app.llm.client import LLM

    class QA(BaseModel):
        id: int
        question: str
        claim: str

    class QAOut(BaseModel):
        items: list[QA]

    con = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True, timeout=10)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT run_id, url, path FROM findings").fetchall()
    rnd = random.Random(SEED)
    rnd.shuffle(rows)
    items, seen = [], set()
    for r in rows:
        p = Path(cfg.research_dir) / r["run_id"] / r["path"]
        if r["url"] in seen or not p.is_file():
            continue
        facts = [m.group(1).strip() for m in _FACT_RE.finditer(p.read_text(encoding="utf-8"))
                 if int(m.group(2)) >= 8 and len(m.group(1)) > 40]
        if facts:
            seen.add(r["url"])
            items.append({"url": r["url"], "run": r["run_id"], "fact": rnd.choice(facts)})
        if len(items) >= N_QUERIES:
            break
    llm = LLM(cfg)
    for start in range(0, len(items), 10):
        batch = items[start:start + 10]
        facts = "\n".join(f"{start + j}. {it['fact']}" for j, it in enumerate(batch))
        out = await llm.chat_json("eval", [{"role": "user", "content": _QA_PROMPT.format(facts=facts)}],
                                  QAOut, max_tokens=3000, temperature=0.3)
        for qa in out.items:
            if start <= qa.id < start + len(batch):
                items[qa.id].update(question=qa.question.strip(), claim=qa.claim.strip())
        log(f"queries {start + len(batch)}/{len(items)}")
    return [it for it in items if it.get("question") and it.get("claim")]


# ---- scoring ----------------------------------------------------------------------
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def distinctive_numbers(fact: str) -> set[str]:
    out = set()
    for n in _NUM.findall(fact):
        n = n.replace(",", "").rstrip(".")
        if len(n) >= 2 and not re.fullmatch(r"(19|20)\d\d", n):
            out.add(n)
    return out


def shown(order, scores, corpus, *, pool, floor, per_run, per_title, limit, soft=False, **_):
    """The app's selection: floor, then per-run and per-title caps, then limit.
    soft=True backfills empty slots from capped candidates, in rank order."""
    taken_run, taken_title, out, held = {}, {}, [], []
    for j in order[:pool]:
        if scores[j] < floor:
            continue
        run, title = corpus[j]["run"], corpus[j]["title"] or corpus[j]["run"]
        if taken_run.get(run, 0) >= per_run or (per_title and taken_title.get(title, 0) >= per_title):
            held.append(j)
            continue
        taken_run[run] = taken_run.get(run, 0) + 1
        taken_title[title] = taken_title.get(title, 0) + 1
        out.append(j)
        if len(out) >= limit:
            return out
    if soft:
        out += held[:limit - len(out)]
    return out


def quantiles(xs) -> dict:
    xs = sorted(xs)
    return {f"p{p}": round(xs[int(p / 100 * (len(xs) - 1))], 4) for p in (10, 25, 50, 75, 90)} if xs else {}


async def embed_corpus(emb, corpus, cache: Path, idle: Idle) -> np.ndarray:
    digest = hashlib.sha1("".join(c["text"] for c in corpus).encode()).hexdigest()[:12]
    meta = cache.with_suffix(".json")
    if cache.is_file() and meta.is_file() and json.loads(meta.read_text()).get("digest") == digest:
        log(f"reusing cached embeddings {cache.name}")
        return np.load(cache)
    t, vecs, step = time.time(), [], 256
    for i in range(0, len(corpus), step):
        idle.wait()
        vecs += await emb.encode_docs([c["text"] for c in corpus[i:i + step]])
        if (i // step) % 8 == 0:
            done = min(i + step, len(corpus))
            rate = done / max(1e-6, time.time() - t)
            log(f"embedded {done}/{len(corpus)} chunks ({rate:.1f}/s, ~{(len(corpus) - done) / max(rate, 1e-6) / 60:.0f} min left)")
    D = np.array(vecs, dtype=np.float32)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    np.save(cache, D)
    meta.write_text(json.dumps({"digest": digest, "chunks": len(corpus), "secs": round(time.time() - t, 1)}))
    return D


RERANK_SECS: list[float] = []


async def rerank(client, base, key, model, query, docs) -> list[float]:
    t = time.time()
    r = await client.post(f"{base}/rerank", headers={"Authorization": f"Bearer {key}"} if key else {},
                          json={"model": model, "query": query, "documents": docs, "top_n": len(docs)})
    r.raise_for_status()
    body = r.json()
    results = body.get("results") or body.get("data") or []
    scores = [float("-inf")] * len(docs)
    for it in results:
        scores[int(it["index"])] = float(it.get("relevance_score", it.get("score", 0.0)))
    RERANK_SECS.append(time.time() - t)
    return scores


async def evaluate(args, cfg, idle: Idle) -> dict:
    data = Path(args.data)
    corpus = [json.loads(l) for l in (data / "corpus.jsonl").read_text().splitlines() if l]
    queries = json.loads((data / "queries.json").read_text())
    base = (getattr(cfg, "embedding_base_url", "") or cfg.resolved_base_url).rstrip("/")
    key = getattr(cfg, "embedding_api_key", "") or cfg.resolved_api_key
    model = args.embed_model or (getattr(cfg, "embedding_model", "") or "")
    if model:
        emb = RemoteEmbedder(base, model, key)
        prefix = args.query_prefix if args.query_prefix is not None else (
            prefixes_for(model)[0] or (QWEN_INSTRUCT if "qwen3-emb" in model.lower() else ""))
        emb.query_prefix = prefix
    else:
        emb, prefix = make_embedder(cfg), "(baked-in default)"
    slug = re.sub(r"[^a-z0-9]+", "-", (model or "baked-in").lower()).strip("-")[:48]
    log(f"embedder {model or 'baked-in'} | query prefix {prefix!r} | {len(corpus)} chunks | {len(queries)} queries")
    D = await embed_corpus(emb, corpus, data / f"emb-{slug}.npy", idle)
    urls = [c["url"] if c["kind"] == "finding" else "" for c in corpus]
    norm = [re.sub(r"(?<=\d),(?=\d)", "", c["text"]) for c in corpus]
    res: dict = {"embed_model": model or "baked-in", "query_prefix": prefix,
                 "rerank_model": args.rerank_model or "", "chunks": len(corpus),
                 "queries": len(queries), "features": {}, "rankings": {}}
    client = httpx.AsyncClient(timeout=120) if args.rerank_model else None
    for qtype, app in APP.items():
        idle.wait()
        Q = np.array([await emb.encode_query(q[qtype]) for q in queries], dtype=np.float32)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        S = Q @ D.T
        counts = {k: 0 for k in ("strict_today", "lenient_today", "lenient_soft", "ceiling_pool",
                                 "ceiling_50", "rerank", "rerank_soft")}
        rel_best, top1, rank_rows = [], [], []
        for i, q in enumerate(queries):
            nums = distinctive_numbers(q["fact"])
            order = np.argsort(-S[i])[:max(60, args.rerank_pool)]
            strict = {int(j) for j in order if urls[j] == q["url"]}
            lenient = strict | {int(j) for j in order if nums and all(n in norm[j] for n in nums)}
            today = shown(order, S[i], corpus, **app)
            counts["strict_today"] += any(j in strict for j in today)
            counts["lenient_today"] += any(j in lenient for j in today)
            counts["lenient_soft"] += any(j in lenient for j in shown(order, S[i], corpus, **app, soft=True))
            counts["ceiling_pool"] += any(j in lenient for j in order[:app["pool"]] if S[i][j] >= app["floor"])
            counts["ceiling_50"] += any(j in lenient for j in order[:50] if S[i][j] >= app["floor"])
            rel = [float(S[i][j]) for j in range(len(corpus)) if urls[j] == q["url"]]
            if rel:
                rel_best.append(max(rel))
            top1.append(float(S[i][order[0]]))
            if client is not None:
                cand = [int(j) for j in order[:args.rerank_pool] if S[i][j] >= app["floor"]]
                if cand:
                    rs = await rerank(client, base, key, args.rerank_model, q[qtype],
                                      [corpus[j]["text"] for j in cand])
                    reordered = [cand[k] for k in np.argsort(-np.array(rs))]
                    ones = {j: 1.0 for j in reordered}  # the floor was applied before reranking
                    sel = dict(app, pool=len(reordered), floor=0.0)
                    counts["rerank"] += any(j in lenient for j in shown(reordered, ones, corpus, **sel))
                    counts["rerank_soft"] += any(j in lenient for j in shown(reordered, ones, corpus, **sel, soft=True))
            rank_rows.append([[int(j), round(float(S[i][j]), 4)] for j in order[:60]])
        n = len(queries)
        res["features"][app["feature"]] = {
            **{k: round(v / n, 3) for k, v in counts.items() if client is not None or not k.startswith("rerank")},
            "relevant_best_score": quantiles(rel_best), "top1_score": quantiles(top1)}
        res["rankings"][qtype] = rank_rows
        log(f"{app['feature']}: " + ", ".join(f"{k} {v}" for k, v in res["features"][app["feature"]].items()
                                              if not isinstance(v, dict)))
    if client is not None:
        await client.aclose()
        res["rerank_secs"] = quantiles(RERANK_SECS)
        log(f"rerank latency per query ({len(RERANK_SECS)} calls, {args.rerank_pool}-chunk pool): {res['rerank_secs']}")
    # Run-level: each run's own question against overview chunks - what the
    # similar-run link (0.55) and the "already researched" hint (0.62) see.
    idle.wait()
    ov = [j for j, c in enumerate(corpus) if c["kind"] == "overview"]
    groups = {}
    for j in ov:
        g = re.sub(r"\s+", " ", corpus[j]["query"].lower()).strip()[:200]
        groups.setdefault(corpus[j]["run"], (g, corpus[j]["query"][:600]))
    runs = sorted(groups)
    Qr = np.array([await emb.encode_query(groups[r][1]) for r in runs], dtype=np.float32)
    Qr /= np.linalg.norm(Qr, axis=1, keepdims=True)
    Sr = Qr @ D[ov].T
    own_first = own_best = other_best = 0
    own_scores, other_scores = [], []
    for i, r in enumerate(runs):
        same = np.array([groups[corpus[j]["run"]][0] == groups[r][0] for j in ov])
        order = np.argsort(-Sr[i])
        own_first += bool(same[order[0]])
        own_scores.append(float(Sr[i][same].max()))
        other_scores.append(float(Sr[i][~same].max()) if (~same).any() else 0.0)
    res["runs"] = {"n": len(runs), "own_run_first": round(own_first / len(runs), 3),
                   "own_best": quantiles(own_scores), "other_best": quantiles(other_scores),
                   "other_best_scores": [round(x, 4) for x in other_scores],
                   "pass_rates": {name: round(float(np.mean(np.array(other_scores) >= c)), 3)
                                  for name, c in RUN_CUTOFFS.items()}}
    log(f"runs: own run first {res['runs']['own_run_first']}, other-run best {res['runs']['other_best']}")
    return res


def cutoff_matching(base: dict, new: dict) -> list[str]:
    """New-model cutoffs that pass the same share the old cutoffs passed."""
    lines = []
    b, n = sorted(base["runs"]["other_best_scores"]), sorted(new["runs"]["other_best_scores"])
    for name, c in RUN_CUTOFFS.items():
        share = float(np.mean(np.array(b) >= c))
        k = min(len(n) - 1, max(0, int(round((1 - share) * (len(n) - 1)))))
        lines.append(f"  {name}: {c} passes {share:.0%} of runs under the baseline -> {n[k]:.2f} passes the same share")
    for feat, app in (("Ask", APP["question"]), ("claim check", APP["claim"])):
        bq, nq = base["features"][feat]["relevant_best_score"], new["features"][feat]["relevant_best_score"]
        lines.append(f"  {feat} floor {app['floor']}: relevant best-score median {bq.get('p50')} -> {nq.get('p50')}, "
                     f"p10 {bq.get('p10')} -> {nq.get('p10')} (keep the floor below the new p10)")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/eval/2026-09-23")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--embed-model", default="")
    ap.add_argument("--query-prefix", default=None)
    ap.add_argument("--rerank-model", default="")
    ap.add_argument("--rerank-pool", type=int, default=50)
    ap.add_argument("--label", default="")
    ap.add_argument("--max-pause", type=float, default=4 * 3600)
    args = ap.parse_args(sys.argv[1:])
    cfg = load_settings()
    idle = Idle(args.max_pause)
    data = Path(args.data)
    if args.freeze:
        data.mkdir(parents=True, exist_ok=True)
        corpus = build_corpus(cfg)
        with (data / "corpus.jsonl").open("w") as fh:
            for c in corpus:
                fh.write(json.dumps(c) + "\n")
        log(f"froze {len(corpus)} chunks from {len({c['run'] for c in corpus})} runs")
        if not (data / "queries.json").is_file():
            idle.wait()
            (data / "queries.json").write_text(json.dumps(asyncio.run(make_queries(cfg)), indent=1))
        return
    res = asyncio.run(evaluate(args, cfg, idle))
    label = args.label or re.sub(r"[^a-z0-9]+", "-", (
        (res["embed_model"] + ("+" + args.rerank_model if args.rerank_model else "")).lower())).strip("-")[:80]
    (data / f"results-{label}.json").write_text(json.dumps(res))
    baseline = data / "results-baseline.json"
    print(f"\n{label}  ({res['chunks']} chunks, {res['queries']} queries)")
    base = json.loads(baseline.read_text()) if baseline.is_file() and baseline.name != f"results-{label}.json" else None
    for feat, m in res["features"].items():
        print(f"  {feat}:")
        for k, v in m.items():
            if isinstance(v, dict):
                continue
            was = f"   (baseline {base['features'][feat][k]:.0%})" if base and k in base["features"][feat] else ""
            print(f"    {k:14} {v:5.0%}{was}")
    print(f"  runs: own run ranked first {res['runs']['own_run_first']:.0%}; other-run best score {res['runs']['other_best']}")
    if res.get("rerank_secs"):
        print(f"  rerank seconds per query: {res['rerank_secs']}")
    if base:
        print("  cutoff retune, if this model is adopted:")
        print("\n".join(cutoff_matching(base, res)))


if __name__ == "__main__":
    main()
