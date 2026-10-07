# Retrieval, reranker and thinking evaluation (2026-09-22/23, models measured 2026-10-07)

A handoff: enough to act without re-reading the session. Everything was
measured on the live library (107 runs, 10,857 chunks) with
Qwen3.6-35B-A3B-oQ4e-mtp and nomicai-modernbert-embed-base-bf16 on oMLX
0.7.0, M4 Pro 48 GB. On 2026-10-07 the three fixes were deployed and merged,
and both Qwen models were downloaded and measured. Neither model is wired
into the app and no Setting changed: the app still embeds with nomic.

## State

| Item | State (2026-10-07) | Decision owner |
|---|---|---|
| Three fixes | **Deployed** (`mkw-app:latest` = `77989d6f5c28`; rollback `mkw-app:rollback-20261007-110126`), smoke-tested, **merged to main**. | Done |
| Qwen3-Reranker-0.6B | Downloaded and measured: claim checks 74% → 85%, Ask 64% → 69%. **Not wired in.** | Luke (recommended: claim checks first) |
| Qwen3-Embedding-4B | Downloaded and measured: claim checks 74% → 82%, Ask 64% → 70%; with the reranker 89% and 79%. **Not adopted.** | Luke, once RAM is sorted |
| Thinking on for synthesis | Tested (A/B below). Not adopted. | Parked |
| EmbeddingGemma 2 | Released 2026-10-06. oMLX 0.7.0 cannot serve it. | Revisit after an oMLX update |
| Eval scripts, this doc | Committed. | Done |

## The branch `retrieval-fixes` (three commits on `7f87c8f`, deployed and merged 2026-10-07)

| Commit | Change | Measured |
|---|---|---|
| `f7234b0` | Ask backfills slots the 3-per-run cap left empty (`app/rag/service.py`). Prior knowledge keeps the strict cap. | Ask shown its source 64% → 67% (retrieval_eval) |
| `6f11d97` | Claim checks show the judge each page's passage about the claim, not its first 1,200 chars (`verify.best_passage`, lexical, no model). | Judge sees the verified evidence 30% → 77% of 141 claims (passage_score) |
| `6ba256f` | Qwen3-Embedding queries carry Qwen's instruction (`app/rag/embeddings.py`). No effect on nomic. | Prerequisite for any Qwen switch |

Review: `git log -p main..retrieval-fixes`. Deploy, dry run first:

```bash
scripts/deploy_overlay.sh --dry-run retrieval-fixes app/rag/service.py app/research/verify.py app/rag/embeddings.py
scripts/deploy_overlay.sh retrieval-fixes app/rag/service.py app/research/verify.py app/rag/embeddings.py
git merge --no-edit retrieval-fixes   # touches only files the diligence WIP does not
```

The script refuses while a run is active, tags a rollback image, recreates
only the app container, waits for /health and hash-checks each file. Smoke
test after: one Ask question, one claim check on a short paragraph. On
2026-10-07 the test gate re-ran (701 passed), all three files hash-matched in
the running container, Ask answered in 13 s, and a three-claim paragraph came
back 6/6 supported in 2 minutes.

## Next decisions, with rules that make them mechanical

**Measuring a model** (done for both Qwen models on 2026-10-07; results under
Measurements). Download it into `~/models/<org>/<name>`, then in the oMLX
dashboard click Models → Manager → ↻ Reload: oMLX does not see new folders
otherwise, and Reload unloads everything and preloads only pinned models.
The eval never calls the 35B, so unpin and unload it first if RAM is short.

```bash
hf download mlx-community/Qwen3-Reranker-0.6B-mxfp8 --local-dir ~/models/mlx-community/Qwen3-Reranker-0.6B-mxfp8     # 0.63 GB
hf download mlx-community/Qwen3-Embedding-4B-4bit-DWQ --local-dir ~/models/mlx-community/Qwen3-Embedding-4B-4bit-DWQ  # 2.28 GB
# confirm oMLX lists them (ids are the directory names), then:
docker compose exec -T app python - --rerank-model Qwen3-Reranker-0.6B-mxfp8 < scripts/eval/retrieval_eval.py      # ~15-30 min
docker compose exec -T app python - --embed-model Qwen3-Embedding-4B-4bit-DWQ < scripts/eval/retrieval_eval.py      # ~1-2 h
docker compose exec -T app python - --embed-model Qwen3-Embedding-4B-4bit-DWQ \
    --rerank-model Qwen3-Reranker-0.6B-mxfp8 < scripts/eval/retrieval_eval.py                                        # stacked
```

Each run prints its numbers beside the baseline and, for an embedder, the
cutoff values that keep today's pass rates. Suggested rules:
- **Embedder:** adopt only if `lenient_today` rises at least 5 points on both
  Ask and claim check, and "own run ranked first" stays at or above 90%.
- **Reranker:** adopt for a feature if `rerank` beats `lenient_today` by at
  least 4 points with p90 latency at or under 3 s.
- **Both:** stack only if the stacked run beats each alone. On Qwen's own
  benchmark the 0.6B reranker (65.8) scores below the 4B embedder alone
  (68.5), so it can reorder a stronger retriever's list for the worse.

Verdicts on 2026-10-07 (numbers under Measurements):
- **Embedder:** gains pass (+6 Ask, +8 claim checks). "Own run ranked first"
  fell to 89.7%, 0.3 under the bar; the misses are sibling runs on the same
  topic, as before.
- **Reranker:** gains pass (+5 Ask, +11 claim checks). p90 is 5.2 s, not 3 s.
  The 3 s bar was written for Ask; a claim check is a background job of
  minutes, where 5 s per claim does not show.
- **Both:** stacking beats each alone on both features (79% and 89%), so the
  worry above did not happen.
- **Recommendation:** wire the reranker into claim checks first (+11 points,
  no rebuild, 0.65 GB). Decide on the 4B once RAM is sorted: with the
  reranker it gives Ask its biggest lift, but it costs 2.4 GB on a Mac that
  ran out of memory on 2026-10-07.

**If the reranker is adopted**, in order of value: (1) library evidence for
claim checks, measured +11 points: rerank the top 50 before the per-title and
per-run caps in `verify.library_evidence`; (2) the same for Ask, +5 points for
about 5 s per question; (3) claim-check web evidence: rerank each fetched
page's chunks against the claim, keeping `best_passage` as the fallback
(unmeasured; `best_passage` already shows the judge the evidence 77% of the
time); (4) search-result ordering before fetching, the only place it could
improve research reports; unmeasured, needs a live A/B. A setting
for it belongs in `app/config.py`, `routes_settings.py` and `settings.html`,
all of which hold the other session's uncommitted edits: coordinate first.

**If the embedder is adopted:**
1. Deploy `6ba256f` first (the query instruction), or the switch loses 1-5%.
2. Pre-build the new collection before flipping the setting. Flipping first
   points Ask, search, claim checks and the similar-run hint at an empty index
   for the whole rebuild, because `app.cli reindex` only builds the configured
   model's collection. Approach (no script exists; write one and test it on a
   scratch Chroma dir first): run `RagService.index_run()` per run with a
   settings object whose `embedding_model` is the new one, run links disabled
   and a read-only repo, reusing retrieval_eval's cached vectors by chunk text.
3. Flip the model in Settings. The old collection stays for switching back.
4. Retune cutoffs to the values the eval printed for the 4B: `LINK_MIN_SCORE`
   0.55 → 0.43 and the similar-run hint (`0.62` in `app/web/routes_runs.py`,
   which holds the other session's uncommitted edits) → 0.65. The Ask floor
   (`ASK_MIN_SCORE`, 0.35) and claim-check floor (`_LIBRARY_MIN_SCORE` in
   `app/research/verify.py`, 0.45) already sit below the 4B's p10s (0.50,
   0.53); check `PRIOR_MIN_SCORE` the same way.
5. Costs, measured: 2.4 GB more oMLX memory; embedding at 1.5 chunks/s, so
   about a minute more indexing per run (~100 chunks) and about 2 h to
   pre-build the collection, less with retrieval_eval's cached vectors.

**Thinking for synthesis (parked).** To settle it, run `synth_ab.py` on three
different runs with `off,on,off,on` (about 45 min of GPU) and grade by hand.
Regex scoring missed three real errors in this round.

## Measurements

### Qwen models on the same set (2026-10-07)

`results-*.json` in the eval dir. Lenient label unless marked; "with
backfill" is `lenient_soft` / `rerank_soft`. The reranker reorders each
embedder's top 50.

| Setup | Ask | Ask with backfill | Claim check | Claim check, strict |
|---|---|---|---|---|
| nomic (serving) | 64% | 67% | 74% | 68% |
| nomic + reranker | 69% | 71% | 85% | n/a |
| Qwen3-Embedding-4B | 70% | 73% | 82% | 80% |
| 4B + reranker | 79% | 80% | 89% | n/a |
| Perfect reranker over the 4B's top 50 | 90% | | 94% | |

- Reranker latency per query (50 chunks): p50 4.5 s, p90 5.2 s over nomic's
  candidates; p50 4.9 s, p90 5.7 s over the 4B's. The 35B was unloaded, so
  the GPU was otherwise idle.
- 4B embedding: 1.3-1.5 chunks/s, 2 h 3 min for 10,857 chunks; 2,560 dims;
  2.38 GB in oMLX. Own run ranked first 89.7% (nomic 92.5%).
- The predictions held: 4B Ask about 70% (measured 70%) and claim checks
  about 80% (82%); reranker +4 to +8 points (+5 Ask, +11 claim checks).

### Thinking on for synthesis (run 20260923_020339, 20 sources, depth 3)

Same stored sources and the same candidate list for every sample; thinking
arm given a 30k budget. Errors graded by hand on five checks from the
original report: benchmark-version mix-up, omits the 4B, dates
Qwen3-Embedding to 2026 (it is June 2025), suggests the embedder as a
reranker, claims a Qwen embedder "aligns" with a Qwen LLM. Half = partly.

| Report | Thinking | Synthesis time | Tokens out | Words | Errors /5 |
|---|---|---|---|---|---|
| Original, in-run | off | n/a | n/a | 1,682 | 5 |
| Sample 1 | off | 141 s | 6,730 | 2,056 | 4 |
| Sample 3 | off | 94 s | 5,550 | 1,543 | 2 |
| Sample 2 | on | 237 s | 14,022 | 1,314 | 2.5 |
| Sample 4 | on | 246 s | 14,364 | 1,640 | 1.5 |

Thinking roughly doubles synthesis time (about 15% of a depth-3 run),
reasons 14-20k chars per call and finishes far inside its budget; oMLX
returns reasoning separately, so nothing leaks into the report. Averages 2.0
errors against 3.0, but two samples an arm, and every sample carried about
four more errors from its sources (all four repeat "4-8 GB" for nomic v2).
The run's real weakness was its "past 3 months" window: no primary sources.
Outputs: `data/eval/2026-09-23/synth-ab/`.

### Library retrieval, nomic baseline (`results-baseline.json`)

160 queries written by the 35B from stored key facts, paraphrased on purpose.
Answered = a shown chunk from the fact's own source ("strict"), or quoting
all its distinctive numbers ("lenient"). Both undercount; compare models on
the same label.

| | Ask (10 shown) | Claim check (6 shown) |
|---|---|---|
| Shown the source today, strict | 56% | 68% |
| Shown the source today, lenient | 64% | 74% |
| With backfill (`f7234b0`) | 67% | 74% |
| Perfect reranker over today's pool (30 / 24) | 80% | 86% |
| Perfect reranker over a top-50 pool | 82% | 91% |

Score floors are not the problem: they cut at most one relevant passage.
Misses split between ranking (in the pool, below the cut: 20 Ask questions,
22 claims) and recall (not in the pool at all: 24% of questions, 16% of
claims). Finding the right past run is saturated: own run ranked first 92%,
and all misses were sibling runs on the same topic. Nothing logs how often
Ask or claim checks are used.

### Benchmarks (`mteb-shared-tasks.txt`, from the published per-task results)

Standard 15-task MTEB retrieval mean nDCG@10: nomic 52.89, Qwen3-0.6B 55.52,
Qwen3-4B 61.58. The cards' headline numbers are on different benchmarks
(56-task vs eng v2) and are not comparable. Gains are uneven: ArguAna +26.7,
FiQA +22.1, SCIDOCS +12.9, SciFact +8.7, HotpotQA +7.6, FEVER +4.3,
MSMARCO +1.3, NQ +1.0, Quora -0.8. Qwen3-Reranker-0.6B over Qwen3-Embedding-0.6B
candidates: 61.8 → 65.8 retrieval, but 75.4 → 73.4 on code.
Prediction, since measured (above): the 4B would lift Ask to about 70% and
claim checks to about 80%; a 0.6B reranker by about +4 to +8 points.

EmbeddingGemma 2 (released 2026-10-06): a 270M-parameter text path, 768
dims, 8K context, MTEB eng v2 68.46 against Qwen3-Embedding-0.6B's 70.70 and
the 4B's 74.60; its gains are in code (+9.9) and image, video and audio.
oMLX 0.7.0 cannot serve it: its mlx_embeddings 0.1.0 has no
`embedding_gemma2`, and the MLX ports need an unreleased mlx-vlm build plus
newer mlx and transformers. Measure it with the harness once oMLX supports
it; it would need a prefix row (`task: question answering | query: `,
`title: none | text: `).

### Pages and evidence (`probe-*.json`, `pages.jsonl`)

- 401 kept sources from the 14 latest runs: median 12,336 chars; 19 (5%)
  over the 56K notes budget; 8 (2%) at the extractor's 80K head cut.
- 341 verified quotes on 127 pages: 45% inside the first 1,200 chars;
  median position 1,383; a quarter past 4,124.
- Reads wasted since 2026-09-14: 30 pages read and rejected vs 217 kept (12%).

### oMLX facts

- The ceiling is dynamic: oMLX's own use + free + inactive memory (+ half of
  other apps' active memory on Aggressive) − a reserve (3.8 GB on Balanced).
  It was 42.9 GB in September and 19-29 GB on 2026-10-07 with Chrome holding
  16 GB. Below about 25 GB the 35B (23.0 GB with nomic) has its prompts
  refused (`prefill_memory_exceeded`), and runs fail at their first LLM call.
  Loads at about 0.5-0.8 s/GB.
- Serves `/v1/embeddings` and `/v1/rerank` (model, query, documents, top_n,
  return_documents, max_chunks_per_doc); Qwen3-Reranker supported through
  yes/no scoring since v0.2.20. Verified 2026-10-07: both Qwen models load as
  `embedding` and `reranker` engines.
- Sustained embedding load starved chat decode until PR #2873 (merged
  2026-09-15; dev4 probably has it). Issue #266: embedding and rerank engines
  recompile after about 3 s idle, so the first request after a pause is slow.
- The 35B has `thinking_default: true`; the app sends `enable_thinking: false`
  per call (`7f87c8f`). Models live in `~/models/<org>/<name>`.

## Pitfalls

- **The working tree holds another session's uncommitted `diligence` work.**
  `docker compose build` bakes it in. Deploy with `scripts/deploy_overlay.sh`.
  Serving since 2026-10-07: `mkw-app:latest` = `77989d6f5c28`, the
  `97f83403a06a` overlay (whose `app/` matched `7f87c8f` file for file) plus
  the three fix files from `6ba256f`, hash-checked. `mkw-app:diligence-wip-20260920`
  is that session's build; `mkw-app:pre-diligence` is the base.
- **A wedged Docker VM looks healthy.** On 2026-10-07 `/health` answered
  while the containers had no outbound network: runs failed at their first
  LLM call ("Connection error") and the tunnel dropped (mattkwade.com 530).
  The engine API hung even for `/_ping`. Restarting Docker Desktop fixed it
  (a force-quit was needed); restarting is Luke's call.
- **Out of RAM looks like a model failure.** oMLX refuses prompts when its
  dynamic ceiling drops below the 35B's needs (see oMLX facts). Compare
  `final_ceiling` with `current_model_memory` in `/v1/models/status`, and
  check swap, before suspecting code. A response with no `choices` then
  crashes the client in `app/llm/client.py` (flagged as its own task).
- **Never test an embedder by changing Settings**: the app switches to that
  model's empty collection at once. Use `retrieval_eval.py --embed-model`.
- **Never re-freeze into `data/eval/2026-09-23/`**: it would overwrite the
  corpus the baseline and caches were built on. Freeze into a new dated dir.
- **Check `/health` for `"active":0,"queued":0` before GPU work.** Matt runs
  depth-10 runs (about 53 min on 2026-09-23).
- **The host venv lacks `sentence_transformers`**, so `tests/test_rag.py`
  skips silently there. Real gate: the dev image with the checkout mounted,
  README included (the image's baked README is stale and fails one test):
  `docker run --rm -v $PWD/app:/srv/app/app -v $PWD/tests:/srv/app/tests -v $PWD/pyproject.toml:/srv/app/pyproject.toml:ro -v $PWD/README.md:/srv/app/README.md:ro -w /srv/app mkw-app-dev:latest python -m pytest -o addopts="" -q -p no:cacheprovider`
- The app requests `Jundot/Qwen3.6-35B-A3B-oQ4e-mtp`; the server lists
  `Qwen3.6-35B-A3B-oQ4e-mtp`. Resolves by loose matching today.
- `sources_uncited` in meta.json goes stale after a re-synthesis
  (`resynthesize` discards `_mark_honestly`'s count). The fix belongs in
  `app/research/pipeline.py`, which holds the other session's edits.
- For "is X outdated?" research, use all-time recency: the 3-month window
  shut out every primary source for the embeddings run.

## Files

- `scripts/eval/` (see its README), `scripts/deploy_overlay.sh`
- `data/eval/2026-09-23/` (gitignored): `corpus.jsonl`, `queries.json`,
  `results-baseline.json`, the three 2026-10-07 results
  (`results-nomicai-modernbert-embed-base-bf16-qwen3-reranker-0-6b-mxfp8.json`,
  `results-qwen3-embedding-4b-4bit-dwq.json`,
  `results-qwen3-embedding-4b-4bit-dwq-qwen3-reranker-0-6b-mxfp8.json`), the
  cached vectors `emb-nomicai-modernbert-embed-base-bf16.npy` and
  `emb-qwen3-embedding-4b-4bit-dwq.npy`, `pages.jsonl`, `probe-lengths.json`,
  `probe-quotes.json`, `mteb-shared-tasks.txt`, `synth-ab/`
- Models (outside the repo): `~/models/mlx-community/Qwen3-Embedding-4B-4bit-DWQ`
  (2.1 GB) and `~/models/mlx-community/Qwen3-Reranker-0.6B-mxfp8` (0.6 GB).
