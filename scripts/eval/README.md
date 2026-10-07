# Evaluation scripts

Measure before changing the knowledge layer or synthesis. Written 2026-09-23;
results and decisions are in `docs/retrieval-eval-2026-09-23.md`. Container
scripts are piped in (the image does not ship `scripts/`), read the library
read-only, and never touch Settings, the live Chroma index or the research
files. The frozen eval set lives in `data/eval/2026-09-23/` (gitignored).

| Script | Where | What it answers |
|---|---|---|
| `retrieval_eval.py` | container | How often Ask and claim checks are shown the right source, for any embedder and optionally a reranker, on the frozen set. Suggests cutoff values for a new model. Pauses while research runs are active. |
| `passage_score.py` | host | How often the claim judge is shown the verified evidence, for whatever `verify.best_passage` is on `PYTHONPATH`. |
| `page_probe.py` | container | Page lengths against the notes budget, where evidence quotes sit in their pages, and fetching the eval claims' source pages for `passage_score.py`. |
| `synth_ab.py` | container | Re-synthesizes one run from its stored sources with thinking off and on, read-only. |
| `mteb_compare.py` | host | Same-task MTEB retrieval scores for several embedders, from the published per-task results. |

```bash
# baseline (nomic), reproduces results-baseline.json
docker compose exec -T app python - --label baseline < scripts/eval/retrieval_eval.py
# a different embedder, once it is on the server (embeds 10,857 chunks; ~2 h for the 4B at 1.5/s)
docker compose exec -T app python - --embed-model Qwen3-Embedding-4B-4bit-DWQ < scripts/eval/retrieval_eval.py
# a reranker over the configured embedder's pool (uses the cached nomic vectors)
docker compose exec -T app python - --rerank-model Qwen3-Reranker-0.6B-mxfp8 < scripts/eval/retrieval_eval.py
# claim-judge evidence visibility for the code in a checkout
PYTHONPATH=. .venv/bin/python scripts/eval/passage_score.py
# thinking A/B on one run: off,on,off,on
docker compose exec -T app python - <run_id> off,on,off,on < scripts/eval/synth_ab.py > synth.jsonl
```
