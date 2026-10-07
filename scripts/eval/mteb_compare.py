"""Same-task MTEB retrieval comparison from the published per-task results.

    python3 scripts/eval/mteb_compare.py        (host; uses curl, no Python deps)

Reads embeddings-benchmark/results on GitHub for the models below and prints
nDCG@10 on every retrieval task all of them were scored on. The model cards'
headline numbers are NOT comparable with each other (nomic reports the
56-task MTEB, Qwen reports MTEB eng v2); these per-task rows are. 2026-09-23:
the standard 15-task mean was nomic 52.89, Qwen3-0.6B 55.52, Qwen3-4B 61.58.
Add a model by appending its results directory to MODELS.
"""
import json, subprocess
API = "https://api.github.com/repos/embeddings-benchmark/results/contents/results/"
RAW = "https://raw.githubusercontent.com/embeddings-benchmark/results/main/results/"
MODELS = {"nomic-modernbert": "nomic-ai__modernbert-embed-base/external",
          "qwen3-0.6B": "Qwen__Qwen3-Embedding-0.6B/b22da495047858cce924d27d76261e96be6febc0",
          "qwen3-4B": "Qwen__Qwen3-Embedding-4B/636cd9bf47d976946cdbb2b0c3ca0cb2f8eea5ff"}
RETRIEVAL = {"ArguAna", "ClimateFEVER", "ClimateFEVERHardNegatives", "DBPedia", "FEVER",
             "FEVERHardNegatives", "FiQA2018", "HotpotQA", "HotpotQAHardNegatives", "MSMARCO",
             "NFCorpus", "NQ", "QuoraRetrieval", "SCIDOCS", "SciFact", "Touche2020",
             "Touche2020Retrieval.v3", "TRECCOVID"} | {f"CQADupstack{s}Retrieval" for s in (
             "Android", "English", "Gaming", "Gis", "Mathematica", "Physics", "Programmers",
             "Stats", "Tex", "Unix", "Webmasters", "Wordpress")}
def get(url):
    out = subprocess.run(["curl", "-sfL", "--max-time", "20", url], capture_output=True, check=True).stdout
    return json.loads(out)
files = {k: {x["name"][:-5] for x in get(API + v) if x["name"].endswith(".json")} for k, v in MODELS.items()}
common = sorted(RETRIEVAL & files["nomic-modernbert"] & files["qwen3-4B"] & files["qwen3-0.6B"])
print("shared retrieval tasks:", len(common))
def score(model, task):
    d = get(RAW + MODELS[model] + "/" + task + ".json")
    sc = d.get("scores", {})
    split = sc.get("test") or sc.get("dev") or next(iter(sc.values()))
    rows = [r for r in split if r.get("hf_subset") in ("default", "en", None)] or split
    return 100 * float(rows[0].get("main_score", rows[0].get("ndcg_at_10")))
rows = []
for t in common:
    try:
        rows.append((t, score("nomic-modernbert", t), score("qwen3-0.6B", t), score("qwen3-4B", t)))
    except Exception as e:
        print("skip", t, e)
print(f"{'task':32} {'nomic':>6} {'q0.6B':>6} {'q4B':>6} {'4B-nomic':>9}")
for t, a, b, c in rows:
    print(f"{t:32} {a:6.2f} {b:6.2f} {c:6.2f} {c-a:+9.2f}")
n = len(rows)
if n:
    A, B, C = (sum(r[i] for r in rows) / n for i in (1, 2, 3))
    print(f"{'MEAN over ' + str(n):32} {A:6.2f} {B:6.2f} {C:6.2f} {C-A:+9.2f}  (relative {100*(C-A)/A:+.0f}%)")
