#!/usr/bin/env bash
# Live end-to-end smoke test: real SearXNG + real DeepSeek, depth-1 run.
# Needs DEEPSEEK_API_KEY in .env (or data/settings.json). Costs a few cents.
set -euo pipefail
cd "$(dirname "$0")/.."

QUERY="${1:-What changed in EU AI Act implementation and enforcement this year?}"

docker compose up -d
echo "→ waiting for app health…"
for i in $(seq 1 30); do
  curl -fs http://127.0.0.1:8090/health >/dev/null 2>&1 && break
  sleep 2
done

echo "→ running depth-1 research: $QUERY"
START=$(date +%s)
docker compose exec -T app python -m app.cli run "$QUERY" --depth 1 --recency 6months
ELAPSED=$(( $(date +%s) - START ))
echo "→ finished in ${ELAPSED}s"

echo "→ checking artifacts…"
docker compose exec -T app python - <<'PY'
import json, pathlib, re, sys

runs = sorted(pathlib.Path("/data/research_data").iterdir())
assert runs, "no run directory created"
run = runs[-1]
meta = json.loads((run / "meta.json").read_text())
assert meta["status"] == "completed", f"status={meta['status']}"

overview = (run / "overview.md").read_text()
assert len(overview) > 1500, f"overview too short ({len(overview)} chars)"
citations = set(re.findall(r"\[(\d+)\]", overview))
assert len(citations) >= 3, f"only {len(citations)} distinct citations"

findings = list((run / "findings").glob("*.md"))
assert len(findings) >= 3, f"only {len(findings)} findings"
assert (run / "sources.md").exists()
assert (run / "further-research.md").exists()

stats = meta.get("stats", {})
llm = stats.get("llm", {})
print(f"OK: {run.name}")
print(f"    sources kept: {stats.get('sources_kept')} · skipped: {stats.get('sources_skipped')}")
print(f"    LLM calls: {llm.get('calls')} · est cost: ${llm.get('est_cost_usd', '?')}")
PY

echo "→ asking the knowledge base…"
docker compose exec -T app python -m app.cli ask "Summarize what we learned about: $QUERY" | head -20
echo "✔ smoke test passed"
