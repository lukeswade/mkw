#!/usr/bin/env bash
# Everything CI runs. Run this before pushing.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "→ building dev image"
docker compose -f docker-compose.yml -f docker-compose.dev.yml build app >/dev/null

echo "→ startup self-check (prompt templates + fresh-database migration)"
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm --no-deps \
  -e DATA_DIR=/tmp/selfcheck-data app \
  python -c "
from app.config import load_settings
from app import selfcheck
cfg = load_settings('/tmp/selfcheck-data'); cfg.ensure_dirs()
selfcheck.run_all(cfg)
print('self-check OK')"

echo "→ test suite"
docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm --no-deps app pytest -q

echo "✔ all checks passed"
