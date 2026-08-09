#!/usr/bin/env bash
# Dev stack: hot-reload app + SearXNG exposed on http://127.0.0.1:8081
set -euo pipefail
cd "$(dirname "$0")/.."
exec docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build "$@"
