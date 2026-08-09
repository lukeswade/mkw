#!/usr/bin/env bash
# Run the test suite inside the dev container image.
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose -f docker-compose.yml -f docker-compose.dev.yml build app
exec docker compose -f docker-compose.yml -f docker-compose.dev.yml run --rm --no-deps app pytest "$@"
