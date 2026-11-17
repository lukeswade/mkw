#!/usr/bin/env bash
# Run scripts/eval/report_eval.py in a throwaway dev container.
#
#   scripts/eval/report_eval.sh freeze | check | score [arm] [--judge] | resynth <label>
#
# The working tree's app/ is mounted, so an arm tests the code in the tree and
# not the deployed image. The library is read-only; only data/eval is
# writable. The serving container is not touched.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p data/eval
rev="$(git rev-parse --short HEAD)"
git diff --quiet -- app || rev="$rev+dirty"
exec docker run --rm -i --add-host host.docker.internal:host-gateway \
  --env-file .env -e DATA_DIR=/data -e REPORT_EVAL_REV="$rev" \
  -v "$PWD/app:/srv/app/app:ro" \
  -v "$PWD/data:/data:ro" \
  -v "$PWD/data/eval:/data/eval" \
  -w /srv/app mkw-app-dev:latest python - "$@" < scripts/eval/report_eval.py
