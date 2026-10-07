#!/usr/bin/env bash
# Deploy files from one git ref as an overlay on the image that is serving now.
#
#   scripts/deploy_overlay.sh [--dry-run] <git-ref> <path> [<path> ...]
#
#   scripts/deploy_overlay.sh retrieval-fixes \
#       app/rag/service.py app/research/verify.py app/rag/embeddings.py
#
# Why not `docker compose build`: the build context is the working tree, and
# the working tree can hold another session's uncommitted work - a plain build
# ships it. This copies exactly the committed versions of the named files from
# <git-ref> onto the running image, so the only change that ships is the one
# named. Pure-source changes only: no new dependencies, no Dockerfile changes.
#
# Refuses while a research run is active or queued (checked again just before
# the recreate). Tags the serving image for rollback first, recreates only the
# app container, waits up to 120s for /health, then proves every file in the
# container matches <git-ref>. --dry-run builds and verifies the image but
# changes nothing that is serving.
set -euo pipefail
cd "$(dirname "$0")/.."
dry=0
if [ "${1:-}" = "--dry-run" ]; then dry=1; shift; fi
[ $# -ge 2 ] || { echo "usage: $0 [--dry-run] <git-ref> <path>..." >&2; exit 2; }
ref="$1"; shift
for p in "$@"; do
  git cat-file -e "$ref:$p" 2>/dev/null || { echo "$p is not in $ref" >&2; exit 2; }
done

health() { curl -sf --max-time 5 localhost:8090/health; }
idle() { health | grep -q '"active":0,"queued":0'; }
h=$(health) || { echo "app is not healthy on :8090 - not deploying" >&2; exit 1; }
[ $dry = 1 ] || idle || { echo "a research run is in progress ($h) - try later" >&2; exit 1; }

running=$(docker inspect mkw-app-1 --format '{{.Image}}')
stamp=$(date +%Y%m%d-%H%M%S)
if [ $dry = 1 ]; then base="mkw-app:dryrun-base-$stamp"; out="mkw-app:dryrun-$stamp"
else base="mkw-app:rollback-$stamp"; out="mkw-app:latest"; fi
docker tag "$running" "$base"      # BuildKit resolves a local tag, not a bare image id

ctx=$(mktemp -d)
trap 'rm -rf "$ctx"' EXIT
{
  echo "FROM $base"
  echo "USER root"
  for p in "$@"; do
    mkdir -p "$ctx/$(dirname "$p")"
    git show "$ref:$p" > "$ctx/$p"
    echo "COPY --chown=app $p /srv/app/$p"
  done
  echo "USER app"
} > "$ctx/Dockerfile"
docker build -q -t "$out" "$ctx" > /dev/null
new=$(docker image inspect "$out" --format '{{.Id}}')
[ "$new" != "$running" ] || { echo "the build produced the serving image - nothing to deploy" >&2; exit 1; }

check() {  # $1 = a function that runs a shell command in the target
  local bad=0 p want got
  for p in "${paths[@]}"; do
    want=$(git show "$ref:$p" | shasum | cut -c1-40)
    got=$($1 "sha1sum /srv/app/$p" | cut -c1-40)
    if [ "$want" = "$got" ]; then echo "MATCH  $p"; else echo "DIFFER $p"; bad=1; fi
  done
  return $bad
}
paths=("$@")

if [ $dry = 1 ]; then
  in_image() { docker run --rm --entrypoint sh "$out" -c "$1"; }
  check in_image || { echo "files in the built image do not match $ref" >&2; exit 1; }
  docker rmi -f "$out" > /dev/null; docker rmi "$base" > /dev/null
  echo "dry run OK: $ref ($(git rev-parse --short "$ref")) builds onto the serving image; nothing changed"
  exit 0
fi

echo "rollback: docker tag $base mkw-app:latest && docker compose up -d --force-recreate --no-build app"
idle || { echo "a run started during the build - image built but not deployed; rerun later" >&2; exit 1; }
docker compose up -d --force-recreate --no-build app
for _ in $(seq 60); do health > /dev/null && break; sleep 2; done
health > /dev/null || { echo "UNHEALTHY after 120s - roll back with the command above" >&2; exit 1; }
in_app() { docker compose exec -T app sh -c "$1"; }
check in_app || { echo "deployed files do not match $ref - roll back with the command above" >&2; exit 1; }
echo "deployed $ref ($(git rev-parse --short "$ref")) onto $running; health: $(health)"
