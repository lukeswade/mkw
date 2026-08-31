#!/usr/bin/env bash
# Which search engines are actually answering right now?
#
# A throttled engine is invisible from the app: results just quietly get
# worse. This asks SearXNG directly, so "is search healthy" and "did the
# router reboot help" are one command instead of a guess.
#
#   ./scripts/check_engines.sh            # general category
#   ./scripts/check_engines.sh science    # any category
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

CATEGORY="${1:-general}"
PROBE="${2:-fly rod repair}"

# searxng's image has no httpx; the app container shares the same egress.
echo "WAN address engines see:"
docker compose exec -T app python -c "
import httpx
try:
    print('   ', httpx.get('https://api.ipify.org', timeout=10).text.strip())
except Exception as e:
    print('    (could not determine:', type(e).__name__, ')')
" 2>/dev/null

echo
echo "Engine health — category '$CATEGORY', probe \"$PROBE\":"
docker compose exec -T app python - "$CATEGORY" "$PROBE" <<'PY' 2>/dev/null
import sys, httpx
from app.config import load_settings

category, probe = sys.argv[1], sys.argv[2]
base = load_settings().searxng_url.rstrip("/")
cfg = httpx.get(base + "/config", timeout=30).json()
engines = sorted(e["name"] for e in cfg.get("engines", [])
                 if e.get("enabled") and category in e.get("categories", []))

ok = blocked = empty = 0
for name in engines:
    try:
        j = httpx.get(base + "/search", timeout=60,
                      params={"q": probe, "format": "json", "engines": name}).json()
        n = len(j.get("results", []))
        dead = j.get("unresponsive_engines") or []
        why = (dead[0][1] if isinstance(dead[0], (list, tuple)) and len(dead[0]) > 1
               else str(dead[0])) if dead else ""
    except Exception as e:
        n, why = 0, f"{type(e).__name__}"
    if why:
        print(f"  REFUSED  {name:<22} {why}"); blocked += 1
    elif n:
        print(f"  ok       {name:<22} {n} results"); ok += 1
    else:
        print(f"  empty    {name:<22} (answered, matched nothing)"); empty += 1

print(f"\n  {ok} answering, {blocked} refusing, {empty} empty, of {len(engines)}.")
if blocked > len(engines) / 3:
    print("  More than a third are refusing — that is a network-address problem,")
    print("  not a tool problem. See README 'When search goes quiet'.")
PY
