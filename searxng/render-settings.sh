#!/bin/sh
# Render settings.yml with API keys taken from the environment, then hand off
# to SearXNG's own entrypoint.
#
# Why this exists: SearXNG gives only `secret_key` an environ_name hook
# (settings_defaults.py:218) — engine api_key values have no env path, and
# folder-mode SEARXNG_SETTINGS_PATH loads sibling config files rather than
# merging settings.yml. So a key in settings.yml would have to be committed,
# and this repo publishes to a public mirror.
#
# The committed settings.yml therefore carries no keys. This copies it
# somewhere writable, appends an engine block per key that is actually set,
# and points SearXNG at the copy. No key set means no block appended and a
# completely normal startup — which is what anyone cloning the repo gets.
set -eu

SRC=/settings.src.yml
DEST_DIR=/etc/searxng
DEST="$DEST_DIR/settings.yml"
SENTINEL='# @@ API_ENGINES @@'

# The image's system python3 has no PyYAML — only SearXNG's venv does. Using
# the wrong one made the validity check below fail on a perfectly good file
# and silently fall back, which would have thrown away an API key.
PY=/usr/local/searxng/.venv/bin/python
[ -x "$PY" ] || PY=python3

# Write where the image already looks rather than redirecting it. The image
# pins its own __SEARXNG_SETTINGS_PATH, so exporting SEARXNG_SETTINGS_PATH at
# a rendered copy elsewhere was silently ignored — the engine block was
# written to a file nothing read. /etc/searxng is a tmpfs (see
# docker-compose.yml) so this is writable and rebuilt on every start.
mkdir -p "$DEST_DIR"
cp "$SRC" "$DEST"

# Engines are appended at the sentinel, not at end-of-file: appending blindly
# would silently produce invalid YAML the day a new top-level section is added
# below the engines list.
if ! grep -q "$SENTINEL" "$DEST"; then
    echo "render-settings: sentinel missing from settings.yml, adding no engines" >&2
else
    BLOCK=""
    if [ -n "${BRAVE_API_KEY:-}" ]; then
        BLOCK="$BLOCK
  - name: braveapi
    engine: braveapi
    shortcut: brapi
    categories: [general, web]
    api_key: '$BRAVE_API_KEY'
    results_per_page: 20
    # Both flags on purpose. Stock settings.yml already declares braveapi with
    # inactive: true (it is useless without a key), and the merge layers this
    # block ONTO that default — so omitting these inherits the off switch and
    # the engine silently never registers, with no error logged anywhere.
    inactive: false
    disabled: false
  # The scraping brave engine covers exactly what braveapi covers, categories
  # general and web, so leaving it on queries Brave twice per search: once on
  # the metered API and once on a scraper that comes back rate-limited. Off
  # while the key is present; it returns on its own if the key is removed.
  #
  # Its three siblings go with it, and must: they declare network: brave to
  # share its connection pool, and SearXNG builds engine networks only for
  # engines it loaded, so leaving brave.news alive over a disabled brave is a
  # dangling reference that kills the whole service at startup with
  # KeyError: 'brave'. They are rate-limited scrapers of the same host
  # regardless, and braveapi is web-only so it never covered them anyway.
  - name: brave
    inactive: true
  - name: brave.news
    inactive: true
  - name: brave.videos
    inactive: true
  - name: brave.images
    inactive: true"
        echo "render-settings: braveapi enabled (scraping 'brave' disabled)"
    fi
    if [ -n "${MARGINALIA_API_KEY:-}" ]; then
        BLOCK="$BLOCK
  - name: marginalia
    engine: marginalia
    shortcut: mar
    categories: [general, web]
    api_key: '$MARGINALIA_API_KEY'
    # Stock declares marginalia with disabled: true; same merge trap as above.
    inactive: false
    disabled: false"
        echo "render-settings: marginalia enabled"
    fi
    if [ -n "${GITHUB_CODE_TOKEN:-}" ]; then
        BLOCK="$BLOCK
  - name: github code
    engine: github_code
    shortcut: ghc
    # Stock declares only 'code', and a search for category 'it' (the app's
    # code scope) selects engines whose categories contain 'it' — a
    # code-only engine is never asked. Both, so the scope reaches it.
    categories: [it, code]
    ghc_auth:
      type: personal_access_token
      token: '$GITHUB_CODE_TOKEN'
    ghc_highlight_matching_lines: true
    # Stock declares it inactive (useless without a token); same merge trap
    # as braveapi above.
    inactive: false
    disabled: false"
        echo "render-settings: github code enabled"
    fi
    if [ -n "${CORE_API_KEY:-}" ]; then
        BLOCK="$BLOCK
  - name: core.ac.uk
    engine: core
    shortcut: cor
    api_key: '$CORE_API_KEY'
    # Measured 2026-09-04: normal answers 2.4-5.6s, one in five past the
    # instance's 8s default — and one timeout suspends the engine for three
    # minutes, most of a round. 12s covers the tail; max_request_timeout is 15.
    timeout: 12.0
    inactive: false
    disabled: false"
        echo "render-settings: core.ac.uk enabled"
    fi

    if [ -n "$BLOCK" ]; then
        # python, not sed: the key is arbitrary text and sed would treat & and
        # \ in a replacement as metacharacters.
        BLOCK="$BLOCK" SENTINEL="$SENTINEL" DEST="$DEST" "$PY" - <<'PY'
import os
dest, sentinel, block = os.environ["DEST"], os.environ["SENTINEL"], os.environ["BLOCK"]
text = open(dest).read()
open(dest, "w").write(text.replace(sentinel, sentinel + block, 1))
PY
    else
        echo "render-settings: no API keys set, using stock engine list"
    fi
fi

# Keys must not reach the logs; only say whether the file parsed.
if "$PY" -c "import yaml; yaml.safe_load(open('$DEST'))" 2>/dev/null; then
    echo "render-settings: settings.yml rendered and parses"
else
    echo "render-settings: rendered settings.yml is INVALID YAML — falling back to the committed file" >&2
    cp "$SRC" "$DEST"
fi

exec /usr/local/searxng/entrypoint.sh "$@"
