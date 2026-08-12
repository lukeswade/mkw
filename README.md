# 🔭 Deep Research

A self-hosted deep-research agent — a personal Perplexity that digs much
deeper. Give it a question, a depth, and a recency window; it unpacks the
question into targeted web searches, reads the results, writes per-source
notes with verbatim evidence, lets the gaps steer further search rounds, and
ends with a cited overview plus follow-up suggestions.

Everything runs on your own machine. The only thing that can leave it is the
LLM call — and even that stays local if you point it at llama.cpp, LM Studio,
Ollama, or MLX.

- **Web UI** with live progress, a research library (keyword + semantic
  search), and an "Ask" page that answers from everything you've researched,
  with citations.
- **Telegram bot**: start runs, watch progress, get the overview delivered.
- **Evergreen topics**: mark a run evergreen and it re-researches itself daily,
  linking each update back to the original.
- **Everything is plain markdown on disk** (`data/research_data/<run>/`).
  SQLite and the vector index are derived — `reindex` rebuilds them from the
  files at any time.

---

## Install

You need Docker (with the compose plugin) and an LLM. Nothing else.

```bash
git clone https://github.com/lukeswade/deep-research.git
cd deep-research

cp .env.example .env
nano .env                 # add an LLM (see below) + a SEARXNG_SECRET

mkdir -p data && sudo chown -R 1000:1000 data   # container runs as uid 1000

docker compose build      # build natively on this machine, don't cross-build
docker compose up -d
```

Open <http://localhost:8090>. First run pulls the SearXNG image and bakes a
small embedding model into the app image, so the initial build takes a few
minutes and produces a ~2.5 GB image.

### Choose an LLM

**Cloud (simplest, costs cents per run).** Get a key from
<https://platform.deepseek.com> and put it in `.env`:

```
DEEPSEEK_API_KEY=sk-...
```

**Local (free, private, slower).** Run any OpenAI-compatible server — LM
Studio, llama.cpp's `llama-server`, Ollama, or an MLX server — and point the
app at it:

```
LLM_PROVIDER=local
LOCAL_LLM_BASE_URL=http://host.docker.internal:1234/v1
LOCAL_LLM_MODEL=your-model-id
LOCAL_LLM_API_KEY=sk-local     # only if your server checks it
LLM_CONCURRENCY=1              # one request at a time suits a single GPU
```

Pick a model that supports **strict `json_schema` output** (LM Studio and
recent llama.cpp both do). The pipeline asks for schema-constrained decoding,
which makes malformed JSON impossible; without it the app still works, but
weaker models occasionally emit unparseable output and those sources get
dropped. A 30B-class instruct model is a good floor for research quality.

You can change any of this later on the **Settings** page, which also has
"Test LLM" and "Test SearXNG" buttons. Settings are stored in
`data/settings.json` (chmod 600) and override `.env`.

---

## Using it

**Depth** is the maximum number of search rounds. Each round runs several
targeted queries, reads what's useful, and then a gap analysis decides what to
search next. Runs stop early when the topic is saturated, after two rounds
that find nothing new, or at per-depth source and LLM-call caps.

- **Depth 0** is quick chat — a straight streamed answer from the model with
  no web search. Good for a fast question.
- **Depth 1–3** is a normal question.
- **Depth 5–10** is a project. Expect many sources and, on a local model, a
  long wall-clock time.

**Recency** maps to search-engine time filters plus a date check on the
documents themselves. Engine date metadata is imperfect, so undated sources
are kept but flagged, and the synthesis is told to prefer dated in-window
material. Treat the windows as best effort.

Every run directory contains `overview.md` (the cited synthesis),
`further-research.md`, `sources.md`, per-source `findings/*.md`, a `rounds/`
log, `meta.json`, and `events.jsonl`. The "Further research" tab turns each
suggestion into a one-click follow-up run.

**Ask** answers questions from everything you've researched so far, citing the
runs it drew on. New runs automatically build on related earlier research.

**Evergreen** — the ☆ button on a finished run. The topic is re-researched
once a day against a recent window, and each refresh appears as a linked child
run. Toggle it off with the same button.

### Telegram (optional)

1. Message **@BotFather** → `/newbot` → put the token in `.env` as
   `TELEGRAM_BOT_TOKEN`.
2. `docker compose restart app`, message your bot `/id`, put the number in
   `TELEGRAM_ALLOWED_USER_IDS`, restart again.
3. Send it a question. It walks you through depth and recency, then delivers
   the overview when the run finishes. `/help` lists every command.

The bot only answers allowlisted user IDs. Without a token the app simply runs
web-only.

### CLI

```bash
docker compose exec app python -m app.cli run "your question" --depth 3 --recency month
docker compose exec app python -m app.cli runs
docker compose exec app python -m app.cli ask "question over everything so far"
docker compose exec app python -m app.cli reindex    # rebuild indexes from the .md files
```

---

## Remote access (optional)

The app binds to port 8090 with no TLS and, by default, no password. That is
fine on your own machine and **not** fine on the open internet.

If you want it reachable from your phone, the intended path is a Cloudflare
Tunnel plus a Cloudflare Access policy — no inbound ports, TLS handled for
you, and an identity check before any request reaches the app:

```bash
# after creating a tunnel in the Cloudflare dashboard and adding an Access policy
echo 'CLOUDFLARE_TUNNEL_TOKEN=...' >> .env
docker compose --profile tunnel up -d
```

Set `WEB_PASSWORD` as well. Access is the lock on the door; the app password is
the lock on the room, and it's what protects you if a tunnel ever points at a
hostname whose policy you forgot to attach.

---

## Development

```bash
./scripts/dev.sh     # hot-reload stack; SearXNG exposed on 127.0.0.1:8081
./scripts/test.sh    # full test suite in the dev image
./scripts/check.sh   # what CI runs — tests plus a startup self-check
```

Build natively on each architecture (arm64 Mac for dev, amd64 server for
prod). Don't cross-build with QEMU: the torch stack is slow and flaky under
emulation. On the server, `git pull && docker compose build && docker compose up -d`.

The dev and prod images are tagged separately (`mkw-app-dev` / `mkw-app`), so
building one never clobbers the other.

---

## Operations

- **Backup**: `tar czf backup.tgz data/` — that is the entire state (settings,
  SQLite, vectors, and all the research markdown).
- **Rebuild indexes**: `docker compose exec app python -m app.cli reindex`
  reconstructs the database, keyword search, and vectors from the markdown on
  disk. The markdown is the source of truth.
- **Deleting a run** removes it everywhere: database row, keyword index,
  vectors, and the run directory. It cancels the run first if it is still
  going. This is not recoverable — the markdown goes with it, so take a backup
  if you might want it later.
- **Interrupted runs** (a restart mid-run) keep everything already gathered and
  offer *Retry with same parameters*. Queued runs survive restarts.
- **Logs**: `docker compose logs -f app`, plus a rotating `data/app.log`.
- **One process, one worker.** The job queue, SSE bus, and run registry live in
  memory in a single process. Never add `--workers` to uvicorn, and run only
  one app container per data directory — a second one would also fight for the
  Telegram token and you'd see 409 Conflict in the logs.
- **Don't open `data/app.sqlite3` with a host `sqlite3` while the app is
  running.** WAL mode over a bind mount leaves stale `-shm`/`-wal` sidecars
  that break the container with "disk I/O error". Use
  `docker compose exec app python -m app.cli runs` instead. If it happens: stop
  the app, delete `data/app.sqlite3-shm` and `-wal`, start again.
- SearXNG engines rate-limit sometimes (`unresponsive_engines` in the logs);
  other engines fill in and the pipeline rephrases on an empty round. The
  SearXNG image is pinned by digest in `docker-compose.yml`.

---

## Security notes

- Secrets live in `.env` (never committed) and `data/settings.json` (mode 600).
  Neither is baked into the image.
- Fetched pages are rendered with raw HTML disabled everywhere, and search
  snippets are escaped before highlighting — page content can't inject markup
  into your UI. Fetched text is framed as untrusted data in every prompt.
- The fetcher refuses private, loopback, and link-local addresses (set
  `ALLOW_PRIVATE_FETCH=true` if you genuinely need intranet sources) and
  honours robots.txt by default.
- Run files are served through a filename allowlist plus a containment check,
  so a run id can't be used to read outside its own directory.

## Licence

MIT — see [LICENSE](LICENSE).
