CREATE TABLE IF NOT EXISTS runs (
  id             TEXT PRIMARY KEY,
  query          TEXT NOT NULL,
  title          TEXT,
  depth          INTEGER NOT NULL,
  recency        TEXT NOT NULL,
  status         TEXT NOT NULL,                 -- queued|running|completed|failed|cancelled|interrupted
  dir            TEXT NOT NULL,
  parent_run_id  TEXT,
  origin         TEXT NOT NULL DEFAULT 'web',   -- web|telegram|cli
  origin_chat_id INTEGER,
  error          TEXT,
  stop_reason    TEXT,
  stats_json     TEXT,
  created_at     TEXT NOT NULL,
  started_at     TEXT,
  finished_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);

CREATE TABLE IF NOT EXISTS findings (
  id             INTEGER PRIMARY KEY,
  run_id         TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  idx            INTEGER NOT NULL,              -- the [n] citation number
  url            TEXT,
  title          TEXT,
  domain         TEXT,
  published_date TEXT,
  relevance      REAL,
  path           TEXT,                          -- run-dir-relative, e.g. findings/001_foo.md
  summary        TEXT,
  created_at     TEXT NOT NULL,
  UNIQUE (run_id, idx)
);

CREATE TABLE IF NOT EXISTS entities (
  id          INTEGER PRIMARY KEY,
  name        TEXT NOT NULL,
  name_norm   TEXT NOT NULL UNIQUE,
  type        TEXT NOT NULL DEFAULT 'other',
  description TEXT,
  created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_entities (
  run_id    TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
  salience  REAL NOT NULL DEFAULT 0.5,
  PRIMARY KEY (run_id, entity_id)
);

CREATE TABLE IF NOT EXISTS run_links (
  src_run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  dst_run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  kind       TEXT NOT NULL,                     -- similar|followup
  score      REAL,
  PRIMARY KEY (src_run_id, dst_run_id, kind)
);

CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
  run_id UNINDEXED,
  kind UNINDEXED,
  title,
  body
);
