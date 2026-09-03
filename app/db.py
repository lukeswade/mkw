"""SQLite persistence: WAL connection, versioned migrations, query helpers.

Single-process, single-event-loop usage — the sync sqlite3 driver is fine at
this write volume (a few rows per research round).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path


# Sentinels for FTS highlighting — control characters, so they cannot occur in
# scraped page text and cannot be confused with markup.
FTS_MARK_OPEN = "\x02"
FTS_MARK_CLOSE = "\x03"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")



def row_get(row, name: str, default=None):
    """A column that may predate its migration.

    Only a missing or NULL column yields the default: 0 and "" are values,
    not absences. Three modules had grown their own copy of this.
    """
    try:
        value = row[name]
    except (KeyError, IndexError):
        return default
    return default if value is None else value

def _schema() -> str:
    return (Path(__file__).with_name("schema.sql")).read_text()


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str,
                           decl: str) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


# One entry per schema version; index i migrates user_version i → i+1.
# Each step must be idempotent: schema.sql already contains every column, so a
# fresh database runs step 0 (which creates them) AND every later step. Steps
# that add columns must therefore check before adding.
def _migrations() -> list:
    return [
        lambda conn: conn.executescript(_schema()),
        lambda conn: _add_column_if_missing(
            conn, "runs", "evergreen", "BOOLEAN NOT NULL DEFAULT 0"),
        # The knowledge-graph feature was removed; these tables were
        # write-only once the graph page went away.
        lambda conn: conn.executescript(
            "DROP TABLE IF EXISTS run_entities;"
            "DROP TABLE IF EXISTS entities;"),
        # Who started the run: a Cloudflare Access email, the LAN label,
        # a Telegram name, or "CLI". Old rows stay NULL and show no tag.
        lambda conn: _add_column_if_missing(
            conn, "runs", "created_by", "TEXT"),
        # Per-run SearXNG categories override (empty = global setting)
        lambda conn: _add_column_if_missing(
            conn, "runs", "categories", "TEXT NOT NULL DEFAULT ''"),
        # 0 = run without prior-run context (fresh diagnosis)
        lambda conn: _add_column_if_missing(
            conn, "runs", "use_prior", "BOOLEAN NOT NULL DEFAULT 1"),
        # "research" (default) or "brief" — which searcher the run uses
        lambda conn: _add_column_if_missing(
            conn, "runs", "kind", "TEXT NOT NULL DEFAULT 'research'"),
        # Named briefs: a saved reading list plus a standing interest. The
        # global FEEDS setting still works and behaves as an unnamed brief.
        lambda conn: conn.executescript("""
            CREATE TABLE IF NOT EXISTS briefs (
              id          INTEGER PRIMARY KEY,
              name        TEXT NOT NULL,
              feeds       TEXT NOT NULL DEFAULT '',
              topic       TEXT NOT NULL DEFAULT '',
              recency     TEXT NOT NULL DEFAULT 'week',
              depth       INTEGER NOT NULL DEFAULT 4,
              daily       BOOLEAN NOT NULL DEFAULT 0,
              created_at  TEXT NOT NULL,
              last_run_at TEXT
            );
        """),
        lambda conn: _add_column_if_missing(conn, "runs", "brief_id", "INTEGER"),
        # Whether the run has a comparison table. Every list page was doing a
        # filesystem stat per row to find out — up to 400 per Library load,
        # and the New tab polled every five seconds. NULL means "not yet
        # checked"; Orchestrator.recover() fills those in once at boot.
        lambda conn: _add_column_if_missing(conn, "runs", "has_matrix", "INTEGER"),
        # Every page read, with its outcome. Rejects used to live only in each
        # run's event file, so the install could not learn that a domain had
        # never produced a source across many runs. Idempotent by design.
        lambda conn: conn.executescript("""
            CREATE TABLE IF NOT EXISTS candidate_outcomes (
                run_id     TEXT NOT NULL,
                url        TEXT NOT NULL,
                domain     TEXT NOT NULL,
                engine     TEXT,
                outcome    TEXT NOT NULL,      -- kept | rejected | fail
                relevance  INTEGER,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_outcomes_domain ON candidate_outcomes(domain);
            CREATE INDEX IF NOT EXISTS idx_outcomes_run ON candidate_outcomes(run_id);
        """),
    ]


def connect(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


def migrate(conn: sqlite3.Connection) -> None:
    migrations = _migrations()
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for i in range(version, len(migrations)):
        migrations[i](conn)
        conn.execute(f"PRAGMA user_version = {i + 1}")
    conn.commit()


_RUN_COLS = {
    "query", "title", "depth", "recency", "status", "dir", "parent_run_id",
    "origin", "origin_chat_id", "error", "stop_reason", "stats_json",
    "evergreen", "created_by", "created_at", "started_at", "finished_at",
    "has_matrix",
}


class Repo:
    """All SQL lives here."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # ---- runs -------------------------------------------------------------
    def create_run(self, *, run_id: str, query: str, depth: int, recency: str,
                   dir: str, origin: str = "web", parent_run_id: str | None = None,
                   origin_chat_id: int | None = None, status: str = "queued",
                   evergreen: bool = False, created_by: str = "",
                   categories: str = "", use_prior: bool = True,
                   kind: str = "research", brief_id: int | None = None) -> None:
        self.conn.execute(
            "INSERT INTO runs (id, query, depth, recency, status, dir, origin,"
            " parent_run_id, origin_chat_id, evergreen, created_by, categories,"
            " use_prior, kind, brief_id,"
            " created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, query, depth, recency, status, dir, origin,
             parent_run_id, origin_chat_id, evergreen, created_by or None,
             categories or "", 1 if use_prior else 0, kind, brief_id,
             utcnow()),
        )
        self.conn.commit()

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()

    def list_runs(self, limit: int = 100, offset: int = 0) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM runs ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()

    def runs_with_status(self, *statuses: str) -> list[sqlite3.Row]:
        marks = ",".join("?" * len(statuses))
        return self.conn.execute(
            f"SELECT * FROM runs WHERE status IN ({marks}) ORDER BY created_at",
            statuses,
        ).fetchall()

    def list_evergreen_runs(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM runs WHERE evergreen = 1 AND status = 'completed'"
            " ORDER BY created_at"
        ).fetchall()

    def evergreen_due(self, interval_hours: int) -> list[sqlite3.Row]:
        """Evergreen runs with no refresh in flight and none inside the window.

        The flag stays on the original run, so 'due' is derived from the age of
        its newest child rather than by moving the flag along a chain (which
        died permanently the first time a child failed).
        """
        cutoff = (datetime.now(timezone.utc)
                  - timedelta(hours=interval_hours)).isoformat(timespec="seconds")
        return self.conn.execute(
            "SELECT r.* FROM runs r"
            " WHERE r.evergreen = 1 AND r.status = 'completed'"
            "   AND COALESCE(r.finished_at, r.created_at) < ?"
            "   AND NOT EXISTS ("
            "     SELECT 1 FROM runs c WHERE c.parent_run_id = r.id"
            "       AND (c.status IN ('queued','running') OR c.created_at >= ?))"
            " ORDER BY r.created_at",
            (cutoff, cutoff),
        ).fetchall()

    def update_run(self, run_id: str, **cols) -> None:
        bad = set(cols) - _RUN_COLS
        if bad:
            raise ValueError(f"unknown run columns: {bad}")
        if not cols:
            return
        assignments = ", ".join(f"{c} = ?" for c in cols)
        self.conn.execute(
            f"UPDATE runs SET {assignments} WHERE id = ?", (*cols.values(), run_id)
        )
        self.conn.commit()

    def delete_run(self, run_id: str) -> None:
        self.conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
        self.conn.execute("DELETE FROM fts WHERE run_id = ?", (run_id,))
        self.conn.commit()

    def set_stats(self, run_id: str, stats: dict) -> None:
        self.update_run(run_id, stats_json=json.dumps(stats))

    def runs_with_unknown_matrix(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id, dir FROM runs WHERE has_matrix IS NULL").fetchall()

    def recover_on_startup(self) -> list[str]:
        """Mark orphaned 'running' rows interrupted; return queued ids to re-enqueue."""
        self.conn.execute(
            "UPDATE runs SET status='interrupted', stop_reason='process restart',"
            " finished_at=? WHERE status='running'",
            (utcnow(),),
        )
        self.conn.commit()
        return [r["id"] for r in self.runs_with_status("queued")]

    # ---- findings ----------------------------------------------------------
    def add_finding(self, *, run_id: str, idx: int, url: str, title: str,
                    domain: str, published_date: str | None, relevance: float,
                    path: str, summary: str) -> None:
        self.conn.execute(
            "INSERT INTO findings (run_id, idx, url, title, domain, published_date,"
            " relevance, path, summary, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, idx, url, title, domain, published_date, relevance, path,
             summary, utcnow()),
        )
        self.conn.commit()

    # ---- named briefs -----------------------------------------------------

    def create_brief(self, *, name: str, feeds: str = "", topic: str = "",
                     recency: str = "week", depth: int = 4) -> int:
        cur = self.conn.execute(
            "INSERT INTO briefs (name, feeds, topic, recency, depth, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (name, feeds, topic, recency, depth, utcnow()))
        self.conn.commit()
        return int(cur.lastrowid)

    def list_briefs(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM briefs ORDER BY name COLLATE NOCASE").fetchall()

    def get_brief(self, brief_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM briefs WHERE id = ?",
                                 (brief_id,)).fetchone()

    def update_brief(self, brief_id: int, **cols) -> None:
        allowed = {"name", "feeds", "topic", "recency", "depth", "daily",
                   "last_run_at"}
        bad = set(cols) - allowed
        if bad:
            raise ValueError(f"cannot update brief columns: {sorted(bad)}")
        if not cols:
            return
        sets = ", ".join(f"{c} = ?" for c in cols)
        self.conn.execute(f"UPDATE briefs SET {sets} WHERE id = ?",
                          (*cols.values(), brief_id))
        self.conn.commit()

    def delete_brief(self, brief_id: int) -> None:
        self.conn.execute("DELETE FROM briefs WHERE id = ?", (brief_id,))
        self.conn.commit()

    def briefs_due(self, interval_hours: int) -> list[sqlite3.Row]:
        """Daily briefs that have not run inside the window."""
        return self.conn.execute(
            "SELECT * FROM briefs WHERE daily = 1 AND ("
            "  last_run_at IS NULL"
            "  OR julianday('now') - julianday(last_run_at) >= ?)",
            (interval_hours / 24.0,)).fetchall()

    def recent_finding_urls(self, kind: str, limit: int = 800) -> set[str]:
        """URLs already reported by recent runs of this kind.

        A daily brief re-reads the same feeds, so yesterday's items are still
        in today's window. Without this the brief repeats itself and reads as
        broken on day two.
        """
        rows = self.conn.execute(
            "SELECT f.url FROM findings f JOIN runs r ON r.id = f.run_id"
            " WHERE r.kind = ? ORDER BY f.id DESC LIMIT ?",
            (kind, limit)).fetchall()
        return {r["url"] for r in rows}

    def findings_for_run(self, run_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM findings WHERE run_id = ? ORDER BY idx", (run_id,)
        ).fetchall()

    # ---- what every read taught us -------------------------------------
    def record_outcome(self, *, run_id: str, url: str, domain: str, engine: str,
                       outcome: str, relevance: int | None) -> None:
        self.conn.execute(
            "INSERT INTO candidate_outcomes (run_id, url, domain, engine, outcome, "
            "relevance, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, url, domain.lower().removeprefix("www."), engine, outcome,
             relevance, utcnow()))
        self.conn.commit()

    def has_outcomes(self, run_id: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM candidate_outcomes WHERE run_id = ? LIMIT 1", (run_id,)
        ).fetchone() is not None

    def dead_domains(self, min_reads: int = 8, min_runs: int = 3) -> list[sqlite3.Row]:
        """Domains read many times, across several runs, that never produced a
        kept source. Yield is topic-bound, so a positive record says little
        globally — but zero over that many reads on that many topics says a
        lot."""
        return self.conn.execute(
            "SELECT domain, COUNT(*) AS reads, COUNT(DISTINCT run_id) AS runs "
            "FROM candidate_outcomes GROUP BY domain "
            "HAVING SUM(outcome = 'kept') = 0 AND reads >= ? AND runs >= ? "
            "ORDER BY reads DESC", (min_reads, min_runs)).fetchall()

    def productive_domains(self, min_reads: int = 6, limit: int = 25) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT domain, COUNT(*) AS reads, SUM(outcome = 'kept') AS kept, "
            "COUNT(DISTINCT run_id) AS runs FROM candidate_outcomes GROUP BY domain "
            "HAVING reads >= ? AND kept > 0 ORDER BY 1.0 * kept / reads DESC, reads DESC LIMIT ?",
            (min_reads, limit)).fetchall()

    def stats_since(self, since_iso: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT created_at, categories, stats_json FROM runs WHERE kind = 'research' "
            "AND stats_json IS NOT NULL AND created_at >= ?", (since_iso,)).fetchall()
        out = []
        for r in rows:
            try:
                d = json.loads(r["stats_json"]); d["_created_at"] = r["created_at"]; d["_categories"] = r["categories"] or ""
                out.append(d)
            except (TypeError, ValueError):
                continue
        return out

    def dead_domains_in_run(self, run_id: str, **kw) -> list[sqlite3.Row]:
        seen = {r["domain"] for r in self.conn.execute(
            "SELECT DISTINCT domain FROM candidate_outcomes WHERE run_id = ?", (run_id,))}
        return [d for d in self.dead_domains(**kw) if d["domain"] in seen]

    def kept_domains_for_runs(self, run_ids: list[str]) -> dict[str, int]:
        """domain -> kept sources across the given runs."""
        if not run_ids:
            return {}
        marks = ",".join("?" * len(run_ids))
        rows = self.conn.execute(
            f"SELECT domain, COUNT(*) AS n FROM findings WHERE run_id IN ({marks}) "
            f"GROUP BY domain", run_ids).fetchall()
        return {r["domain"]: r["n"] for r in rows}

    # ---- cross-run links ------------------------------------------------
    def add_run_link(self, src: str, dst: str, kind: str, score: float | None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO run_links (src_run_id, dst_run_id, kind, score)"
            " VALUES (?,?,?,?)",
            (src, dst, kind, score),
        )
        self.conn.commit()

    def links_for_run(self, run_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT l.*, r1.title AS src_title, r2.title AS dst_title"
            " FROM run_links l"
            " JOIN runs r1 ON r1.id = l.src_run_id"
            " JOIN runs r2 ON r2.id = l.dst_run_id"
            " WHERE l.src_run_id = ? OR l.dst_run_id = ?",
            (run_id, run_id),
        ).fetchall()

    # ---- full-text search ------------------------------------------------------
    def fts_add(self, run_id: str, kind: str, title: str, body: str) -> None:
        self.conn.execute(
            "INSERT INTO fts (run_id, kind, title, body) VALUES (?,?,?,?)",
            (run_id, kind, title, body),
        )
        self.conn.commit()

    def fts_delete_run(self, run_id: str) -> None:
        self.conn.execute("DELETE FROM fts WHERE run_id = ?", (run_id,))
        self.conn.commit()

    def fts_search(self, query: str, limit: int = 20) -> list[sqlite3.Row]:
        # Quote each term so user input can't hit fts5 query syntax errors.
        terms = " ".join(f'"{t}"' for t in query.replace('"', " ").split() if t)
        if not terms:
            return []
        # Highlight with control-char sentinels, never raw HTML: the indexed
        # body is text lifted from fetched pages, and sqlite does not escape
        # it. app.web.markdown.highlight_snippet turns these into <mark> after
        # the surrounding text has been escaped.
        return self.conn.execute(
            "SELECT run_id, kind, title,"
            f" snippet(fts, 3, '{FTS_MARK_OPEN}', '{FTS_MARK_CLOSE}',"
            " ' … ', 16) AS snip"
            " FROM fts WHERE fts MATCH ? ORDER BY rank LIMIT ?",
            (terms, limit),
        ).fetchall()

    def delete_run_index(self, run_id: str) -> None:
        """Remove derived index data for a run (used by reindex)."""
        self.conn.execute(
            "DELETE FROM run_links WHERE src_run_id = ? OR dst_run_id = ?",
            (run_id, run_id),
        )
        self.conn.execute("DELETE FROM fts WHERE run_id = ?", (run_id,))
        self.conn.commit()
