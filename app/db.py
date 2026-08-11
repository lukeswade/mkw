"""SQLite persistence: WAL connection, versioned migrations, query helpers.

Single-process, single-event-loop usage — the sync sqlite3 driver is fine at
this write volume (a few rows per research round).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _schema() -> str:
    return (Path(__file__).with_name("schema.sql")).read_text()


# One entry per schema version; index i migrates user_version i → i+1.
def _migrations() -> list[str]:
    return [
        _schema(),
        "ALTER TABLE runs ADD COLUMN evergreen BOOLEAN NOT NULL DEFAULT 0;"
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
        conn.executescript(migrations[i])
        conn.execute(f"PRAGMA user_version = {i + 1}")
    conn.commit()


_RUN_COLS = {
    "query", "title", "depth", "recency", "status", "dir", "parent_run_id",
    "origin", "origin_chat_id", "error", "stop_reason", "stats_json",
    "evergreen", "created_at", "started_at", "finished_at",
}


class Repo:
    """All SQL lives here."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # ---- runs -------------------------------------------------------------
    def create_run(self, *, run_id: str, query: str, depth: int, recency: str,
                   dir: str, origin: str = "web", parent_run_id: str | None = None,
                   origin_chat_id: int | None = None, status: str = "queued", evergreen: bool = False) -> None:
        self.conn.execute(
            "INSERT INTO runs (id, query, depth, recency, status, dir, origin,"
            " parent_run_id, origin_chat_id, evergreen, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, query, depth, recency, status, dir, origin,
             parent_run_id, origin_chat_id, evergreen, utcnow()),
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
            "SELECT * FROM runs WHERE evergreen = 1 AND status = 'completed' ORDER BY created_at"
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

    def findings_for_run(self, run_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM findings WHERE run_id = ? ORDER BY idx", (run_id,)
        ).fetchall()

    # ---- entities & graph ----------------------------------------------------
    def upsert_entity(self, name: str, name_norm: str, type_: str,
                      description: str) -> int:
        row = self.conn.execute(
            "SELECT id FROM entities WHERE name_norm = ?", (name_norm,)
        ).fetchone()
        if row:
            return int(row["id"])
        cur = self.conn.execute(
            "INSERT INTO entities (name, name_norm, type, description, created_at)"
            " VALUES (?,?,?,?,?)",
            (name, name_norm, type_, description, utcnow()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def set_run_entity(self, run_id: str, entity_id: int, salience: float) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO run_entities (run_id, entity_id, salience)"
            " VALUES (?,?,?)",
            (run_id, entity_id, salience),
        )
        self.conn.commit()

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

    def graph_rows(self) -> tuple[list[sqlite3.Row], list[sqlite3.Row], list[sqlite3.Row]]:
        runs = self.conn.execute(
            "SELECT r.id, r.title, r.query, r.status,"
            " (SELECT COUNT(*) FROM findings f WHERE f.run_id = r.id) AS n_findings"
            " FROM runs r WHERE r.status IN ('completed','interrupted','cancelled')"
        ).fetchall()
        entities = self.conn.execute(
            "SELECT e.id, e.name, e.type, e.description, re.run_id, re.salience"
            " FROM entities e JOIN run_entities re ON re.entity_id = e.id"
        ).fetchall()
        links = self.conn.execute("SELECT * FROM run_links").fetchall()
        return runs, entities, links

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
        return self.conn.execute(
            "SELECT run_id, kind, title,"
            " snippet(fts, 3, '<mark>', '</mark>', ' … ', 16) AS snip"
            " FROM fts WHERE fts MATCH ? ORDER BY rank LIMIT ?",
            (terms, limit),
        ).fetchall()

    def delete_run_index(self, run_id: str) -> None:
        """Remove derived index data for a run (used by reindex)."""
        self.conn.execute("DELETE FROM run_entities WHERE run_id = ?", (run_id,))
        self.conn.execute(
            "DELETE FROM run_links WHERE src_run_id = ? OR dst_run_id = ?",
            (run_id, run_id),
        )
        self.conn.execute("DELETE FROM fts WHERE run_id = ?", (run_id,))
        self.conn.commit()
