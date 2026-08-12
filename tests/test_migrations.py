"""Migrations must be idempotent: schema.sql already contains every column, so
a fresh database runs step 0 AND every later step. These tests cover the three
states a real deployment can be in."""
import sqlite3

from app.db import Repo, _migrations, _schema, connect, migrate

CURRENT_VERSION = len(_migrations())


def _legacy_v1_db(path) -> None:
    """A database created before the `evergreen` column existed."""
    conn = sqlite3.connect(str(path))
    conn.executescript(_schema().replace(
        "  evergreen      BOOLEAN NOT NULL DEFAULT 0,\n", ""))
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()


def _columns(conn, table="runs") -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def test_fresh_database_boots(data_dir):
    conn = connect(data_dir / "fresh.sqlite3")
    assert conn.execute("PRAGMA user_version").fetchone()[0] == CURRENT_VERSION
    assert "evergreen" in _columns(conn)


def test_legacy_v1_database_upgrades(data_dir):
    path = data_dir / "legacy.sqlite3"
    _legacy_v1_db(path)
    conn = connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == CURRENT_VERSION
    assert "evergreen" in _columns(conn)


def test_migrate_is_reentrant(data_dir):
    """Re-running against an already-current DB must be a no-op, not an error."""
    conn = connect(data_dir / "again.sqlite3")
    migrate(conn)
    migrate(conn)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == CURRENT_VERSION


def test_legacy_rows_survive_upgrade(data_dir):
    path = data_dir / "withdata.sqlite3"
    _legacy_v1_db(path)
    raw = sqlite3.connect(str(path))
    raw.execute(
        "INSERT INTO runs (id, query, depth, recency, status, dir, origin,"
        " created_at) VALUES ('r1','q',2,'all','completed','r1','web','2026-01-01')")
    raw.commit()
    raw.close()

    repo = Repo(connect(path))
    row = repo.get_run("r1")
    assert row is not None
    assert row["evergreen"] == 0  # backfilled by the DEFAULT
