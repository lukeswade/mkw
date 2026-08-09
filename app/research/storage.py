"""Run directory management and atomic file writing.

The .md files on disk are the source of truth — SQLite/Chroma are derived
indexes that `python -m app.cli reindex` can rebuild from these directories.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path

# Filenames the web layer may serve from a run directory (anything else → 404).
SERVABLE_RE = re.compile(
    r"^(overview\.md|further-research\.md|sources\.md|meta\.json|events\.jsonl"
    r"|findings/\d{3}_[a-z0-9-]*\.md|rounds/round-\d{2}\.md)$"
)

_CITATION_RE = re.compile(r"\[(\d{1,3})\]")


def slugify(text: str, max_len: int = 40) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    text = re.sub(r"-{2,}", "-", text)
    return text[:max_len].rstrip("-") or "research"


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def validate_citations(md: str, n_sources: int) -> tuple[str, set[int]]:
    """Strip citation markers pointing beyond the bibliography ([7] when n=5)."""
    removed: set[int] = set()

    def repl(m: re.Match) -> str:
        k = int(m.group(1))
        if 1 <= k <= n_sources:
            return m.group(0)
        removed.add(k)
        return ""

    return _CITATION_RE.sub(repl, md), removed


class RunStore:
    """Filesystem layout of one research run."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir)

    @classmethod
    def create(cls, research_dir: Path, query: str,
               now: datetime | None = None) -> "RunStore":
        now = now or datetime.now()
        base = f"{now.strftime('%Y%m%d_%H%M%S')}_{slugify(query)}"
        d = research_dir / base
        i = 2
        while d.exists():
            d = research_dir / f"{base}-{i}"
            i += 1
        (d / "findings").mkdir(parents=True)
        (d / "rounds").mkdir()
        return cls(d)

    @property
    def run_id(self) -> str:
        return self.dir.name

    # --- well-known paths ---
    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    @property
    def overview_path(self) -> Path:
        return self.dir / "overview.md"

    @property
    def further_path(self) -> Path:
        return self.dir / "further-research.md"

    @property
    def sources_path(self) -> Path:
        return self.dir / "sources.md"

    @property
    def events_path(self) -> Path:
        return self.dir / "events.jsonl"

    # --- writers ---
    def write_meta(self, meta: dict) -> None:
        atomic_write_text(self.meta_path, json.dumps(meta, indent=2, ensure_ascii=False))

    def read_meta(self) -> dict:
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def update_meta(self, **kv) -> dict:
        meta = self.read_meta()
        meta.update(kv)
        self.write_meta(meta)
        return meta

    def finding_relpath(self, idx: int, title: str) -> str:
        return f"findings/{idx:03d}_{slugify(title, 32)}.md"

    def write_finding(self, idx: int, title: str, md: str) -> str:
        rel = self.finding_relpath(idx, title)
        atomic_write_text(self.dir / rel, md)
        return rel

    def write_round(self, n: int, md: str) -> str:
        rel = f"rounds/round-{n:02d}.md"
        atomic_write_text(self.dir / rel, md)
        return rel

    def write_overview(self, md: str) -> None:
        atomic_write_text(self.overview_path, md)

    def write_further(self, md: str) -> None:
        atomic_write_text(self.further_path, md)

    def write_sources(self, md: str) -> None:
        atomic_write_text(self.sources_path, md)

    def append_event(self, obj: dict) -> None:
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def read_events(self, after_seq: int = 0) -> list[dict]:
        events: list[dict] = []
        try:
            with self.events_path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # torn final line after a crash
                    if e.get("seq", 0) > after_seq:
                        events.append(e)
        except OSError:
            pass
        return events
