"""Knowledge-layer facade: prior-knowledge lookup, run indexing, cross-run
linking, semantic search, RAG ask, and disk-based reindex.

The .md files in research_data/ are the source of truth — everything here is
a derived index that reindex_all() can rebuild.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

from app.config import Settings, load_settings
from app.db import Repo, utcnow
from app.llm import prompts
from app.rag.chunking import chunk_markdown
from app.rag.embeddings import embedder_id, make_embedder
from app.rag.index import VectorIndex

log = logging.getLogger(__name__)

PRIOR_MIN_SCORE = 0.45
PRIOR_MAX_CHUNKS = 8
LINK_MIN_SCORE = 0.55
LINK_TOP_N = 3
ASK_MIN_SCORE = 0.35
ASK_MAX_CHUNKS = 10
# Chunks any one run may contribute to Ask or to the planner's prior block.
# Top-N by score hands every slot to the richest run: a probe of the live
# index returned 12 of 12 chunks from a single run. The same per-source cap
# already applies per domain, per feed and per source elsewhere in the
# pipeline; this is the same rule for the knowledge layer.
ASK_PER_RUN = 3
PRIOR_PER_RUN = 3


def diversify(hits, *, per_run: int, limit: int, backfill: bool = False) -> list:
    """Best-first, but no run may take more than per_run slots.

    With backfill, slots the other runs could not fill go to the capped
    candidates, best first. The cap is there so one rich run cannot crowd the
    rest out; when nothing else clears the floor it only threw relevant
    chunks away. Measured 2026-09-23 on 160 synthetic Ask questions
    (scripts/eval/retrieval_eval.py): 11 were shown about 3 chunks instead of
    10, and backfill lifted the share shown their source from 64% to 67%.
    """
    taken: dict[str, int] = defaultdict(int)
    out, held = [], []
    for h in hits:
        rid = h.meta.get("run_id", "")
        if taken[rid] >= per_run:
            held.append(h)
            continue
        taken[rid] += 1
        out.append(h)
        if len(out) >= limit:
            return out
    if backfill:
        out += held[:limit - len(out)]
    return out


class RagService:
    """Embeddings + vector index.

    The web app builds this once at boot, but embedding settings can change
    on the Settings page afterwards. The embedder and its index are therefore
    re-resolved whenever the configured model changes — otherwise a running
    server keeps writing vectors with the old model into the old collection
    while everything else uses the new one, silently splitting the library
    across two indexes.
    """

    def __init__(self, cfg: Settings, llm_factory=None):
        import sentence_transformers  # noqa: F401 — fail fast if ML deps absent
        import chromadb                # noqa: F401
        self.cfg = cfg
        self._embedder_id = ""
        self.embedder = None
        self.index = None
        self._refresh()
        self._llm_factory = llm_factory

    def _refresh(self) -> None:
        """Point at whichever embedder the current settings ask for."""
        try:
            cfg = load_settings(self.cfg.data_dir)
        except Exception:
            cfg = self.cfg
        wanted = embedder_id(cfg)
        if wanted == self._embedder_id and self.embedder is not None:
            return
        if self._embedder_id:
            log.info("embedding model changed (%s -> %s); switching index",
                     self._embedder_id, wanted)
        self.cfg = cfg
        self.embedder = make_embedder(cfg)
        self.index = VectorIndex(cfg.chroma_dir, wanted)
        self._embedder_id = wanted

    def _llm(self):
        if self._llm_factory is not None:
            return self._llm_factory()
        from app.llm.client import LLM
        return LLM(load_settings(self.cfg.data_dir))

    # ---- pipeline hook: before planning -------------------------------------
    async def prior_knowledge(self, query: str, exclude_run: str | None = None
                              ) -> tuple[str, list[tuple[str, float]]]:
        self._refresh()
        if self.index.count() == 0:
            return "", []
        emb = await self.embedder.encode_query(query)
        hits = diversify(
            [h for h in self.index.query(emb, n=PRIOR_MAX_CHUNKS * 4)
             if h.meta.get("run_id") != exclude_run
             and h.score >= PRIOR_MIN_SCORE],
            per_run=PRIOR_PER_RUN, limit=PRIOR_MAX_CHUNKS)
        if not hits:
            return "", []
        lines = []
        best_per_run: dict[str, float] = defaultdict(float)
        for h in hits:
            run_id = h.meta.get("run_id", "")
            title = h.meta.get("run_title") or run_id
            best_per_run[run_id] = max(best_per_run[run_id], h.score)
            lines.append(f"- (from “{title}”) {h.text[:600]}")
        related = sorted(
            ((rid, s) for rid, s in best_per_run.items() if s >= LINK_MIN_SCORE),
            key=lambda t: -t[1])[:LINK_TOP_N]
        return "\n".join(lines), related

    async def related_runs(self, query: str, exclude_run: str | None = None
                           ) -> list[tuple[str, float]]:
        """Which earlier runs this question resembles — the sourcing signal,
        usable even when the planner is told nothing about them."""
        _text, related = await self.prior_knowledge(query, exclude_run=exclude_run)
        return related

    # ---- pipeline hook: after synthesis ----------------------------------------
    async def index_run(self, repo: Repo, run_id: str) -> int:
        self._refresh()
        row = repo.get_run(run_id)
        if row is None:
            return 0
        run_dir = self.cfg.research_dir / row["dir"]
        title = row["title"] or row["query"]

        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []

        overview_path = run_dir / "overview.md"
        if overview_path.exists():
            for i, chunk in enumerate(chunk_markdown(overview_path.read_text(encoding="utf-8"))):
                ids.append(f"{run_id}:ov:{i}")
                docs.append(chunk)
                metas.append({"run_id": run_id, "kind": "overview",
                              "run_title": title, "title": title})

        for f in repo.findings_for_run(run_id):
            p = run_dir / f["path"]
            if not p.is_file():
                continue
            for i, chunk in enumerate(chunk_markdown(p.read_text(encoding="utf-8"))):
                ids.append(f"{run_id}:f{f['idx']}:{i}")
                docs.append(chunk)
                metas.append({"run_id": run_id, "kind": "finding",
                              "run_title": title, "title": f["title"] or "",
                              "url": f["url"] or "",
                              "published": f["published_date"] or ""})

        if not ids:
            return 0
        embeddings = await self.embedder.encode_docs(docs)
        self.index.delete_run(run_id)
        self.index.add(ids, embeddings, docs, metas)
        await self._link_similar(repo, run_id, ids, embeddings, metas)
        return len(ids)

    async def _link_similar(self, repo: Repo, run_id: str, ids, embeddings,
                            metas) -> None:
        ov_vecs = [e for e, m in zip(embeddings, metas)
                   if m["kind"] == "overview"]
        if not ov_vecs:
            return
        centroid = [sum(col) / len(ov_vecs) for col in zip(*ov_vecs)]
        hits = self.index.query(centroid, n=20, where={
            "$and": [{"kind": "overview"}, {"run_id": {"$ne": run_id}}]})
        best: dict[str, float] = defaultdict(float)
        for h in hits:
            rid = h.meta.get("run_id", "")
            best[rid] = max(best[rid], h.score)
        for rid, score in sorted(best.items(), key=lambda t: -t[1])[:LINK_TOP_N]:
            if score >= LINK_MIN_SCORE and repo.get_run(rid) is not None:
                repo.add_run_link(run_id, rid, "similar", score)

    # ---- library semantic mode ---------------------------------------------------
    async def semantic_search(self, query: str, limit: int = 20) -> list[dict]:
        self._refresh()
        emb = await self.embedder.encode_query(query)
        hits = self.index.query(emb, n=limit)
        return [{"run_id": h.meta.get("run_id", ""), "text": h.text[:400],
                 "score": h.score, "kind": h.meta.get("kind", ""),
                 "title": h.meta.get("title", "")} for h in hits]

    # ---- ask ------------------------------------------------------------------------
    async def ask(self, question: str, repo: Repo) -> dict:
        self._refresh()
        emb = await self.embedder.encode_query(question)
        hits = diversify(
            [h for h in self.index.query(emb, n=ASK_MAX_CHUNKS * 3)
             if h.score >= ASK_MIN_SCORE],
            per_run=ASK_PER_RUN, limit=ASK_MAX_CHUNKS, backfill=True)
        if not hits:
            return {"answer": "The research corpus doesn't cover this yet — "
                              "try running a research on it first.",
                    "sources": []}
        excerpt_lines = []
        best_per_run: dict[str, float] = defaultdict(float)
        titles: dict[str, str] = {}
        for h in hits:
            rid = h.meta.get("run_id", "")
            row = repo.get_run(rid)
            title = (row["title"] if row and row["title"] else
                     h.meta.get("run_title") or rid)
            titles[rid] = title
            best_per_run[rid] = max(best_per_run[rid], h.score)
            excerpt_lines.append(f"[run: {title}]\n{h.text}\n")
        prompt = prompts.ASK.format(question=question,
                                    excerpts="\n".join(excerpt_lines))
        llm = self._llm()
        answer = await llm.chat("ask", [{"role": "user", "content": prompt}],
                                max_tokens=2500, temperature=0.3)
        sources = [{"run_id": rid, "title": titles[rid], "score": score}
                   for rid, score in
                   sorted(best_per_run.items(), key=lambda t: -t[1])]
        return {"answer": answer, "sources": sources}

    # ---- reindex from disk -------------------------------------------------------------
    async def reindex_all(self, repo: Repo) -> int:
        self._refresh()
        count = 0
        for run_dir in sorted(self.cfg.research_dir.iterdir()):
            if not (run_dir / "meta.json").is_file():
                continue
            run_id = run_dir.name
            self._restore_db_rows(repo, run_dir, run_id)
            repo.fts_delete_run(run_id)
            row = repo.get_run(run_id)
            if row is None:
                continue
            title = row["title"] or row["query"]
            overview_path = run_dir / "overview.md"
            if overview_path.exists():
                repo.fts_add(run_id, "overview", title,
                             overview_path.read_text(encoding="utf-8"))
            for f in repo.findings_for_run(run_id):
                p = run_dir / f["path"]
                if p.is_file():
                    repo.fts_add(run_id, "finding", f["title"] or "",
                                 p.read_text(encoding="utf-8"))
            await self.index_run(repo, run_id)
            count += 1
        return count

    def _restore_db_rows(self, repo: Repo, run_dir: Path, run_id: str) -> None:
        """Recreate runs/findings rows from disk when the DB was lost."""
        import json
        if repo.get_run(run_id) is not None:
            return
        try:
            meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        log.info("restoring DB row for %s from meta.json", run_id)
        repo.create_run(
            run_id=run_id, query=meta.get("query", run_id),
            depth=int(meta.get("depth", 1)), recency=meta.get("recency", "all"),
            dir=run_id, origin=meta.get("origin", "web"),
            parent_run_id=meta.get("parent_run_id"),
            status=meta.get("status", "completed"))
        repo.update_run(run_id, title=meta.get("title"),
                        stop_reason=meta.get("stop_reason"),
                        finished_at=meta.get("finished_at"))
        for p in sorted((run_dir / "findings").glob("*.md")):
            parsed = _parse_finding_md(p.read_text(encoding="utf-8"))
            if parsed:
                try:
                    repo.add_finding(run_id=run_id, path=f"findings/{p.name}",
                                     **parsed)
                except Exception:
                    log.debug("could not restore finding %s", p, exc_info=True)


_FINDING_HEAD_RE = re.compile(r"^#\s*\[(\d+)\]\s*(.+)$", re.MULTILINE)


def _field(md: str, name: str) -> str:
    m = re.search(rf"^\-\s*\*\*{name}:\*\*\s*(.+)$", md, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _parse_finding_md(md: str) -> dict | None:
    head = _FINDING_HEAD_RE.search(md)
    if not head:
        return None
    published = _field(md, "Published")
    relevance = re.sub(r"/10.*", "", _field(md, "Relevance"))
    summary_m = re.search(r"^\*\*Summary:\*\*\s*(.+)$", md, re.MULTILINE)
    url = _field(md, "URL")
    return {
        "idx": int(head.group(1)),
        "title": head.group(2).strip(),
        "url": url,
        "domain": _field(md, "Domain"),
        "published_date": None if published in ("", "unknown") else published,
        "relevance": float(relevance) if relevance.replace(".", "").isdigit() else 0,
        "summary": summary_m.group(1).strip() if summary_m else "",
    }
