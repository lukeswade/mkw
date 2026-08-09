"""Chroma persistent vector store.

Embeddings are always supplied explicitly — Chroma's default embedding
function would download a model at runtime, breaking the offline guarantee.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

COLLECTION = "research"


@dataclass
class Hit:
    id: str
    text: str
    meta: dict
    score: float  # cosine similarity (1 - distance)


class VectorIndex:
    def __init__(self, chroma_dir: Path):
        self.chroma_dir = Path(chroma_dir)
        self._coll = None

    def _collection(self):
        if self._coll is None:
            import chromadb
            client = chromadb.PersistentClient(
                path=str(self.chroma_dir),
                settings=chromadb.Settings(anonymized_telemetry=False))
            self._coll = client.get_or_create_collection(
                COLLECTION, metadata={"hnsw:space": "cosine"})
        return self._coll

    def count(self) -> int:
        return self._collection().count()

    def add(self, ids: list[str], embeddings: list[list[float]],
            documents: list[str], metadatas: list[dict]) -> None:
        if not ids:
            return
        self._collection().add(ids=ids, embeddings=embeddings,
                               documents=documents, metadatas=metadatas)

    def delete_run(self, run_id: str) -> None:
        self._collection().delete(where={"run_id": run_id})

    def query(self, embedding: list[float], n: int = 10,
              where: dict | None = None) -> list[Hit]:
        coll = self._collection()
        if coll.count() == 0:
            return []
        res = coll.query(query_embeddings=[embedding],
                         n_results=min(n, coll.count()), where=where,
                         include=["documents", "metadatas", "distances"])
        hits: list[Hit] = []
        for i, hid in enumerate(res["ids"][0]):
            hits.append(Hit(
                id=hid,
                text=res["documents"][0][i] or "",
                meta=res["metadatas"][0][i] or {},
                score=round(1.0 - float(res["distances"][0][i]), 4),
            ))
        return hits
