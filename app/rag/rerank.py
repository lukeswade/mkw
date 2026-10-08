"""Reranking through the model server's /rerank endpoint (oMLX, vLLM, ...).

A reranker reads a question and one passage together and scores the pair;
embeddings encode them apart and compare afterwards. Measured 2026-10-07 on
this library (docs/retrieval-eval-2026-09-23.md): Qwen3-Reranker-0.6B over the
top 50 library chunks raised claim checks shown the right source from 74% to
85%, at about 5 s per 50 passages on an M4 Pro.

A reranker is an improvement, never a dependency: when the server cannot
answer, callers get None and keep their own order.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_BATCH = 64       # passages per request; the measured calls carried 50


class Reranker:
    def __init__(self, base_url: str, model: str, api_key: str = "",
                 timeout: float = 90.0):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    async def scores(self, query: str, docs: list[str]) -> list[float] | None:
        """Relevance of each doc to the query, in the docs' order, or None."""
        if not docs:
            return []
        import httpx
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        out: list[float | None] = [None] * len(docs)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                for start in range(0, len(docs), _BATCH):
                    batch = docs[start:start + _BATCH]
                    resp = await client.post(
                        f"{self.base_url}/rerank", headers=headers,
                        json={"model": self.model, "query": query,
                              "documents": batch, "top_n": len(batch)})
                    resp.raise_for_status()
                    body = resp.json()
                    for item in body.get("results") or body.get("data") or []:
                        i = int(item.get("index", -1))
                        if 0 <= i < len(batch):
                            out[start + i] = float(
                                item.get("relevance_score", item.get("score", 0.0)))
        except Exception as e:
            log.warning("rerank via %s failed: %s", self.model, e)
            return None
        if any(s is None for s in out):
            log.warning("rerank via %s scored %d of %d passages", self.model,
                        sum(s is not None for s in out), len(out))
            return None
        return out  # type: ignore[return-value]

    async def order(self, query: str, docs: list[str]) -> list[int] | None:
        """Indices of docs, best first, or None when there are no scores."""
        scores = await self.scores(query, docs)
        if scores is None:
            return None
        return sorted(range(len(docs)), key=lambda i: -scores[i])


def make_reranker(cfg, feature: str) -> Reranker | None:
    """The reranker for one feature ("claims", "search", "passages"), or None.

    Off unless a model is named and the feature's switch is on. It uses the
    embedding endpoint, which falls back to the LLM's, like the embedder.
    """
    model = (getattr(cfg, "rerank_model", "") or "").strip()
    if not model or not getattr(cfg, f"rerank_{feature}", False):
        return None
    base = (getattr(cfg, "embedding_base_url", "") or "").strip() \
        or cfg.resolved_base_url
    key = (getattr(cfg, "embedding_api_key", "") or "").strip() \
        or cfg.resolved_api_key
    return Reranker(base, model, key)
