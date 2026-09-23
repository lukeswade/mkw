"""Local embeddings via sentence-transformers (BAAI/bge-small-en-v1.5).

The model is baked into the Docker image and loaded lazily — web-only actions
never pay the load cost. Encoding runs in a thread so the event loop stays
responsive. The loaded model is cached at class level so every RagService
(and every test) shares one instance.
"""
from __future__ import annotations

import asyncio
import logging

log = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
# bge v1.5 English retrieval instruction — queries only, documents encoded bare
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    _models: dict[str, object] = {}

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name

    def _model(self):
        if self.model_name not in self._models:
            from sentence_transformers import SentenceTransformer
            log.info("loading embedding model %s", self.model_name)
            self._models[self.model_name] = SentenceTransformer(self.model_name)
        return self._models[self.model_name]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._model()
        vecs = model.encode(texts, normalize_embeddings=True,
                            show_progress_bar=False, batch_size=32)
        return [v.tolist() for v in vecs]

    async def encode_docs(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, list(texts))

    async def encode_query(self, text: str) -> list[float]:
        return (await asyncio.to_thread(self._encode, [QUERY_PREFIX + text]))[0]


# ---- remote embeddings (OpenAI-compatible /embeddings) -----------------------

# Qwen3-Embedding is instruction-aware on the query side only (documents go
# bare); its model card measures a 1-5% retrieval drop without one. One
# instruction serves every lookup the app makes: Ask, claim checks, the
# similar-run hint and the planner's prior block. scripts/eval/retrieval_eval.py
# uses the same string, so its numbers are what the app would do.
QWEN3_QUERY_INSTRUCT = ("Instruct: Given a research question or a claim, retrieve "
                        "research notes that answer or verify it\nQuery: ")

# Retrieval models want their own instruction prefixes; using the wrong ones
# (or none) measurably degrades recall. Keyed by substring of the model id.
_PREFIXES: tuple[tuple[str, str, str], ...] = (
    ("nomic", "search_query: ", "search_document: "),
    ("modernbert", "search_query: ", "search_document: "),
    ("bge", QUERY_PREFIX, ""),
    ("e5", "query: ", "passage: "),
    ("gte", "", ""),
    ("qwen3-emb", QWEN3_QUERY_INSTRUCT, ""),
)


def prefixes_for(model: str) -> tuple[str, str]:
    m = (model or "").lower()
    for key, q, d in _PREFIXES:
        if key in m:
            return q, d
    return "", ""


class RemoteEmbedder:
    """Embeddings from an OpenAI-compatible /embeddings endpoint.

    Lets a local inference server (oMLX, LM Studio, llama.cpp) supply a
    stronger retrieval model than the small one baked into the image — often
    one that is already resident in memory for other work.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "",
                 timeout: float = 120.0, batch_size: int = 32):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.batch_size = max(1, batch_size)
        self.query_prefix, self.doc_prefix = prefixes_for(model)

    async def _embed(self, inputs: list[str]) -> list[list[float]]:
        import httpx
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        out: list[list[float]] = []
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for i in range(0, len(inputs), self.batch_size):
                batch = inputs[i:i + self.batch_size]
                resp = await client.post(f"{self.base_url}/embeddings",
                                         headers=headers,
                                         json={"model": self.model,
                                               "input": batch})
                resp.raise_for_status()
                data = resp.json().get("data") or []
                if len(data) != len(batch):
                    raise RuntimeError(
                        f"embedding endpoint returned {len(data)} vectors "
                        f"for {len(batch)} inputs")
                for item in sorted(data, key=lambda d: d.get("index", 0)):
                    out.append([float(x) for x in item["embedding"]])
        return out

    async def encode_docs(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await self._embed([self.doc_prefix + t for t in texts])

    async def encode_query(self, text: str) -> list[float]:
        return (await self._embed([self.query_prefix + text]))[0]


def make_embedder(cfg):
    """Remote embedder when a model is configured, else the baked-in one."""
    model = (getattr(cfg, "embedding_model", "") or "").strip()
    if not model:
        return Embedder()
    base = (getattr(cfg, "embedding_base_url", "") or "").strip() \
        or cfg.resolved_base_url
    key = (getattr(cfg, "embedding_api_key", "") or "").strip() \
        or cfg.resolved_api_key
    log.info("embeddings via %s (model %s)", base, model)
    return RemoteEmbedder(base, model, key)


def embedder_id(cfg) -> str:
    """Short identity of the active embedding model.

    The vector index is namespaced by this: two models produce incompatible
    vector spaces (and usually different dimensions), so they must not share
    a collection. Switching models is then non-destructive and reversible —
    the old collection stays put until you switch back.
    """
    import re as _re
    model = (getattr(cfg, "embedding_model", "") or "").strip() or MODEL_NAME
    return _re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")[:48]
