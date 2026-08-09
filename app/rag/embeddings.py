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
