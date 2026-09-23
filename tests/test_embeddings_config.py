"""Pluggable embeddings: remote endpoint, prefixes, index namespacing."""
from __future__ import annotations

import httpx
import pytest
import respx

from app.config import Settings
from app.rag.embeddings import (Embedder, RemoteEmbedder, embedder_id,
                                make_embedder, prefixes_for)


def test_prefixes_match_the_model_family():
    assert prefixes_for("nomicai-modernbert-embed-base-bf16") == (
        "search_query: ", "search_document: ")
    assert prefixes_for("BAAI/bge-small-en-v1.5")[1] == ""
    assert prefixes_for("intfloat/e5-large") == ("query: ", "passage: ")
    assert prefixes_for("something-unknown") == ("", "")


def test_qwen3_embedding_queries_carry_its_instruction():
    """Qwen measures a 1-5% retrieval drop without the query instruction;
    documents are embedded bare."""
    q, d = prefixes_for("mlx-community/Qwen3-Embedding-4B-4bit-DWQ")
    assert q.startswith("Instruct: ") and q.endswith("\nQuery: ")
    assert d == ""
    assert prefixes_for("Qwen3-Embedding-0.6B")[0] == q


def test_factory_defaults_to_the_baked_in_model(data_dir):
    cfg = Settings(data_dir=str(data_dir))
    assert isinstance(make_embedder(cfg), Embedder)


def test_factory_uses_remote_when_a_model_is_named(data_dir):
    cfg = Settings(data_dir=str(data_dir), llm_provider="local",
                   local_llm_base_url="http://mlx.test/v1",
                   local_llm_api_key="sk-x",
                   embedding_model="nomicai-modernbert-embed-base-bf16")
    e = make_embedder(cfg)
    assert isinstance(e, RemoteEmbedder)
    assert e.base_url == "http://mlx.test/v1"      # falls back to the LLM endpoint
    assert e.api_key == "sk-x"
    assert e.query_prefix == "search_query: "


def test_index_namespace_changes_with_the_model(data_dir):
    local = Settings(data_dir=str(data_dir))
    remote = Settings(data_dir=str(data_dir),
                      embedding_model="nomicai-modernbert-embed-base-bf16")
    a, b = embedder_id(local), embedder_id(remote)
    assert a and b and a != b
    assert "/" not in b and " " not in b


@respx.mock
async def test_remote_embedder_batches_and_applies_prefixes():
    seen: list[dict] = []

    def handler(req):
        import json as _json
        body = _json.loads(req.content)
        seen.append(body)
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": [0.1, 0.2, 0.3]}
            for i in range(len(body["input"]))]})

    respx.post("http://mlx.test/v1/embeddings").mock(side_effect=handler)
    e = RemoteEmbedder("http://mlx.test/v1", "nomic-embed", batch_size=2)

    docs = await e.encode_docs(["alpha", "beta", "gamma"])
    assert len(docs) == 3 and len(docs[0]) == 3
    assert [len(b["input"]) for b in seen] == [2, 1]          # batched
    assert seen[0]["input"][0] == "search_document: alpha"    # doc prefix

    seen.clear()
    await e.encode_query("how tight?")
    assert seen[0]["input"] == ["search_query: how tight?"]   # query prefix


@respx.mock
async def test_remote_embedder_rejects_a_short_response():
    respx.post("http://mlx.test/v1/embeddings").mock(
        return_value=httpx.Response(200, json={"data": [
            {"index": 0, "embedding": [0.1]}]}))
    e = RemoteEmbedder("http://mlx.test/v1", "nomic-embed")
    with pytest.raises(RuntimeError, match="returned 1 vectors"):
        await e.encode_docs(["a", "b"])


def test_long_queries_are_accepted():
    from app.models import RunParams
    q = "Research this in depth. " * 300          # ~7200 chars
    assert 2000 < len(q) <= 8000
    assert RunParams(query=q, depth=3).query.startswith("Research")
    with pytest.raises(Exception):
        RunParams(query="x" * 8001, depth=3)
