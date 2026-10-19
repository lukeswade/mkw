from datetime import datetime, timedelta

import httpx
import pytest
import respx

from app.research.searcher import (
    RECENCY_CUTOFF_DAYS,
    RECENCY_TO_TIME_RANGE,
    Searcher,
    SearxngError,
    categories_for,
    cutoff_for,
    parse_published,
)

BASE = "http://sx.test"


def test_recency_tables_cover_all_options():
    options = {"week", "month", "3months", "6months", "1year", "3years", "all"}
    assert set(RECENCY_TO_TIME_RANGE) == options
    assert set(RECENCY_CUTOFF_DAYS) == options
    assert RECENCY_TO_TIME_RANGE["week"] == "week"
    assert RECENCY_TO_TIME_RANGE["3months"] == "year"
    assert RECENCY_TO_TIME_RANGE["3years"] is None
    assert RECENCY_CUTOFF_DAYS["6months"] == 186
    # engines don't reliably honor time_range → every window has a cutoff
    assert all(RECENCY_CUTOFF_DAYS[r] for r in options - {"all"})


def test_cutoff_for():
    now = datetime(2026, 8, 9)
    assert cutoff_for("all", now) is None
    assert cutoff_for("week", now) == now - timedelta(days=8)
    assert cutoff_for("3months", now) == now - timedelta(days=93)


def test_categories_for():
    # science/it reach engines that don't CAPTCHA and are better research
    # sources; news is added only where freshness is the point
    assert categories_for("week").endswith(",news")
    assert "science" in categories_for("1year")
    assert categories_for("1year") == "general,science"


def test_parse_published():
    assert parse_published("2026-01-02T10:00:00Z") == datetime(2026, 1, 2, 10)
    assert parse_published("2026-01-02T10:00:00+02:00").hour == 10
    assert parse_published("not a date") is None
    assert parse_published(None) is None


@respx.mock
async def test_search_prefilters_dated_results():
    old = (datetime.now() - timedelta(days=400)).isoformat()
    respx.get(f"{BASE}/search").mock(return_value=httpx.Response(200, json={
        "results": [
            {"url": "https://a.com/new", "title": "New", "content": "c",
             "engine": "g", "publishedDate": datetime.now().isoformat()},
            {"url": "https://a.com/old", "title": "Old", "content": "c",
             "engine": "g", "publishedDate": old},
            {"url": "https://a.com/undated", "title": "Undated", "content": "c",
             "engine": "g", "publishedDate": None},
            {"url": "ftp://a.com/bad-scheme", "title": "Bad", "content": "c",
             "engine": "g"},
        ],
        "unresponsive_engines": [["brave", "too many requests"]],
    }))
    async with httpx.AsyncClient() as client:
        results = await Searcher(BASE, client).search("q", "6months")
    urls = [r.url for r in results]
    assert "https://a.com/new" in urls
    assert "https://a.com/undated" in urls      # undated kept, tagged later
    assert "https://a.com/old" not in urls      # dated outside window dropped
    assert all(u.startswith("http") for u in urls)


@respx.mock
async def test_search_time_range_param():
    route = respx.get(f"{BASE}/search").mock(
        return_value=httpx.Response(200, json={"results": []}))
    async with httpx.AsyncClient() as client:
        await Searcher(BASE, client).search("q", "month")
        assert route.calls.last.request.url.params["time_range"] == "month"
        await Searcher(BASE, client).search("q", "all")
        assert "time_range" not in route.calls.last.request.url.params


@respx.mock
async def test_search_403_gives_settings_hint():
    respx.get(f"{BASE}/search").mock(return_value=httpx.Response(403))
    async with httpx.AsyncClient() as client:
        with pytest.raises(SearxngError, match="formats"):
            await Searcher(BASE, client).search("q", "all")


# ---- a project's own page is not dated by when the project began ------------

def test_a_repository_or_model_card_is_a_living_page():
    """44 such pages were skipped as "outside recency window" in past runs,
    among them ggml-org/llama.cpp (2023-03-10) and ollama/ollama (2023-06-26):
    the date a reader extracts from them is when the project started."""
    from app.research.dedupe import is_living_page
    for url in ("https://github.com/ggml-org/llama.cpp",
                "https://github.com/SYSTRAN/faster-whisper/",
                "https://github.com/jundot/omlx/blob/main/README.md",
                "https://github.com/jundot/omlx/releases",
                "https://gitlab.com/owner/project",
                "https://huggingface.co/Qwen/Qwen3-8B",
                "https://huggingface.co/datasets/org/name",
                "https://huggingface.co/spaces/org/demo"):
        assert is_living_page(url), url
    # dated events, a profile, a blog and anything elsewhere keep their dates
    for url in ("https://github.com/ml-explore/mlx-lm/issues/1060",
                "https://github.com/ggml-org/llama.cpp/pull/9999",
                "https://github.com/ggml-org/llama.cpp/discussions/42",
                "https://github.com/ggml-org/llama.cpp/commit/abc123",
                "https://github.com/ggml-org",
                "https://huggingface.co/blog/ettin-reranker",
                "https://huggingface.co/papers/2501.00001",
                "https://example.com/owner/repo"):
        assert not is_living_page(url), url


@respx.mock
async def test_an_old_repository_page_is_read_and_an_old_post_is_not(data_dir):
    from app.db import Repo, connect
    from app.models import RunParams
    from app.research.orchestrator import Orchestrator
    from app.research.progress import ProgressBus
    from tests.fake_llm import FakeLLM
    from tests.test_pipeline_e2e import SX, article, make_cfg, script, sx_payload, sx_result
    cfg = make_cfg(data_dir)

    def dated(title):     # the extractor finds the project's first date
        return article(title).replace(
            "<head>", '<head><meta property="article:published_time" content="2023-03-10">')

    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload([
        sx_result("https://github.com/ggml-org/llama.cpp", "llama.cpp repository"),
        sx_result("https://oldblog.example.com/llama-post", "An old llama.cpp post")])))
    respx.get("https://github.com/ggml-org/llama.cpp").mock(
        return_value=httpx.Response(200, html=dated("llama.cpp repository")))
    respx.get("https://oldblog.example.com/llama-post").mock(
        return_value=httpx.Response(200, html=dated("An old llama.cpp post")))
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(), llm_factory=lambda: FakeLLM(
        script([{"state_md": "s", "saturated": True, "next_queries": []}])))
    rid = orch.enqueue(RunParams(query="llama.cpp vulkan backend", depth=1,
                                 recency="1year", origin="cli"))
    await orch.execute_now(rid)
    kept = {f["url"] for f in repo.findings_for_run(rid)}
    assert kept == {"https://github.com/ggml-org/llama.cpp"}
    events = (cfg.research_dir / rid / "events.jsonl").read_text()
    assert "outside recency window (2023-03-10)" in events      # the post, still dated
