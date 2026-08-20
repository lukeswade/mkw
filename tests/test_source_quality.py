"""The source-quality batch: old-reddit rewrite, login-walled skip, reddit
thread reading via the .json API, YouTube caption transcripts via InnerTube,
and near-duplicate collapse."""
from __future__ import annotations

import json

import httpx
import pytest
import respx

from app.db import Repo, connect
from app.models import RunParams
from app.research.dedupe import similarity, text_fingerprint
from app.research.extractor import extract
from app.research.fetcher import Fetched, Fetcher, SkipReason, rewrite_host
from app.research.orchestrator import Orchestrator
from app.research.progress import ProgressBus
from app.research.reddit import _json_url, is_thread
from app.research.youtube import _caption_text, _pick_track, video_id
from tests.fake_llm import FakeLLM
from tests.test_next_level import _script
from tests.test_pipeline_e2e import SX, article, make_cfg, sx_payload, sx_result


# ---- reddit rewrite + login-walled skip --------------------------------------

def test_rewrite_host_targets_old_reddit():
    assert rewrite_host(
        "https://www.reddit.com/r/GXOR/comments/abc/plugs/?share_id=x"
    ) == "https://old.reddit.com/r/GXOR/comments/abc/plugs/?share_id=x"
    assert rewrite_host("https://old.reddit.com/r/GXOR/") == \
        "https://old.reddit.com/r/GXOR/"
    assert rewrite_host("https://example.com/reddit.com") == \
        "https://example.com/reddit.com"


def _fetcher(data_dir) -> tuple[Fetcher, httpx.AsyncClient]:
    cfg = make_cfg(data_dir)
    client = httpx.AsyncClient()
    return Fetcher(cfg, client), client


@respx.mock
async def test_reddit_fetches_go_to_old_reddit(data_dir):
    # only the old.reddit route exists — hitting www.reddit would blow up
    respx.get("https://old.reddit.com/r/GXOR/comments/abc/plugs").mock(
        return_value=httpx.Response(200, html=article("Plug thread")))
    fetcher, client = _fetcher(data_dir)
    fetched = await fetcher.fetch(
        "https://www.reddit.com/r/GXOR/comments/abc/plugs")
    await client.aclose()
    assert fetched.url.startswith("https://old.reddit.com/")


@respx.mock
async def test_redirect_back_to_www_reddit_is_rewritten_again(data_dir):
    respx.get("https://l.example.com/x").mock(return_value=httpx.Response(
        302, headers={"location": "https://www.reddit.com/r/GXOR/top"}))
    respx.get("https://old.reddit.com/r/GXOR/top").mock(
        return_value=httpx.Response(200, html=article("Top thread")))
    fetcher, client = _fetcher(data_dir)
    fetched = await fetcher.fetch("https://l.example.com/x")
    await client.aclose()
    assert fetched.url.startswith("https://old.reddit.com/")


async def test_login_walled_domains_are_skipped_without_a_fetch(data_dir):
    fetcher, client = _fetcher(data_dir)
    for url in ("https://www.instagram.com/p/abc/",
                "https://www.facebook.com/groups/fish/posts/123/",
                "https://x.com/someone/status/1"):
        with pytest.raises(SkipReason, match="login-walled"):
            await fetcher.fetch(url)
    await client.aclose()


# ---- reddit threads via the .json API ------------------------------------------

def test_is_thread_and_json_url():
    assert is_thread("https://www.reddit.com/r/GXOR/comments/abc/plugs/")
    assert is_thread("https://old.reddit.com/r/GXOR/comments/abc/plugs")
    assert not is_thread("https://www.reddit.com/r/GXOR/")
    assert not is_thread("https://example.com/r/GXOR/comments/abc/")
    assert _json_url("https://www.reddit.com/r/GXOR/comments/abc/plugs/") == \
        "https://www.reddit.com/r/GXOR/comments/abc/plugs.json?limit=100"


def _thread_json() -> list:
    def comment(body, score, replies=None):
        data = {"body": body, "score": score}
        if replies:
            data["replies"] = {"data": {"children": replies}}
        return {"kind": "t1", "data": data}
    post = {"kind": "t3", "data": {
        "title": "Rear bank plugs on a GX470 — what worked",
        "subreddit": "GXOR", "selftext": "Did all 8 today. Notes below.",
        "created_utc": 1730000000,
        "permalink": "/r/GXOR/comments/abc/rear_bank_plugs/"}}
    comments = [
        comment("Use a 12in extension plus a wobble on cylinder 8; going in "
                "blind from the top is easier than it looks once the coil is "
                "out of the way.", 57,
                replies=[comment("Seconding the wobble joint, the u-joint "
                                 "binds at that angle.", 21)]),
        comment("[deleted]", 2),
        comment("Torque is 13 ft-lb on the 2UZ, do not anti-seize the "
                "modern plated threads.", 33),
    ]
    return [{"data": {"children": [post]}}, {"data": {"children": comments}}]


@respx.mock
async def test_reddit_thread_is_read_through_json_api(data_dir):
    cfg = make_cfg(data_dir)
    thread_url = "https://www.reddit.com/r/GXOR/comments/abc/rear_bank_plugs/"
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result(thread_url, "Rear bank plugs")])))
    # only the old.reddit .json route exists — anything else would blow up
    respx.get("https://old.reddit.com/r/GXOR/comments/abc/rear_bank_plugs.json").mock(
        return_value=httpx.Response(
            200, json=_thread_json(),
            headers={"content-type": "application/json"}))

    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(_script()))
    run_id = orch.enqueue(RunParams(query="gx470 spark plugs", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)

    findings = repo.findings_for_run(run_id)
    assert len(findings) == 1
    f = findings[0]
    assert f["domain"] == "reddit.com"
    assert "(reddit thread)" in f["title"]
    assert f["published_date"] is None or f["published_date"].startswith("2024")


_OLD_REDDIT_HTML = """<html><body>
<div id="siteTable"><div class="thing link">
  <a class="title">Rear bank plugs on a GX470 — what worked</a>
  <div class="expando"><div class="usertext-body">Did all 8 today, notes on
  extensions and torque below for anyone searching later.</div></div>
</div></div>
<div class="commentarea">
  <div class="thing comment"><div class="entry">
    <span class="score">57 points</span>
    <div class="usertext-body">Use a 12in extension plus a wobble on cylinder
    8; going in blind from the top is easier than it looks.</div></div>
    <div class="child"><div class="thing comment"><div class="entry">
      <span class="score">21 points</span>
      <div class="usertext-body">Seconding the wobble joint, the u-joint
      binds at that angle.</div></div></div></div>
  </div>
  <div class="thing comment"><div class="entry">
    <span class="score">33 points</span>
    <div class="usertext-body">Torque is 13 ft-lb on the 2UZ, do not
    anti-seize the plated threads.</div></div></div>
</div></body></html>"""


@respx.mock
async def test_blocked_json_api_falls_back_to_old_reddit_html(data_dir):
    """Reddit revokes anonymous .json access for days at a time while still
    serving HTML — thread reading must survive that."""
    cfg = make_cfg(data_dir)
    thread_url = "https://www.reddit.com/r/GXOR/comments/abc/x/"
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result(thread_url, "Thread")])))
    respx.get("https://old.reddit.com/r/GXOR/comments/abc/x.json").mock(
        return_value=httpx.Response(403))
    respx.get("https://old.reddit.com/r/GXOR/comments/abc/x/").mock(
        return_value=httpx.Response(200, html=_OLD_REDDIT_HTML))
    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(_script()))
    orch_cfg = cfg
    orch_cfg.browser_impersonation = False   # keep the 403 path deterministic
    run_id = orch.enqueue(RunParams(query="gx470 plugs", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)
    findings = repo.findings_for_run(run_id)
    assert len(findings) == 1
    assert "reddit thread" in findings[0]["title"]
    md = (cfg.research_dir / run_id / findings[0]["path"]).read_text()
    assert "wobble" in md or findings[0]["summary"]   # content flowed through


def test_thread_from_html_parses_post_and_nested_comments():
    from app.research.fetcher import Fetched
    from app.research.reddit import _thread_from_html
    page = Fetched(url="u", final_url="u", content_type="text/html",
                   body=_OLD_REDDIT_HTML.encode())
    title, selftext, comments = _thread_from_html(page)
    assert title.startswith("Rear bank plugs")
    assert "notes on" in selftext
    assert len(comments) == 3
    assert comments[0].startswith("[57 points]")
    assert comments[1].startswith("  [21 points]")   # nested reply indented


# ---- youtube: url recognition and caption parsing ------------------------------

def test_video_id_recognizes_video_pages():
    assert video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert video_id("https://youtu.be/dQw4w9WgXcQ?t=42") == "dQw4w9WgXcQ"
    assert video_id("https://m.youtube.com/watch?v=dQw4w9WgXcQ&list=PL1") == "dQw4w9WgXcQ"
    assert video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_video_id_rejects_non_video_pages():
    assert video_id("https://www.youtube.com/@channel") is None
    assert video_id("https://www.youtube.com/playlist?list=PL1") is None
    assert video_id("https://www.youtube.com/watch?v=short") is None
    assert video_id("https://example.com/watch?v=dQw4w9WgXcQ") is None


def test_pick_track_prefers_authored_english():
    tracks = [{"languageCode": "de"},
              {"languageCode": "en", "kind": "asr"},
              {"languageCode": "en-US"}]
    assert _pick_track(tracks) == {"languageCode": "en-US"}
    assert _pick_track(tracks[:2]) == {"languageCode": "en", "kind": "asr"}


def test_caption_text_reads_json3_and_xml():
    json3 = json.dumps({"events": [
        {"segs": [{"utf8": "step one "}, {"utf8": "remove the coil"}]},
        {"segs": []},
        {"segs": [{"utf8": "step two\ntorque to spec"}]},
    ]}).encode()
    assert _caption_text(json3) == \
        "step one remove the coil\nstep two torque to spec"
    srv3 = (b'<?xml version="1.0"?><timedtext format="3"><body>'
            b'<p t="0"><s>step one</s><s> remove the coil</s></p>'
            b'<p t="5">step two</p></body></timedtext>')
    assert _caption_text(srv3) == "step one remove the coil\nstep two"
    fmt1 = (b'<?xml version="1.0"?><transcript>'
            b'<text start="0">it &amp;#39;s easy</text></transcript>')
    assert "easy" in _caption_text(fmt1)
    assert _caption_text(b"garbage") == ""


# ---- youtube: end to end ------------------------------------------------------

def _innertube_response(vid: str) -> dict:
    return {
        "captions": {"playerCaptionsTracklistRenderer": {"captionTracks": [
            {"baseUrl": f"https://www.youtube.com/api/timedtext?v={vid}&lang=en",
             "languageCode": "en", "kind": "asr"},
        ]}},
        "videoDetails": {"title": "2UZ-FE Spark Plug Replacement",
                         "author": "GarageChannel"},
        "microformat": {"playerMicroformatRenderer":
                        {"publishDate": "2025-11-02"}},
    }


def _json3(sentences: list[str]) -> dict:
    return {"events": [{"segs": [{"utf8": s}]} for s in sentences]}


@respx.mock
async def test_youtube_candidate_is_kept_via_its_transcript(data_dir):
    cfg = make_cfg(data_dir)
    vid = "dQw4w9WgXcQ"
    watch = f"https://www.youtube.com/watch?v={vid}"
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result(watch, "Spark plug video")])))
    respx.post("https://www.youtube.com/youtubei/v1/player").mock(
        return_value=httpx.Response(200, json=_innertube_response(vid)))
    respx.get(url__startswith="https://www.youtube.com/api/timedtext").mock(
        return_value=httpx.Response(200, json=_json3(
            [f"step {i}: remove the coil pack and use a long extension "
             f"on the rear bank plug number {i}" for i in range(12)])))

    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(_script()))
    run_id = orch.enqueue(RunParams(query="spark plug replacement", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)

    findings = repo.findings_for_run(run_id)
    assert len(findings) == 1
    assert findings[0]["domain"] == "youtube.com"
    assert "video transcript" in findings[0]["title"]
    assert "GarageChannel" in findings[0]["title"]


@respx.mock
async def test_youtube_without_captions_reports_no_transcript(data_dir):
    cfg = make_cfg(data_dir)
    watch = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result(watch, "Video without captions")])))
    respx.post("https://www.youtube.com/youtubei/v1/player").mock(
        return_value=httpx.Response(200, json={"playabilityStatus":
                                               {"status": "OK"}}))

    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(_script()))
    run_id = orch.enqueue(RunParams(query="spark plug replacement", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)
    assert repo.findings_for_run(run_id) == []
    events = (cfg.research_dir / run_id / "events.jsonl").read_text()
    assert "no caption transcript" in events


# ---- near-duplicate collapse ---------------------------------------------------

def test_fingerprint_similarity_separates_clones_from_neighbors():
    a = extract(Fetched(url="u", final_url="u", content_type="text/html",
                        body=article("Alpha Study").encode()))
    b = extract(Fetched(url="u", final_url="u", content_type="text/html",
                        body=article("Beta Study").encode()))
    clone = extract(Fetched(url="u", final_url="u", content_type="text/html",
                            body=article("Alpha Study").encode()))
    fa, fb, fc = (text_fingerprint(d.text) for d in (a, b, clone))
    assert similarity(fa, fc) == 1.0                    # scraped clone
    assert similarity(fa, fb) < 0.5                     # same topic, distinct page
    assert similarity(fa, frozenset()) == 0.0


@respx.mock
async def test_duplicate_content_across_domains_is_kept_once(data_dir):
    cfg = make_cfg(data_dir)
    clone_html = article("Best Spark Plugs 2uzfe")
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload([
        sx_result("https://rvcontext.com/best-spark-plugs", "Best Spark Plugs"),
        sx_result("https://weldingresource.com/best-spark-plugs", "Best Spark Plugs"),
    ])))
    for url in ("https://rvcontext.com/best-spark-plugs",
                "https://weldingresource.com/best-spark-plugs"):
        respx.get(url).mock(return_value=httpx.Response(200, html=clone_html))

    repo = Repo(connect(cfg.db_path))
    llm = FakeLLM(_script())
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: llm)
    run_id = orch.enqueue(RunParams(query="spark plugs", depth=1,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)

    findings = repo.findings_for_run(run_id)
    assert len(findings) == 1                      # one copy kept, one collapsed
    assert llm.calls["notes"] == 1                 # the clone never cost a call
    events = (cfg.research_dir / run_id / "events.jsonl").read_text()
    assert "duplicate of" in events


# ---- fetch escalation: impersonation and the browser solver --------------------

from app.research.fetcher import _challenge_status


def test_challenge_status_recognizes_bot_walls():
    for reason in ("http 403", "http 202", "http 429", "http 522",
                   "http 403 (impersonated)"):
        assert _challenge_status(reason)
    for reason in ("http 404", "http 500", "no extractable text",
                   "fetch failed: ReadTimeout"):
        assert not _challenge_status(reason)


@respx.mock
async def test_blocked_fetch_escalates_to_impersonation(data_dir, monkeypatch):
    respx.get("https://walled.example.com/page").mock(
        return_value=httpx.Response(403))
    fetcher, client = _fetcher(data_dir)
    calls = []

    async def fake_curl(url, extra_types=()):
        calls.append(url)
        return Fetched(url=url, final_url=url, content_type="text/html",
                       body=article("Recovered Page").encode())
    monkeypatch.setattr(fetcher, "_curl_get", fake_curl)

    fetched = await fetcher.fetch("https://walled.example.com/page")
    await client.aclose()
    assert calls == ["https://walled.example.com/page"]
    assert b"Recovered Page" in fetched.body


@respx.mock
async def test_non_challenge_failures_do_not_escalate(data_dir, monkeypatch):
    respx.get("https://gone.example.com/x").mock(return_value=httpx.Response(404))
    fetcher, client = _fetcher(data_dir)

    async def fake_curl(url, extra_types=()):
        raise AssertionError("impersonation must not run for a 404")
    monkeypatch.setattr(fetcher, "_curl_get", fake_curl)

    with pytest.raises(SkipReason, match="http 404"):
        await fetcher.fetch("https://gone.example.com/x")
    await client.aclose()


@respx.mock
async def test_escalation_can_be_disabled(data_dir, monkeypatch):
    respx.get("https://walled.example.com/page").mock(
        return_value=httpx.Response(403))
    fetcher, client = _fetcher(data_dir)
    fetcher.cfg.browser_impersonation = False

    async def fake_curl(url, extra_types=()):
        raise AssertionError("impersonation is disabled")
    monkeypatch.setattr(fetcher, "_curl_get", fake_curl)

    with pytest.raises(SkipReason, match="http 403"):
        await fetcher.fetch("https://walled.example.com/page")
    await client.aclose()


@respx.mock
async def test_solver_is_last_resort_after_impersonation(data_dir, monkeypatch):
    respx.get("https://walled.example.com/page").mock(
        return_value=httpx.Response(403))
    respx.post("http://solver.test:8191/v1").mock(
        return_value=httpx.Response(200, json={
            "status": "ok",
            "solution": {"status": 200,
                         "url": "https://walled.example.com/page",
                         "response": article("Solved Page")}}))
    fetcher, client = _fetcher(data_dir)
    fetcher.cfg.browser_solver_url = "http://solver.test:8191"

    async def fake_curl(url, extra_types=()):
        raise SkipReason("http 403 (impersonated)")
    monkeypatch.setattr(fetcher, "_curl_get", fake_curl)

    fetched = await fetcher.fetch("https://walled.example.com/page")
    await client.aclose()
    assert b"Solved Page" in fetched.body
    assert fetched.content_type == "text/html"


@respx.mock
async def test_solver_is_skipped_for_api_fetches(data_dir, monkeypatch):
    respx.get("https://old.reddit.com/r/GXOR/comments/abc/x.json").mock(
        return_value=httpx.Response(403))
    fetcher, client = _fetcher(data_dir)
    fetcher.cfg.browser_solver_url = "http://solver.test:8191"
    fetcher.cfg.browser_impersonation = False

    with pytest.raises(SkipReason, match="http 403"):
        await fetcher.fetch(
            "https://www.reddit.com/r/GXOR/comments/abc/x.json",
            extra_types=("application/json",))
    await client.aclose()


@respx.mock
async def test_solver_failure_reports_honestly(data_dir, monkeypatch):
    respx.get("https://walled.example.com/page").mock(
        return_value=httpx.Response(403))
    respx.post("http://solver.test:8191/v1").mock(
        return_value=httpx.Response(200, json={
            "status": "error", "message": "challenge not solved"}))
    fetcher, client = _fetcher(data_dir)
    fetcher.cfg.browser_impersonation = False
    fetcher.cfg.browser_solver_url = "http://solver.test:8191"

    with pytest.raises(SkipReason, match="browser solver failed"):
        await fetcher.fetch("https://walled.example.com/page")
    await client.aclose()


# ---- JS-shell fallback: render-then-extract ------------------------------------

_JS_SHELL = ('<html><head><title>App</title></head><body>'
             '<div id="root"></div><script>window.__APP__=1</script>'
             '</body></html>')


@respx.mock
async def test_js_shell_page_is_rendered_and_recovered(data_dir):
    cfg = make_cfg(data_dir)
    cfg.browser_solver_url = "http://solver.test:8191"
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result("https://spa.example.com/guide", "SPA Guide")])))
    respx.get("https://spa.example.com/guide").mock(
        return_value=httpx.Response(200, html=_JS_SHELL))
    solver_calls = []

    def solver(req):
        solver_calls.append(json.loads(req.content)["url"])
        return httpx.Response(200, json={
            "status": "ok",
            "solution": {"status": 200, "url": "https://spa.example.com/guide",
                         "response": article("Rendered Guide")}})

    respx.post("http://solver.test:8191/v1").mock(side_effect=solver)

    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(_script()))
    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=2,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)

    assert solver_calls == ["https://spa.example.com/guide"]
    findings = repo.findings_for_run(run_id)
    assert len(findings) == 1 and findings[0]["domain"] == "spa.example.com"


@respx.mock
async def test_js_shell_without_solver_skips_honestly(data_dir):
    cfg = make_cfg(data_dir)
    assert not cfg.browser_solver_url
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result("https://spa.example.com/guide", "SPA Guide")])))
    respx.get("https://spa.example.com/guide").mock(
        return_value=httpx.Response(200, html=_JS_SHELL))
    # NOTE: no solver route — a POST there would fail the test

    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(_script()))
    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=2,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)
    assert repo.findings_for_run(run_id) == []
    events = (cfg.research_dir / run_id / "events.jsonl").read_text()
    assert "no extractable text" in events


@respx.mock
async def test_solver_fetched_pages_are_not_rendered_twice(data_dir):
    """A page whose fetch already came through the solver must not re-render
    when it still extracts to nothing (the solver already had its shot)."""
    cfg = make_cfg(data_dir)
    cfg.browser_solver_url = "http://solver.test:8191"
    cfg.browser_impersonation = False
    respx.get(f"{SX}/search").mock(return_value=httpx.Response(200, json=sx_payload(
        [sx_result("https://walled.example.com/page", "Walled")])))
    respx.get("https://walled.example.com/page").mock(
        return_value=httpx.Response(403))
    solver_calls = []

    def solver(req):
        solver_calls.append(1)
        return httpx.Response(200, json={
            "status": "ok",
            "solution": {"status": 200, "url": "https://walled.example.com/page",
                         "response": _JS_SHELL}})   # solved, but still a shell

    respx.post("http://solver.test:8191/v1").mock(side_effect=solver)

    repo = Repo(connect(cfg.db_path))
    orch = Orchestrator(lambda: cfg, repo, ProgressBus(),
                        llm_factory=lambda: FakeLLM(_script()))
    run_id = orch.enqueue(RunParams(query="solid state batteries", depth=2,
                                    recency="all", origin="cli"))
    await orch.execute_now(run_id)
    assert len(solver_calls) == 1                  # exactly one render
    assert repo.findings_for_run(run_id) == []
