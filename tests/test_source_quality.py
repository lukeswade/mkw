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
from app.research import reddit
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
    # The solver now defaults to the compose service, so a test about the
    # impersonation rung has to say it is testing that rung alone.
    fetcher.cfg.browser_solver_url = ""

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
    # Solver off for this one: the point is what happens with no last resort.
    cfg.browser_solver_url = ""
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


async def test_both_escalation_rungs_are_on_by_default(data_dir):
    """A challenged page should get the cheap Chrome-fingerprint retry and,
    failing that, the browser — without anyone visiting Settings first."""
    from app.config import load_settings
    cfg = load_settings(str(data_dir))
    assert cfg.browser_impersonation is True
    assert cfg.browser_solver_url == "http://flaresolverr:8191"
    # Reference chasing is the opposite case: on for a fortnight it kept 3 of
    # 73 pages it read, so a fresh install leaves it off.
    assert cfg.reference_chasing is False


@respx.mock
async def test_a_run_counts_which_escalation_rungs_it_needed(data_dir, monkeypatch):
    """130 browser solves had happened across past runs with no mention
    anywhere — there was no way to know which pages fought back, nor whether
    the cheap rung would have been enough."""
    respx.get("https://walled.example.com/page").mock(
        return_value=httpx.Response(403))
    respx.post("http://solver.test:8191/v1").mock(
        return_value=httpx.Response(200, json={
            "status": "ok",
            "solution": {"status": 200, "url": "https://walled.example.com/page",
                         "response": article("Solved Page")}}))
    fetcher, client = _fetcher(data_dir)
    fetcher.cfg.browser_solver_url = "http://solver.test:8191"
    assert fetcher.impersonated == 0 and fetcher.solved == 0

    # curl_cffi is a separate HTTP stack respx cannot intercept, so the cheap
    # rung is stood in for — its counter lives in the real one either way.
    async def fake_curl(url, extra_types=()):
        fetcher.impersonated += 1
        raise SkipReason("http 403 (impersonated)")
    monkeypatch.setattr(fetcher, "_curl_get", fake_curl)

    await fetcher.fetch("https://walled.example.com/page")
    await client.aclose()
    assert fetcher.impersonated == 1, "the cheap rung is tried before the browser"
    assert fetcher.solved == 1


# ---- what a forum thread costs when the reader gives up ---------------------

_SMF_THREAD = """<!DOCTYPE html><html><head><title>Loose Reel Seat Repair</title></head>
<body><div id="wrapper"><div class="navigate_section">Main Menu Home Search Login</div>
<div id="forumposts"><div class="post">%s</div></div></body></html>""" % (
    "The reel seat on my rod worked loose after years of use. I removed the old "
    "epoxy with a heat gun set low, cleaned the blank with acetone, and re-bedded "
    "the seat with a slow-cure two part epoxy. Masking tape arbors keep it "
    "concentric while it sets. " * 12)


def test_a_page_trafilatura_reads_as_empty_falls_back(monkeypatch):
    """trafilatura returns nothing on some forum software while the same bytes
    hold thousands of characters of real discussion — 37 pages were discarded
    that way in one run, and forums are the best source a repair question has."""
    from app.research import extractor
    from app.research.fetcher import Fetched
    page = Fetched(url="https://forum.test/t/1", final_url="https://forum.test/t/1",
                   content_type="text/html", body=_SMF_THREAD.encode())
    monkeypatch.setattr(extractor.trafilatura, "bare_extraction",
                        lambda *a, **k: None)
    doc = extractor.extract(page)
    assert doc is not None, "the fallback should have recovered this"
    assert "epoxy" in doc.text


def test_the_fallback_does_not_displace_trafilatura():
    """Second reader, not first: trafilatura is the better one on articles."""
    from app.research import extractor
    from app.research.fetcher import Fetched
    article = ("<html><head><title>T</title></head><body><article><p>"
               + ("A real article paragraph about rod building. " * 40)
               + "</p></article></body></html>")
    doc = extractor.extract(Fetched(url="https://site.test/a",
                                    final_url="https://site.test/a",
                                    content_type="text/html",
                                    body=article.encode()))
    assert doc is not None and "real article paragraph" in doc.text


def test_a_bot_wall_is_not_reported_as_a_thin_page():
    """Eight forums in one run answered 200 with a ~2.6KB proof-of-work
    challenge. Filed as "no extractable text", that reads as a thin page
    rather than a locked door."""
    from app.research.extractor import looks_bot_walled
    from app.research.fetcher import Fetched
    def page(body):
        return Fetched(url="https://f.test/t", final_url="https://f.test/t",
                       content_type="text/html", body=body.encode())
    assert looks_bot_walled(page(
        "<html><head><script>window.POW_CHALLENGE_DATA={difficulty:'3'};</script>"))
    assert looks_bot_walled(page("<html><title>Just a moment...</title>"))
    assert not looks_bot_walled(page(_SMF_THREAD))


def test_a_pdf_gets_a_larger_ceiling_than_html():
    """A 14MB rod-building manual was rejected by the shared 3MB cap — the
    kind of primary document search engines never surface. Only the first
    pages are read either way, so the download is the whole cost."""
    from app.research.fetcher import MAX_BYTES, MAX_PDF_BYTES
    assert MAX_PDF_BYTES > MAX_BYTES
    assert MAX_PDF_BYTES >= 15_000_000

# ---- when reddit refuses on every surface it owns ---------------------------

@respx.mock
async def test_a_public_archive_rebuilds_a_thread_reddit_refuses(data_dir):
    """The .json API 403s for whole address ranges and old.reddit redirects
    those same addresses to a login wall, so a run could lose every reddit
    source with no recourse. Two free archives cover it."""
    url = "https://www.reddit.com/r/rodbuilding/comments/abc123/reel_seat/"
    respx.get(url__regex=r"https://old\.reddit\.com/r/rodbuilding/comments/abc123/reel_seat\.json.*").mock(
        return_value=httpx.Response(403))
    respx.get("https://old.reddit.com/r/rodbuilding/comments/abc123/reel_seat/").mock(
        return_value=httpx.Response(302, headers={"location": "https://old.reddit.com/login/?reason=lor2"}))
    respx.get("https://old.reddit.com/login/").mock(return_value=httpx.Response(403))
    respx.get("https://api.pullpush.io/reddit/search/submission/").mock(
        return_value=httpx.Response(200, json={"data": [{
            "title": "Reel seat repair", "subreddit": "rodbuilding",
            "selftext": "The threaded barrel is backing off the spacer. " * 8,
            "permalink": "/r/rodbuilding/comments/abc123/reel_seat/"}]}))
    respx.get("https://api.pullpush.io/reddit/search/comment/").mock(
        return_value=httpx.Response(200, json={"data": [
            {"body": "Rough up the spacer and re-bed it with slow-cure epoxy. " * 4},
            {"body": "[deleted]"}]}))

    cfg = make_cfg(data_dir)
    # This is about the archive rung, not the escalation ladder above it.
    cfg.browser_impersonation, cfg.browser_solver_url = False, ""
    async with httpx.AsyncClient() as client:
        doc, canonical = await reddit.thread(Fetcher(cfg, client), url)
    assert "threaded barrel" in doc.text
    assert "slow-cure epoxy" in doc.text          # comments came too
    assert "[deleted]" not in doc.text            # and the dead ones did not
    assert canonical.endswith("/r/rodbuilding/comments/abc123/reel_seat/")


@respx.mock
async def test_the_second_archive_covers_what_the_first_has_not_ingested(data_dir):
    """PullPush lags on recent posts; Arctic Shift had one it was missing."""
    url = "https://www.reddit.com/r/rodbuilding/comments/xyz789/help/"
    respx.get(url__regex=r"https://old\.reddit\.com/r/rodbuilding/comments/xyz789/help\.json.*").mock(
        return_value=httpx.Response(403))
    respx.get("https://old.reddit.com/r/rodbuilding/comments/xyz789/help/").mock(
        return_value=httpx.Response(403))
    respx.get("https://api.pullpush.io/reddit/search/submission/").mock(
        return_value=httpx.Response(200, json={"data": []}))     # not ingested
    respx.get("https://arctic-shift.photon-reddit.com/api/posts/ids").mock(
        return_value=httpx.Response(200, json={"data": [{
            "title": "Help removing reel seat", "subreddit": "rodbuilding",
            "selftext": "Heat gun on low, then acetone on the blank. " * 8}]}))
    respx.get("https://arctic-shift.photon-reddit.com/api/comments/search").mock(
        return_value=httpx.Response(200, json={"data": []}))

    cfg = make_cfg(data_dir)
    cfg.browser_impersonation, cfg.browser_solver_url = False, ""
    async with httpx.AsyncClient() as client:
        doc, _ = await reddit.thread(Fetcher(cfg, client), url)
    assert "Heat gun" in doc.text


def test_the_submission_id_comes_out_of_any_thread_url():
    assert reddit.thread_id(
        "https://www.reddit.com/r/rodbuilding/comments/1hc9je3/reel_seat/") == "1hc9je3"
    assert reddit.thread_id(
        "https://old.reddit.com/r/x/comments/abc/") == "abc"
    assert reddit.thread_id("https://www.reddit.com/r/rodbuilding/") is None


@respx.mock
async def test_a_video_without_captions_is_read_from_its_description(data_dir):
    """A demonstration video with no captions was discarded whole — the Mana
    Ball DIY trackball build among them — though its description says what it
    shows and links the repo. Captions lead when they carry enough; the
    description completes or replaces them."""
    from app.research import youtube
    desc = ("Full build of my open-source modular trackball: PMW3610 sensor, nRF52840, ZMK firmware. "
            "3D print files and PCB on GitHub: https://github.com/example/mana-ball — parts list in the pinned comment. "
            "Chapters: sensor board, case, firmware flashing, calibration.")
    respx.post("https://www.youtube.com/youtubei/v1/player").mock(return_value=httpx.Response(200, json={
        "playabilityStatus": {"status": "OK"},
        "videoDetails": {"title": "Mana Ball - Open Source DIY Modular Trackball", "author": "Maker",
                         "shortDescription": desc, "keywords": ["trackball", "zmk"]},
        "microformat": {"playerMicroformatRenderer": {"publishDate": "2026-05-01"}}}))
    async with httpx.AsyncClient() as client:
        doc = await youtube.transcript(client, "vn09xzBtV5k")
    assert doc is not None and "(video description)" in doc.title
    assert "github.com/example/mana-ball" in doc.text and "Tags: trackball, zmk" in doc.text
    assert doc.date == "2026-05-01"


@respx.mock
async def test_a_video_with_only_a_title_is_still_skipped(data_dir):
    from app.research import youtube
    respx.post("https://www.youtube.com/youtubei/v1/player").mock(return_value=httpx.Response(200, json={
        "playabilityStatus": {"status": "OK"}, "videoDetails": {"title": "Untitled", "author": "x", "shortDescription": "short"}}))
    async with httpx.AsyncClient() as client:
        assert await youtube.transcript(client, "dQw4w9WgXcQ") is None


@respx.mock
async def test_a_comment_less_archive_rebuild_yields_to_one_with_the_comments(data_dir):
    """Two threads came back from the first archive with zero comments and
    scored 2/10 while the other archive held them."""
    from app.research import reddit
    post = {"title": "Reel seat came loose — how do I fix it?", "subreddit": "flyfishing",
            "selftext": "The reel seat on my 6wt spins freely on the blank. " * 6, "permalink": "/r/flyfishing/comments/abc12/x/"}
    respx.get("https://api.pullpush.io/reddit/search/submission/").mock(return_value=httpx.Response(200, json={"data": [post]}))
    respx.get("https://api.pullpush.io/reddit/search/comment/").mock(return_value=httpx.Response(200, json={"data": []}))
    respx.get("https://arctic-shift.photon-reddit.com/api/posts/ids").mock(return_value=httpx.Response(200, json={"data": [post]}))
    respx.get("https://arctic-shift.photon-reddit.com/api/comments/search").mock(return_value=httpx.Response(200, json={"data": [
        {"body": "Inject rod bond epoxy through a small hole and rotate the seat to spread it."},
        {"body": "Or heat the seat gently and pull it, then re-epoxy with a proper arbor."}]}))
    async with httpx.AsyncClient() as client:
        doc, url = await reddit._archive_fallback(client, "https://www.reddit.com/r/flyfishing/comments/abc12/x/", SkipReason("blocked"))
    assert "## Comments" in doc.text and "rod bond epoxy" in doc.text


@respx.mock
async def test_a_comment_less_rebuild_is_still_kept_when_no_archive_has_more(data_dir):
    from app.research import reddit
    post = {"title": "Reel seat came loose", "subreddit": "flyfishing", "selftext": "Long description of the problem. " * 10, "permalink": "/r/x/comments/abc12/y/"}
    respx.get("https://api.pullpush.io/reddit/search/submission/").mock(return_value=httpx.Response(200, json={"data": [post]}))
    respx.get("https://api.pullpush.io/reddit/search/comment/").mock(return_value=httpx.Response(200, json={"data": []}))
    respx.get("https://arctic-shift.photon-reddit.com/api/posts/ids").mock(return_value=httpx.Response(200, json={"data": []}))
    async with httpx.AsyncClient() as client:
        doc, url = await reddit._archive_fallback(client, "https://www.reddit.com/r/x/comments/abc12/y/", SkipReason("blocked"))
    assert doc is not None and "## Comments" not in doc.text
