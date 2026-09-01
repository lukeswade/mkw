import json
import os
import stat

import pytest
from fastapi.testclient import TestClient

from app.config import ENV_MAP, load_settings
from app.db import Repo, connect, utcnow
from app.research.storage import RunStore
from app.web.server import create_app


def make_app(data_dir, monkeypatch, **cfg_kw):
    """App with production-style dynamic settings loading (env + settings.json)."""
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    for field, value in cfg_kw.items():
        monkeypatch.setenv(ENV_MAP[field], str(value))
    app = create_app(enable_worker=False, enable_bot=False)
    return app, load_settings(str(data_dir))


def seed_completed_run(cfg, *, evil=False) -> str:
    """Create a finished run directly on disk + DB, as the pipeline would."""
    repo = Repo(connect(cfg.db_path))
    store = RunStore.create(cfg.research_dir, "seeded research run")
    run_id = store.run_id
    repo.create_run(run_id=run_id, query="seeded research run", depth=2,
                    recency="month", dir=run_id, origin="web", status="queued")
    notes = "Notes body."
    if evil:
        notes = 'Injected <script>alert("xss")</script> <img src=x onerror=alert(1)>'
    store.write_overview("# Seeded Research\n\n## TL;DR\n\n- A claim [1]\n\n" + notes)
    store.write_sources("# Sources\n\n1. Example\n")
    store.write_further("# Further research\n\n1. **More**\n")
    store.write_finding(1, "Example Finding", f"# [1] Example Finding\n\n{notes}\n")
    store.write_meta({"run_id": run_id, "status": "completed",
                      "title": "Seeded Research",
                      "followups": [{"query": "Dig into more detail",
                                     "rationale": "gap", "depth": 4,
                                     "recency": "1year"}]})
    store.append_event({"seq": 1, "ts": 0, "type": "status", "status": "running"})
    store.append_event({"seq": 2, "ts": 0, "type": "done", "status": "completed"})
    repo.add_finding(run_id=run_id, idx=1, url="https://example.com/x",
                     title="Example Finding", domain="example.com",
                     published_date="2026-01-01", relevance=8,
                     path="findings/001_example-finding.md", summary="sum")
    repo.update_run(run_id, status="completed", title="Seeded Research",
                    stop_reason="saturated", finished_at=utcnow(),
                    stats_json=json.dumps({
                        "rounds": 2, "sources_kept": 1, "sources_skipped": 0,
                        "llm": {"calls": 5, "prompt_tokens": 10,
                                "completion_tokens": 5}}))
    return run_id


def test_health_and_index_without_password(data_dir, monkeypatch):
    app, _ = make_app(data_dir, monkeypatch)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        r = client.get("/")
        assert r.status_code == 200
        assert "What should I research?" in r.text


def test_auth_gate_and_login(data_dir, monkeypatch):
    app, _ = make_app(data_dir, monkeypatch, web_password="hunter2")
    with TestClient(app) as client:
        r = client.get("/", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"].startswith("/login")
        assert client.get("/health").status_code == 200  # exempt

        bad = client.post("/login", data={"password": "wrong", "next": "/"})
        assert bad.status_code == 401

        ok = client.post("/login", data={"password": "hunter2", "next": "/"},
                         follow_redirects=False)
        assert ok.status_code == 303
        assert client.get("/").status_code == 200  # cookie now set

        evil_next = client.post(
            "/login", data={"password": "hunter2", "next": "//evil.com/x"},
            follow_redirects=False)
        assert evil_next.headers["location"] == "/"


def test_create_run_and_validation(data_dir, monkeypatch):
    app, cfg = make_app(data_dir, monkeypatch)
    with TestClient(app) as client:
        r = client.post("/runs", data={"query": "a proper research question",
                                       "depth": 4, "recency": "month"},
                        follow_redirects=False)
        assert r.status_code == 303
        run_id = r.headers["location"].split("/runs/")[1]
        repo = Repo(connect(cfg.db_path))
        row = repo.get_run(run_id)
        assert row["status"] == "queued"  # worker disabled in tests
        assert row["depth"] == 4

        bad = client.post("/runs", data={"query": "xy", "depth": 3,
                                         "recency": "month"})
        assert bad.status_code == 422


def test_run_page_renders_and_escapes(data_dir, monkeypatch):
    app, cfg = make_app(data_dir, monkeypatch)
    run_id = seed_completed_run(cfg, evil=True)
    with TestClient(app) as client:
        r = client.get(f"/runs/{run_id}")
        assert r.status_code == 200
        assert "Seeded Research" in r.text
        assert 'id="src-1"' in r.text            # citation anchor
        assert "Dig into more detail" in r.text  # follow-up card
        # stored XSS: tags from fetched content must arrive escaped, never raw
        assert "<script>alert" not in r.text
        assert "<img src=x" not in r.text
        assert "&lt;script&gt;" in r.text
        assert client.get("/runs/nope").status_code == 404


def test_file_serving_blocks_traversal(data_dir, monkeypatch):
    app, cfg = make_app(data_dir, monkeypatch)
    run_id = seed_completed_run(cfg)
    outside = cfg.data_path / "secret.txt"
    outside.write_text("secret")
    with TestClient(app) as client:
        ok = client.get(f"/runs/{run_id}/file/overview.md")
        assert ok.status_code == 200 and "Seeded" in ok.text
        # encoded forms reach the server un-normalized; SERVABLE_RE must reject
        for bad in ("..%2Fsecret.txt", "..%2F..%2Fsecret.txt", "%2Fetc%2Fpasswd",
                    "findings%2F..%2F..%2Fsecret.txt",
                    "meta.json%2F..%2Foverview.md", "overview.md.tmp",
                    "findings/999_%2E%2E.md"):
            r = client.get(f"/runs/{run_id}/file/{bad}")
            assert r.status_code == 404, f"{bad} → {r.status_code}"


def test_sse_replay_for_finished_run(data_dir, monkeypatch):
    app, cfg = make_app(data_dir, monkeypatch)
    run_id = seed_completed_run(cfg)
    with TestClient(app) as client:
        with client.stream("GET", f"/runs/{run_id}/events") as r:
            body = "".join(r.iter_text())
    assert "id: 1" in body and "id: 2" in body
    assert '"type": "done"' in body


def test_settings_save_masking_and_mode(data_dir, monkeypatch):
    app, cfg = make_app(data_dir, monkeypatch)
    with TestClient(app) as client:
        r = client.post("/settings", data={
            "llm_provider": "deepseek",
            "llm_base_url": "https://api.deepseek.com",
            "llm_model": "deepseek-chat",
            "fast_model": "",
            "telegram_allowed_user_ids": "123",
            "searxng_url": "http://searxng:8080",
            "llm_concurrency": "9",
            "respect_robots": "on",
            "llm_api_key": "sk-supersecret-9876",
            "telegram_bot_token": "",
            "web_password": "",
        }, follow_redirects=False)
        assert r.status_code == 303

        settings_file = cfg.data_path / "settings.json"
        assert stat.S_IMODE(os.stat(settings_file).st_mode) == 0o600
        saved = json.loads(settings_file.read_text())
        assert saved["llm_api_key"] == "sk-supersecret-9876"
        assert saved["llm_concurrency"] == 9
        assert "telegram_bot_token" not in saved  # blank secret untouched

        page = client.get("/settings")
        assert "sk-supersecret-9876" not in page.text  # masked
        assert "9876" in page.text                     # last 4 shown

        # blank key on re-save keeps the stored secret
        client.post("/settings", data={"llm_provider": "deepseek",
                                       "llm_api_key": ""})
        saved = json.loads(settings_file.read_text())
        assert saved["llm_api_key"] == "sk-supersecret-9876"


def test_library_keyword_search(data_dir, monkeypatch):
    app, cfg = make_app(data_dir, monkeypatch)
    run_id = seed_completed_run(cfg)
    repo = Repo(connect(cfg.db_path))
    repo.fts_add(run_id, "overview", "Seeded Research",
                 "unique zanzibar content body")
    with TestClient(app) as client:
        r = client.get("/library", params={"q": "zanzibar"})
        assert r.status_code == 200
        assert "Seeded Research" in r.text
        none = client.get("/library", params={"q": "missingword12345"})
        assert "Nothing found" in none.text


def test_readme_page_renders(data_dir, monkeypatch):
    app, _ = make_app(data_dir, monkeypatch)
    with TestClient(app) as client:
        r = client.get("/readme")
    assert r.status_code == 200
    assert "Deep Research" in r.text


def test_keyword_snippet_is_escaped(data_dir, monkeypatch):
    """FTS snippets are page text sqlite copies verbatim — never raw HTML."""
    app, cfg = make_app(data_dir, monkeypatch)
    run_id = seed_completed_run(cfg)
    repo = Repo(connect(cfg.db_path))
    repo.fts_add(run_id, "finding", "Evil Source",
                 'text <img src=x onerror=alert(1)> about zanzibar batteries')
    with TestClient(app) as client:
        r = client.get("/library", params={"q": "zanzibar"})
    assert r.status_code == 200
    # the angle brackets are escaped, so the payload is inert text, not a tag
    assert "<img" not in r.text
    assert "&lt;img src=x onerror=alert(1)&gt;" in r.text
    assert "<mark>zanzibar</mark>" in r.text   # highlight still works


def test_delete_run_removes_every_store(data_dir, monkeypatch):
    app, cfg = make_app(data_dir, monkeypatch)
    run_id = seed_completed_run(cfg)
    run_dir = cfg.research_dir / run_id
    repo = Repo(connect(cfg.db_path))
    repo.fts_add(run_id, "overview", "Seeded Research", "zanzibar content")
    assert run_dir.is_dir()

    with TestClient(app) as client:
        r = client.delete(f"/runs/{run_id}")
        assert r.status_code == 200
        assert r.headers["HX-Redirect"] == "/library"
        assert client.get(f"/runs/{run_id}").status_code == 404

    assert repo.get_run(run_id) is None
    assert repo.findings_for_run(run_id) == []
    assert repo.fts_search("zanzibar") == []
    assert not run_dir.exists()
    assert cfg.research_dir.is_dir()          # only the run went
    assert client.app.state is not None


def test_delete_missing_run_is_404(data_dir, monkeypatch):
    app, _ = make_app(data_dir, monkeypatch)
    with TestClient(app) as client:
        assert client.delete("/runs/does-not-exist").status_code == 404


def test_login_throttles_brute_force(data_dir, monkeypatch):
    app, _ = make_app(data_dir, monkeypatch, web_password="hunter2")
    with TestClient(app) as client:
        for _ in range(5):
            r = client.post("/login", data={"password": "wrong", "next": "/"})
            assert r.status_code == 401
        blocked = client.post("/login", data={"password": "wrong", "next": "/"})
        assert blocked.status_code == 429
        assert "Retry-After" in blocked.headers
        # the correct password is refused too while locked out
        assert client.post("/login",
                           data={"password": "hunter2", "next": "/"}
                           ).status_code == 429


def test_pdf_export(data_dir, monkeypatch):
    app, cfg = make_app(data_dir, monkeypatch)
    run_id = seed_completed_run(cfg)
    with TestClient(app) as client:
        r = client.get(f"/runs/{run_id}/export.pdf")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"
    assert run_id in r.headers["content-disposition"]

    import pymupdf
    doc = pymupdf.open(stream=r.content, filetype="pdf")
    text = "".join(page.get_text() for page in doc)
    assert "Seeded Research" in text            # title
    assert "seeded research run" in text        # the original question
    assert "example.com" in text                # bibliography made it in


def test_run_page_shows_original_query_and_merged_tabs(data_dir, monkeypatch):
    app, cfg = make_app(data_dir, monkeypatch)
    run_id = seed_completed_run(cfg)
    with TestClient(app) as client:
        r = client.get(f"/runs/{run_id}")
    assert "seeded research run" in r.text      # the question, verbatim
    # one Sources tab with the rich cards — no separate Findings tab
    assert 'data-tab="sources"' in r.text
    assert 'data-tab="findings"' not in r.text
    assert r.text.count('class="tab-panel') == 4
    assert "export.pdf" in r.text               # export reachable from files
    assert "Export PDF" in r.text               # ...and from the header


def test_evergreen_star_in_lists(data_dir, monkeypatch):
    app, cfg = make_app(data_dir, monkeypatch)
    run_id = seed_completed_run(cfg)
    repo = Repo(connect(cfg.db_path))
    with TestClient(app) as client:
        home = client.get("/")
        assert "evergreen-star" in home.text
        assert "☆" in home.text                  # off state

        # toggling from a list swaps just the star, not a page header
        r = client.post(f"/runs/{run_id}/evergreen?view=star")
        assert r.status_code == 200
        assert "★" in r.text and "run-header" not in r.text
        assert repo.get_run(run_id)["evergreen"] == 1

        lib = client.get("/library")
        assert "★" in lib.text                   # on state visible in library

        r = client.post(f"/runs/{run_id}/evergreen?view=star")
        assert "☆" in r.text
        assert repo.get_run(run_id)["evergreen"] == 0


def test_run_attribution_tags(data_dir, monkeypatch):
    """Tunneled runs carry the Cloudflare Access identity; LAN runs carry the
    configurable local label; retries inherit whoever pressed retry."""
    app, cfg = make_app(data_dir, monkeypatch, lan_user_label="Luke")
    repo = Repo(connect(cfg.db_path))
    with TestClient(app) as client:
        # via the tunnel: Access injects the authenticated email
        r = client.post("/runs", data={"query": "a tunneled research question",
                                       "depth": 2, "recency": "all"},
                        headers={"Cf-Access-Authenticated-User-Email":
                                 "matt.wade@example.com"},
                        follow_redirects=False)
        tunneled = r.headers["location"].split("/runs/")[1]
        assert repo.get_run(tunneled)["created_by"] == "matt.wade"

        # from the LAN: no header, configured label applies
        r = client.post("/runs", data={"query": "a local research question",
                                       "depth": 2, "recency": "all"},
                        follow_redirects=False)
        local = r.headers["location"].split("/runs/")[1]
        assert repo.get_run(local)["created_by"] == "Luke"

        # tags render in the list views, coloured per user
        home = client.get("/")
        assert 'class="user-tag"' in home.text
        assert ">matt.wade</span>" in home.text
        assert ">Luke</span>" in home.text
        assert "--hue:" in home.text

        # a retry belongs to whoever pressed retry, not the original owner
        repo.update_run(tunneled, status="completed")
        r = client.post(f"/runs/{tunneled}/retry", follow_redirects=False)
        retried = r.headers["location"].split("/runs/")[1]
        assert repo.get_run(retried)["created_by"] == "Luke"


def test_a_follow_up_inherits_how_its_parent_was_run(data_dir, monkeypatch):
    """The "Run this" form posted only query/depth/recency/parent, and
    create_run reads an absent use_prior as OFF — so every follow-up ran with
    memory disabled and the parent's categories dropped. Exercised through
    the rendered form, not by posting fields the form does not send."""
    import re
    from fastapi.testclient import TestClient
    app, cfg = make_app(data_dir, monkeypatch)
    parent = seed_completed_run(cfg)
    repo = Repo(connect(cfg.db_path))
    repo.conn.execute("UPDATE runs SET categories='general,science', use_prior=1"
                      " WHERE id=?", (parent,))
    repo.conn.commit()

    with TestClient(app) as client:
        page = client.get(f"/runs/{parent}").text
        form = re.search(r'<form method="post" action="/runs">(.*?)</form>',
                         page, re.S).group(1)
        fields = dict(re.findall(r'name="(\w+)" value="([^"]*)"', form))
        assert fields["parent_run_id"] == parent
        r = client.post("/runs", data=fields, follow_redirects=False)
        assert r.status_code == 303, r.text
        child = repo.get_run(r.headers["location"].rsplit("/", 1)[-1])

    assert child["use_prior"] == 1, "a follow-up must build on earlier research"
    assert child["categories"] == "general,science"
    assert child["parent_run_id"] == parent


def test_a_follow_up_of_a_memory_off_run_stays_memory_off(data_dir, monkeypatch):
    """Inherit, don't force: a deliberately fresh diagnosis stays fresh."""
    import re
    from fastapi.testclient import TestClient
    app, cfg = make_app(data_dir, monkeypatch)
    parent = seed_completed_run(cfg)
    repo = Repo(connect(cfg.db_path))
    repo.conn.execute("UPDATE runs SET use_prior=0 WHERE id=?", (parent,))
    repo.conn.commit()
    with TestClient(app) as client:
        page = client.get(f"/runs/{parent}").text
        form = re.search(r'<form method="post" action="/runs">(.*?)</form>',
                         page, re.S).group(1)
        fields = dict(re.findall(r'name="(\w+)" value="([^"]*)"', form))
        r = client.post("/runs", data=fields, follow_redirects=False)
        child = repo.get_run(r.headers["location"].rsplit("/", 1)[-1])
    assert child["use_prior"] == 0


def test_a_retry_is_the_same_run_again(data_dir, monkeypatch):
    """Retry carried only the query and categories: a failed brief retried as
    a web search, a named brief lost its list, a claim check lost its
    document, a memory-off run came back with memory on."""
    from fastapi.testclient import TestClient
    app, cfg = make_app(data_dir, monkeypatch)
    repo = Repo(connect(cfg.db_path))
    with TestClient(app) as client:
        # a named brief that ran memory-off
        store = RunStore.create(cfg.research_dir, "Brief: Local LLM")
        bid = repo.create_brief(name="Local LLM", feeds="https://a.test/f\n")
        repo.create_run(run_id=store.run_id, query="Brief: Local LLM", depth=4,
                        recency="week", dir=store.run_id, status="failed",
                        kind="brief", brief_id=bid, use_prior=False,
                        categories="general")
        r = client.post(f"/runs/{store.run_id}/retry", follow_redirects=False)
        child = repo.get_run(r.headers["location"].rsplit("/", 1)[-1])
        assert child["kind"] == "brief"
        assert child["brief_id"] == bid
        assert child["use_prior"] == 0
        assert child["categories"] == "general"

        # a claim check keeps the document it was checking
        vstore = RunStore.create(cfg.research_dir, "Claim check: x")
        vstore.write_document("MLX doubles decode throughput on M4.")
        repo.create_run(run_id=vstore.run_id, query="Claim check: x", depth=0,
                        recency="all", dir=vstore.run_id, status="failed",
                        kind="verify")
        r = client.post(f"/runs/{vstore.run_id}/retry", follow_redirects=False)
        vchild = repo.get_run(r.headers["location"].rsplit("/", 1)[-1])
        assert vchild["kind"] == "verify"
        assert "doubles decode" in RunStore(
            cfg.research_dir / vchild["dir"]).read_document()


def test_every_ui_editable_setting_has_a_field(data_dir, monkeypatch):
    """Three settings were UI_EDITABLE with no field on the page, reachable
    only through .env: the default categories, the relevance threshold, and
    the embedding API key. The six legacy provider fields are back-compat
    only and stay off the page on purpose."""
    from fastapi.testclient import TestClient
    from app.config import UI_EDITABLE
    legacy = {"deepseek_api_key", "deepseek_base_url", "deepseek_model",
              "local_llm_api_key", "local_llm_base_url", "local_llm_model"}
    app, _cfg = make_app(data_dir, monkeypatch)
    with TestClient(app) as client:
        page = client.get("/settings").text
    missing = sorted(k for k in UI_EDITABLE - legacy if f'name="{k}"' not in page)
    assert missing == [], missing


def test_the_new_settings_fields_round_trip_and_the_key_is_masked(data_dir, monkeypatch):
    from fastapi.testclient import TestClient
    app, cfg = make_app(data_dir, monkeypatch)
    with TestClient(app) as client:
        r = client.post("/settings", data={
            "search_categories": " general,it,q&a ",
            "relevance_threshold": "7",
            "embedding_api_key": "sk-embed-secret-42",
            "llm_concurrency": "3",
        }, follow_redirects=False)
        assert r.status_code == 303
        saved = json.loads((cfg.data_path / "settings.json").read_text())
        assert saved["search_categories"] == "general,it,q&a"
        assert saved["relevance_threshold"] == 7
        assert saved["embedding_api_key"] == "sk-embed-secret-42"
        page = client.get("/settings").text
        assert "sk-embed-secret-42" not in page              # masked, like every secret
        # out-of-range and garbage leave the threshold alone
        client.post("/settings", data={"relevance_threshold": "99"}, follow_redirects=False)
        assert json.loads((cfg.data_path / "settings.json").read_text())["relevance_threshold"] == 10
        client.post("/settings", data={"relevance_threshold": "lots"}, follow_redirects=False)
        assert json.loads((cfg.data_path / "settings.json").read_text())["relevance_threshold"] == 10


def test_the_run_page_tells_the_whole_story_in_one_place(data_dir, monkeypatch):
    """Five counters existed — refused engines, candidates dropped before a
    fetch, pages needing a fingerprint, a browser, or a proof-of-work solve —
    across three surfaces, and pow_solved never reached the stats at all."""
    from fastapi.testclient import TestClient
    app, cfg = make_app(data_dir, monkeypatch)
    rid = seed_completed_run(cfg)
    repo = Repo(connect(cfg.db_path))
    repo.set_stats(rid, {
        "rounds": 3, "searches": 14, "urls_considered": 120, "pre_dropped": 31,
        "sources_kept": 9, "sources_skipped": 44, "sources_expected": 36,
        "blocked_engines": {"duckduckgo": "CAPTCHA", "brave": "too many requests"},
        "impersonated": 6, "browser_solved": 2, "pow_solved": 3,
        "llm": {"calls": 40, "prompt_tokens": 90000, "completion_tokens": 8000,
                "est_cost_usd": 0.12}})
    with TestClient(app) as client:
        page = client.get(f"/runs/{rid}").text
    assert "How this run went" in page
    for expected in ("3 rounds", "14 searches", "2 refused", "duckduckgo", "brave",
                     "120 considered", "31 dropped before a fetch", "44 skipped",
                     "9 kept", "of up to 36", "6 retried with a Chrome fingerprint",
                     "2 needed the browser solver", "3 proof-of-work walls solved",
                     "40 calls", "est $0.12"):
        assert expected in page, expected
    assert "11 pages fought back" in page          # 6 + 2 + 3, in the one-line brief
    assert 'class="stats-footer"' not in page       # the old scattered pieces are gone
    assert 'class="hint escalation-note"' not in page


def test_a_quiet_run_does_not_invent_a_fight(data_dir, monkeypatch):
    from fastapi.testclient import TestClient
    app, cfg = make_app(data_dir, monkeypatch)
    rid = seed_completed_run(cfg)
    Repo(connect(cfg.db_path)).set_stats(rid, {
        "rounds": 1, "searches": 4, "urls_considered": 20, "sources_kept": 5,
        "sources_skipped": 3, "blocked_engines": {},
        "llm": {"calls": 9, "prompt_tokens": 100, "completion_tokens": 50}})
    with TestClient(app) as client:
        page = client.get(f"/runs/{rid}").text
    assert "every engine answered" in page
    assert "fought back" not in page
    assert "Pages that fought back" not in page


def test_the_new_tab_poll_pauses_when_the_tab_is_hidden(data_dir, monkeypatch):
    from fastapi.testclient import TestClient
    app, _cfg = make_app(data_dir, monkeypatch)
    with TestClient(app) as client:
        page = client.get("/").text
    assert "every 5s [document.visibilityState=='visible']" in page


def test_has_matrix_is_read_from_the_row_not_the_disk(data_dir, monkeypatch):
    """Both list pages did a filesystem stat per row — up to 400 per Library
    load, and the New tab polled every five seconds. The column is settled
    once at boot; afterwards the pages must not touch the disk at all."""
    from pathlib import Path
    from fastapi.testclient import TestClient
    from app.research.orchestrator import Orchestrator
    from app.research.progress import ProgressBus
    app, cfg = make_app(data_dir, monkeypatch)
    repo = Repo(connect(cfg.db_path))
    with_table = seed_completed_run(cfg)
    without = seed_completed_run(cfg)
    (cfg.research_dir / with_table / "matrix.md").write_text("# T\n\n| a | b |\n")
    repo.conn.execute("UPDATE runs SET has_matrix = NULL")      # predates the column
    repo.conn.commit()

    # the one-time settle at boot
    Orchestrator(lambda: cfg, repo, ProgressBus()).recover()
    assert repo.get_run(with_table)["has_matrix"] == 1
    assert repo.get_run(without)["has_matrix"] == 0

    # from here on, a page load must not stat the run directories
    real_exists = Path.exists
    def no_disk(self):
        if self.name == "matrix.md":
            raise AssertionError(f"list page touched the disk for {self}")
        return real_exists(self)
    monkeypatch.setattr(Path, "exists", no_disk)
    with TestClient(app) as client:
        home = client.get("/partials/recent-runs").text
        library = client.get("/library").text
        narrowed = client.get("/library", params={"kind": "research+matrix"}).text
    assert home.count("kind-matrix") == 1 and library.count("kind-matrix") == 1
    assert with_table in narrowed and without not in narrowed
