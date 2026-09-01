import asyncio

from app.config import Settings
from app.telegram.bot import start_bot, stop_bot
from app.telegram.handlers import extract_tldr, split_message, status_line


def test_split_message_short():
    assert split_message("hello") == ["hello"]
    assert split_message("") == []


def test_split_message_prefers_paragraphs():
    paras = [f"Paragraph {i}. " + "x" * 900 for i in range(10)]
    text = "\n\n".join(paras)
    chunks = split_message(text, limit=4000)
    assert all(len(c) <= 4000 for c in chunks)
    assert "".join(chunks).replace("\n", "").replace(" ", "") == \
        text.replace("\n", "").replace(" ", "")  # nothing lost
    assert all(c.startswith("Paragraph") for c in chunks)  # clean boundaries


def test_split_message_giant_line_hard_splits():
    chunks = split_message("y" * 9000, limit=4000)
    assert [len(c) for c in chunks] == [4000, 4000, 1000]


def test_extract_tldr_section():
    md = ("# Title\n\n## TL;DR\n\n- point one\n- point two\n\n"
          "## Detail\n\nlots more text")
    tldr = extract_tldr(md)
    assert "point one" in tldr and "point two" in tldr
    assert "lots more" not in tldr


def test_extract_tldr_fallback():
    assert extract_tldr("plain overview text without sections")\
        .startswith("plain overview")


def test_status_line():
    assert status_line({"status": "running", "round": 2, "depth": 5,
                        "findings": 7}) == "🔎 running · round 2/5 · 7 sources kept"
    assert status_line({"status": "completed"}) == "✅ completed"


async def test_start_bot_without_token_is_none(data_dir):
    cfg = Settings(data_dir=str(data_dir), telegram_bot_token="")
    bot = await start_bot(lambda: cfg, None, None, None)
    assert bot is None
    await stop_bot(bot)  # no-op, must not raise


def test_cancel_targets_the_research_run_not_a_post_hoc_job():
    """orch.active also holds resynth/matrix jobs on completed runs, and the
    first key was whichever started first — so /cancel_run could kill a
    matrix build while the research it was asked to stop carried on."""
    from app.telegram.handlers import cancellable_run

    class Orch:
        active = {"done-run-matrix-job": None, "live-run": None}

    class Repo:
        rows = {"done-run-matrix-job": {"status": "completed"},
                "live-run": {"status": "running"}}
        def get_run(self, rid):
            return self.rows.get(rid)

    assert cancellable_run(Orch(), Repo()) == "live-run"

    class OnlyJobs:
        active = {"done-run-matrix-job": None}
    assert cancellable_run(OnlyJobs(), Repo()) is None
