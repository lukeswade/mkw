"""The self-check must actually catch the two failures it exists for."""
import pytest

from app import selfcheck
from app.config import Settings
from app.selfcheck import SelfCheckError


def test_passes_on_current_code(data_dir):
    selfcheck.run_all(Settings(data_dir=str(data_dir)))


def test_catches_unescaped_brace_in_template(monkeypatch):
    class FakePrompts:
        BROKEN = 'Do the thing.\n\nExample:\n{\n  "relevance": 8\n}'

    monkeypatch.setattr(selfcheck, "prompts", FakePrompts)
    with pytest.raises(SelfCheckError, match="unescaped"):
        selfcheck.check_prompt_templates()


def test_accepts_properly_escaped_template(monkeypatch):
    class FakePrompts:
        FINE = 'Use {query}.\n\nExample:\n{{\n  "relevance": 8\n}}'

    monkeypatch.setattr(selfcheck, "prompts", FakePrompts)
    selfcheck.check_prompt_templates()


def test_database_check_runs_migrations_on_a_fresh_volume(data_dir):
    cfg = Settings(data_dir=str(data_dir))
    cfg.ensure_dirs()
    selfcheck.check_database(cfg)          # would raise on a bad migration
    assert cfg.db_path.exists()
