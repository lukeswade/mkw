import json
import os
import stat

import pytest

from app.config import Settings, load_settings, mask_secret, save_settings


def test_defaults(data_dir):
    s = load_settings(str(data_dir))
    assert s.llm_provider == "deepseek"
    assert s.resolved_base_url == "https://api.deepseek.com"
    assert s.results_per_query == 8
    assert s.respect_robots is False


def test_env_overrides_default(data_dir, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
    monkeypatch.setenv("RESULTS_PER_QUERY", "5")
    monkeypatch.setenv("RESPECT_ROBOTS", "true")
    s = load_settings(str(data_dir))
    assert s.deepseek_api_key == "sk-env"
    assert s.results_per_query == 5
    assert s.respect_robots is True


def test_settings_json_overrides_env(data_dir, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
    (data_dir / "settings.json").write_text(json.dumps({"deepseek_api_key": "sk-file"}))
    s = load_settings(str(data_dir))
    assert s.deepseek_api_key == "sk-file"


def test_empty_json_value_falls_back_to_env(data_dir, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
    (data_dir / "settings.json").write_text(json.dumps({"deepseek_api_key": ""}))
    s = load_settings(str(data_dir))
    assert s.deepseek_api_key == "sk-env"


def test_corrupt_settings_json_ignored(data_dir):
    (data_dir / "settings.json").write_text("{not json")
    s = load_settings(str(data_dir))
    assert s.llm_provider == "deepseek"


def test_int_coercion_garbage_falls_back(data_dir, monkeypatch):
    monkeypatch.setenv("RESULTS_PER_QUERY", "lots")
    s = load_settings(str(data_dir))
    assert s.results_per_query == 8


def test_save_settings_atomic_0600_and_merge(data_dir):
    path = data_dir / "settings.json"
    save_settings(path, {"deepseek_api_key": "sk-1"})
    save_settings(path, {"llm_provider": "local"})
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600
    data = json.loads(path.read_text())
    assert data == {"deepseek_api_key": "sk-1", "llm_provider": "local"}
    assert not list(data_dir.glob("*.tmp"))


def test_save_settings_rejects_unknown_keys(data_dir):
    with pytest.raises(ValueError):
        save_settings(data_dir / "settings.json", {"data_dir": "/etc"})


def test_allowed_telegram_ids():
    s = Settings(telegram_allowed_user_ids=" 123, 456;789 , nope ")
    assert s.allowed_telegram_ids == {123, 456, 789}


def test_mask_secret():
    assert mask_secret("") == ""
    assert mask_secret("abc") == "•" * 8
    assert mask_secret("sk-abcdef123456").endswith("3456")
    assert "abcdef" not in mask_secret("sk-abcdef123456")
