import pytest

from app.config import ENV_MAP


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch):
    """Tests must not see the developer's .env.

    docker-compose passes `env_file: .env` into the test container, so without
    this a machine configured for a local LLM fails the default-value tests
    while CI passes (and vice versa). Tests that want an env var set it
    explicitly with monkeypatch after this fixture has cleared the slate.
    """
    for var in ENV_MAP.values():
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def research_dir(data_dir):
    d = data_dir / "research_data"
    d.mkdir()
    return d
