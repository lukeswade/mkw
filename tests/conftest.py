import pytest


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
