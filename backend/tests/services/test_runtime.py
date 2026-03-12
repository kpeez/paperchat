from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from paperchat import config as paperchat_config
from paperchat.db import engine as db_engine
from paperchat.services import runtime as runtime_service


def _clear_runtime_caches() -> None:
    paperchat_config.get_database_path.cache_clear()
    paperchat_config.get_database_url.cache_clear()
    paperchat_config.get_cache_dir.cache_clear()
    paperchat_config.get_embedding_model_name.cache_clear()
    db_engine.get_engine.cache_clear()


@pytest.fixture(autouse=True)
def clear_runtime_caches() -> Generator[None, None, None]:
    _clear_runtime_caches()
    yield
    _clear_runtime_caches()


def test_build_runtime_response_prefers_database_url(monkeypatch, tmp_path: Path) -> None:
    database_path = tmp_path / "runtime" / "paperchat.db"
    cache_dir = tmp_path / "cache"

    monkeypatch.setenv("PAPERCHAT_DATABASE_URL", f"sqlite:///{database_path}")
    monkeypatch.setenv("PAPERCHAT_DB_PATH", str(tmp_path / "ignored.db"))
    monkeypatch.setenv("PAPERCHAT_CACHE_DIR", str(cache_dir))

    response = runtime_service.build_runtime_response()

    assert response.database_path == str(database_path.resolve())
    assert response.data_dir == str(database_path.resolve().parent)
    assert response.cache_dir == str(cache_dir.resolve())


def test_get_engine_creates_parent_dir_from_database_url(monkeypatch, tmp_path: Path) -> None:
    database_path = tmp_path / "nested" / "paperchat.db"
    monkeypatch.setenv("PAPERCHAT_DATABASE_URL", f"sqlite:///{database_path}")

    engine = db_engine.get_engine()
    try:
        assert database_path.parent.is_dir()
    finally:
        engine.dispose()
        db_engine.get_engine.cache_clear()


def test_database_url_must_remain_sqlite(monkeypatch) -> None:
    monkeypatch.setenv("PAPERCHAT_DATABASE_URL", "postgresql://paperchat:paperchat@localhost/db")

    with pytest.raises(ValueError, match="sqlite URLs"):
        paperchat_config.get_database_url()
