from __future__ import annotations

from paperchat import __version__
from paperchat.config import get_cache_dir, get_database_path, get_embedding_model_name
from paperchat.models.runtime import RuntimeResponse


def build_runtime_response() -> RuntimeResponse:
    database_path = get_database_path().resolve()
    cache_dir = get_cache_dir().resolve()
    return RuntimeResponse(
        app_version=__version__,
        data_dir=str(database_path.parent),
        database_path=str(database_path),
        cache_dir=str(cache_dir),
        embedding_model=get_embedding_model_name(),
    )
