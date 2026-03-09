from functools import lru_cache

from sqlalchemy import Engine, create_engine

from paperchat_backend.config import get_database_url


@lru_cache
def get_engine() -> Engine:
    return create_engine(get_database_url(), pool_pre_ping=True)
