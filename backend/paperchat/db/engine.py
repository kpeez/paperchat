from functools import lru_cache

import sqlite_vec
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from paperchat.config import get_database_path, get_database_url


def _configure_sqlite_connection(dbapi_conn, _connection_record):
    """Enable WAL mode, foreign keys, and load the sqlite-vec extension."""
    dbapi_conn.execute("PRAGMA journal_mode=WAL")
    dbapi_conn.execute("PRAGMA foreign_keys=ON")
    dbapi_conn.enable_load_extension(True)
    sqlite_vec.load(dbapi_conn)
    dbapi_conn.enable_load_extension(False)


@lru_cache
def get_engine() -> Engine:
    db_path = get_database_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(get_database_url())
    event.listen(engine, "connect", _configure_sqlite_connection)
    return engine


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)
