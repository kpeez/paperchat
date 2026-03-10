from __future__ import annotations

from collections.abc import Generator

import pytest
import sqlite_vec
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from paperchat.db.schema import Base


def _configure_test_connection(dbapi_conn, _connection_record):
    dbapi_conn.execute("PRAGMA foreign_keys=ON")
    dbapi_conn.enable_load_extension(True)
    sqlite_vec.load(dbapi_conn)
    dbapi_conn.enable_load_extension(False)


@pytest.fixture(scope="session")
def database_engine() -> Generator[Engine]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    event.listen(engine, "connect", _configure_test_connection)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session_factory(database_engine: Engine) -> sessionmaker[Session]:
    Base.metadata.drop_all(database_engine)
    Base.metadata.create_all(database_engine)
    return sessionmaker(bind=database_engine, expire_on_commit=False)


@pytest.fixture
def db_session(db_session_factory: sessionmaker[Session]) -> Generator[Session]:
    session = db_session_factory()
    try:
        yield session
    finally:
        session.close()
