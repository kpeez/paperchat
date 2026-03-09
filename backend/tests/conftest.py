from __future__ import annotations

from collections.abc import Generator
from uuid import uuid4

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from paperchat.config import get_database_url
from paperchat.db.schema import Base


@pytest.fixture(scope="session")
def database_engine() -> Generator[Engine]:
    schema_name = f"paperchat_test_{uuid4().hex}"
    admin_engine = create_engine(get_database_url(), pool_pre_ping=True)
    with admin_engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema_name}"'))
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    engine = create_engine(
        get_database_url(),
        pool_pre_ping=True,
        connect_args={"options": f"-csearch_path={schema_name},public"},
    )

    try:
        yield engine
    finally:
        engine.dispose()
        with admin_engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE'))
        admin_engine.dispose()


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
