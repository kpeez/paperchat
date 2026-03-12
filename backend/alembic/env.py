"""Alembic environment configuration for backend schema migrations."""

from __future__ import annotations

from logging.config import fileConfig
from pathlib import Path

import sqlite_vec
from sqlalchemy import engine_from_config, event, pool
from sqlalchemy.engine import make_url

from alembic import context
from paperchat.config import get_database_url
from paperchat.db.schema import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    return get_database_url()


def _configure_sqlite_connection(dbapi_conn, _connection_record):
    dbapi_conn.execute("PRAGMA foreign_keys=ON")
    dbapi_conn.enable_load_extension(True)
    sqlite_vec.load(dbapi_conn)
    dbapi_conn.enable_load_extension(False)


def _ensure_sqlite_parent_dir(database_url: str) -> None:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite":
        return

    database = url.database
    if database in {None, "", ":memory:"}:
        return

    Path(database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def run_migrations_offline() -> None:
    """Run migrations without creating an Engine."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        render_as_batch=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations using a live database connection."""
    configuration = config.get_section(config.config_ini_section, {})
    database_url = _database_url()
    configuration["sqlalchemy.url"] = database_url
    _ensure_sqlite_parent_dir(database_url)

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    event.listen(connectable, "connect", _configure_sqlite_connection)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            render_as_batch=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
