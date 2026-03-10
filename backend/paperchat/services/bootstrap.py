from dataclasses import dataclass

from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from alembic.util.exc import CommandError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from paperchat import __version__
from paperchat.config import (
    ALEMBIC_DIR,
    ALEMBIC_INI_PATH,
    get_database_url,
)
from paperchat.db.engine import get_engine
from paperchat.models.bootstrap import (
    BootstrapError,
    BootstrapResponse,
    BootstrapStatus,
    CheckResult,
)

CHECK_ACTIONS = {
    "database": "Confirm the SQLite database path is writable and the file is not corrupt.",
    "migrations": "Run the backend migrations so the database revision matches the application.",
    "sqlite_vec": "Reinstall the sqlite-vec Python package (pip install sqlite-vec).",
}
DATABASE_BLOCKED_MESSAGE = "Database must be reachable before this check can run."


@dataclass(frozen=True, slots=True)
class CompletedCheck:
    name: str
    result: CheckResult


def _completed_check(name: str, *, ok: bool, message: str) -> CompletedCheck:
    return CompletedCheck(name=name, result=CheckResult(ok=ok, message=message))


def build_bootstrap_response() -> BootstrapResponse:
    checks = {"backend": CheckResult(ok=True, message=f"Backend {__version__} is running.")}

    database_check = _check_database()
    checks[database_check.name] = database_check.result

    if checks["database"].ok:
        for check in (_check_migrations(), _check_sqlite_vec()):
            checks[check.name] = check.result
    else:
        for name in ("migrations", "sqlite_vec"):
            checks[name] = CheckResult(ok=False, message=DATABASE_BLOCKED_MESSAGE)

    errors = [
        BootstrapError(
            code=f"{name}_failed",
            message=check.message or f"{name} check failed.",
            action=CHECK_ACTIONS[name],
        )
        for name, check in checks.items()
        if name != "backend" and not check.ok
    ]

    return BootstrapResponse(
        app_version=__version__,
        status=BootstrapStatus.ready if not errors else BootstrapStatus.error,
        checks=checks,
        errors=errors,
    )


def _check_database() -> CompletedCheck:
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError as error:
        return _completed_check(
            "database",
            ok=False,
            message=str(error.__cause__ or error),
        )

    return _completed_check("database", ok=True, message="Connected to SQLite.")


def _check_migrations() -> CompletedCheck:
    message: str | None = None
    head_revision: str | None = None

    if not ALEMBIC_INI_PATH.is_file():
        message = f"Alembic config not found at {ALEMBIC_INI_PATH}."
    elif not ALEMBIC_DIR.is_dir():
        message = f"Alembic scripts not found at {ALEMBIC_DIR}."
    else:
        try:
            alembic_config = Config(str(ALEMBIC_INI_PATH))
            alembic_config.set_main_option("script_location", str(ALEMBIC_DIR))
            alembic_config.set_main_option("sqlalchemy.url", get_database_url())
            head_revision = ScriptDirectory.from_config(alembic_config).get_current_head()
        except CommandError as error:
            message = str(error)

    if message is None and head_revision is None:
        message = "No Alembic head revision is defined."

    if message is None:
        try:
            with get_engine().connect() as connection:
                current_revision = MigrationContext.configure(connection).get_current_revision()
        except SQLAlchemyError as error:
            message = str(error.__cause__ or error)
        else:
            if current_revision != head_revision:
                message = (
                    f"Database revision {current_revision or 'none'} does not match "
                    f"head {head_revision}."
                )

    if message is not None:
        return _completed_check("migrations", ok=False, message=message)

    return _completed_check(
        "migrations", ok=True, message=f"Migration revision {head_revision} is current."
    )


def _check_sqlite_vec() -> CompletedCheck:
    try:
        with get_engine().connect() as connection:
            version = connection.scalar(text("SELECT vec_version()"))
    except SQLAlchemyError as error:
        return _completed_check(
            "sqlite_vec",
            ok=False,
            message=str(error.__cause__ or error),
        )

    if version is None:
        return _completed_check(
            "sqlite_vec",
            ok=False,
            message="sqlite-vec extension did not load.",
        )

    return _completed_check(
        "sqlite_vec",
        ok=True,
        message=f"sqlite-vec {version} is loaded.",
    )
