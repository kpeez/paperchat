import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from shutil import which

from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from alembic.util.exc import CommandError
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from paperchat_backend import __version__
from paperchat_backend.config import (
    ALEMBIC_DIR,
    ALEMBIC_INI_PATH,
    COMPOSE_FILE,
    DOCKER_BINARY,
    DOCKER_SERVICE,
    get_database_url,
)
from paperchat_backend.db.engine import get_engine
from paperchat_backend.models.bootstrap import (
    BootstrapError,
    BootstrapResponse,
    BootstrapStatus,
    CheckResult,
)

CHECK_ACTIONS = {
    "docker": "Install Docker Desktop and start the local PaperChat database container.",
    "database": "Start the local Postgres container and confirm the configured port is reachable.",
    "migrations": "Run the backend migrations so the database revision matches the application.",
    "pgvector": "Recreate the local database with the pgvector image or install the vector extension.",
}
DATABASE_BLOCKED_MESSAGE = "Database must be reachable before this check can run."
DOCKER_COMMAND_TIMEOUT_SECONDS = 5


@dataclass(frozen=True, slots=True)
class CompletedCheck:
    name: str
    result: CheckResult


def _completed_check(name: str, *, ok: bool, message: str) -> CompletedCheck:
    return CompletedCheck(name=name, result=CheckResult(ok=ok, message=message))


def build_bootstrap_response() -> BootstrapResponse:
    checks = {"backend": CheckResult(ok=True, message=f"Backend {__version__} is running.")}

    for check in (_check_docker_runtime(), _check_database()):
        checks[check.name] = check.result

    if checks["database"].ok:
        for check in (_check_migrations(), _check_pgvector()):
            checks[check.name] = check.result
    else:
        for name in ("migrations", "pgvector"):
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


def _check_docker_runtime() -> CompletedCheck:
    if not COMPOSE_FILE.is_file():
        return _completed_check(
            "docker",
            ok=False,
            message=f"Compose file not found at {COMPOSE_FILE}.",
        )

    docker_binary = which(DOCKER_BINARY)
    if docker_binary is None:
        return _completed_check("docker", ok=False, message="Docker CLI is not installed.")

    version_result = _run_docker_command([docker_binary, "compose", "version"])
    if isinstance(version_result, CompletedCheck):
        return version_result

    compose_command = [docker_binary, "compose", "-f", str(COMPOSE_FILE)]
    ps_result = _run_docker_command([*compose_command, "ps", "--services", "--status", "running"])
    if isinstance(ps_result, CompletedCheck):
        return ps_result

    if DOCKER_SERVICE not in ps_result.stdout.split():
        return _completed_check(
            "docker",
            ok=False,
            message=f"Compose service '{DOCKER_SERVICE}' is not running.",
        )

    return _completed_check("docker", ok=True, message="Docker runtime is ready.")


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

    return _completed_check("database", ok=True, message="Connected to Postgres.")


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


def _check_pgvector() -> CompletedCheck:
    try:
        with get_engine().connect() as connection:
            extension_version = connection.scalar(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
    except SQLAlchemyError as error:
        return _completed_check(
            "pgvector",
            ok=False,
            message=str(error.__cause__ or error),
        )

    if extension_version is None:
        return _completed_check(
            "pgvector",
            ok=False,
            message="pgvector extension is not installed.",
        )

    return _completed_check(
        "pgvector",
        ok=True,
        message=f"pgvector extension {extension_version} is installed.",
    )


def _run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=DOCKER_COMMAND_TIMEOUT_SECONDS,
    )


def _run_docker_command(
    command: Sequence[str],
) -> subprocess.CompletedProcess[str] | CompletedCheck:
    try:
        result = _run_command(command)
    except subprocess.TimeoutExpired:
        return _completed_check("docker", ok=False, message="Docker command timed out.")

    if result.returncode == 0:
        return result

    return _completed_check(
        "docker",
        ok=False,
        message=_command_error_message(result.stderr, result.stdout),
    )


def _command_error_message(stderr: str, stdout: str) -> str:
    message = stderr.strip() or stdout.strip()
    if message:
        return message

    return "Docker command failed without output."
