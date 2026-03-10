from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from paperchat.main import create_app
from paperchat.models.bootstrap import CheckResult
from paperchat.services import bootstrap as bootstrap_service

client = TestClient(create_app())


def make_check(name: str, *, ok: bool, message: str) -> bootstrap_service.CompletedCheck:
    return bootstrap_service.CompletedCheck(name=name, result=CheckResult(ok=ok, message=message))


DEFAULT_READY_CHECKS = {
    "database": make_check("database", ok=True, message="Connected to SQLite."),
    "migrations": make_check(
        "migrations",
        ok=True,
        message="Migration revision 0001_initial_schema is current.",
    ),
    "sqlite_vec": make_check("sqlite_vec", ok=True, message="sqlite-vec 0.1.6 is loaded."),
}


def patch_bootstrap_checks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    database: bootstrap_service.CompletedCheck | None = None,
    migrations: bootstrap_service.CompletedCheck | None = None,
    sqlite_vec: bootstrap_service.CompletedCheck | None = None,
) -> None:
    checks = {
        **DEFAULT_READY_CHECKS,
        "database": database or DEFAULT_READY_CHECKS["database"],
        "migrations": migrations or DEFAULT_READY_CHECKS["migrations"],
        "sqlite_vec": sqlite_vec or DEFAULT_READY_CHECKS["sqlite_vec"],
    }
    monkeypatch.setattr(
        bootstrap_service,
        "_check_database",
        lambda: checks["database"],
    )
    monkeypatch.setattr(
        bootstrap_service,
        "_check_migrations",
        lambda: checks["migrations"],
    )
    monkeypatch.setattr(
        bootstrap_service,
        "_check_sqlite_vec",
        lambda: checks["sqlite_vec"],
    )


def get_bootstrap_data(
    monkeypatch: pytest.MonkeyPatch,
    **checks: bootstrap_service.CompletedCheck | None,
) -> dict[str, Any]:
    patch_bootstrap_checks(monkeypatch, **checks)
    return cast(dict[str, Any], client.get("/api/bootstrap").json())


def test_bootstrap_returns_200(monkeypatch: pytest.MonkeyPatch):
    patch_bootstrap_checks(monkeypatch)
    response = client.get("/api/bootstrap")
    assert response.status_code == 200


def test_bootstrap_has_valid_structure(monkeypatch: pytest.MonkeyPatch):
    data = get_bootstrap_data(monkeypatch)
    assert "app_version" in data
    assert "status" in data
    assert "checks" in data
    assert "errors" in data


def test_backend_check_is_ok(monkeypatch: pytest.MonkeyPatch):
    data = get_bootstrap_data(monkeypatch)
    assert data["checks"]["backend"]["ok"] is True


def test_status_is_ready(monkeypatch: pytest.MonkeyPatch):
    data = get_bootstrap_data(monkeypatch)
    assert data["status"] == "ready"


def test_all_expected_checks_present(monkeypatch: pytest.MonkeyPatch):
    data = get_bootstrap_data(monkeypatch)
    expected = {"backend", "database", "migrations", "sqlite_vec"}
    assert set(data["checks"].keys()) == expected


@pytest.mark.parametrize(
    ("check_patch", "expected_checks", "expected_errors"),
    [
        (
            {
                "database": make_check(
                    "database",
                    ok=False,
                    message="Database connection failed",
                )
            },
            {
                "database": {"ok": False, "message": "Database connection failed"},
                "migrations": {
                    "ok": False,
                    "message": bootstrap_service.DATABASE_BLOCKED_MESSAGE,
                },
                "sqlite_vec": {
                    "ok": False,
                    "message": bootstrap_service.DATABASE_BLOCKED_MESSAGE,
                },
            },
            {
                "code": "database_failed",
                "message": "Database connection failed",
                "action": bootstrap_service.CHECK_ACTIONS["database"],
            },
        ),
        (
            {"migrations": make_check("migrations", ok=False, message="Pending migrations")},
            {
                "migrations": {"ok": False, "message": "Pending migrations"},
            },
            {
                "code": "migrations_failed",
                "message": "Pending migrations",
                "action": bootstrap_service.CHECK_ACTIONS["migrations"],
            },
        ),
        (
            {
                "sqlite_vec": make_check(
                    "sqlite_vec",
                    ok=False,
                    message="sqlite-vec extension did not load.",
                )
            },
            {
                "sqlite_vec": {"ok": False, "message": "sqlite-vec extension did not load."},
            },
            {
                "code": "sqlite_vec_failed",
                "message": "sqlite-vec extension did not load.",
                "action": bootstrap_service.CHECK_ACTIONS["sqlite_vec"],
            },
        ),
    ],
)
def test_bootstrap_reports_storage_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
    check_patch: dict[str, bootstrap_service.CompletedCheck],
    expected_checks: dict[str, dict[str, str | bool | None]],
    expected_errors: dict[str, str],
):
    data = get_bootstrap_data(monkeypatch, **check_patch)

    assert data["status"] != "ready"
    for check_name, expected_check in expected_checks.items():
        assert data["checks"][check_name] == expected_check

    assert expected_errors in data["errors"]
