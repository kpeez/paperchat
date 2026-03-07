from fastapi import APIRouter

from paperchat_backend.models.bootstrap import (
    BootstrapError,
    BootstrapResponse,
    BootstrapStatus,
    CheckResult,
    HealthResponse,
)

router = APIRouter()


@router.get("/api/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/api/bootstrap")
def bootstrap() -> BootstrapResponse:
    checks = {
        "backend": CheckResult(ok=True),
        "database": CheckResult(ok=True),
        "docker": CheckResult(ok=True),
        "migrations": CheckResult(ok=True),
    }

    errors: list[BootstrapError] = []
    for name, check in checks.items():
        if not check.ok:
            errors.append(
                BootstrapError(
                    code=f"{name}_failed",
                    message=check.message or f"{name} check failed",
                    action=f"Check {name} configuration",
                )
            )

    status = BootstrapStatus.ready if all(c.ok for c in checks.values()) else BootstrapStatus.error

    return BootstrapResponse(
        app_version="0.1.0",
        status=status,
        checks=checks,
        errors=errors,
    )
