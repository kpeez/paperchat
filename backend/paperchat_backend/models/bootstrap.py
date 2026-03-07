from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"]


class CheckResult(BaseModel):
    ok: bool
    message: str | None = None


class BootstrapError(BaseModel):
    code: str
    message: str
    action: str


class BootstrapStatus(StrEnum):
    ready = "ready"
    starting = "starting"
    degraded = "degraded"
    error = "error"


class BootstrapResponse(BaseModel):
    app_version: str
    status: BootstrapStatus
    checks: dict[str, CheckResult]
    errors: list[BootstrapError]
