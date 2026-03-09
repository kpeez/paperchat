from fastapi import APIRouter

from paperchat_backend.models.bootstrap import BootstrapResponse, HealthResponse
from paperchat_backend.services import build_bootstrap_response

router = APIRouter()


@router.get("/api/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/api/bootstrap")
def bootstrap() -> BootstrapResponse:
    return build_bootstrap_response()
