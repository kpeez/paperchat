from fastapi import APIRouter

from paperchat.models.bootstrap import BootstrapResponse, HealthResponse
from paperchat.services import build_bootstrap_response

router = APIRouter()


@router.get("/api/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/api/bootstrap")
def bootstrap() -> BootstrapResponse:
    return build_bootstrap_response()
