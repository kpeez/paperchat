from fastapi import APIRouter

from paperchat.models.runtime import RuntimeResponse
from paperchat.services import runtime as runtime_service

router = APIRouter()


@router.get("/api/runtime", response_model=RuntimeResponse)
def runtime() -> RuntimeResponse:
    return runtime_service.build_runtime_response()
