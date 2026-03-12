from fastapi import APIRouter, HTTPException, status

from paperchat.models.local_files import PickDocumentsResponse
from paperchat.services import local_files as local_files_service

router = APIRouter(prefix="/api/local-files")


@router.post("/pick-documents", response_model=PickDocumentsResponse)
def pick_local_documents() -> PickDocumentsResponse:
    try:
        paths = list(local_files_service.pick_documents())
    except local_files_service.LocalFilePickerUnavailableError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error

    return PickDocumentsResponse(paths=paths)
