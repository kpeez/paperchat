from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, Response, status

from paperchat.models.documents import (
    DocumentImportRequest,
    DocumentListResponse,
    DocumentMutationResponse,
    DocumentResponse,
    IngestionJobResponse,
)
from paperchat.services.documents import (
    DocumentActionResult,
    DocumentRecord,
    DocumentServiceProtocol,
    IngestionJobRecord,
)

router = APIRouter(prefix="/api/documents")


def _get_document_service(request: Request) -> DocumentServiceProtocol:
    service = getattr(request.app.state, "document_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document service is not configured.",
        )
    return service


def _to_job_response(job: IngestionJobRecord | None) -> IngestionJobResponse | None:
    if job is None:
        return None
    return IngestionJobResponse(
        id=job.id,
        document_id=job.document_id,
        attempt_number=job.attempt_number,
        status=job.status,
        stage=job.stage,
        error_code=job.error_code,
        error_message=job.error_message,
    )


def _to_document_response(document: DocumentRecord) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        content_hash=document.content_hash,
        original_filename=document.original_filename,
        display_name=document.display_name,
        file_path=document.file_path,
        status=document.status,
        chunk_count=document.chunk_count,
        parser_id=document.parser_id,
        chunker_id=document.chunker_id,
        embedding_model_id=document.embedding_model_id,
        error_code=document.error_code,
        error_message=document.error_message,
        latest_job=_to_job_response(document.latest_job),
    )


def _to_mutation_response(result: DocumentActionResult) -> DocumentMutationResponse:
    return DocumentMutationResponse(
        document=_to_document_response(result.document),
        job_enqueued=result.job_enqueued,
    )


@router.post("/import", response_model=DocumentMutationResponse)
def import_document(payload: DocumentImportRequest, request: Request) -> DocumentMutationResponse:
    service = _get_document_service(request)
    result = service.import_document(file_path=Path(payload.file_path))
    return _to_mutation_response(result)


@router.get("", response_model=DocumentListResponse)
def list_documents(request: Request) -> DocumentListResponse:
    service = _get_document_service(request)
    documents = [_to_document_response(document) for document in service.list_documents()]
    return DocumentListResponse(documents=documents)


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(document_id: UUID, request: Request) -> DocumentResponse:
    service = _get_document_service(request)
    document = service.get_document(document_id=str(document_id))
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return _to_document_response(document)


@router.post("/{document_id}/retry", response_model=DocumentMutationResponse)
def retry_document(document_id: UUID, request: Request) -> DocumentMutationResponse:
    service = _get_document_service(request)
    result = service.retry_document(document_id=str(document_id))
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return _to_mutation_response(result)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: UUID, request: Request) -> Response:
    service = _get_document_service(request)
    deleted = service.delete_document(document_id=str(document_id))
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
