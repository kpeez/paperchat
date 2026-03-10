from enum import StrEnum

from pydantic import BaseModel


class DocumentStatus(StrEnum):
    pending = "pending"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class IngestionJobStatus(StrEnum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class DocumentImportRequest(BaseModel):
    file_path: str


class IngestionJobResponse(BaseModel):
    id: str
    document_id: str
    attempt_number: int
    status: IngestionJobStatus
    stage: str | None = None
    error_code: str | None = None
    error_message: str | None = None


class DocumentResponse(BaseModel):
    id: str
    content_hash: str
    original_filename: str
    display_name: str
    file_path: str
    status: DocumentStatus
    chunk_count: int
    parser_id: str | None = None
    chunker_id: str | None = None
    embedding_model_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    latest_job: IngestionJobResponse | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


class DocumentMutationResponse(BaseModel):
    document: DocumentResponse
    job_enqueued: bool
