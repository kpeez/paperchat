from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from sqlalchemy.orm import Session

from paperchat.db.schema import Document, DocumentChunk, IngestionJob
from paperchat.repositories.documents import DocumentRepository, IngestionJobRepository


@dataclass(frozen=True, slots=True)
class StoredChunk:
    chunk_index: int
    text: str
    retrieval_text: str
    page_numbers: tuple[int, ...]
    headings: tuple[str, ...]
    warning_codes: tuple[str, ...]
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DocumentRegistrationResult:
    mode: Literal["created", "existing", "retried"]
    document: Document
    job: IngestionJob


class DocumentRegistryRepository:
    """High-level persistence helpers for document registration flows."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._documents = DocumentRepository(session)
        self._jobs = IngestionJobRepository(session)

    def get_document(self, document_id: str) -> Document | None:
        return self._documents.get(document_id)

    def list_documents(self) -> tuple[Document, ...]:
        return self._documents.list_all()

    def get_job(self, job_id: str) -> IngestionJob | None:
        return self._jobs.get(job_id)

    def get_latest_job(self, document_id: str) -> IngestionJob | None:
        return self._jobs.get_latest_for_document(document_id)

    def register_document(
        self,
        *,
        file_path: Path,
        content_hash: str,
        original_filename: str,
        display_name: str,
    ) -> DocumentRegistrationResult:
        resolved_path = str(file_path.resolve())
        document = self._documents.get_by_content_hash(content_hash)

        if document is None:
            document = self._documents.add(
                Document(
                    content_hash=content_hash,
                    original_filename=original_filename,
                    display_name=display_name,
                    file_path=resolved_path,
                    status="pending",
                )
            )
            job = self._create_job(document_id=document.id)
            self._session.commit()
            return DocumentRegistrationResult(mode="created", document=document, job=job)

        document.file_path = resolved_path
        document.display_name = display_name

        latest_job = self._jobs.get_latest_for_document(document.id)
        if latest_job is None:
            latest_job = self._create_job(document_id=document.id)
            self._session.commit()
            return DocumentRegistrationResult(mode="retried", document=document, job=latest_job)

        if document.status == "failed":
            self._session.commit()
            return DocumentRegistrationResult(mode="existing", document=document, job=latest_job)

        self._session.commit()
        return DocumentRegistrationResult(mode="existing", document=document, job=latest_job)

    def replace_chunks(self, *, document_id: str, chunks: tuple[StoredChunk, ...]) -> None:
        rows = [
            DocumentChunk(
                document_id=document_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                retrieval_text=chunk.retrieval_text,
                page_numbers=list(chunk.page_numbers),
                headings=list(chunk.headings),
                warning_codes=list(chunk.warning_codes),
                embedding=list(chunk.embedding),
            )
            for chunk in chunks
        ]
        self._documents.replace_chunks(document_id=document_id, chunks=rows)

    def mark_job_running(self, *, job_id: str, stage: str) -> IngestionJob:
        job = self._require_job(job_id)
        document = self._require_document(job.document_id)
        job.status = "running"
        job.stage = stage
        job.started_at = datetime.now(UTC)
        job.error_code = None
        job.error_message = None
        document.status = "processing"
        document.error_code = None
        document.error_message = None
        self._session.commit()
        return job

    def mark_job_failed(
        self,
        *,
        job_id: str,
        error_code: str,
        error_message: str,
    ) -> IngestionJob:
        job = self._require_job(job_id)
        document = self._require_document(job.document_id)
        job.status = "failed"
        job.stage = "failed"
        job.error_code = error_code
        job.error_message = error_message
        job.finished_at = datetime.now(UTC)
        self._documents.replace_chunks(document_id=document.id, chunks=())
        document.status = "failed"
        document.parser_id = None
        document.chunker_id = None
        document.embedding_model_id = None
        document.chunk_count = 0
        document.error_code = error_code
        document.error_message = error_message
        self._session.commit()
        return job

    def mark_job_succeeded(
        self,
        *,
        job_id: str,
        parser_id: str,
        chunker_id: str,
        embedding_model_id: str,
        chunk_count: int,
    ) -> IngestionJob:
        job = self._require_job(job_id)
        document = self._require_document(job.document_id)
        job.status = "succeeded"
        job.stage = "complete"
        job.error_code = None
        job.error_message = None
        job.finished_at = datetime.now(UTC)
        document.status = "ready"
        document.parser_id = parser_id
        document.chunker_id = chunker_id
        document.embedding_model_id = embedding_model_id
        document.chunk_count = chunk_count
        document.error_code = None
        document.error_message = None
        self._session.commit()
        return job

    def mark_running_jobs_failed(self) -> int:
        failed = 0
        for job in self._jobs.list_running():
            self.mark_job_failed(
                job_id=job.id,
                error_code="worker_interrupted",
                error_message="Ingestion stopped before the job completed.",
            )
            failed += 1
        return failed

    def delete_document(self, document_id: str) -> None:
        document = self._require_document(document_id)
        self._documents.delete(document)
        self._session.commit()

    def _create_job(self, *, document_id: str) -> IngestionJob:
        attempt = self._jobs.next_attempt(document_id)
        return self._jobs.add(
            IngestionJob(
                document_id=document_id,
                attempt=attempt,
                status="queued",
                stage="queued",
            )
        )

    def _require_document(self, document_id: str) -> Document:
        document = self._documents.get(document_id)
        if document is None:
            msg = f"Document {document_id} was not found."
            raise LookupError(msg)
        return document

    def _require_job(self, job_id: str) -> IngestionJob:
        job = self._jobs.get(job_id)
        if job is None:
            msg = f"Ingestion job {job_id} was not found."
            raise LookupError(msg)
        return job
