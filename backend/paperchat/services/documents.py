from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from sqlalchemy.orm import Session, sessionmaker

from paperchat.db.schema import Document, IngestionJob
from paperchat.models.documents import DocumentStatus, IngestionJobStatus
from paperchat.repositories import (
    DocumentRegistryRepository,
    DocumentRepository,
    IngestionJobRepository,
)


@dataclass(frozen=True, slots=True)
class IngestionJobRecord:
    id: str
    document_id: str
    attempt_number: int
    status: IngestionJobStatus
    stage: str | None = None
    error_code: str | None = None
    error_message: str | None = None

    @property
    def attempt(self) -> int:
        return self.attempt_number


@dataclass(frozen=True, slots=True)
class DocumentRecord:
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
    latest_job: IngestionJobRecord | None = None


@dataclass(frozen=True, slots=True)
class DocumentActionResult:
    document: DocumentRecord
    job_enqueued: bool
    job: IngestionJobRecord | None = None

    @property
    def job_id(self) -> str | None:
        if self.job_enqueued and self.job is not None:
            return self.job.id
        return None


class IngestionCoordinatorProtocol(Protocol):
    def enqueue(self, job_id: str) -> None: ...


class DocumentLifecycleProtocol(Protocol):
    def import_document(self, *, file_path: Path) -> DocumentActionResult: ...

    def list_documents(self) -> tuple[DocumentRecord, ...]: ...

    def get_document(self, *, document_id: str) -> DocumentRecord | None: ...

    def retry_document(self, *, document_id: str) -> DocumentActionResult | None: ...

    def delete_document(self, *, document_id: str) -> bool: ...

    def recover_interrupted_jobs(self) -> int: ...


class LifecycleBackendProtocol(DocumentLifecycleProtocol, Protocol):
    pass


class DocumentServiceProtocol(DocumentLifecycleProtocol, Protocol):
    pass


class DocumentService:
    """Session-bound lifecycle service used by integration tests and simple callers."""

    def __init__(
        self,
        session: Session,
        *,
        coordinator: IngestionCoordinatorProtocol,
    ) -> None:
        self._session = session
        self._registry = DocumentRegistryRepository(session)
        self._coordinator = coordinator

    def import_document(self, *, file_path: Path) -> DocumentActionResult:
        resolved_path = _resolve_file_path(file_path)
        registration = self._registry.register_document(
            file_path=resolved_path,
            content_hash=_read_content_hash(resolved_path),
            original_filename=resolved_path.name,
            display_name=resolved_path.name,
        )
        result = _to_action_result(
            registration.document,
            registration.job,
            enqueue=registration.mode != "existing",
        )
        _enqueue_result(self._coordinator, result)
        return result

    def list_documents(self) -> tuple[DocumentRecord, ...]:
        return tuple(
            _to_document_record(
                document,
                latest_job=self._registry.get_latest_job(document.id),
            )
            for document in self._registry.list_documents()
        )

    def get_document(self, *, document_id: str) -> DocumentRecord | None:
        document = self._registry.get_document(document_id)
        if document is None:
            return None
        return _to_document_record(
            document,
            latest_job=self._registry.get_latest_job(document_id),
        )

    def retry_document(self, *, document_id: str) -> DocumentActionResult | None:
        document = self._registry.get_document(document_id)
        if document is None:
            return None

        jobs = IngestionJobRepository(self._session)
        latest_job = jobs.get_latest_for_document(document_id)
        if latest_job is not None and latest_job.status in {
            IngestionJobStatus.queued,
            IngestionJobStatus.running,
        }:
            return _to_action_result(document, latest_job, enqueue=False)

        if document.status != DocumentStatus.failed:
            return _to_action_result(document, latest_job, enqueue=False)

        job = _create_queued_job(jobs, document.id)
        document.status = DocumentStatus.pending
        document.error_code = None
        document.error_message = None
        self._session.commit()
        result = _to_action_result(document, job, enqueue=True)
        _enqueue_result(self._coordinator, result)
        return result

    def delete_document(self, *, document_id: str) -> bool:
        document = self._registry.get_document(document_id)
        if document is None:
            return False
        self._registry.delete_document(document_id)
        return True

    def recover_interrupted_jobs(self) -> int:
        return self._registry.mark_running_jobs_failed()


class DocumentLifecycleService:
    def __init__(
        self,
        *,
        backend: LifecycleBackendProtocol,
        coordinator: IngestionCoordinatorProtocol,
    ) -> None:
        self._backend = backend
        self._coordinator = coordinator

    def import_document(self, *, file_path: Path) -> DocumentActionResult:
        result = self._backend.import_document(file_path=file_path)
        _enqueue_result(self._coordinator, result)
        return result

    def list_documents(self) -> tuple[DocumentRecord, ...]:
        return self._backend.list_documents()

    def get_document(self, *, document_id: str) -> DocumentRecord | None:
        return self._backend.get_document(document_id=document_id)

    def retry_document(self, *, document_id: str) -> DocumentActionResult | None:
        result = self._backend.retry_document(document_id=document_id)
        if result is None:
            return None
        _enqueue_result(self._coordinator, result)
        return result

    def delete_document(self, *, document_id: str) -> bool:
        return self._backend.delete_document(document_id=document_id)

    def recover_interrupted_jobs(self) -> int:
        return self._backend.recover_interrupted_jobs()


class DocumentLifecycleBackend:
    """Database-backed document lifecycle behavior."""

    def __init__(self, *, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def import_document(self, *, file_path: Path) -> DocumentActionResult:
        resolved_path = _resolve_file_path(file_path)
        content_hash = _read_content_hash(resolved_path)

        with self._session_factory.begin() as session:
            documents = DocumentRepository(session)
            jobs = IngestionJobRepository(session)
            existing = documents.get_by_content_hash(content_hash)
            if existing is not None:
                existing.file_path = str(resolved_path)
                latest_job = jobs.get_latest_for_document(existing.id)
                return DocumentActionResult(
                    document=_to_document_record(existing, latest_job=latest_job),
                    job_enqueued=False,
                    job=_to_job_record(latest_job),
                )

            document = documents.add(
                Document(
                    content_hash=content_hash,
                    original_filename=resolved_path.name,
                    display_name=resolved_path.name,
                    file_path=str(resolved_path),
                    status=DocumentStatus.pending,
                )
            )
            job = _create_queued_job(jobs, document.id)
            return DocumentActionResult(
                document=_to_document_record(document, latest_job=job),
                job_enqueued=True,
                job=_to_job_record(job),
            )

    def list_documents(self) -> tuple[DocumentRecord, ...]:
        with self._session_factory() as session:
            documents = DocumentRepository(session)
            jobs = IngestionJobRepository(session)
            return tuple(
                _to_document_record(document, latest_job=jobs.get_latest_for_document(document.id))
                for document in documents.list_all()
            )

    def get_document(self, *, document_id: str) -> DocumentRecord | None:
        with self._session_factory() as session:
            documents = DocumentRepository(session)
            document = documents.get(document_id)
            if document is None:
                return None
            latest_job = IngestionJobRepository(session).get_latest_for_document(document_id)
            return _to_document_record(document, latest_job=latest_job)

    def retry_document(self, *, document_id: str) -> DocumentActionResult | None:
        with self._session_factory.begin() as session:
            documents = DocumentRepository(session)
            jobs = IngestionJobRepository(session)
            document = documents.get(document_id)
            if document is None:
                return None

            latest_job = jobs.get_latest_for_document(document_id)
            if latest_job is not None and latest_job.status in {
                IngestionJobStatus.queued,
                IngestionJobStatus.running,
            }:
                return DocumentActionResult(
                    document=_to_document_record(document, latest_job=latest_job),
                    job_enqueued=False,
                    job=_to_job_record(latest_job),
                )

            if document.status != DocumentStatus.failed:
                return DocumentActionResult(
                    document=_to_document_record(document, latest_job=latest_job),
                    job_enqueued=False,
                    job=_to_job_record(latest_job),
                )

            job = _create_queued_job(jobs, document.id)
            document.status = DocumentStatus.pending
            document.error_code = None
            document.error_message = None
            return DocumentActionResult(
                document=_to_document_record(document, latest_job=job),
                job_enqueued=True,
                job=_to_job_record(job),
            )

    def delete_document(self, *, document_id: str) -> bool:
        with self._session_factory.begin() as session:
            documents = DocumentRepository(session)
            document = documents.get(document_id)
            if document is None:
                return False
            documents.delete(document)
            return True

    def recover_interrupted_jobs(self) -> int:
        with self._session_factory.begin() as session:
            jobs = IngestionJobRepository(session)
            documents = DocumentRepository(session)
            running_jobs = jobs.list_running()
            for job in running_jobs:
                job.status = IngestionJobStatus.failed
                job.stage = "failed"
                job.error_code = "worker_interrupted"
                job.error_message = "The backend restarted before ingestion completed."
                job.finished_at = datetime.now(tz=UTC)
                document = documents.get(job.document_id)
                if document is not None:
                    document.status = DocumentStatus.failed
                    document.error_code = job.error_code
                    document.error_message = job.error_message
            return len(running_jobs)


def _resolve_file_path(file_path: Path) -> Path:
    resolved_path = file_path.expanduser().resolve()
    if not resolved_path.is_file():
        msg = f"Document path does not exist: {resolved_path}"
        raise FileNotFoundError(msg)
    return resolved_path


def _read_content_hash(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def _to_document_record(
    document: Document,
    *,
    latest_job: IngestionJob | None,
) -> DocumentRecord:
    return DocumentRecord(
        id=document.id,
        content_hash=document.content_hash,
        original_filename=document.original_filename,
        display_name=document.display_name,
        file_path=document.file_path,
        status=DocumentStatus(document.status),
        chunk_count=document.chunk_count,
        parser_id=document.parser_id,
        chunker_id=document.chunker_id,
        embedding_model_id=document.embedding_model_id,
        error_code=document.error_code,
        error_message=document.error_message,
        latest_job=_to_job_record(latest_job),
    )


def _to_job_record(job: IngestionJob | None) -> IngestionJobRecord | None:
    if job is None:
        return None
    return IngestionJobRecord(
        id=job.id,
        document_id=job.document_id,
        attempt_number=job.attempt,
        status=IngestionJobStatus(job.status),
        stage=job.stage,
        error_code=job.error_code,
        error_message=job.error_message,
    )


def _to_action_result(
    document: Document,
    job: IngestionJob | None,
    *,
    enqueue: bool,
) -> DocumentActionResult:
    return DocumentActionResult(
        document=_to_document_record(document, latest_job=job),
        job_enqueued=enqueue,
        job=_to_job_record(job),
    )


def _create_queued_job(jobs: IngestionJobRepository, document_id: str) -> IngestionJob:
    return jobs.add(
        IngestionJob(
            document_id=document_id,
            attempt=jobs.next_attempt(document_id),
            status=IngestionJobStatus.queued,
            stage="queued",
        )
    )


def _enqueue_result(
    coordinator: IngestionCoordinatorProtocol,
    result: DocumentActionResult,
) -> None:
    if result.job_id is not None:
        coordinator.enqueue(result.job_id)
