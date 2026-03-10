from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Protocol

from sqlalchemy.orm import Session, sessionmaker

from paperchat.db.schema import DocumentChunk
from paperchat.models.documents import DocumentStatus, IngestionJobStatus
from paperchat.repositories import DocumentRepository, IngestionJobRepository
from paperchat.services.documents import IngestionCoordinatorProtocol


@dataclass(frozen=True, slots=True)
class IngestionError:
    code: str
    message: str


class DocumentParserProtocol(Protocol):
    def parse_document(self, *, document_id: str, pdf_path: Path) -> Any: ...


class EmbeddingServiceProtocol(Protocol):
    @property
    def model_name(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...


class IngestionProcessor:
    """Runs one ingestion attempt from document bytes to persisted chunks."""

    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        parser: DocumentParserProtocol,
        embedder: EmbeddingServiceProtocol,
    ) -> None:
        self._session_factory = session_factory
        self._parser = parser
        self._embedder = embedder

    def process(self, job_id: str) -> None:
        document_id, file_path = self._mark_job_running(job_id)
        if document_id is None or file_path is None:
            return

        try:
            parsed = self._parser.parse_document(document_id=document_id, pdf_path=Path(file_path))
            parse_error = getattr(parsed, "error", None)
            if parse_error:
                raise RuntimeError(str(parse_error))
            if not parsed.chunks:
                raise RuntimeError("Docling returned no chunks.")
            self._update_job_stage(job_id, stage="embedding")
            embeddings = self._embedder.embed_documents(
                tuple(chunk.retrieval_text for chunk in parsed.chunks)
            )
            self._update_job_stage(job_id, stage="persisting")
            self._persist_success(job_id=job_id, parsed=parsed, embeddings=embeddings)
        except Exception as error:
            self._persist_failure(job_id=job_id, error=_classify_error(error))

    def _mark_job_running(self, job_id: str) -> tuple[str | None, str | None]:
        with self._session_factory.begin() as session:
            jobs = IngestionJobRepository(session)
            documents = DocumentRepository(session)
            job = jobs.get(job_id)
            if job is None:
                return None, None
            document = documents.get(job.document_id)
            if document is None:
                return None, None
            job.status = IngestionJobStatus.running
            job.stage = "parsing"
            job.started_at = datetime.now(tz=UTC)
            document.status = DocumentStatus.processing
            document.error_code = None
            document.error_message = None
            return document.id, document.file_path

    def _update_job_stage(self, job_id: str, *, stage: str) -> None:
        with self._session_factory.begin() as session:
            job = IngestionJobRepository(session).get(job_id)
            if job is None:
                return
            job.stage = stage

    def _persist_success(
        self,
        *,
        job_id: str,
        parsed,
        embeddings: tuple[tuple[float, ...], ...],
    ) -> None:
        with self._session_factory.begin() as session:
            jobs = IngestionJobRepository(session)
            documents = DocumentRepository(session)
            job = jobs.get(job_id)
            if job is None:
                return
            document = documents.get(job.document_id)
            if document is None:
                return
            chunks = [
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    retrieval_text=chunk.retrieval_text,
                    page_numbers=list(chunk.page_numbers),
                    headings=list(chunk.headings),
                    warning_codes=list(chunk.warning_codes),
                    embedding=list(embeddings[index]),
                )
                for index, chunk in enumerate(parsed.chunks)
            ]
            documents.replace_chunks(document.id, chunks)
            document.status = DocumentStatus.ready
            document.parser_id = parsed.parser_id
            document.chunker_id = parsed.chunker_id
            document.embedding_model_id = self._embedder.model_name
            document.chunk_count = len(chunks)
            document.error_code = None
            document.error_message = None
            job.status = IngestionJobStatus.succeeded
            job.stage = "complete"
            job.error_code = None
            job.error_message = None
            job.finished_at = datetime.now(tz=UTC)

    def _persist_failure(self, *, job_id: str, error: IngestionError) -> None:
        with self._session_factory.begin() as session:
            jobs = IngestionJobRepository(session)
            documents = DocumentRepository(session)
            job = jobs.get(job_id)
            if job is None:
                return
            document = documents.get(job.document_id)
            if document is None:
                return
            documents.replace_chunks(document.id, ())
            document.status = DocumentStatus.failed
            document.parser_id = None
            document.chunker_id = None
            document.embedding_model_id = self._embedder.model_name
            document.chunk_count = 0
            document.error_code = error.code
            document.error_message = error.message
            job.status = IngestionJobStatus.failed
            job.stage = "failed"
            job.error_code = error.code
            job.error_message = error.message
            job.finished_at = datetime.now(tz=UTC)


class IngestionCoordinator(IngestionCoordinatorProtocol):
    """Single-process background worker for queued ingestion jobs."""

    def __init__(self, *, processor: IngestionProcessor) -> None:
        self._processor = processor
        self._queue: Queue[str] = Queue()
        self._stop_event = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, name="paperchat-ingestion", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def enqueue(self, job_id: str) -> None:
        self._queue.put(job_id)

    def process_now(self, job_id: str) -> None:
        self._processor.process(job_id)

    def process_job(self, job_id: str) -> None:
        self.process_now(job_id)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                job_id = self._queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                self._processor.process(job_id)
            finally:
                self._queue.task_done()


def _classify_error(error: Exception) -> IngestionError:
    if isinstance(error, FileNotFoundError):
        return IngestionError(code="file_missing", message=str(error))
    if _is_embedding_runtime_error(error):
        return IngestionError(code="embedding_unavailable", message=str(error))
    if str(error).strip().lower().startswith("parse "):
        return IngestionError(code="parse_failed", message=str(error))
    return IngestionError(code="ingestion_failed", message=str(error))


def _is_embedding_runtime_error(error: Exception) -> bool:
    return error.__class__.__name__ in {"EmbeddingRuntimeError", "EmbeddingDependencyError"}
