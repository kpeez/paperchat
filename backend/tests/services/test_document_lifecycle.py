from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

from paperchat.db.schema import Document, DocumentChunk, IngestionJob
from paperchat.repositories.document_registry import DocumentRegistryRepository
from paperchat.services.documents import DocumentLifecycleBackend, DocumentService
from paperchat.services.embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingRuntimeError
from paperchat.services.ingestion import IngestionCoordinator, IngestionProcessor
from paperchat.services.ingestion_parsing import IngestionChunk, ParseResult


class FakeCoordinator:
    def __init__(self) -> None:
        self.job_ids: list[str] = []

    def enqueue(self, job_id: str) -> None:
        self.job_ids.append(job_id)


class FakeParser:
    def __init__(self, *, error: str | None = None) -> None:
        self.error = error

    def parse_document(self, *, document_id: str, pdf_path: Path) -> ParseResult:
        del document_id
        if self.error is not None:
            return ParseResult(
                parser_id="docling",
                chunker_id="hierarchical",
                parse_seconds=0.1,
                parser_warning_codes=(),
                chunks=(),
                error=self.error,
            )

        return ParseResult(
            parser_id="docling",
            chunker_id="hierarchical",
            parse_seconds=0.1,
            parser_warning_codes=(),
            chunks=(
                IngestionChunk(
                    chunk_index=0,
                    text="raw text",
                    retrieval_text="retrieval text",
                    page_numbers=(1,),
                    headings=("Intro",),
                    warning_codes=(),
                ),
            ),
        )


class FakeEmbedder:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.model_name = DEFAULT_EMBEDDING_MODEL

    def embed_documents(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if self.error is not None:
            raise self.error
        assert tuple(texts) == ("retrieval text",)
        return ((0.1, 0.2),)


def test_import_document_hashes_file_and_enqueues_job(db_session, tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    content = b"%PDF-1.4\nhello"
    pdf_path.write_bytes(content)
    coordinator = FakeCoordinator()
    service = DocumentService(db_session, coordinator=coordinator)

    result = service.import_document(file_path=pdf_path)

    assert result.job is not None
    assert result.document.content_hash == sha256(content).hexdigest()
    assert coordinator.job_ids == [result.job.id]


def test_duplicate_import_returns_existing_document_without_second_job(
    db_session,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nhello")
    coordinator = FakeCoordinator()
    service = DocumentService(db_session, coordinator=coordinator)

    created = service.import_document(file_path=pdf_path)
    duplicate = service.import_document(file_path=pdf_path)

    assert created.job is not None
    assert duplicate.job is not None
    assert duplicate.document.id == created.document.id
    assert duplicate.job.id == created.job.id
    assert duplicate.job_enqueued is False
    assert coordinator.job_ids == [created.job.id]


def test_failed_duplicate_import_returns_existing_failed_document_without_second_job(
    db_session,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nhello")
    coordinator = FakeCoordinator()
    service = DocumentService(db_session, coordinator=coordinator)

    created = service.import_document(file_path=pdf_path)
    assert created.job is not None

    repository = DocumentRegistryRepository(db_session)
    repository.mark_job_running(job_id=created.job.id, stage="parsing")
    repository.mark_job_failed(
        job_id=created.job.id,
        error_code="parse_failed",
        error_message="parse failed",
    )

    duplicate = service.import_document(file_path=pdf_path)

    assert duplicate.job is not None
    assert duplicate.document.id == created.document.id
    assert duplicate.document.status == "failed"
    assert duplicate.job.id == created.job.id
    assert duplicate.job_enqueued is False
    assert coordinator.job_ids == [created.job.id]


def test_retry_document_enqueues_new_attempt_for_failed_document(
    db_session, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nhello")
    coordinator = FakeCoordinator()
    service = DocumentService(db_session, coordinator=coordinator)
    created = service.import_document(file_path=pdf_path)
    assert created.job is not None

    repository = DocumentRegistryRepository(db_session)
    repository.mark_job_running(job_id=created.job.id, stage="parsing")
    repository.mark_job_failed(
        job_id=created.job.id,
        error_code="parse_failed",
        error_message="parse failed",
    )

    retried = service.retry_document(document_id=created.document.id)
    assert retried is not None
    assert retried.job is not None

    assert retried.document.id == created.document.id
    assert retried.job.attempt == 2
    assert coordinator.job_ids == [created.job.id, retried.job.id]


def test_retry_document_returns_existing_ready_document_without_second_job(
    db_session_factory,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nhello")

    backend = DocumentLifecycleBackend(session_factory=db_session_factory)
    created = backend.import_document(file_path=pdf_path)
    assert created.job is not None

    IngestionCoordinator(
        processor=IngestionProcessor(
            session_factory=db_session_factory,
            parser=FakeParser(),
            embedder=FakeEmbedder(),
        )
    ).process_now(created.job.id)

    retried = backend.retry_document(document_id=created.document.id)

    assert retried is not None
    assert retried.job_enqueued is False
    assert retried.job is not None
    assert retried.job.id == created.job.id
    assert retried.document.status == "ready"


def test_delete_document_hard_deletes_and_reimport_creates_new_document(
    db_session,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nhello")
    coordinator = FakeCoordinator()
    service = DocumentService(db_session, coordinator=coordinator)

    created = service.import_document(file_path=pdf_path)
    assert created.job is not None
    assert service.delete_document(document_id=created.document.id) is True
    assert service.get_document(document_id=created.document.id) is None

    reimported = service.import_document(file_path=pdf_path)

    assert reimported.job is not None
    assert reimported.document.id != created.document.id
    assert reimported.job.attempt == 1
    assert coordinator.job_ids == [created.job.id, reimported.job.id]


def test_recover_interrupted_jobs_marks_running_document_failed(
    db_session,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nhello")
    service = DocumentService(db_session, coordinator=FakeCoordinator())

    created = service.import_document(file_path=pdf_path)
    assert created.job is not None
    repository = DocumentRegistryRepository(db_session)
    repository.mark_job_running(job_id=created.job.id, stage="parsing")

    recovered = service.recover_interrupted_jobs()
    recovered_document = service.get_document(document_id=created.document.id)

    assert recovered == 1
    assert recovered_document is not None
    assert recovered_document.status == "failed"
    assert recovered_document.error_code == "worker_interrupted"


def test_document_lifecycle_backend_returns_existing_ready_document_without_second_job(
    db_session_factory,
    tmp_path: Path,
) -> None:
    original_path = tmp_path / "paper.pdf"
    duplicate_path = tmp_path / "paper-copy.pdf"
    payload = b"%PDF-1.4\nhello"
    original_path.write_bytes(payload)
    duplicate_path.write_bytes(payload)

    backend = DocumentLifecycleBackend(session_factory=db_session_factory)
    created = backend.import_document(file_path=original_path)
    assert created.job is not None

    IngestionCoordinator(
        processor=IngestionProcessor(
            session_factory=db_session_factory,
            parser=FakeParser(),
            embedder=FakeEmbedder(),
        )
    ).process_now(created.job.id)

    duplicate = backend.import_document(file_path=duplicate_path)

    assert duplicate.document.id == created.document.id
    assert duplicate.job is not None
    assert duplicate.job.id == created.job.id
    assert duplicate.job_enqueued is False

    with db_session_factory() as session:
        stored_document = session.get(Document, created.document.id)
        assert stored_document is not None
        assert stored_document.file_path == str(duplicate_path.resolve())
        assert session.query(IngestionJob).count() == 1


def test_ingestion_coordinator_processes_job_and_persists_chunks(
    db_session_factory,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nhello")
    with db_session_factory() as session:
        service = DocumentService(session, coordinator=FakeCoordinator())
        created = service.import_document(file_path=pdf_path)
        assert created.job is not None
    coordinator = IngestionCoordinator(
        processor=IngestionProcessor(
            session_factory=db_session_factory,
            parser=FakeParser(),
            embedder=FakeEmbedder(),
        )
    )

    coordinator.process_now(created.job.id)

    with db_session_factory() as session:
        document = session.get(Document, created.document.id)
        job = session.get(IngestionJob, created.job.id)
        assert document is not None
        assert job is not None
        assert document.status == "ready"
        assert document.chunk_count == 1
        assert document.embedding_model_id == DEFAULT_EMBEDDING_MODEL
        assert job.status == "succeeded"
        chunk = session.query(DocumentChunk).one()
        assert chunk.retrieval_text == "retrieval text"


def test_ingestion_coordinator_marks_job_failed_when_parser_errors(
    db_session_factory,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nhello")
    with db_session_factory() as session:
        service = DocumentService(session, coordinator=FakeCoordinator())
        created = service.import_document(file_path=pdf_path)
        assert created.job is not None
    coordinator = IngestionCoordinator(
        processor=IngestionProcessor(
            session_factory=db_session_factory,
            parser=FakeParser(error="parse failed"),
            embedder=FakeEmbedder(),
        )
    )
    coordinator.process_now(created.job.id)

    with db_session_factory() as session:
        document = session.get(Document, created.document.id)
        assert document is not None
        assert document.status == "failed"
        assert document.error_code == "parse_failed"


def test_ingestion_coordinator_marks_job_failed_when_embeddings_are_unavailable(
    db_session_factory,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\nhello")
    with db_session_factory() as session:
        service = DocumentService(session, coordinator=FakeCoordinator())
        created = service.import_document(file_path=pdf_path)
        assert created.job is not None
    coordinator = IngestionCoordinator(
        processor=IngestionProcessor(
            session_factory=db_session_factory,
            parser=FakeParser(),
            embedder=FakeEmbedder(error=EmbeddingRuntimeError("model cache missing")),
        )
    )

    coordinator.process_now(created.job.id)

    with db_session_factory() as session:
        document = session.get(Document, created.document.id)
        job = session.get(IngestionJob, created.job.id)
        assert document is not None
        assert job is not None
        assert document.status == "failed"
        assert document.error_code == "embedding_unavailable"
        assert document.embedding_model_id == DEFAULT_EMBEDDING_MODEL
        assert document.chunk_count == 0
        assert job.status == "failed"
        assert session.query(DocumentChunk).count() == 0
