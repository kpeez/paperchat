from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError

from paperchat.db.schema import Document, DocumentChunk, IngestionJob
from paperchat.repositories.document_registry import DocumentRegistryRepository, StoredChunk
from paperchat.services.embeddings import DEFAULT_EMBEDDING_MODEL


def content_hash_for(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_register_document_creates_pending_document_and_queued_job(
    db_session, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    payload = b"%PDF-1.4\n"
    pdf_path.write_bytes(payload)

    repository = DocumentRegistryRepository(db_session)
    result = repository.register_document(
        file_path=pdf_path,
        content_hash=content_hash_for(payload),
        original_filename=pdf_path.name,
        display_name=pdf_path.name,
    )

    assert result.mode == "created"
    assert result.document.status == "pending"
    assert result.job.status == "queued"
    assert result.job.attempt == 1


def test_register_document_returns_existing_ready_document_without_second_job(
    db_session,
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "paper.pdf"
    payload = b"%PDF-1.4\n"
    first_path.write_bytes(payload)
    second_path = tmp_path / "paper-copy.pdf"
    second_path.write_bytes(payload)

    repository = DocumentRegistryRepository(db_session)
    created = repository.register_document(
        file_path=first_path,
        content_hash=content_hash_for(payload),
        original_filename=first_path.name,
        display_name=first_path.name,
    )
    repository.mark_job_running(job_id=created.job.id, stage="parsing")
    repository.mark_job_succeeded(
        job_id=created.job.id,
        parser_id="docling",
        chunker_id="hierarchical",
        embedding_model_id=DEFAULT_EMBEDDING_MODEL,
        chunk_count=0,
    )

    duplicate = repository.register_document(
        file_path=second_path,
        content_hash=content_hash_for(payload),
        original_filename=second_path.name,
        display_name=second_path.name,
    )

    assert duplicate.mode == "existing"
    assert duplicate.document.id == created.document.id
    assert db_session.query(IngestionJob).count() == 1


def test_register_document_returns_existing_failed_document_without_second_job(
    db_session,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    payload = b"%PDF-1.4\n"
    pdf_path.write_bytes(payload)

    repository = DocumentRegistryRepository(db_session)
    created = repository.register_document(
        file_path=pdf_path,
        content_hash=content_hash_for(payload),
        original_filename=pdf_path.name,
        display_name=pdf_path.name,
    )
    repository.mark_job_running(job_id=created.job.id, stage="parsing")
    repository.mark_job_failed(
        job_id=created.job.id,
        error_code="parse_failed",
        error_message="parse failed",
    )

    duplicate = repository.register_document(
        file_path=pdf_path,
        content_hash=content_hash_for(payload),
        original_filename=pdf_path.name,
        display_name=pdf_path.name,
    )

    assert duplicate.mode == "existing"
    assert duplicate.document.id == created.document.id
    assert duplicate.job.attempt == 1
    assert duplicate.document.status == "failed"
    assert duplicate.document.error_code == "parse_failed"
    assert db_session.query(IngestionJob).count() == 1


def test_replace_chunks_persists_retrieval_text_and_warning_codes_and_delete_cascades(
    db_session,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    payload = b"%PDF-1.4\n"
    pdf_path.write_bytes(payload)

    repository = DocumentRegistryRepository(db_session)
    created = repository.register_document(
        file_path=pdf_path,
        content_hash=content_hash_for(payload),
        original_filename=pdf_path.name,
        display_name=pdf_path.name,
    )

    repository.replace_chunks(
        document_id=created.document.id,
        chunks=(
            StoredChunk(
                chunk_index=0,
                text="raw text",
                retrieval_text="retrieval text",
                page_numbers=(1,),
                headings=("Intro",),
                warning_codes=("missing_page_numbers",),
                embedding=(0.1, 0.2),
            ),
        ),
    )
    repository.mark_job_running(job_id=created.job.id, stage="persisting")
    repository.mark_job_succeeded(
        job_id=created.job.id,
        parser_id="docling",
        chunker_id="hierarchical",
        embedding_model_id=DEFAULT_EMBEDDING_MODEL,
        chunk_count=1,
    )

    stored_document = db_session.query(Document).one()
    stored_chunk = db_session.query(DocumentChunk).one()
    assert stored_document.chunk_count == 1
    assert stored_document.parser_id == "docling"
    assert stored_document.chunker_id == "hierarchical"
    assert stored_document.embedding_model_id == DEFAULT_EMBEDDING_MODEL
    assert stored_chunk.retrieval_text == "retrieval text"
    assert stored_chunk.warning_codes == ["missing_page_numbers"]

    repository.delete_document(created.document.id)

    assert db_session.query(Document).count() == 0
    assert db_session.query(DocumentChunk).count() == 0
    assert db_session.query(IngestionJob).count() == 0


def test_mark_job_failed_clears_partial_chunks_and_registry_metadata(
    db_session,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    payload = b"%PDF-1.4\n"
    pdf_path.write_bytes(payload)

    repository = DocumentRegistryRepository(db_session)
    created = repository.register_document(
        file_path=pdf_path,
        content_hash=content_hash_for(payload),
        original_filename=pdf_path.name,
        display_name=pdf_path.name,
    )

    repository.replace_chunks(
        document_id=created.document.id,
        chunks=(
            StoredChunk(
                chunk_index=0,
                text="raw text",
                retrieval_text="retrieval text",
                page_numbers=(1,),
                headings=("Intro",),
                warning_codes=("missing_page_numbers",),
                embedding=(0.1, 0.2),
            ),
        ),
    )
    created.document.parser_id = "docling"
    created.document.chunker_id = "hierarchical"
    created.document.chunk_count = 1
    db_session.commit()

    repository.mark_job_running(job_id=created.job.id, stage="persisting")
    repository.mark_job_failed(
        job_id=created.job.id,
        error_code="persist_failed",
        error_message="persist failed",
    )

    db_session.refresh(created.document)

    assert created.document.status == "failed"
    assert created.document.chunk_count == 0
    assert created.document.parser_id is None
    assert created.document.chunker_id is None
    assert created.document.embedding_model_id is None
    assert db_session.query(DocumentChunk).count() == 0


def test_delete_then_reimport_same_hash_creates_new_document_uuid(
    db_session,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    payload = b"%PDF-1.4\n"
    pdf_path.write_bytes(payload)

    repository = DocumentRegistryRepository(db_session)
    created = repository.register_document(
        file_path=pdf_path,
        content_hash=content_hash_for(payload),
        original_filename=pdf_path.name,
        display_name=pdf_path.name,
    )

    repository.delete_document(created.document.id)

    recreated = repository.register_document(
        file_path=pdf_path,
        content_hash=content_hash_for(payload),
        original_filename=pdf_path.name,
        display_name=pdf_path.name,
    )

    assert recreated.mode == "created"
    assert recreated.document.id != created.document.id
    assert recreated.job.attempt == 1


def test_register_document_rejects_non_sha256_content_hash(db_session, tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")

    repository = DocumentRegistryRepository(db_session)

    with pytest.raises(IntegrityError):
        repository.register_document(
            file_path=pdf_path,
            content_hash="hash-1",
            original_filename=pdf_path.name,
            display_name=pdf_path.name,
        )
