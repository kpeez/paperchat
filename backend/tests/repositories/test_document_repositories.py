from __future__ import annotations

from paperchat.repositories.documents import (
    DocumentChunkRepository,
    DocumentRepository,
    IngestionJobRepository,
    NewChunk,
    VectorChunkRepository,
)

VALID_CONTENT_HASH = "a" * 64


def test_document_repository_creates_and_fetches_document_by_hash(db_session) -> None:
    repository = DocumentRepository(db_session)

    created = repository.create(
        content_hash=VALID_CONTENT_HASH,
        original_filename="paper.pdf",
        display_name="Paper",
        file_path="/tmp/paper.pdf",
    )

    fetched = repository.get_by_content_hash(VALID_CONTENT_HASH)

    assert created.status == "pending"
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.original_filename == "paper.pdf"


def test_ingestion_job_repository_increments_attempt_numbers(db_session) -> None:
    document_repository = DocumentRepository(db_session)
    job_repository = IngestionJobRepository(db_session)
    document = document_repository.create(
        content_hash=VALID_CONTENT_HASH,
        original_filename="paper.pdf",
        display_name="Paper",
        file_path="/tmp/paper.pdf",
    )

    first_job = job_repository.create(document_id=document.id)
    second_job = job_repository.create(document_id=document.id)

    latest = job_repository.get_latest_for_document(document.id)

    assert first_job.status == "queued"
    assert first_job.attempt_number == 1
    assert second_job.attempt_number == 2
    assert latest is not None
    assert latest.id == second_job.id


def test_chunk_repository_replaces_chunks_with_retrieval_text_and_warning_codes(db_session) -> None:
    document_repository = DocumentRepository(db_session)
    chunk_repository = DocumentChunkRepository(db_session)
    document = document_repository.create(
        content_hash=VALID_CONTENT_HASH,
        original_filename="paper.pdf",
        display_name="Paper",
        file_path="/tmp/paper.pdf",
    )

    chunk_repository.replace_for_document(
        document.id,
        (
            NewChunk(
                chunk_index=0,
                text="raw text",
                retrieval_text="retrieval text",
                page_numbers=(1, 2),
                headings=("Methods",),
                warning_codes=("missing_headings",),
                embedding=(0.1, 0.2),
            ),
        ),
    )

    stored = chunk_repository.list_for_document(document.id)

    assert len(stored) == 1
    assert stored[0].retrieval_text == "retrieval text"
    assert stored[0].warning_codes == ["missing_headings"]


def test_delete_document_hard_deletes_related_chunks_and_jobs(db_session) -> None:
    document_repository = DocumentRepository(db_session)
    chunk_repository = DocumentChunkRepository(db_session)
    job_repository = IngestionJobRepository(db_session)
    document = document_repository.create(
        content_hash=VALID_CONTENT_HASH,
        original_filename="paper.pdf",
        display_name="Paper",
        file_path="/tmp/paper.pdf",
    )
    job_repository.create(document_id=document.id)
    chunk_repository.replace_for_document(
        document.id,
        (
            NewChunk(
                chunk_index=0,
                text="raw text",
                retrieval_text="retrieval text",
                page_numbers=(1,),
                headings=("Intro",),
                warning_codes=(),
                embedding=(0.1, 0.2),
            ),
        ),
    )

    document_repository.delete(document.id)
    db_session.flush()

    assert document_repository.get(document.id) is None
    assert chunk_repository.list_for_document(document.id) == ()
    assert job_repository.get_latest_for_document(document.id) is None


def test_vector_chunk_repository_searches_ready_documents(db_session) -> None:
    document_repository = DocumentRepository(db_session)
    chunk_repository = DocumentChunkRepository(db_session)

    first = document_repository.create(
        content_hash="b" * 64,
        original_filename="attention.pdf",
        display_name="Attention",
        file_path="/tmp/attention.pdf",
        status="ready",
    )
    second = document_repository.create(
        content_hash="c" * 64,
        original_filename="bert.pdf",
        display_name="BERT",
        file_path="/tmp/bert.pdf",
        status="ready",
    )
    chunk_repository.replace_for_document(
        first.id,
        (
            NewChunk(
                chunk_index=0,
                text="attention text",
                retrieval_text="attention text",
                page_numbers=(1,),
                headings=("Overview",),
                warning_codes=(),
                embedding=(1.0, 0.0),
            ),
        ),
    )
    chunk_repository.replace_for_document(
        second.id,
        (
            NewChunk(
                chunk_index=0,
                text="bert text",
                retrieval_text="bert text",
                page_numbers=(2,),
                headings=("Methods",),
                warning_codes=(),
                embedding=(0.0, 1.0),
            ),
        ),
    )

    results = VectorChunkRepository(db_session).search(
        query_embedding=(1.0, 0.0),
        document_ids=(first.id, second.id),
        limit=2,
    )

    assert [result.document_id for result in results] == [first.id, second.id]
