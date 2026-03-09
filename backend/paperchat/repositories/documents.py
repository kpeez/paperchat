from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import Select, delete, func, select
from sqlalchemy.orm import Session

from paperchat.db.schema import Document, DocumentChunk, IngestionJob


@dataclass(frozen=True, slots=True)
class NewChunk:
    chunk_index: int
    text: str
    retrieval_text: str
    page_numbers: tuple[int, ...]
    headings: tuple[str, ...]
    warning_codes: tuple[str, ...]
    embedding: tuple[float, ...]


class DocumentRepository:
    """Persistence helpers for document rows and chunks."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, document_id: UUID) -> Document | None:
        return self._session.get(Document, document_id)

    def get_by_content_hash(self, content_hash: str) -> Document | None:
        return self._session.scalar(select(Document).where(Document.content_hash == content_hash))

    def list_all(self) -> tuple[Document, ...]:
        statement = select(Document).order_by(Document.created_at.desc())
        return tuple(self._session.scalars(statement))

    def add(self, document: Document) -> Document:
        self._session.add(document)
        self._session.flush()
        return document

    def create(
        self,
        *,
        content_hash: str,
        original_filename: str,
        display_name: str,
        file_path: str,
        status: str = "pending",
    ) -> Document:
        return self.add(
            Document(
                content_hash=content_hash,
                original_filename=original_filename,
                display_name=display_name,
                file_path=file_path,
                status=status,
            )
        )

    def delete(self, document: Document | UUID) -> None:
        if isinstance(document, UUID):
            target = self.get(document)
            if target is None:
                return
            self._session.delete(target)
            return
        self._session.delete(document)

    def replace_chunks(
        self,
        document_id: UUID,
        chunks: Sequence[DocumentChunk],
    ) -> None:
        self._session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
        self._session.add_all(chunks)
        self._session.flush()


class DocumentChunkRepository:
    """Persistence helpers for chunk rows."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def list_for_document(self, document_id: UUID) -> tuple[DocumentChunk, ...]:
        statement = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index.asc())
        )
        return tuple(self._session.scalars(statement))

    def replace_for_document(self, document_id: UUID, chunks: Sequence[NewChunk]) -> None:
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
        DocumentRepository(self._session).replace_chunks(document_id=document_id, chunks=rows)


class IngestionJobRepository:
    """Persistence helpers for ingestion attempts."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, job_id: UUID) -> IngestionJob | None:
        return self._session.get(IngestionJob, job_id)

    def get_latest_for_document(self, document_id: UUID) -> IngestionJob | None:
        statement = (
            select(IngestionJob)
            .where(IngestionJob.document_id == document_id)
            .order_by(IngestionJob.attempt.desc())
        )
        return self._session.scalar(statement.limit(1))

    def list_running(self) -> tuple[IngestionJob, ...]:
        statement = select(IngestionJob).where(IngestionJob.status == "running")
        return tuple(self._session.scalars(statement))

    def next_attempt(self, document_id: UUID) -> int:
        statement: Select[tuple[int | None]] = select(func.max(IngestionJob.attempt)).where(
            IngestionJob.document_id == document_id
        )
        current = self._session.scalar(statement)
        return (current or 0) + 1

    def add(self, job: IngestionJob) -> IngestionJob:
        self._session.add(job)
        self._session.flush()
        return job

    def create(
        self,
        *,
        document_id: UUID,
        status: str = "queued",
        stage: str = "queued",
    ) -> IngestionJob:
        return self.add(
            IngestionJob(
                document_id=document_id,
                attempt=self.next_attempt(document_id),
                status=status,
                stage=stage,
            )
        )
