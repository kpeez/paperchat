from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import sqlite_vec
from sqlalchemy import Select, delete, func, select, text
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


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_name: str
    text: str
    retrieval_text: str
    page_numbers: tuple[int, ...]
    headings: tuple[str, ...]
    warning_codes: tuple[str, ...]
    distance: float


VECTOR_TABLE_NAME = "vec_chunks"


class DocumentRepository:
    """Persistence helpers for document rows and chunks."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, document_id: str) -> Document | None:
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

    def delete(self, document: Document | str) -> None:
        if isinstance(document, str):
            target = self.get(document)
            if target is None:
                return
            VectorChunkRepository(self._session).delete_for_document(target.id)
            self._session.delete(target)
            return
        VectorChunkRepository(self._session).delete_for_document(document.id)
        self._session.delete(document)

    def replace_chunks(
        self,
        document_id: str,
        chunks: Sequence[DocumentChunk],
    ) -> None:
        vector_chunks = VectorChunkRepository(self._session)
        vector_chunks.delete_for_document(document_id)
        self._session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
        self._session.add_all(chunks)
        self._session.flush()
        vector_chunks.upsert_chunks(chunks)


class DocumentChunkRepository:
    """Persistence helpers for chunk rows."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def list_for_document(self, document_id: str) -> tuple[DocumentChunk, ...]:
        statement = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index.asc())
        )
        return tuple(self._session.scalars(statement))

    def replace_for_document(self, document_id: str, chunks: Sequence[NewChunk]) -> None:
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

    def get(self, job_id: str) -> IngestionJob | None:
        return self._session.get(IngestionJob, job_id)

    def get_latest_for_document(self, document_id: str) -> IngestionJob | None:
        statement = (
            select(IngestionJob)
            .where(IngestionJob.document_id == document_id)
            .order_by(IngestionJob.attempt.desc())
        )
        return self._session.scalar(statement.limit(1))

    def list_running(self) -> tuple[IngestionJob, ...]:
        statement = select(IngestionJob).where(IngestionJob.status == "running")
        return tuple(self._session.scalars(statement))

    def next_attempt(self, document_id: str) -> int:
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
        document_id: str,
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


class VectorChunkRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def ensure_search_index_synced(self) -> None:
        chunk_rows = tuple(
            self._session.execute(select(DocumentChunk.id, DocumentChunk.embedding)).all()
        )
        if not chunk_rows:
            return

        dimension = len(chunk_rows[0].embedding)
        if not self._index_exists() or self._count_index_rows() != len(chunk_rows):
            self._drop_index()
            self._create_index(dimension)
            self._session.execute(
                text(
                    f"INSERT INTO {VECTOR_TABLE_NAME}(chunk_id, embedding) VALUES "
                    "(:chunk_id, :embedding)"
                ),
                [
                    {
                        "chunk_id": chunk_id,
                        "embedding": sqlite_vec.serialize_float32(list(embedding)),
                    }
                    for chunk_id, embedding in chunk_rows
                ],
            )

    def upsert_chunks(self, chunks: Sequence[DocumentChunk]) -> None:
        if not chunks:
            return
        if not self._index_exists():
            self._create_index(len(chunks[0].embedding))
        self._session.execute(
            text(
                f"INSERT OR REPLACE INTO {VECTOR_TABLE_NAME}(chunk_id, embedding) VALUES "
                "(:chunk_id, :embedding)"
            ),
            [
                {
                    "chunk_id": chunk.id,
                    "embedding": sqlite_vec.serialize_float32(list(chunk.embedding)),
                }
                for chunk in chunks
            ],
        )

    def delete_for_document(self, document_id: str) -> None:
        if not self._index_exists():
            return
        chunk_ids = tuple(
            self._session.scalars(
                select(DocumentChunk.id).where(DocumentChunk.document_id == document_id)
            )
        )
        if not chunk_ids:
            return
        self._delete_chunk_ids(chunk_ids)

    def search(
        self,
        *,
        query_embedding: Sequence[float],
        document_ids: tuple[str, ...],
        limit: int,
    ) -> tuple[RetrievedChunk, ...]:
        if not self._index_exists():
            return ()

        params: dict[str, object] = {
            "query_embedding": sqlite_vec.serialize_float32(list(query_embedding)),
            "search_k": self._count_index_rows(),
        }
        statement = text(
            f"""
            SELECT chunk_id, distance
            FROM {VECTOR_TABLE_NAME}
            WHERE {VECTOR_TABLE_NAME}.embedding MATCH :query_embedding AND k = :search_k
            ORDER BY distance
            """
        )
        distance_rows = tuple(self._session.execute(statement, params).mappings())
        if not distance_rows:
            return ()

        chunk_ids = [str(row["chunk_id"]) for row in distance_rows]
        metadata_rows = self._session.execute(
            select(DocumentChunk, Document.display_name)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.id.in_(chunk_ids))
        )
        metadata_by_chunk_id = {
            chunk.id: (chunk, document_name) for chunk, document_name in metadata_rows
        }
        results: list[RetrievedChunk] = []
        allowed_document_ids = set(document_ids)
        for row in distance_rows:
            chunk_id = str(row["chunk_id"])
            metadata = metadata_by_chunk_id.get(chunk_id)
            if metadata is None:
                continue
            chunk, document_name = metadata
            document = self._session.get(Document, chunk.document_id)
            if document is None or document.status != "ready":
                continue
            if allowed_document_ids and chunk.document_id not in allowed_document_ids:
                continue
            results.append(
                _to_retrieved_chunk(
                    chunk_id=chunk_id,
                    distance=float(row["distance"]),
                    chunk=chunk,
                    document_name=document_name,
                )
            )
            if len(results) >= limit:
                break
        return tuple(results)

    def _index_exists(self) -> bool:
        return (
            self._session.execute(
                text("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = :name LIMIT 1"),
                {"name": VECTOR_TABLE_NAME},
            ).scalar()
            is not None
        )

    def _count_index_rows(self) -> int:
        if not self._index_exists():
            return 0
        return (
            self._session.execute(text(f"SELECT COUNT(*) FROM {VECTOR_TABLE_NAME}")).scalar() or 0
        )

    def _create_index(self, dimension: int) -> None:
        self._session.execute(
            text(
                f"""
                CREATE VIRTUAL TABLE {VECTOR_TABLE_NAME}
                USING vec0(chunk_id TEXT PRIMARY KEY, embedding FLOAT[{dimension}])
                """
            )
        )

    def _drop_index(self) -> None:
        if not self._index_exists():
            return
        self._session.execute(text(f"DROP TABLE {VECTOR_TABLE_NAME}"))

    def _delete_chunk_ids(self, chunk_ids: Sequence[str]) -> None:
        if not chunk_ids:
            return
        placeholders = []
        params: dict[str, object] = {}
        for index, chunk_id in enumerate(chunk_ids):
            key = f"chunk_id_{index}"
            placeholders.append(f":{key}")
            params[key] = chunk_id
        self._session.execute(
            text(f"DELETE FROM {VECTOR_TABLE_NAME} WHERE chunk_id IN ({', '.join(placeholders)})"),
            params,
        )


def _to_retrieved_chunk(
    *,
    chunk_id: str,
    distance: float,
    chunk: DocumentChunk,
    document_name: str,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=chunk.document_id,
        document_name=document_name,
        text=chunk.text,
        retrieval_text=chunk.retrieval_text,
        page_numbers=tuple(chunk.page_numbers),
        headings=tuple(chunk.headings),
        warning_codes=tuple(chunk.warning_codes),
        distance=distance,
    )
