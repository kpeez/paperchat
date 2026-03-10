"""Database schema for the local SQLite persistence contract."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON

DOCUMENT_STATUS_LENGTH = 32
INGESTION_JOB_STATUS_LENGTH = 32
INGESTION_STAGE_LENGTH = 32
INGESTION_ERROR_CODE_LENGTH = 64
MESSAGE_ROLE_LENGTH = 16
PARSER_ID_LENGTH = 64
CHUNKER_ID_LENGTH = 64
EMBEDDING_MODEL_ID_LENGTH = 128

DOCUMENT_STATUSES = ("pending", "processing", "ready", "failed")
INGESTION_JOB_STATUSES = ("queued", "running", "succeeded", "failed")


def _new_uuid() -> str:
    return str(uuid4())


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Base metadata for backend database tables."""


class TimestampMixin:
    """Shared timestamp columns for mutable tables."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
        onupdate=_utcnow,
    )


class Document(TimestampMixin, Base):
    """Local document metadata and ingestion lifecycle state."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=_new_uuid,
    )
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    original_filename: Mapped[str] = mapped_column(Text, nullable=False)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(DOCUMENT_STATUS_LENGTH),
        nullable=False,
        default="pending",
    )
    parser_id: Mapped[str | None] = mapped_column(String(PARSER_ID_LENGTH), nullable=True)
    chunker_id: Mapped[str | None] = mapped_column(String(CHUNKER_ID_LENGTH), nullable=True)
    embedding_model_id: Mapped[str | None] = mapped_column(
        String(EMBEDDING_MODEL_ID_LENGTH),
        nullable=True,
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    error_code: Mapped[str | None] = mapped_column(
        String(INGESTION_ERROR_CODE_LENGTH),
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_documents_status", "status"),
        CheckConstraint(
            "length(content_hash) = 64 AND content_hash NOT GLOB '*[^0-9a-f]*'",
            name="ck_documents_content_hash_sha256",
        ),
        CheckConstraint(
            f"status IN {DOCUMENT_STATUSES}",
            name="ck_documents_status_valid",
        ),
    )


class DocumentChunk(Base):
    """Chunk rows linked to a durable document id."""

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=_new_uuid,
    )
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_text: Mapped[str] = mapped_column(Text, nullable=False)
    page_numbers: Mapped[list[int]] = mapped_column(JSON, nullable=False)
    headings: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    warning_codes: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_document_chunks_document_id_chunk_index",
        ),
        Index("ix_document_chunks_document_id", "document_id"),
    )


class IngestionJob(TimestampMixin, Base):
    """Document ingestion attempt tracking."""

    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=_new_uuid,
    )
    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(INGESTION_JOB_STATUS_LENGTH),
        nullable=False,
        default="queued",
    )
    stage: Mapped[str] = mapped_column(
        String(INGESTION_STAGE_LENGTH),
        nullable=False,
        default="queued",
    )
    error_code: Mapped[str | None] = mapped_column(
        String(INGESTION_ERROR_CODE_LENGTH),
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "attempt",
            name="uq_ingestion_jobs_document_id_attempt",
        ),
        Index("ix_ingestion_jobs_document_id", "document_id"),
        Index("ix_ingestion_jobs_status", "status"),
        CheckConstraint(
            f"status IN {INGESTION_JOB_STATUSES}",
            name="ck_ingestion_jobs_status_valid",
        ),
    )

    @property
    def attempt_number(self) -> int:
        return self.attempt


class Conversation(TimestampMixin, Base):
    """Chat conversation metadata."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=_new_uuid,
    )
    title: Mapped[str | None] = mapped_column(Text, nullable=True)


class Message(Base):
    """Persisted chat messages for a local conversation."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=_new_uuid,
    )
    conversation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    message_index: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(MESSAGE_ROLE_LENGTH), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "message_index",
            name="uq_messages_conversation_id_message_index",
        ),
        Index("ix_messages_conversation_id", "conversation_id"),
    )


class AppState(Base):
    """Small key-value store for runtime and schema markers."""

    __tablename__ = "app_state"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        server_default="{}",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=_utcnow,
        server_default=func.now(),
    )
