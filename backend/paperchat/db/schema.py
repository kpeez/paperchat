"""Database schema for the local Postgres persistence contract."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import expression
from sqlalchemy.types import UserDefinedType

DOCUMENT_STATUS_LENGTH = 32
MESSAGE_ROLE_LENGTH = 16


class Base(DeclarativeBase):
    """Base metadata for backend database tables."""


class Vector(UserDefinedType[Any]):
    """Minimal SQLAlchemy type for pgvector-backed columns."""

    cache_ok = True

    def __init__(self, dimensions: int | None = None) -> None:
        self.dimensions = dimensions

    def get_col_spec(self, **_: Any) -> str:
        if self.dimensions is None:
            return "vector"
        return f"vector({self.dimensions})"


class TimestampMixin:
    """Shared timestamp columns for mutable tables."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class Document(TimestampMixin, Base):
    """Local document metadata and ingestion lifecycle state."""

    __tablename__ = "documents"

    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        server_default=expression.text("gen_random_uuid()"),
    )
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    original_filename: Mapped[str] = mapped_column(Text, nullable=False)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(DOCUMENT_STATUS_LENGTH),
        nullable=False,
        server_default=expression.text("'pending'"),
    )

    __table_args__ = (Index("ix_documents_status", "status"),)


class DocumentChunk(Base):
    """Chunk rows linked to a durable document id."""

    __tablename__ = "document_chunks"

    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        server_default=expression.text("gen_random_uuid()"),
    )
    document_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    page_numbers: Mapped[list[int]] = mapped_column(ARRAY(Integer()), nullable=False)
    headings: Mapped[list[str]] = mapped_column(ARRAY(Text()), nullable=False)
    embedding: Mapped[Any] = mapped_column(Vector(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
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


class Conversation(TimestampMixin, Base):
    """Chat conversation metadata."""

    __tablename__ = "conversations"

    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        server_default=expression.text("gen_random_uuid()"),
    )
    title: Mapped[str | None] = mapped_column(Text, nullable=True)


class Message(Base):
    """Persisted chat messages for a local conversation."""

    __tablename__ = "messages"

    id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        primary_key=True,
        server_default=expression.text("gen_random_uuid()"),
    )
    conversation_id: Mapped[UUID] = mapped_column(
        PostgresUUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    message_index: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(MESSAGE_ROLE_LENGTH), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
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
        JSONB,
        nullable=False,
        server_default=expression.text("'{}'::jsonb"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
