"""Expand the document registry for PR 4 ingestion."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op
from paperchat.db.schema import (
    CHUNKER_ID_LENGTH,
    EMBEDDING_MODEL_ID_LENGTH,
    INGESTION_ERROR_CODE_LENGTH,
    INGESTION_JOB_STATUS_LENGTH,
    INGESTION_STAGE_LENGTH,
    PARSER_ID_LENGTH,
)

revision = "0002_ingestion_registry"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("parser_id", sa.String(length=PARSER_ID_LENGTH), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("chunker_id", sa.String(length=CHUNKER_ID_LENGTH), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column(
            "embedding_model_id",
            sa.String(length=EMBEDDING_MODEL_ID_LENGTH),
            nullable=True,
        ),
    )
    op.add_column(
        "documents",
        sa.Column(
            "chunk_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.add_column(
        "documents",
        sa.Column(
            "error_code",
            sa.String(length=INGESTION_ERROR_CODE_LENGTH),
            nullable=True,
        ),
    )
    op.add_column(
        "documents",
        sa.Column("error_message", sa.Text(), nullable=True),
    )
    op.create_check_constraint(
        "ck_documents_content_hash_sha256",
        "documents",
        "content_hash ~ '^[0-9a-f]{64}$'",
    )
    op.create_check_constraint(
        "ck_documents_status_valid",
        "documents",
        "status IN ('pending', 'processing', 'ready', 'failed')",
    )

    op.add_column(
        "document_chunks",
        sa.Column("retrieval_text", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "document_chunks",
        sa.Column(
            "warning_codes",
            postgresql.ARRAY(sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
    )
    op.alter_column("document_chunks", "retrieval_text", server_default=None)
    op.alter_column("document_chunks", "warning_codes", server_default=None)

    op.create_table(
        "ingestion_jobs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=INGESTION_JOB_STATUS_LENGTH),
            nullable=False,
            server_default=sa.text("'queued'"),
        ),
        sa.Column(
            "stage",
            sa.String(length=INGESTION_STAGE_LENGTH),
            nullable=False,
            server_default=sa.text("'queued'"),
        ),
        sa.Column(
            "error_code",
            sa.String(length=INGESTION_ERROR_CODE_LENGTH),
            nullable=True,
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id",
            "attempt",
            name="uq_ingestion_jobs_document_id_attempt",
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed')",
            name="ck_ingestion_jobs_status_valid",
        ),
    )
    op.create_index(
        "ix_ingestion_jobs_document_id", "ingestion_jobs", ["document_id"], unique=False
    )
    op.create_index("ix_ingestion_jobs_status", "ingestion_jobs", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_ingestion_jobs_status", table_name="ingestion_jobs")
    op.drop_index("ix_ingestion_jobs_document_id", table_name="ingestion_jobs")
    op.drop_table("ingestion_jobs")

    op.drop_column("document_chunks", "warning_codes")
    op.drop_column("document_chunks", "retrieval_text")

    op.drop_constraint("ck_documents_status_valid", "documents", type_="check")
    op.drop_constraint("ck_documents_content_hash_sha256", "documents", type_="check")
    op.drop_column("documents", "error_message")
    op.drop_column("documents", "error_code")
    op.drop_column("documents", "chunk_count")
    op.drop_column("documents", "embedding_model_id")
    op.drop_column("documents", "chunker_id")
    op.drop_column("documents", "parser_id")
