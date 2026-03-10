"""Expand the document registry for ingestion tracking."""

import sqlalchemy as sa

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
    with op.batch_alter_table("documents") as batch_op:
        batch_op.add_column(
            sa.Column("parser_id", sa.String(length=PARSER_ID_LENGTH), nullable=True)
        )
        batch_op.add_column(
            sa.Column("chunker_id", sa.String(length=CHUNKER_ID_LENGTH), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "embedding_model_id",
                sa.String(length=EMBEDDING_MODEL_ID_LENGTH),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0")
        )
        batch_op.add_column(
            sa.Column(
                "error_code",
                sa.String(length=INGESTION_ERROR_CODE_LENGTH),
                nullable=True,
            )
        )
        batch_op.add_column(sa.Column("error_message", sa.Text(), nullable=True))
        batch_op.create_check_constraint(
            "ck_documents_content_hash_sha256",
            "length(content_hash) = 64 AND content_hash NOT GLOB '*[^0-9a-f]*'",
        )
        batch_op.create_check_constraint(
            "ck_documents_status_valid",
            "status IN ('pending', 'processing', 'ready', 'failed')",
        )

    with op.batch_alter_table("document_chunks") as batch_op:
        batch_op.add_column(
            sa.Column("retrieval_text", sa.Text(), nullable=False, server_default="")
        )
        batch_op.add_column(
            sa.Column("warning_codes", sa.JSON(), nullable=False, server_default="[]")
        )

    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("document_id", sa.String(length=36), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=INGESTION_JOB_STATUS_LENGTH),
            nullable=False,
            server_default="queued",
        ),
        sa.Column(
            "stage",
            sa.String(length=INGESTION_STAGE_LENGTH),
            nullable=False,
            server_default="queued",
        ),
        sa.Column(
            "error_code",
            sa.String(length=INGESTION_ERROR_CODE_LENGTH),
            nullable=True,
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
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

    with op.batch_alter_table("document_chunks") as batch_op:
        batch_op.drop_column("warning_codes")
        batch_op.drop_column("retrieval_text")

    with op.batch_alter_table("documents") as batch_op:
        batch_op.drop_constraint("ck_documents_status_valid", type_="check")
        batch_op.drop_constraint("ck_documents_content_hash_sha256", type_="check")
        batch_op.drop_column("error_message")
        batch_op.drop_column("error_code")
        batch_op.drop_column("chunk_count")
        batch_op.drop_column("embedding_model_id")
        batch_op.drop_column("chunker_id")
        batch_op.drop_column("parser_id")
