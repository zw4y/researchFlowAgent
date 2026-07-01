"""Add production embedding index lifecycle fields."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_embedding_index_profile"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "papers",
        sa.Column("index_status", sa.String(32), nullable=False, server_default="pending"),
    )
    op.add_column("papers", sa.Column("index_profile", sa.String(64), nullable=True))
    op.add_column("papers", sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_papers_index_status", "papers", ["index_status"])
    op.create_index("ix_papers_index_profile", "papers", ["index_profile"])
    op.execute(
        "UPDATE papers SET index_status = CASE "
        "WHEN status = 'ready' THEN 'stale' "
        "WHEN status = 'failed' THEN 'failed' ELSE 'pending' END"
    )

    op.add_column(
        "chunks",
        sa.Column(
            "index_profile",
            sa.String(64),
            nullable=False,
            server_default="legacy-unindexed",
        ),
    )
    op.create_index("ix_chunks_index_profile", "chunks", ["index_profile"])

    op.add_column(
        "ingestion_jobs",
        sa.Column("job_type", sa.String(32), nullable=False, server_default="ingest"),
    )
    op.add_column(
        "ingestion_jobs",
        sa.Column("details", sa.JSON(), nullable=False, server_default="{}"),
    )


def downgrade() -> None:
    op.drop_column("ingestion_jobs", "details")
    op.drop_column("ingestion_jobs", "job_type")
    op.drop_index("ix_chunks_index_profile", table_name="chunks")
    op.drop_column("chunks", "index_profile")
    op.drop_index("ix_papers_index_profile", table_name="papers")
    op.drop_index("ix_papers_index_status", table_name="papers")
    op.drop_column("papers", "indexed_at")
    op.drop_column("papers", "index_profile")
    op.drop_column("papers", "index_status")