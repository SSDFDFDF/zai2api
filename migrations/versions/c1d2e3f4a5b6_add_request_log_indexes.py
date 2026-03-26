"""add request log indexes

Revision ID: c1d2e3f4a5b6
Revises: b4f7c2a19d21
Create Date: 2026-03-26 10:40:00.000000
"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "c1d2e3f4a5b6"
down_revision = "b4f7c2a19d21"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_request_logs_timestamp",
        "request_logs",
        ["timestamp"],
        unique=False,
    )
    op.create_index(
        "ix_request_logs_provider_timestamp",
        "request_logs",
        ["provider", "timestamp"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_request_logs_provider_timestamp", table_name="request_logs")
    op.drop_index("ix_request_logs_timestamp", table_name="request_logs")
