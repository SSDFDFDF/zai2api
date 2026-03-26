"""add upstream_auth_token to request_logs

Revision ID: b4f7c2a19d21
Revises: 9a6b3d4c2e11
Create Date: 2026-03-14 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "b4f7c2a19d21"
down_revision = "9a6b3d4c2e11"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "request_logs",
        sa.Column("upstream_auth_token", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("request_logs", "upstream_auth_token")
