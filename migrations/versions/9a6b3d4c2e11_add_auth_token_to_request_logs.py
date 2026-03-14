"""add auth_token to request_logs

Revision ID: 9a6b3d4c2e11
Revises: 82a1fdd3f5f3
Create Date: 2026-03-13 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9a6b3d4c2e11"
down_revision: Union[str, Sequence[str], None] = "82a1fdd3f5f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("request_logs", sa.Column("auth_token", sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("request_logs", "auth_token")
