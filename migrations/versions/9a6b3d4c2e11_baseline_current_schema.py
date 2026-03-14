"""baseline current schema

Revision ID: 9a6b3d4c2e11
Revises:
Create Date: 2026-03-14 10:30:00.000000
This is the single retained migration baseline.

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9a6b3d4c2e11"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "config_items",
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint("key"),
    )
    op.create_table(
        "request_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("endpoint", sa.String(), server_default="", nullable=False),
        sa.Column("source", sa.String(), server_default="unknown", nullable=False),
        sa.Column("protocol", sa.String(), server_default="unknown", nullable=False),
        sa.Column("client_name", sa.String(), server_default="Unknown", nullable=False),
        sa.Column("auth_token", sa.Text(), nullable=True),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("status_code", sa.Integer(), server_default=sa.text("200"), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("duration", sa.Float(), server_default=sa.text("0.0"), nullable=False),
        sa.Column("first_token_time", sa.Float(), server_default=sa.text("0.0"), nullable=False),
        sa.Column("input_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("output_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("cache_creation_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("cache_read_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("total_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "tokens",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("token", sa.String(), nullable=False),
        sa.Column("token_type", sa.String(), nullable=False),
        sa.Column("priority", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("is_enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("last_chat_cleanup", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("provider", "token", name="uq_provider_token"),
    )
    op.create_table(
        "token_stats",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("token_id", sa.Integer(), nullable=False),
        sa.Column("total_requests", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("successful_requests", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("failed_requests", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("last_success_time", sa.DateTime(), nullable=True),
        sa.Column("last_failure_time", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["token_id"], ["tokens.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token_id"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("token_stats")
    op.drop_table("tokens")
    op.drop_table("request_logs")
    op.drop_table("config_items")
