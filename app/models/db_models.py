from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, mapped_column, relationship
from sqlalchemy.sql import func, text


class Base(DeclarativeBase):
    pass


class Token(Base):
    __tablename__ = "tokens"

    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider = mapped_column(String, nullable=False)
    token = mapped_column(String, nullable=False)
    token_type = mapped_column(String, default="user", nullable=False)
    priority = mapped_column(Integer, default=0, server_default=text("0"), nullable=False)
    is_enabled = mapped_column(Boolean, default=True, server_default=text("true"), nullable=False)
    created_at = mapped_column(DateTime, default=func.now(), server_default=func.now(), nullable=False)
    last_chat_cleanup = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("provider", "token", name="uq_provider_token"),
    )

    stats = relationship("TokenStats", back_populates="token_rel", uselist=False, cascade="all, delete-orphan")


class TokenStats(Base):
    __tablename__ = "token_stats"

    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    token_id = mapped_column(
        Integer, ForeignKey("tokens.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    total_requests = mapped_column(Integer, default=0, server_default=text("0"), nullable=False)
    successful_requests = mapped_column(Integer, default=0, server_default=text("0"), nullable=False)
    failed_requests = mapped_column(Integer, default=0, server_default=text("0"), nullable=False)
    last_success_time = mapped_column(DateTime, nullable=True)
    last_failure_time = mapped_column(DateTime, nullable=True)

    token_rel = relationship("Token", back_populates="stats")


class RequestLog(Base):
    __tablename__ = "request_logs"

    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider = mapped_column(String, nullable=False)
    endpoint = mapped_column(String, nullable=False, default="", server_default=text("''"))
    source = mapped_column(String, nullable=False, default="unknown", server_default=text("'unknown'"))
    protocol = mapped_column(String, nullable=False, default="unknown", server_default=text("'unknown'"))
    client_name = mapped_column(String, nullable=False, default="Unknown", server_default=text("'Unknown'"))
    auth_token = mapped_column(Text, nullable=True)
    model = mapped_column(String, nullable=False)
    status_code = mapped_column(Integer, nullable=False, default=200, server_default=text("200"))
    success = mapped_column(Boolean, nullable=False)
    duration = mapped_column(Float, nullable=False, default=0.0, server_default=text("0.0"))
    first_token_time = mapped_column(Float, nullable=False, default=0.0, server_default=text("0.0"))
    input_tokens = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    output_tokens = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    cache_creation_tokens = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    cache_read_tokens = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    total_tokens = mapped_column(Integer, nullable=False, default=0, server_default=text("0"))
    error_message = mapped_column(Text, nullable=True)
    timestamp = mapped_column(DateTime, default=func.now(), server_default=func.now(), nullable=False)


class ConfigItem(Base):
    __tablename__ = "config_items"

    key = mapped_column(String, primary_key=True)
    value = mapped_column(Text, nullable=False)
