"""本地目录 / 上传文件 token 导入服务。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from app.services.token_dao import TokenDAO, get_token_dao
from app.utils.logger import logger


@dataclass(frozen=True)
class TokenImportSummary:
    source_dir: str
    scanned_files: int
    imported_count: int
    duplicate_count: int
    invalid_json_count: int
    missing_token_count: int
    invalid_token_count: int

    @property
    def failed_count(self) -> int:
        return (
            self.duplicate_count
            + self.invalid_json_count
            + self.missing_token_count
            + self.invalid_token_count
        )


def _load_token_payload(file_path: Path) -> dict:
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败: {exc}") from exc


def _parse_token_bytes(data: bytes, filename: str = "<upload>") -> dict:
    """将上传的 bytes 解析为 dict，失败时抛出 ValueError。"""
    try:
        return json.loads(data)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 解析失败 ({filename}): {exc}") from exc


async def import_tokens_from_directory(
    source_dir: str | Path,
    *,
    provider: str = "zai",
    validate: bool = True,
    dao: Optional[TokenDAO] = None,
) -> TokenImportSummary:
    """
    从本地目录导入 token。

    目录中的每个 JSON 文件应至少包含 `token` 字段。
    """
    source_path = Path(source_dir).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"导入目录不存在: {source_path}")
    if not source_path.is_dir():
        raise NotADirectoryError(f"导入路径不是目录: {source_path}")

    token_dao = dao or get_token_dao()
    token_files = sorted(source_path.rglob("*.json"))
    seen_tokens: set[str] = set()
    imported_count = 0
    duplicate_count = 0
    invalid_json_count = 0
    missing_token_count = 0
    invalid_token_count = 0

    for file_path in token_files:
        try:
            payload = _load_token_payload(file_path)
        except ValueError as exc:
            invalid_json_count += 1
            logger.warning(f"⚠️ 跳过无效 JSON 文件: {file_path} - {exc}")
            continue

        if not isinstance(payload, dict):
            invalid_json_count += 1
            logger.warning(f"⚠️ 跳过非对象 JSON 文件: {file_path}")
            continue

        token = str(payload.get("token") or "").strip()
        email = str(payload.get("email") or "").strip()
        if not token:
            missing_token_count += 1
            logger.warning(f"⚠️ 文件缺少 token 字段: {file_path}")
            continue

        if token in seen_tokens:
            duplicate_count += 1
            logger.debug("跳过本批次重复 Token: %s", file_path.name)
            continue
        seen_tokens.add(token)

        existing = await token_dao.get_token_by_value(provider, token)
        if existing is not None:
            duplicate_count += 1
            logger.debug(
                "Token 已存在，跳过导入: %s (%s)",
                file_path.name,
                email or "unknown",
            )
            continue

        token_id = await token_dao.add_token(
            provider=provider,
            token=token,
            token_type="user",
            validate=validate,
        )
        if token_id is None:
            invalid_token_count += 1
            logger.warning(f"⚠️ Token 导入失败: {file_path.name} ({email or 'unknown'})")
            continue

        imported_count += 1
        logger.debug("已导入 Token: %s (%s)", file_path.name, email or "unknown")

    summary = TokenImportSummary(
        source_dir=str(source_path),
        scanned_files=len(token_files),
        imported_count=imported_count,
        duplicate_count=duplicate_count,
        invalid_json_count=invalid_json_count,
        missing_token_count=missing_token_count,
        invalid_token_count=invalid_token_count,
    )
    logger.info(
        "Token 目录导入完成: "
        "scanned=%s, imported=%s, duplicate=%s, invalid_json=%s, "
        "missing_token=%s, invalid_token=%s",
        summary.scanned_files,
        summary.imported_count,
        summary.duplicate_count,
        summary.invalid_json_count,
        summary.missing_token_count,
        summary.invalid_token_count,
    )
    return summary


async def import_tokens_from_uploaded_files(
    files: List[Tuple[str, bytes]],
    *,
    provider: str = "zai",
    validate: bool = True,
    dao: Optional[TokenDAO] = None,
) -> TokenImportSummary:
    """
    从上传的 JSON 文件列表导入 token。

    每个元素为 (filename, file_bytes)，JSON 应至少包含 `token` 字段。
    与 import_tokens_from_directory 保持一致的去重 / 验证逻辑。
    """
    token_dao = dao or get_token_dao()
    seen_tokens: set[str] = set()
    imported_count = 0
    duplicate_count = 0
    invalid_json_count = 0
    missing_token_count = 0
    invalid_token_count = 0

    for filename, data in files:
        try:
            payload = _parse_token_bytes(data, filename)
        except ValueError as exc:
            invalid_json_count += 1
            logger.warning(f"⚠️ 跳过无效 JSON 文件: {filename} - {exc}")
            continue

        if not isinstance(payload, dict):
            invalid_json_count += 1
            logger.warning(f"⚠️ 跳过非对象 JSON 文件: {filename}")
            continue

        token = str(payload.get("token") or "").strip()
        email = str(payload.get("email") or "").strip()
        if not token:
            missing_token_count += 1
            logger.warning(f"⚠️ 文件缺少 token 字段: {filename}")
            continue

        if token in seen_tokens:
            duplicate_count += 1
            logger.debug("跳过本批次重复 Token: %s", filename)
            continue
        seen_tokens.add(token)

        existing = await token_dao.get_token_by_value(provider, token)
        if existing is not None:
            duplicate_count += 1
            logger.debug(
                "Token 已存在，跳过导入: %s (%s)",
                filename,
                email or "unknown",
            )
            continue

        token_id = await token_dao.add_token(
            provider=provider,
            token=token,
            token_type="user",
            validate=validate,
        )
        if token_id is None:
            invalid_token_count += 1
            logger.warning(f"⚠️ Token 导入失败: {filename} ({email or 'unknown'})")
            continue

        imported_count += 1
        logger.debug("已导入 Token: %s (%s)", filename, email or "unknown")

    summary = TokenImportSummary(
        source_dir="upload",
        scanned_files=len(files),
        imported_count=imported_count,
        duplicate_count=duplicate_count,
        invalid_json_count=invalid_json_count,
        missing_token_count=missing_token_count,
        invalid_token_count=invalid_token_count,
    )
    logger.info(
        "Token 文件上传导入完成: "
        "scanned=%s, imported=%s, duplicate=%s, invalid_json=%s, "
        "missing_token=%s, invalid_token=%s",
        summary.scanned_files,
        summary.imported_count,
        summary.duplicate_count,
        summary.invalid_json_count,
        summary.missing_token_count,
        summary.invalid_token_count,
    )
    return summary
