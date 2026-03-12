#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""文件上传模块。

将原 UpstreamClient.upload_file() 提取为独立函数，
解耦对 UpstreamClient 实例的依赖，便于独立测试和复用。
所有原参数保留，增加 client 和 base_url 两个注入参数。
"""

import base64
import time
import uuid
from typing import Any, Dict, Optional
from urllib.parse import quote

import httpx

from app.utils.logger import get_logger
from app.core.config import settings

logger = get_logger()


async def upload_file(
    client: httpx.AsyncClient,
    base_url: str,
    data_url: str,
    chat_id: str,
    token: str,
    user_id: str,
    auth_mode: str = "authenticated",
    message_id: str = "",
) -> Optional[Dict[str, Any]]:
    """上传文件（图片/文档）到上游服务器。

    Args:
        client: 可复用的 httpx.AsyncClient 实例。
        base_url: 上游服务基础 URL（如 "https://chat.z.ai"）。
        data_url: ``data:mime/type;base64,...`` 格式的文件数据。
        chat_id: 当前对话 ID。
        token: 认证令牌。
        user_id: 用户 ID。
        auth_mode: 当前鉴权模式，``guest`` 模式下禁止上传。
        message_id: 关联的用户消息 ID，用于文件引用溯源。

    Returns:
        上传成功返回完整的文件信息字典，失败或 guest 模式返回 None。
    """
    if auth_mode == "guest" or not data_url.startswith("data:"):
        return None

    try:
        # 解析 data URL
        header, encoded = data_url.split(",", 1)

        # 解码前先预估文件大小，避免大文件 OOM
        if settings.MAX_UPLOAD_FILE_SIZE > 0:
            # base64 编码后大小约为原始大小的 4/3，反推原始大小估算
            estimated_size = len(encoded) * 3 // 4
            if estimated_size > settings.MAX_UPLOAD_FILE_SIZE:
                limit_kb = settings.MAX_UPLOAD_FILE_SIZE // 1024
                logger.warning(
                    f"⚠️ 文件过大，预估 {estimated_size // 1024}KB 超过限制 {limit_kb}KB，已跳过上传"
                )
                return None

        mime_type = (
            header.split(";")[0].split(":")[1]
            if ":" in header
            else "application/octet-stream"
        )
        file_data = base64.b64decode(encoded)

        # 判断文件类型
        is_image = mime_type.startswith("image/")
        media_type = "image" if is_image else "file"

        # 生成文件名
        ext_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "application/pdf": ".pdf",
            "text/plain": ".txt",
            "text/markdown": ".md",
            "application/octet-stream": "",
        }
        ext = ext_map.get(mime_type, "")
        filename = f"{uuid.uuid4()}{ext}"

        logger.debug(
            f"📤 上传{media_type}: {filename}, 大小: {len(file_data)} bytes"
        )

        # 构建上传请求头（保留所有原 header 字段）
        upload_url = f"{base_url}/api/v1/files/"
        headers = {
            "Accept": "application/json",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
            "Connection": "keep-alive",
            "Origin": base_url,
            "Referer": f"{base_url}/",
            "Sec-Ch-Ua": '"Not:A-Brand";v="99", "Microsoft Edge";v="145", "Chromium";v="145"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Authorization": f"Bearer {token}",
        }

        upload_files = {"file": (filename, file_data, mime_type)}
        response = await client.post(upload_url, files=upload_files, headers=headers)

        if response.status_code != 200:
            logger.error(
                f"❌ 文件上传失败: {response.status_code} - {response.text}"
            )
            return None

        result = response.json()
        file_id = result.get("id")
        file_name = result.get("filename", filename)
        file_meta = result.get("meta", {})

        logger.info(f"[file] upload success: {file_id} ({file_name})")

        # 使用上游返回的完整 meta（包含 oss_endpoint, cdn_url 等）
        return {
            "type": media_type,
            "file": {
                "id": file_id,
                "user_id": result.get("user_id", user_id),
                "hash": result.get("hash"),
                "filename": file_name,
                "data": result.get("data", {}),
                "meta": file_meta,
                "created_at": result.get("created_at", int(time.time())),
                "updated_at": result.get("updated_at", int(time.time())),
            },
            "id": file_id,
            "url": f"/api/v1/files/{file_id}",
            "name": quote(file_name),
            "status": "uploaded",
            "size": file_meta.get("size", len(file_data)),
            "error": "",
            "itemId": str(uuid.uuid4()),
            "media": media_type,
            "ref_user_msg_id": message_id,
        }

    except Exception as e:
        logger.error(f"❌ 文件上传异常: {e}")
        return None
