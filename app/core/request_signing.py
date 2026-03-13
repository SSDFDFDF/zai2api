#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""请求构建与双层 HMAC 签名模块。

将原 UpstreamClient.transform_request() 中的请求体构建、多模态消息处理
和签名逻辑提取为独立函数。
"""

import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx

from app.core.config import settings
from app.core.headers import build_dynamic_headers
from app.core.file_upload import upload_file
from app.utils.fe_version import get_latest_fe_version
from app.utils.logger import get_logger
from app.utils.signature import generate_signature

logger = get_logger()


async def process_multimodal_messages(
    normalized_messages: List[Dict[str, Any]],
    token: str,
    user_id: str,
    chat_id: str,
    auth_mode: str,
    http_client: httpx.AsyncClient,
    base_url: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """处理多模态消息：分离文本与图片，并上传 base64 图片。

    Args:
        normalized_messages: 经过预处理的消息列表。
        token: 认证令牌。
        user_id: 用户 ID。
        chat_id: 对话 ID。
        auth_mode: 鉴权模式，guest 模式禁止上传。
        http_client: 可复用 HTTP 客户端。
        base_url: 上游服务基础 URL。

    Returns:
        ``(messages, files)`` 元组：
        - ``messages``: 上游格式的消息列表。
        - ``files``: 已上传文件信息列表（仅认证模式且有图片时非空）。
    """
    messages: List[Dict[str, Any]] = []
    files: List[Dict[str, Any]] = []

    for msg in normalized_messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content")

        if isinstance(content, str):
            # 纯文本消息
            messages.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            # 多模态内容：分离文本和图片
            text_parts: List[str] = []
            image_parts: List[Dict[str, Any]] = []

            for part in content:
                if hasattr(part, "type"):
                    if part.type == "text" and hasattr(part, "text"):
                        text_parts.append(part.text or "")
                    elif part.type == "image_url" and hasattr(part, "image_url"):
                        image_url = None
                        if hasattr(part.image_url, "url"):
                            image_url = part.image_url.url
                        elif isinstance(part.image_url, dict) and "url" in part.image_url:
                            image_url = part.image_url["url"]

                        if image_url:
                            logger.debug(f"✅ 检测到图片: {image_url[:50]}...")
                            if image_url.startswith("data:") and auth_mode != "guest":
                                logger.info("[image] upload base64 image to upstream service")
                                file_info = await upload_file(
                                    http_client, base_url, image_url,
                                    chat_id, token, user_id, auth_mode=auth_mode,
                                )
                                if file_info:
                                    files.append(file_info)
                                    logger.info("[image] image added to files array")
                                    image_ref = f"{file_info['id']}_{file_info['name']}"
                                    image_parts.append({
                                        "type": "image_url",
                                        "image_url": {"url": image_ref},
                                    })
                                    logger.debug(f"📎 图片引用: {image_ref}")
                                else:
                                    logger.warning("⚠️ 图片上传失败")
                                    text_parts.append("[系统提示: 图片上传失败]")
                            else:
                                if auth_mode != "guest":
                                    logger.warning("⚠️ 非 base64 图片或匿名模式，保留原始URL")
                                image_parts.append({
                                    "type": "image_url",
                                    "image_url": {"url": image_url},
                                })

                elif isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        image_url = part.get("image_url", {}).get("url", "")
                        if image_url:
                            logger.debug(f"✅ 检测到图片: {image_url[:50]}...")
                            if image_url.startswith("data:") and auth_mode != "guest":
                                logger.info("[image] upload base64 image to upstream service")
                                file_info = await upload_file(
                                    http_client, base_url, image_url,
                                    chat_id, token, user_id, auth_mode=auth_mode,
                                )
                                if file_info:
                                    files.append(file_info)
                                    logger.info("[image] image added to files array")
                                    image_ref = f"{file_info['id']}_{file_info['name']}"
                                    image_parts.append({
                                        "type": "image_url",
                                        "image_url": {"url": image_ref},
                                    })
                                    logger.debug(f"📎 图片引用: {image_ref}")
                                else:
                                    logger.warning("⚠️ 图片上传失败")
                                    text_parts.append("[系统提示: 图片上传失败]")
                            else:
                                if auth_mode != "guest":
                                    logger.warning("⚠️ 非 base64 图片或匿名模式，保留原始URL")
                                image_parts.append({
                                    "type": "image_url",
                                    "image_url": {"url": image_url},
                                })

                elif isinstance(part, str):
                    text_parts.append(part)

            # 构建多模态消息内容
            message_content: List[Dict[str, Any]] = []
            combined_text = " ".join(text_parts).strip()
            if combined_text:
                message_content.append({"type": "text", "text": combined_text})
            message_content.extend(image_parts)

            if message_content:
                messages.append({"role": role, "content": message_content})

    return messages, files


def build_upstream_body(
    messages: List[Dict[str, Any]],
    files: List[Dict[str, Any]],
    upstream_model_id: str,
    last_user_text: str,
    chat_id: str,
    message_id: str,
    enable_thinking: bool,
    web_search: bool,
    auto_web_search: bool,
    flags: List[str],
    extra: Dict[str, Any],
    mcp_servers: List[str],
    temperature: Optional[float],
    max_tokens: Optional[int],
    parent_message_id: Optional[str] = None,
) -> Dict[str, Any]:
    """构建发送到上游的请求 JSON body。

    采用 Toolify XML 方案后，工具定义通过 system prompt 注入，
    不再通过 body 透传 tools/tool_choice 字段。

    Args:
        messages: 上游格式消息列表。
        files: 已上传文件信息列表。
        upstream_model_id: 上游实际模型 ID。
        last_user_text: 最后一条用户消息（用于签名占位符）。
        chat_id: 对话 ID。
        message_id: 消息 ID（当前消息的 UUID）。
        enable_thinking: 是否启用思考模式。
        web_search: 是否启用网络搜索。
        auto_web_search: 是否启用自动网络搜索。
        flags: 功能标志列表（如 ["general_agent"]）。
        extra: 额外参数字典。
        mcp_servers: MCP 服务器 ID 列表。
        temperature: 采样温度，None 时不添加到 params。
        max_tokens: 最大 token 数，None 时不添加到 params。

    Returns:
        完整的上游请求 body 字典。
    """
    body: Dict[str, Any] = {
        "stream": True,  # 总是使用流式
        "model": upstream_model_id,
        "messages": messages,
        "signature_prompt": last_user_text,  # 用于签名的最后一条用户消息
        "params": {},
        "extra": extra,
        "features": {
            "image_generation": False,
            "web_search": web_search,
            "auto_web_search": auto_web_search,
            "preview_mode": True,
            "flags": flags,
            "enable_thinking": enable_thinking,
        },
        "background_tasks": {
            "title_generation": settings.UPSTREAM_BACKGROUND_TASKS,
            "tags_generation": settings.UPSTREAM_BACKGROUND_TASKS,
        },
        "variables": {
            "{{USER_NAME}}": settings.UPSTREAM_USER_NAME,
            "{{USER_LOCATION}}": settings.UPSTREAM_USER_LOCATION,
            "{{CURRENT_DATETIME}}": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "{{CURRENT_DATE}}": datetime.now().strftime("%Y-%m-%d"),
            "{{CURRENT_TIME}}": datetime.now().strftime("%H:%M:%S"),
            "{{CURRENT_WEEKDAY}}": datetime.now().strftime("%A"),
            "{{CURRENT_TIMEZONE}}": settings.UPSTREAM_USER_TIMEZONE,
            "{{USER_LANGUAGE}}": settings.UPSTREAM_USER_LANGUAGE,
        },
        "chat_id": chat_id,
        "id": message_id,
        "current_user_message_id": str(uuid.uuid4()),
        "current_user_message_parent_id": parent_message_id,
    }

    # 只在有图片时才包含 files 字段
    if files:
        body["files"] = files

    # 只在有 MCP 服务器时才包含
    if mcp_servers:
        body["mcp_servers"] = mcp_servers

    # NOTE: tools/tool_choice 不再透传到 body。
    # Toolify XML 方案通过 system prompt 注入工具定义，
    # 避免与上游原生工具处理冲突。

    # 处理其他参数
    # if temperature is not None:
    #     body["params"]["temperature"] = temperature
    # if max_tokens is not None:
    #     body["params"]["max_tokens"] = max_tokens

    return body


async def sign_request(
    api_endpoint: str,
    user_id: str,
    last_user_text: str,
    chat_id: str,
    token: str,
    fe_version: Optional[str] = None,
) -> tuple[str, Dict[str, str], str]:
    """生成双层 HMAC 签名并构造带签名的 URL 和请求头。

    Args:
        api_endpoint: 上游 API 端点 URL（不含 query 参数）。
        user_id: 用户 ID，用于签名元数据。
        last_user_text: 最后一条用户消息文本，用于签名载荷。
        chat_id: 对话 ID，用于构建 Referer 和 URL 路径。
        token: 认证令牌，添加到请求头 Authorization。
        fe_version: 可选前端版本号，传入时跳过网络拉取（用于并行预取）。

    Returns:
        ``(signed_url, headers, fe_version)`` 三元组：
        - ``signed_url``: 带完整 query 参数的签名 URL。
        - ``headers``: 包含 Authorization、X-Signature 等所有请求头的字典。
        - ``fe_version``: 使用的前端版本号。
    """
    timestamp_ms = int(time.time() * 1000)
    request_id = str(uuid.uuid4())
    if fe_version is None:
        fe_version = await get_latest_fe_version()

    try:
        signing_metadata = f"requestId,{request_id},timestamp,{timestamp_ms},user_id,{user_id}"
        prompt_for_signature = last_user_text or ""
        signature_result = generate_signature(
            e=signing_metadata,
            t=prompt_for_signature,
            s=timestamp_ms,
        )
        signature = signature_result["signature"]
        logger.debug(
            f"[上游] 生成签名成功: {signature[:16]}... "
            f"(user_id={user_id}, request_id={request_id})"
        )
    except Exception as e:
        logger.error(f"[上游] 签名生成失败: {e}")
        raise RuntimeError(f"签名生成失败: {e}") from e

    # 构建请求头（保留所有字段）
    headers = build_dynamic_headers(fe_version, chat_id)
    user_agent = headers.get("User-Agent", "")

    # 从 User-Agent 推断浏览器和操作系统名
    if "Edg/" in user_agent:
        browser_name = "Edge"
    elif "Firefox/" in user_agent:
        browser_name = "Firefox"
    elif "Safari/" in user_agent and "Chrome/" not in user_agent:
        browser_name = "Safari"
    else:
        browser_name = "Chrome"

    if "Windows" in user_agent:
        os_name = "Windows"
    elif "Macintosh" in user_agent or "Mac OS" in user_agent:
        os_name = "macOS"
    elif "Linux" in user_agent:
        os_name = "Linux"
    else:
        os_name = "Windows"

    headers.update(
        {
            "Authorization": f"Bearer {token}",
            "X-Signature": signature,
        }
    )

    # 构建完整的 URL query params（对齐浏览器指纹字段）
    now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
    now_utc = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")

    query_params = {
        "timestamp": str(timestamp_ms),
        "requestId": request_id,
        "user_id": user_id,
        "version": "0.0.1",
        "platform": "web",
        "token": token,
        "user_agent": user_agent,
        "language": "zh-CN",
        "languages": "zh-CN,en,en-GB,en-US",
        "timezone": "Asia/Shanghai",
        "cookie_enabled": "true",
        "screen_width": "2560",
        "screen_height": "1440",
        "screen_resolution": "2560x1440",
        "viewport_height": "1274",
        "viewport_width": "1574",
        "viewport_size": "1574x1274",
        "color_depth": "32",
        "pixel_ratio": "1",
        "current_url": f"https://chat.z.ai/c/{chat_id}",
        "pathname": f"/c/{chat_id}",
        "search": "",
        "hash": "",
        "host": "chat.z.ai",
        "hostname": "chat.z.ai",
        "protocol": "https:",
        "referrer": "",
        "title": "Z.ai - Free AI Chatbot & Agent powered by GLM-5 & GLM-4.7",
        "timezone_offset": "-480",
        "local_time": now_iso,
        "utc_time": now_utc,
        "is_mobile": "false",
        "is_touch": "false",
        "max_touch_points": "0",
        "browser_name": browser_name,
        "os_name": os_name,
        "signature_timestamp": str(timestamp_ms),
    }
    signed_url = f"{api_endpoint}?{urlencode(query_params)}"

    logger.debug(
        f"[上游] 请求头: Authorization=Bearer {token}, "
        f"X-Signature={signature[:16] if signature else '(空)'}..."
    )
    logger.debug(
        f"[上游] URL 参数: timestamp={timestamp_ms}, "
        f"requestId={request_id}, user_id={user_id}"
    )

    return signed_url, headers, fe_version
