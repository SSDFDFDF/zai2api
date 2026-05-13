#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""请求构建与双层 HMAC 签名模块。

将原 UpstreamClient.transform_request() 中的请求体构建、多模态消息处理
和签名逻辑提取为独立函数。
"""

import asyncio
import time
import uuid
from datetime import datetime
import mimetypes
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import httpx

from app.core.config import settings
from app.core.headers import build_dynamic_headers
from app.core.file_upload import upload_file
from app.utils.fe_version import get_latest_fe_version
from app.utils.logger import logger, log_exception
from app.utils.signature import generate_signature
from app.core.resin_compat import apply_resin_account_header


def _extract_image_url(part: Dict[str, Any]) -> str:
    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        return str(image_url.get("url") or "")
    if isinstance(image_url, str):
        return image_url
    return ""


def _extract_file_payload(part: Dict[str, Any]) -> Dict[str, Any]:
    payload = part.get("file")
    if isinstance(payload, dict):
        return {
            key: value
            for key, value in payload.items()
            if value is not None
        }

    return {
        key: value
        for key in ("file_id", "file_url", "file_data", "filename")
        for value in [part.get(key)]
        if value is not None
    }


def _build_file_data_url(file_payload: Dict[str, Any]) -> Optional[str]:
    file_data = file_payload.get("file_data")
    if isinstance(file_data, str) and file_data.strip():
        if file_data.startswith("data:"):
            return file_data
        filename = str(file_payload.get("filename") or "attachment").strip()
        mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        return f"data:{mime_type};base64,{file_data}"

    file_url = file_payload.get("file_url")
    if isinstance(file_url, str) and file_url.startswith("data:"):
        return file_url

    return None


async def _append_image_part(
    *,
    image_url: str,
    token: str,
    user_id: str,
    chat_id: str,
    auth_mode: str,
    http_client: httpx.AsyncClient,
    base_url: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    """Upload an image and return (image_part, file_info, error_text)."""
    if not image_url:
        return None, None, None

    logger.debug("检测到图片: %s...", image_url[:50])
    if image_url.startswith("data:") and auth_mode != "guest":
        logger.debug("[image] upload base64 image to upstream service")
        file_info = await upload_file(
            http_client,
            base_url,
            image_url,
            chat_id,
            token,
            user_id,
            auth_mode=auth_mode,
        )
        if file_info:
            logger.debug("[image] image added to files array")
            image_ref = f"{file_info['id']}_{file_info['name']}"
            image_part = {
                "type": "image_url",
                "image_url": {"url": image_ref},
            }
            logger.debug("图片引用: %s", image_ref)
            return image_part, file_info, None

        logger.warning("⚠️ 图片上传失败")
        return None, None, "[系统提示: 图片上传失败]"

    if auth_mode != "guest":
        logger.warning("⚠️ 非 base64 图片或匿名模式，保留原始URL")
    image_part = {
        "type": "image_url",
        "image_url": {"url": image_url},
    }
    return image_part, None, None


async def _append_file_part(
    *,
    part: Dict[str, Any],
    token: str,
    user_id: str,
    chat_id: str,
    auth_mode: str,
    http_client: httpx.AsyncClient,
    base_url: str,
) -> Dict[str, Any]:
    """Upload a file attachment and return the file_info dict."""
    file_payload = _extract_file_payload(part)
    if not file_payload:
        raise ValueError("file content part requires file_id, file_url, or file_data")

    data_url = _build_file_data_url(file_payload)
    if not data_url:
        if file_payload.get("file_id"):
            raise ValueError(
                "file_id references are not supported on this upstream; provide file_data instead"
            )
        raise ValueError(
            "file_url is only supported when provided as a data URL"
        )

    if auth_mode == "guest":
        raise ValueError("file attachments require an authenticated upstream session")

    file_info = await upload_file(
        http_client,
        base_url,
        data_url,
        chat_id,
        token,
        user_id,
        auth_mode=auth_mode,
    )
    if not file_info:
        raise ValueError("file upload failed")

    logger.debug("[file] file added to files array")
    return file_info


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

    Uploads are collected and executed concurrently via asyncio.gather,
    then merged back in original order.

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

    # Per-message state for deferred upload assembly
    # Each entry: (msg_index, role, text_parts, upload_slots)
    # upload_slots: list of (slot_index, type, coroutine)
    deferred_messages: List[Tuple[str, List[str], List]] = []
    pending_coroutines: List = []  # (msg_idx, slot_idx, type, coro)

    for msg in normalized_messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            text_parts: List[str] = []
            upload_slots: List = []  # (slot_idx, type("image"|"file"), part_data)

            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(str(part.get("text") or ""))
                    continue

                if isinstance(part, dict) and part.get("type") == "image_url":
                    slot_idx = len(upload_slots)
                    image_url = _extract_image_url(part)
                    coro = _append_image_part(
                        image_url=image_url,
                        token=token,
                        user_id=user_id,
                        chat_id=chat_id,
                        auth_mode=auth_mode,
                        http_client=http_client,
                        base_url=base_url,
                    )
                    upload_slots.append(("image", coro))
                    pending_coroutines.append((len(deferred_messages), slot_idx, "image", coro))
                    continue

                if isinstance(part, dict) and part.get("type") in ("file", "input_file"):
                    slot_idx = len(upload_slots)
                    coro = _append_file_part(
                        part=part,
                        token=token,
                        user_id=user_id,
                        chat_id=chat_id,
                        auth_mode=auth_mode,
                        http_client=http_client,
                        base_url=base_url,
                    )
                    upload_slots.append(("file", coro))
                    pending_coroutines.append((len(deferred_messages), slot_idx, "file", coro))
                    continue

                if isinstance(part, dict) and part.get("type") == "input_audio":
                    raise ValueError(
                        "input_audio is not supported by this upstream yet"
                    )

                if isinstance(part, str):
                    text_parts.append(part)

            deferred_messages.append((role, text_parts, upload_slots))
            continue

    # Execute all uploads concurrently
    if pending_coroutines:
        coroutines = [item[3] for item in pending_coroutines]
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Build a lookup: (msg_idx, slot_idx) -> result
        result_map: Dict[Tuple[int, int], Any] = {}
        for (msg_idx, slot_idx, upload_type, _coro), result in zip(pending_coroutines, results):
            result_map[(msg_idx, slot_idx)] = (upload_type, result)

        # Assemble deferred messages
        for msg_idx, (role, text_parts, upload_slots) in enumerate(deferred_messages):
            image_parts: List[Dict[str, Any]] = []

            for slot_idx, (slot_type, _) in enumerate(upload_slots):
                key = (msg_idx, slot_idx)
                upload_type, result = result_map[key]

                if isinstance(result, Exception):
                    if upload_type == "image":
                        text_parts.append("[系统提示: 图片上传失败]")
                        logger.warning("⚠️ 图片上传异常: %s", result)
                    else:
                        logger.warning("⚠️ 文件上传异常: %s", result)
                    continue

                if upload_type == "image":
                    # result is (image_part, file_info, error_text)
                    image_part, file_info, error_text = result
                    if error_text:
                        text_parts.append(error_text)
                    if file_info:
                        files.append(file_info)
                    if image_part:
                        image_parts.append(image_part)
                else:
                    # result is file_info dict
                    files.append(result)

            message_content: List[Dict[str, Any]] = []
            combined_text = " ".join(text_parts).strip()
            if combined_text:
                message_content.append({"type": "text", "text": combined_text})
            message_content.extend(image_parts)

            if message_content:
                messages.append({"role": role, "content": message_content})
    else:
        # No uploads needed — assemble deferred messages directly
        for role, text_parts, upload_slots in deferred_messages:
            message_content_list: List[Dict[str, Any]] = []
            combined_text = " ".join(text_parts).strip()
            if combined_text:
                message_content_list.append({"type": "text", "text": combined_text})
            if message_content_list:
                messages.append({"role": role, "content": message_content_list})

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
    captcha_verify_param: Optional[str] = None,
) -> Dict[str, Any]:
    """构建发送到上游的请求 JSON body。

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
        parent_message_id: 父消息 ID（用于链式会话）。

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
            "vlm_tools_enable": False,
            "vlm_web_search_enable": False,
            "vlm_website_mode": False,
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

    # Captcha 验证参数
    if captcha_verify_param:
        body["captcha_verify_param"] = captcha_verify_param

    # 只在有图片时才包含 files 字段
    if files:
        body["files"] = files

    # 只在有 MCP 服务器时才包含
    if mcp_servers:
        body["mcp_servers"] = mcp_servers

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
            "[上游] 生成签名成功: %s... "
            "(user_id=%s, request_id=%s)",
            signature[:16], user_id, request_id,
        )
    except Exception as e:
        log_exception(logger, "[上游] 签名生成失败")
        raise RuntimeError(f"签名生成失败: {e}") from e

    # 构建请求头
    headers = build_dynamic_headers(fe_version, chat_id)
    headers.update(
        {
            "Authorization": f"Bearer {token}",
            "X-Signature": signature,
        }
    )
    apply_resin_account_header(headers, token)

    # 构建 URL query params。
    # 仅保留签名核心字段（timestamp / requestId / user_id / token / platform / version /
    # signature_timestamp）。浏览器指纹类字段（user_agent、screen_*、current_url、title、
    # local_time、…）放在 URL 上会让请求行超过 1KB，并触发阿里云 ESA WAF 的"嵌套 URL +
    # 长 URL"规则，拒绝码就是 HTTP 405 + 拦截 HTML。
    query_params = {
        "timestamp": str(timestamp_ms),
        "requestId": request_id,
        "user_id": user_id,
        "version": "0.0.1",
        "platform": "web",
        "token": token,
        "signature_timestamp": str(timestamp_ms),
    }
    signed_url = f"{api_endpoint}?{urlencode(query_params)}"

    logger.debug(
        "[上游] 请求头: Authorization=Bearer %s, "
        "X-Signature=%s...",
        token, signature[:16] if signature else '(空)',
    )
    logger.debug(
        "[上游] URL 参数: timestamp=%s, "
        "requestId=%s, user_id=%s",
        timestamp_ms, request_id, user_id,
    )

    return signed_url, headers, fe_version
