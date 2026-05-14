#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""上游适配器。

UpstreamClient 作为薄集成层，组合各子模块完成完整的请求/响应处理流程：
- http_client.py     — HTTP 客户端管理
- headers.py         — 动态浏览器 headers
- models.py          — 模型映射与特性解析
- message_utils.py   — 消息预处理
- request_signing.py — 请求体构建与签名
- retry_policy.py    — 双池重试策略
- response_handler.py — 流式/非流式响应处理
- file_upload.py     — 文件上传

所有公有方法签名与原实现完全一致。
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Union

import httpx

from app.core.config import settings
from app.core.http_client import SharedHttpClients
from app.core.headers import build_dynamic_headers
from app.core.models import ModelManager
from app.utils.jwt_utils import extract_user_id_from_token
from app.core.message_utils import extract_last_user_text, preprocess_openai_messages
from app.core.request_signing import (
    process_multimodal_messages,
    build_upstream_body,
    sign_request,
)
from app.core.retry_policy import (
    RetryPolicy,
    extract_upstream_error_details,
    get_upstream_page_error_message,
    is_concurrency_limited,
    is_upstream_page_response,
)
from app.core.response_handler import ResponseHandler
from app.core.session import SessionManager
from app.core.session.session_content import (
    build_session_body_messages,
    extract_turn_content,
    get_precreate_content,
    inject_system_prompt,
)
from app.core.file_upload import upload_file as _upload_file
from app.core.httpx_errors import normalize_httpx_exception, normalize_httpx_response
from app.core.openai_compat import (
    get_error_message,
    handle_error,
)
from app.core.resin_compat import apply_resin_account_header
from app.core.upstream_urls import build_upstream_url, get_api_endpoint, get_upstream_base_url
from app.models.schemas import OpenAIRequest
from app.utils.fe_version import get_latest_fe_version
from app.core.captcha_client import get_captcha_client
from app.utils.logger import logger
from app.utils.token_pool import get_token_pool
from app.utils.guest_session_pool import get_guest_session_pool
from app.utils.request_logging import _openai_response_has_output, _sse_chunk_has_output, write_request_log
from app.utils.request_source import RequestSourceInfo


def generate_uuid() -> str:
    """生成UUID v4"""
    return str(uuid.uuid4())


@dataclass
class PreparedConversationRequest:
    """请求准备阶段收敛后的统一结果。"""

    token: str
    user_id: str
    auth_mode: str
    token_source: str
    guest_user_id: Optional[str]
    chat_id: str
    message_id: str
    user_message_id: str
    parent_id: Optional[str]
    body_messages: List[Dict[str, Any]]
    session_commit: Optional[Dict[str, Any]]



# --------------------------------------------------------------------------
# UpstreamClient
# --------------------------------------------------------------------------

class UpstreamClient:
    """当前服务使用的上游适配器（薄集成层）。"""

    def __init__(self):
        self.name = "upstream"
        self.logger = logger
        self.api_endpoint = get_api_endpoint()

        # 当前上游特定配置
        self.base_url = get_upstream_base_url()
        self.auth_url = build_upstream_url("/api/v1/auths/")

        # 子模块
        self._http_clients = SharedHttpClients()
        self._model_manager = ModelManager()
        self._retry_policy = RetryPolicy()
        self._response_handler = ResponseHandler()
        self._session_manager = SessionManager(
            session_ttl=settings.SESSION_TTL,
            max_sessions_per_client=settings.SESSION_MAX_PER_CLIENT,
            cleanup_interval=settings.SESSION_CLEANUP_INTERVAL,
        )

        # 在线模型缓存（实例变量，避免多实例混用）
        self._online_models: Optional[List[Dict[str, Any]]] = None
        self._online_models_time: float = 0.0
        self._online_models_lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # HTTP 客户端访问
    # ------------------------------------------------------------------

    def _get_shared_client(self) -> httpx.AsyncClient:
        return self._http_clients.get_client()

    def _get_shared_stream_client(self) -> httpx.AsyncClient:
        return self._http_clients.get_stream_client()

    async def close(self) -> None:
        """关闭共享 HTTP 客户端连接和会话管理器。"""
        await self._http_clients.close()
        await self._session_manager.close()

    # ------------------------------------------------------------------
    # 重试预算（委托到 RetryPolicy）
    # ------------------------------------------------------------------

    async def _get_guest_retry_limit(self) -> int:
        """匿名号池可提供的最大重试预算。"""
        return await self._retry_policy.get_guest_retry_limit()

    async def _get_authenticated_retry_limit(self) -> int:
        """认证号池与静态 Token 可提供的最大重试预算。"""
        return await self._retry_policy.get_authenticated_retry_limit()

    async def _get_total_retry_limit(self) -> int:
        """综合认证号池与匿名号池的最大尝试次数。"""
        return await self._retry_policy.get_total_retry_limit()

    # ------------------------------------------------------------------
    # 重试决策（委托到 RetryPolicy）
    # ------------------------------------------------------------------

    def _is_guest_auth(self, transformed: Dict[str, Any]) -> bool:
        """判断当前请求是否使用匿名会话。"""
        return self._retry_policy.is_guest_auth(transformed)

    def _should_retry_guest_session(
        self,
        status_code: int,
        is_concurrency_limited_flag: bool,
        attempt: int,
        max_attempts: int,
        transformed: Dict[str, Any],
    ) -> bool:
        """判断匿名号池是否需要刷新会话后重试。"""
        return self._retry_policy.should_retry_guest_session(
            status_code, is_concurrency_limited_flag, attempt, max_attempts, transformed
        )

    def _should_retry_authenticated_session(
        self,
        status_code: int,
        is_concurrency_limited_flag: bool,
        attempt: int,
        max_attempts: int,
        transformed: Dict[str, Any],
    ) -> bool:
        """判断认证号池是否需要切号重试。"""
        return self._retry_policy.should_retry_authenticated_session(
            status_code, is_concurrency_limited_flag, attempt, max_attempts, transformed
        )

    async def _release_guest_session(self, transformed: Dict[str, Any]) -> None:
        """释放当前匿名会话占用。"""
        await self._retry_policy.release_guest_session(transformed)

    async def _release_authenticated_token_allocation(
        self,
        transformed: Optional[Dict[str, Any]],
    ) -> None:
        """释放认证 token 的并发占用，不记录成功或失败。"""
        if not transformed or self._is_guest_auth(transformed):
            return

        token = str(transformed.get("token") or "")
        if not token:
            return

        token_pool = get_token_pool()
        if token_pool:
            await token_pool.release_token_allocation(token)

    async def _report_guest_session_failure(
        self,
        transformed: Dict[str, Any],
        *,
        is_concurrency_limited: bool = False,
    ) -> None:
        """上报匿名会话失败并补齐新会话。"""
        await self._retry_policy.report_guest_session_failure(
            transformed, is_concurrency_limited_flag=is_concurrency_limited
        )

    async def _refresh_guest_request(
        self,
        request: OpenAIRequest,
        attempt: int,
        excluded_tokens: Set[str],
        excluded_guest_user_ids: Set[str],
        failed_transformed: Dict[str, Any],
        is_concurrency_limited: bool = False,
    ) -> Dict[str, Any]:
        """匿名会话失效或并发受限后切换会话并重签请求。"""
        retry_number = attempt + 2
        self.logger.warning(
            "匿名会话不可用，正在切换匿名会话并进行第 "
            "%s 次请求",
            retry_number,
        )
        await self._report_guest_session_failure(
            failed_transformed,
            is_concurrency_limited=is_concurrency_limited,
        )
        return await self.transform_request(
            request,
            excluded_tokens=excluded_tokens,
            excluded_guest_user_ids=excluded_guest_user_ids,
        )

    async def _refresh_authenticated_request(
        self,
        request: OpenAIRequest,
        attempt: int,
        excluded_tokens: Set[str],
        excluded_guest_user_ids: Set[str],
    ) -> Dict[str, Any]:
        """认证模式下切换到下一枚 Token，并允许回退匿名池。"""
        retry_number = attempt + 2
        self.logger.warning(
            "检测到认证会话不可用，正在切换认证 Token/回退匿名池并进行第 "
            "%s 次请求",
            retry_number,
        )
        return await self.transform_request(
            request,
            excluded_tokens=excluded_tokens,
            excluded_guest_user_ids=excluded_guest_user_ids,
        )

    async def _commit_session_if_needed(self, transformed: Dict[str, Any]) -> None:
        """在上游已接受请求后，提交连续会话状态。"""
        commit_info = transformed.pop("session_commit", None)
        if not isinstance(commit_info, dict):
            return

        action = str(commit_info.get("action") or "")
        try:
            if action == "reuse":
                await self._session_manager.commit_session_turn(
                    model=str(commit_info["model"]),
                    messages=list(commit_info["messages"]),
                    chat_id=str(commit_info["chat_id"]),
                    message_id=str(commit_info["message_id"]),
                )
                return

            self.logger.warning("未知会话提交动作: %s", action)
        except Exception as exc:
            self.logger.warning(
                "提交会话状态失败",
                exc_info=settings.DEBUG_LOGGING,
            )

    # ------------------------------------------------------------------
    # 错误解析（委托到 retry_policy 工具函数）
    # ------------------------------------------------------------------

    def _extract_upstream_error_details(
        self,
        status_code: int,
        error_text: str,
        content_type: Optional[str] = None,
    ):
        """解析上游错误响应中的 code/message。"""
        return extract_upstream_error_details(
            status_code,
            error_text,
            content_type=content_type,
        )

    def _is_concurrency_limited(
        self,
        status_code: int,
        error_code,
        error_message: str,
    ) -> bool:
        """判断是否为上游并发限制/429 场景。"""
        return is_concurrency_limited(status_code, error_code, error_message)

    async def _try_get_captcha_token(
        self, transformed: Dict[str, Any]
    ) -> bool:
        """获取 captcha token 并写入 transformed body。

        Returns:
            True 表示成功获取并写入，False 表示失败，调用方自行决定后续操作。
        """
        captcha_client = get_captcha_client()
        if not captcha_client:
            return False
        try:
            fresh_token = await captcha_client.get_token(
                str(transformed.get("token") or "")
            )
            transformed["body"]["captcha_verify_param"] = fresh_token
            return True
        except Exception as e:
            self.logger.warning("[captcha] get token failed: %s", e)
            return False

    @staticmethod
    def _build_upstream_error_response(
        status_code: int,
        error_code: Optional[int],
        error_message: str,
        *,
        is_page_error: bool = False,
    ) -> Dict[str, Any]:
        """统一构建对客户端返回的上游错误对象。"""
        normalized_status = (
            status_code if isinstance(status_code, int) and status_code >= 400 else 502
        )

        if is_page_error:
            return {
                "error": {
                    "message": get_upstream_page_error_message(normalized_status),
                    "type": "upstream_page_error",
                    "code": normalized_status,
                }
            }

        message = (error_message or "").strip() or f"Upstream error: {status_code}"
        code = error_code if isinstance(error_code, int) else status_code

        return {
            "error": {
                "message": message,
                "type": "upstream_error",
                "code": code,
            }
        }

    # ------------------------------------------------------------------
    # 会话预创建（对齐浏览器 /api/v1/chats/new 流程）
    # ------------------------------------------------------------------

    async def _precreate_chat(
        self,
        token: str,
        fe_version: str,
        model: str,
        user_msg_id: str,
        content: str,
        enable_thinking: bool = False,
        auto_web_search: bool = False,
        mcp_servers: Optional[List[str]] = None,
        allow_fallback: bool = True,
        max_retries: int = 3,
    ) -> str:
        """调用 /api/v1/chats/new 预创建会话，返回服务端分配的 chat_id。

        浏览器在发 completions 之前会先调用此接口创建会话；
        缺少此步骤会导致上游在 done 阶段返回 INTERNAL_ERROR。
        """
        now_ts = int(time.time())
        body = {
            "chat": {
                "id": "",
                "title": "新聊天",
                "models": [model],
                "params": {},
                "history": {
                    "messages": {
                        user_msg_id: {
                            "id": user_msg_id,
                            "parentId": None,
                            "childrenIds": [],
                            "role": "user",
                            "content": content or "hi",
                            "timestamp": now_ts,
                            "models": [model],
                        }
                    },
                    "currentId": user_msg_id,
                },
                "tags": [],
                "flags": [],
                "features": [
                    {
                        "type": "tool_selector",
                        "server": "tool_selector_h",
                        "status": "hidden",
                    }
                ],
                "mcp_servers": mcp_servers or [],
                "enable_thinking": enable_thinking,
                "auto_web_search": auto_web_search,
                "message_version": 1,
                "extra": {},
                "timestamp": int(time.time() * 1000),
            }
        }

        headers = build_dynamic_headers(fe_version)
        headers["Authorization"] = f"Bearer {token}"
        apply_resin_account_header(headers, token)

        client = self._get_shared_client()
        attempts = max(1, max_retries)
        for attempt in range(attempts):
            try:
                resp = await client.post(
                    f"{self.base_url}/api/v1/chats/new",
                    json=body,
                    headers=headers,
                )
                if resp.status_code == 200:
                    chat_id = resp.json().get("id", "")
                    if chat_id:
                        self.logger.debug(
                            "[chat] pre-created chat_id=%s", chat_id
                        )
                        return chat_id
                trace_id = (
                    resp.headers.get("eagleid")
                    or resp.headers.get("x-trace-id")
                    or resp.headers.get("cf-ray")
                    or "-"
                )
                self.logger.warning(
                    "[chat] pre-create failed: HTTP %s (attempt %s/%s) "
                    "server=%s trace=%s body=%r",
                    resp.status_code,
                    attempt + 1,
                    attempts,
                    resp.headers.get("server", "-"),
                    trace_id,
                    resp.text[:200],
                )
            except Exception as e:
                self.logger.warning(
                    "[chat] pre-create error (attempt %s/%s)",
                    attempt + 1,
                    attempts,
                    exc_info=settings.DEBUG_LOGGING,
                )

            if attempt + 1 < attempts:
                await asyncio.sleep(min(2 ** attempt, 8))

        if allow_fallback:
            # 降级：使用随机 chat_id（会触发 done 阶段 INTERNAL_ERROR，但内容不受影响）
            self.logger.warning(
                "[chat] pre-create failed after %s attempts, fallback to random chat_id",
                attempts,
            )
            return generate_uuid()

        raise RuntimeError(
            f"pre-create chat_id failed after {attempts} attempts"
        )

    async def _prepare_session_request(
        self,
        request: OpenAIRequest,
        *,
        raw_messages: List[Dict[str, Any]],
        normalized_messages: List[Dict[str, Any]],
        session_turn_content: str,
        features: Dict[str, Any],
        token: str,
        user_id: str,
        auth_mode: str,
        token_source: str,
        guest_user_id: Optional[str],
        fe_version: str,
        auth_info: Dict[str, Any],
    ) -> PreparedConversationRequest:
        session_result = await self._session_manager.find_session(
            model=request.model,
            messages=raw_messages,
        )

        if (
            session_result
            and not session_result.parent_id
            and session_result.bound_token
            and session_result.bound_token != token
        ):
            self.logger.debug(
                "Token 已切换，跳过待确认会话 chat_id=%s",
                session_result.chat_id[:8],
            )
            session_result = None

        if session_result:
            chat_id = session_result.chat_id
            if session_result.bound_token and session_result.bound_token != token:
                self.logger.debug(
                    "链式会话 token 亲和: 切换到 bound_token (chat_id=%s)",
                    chat_id[:8],
                )
                await self._release_guest_session(auth_info)
                token = session_result.bound_token
                user_id = extract_user_id_from_token(token)
                auth_mode = "authenticated"
                token_source = "session_bound"
                guest_user_id = None
                auth_info["token"] = token
                auth_info["user_id"] = user_id
                auth_info["auth_mode"] = auth_mode
                auth_info["token_source"] = token_source
                auth_info["guest_user_id"] = guest_user_id

            user_message_id = session_result.message_id
            message_id = generate_uuid()
            parent_id = session_result.parent_id
            is_first_turn_retry = not parent_id
            self.logger.debug(
                "%s chat_id=%s, parent_id=%s",
                "首轮重试复用" if is_first_turn_retry else "复用会话",
                chat_id[:8],
                parent_id[:8] if parent_id else "None",
            )
            body_messages = build_session_body_messages(
                normalized_messages=normalized_messages,
                session_turn_content=session_turn_content,
                is_new_session=is_first_turn_retry,
                inject_system=settings.SESSION_SYSTEM_INJECT,
            )
            return PreparedConversationRequest(
                token=token,
                user_id=user_id,
                auth_mode=auth_mode,
                token_source=token_source,
                guest_user_id=guest_user_id,
                chat_id=chat_id,
                message_id=message_id,
                user_message_id=user_message_id,
                parent_id=parent_id,
                body_messages=body_messages,
                session_commit={
                    "action": "reuse",
                    "model": request.model,
                    "messages": raw_messages,
                    "chat_id": chat_id,
                    "message_id": user_message_id,
                },
            )

        message_id = generate_uuid()
        user_message_id = generate_uuid()
        body_messages = build_session_body_messages(
            normalized_messages=normalized_messages,
            session_turn_content=session_turn_content,
            is_new_session=True,
            inject_system=settings.SESSION_SYSTEM_INJECT,
        )
        precreate_content = get_precreate_content(body_messages)
        chat_id = await self._precreate_chat(
            token=token,
            fe_version=fe_version,
            model=features["upstream_model_id"],
            user_msg_id=user_message_id,
            content=precreate_content,
            enable_thinking=features["enable_thinking"],
            auto_web_search=features["auto_web_search"],
            mcp_servers=features.get("mcp_servers", []),
            allow_fallback=False,
        )
        await self._session_manager.create_session(
            auth_token=token,
            model=request.model,
            messages=raw_messages,
            chat_id=chat_id,
            message_id="",
        )
        return PreparedConversationRequest(
            token=token,
            user_id=user_id,
            auth_mode=auth_mode,
            token_source=token_source,
            guest_user_id=guest_user_id,
            chat_id=chat_id,
            message_id=message_id,
            user_message_id=user_message_id,
            parent_id=None,
            body_messages=body_messages,
            session_commit={
                "action": "reuse",
                "model": request.model,
                "messages": raw_messages,
                "chat_id": chat_id,
                "message_id": user_message_id,
            },
        )

    async def _prepare_direct_request(
        self,
        *,
        normalized_messages: List[Dict[str, Any]],
        features: Dict[str, Any],
        token: str,
        user_id: str,
        auth_mode: str,
        token_source: str,
        guest_user_id: Optional[str],
        fe_version: str,
    ) -> PreparedConversationRequest:
        message_id = generate_uuid()
        user_message_id = generate_uuid()
        body_messages = (
            inject_system_prompt(normalized_messages)
            if settings.SESSION_SYSTEM_INJECT
            else normalized_messages
        )
        if settings.PRECREATE_CHAT:
            precreate_content = get_precreate_content(body_messages)
            chat_id = await self._precreate_chat(
                token=token,
                fe_version=fe_version,
                model=features["upstream_model_id"],
                user_msg_id=user_message_id,
                content=precreate_content,
                enable_thinking=features["enable_thinking"],
                auto_web_search=features["auto_web_search"],
                mcp_servers=features.get("mcp_servers", []),
                allow_fallback=False,
            )
        else:
            chat_id = generate_uuid()

        return PreparedConversationRequest(
            token=token,
            user_id=user_id,
            auth_mode=auth_mode,
            token_source=token_source,
            guest_user_id=guest_user_id,
            chat_id=chat_id,
            message_id=message_id,
            user_message_id=user_message_id,
            parent_id=None,
            body_messages=body_messages,
            session_commit=None,
        )

    # 在线模型（缓存层保留在本类）
    # ------------------------------------------------------------------

    async def get_online_models(self) -> List[Dict[str, Any]]:
        """获取上游在线模型详细信息（缓存1小时，asyncio.Lock 防并发刷新）。"""
        now = time.time()
        # 快速路径：缓存命中，无需加锁
        if self._online_models and (now - self._online_models_time < 3600):
            return self._online_models

        async with self._online_models_lock:
            # 加锁后二次检查，避免多协程同时刷新
            now = time.time()
            if self._online_models and (now - self._online_models_time < 3600):
                return self._online_models

            try:
                fe_version = await get_latest_fe_version()
                headers = build_dynamic_headers(fe_version=fe_version)
                auth_info = await self.get_auth_info()
                token = auth_info.get("token", "")
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                    apply_resin_account_header(headers, token)
                client = self._get_shared_client()
                response = await client.get(
                    f"{self.base_url}/api/models", headers=headers, timeout=10.0
                )

                if response.status_code == 200:
                    data = response.json()
                    models_data = data.get("data", [])

                    parsed_models = []
                    for item in models_data:
                        model_id = item.get("id")
                        if not model_id:
                            continue

                        owned_by = item.get("owned_by", "openai")

                        info = item.get("info", {})
                        display_name = info.get("name") or item.get("name") or model_id
                        is_active = info.get("is_active", True)
                        created_at = info.get("created_at") or int(now)
                        updated_at = info.get("updated_at")

                        meta = info.get("meta") or {}
                        capabilities = meta.get("capabilities") or {}
                        mcp_servers = meta.get("mcpServerIds") or []

                        raw_tags = meta.get("tags") or []
                        tags = [
                            tag.get("name")
                            for tag in raw_tags
                            if isinstance(tag, dict) and tag.get("name")
                        ]

                        parsed_models.append({
                            "id": model_id,
                            "name": display_name,
                            "owned_by": owned_by,
                            "is_active": is_active,
                            "created": created_at,
                            "updated_at": updated_at,
                            "capabilities": capabilities,
                            "mcpServerIds": mcp_servers,
                            "tags": tags,
                        })

                    self._online_models = parsed_models
                    self._online_models_time = now
                    self._model_manager.load_from_online_models(parsed_models)
                    # 持久化到数据库
                    await self._save_models_cache(parsed_models)
                    self.logger.debug(
                        "在线模型同步成功，共获取 %s 个模型", len(parsed_models)
                    )
                    return parsed_models
                else:
                    self.logger.warning(
                        "%s",
                        normalize_httpx_response(
                            response.status_code,
                            response.text,
                            content_type=response.headers.get("content-type"),
                            method="GET",
                            url=f"{self.base_url}/api/models",
                            context="upstream.models",
                        ),
                    )
            except Exception as exc:
                self.logger.warning(
                    "%s",
                    normalize_httpx_exception(
                        exc,
                        method="GET",
                        url=f"{self.base_url}/api/models",
                        context="upstream.models",
                    ),
                    exc_info=settings.DEBUG_LOGGING,
                )

        return self._online_models or []

    _MODELS_CACHE_KEY = "online_models_cache"

    async def _save_models_cache(self, models: List[Dict[str, Any]]) -> None:
        """将在线模型数据持久化到 config_items 表。"""
        try:
            from app.services.config_dao import get_config_dao
            dao = get_config_dao()
            await dao.set(self._MODELS_CACHE_KEY, json.dumps(models, ensure_ascii=False))
            self.logger.debug("在线模型缓存已写入数据库")
        except Exception as exc:
            self.logger.warning(
                "在线模型缓存写入数据库失败",
                exc_info=settings.DEBUG_LOGGING,
            )

    async def load_cached_models(self) -> bool:
        """从数据库加载缓存的在线模型数据，成功返回 True。"""
        try:
            from app.services.config_dao import get_config_dao
            dao = get_config_dao()
            raw = await dao.get(self._MODELS_CACHE_KEY)
            if not raw:
                return False
            models = json.loads(raw)
            if not isinstance(models, list) or not models:
                return False
            self._online_models = models
            self._online_models_time = time.time()
            self._model_manager.load_from_online_models(models)
            self.logger.debug(
                "从数据库缓存加载 %s 个在线模型，"
                "生成 %s 个变体",
                len(models),
                len(self._model_manager.get_supported_models()),
            )
            return True
        except Exception as exc:
            self.logger.warning(
                "从数据库加载在线模型缓存失败",
                exc_info=settings.DEBUG_LOGGING,
            )
            return False

    def get_supported_models(self) -> List[str]:
        """获取支持的模型列表"""
        return self._model_manager.get_supported_models()

    # ------------------------------------------------------------------
    # 鉴权
    # ------------------------------------------------------------------

    async def _fetch_direct_guest_auth(self) -> Dict[str, Any]:
        """匿名号池缺席时，兜底直连拉取一个访客令牌。"""
        max_retries = 3

        for retry_count in range(max_retries):
            try:
                fe_version = await get_latest_fe_version()
                headers = build_dynamic_headers(fe_version=fe_version)
                self.logger.debug(
                    "尝试获取访客令牌 (第%s次): %s", retry_count + 1, self.auth_url
                )

                client = self._get_shared_client()
                response = await client.get(self.auth_url, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    token = str(data.get("token") or "").strip()
                    if token:
                        user_id = str(
                            data.get("id")
                            or data.get("user_id")
                            or extract_user_id_from_token(token)
                        )
                        username = str(
                            data.get("name")
                            or str(data.get("email") or "").split("@")[0]
                            or "Guest"
                        )
                        self.logger.debug(
                            "直连获取匿名令牌成功: %s...", token[:20]
                        )
                        return {
                            "token": token,
                            "user_id": user_id,
                            "username": username or "Guest",
                            "auth_mode": "guest",
                            "token_source": "guest_direct",
                            "guest_user_id": user_id,
                        }

                    self.logger.warning("响应中未找到 token 字段: %s", data)
                elif response.status_code == 405:
                    self.logger.error(
                        "🚫 请求被 WAF 拦截 (405)，无法直连获取匿名令牌"
                    )
                    break
                else:
                    self.logger.warning(
                        "%s",
                        normalize_httpx_response(
                            response.status_code,
                            response.text,
                            content_type=response.headers.get("content-type"),
                            method="GET",
                            url=self.auth_url,
                            context="guest_auth.direct",
                        ),
                    )
            except httpx.TimeoutException as exc:
                self.logger.warning(
                    "%s",
                    normalize_httpx_exception(
                        exc,
                        method="GET",
                        url=self.auth_url,
                        context="guest_auth.direct",
                        attempt=retry_count + 1,
                    ),
                )
            except httpx.ConnectError as exc:
                self.logger.warning(
                    "%s",
                    normalize_httpx_exception(
                        exc,
                        method="GET",
                        url=self.auth_url,
                        context="guest_auth.direct",
                        attempt=retry_count + 1,
                    ),
                )
            except json.JSONDecodeError as exc:
                self.logger.warning(
                    "直连获取匿名令牌 JSON 解析错误 (第%s次): %s", retry_count + 1, exc
                )
            except Exception as exc:
                self.logger.warning(
                    "%s",
                    normalize_httpx_exception(
                        exc,
                        method="GET",
                        url=self.auth_url,
                        context="guest_auth.direct",
                        attempt=retry_count + 1,
                    ),
                    exc_info=settings.DEBUG_LOGGING,
                )

            if retry_count + 1 < max_retries:
                # 指数退避: 1s → 2s → 4s（最大 8s）
                await asyncio.sleep(min(2 ** retry_count, 8))

        return {
            "token": "",
            "user_id": "guest",
            "username": "Guest",
            "auth_mode": "guest",
            "token_source": "guest_direct",
            "guest_user_id": None,
        }

    async def get_auth_info(
        self,
        excluded_tokens: Optional[Set[str]] = None,
        excluded_guest_user_ids: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """优先获取认证 Token，必要时回退匿名会话池。"""
        token_pool = get_token_pool()
        if token_pool:
            token = await token_pool.get_next_token(exclude_tokens=excluded_tokens)
            if token:
                user_id = extract_user_id_from_token(token)
                self.logger.debug("从认证号池获取令牌: %s...", token[:20])
                return {
                    "token": token,
                    "user_id": user_id,
                    "username": "User",
                    "auth_mode": "authenticated",
                    "token_source": "auth_pool",
                    "guest_user_id": None,
                }

        if settings.ANONYMOUS_MODE:
            guest_pool = get_guest_session_pool()
            if guest_pool:
                try:
                    session = await guest_pool.acquire(
                        exclude_user_ids=excluded_guest_user_ids
                    )
                    self.logger.debug(
                        "认证池不可用，回退匿名会话池: "
                        "user_id=%s",
                        session.user_id,
                    )
                    return {
                        "token": session.token,
                        "user_id": session.user_id,
                        "username": session.username,
                        "auth_mode": "guest",
                        "token_source": "guest_pool",
                        "guest_user_id": session.user_id,
                    }
                except Exception as exc:
                    self.logger.warning(
                        "匿名会话池获取失败，转为直连访客鉴权",
                        exc_info=settings.DEBUG_LOGGING,
                    )

            return await self._fetch_direct_guest_auth()

        self.logger.info(
            "[guest_session.disabled] ANONYMOUS_MODE=false, skip guest pool and direct guest auth fallback"
        )

        self.logger.error("❌ 无法获取有效的上游令牌")
        return {
            "token": "",
            "user_id": "",
            "username": "",
            "auth_mode": "authenticated",
            "token_source": "none",
            "guest_user_id": None,
        }

    async def mark_token_failure(self, token: str, error: Exception = None):
        """标记token使用失败"""
        token_pool = get_token_pool()
        if token_pool:
            await token_pool.record_token_failure(token, error)

    # ------------------------------------------------------------------
    # 文件上传（委托到 file_upload 模块）
    # ------------------------------------------------------------------

    async def upload_file(
        self,
        data_url: str,
        chat_id: str,
        token: str,
        user_id: str,
        auth_mode: str = "authenticated",
        message_id: str = "",
    ) -> Optional[Dict]:
        """上传文件（图片/文档）到上游服务器。

        Args:
            data_url: data:mime/type;base64,... 格式的文件数据
            chat_id: 当前对话ID
            token: 认证令牌
            user_id: 用户ID
            auth_mode: 当前鉴权模式，guest 模式下禁止上传
            message_id: 关联的用户消息ID

        Returns:
            上传成功返回完整的文件信息字典，失败返回 None
        """
        return await _upload_file(
            client=self._get_shared_client(),
            base_url=self.base_url,
            data_url=data_url,
            chat_id=chat_id,
            token=token,
            user_id=user_id,
            auth_mode=auth_mode,
            message_id=message_id,
        )

    # ------------------------------------------------------------------
    # 请求转换
    # ------------------------------------------------------------------

    async def transform_request(
        self,
        request: OpenAIRequest,
        excluded_tokens: Optional[Set[str]] = None,
        excluded_guest_user_ids: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """转换 OpenAI 请求为上游格式。"""
        self.logger.debug("转换 OpenAI 请求到上游格式: %s", request.model)

        auth_info: Optional[Dict[str, Any]] = None  # 用于 finally 中 guest session 清理

        raw_messages = [
            message.model_dump(exclude_none=True) for message in request.messages
        ]

        normalized_messages = preprocess_openai_messages(raw_messages)

        # 并行拉取 auth_info 和 fe_version，减少 TTFB
        auth_info, fe_version = await asyncio.gather(
            self.get_auth_info(
                excluded_tokens=excluded_tokens,
                excluded_guest_user_ids=excluded_guest_user_ids,
            ),
            get_latest_fe_version(),
        )
        token = str(auth_info.get("token") or "")
        if not token:
            # guest session 已 acquire，失败时需归还
            await self._release_guest_session(auth_info)
            raise RuntimeError("无法获取上游认证令牌")

        user_id = str(
            auth_info.get("user_id") or extract_user_id_from_token(token)
        )
        auth_mode = str(auth_info.get("auth_mode") or "authenticated")
        token_source = str(auth_info.get("token_source") or "unknown")
        guest_user_id = auth_info.get("guest_user_id")

        try:
            # 提取最后一条用户消息（用于签名和会话预创建，两种模式均需要）
            last_user_text = extract_last_user_text(raw_messages)
            session_turn_content = extract_turn_content(
                raw_messages=raw_messages,
                normalized_messages=normalized_messages,
                fallback_user_text=last_user_text,
            )

            # 解析模型特性（两种模式均需要，precreate 路径也依赖 features）
            features = self._model_manager.resolve_model_features(request)
            self.logger.debug(
                "Resolved model features for %s: %s, temperature=%s, max_tokens=%s",
                request.model,
                features,
                request.temperature,
                request.max_tokens,
            )

            if settings.SESSION_ENABLED:
                prepared = await self._prepare_session_request(
                    request,
                    raw_messages=raw_messages,
                    normalized_messages=normalized_messages,
                    session_turn_content=session_turn_content,
                    features=features,
                    token=token,
                    user_id=user_id,
                    auth_mode=auth_mode,
                    token_source=token_source,
                    guest_user_id=guest_user_id,
                    fe_version=fe_version,
                    auth_info=auth_info,
                )
            else:
                prepared = await self._prepare_direct_request(
                    normalized_messages=normalized_messages,
                    features=features,
                    token=token,
                    user_id=user_id,
                    auth_mode=auth_mode,
                    token_source=token_source,
                    guest_user_id=guest_user_id,
                    fe_version=fe_version,
                )

            messages, files = await process_multimodal_messages(
                normalized_messages=prepared.body_messages,
                token=prepared.token,
                user_id=prepared.user_id,
                chat_id=prepared.chat_id,
                auth_mode=prepared.auth_mode,
                http_client=self._get_shared_client(),
                base_url=self.base_url,
            )

            # 构建请求体
            body = build_upstream_body(
                messages=messages,
                files=files,
                upstream_model_id=features["upstream_model_id"],
                last_user_text=last_user_text,
                chat_id=prepared.chat_id,
                message_id=prepared.message_id,
                enable_thinking=features["enable_thinking"],
                web_search=features["web_search"],
                auto_web_search=features["auto_web_search"],
                flags=features["flags"],
                extra=features["extra"],
                mcp_servers=features["mcp_servers"],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                parent_message_id=prepared.parent_id,
            )
            # 对齐浏览器：current_user_message_id 使用和 chats/new 一致的 ID
            body["current_user_message_id"] = prepared.user_message_id

            if settings.CAPTCHA_ENABLED:
                transformed_ctx = {"token": token, "body": body}
                if await self._try_get_captcha_token(transformed_ctx):
                    self.logger.info("[captcha] token added to body")

            self.logger.debug("Upstream request body: %s", body)

            # 签名并生成最终 URL 和 headers（复用已并行拉取的 fe_version）
            signed_url, headers, _fe_version = await sign_request(
                api_endpoint=self.api_endpoint,
                user_id=prepared.user_id,
                last_user_text=last_user_text,
                chat_id=prepared.chat_id,
                token=prepared.token,
                fe_version=fe_version,
            )

        except Exception:
            # 签名/构建失败时归还 guest session，避免永久占用
            await self._release_authenticated_token_allocation(auth_info)
            await self._release_guest_session(auth_info)
            raise

        return {
            "url": signed_url,
            "headers": headers,
            "body": body,
            "token": prepared.token,
            "chat_id": prepared.chat_id,
            "model": request.model,
            "user_id": prepared.user_id,
            "auth_mode": prepared.auth_mode,
            "token_source": prepared.token_source,
            "guest_user_id": prepared.guest_user_id,
            "session_commit": prepared.session_commit,
        }


    # ------------------------------------------------------------------
    # 聊天完成
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        request: OpenAIRequest,
        *,
        http_request=None,
        **kwargs
    ) -> tuple[Union[Dict[str, Any], AsyncGenerator[str, None]], Optional[str]]:
        """聊天完成接口。

        Args:
            request: OpenAI 请求对象。
            http_request: FastAPI Request 对象，用于检测客户端断开。
        """
        self.logger.debug("%s 处理请求: %s", self.name, request.model)
        self.logger.debug("  消息数量: %s", len(request.messages))
        self.logger.debug("  流式模式: %s", request.stream)

        transformed: Dict[str, Any] = {}
        try:
            transformed = await self.transform_request(request)
            max_attempts = await self._get_total_retry_limit()

            if request.stream:
                result = await self._create_stream_response(
                    request, transformed, http_request=http_request,
                )
                return result, str(transformed.get("token") or "") or None

            client = self._get_shared_client()
            excluded_tokens: Set[str] = set()
            excluded_guest_user_ids: Set[str] = set()
            empty_retries = 0
            captcha_retries = 0

            async with asyncio.timeout(settings.CHAT_TOTAL_TIMEOUT):
                for attempt in range(max_attempts):
                    response = await client.post(
                        transformed["url"],
                        headers=transformed["headers"],
                        json=transformed["body"],
                    )
                    response_text = response.text
                    content_type = response.headers.get("content-type")
                    is_page_error = is_upstream_page_response(
                        content_type,
                        response_text,
                    )

                    error_code, error_message = extract_upstream_error_details(
                        response.status_code,
                        response_text,
                        content_type=content_type,
                    )
                    is_concurrency_limited_flag = (
                        not is_page_error
                        and is_concurrency_limited(
                            response.status_code,
                            error_code,
                            error_message,
                        )
                    )

                    if (
                        settings.CAPTCHA_ENABLED
                        and error_message
                        and "FRONTEND_CAPTCHA_REQUIRED" in str(error_message)
                        and captcha_retries < settings.CAPTCHA_MAX_RETRIES
                    ):
                        captcha_retries += 1
                        if await self._try_get_captcha_token(transformed):
                            self.logger.warning(
                                "captcha required, retrying with fresh token "
                                "(captcha_retry %s/%s)",
                                captcha_retries,
                                settings.CAPTCHA_MAX_RETRIES,
                            )
                            continue

                    if is_page_error:
                        await self._release_authenticated_token_allocation(
                            transformed
                        )
                        await self._release_guest_session(transformed)
                        self.logger.error(
                            "%s",
                            normalize_httpx_response(
                                response.status_code,
                                response_text,
                                content_type=content_type,
                                method="POST",
                                url=transformed["url"],
                                context="upstream.non_stream.page",
                            ),
                        )
                        return (
                            self._build_upstream_error_response(
                                response.status_code,
                                error_code,
                                error_message,
                                is_page_error=True,
                            ),
                            str(transformed.get("token") or "") or None,
                        )

                    if self._should_retry_guest_session(
                        response.status_code,
                        is_concurrency_limited_flag,
                        attempt,
                        max_attempts,
                        transformed,
                    ):
                        guest_user_id = str(
                            transformed.get("guest_user_id")
                            or transformed.get("user_id")
                            or ""
                        )
                        if guest_user_id:
                            excluded_guest_user_ids.add(guest_user_id)
                        transformed = await self._refresh_guest_request(
                            request,
                            attempt,
                            excluded_tokens,
                            excluded_guest_user_ids,
                            transformed,
                            is_concurrency_limited=is_concurrency_limited_flag,
                        )
                        continue

                    if self._should_retry_authenticated_session(
                        response.status_code,
                        is_concurrency_limited_flag,
                        attempt,
                        max_attempts,
                        transformed,
                    ):
                        current_token = str(transformed.get("token") or "")
                        if current_token:
                            excluded_tokens.add(current_token)
                            await self.mark_token_failure(
                                current_token,
                                Exception(error_message or "上游认证会话不可用"),
                            )
                            self.logger.warning(
                                "认证会话不可用，准备切换认证 Token/回退匿名池: %s...",
                                current_token[:20],
                            )
                        transformed = await self._refresh_authenticated_request(
                            request,
                            attempt,
                            excluded_tokens,
                            excluded_guest_user_ids,
                        )
                        continue

                    if not response.is_success:
                        error_msg = f"上游 API 错误: {response.status_code}"
                        if not self._is_guest_auth(transformed):
                            current_token = str(transformed.get("token") or "")
                            if current_token:
                                await self.mark_token_failure(
                                    current_token,
                                    Exception(error_message or error_msg),
                                )
                        await self._release_guest_session(transformed)
                        self.logger.error(
                            "%s",
                            normalize_httpx_response(
                                response.status_code,
                                response_text,
                                content_type=content_type,
                                method="POST",
                                url=transformed["url"],
                                context="upstream.non_stream.response",
                            ),
                        )
                        return (
                            self._build_upstream_error_response(
                                response.status_code,
                                error_code,
                                error_message or error_msg,
                            ),
                            str(transformed.get("token") or "") or None,
                        )

                    try:
                        result = await self._response_handler.handle_non_stream_response(
                            response,
                            transformed["chat_id"],
                            transformed["model"],
                        )
                    finally:
                        await self._release_guest_session(transformed)

                    current_token = str(transformed.get("token") or "")

                    # 空回检测：上游返回 200 但 choices 无实际内容
                    if not _openai_response_has_output(result):
                        self.logger.warning(
                            "空回响应 (attempt %s/%s, empty_retry %s/%s), token: %s...",
                            attempt + 1,
                            max_attempts,
                            empty_retries + 1,
                            settings.EMPTY_RESPONSE_MAX_RETRIES,
                            current_token[:20] if current_token else "guest",
                        )
                        await write_request_log(
                            provider="zai",
                            model=str(transformed.get("model") or ""),
                            source_info=RequestSourceInfo(
                                source="upstream", protocol="openai",
                                client_name="retry", endpoint="/v1/chat/completions",
                                user_agent="",
                            ),
                            auth_token=None,
                            upstream_auth_token=current_token or None,
                            success=False,
                            started_at=time.perf_counter(),
                            status_code=200,
                            error_message="Empty response: no content in choices",
                        )
                        if self._is_guest_auth(transformed):
                            guest_user_id = str(
                                transformed.get("guest_user_id")
                                or transformed.get("user_id")
                                or ""
                            )
                            if guest_user_id:
                                excluded_guest_user_ids.add(guest_user_id)
                        elif current_token:
                            excluded_tokens.add(current_token)
                            await self.mark_token_failure(
                                current_token,
                                Exception("Empty response: no content in choices"),
                            )
                        if attempt + 1 < max_attempts and empty_retries < settings.EMPTY_RESPONSE_MAX_RETRIES:
                            empty_retries += 1
                            if self._is_guest_auth(transformed):
                                transformed = await self._refresh_guest_request(
                                    request,
                                    attempt,
                                    excluded_tokens,
                                    excluded_guest_user_ids,
                                    transformed,
                                )
                            else:
                                transformed = await self._refresh_authenticated_request(
                                    request,
                                    attempt,
                                    excluded_tokens,
                                    excluded_guest_user_ids,
                                )
                            continue
                        # 所有重试已耗尽
                        return result, current_token or None

                    await self._commit_session_if_needed(transformed)

                    if not self._is_guest_auth(transformed):
                        if current_token:
                            token_pool = get_token_pool()
                            if token_pool:
                                await token_pool.record_token_success(current_token)

                    return result, current_token or None

        except Exception as e:
            self.logger.error(
                "%s 响应失败: %s",
                self.name,
                get_error_message(e),
                exc_info=settings.DEBUG_LOGGING,
            )
            try:
                await self._release_authenticated_token_allocation(transformed)
                await self._release_guest_session(transformed)
            except Exception:
                pass
            return handle_error(e, "请求处理"), None

    async def _create_stream_response(
        self,
        request: OpenAIRequest,
        transformed: Dict[str, Any],
        *,
        http_request=None,
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """创建流式响应，并在首包前支持双池重试。

        Args:
            request: OpenAI 请求对象。
            transformed: 已转换的上游请求参数字典。
            http_request: FastAPI Request 对象，用于检测客户端断开。
        """
        max_attempts = await self._get_total_retry_limit()
        excluded_tokens: Set[str] = set()
        excluded_guest_user_ids: Set[str] = set()
        current_token = str(transformed.get("token") or "")
        empty_retries = 0
        captcha_retries = 0

        client = self._get_shared_stream_client()
        loop = asyncio.get_running_loop()
        stream_deadline = loop.time() + settings.HTTP_STREAM_TOTAL_TIMEOUT

        def _remaining_timeout() -> float:
            return stream_deadline - loop.time()

        for attempt in range(max_attempts):
            if _remaining_timeout() <= 0:
                await self._release_authenticated_token_allocation(transformed)
                await self._release_guest_session(transformed)
                return {
                    "error": {
                        "message": "流式请求总超时，请重试。",
                        "type": "stream_timeout",
                        "code": 504,
                    }
                }

            self.logger.debug("发送请求到上游: %s", transformed["url"])
            req = client.build_request(
                "POST",
                transformed["url"],
                json=transformed["body"],
                headers=transformed["headers"],
            )

            try:
                response = await asyncio.wait_for(
                    client.send(req, stream=True),
                    timeout=max(0.1, _remaining_timeout()),
                )
            except asyncio.TimeoutError as e:
                self.logger.error(
                    "%s",
                    normalize_httpx_exception(
                        e,
                        method="POST",
                        url=transformed["url"],
                        context="upstream.stream.connect",
                    ),
                )
                if self._is_guest_auth(transformed):
                    await self._release_guest_session(transformed)
                elif current_token:
                    await self.mark_token_failure(current_token, e)
                return {
                    "error": {
                        "message": "上游连接超时，请重试。",
                        "type": "stream_timeout",
                        "code": 504,
                    }
                }
            except Exception as e:
                friendly_msg = get_error_message(e)
                self.logger.error(
                    "%s (raw: %s)",
                    normalize_httpx_exception(
                        e,
                        method="POST",
                        url=transformed["url"],
                        context="upstream.stream.connect",
                    ),
                    e,
                )
                if self._is_guest_auth(transformed):
                    await self._release_guest_session(transformed)
                elif current_token:
                    await self.mark_token_failure(current_token, e)
                return {
                    "error": {
                        "message": f"上游连接异常: {friendly_msg}",
                        "type": "stream_error",
                    }
                }

            try:
                error_text = b""
                content_type = response.headers.get("content-type")
                should_read_error_body = (
                    response.status_code != 200
                    or is_upstream_page_response(content_type, "")
                )
                if should_read_error_body:
                    error_text = await asyncio.wait_for(
                        response.aread(),
                        timeout=max(0.1, _remaining_timeout()),
                    )
                error_msg = error_text.decode("utf-8", errors="ignore")
                is_page_error = is_upstream_page_response(content_type, error_msg)
                error_code, parsed_error_message = (
                    extract_upstream_error_details(
                        response.status_code,
                        error_msg,
                        content_type=content_type,
                    )
                    if should_read_error_body
                    else (None, "")
                )
                is_concurrency_limited_flag = (
                    not is_page_error
                    and is_concurrency_limited(
                        response.status_code,
                        error_code,
                        parsed_error_message,
                    )
                )

                if (
                    settings.CAPTCHA_ENABLED
                    and parsed_error_message
                    and "FRONTEND_CAPTCHA_REQUIRED" in str(parsed_error_message)
                    and captcha_retries < settings.CAPTCHA_MAX_RETRIES
                ):
                    captcha_retries += 1
                    await response.aclose()
                    if await self._try_get_captcha_token(transformed):
                        self.logger.warning(
                            "captcha required, retrying with fresh token "
                            "(captcha_retry %s/%s)",
                            captcha_retries,
                            settings.CAPTCHA_MAX_RETRIES,
                        )
                        continue

                if is_page_error:
                    await response.aclose()
                    await self._release_authenticated_token_allocation(
                        transformed
                    )
                    self.logger.error(
                        "%s",
                        normalize_httpx_response(
                            response.status_code,
                            error_msg,
                            content_type=content_type,
                            method="POST",
                            url=transformed["url"],
                            context="upstream.stream.page",
                        ),
                    )
                    await self._release_guest_session(transformed)
                    return self._build_upstream_error_response(
                        response.status_code,
                        error_code,
                        parsed_error_message,
                        is_page_error=True,
                    )

                if self._should_retry_guest_session(
                    response.status_code,
                    is_concurrency_limited_flag,
                    attempt,
                    max_attempts,
                    transformed,
                ):
                    await response.aclose()
                    guest_user_id = str(
                        transformed.get("guest_user_id")
                        or transformed.get("user_id")
                        or ""
                    )
                    if guest_user_id:
                        excluded_guest_user_ids.add(guest_user_id)
                    transformed = await self._refresh_guest_request(
                        request,
                        attempt,
                        excluded_tokens,
                        excluded_guest_user_ids,
                        transformed,
                        is_concurrency_limited=is_concurrency_limited_flag,
                    )
                    current_token = str(transformed.get("token") or "")
                    continue

                if self._should_retry_authenticated_session(
                    response.status_code,
                    is_concurrency_limited_flag,
                    attempt,
                    max_attempts,
                    transformed,
                ):
                    await response.aclose()
                    if current_token:
                        excluded_tokens.add(current_token)
                        await self.mark_token_failure(
                            current_token,
                            Exception(
                                parsed_error_message or "上游认证会话不可用"
                            ),
                        )
                        self.logger.warning(
                            "流式请求命中认证会话限制，准备切号/回退匿名池: %s...",
                            current_token[:20],
                        )
                    transformed = await self._refresh_authenticated_request(
                        request,
                        attempt,
                        excluded_tokens,
                        excluded_guest_user_ids,
                    )
                    current_token = str(transformed.get("token") or "")
                    continue

                if response.status_code != 200:
                    await response.aclose()
                    self.logger.error(
                        "%s",
                        normalize_httpx_response(
                            response.status_code,
                            error_msg,
                            content_type=content_type,
                            method="POST",
                            url=transformed["url"],
                            context="upstream.stream.response",
                        ),
                    )

                    if not self._is_guest_auth(transformed) and current_token:
                        await self.mark_token_failure(
                            current_token,
                            Exception(
                                parsed_error_message
                                or f"Upstream error: {response.status_code}"
                            ),
                        )
                    await self._release_guest_session(transformed)
                    return self._build_upstream_error_response(
                        response.status_code,
                        error_code,
                        parsed_error_message,
                    )

                chat_id = transformed["chat_id"]
                model = transformed["model"]

                # Peek at stream to detect empty response before committing
                response_iter = self._response_handler.handle_stream_response(
                    response,
                    chat_id,
                    model,
                    started_at=getattr(request, "started_at", None),
                )
                buffer: list[str] = []
                found_output = False
                try:
                    remaining = _remaining_timeout()
                    if remaining > 0:
                        async with asyncio.timeout(max(0.1, remaining)):
                            async for chunk in response_iter:
                                buffer.append(chunk)
                                if _sse_chunk_has_output(chunk):
                                    found_output = True
                                    break
                except (asyncio.TimeoutError, Exception):
                    pass

                if not found_output:
                    await response.aclose()
                    self.logger.warning(
                        "流式空回响应 (attempt %s/%s, empty_retry %s/%s), token: %s...",
                        attempt + 1, max_attempts,
                        empty_retries + 1, settings.EMPTY_RESPONSE_MAX_RETRIES,
                        current_token[:20] if current_token else "guest",
                    )
                    await write_request_log(
                        provider="zai",
                        model=str(transformed.get("model") or ""),
                        source_info=RequestSourceInfo(
                            source="upstream", protocol="openai",
                            client_name="retry", endpoint="/v1/chat/completions",
                            user_agent="",
                        ),
                        auth_token=None,
                        upstream_auth_token=current_token or None,
                        success=False,
                        started_at=time.perf_counter(),
                        status_code=200,
                        error_message="Empty stream: no content in response",
                    )
                    if self._is_guest_auth(transformed):
                        guest_user_id = str(
                            transformed.get("guest_user_id")
                            or transformed.get("user_id")
                            or ""
                        )
                        if guest_user_id:
                            excluded_guest_user_ids.add(guest_user_id)
                    elif current_token:
                        excluded_tokens.add(current_token)
                        await self.mark_token_failure(
                            current_token,
                            Exception("Empty stream: no content"),
                        )
                    await self._release_guest_session(transformed)
                    if attempt + 1 < max_attempts and empty_retries < settings.EMPTY_RESPONSE_MAX_RETRIES:
                        empty_retries += 1
                        if self._is_guest_auth(transformed):
                            transformed = await self._refresh_guest_request(
                                request, attempt, excluded_tokens,
                                excluded_guest_user_ids, transformed,
                            )
                        else:
                            transformed = await self._refresh_authenticated_request(
                                request, attempt, excluded_tokens,
                                excluded_guest_user_ids,
                            )
                        current_token = str(transformed.get("token") or "")
                        continue
                    return {
                        "error": {
                            "message": "Empty stream: no content in response",
                            "type": "stream_error",
                            "code": 502,
                        }
                    }

                await self._commit_session_if_needed(transformed)

                async def stream_generator() -> AsyncGenerator[str, None]:
                    success = False
                    disconnect_task: Optional[asyncio.Task] = None

                    async def _wait_for_disconnect() -> None:
                        """后台任务：每 0.5s 轮询一次，检测到客户端断开后
                        立即关闭上游 HTTP 响应，使 aiter_lines() 退出。"""
                        try:
                            while True:
                                if await http_request.is_disconnected():
                                    self.logger.debug(
                                        "[stream] client disconnected, closing upstream stream (chat_id=%s)",
                                        chat_id,
                                    )
                                    await response.aclose()
                                    return
                                await asyncio.sleep(0.5)
                        except asyncio.CancelledError:
                            pass

                    try:
                        if http_request is not None:
                            disconnect_task = asyncio.create_task(
                                _wait_for_disconnect()
                            )

                        remaining = _remaining_timeout()
                        if remaining <= 0:
                            raise asyncio.TimeoutError(
                                "stream total timeout before consume"
                            )

                        async with asyncio.timeout(remaining):
                            for chunk in buffer:
                                yield chunk
                            async for chunk in response_iter:
                                yield chunk
                        success = True
                    except asyncio.TimeoutError as e:
                        self.logger.error("流处理超时: %s", e)
                        if not self._is_guest_auth(transformed) and current_token:
                            await self.mark_token_failure(current_token, e)
                        error_response = {
                            "error": {
                                "message": "流处理超时，请重试。",
                                "type": "stream_timeout",
                                "code": 504,
                            }
                        }
                        yield f"data: {json.dumps(error_response)}\n\n"
                        yield "data: [DONE]\n\n"
                    except asyncio.CancelledError:
                        self.logger.debug(
                            "[stream] stream task cancelled (chat_id=%s)",
                            chat_id,
                        )
                        if (
                            not self._is_guest_auth(transformed)
                            and current_token
                        ):
                            await self._release_authenticated_token_allocation(
                                transformed
                            )
                    except Exception as e:
                        friendly_msg = get_error_message(e)
                        self.logger.error(
                            "流处理错误: %s (raw: %s)",
                            friendly_msg,
                            e,
                        )
                        if not self._is_guest_auth(transformed) and current_token:
                            await self.mark_token_failure(current_token, e)
                        error_response = {
                            "error": {
                                "message": f"流处理错误: {friendly_msg}",
                                "type": "stream_error",
                            }
                        }
                        yield f"data: {json.dumps(error_response)}\n\n"
                        yield "data: [DONE]\n\n"
                    finally:
                        if disconnect_task is not None and not disconnect_task.done():
                            disconnect_task.cancel()
                            try:
                                await disconnect_task
                            except asyncio.CancelledError:
                                pass
                        await response.aclose()
                        await self._release_guest_session(transformed)
                        if (
                            success
                            and not self._is_guest_auth(transformed)
                            and current_token
                        ):
                            token_pool = get_token_pool()
                            if token_pool:
                                await token_pool.record_token_success(current_token)

                return stream_generator()

            except asyncio.TimeoutError as e:
                await response.aclose()
                self.logger.error("流处理超时: %s", e)
                if self._is_guest_auth(transformed):
                    await self._release_guest_session(transformed)
                elif current_token:
                    await self.mark_token_failure(current_token, e)
                return {
                    "error": {
                        "message": "流处理超时，请重试。",
                        "type": "stream_timeout",
                        "code": 504,
                    }
                }
            except Exception as e:
                await response.aclose()
                friendly_msg = get_error_message(e)
                self.logger.error(
                    "流处理错误: %s (raw: %s)",
                    friendly_msg,
                    e,
                )
                if self._is_guest_auth(transformed):
                    await self._release_guest_session(transformed)
                elif current_token:
                    await self.mark_token_failure(current_token, e)

                return {
                    "error": {
                        "message": f"流处理错误: {friendly_msg}",
                        "type": "stream_error",
                    }
                }

        await self._release_authenticated_token_allocation(transformed)
        await self._release_guest_session(transformed)
        return {
            "error": {
                "message": "Max retry attempts exhausted.",
                "type": "stream_error",
                "code": 500
            }
        }
