#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""上游适配器。

UpstreamClient 作为薄集成层，组合各子模块完成完整的请求/响应处理流程：
- http_client.py     — HTTP 客户端管理
- headers.py         — 动态浏览器 headers
- models.py          — 模型映射与特性解析
- toolify/message.py  — 消息预处理
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
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Union

import httpx

from app.core.config import settings
from app.core.http_client import SharedHttpClients
from app.core.headers import build_dynamic_headers
from app.core.models import ModelManager
from app.utils.jwt_utils import extract_user_id_from_token
from app.core.toolify.message import extract_last_user_text
from app.core.request_signing import (
    process_multimodal_messages,
    build_upstream_body,
    sign_request,
)
from app.core.retry_policy import (
    RetryPolicy,
    extract_upstream_error_details,
    is_concurrency_limited,
)
from app.core.response_handler import ResponseHandler
from app.core.toolify import ToolifyRequestHandler
from app.core.session import SessionManager
from app.core.session.session_content import (
    build_session_body_messages,
    extract_turn_content,
    get_precreate_content,
    inject_system_prompt,
    resolve_trigger_signal,
)
from app.core.file_upload import upload_file as _upload_file
from app.core.openai_compat import (
    get_error_message,
    handle_error,
)
from app.models.schemas import OpenAIRequest
from app.utils.fe_version import get_latest_fe_version
from app.utils.logger import get_logger
from app.utils.token_pool import get_token_pool
from app.utils.guest_session_pool import get_guest_session_pool

logger = get_logger()


def generate_uuid() -> str:
    """生成UUID v4"""
    return str(uuid.uuid4())



# --------------------------------------------------------------------------
# UpstreamClient
# --------------------------------------------------------------------------

class UpstreamClient:
    """当前服务使用的上游适配器（薄集成层）。"""

    def __init__(self):
        self.name = "upstream"
        self.logger = logger
        self.api_endpoint = settings.API_ENDPOINT

        # 当前上游特定配置
        self.base_url = "https://chat.z.ai"
        self.auth_url = f"{self.base_url}/api/v1/auths/"

        # 子模块
        self._http_clients = SharedHttpClients()
        self._model_manager = ModelManager()
        self._retry_policy = RetryPolicy()
        self._response_handler = ResponseHandler()
        self._toolify_request_handler = ToolifyRequestHandler()
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
            "🔄 匿名会话不可用，正在切换匿名会话并进行第 "
            f"{retry_number} 次请求"
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
            "🔄 检测到认证会话不可用，正在切换认证 Token/回退匿名池并进行第 "
            f"{retry_number} 次请求"
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

            self.logger.warning("⚠️ 未知会话提交动作: {}", action)
        except Exception as exc:
            self.logger.warning("⚠️ 提交会话状态失败: {}", exc)

    # ------------------------------------------------------------------
    # 错误解析（委托到 retry_policy 工具函数）
    # ------------------------------------------------------------------

    def _extract_upstream_error_details(
        self,
        status_code: int,
        error_text: str,
    ):
        """解析上游错误响应中的 code/message。"""
        return extract_upstream_error_details(status_code, error_text)

    def _is_concurrency_limited(
        self,
        status_code: int,
        error_code,
        error_message: str,
    ) -> bool:
        """判断是否为上游并发限制/429 场景。"""
        return is_concurrency_limited(status_code, error_code, error_message)

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
                            "[chat] pre-created chat_id={}", chat_id
                        )
                        return chat_id
                self.logger.warning(
                    "[chat] pre-create failed: HTTP {} (attempt {}/{})",
                    resp.status_code,
                    attempt + 1,
                    attempts,
                )
            except Exception as e:
                self.logger.warning(
                    "[chat] pre-create error: {} (attempt {}/{})",
                    e,
                    attempt + 1,
                    attempts,
                )

            if attempt + 1 < attempts:
                await asyncio.sleep(min(2 ** attempt, 8))

        if allow_fallback:
            # 降级：使用随机 chat_id（会触发 done 阶段 INTERNAL_ERROR，但内容不受影响）
            self.logger.warning(
                "[chat] pre-create failed after {} attempts, fallback to random chat_id",
                attempts,
            )
            return generate_uuid()

        raise RuntimeError(
            f"pre-create chat_id failed after {attempts} attempts"
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
                        f"✅ 在线模型同步成功，共获取 {len(parsed_models)} 个模型"
                    )
                    return parsed_models
                else:
                    self.logger.warning("获取在线模型失败，状态码: {}", response.status_code)
            except Exception as exc:
                self.logger.warning("获取在线模型异常: {}", exc)

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
            self.logger.warning("在线模型缓存写入数据库失败: {}", exc)

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
            self.logger.info(
                f"从数据库缓存加载 {len(models)} 个在线模型，"
                f"生成 {len(self._model_manager.get_supported_models())} 个变体"
            )
            return True
        except Exception as exc:
            self.logger.warning("从数据库加载在线模型缓存失败: {}", exc)
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
                    f"尝试获取访客令牌 (第{retry_count + 1}次): {self.auth_url}"
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
                            f"✅ 直连获取匿名令牌成功: {token[:20]}..."
                        )
                        return {
                            "token": token,
                            "user_id": user_id,
                            "username": username or "Guest",
                            "auth_mode": "guest",
                            "token_source": "guest_direct",
                            "guest_user_id": user_id,
                        }

                    self.logger.warning("响应中未找到 token 字段: {}", data)
                elif response.status_code == 405:
                    self.logger.error(
                        "🚫 请求被 WAF 拦截 (405)，无法直连获取匿名令牌"
                    )
                    break
                else:
                    self.logger.warning(
                        f"直连获取匿名令牌失败，状态码: {response.status_code}"
                    )
            except httpx.TimeoutException as exc:
                self.logger.warning(
                    f"直连获取匿名令牌超时 (第{retry_count + 1}次): {exc}"
                )
            except httpx.ConnectError as exc:
                self.logger.warning(
                    f"直连获取匿名令牌连接错误 (第{retry_count + 1}次): {exc}"
                )
            except json.JSONDecodeError as exc:
                self.logger.warning(
                    f"直连获取匿名令牌 JSON 解析错误 (第{retry_count + 1}次): {exc}"
                )
            except Exception as exc:
                self.logger.warning(
                    f"直连获取匿名令牌失败 (第{retry_count + 1}次): {exc}"
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
                self.logger.debug("从认证号池获取令牌: {}...", token[:20])
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
                        "🫥 认证池不可用，回退匿名会话池: "
                        f"user_id={session.user_id}"
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
                    self.logger.warning("匿名会话池获取失败，转为直连访客鉴权: {}", exc)

            return await self._fetch_direct_guest_auth()

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
        self.logger.debug("🔄 转换 OpenAI 请求到上游格式: {}", request.model)

        auth_info: Optional[Dict[str, Any]] = None  # 用于 finally 中 guest session 清理

        raw_messages = [
            message.model_dump(exclude_none=True) for message in request.messages
        ]

        tool_choice = getattr(request, "tool_choice", None)
        toolify_prepared = self._toolify_request_handler.prepare(
            raw_messages=raw_messages,
            request_tools=request.tools,
            tool_choice=tool_choice,
        )
        tool_strategy = toolify_prepared.tool_strategy
        tools = toolify_prepared.tools
        trigger_signal = toolify_prepared.trigger_signal
        normalized_messages = toolify_prepared.normalized_messages
        session_commit: Optional[Dict[str, Any]] = None

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
                "Resolved model features for {}: {}, temperature={}, max_tokens={}",
                request.model,
                features,
                request.temperature,
                request.max_tokens,
            )

            # ── Session mode vs Direct mode ──
            if settings.SESSION_ENABLED:
                # Try to find an existing continuous session
                session_result = await self._session_manager.find_session(
                    model=request.model,
                    messages=raw_messages,
                )

                # Token 亲和性：待确认会话（首轮重试）如果 token 已切换，
                # 则原 chat_id 不可用，需重新 precreate。
                if (
                    session_result
                    and not session_result.parent_id
                    and session_result.bound_token
                    and session_result.bound_token != token
                ):
                    self.logger.debug(
                        "⚠️ Token 已切换，跳过待确认会话 chat_id={}",
                        session_result.chat_id[:8],
                    )
                    session_result = None

                if session_result:
                    # Continuous session: reuse chat_id, chain via parent_id
                    chat_id = session_result.chat_id
                    # ── Token 亲和性：链式会话强制使用创建时的 token ──
                    if session_result.bound_token and session_result.bound_token != token:
                        self.logger.debug(
                            "🔒 链式会话 token 亲和: 切换到 bound_token (chat_id={})",
                            chat_id[:8],
                        )
                        await self._release_guest_session(auth_info)
                        token = session_result.bound_token
                        user_id = extract_user_id_from_token(token)
                    user_msg_id = session_result.message_id
                    message_id = generate_uuid()
                    parent_id = session_result.parent_id
                    is_first_turn_retry = not parent_id
                    if tools and tool_strategy == "xmlfc":
                        trigger_signal = resolve_trigger_signal(
                            session_result, trigger_signal, self.logger,
                        )
                    self.logger.debug(
                        "♻️ {} chat_id={}, parent_id={}",
                        "首轮重试复用" if is_first_turn_retry else "复用会话",
                        chat_id[:8],
                        parent_id[:8] if parent_id else "None",
                    )
                    session_body_messages = build_session_body_messages(
                        normalized_messages=normalized_messages,
                        session_turn_content=session_turn_content,
                        is_new_session=is_first_turn_retry,
                        inject_system=settings.SESSION_SYSTEM_INJECT,
                    )
                    session_commit = {
                        "action": "reuse",
                        "model": request.model,
                        "messages": raw_messages,
                        "chat_id": chat_id,
                        "message_id": user_msg_id,
                    }
                else:
                    # New session: precreate chat on upstream
                    message_id = generate_uuid()
                    user_msg_id = generate_uuid()
                    session_body_messages = build_session_body_messages(
                        normalized_messages=normalized_messages,
                        session_turn_content=session_turn_content,
                        is_new_session=True,
                        inject_system=settings.SESSION_SYSTEM_INJECT,
                    )
                    precreate_content = get_precreate_content(session_body_messages)
                    chat_id = await self._precreate_chat(
                        token=token,
                        fe_version=fe_version,
                        model=features["upstream_model_id"],
                        user_msg_id=user_msg_id,
                        content=precreate_content,
                        enable_thinking=features["enable_thinking"],
                        auto_web_search=features["auto_web_search"],
                        mcp_servers=features.get("mcp_servers", []),
                        allow_fallback=False,
                    )
                    parent_id = None
                    # 立即注册待确认会话（last_message_id 为空），
                    # 使重试时 find_session 能复用此 chat_id，避免重复 precreate。
                    await self._session_manager.create_session(
                        auth_token=token,
                        model=request.model,
                        messages=raw_messages,
                        chat_id=chat_id,
                        message_id="",
                        trigger_signal=trigger_signal if tools and tool_strategy == "xmlfc" else "",
                    )
                    session_commit = {
                        "action": "reuse",
                        "model": request.model,
                        "messages": raw_messages,
                        "chat_id": chat_id,
                        "message_id": user_msg_id,
                    }

                # Session mode: send prepared messages
                messages, files = await process_multimodal_messages(
                    normalized_messages=session_body_messages,
                    token=token,
                    user_id=user_id,
                    chat_id=chat_id,
                    auth_mode=auth_mode,
                    http_client=self._get_shared_client(),
                    base_url=self.base_url,
                )
            else:
                # Direct mode: no session management
                message_id = generate_uuid()
                user_msg_id = generate_uuid()
                parent_id = None
                if settings.PRECREATE_CHAT:
                    precreate_content = get_precreate_content(
                        inject_system_prompt(normalized_messages)
                        if settings.SESSION_SYSTEM_INJECT
                        else normalized_messages
                    )
                    chat_id = await self._precreate_chat(
                        token=token,
                        fe_version=fe_version,
                        model=features["upstream_model_id"],
                        user_msg_id=user_msg_id,
                        content=precreate_content,
                        enable_thinking=features["enable_thinking"],
                        auto_web_search=features["auto_web_search"],
                        mcp_servers=features.get("mcp_servers", []),
                        allow_fallback=False,
                    )
                else:
                    chat_id = generate_uuid()
                direct_messages = (
                    inject_system_prompt(normalized_messages)
                    if settings.SESSION_SYSTEM_INJECT
                    else normalized_messages
                )
                messages, files = await process_multimodal_messages(
                    normalized_messages=direct_messages,
                    token=token,
                    user_id=user_id,
                    chat_id=chat_id,
                    auth_mode=auth_mode,
                    http_client=self._get_shared_client(),
                    base_url=self.base_url,
                )

            if tools:
                self.logger.debug(
                    "工具定义: {} 个工具, 策略: {}",
                    len(tools),
                    tool_strategy,
                )

            # 构建请求体
            passthrough_tools = tools if tool_strategy in ("native", "glmnative") else None
            passthrough_tool_choice = tool_choice if tool_strategy in ("native", "glmnative") else None
            body = build_upstream_body(
                messages=messages,
                files=files,
                upstream_model_id=features["upstream_model_id"],
                last_user_text=last_user_text,
                chat_id=chat_id,
                message_id=message_id,
                enable_thinking=features["enable_thinking"],
                web_search=features["web_search"],
                auto_web_search=features["auto_web_search"],
                flags=features["flags"],
                extra=features["extra"],
                mcp_servers=features["mcp_servers"],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                parent_message_id=parent_id,
                tools=passthrough_tools,
                tool_choice=passthrough_tool_choice,
            )
            # 对齐浏览器：current_user_message_id 使用和 chats/new 一致的 ID
            body["current_user_message_id"] = user_msg_id
            # DEBUG 日志脱敏：仅记录 body 结构，不记录消息内容
            if settings.DEBUG_LOGGING:
                sanitized = {
                    k: (f"[{len(v)} messages]" if k == "messages" else v)
                    for k, v in body.items()
                }
                self.logger.debug("Upstream request body (sanitized): {}", sanitized)

            # 签名并生成最终 URL 和 headers（复用已并行拉取的 fe_version）
            signed_url, headers, _fe_version = await sign_request(
                api_endpoint=self.api_endpoint,
                user_id=user_id,
                last_user_text=last_user_text,
                chat_id=chat_id,
                token=token,
                fe_version=fe_version,
            )

        except Exception:
            # 签名/构建失败时归还 guest session，避免永久占用
            await self._release_guest_session(auth_info)
            raise

        return {
            "url": signed_url,
            "headers": headers,
            "body": body,
            "token": token,
            "chat_id": chat_id,
            "model": request.model,
            "user_id": user_id,
            "auth_mode": auth_mode,
            "token_source": token_source,
            "guest_user_id": guest_user_id,
            "trigger_signal": trigger_signal,
            "tools": tools,
            "tool_strategy": tool_strategy,
            "session_commit": session_commit,
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
        self.logger.debug("🔄 {} 处理请求: {}", self.name, request.model)
        self.logger.debug("  消息数量: {}", len(request.messages))
        self.logger.debug("  流式模式: {}", request.stream)

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

            async with asyncio.timeout(settings.CHAT_TOTAL_TIMEOUT):
                for attempt in range(max_attempts):
                    response = await client.post(
                        transformed["url"],
                        headers=transformed["headers"],
                        json=transformed["body"],
                    )

                    error_code, error_message = extract_upstream_error_details(
                        response.status_code,
                        response.text,
                    )
                    is_concurrency_limited_flag = is_concurrency_limited(
                        response.status_code,
                        error_code,
                        error_message,
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
                                "⚠️ 认证会话不可用，准备切换认证 Token/回退匿名池: {}...",
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
                        self.logger.error("❌ {} 响应失败: {}", self.name, error_msg)
                        return (
                            handle_error(Exception(error_message or error_msg)),
                            str(transformed.get("token") or "") or None,
                        )

                    await self._commit_session_if_needed(transformed)
                    try:
                        result = await self.transform_response(
                            response, request, transformed
                        )
                    finally:
                        await self._release_guest_session(transformed)

                    if not self._is_guest_auth(transformed):
                        current_token = str(transformed.get("token") or "")
                        if current_token:
                            token_pool = get_token_pool()
                            if token_pool:
                                await token_pool.record_token_success(current_token)

                    return result, str(transformed.get("token") or "") or None

        except Exception as e:
            self.logger.error("❌ {} 响应失败: {}", self.name, str(e))
            try:
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

        client = self._get_shared_stream_client()
        loop = asyncio.get_running_loop()
        stream_deadline = loop.time() + settings.HTTP_STREAM_TOTAL_TIMEOUT

        def _remaining_timeout() -> float:
            return stream_deadline - loop.time()

        for attempt in range(max_attempts):
            if _remaining_timeout() <= 0:
                await self._release_guest_session(transformed)
                return {
                    "error": {
                        "message": "流式请求总超时，请重试。",
                        "type": "stream_timeout",
                        "code": 504,
                    }
                }

            self.logger.debug("🎯 发送请求到上游: {}", transformed["url"])
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
                self.logger.error("❌ 上游连接超时: {}", e)
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
                    "❌ 上游连接异常: {} (raw: {})",
                    friendly_msg,
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
                if response.status_code != 200:
                    error_text = await asyncio.wait_for(
                        response.aread(),
                        timeout=max(0.1, _remaining_timeout()),
                    )
                error_msg = error_text.decode("utf-8", errors="ignore")
                error_code, parsed_error_message = (
                    extract_upstream_error_details(
                        response.status_code,
                        error_msg,
                    )
                    if response.status_code != 200
                    else (None, "")
                )
                is_concurrency_limited_flag = is_concurrency_limited(
                    response.status_code,
                    error_code,
                    parsed_error_message,
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
                            "⚠️ 流式请求命中认证会话限制，准备切号/回退匿名池: {}...",
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
                    self.logger.error("❌ 上游返回错误: {}", response.status_code)
                    if error_msg:
                        self.logger.error("❌ 错误详情: {}", error_msg)

                    if not self._is_guest_auth(transformed) and current_token:
                        await self.mark_token_failure(
                            current_token,
                            Exception(
                                parsed_error_message
                                or f"Upstream error: {response.status_code}"
                            ),
                        )
                    await self._release_guest_session(transformed)

                    if response.status_code == 405:
                        self.logger.error(
                            "🚫 请求被上游 WAF 拦截，可能是请求头或签名异常"
                        )
                        return {
                            "error": {
                                "message": (
                                    "请求被上游WAF拦截(405 Method Not Allowed),"
                                    "可能是请求头或签名异常,请稍后重试..."
                                ),
                                "type": "waf_blocked",
                                "code": 405,
                            }
                        }
                    return {
                        "error": {
                            "message": parsed_error_message
                            or f"Upstream error: {response.status_code}",
                            "type": "upstream_error",
                            "code": error_code or response.status_code,
                        }
                    }

                await self._commit_session_if_needed(transformed)
                chat_id = transformed["chat_id"]
                model = transformed["model"]

                async def stream_generator() -> AsyncGenerator[str, None]:
                    success = False
                    disconnect_task: Optional[asyncio.Task] = None

                    async def _wait_for_disconnect() -> None:
                        """后台任务：每 0.5s 轮询一次，检测到客户端断开后
                        立即关闭上游 HTTP 响应，使 aiter_lines() 退出。"""
                        try:
                            while True:
                                if await http_request.is_disconnected():
                                    self.logger.info(
                                        "[stream] client disconnected, closing upstream stream (chat_id={})",
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
                            async for chunk in self._response_handler.handle_stream_response(
                                response,
                                chat_id,
                                model,
                                request,
                                transformed,
                            ):
                                yield chunk
                        success = True
                    except asyncio.TimeoutError as e:
                        self.logger.error("❌ 流处理超时: {}", e)
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
                        self.logger.info(
                            "[stream] stream task cancelled (chat_id={})",
                            chat_id,
                        )
                    except Exception as e:
                        friendly_msg = get_error_message(e)
                        self.logger.error(
                            "❌ 流处理错误: {} (raw: {})",
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
                self.logger.error("❌ 流处理超时: {}", e)
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
                    "❌ 流处理错误: {} (raw: {})",
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

        await self._release_guest_session(transformed)
        return {
            "error": {
                "message": "Max retry attempts exhausted.",
                "type": "stream_error",
                "code": 500
            }
        }

    async def transform_response(
        self,
        response: httpx.Response,
        request: OpenAIRequest,
        transformed: Dict[str, Any],
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """转换上游响应为 OpenAI 格式。"""
        chat_id = transformed["chat_id"]
        model = transformed["model"]

        if request.stream:
            return self._response_handler.handle_stream_response(
                response, chat_id, model, request, transformed
            )
        else:
            return await self._response_handler.handle_non_stream_response(
                response, chat_id, model,
                trigger_signal=transformed.get("trigger_signal", ""),
                tools_defs=transformed.get("tools"),
                tool_strategy=transformed.get("tool_strategy", "xmlfc"),
            )
