#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""上游适配器。

UpstreamClient 作为薄集成层，组合各子模块完成完整的请求/响应处理流程：
- http_client.py     — HTTP 客户端管理
- headers.py         — 动态浏览器 headers
- models.py          — 模型映射与特性解析
- message_preprocessing.py — 消息预处理与 JWT 工具
- request_signing.py — 请求体构建与签名
- retry_policy.py    — 双池重试策略
- response_handler.py — 流式/非流式响应处理
- file_upload.py     — 文件上传

所有公有方法签名与原实现完全一致。
"""

import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Union

import httpx

from app.core.config import settings
from app.core.http_client import SharedHttpClients
from app.core.headers import build_dynamic_headers
from app.core.models import ModelManager
from app.core.message_preprocessing import (
    preprocess_openai_messages,
    extract_user_id_from_token,
    extract_last_user_text,
)
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
from app.core.file_upload import upload_file as _upload_file
from app.core.openai_compat import (
    create_openai_chunk,
    create_openai_response_with_reasoning,
    format_sse_chunk,
    handle_error,
)
from app.utils.tool_call_handler import (
    generate_trigger_signal,
    process_messages_with_tools,
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
# 模块级工具函数（向后兼容：原先是模块级函数，保留对外可见性）
# --------------------------------------------------------------------------

def get_dynamic_headers(fe_version: str, chat_id: str = "") -> Dict[str, str]:
    """生成上游请求所需的动态浏览器 headers。（委托到 headers.py）"""
    return build_dynamic_headers(fe_version, chat_id)


def _urlsafe_b64decode(data: str) -> bytes:
    from app.core.message_preprocessing import _urlsafe_b64decode as _impl
    return _impl(data)


def _decode_jwt_payload(token: str) -> Dict[str, Any]:
    from app.core.message_preprocessing import _decode_jwt_payload as _impl
    return _impl(token)


def _extract_user_id_from_token(token: str) -> str:
    return extract_user_id_from_token(token)


def _extract_text_from_content(content: Any) -> str:
    from app.core.message_preprocessing import _extract_text_from_content as _impl
    return _impl(content)


def _stringify_tool_arguments(arguments: Any) -> str:
    from app.core.message_preprocessing import _stringify_tool_arguments as _impl
    return _impl(arguments)


def _build_tool_call_index(messages: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    from app.core.message_preprocessing import _build_tool_call_index as _impl
    return _impl(messages)


def _format_tool_result_message(
    tool_name: str, tool_arguments: str, result_content: str
) -> str:
    from app.core.message_preprocessing import _format_tool_result_message as _impl
    return _impl(tool_name, tool_arguments, result_content)


def _format_assistant_tool_calls(tool_calls: List[Dict[str, Any]]) -> str:
    from app.core.message_preprocessing import _format_assistant_tool_calls as _impl
    return _impl(tool_calls)


def _preprocess_openai_messages(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return preprocess_openai_messages(messages)


def _extract_last_user_text(messages: List[Dict[str, Any]]) -> str:
    return extract_last_user_text(messages)


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

        # 向后兼容：旧代码可能访问这些属性
        self.model_mapping = self._model_manager.model_mapping
        self.model_mcp_servers = self._model_manager.model_mcp_servers
        self.model_scene_defaults = self._model_manager.model_scene_defaults

        self._shared_client = None
        self._shared_stream_client = None

    # ------------------------------------------------------------------
    # HTTP 客户端访问（向后兼容）
    # ------------------------------------------------------------------

    def _get_shared_client(self) -> httpx.AsyncClient:
        return self._http_clients.get_client()

    def _get_shared_stream_client(self) -> httpx.AsyncClient:
        return self._http_clients.get_stream_client()

    async def close(self) -> None:
        """关闭共享 HTTP 客户端连接。"""
        await self._http_clients.close()

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
    # 响应内容辅助方法（委托到 ResponseHandler）
    # ------------------------------------------------------------------

    def _clean_reasoning_delta(self, delta_content: str) -> str:
        """清理思考阶段的 details 包裹内容。"""
        return self._response_handler.clean_reasoning_delta(delta_content)

    def _extract_answer_content(self, text: str) -> str:
        """提取思考结束后的答案正文。"""
        return self._response_handler.extract_answer_content(text)

    def _normalize_tool_calls(
        self,
        raw_tool_calls: Any,
        start_index: int = 0,
    ) -> List[Dict[str, Any]]:
        """标准化上游工具调用为 OpenAI 兼容格式。"""
        return self._response_handler.normalize_tool_calls(raw_tool_calls, start_index)

    def _format_search_results(self, data: Dict[str, Any]) -> str:
        """将上游搜索结果格式化为可追加的 Markdown 引用。"""
        return self._response_handler.format_search_results(data)

    # ------------------------------------------------------------------
    # HTTP 客户端工具方法（向后兼容）
    # ------------------------------------------------------------------

    def _get_proxy_config(self) -> Optional[str]:
        """Get proxy configuration from settings"""
        from app.core.http_client import get_proxy_config
        return get_proxy_config()

    def _build_timeout(self, read_timeout: float = 300.0) -> httpx.Timeout:
        """Create httpx timeout settings tuned for upstream chat traffic."""
        from app.core.http_client import build_timeout
        return build_timeout(read_timeout)

    def _build_limits(self) -> httpx.Limits:
        """Create httpx connection limits."""
        from app.core.http_client import build_limits
        return build_limits()

    # ------------------------------------------------------------------
    # 在线模型（缓存层保留在本类）
    # ------------------------------------------------------------------

    _online_models: Optional[List[Dict[str, Any]]] = None
    _online_models_time: float = 0

    async def get_online_models(self) -> List[Dict[str, Any]]:
        """获取上游在线模型详细信息"""
        now = time.time()
        # 缓存1小时
        if self._online_models and (now - self._online_models_time < 3600):
            return self._online_models

        try:
            fe_version = await get_latest_fe_version()
            headers = build_dynamic_headers(fe_version=fe_version)
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

                    meta = info.get("meta", {})
                    capabilities = meta.get("capabilities", {})
                    mcp_servers = meta.get("mcpServerIds", [])

                    raw_tags = meta.get("tags", [])
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
                self.logger.debug(
                    f"✅ 在线模型同步成功，共获取 {len(parsed_models)} 个模型"
                )
                return parsed_models
            else:
                self.logger.warning(f"获取在线模型失败，状态码: {response.status_code}")
        except Exception as exc:
            self.logger.warning(f"获取在线模型异常: {exc}")

        return self._online_models or []

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

                    self.logger.warning(f"响应中未找到 token 字段: {data}")
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
                import asyncio
                await asyncio.sleep(2)

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
                self.logger.debug(f"从认证号池获取令牌: {token[:20]}...")
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
                    self.logger.warning(f"匿名会话池获取失败，转为直连访客鉴权: {exc}")

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
        self.logger.debug(f"🔄 转换 OpenAI 请求到上游格式: {request.model}")

        raw_messages = [
            message.model_dump(exclude_none=True) for message in request.messages
        ]

        # NOTE: chat.z.ai 对 OpenAI-style `tools` 的原生支持并不稳定。
        # 采用 Toolify XML 方案：生成随机触发信号，将工具定义注入 system prompt，
        # 模型会输出 XML 格式的工具调用，由响应处理器实时检测并转换为 OpenAI tool_calls。
        tools = request.tools if settings.TOOL_SUPPORT and request.tools else None
        tool_choice = getattr(request, "tool_choice", None)

        # 生成 Toolify XML 触发信号（每次请求唯一）
        trigger_signal = generate_trigger_signal() if tools else ""
        if trigger_signal:
            self.logger.debug(f"🔧 生成 XML 触发信号: {trigger_signal}")

        # 预处理消息（传入 trigger_signal 以便历史工具调用使用正确格式）
        normalized_messages = preprocess_openai_messages(
            raw_messages,
            trigger_signal=trigger_signal,
        )

        # 注入 XML 工具提示词到 system prompt
        normalized_messages = process_messages_with_tools(
            normalized_messages,
            tools=tools,
            tool_choice=tool_choice,
            trigger_signal=trigger_signal,
        )

        auth_info = await self.get_auth_info(
            excluded_tokens=excluded_tokens,
            excluded_guest_user_ids=excluded_guest_user_ids,
        )
        token = str(auth_info.get("token") or "")
        if not token:
            raise RuntimeError("无法获取上游认证令牌")

        user_id = str(
            auth_info.get("user_id") or extract_user_id_from_token(token)
        )
        auth_mode = str(auth_info.get("auth_mode") or "authenticated")
        token_source = str(auth_info.get("token_source") or "unknown")
        guest_user_id = auth_info.get("guest_user_id")

        # 生成 chat_id
        chat_id = generate_uuid()

        # 处理多模态消息（图片上传）
        messages, files = await process_multimodal_messages(
            normalized_messages=normalized_messages,
            token=token,
            user_id=user_id,
            chat_id=chat_id,
            auth_mode=auth_mode,
            http_client=self._get_shared_client(),
            base_url=self.base_url,
        )

        # 提取最后一条用户消息（用于签名）
        last_user_text = extract_last_user_text(raw_messages)

        # 解析模型特性
        features = self._model_manager.resolve_model_features(request)
        self.logger.debug(
            f"Resolved model features for {request.model}: {features}, "
            f"temperature={request.temperature}, max_tokens={request.max_tokens}"
        )

        message_id = generate_uuid()
        if tools:
            self.logger.info(
                f"工具定义: {len(tools)} 个工具；"
                "XML 提示已注入，tools/tool_choice 不透传到上游 body"
            )

        # 构建请求体（Toolify 方案：不透传 tools/tool_choice）
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
        )
        self.logger.debug(f"Upstream request body: {body}")

        # 签名并生成最终 URL 和 headers
        signed_url, headers, _fe_version = await sign_request(
            api_endpoint=self.api_endpoint,
            user_id=user_id,
            last_user_text=last_user_text,
            chat_id=chat_id,
            token=token,
        )

        # 存储当前token用于错误处理
        self._current_token = token

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
        }

    # ------------------------------------------------------------------
    # 聊天完成
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        request: OpenAIRequest,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """聊天完成接口。"""
        self.logger.debug(f"🔄 {self.name} 处理请求: {request.model}")
        self.logger.debug(f"  消息数量: {len(request.messages)}")
        self.logger.debug(f"  流式模式: {request.stream}")

        try:
            transformed = await self.transform_request(request)
            max_attempts = await self._get_total_retry_limit()

            if request.stream:
                return await self._create_stream_response(request, transformed)

            client = self._get_shared_client()
            excluded_tokens: Set[str] = set()
            excluded_guest_user_ids: Set[str] = set()

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
                            "⚠️ 认证会话不可用，准备切换认证 Token/回退匿名池: "
                            f"{current_token[:20]}..."
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
                    self.logger.error(f"❌ {self.name} 响应失败: {error_msg}")
                    return handle_error(Exception(error_message or error_msg))

                try:
                    result = await self.transform_response(response, request, transformed)
                finally:
                    await self._release_guest_session(transformed)

                if not self._is_guest_auth(transformed):
                    current_token = str(transformed.get("token") or "")
                    if current_token:
                        token_pool = get_token_pool()
                        if token_pool:
                            await token_pool.record_token_success(current_token)

                return result

        except Exception as e:
            self.logger.error(f"❌ {self.name} 响应失败: {str(e)}")
            return handle_error(e, "请求处理")

    async def _create_stream_response(
        self,
        request: OpenAIRequest,
        transformed: Dict[str, Any],
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """创建流式响应，并在首包前支持双池重试。"""
        max_attempts = await self._get_total_retry_limit()
        excluded_tokens: Set[str] = set()
        excluded_guest_user_ids: Set[str] = set()
        current_token = str(transformed.get("token") or "")

        client = self._get_shared_stream_client()

        for attempt in range(max_attempts):
            self.logger.debug(f"🎯 发送请求到上游: {transformed['url']}")
            req = client.build_request(
                "POST",
                transformed["url"],
                json=transformed["body"],
                headers=transformed["headers"],
            )

            try:
                response = await client.send(req, stream=True)
            except Exception as e:
                self.logger.error(f"❌ 连接异常: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                if self._is_guest_auth(transformed):
                    await self._release_guest_session(transformed)
                elif current_token:
                    await self.mark_token_failure(current_token, e)
                return {
                    "error": {
                        "message": str(e),
                        "type": "stream_error",
                    }
                }

            try:
                error_text = (
                    await response.aread() if response.status_code != 200 else b""
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
                            "⚠️ 流式请求命中认证会话限制，准备切号/回退匿名池: "
                            f"{current_token[:20]}..."
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
                    self.logger.error(f"❌ 上游返回错误: {response.status_code}")
                    if error_msg:
                        self.logger.error(f"❌ 错误详情: {error_msg}")

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
                    else:
                        return {
                            "error": {
                                "message": parsed_error_message
                                or f"Upstream error: {response.status_code}",
                                "type": "upstream_error",
                                "code": error_code or response.status_code,
                            }
                        }

                chat_id = transformed["chat_id"]
                model = transformed["model"]
                
                async def stream_generator() -> AsyncGenerator[str, None]:
                    try:
                        async for chunk in self._response_handler.handle_stream_response(
                            response,
                            chat_id,
                            model,
                            request,
                            transformed,
                        ):
                            yield chunk
                    except Exception as e:
                        self.logger.error(f"❌ 流处理错误: {e}")
                        import traceback
                        self.logger.error(traceback.format_exc())
                        if self._is_guest_auth(transformed):
                            await self._release_guest_session(transformed)
                        elif current_token:
                            await self.mark_token_failure(current_token, e)
                        
                        error_response = {
                            "error": {
                                "message": str(e),
                                "type": "stream_error",
                            }
                        }
                        yield f"data: {json.dumps(error_response)}\n\n"
                        yield "data: [DONE]\n\n"
                    finally:
                        await response.aclose()
                        await self._release_guest_session(transformed)
                        if not self._is_guest_auth(transformed) and current_token:
                            token_pool = get_token_pool()
                            if token_pool:
                                await token_pool.record_token_success(current_token)

                return stream_generator()

            except Exception as e:
                if 'response' in locals() and response:
                    await response.aclose()
                self.logger.error(f"❌ 流处理错误: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                if self._is_guest_auth(transformed):
                    await self._release_guest_session(transformed)
                elif current_token:
                    await self.mark_token_failure(current_token, e)
                
                return {
                    "error": {
                        "message": str(e),
                        "type": "stream_error",
                    }
                }

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
            )

    async def _handle_stream_response(
        self,
        response: httpx.Response,
        chat_id: str,
        model: str,
        request: OpenAIRequest,
        transformed: Dict[str, Any],
    ) -> AsyncGenerator[str, None]:
        """处理上游流式响应（委托到 ResponseHandler）。"""
        async for chunk in self._response_handler.handle_stream_response(
            response, chat_id, model, request, transformed
        ):
            yield chunk

    async def _handle_non_stream_response(
        self,
        response: httpx.Response,
        chat_id: str,
        model: str,
        transformed: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """处理非流式响应（委托到 ResponseHandler）。"""
        trigger_signal = (transformed or {}).get("trigger_signal", "")
        tools_defs = (transformed or {}).get("tools")
        return await self._response_handler.handle_non_stream_response(
            response, chat_id, model,
            trigger_signal=trigger_signal,
            tools_defs=tools_defs,
        )
