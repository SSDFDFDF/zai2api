#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""统一 HTTP 客户端工厂。

提供代理配置、超时设置、连接池配置的统一构建函数，
以及管理共享 AsyncClient 生命周期的 SharedHttpClients 类。
可被 upstream.py 和 guest_session_pool.py 等模块复用。
"""

from typing import Optional

import httpx

from app.core.config import settings


def get_proxy_config() -> Optional[str]:
    """获取代理配置。

    统一获取全局的网络代理。支持 http(s):// 或 socks5://。
    返回 httpx 接受的代理 URL 字符串，无代理时返回 None。
    """
    proxy = settings.HTTP_PROXY
    return proxy if proxy else None


def build_timeout(read_timeout: float = 300.0) -> httpx.Timeout:
    """构建 httpx 超时配置。

    Args:
        read_timeout: 读取超时（秒），默认 300s 适配流式长响应。
            非流式短请求可传入 60.0。
    """
    return httpx.Timeout(
        connect=5.0,
        read=read_timeout,
        write=10.0,
        pool=5.0,
    )


def build_limits(
    max_keepalive_connections: Optional[int] = None,
    max_connections: Optional[int] = None,
) -> httpx.Limits:
    """构建 httpx 连接池限制。

    Args:
        max_keepalive_connections: 最大持久连接数，None 时使用全局配置。
        max_connections: 最大连接数，None 时使用全局配置。
    """
    keepalive = max_keepalive_connections if max_keepalive_connections is not None else settings.GUEST_HTTP_MAX_KEEPALIVE_CONNECTIONS
    connections = max_connections if max_connections is not None else settings.GUEST_HTTP_MAX_CONNECTIONS
    return httpx.Limits(
        max_keepalive_connections=max(1, keepalive),
        max_connections=max(1, connections),
    )


class SharedHttpClients:
    """管理共享 httpx.AsyncClient 生命周期。

    维护两类客户端：
    - ``client``：用于短请求（鉴权、文件上传、模型列表），读取超时 60s。
    - ``stream_client``：用于流式聊天，读取超时 300s，启用 HTTP/2。

    使用示例::

        clients = SharedHttpClients()
        client = clients.get_client()
        stream_client = clients.get_stream_client()
        await clients.close()
    """

    def __init__(self) -> None:
        self._client: Optional[httpx.AsyncClient] = None
        self._stream_client: Optional[httpx.AsyncClient] = None

    def get_client(self) -> httpx.AsyncClient:
        """获取通用共享客户端（读取超时 60s）。

        首次调用时惰性创建，后续调用复用同一实例（除非已关闭）。
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=build_timeout(read_timeout=60.0),
                limits=build_limits(
                    max_keepalive_connections=20,
                    max_connections=50,
                ),
                proxy=get_proxy_config(),
            )
        return self._client

    def get_stream_client(self) -> httpx.AsyncClient:
        """获取流式专用客户端（读取超时 300s，启用 HTTP/2）。

        首次调用时惰性创建，后续调用复用同一实例（除非已关闭）。
        """
        if self._stream_client is None or self._stream_client.is_closed:
            self._stream_client = httpx.AsyncClient(
                timeout=build_timeout(read_timeout=300.0),
                http2=True,
                limits=build_limits(
                    max_keepalive_connections=20,
                    max_connections=50,
                ),
                proxy=get_proxy_config(),
            )
        return self._stream_client

    async def close(self) -> None:
        """关闭所有共享客户端连接。"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        if self._stream_client and not self._stream_client.is_closed:
            await self._stream_client.aclose()
