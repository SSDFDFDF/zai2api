#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Captcha 服务客户端。

通过 HTTP 调用独立的 Captcha Token Service 获取 captcha_verify_param。
支持重试、超时、连接错误处理，与项目 SharedHttpClients 模式一致。
"""

import asyncio
from typing import Optional

import httpx

from app.core.config import settings
from app.core.http_client import build_timeout, get_proxy_config
from app.utils.logger import logger


class CaptchaServiceError(Exception):
    """Captcha 服务不可用（连接失败、超时、非 200 响应）。"""


class CaptchaTokenError(Exception):
    """Captcha token 获取失败（服务返回错误）。"""


class CaptchaClient:
    """Captcha Token Service 的 HTTP 客户端。

    按需调用远端 captcha 服务获取验证码 token，
    内置重试和指数退避。
    """

    def __init__(
        self,
        service_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._service_url = (service_url or settings.CAPTCHA_SERVICE_URL).rstrip("/")
        self._timeout = timeout or settings.CAPTCHA_SERVICE_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None
        self._max_retries = 3

    @property
    def token_url(self) -> str:
        return f"{self._service_url}/token"

    @property
    def health_url(self) -> str:
        return f"{self._service_url}/health"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=build_timeout(read_timeout=self._timeout),
                proxy=get_proxy_config(),
                trust_env=False,
            )
        return self._client

    async def get_token(self, jwt_token: str) -> str:
        """获取一个新鲜的 captcha_verify_param。

        Args:
            jwt_token: 用户的 Z.ai JWT token。

        Returns:
            captcha_verify_param 字符串（base64 编码的 captcha 数据）。

        Raises:
            CaptchaServiceError: 服务不可用或超时。
            CaptchaTokenError: 服务返回错误。
        """
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                client = await self._get_client()
                response = await client.post(
                    self.token_url,
                    json={"jwt_token": jwt_token},
                )

                if response.status_code == 200:
                    data = response.json()
                    token = data.get("captcha_verify_param")
                    if token:
                        logger.info(
                            "[captcha] token obtained certify_id=%s attempt=%d",
                            data.get("certify_id", "unknown"),
                            attempt + 1,
                        )
                        return token
                    raise CaptchaTokenError(
                        f"Captcha service returned empty token: {data}"
                    )

                error_text = response.text[:500]
                if response.status_code == 503:
                    if attempt < self._max_retries - 1:
                        wait = 2 ** attempt
                        logger.warning(
                            "[captcha] service unavailable (503), retry in %ds (attempt %d/%d): %s",
                            wait, attempt + 1, self._max_retries, error_text,
                        )
                        await asyncio.sleep(wait)
                        last_error = CaptchaServiceError(
                            f"Captcha service unavailable: {error_text}"
                        )
                        continue
                    raise CaptchaServiceError(
                        f"Captcha service unavailable after {self._max_retries} attempts: {error_text}"
                    )

                raise CaptchaTokenError(
                    f"Captcha service returned {response.status_code}: {error_text}"
                )

            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as e:
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "[captcha] connection failed, retry in %ds (attempt %d/%d): %s",
                        wait, attempt + 1, self._max_retries, e,
                    )
                    await asyncio.sleep(wait)
                    last_error = e
                    continue
                raise CaptchaServiceError(
                    f"Cannot reach captcha service at {self._service_url}: {e}"
                ) from e

            except (CaptchaServiceError, CaptchaTokenError):
                raise
            except Exception as e:
                if attempt < self._max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "[captcha] unexpected error, retry in %ds (attempt %d/%d): %s",
                        wait, attempt + 1, self._max_retries, e,
                    )
                    await asyncio.sleep(wait)
                    last_error = e
                    continue
                raise CaptchaServiceError(f"Captcha client error: {e}") from e

        raise CaptchaServiceError(
            f"Failed to get captcha token after {self._max_retries} attempts"
        ) from last_error

    async def health(self) -> dict:
        """检查 captcha 服务健康状态。"""
        try:
            client = await self._get_client()
            response = await client.get(self.health_url)
            return response.json() if response.status_code == 200 else {"status": "error"}
        except Exception:
            return {"status": "unreachable"}

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# Module-level singleton
_captcha_client: Optional[CaptchaClient] = None


def get_captcha_client() -> Optional[CaptchaClient]:
    """获取 captcha 客户端单例。未初始化时返回 None。"""
    return _captcha_client


def create_captcha_client(
    service_url: Optional[str] = None,
    timeout: Optional[float] = None,
) -> CaptchaClient:
    """创建并注册 captcha 客户端单例。"""
    global _captcha_client
    _captcha_client = CaptchaClient(service_url=service_url, timeout=timeout)
    logger.info(
        "[captcha] client initialized service_url=%s timeout=%s",
        _captcha_client._service_url,
        _captcha_client._timeout,
    )
    return _captcha_client


async def close_captcha_client() -> None:
    """关闭 captcha 客户端连接。"""
    global _captcha_client
    if _captcha_client:
        await _captcha_client.close()
        _captcha_client = None
        logger.info("[captcha] client closed")
