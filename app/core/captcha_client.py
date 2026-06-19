#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Captcha provider 客户端。

通过 HTTP 调用 Node captcha-provider 获取 captcha_verify_param。
支持重试、超时、连接错误处理，与项目 SharedHttpClients 模式一致。
"""

import asyncio
from typing import Optional

import httpx

from app.core.config import settings
from app.core.http_client import build_timeout
from app.utils.logger import logger


class CaptchaProviderError(Exception):
    """Captcha provider 不可用（连接失败、超时、非 200 响应）。"""


class CaptchaTokenError(Exception):
    """Captcha token 获取失败（provider 返回错误）。"""


class CaptchaClient:
    """Node captcha-provider 的 HTTP 客户端。

    按需调用 provider 获取验证码 token，
    内置重试和指数退避。
    """

    def __init__(
        self,
        provider_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        secret: Optional[str] = None,
    ) -> None:
        self._provider_url = (provider_url or settings.CAPTCHA_PROVIDER_URL).rstrip(
            "/"
        )
        self._timeout = timeout or settings.CAPTCHA_PROVIDER_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None
        self._max_retries = (
            max_retries if max_retries is not None else settings.CAPTCHA_MAX_RETRIES
        )
        self._secret = (
            secret if secret is not None else settings.CAPTCHA_PROVIDER_SECRET
        )

    @property
    def token_url(self) -> str:
        return f"{self._provider_url}/token"

    @property
    def health_url(self) -> str:
        return f"{self._provider_url}/health"

    @property
    def _headers(self) -> dict[str, str]:
        return {"x-secret": self._secret} if self._secret else {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=build_timeout(read_timeout=self._timeout),
                trust_env=False,
            )
        return self._client

    async def get_token(self) -> str:
        """获取一个新鲜的 captcha_verify_param。

        Returns:
            captcha_verify_param 字符串。

        Raises:
            CaptchaProviderError: provider 不可用或超时。
            CaptchaTokenError: provider 返回错误。
        """
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                client = await self._get_client()
                response = await client.get(
                    self.token_url,
                    headers=self._headers,
                )

                if response.status_code == 200:
                    data = response.json()
                    token = data.get("token")
                    if data.get("ok") is True and token:
                        logger.info(
                            "[captcha] provider token obtained cached=%s attempt=%d",
                            data.get("cached", False),
                            attempt + 1,
                        )
                        return token
                    raise CaptchaTokenError(
                        f"Captcha provider returned invalid token response: {data}"
                    )

                error_text = response.text[:500]
                if response.status_code >= 500:
                    if attempt < self._max_retries - 1:
                        wait = 2 ** attempt
                        logger.warning(
                            "[captcha] provider unavailable (%d), retry in %ds "
                            "(attempt %d/%d): %s",
                            response.status_code,
                            wait,
                            attempt + 1,
                            self._max_retries,
                            error_text,
                        )
                        await asyncio.sleep(wait)
                        last_error = CaptchaProviderError(
                            f"Captcha provider unavailable: {error_text}"
                        )
                        continue
                    raise CaptchaProviderError(
                        "Captcha provider unavailable after "
                        f"{self._max_retries} attempts: {error_text}"
                    )

                raise CaptchaTokenError(
                    f"Captcha provider returned {response.status_code}: {error_text}"
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
                raise CaptchaProviderError(
                    f"Cannot reach captcha provider at {self._provider_url}: {e}"
                ) from e

            except (CaptchaProviderError, CaptchaTokenError):
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
                raise CaptchaProviderError(f"Captcha client error: {e}") from e

        raise CaptchaProviderError(
            f"Failed to get captcha provider token after {self._max_retries} attempts"
        ) from last_error

    async def health(self) -> dict:
        """检查 captcha provider 健康状态。"""
        try:
            client = await self._get_client()
            response = await client.get(self.health_url, headers=self._headers)
            return (
                response.json()
                if response.status_code == 200
                else {"status": "error"}
            )
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
    provider_url: Optional[str] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    secret: Optional[str] = None,
) -> CaptchaClient:
    """创建并注册 captcha 客户端单例。"""
    global _captcha_client
    _captcha_client = CaptchaClient(
        provider_url=provider_url,
        timeout=timeout,
        max_retries=max_retries,
        secret=secret,
    )
    logger.info(
        "[captcha] provider client initialized url=%s timeout=%s",
        _captcha_client._provider_url,
        _captcha_client._timeout,
    )
    return _captcha_client


async def close_captcha_client() -> None:
    """关闭 captcha 客户端连接。"""
    global _captcha_client
    if _captcha_client:
        await _captcha_client.close()
        _captcha_client = None
        logger.info("[captcha] provider client closed")
