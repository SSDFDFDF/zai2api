#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""统一浏览器 headers 生成。

提供 build_dynamic_headers() 作为唯一入口。

⚠️ 此处只生成真实浏览器同源 fetch 请求中实际带的最小集合，避免被阿里云 ESA/WAF
按"非浏览器指纹"判别拦截（参见 chat.z.ai 抓包对比）。
- 不发 Origin（同源请求浏览器不发）
- 不发 Cache-Control / Pragma / Connection（fetch 不发）
- 不发 Sec-Fetch-*（fetch 自身管理，写死反而成为指纹）
- 不发 Referer 路径（chat.z.ai 前端的 referrer policy 让其为空）
"""

from typing import Dict

CAPTCHA_PROVIDER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
)

CAPTCHA_PROVIDER_SEC_CH_UA = (
    '"Google Chrome";v="133", "Chromium";v="133", "Not A(Brand";v="24"'
)


def build_dynamic_headers(fe_version: str, chat_id: str = "") -> Dict[str, str]:
    """生成上游请求所需的最小浏览器 headers。

    Args:
        fe_version: 前端版本号，填充到 X-FE-Version header。
        chat_id: 当前对话 ID（保留参数以兼容现有调用方，不再用于构造 Referer）。

    Returns:
        与 captcha-provider 的 Chrome 133 指纹保持一致的最小 header 集合。
    """
    del chat_id  # 兼容旧签名；真实浏览器同源请求里 Referer 为空，不再据此拼接

    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "User-Agent": CAPTCHA_PROVIDER_USER_AGENT,
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "X-FE-Version": fe_version,
        "X-Region": "overseas",
        "sec-ch-ua": CAPTCHA_PROVIDER_SEC_CH_UA,
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
    }

    return headers
