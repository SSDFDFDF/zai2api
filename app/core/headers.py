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

import random
from typing import Dict

from app.utils.user_agent import get_random_user_agent


def build_dynamic_headers(fe_version: str, chat_id: str = "") -> Dict[str, str]:
    """生成上游请求所需的最小浏览器 headers。

    Args:
        fe_version: 前端版本号，填充到 X-FE-Version header。
        chat_id: 当前对话 ID（保留参数以兼容现有调用方，不再用于构造 Referer）。

    Returns:
        与真实浏览器同源 fetch 一致的最小 header 集合。
        Firefox UA 不带 sec-ch-ua 系列；Chromium 系带。
    """
    del chat_id  # 兼容旧签名；真实浏览器同源请求里 Referer 为空，不再据此拼接
    browser_choices = ["chrome", "chrome", "chrome", "edge", "edge", "firefox", "safari"]
    browser_type = random.choice(browser_choices)
    user_agent = get_random_user_agent(browser_type)

    chrome_version = "139"
    edge_version = "139"

    if "Chrome/" in user_agent:
        try:
            chrome_version = user_agent.split("Chrome/")[1].split(".")[0]
        except Exception:
            pass

    if "Edg/" in user_agent:
        try:
            edge_version = user_agent.split("Edg/")[1].split(".")[0]
            sec_ch_ua = (
                f'"Microsoft Edge";v="{edge_version}", '
                f'"Chromium";v="{chrome_version}", "Not_A Brand";v="24"'
            )
        except Exception:
            sec_ch_ua = (
                f'"Not_A Brand";v="8", "Chromium";v="{chrome_version}", '
                f'"Google Chrome";v="{chrome_version}"'
            )
    elif "Firefox/" in user_agent:
        sec_ch_ua = None
    else:
        sec_ch_ua = (
            f'"Not_A Brand";v="8", "Chromium";v="{chrome_version}", '
            f'"Google Chrome";v="{chrome_version}"'
        )

    if "Windows" in user_agent:
        platform = '"Windows"'
    elif "Macintosh" in user_agent or "Mac OS X" in user_agent:
        platform = '"macOS"'
    elif "Linux" in user_agent:
        platform = '"Linux"'
    else:
        platform = '"Windows"'

    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "User-Agent": user_agent,
        "Accept-Language": "en-US",
        "X-FE-Version": fe_version,
        "X-Region": "overseas",
    }

    if sec_ch_ua:
        headers["sec-ch-ua"] = sec_ch_ua
        headers["sec-ch-ua-mobile"] = "?0"
        headers["sec-ch-ua-platform"] = platform

    return headers
