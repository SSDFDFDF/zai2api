#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""统一浏览器 headers 生成。

提供 build_dynamic_headers() 作为唯一入口。
所有 header 字段和参数均保留。
"""

import random
from typing import Dict

from app.utils.user_agent import get_random_user_agent


def build_dynamic_headers(fe_version: str, chat_id: str = "") -> Dict[str, str]:
    """生成上游请求所需的动态浏览器 headers。

    随机选择浏览器类型（chrome/edge/firefox/safari），根据 User-Agent 动态生成
    对应的 sec-ch-ua 等安全相关头。

    Args:
        fe_version: 前端版本号，填充到 X-FE-Version header。
        chat_id: 当前对话 ID，非空时设置 Referer 为对话页。

    Returns:
        包含所有必要浏览器 headers 的字典。
        Firefox 不包含 sec-ch-ua 相关头（浏览器行为对齐）。
    """
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

    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "*/*",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "User-Agent": user_agent,
        "Accept-Language": "zh-CN",
        "X-FE-Version": fe_version,
        "Origin": "https://chat.z.ai",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }

    if sec_ch_ua:
        headers["sec-ch-ua"] = sec_ch_ua
        headers["sec-ch-ua-mobile"] = "?0"
        headers["sec-ch-ua-platform"] = '"Windows"'

    if chat_id:
        headers["Referer"] = f"https://chat.z.ai/c/{chat_id}"
    else:
        headers["Referer"] = "https://chat.z.ai/"

    return headers
