#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用户代理工具模块
提供动态随机用户代理生成功能，包含更丰富的浏览器指纹特征
"""

import random
import re
from typing import Dict, Optional, Tuple
from fake_useragent import UserAgent

# 全局 UserAgent 实例（单例模式）
_user_agent_instance: Optional[UserAgent] = None


def get_user_agent_instance() -> UserAgent:
    """获取或创建 UserAgent 实例（单例模式）"""
    global _user_agent_instance
    if _user_agent_instance is None:
        try:
            # 开启缓存以提高性能
            _user_agent_instance = UserAgent(use_external_data=True, cache=True)
        except Exception:
            # 如果外部数据加载失败，提供降级机制
            _user_agent_instance = UserAgent(use_external_data=False)
    return _user_agent_instance


def get_random_user_agent(browser_type: Optional[str] = None) -> str:
    """
    获取随机用户代理字符串

    Args:
        browser_type: 指定浏览器类型 ('chrome', 'firefox', 'safari', 'edge', 'mobile_safari', 'mobile_chrome')
                     如果为 None，则随机选择

    Returns:
        str: 用户代理字符串
    """
    try:
        ua = get_user_agent_instance()
        
        # 定义更多的浏览器选择及其权重
        if browser_type is None:
            browser_choices = [
                "chrome", "chrome", "chrome", "chrome", 
                "edge", "edge", 
                "firefox", "safari", 
                "mobile_chrome", "mobile_safari"
            ]
            browser_type = random.choice(browser_choices)

        # 根据浏览器类型获取用户代理
        if browser_type == "chrome":
            user_agent = ua.chrome
        elif browser_type == "edge":
            user_agent = ua.edge
        elif browser_type == "firefox":
            user_agent = ua.firefox
        elif browser_type == "safari":
            user_agent = ua.safari
        # 对于移动端特定的 UA, fake_useragent.random 支持通过 OS 等获取，这里为增加可靠性可以显式定义些移动端特征
        elif browser_type == "mobile_chrome":
            # 尝试生成包含 Android 的 Chrome UA
            user_agent = f"Mozilla/5.0 (Linux; Android {random.randint(10, 14)}; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(110, 125)}.0.0.0 Mobile Safari/537.36"
        elif browser_type == "mobile_safari":
            ios_ver = f"{random.randint(15, 17)}_{random.randint(0, 5)}"
            user_agent = f"Mozilla/5.0 (iPhone; CPU iPhone OS {ios_ver} like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/{random.randint(15, 17)}.0 Mobile/15E148 Safari/604.1"
        else:
            user_agent = ua.random

        # 确保不返回 None 或空字符串
        if not user_agent:
            raise ValueError("获取到的 User-Agent 为空")
            
        return user_agent

    except Exception:
        # Fallback 机制：如果 fake_useragent 失败，使用硬编码的常用 UA 列表
        fallback_uas = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15"
        ]
        return random.choice(fallback_uas)


def parse_ua_features(user_agent: str) -> Tuple[str, str, str, str, str]:
    """
    解析 UA 字符串以提取平台、移动端标识、Chrome/Edge/Firefox 等版本号
    """
    platform = '"Windows"'
    if "Windows" in user_agent:
        platform = '"Windows"'
    elif "Macintosh" in user_agent or "Mac OS X" in user_agent:
        platform = '"macOS"'
    elif "Android" in user_agent:
        platform = '"Android"'
    elif "Linux" in user_agent:
        platform = '"Linux"'
    elif "iPhone" in user_agent or "iPad" in user_agent:
        platform = '"iOS"'
    elif "CrOS" in user_agent:
        platform = '"Chrome OS"'

    is_mobile = "?1" if "Mobile" in user_agent or "Android" in user_agent or platform in ('"Android"', '"iOS"') else "?0"

    chrome_version = "124"
    edge_version = "124"
    firefox_version = ""

    # 解析 Chrome 版本
    chrome_match = re.search(r"Chrome/(\d+)\.", user_agent)
    if chrome_match:
        chrome_version = chrome_match.group(1)

    # 解析 Edge 版本 (Edge, Edg, EdgiOS, EdgA)
    edge_match = re.search(r"Edg[a-zA-Z]*/(\d+)\.", user_agent)
    if edge_match:
        edge_version = edge_match.group(1)
        
    # 解析 Firefox 版本
    firefox_match = re.search(r"Firefox/(\d+)\.", user_agent)
    if firefox_match:
        firefox_version = firefox_match.group(1)

    return platform, is_mobile, chrome_version, edge_version, firefox_version


def get_dynamic_headers(
    referer: Optional[str] = None,
    origin: Optional[str] = None,
    browser_type: Optional[str] = None,
    additional_headers: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    生成动态浏览器 headers，包含随机 User-Agent 以及逼真的客户端提示 (Client Hints)

    Args:
        referer: 引用页面 URL
        origin: 源站 URL
        browser_type: 指定浏览器类型
        additional_headers: 额外的 headers

    Returns:
        Dict[str, str]: 包含动态 User-Agent 的 headers
    """
    user_agent = get_random_user_agent(browser_type)
    
    # 随机选择常见的支持语言组合
    accept_languages = [
        "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
        "zh-CN,zh;q=0.9,en;q=0.8",
        "zh-CN,zh-TW;q=0.9,zh;q=0.8,en-US;q=0.7,en;q=0.6",
        "zh-CN,zh;q=0.9",
        "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    ]

    # 基础 headers
    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Accept-Language": random.choice(accept_languages),
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Cache-Control": "max-age=0",
        "Connection": "keep-alive",
        "DNT": "1" if random.random() > 0.5 else "0", # 随机的 Do Not Track
        "Upgrade-Insecure-Requests": "1",
    }

    # 添加可选的 headers
    if referer:
        headers["Referer"] = referer

    if origin:
        headers["Origin"] = origin
        
    # 解析特征
    platform, is_mobile, chrome_version, edge_version, firefox_version = parse_ua_features(user_agent)

    # 识别是否是 Chromium 内核浏览器 (包含 Chrome, Edge 等)
    is_chromium = "Chrome/" in user_agent or "CrOS" in user_agent
    is_edge = re.search(r"Edg[a-zA-Z]*/\d+\.", user_agent) is not None
    is_safari = "Safari/" in user_agent and "Chrome/" not in user_agent and "Chromium/" not in user_agent
    is_firefox = "Firefox/" in user_agent

    # 针对 Chromium 内核（Chrome, Edge, Opera 等）添加特有的 Sec-CH-UA 头部
    if is_chromium:
        # Chromium 随机品牌策略
        brand_list = [
            f'"Not-A.Brand";v="99"',
            f'"Not/A)Brand";v="8"',
            f'"Chromium";v="{chrome_version}"'
        ]
        
        if is_edge:
            brand_list.append(f'"Microsoft Edge";v="{edge_version}"')
        else:
            brand_list.append(f'"Google Chrome";v="{chrome_version}"')
            
        # 打乱品牌顺序使其更随机
        random.shuffle(brand_list)
        # 通常 Sec-CH-UA 有 3 个品牌
        sec_ch_ua = ", ".join(brand_list[:3])

        headers.update({
            "sec-ch-ua": sec_ch_ua,
            "sec-ch-ua-mobile": is_mobile,
            "sec-ch-ua-platform": platform,
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site" if origin else "same-origin",
            "Sec-Fetch-User": "?1",
        })
    elif is_firefox:
        # Firefox 特有请求头
        headers.update({
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site" if origin else "same-origin",
            "Sec-Fetch-User": "?1",
            "TE": "trailers",
        })
    elif is_safari:
        # Safari 特有请求头
        headers.update({
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site" if origin else "same-origin",
        })
        
    # 添加现代浏览器常见的优先度请求头
    if "Priority" not in headers and (is_chromium or is_firefox):
        headers["Priority"] = "u=0, i"

    # 添加额外的 headers (覆盖默认值)
    if additional_headers:
        headers.update(additional_headers)

    return headers


