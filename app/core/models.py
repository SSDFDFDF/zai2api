#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""模型管理模块。

集中管理模型映射、MCP 服务器配置、场景默认值，
以及模型特性解析逻辑。原先散落在 UpstreamClient.__init__ 和
transform_request 中的模型相关代码统一迁移至此。
"""

from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.models.schemas import OpenAIRequest


class ModelManager:
    """模型配置管理器。

    管理三类配置：
    1. ``model_mapping`` — 外部模型名 → 上游实际模型 ID
    2. ``model_mcp_servers`` — 模型 → 默认 MCP 服务器列表
    3. ``model_scene_defaults`` — 模型 → 场景默认参数
    """

    def __init__(self) -> None:
        # 外部模型名 → 上游模型 ID
        self.model_mapping: Dict[str, str] = {
            settings.GLM45_MODEL: "0727-360B-API",           # GLM-4.5
            settings.GLM45_THINKING_MODEL: "0727-360B-API",  # GLM-4.5-Thinking
            settings.GLM45_SEARCH_MODEL: "0727-360B-API",    # GLM-4.5-Search
            settings.GLM45_AIR_MODEL: "0727-106B-API",       # GLM-4.5-Air
            settings.GLM46V_MODEL: "glm-4.6v",               # GLM-4.6V
            settings.GLM46V_ADVANCED_SEARCH_MODEL: "glm-4.6v",  # GLM-4.6V-advanced-search
            settings.GLM5_MODEL: "glm-5",                    # GLM-5
            settings.GLM5_THINKING_MODEL: "glm-5",           # GLM-5-Thinking
            settings.GLM5_AGENT_MODEL: "glm-5",              # GLM-5-Agent
            settings.GLM5_ADVANCED_SEARCH_MODEL: "glm-5",   # GLM-5-advanced-search
            settings.GLM47_MODEL: "glm-4.7",                 # GLM-4.7
            settings.GLM47_THINKING_MODEL: "glm-4.7",        # GLM-4.7-Thinking
            settings.GLM47_SEARCH_MODEL: "glm-4.7",          # GLM-4.7-Search
            settings.GLM47_ADVANCED_SEARCH_MODEL: "glm-4.7", # GLM-4.7-advanced-search
        }

        # 每个模型的默认 MCP 服务器（基于浏览器抓包分析）
        # 基础模型和其他变体不需要 MCP 服务器
        self.model_mcp_servers: Dict[str, List[str]] = {
            settings.GLM5_ADVANCED_SEARCH_MODEL: ["advanced-search"],
            settings.GLM46V_ADVANCED_SEARCH_MODEL: ["advanced-search"],
            settings.GLM47_ADVANCED_SEARCH_MODEL: ["advanced-search"],
        }

        # 模型场景默认值（覆盖通用默认值）
        # 可配置: flags, enable_thinking, auto_web_search, extra
        self.model_scene_defaults: Dict[str, Dict[str, Any]] = {
            settings.GLM5_AGENT_MODEL: {
                "flags": ["general_agent"],
                "enable_thinking": True,
                "auto_web_search": False,
            },
            settings.GLM5_ADVANCED_SEARCH_MODEL: {
                "enable_thinking": True,
                "web_search": True,
                "auto_web_search": True,
            },
            settings.GLM46V_ADVANCED_SEARCH_MODEL: {
                "enable_thinking": True,
                "web_search": True,
                "auto_web_search": True,
            },
            settings.GLM47_ADVANCED_SEARCH_MODEL: {
                "enable_thinking": True,
                "web_search": True,
                "auto_web_search": True,
            },
            settings.GLM45_SEARCH_MODEL: {
                "web_search": True,
                "auto_web_search": True,
            },
            settings.GLM47_SEARCH_MODEL: {
                "web_search": True,
                "auto_web_search": True,
            },
        }

        # 模型能力声明 — 让客户端（Cursor / Continue / Cherry Studio 等）
        # 通过 /v1/models 响应识别模型支持的特性
        # 能力键说明:
        #   tool_use     — 支持 function calling / tool_calls
        #   vision       — 支持图像输入（多模态）
        #   web_search   — 支持联网搜索
        #   thinking     — 支持思维链 / 推理过程输出
        #   agent        — 支持 Agent 模式
        self.model_capabilities: Dict[str, Dict[str, bool]] = {
            settings.GLM45_MODEL: {
                "tool_use": True,
                "vision": False,
                "web_search": False,
                "thinking": False,
            },
            settings.GLM45_THINKING_MODEL: {
                "tool_use": True,
                "vision": False,
                "web_search": False,
                "thinking": True,
            },
            settings.GLM45_SEARCH_MODEL: {
                "tool_use": True,
                "vision": False,
                "web_search": True,
                "thinking": False,
            },
            settings.GLM45_AIR_MODEL: {
                "tool_use": True,
                "vision": False,
                "web_search": False,
                "thinking": False,
            },
            settings.GLM46V_MODEL: {
                "tool_use": True,
                "vision": True,
                "web_search": False,
                "thinking": False,
            },
            settings.GLM46V_ADVANCED_SEARCH_MODEL: {
                "tool_use": True,
                "vision": True,
                "web_search": True,
                "thinking": True,
            },
            settings.GLM5_MODEL: {
                "tool_use": True,
                "vision": False,
                "web_search": False,
                "thinking": False,
            },
            settings.GLM5_THINKING_MODEL: {
                "tool_use": True,
                "vision": False,
                "web_search": False,
                "thinking": True,
            },
            settings.GLM5_AGENT_MODEL: {
                "tool_use": True,
                "vision": False,
                "web_search": False,
                "thinking": True,
                "agent": True,
            },
            settings.GLM5_ADVANCED_SEARCH_MODEL: {
                "tool_use": True,
                "vision": False,
                "web_search": True,
                "thinking": True,
            },
            settings.GLM47_MODEL: {
                "tool_use": True,
                "vision": False,
                "web_search": False,
                "thinking": False,
            },
            settings.GLM47_THINKING_MODEL: {
                "tool_use": True,
                "vision": False,
                "web_search": False,
                "thinking": True,
            },
            settings.GLM47_SEARCH_MODEL: {
                "tool_use": True,
                "vision": False,
                "web_search": True,
                "thinking": False,
            },
            settings.GLM47_ADVANCED_SEARCH_MODEL: {
                "tool_use": True,
                "vision": False,
                "web_search": True,
                "thinking": True,
            },
        }

    def get_upstream_model_id(self, model_name: str) -> Optional[str]:
        """获取上游实际模型 ID。

        Args:
            model_name: 外部模型名称（如 "GLM-4.5"）。

        Returns:
            上游模型 ID，未匹配时返回 None。
        """
        return self.model_mapping.get(model_name)

    def get_mcp_servers(self, model_name: str) -> List[str]:
        """获取模型的默认 MCP 服务器列表。

        Args:
            model_name: 外部模型名称。

        Returns:
            MCP 服务器 ID 列表，无配置时返回空列表。
        """
        return self.model_mcp_servers.get(model_name, [])

    def get_scene_defaults(self, model_name: str) -> Dict[str, Any]:
        """获取模型的场景默认配置。

        Args:
            model_name: 外部模型名称。

        Returns:
            场景默认配置字典，无配置时返回空字典。
        """
        return self.model_scene_defaults.get(model_name, {})

    def get_model_capabilities(self, model_name: str) -> Dict[str, bool]:
        """获取模型的能力声明。

        Args:
            model_name: 外部模型名称。

        Returns:
            能力字典，例如 {"tool_use": True, "vision": True, ...}。
            未注册的模型返回基础能力集。
        """
        return self.model_capabilities.get(model_name, {
            "tool_use": True,
            "vision": False,
            "web_search": False,
            "thinking": False,
        })

    def get_supported_models(self) -> List[str]:
        """获取对外暴露的支持模型列表。

        Returns:
            支持的模型名称列表。
        """
        return [
            settings.GLM45_MODEL,
            settings.GLM45_THINKING_MODEL,
            settings.GLM45_SEARCH_MODEL,
            settings.GLM45_AIR_MODEL,
            settings.GLM46V_MODEL,
            settings.GLM46V_ADVANCED_SEARCH_MODEL,
            settings.GLM5_MODEL,
            settings.GLM5_THINKING_MODEL,
            settings.GLM5_AGENT_MODEL,
            settings.GLM5_ADVANCED_SEARCH_MODEL,
            settings.GLM47_MODEL,
            settings.GLM47_THINKING_MODEL,
            settings.GLM47_SEARCH_MODEL,
            settings.GLM47_ADVANCED_SEARCH_MODEL,
        ]

    def resolve_model_features(self, request: OpenAIRequest) -> Dict[str, Any]:
        """解析请求的模型特性参数。

        按优先级（客户端显式参数 > 场景默认 > 模型后缀推断）解析以下特性：
        - ``enable_thinking``
        - ``web_search``
        - ``auto_web_search``
        - ``flags``
        - ``extra``
        - ``mcp_servers``
        - ``upstream_model_id``

        Args:
            request: OpenAI 请求对象。

        Returns:
            包含所有已解析模型特性的字典。
        """
        requested_model = request.model
        is_thinking_model = "-thinking" in requested_model.casefold()
        is_search_model = "-search" in requested_model.casefold()

        scene_defaults = self.get_scene_defaults(requested_model)

        # enable_thinking: 客户端 > 场景默认 > 模型后缀判断
        enable_thinking = request.enable_thinking
        if enable_thinking is None:
            enable_thinking = scene_defaults.get("enable_thinking", is_thinking_model)

        # web_search: 客户端 > 场景默认 > 模型后缀推断 (默认已包含在 is_search_model 中，非搜索模型即为 False)
        web_search = request.web_search
        if web_search is None:
            web_search = scene_defaults.get("web_search", is_search_model)

        # auto_web_search: 场景默认 > False
        auto_web_search = scene_defaults.get("auto_web_search", False)

        # 获取上游模型 ID
        upstream_model_id = self.get_upstream_model_id(requested_model)
        if upstream_model_id is None:
            raise ValueError(f"不支持的模型: {requested_model}")

        # mcp_servers: 客户端透传优先
        mcp_servers = (
            request.mcp_servers
            if request.mcp_servers is not None
            else self.get_mcp_servers(requested_model)
        )

        # extra 和 flags: 客户端 > 场景默认 > 空值
        extra = request.extra if request.extra is not None else scene_defaults.get("extra", {})
        flags = request.flags if request.flags is not None else scene_defaults.get("flags", [])

        return {
            "upstream_model_id": upstream_model_id,
            "enable_thinking": enable_thinking,
            "web_search": web_search,
            "auto_web_search": auto_web_search,
            "mcp_servers": mcp_servers,
            "extra": extra,
            "flags": flags,
        }
