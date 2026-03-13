#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Model management module.

Centrally manages model mapping, MCP server configuration, scene defaults,
model capabilities, and model feature resolution logic.  Supports both
hardcoded fallback data and dynamic parsing from online model metadata.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.models.schemas import OpenAIRequest

logger = logging.getLogger(__name__)


@dataclass
class ParsedModel:
    """Structured representation of an online model after parsing."""

    upstream_id: str
    display_name: str
    capabilities: Dict[str, Any] = field(default_factory=dict)
    mcp_server_ids: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    tags: List[str] = field(default_factory=list)


class ModelManager:
    """Model configuration manager.

    Manages four configuration dictionaries:
    1. ``model_mapping``       -- external model name -> upstream model ID
    2. ``model_mcp_servers``   -- model -> default MCP server list
    3. ``model_scene_defaults``-- model -> scene default parameters
    4. ``model_capabilities``  -- model -> capability declarations

    On startup, hardcoded defaults are loaded.  When
    :meth:`load_from_online_models` is called with fresh upstream data the
    hardcoded values are replaced with dynamically generated variants.
    """

    def __init__(self) -> None:
        self._init_hardcoded_defaults()
        self._parsed: bool = False  # True after successful online parse
        # Apply aliases to hardcoded defaults (online path re-applies after parse)
        self.apply_aliases()

    # ------------------------------------------------------------------
    # Hardcoded fallback (kept for offline / pre-fetch scenarios)
    # ------------------------------------------------------------------

    def _init_hardcoded_defaults(self) -> None:
        """Initialise all model dictionaries with hardcoded defaults."""

        # External model name -> upstream model ID
        self.model_mapping: Dict[str, str] = {
            settings.GLM45_MODEL: "0727-360B-API",
            settings.GLM45_THINKING_MODEL: "0727-360B-API",
            settings.GLM45_SEARCH_MODEL: "0727-360B-API",
            settings.GLM45_AIR_MODEL: "0727-106B-API",
            settings.GLM46V_MODEL: "glm-4.6v",
            settings.GLM46V_ADVANCED_SEARCH_MODEL: "glm-4.6v",
            settings.GLM5_MODEL: "glm-5",
            settings.GLM5_THINKING_MODEL: "glm-5",
            settings.GLM5_AGENT_MODEL: "glm-5",
            settings.GLM5_ADVANCED_SEARCH_MODEL: "glm-5",
            settings.GLM47_MODEL: "glm-4.7",
            settings.GLM47_THINKING_MODEL: "glm-4.7",
            settings.GLM47_SEARCH_MODEL: "glm-4.7",
            settings.GLM47_ADVANCED_SEARCH_MODEL: "glm-4.7",
        }

        # Default MCP servers per model
        self.model_mcp_servers: Dict[str, List[str]] = {
            settings.GLM5_ADVANCED_SEARCH_MODEL: ["advanced-search"],
            settings.GLM46V_ADVANCED_SEARCH_MODEL: ["advanced-search"],
            settings.GLM47_ADVANCED_SEARCH_MODEL: ["advanced-search"],
        }

        # Scene defaults per model
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

        # Model capability declarations
        self.model_capabilities: Dict[str, Dict[str, bool]] = {
            settings.GLM45_MODEL: {
                "tool_use": True, "vision": False,
                "web_search": False, "thinking": False,
            },
            settings.GLM45_THINKING_MODEL: {
                "tool_use": True, "vision": False,
                "web_search": False, "thinking": True,
            },
            settings.GLM45_SEARCH_MODEL: {
                "tool_use": True, "vision": False,
                "web_search": True, "thinking": False,
            },
            settings.GLM45_AIR_MODEL: {
                "tool_use": True, "vision": False,
                "web_search": False, "thinking": False,
            },
            settings.GLM46V_MODEL: {
                "tool_use": True, "vision": True,
                "web_search": False, "thinking": False,
            },
            settings.GLM46V_ADVANCED_SEARCH_MODEL: {
                "tool_use": True, "vision": True,
                "web_search": True, "thinking": True,
            },
            settings.GLM5_MODEL: {
                "tool_use": True, "vision": False,
                "web_search": False, "thinking": False,
            },
            settings.GLM5_THINKING_MODEL: {
                "tool_use": True, "vision": False,
                "web_search": False, "thinking": True,
            },
            settings.GLM5_AGENT_MODEL: {
                "tool_use": True, "vision": False,
                "web_search": False, "thinking": True,
                "agent": True,
            },
            settings.GLM5_ADVANCED_SEARCH_MODEL: {
                "tool_use": True, "vision": False,
                "web_search": True, "thinking": True,
            },
            settings.GLM47_MODEL: {
                "tool_use": True, "vision": False,
                "web_search": False, "thinking": False,
            },
            settings.GLM47_THINKING_MODEL: {
                "tool_use": True, "vision": False,
                "web_search": False, "thinking": True,
            },
            settings.GLM47_SEARCH_MODEL: {
                "tool_use": True, "vision": False,
                "web_search": True, "thinking": False,
            },
            settings.GLM47_ADVANCED_SEARCH_MODEL: {
                "tool_use": True, "vision": False,
                "web_search": True, "thinking": True,
            },
        }

        # Hardcoded supported model list (fallback order)
        self._hardcoded_supported_models: List[str] = [
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

        # Will be populated by load_from_online_models
        self._dynamic_supported_models: List[str] = []

    # ------------------------------------------------------------------
    # Online model parsing
    # ------------------------------------------------------------------

    def load_from_online_models(self, online_models: List[Dict[str, Any]]) -> None:
        """Parse online model metadata and generate variant configurations.

        For each non-blacklisted active model, up to five variants are
        generated (base, Thinking, Search, Agent, advanced-search) depending
        on the upstream capability flags.  The four core dictionaries are
        fully replaced on success.

        Args:
            online_models: List of model dicts as returned by
                :meth:`upstream.UpstreamClient.get_online_models`.
        """
        # Read blacklist from settings (comma-separated upstream IDs)
        blacklist_raw = getattr(settings, "MODEL_BLACKLIST", "glm-4-flash")
        blacklist = {
            item.strip().lower()
            for item in blacklist_raw.split(",")
            if item.strip()
        }

        # Parse each raw dict into a ParsedModel
        parsed: List[ParsedModel] = []
        skipped_inactive = 0
        skipped_blacklisted = 0

        for raw in online_models:
            upstream_id = raw.get("id", "")
            if not upstream_id:
                continue

            # Blacklist check (case-insensitive on upstream ID)
            if upstream_id.lower() in blacklist:
                skipped_blacklisted += 1
                continue

            is_active = raw.get("is_active", True)
            if not is_active:
                skipped_inactive += 1
                continue

            display_name = raw.get("name") or upstream_id
            capabilities = raw.get("capabilities") or {}
            mcp_server_ids = raw.get("mcpServerIds") or []
            tags = raw.get("tags") or []

            parsed.append(ParsedModel(
                upstream_id=upstream_id,
                display_name=display_name,
                capabilities=capabilities,
                mcp_server_ids=mcp_server_ids,
                params={},
                is_active=is_active,
                tags=tags,
            ))

        # Build fresh dictionaries from parsed models
        new_mapping: Dict[str, str] = {}
        new_capabilities: Dict[str, Dict[str, bool]] = {}
        new_mcp_servers: Dict[str, List[str]] = {}
        new_scene_defaults: Dict[str, Dict[str, Any]] = {}
        new_supported: List[str] = []

        total_variants = 0

        for model in parsed:
            variants = self._generate_variants(model)
            for v_name, v_caps, v_scene, v_mcp in variants:
                new_mapping[v_name] = model.upstream_id
                new_capabilities[v_name] = v_caps
                if v_scene:
                    new_scene_defaults[v_name] = v_scene
                if v_mcp:
                    new_mcp_servers[v_name] = v_mcp
                new_supported.append(v_name)
                total_variants += 1

        # Atomically replace all dictionaries
        self.model_mapping = new_mapping
        self.model_capabilities = new_capabilities
        self.model_mcp_servers = new_mcp_servers
        self.model_scene_defaults = new_scene_defaults
        self._dynamic_supported_models = new_supported
        self._parsed = True

        logger.info(
            "Online model parsing complete: %d models parsed, "
            "%d variants generated, %d blacklisted, %d inactive",
            len(parsed), total_variants,
            skipped_blacklisted, skipped_inactive,
        )

        # Apply user-defined aliases on top of generated variants
        self.apply_aliases()

    @staticmethod
    def _generate_variants(
        model: ParsedModel,
    ) -> List[tuple]:
        """Generate variant tuples for a single parsed model.

        Returns:
            List of (variant_name, capabilities_dict, scene_defaults_dict,
            mcp_servers_list) tuples.
        """
        caps = model.capabilities
        name = model.display_name
        name_lower = name.lower()
        mcp_ids = model.mcp_server_ids

        # Derive base capability flags from upstream capabilities
        has_return_fc = bool(caps.get("returnFc", False))
        has_vision = bool(caps.get("vision", False))
        has_think = bool(caps.get("think", False))
        has_web_search = bool(caps.get("web_search", False))
        has_agent_mode = bool(caps.get("agent_mode", False))
        has_deep_web_search = "deep-web-search" in mcp_ids
        has_advanced_search = "advanced-search" in mcp_ids

        base_caps: Dict[str, bool] = {
            "tool_use": has_return_fc,
            "vision": has_vision,
        }

        variants: List[tuple] = []

        # --- 1. Base variant (always generated) ---
        variants.append((
            name,
            {**base_caps, "web_search": False, "thinking": False},
            {},  # no scene defaults for base
            [],  # no MCP servers for base
        ))

        # --- 2. Thinking variant ---
        if has_think and "thinking" not in name_lower:
            thinking_name = f"{name}-Thinking"
            variants.append((
                thinking_name,
                {**base_caps, "web_search": False, "thinking": True},
                {"enable_thinking": True},
                [],
            ))

        # --- 3. Search variant ---
        if (has_web_search or has_deep_web_search) and "search" not in name_lower:
            search_name = f"{name}-Search"
            variants.append((
                search_name,
                {**base_caps, "web_search": True, "thinking": False},
                {"web_search": True, "auto_web_search": True},
                [],
            ))

        # --- 4. Agent variant ---
        if has_agent_mode and "agent" not in name_lower:
            agent_name = f"{name}-Agent"
            variants.append((
                agent_name,
                {**base_caps, "web_search": False, "thinking": True, "agent": True},
                {"flags": ["general_agent"], "enable_thinking": True},
                [],
            ))

        # --- 5. advanced-search variant ---
        if has_advanced_search and "advanced-search" not in name_lower:
            adv_search_name = f"{name}-advanced-search"
            variants.append((
                adv_search_name,
                {**base_caps, "web_search": True, "thinking": True},
                {"enable_thinking": True, "web_search": True, "auto_web_search": True},
                ["advanced-search"],
            ))

        return variants

    # ------------------------------------------------------------------
    # Alias mapping
    # ------------------------------------------------------------------

    def apply_aliases(self, aliases_str: Optional[str] = None) -> None:
        """Apply model aliases from a comma-separated ``alias=target`` string.

        Each alias inherits the target model's upstream mapping, capabilities,
        MCP servers, and scene defaults.  Unknown targets are silently skipped.

        Args:
            aliases_str: Alias definitions.  Falls back to
                ``settings.MODEL_ALIASES`` when *None*.
        """
        if aliases_str is None:
            aliases_str = getattr(settings, "MODEL_ALIASES", "")
        if not aliases_str or not aliases_str.strip():
            return

        applied = 0
        for pair in aliases_str.split(","):
            pair = pair.strip()
            if "=" not in pair:
                continue
            alias, target = pair.split("=", 1)
            alias = alias.strip()
            target = target.strip()
            if not alias or not target:
                continue

            # Target must exist in current mapping
            if target not in self.model_mapping:
                logger.warning(
                    "Alias target %r not found in model mapping, "
                    "skipping alias %r",
                    target, alias,
                )
                continue

            # Copy all properties from target
            self.model_mapping[alias] = self.model_mapping[target]
            if target in self.model_capabilities:
                self.model_capabilities[alias] = dict(self.model_capabilities[target])
            if target in self.model_mcp_servers:
                self.model_mcp_servers[alias] = list(self.model_mcp_servers[target])
            if target in self.model_scene_defaults:
                self.model_scene_defaults[alias] = dict(self.model_scene_defaults[target])

            # Add to supported models list
            if self._parsed and alias not in self._dynamic_supported_models:
                self._dynamic_supported_models.append(alias)
            elif not self._parsed and alias not in self._hardcoded_supported_models:
                self._hardcoded_supported_models.append(alias)

            applied += 1

        if applied:
            logger.info("Applied %d model alias(es)", applied)

    # ------------------------------------------------------------------
    # Public query API (signatures unchanged)
    # ------------------------------------------------------------------

    def get_upstream_model_id(self, model_name: str) -> Optional[str]:
        """Return the upstream model ID for *model_name*, or None."""
        return self.model_mapping.get(model_name)

    def get_mcp_servers(self, model_name: str) -> List[str]:
        """Return the default MCP server list for *model_name*."""
        return self.model_mcp_servers.get(model_name, [])

    def get_scene_defaults(self, model_name: str) -> Dict[str, Any]:
        """Return scene default parameters for *model_name*."""
        return self.model_scene_defaults.get(model_name, {})

    def get_model_capabilities(self, model_name: str) -> Dict[str, bool]:
        """Return capability declarations for *model_name*.

        Unregistered models receive a minimal base capability set.
        """
        return self.model_capabilities.get(model_name, {
            "tool_use": True,
            "vision": False,
            "web_search": False,
            "thinking": False,
        })

    def get_supported_models(self) -> List[str]:
        """Return the list of externally exposed supported models.

        Uses dynamically generated list when online parsing has succeeded,
        otherwise falls back to the hardcoded list.
        """
        if self._parsed and self._dynamic_supported_models:
            return list(self._dynamic_supported_models)
        return list(self._hardcoded_supported_models)

    # ------------------------------------------------------------------
    # Feature resolution (signature unchanged)
    # ------------------------------------------------------------------

    def resolve_model_features(self, request: OpenAIRequest) -> Dict[str, Any]:
        """Resolve model feature parameters from request context.

        Priority order: explicit client parameters > scene defaults >
        model-suffix inference.

        Args:
            request: OpenAI-compatible request object.

        Returns:
            Dict with keys: upstream_model_id, enable_thinking, web_search,
            auto_web_search, mcp_servers, extra, flags.
        """
        requested_model = request.model
        is_thinking_model = "-thinking" in requested_model.casefold()
        is_search_model = "-search" in requested_model.casefold()

        scene_defaults = self.get_scene_defaults(requested_model)

        # enable_thinking: client > scene default > suffix inference
        enable_thinking = request.enable_thinking
        if enable_thinking is None:
            enable_thinking = scene_defaults.get("enable_thinking", is_thinking_model)

        # web_search: client > scene default > suffix inference
        web_search = request.web_search
        if web_search is None:
            web_search = scene_defaults.get("web_search", is_search_model)

        # auto_web_search: scene default > False
        auto_web_search = scene_defaults.get("auto_web_search", False)

        # Upstream model ID
        upstream_model_id = self.get_upstream_model_id(requested_model)
        if upstream_model_id is None:
            raise ValueError(f"Unsupported model: {requested_model}")

        # mcp_servers: client pass-through takes priority
        mcp_servers = (
            request.mcp_servers
            if request.mcp_servers is not None
            else self.get_mcp_servers(requested_model)
        )

        # extra and flags: client > scene default > empty
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
