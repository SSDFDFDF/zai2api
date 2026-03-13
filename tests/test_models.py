#!/usr/bin/env python
"""Tests for ModelManager online model parsing and variant generation."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.models import ModelManager, ParsedModel


# ---------------------------------------------------------------------------
# Fixtures: simulated parsed model data (same format as upstream.py produces)
# ---------------------------------------------------------------------------

def _make_model(
    model_id: str,
    name: str,
    *,
    think: bool = False,
    web_search: bool = False,
    agent_mode: bool = False,
    returnFc: bool = True,
    vision: bool = False,
    mcp_server_ids: list | None = None,
    is_active: bool = True,
    tags: list | None = None,
) -> dict:
    caps = {
        "vision": vision,
        "web_search": web_search,
        "returnFc": returnFc,
        "think": think,
    }
    if agent_mode:
        caps["agent_mode"] = True
    return {
        "id": model_id,
        "name": name,
        "is_active": is_active,
        "capabilities": caps,
        "mcpServerIds": mcp_server_ids or [],
        "tags": tags or [],
    }


SAMPLE_GLM5 = _make_model(
    "glm-5", "GLM-5",
    think=True,
    web_search=True,
    agent_mode=True,
    returnFc=True,
    mcp_server_ids=[
        "deep-web-search", "ppt-maker", "vibe-coding",
        "image-search", "deep-research", "advanced-search",
    ],
    tags=["New"],
)

SAMPLE_GLM47 = _make_model(
    "glm-4.7", "GLM-4.7",
    think=True,
    web_search=True,
    agent_mode=False,
    returnFc=True,
    mcp_server_ids=[
        "deep-web-search", "ppt-maker", "vibe-coding",
        "image-search", "deep-research", "advanced-search",
    ],
)

SAMPLE_GLM46V = _make_model(
    "glm-4.6v", "GLM-4.6V",
    think=True,
    vision=True,
    web_search=False,
    returnFc=True,
    mcp_server_ids=[
        "shopping-search", "vlm-image-search",
        "vlm-image-processing", "vlm-image-recognition",
    ],
)

SAMPLE_GLM4_FLASH = _make_model(
    "glm-4-flash", "任务专用",
    returnFc=False,
)

SAMPLE_INACTIVE = _make_model(
    "test-inactive", "Inactive Model",
    is_active=False,
)


# ---------------------------------------------------------------------------
# Tests: _generate_variants
# ---------------------------------------------------------------------------

class TestGenerateVariants:
    """Test variant generation for individual models."""

    def test_glm5_generates_all_five_variants(self):
        parsed = ParsedModel(
            upstream_id="glm-5",
            display_name="GLM-5",
            capabilities=SAMPLE_GLM5["capabilities"],
            mcp_server_ids=SAMPLE_GLM5["mcpServerIds"],
        )
        variants = ModelManager._generate_variants(parsed)
        names = [v[0] for v in variants]
        assert names == [
            "GLM-5",
            "GLM-5-Thinking",
            "GLM-5-Search",
            "GLM-5-Agent",
            "GLM-5-advanced-search",
        ]

    def test_glm47_no_agent_variant(self):
        parsed = ParsedModel(
            upstream_id="glm-4.7",
            display_name="GLM-4.7",
            capabilities=SAMPLE_GLM47["capabilities"],
            mcp_server_ids=SAMPLE_GLM47["mcpServerIds"],
        )
        variants = ModelManager._generate_variants(parsed)
        names = [v[0] for v in variants]
        assert "GLM-4.7-Agent" not in names
        assert "GLM-4.7" in names
        assert "GLM-4.7-Thinking" in names
        assert "GLM-4.7-Search" in names
        assert "GLM-4.7-advanced-search" in names

    def test_glm46v_vision_model_no_search_or_agent(self):
        parsed = ParsedModel(
            upstream_id="glm-4.6v",
            display_name="GLM-4.6V",
            capabilities=SAMPLE_GLM46V["capabilities"],
            mcp_server_ids=SAMPLE_GLM46V["mcpServerIds"],
        )
        variants = ModelManager._generate_variants(parsed)
        names = [v[0] for v in variants]
        assert "GLM-4.6V" in names
        assert "GLM-4.6V-Thinking" in names
        # no deep-web-search or web_search=True → no Search variant
        assert "GLM-4.6V-Search" not in names
        # no agent_mode → no Agent variant
        assert "GLM-4.6V-Agent" not in names
        # no advanced-search in mcp → no advanced-search variant
        assert "GLM-4.6V-advanced-search" not in names

    def test_base_variant_always_generated(self):
        parsed = ParsedModel(
            upstream_id="basic",
            display_name="Basic",
            capabilities={"returnFc": False, "vision": False},
            mcp_server_ids=[],
        )
        variants = ModelManager._generate_variants(parsed)
        assert len(variants) == 1
        assert variants[0][0] == "Basic"

    def test_no_duplicate_thinking_suffix(self):
        """Model already named with Thinking should not get -Thinking variant."""
        parsed = ParsedModel(
            upstream_id="GLM-4.1V-Thinking-FlashX",
            display_name="GLM-4.1V-9B-Thinking",
            capabilities={"returnFc": True, "vision": True, "think": True},
            mcp_server_ids=[],
        )
        variants = ModelManager._generate_variants(parsed)
        names = [v[0] for v in variants]
        # Should NOT generate "GLM-4.1V-9B-Thinking-Thinking"
        assert not any(n.endswith("-Thinking") and n != "GLM-4.1V-9B-Thinking" for n in names)


# ---------------------------------------------------------------------------
# Tests: variant capabilities mapping
# ---------------------------------------------------------------------------

class TestVariantCapabilities:
    """Test that capabilities are correctly set for each variant type."""

    def _get_variants_dict(self, model_dict: dict) -> dict:
        parsed = ParsedModel(
            upstream_id=model_dict["id"],
            display_name=model_dict["name"],
            capabilities=model_dict["capabilities"],
            mcp_server_ids=model_dict["mcpServerIds"],
        )
        variants = ModelManager._generate_variants(parsed)
        return {v[0]: {"caps": v[1], "scene": v[2], "mcp": v[3]} for v in variants}

    def test_base_caps(self):
        vd = self._get_variants_dict(SAMPLE_GLM5)
        caps = vd["GLM-5"]["caps"]
        assert caps["tool_use"] is True
        assert caps["vision"] is False
        assert caps["web_search"] is False
        assert caps["thinking"] is False

    def test_thinking_caps(self):
        vd = self._get_variants_dict(SAMPLE_GLM5)
        caps = vd["GLM-5-Thinking"]["caps"]
        assert caps["thinking"] is True
        assert caps["web_search"] is False
        scene = vd["GLM-5-Thinking"]["scene"]
        assert scene["enable_thinking"] is True

    def test_search_caps(self):
        vd = self._get_variants_dict(SAMPLE_GLM5)
        caps = vd["GLM-5-Search"]["caps"]
        assert caps["web_search"] is True
        assert caps["thinking"] is False
        scene = vd["GLM-5-Search"]["scene"]
        assert scene["web_search"] is True
        assert scene["auto_web_search"] is True

    def test_agent_caps(self):
        vd = self._get_variants_dict(SAMPLE_GLM5)
        caps = vd["GLM-5-Agent"]["caps"]
        assert caps["thinking"] is True
        assert caps["agent"] is True
        scene = vd["GLM-5-Agent"]["scene"]
        assert scene["flags"] == ["general_agent"]
        assert scene["enable_thinking"] is True

    def test_advanced_search_caps(self):
        vd = self._get_variants_dict(SAMPLE_GLM5)
        caps = vd["GLM-5-advanced-search"]["caps"]
        assert caps["web_search"] is True
        assert caps["thinking"] is True
        scene = vd["GLM-5-advanced-search"]["scene"]
        assert scene["enable_thinking"] is True
        assert scene["web_search"] is True
        mcp = vd["GLM-5-advanced-search"]["mcp"]
        assert mcp == ["advanced-search"]

    def test_vision_flag(self):
        vd = self._get_variants_dict(SAMPLE_GLM46V)
        caps = vd["GLM-4.6V"]["caps"]
        assert caps["vision"] is True


# ---------------------------------------------------------------------------
# Tests: load_from_online_models (full pipeline)
# ---------------------------------------------------------------------------

class TestLoadFromOnlineModels:
    """Test the full parsing pipeline including blacklist and mapping."""

    def _make_manager(self) -> ModelManager:
        return ModelManager()

    def test_basic_load(self):
        mgr = self._make_manager()
        mgr.load_from_online_models([SAMPLE_GLM5])
        supported = mgr.get_supported_models()
        assert "GLM-5" in supported
        assert "GLM-5-Thinking" in supported
        assert "GLM-5-Agent" in supported

    def test_mapping_all_point_to_upstream_id(self):
        mgr = self._make_manager()
        mgr.load_from_online_models([SAMPLE_GLM5])
        for model_name in mgr.get_supported_models():
            assert mgr.get_upstream_model_id(model_name) == "glm-5"

    def test_blacklist_filters_model(self):
        mgr = self._make_manager()
        mgr.load_from_online_models([SAMPLE_GLM5, SAMPLE_GLM4_FLASH])
        supported = mgr.get_supported_models()
        # glm-4-flash should be filtered by default blacklist
        assert "任务专用" not in supported
        assert "GLM-5" in supported

    def test_inactive_model_filtered(self):
        mgr = self._make_manager()
        mgr.load_from_online_models([SAMPLE_GLM5, SAMPLE_INACTIVE])
        supported = mgr.get_supported_models()
        assert "Inactive Model" not in supported

    def test_parsed_flag(self):
        mgr = self._make_manager()
        assert mgr._parsed is False
        mgr.load_from_online_models([SAMPLE_GLM5])
        assert mgr._parsed is True

    def test_dynamic_overrides_hardcoded(self):
        mgr = self._make_manager()
        # Before: hardcoded defaults
        old_supported = mgr.get_supported_models()
        mgr.load_from_online_models([SAMPLE_GLM5])
        new_supported = mgr.get_supported_models()
        assert old_supported != new_supported

    def test_get_model_capabilities_for_unknown_model(self):
        mgr = self._make_manager()
        mgr.load_from_online_models([SAMPLE_GLM5])
        caps = mgr.get_model_capabilities("nonexistent-model")
        assert caps["tool_use"] is True
        assert caps["thinking"] is False

    def test_mcp_servers(self):
        mgr = self._make_manager()
        mgr.load_from_online_models([SAMPLE_GLM5])
        assert mgr.get_mcp_servers("GLM-5") == []
        assert mgr.get_mcp_servers("GLM-5-advanced-search") == ["advanced-search"]

    def test_scene_defaults(self):
        mgr = self._make_manager()
        mgr.load_from_online_models([SAMPLE_GLM5])
        assert mgr.get_scene_defaults("GLM-5") == {}
        sd = mgr.get_scene_defaults("GLM-5-Thinking")
        assert sd["enable_thinking"] is True

    def test_multiple_models(self):
        mgr = self._make_manager()
        mgr.load_from_online_models([
            SAMPLE_GLM5, SAMPLE_GLM47, SAMPLE_GLM46V,
        ])
        supported = mgr.get_supported_models()
        assert "GLM-5" in supported
        assert "GLM-4.7" in supported
        assert "GLM-4.6V" in supported
        # GLM-5 variants
        assert "GLM-5-Agent" in supported
        # GLM-4.7 no agent
        assert "GLM-4.7-Agent" not in supported
        # GLM-4.6V no search
        assert "GLM-4.6V-Search" not in supported


# ---------------------------------------------------------------------------
# Tests: load from real models.json (integration)
# ---------------------------------------------------------------------------

class TestLoadFromModelsJson:
    """Integration test using real models.json from project root."""

    @pytest.fixture
    def parsed_from_json(self):
        """Load models.json and simulate upstream.py parsing."""
        json_path = Path(__file__).resolve().parent.parent / "models.json"
        if not json_path.exists():
            pytest.skip("models.json not found")
        with open(json_path) as f:
            data = json.load(f)
        parsed = []
        for item in data.get("data", []):
            model_id = item.get("id")
            if not model_id:
                continue
            info = item.get("info", {})
            meta = info.get("meta") or {}
            raw_tags = meta.get("tags") or []
            tags = [
                t.get("name") for t in raw_tags
                if isinstance(t, dict) and t.get("name")
            ]
            parsed.append({
                "id": model_id,
                "name": info.get("name") or item.get("name") or model_id,
                "is_active": info.get("is_active", True),
                "capabilities": meta.get("capabilities") or {},
                "mcpServerIds": meta.get("mcpServerIds") or [],
                "tags": tags,
            })
        return parsed

    def test_all_active_non_blacklisted_models_present(self, parsed_from_json):
        mgr = ModelManager()
        mgr.load_from_online_models(parsed_from_json)
        supported = mgr.get_supported_models()

        # glm-4-flash should be blacklisted
        for name in supported:
            assert mgr.get_upstream_model_id(name) != "glm-4-flash"

        # GLM-5 should have all 5 variants
        assert "GLM-5" in supported
        assert "GLM-5-Thinking" in supported
        assert "GLM-5-Search" in supported
        assert "GLM-5-Agent" in supported
        assert "GLM-5-advanced-search" in supported

    def test_variant_count_reasonable(self, parsed_from_json):
        mgr = ModelManager()
        mgr.load_from_online_models(parsed_from_json)
        supported = mgr.get_supported_models()
        # 12 non-blacklisted models → each with 1-5 variants
        assert len(supported) >= 12
        assert len(supported) <= 60

    def test_every_model_has_mapping(self, parsed_from_json):
        mgr = ModelManager()
        mgr.load_from_online_models(parsed_from_json)
        for name in mgr.get_supported_models():
            assert mgr.get_upstream_model_id(name) is not None, f"{name} has no mapping"


# ---------------------------------------------------------------------------
# Tests: alias mapping
# ---------------------------------------------------------------------------

class TestAliasMapping:
    """Test model alias functionality."""

    def test_alias_inherits_all_properties(self):
        mgr = ModelManager()
        mgr.load_from_online_models([SAMPLE_GLM5])
        mgr.apply_aliases("gpt-4o=GLM-5-Thinking")

        assert "gpt-4o" in mgr.get_supported_models()
        assert mgr.get_upstream_model_id("gpt-4o") == "glm-5"
        caps = mgr.get_model_capabilities("gpt-4o")
        assert caps["thinking"] is True
        scene = mgr.get_scene_defaults("gpt-4o")
        assert scene["enable_thinking"] is True

    def test_alias_to_base_model(self):
        mgr = ModelManager()
        mgr.load_from_online_models([SAMPLE_GLM5])
        mgr.apply_aliases("my-model=GLM-5")

        assert "my-model" in mgr.get_supported_models()
        assert mgr.get_upstream_model_id("my-model") == "glm-5"
        assert mgr.get_scene_defaults("my-model") == {}
        assert mgr.get_mcp_servers("my-model") == []

    def test_alias_to_advanced_search(self):
        mgr = ModelManager()
        mgr.load_from_online_models([SAMPLE_GLM5])
        mgr.apply_aliases("deep=GLM-5-advanced-search")

        assert mgr.get_mcp_servers("deep") == ["advanced-search"]
        caps = mgr.get_model_capabilities("deep")
        assert caps["web_search"] is True
        assert caps["thinking"] is True

    def test_multiple_aliases(self):
        mgr = ModelManager()
        mgr.load_from_online_models([SAMPLE_GLM5])
        mgr.apply_aliases("gpt-4o=GLM-5,gpt-4o-mini=GLM-5-Thinking")

        assert "gpt-4o" in mgr.get_supported_models()
        assert "gpt-4o-mini" in mgr.get_supported_models()
        assert mgr.get_upstream_model_id("gpt-4o") == "glm-5"
        assert mgr.get_upstream_model_id("gpt-4o-mini") == "glm-5"

    def test_alias_unknown_target_skipped(self):
        mgr = ModelManager()
        mgr.load_from_online_models([SAMPLE_GLM5])
        mgr.apply_aliases("foo=nonexistent-model")

        assert "foo" not in mgr.get_supported_models()

    def test_empty_aliases_no_effect(self):
        mgr = ModelManager()
        mgr.load_from_online_models([SAMPLE_GLM5])
        before = mgr.get_supported_models()
        mgr.apply_aliases("")
        mgr.apply_aliases(None)
        assert mgr.get_supported_models() == before

    def test_alias_no_duplicates_in_supported(self):
        mgr = ModelManager()
        mgr.load_from_online_models([SAMPLE_GLM5])
        mgr.apply_aliases("gpt-4o=GLM-5")
        mgr.apply_aliases("gpt-4o=GLM-5")  # apply twice
        supported = mgr.get_supported_models()
        assert supported.count("gpt-4o") == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
