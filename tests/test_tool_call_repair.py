#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for XML/JSON Wildcard Repair Pipeline in tool_call_handler.
"""

import sys
import os
# 将项目根目录加入 sys.path, 允许直接 python 运行
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.utils.tool_call_handler import (
    normalize_cdata_markers,
    normalize_xml_tag_names,
    normalize_xml_structure,
    repair_json_payload,
    _is_xml_noise,
    _extract_cdata_text,
    _parse_args_json_payload,
    parse_function_calls_xml,
    looks_like_complete_function_calls,
)


# ===========================================================================
# 1. normalize_cdata_markers
# ===========================================================================

class TestNormalizeCdataMarkers:
    def test_normal_cdata_unchanged(self):
        raw = '<![CDATA[{"a":1}]]>'
        assert normalize_cdata_markers(raw) == raw

    def test_duplicated_keyword(self):
        """<![CDATACDATA[ → <![CDATA[  (报告的 bug)"""
        raw = '<![CDATACDATA[{"a":1}]]>'
        assert normalize_cdata_markers(raw) == '<![CDATA[{"a":1}]]>'

    def test_triple_keyword(self):
        """<![CDATACDATACDATA[ → <![CDATA["""
        raw = '<![CDATACDATACDATA[{"a":1}]]>'
        assert normalize_cdata_markers(raw) == '<![CDATA[{"a":1}]]>'

    def test_spaces_in_cdata(self):
        """<![ CDATA [ → <![CDATA["""
        raw = '<![ CDATA [{"a":1}]]>'
        assert normalize_cdata_markers(raw) == '<![CDATA[{"a":1}]]>'

    def test_mixed_case_cdata(self):
        """<![CData[ → <![CDATA["""
        raw = '<![CData[{"a":1}]]>'
        assert normalize_cdata_markers(raw) == '<![CDATA[{"a":1}]]>'

    def test_space_in_close(self):
        """]] > → ]]>"""
        raw = '<![CDATA[{"a":1}] ] >'
        result = normalize_cdata_markers(raw)
        assert result == '<![CDATA[{"a":1}]]>'

    def test_none_input(self):
        assert normalize_cdata_markers(None) == ""

    def test_no_cdata(self):
        raw = '{"a":1}'
        assert normalize_cdata_markers(raw) == raw

    def test_partial_keyword_with_space(self):
        """<![CDATA CDATA[ → <![CDATA["""
        raw = '<![CDATA CDATA[{"a":1}]]>'
        assert normalize_cdata_markers(raw) == '<![CDATA[{"a":1}]]>'


# ===========================================================================
# 2. normalize_xml_tag_names
# ===========================================================================

class TestNormalizeXmlTagNames:
    def test_normal_tags_unchanged(self):
        xml = '<function_calls><function_call><tool>Glob</tool><args_json>{}</args_json></function_call></function_calls>'
        assert normalize_xml_tag_names(xml) == xml

    def test_uppercase_tags(self):
        xml = '<FUNCTION_CALLS><FUNCTION_CALL><TOOL>Glob</TOOL></FUNCTION_CALL></FUNCTION_CALLS>'
        result = normalize_xml_tag_names(xml)
        assert '<function_calls>' in result
        assert '<function_call>' in result
        assert '<tool>' in result
        assert '</tool>' in result

    def test_hyphen_tags(self):
        xml = '<function-calls><function-call><tool>X</tool></function-call></function-calls>'
        result = normalize_xml_tag_names(xml)
        assert '<function_calls>' in result
        assert '<function_call>' in result

    def test_space_in_tag(self):
        xml = '< tool >Glob</ tool >'
        result = normalize_xml_tag_names(xml)
        assert '<tool>' in result
        assert '</tool>' in result

    def test_no_separator_tags(self):
        xml = '<functioncalls><functioncall><argsjson>{}</argsjson></functioncall></functioncalls>'
        result = normalize_xml_tag_names(xml)
        assert '<function_calls>' in result
        assert '<function_call>' in result
        assert '<args_json>' in result


# ===========================================================================
# 3. repair_json_payload
# ===========================================================================

class TestRepairJsonPayload:
    def test_normal_json_unchanged(self):
        s = '{"pattern": "**/*.py"}'
        assert repair_json_payload(s) == s

    def test_trailing_comma_object(self):
        assert repair_json_payload('{"a":1,}') == '{"a":1}'

    def test_trailing_comma_array(self):
        assert repair_json_payload('[1,2,3,]') == '[1,2,3]'

    def test_python_true(self):
        assert repair_json_payload('{"flag": True}') == '{"flag": true}'

    def test_python_false(self):
        assert repair_json_payload('{"flag": False}') == '{"flag": false}'

    def test_python_none(self):
        assert repair_json_payload('{"val": None}') == '{"val": null}'

    def test_empty_string(self):
        assert repair_json_payload("") == ""

    def test_combined_repairs(self):
        s = '{"a": True, "b": None,}'
        result = repair_json_payload(s)
        assert result == '{"a": true, "b": null}'


# ===========================================================================
# 4. _is_xml_noise
# ===========================================================================

class TestIsXmlNoise:
    def test_empty(self):
        assert _is_xml_noise("") is True
        assert _is_xml_noise(None) is True
        assert _is_xml_noise("   ") is True

    def test_cdata_open_normal(self):
        assert _is_xml_noise("<![CDATA[") is True

    def test_cdata_close(self):
        assert _is_xml_noise("]]>") is True

    def test_cdata_open_malformed(self):
        assert _is_xml_noise("<![CDATACDATA[") is True

    def test_meaningful_text(self):
        assert _is_xml_noise("hello world") is False

    def test_mixed_cdata_and_text(self):
        assert _is_xml_noise("<![CDATA[ some text") is False


# ===========================================================================
# 5. _extract_cdata_text — 集成测试
# ===========================================================================

class TestExtractCdataTextIntegration:
    def test_normal_cdata(self):
        assert _extract_cdata_text('<![CDATA[hello]]>') == 'hello'

    def test_malformed_cdata_duplicated(self):
        """修复 <![CDATACDATA[ 后再提取"""
        raw = '<![CDATACDATA[{"pattern":"**/*.py"}]]>'
        assert _extract_cdata_text(raw) == '{"pattern":"**/*.py"}'

    def test_no_cdata(self):
        assert _extract_cdata_text('{"a":1}') == '{"a":1}'


# ===========================================================================
# 6. _parse_args_json_payload — 集成测试
# ===========================================================================

class TestParseArgsJsonPayloadIntegration:
    def test_normal_json(self):
        assert _parse_args_json_payload('{"a":1}') == {"a": 1}

    def test_json_with_xml_noise(self):
        """JSON 被 CDATA 残留包裹时应接受"""
        payload = '<![CDATACDATA[{"command": "ls -la"}]]>'
        result = _parse_args_json_payload(payload)
        assert result == {"command": "ls -la"}

    def test_trailing_comma(self):
        assert _parse_args_json_payload('{"a": 1,}') == {"a": 1}

    def test_python_booleans(self):
        assert _parse_args_json_payload('{"flag": True}') == {"flag": True}

    def test_empty_string(self):
        assert _parse_args_json_payload("") == {}

    def test_reject_real_extra_content(self):
        """JSON 外有有意义文本时仍应拒绝"""
        payload = 'some text {"a":1} more text'
        assert _parse_args_json_payload(payload) is None


# ===========================================================================
# 7. parse_function_calls_xml — 端到端回归测试
# ===========================================================================

class TestParseEndToEnd:
    TRIGGER = "<Function_S5Wm_Start/>"

    def test_reported_bug_malformed_cdata(self):
        """完整重现: 3个function_call, 第3个有 <![CDATACDATA["""
        xml = f"""{self.TRIGGER}
<function_calls>
<function_call>
<tool>Glob</tool>
<args_json><![CDATA[{{"pattern": "**/*.py"}}]]></args_json>
</function_call>
<function_call>
<tool>Glob</tool>
<args_json><![CDATA[{{"pattern": "**/*.json"}}]]></args_json>
</function_call>
<function_call>
<tool>Bash</tool>
<args_json><![CDATACDATA[{{"command": "ls -la"}}]]></args_json>
</function_call>
</function_calls>"""

        result = parse_function_calls_xml(xml, self.TRIGGER)
        assert result is not None, "应成功解析"
        assert len(result) == 3
        assert result[0]["name"] == "Glob"
        assert result[0]["args"] == {"pattern": "**/*.py"}
        assert result[2]["name"] == "Bash"
        assert result[2]["args"] == {"command": "ls -la"}

    def test_uppercase_tags(self):
        """标签全大写也能解析"""
        xml = f"""{self.TRIGGER}
<FUNCTION_CALLS>
<FUNCTION_CALL>
<TOOL>search</TOOL>
<ARGS_JSON>{{"query": "test"}}</ARGS_JSON>
</FUNCTION_CALL>
</FUNCTION_CALLS>"""

        result = parse_function_calls_xml(xml, self.TRIGGER)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "search"
        assert result[0]["args"] == {"query": "test"}

    def test_python_booleans_in_args(self):
        """args_json 中的 Python True/False 也能解析"""
        xml = f"""{self.TRIGGER}
<function_calls>
<function_call>
<tool>config</tool>
<args_json><![CDATA[{{"enabled": True, "debug": False, "value": None}}]]></args_json>
</function_call>
</function_calls>"""

        result = parse_function_calls_xml(xml, self.TRIGGER)
        assert result is not None
        assert result[0]["args"]["enabled"] is True
        assert result[0]["args"]["debug"] is False
        assert result[0]["args"]["value"] is None

    def test_normal_xml_still_works(self):
        """正常 XML 不受影响"""
        xml = f"""{self.TRIGGER}
<function_calls>
<function_call>
<tool>Grep</tool>
<args_json><![CDATA[{{"pattern": "hello"}}]]></args_json>
</function_call>
</function_calls>"""

        result = parse_function_calls_xml(xml, self.TRIGGER)
        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "Grep"
        assert result[0]["args"] == {"pattern": "hello"}


# ===========================================================================
# 8. looks_like_complete_function_calls — 畸形标签计数
# ===========================================================================

class TestLooksLikeComplete:
    def test_normal(self):
        buf = '<function_calls><function_call><tool>X</tool><args_json><![CDATA[{}]]></args_json></function_call></function_calls>'
        assert looks_like_complete_function_calls(buf) is True

    def test_malformed_cdata(self):
        buf = '<function_calls><function_call><tool>X</tool><args_json><![CDATACDATA[{}]]></args_json></function_call></function_calls>'
        assert looks_like_complete_function_calls(buf) is True

    def test_uppercase_tags(self):
        buf = '<FUNCTION_CALLS><FUNCTION_CALL><TOOL>X</TOOL></FUNCTION_CALL></FUNCTION_CALLS>'
        assert looks_like_complete_function_calls(buf) is True

    def test_incomplete(self):
        buf = '<function_calls><function_call><tool>X</tool>'
        assert looks_like_complete_function_calls(buf) is False


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
