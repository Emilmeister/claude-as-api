import json

from claude_proxy.app import _schema_uses_union
from claude_proxy.schemas import FunctionDef, ToolDef
from claude_proxy.tools_bridge import build_response_schema, parse_structured_output


def _tools():
    return [
        ToolDef(function=FunctionDef(name="a", parameters={"type": "object"})),
        ToolDef(function=FunctionDef(name="b", parameters={"type": "object"})),
    ]


def test_schema_auto_uses_flat_discriminator():
    """auto-mode must avoid oneOf/anyOf to dodge Claude Code's tool-injection bug."""
    schema = build_response_schema(_tools(), tool_choice="auto")
    assert "oneOf" not in schema and "anyOf" not in schema
    assert schema["properties"]["kind"]["enum"] == ["tool_calls", "final"]
    assert "tool_calls" in schema["properties"]
    assert "content" in schema["properties"]
    assert schema["required"] == ["kind"]


def test_schema_required_keeps_only_tool_branch():
    schema = build_response_schema(_tools(), tool_choice="required")
    assert schema["properties"]["kind"]["const"] == "tool_calls"
    enum = schema["properties"]["tool_calls"]["items"]["properties"]["name"]["enum"]
    assert enum == ["a", "b"]


def test_schema_none_keeps_only_final():
    schema = build_response_schema(_tools(), tool_choice="none")
    assert schema["properties"]["kind"]["const"] == "final"


def test_schema_forced_function_pins_name():
    schema = build_response_schema(_tools(), tool_choice={"type": "function", "function": {"name": "a"}})
    name_schema = schema["properties"]["tool_calls"]["items"]["properties"]["name"]
    assert name_schema == {"const": "a"}
    assert schema["properties"]["tool_calls"]["maxItems"] == 1


def test_parse_tool_calls():
    raw = {
        "kind": "tool_calls",
        "tool_calls": [
            {"name": "a", "arguments": {"x": 1}},
            {"name": "b", "arguments": {}},
        ],
    }
    finish, content, tcs = parse_structured_output(raw)
    assert finish == "tool_calls"
    assert content is None
    assert len(tcs) == 2
    assert tcs[0]["function"]["name"] == "a"
    assert json.loads(tcs[0]["function"]["arguments"]) == {"x": 1}
    assert tcs[0]["id"].startswith("call_")
    assert tcs[0]["id"] != tcs[1]["id"]


def test_parse_final():
    finish, content, tcs = parse_structured_output({"kind": "final", "content": "hello"})
    assert finish == "stop"
    assert content == "hello"
    assert tcs is None


def test_schema_uses_union_detects_oneof_and_anyof():
    assert _schema_uses_union({"oneOf": []}) is True
    assert _schema_uses_union({"anyOf": []}) is True
    assert _schema_uses_union({
        "type": "object",
        "properties": {"x": {"oneOf": [{"type": "string"}]}},
    }) is True
    assert _schema_uses_union({
        "type": "object",
        "properties": {"x": {"type": "integer"}},
    }) is False
    assert _schema_uses_union({}) is False


def test_parse_handles_forced_branch_without_kind():
    # When tool_choice forces single branch, Claude may omit `kind`.
    finish, content, tcs = parse_structured_output({
        "tool_calls": [{"name": "a", "arguments": {}}]
    })
    assert finish == "tool_calls"
    assert tcs and tcs[0]["function"]["name"] == "a"
