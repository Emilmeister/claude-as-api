"""Bridge OpenAI-style tool calling onto Claude's --json-schema structured output.

Strategy: when the caller passes `tools`, we force Claude's response into a discriminated
union over {kind: "tool_calls", tool_calls: [...]} | {kind: "final", content: "..."}.
We then parse the structured_output back into an OpenAI ChatCompletion shape.
"""
from __future__ import annotations

import json
import secrets
from typing import Any

from .schemas import ToolDef


def build_response_schema(tools: list[ToolDef], tool_choice: Any) -> dict[str, Any]:
    """Build a JSON schema that forces Claude into either a tool_calls or a final answer.

    For `tool_choice` values "required", "none", or `{type: function, function: {name}}`
    we return a single-branch schema (no `oneOf`). For "auto" we return a flat schema
    with a `kind` discriminator and both `tool_calls` and `content` optional —
    deliberately avoiding `oneOf`/`anyOf`, which causes Claude Code's structured output
    pipeline to inject built-in tools that some Claude Code versions ship with malformed
    `input_schema` (Anthropic API rejects with `tools.N.custom.input_schema.type:
    Field required`). The system prompt instructs the model on how to choose `kind`.
    """
    tool_names = [t.function.name for t in tools]

    forced_name: str | None = None
    if isinstance(tool_choice, dict):
        fn = tool_choice.get("function") or {}
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            forced_name = fn["name"]
        choice_kind = tool_choice.get("type")
    else:
        choice_kind = tool_choice

    if forced_name:
        return _tool_calls_only_schema([forced_name], max_items=1)
    if choice_kind == "required":
        return _tool_calls_only_schema(tool_names)
    if choice_kind == "none":
        return _final_only_schema()

    # auto / unset: flat schema with kind discriminator, no oneOf.
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "kind": {"enum": ["tool_calls", "final"]},
            "tool_calls": {
                "type": "array",
                "items": _tool_call_item_schema(tool_names),
            },
            "content": {"type": "string"},
        },
        "required": ["kind"],
    }


def _tool_call_item_schema(tool_names: list[str]) -> dict[str, Any]:
    name_constraint: dict[str, Any]
    if len(tool_names) == 1:
        name_constraint = {"const": tool_names[0]}
    else:
        name_constraint = {"enum": tool_names}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": name_constraint,
            "arguments": {"type": "object"},
        },
        "required": ["name", "arguments"],
    }


def _tool_calls_only_schema(tool_names: list[str], max_items: int | None = None) -> dict[str, Any]:
    arr: dict[str, Any] = {
        "type": "array",
        "minItems": 1,
        "items": _tool_call_item_schema(tool_names),
    }
    if max_items is not None:
        arr["maxItems"] = max_items
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "kind": {"const": "tool_calls"},
            "tool_calls": arr,
        },
        "required": ["kind", "tool_calls"],
    }


def _final_only_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "kind": {"const": "final"},
            "content": {"type": "string"},
        },
        "required": ["kind", "content"],
    }


def parse_structured_output(raw: dict[str, Any] | None) -> tuple[str, str | None, list[dict[str, Any]] | None]:
    """Return (finish_reason, content, tool_calls) for an OpenAI Choice.message.

    Tool call ids are generated here in OpenAI's `call_<hex>` format.
    """
    if not raw or not isinstance(raw, dict):
        return "stop", "", None

    kind = raw.get("kind")
    if kind == "tool_calls":
        items = raw.get("tool_calls") or []
        tool_calls: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            args = item.get("arguments") or {}
            if not isinstance(name, str):
                continue
            tool_calls.append({
                "id": _new_tool_call_id(),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            })
        if tool_calls:
            return "tool_calls", None, tool_calls
        return "stop", "", None

    if kind == "final":
        return "stop", str(raw.get("content") or ""), None

    # Forced single-branch (kind missing): treat as tool_calls if it has tool_calls key.
    if "tool_calls" in raw:
        return parse_structured_output({"kind": "tool_calls", **raw})
    if "content" in raw:
        return parse_structured_output({"kind": "final", **raw})
    return "stop", "", None


def _new_tool_call_id() -> str:
    return f"call_{secrets.token_hex(12)}"
