import json
from typing import Iterable

from .schemas import ChatMessage, ToolDef


BASE_SYSTEM = (
    "You are an AI assistant accessed through an OpenAI-compatible API. "
    "Reply concisely and directly. Do not use any tools unless explicitly enabled below. "
    "Do not edit files, run shell commands, or read the filesystem. "
    "Answer the user's last message; prior turns are conversation history for context."
)


def _content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif "text" in block:
                parts.append(str(block["text"]))
    return "\n".join(parts)


def build_system_prompt(messages: Iterable[ChatMessage], tools: list[ToolDef] | None) -> str:
    parts = [BASE_SYSTEM]
    sys_texts = [
        _content_to_text(m.content) for m in messages if m.role == "system"
    ]
    sys_texts = [t for t in sys_texts if t.strip()]
    if sys_texts:
        parts.append("## Caller-provided system instructions\n" + "\n\n".join(sys_texts))
    if tools:
        parts.append(_render_tools_block(tools))
    return "\n\n".join(parts)


def _render_tools_block(tools: list[ToolDef]) -> str:
    lines = [
        "## Available tools",
        "You may call any of the following tools by returning a JSON object that conforms to the response schema (kind=tool_calls).",
        "Only call a tool if it is genuinely required to answer the user. Otherwise return kind=final with your text answer.",
        "",
    ]
    for t in tools:
        f = t.function
        lines.append(f"### {f.name}")
        if f.description:
            lines.append(f.description.strip())
        lines.append("Parameters JSON Schema:")
        lines.append("```json")
        lines.append(json.dumps(f.parameters or {"type": "object"}, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip()


def build_user_prompt(messages: list[ChatMessage]) -> str:
    """Render the conversation (excluding system messages) as a single markdown prompt."""
    blocks: list[str] = []
    for m in messages:
        if m.role == "system":
            continue
        if m.role == "user":
            text = _content_to_text(m.content)
            if text.strip():
                blocks.append(f"### User\n{text}")
        elif m.role == "assistant":
            text = _content_to_text(m.content)
            if text.strip():
                blocks.append(f"### Assistant\n{text}")
            if m.tool_calls:
                tc_lines = ["### Assistant (tool calls)"]
                for tc in m.tool_calls:
                    tc_lines.append(
                        f"- id={tc.id} name={tc.function.name} args={tc.function.arguments}"
                    )
                blocks.append("\n".join(tc_lines))
        elif m.role == "tool":
            text = _content_to_text(m.content)
            header = f"### Tool result (id={m.tool_call_id or '?'}, name={m.name or '?'})"
            blocks.append(f"{header}\n{text}")

    blocks.append(
        "### Task\nReply to the user's last message. "
        "If tools are listed in the system prompt, you may call them via the structured response."
    )
    return "\n\n".join(blocks)
