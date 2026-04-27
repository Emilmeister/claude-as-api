from claude_proxy.prompt_builder import build_system_prompt, build_user_prompt
from claude_proxy.schemas import ChatMessage, FunctionDef, ToolCall, ToolCallFunction, ToolDef


def test_system_prompt_includes_caller_system_messages():
    msgs = [
        ChatMessage(role="system", content="You are a Russian-speaking assistant."),
        ChatMessage(role="user", content="hi"),
    ]
    sys = build_system_prompt(msgs, tools=None)
    assert "You are an AI assistant" in sys
    assert "Russian-speaking" in sys
    assert "Available tools" not in sys


def test_system_prompt_renders_tools():
    tools = [
        ToolDef(function=FunctionDef(
            name="get_weather",
            description="Lookup the weather.",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        )),
    ]
    sys = build_system_prompt([ChatMessage(role="user", content="?")], tools)
    assert "Available tools" in sys
    assert "### get_weather" in sys
    assert "Lookup the weather" in sys
    assert '"city"' in sys


def test_user_prompt_renders_history_and_tool_results():
    msgs = [
        ChatMessage(role="system", content="ignore me"),
        ChatMessage(role="user", content="weather?"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="call_abc", function=ToolCallFunction(name="get_weather", arguments='{"city":"Berlin"}'))],
        ),
        ChatMessage(role="tool", tool_call_id="call_abc", name="get_weather", content="22C sunny"),
        ChatMessage(role="user", content="thanks!"),
    ]
    rendered = build_user_prompt(msgs)
    assert "ignore me" not in rendered
    assert "### User\nweather?" in rendered
    assert "### Assistant (tool calls)" in rendered
    assert "call_abc" in rendered
    assert "### Tool result (id=call_abc, name=get_weather)" in rendered
    assert "22C sunny" in rendered
    assert "thanks!" in rendered
    assert rendered.rstrip().endswith("structured response.")


def test_content_blocks_are_flattened_to_text():
    msgs = [
        ChatMessage(role="user", content=[
            {"type": "text", "text": "part one"},
            {"type": "text", "text": "part two"},
        ]),
    ]
    rendered = build_user_prompt(msgs)
    assert "part one" in rendered and "part two" in rendered
