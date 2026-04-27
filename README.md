# claude-as-api

OpenAI-compatible HTTP API on top of `claude -p` (Claude Code CLI). Routes `/v1/chat/completions`, `/v1/models` and `/health`. Tool calls, streaming, and `response_format` are supported.

The point: pay through your **Claude Code subscription**, not pay-as-you-go API rates.

## Setup

Requires `claude` CLI (`>= 2.x`), Python `>= 3.11`, `uv`.

```bash
uv sync --extra dev
```

## Subscription billing — verify before running

`claude -p` uses `ANTHROPIC_API_KEY` if it sees it in the env. You must NOT have it set:

```bash
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
claude /status     # interactive — should report "Claude subscription auth"
```

The runner strips these env vars from the subprocess for safety, but you still need OAuth in `~/.claude/` to be valid (i.e. you must have logged in via `claude login`).

## Debugging

Set `LOG_LEVEL=DEBUG` to see full prompts, schemas, argv flag shape, and parsed structured outputs in the server log:

```bash
LOG_LEVEL=DEBUG uv run python main.py
```

At INFO (default) you still see one line per `claude -p` call with rc, duration, turn count, token counts, and cost. On failures the actual API error message (from the JSON envelope's `result`) is now surfaced through the 502 — you no longer get a bare `exited with code 1: ` with empty stderr.

## Run

```bash
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN
uv run python main.py            # listens on 127.0.0.1:8000

# Optional env:
#   PORT=8765 HOST=0.0.0.0
#   PROXY_API_KEY=xyz             # require Bearer auth on the proxy itself
#   MAX_CONCURRENT=4              # parallel claude -p subprocesses
#   CLAUDE_TIMEOUT_S=300
#   DEFAULT_MODEL=claude-sonnet-4-6
#   SANDBOX_DIR=/tmp/claude-as-api/sandbox  # cwd for subprocesses; keep empty
```

## Use from any OpenAI-compatible client

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="dummy")

# chat
r = c.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "hi"}],
)

# streaming
for chunk in c.chat.completions.create(
    model="claude-haiku-4-5",
    messages=[{"role": "user", "content": "count to 5"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)

# tools
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}]
r = c.chat.completions.create(
    model="claude-haiku-4-5",
    messages=[{"role": "user", "content": "weather in Tokyo?"}],
    tools=tools, tool_choice="auto",
)
# r.choices[0].message.tool_calls -> client executes -> appends tool result -> next call.
```

Models exposed (`GET /v1/models`): `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5`. Common OpenAI model ids (`gpt-4o`, `gpt-4`, `gpt-4o-mini`, `gpt-3.5-turbo`) are mapped onto the closest Claude tier.

## Structured output (`response_format`)

Both OpenAI shapes are supported:

```python
# Free-form JSON
c.chat.completions.create(
    model="claude-haiku-4-5",
    messages=[{"role": "user", "content": "Return a JSON object summarizing X."}],
    response_format={"type": "json_object"},
)

# Strict schema
c.chat.completions.create(
    model="claude-haiku-4-5",
    messages=[{"role": "user", "content": "Imagine the weather in Tokyo."}],
    response_format={"type": "json_schema", "json_schema": {
        "name": "WeatherInfo",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "city": {"type": "string"},
                "temperature_c": {"type": "number"},
                "conditions": {"type": "string"},
            },
            "required": ["city", "temperature_c", "conditions"],
        },
    }},
)
# message.content is a JSON string conforming to the schema.
```

### `oneOf` / `anyOf` are auto-rewritten

Claude Code's structured-output pipeline injects built-in tools whose `input_schema` the Anthropic API rejects whenever the user schema contains a union. The service transparently rewrites unions before sending them to `claude -p` (see `claude_proxy/schema_normalizer.py`):

| Input shape | Rewrite |
|---|---|
| Discriminated union — every branch is an object with a shared `const` field (`kind`, `type`, `tag`, ...) | Flat object: discriminator becomes `{"enum": [...]}`, all branch properties merged in. Only the discriminator stays `required`. |
| Object union without a discriminator | Single object whose `properties` are the union of all branches. No `required`. |
| Heterogeneous (mixed types) | Falls back to the first object-typed branch. |

What you lose: per-branch `required` fields and strict mutual exclusion. The model still picks one branch's payload at a time and the discriminator in the prompt usually steers it correctly. If you need stricter guarantees, write the flat schema yourself.

## How tools are emulated

OpenAI tool calling has the model return `tool_calls` and the *client* execute them. Claude Code's native paradigm is the agent executing its own internal tools.

This service bridges the two by passing `--json-schema` to `claude -p`, forcing the model to return either:

```json
{"kind": "tool_calls", "tool_calls": [{"name": "...", "arguments": {...}}]}
```

or:

```json
{"kind": "final", "content": "..."}
```

The schema avoids `oneOf`/`anyOf` — those trigger a Claude Code internal-tool injection path that can fail with `tools.N.custom.input_schema.type: Field required` against the Anthropic API. See `claude_proxy/tools_bridge.py`.

## Caveats

- **Streaming + tools is non-streaming under the hood.** When `tools` are present, structured output requires the full JSON to assemble before parsing, so the response is delivered as a single OpenAI chunk at the end. Plain chat streams normally.
- **Stateless conversations.** Send the full message history each call (matches OpenAI semantics). The service does not retain `session_id`s.
- **Cost overhead per call.** Claude Code loads its tool registry into context regardless of `--allowedTools` or empty `--mcp-config`, so each call has a few-thousand-token baseline. Within the 5-minute prompt cache TTL these are cache reads (cheap on the API but still billed at your subscription's rate).
- **No `--bare`.** It would skip OAuth and fail subscription auth.

## Tests

```bash
uv run pytest tests/                      # unit (no claude required)
```
