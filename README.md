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

If responses ever come back nonsensical (e.g. mentioning "hooks", "feedback", or "stopped" out of context), set `CLAUDE_DEBUG=1` to add `--debug=api,hooks` to the subprocess; subprocess stderr is then dumped at DEBUG level and you can see which user-level hook from `~/.claude/settings.json` is firing. The proxy already passes `--disable-slash-commands` and `--no-session-persistence` to neutralize skills and session writes, but hooks in `~/.claude/settings.json` cannot be turned off without losing OAuth (`--bare`).

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
#   EFFORT=medium                 # thinking effort: low|medium|high|xhigh|max|auto (auto = let model pick)
#   SANDBOX_DIR=/tmp/claude-as-api/sandbox  # cwd for subprocesses; keep empty
#   CLAUDE_DEBUG=1                # forward `--debug=api,hooks` and dump subprocess stderr at DEBUG
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

OpenAI semantics: the model returns `tool_calls`, **your client executes them**, and posts back a `role: tool` message on the next request. **The proxy never executes your tools** — `obsidian_search`, `get_weather`, etc. are just names + JSON schemas that get round-tripped to the model.

To make Claude follow those semantics (its native loop runs tools internally instead), the service passes `--json-schema` to `claude -p`, forcing the model to return either:

```json
{"kind": "tool_calls", "tool_calls": [{"name": "...", "arguments": {...}}]}
```

or:

```json
{"kind": "final", "content": "..."}
```

The schema avoids `oneOf`/`anyOf` — those trigger a Claude Code internal-tool injection path that can fail with `tools.N.custom.input_schema.type: Field required` against the Anthropic API. See `claude_proxy/tools_bridge.py`.

> **About `num_turns=2-3` you'll see in the logs:** that's an artifact of Claude Code's structured-output implementation, not the number of tools your model called. When `--json-schema` is passed, Claude Code wraps the response generation in its own internal tool-calling loop (a hidden `StructuredOutput` tool that Claude Code "executes" locally by extracting the JSON payload). Plain chat without `--json-schema` stays at `num_turns=1`. None of this involves your OpenAI tools — those round-trip through your client as usual.

## Thinking / reasoning

When the model thinks (Claude Haiku 4.5 does so by default; Sonnet/Opus on demand), thinking deltas are streamed as `delta.reasoning_content` in SSE chunks — the field used by DeepSeek and supported by Open WebUI, LibreChat, etc. It works in **both** plain streaming and `stream=True + tools`/`response_format` modes:

```python
for chunk in c.chat.completions.create(
    model="claude-haiku-4-5",
    messages=[{"role": "user", "content": "explain RSA briefly"}],
    stream=True,
):
    d = chunk.choices[0].delta
    if getattr(d, "reasoning_content", None):
        print(f"[think] {d.reasoning_content}", end="", flush=True)
    if d.content:
        print(d.content, end="", flush=True)
```

### Effort level

Set how hard the model thinks via Claude Code's `--effort`. Levels: `low | medium | high | xhigh | max`. Two ways:

```bash
EFFORT=medium uv run python main.py    # server-wide default; this is the default-default
EFFORT=auto   uv run python main.py    # let the model choose (no --effort passed)
```

```python
# Per-request override (OpenAI o1 convention):
c.chat.completions.create(
    model="claude-sonnet-4-6",
    messages=[...],
    extra_body={"reasoning_effort": "high"},
)
```

Per-request `reasoning_effort` wins over the env default. Unrecognized values fall back to "auto".

## Caveats

- **`stream=True && tools` streams thinking, not text.** The final tool_calls/content arrives as a single SSE chunk at the end (because structured output assembles a complete JSON payload). Thinking still streams live in `reasoning_content` deltas while the model works.
- **Stateless conversations.** Send the full message history each call (matches OpenAI semantics). The service does not retain `session_id`s.
- **Cost overhead per call.** Claude Code loads its tool registry into context regardless of `--allowedTools` or empty `--mcp-config`, so each call has a few-thousand-token baseline. Within the 5-minute prompt cache TTL these are cache reads (cheap on the API but still billed at your subscription's rate).
- **No `--bare` mode.** It would skip OAuth and break subscription auth. Side effect: hooks defined in `~/.claude/settings.json` still run (skills are disabled via `--disable-slash-commands`, but hooks can't be turned off without `--bare`). If a hook starts hijacking responses with text like "Hook feedback stopped", set `CLAUDE_DEBUG=1` to identify it.
- **Encoding.** On VMs with `LANG=C`, multi-byte chars (Cyrillic, emoji) can be mangled in argv. The runner forces `LC_ALL=C.UTF-8` in the subprocess; for the parent uvicorn process run with `PYTHONUTF8=1` if you suspect host locale issues. DEBUG logs include hex of the prompt's first bytes and a count of `�` chars to localize where corruption enters.
- **Some flag forms are mandatory.** `claude -p` greedy-multi-value flags (`--mcp-config`, `--system-prompt`, `--json-schema`) must be passed as `--flag=value`, not `--flag value`, or they swallow the prompt as their argument. The runner already does this; do not refactor argv assembly to space-separated form.

## Tests

```bash
uv run pytest tests/                      # unit (no claude required)
```
