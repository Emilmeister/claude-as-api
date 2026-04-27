# claude-as-api — context for future Claude Code sessions

## What this is

A FastAPI service that speaks the **OpenAI Chat Completions API** but proxies every call to `claude -p` (the Claude Code CLI) so requests bill against the user's **Claude subscription**, not the Anthropic API. Targets local OpenAI-compatible clients (Open WebUI, LangChain, OpenAI SDK, IDE plugins).

## Layout

```
claude_proxy/
├── app.py                  # FastAPI routes, request dispatch (3 paths: blocking / true_stream / pseudo_stream)
├── claude_runner.py        # subprocess manager: env sanitization, sterile cwd, argv assembly, parsing
├── prompt_builder.py       # OpenAI messages -> single system + user prompt strings
├── tools_bridge.py         # OpenAI `tools` -> `--json-schema`; structured_output -> OpenAI tool_calls
├── schema_normalizer.py    # rewrites user-provided oneOf/anyOf into flat schemas
├── stream_translator.py    # claude stream-json events -> OpenAI SSE chunks (text + thinking)
├── schemas.py              # pydantic models for OpenAI shapes
└── config.py               # env -> Config; model id mapping; effort normalization
main.py                     # uvicorn entry: `app = create_app()`
tests/                      # pytest unit tests (no claude needed)
```

## Three response paths

The handler in `app.py` picks one based on the request:

| condition | path | what runs |
|---|---|---|
| `stream=False` (any tools/format) | `_blocking_response` | `run_oneshot` with `--output-format=json` (+ `--json-schema=...` if needed); single JSON response |
| `stream=True && !tools && !response_format=json_*` | `_stream_response` | `run_streaming` with `--output-format=stream-json`; `translate_stream` forwards `text_delta`→content, `thinking_delta`→reasoning_content |
| `stream=True && (tools OR response_format=json_*)` | `_pseudo_stream_response` | `run_streaming` with `--output-format=stream-json --json-schema=...`; `translate_schema_stream` forwards `thinking_delta`→reasoning_content, ignores `text_delta` and `input_json_delta`, then emits final `tool_calls`/`content` from `result` event's `structured_output` |

Don't add a fourth path. The split is deliberate: only the schema-mode `result` event has the authoritative answer; intermediate `text_delta`s are pre-StructuredOutput-call noise.

## Critical CLI flags (do not regress)

`claude_runner._base_args` always passes:

- **`--mcp-config=<empty-file>`** — suppresses `~/.claude/`-level MCP servers loading. Without this the cached context inflates by 15k+ tokens.
- **`--system-prompt=<...>`** (not `--append-`) — fully replaces Claude Code's default agentic system prompt; we control what the model sees.
- **`--allowedTools=`** (empty) — blocks the *injection* path that would otherwise add Claude Code's built-in tools (Bash, Read, Edit, …) into the Anthropic API request body. One of those built-ins (`TaskOutput` historically) ships a malformed `input_schema` that the API rejects when **combined with `--json-schema`** — single-branch schemas were the trigger we hit. `--disallowedTools=` does not help here because it only blocks *execution*, not the API tool list.
- **`--permission-mode=dontAsk`** — paired with the empty allow-list, prevents any tool actually executing.
- **`--disable-slash-commands`** — disables auto-loading of user-level *skills*. This was the cause of the "Hook feedback stopped" non-sequitur replies: a user-level skill (e.g. `claude-api`, `update-config`) was firing and contaminating the response. Always keep this on.
- **`--no-session-persistence`** — we're stateless; don't litter the disk.
- **`--effort=<level>`** — passed only when configured (env `EFFORT`, default `medium`) or per-request `reasoning_effort`. Omit the flag for "auto" (model decides).
- **`--output-format`** — `json` for blocking, `stream-json --verbose --include-partial-messages` for streaming.

`claude_runner.sanitized_env` strips billing-related env vars (`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`, `CLAUDE_CODE_USE_BEDROCK/VERTEX`, `AWS_BEARER_TOKEN_BEDROCK`) — without this, `claude -p` would silently switch to API billing if the parent process happens to have one of these set. It also forces `LC_ALL=C.UTF-8`/`LANG=C.UTF-8` if the parent locale isn't already UTF-8.

Subprocess args use **`--flag=value`** form (not space-separated). `--mcp-config` and similar greedy-multi-value flags will swallow the prompt as their argument otherwise — observed and verified.

`stdin=asyncio.subprocess.DEVNULL` is required: `claude -p` waits 3 seconds for stdin data otherwise. This shaves ~3s off every call.

## Why we cannot use `--bare`

`--bare` would skip skills, hooks, MCP discovery, and CLAUDE.md. **But it also skips OAuth/keychain reads.** Subscription billing requires OAuth. So we live with the Claude Code default-loaded context and suppress what we can via individual flags. `~/.claude/settings.json` hooks still run and cannot be turned off without `--bare`. If a hook starts hijacking responses, set `CLAUDE_DEBUG=1` to add `--debug=api,hooks` and surface what's firing in subprocess stderr.

## Tools mapping (OpenAI-style)

OpenAI semantics: model returns `tool_calls`, *the caller's client* executes them and posts back a `role: tool` message. Claude Code's native loop runs tools internally, which is the wrong shape.

We bridge by forcing structured output:

1. `tools_bridge.build_response_schema(tools, tool_choice)` builds a JSON schema with a flat `kind` discriminator (`enum: ["tool_calls", "final"]`) and optional `tool_calls`/`content` fields. **Never `oneOf` / `anyOf`** — those trigger the Claude Code internal-tool injection bug described above.
2. The schema goes into `--json-schema=`. The system prompt also contains markdown-rendered tool descriptions so the model knows what each tool does.
3. `tools_bridge.parse_structured_output` reads `envelope["structured_output"]`, generates `call_<hex>` IDs, JSON-encodes arguments, and returns `(finish_reason, content, tool_calls)` shaped for OpenAI.

Per-tool-choice handling:
- `auto` → flat schema with both branches available
- `required` → schema where `kind: "tool_calls"` only
- `none` → schema where `kind: "final"` only
- `{type: function, function: {name: "X"}}` → tool_calls-only with `name: {const: "X"}`

## Schema normalization (`schema_normalizer.py`)

Users may send arbitrary JSON schemas in `response_format.json_schema`. If they include `oneOf`/`anyOf`, claude -p's structured-output pipeline pulls in the malformed-built-in-tool path and the API rejects with `tools.N.custom.input_schema.type: Field required`. Rather than 400 the user, we **rewrite the schema before sending**:

- Discriminated union (every branch is an object with the same `const`-valued field): flatten to `{kind: {enum: [...]}, ...union of branches}`, drop per-branch `required`.
- Object union without a discriminator: merge `properties`, no `required`.
- Heterogeneous: pick the first object branch.

Walks the tree recursively; nested unions get rewritten too. `properties` and `required` siblings of the union are preserved.

## Why `num_turns` is often 2-3, not 1

Whenever `--json-schema` is passed (= any tools call OR `response_format=json_*`), Claude Code's internal flow becomes:

1. Turn 1 — API call with the user's prompt + an internal `StructuredOutput` tool whose `input_schema` is the user's schema. Claude responds with `tool_use(StructuredOutput, {...payload...})`.
2. Turn 2 — Claude Code locally captures the payload (this is what becomes `envelope.structured_output`), feeds an empty `tool_result` back. Claude responds with a brief acknowledgement like "Done."
3. Turn 3 — for thinking-enabled models (Haiku 4.5 by default) sometimes a final extra reasoning step.

This is unavoidable cost in `claude -p` headless mode. Plain (no schema) calls stay at 1 turn.

## Thinking / reasoning forwarding

Claude streams `thinking_delta` events in stream-json. We forward them as `delta.reasoning_content` (de-facto field name used by DeepSeek and supported by Open WebUI/LibreChat/etc). Both `translate_stream` (plain) and `translate_schema_stream` (tools/schema) do this.

In schema mode, intermediate `text_delta` events are **ignored** because they are pre-`StructuredOutput`-call thinking-out-loud, not the final answer. The final answer comes from the `result` event's `structured_output`, parsed in `_parse_schema_result`.

Effort levels (`--effort`): `low | medium | high | xhigh | max`. Server-wide default via `EFFORT` env (defaults to `medium`); per-request override via OpenAI's `reasoning_effort`. `auto` (or empty/unrecognized) means "don't pass `--effort`, model defaults".

## Encoding

`asyncio.create_subprocess_exec` encodes argv using the parent process's filesystem encoding. On VMs with `LANG=C` this drops non-ASCII (Cyrillic, emoji). Mitigations already in place:

- `sanitized_env()` forces UTF-8 locale in subprocess if parent isn't UTF-8.
- For the **parent** process, run with `PYTHONUTF8=1` if the host locale is suspect.

DEBUG-level logs include hex of the first 96 bytes of the user prompt and a count of replacement chars (`�`) — useful to localize where corruption enters (request body vs our pipeline).

## Tests

```bash
uv run pytest tests/ -q
```

All units run without `claude` installed (pure logic: prompt building, schema normalization, tool bridging, union detection). E2E smoke tests live ad-hoc in commit history; rerun manually when changing the runner.

## Don't break

- The non-`oneOf` rule for tools schemas (`tools_bridge.build_response_schema`).
- The flag set in `_base_args`. Each one earned its place by debugging a specific failure mode.
- The dispatch split between `_stream_response`, `_pseudo_stream_response`, `_blocking_response` — schema mode and plain mode have fundamentally different "where is the answer" assumptions.
- Subscription billing: never let `ANTHROPIC_API_KEY` or its siblings into the subprocess env. `sanitized_env()` is the single chokepoint.
