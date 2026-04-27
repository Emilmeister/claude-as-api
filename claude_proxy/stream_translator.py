"""Translate Claude `stream-json` events into OpenAI Chat Completion SSE chunks.

Two modes:

* **text mode** (no `--json-schema`): forward `text_delta` as `delta.content` and
  `thinking_delta` as `delta.reasoning_content`. Used for plain chat streaming.

* **schema mode** (with `--json-schema`, e.g. tools or `response_format`): only
  `thinking_delta` becomes `reasoning_content`; intermediate `text_delta` and
  `input_json_delta` events are ignored because the authoritative answer lives in
  the final `result` event's `structured_output`. The caller passes a
  `final_emit` callback that runs once the result event is seen and yields the
  closing chunk(s) — typically a `tool_calls` delta or a `content` delta with the
  finish reason.

`reasoning_content` is the de-facto field name used by DeepSeek's API and
supported by Open WebUI, LibreChat, and most other OpenAI-compatible clients.
OpenAI's own o1 doesn't expose deltas, but no client errors on an extra delta
field, so it's a safe default.
"""
from __future__ import annotations

import json
import time
from typing import AsyncIterator, Awaitable, Callable, Iterable

from .schemas import ChatCompletionChunk, DeltaMessage, StreamChoice


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _chunk(completion_id: str, model: str, delta: dict,
           finish_reason: str | None = None) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }


async def translate_stream(
    events: AsyncIterator[dict],
    *,
    completion_id: str,
    model: str,
) -> AsyncIterator[bytes]:
    """Plain text mode: stream both content and thinking from Claude stream-json."""
    role_sent = False
    finish_reason: str | None = "stop"

    async for ev in events:
        thought = _extract_thinking_delta(ev)
        if thought:
            if not role_sent:
                yield _sse(_chunk(completion_id, model, {"role": "assistant", "content": ""}))
                role_sent = True
            yield _sse(_chunk(completion_id, model, {"reasoning_content": thought}))
            continue

        text = _extract_text_delta(ev)
        if text:
            if not role_sent:
                yield _sse(_chunk(completion_id, model, {"role": "assistant", "content": ""}))
                role_sent = True
            yield _sse(_chunk(completion_id, model, {"content": text}))
            continue

        if ev.get("type") == "result":
            sub = ev.get("subtype")
            if sub == "error_max_turns":
                finish_reason = "length"

    if not role_sent:
        yield _sse(_chunk(completion_id, model, {"role": "assistant", "content": ""}))

    yield _sse(_chunk(completion_id, model, {}, finish_reason=finish_reason))
    yield b"data: [DONE]\n\n"


async def translate_schema_stream(
    events: AsyncIterator[dict],
    *,
    completion_id: str,
    model: str,
    final_emit: Callable[[dict], Iterable[bytes]],
) -> AsyncIterator[bytes]:
    """Schema mode: stream thinking, then emit final structured payload from `result`.

    `final_emit(result_event)` receives the terminal `result` envelope (containing
    `structured_output`) and returns the closing SSE bytes — typically a delta
    with `tool_calls` or `content` plus a finish-reason chunk and the `[DONE]`
    sentinel. We delegate that to the caller because `tool_calls` vs `content`
    requires the same parsing the blocking path does.
    """
    role_sent = False
    final_result: dict | None = None

    async for ev in events:
        thought = _extract_thinking_delta(ev)
        if thought:
            if not role_sent:
                yield _sse(_chunk(completion_id, model, {"role": "assistant", "content": ""}))
                role_sent = True
            yield _sse(_chunk(completion_id, model, {"reasoning_content": thought}))
            continue

        if ev.get("type") == "result":
            final_result = ev
            # Don't break — keep draining in case of trailing events.

    if not role_sent:
        yield _sse(_chunk(completion_id, model, {"role": "assistant", "content": ""}))

    for piece in final_emit(final_result or {}):
        yield piece


def _extract_text_delta(ev: dict) -> str | None:
    if ev.get("type") != "stream_event":
        return None
    inner = ev.get("event") or {}
    delta = inner.get("delta") or {}
    if delta.get("type") == "text_delta":
        t = delta.get("text")
        if isinstance(t, str) and t:
            return t
    if inner.get("type") == "content_block_delta":
        d = inner.get("delta") or {}
        if d.get("type") == "text_delta" and isinstance(d.get("text"), str):
            return d["text"] or None
    return None


def _extract_thinking_delta(ev: dict) -> str | None:
    if ev.get("type") != "stream_event":
        return None
    inner = ev.get("event") or {}
    delta = inner.get("delta") or {}
    if delta.get("type") == "thinking_delta":
        t = delta.get("thinking")
        if isinstance(t, str) and t:
            return t
    if inner.get("type") == "content_block_delta":
        d = inner.get("delta") or {}
        if d.get("type") == "thinking_delta" and isinstance(d.get("thinking"), str):
            return d["thinking"] or None
    return None
