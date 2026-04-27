"""Translate Claude `stream-json` events into OpenAI Chat Completion SSE chunks."""
from __future__ import annotations

import json
import time
from typing import AsyncIterator, Iterable

from .schemas import ChatCompletionChunk, DeltaMessage, StreamChoice


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _chunk(completion_id: str, model: str, delta: DeltaMessage,
           finish_reason: str | None = None) -> dict:
    chunk = ChatCompletionChunk(
        id=completion_id,
        created=int(time.time()),
        model=model,
        choices=[StreamChoice(delta=delta, finish_reason=finish_reason)],
    )
    return chunk.model_dump(exclude_none=True)


async def translate_stream(
    events: AsyncIterator[dict],
    *,
    completion_id: str,
    model: str,
) -> AsyncIterator[bytes]:
    """Consume Claude stream-json events; yield raw SSE bytes for the OpenAI client."""
    role_sent = False
    saw_text = False
    finish_reason: str | None = "stop"

    async for ev in events:
        ev_type = ev.get("type")

        # First content delta — emit role chunk first per OpenAI convention.
        delta_text = _extract_text_delta(ev)
        if delta_text is not None:
            if not role_sent:
                yield _sse(_chunk(completion_id, model, DeltaMessage(role="assistant", content="")))
                role_sent = True
            saw_text = True
            yield _sse(_chunk(completion_id, model, DeltaMessage(content=delta_text)))
            continue

        if ev_type == "result":
            # Final envelope; map subtype to finish_reason.
            sub = ev.get("subtype")
            if sub == "error_max_turns":
                finish_reason = "length"
            elif sub and "error" in sub:
                finish_reason = "stop"

    if not role_sent:
        # No text streamed (e.g. tool use or empty); still emit a minimal opener so clients are happy.
        yield _sse(_chunk(completion_id, model, DeltaMessage(role="assistant", content="")))

    yield _sse(_chunk(completion_id, model, DeltaMessage(), finish_reason=finish_reason))
    yield b"data: [DONE]\n\n"


def _extract_text_delta(ev: dict) -> str | None:
    """Pull a text delta out of a stream-json event, if present.

    Handles both the partial-message form (`stream_event.delta.text_delta.text`)
    and assistant content blocks (`assistant.message.content[*].text`) as a fallback.
    """
    if ev.get("type") == "stream_event":
        inner = ev.get("event") or {}
        delta = inner.get("delta") or {}
        if delta.get("type") == "text_delta":
            t = delta.get("text")
            if isinstance(t, str) and t:
                return t
        # content_block_delta nested form
        if inner.get("type") == "content_block_delta":
            d = inner.get("delta") or {}
            if d.get("type") == "text_delta" and isinstance(d.get("text"), str):
                return d["text"]
    return None
