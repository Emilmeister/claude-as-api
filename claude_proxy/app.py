from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import claude_runner, prompt_builder, schema_normalizer, tools_bridge
from .config import EXPOSED_MODELS, Config, map_model, normalize_effort
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ModelInfo,
    ModelList,
    Usage,
)
from .stream_translator import translate_schema_stream, translate_stream

log = logging.getLogger("claude_proxy")


def _new_completion_id() -> str:
    return f"chatcmpl-{secrets.token_hex(12)}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg: Config = app.state.config
    app.state.semaphore = asyncio.Semaphore(cfg.max_concurrent)
    log.info(
        "claude-as-api ready: cwd=%s, max_concurrent=%d, default_model=%s, auth=%s",
        cfg.sandbox_dir, cfg.max_concurrent, cfg.default_model,
        "on" if cfg.proxy_api_key else "off",
    )
    yield


def create_app(config: Config | None = None) -> FastAPI:
    cfg = config or Config.from_env()
    logging.basicConfig(level=cfg.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = FastAPI(title="claude-as-api", lifespan=lifespan)
    app.state.config = cfg

    def _check_auth(request: Request) -> None:
        if not cfg.proxy_api_key:
            return
        header = request.headers.get("authorization", "")
        if not header.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = header.split(" ", 1)[1].strip()
        if not secrets.compare_digest(token, cfg.proxy_api_key):
            raise HTTPException(status_code=401, detail="invalid api key")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models(_: None = Depends(_check_auth)) -> ModelList:
        now = int(time.time())
        return ModelList(data=[ModelInfo(id=m, created=now) for m in EXPOSED_MODELS])

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request,
                               _: None = Depends(_check_auth)):
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        if log.isEnabledFor(logging.DEBUG):
            # Surface the raw body's UTF-8 head so we can tell whether mojibake
            # is arriving from the client or being introduced by our pipeline.
            try:
                raw = await request.body()
                bad = raw.decode("utf-8", errors="replace").count("�")
                log.debug("raw body: %d bytes, head=%s, replacement_chars=%d",
                          len(raw), raw[:96].hex(" "), bad)
            except Exception as e:
                log.debug("could not snapshot raw body: %s", e)

        model = map_model(req.model, cfg.default_model)
        # Per-request `reasoning_effort` (OpenAI o1 convention) wins over the
        # server default; if the request says "auto" we honor that and pass no
        # `--effort` so the model picks for itself.
        if req.reasoning_effort is not None:
            effort = normalize_effort(req.reasoning_effort)
        else:
            effort = cfg.default_effort
        roles = [m.role for m in req.messages]
        rf_type = req.response_format.type if req.response_format else None
        log.info(
            "chat req: requested_model=%s -> %s msgs=%d roles=%s stream=%s tools=%d tool_choice=%s response_format=%s effort=%s",
            req.model, model, len(req.messages), roles, req.stream,
            len(req.tools or []), req.tool_choice, rf_type, effort or "auto",
        )

        system_prompt = prompt_builder.build_system_prompt(req.messages, req.tools)
        user_prompt = prompt_builder.build_user_prompt(req.messages)

        semaphore: asyncio.Semaphore = request.app.state.semaphore
        # Tools force a structured-output (--json-schema) call that has no useful
        # mid-flight token deltas. When the caller requested SSE we honor that by
        # emitting the final result as a single SSE chunk instead of switching
        # them silently to a JSON response (which OpenAI clients can't parse).
        pseudo_stream = req.stream and bool(req.tools)
        true_stream = req.stream and not req.tools
        if pseudo_stream:
            log.info("stream + tools: running blocking claude -p, emitting result as one SSE chunk")

        if true_stream:
            return _stream_response(request, cfg, semaphore, model, system_prompt, user_prompt, effort)
        if pseudo_stream:
            return _pseudo_stream_response(request, cfg, semaphore, req, model, system_prompt, user_prompt, effort)
        return await _blocking_response(cfg, semaphore, req, model, system_prompt, user_prompt, effort)

    return app


def _build_json_schema(req: ChatCompletionRequest) -> dict | None:
    if req.tools:
        return tools_bridge.build_response_schema(req.tools, req.tool_choice)
    if req.response_format and req.response_format.type == "json_schema":
        rf_schema = req.response_format.json_schema or {}
        raw_schema = rf_schema.get("schema", rf_schema) if isinstance(rf_schema, dict) else None
        if not raw_schema:
            return None
        normalized = schema_normalizer.normalize(raw_schema)
        if _schema_uses_union(raw_schema):
            log.info("normalized user schema: stripped oneOf/anyOf -> flat form")
        return normalized
    if req.response_format and req.response_format.type == "json_object":
        return {"type": "object"}
    return None


async def _run_and_parse(
    cfg: Config,
    semaphore: asyncio.Semaphore,
    req: ChatCompletionRequest,
    model: str,
    system_prompt: str,
    user_prompt: str,
    effort: str | None,
) -> tuple[str, str | None, list | None, dict]:
    """Run claude -p once and translate the envelope into (finish_reason, content, tool_calls, envelope)."""
    json_schema = _build_json_schema(req)

    async with semaphore:
        try:
            result = await claude_runner.run_oneshot(
                claude_bin=cfg.claude_bin,
                cwd=cfg.sandbox_dir,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=json_schema,
                timeout_s=cfg.claude_timeout_s,
                effort=effort,
            )
        except claude_runner.ClaudeError as e:
            log.error("claude error: %s", e)
            raise HTTPException(status_code=502, detail=f"claude backend error: {e}")

    envelope = result.raw
    finish_reason: str = "stop"
    content: str | None = None
    tool_calls = None

    if json_schema is not None and req.tools:
        finish_reason, content, tool_calls = tools_bridge.parse_structured_output(
            envelope.get("structured_output")
        )
    elif json_schema is not None:
        content = _coerce_to_text(envelope.get("structured_output"))
    else:
        content = envelope.get("result") or ""

    log.info(
        "chat resp: finish=%s content_chars=%d tool_calls=%d",
        finish_reason, len(content or ""), len(tool_calls or []),
    )
    if not content and not tool_calls:
        log.warning(
            "empty response: envelope subtype=%s is_error=%s result=%r structured=%r",
            envelope.get("subtype"), envelope.get("is_error"),
            (envelope.get("result") or "")[:200],
            envelope.get("structured_output"),
        )

    return finish_reason, content, tool_calls, envelope


async def _blocking_response(
    cfg: Config,
    semaphore: asyncio.Semaphore,
    req: ChatCompletionRequest,
    model: str,
    system_prompt: str,
    user_prompt: str,
    effort: str | None,
) -> JSONResponse:
    finish_reason, content, tool_calls, envelope = await _run_and_parse(
        cfg, semaphore, req, model, system_prompt, user_prompt, effort,
    )
    message = ChatMessage(
        role="assistant",
        content=content if tool_calls is None else None,
        tool_calls=tool_calls,
    )
    response = ChatCompletionResponse(
        id=_new_completion_id(),
        created=int(time.time()),
        model=model,
        choices=[Choice(index=0, message=message, finish_reason=finish_reason)],
        usage=_extract_usage(envelope),
    )
    return JSONResponse(content=response.model_dump(exclude_none=True))


def _pseudo_stream_response(
    request: Request,
    cfg: Config,
    semaphore: asyncio.Semaphore,
    req: ChatCompletionRequest,
    model: str,
    system_prompt: str,
    user_prompt: str,
    effort: str | None,
) -> StreamingResponse:
    """Stream thinking deltas live, then emit the final structured payload.

    Used when the caller requested `stream=True` with `tools` or `response_format`.
    `claude -p --output-format=stream-json --json-schema=...` is happy to combine
    those: thinking comes through as `thinking_delta` events (forwarded as
    `reasoning_content`), the model's intermediate text and the StructuredOutput
    tool's input_json are ignored, and the final `result` event carries the
    authoritative `structured_output` we translate to OpenAI tool_calls/content.
    """
    completion_id = _new_completion_id()
    json_schema = _build_json_schema(req)

    def _final_emit(result_event: dict):
        finish_reason, content, tool_calls = _parse_schema_result(req, result_event)
        if tool_calls:
            tc_delta = [
                {
                    "index": i,
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                for i, tc in enumerate(tool_calls)
            ]
            yield _sse_raw(_chunk_dict(completion_id, model, {"tool_calls": tc_delta}))
        elif content:
            yield _sse_raw(_chunk_dict(completion_id, model, {"content": content}))

        log.info(
            "chat resp (stream): finish=%s content_chars=%d tool_calls=%d",
            finish_reason, len(content or ""), len(tool_calls or []),
        )
        yield _sse_raw(_chunk_dict(completion_id, model, {}, finish_reason=finish_reason))
        yield b"data: [DONE]\n\n"

    async def gen():
        async with semaphore:
            events = claude_runner.run_streaming(
                claude_bin=cfg.claude_bin,
                cwd=cfg.sandbox_dir,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout_s=cfg.claude_timeout_s,
                json_schema=json_schema,
                effort=effort,
            )
            try:
                async for chunk in translate_schema_stream(
                    events,
                    completion_id=completion_id,
                    model=model,
                    final_emit=_final_emit,
                ):
                    if await request.is_disconnected():
                        log.info("client disconnected; aborting schema stream")
                        await events.aclose()
                        return
                    yield chunk
            except claude_runner.ClaudeError as e:
                log.error("claude schema stream error: %s", e)
                err = {"error": {"message": str(e), "type": "backend_error"}}
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _parse_schema_result(req: ChatCompletionRequest, result_event: dict) -> tuple[str, str | None, list | None]:
    """Mirror the parsing logic from `_run_and_parse` for a stream-json `result` envelope."""
    if req.tools:
        return tools_bridge.parse_structured_output(result_event.get("structured_output"))
    return "stop", _coerce_to_text(result_event.get("structured_output")), None


def _chunk_dict(completion_id: str, model: str, delta: dict,
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


def _sse_raw(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _stream_response(
    request: Request,
    cfg: Config,
    semaphore: asyncio.Semaphore,
    model: str,
    system_prompt: str,
    user_prompt: str,
    effort: str | None,
) -> StreamingResponse:
    completion_id = _new_completion_id()

    async def gen() -> AsyncIterator[bytes]:
        async with semaphore:
            events = claude_runner.run_streaming(
                claude_bin=cfg.claude_bin,
                cwd=cfg.sandbox_dir,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout_s=cfg.claude_timeout_s,
                effort=effort,
            )
            try:
                async for chunk in translate_stream(events, completion_id=completion_id, model=model):
                    if await request.is_disconnected():
                        log.info("client disconnected; aborting stream")
                        await events.aclose()
                        return
                    yield chunk
            except claude_runner.ClaudeError as e:
                log.error("claude stream error: %s", e)
                err = {"error": {"message": str(e), "type": "backend_error"}}
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _schema_uses_union(schema) -> bool:
    """Recursively check for `oneOf` / `anyOf` keys anywhere in a JSON schema."""
    if isinstance(schema, dict):
        if "oneOf" in schema or "anyOf" in schema:
            return True
        return any(_schema_uses_union(v) for v in schema.values())
    if isinstance(schema, list):
        return any(_schema_uses_union(v) for v in schema)
    return False


def _coerce_to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _extract_usage(envelope: dict) -> Usage:
    u = envelope.get("usage") or {}
    prompt = (
        int(u.get("input_tokens") or 0)
        + int(u.get("cache_read_input_tokens") or 0)
        + int(u.get("cache_creation_input_tokens") or 0)
    )
    completion = int(u.get("output_tokens") or 0)
    return Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion)
