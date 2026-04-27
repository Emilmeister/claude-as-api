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
from .config import EXPOSED_MODELS, Config, map_model
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ModelInfo,
    ModelList,
    Usage,
)
from .stream_translator import translate_stream

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

        model = map_model(req.model, cfg.default_model)
        system_prompt = prompt_builder.build_system_prompt(req.messages, req.tools)
        user_prompt = prompt_builder.build_user_prompt(req.messages)

        semaphore: asyncio.Semaphore = request.app.state.semaphore
        force_nonstream = bool(req.tools)
        wants_stream = req.stream and not force_nonstream

        if wants_stream:
            return _stream_response(request, cfg, semaphore, model, system_prompt, user_prompt)
        return await _blocking_response(cfg, semaphore, req, model, system_prompt, user_prompt)

    return app


async def _blocking_response(
    cfg: Config,
    semaphore: asyncio.Semaphore,
    req: ChatCompletionRequest,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> JSONResponse:
    json_schema: dict | None = None
    if req.tools:
        json_schema = tools_bridge.build_response_schema(req.tools, req.tool_choice)
    elif req.response_format and req.response_format.type == "json_schema":
        rf_schema = req.response_format.json_schema or {}
        raw_schema = rf_schema.get("schema", rf_schema) if isinstance(rf_schema, dict) else None
        if raw_schema:
            json_schema = schema_normalizer.normalize(raw_schema)
            if _schema_uses_union(raw_schema):
                log.info("normalized user schema: stripped oneOf/anyOf -> flat form")
    elif req.response_format and req.response_format.type == "json_object":
        json_schema = {"type": "object"}

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


def _stream_response(
    request: Request,
    cfg: Config,
    semaphore: asyncio.Semaphore,
    model: str,
    system_prompt: str,
    user_prompt: str,
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
