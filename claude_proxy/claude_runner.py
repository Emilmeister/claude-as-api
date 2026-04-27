"""Spawn `claude -p` subprocesses with subscription-safe env and a sterile cwd."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

log = logging.getLogger(__name__)


def _truncate(s: str, n: int = 800) -> str:
    if len(s) <= n:
        return s
    return f"{s[:n]} ...[+{len(s) - n} chars]"


def _hex_head(s: str, n: int = 96) -> str:
    """Hex of the first `n` UTF-8 bytes plus a count of replacement chars (\\ufffd).

    Useful for checking whether Cyrillic / emoji round-trips intact. If the
    string already contains U+FFFD, the corruption happened upstream of us
    (request body or client) — not in our subprocess pipeline.
    """
    head = s[:n].encode("utf-8", errors="replace")
    bad = s.count("�")
    return f"{head.hex(' ')} (len={len(s)} chars, replacement_chars={bad})"


# Note on tool blocking:
# `--disallowedTools` blocks *execution* but does not strip tools from the Anthropic API
# request body — and one of Claude Code's built-in tools (`TaskOutput`) ships with an
# input_schema that the API rejects when combined with `--json-schema`. Passing
# `--allowedTools=` empty avoids that path and (with `--permission-mode=dontAsk`) prevents
# any tool execution. Cost: a small startup overhead from Claude still loading its tool
# registry into context, but no API rejections.


def _empty_mcp_config_path() -> str:
    """Path to an empty MCP config file used to suppress user-level MCP servers.

    Without this, Claude Code auto-loads MCP servers from `~/.claude/` and inflates the
    cached context by ~15k+ tokens. Writing a `{"mcpServers":{}}` file and passing it via
    `--mcp-config=<path>` overrides discovery and keeps the run lean.
    """
    global _MCP_PATH
    if _MCP_PATH and Path(_MCP_PATH).exists():
        return _MCP_PATH
    fd, path = tempfile.mkstemp(prefix="claude-as-api-mcp-", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump({"mcpServers": {}}, f)
    _MCP_PATH = path
    return path


_MCP_PATH: str | None = None

# Env vars that would force pay-as-you-go billing or non-Anthropic providers.
# We strip these from the subprocess so OAuth / Claude Code subscription is used.
_BILLING_OVERRIDES = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "AWS_BEARER_TOKEN_BEDROCK",
)


def sanitized_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _BILLING_OVERRIDES}
    # Force a UTF-8 locale for the subprocess so Cyrillic / emoji / other multi-byte
    # text round-trips cleanly. Many minimal VM/container images come with `LANG=C`
    # which makes argv / stdout encoding fall back to ASCII and replace bytes with
    # `?` or `�`. We only override when the parent locale isn't already UTF-8.
    if not _looks_utf8(env.get("LC_ALL", "")) and not _looks_utf8(env.get("LANG", "")):
        env["LC_ALL"] = "C.UTF-8"
        env["LANG"] = "C.UTF-8"
    return env


def _looks_utf8(value: str) -> bool:
    v = value.lower()
    return "utf-8" in v or "utf8" in v


@dataclass
class ClaudeResult:
    raw: dict
    stderr: str


class ClaudeError(RuntimeError):
    pass


def _base_args(model: str, system_prompt: str, effort: str | None = None) -> list[str]:
    args = [
        "-p",
        f"--model={model}",
        f"--mcp-config={_empty_mcp_config_path()}",
        f"--system-prompt={system_prompt}",
        "--allowedTools=",
        "--permission-mode=dontAsk",
        # Without these, Claude Code auto-loads user-level skills and persists sessions
        # — both can inject content into the subprocess (skills like `claude-api` may
        # auto-trigger on certain prompts; persisted sessions waste disk and pollute history).
        "--disable-slash-commands",
        "--no-session-persistence",
    ]
    if effort:
        args.append(f"--effort={effort}")
    if os.environ.get("CLAUDE_DEBUG"):
        # Streams diagnostics for hooks / API / file events to the subprocess stderr.
        # Useful when a stray hook in `~/.claude/settings.json` is hijacking responses.
        args.append("--debug=api,hooks")
    return args


async def run_oneshot(
    *,
    claude_bin: str,
    cwd: Path,
    model: str,
    system_prompt: str,
    user_prompt: str,
    json_schema: dict | None,
    timeout_s: float,
    effort: str | None = None,
) -> ClaudeResult:
    """Run claude -p once and return the parsed JSON envelope."""
    args = [claude_bin, *_base_args(model, system_prompt, effort), "--output-format=json"]
    if json_schema is not None:
        args.append(f"--json-schema={json.dumps(json_schema, ensure_ascii=False)}")
    args.append(user_prompt)

    log.info(
        "claude -p spawn: model=%s cwd=%s system_chars=%d user_chars=%d schema=%s",
        model, cwd, len(system_prompt), len(user_prompt),
        "yes" if json_schema is not None else "no",
    )
    if log.isEnabledFor(logging.DEBUG):
        log.debug("system prompt: %s", _truncate(system_prompt))
        log.debug("user prompt: %s", _truncate(user_prompt))
        log.debug("user prompt utf8 bytes head: %s", _hex_head(user_prompt))
        if json_schema is not None:
            log.debug("json_schema: %s", _truncate(json.dumps(json_schema, ensure_ascii=False)))
        # Mask huge prompt args in argv but show flag shape.
        log.debug("argv flags: %s", [a if not a.startswith(("--system-prompt=", "--json-schema=")) else a.split("=", 1)[0] + "=<...>" for a in args[:-1]])

    started = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.DEVNULL,  # claude -p otherwise waits 3s on stdin
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=sanitized_env(),
        cwd=str(cwd),
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=2)
        except asyncio.TimeoutError:
            proc.kill()
        raise ClaudeError(f"claude -p timed out after {timeout_s}s")

    duration = time.monotonic() - started
    stderr = stderr_b.decode("utf-8", errors="replace")
    stdout = stdout_b.decode("utf-8", errors="replace").strip()

    # Try to parse stdout regardless of exit code — claude -p often emits a JSON
    # envelope with `is_error: true` on Anthropic API failures while exiting non-zero.
    envelope: dict | None = None
    if stdout:
        try:
            envelope = json.loads(stdout)
        except json.JSONDecodeError:
            envelope = None

    if proc.returncode != 0 or (isinstance(envelope, dict) and envelope.get("is_error")):
        # Surface the most useful error string we can find.
        api_msg = ""
        if isinstance(envelope, dict):
            api_msg = (envelope.get("result") or "").strip()
        log.error(
            "claude -p failed: rc=%s duration=%.2fs is_error=%s api_msg=%s stderr=%s",
            proc.returncode, duration,
            envelope.get("is_error") if isinstance(envelope, dict) else None,
            _truncate(api_msg, 600),
            _truncate(stderr.strip(), 600),
        )
        if log.isEnabledFor(logging.DEBUG) and isinstance(envelope, dict):
            log.debug("full error envelope: %s", _truncate(json.dumps(envelope, ensure_ascii=False), 4000))
        detail = api_msg or stderr.strip() or f"exit code {proc.returncode}, no output"
        raise ClaudeError(detail)

    if envelope is None:
        raise ClaudeError(
            f"empty/invalid stdout from claude -p; stderr={_truncate(stderr.strip(), 500)}"
        )

    usage = envelope.get("usage") or {}
    log.info(
        "claude -p ok: rc=0 duration=%.2fs turns=%s subtype=%s in_tok=%s cache_creation=%s cache_read=%s out_tok=%s cost=$%.5f",
        duration,
        envelope.get("num_turns"),
        envelope.get("subtype"),
        usage.get("input_tokens"),
        usage.get("cache_creation_input_tokens"),
        usage.get("cache_read_input_tokens"),
        usage.get("output_tokens"),
        envelope.get("total_cost_usd") or 0.0,
    )
    if log.isEnabledFor(logging.DEBUG):
        result_text = envelope.get("result") or ""
        struct = envelope.get("structured_output")
        log.debug("result text: %s", _truncate(result_text, 600))
        if struct is not None:
            log.debug("structured_output: %s", _truncate(json.dumps(struct, ensure_ascii=False), 800))
        if stderr.strip():
            log.debug("stderr (success): %s", _truncate(stderr.strip(), 2000))
    return ClaudeResult(raw=envelope, stderr=stderr)


async def run_streaming(
    *,
    claude_bin: str,
    cwd: Path,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_s: float,
    json_schema: dict | None = None,
    effort: str | None = None,
) -> AsyncIterator[dict]:
    """Yield parsed stream-json events from claude -p.

    Caller is responsible for terminating iteration on disconnect; the subprocess
    is wrapped so it gets killed when the async generator is closed.
    """
    args = [
        claude_bin,
        *_base_args(model, system_prompt, effort),
        "--output-format=stream-json",
        "--verbose",
        "--include-partial-messages",
    ]
    if json_schema is not None:
        args.append(f"--json-schema={json.dumps(json_schema, ensure_ascii=False)}")
    args.append(user_prompt)

    log.info(
        "claude -p stream spawn: model=%s cwd=%s system_chars=%d user_chars=%d",
        model, cwd, len(system_prompt), len(user_prompt),
    )
    if log.isEnabledFor(logging.DEBUG):
        log.debug("system prompt: %s", _truncate(system_prompt))
        log.debug("user prompt: %s", _truncate(user_prompt))

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.DEVNULL,  # claude -p otherwise waits 3s on stdin
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=sanitized_env(),
        cwd=str(cwd),
    )

    async def _kill():
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
            except asyncio.TimeoutError:
                proc.kill()

    assert proc.stdout is not None
    started = time.monotonic()
    n_events = 0
    last_result: dict | None = None
    try:
        while True:
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=timeout_s)
            except asyncio.TimeoutError:
                await _kill()
                raise ClaudeError(f"claude -p stream stalled (timeout {timeout_s}s)")
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                log.warning("non-JSON line on stream: %r", text[:200])
                continue
            n_events += 1
            if isinstance(event, dict) and event.get("type") == "result":
                last_result = event
            if log.isEnabledFor(logging.DEBUG) and n_events <= 5:
                log.debug("stream event #%d type=%s", n_events, event.get("type") if isinstance(event, dict) else "?")
            yield event
        rc = await proc.wait()
        duration = time.monotonic() - started
        if rc != 0 or (last_result and last_result.get("is_error")):
            stderr = (await proc.stderr.read()).decode("utf-8", errors="replace") if proc.stderr else ""
            api_msg = (last_result.get("result") or "").strip() if last_result else ""
            log.error(
                "claude -p stream failed: rc=%s duration=%.2fs events=%d api_msg=%s stderr=%s",
                rc, duration, n_events, _truncate(api_msg, 600), _truncate(stderr.strip(), 600),
            )
            raise ClaudeError(api_msg or stderr.strip() or f"exit code {rc}, no output")
        log.info("claude -p stream ok: duration=%.2fs events=%d", duration, n_events)
    finally:
        await _kill()
