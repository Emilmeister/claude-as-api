"""Spawn `claude -p` subprocesses with subscription-safe env and a sterile cwd."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

log = logging.getLogger(__name__)


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
    return env


@dataclass
class ClaudeResult:
    raw: dict
    stderr: str


class ClaudeError(RuntimeError):
    pass


def _base_args(model: str, system_prompt: str) -> list[str]:
    return [
        "-p",
        f"--model={model}",
        f"--mcp-config={_empty_mcp_config_path()}",
        f"--system-prompt={system_prompt}",
        "--allowedTools=",
        "--permission-mode=dontAsk",
    ]


async def run_oneshot(
    *,
    claude_bin: str,
    cwd: Path,
    model: str,
    system_prompt: str,
    user_prompt: str,
    json_schema: dict | None,
    timeout_s: float,
) -> ClaudeResult:
    """Run claude -p once and return the parsed JSON envelope."""
    args = [claude_bin, *_base_args(model, system_prompt), "--output-format=json"]
    if json_schema is not None:
        args.append(f"--json-schema={json.dumps(json_schema, ensure_ascii=False)}")
    args.append(user_prompt)

    log.debug("spawning claude: cwd=%s argv[0..3]=%s", cwd, args[:4])

    proc = await asyncio.create_subprocess_exec(
        *args,
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

    stderr = stderr_b.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        raise ClaudeError(
            f"claude -p exited with code {proc.returncode}: {stderr.strip()[:500]}"
        )

    stdout = stdout_b.decode("utf-8", errors="replace").strip()
    if not stdout:
        raise ClaudeError(f"empty stdout from claude -p; stderr={stderr.strip()[:500]}")
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise ClaudeError(f"failed to parse claude JSON envelope: {e}; head={stdout[:300]!r}")
    return ClaudeResult(raw=envelope, stderr=stderr)


async def run_streaming(
    *,
    claude_bin: str,
    cwd: Path,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_s: float,
) -> AsyncIterator[dict]:
    """Yield parsed stream-json events from claude -p.

    Caller is responsible for terminating iteration on disconnect; the subprocess
    is wrapped so it gets killed when the async generator is closed.
    """
    args = [
        claude_bin,
        *_base_args(model, system_prompt),
        "--output-format=stream-json",
        "--verbose",
        "--include-partial-messages",
        user_prompt,
    ]

    proc = await asyncio.create_subprocess_exec(
        *args,
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
            yield event
        rc = await proc.wait()
        if rc != 0:
            stderr = (await proc.stderr.read()).decode("utf-8", errors="replace") if proc.stderr else ""
            raise ClaudeError(f"claude -p (stream) exited with {rc}: {stderr.strip()[:500]}")
    finally:
        await _kill()
