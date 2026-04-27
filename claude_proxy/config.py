import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    claude_bin: str
    sandbox_dir: Path
    proxy_api_key: str | None
    max_concurrent: int
    claude_timeout_s: float
    default_model: str
    log_level: str

    @classmethod
    def from_env(cls) -> "Config":
        sandbox = Path(os.environ.get("SANDBOX_DIR", "/tmp/claude-as-api/sandbox")).resolve()
        sandbox.mkdir(parents=True, exist_ok=True)
        return cls(
            claude_bin=os.environ.get("CLAUDE_BIN", "claude"),
            sandbox_dir=sandbox,
            proxy_api_key=os.environ.get("PROXY_API_KEY") or None,
            max_concurrent=int(os.environ.get("MAX_CONCURRENT", "4")),
            claude_timeout_s=float(os.environ.get("CLAUDE_TIMEOUT_S", "300")),
            default_model=os.environ.get("DEFAULT_MODEL", "claude-sonnet-4-6"),
            log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        )


MODEL_MAP: dict[str, str] = {
    "gpt-4o": "claude-opus-4-7",
    "gpt-4": "claude-opus-4-7",
    "opus": "claude-opus-4-7",
    "claude-opus-4-7": "claude-opus-4-7",
    "claude-opus": "claude-opus-4-7",
    "gpt-4o-mini": "claude-sonnet-4-6",
    "gpt-3.5-turbo": "claude-sonnet-4-6",
    "sonnet": "claude-sonnet-4-6",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "claude-sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5",
    "claude-haiku-4-5": "claude-haiku-4-5",
    "claude-haiku": "claude-haiku-4-5",
}

EXPOSED_MODELS = ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"]


def map_model(requested: str | None, default: str) -> str:
    if not requested:
        return default
    key = requested.lower().strip()
    if key in MODEL_MAP:
        return MODEL_MAP[key]
    for prefix, target in (("claude-opus", "claude-opus-4-7"),
                           ("claude-sonnet", "claude-sonnet-4-6"),
                           ("claude-haiku", "claude-haiku-4-5")):
        if key.startswith(prefix):
            return target
    return default
