"""Entry point for `uvicorn main:app` and `python main.py`."""
from __future__ import annotations

import os

import uvicorn

from claude_proxy.app import create_app

app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=bool(os.environ.get("RELOAD")),
    )
