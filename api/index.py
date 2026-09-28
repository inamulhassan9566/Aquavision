"""
AQUAVISION: AI Maritime Intelligence Platform
Vercel Serverless Function Entry Point (ASGI)

Mounts the FastAPI application for Vercel Serverless Python runtime
and automatically resolves Vercel rewrite paths.
"""

import sys
import os
from pathlib import Path

# Ensure project root is at the head of sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import FastAPI app from backend
from backend.main import app
from starlette.types import ASGIApp, Receive, Scope, Send


class VercelRewriteMiddleware:
    """
    Vercel ASGI rewrite handler:
    Restores the original requested URL from x-matched-path when Vercel forwards
    requests to /api/index.py so FastAPI routes match cleanly.
    """
    def __init__(self, asgi_app: ASGIApp):
        self.asgi_app = asgi_app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope.get("type") == "http":
            raw_path = scope.get("path", "")
            if raw_path == "/api/index.py" or raw_path.startswith("/api/index.py"):
                headers = dict(scope.get("headers", []))
                matched = headers.get(b"x-matched-path", b"").decode("utf-8")
                if matched and not matched.startswith("/api/index.py"):
                    scope["path"] = matched
                else:
                    scope["path"] = "/"
        await self.asgi_app(scope, receive, send)


app.add_middleware(VercelRewriteMiddleware)

# Expose app for Vercel Python runtime
__all__ = ["app"]
