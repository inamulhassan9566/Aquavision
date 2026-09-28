"""
AQUAVISION: AI Maritime Intelligence Platform
Vercel Serverless Function Entry Point (ASGI)

Mounts the FastAPI application for Vercel Serverless Python runtime.
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

# Expose app for Vercel Python runtime
__all__ = ["app"]
