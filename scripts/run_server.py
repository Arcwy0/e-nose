#!/usr/bin/env python3
"""Launch the FastAPI server with uvicorn. Honors SERVER_HOST / SERVER_PORT env vars."""

from __future__ import annotations

import argparse
import os

import uvicorn

from enose.config import SERVER_HOST, SERVER_PORT


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the E-Nose FastAPI server")
    parser.add_argument("--host", default=os.environ.get("SERVER_HOST", SERVER_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SERVER_PORT", SERVER_PORT)))
    parser.add_argument("--reload", action="store_true", help="Dev auto-reload (watches enose/)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    uvicorn.run(
        "enose.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()
