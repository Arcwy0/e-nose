#!/usr/bin/env python3
"""Launch the FastAPI server with uvicorn. Honors SERVER_HOST / SERVER_PORT env vars."""

from __future__ import annotations

from enose.server.cli import main


if __name__ == "__main__":
    main()
