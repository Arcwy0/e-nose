"""FastAPI server. Run via scripts/run_server.py or `uvicorn enose.server.app:app`."""

from .app import create_app, app

__all__ = ["create_app", "app"]
