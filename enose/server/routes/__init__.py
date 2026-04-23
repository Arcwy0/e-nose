"""FastAPI routers. Register a new router in `enose.server.app.create_app`."""

from . import analytics, health, live, smell, training, ui

__all__ = ["health", "smell", "training", "analytics", "live", "ui"]
