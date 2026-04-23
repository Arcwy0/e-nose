"""FastAPI app factory — `create_app()` builds the full server with routes mounted.

Run via `uvicorn enose.server.app:app` or the `scripts/run_server.py` helper.
Startup lifecycle loads Florence-2 VLM and the smell classifier into `state`.
"""

from __future__ import annotations

import os
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from enose.config import DATA_DIR, PLOTS_DIR, TRAINED_MODELS_DIR, VLM_MODEL_PATH
from enose.vision.gpu import check_gpu_availability

from . import state
from .model_loader import load_or_create_classifier
from .routes import analytics, health, live, smell, training, ui

# Set ENOSE_NO_VLM=1 to start without loading Florence-2 (classifier-only / offline testing mode).
_NO_VLM = os.environ.get("ENOSE_NO_VLM", "0").lower() in ("1", "true", "yes")


async def startup_logic() -> None:
    """Create dirs, detect GPU, optionally load VLM, load classifier. Populates `state`."""
    for d in ("data", "models", TRAINED_MODELS_DIR, "plots"):
        os.makedirs(d, exist_ok=True)

    device = check_gpu_availability()

    if _NO_VLM:
        print("ENOSE_NO_VLM=1 — skipping Florence-2 VLM (vision endpoints will return 503).")
    else:
        from enose.vision.florence import load_florence_model
        print("Loading Florence-2 VLM …")
        try:
            model, processor = load_florence_model(device, model_path=VLM_MODEL_PATH)
            state.set_vlm(model, processor)
        except Exception as e:
            print(f"✗ Critical: VLM load failed: {e}")
            traceback.print_exc()
            raise

    print("Loading smell classifier …")
    try:
        state.set_classifier(load_or_create_classifier())
    except Exception as e:
        print(f"✗ Smell classifier load error: {e}")
        traceback.print_exc()
        from enose.classifier import BalancedRFClassifier
        state.set_classifier(BalancedRFClassifier())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: run startup, yield, then cleanup on shutdown."""
    await startup_logic()
    print("=" * 60)
    print("E-Nose Multimodal Server v3.1.0 ready on http://0.0.0.0:8080")
    print("Docs at /docs")
    print("=" * 60)
    yield
    print("Server shutting down…")


def create_app() -> FastAPI:
    """Build the FastAPI instance with static mounts, routers, and the global exception handler."""
    app = FastAPI(
        title="E-Nose Multimodal Server",
        description="Florence-2 vision + 22-feature smell classification",
        version="3.1.0",
        lifespan=lifespan,
    )

    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    app.mount(f"/{PLOTS_DIR}", StaticFiles(directory=PLOTS_DIR), name="plots")
    app.mount(f"/{DATA_DIR}", StaticFiles(directory=DATA_DIR), name="data")

    app.include_router(health.router)
    app.include_router(smell.router)
    app.include_router(training.router)
    app.include_router(analytics.router)
    app.include_router(live.router)
    app.include_router(ui.router)

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        print(f"Unhandled error: {exc}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc), "type": type(exc).__name__},
        )

    return app


# Module-level instance so uvicorn can target `enose.server.app:app` directly.
app = create_app()
