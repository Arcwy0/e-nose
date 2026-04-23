# enose/server — FastAPI Server

## Starting the server

```bash
# Smell-only (no VLM, fast start):
ENOSE_NO_VLM=1 python scripts/run_server.py

# Full (VLM + classifier):
python scripts/run_server.py --port 8080 --reload
```

The web UI is at **http://localhost:8080/ui**.  
Interactive API docs are at **http://localhost:8080/docs**.

## Module layout

```
server/
├── app.py           — create_app(), lifespan, startup_logic()
├── state.py         — module-level singletons (vlm_model, smell_classifier)
├── live_buffer.py   — thread-safe ring buffer for live samples
├── model_loader.py  — load_or_create_classifier(), save_training_data()
├── schemas.py       — Pydantic request models
└── routes/
    ├── health.py    — GET /  /health  /smell/model_info
    ├── smell.py     — POST /smell/classify (now with OOD)  /smell/test_console  /smell/debug_input
    ├── training.py  — POST /smell/online_learning  /smell/learn_from_csv  /training_pipeline
    ├── analytics.py — GET /smell/visualize_data  /smell/analyze_data  /smell/environmental_analysis  /smell/drift
    ├── live.py      — POST /sensor/live/push · GET /sensor/live/recent · /stats · DELETE /clear
    └── ui.py        — GET /ui  (browser interface, Classify/Train/Live/Model-Info tabs)
```

## State management

`state.py` holds three module-level variables:

```python
vlm_model      # Florence-2 model or None
vlm_processor  # Florence-2 processor or None
smell_classifier  # BalancedRFClassifier or None
```

Use `state.require_classifier()` (raises 503 if None) or `state.require_fitted_classifier()` (raises 503 if not fitted) inside route handlers.

## Startup lifecycle

`startup_logic()` runs during the FastAPI lifespan:

1. Create `data/`, `models/`, `trained_models/`, `plots/` directories.
2. Detect GPU via `nvidia-smi`.
3. If `ENOSE_NO_VLM != 1`: load Florence-2. Any failure here aborts startup.
4. Load or create classifier (failure → empty unfitted classifier, server still starts).

## Adding a new route

1. Create `enose/server/routes/my_route.py` with a `router = APIRouter(...)`.
2. Add `from . import my_route` to `routes/__init__.py`.
3. Add `app.include_router(my_route.router)` in `app.py:create_app()`.

## Environment variables

| Variable | Effect |
|---|---|
| `ENOSE_NO_VLM=1` | Skip Florence-2, start with smell classifier only |
| `ENOSE_CLASSIFIER` | Classifier backend: `balanced_rf` (default) or `xgb` |
| `SERVER_HOST` | Override bind host (default `0.0.0.0`) |
| `SERVER_PORT` | Override port (default `8080`) |

## Running in Docker

Typical setup: GPU Docker container for the server, reachable on the
local machine at `http://localhost:8016/` or across a LAN / private
tunnel at whatever address the host is exposed on.

### 1. Network and server container

```bash
docker network create enose-net         # once

docker run -it --name florence2 \
    --gpus '"device=0"' \
    --network enose-net \
    -p 8016:8080 \
    -v "$(pwd)":/app \
    enose-server
```

Inside the container:

```bash
ENOSE_CLASSIFIER=xgb python scripts/run_server.py
# add ENOSE_NO_VLM=1 at the front for smell-only mode
```

The server now listens on container port 8080, mapped to host port **8016**.

### 2. Remote access (optional)

If colleagues need to reach the server from another machine, expose host
port 8016 via your preferred mechanism (LAN IP, VPN like WireGuard or
Tailscale, or a private TCP tunnel). The repo is deliberately tunnel-
agnostic — pick whatever fits your network.

The UI uses relative paths for every `fetch()` call, so it works
identically behind any reverse proxy or TCP forwarder.

### 3. Teardown

```bash
docker stop florence2
docker rm florence2
```

See [`docs/SERVER_CLIENT_GUIDE.md`](../../docs/SERVER_CLIENT_GUIDE.md) for the client-side instructions, including Docker with UART pass-through.

## Schemas

| Schema | Used by | Fields |
|---|---|---|
| `SensorData` | `/smell/classify` | 22 float fields (R1–R17, T, H, CO2, H2S, CH2O) |
| `ConsoleSensorData` | `/smell/test_console` | `values: str` (comma-separated) |
| `OnlineLearningData` | `/smell/online_learning` | `sensor_data: list[dict]`, `labels: list[str]` |
| `CSVLearningData` | `/smell/learn_from_csv` | `csv_data`, `target_column`, augmentation params |
