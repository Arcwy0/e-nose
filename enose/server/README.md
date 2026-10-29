# enose/server — FastAPI Server

## Starting the server

Install the full shared lock from the repository root (Python 3.10–3.12):

```bash
python -m pip install -r requirements-server.txt
python -m pip install --no-deps -e .
# uv alternative: uv pip sync --python .venv requirements-server.txt
#                 uv pip install --python .venv --no-deps -e .
```

The same pinned versions are used by Docker. Docker/Linux CUDA retain the
separate FlashAttention 2.7.3 extension; native Windows uses eager attention.
See [`docs/SETUP.md`](../../docs/SETUP.md) for environment creation and weights.

```bash
# Smell-only (no VLM, fast start):
ENOSE_NO_VLM=1 python scripts/run_server.py

# Full (VLM + classifier):
unset ENOSE_NO_VLM
python scripts/verify_server_environment.py --vlm
python scripts/run_server.py --port 8080
```

Windows PowerShell, full mode on an alternate port:

```powershell
$env:ENOSE_NO_VLM = "0"
python scripts/verify_server_environment.py --vlm
python scripts/run_server.py --host 0.0.0.0 --port 8016
```

For smell-only PowerShell use `$env:ENOSE_NO_VLM = "1"`. Change `--host` to
`127.0.0.1` for local-only access. The default port is 8080; `--port 8016` uses
`http://localhost:8016/ui`. Follow Uvicorn's listening-address line if the app
banner still shows its fixed example address. Each server worker loads its
own model, so keep the default single worker unless memory permits more.

The web UI is at **http://localhost:8080/ui**.  
Interactive API docs are at **http://localhost:8080/docs**.

Version 3.3.0 also installs a working `enose-server --host ... --port ...`
console command. Use one worker: buffers, live baseline identity, and deployment
locks are process-local. The Train tab previews configurable CSV extraction and
stages candidates by default, preserving the active model/history until explicit
promotion. Model Info supports independent paired evaluation and rollback.
Missing independent exposures produce null metrics with an explanation, not a
row-wise substitute. These endpoints have no built-in authentication: restrict
the server to a trusted private network/proxy. No HTTP route enables heaters.

Vision request deadlines are client settings (`--vision-timeout`,
`--scene-timeout`, or `ENOSE_VISION_TIMEOUT` / `ENOSE_SCENE_TIMEOUT`). Raising
them permits slower CPU inference; it does not make the model faster. PCB
heater controls and preflight live in the acquisition client, not this server.

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
| `CSVLearningData` | `/smell/learn_from_csv` | `csv_data`, `target_column`, augmentation params, `merge_history` (default `true`) |
