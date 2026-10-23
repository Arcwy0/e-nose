# E-Nose — Vision + Smell Multimodal Classification

A robotic olfaction system combining a **Florence-2** vision model with a **17-channel electronic nose**, with five optional environmental channels. Runs on a Unitree Go-1 robot dog; smell models can use R1–R17 only.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  FastAPI Server  (GPU machine / Docker)             │
│  ├── Florence-2 VLM   (object detection)            │
│  └── BalancedRF classifier  (smell classification)  │
└──────────────────┬──────────────────────────────────┘
                   │ REST / JSON
┌──────────────────▼──────────────────────────────────┐
│  Client  (laptop / robot)                           │
│  ├── ENoseSensor   (e-nose UART; env optional)     │
│  ├── WebcamHandler (OpenCV capture)                 │
│  └── TrainingPipeline (interactive CLI)             │
└─────────────────────────────────────────────────────┘
```

## Quick start

> **First time on this machine?** Read [`docs/SETUP.md`](docs/SETUP.md) —
> it walks through the venv, `pip install`, Florence-2 download, and
> `.env` setup. The steps below assume that's done.

### Install the server — one lock for Docker, pip, and uv

Use **64-bit Python 3.10–3.12** on Windows/Linux; Python 3.10 matches Docker.
`requirements-server.txt` locks the complete runtime, including Torch
**2.4.1+cu124**, torchvision **0.19.1+cu124**, Transformers **4.44.2**, and
tokenizers **0.19.1**. Keep the server and hardware client in separate environments.

**Windows PowerShell, fresh server environment:**

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade "pip==24.3.1"
python -m pip install -r requirements-server.txt
python -m pip install --no-deps -e .
python -m pip check
```

**Linux, fresh server environment:**

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade "pip==24.3.1"
python -m pip install -r requirements-server.txt
python -m pip install --no-deps -e .
python -m pip check
```

**uv alternative, same runtime versions:**

```bash
uv venv --python 3.10 .venv
uv pip sync --python .venv requirements-server.txt
uv pip install --python .venv --no-deps -e .
uv pip check --python .venv
```

Use `uv pip sync` with this lock; `uv sync` is not the documented installation
path. Activate the environment before the commands below. Docker installs
this same lock and retains **FlashAttention 2.7.3**. Native Windows, or Linux
without a CUDA compiler, uses the tested eager-attention fallback. Native Linux
CUDA extension installation is described in [`docs/SETUP.md`](docs/SETUP.md#flashattention--retained-for-docker--linux-cuda).

For an existing installation, create a fresh environment from the lock rather
than upgrading individual ML libraries. The old **3.2.0 wheel does not contain
the 3.2.1 dependency/loader fixes**; client-only users can continue using it.
Use [release v3.2.2](https://github.com/Arcwy0/e-nose/releases/tag/v3.2.2) for
the additional tokenizer-file recovery and detection decoding/display fixes.
Update both the server and camera client; the dependency lock is unchanged.

Florence weights, trained smell models, and experimental datasets are separate
from the wheel and source checkout. Follow the [download instructions](docs/SETUP.md#3-download-florence-2-large)
and [migration guide](docs/MIGRATION.md) when moving PCs.

### 1 — Server (classifier only, no GPU / VLM required)

```bash
# Start without Florence-2 — all smell endpoints work normally
ENOSE_NO_VLM=1 python scripts/run_server.py
# Or with auto-reload for development:
ENOSE_NO_VLM=1 python scripts/run_server.py --reload
```

Windows PowerShell:

```powershell
$env:ENOSE_NO_VLM = "1"
python scripts/run_server.py --host 0.0.0.0 --port 8016
```

Open **http://localhost:8080/ui** for the default port, or
**http://localhost:8016/ui** for the PowerShell example.

### 2 — Server (full, with VLM)

```bash
unset ENOSE_NO_VLM
python scripts/verify_server_environment.py --vlm
python scripts/run_server.py
```

Windows PowerShell:

```powershell
$env:ENOSE_NO_VLM = "0"
python scripts/verify_server_environment.py --vlm
python scripts/run_server.py --host 0.0.0.0 --port 8016
```

Requires Florence-2 weights at `model/Florence-2-Large/`. The verification command
checks library versions, runs real server startup, and generates a caption.
Some official downloads omit `merges.txt`; the loader reconstructs missing
slow-tokenizer files from that snapshot's own `tokenizer.json`. The model folder
must be writable for this one-time recovery. Separate `added_tokens.json` and
`special_tokens_map.json` files are not required for the official snapshot.
See [Florence model files](docs/FLORENCE_MODEL_FILES.md) for why folders differ,
the pinned download command, and repair/export instructions for existing installs.
Use `--device cpu` or `--device cuda` to select the verification device explicitly.
CPU inference is supported; a CUDA GPU improves latency. Set `--host 127.0.0.1`
for access only from this PC, or `0.0.0.0` for LAN access. If a port is busy,
choose another with `--port`; remote clients connect to the PC's IP, not `0.0.0.0`.

Docker uses the same runtime lock:

```bash
docker build -t enose-server -f docker/Dockerfile-cu124 .
docker run --rm --gpus all -p 8016:8080 -v "$PWD":/app -w /app \
  -e PYTHONPATH=/app -e ENOSE_NO_VLM=0 enose-server python scripts/run_server.py
```

For Docker smell-only mode, omit `--gpus all` and set `ENOSE_NO_VLM=1`.

### 3 — Client (offline / simulated sensors)

```bash
python scripts/run_client.py --offline --server http://localhost:8080
```

### 4 — Client (hardware)

```bash
# Linux
python scripts/run_client.py --port_enose /dev/ttyUSB0 --port_UART /dev/ttyUSB1 --server http://localhost:8080

# Windows
python scripts/run_client.py --port_enose COM3 --port_UART COM4 --server http://localhost:8080
```

### 5 — Remote clients (colleagues on the LAN or a private tunnel)

Point `--server` at whatever base URL the server is reachable at (LAN
IP, VPN, private TCP tunnel — see [`docs/SERVER_CLIENT_GUIDE.md`](docs/SERVER_CLIENT_GUIDE.md)
for Docker pass-through and other deployment notes). The client-side
command is identical to the local case above, just with a different
`--server` value.

## Testing without any hardware

| What you want | How |
|---|---|
| Classify a single sample | Browser UI → **Classify** tab, paste 17 resistance values or all 22 values |
| Classify via API | `POST /smell/test_console` with `{"values": "v1,v2,...,v22"}` |
| Train on CSV data | Browser UI → **Train** tab, upload/paste CSV |
| Train via API | `POST /smell/learn_from_csv` |
| Simulate full pipeline | `python scripts/run_client.py --offline` |
| Live stream to UI | Client: `--live` · UI: **Live** tab |
| Replay a recorded session | `python scripts/replay.py session.npz --server http://…` |

For continuous labelled recordings, use the UI's **Stable plateaus** profile
with **Balanced RF + baseline** and **Log ratio to clean air**. It removes
transition/recovery leakage and supports R1–R17-only datasets. See
[`docs/MODEL_IMPROVEMENTS.md`](docs/MODEL_IMPROVEMENTS.md) for the live-baseline
workflow and independent-session benchmark.

## Live streaming, drift, and replay

* **Client — live dashboard:** run the client with `--live` (or pick menu
  option 8) to open a matplotlib window with the rolling sensor trace and
  push every sample to the server for everyone else to watch in the UI.
  ```bash
  python scripts/run_client.py --offline --live --live-classify \
      --live-label coffee --record-session runs/coffee_2026-04-23.npz
  ```
* **UI — Live tab:** polls `/sensor/live/recent` once per second and draws
  R-sensor / env / confidence charts plus a drift panel comparing the live
  buffer against the training distribution.
* **`scripts/replay.py`:** reads a `.npz` recorded by `--record-session`
  and re-runs it through `/smell/classify` so you can verify that a saved
  session still predicts the expected label after a retrain.
* **Out-of-distribution indicator:** every `/smell/classify` response now
  includes an `ood` block (score, nearest centroid, status). The UI
  renders it as a green / yellow / red badge above the result.
* **Per-class metrics:** `/smell/model_info` now returns
  `per_class_metrics` and `confusion_matrix` from the last training split,
  shown as a table in the UI's Model Info tab.
* **New plots** from `/smell/visualize_data`: per-class feature importance
  (one-vs-rest F-ratio heatmap) and calibration / reliability diagram with
  Expected Calibration Error (ECE).

## Package layout

```
enose/
├── config.py              — all constants and paths
├── classifier/            — BalancedRFClassifier, XGBOdorClassifier
├── server/                — FastAPI app, routes, state
├── client/                — sensors, webcam, API client, pipeline, CLI
├── visualization/         — matplotlib plot functions
├── vision/                — Florence-2 loader + GPU helper
└── utils/                 — CSV I/O helpers

scripts/
├── run_server.py          — uvicorn launcher
├── run_client.py          — client entry point
├── verify_server_environment.py — locked versions + real VLM startup/inference
└── smoke_test.py          — AST + export validation

docs/
├── CHEMIST_GUIDE.md       — step-by-step for non-Python users
└── SERVER_CLIENT_GUIDE.md — developer integration reference
```

## Sensor features (22 total)

| Group | Sensors | Preprocessing |
|---|---|---|
| Resistance (R1–R17) | 17 MOS sensors | ADC → Ω on client; StandardScaler on server |
| Environmental | T, H, CO2, H2S, CH2O | Raw engineering units, sanity-clamped |

## API overview

| Endpoint | Method | Purpose |
|---|---|---|
| `/ui` | GET | Browser web interface |
| `/` | GET | Server status JSON |
| `/health` | GET | Liveness check |
| `/smell/classify` | POST | Classify named-sensor dict (+ OOD block) |
| `/smell/baseline/live` | POST | Capture a stable clean-air baseline from the live buffer |
| `/smell/classify_window` | POST | Fast aggregated 5–120 s prediction with confidence abstention |
| `/smell/classify_stable` | POST | Classify the median of a stable live window |
| `/smell/recovery` | GET | Check return to the captured clean-air baseline |
| `/smell/test_console` | POST | Classify 17 or 22 comma-separated values |
| `/smell/debug_input` | POST | Diagnostics (z-scores, OOD, centroids) |
| `/smell/online_learning` | POST | Incremental training |
| `/smell/learn_from_csv` | POST | Batch CSV training |
| `/smell/model_info` | GET | Classes, accuracy, per-class P/R/F1, confusion matrix |
| `/smell/visualize_data` | GET | Generate plots (incl. per-class importance, calibration) |
| `/smell/drift` | GET | Live-vs-training per-sensor drift report |
| `/sensor/live/push` | POST | Client → server: one live sample |
| `/sensor/live/recent` | GET | Incremental live-buffer poll (`?since=<id>`) |
| `/sensor/live/stats` | GET | Aggregate stats over the live buffer |
| `/sensor/live/clear` | DELETE | Reset the live ring buffer |
| `/docs` | GET | Interactive Swagger UI |

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `ENOSE_NO_VLM` | `0` | Set to `1` to skip Florence-2 (smell-only mode) |
| `ENOSE_CLASSIFIER` | `balanced_rf` | Classifier backend: `balanced_rf` or `xgb` |
| `SERVER_HOST` | `0.0.0.0` | Bind address |
| `SERVER_PORT` | `8080` | Listen port |
| `ENOSE_RLOW` | `1.0` | Client-side load resistance for ADC conversion; `--rlow` overrides it |

## License

This project is released under the [MIT License](LICENSE) — use, modify,
distribute, and sublicense freely. Attribution (the copyright notice in
[`LICENSE`](LICENSE)) is the only requirement.

Third-party components keep their own licenses:

- **Florence-2-Large** weights (downloaded separately, see
  [`docs/SETUP.md`](docs/SETUP.md#3-download-florence-2-large)) — MIT,
  © Microsoft.
- All Python dependencies listed in `pyproject.toml` are permissively
  licensed (MIT / BSD / Apache 2.0) and compatible with MIT
  redistribution.

## Further reading

- [**Setup guide**](docs/SETUP.md) — clone → venv → download Florence-2 → first classification
- [**PC migration guide**](docs/MIGRATION.md) — move code, weights, trained models, data, and configuration safely
- [**Stable/drift-robust training**](docs/MODEL_IMPROVEMENTS.md) — plateau extraction, recommended settings, live workflow, and benchmark
- [Chemist guide](docs/CHEMIST_GUIDE.md) — no Python needed
- [Server + Client integration guide](docs/SERVER_CLIENT_GUIDE.md) — developer reference
- [Classifier internals](enose/classifier/README.md)
- [Server routes](enose/server/README.md)
- [Client modules](enose/client/README.md)
