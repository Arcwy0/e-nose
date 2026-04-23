# E-Nose — Vision + Smell Multimodal Classification

A robotic olfaction system combining a **Florence-2** vision model with a **22-feature electronic nose** (17 resistance sensors + 5 environmental). Runs on a Unitree Go-1 robot dog.

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
│  ├── ENoseSensor   (dual UART: e-nose + env)        │
│  ├── WebcamHandler (OpenCV capture)                 │
│  └── TrainingPipeline (interactive CLI)             │
└─────────────────────────────────────────────────────┘
```

## Quick start

> **First time on this machine?** Read [`docs/SETUP.md`](docs/SETUP.md) —
> it walks through the venv, `pip install`, Florence-2 download, and
> `.env` setup. The steps below assume that's done.

### 1 — Server (classifier only, no GPU / VLM required)

```bash
# Start without Florence-2 — all smell endpoints work normally
ENOSE_NO_VLM=1 python scripts/run_server.py
# Or with auto-reload for development:
ENOSE_NO_VLM=1 python scripts/run_server.py --reload
```

Open **http://localhost:8080/ui** in any browser for the point-and-click interface.

### 2 — Server (full, with VLM)

```bash
python scripts/run_server.py
```

Requires Florence-2 weights at `model/Florence-2-Large/`.

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
| Classify a single sample | Browser UI → **Classify** tab, paste 22 values |
| Classify via API | `POST /smell/test_console` with `{"values": "v1,v2,...,v22"}` |
| Train on CSV data | Browser UI → **Train** tab, upload/paste CSV |
| Train via API | `POST /smell/learn_from_csv` |
| Simulate full pipeline | `python scripts/run_client.py --offline` |
| Live stream to UI | Client: `--live` · UI: **Live** tab |
| Replay a recorded session | `python scripts/replay.py session.npz --server http://…` |

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
| `/smell/test_console` | POST | Classify 22 comma-separated values |
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
- [Chemist guide](docs/CHEMIST_GUIDE.md) — no Python needed
- [Server + Client integration guide](docs/SERVER_CLIENT_GUIDE.md) — developer reference
- [Classifier internals](enose/classifier/README.md)
- [Server routes](enose/server/README.md)
- [Client modules](enose/client/README.md)
