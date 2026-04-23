# Server + Client Integration Guide

## Overview

The system has two independent parts that communicate over HTTP:

- **Server** (`enose/server/`) — FastAPI app running Florence-2 VLM + smell classifier. Runs on the GPU machine or in Docker.
- **Client** (`enose/client/`) — Python CLI that reads from physical sensors and drives the server. Runs on the robot laptop.

---

## Running the server

### Classifier-only mode (no GPU, no VLM)

Use `ENOSE_NO_VLM=1` to skip loading Florence-2. All smell endpoints work normally. Vision endpoints return HTTP 503.

```bash
ENOSE_NO_VLM=1 python scripts/run_server.py
```

### Full mode (VLM + classifier)

```bash
python scripts/run_server.py
# Optional flags:
#   --host 0.0.0.0      (default)
#   --port 8080         (default)
#   --reload            (dev auto-reload)
#   --workers 4         (production)
```

Florence-2 weights must be in `model/Florence-2-Large/` (set `VLM_MODEL_PATH` in `enose/config.py` to change).

### Via uvicorn directly

```bash
ENOSE_NO_VLM=1 uvicorn enose.server.app:app --host 0.0.0.0 --port 8080
```

---

## Running the client

### Offline / simulated mode — no hardware required

```bash
python scripts/run_client.py --offline --server http://localhost:8080
```

All sensor readings are generated synthetically using smell profiles defined in `enose/client/sensors.py`. The full menu works: train, classify, CSV import, manual testing.

### Hardware mode

```bash
# Linux
python scripts/run_client.py \
    --port_enose /dev/ttyUSB0 \
    --port_UART  /dev/ttyUSB1 \
    --server     http://SERVER_IP:8080 \
    --time       60 \
    --samples    100

# Windows
python scripts/run_client.py \
    --port_enose COM3 \
    --port_UART  COM4 \
    --server     http://SERVER_IP:8080
```

The client discovers serial ports interactively if `--port_enose` / `--port_UART` are omitted. Typical Linux device names are `/dev/ttyUSB0`, `/dev/ttyUSB1`, `/dev/ttyACM0`; on macOS `/dev/tty.usbserial-XXXX`; on Windows `COM3`, `COM4`, etc. Run `ls /dev/tty*` (Linux/macOS) or check Device Manager → Ports (Windows) to confirm.

If you see `permission denied` opening the serial device on Linux, add your user to the `dialout` group once and re-log:

```bash
sudo usermod -aG dialout $USER
# log out and back in
```

### Live streaming from the client

The client can publish samples to the server in real time so anyone with
the UI open sees the live charts update. This replaces the older "record,
upload, classify" flow for ad-hoc experiments.

```bash
# Stream synthetic samples, classify each one, tag them as "coffee",
# and save a .npz for later replay.
python scripts/run_client.py --offline \
    --live --live-classify --live-label coffee \
    --record-session runs/coffee_2026-04-23.npz \
    --server http://localhost:8080
```

New flags:

| Flag | Purpose |
|---|---|
| `--live` | Skip the menu; go straight to the live dashboard. |
| `--live-rate` (default `5.0`) | Publisher rate in Hz. The server buffer is sized so 1000 / rate ≈ minutes of history. |
| `--live-window` (default `60.0`) | Seconds visible in the client-side matplotlib window. |
| `--live-baseline` | Subtract the first few seconds as an "air" baseline (sensor-drift compensation on the plot only — not sent to the server). |
| `--live-classify` | Classify every sample through `/smell/classify` and show confidence on the bottom axis. |
| `--live-label coffee` | Tag each published sample with this label so the server can separate runs in the ring buffer. |
| `--record-session path.npz` | Tee the sample stream into a `.npz` file while publishing. Works with `--live` or menu option 8. |

### Replaying a recorded session

`scripts/replay.py` reads a `.npz` from `--record-session` and re-plays
every sample through `/smell/classify`, reporting:

- Label distribution and majority fraction
- Confidence: mean, median, p10, p90
- Accuracy vs. `session_label` (if the session was recorded with a tag)
- Agreement with the predictions originally recorded at capture time

```bash
python scripts/replay.py runs/coffee_2026-04-23.npz \
    --server http://localhost:8080 \
    --rate 10            # classifications per second, 0 = as fast as possible
    --limit 500          # cap total samples
    --verbose            # print each sample's result
```

Use case: after retraining, re-run a known-good session to confirm the
new model still predicts the expected label on captured data — no
re-measurement needed.

---

## Remote access (LAN / private tunnel)

If colleagues need to drive the server from off-host, bind the server to
`0.0.0.0` (the default) and reach it at `http://<LAN-IP>:8080`. For
off-site access, put a VPN (WireGuard, Tailscale, etc.) or a private TCP
tunnel in front of it — the repo is deliberately tunnel-agnostic, since
every deployment picks something different.

The client side is identical to a local run: only `--server` changes.
The browser UI uses relative paths for every `fetch()` call, so it works
identically behind any reverse proxy or TCP forwarder. The status bar
shows the origin URL the page was loaded from, so users can always
confirm which instance they are talking to.

### Docker client with UART pass-through

Colleagues who want a reproducible Python environment can run the client
from the provided Docker image. The only hardware-specific part is
passing the USB serial device into the container.

```bash
git clone <repo-url> e-nose && cd e-nose
docker build -t enose-client -f docker/Dockerfile-client .

# Offline (verify connectivity first)
docker run --rm -it \
  -v "$(pwd)":/app -e PYTHONPATH=/app \
  enose-client \
  python scripts/run_client.py --offline --server http://<SERVER_HOST>:8080

# Hardware — map both UART devices into the container
docker run --rm -it \
  --device=/dev/ttyUSB0 --device=/dev/ttyUSB1 \
  -v "$(pwd)":/app -e PYTHONPATH=/app \
  enose-client \
  python scripts/run_client.py \
      --port_enose /dev/ttyUSB0 --port_UART /dev/ttyUSB1 \
      --server http://<SERVER_HOST>:8080
```

Notes:
- `--device` grants direct access to host serial nodes — no `--privileged` needed.
- `Dockerfile-client` is headless — menu options that need the webcam
  (1 — full training cycle, 2 — vision only) won't work inside it unless
  you also pass `--device=/dev/video0` and wire up display forwarding.
  Usually not worth it; use menu **3** (smell identification) or **6**
  (manual 22-feature input) when driving a remote server from Docker.
- Windows/macOS cannot pass USB serial into a Linux container directly.
  On those hosts run the client **outside** Docker.

### Quick troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Client can't reach `/health` | Server not running, wrong URL, or firewall blocking the port. Confirm with `curl -v http://<SERVER_HOST>:8080/health` from the client host. |
| Client reaches `/health` but training is slow | Expected for large CSV uploads over a slow link. For very large datasets, keep training local to the server host. |
| `permission denied: /dev/ttyUSB0` inside Docker client | Either `--device=/dev/ttyUSB0` is missing from the `docker run`, or host-side permissions block the container user. Easiest fix: `sudo chmod 666 /dev/ttyUSB0` for one session. |

---

## Testing without hardware — all options

### Option 1: Browser UI

```
http://localhost:8080/ui
```

- Paste 22 values → Classify
- Upload/paste CSV → Train
- No Python, no terminal

### Option 2: Swagger / ReDoc

```
http://localhost:8080/docs    ← interactive
http://localhost:8080/redoc   ← readable
```

### Option 3: curl (single sample)

```bash
# Classify by name
curl -X POST http://localhost:8080/smell/classify \
  -H "Content-Type: application/json" \
  -d '{"R1":15.2,"R2":8.3,"R3":12.1,"R4":3.4,"R5":18.9,"R6":11.2,"R7":9.8,"R8":6.7,"R9":14.3,"R10":10.5,"R11":13.2,"R12":7.8,"R13":5.6,"R14":12.4,"R15":8.9,"R16":6.3,"R17":11.7,"T":21.0,"H":49.0,"CO2":400,"H2S":0.0,"CH2O":5.0}'

# Classify by position (comma-separated, R1…R17, T, H, CO2, H2S, CH2O)
curl -X POST http://localhost:8080/smell/test_console \
  -H "Content-Type: application/json" \
  -d '{"values": "15.2,8.3,12.1,3.4,18.9,11.2,9.8,6.7,14.3,10.5,13.2,7.8,5.6,12.4,8.9,6.3,11.7,21.0,49.0,400,0.0,5.0"}'
```

### Option 4: Batch CSV training

```bash
curl -X POST http://localhost:8080/smell/learn_from_csv \
  -H "Content-Type: application/json" \
  -d "{\"csv_data\": \"$(cat your_data.csv)\", \"target_column\": \"Gas name\", \"use_augmentation\": true, \"n_augmentations\": 5}"
```

### Option 5: Offline client menu

```bash
python scripts/run_client.py --offline
# Menu option 6 → Manual 22-feature input testing
# Menu option 5 → Learn from CSV file
```

---

## API reference

### Smell classification

```
POST /smell/classify
Body: {"R1": float, "R2": float, ..., "T": float, "H": float, "CO2": float, "H2S": float, "CH2O": float}
Returns: {
  predicted_smell, probabilities, confidence, features_used,
  ood: {
    available: bool,
    score: float,                  # mean |z| across features
    min_centroid_L2: float,        # L2 to the nearest class centroid
    nearest_centroid_L2: float,    # same value, kept for legacy consumers
    status: "ok" | "warn" | "out",
    thresholds: {ood_warn, ood_out, centroid_warn, centroid_out}
  }
}
```

The `ood` block is a best-effort addition — if diagnostics fail for any
reason (missing centroids on an un-fit model, numerical issues, etc.) the
response still carries the prediction and `ood.available = false`. The
status enum uses the same thresholds as the classifier's internal
`predict_with_ood`, so the browser traffic light and the confidence-damping
path agree on what "OOD" means.

Status rules:

| Condition | Status |
|---|---|
| `score > 3.0` OR `min_centroid_L2 > 5.0` | `out` (red) |
| `score > 2.0` OR `min_centroid_L2 > 3.5` | `warn` (yellow) |
| otherwise | `ok` (green) |

```
POST /smell/test_console
Body: {"values": "v1,v2,...,v22"}   ← comma-separated, R1–R17 then T,H,CO2,H2S,CH2O
Returns: {predicted_smell, confidence, all_probabilities, sorted_probabilities, ood: {...}}
```

```
POST /smell/debug_input
Body: {"values": [22 floats]}  OR named sensor dict
Returns: {predicted, top3, ood_score, nearest_centroid_L2, z_scores}
```

### Training

```
POST /smell/online_learning
Body: {"sensor_data": [{...}, ...], "labels": ["coffee", ...]}
```

```
POST /smell/learn_from_csv
Body: {
  "csv_data": "<full CSV string>",
  "target_column": "Gas name",
  "use_augmentation": true,
  "n_augmentations": 5,
  "noise_std": 0.0015,
  "lowercase_labels": true
}
```

### Analytics & info

```
GET /smell/model_info
  Returns: {
    classes, accuracy, n_training_runs,
    per_class_metrics: {class: {precision, recall, "f1-score", support}, ...},
    confusion_matrix: [[int, ...], ...],
    confusion_labels: [str, ...],
    last_test_size: int,
    feature_importances, feature_names
  }

GET /smell/visualize_data    → writes PNGs to data/plots/
  Returns: {
    confusion_matrix, class_counts, feature_importances,
    per_class_feature_importance, calibration,
    calibration_ece: float,
    environmental_histograms, ...
  }

GET /smell/drift             → per-sensor live-vs-training drift report
  Returns: {
    available: bool,
    overall_status: "ok" | "warn" | "out" | "na",
    sensors: [
      {
        name, train_mean, train_std, live_mean, live_std,
        z_shift: (live_mean - train_mean) / train_std,
        std_ratio: live_std / train_std,
        status: "ok" | "warn" | "out" | "na"
      }, ...
    ],
    top_shifts: [name, ...],
    thresholds: {z_warn: 1.5, z_out: 3.0,
                 std_ratio_warn: [0.5, 2.0], std_ratio_out: [0.25, 4.0]}
  }

  status="na" means the classifier has no `last_training_data` yet
  (never trained) or the live buffer is empty — the UI shows a dash.

GET /smell/analyze_data
GET /smell/environmental_analysis
GET /health
GET /
```

### Live streaming

The server exposes a thread-safe ring buffer (default 1000 slots, see
`enose/server/live_buffer.py`) that the client fills via HTTP POSTs and
the UI drains via polling. There is no WebSocket — polling keeps things
compatible with plain TCP forwarders / proxies that may not handle WS.

```
POST /sensor/live/push
Body: {
  "entry": {
    "ts": float,                       # epoch seconds
    "values": {"R1": float, ..., "CH2O": float},
    "label": str | null,               # optional tag (e.g. "coffee")
    "prediction": str | null,          # optional, from --live-classify
    "confidence": float | null         # optional, 0..1
  }
}
Returns: {id: int}     # monotonic buffer ID assigned to this entry

GET /sensor/live/recent?since=<id>&limit=<n>
Returns: {
  entries: [{id, ts, values, label, prediction, confidence}, ...],
  next_since: int,     # poll with this value next time
  size: int            # current buffer size
}

GET /sensor/live/stats
Returns: {
  size, first_ts, last_ts, hz,
  per_sensor_mean, per_sensor_std,
  label_counts, prediction_counts
}

DELETE /sensor/live/clear
Returns: {cleared: int}   # number of entries removed
```

The UI's **Live** tab polls `/sensor/live/recent` once per second with the
last-seen `next_since`, then draws three inline-SVG charts (R sensors,
environmental, confidence over time coloured by prediction) plus a drift
panel that re-hits `/smell/drift`.

---

## Training the model

### From CSV (recommended for batch data)

Your CSV must have columns matching the sensor names. Unknown columns are ignored; missing sensor columns are filled with defaults.

```csv
R1,R2,...,R17,T,H,CO2,H2S,CH2O,Gas name
15.2,8.3,...,21.0,49.0,400,0,5,coffee
0.1,0.05,...,21.0,49.0,400,0,5,air
```

Minimum recommended: **20 samples per class**. More is better.

### From the client (hardware or offline)

1. Connect sensors and run the full training pipeline from the client menu (option 1).
2. The pipeline captures an image, detects the object, records sensor data, and calls `POST /smell/online_learning`.

### Auto retrain logic

- First upload → initial training.
- Subsequent uploads with **existing classes** → incremental online update.
- Uploads with **new classes** → full retrain to incorporate the new class correctly.

---

## Adding a new smell class

You do not need to change any code. Just:

1. Collect 20+ labelled sensor samples for the new smell.
2. Upload the CSV via the web UI or `/smell/learn_from_csv`.
3. The server detects the new class and retrains automatically.

---

## Configuration

All constants live in `enose/config.py`:

| Constant | Default | Purpose |
|---|---|---|
| `VLM_MODEL_PATH` | `model/Florence-2-Large` | Florence-2 weights directory |
| `SMELL_MODEL_PATH` | `trained_models/smell_classifier_sgd_latest.joblib` | Saved classifier |
| `TRAINING_DATA_PATH` | `database_robodog.csv` | Training data log |
| `SERVER_HOST` | `0.0.0.0` | Bind address |
| `SERVER_PORT` | `8080` | Listen port |
| `DEFAULT_RECORDING_TIME` | `60` | Recording duration (seconds) |
| `DEFAULT_TARGET_SAMPLES` | `100` | Target samples per recording |

---

## Programmatic use (Python)

```python
from enose.client.api import ServerAPI

api = ServerAPI("http://localhost:8080")

# Single-sample classification
result, err = api.classify_smell({
    "R1": 15.2, "R2": 8.3, ..., "T": 21.0, "H": 49.0, "CO2": 400, "H2S": 0, "CH2O": 5
})
print(result["predicted_smell"], result["confidence"])

# CSV training
result, err = api.learn_from_csv("my_data.csv", target_column="Gas name")
```
