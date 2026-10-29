# E-Nose — Repository Overview

A map of what lives in this repo, what each part does, how to run it, and the
known limitations. Current operational reference: **3.3.0**. Written for the
robotic-olfaction stack (electronic nose + Unitree Go-1 + Florence-2 VLM).

For deeper docs see: [`SETUP.md`](SETUP.md), [`SERVER_CLIENT_GUIDE.md`](SERVER_CLIENT_GUIDE.md),
[`CLIENT_QUICKSTART.md`](CLIENT_QUICKSTART.md), [`CHEMIST_GUIDE.md`](CHEMIST_GUIDE.md),
[`TESTING_GUIDE.md`](TESTING_GUIDE.md), [`IDEA1_PLAN.md`](IDEA1_PLAN.md),
[`IDEA1_PROGRESS.md`](IDEA1_PROGRESS.md), [`NEXT_STEPS.md`](NEXT_STEPS.md),
and the Windows client guide [`CLIENT_WINDOWS_INSTALL.md`](CLIENT_WINDOWS_INSTALL.md).

---

## 1. What the system is

A robot dog (Unitree Go-1 EDU) carries an **electronic nose** (17 resistance MOS
sensors plus 5 optional environmental transport channels) and RGB/depth cameras. A
**FastAPI server** on a lab GPU box runs:

- a **Florence-2 VLM** for open-vocabulary object detection / phrase grounding, and
- a **smell classifier** (`BalancedRandomForest`, calibrated, with OOD detection).

The dog (client) talks to the server over REST/JSON on Wi-Fi. Two research ideas
drive the work:

- **Idea 1 — on-the-fly scent learning** (current focus): operator drives the dog to
  an object, says "learn this scent"; the dog approaches, settles, records ~30 s of
  pose-tagged sensor data, and commits a labelled batch that retrains the classifier.
- **Idea 2 — olfactory + visual navigation** (later): fuse smell gradient/plume cues
  with vision/LIDAR to locate an odour source. Not started; design in `IDEA1_PLAN.md §4`.

### Data flow

```
Go-1 Jetson (client)                         Lab GPU box (server)
──────────────────────                       ─────────────────────────────────
PCB USB CDC ─┐                     REST/JSON
legacy UART ─┤
env UART   ──┼─► sensors.py ──►  ServerAPI ───────────►  FastAPI routes
camera     ──┘   live.py (stream)                        ├─ vision  (Florence-2)
robot/policy.py (Idea-1 FSM) ──────────────────────────► ├─ smell   (classify)
                                                         ├─ training(learn/online)
                                                         ├─ live    (ring buffer)
                                                         └─ analytics(drift/settling)
```

---

## 2. Package map (`enose/`)

| Package | Main responsibility | Key modules |
|---|---|---|
| `enose/config.py` | Central constants: sensor names, feature order (`ALL_SENSORS`), model/data paths, defaults | — |
| `enose/server/` | FastAPI app, request/response schemas, state, model persistence glue, routes | `app.py`, `state.py`, `schemas.py`, `model_loader.py`, `routes/` |
| `enose/client/` | Interactive client: sensor I/O, camera, REST wrapper, live streaming, pipeline | `api.py`, `live.py`, `sensors.py`, `webcam.py`, `pipeline.py`, `session.py`, `main.py` |
| `enose/classifier/` | Smell classifiers + preprocessing + persistence + OOD | `balanced_rf.py`, `xgb_tabular.py`, `preprocessing.py`, `persistence.py` |
| `enose/vision/` | Florence-2 loading + inference + GPU detection | `florence.py`, `gpu.py` |
| `enose/visualization/` | Matplotlib plots (confusion matrix, feature importance, data quality) | — |
| `enose/robot/` | **Idea-1** state machine + adapters (sim now, ROS2 later) | `policy.py`, `mission.py`, `interfaces.py`, `visual_servo.py`, `manual_pump.py`, `vision_http.py`, `sim/`, `ros2/` |

### Server modules of note
- **`state.py`** — global singletons `vlm_model`, `vlm_processor`, `smell_classifier`, with
  `set_classifier()` and `require_fitted_classifier()` helpers.
- **`model_loader.py`** — `reload_smell_classifier()` (swap in the freshly-saved model),
  `save_training_data()` (append labelled batch to CSV), `load_or_create_classifier()` (startup).
- **`live_buffer.py`** — thread-safe ring buffer of pose-tagged live samples.

### Classifier internals (`balanced_rf.py`)
- `train()` — split raw independent groups first, learn preprocessing/calibration
  only within training folds, freeze holdout diagnostics, then refit deployment.
  No usable independent group split means null accuracy, not a row-wise fallback.
- `online_update()` — append a batch to retained training data and refit from scratch (supports new classes).
- `predict()` / `predict_proba()` / `diagnose_sample()` (OOD z-scores + centroid distances).
- `get_model_info()` — classes, per-class P/R/F1, confusion matrix, training history,
  **class distribution + total training samples**, calibration/env config.
- `class_example_vectors()` — per-class mean 22-feature vector (powers the UI's dynamic "Try" buttons).
- Persistence (`persistence.py`): saves a timestamped joblib **and** keeps
  `smell_classifier_sgd_latest.joblib` in sync (this is what the server reloads).

---

## 3. HTTP endpoints

| Route | Method | Purpose |
|---|---|---|
| `/`, `/health` | GET | Status / liveness; VLM & classifier load state, backend name |
| `/smell/model_info` | GET | Full model metadata (classes, confusion matrix, metrics, distribution). **No-store.** |
| `/smell/class_examples` | GET | Per-class representative vectors for the UI's dynamic "Try" buttons |
| `/smell/classify` | POST | Single 22-feature classification + OOD block |
| `/smell/test_console` | POST | Comma-separated values → prediction (console/UI ergonomic) |
| `/smell/debug_input` | POST | OOD diagnostics for one sample |
| `/smell/learn_from_csv` | POST | Train/retrain from CSV (UI Train tab) |
| `/smell/online_learning` | POST | Incremental commit of a labelled batch (+ optional provenance) |
| `/training_pipeline` | POST | Vision + smell (detect object then train) |
| `/smell/provenance` | GET | List autonomous-training commits |
| `/sensor/live/push` `/recent` `/clear` `/stats` | POST/GET/DELETE/GET | Live ring buffer |
| `/smell/drift` | GET | Live buffer vs training distribution |
| `/smell/settling` | GET | Per-sensor stability (robot SETTLE state) |
| `/smell/visualize_data` `/analyze_data` `/environmental_analysis` | GET | Plots / data quality |
| `/predict/object` | POST | Florence-2 single-label open-vocabulary detection |
| `/predict/scene` | POST | Florence-2 multi-label phrase grounding (robot SEARCH) |
| `/ui` | GET | Single-page browser UI (classify / train / live / model info) |

---

## 4. Scripts (`scripts/`)

| Script | What it does |
|---|---|
| `run_server.py` | Start the FastAPI server (respects `ENOSE_NO_VLM=1`, `ENOSE_CLASSIFIER`) |
| `run_client.py` | Interactive client (offline sim or real serial hardware; `--live`, `--live-classify`) |
| `run_robot_trainer.py` | Idea-1 mission entry point; `--backend sim` (works now) or `ros2` (stubbed) |
| `sim_run.py` | Standalone end-to-end sim mission using `ScriptedVision` + sim adapters |
| `sim_targets.json` | Example target list (label, aliases, waypoints, world pose, detection score) |
| `audit_auto_labels.py` | Inspect autonomous-training commits (`list` / `inspect <id>` / `low-score`) |
| `smoke_test.py` | AST-parse all modules; catches syntax/import-surface breakage without heavy deps |
| `verify_server_environment.py` | Check the shared version lock and, with `--vlm`, load real weights and generate a caption |

---

## 5. How to run & test (Docker-first)

```bash
# Server, smell-only (no VLM) — seconds to boot
docker run --rm -d --name enose-srv --network enose-net -p 18080:8080 \
  -v "$(pwd)":/app -e PYTHONPATH=/app -e ENOSE_NO_VLM=1 \
  enose-server python scripts/run_server.py
curl http://localhost:18080/health

# Browser UI
#   http://localhost:18080/ui   (Classify / Train / Live / Model Info)

# End-to-end sim mission (no robot, no Florence-2)
docker run --rm --network enose-net -v "$(pwd)":/app -e PYTHONPATH=/app \
  enose-server python scripts/sim_run.py \
  --server http://enose-srv:8080 --targets /app/scripts/sim_targets.json --speed 8

# Client (offline) — from a client install (see CLIENT_QUICKSTART.md)
python scripts/run_client.py --offline --server http://localhost:18080
```

Install the server lock (see `SETUP.md`): `pip install -r requirements-server.txt`
then `pip install --no-deps -e .`. Docker and uv use the same complete pins.
Client-only = `pip install -e ".[client]"` in a separate environment.

Prebuilt client wheel for Windows colleagues: `enose-3.3.0-py3-none-any.whl`
from [GitHub Releases](https://github.com/Arcwy0/e-nose/releases/latest).
→ see [`CLIENT_WINDOWS_INSTALL.md`](CLIENT_WINDOWS_INSTALL.md).

---

## 6. Component readiness (as of this review)

| Component | Status | Notes |
|---|---|---|
| Smell classifier + server API | ✅ Ready | Trains, classifies, retrains, persists; OOD works |
| Web UI (classify/train/live/model-info) | ✅ Ready | Model Info auto-refreshes; dynamic per-class "Try" buttons |
| **VLM (Florence-2)** | ✅ Ready to test | Loader + `/predict/object` + `/predict/scene` wired; needs weights + GPU (or slow CPU). `ENOSE_NO_VLM=1` disables cleanly |
| **Robot simulation** | ✅ Ready to test | `sim_run.py` runs the full policy FSM vs a no-VLM server; no hardware needed |
| Client packaging | Software-tested | Separate extras and universal package wheel; native Windows GUI/USB requires field validation |
| Calibration/holdout isolation | Implemented | Group-disjoint folds; no independent groups means calibration disabled / metrics not estimable |
| New PCB acquisition/heating | Software-tested only | Default USB CDC, preflight, opt-in monitored PI; no host-loss watchdog guarantee |
| CSV/model deployment | Implemented | Preview, annotations, staged candidate, paired evaluation, explicit promotion and recoverable rollback |
| Real-robot ROS2 path | ⛔ Not ready | `enose/robot/ros2/*` raise `NotImplementedError` (Phase-0 work) |
| Visual-servo integration | ⛔ Not ready | `policy._approach` sim-teleports; real depth-servo loop not wired |

---

## 7. Known problems / bugs & where they live

| Issue | Severity | Status | Where |
|---|---|---|---|
| Calibration without independent groups | Evaluation | Disabled rather than replacing group CV with row CV | `enose/classifier/balanced_rf.py` `_fit_estimator()` |
| Model Info UI stale after retrain | MED | **Fixed** (no-store fetches + `Cache-Control` + auto-refresh after train) | `enose/server/routes/ui.py`, `health.py` |
| Hardcoded "Try coffee/air" buttons | LOW | **Fixed** (dynamic per-class via `/smell/class_examples`) | `ui.py`, `routes/smell.py` |
| Native Windows dependency/GUI/USB behavior | Field | Universal wheel is available; native execution still requires validation | `docs/CLIENT_WINDOWS_INSTALL.md` |
| `/predict/object` previously missing (404) | — | Fixed earlier | `enose/server/routes/vision.py` |
| Grounding thresholds not calibrated (`τ_grounding`, `τ_commit`) | HIGH (field) | Open — defaults are placeholders | `enose/robot/policy.PolicyConfig` |
| ROS2 adapters unimplemented | HIGH (real robot) | Open — Phase-0 | `enose/robot/ros2/{unitree_adapter,localization,pump}.py` |
| Visual-servo loop not integrated | MED | Open | `enose/robot/policy.py::_approach`, `visual_servo.py` |
| Pump is mechanical-only (no relay) | MED | By design — `ManualPump` logs prompts | `enose/robot/manual_pump.py` |
| Wi-Fi reconnect hardening unproven | MED | Open — needs real-dog testing | `enose/client/live.py` |

Out of scope for the current round (Idea 2): `/smell/intensity`, `/smell/spatial_map`,
plume tracers, fusion policy.

## 8. Acquisition and deployment controls

Read [PCB acquisition](PCB_ACQUISITION.md) for new-board units, firmware
conversion, heater limits and legacy compatibility. New PCB is default;
`--sensor-backend legacy` explicitly retains original ADC/RLOW behavior.
Environmental UART is optional. Replay never actuates hardware.

The [client reference](../enose/client/README.md#complete-cli-reference) lists
all CLI options and menu 0–13, including configurable vision timeouts, preflight,
external/client heating and headless recording. Browser heater state is read-only.
Graceful cooldown is monitored, but USB/host loss cannot guarantee off.

The UI stages candidates by default; active predictions change only after
promotion. Historical merge is independently default-on. Follow the
[deployment workflow](SERVER_CLIENT_GUIDE.md#training-and-candidate-deployment)
and migrate active model/root history/candidate snapshots together.
Keep one server worker and private authenticated/network access.
