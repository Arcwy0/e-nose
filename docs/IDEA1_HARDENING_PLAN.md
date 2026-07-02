# Idea 1 hardening: repo doc, UI fixes, small-batch retrain fix, client wheel

## Context

The e-nose + Go-1 + Florence-2 stack has Idea 1 (on-the-fly scent learning) essentially
implemented and verified in simulation. Before moving to Idea 2, several concrete problems
must be fixed and the repo documented. This plan covers, in order:

1. A repository overview doc (contents, main functions, known bugs).
2. Three UI problems on the web UI (`enose/server/routes/ui.py`):
   - **Model Info page never updates** after retraining (confusion matrix, known classes,
     distribution).
   - **"Try coffee" / "Try air" buttons are hardcoded** — must be dynamic per known class.
   - Other UI staleness issues found during review.
3. The **small-batch retrain crash** (Idea-1 blocker): `online_learning` on a tiny new-class
   batch intermittently 500s because `CalibratedClassifierCV(cv=3)` fails when a class has
   fewer samples than the fold count.
4. A **prebuilt client wheel** for Windows colleagues, by splitting dependencies so a client
   install is lightweight.
5. A **readiness review** of the VLM and robot-dog simulation paths.

### Root causes established during exploration (important)

- **Model Info is a FRONTEND bug, not backend.** The classifier persists metrics
  (`per_class_metrics_`, `confusion_matrix_`, `confusion_labels_`, `last_test_size_` in
  `persistence.build_save_payload`), `persistence.save()` keeps `smell_classifier_sgd_latest.joblib`
  in sync, and `reload_smell_classifier()` reloads it. So `/smell/model_info` returns fresh data.
  The UI is stale because: (a) `fetch('/smell/model_info')` GETs are **browser-cached**, and
  (b) `trainCsv()` only calls `refreshStatus()` (updates the top badge), never `loadInfo()`.
- **"Try" buttons** are a static JS object `EX` (ui.py:267–270) with only `coffee`/`air`.
- **Client is cleanly decoupled**: `enose/client/*` imports only `enose.config` +
  `requests/pandas/numpy/serial/matplotlib/cv2/PIL`. No fastapi/scikit-learn/torch. The
  `numpy<2` / `scikit-learn<1.6` pins in the base deps are server-only and are the likely cause
  of the colleagues' Windows source-build failure.
- **CV crash**: `enose/classifier/balanced_rf.py:95` builds the calibrator with a fixed `cv=3`
  via `_make_calibrator`. Small new-class batches (< 3 samples in a class) break the internal
  stratified CV.

---

## Part 1 — Repository overview doc

Create `docs/REPO_OVERVIEW.md` (per the docs/ convention). Contents:
- Architecture & data flow (server on lab GPU ↔ Go-1 client over REST; classifier + Florence-2 + live buffer).
- Package-by-package map (`enose/server`, `client`, `classifier`, `vision`, `robot`, `visualization`, `config.py`) and the main function of each key module.
- Endpoint table (all routes under `enose/server/routes/`).
- Scripts (`run_server.py`, `run_client.py`, `run_robot_trainer.py`, `sim_run.py`, `audit_auto_labels.py`, `smoke_test.py`).
- Known problems / bugs section (the CV crash, uncalibrated thresholds, manual pump, ROS2 stubs, Wi-Fi hardening, the UI issues fixed here) with file pointers.
- How to run/test (Docker server, sim mission, client offline).

This is documentation only; no code impact.

---

## Part 2 — UI fixes (`enose/server/routes/ui.py`, plus one small server change)

All UI is a single embedded HTML string in `ui.py`. Changes are contained there except one
new/extended endpoint.

### 2a. Stop the Model Info page from going stale
- Add `cache: 'no-store'` (and a `?t=` cache-buster) to the dynamic GET fetches: `/smell/model_info`, `/` , `/smell/drift`, `/sensor/live/recent`. Primary fix targets `loadInfo()` and `refreshStatus()`.
- Server-side: set `Cache-Control: no-store` on `GET /smell/model_info` (in `enose/server/routes/health.py`) as defense-in-depth. Use a `JSONResponse` with headers, or a small dependency.
- After a successful train in `trainCsv()`, call `loadInfo()` in addition to `refreshStatus()` so the Model Info tab and the dynamic buttons refresh immediately.

### 2b. Add class distribution + richer model info to the Model Info tab
- Extend `BalancedRFClassifier.get_model_info()` (`enose/classifier/balanced_rf.py:557`) with:
  - `class_distribution`: per-class training-sample counts from `self.last_training_data[self._label_col].value_counts()` (guarded for None).
  - `total_training_samples`, and surface existing-but-unshown fields already returned (`calibration`, `features`, `env_config`, `class_weights`).
- In `loadInfo()` render:
  - A **class distribution** bar list (class → count, % of total).
  - A small facts block: backend type (from `/`), n_features, calibration method, training-runs count, last accuracy/balanced accuracy (already present), total samples.
  Keep the existing per-class P/R/F1 table and confusion matrix.

### 2c. Dynamic "Try: <class>" buttons for every known class
- Add a server endpoint `GET /smell/class_examples` (in `routes/smell.py` or `analytics.py`) returning, per known class, a representative 22-feature vector = per-class mean of `last_training_data` (raw, ordered to `ALL_SENSORS`). Returns `{}` when unfitted.
  - Reuse `ALL_SENSORS` ordering from `enose.config`; reuse `last_training_data` already held by the classifier.
- In the UI: on load and after training, fetch class examples and **generate one `Try: <class>` button per class** into the `.brow` container (replacing the hardcoded coffee/air buttons). `fillEx(label)` looks up the fetched vector. Buttons wrap; falls back to the two static examples only if the model is unfitted/no data.

### 2d. Other UI issues found (fold into the same pass)
- "Known smells" badge shows only a count — also show/refresh it after training (covered by 2a auto-refresh).
- Classify-tab "Try" buttons stale after a retrain that changes classes — covered by re-fetching class examples after training (2c).

**Critical files:** `enose/server/routes/ui.py`, `enose/server/routes/health.py`,
`enose/classifier/balanced_rf.py` (get_model_info), `enose/server/routes/smell.py` (new endpoint).

---

## Part 3 — Small-batch retrain fix (Idea-1 blocker)

In `enose/classifier/balanced_rf.py`:
- Make the calibration CV adaptive instead of the fixed `cv=3` at line 95. Before building the
  calibrator in the training path, compute the minimum per-class count of the training split;
  set `cv = max(2, min(3, min_class_count))` and, if `min_class_count < 2`, skip calibration
  (return the bare `BalancedRandomForestClassifier`) so a single-sample class can't crash the fit.
- Keep it centralized so both `train()` and `online_update()` (which refits from scratch) benefit.
- This complements the existing policy-level graceful skip; the goal is that a legitimate small
  new-scent batch **succeeds** instead of 500ing.

**Verification:** unit-style check — fit on a dataset where one class has 1–2 samples; assert no
exception and `is_fitted` is true. Then exercise `POST /smell/online_learning` with a tiny
new-class batch against the running server and assert HTTP 200.

**Critical files:** `enose/classifier/balanced_rf.py` (`_build_model` / `train` calibration path).

---

## Part 4 — Prebuilt client wheel (Windows)

Approach: split dependencies so a client install is light, then build the universal wheel.

### 4a. Restructure `pyproject.toml`
- Shrink base `dependencies` to shared light deps that always have prebuilt wheels:
  `numpy>=1.24`, `pandas>=2.0,<3.0`, `requests>=2.31`, `pillow>=10.0`.
- Add extras:
  - `client = ["pyserial>=3.5", "matplotlib>=3.7", "opencv-python>=4.8"]` (matches `docker/Dockerfile-client`).
  - `server = ["fastapi>=0.110,<0.120", "uvicorn[standard]>=0.29", "pydantic>=2.5", "python-multipart>=0.0.9", "numpy>=1.24,<2.0", "scikit-learn>=1.3,<1.6", "joblib>=1.3"]` — the numpy/sklearn pins move here (server-only).
  - Keep `vision`, `classifier-extras`, `dev` as-is.
- Net effect: `pip install "enose[client]"` pulls **no** fastapi/scikit-learn<1.6/numpy<2 — the exact pins that fail to build on Windows Python.
- Docker is unaffected: `Dockerfile-cu124` and `Dockerfile-client` install deps explicitly, not via `.[extras]`.

### 4b. Build the wheel
- `enose` is pure Python → the wheel is `enose-3.1.0-py3-none-any.whl` (built once on Linux, installs on Windows). Build with `python -m build --wheel` (install `build` into a venv) or inside the client Docker image. Output to `dist/`.

### 4c. Windows deliverables
- Add `requirements-client.txt` (pinned, wheel-available versions) as a fallback.
- Add `docs/CLIENT_WINDOWS_INSTALL.md`: create a Python 3.11/3.12 venv, then
  `pip install "enose-3.1.0-py3-none-any.whl[client]"` (recommend 3.11/3.12 where all deps have prebuilt wheels; note 3.13 may lack some).
- Update install commands in `docs/CLIENT_QUICKSTART.md` / `docs/SETUP.md` / `docs/SERVER_CLIENT_GUIDE.md` to use the new extras (`[client]`, `[server]`, `[server,vision,classifier-extras]`).

**Critical files:** `pyproject.toml`, new `requirements-client.txt`, new `docs/CLIENT_WINDOWS_INSTALL.md`, doc edits.

---

## Part 5 — VLM & simulation readiness review (assessment, minimal/no code)

Findings from code review (to be stated to the user, and captured in `docs/REPO_OVERVIEW.md`):
- **VLM (Florence-2): READY to test.** Loader (`enose/vision/florence.py`) and routes
  `/predict/object` + `/predict/scene` (`routes/vision.py`) are wired; `ENOSE_NO_VLM=1` disables
  cleanly. Needs model weights + GPU (or slow CPU). Verify with a `POST /predict/scene` on a test image.
- **Robot simulation: READY to test.** `scripts/sim_run.py` runs the full policy state machine
  against a `ENOSE_NO_VLM=1` server using `ScriptedVision` + sim adapters; no hardware/Florence needed.
  Verify end-to-end with the sim mission command below.
- **NOT ready (out of scope now):** ROS2 adapters (`enose/robot/ros2/*` raise `NotImplementedError`)
  and visual-servo integration in `policy._approach` (sim teleports instead). These gate the real
  robot, not sim.

I will run the sim + VLM-off smoke checks during implementation to confirm the "ready" claims.

---

## End-to-end verification (Docker-first, per workflow preference)

1. **Server up (no VLM):** `ENOSE_NO_VLM=1 python scripts/run_server.py` → `curl /health`.
2. **UI fixes:** open `/ui`; train a small CSV on the Train tab → confirm Model Info updates
   automatically (classes, class distribution, confusion matrix) with **no manual refresh**;
   confirm a `Try: <class>` button appears for every known class and fills correct values;
   retrain with a new class → buttons + info update.
3. **CV fix:** `POST /smell/online_learning` with a tiny new-class batch → expect HTTP 200
   (previously intermittent 500); Model Info shows the new class.
4. **Sim mission:** `python scripts/sim_run.py --server http://localhost:8080 --targets scripts/sim_targets.json --speed 8` → mission reaches `done` with commits.
5. **VLM (if GPU/weights available):** start server without `ENOSE_NO_VLM`; `POST /predict/scene` with a test image → detections returned.
6. **Wheel:** build `dist/enose-3.1.0-py3-none-any.whl`; in a clean venv run
   `pip install "dist/enose-3.1.0-py3-none-any.whl[client]"` and `python -c "import enose.client.api"`
   to confirm the client imports with only the light deps.

## Out of scope (Idea 2 / later)
ROS2 adapter implementation, visual-servo wiring, threshold calibration on real photos,
`/smell/intensity` & `/smell/spatial_map`, plume tracers.
