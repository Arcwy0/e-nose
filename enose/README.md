# enose — Package Overview

Python package implementing the full e-nose multimodal system. Every feature
the UI exposes is backed by a module in one of these sub-packages; this file
is the map that points you at the right one.

## Sub-packages

| Package | What lives there | Touch when… |
|---|---|---|
| [`classifier/`](classifier/README.md) | `BalancedRFClassifier`, `XGBTabularClassifier`, preprocessing, diagnostics, persistence, per-class metrics capture | Changing the model / features, adding a new backend, tweaking OOD gating |
| [`server/`](server/README.md) | FastAPI routes including CSV preview and candidate deployment; process-local state/buffers/locks | API/UI and server state |
| [`client/`](client/README.md) | PCB/legacy/replay acquisition, heater/lifecycle/preflight, HTTP, camera, recording and CLI | Acquisition, controls and live dashboard |
| [`visualization/`](visualization/README.md) | Matplotlib plots: confusion matrix, feature importance, per-class F-ratio heatmap, calibration / reliability diagram, environmental histograms | Adding a plot or changing how `generate_all_plots` wires them up |
| [`vision/`](#vision) | Florence-2 loader, GPU detection | Swapping the VLM or GPU check |
| [`utils/`](#utils) | CSV I/O helpers | New CSV format / label column rules |

## Cross-package data flows

### Live streaming loop (`--live` on the client)

```
enose/client/pcb.py / sensors.py / pcb_replay.py
      │    read_single_measurement()            22-feature dict
      ▼
enose/client/live.py  (acquisition + bounded latest-sample network queue)
      │    POST /sensor/live/push               best-effort HTTP publication
      ▼
enose/server/routes/live.py
      │    buffer.push(entry)
      ▼
enose/server/live_buffer.py  (thread-safe ring, default 1000 slots)
      ▲
      │    GET /sensor/live/recent?since=<id>   polled by UI
enose/server/routes/ui.py  (Live tab, inline SVG charts)
```

`enose/client/session.py` records locally acquired readings before HTTP;
network-skipped frames remain without predictions. It does not guarantee every
MCU frame is sampled. `python -m enose.client.replay` reclassifies NPZ without
converting units. `scripts/replay.py` is a checkout-only wrapper.

For all client settings/defaults and menu/UI equivalents see the
[complete CLI reference](client/README.md#complete-cli-reference).

### CSV and deployment

`utils/training_preview.py` prepares both preview and training. Preview never
fits or changes active state. UI/menu stage by default; `classifier/registry.py`
stores candidate model/history pairs. Paired independent evaluation, explicit
promotion and rollback are in `server/routes/models.py`. Historical merging is
a separate default-on policy. API defaults remain immediate for compatibility.
Group-isolated validation reports null metrics when independent groups are
insufficient, rather than silently substituting a row split.

### Classify + OOD loop (`/smell/classify`)

```
enose/server/routes/smell.py
      │   clf.predict / predict_proba / diagnose_sample
      ▼
enose/classifier/balanced_rf.py
      │   process_sensor_data → model → calibrated probs
      │   diagnostics.{compute_z_scores, compute_ood_score, nearest_centroid}
      ▼
JSON: {predicted_smell, probabilities, confidence, ood:{score, status, nearest_centroid_L2}}
```

The `ood.status` enum (`ok` / `warn` / `out`) is the same one
`predict_with_ood` uses internally, so the browser traffic light and the
confidence-damping path agree on what "OOD" means.

### Drift loop (`/smell/drift`)

```
enose/server/routes/analytics.py
      │   clf.last_training_data                reference distribution
      │   live_buffer.snapshot()                current distribution
      ▼
per-sensor {train_mean, live_mean, z_shift, std_ratio, status}
```

## Shared configuration

All constants and paths are in [`config.py`](config.py). Edit that file — not
individual modules — when changing sensor counts, paths, or transform
coefficients.

```python
from enose.config import ALL_SENSORS, RLOW, VCC, SERVER_PORT, DEFAULT_SERVER_URL
```

Frequently touched keys:

| Constant | Used by |
|---|---|
| `ALL_SENSORS` / `RESISTANCE_SENSORS` / `ENVIRONMENTAL_SENSORS` | Feature ordering everywhere. Change = schema change. |
| `N_FEATURES` | Server exposes this via `/` so the client's feature count check has something to compare against. |
| `RLOW`, `VCC`, `EG`, `VREF`, `COEF` | Legacy client ADC transform only; PCB firmware supplies calibrated resistance. |
| `ENV_SANITY_RANGES` / `ENV_SANITY_MEDIANS` / `ENV_DEFAULTS` | Preprocessing (`preprocessing.sanitize_environmentals`). |
| `DEFAULT_SERVER_URL` | Default for the client's `--server` flag and `scripts/replay.py`. |

## Vision

`enose/vision/florence.py` — `load_florence_model(device, model_path)` returns `(model, processor)`.
`enose/vision/gpu.py` — `check_gpu_availability()` returns `"cuda"` or `"cpu"`.

These are called by the server startup. The client does not import them.
Set `ENOSE_NO_VLM=1` to skip the Florence-2 load entirely — the smell
classifier, live streaming, drift, and every other endpoint still work.

## Utils

`enose/utils/csv_io.py` — CSV loading helpers:

| Function | Description |
|---|---|
| `load_training_csv(path)` | Auto-detect label column, return `(X_df, y_series)` |
| `parse_semicolon_enose_csv(path)` | Handle semicolon-delimited exports |
| `save_training_samples(data, labels, path)` | Append samples to the training log CSV |
| `ensure_canonical_columns(df)` | Fill missing sensor columns with defaults |

`pcb_csv.py` normalizes supplier Rs-mode exports without inventing smell
labels; heater temperature stays diagnostic, not ambient `T`.
`training_preview.py` handles verified row annotations, recording identity and
retained/rejected extraction windows. `acquisition.py` guards units, physical
sample validity and baseline identity. These safeguards do not prove electrical
cross-board equivalence or new chemical-recognition accuracy.
