# enose/classifier — Smell Classifiers

Two interchangeable classifier backends share the same public interface, the
same preprocessing chain, and the same persistence format. Switching between
them is a single env var: `ENOSE_CLASSIFIER=balanced_rf` (default) or
`ENOSE_CLASSIFIER=xgb`.

Everything the server and UI know about the model — per-class P/R/F1, the
confusion matrix, OOD diagnostics, the last-seen training distribution used
for drift scoring — comes out of the objects in this sub-package.

## Classes

### `BalancedRFClassifier` (default)

`balanced_rf.py` — `BalancedRandomForestClassifier` wrapped in
`CalibratedClassifierCV` (sigmoid / Platt scaling). Used by the server by
default; also the parent class for `XGBTabularClassifier` so the two share
the same preprocessing, diagnostics, persistence, and metric-capture paths.

**Pipeline:**
1. `preprocessing.sanitize_environmentals` clamps T/H/CO2/H2S/CH2O into the
   sane ranges declared in `config.ENV_SANITY_RANGES`. NaNs and
   out-of-range values get the per-sensor median or the configured default.
2. `ensure_canonical_columns` fills missing sensor columns with defaults so
   a partial dict still classifies.
3. `StandardScaler` is fit on R1–R17 **only** — environmental sensors stay
   in raw engineering units so drift and OOD distances remain interpretable.
4. `BalancedRandomForestClassifier` → `CalibratedClassifierCV(method="sigmoid")`.
5. After fitting, a held-out test report is written via `_log_test_report` —
   this is where per-class P/R/F1 and the confusion matrix are captured
   (see **Metrics capture** below).
6. Online update = merge new rows with the stored history
   (`last_training_data`) and refit from scratch. This keeps the model
   deterministic rather than accumulating hidden state from partial fits.

**Key methods:**

| Method | Description |
|---|---|
| `train(X, y, ...)` | Full (re)training. Returns balanced accuracy. Populates `per_class_metrics_`, `confusion_matrix_`, `confusion_labels_`, `last_test_size_`. |
| `online_update(X, y, ...)` | Appends rows to `last_training_data` and retrains. New classes are handled automatically (full retrain, not a partial fit). |
| `predict(X)` | Returns list of label strings. |
| `predict_proba(X)` | Returns `(n_samples, n_classes)` calibrated probabilities. |
| `process_sensor_data(X)` | Sanitize + scale — call this explicitly when you want the preprocessed feature vector. |
| `diagnose_sample(X)` | Returns `{z_scores, ood_score, nearest_centroid_L2, nearest_class, ...}`. Used by `/smell/classify` to build the `ood` block. |
| `predict_with_ood(X)` | Predict + damp confidence when OOD. Uses the same thresholds as the `/smell/classify` status enum — single source of truth for "OOD". |
| `get_model_info()` | Returns the dict consumed by `/smell/model_info`. Includes `per_class_metrics`, `confusion_matrix`, `confusion_labels`, `last_test_size`, `classes`, `accuracy`, `n_training_runs`, plus the legacy feature-importances block. |
| `generate_visualizations(out_dir)` | Writes confusion matrix, class counts, feature importances, environmental histograms, per-class feature importance heatmap, and reliability diagram to `out_dir/plots/`. |
| `save_model(out_dir)` / `load_model(path)` | Joblib persistence. `load_model` is backward-compatible with artifacts that pre-date the metrics-capture fields. |

### `XGBTabularClassifier`

`xgb_tabular.py` — subclasses `BalancedRFClassifier` and swaps the estimator
for `XGBClassifier` via `_XGBStringAdapter` (translates string labels to the
integer codes XGBoost expects, and back). Everything else — preprocessing,
metrics capture, OOD, persistence — is inherited unchanged.

This inheritance is deliberate: an apples-to-apples A/B between BRF and XGB
means the two paths **must** diverge only at the estimator call. If you
change preprocessing, drift, or OOD for one, you get it for the other for
free.

Selected at startup by `ENOSE_CLASSIFIER=xgb`.

### `XGBOdorClassifier` (legacy, separate path)

`xgb.py` — the earlier window-based XGBoost model with exemplar replay and
EMA drift correction. Kept for backward compatibility with older saved
artifacts; **not** on the A/B path. If you are adding new features, put
them on `BalancedRFClassifier` so both `balanced_rf` and `xgb` backends
pick them up.

---

## Metrics capture

Every `train()` call ends with `_log_test_report(X_test, y_test, y_pred)`.
On top of printing the sklearn report, it persists four structured fields
on `self`:

```python
self.per_class_metrics_   # dict[class -> {precision, recall, f1-score, support}]
self.confusion_matrix_    # list[list[int]]  — JSON-serialisable
self.confusion_labels_    # list[str]         — row/column order
self.last_test_size_      # int               — held-out sample count
```

These are what the UI's Model Info tab renders:

- Per-class F1 is colour-coded so a chemist can immediately see which
  classes are under-collected (green ≥ 0.85, amber 0.70–0.85, red < 0.70).
- The confusion matrix is drawn as a heat-map so off-diagonal clusters
  like "rose predicted as lemon" pop out visually.

### Why `classification_report(output_dict=True)` rather than computing P/R/F1 by hand?

It handles zero-support classes, weighted vs. macro averaging, and edge
cases (e.g. a class appears in `y_true` but not in `y_pred`) in one call,
so the numbers in the UI match what any sklearn user would get if they
recomputed them from the same split.

### Why store the confusion matrix as a list of lists?

It has to survive `json.dumps` in the `/smell/model_info` response and
`joblib.dump` on disk without numpy-vs-plain-list shenanigans. List of
lists is the least-surprising representation for both consumers.

---

## OOD diagnostics

`diagnostics.py` computes three per-sample scalars:

| Metric | Meaning |
|---|---|
| `z_scores` | Per-feature `(x - mean) / std` against the training distribution |
| `ood_score` | Mean of `|z_scores|` across features — a single scalar "how unusual is this sample" |
| `nearest_centroid_L2` | L2 distance in the scaled feature space to the nearest class centroid |

`/smell/classify` and `/smell/test_console` both surface these through an
`ood` block in the response:

```json
"ood": {
  "available": true,
  "score": 1.42,
  "min_centroid_L2": 2.87,
  "nearest_centroid_L2": 2.87,
  "status": "ok",
  "thresholds": {
    "ood_warn": 2.0, "ood_out": 3.0,
    "centroid_warn": 3.5, "centroid_out": 5.0
  }
}
```

Status logic (same thresholds used by `predict_with_ood` for
confidence damping, so the traffic light and the model's own confidence
decisions agree):

- `out`  — `ood_score > 3.0` **or** `min_centroid_L2 > 5.0`
- `warn` — `ood_score > 2.0` **or** `min_centroid_L2 > 3.5`
- `ok`   — otherwise

The UI renders the status as a green / yellow / red badge above the
predicted smell.

---

## Training-distribution retention

`self.last_training_data` keeps the full `(X, y)` most recently used for
training. Two consumers rely on it:

1. **Online update** — merges new rows with this history before refitting,
   so incremental learning never silently drops older classes.
2. **`/smell/drift`** — `routes/analytics.py` reads `last_training_data` as
   the "reference distribution" and compares the live ring buffer against
   it per-sensor. If this field is empty, the drift endpoint returns
   `status: "na"` for every sensor rather than crashing.

---

## Persistence

`persistence.py` builds the joblib payload with every field a consumer
might need after a server restart:

```python
payload = {
    "model": ..., "scaler": ..., "classes_": ...,
    "last_training_data": ...,                  # → drift + online update
    "per_class_metrics_": ...,                  # → UI Model Info table
    "confusion_matrix_": ..., "confusion_labels_": ...,
    "last_test_size_": ...,
    "feature_importances_": ..., "accuracy_": ...,
    "n_training_runs_": ...,
}
```

`load()` restores each field with a `.get(key, default)` so artifacts
saved before the metrics-capture fields existed still load cleanly — they
just come back with empty `per_class_metrics_ = {}` and a zeroed test
size until the next retrain.

---

## Submodules

| File | Contents |
|---|---|
| `base.py` | `SmellClassifierBase` abstract interface |
| `config.py` | `SmellClassifierConfig` dataclass (n_estimators, calibration, augmentation settings) |
| `preprocessing.py` | Pure functions: `sanitize_environmentals`, `ensure_canonical_columns`, `scale_resistances`, augmentation, clip bounds |
| `diagnostics.py` | `diagnose_sample`, `compute_z_scores`, `compute_ood_score`, `nearest_centroid` |
| `persistence.py` | `build_save_payload`, `save`, `load` (with legacy fallback) |
| `training.py` | `retrain_with_all_data()` — server-level retrain helper used by `/smell/online_learning` when new classes appear |
| `xgb_config.py` | `XGBClassifierConfig` dataclass |
| `xgb_features.py` | Pure window-feature functions for the legacy windowed path (no state) |

---

## Adding a new classifier backend

1. Create `enose/classifier/my_clf.py`. The path of least resistance is
   subclassing `BalancedRFClassifier` and overriding the estimator
   construction — you inherit preprocessing, metrics capture, OOD, drift
   retention, and persistence for free.
2. Export it from `enose/classifier/__init__.py`.
3. Wire it into `load_or_create_classifier()` in
   `enose/server/model_loader.py`, keyed on a new `ENOSE_CLASSIFIER` value.

If you instead implement `SmellClassifierBase` from scratch, you **must**
also populate `per_class_metrics_`, `confusion_matrix_`,
`confusion_labels_`, `last_test_size_`, and `last_training_data` yourself
— the server and UI read those fields directly.

---

## Training data format

```python
import pandas as pd
from enose.classifier import BalancedRFClassifier

X = pd.DataFrame([
    {"R1": 15.2, "R2": 8.3, ..., "T": 21.0, "H": 49.0, "CO2": 400, "H2S": 0, "CH2O": 5},
])
y = pd.Series(["coffee"])

clf = BalancedRFClassifier()
acc = clf.train(X, y, use_augmentation=True, n_augmentations=5)
print(f"Balanced accuracy: {acc:.3f}")
print(clf.per_class_metrics_)   # newly populated
print(clf.confusion_matrix_)

clf.save_model("trained_models/")
```
