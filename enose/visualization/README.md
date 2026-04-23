# enose/visualization — Plot Functions

Pure matplotlib plotting utilities. Each function takes data and an output
path; nothing is shown on screen (all use `matplotlib.use("Agg")`).

Every plot in this package is designed to be **rendered server-side once,
saved to `data/plots/`, and served as a static PNG** — the browser UI never
re-renders them on the client. This is what keeps the UI dependency-free
(no Plotly, no Chart.js) and consistent across desktop/tablet/phone.

## Module layout

| File | Functions | Output |
|---|---|---|
| `confusion.py` | `plot_confusion_matrix(y_true, y_pred, out_path, ...)` | Heat-map PNG, diagonal = correct |
| `feature_plots.py` | `plot_class_counts(y, out_path, ...)` | Bar chart PNG |
| | `plot_feature_importances(importances, names, out_path, ...)` | Bar chart PNG — model-global feature ranking |
| `per_class_importance.py` | `compute_per_class_f_ratio(X, y, features)` | Pure function → `(n_classes, n_features)` F-ratio array |
| | `plot_per_class_feature_importance(X, y, features, out_path, ...)` | Heat-map PNG, one row per class, top-3 features annotated |
| | `summarize_per_class_importance(X, y, features, top_k=5)` | `{class: [top-k feature names]}` |
| `calibration.py` | `reliability_bins(y_true, y_pred, confidences, n_bins=10)` | Pure stats: per-bin accuracy, mean confidence, support, ECE |
| | `plot_reliability_diagram(y_true, y_pred, confidences, out_path, ...)` | Two-panel PNG (reliability + support histogram); returns `(path, ece)` |
| `environmental.py` | `plot_environmental_histograms(df, out_path, ...)` | 5-panel histogram PNG |
| | `plot_environmental_by_class(df, label_col, out_dir, ...)` | One box-plot PNG per sensor |
| `data_quality.py` | `analyze_data_quality(df, label_col, sensors)` | Pure stats dict (no files) |
| `generate.py` | `generate_all_plots(clf, out_dir)` | Writes the full plot set; returns `{name: rel_path, "calibration_ece": float}` |
| | `generate_environmental_by_class(clf, out_dir)` | Environmental box-plots; returns stat dict |

---

## What `generate_all_plots` produces

All paths are relative to `out_dir` so they map directly to the server's
static-file URLs (`/data/plots/confusion_matrix.png`).

| Key | PNG | Why it exists |
|---|---|---|
| `confusion_matrix` | `confusion_matrix.png` | Which class pairs the model confuses. |
| `class_counts` | `class_counts.png` | Imbalance check — are some classes under-sampled? |
| `feature_importances` | `feature_importances.png` | Global feature ranking from the tree model. |
| `per_class_feature_importance` | `per_class_feature_importance.png` | Per-class discriminative profile (see reasoning below). |
| `calibration` | `calibration.png` | Does the predicted confidence match the actual accuracy? |
| `calibration_ece` | (float, not a path) | Expected Calibration Error — a single scalar summary. |
| `environmental_histograms` | `environmental_histograms.png` | Distribution of T/H/CO2/H2S/CH2O across the training log. |

Each plot is wrapped in `try/except` inside `generate_all_plots` — a
failure in one (e.g. reliability diagram needs ≥ 2 classes in y_test)
does not block the others from rendering.

---

## Per-class feature importance — why F-ratio and not `model.feature_importances_`?

`model.feature_importances_` is **one vector** — a single ranking averaged
across all classes. That is useful for "which sensors matter at all" but
tells a chemist nothing about "which sensor distinguishes coffee from
chocolate."

`compute_per_class_f_ratio` reframes the question as one-vs-rest ANOVA:
for each class `c`, compute `f_classif(X, y == c)` giving a
per-feature F-score, then row-normalise so every class's row sums to 1.
The resulting `(n_classes, n_features)` matrix is rendered as a heat-map
with the top-3 features of each row annotated.

This approach has two advantages:

1. **Model-agnostic.** It works identically for `BalancedRFClassifier`
   and `XGBTabularClassifier`, so the A/B comparison is fair.
2. **Chemically interpretable.** An F-ratio is "how much does this sensor's
   mean shift for this class vs. everything else", which maps directly onto
   a chemist's intuition about sensor selectivity.

---

## Reliability diagram — why two panels and ECE?

`plot_reliability_diagram` produces a single PNG with two stacked axes:

1. **Top panel: accuracy-vs-confidence.** Bars at each confidence bin
   (0.0–0.1, 0.1–0.2, …) showing the empirical accuracy of predictions
   whose confidence falls in that bin, against the `y = x` diagonal.
   Bars below the diagonal = overconfident, above = underconfident.
2. **Bottom panel: support histogram.** How many test samples fell into
   each confidence bin. A reliability bar with only two samples behind it
   is noise, not a calibration signal — this panel lets you see that at
   a glance.

`reliability_bins` also returns the **Expected Calibration Error (ECE)**
— the sample-weighted mean gap between accuracy and confidence across
bins. One scalar, ≥ 0, lower is better. The server surfaces it as
`calibration_ece` in the `/smell/visualize_data` response so the UI can
show "ECE = 0.043" without re-reading the PNG.

`Platt / sigmoid calibration` (in `CalibratedClassifierCV`) should pull
ECE toward zero. If ECE stays large after calibration, that is a signal
the model is mis-specified or the training set is too small for the
calibration fold to converge.

---

## Adding a new plot

1. Create `enose/visualization/my_plot.py` with a `plot_*` function that
   saves a PNG and returns the path. Follow the convention of a pure
   function that takes data in and a file path out — no hidden state.
2. Export the function from `__init__.py`.
3. Optionally wire it into `generate.py:generate_all_plots()`:
   ```python
   try:
       plots["my_plot"] = _rel(plot_my_thing(...))
   except Exception as e:
       print(f"[generate_all_plots] my_plot failed: {e}")
   ```
   The `try/except` is important — one failing plot must not kill the
   whole visualization endpoint.
4. Optionally add a new GET endpoint in
   `enose/server/routes/analytics.py` if the plot is useful independently
   of the main visualization sweep.

## Usage

```python
from enose.visualization import (
    plot_confusion_matrix,
    plot_per_class_feature_importance,
    plot_reliability_diagram,
    generate_all_plots,
)

# Standalone
plot_confusion_matrix(y_test, y_pred, "out/confusion.png", labels=["air", "coffee"])

# Via classifier (writes to data/plots/)
plots = clf.generate_visualizations("data")
# plots = {
#   "confusion_matrix": "plots/confusion_matrix.png",
#   "per_class_feature_importance": "plots/per_class_feature_importance.png",
#   "calibration": "plots/calibration.png",
#   "calibration_ece": 0.043,
#   ...
# }
```

## Notes

- `generate_all_plots` temporarily refits `clf.model` on a train/test
  split to compute held-out predictions for the confusion matrix and the
  reliability diagram. The classifier's own weights on disk are not
  touched — this is a throwaway fit purely for plotting.
- All paths returned are relative to `out_dir`, matching how the server
  serves `/data/plots/*` statically. The browser UI constructs URLs as
  `/data/<returned-rel-path>` and appends `?t=<timestamp>` for cache
  busting after each retrain.
