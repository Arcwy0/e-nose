"""Per-class feature importance (one-vs-rest F-ratio / discriminative profile).

The standard ``feature_importances_`` on a tree model is a single global
ranking: "which features matter on average, across every class." That answers
"is R4 useful?" but not "is R4 useful *for coffee*?" — which is the question
a chemist actually has when staring at a bottle.

For every class we compute a one-vs-rest ANOVA F-ratio per feature (higher =
the feature's mean differs more between that class and everything else).
Normalising each class to its own max keeps features comparable within a
class and makes the heatmap chemically readable: rows are classes, columns
are sensors, bright cells show which sensors *drive* each class.

Caveat: this is a feature-space heuristic, not the model's actual decision
path. For a tree-based model the two agree in broad strokes but not exactly
(the model can use interactions the ANOVA can't see). We plot the ANOVA
version because it is model-agnostic and therefore informative across the
BalancedRF / XGB A/B.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif


def compute_per_class_f_ratio(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
) -> pd.DataFrame:
    """Return a (n_classes, n_features) frame of one-vs-rest F-scores.

    Rows are class labels in sorted order; columns follow ``features``.
    Rows are L2-normalised per class (each class's max → 1) so the heatmap
    doesn't get swamped by whichever class happens to have the largest
    absolute F-ratios.
    """
    X = X[list(features)].copy()
    y_arr = np.asarray(y)
    classes = sorted(pd.unique(y_arr).tolist())

    rows: List[np.ndarray] = []
    for c in classes:
        y_ovr = (y_arr == c).astype(int)
        # f_classif is happy with constant features (returns NaN); mask them
        # out so they don't poison the row normalization.
        f, _ = f_classif(X.values, y_ovr)
        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
        peak = f.max() if f.size else 1.0
        if peak > 0:
            f = f / peak
        rows.append(f)
    return pd.DataFrame(rows, index=classes, columns=list(features))


def plot_per_class_feature_importance(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    out_path: str,
    dpi: int = 120,
    title: str = "Per-class feature importance (one-vs-rest F-ratio, normalised)",
) -> str:
    """Heatmap: rows = class, cols = sensor, colour = normalised F-ratio."""
    heat = compute_per_class_f_ratio(X, y, features)
    if heat.empty:
        return out_path

    n_cls, n_feat = heat.shape
    fig_w = max(7.0, 0.35 * n_feat + 2.5)
    fig_h = max(3.0, 0.45 * n_cls + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    im = ax.imshow(heat.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n_feat))
    ax.set_xticklabels(list(heat.columns), rotation=45, ha="right")
    ax.set_yticks(range(n_cls))
    ax.set_yticklabels(list(heat.index))
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("F-ratio (row-normalised)")

    # Annotate the top-3 per row so chemists can jump straight to the
    # dominant sensors without reading the colour scale.
    for i, cls in enumerate(heat.index):
        row = heat.iloc[i].values
        top_idx = np.argsort(row)[::-1][:3]
        for j in top_idx:
            ax.text(j, i, heat.columns[j], ha="center", va="center",
                    fontsize=7, color="white" if row[j] > 0.5 else "black")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def summarize_per_class_importance(
    X: pd.DataFrame,
    y: pd.Series,
    features: Sequence[str],
    top_k: int = 5,
) -> Dict[str, List[str]]:
    """Return ``{class -> [top-k feature names]}`` without drawing anything.

    Handy for surfacing a quick "for coffee, look at R4 / R12 / CO2" hint in
    the UI without round-tripping an image.
    """
    heat = compute_per_class_f_ratio(X, y, features)
    out: Dict[str, List[str]] = {}
    for cls in heat.index:
        row = heat.loc[cls]
        out[str(cls)] = row.sort_values(ascending=False).head(top_k).index.tolist()
    return out
