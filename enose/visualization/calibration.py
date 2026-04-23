"""Calibration / reliability diagram for probabilistic multi-class classifiers.

For every prediction we collect (predicted-confidence, correct?) pairs, bin
by confidence, and plot empirical accuracy vs. confidence. A perfectly
calibrated model sits on the diagonal: "when I say 70%, I'm right 70% of
the time." Platt-sigmoid-calibrated BRF usually sits close; raw XGBoost with
logloss is close enough that we turn off the calibrator there.

Also returns the Expected Calibration Error (ECE) — a one-number summary of
how far the curve strays from the diagonal, weighted by bin support.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def reliability_bins(
    y_true: Iterable,
    y_pred: Iterable,
    confidences: Iterable[float],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Bin (conf, correct) and return per-bin accuracy + mean-conf + support.

    Bins are equal-width over [0, 1]. Empty bins land as NaN so the plot
    can skip them without drawing a misleading zero.
    """
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    conf = np.asarray(list(confidences), dtype=float)
    correct = (y_true == y_pred).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    bin_acc = np.full(n_bins, np.nan)
    bin_conf = np.full(n_bins, np.nan)
    bin_n = np.zeros(n_bins, dtype=int)

    # Use searchsorted on the right so p=1.0 lands in the last bin.
    idx = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, n_bins - 1)
    for b in range(n_bins):
        mask = idx == b
        bin_n[b] = int(mask.sum())
        if bin_n[b] > 0:
            bin_acc[b] = correct[mask].mean()
            bin_conf[b] = conf[mask].mean()

    # ECE = sum_b (n_b / N) * |acc_b - conf_b|, skipping empty bins.
    total = bin_n.sum()
    if total > 0:
        valid = ~np.isnan(bin_acc)
        ece = float(
            ((bin_n[valid] / total) * np.abs(bin_acc[valid] - bin_conf[valid])).sum()
        )
    else:
        ece = float("nan")

    return {
        "edges": edges.tolist(),
        "midpoints": mids.tolist(),
        "bin_accuracy": bin_acc.tolist(),
        "bin_confidence": bin_conf.tolist(),
        "bin_count": bin_n.tolist(),
        "n_samples": int(total),
        "ece": ece,
    }


def plot_reliability_diagram(
    y_true: Iterable,
    y_pred: Iterable,
    confidences: Iterable[float],
    out_path: str,
    n_bins: int = 10,
    dpi: int = 120,
    title: Optional[str] = None,
) -> Tuple[str, float]:
    """Save the reliability diagram PNG. Returns (path, ECE)."""
    stats = reliability_bins(y_true, y_pred, confidences, n_bins=n_bins)
    mids = np.asarray(stats["midpoints"])
    acc = np.asarray(stats["bin_accuracy"])
    conf = np.asarray(stats["bin_confidence"])
    n = np.asarray(stats["bin_count"])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.5, 6.2), dpi=dpi, gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # Diagonal reference + per-bin bars.
    ax1.plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1, label="Perfect calibration")
    valid = ~np.isnan(acc)
    if valid.any():
        ax1.bar(mids[valid], acc[valid], width=(1.0 / n_bins) * 0.9,
                alpha=0.6, color="#2563eb", label="Empirical accuracy",
                edgecolor="#1e40af")
        ax1.plot(conf[valid], acc[valid], "o-", color="#1d4ed8",
                 markersize=5, linewidth=1.2, label="Accuracy at mean-conf")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Accuracy")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left", fontsize=8)
    ttl = title or f"Reliability diagram  (ECE = {stats['ece']:.3f}, N = {stats['n_samples']})"
    ax1.set_title(ttl)

    # Support histogram — shows how many predictions actually land in each bin.
    # A gorgeous curve on 3 samples in the 0.9 bin is meaningless, so we want
    # this front and centre.
    ax2.bar(mids, n, width=(1.0 / n_bins) * 0.9, color="#64748b", alpha=0.7)
    ax2.set_xlabel("Predicted confidence")
    ax2.set_ylabel("Count")
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path, stats["ece"]
