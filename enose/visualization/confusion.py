"""Confusion-matrix heatmap. Independent of classifier internals — takes y_true/y_pred."""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true,
    y_pred,
    out_path: str,
    labels: Iterable[str] | None = None,
    dpi: int = 120,
    title: str = "Confusion Matrix",
) -> str:
    """Save a confusion matrix PNG. Returns the written path."""
    if labels is None:
        labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    labels = list(labels)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig = plt.figure(dpi=dpi)
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick = np.arange(len(labels))
    plt.xticks(tick, labels, rotation=45, ha="right")
    plt.yticks(tick, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
