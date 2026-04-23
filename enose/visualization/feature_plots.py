"""Class counts and feature-importance bar charts."""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_class_counts(
    y: pd.Series,
    out_path: str,
    dpi: int = 120,
    title: str = "Class Counts",
) -> str:
    fig = plt.figure(dpi=dpi)
    y.value_counts().sort_values(ascending=False).plot(kind="bar")
    plt.ylabel("Samples")
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_feature_importances(
    importances: Sequence[float],
    feature_names: Sequence[str],
    out_path: str,
    dpi: int = 120,
    title: str = "Feature Importances",
) -> str:
    """Sort features by importance descending and save a bar chart."""
    imp = np.asarray(importances)
    names = np.asarray(feature_names)
    idx = np.argsort(imp)[::-1]

    fig = plt.figure(dpi=dpi)
    plt.bar(range(len(imp)), imp[idx])
    plt.xticks(range(len(imp)), names[idx], rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
