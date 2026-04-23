"""Environmental-sensor plots: overall histograms + per-class boxplots."""

from __future__ import annotations

import os
from typing import Dict, Sequence

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from enose.config import ENVIRONMENTAL_SENSORS


def plot_environmental_histograms(
    df: pd.DataFrame,
    out_path: str,
    env_sensors: Sequence[str] = ENVIRONMENTAL_SENSORS,
    dpi: int = 120,
    bins: int = 30,
) -> str:
    """One subplot per env sensor, each a histogram of that column."""
    fig = plt.figure(dpi=dpi)
    for i, col in enumerate(env_sensors):
        if col not in df.columns:
            continue
        plt.subplot(3, 2, i + 1)
        df[col].hist(bins=bins)
        plt.title(col)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_environmental_by_class(
    df: pd.DataFrame,
    label_col: str,
    out_dir: str,
    env_sensors: Sequence[str] = ENVIRONMENTAL_SENSORS,
    dpi: int = 120,
) -> Dict[str, str]:
    """Per-env-sensor boxplot grouped by class. Returns {sensor → filepath}."""
    os.makedirs(out_dir, exist_ok=True)
    classes = sorted(df[label_col].unique())
    out: Dict[str, str] = {}

    for col in env_sensors:
        if col not in df.columns:
            continue
        fig = plt.figure(dpi=dpi)
        data = [df[df[label_col] == c][col].values for c in classes]
        plt.boxplot(data, labels=classes, vert=True, showmeans=True)
        plt.title(f"{col} by class")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        fp = os.path.join(out_dir, f"{col}_by_class.png")
        fig.savefig(fp, bbox_inches="tight")
        plt.close(fig)
        out[col] = fp
    return out
