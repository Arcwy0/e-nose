"""Plotting and analysis helpers.

Each file adds one kind of plot. To add a new visualization:
    1. Create a new module here (e.g. `tsne.py`)
    2. Expose a `plot_*` function that takes a DataFrame + output path
    3. Register it in `enose/server/routes/analytics.py` if it should be an endpoint
"""

from .calibration import plot_reliability_diagram, reliability_bins
from .confusion import plot_confusion_matrix
from .data_quality import analyze_data_quality
from .environmental import plot_environmental_by_class, plot_environmental_histograms
from .feature_plots import plot_class_counts, plot_feature_importances
from .generate import generate_all_plots, generate_environmental_by_class
from .per_class_importance import (
    compute_per_class_f_ratio,
    plot_per_class_feature_importance,
    summarize_per_class_importance,
)

__all__ = [
    "plot_confusion_matrix",
    "plot_class_counts",
    "plot_feature_importances",
    "plot_environmental_histograms",
    "plot_environmental_by_class",
    "analyze_data_quality",
    "generate_all_plots",
    "generate_environmental_by_class",
    "plot_reliability_diagram",
    "reliability_bins",
    "plot_per_class_feature_importance",
    "compute_per_class_f_ratio",
    "summarize_per_class_importance",
]
