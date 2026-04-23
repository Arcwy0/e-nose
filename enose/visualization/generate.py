"""Orchestrator: run every standard plot over a fitted classifier's last training batch.

Calling `generate_all_plots(clf, out_dir)` writes PNGs under `{out_dir}/plots/` and
returns a `{plot_name → relative_path}` dict. This is what the server's
`/smell/visualize_data` endpoint ultimately invokes via `clf.generate_visualizations`.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from enose.classifier import preprocessing

from .calibration import plot_reliability_diagram
from .confusion import plot_confusion_matrix
from .environmental import plot_environmental_by_class, plot_environmental_histograms
from .feature_plots import plot_class_counts, plot_feature_importances
from .per_class_importance import plot_per_class_feature_importance


def generate_all_plots(clf: Any, out_dir: str = "data") -> Dict[str, str]:
    """Confusion matrix, class counts, feature importances, env hists. Mutates `clf.model`.

    The confusion-matrix panel re-splits the stored training data and refits `clf.model`
    to compute held-out predictions. This matches the legacy behavior the client relies on.
    """
    if clf.last_training_data is None or not clf.is_fitted:
        return {}

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out: Dict[str, str] = {}

    X = clf.last_training_data[clf.ALL_SENSORS]
    y = clf.last_training_data[clf._label_col]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=clf.config.test_size,
        random_state=clf.config.random_state,
        stratify=y,
    )
    scaler = StandardScaler()
    X_trs = preprocessing.scale_resistances(X_tr, scaler, fit=True)
    X_tes = preprocessing.scale_resistances(X_te, scaler, fit=False)
    clf.model.fit(X_trs[clf.ALL_SENSORS].values, y_tr.values)
    y_pred = clf.model.predict(X_tes[clf.ALL_SENSORS].values)

    labels = np.unique(y)
    cm_path = os.path.join(plots_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_te, y_pred, cm_path, labels=labels, dpi=clf.config.figure_dpi)
    out["confusion_matrix"] = os.path.relpath(cm_path, out_dir)

    cc_path = os.path.join(plots_dir, "class_counts.png")
    plot_class_counts(y, cc_path, dpi=clf.config.figure_dpi)
    out["class_counts"] = os.path.relpath(cc_path, out_dir)

    if hasattr(clf.model, "feature_importances_"):
        fi_path = os.path.join(plots_dir, "feature_importances.png")
        plot_feature_importances(
            clf.model.feature_importances_,
            clf.ALL_SENSORS,
            fi_path,
            dpi=clf.config.figure_dpi,
            title="Feature Importances (Balanced RF)",
        )
        out["feature_importances"] = os.path.relpath(fi_path, out_dir)

    # Per-class feature importance (one-vs-rest F-ratio heatmap). Computed on
    # the *scaled* training split so the magnitudes are comparable to the
    # feature values the model actually sees.
    try:
        pci_path = os.path.join(plots_dir, "per_class_feature_importance.png")
        plot_per_class_feature_importance(
            X_trs[clf.ALL_SENSORS], y_tr, clf.ALL_SENSORS, pci_path,
            dpi=clf.config.figure_dpi,
        )
        out["per_class_feature_importance"] = os.path.relpath(pci_path, out_dir)
    except Exception as e:
        print(f"[plots] per-class feature importance failed: {e}")

    # Calibration / reliability diagram. Requires predict_proba — skipped
    # silently when the estimator only exposes hard predictions.
    try:
        if hasattr(clf.model, "predict_proba"):
            proba = clf.model.predict_proba(X_tes[clf.ALL_SENSORS].values)
            conf = np.max(proba, axis=1)
            cal_path = os.path.join(plots_dir, "calibration.png")
            _, ece = plot_reliability_diagram(
                y_te.values, y_pred, conf, cal_path,
                dpi=clf.config.figure_dpi,
            )
            out["calibration"] = os.path.relpath(cal_path, out_dir)
            out["calibration_ece"] = float(ece)
    except Exception as e:
        print(f"[plots] calibration plot failed: {e}")

    eh_path = os.path.join(plots_dir, "environmental_hist.png")
    plot_environmental_histograms(X, eh_path, env_sensors=clf.ENVIRONMENTAL_SENSORS,
                                  dpi=clf.config.figure_dpi)
    out["environmental_hist"] = os.path.relpath(eh_path, out_dir)

    return out


def generate_environmental_by_class(clf: Any, out_dir: str = "data") -> Dict[str, Any]:
    """Per-env-sensor boxplot grouped by class. Used by `/smell/environmental_analysis`."""
    if clf.last_training_data is None:
        return {"summary": "No training data available."}

    plots_dir = os.path.join(out_dir, "plots")
    paths = plot_environmental_by_class(
        clf.last_training_data,
        label_col=clf._label_col,
        out_dir=plots_dir,
        env_sensors=clf.ENVIRONMENTAL_SENSORS,
        dpi=clf.config.figure_dpi,
    )
    out: Dict[str, Any] = {
        f"{col}_by_class": os.path.relpath(p, out_dir) for col, p in paths.items()
    }
    out["summary"] = "Environmental sensor distributions saved."
    return out
