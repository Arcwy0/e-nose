"""Per-sample diagnostics: feature z-scores, OOD score, nearest-centroid distances.

These don't affect predictions directly — they help answer "is this sample weird?"
and power the /smell/debug_input endpoint. Also used to damp confidence for
out-of-distribution inputs in predict_with_ood().
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from enose.config import ALL_SENSORS


def _resolve_features(features: Optional[Sequence[str]]) -> List[str]:
    return list(features) if features is not None else list(ALL_SENSORS)


def compute_z_scores(
    sample_row: pd.Series,
    feature_means: Dict[str, float],
    feature_stds: Dict[str, float],
    features: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """z = (x - train_mean) / max(train_std, eps) per feature."""
    zs = {}
    for f in _resolve_features(features):
        mu = feature_means.get(f, 0.0)
        sd = max(feature_stds.get(f, 1.0), 1e-9)
        zs[f] = float((sample_row[f] - mu) / sd)
    return zs


def compute_ood_score(
    z_scores: Dict[str, float],
    features: Optional[Sequence[str]] = None,
) -> float:
    """Global OOD score = mean absolute z across all features."""
    feats = [f for f in _resolve_features(features) if f in z_scores]
    if not feats:
        return 0.0
    return float(np.mean(np.abs([z_scores[f] for f in feats])))


def nearest_centroid_distances(
    sample_row: pd.Series,
    class_centroids: Optional[pd.DataFrame],
    top_k: int = 5,
    features: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """L2 distance from sample to each class centroid, sorted ascending, top-k only."""
    if class_centroids is None or len(class_centroids) == 0:
        return {}
    feats = [f for f in _resolve_features(features) if f in class_centroids.columns]
    v = sample_row[feats].values.reshape(1, -1)
    dists = {}
    for cls_name, row in class_centroids.iterrows():
        c = row[feats].values.reshape(1, -1)
        dists[str(cls_name)] = float(np.linalg.norm(v - c, ord=2))
    return dict(sorted(dists.items(), key=lambda kv: kv[1])[:top_k])


def diagnose(
    scaled_sample: pd.DataFrame,
    feature_means: Dict[str, float],
    feature_stds: Dict[str, float],
    class_centroids: Optional[pd.DataFrame],
    probs: np.ndarray,
    classes: np.ndarray,
    features: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Bundle z-scores + OOD + centroid distances + probs for one sample."""
    row = scaled_sample.iloc[0]
    zs = compute_z_scores(row, feature_means, feature_stds, features=features)
    return {
        "z_scores": zs,
        "ood_score": compute_ood_score(zs, features=features),
        "nearest_centroid_L2": nearest_centroid_distances(row, class_centroids, features=features),
        "classes": list(map(str, classes)),
        "probs": probs.tolist() if hasattr(probs, "tolist") else list(probs),
    }
