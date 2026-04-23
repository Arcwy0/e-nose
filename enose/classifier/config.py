"""Tunables for the Balanced RF smell classifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from enose.config import ENV_SANITY_RANGES, ENV_SANITY_MEDIANS, RANDOM_STATE, TEST_SIZE


# Env sensors that are dead (constant or sensor-absent) in the collected data
# and should never be fed to the model, even when `use_env_sensors` is True.
# Analysis of database_robodog_new_cleaned.csv: H2S is 0.0 for every single
# row across 121k samples. Keeping it as a feature just adds a zero column
# that the trees can't use and a scaler might divide by.
DEAD_ENV_FEATURES_DEFAULT: Tuple[str, ...] = ("H2S",)


@dataclass
class SmellClassifierConfig:
    """All hyperparameters for BalancedRFClassifier.

    Edit here, not in the classifier body, when tuning.
    """

    model_type: str = "balanced_rf"  # kept for API compatibility
    random_state: int = RANDOM_STATE
    test_size: float = TEST_SIZE

    # Balanced RF
    n_estimators: int = 800
    max_depth: Optional[int] = None
    n_jobs: int = -1

    # Augmentation (uniform noise on R1-R17 only).
    # Keep this small — noise on an already-large, leaky dataset just inflates
    # near-duplicates and inflates reported accuracy.
    use_augmentation: bool = True
    n_augmentations: int = 1
    noise_max: float = 0.005

    # Feature shaping
    use_log1p: bool = True          # log1p on R1-R17 before scaling
    scaler_kind: str = "robust"     # "robust" (RobustScaler) or "standard" (StandardScaler)

    # Feature selection. When False, T/H/CO2/H2S/CH2O are excluded from the
    # model's input. They still pass through the preprocessing pipeline (so
    # sanity ranges and median fills still apply), but are dropped before
    # fit / predict. Recommended for the current dataset where env sensors
    # are near-constant across classes and out-of-range real-world readings
    # get silently replaced with medians, which confuses inference.
    use_env_sensors: bool = False

    # Env features to exclude from the model even when `use_env_sensors=True`.
    # See DEAD_ENV_FEATURES_DEFAULT for rationale.
    dead_env_features: Tuple[str, ...] = DEAD_ENV_FEATURES_DEFAULT

    # Data cleanup applied inside retrain_with_all_data before splitting:
    #
    # - drop_sensor_off_air_rows: 'air' samples where R1==0 AND R4==0 are
    #   sensor-off artifacts (~21% of the historical air class). They teach
    #   the model "zeros -> air" which is not the chemistry we want to learn.
    # - per_class_cap_multiplier: cap EVERY class at
    #   ``min_class_count * multiplier``. A value of 3.0 means the largest
    #   class can't exceed 3x the smallest. This is stricter than the
    #   old "runner-up + 15%" rule because "air" and the next-largest
    #   class were historically both huge, leaving minority classes still
    #   drowned out.
    drop_sensor_off_air_rows: bool = True
    per_class_cap_multiplier: float = 3.0

    # Train/test splitting
    dedup_before_split: bool = True      # drop near-identical rows before splitting
    dedup_round_decimals: int = 2        # precision used to detect near-duplicates
    group_shuffle_when_available: bool = True  # GroupShuffleSplit if groups are passed

    # Probability calibration
    calibrated: bool = True
    calibration_method: str = "sigmoid"  # "isotonic" or "sigmoid" (Platt)

    # Environmental sanity
    env_ranges: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(ENV_SANITY_RANGES)
    )
    env_medians: Dict[str, float] = field(
        default_factory=lambda: dict(ENV_SANITY_MEDIANS)
    )

    # Plotting
    figure_dpi: int = 140
