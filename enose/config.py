"""Central configuration: sensor lists, paths, transform coefficients, defaults.

All magic numbers and paths live here. Edit this file, not individual modules,
when tuning the 22-feature setup or moving data around.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ── Sensor lists ──────────────────────────────────────────────────────────────
RESISTANCE_SENSORS: List[str] = [f"R{i}" for i in range(1, 18)]  # 17 sensors
ENVIRONMENTAL_SENSORS: List[str] = ["T", "H", "CO2", "H2S", "CH2O"]  # 5 sensors
ALL_SENSORS: List[str] = RESISTANCE_SENSORS + ENVIRONMENTAL_SENSORS  # 22 total
N_FEATURES: int = len(ALL_SENSORS)

# ── Raw-ADC → resistance transform (client-side, R1–R17 only) ────────────────
RLOW: float = 1.0
VCC: float = 4.49
EG: float = 1.02
VREF: float = 1.227
COEF: float = 65536.0

# ── Environmental sensor sanity ──────────────────────────────────────────────
ENV_SANITY_RANGES: Dict[str, Tuple[float, float]] = {
    "T": (5.0, 30.0),
    "H": (25.0, 65.0),
    "CO2": (300.0, 2500.0),
    "H2S": (0.0, float("inf")),
    "CH2O": (0.0, 200.0),
}
ENV_SANITY_MEDIANS: Dict[str, float] = {
    "T": 20.0,
    "H": 45.0,
    "CO2": 200.0,
    "H2S": 0.0,
    "CH2O": 10.0,
}
ENV_DEFAULTS: Dict[str, float] = {
    "T": 21.0, "H": 49.0, "CO2": 400.0, "H2S": 0.0, "CH2O": 5.0,
}

# ── Recording defaults ───────────────────────────────────────────────────────
DEFAULT_RECORDING_TIME: int = 60
DEFAULT_TARGET_SAMPLES: int = 100
DEFAULT_BAUD_RATE: int = 9600

# ── Paths ────────────────────────────────────────────────────────────────────
VLM_MODEL_PATH: str = "model/Florence-2-Large"
SMELL_MODEL_PATH: str = "trained_models/smell_classifier_sgd_latest.joblib"
TRAINING_DATA_PATH: str = "database_robodog.csv"
TRAINED_MODELS_DIR: str = "trained_models"
DATA_DIR: str = "data"
PLOTS_DIR: str = "plots"
TEMP_IMAGE_PATH: str = "temp_capture.png"

# ── Classifier training defaults ─────────────────────────────────────────────
AUGMENT_N: int = 5
AUGMENT_NOISE_STD: float = 0.0015
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42

# ── Server ───────────────────────────────────────────────────────────────────
SERVER_HOST: str = "0.0.0.0"
SERVER_PORT: int = 8080
# Inside Docker the server listens on 8080; the run command maps it to host port 8016.
# Pass --server http://HOST:8016 when connecting from outside the container.
DEFAULT_SERVER_URL: str = "http://localhost:8080"
