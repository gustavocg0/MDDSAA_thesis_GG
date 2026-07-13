"""Central configuration for the IDS evaluation protocol thesis.

All constants, paths, and hyperparameters live here.
No module outside this file may define its own magic numbers.

══════════════════════════════════════════════════════════════
FROZEN SCOPE  (do not expand without explicit decision)
══════════════════════════════════════════════════════════════
Datasets   : BCCC-CIC-IDS-2017, CSE-CIC-IDS2018
Task       : Binary classification only (benign vs attack)
Protocols  : random stratified split, temporal holdout
Contributions:
  C1 — inflation gap (random vs temporal metric delta)
  C2 — latent-space drift (PSI + KS on autoencoder latent dims)

OUT OF SCOPE:
  ✗ Sliding window protocol
  ✗ Multiclass classification
  ✗ Feature attribution / SHAP
  ✗ Calibration analysis
══════════════════════════════════════════════════════════════
"""

from pathlib import Path

# ── Reproducibility ───────────────────────────────────────────────────────
SEEDS: list[int] = [42, 43, 44, 45, 46]

# ── Paths ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
DATA_DIR = _ROOT.parent / "data" / "raw"
OUTPUT_DIR = _ROOT.parent / "results"

# ── Dataset configurations ────────────────────────────────────────────────
DATASETS: dict = {
    "CIC2017": {
        "path": DATA_DIR / "BCCC-CIC-IDS-2017",
        "type": "cic2017",
        "label_col": "label",
    },
    "CIC2018": {
        "path": DATA_DIR / "CSE-CIC-IDS2018",
        "type": "cic2018",
        "label_col": "label",
    },
}

# ── Protocols ─────────────────────────────────────────────────────────────
PROTOCOLS: list[str] = ["random", "temporal"]

# ── XGBoost hyperparameters (FROZEN across all runs) ─────────────────────
MODEL_PARAMS: dict = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "verbosity": 0,
}

EARLY_STOPPING_ROUNDS: int = 20

# ── Autoencoder ───────────────────────────────────────────────────────────
LATENT_DIM: int = 16
AUTOENCODER_HIDDEN_DIM: int = 64
AUTOENCODER_EPOCHS: int = 50
AUTOENCODER_BATCH_SIZE: int = 256
AUTOENCODER_LR: float = 1e-3
# Cap training rows for the AE (scalability for CIC2018 ~9.7M train rows).
AE_MAX_TRAIN_SAMPLES: int = 500_000

# ── Split ratios (random protocol) ───────────────────────────────────────
RANDOM_SPLIT: dict = {"train": 0.7, "val": 0.1, "test": 0.2}

# ── Temporal splits ───────────────────────────────────────────────────────
# CIC2017: five-day dataset — natural boundaries determined by attack calendar.
# Monday+Tuesday → training (stable benign + first attack category).
# Wednesday       → validation (DoS variants; hyperparameter tuning only).
# Thursday+Friday → test (Web Attacks, Botnet, PortScan, DDoS — unseen categories).
TEMPORAL_SPLIT_2017: dict = {
    "train": ["Monday", "Tuesday"],
    "val":   ["Wednesday"],
    "test":  ["Thursday", "Friday"],
}

# CIC2018: ten date files in chronological order.
# Cutoff logic: first 60 % → train, next 20 % → val, last 20 % → test.
# With 10 files: train = files 1-6, val = files 7-8, test = files 9-10.
# File stem is used as the "session" identifier (e.g. "02-14-2018").
TEMPORAL_SPLIT_2018: dict = {
    "train": [
        "02-14-2018", "02-15-2018", "02-16-2018",
        "02-20-2018", "02-21-2018", "02-22-2018",
    ],
    "val":  ["02-23-2018", "02-28-2018"],
    "test": ["03-01-2018", "03-02-2018"],
}

# ── Preprocessing ─────────────────────────────────────────────────────────
N_FEATURES_SELECT: int = 30
VARIANCE_THRESHOLD: float = 0.0
MISSING_THRESHOLD: float = 0.5      # drop columns with > 50 % missing values
# Cap rows used to FIT the sklearn pipeline (scalability for CIC2018).
# Transform is still applied to all rows; only the fit (imputer/scaler/selector) is subsampled.
PIPELINE_FIT_MAX_SAMPLES: int = 500_000
# Clip scaled features to [-CLIP_VALUE, +CLIP_VALUE] after SelectKBest.
# Network flow features after RobustScaler can have extreme outliers
# (sparse features with IQR ≈ 0 produce scaled values in the millions).
# Clipping is applied after selection and is disclosed in the audit.
CLIP_VALUE: float = 10.0

# ── Drift ─────────────────────────────────────────────────────────────────
PSI_N_BINS: int = 10
PSI_THRESHOLDS: dict = {"low": 0.1, "moderate": 0.25}
ALPHA: float = 0.05

# ── Reporting ─────────────────────────────────────────────────────────────
FIGURE_DPI: int = 150

# ── Leakage columns ───────────────────────────────────────────────────────
# CIC2017 files already use snake_case column names.
LEAKAGE_COLS_2017: list[str] = [
    "flow_id", "timestamp", "src_ip", "dst_ip",
    "src_port", "dst_port", "protocol",
]

# CIC2018 columns are Title Case; normalised names are used here.
LEAKAGE_COLS_2018: list[str] = [
    "timestamp", "dst_port", "protocol",
]

# Columns added by the loader that must be excluded from the feature matrix.
META_COLS: list[str] = ["label", "day", "session", "attack_type"]
