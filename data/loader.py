"""Load raw CSV files for CIC2017 and CIC2018 datasets.

Returns a single pandas DataFrame with normalised column names,
a temporal identifier column ('day' for 2017, 'session' for 2018),
and a binary integer 'label' column (0 = benign, 1 = attack).
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── CIC2017 file-to-day mapping ───────────────────────────────────────────

_BENIGN_FILE_TO_DAY: dict[str, str] = {
    "monday_benign.csv":    "Monday",
    "tuesday_benign.csv":   "Tuesday",
    "wednesday_benign.csv": "Wednesday",
    "thursday_benign.csv":  "Thursday",
}

_ATTACK_FILE_TO_DAY: dict[str, str] = {
    "ssh_patator-new.csv":    "Tuesday",
    "dos_golden_eye.csv":     "Wednesday",
    "dos_hulk.csv":           "Wednesday",
    "dos_slowhttptest.csv":   "Wednesday",
    "dos_slowloris.csv":      "Wednesday",
    "heartbleed.csv":         "Wednesday",
    "web_brute_force.csv":    "Thursday",
    "web_sql_injection.csv":  "Thursday",
    "web_xss.csv":            "Thursday",
    "botnet_ares.csv":        "Friday",
    "portscan.csv":           "Friday",
    "ddos_loit.csv":          "Friday",
}


def _norm_name(name: str) -> str:
    """Scalar twin of _normalise_columns for a single raw column name."""
    s = re.sub(r"\s+", "_", str(name).strip().lower())
    return re.sub(r"[^\w]", "_", s)


def _skip_at_read(dataset: str) -> "set[str]":
    """Columns excluded at read_csv time (normalised names).

    Exactly the columns the pipeline removes anyway — the per-dataset
    leakage list (dropped by data.cleaner.clean_global) plus the identity
    columns absent from some files (dropped by the schema intersection of
    _align_and_concat).  Skipping them at read time is therefore
    value-neutral, but avoids materialising tens of millions of timestamp
    and address strings that OOM-kill the CIC2018 load on 32 GB hosts.
    """
    from config import LEAKAGE_COLS_2017, LEAKAGE_COLS_2018

    leakage = LEAKAGE_COLS_2017 if dataset == "CIC2017" else LEAKAGE_COLS_2018
    return set(leakage) | {"flow_id", "src_ip", "dst_ip", "src_port"}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, lowercase, replace spaces with underscores."""
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"\s+", "_", regex=True)
                  .str.replace(r"[^\w]", "_", regex=True)
    )
    return df


def _load_cic2017(data_path: Path) -> pd.DataFrame:
    """Load all CIC2017 per-day CSV files and tag each row with day + binary label."""
    skip = _skip_at_read("CIC2017")
    parts: list[pd.DataFrame] = []

    for filename, day in _BENIGN_FILE_TO_DAY.items():
        fp = data_path / filename
        if not fp.exists():
            logger.warning("Benign file not found, skipping: %s", fp)
            continue
        df = pd.read_csv(fp, low_memory=False, usecols=lambda c: _norm_name(c) not in skip)
        df = _normalise_columns(df)
        df = df.drop(columns=["label"], errors="ignore")
        df = df.assign(day=day, label=0, attack_type="BENIGN")
        df = _downcast_features(df)
        parts.append(df)
        logger.debug("Loaded %s: %d rows", filename, len(df))

    for filename, day in _ATTACK_FILE_TO_DAY.items():
        fp = data_path / filename
        if not fp.exists():
            logger.warning("Attack file not found, skipping: %s", fp)
            continue
        df = pd.read_csv(fp, low_memory=False, usecols=lambda c: _norm_name(c) not in skip)
        df = _normalise_columns(df)
        # Binarize labels: benign → 0, everything else → 1 (same logic as CIC2018).
        # Attack files can contain co-mingled BENIGN rows; assigning label=1
        # unconditionally would corrupt training labels.
        if "label" in df.columns:
            raw_labels   = df["label"].astype(str).str.strip()
            binary_label = (~raw_labels.str.lower().isin(["benign"])).astype(int)
            n_benign_in_attack = int(binary_label.eq(0).sum())
            if n_benign_in_attack:
                logger.warning(
                    "%s contains %d BENIGN row(s) — assigning label=0 to those rows",
                    filename, n_benign_in_attack,
                )
            df = df.drop(columns=["label"])
            df = df.assign(day=day, label=binary_label, attack_type=raw_labels)
        else:
            df = df.assign(
                day=day, label=1,
                attack_type=filename.replace(".csv", "").upper(),
            )
        df = _downcast_features(df)
        parts.append(df)
        logger.debug("Loaded %s: %d rows", filename, len(df))

    if not parts:
        raise FileNotFoundError(f"No CSV files found in {data_path}")

    # Align schemas: cast all numeric columns to float32 for consistency
    combined = _align_and_concat(parts)
    logger.info("CIC2017 loaded: %d rows, %d columns", len(combined), combined.shape[1])
    return combined


def _load_cic2018(data_path: Path) -> pd.DataFrame:
    """Load all CIC2018 date-based CSV files and tag each row with session + binary label."""
    skip = _skip_at_read("CIC2018")
    parts: list[pd.DataFrame] = []

    for fp in sorted(data_path.glob("*.csv")):
        session = fp.stem   # e.g. "02-14-2018"
        df = pd.read_csv(fp, low_memory=False, usecols=lambda c: _norm_name(c) not in skip)
        df = _normalise_columns(df)

        if "label" not in df.columns:
            logger.warning("No 'label' column in %s — skipping", fp.name)
            continue

        # Binarise: "Benign" / "benign" -> 0, anything else -> 1
        raw_labels = df["label"].astype(str).str.strip()
        binary_label = (~raw_labels.str.lower().isin(["benign"])).astype(int)
        df = df.drop(columns=["label"])
        df = df.assign(label=binary_label, attack_type=raw_labels, session=session)
        df = _downcast_features(df)
        parts.append(df)
        logger.debug("Loaded %s: %d rows", fp.name, len(df))

    if not parts:
        raise FileNotFoundError(f"No CSV files found in {data_path}")

    combined = _align_and_concat(parts)
    logger.info("CIC2018 loaded: %d rows, %d columns", len(combined), combined.shape[1])
    return combined


def _downcast_features(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce every non-meta column to float32 and ±inf to NaN, per file.

    Identical to the coercion _align_and_concat applies later (to_numeric with
    errors="coerce", then float32) and to the ±inf→NaN replacement
    data.cleaner.clean_global applies afterwards, just performed as each file
    is read: per-file temporaries are ~1M rows, whereas the same operations on
    the concatenated ~16M-row CIC2018 frame OOM-kill the run on 32 GB hosts.
    to_numeric, the float32 cast, and inf→NaN all commute with concatenation,
    so values are unchanged.
    """
    from config import META_COLS

    for col in df.columns:
        if col not in META_COLS:
            s = pd.to_numeric(df[col], errors="coerce").astype("float32")
            df[col] = s.replace([np.inf, -np.inf], np.nan)
    return df


def _align_and_concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate DataFrames with potentially different column sets.

    Feature columns are coerced to float32 via pd.to_numeric (non-numeric
    values become NaN — this safely handles any residual string columns that
    were not listed in META_COLS, such as flow_id).  Meta columns are kept
    as-is. Only columns present in every part are retained.
    """
    from config import META_COLS

    # Intersection of feature columns across all parts (excludes meta cols)
    all_cols = [set(df.columns) - set(META_COLS) for df in parts]
    # Sorted for determinism: set iteration order depends on PYTHONHASHSEED,
    # and column order feeds MI noise injection, XGBoost, and the autoencoder.
    feature_cols: list[str] = sorted(set.intersection(*all_cols))

    # Log any features lost due to schema mismatch across files
    union_cols = set.union(*all_cols)
    lost = union_cols - set(feature_cols)
    if lost:
        logger.warning(
            "Schema intersection dropped %d feature(s) not present in all files: %s",
            len(lost), sorted(lost),
        )

    aligned = []
    for i in range(len(parts)):
        df = parts[i]
        keep = feature_cols + [c for c in META_COLS if c in df.columns]
        sub = df[keep].copy()
        # Release the full-width float64 original immediately; keeping every
        # part alive alongside its float32 copy doubles peak memory and
        # OOM-kills the CIC2018 load on 32 GB hosts.
        parts[i] = None
        del df
        for col in feature_cols:
            # Always coerce: handles both already-numeric columns and any
            # residual string columns (they become NaN, dropped by cleaner).
            sub[col] = pd.to_numeric(sub[col], errors="coerce").astype("float32")
        aligned.append(sub)

    combined = pd.concat(aligned, ignore_index=True)
    aligned.clear()
    return combined


# ── Public API ────────────────────────────────────────────────────────────

def load_dataset(name: str) -> pd.DataFrame:
    """Load the named dataset from raw CSV files.

    Args:
        name: Dataset key from config.DATASETS (e.g. "CIC2017", "CIC2018").

    Returns:
        DataFrame with normalised column names, a temporal identifier column
        ('day' for CIC2017, 'session' for CIC2018), 'label' (0/1 int),
        and 'attack_type' (original label string).

    Raises:
        KeyError: If name is not in config.DATASETS.
        FileNotFoundError: If the dataset directory or files are missing.
    """
    from config import DATASETS

    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list(DATASETS)}")

    cfg = DATASETS[name]
    data_path = Path(cfg["path"])

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")

    dtype = cfg["type"]
    if dtype == "cic2017":
        return _load_cic2017(data_path)
    if dtype == "cic2018":
        return _load_cic2018(data_path)
    raise ValueError(f"Unknown dataset type '{dtype}'")
