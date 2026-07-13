"""Global (non-learned) cleaning: leakage removal, inf replacement, deduplication.

RESPONSIBILITY SPLIT
────────────────────
clean_global()  ← this module
    Fixed, deterministic operations that do NOT depend on any data statistics.
    Safe to run on the full dataset before splitting.
    • Drop fixed leakage/identity columns (known at design time)
    • Replace ±inf with NaN
    • Drop exact duplicate rows
    • Validate binary label

preprocessing.pipeline.fit_transform_splits()  ← NOT here
    Data-dependent decisions fitted on the TRAINING split only.
    • Drop high-missing columns  (threshold computed on train)
    • Impute missing values       (median fitted on train)
    • Scale features              (RobustScaler fitted on train)
    • Feature selection           (mutual info fitted on train)

Mixing these two responsibilities was the original leakage risk:
dropping high-missing columns over the full dataset before splitting
means the test set's missingness pattern influences which columns survive.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_global(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Apply fixed, non-learned cleaning to the full dataset.

    Every operation here is deterministic and dataset-statistics-free.
    No learned threshold (e.g. missing rate) is applied.

    Steps (in order):
        1. Drop fixed leakage/identity columns (hardcoded per dataset).
        2. Replace ±inf with NaN in feature columns.
        3. Drop exact duplicate rows.
        4. Validate that 'label' is binary (0/1 integer).

    Args:
        df:      Raw DataFrame from data.loader.load_dataset.
        dataset: Dataset key ("CIC2017" or "CIC2018") — selects leakage list.

    Returns:
        Cleaned DataFrame. The input frame is consumed (mutated in place):
        an upfront defensive copy of the ~16M-row CIC2018 frame, on top of
        the caller's still-live reference, OOM-kills the run on 32 GB hosts.
        The only caller (experiment.runner) immediately rebinds the result.

    Raises:
        ValueError: If 'label' column is missing or contains non-binary values.
        ValueError: If dataset is not recognised.
    """
    from config import LEAKAGE_COLS_2017, LEAKAGE_COLS_2018, META_COLS

    n_initial = len(df)

    # ── 1. Drop fixed leakage columns ────────────────────────────────────
    if dataset == "CIC2017":
        leakage_cols = LEAKAGE_COLS_2017
    elif dataset == "CIC2018":
        leakage_cols = LEAKAGE_COLS_2018
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected 'CIC2017' or 'CIC2018'.")

    to_drop = [c for c in leakage_cols if c in df.columns]
    if to_drop:
        df.drop(columns=to_drop, inplace=True)
        logger.debug("Dropped %d leakage columns: %s", len(to_drop), to_drop)

    # ── 2. Replace ±inf with NaN ──────────────────────────────────────────
    # The loader already maps ±inf to NaN per file, so this pass is normally
    # a no-op; the isinf guard keeps it from copying blocks of the ~16M-row
    # CIC2018 frame (under copy-on-write, every column assignment copies)
    # while preserving the invariant for any loader path that skips it.
    feature_cols = [c for c in df.columns if c not in META_COLS]
    numeric_feature_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
    for col in numeric_feature_cols:
        if np.isinf(df[col].to_numpy()).any():
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    # ── 3. Drop exact duplicate rows ──────────────────────────────────────
    n_before = len(df)
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.debug("Dropped %d duplicate rows", n_dropped)

    # ── 4. Validate binary label ──────────────────────────────────────────
    if "label" not in df.columns:
        raise ValueError("'label' column missing after cleaning.")
    unique_labels = set(df["label"].unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"'label' must contain only 0 and 1 (binary), found: {unique_labels}"
        )

    n_features = len([c for c in df.columns if c not in META_COLS])
    logger.info(
        "clean_global [%s]: %d → %d rows | %d features | %.1f%% attack",
        dataset,
        n_initial,
        len(df),
        n_features,
        df["label"].mean() * 100,
    )
    return df
