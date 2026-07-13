"""Construct train/val/test splits for both evaluation protocols.

Protocol 1 — random stratified split: ignores temporal structure.
Protocol 2 — temporal holdout: uses the 'day' or 'session' column.

INVARIANT: Leakage columns must be removed (via data.cleaner.clean)
BEFORE calling any function here.
"""

import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

SplitTriple = Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]

logger = logging.getLogger(__name__)


def random_split(df: pd.DataFrame, seed: int) -> SplitTriple:
    """Stratified random split, ignoring temporal ordering.

    Split ratios are taken from config.RANDOM_SPLIT (train=0.7, val=0.1, test=0.2).
    Stratification is on the binary 'label' column.

    Args:
        df:   Cleaned DataFrame with a 'label' column.
        seed: Random seed for reproducibility.

    Returns:
        (train_df, val_df, test_df)
    """
    from config import RANDOM_SPLIT

    train_frac = RANDOM_SPLIT["train"]
    val_frac   = RANDOM_SPLIT["val"]
    test_frac  = RANDOM_SPLIT["test"]

    # First split: train+val vs test
    trainval_df, test_df = train_test_split(
        df,
        test_size=test_frac,
        stratify=df["label"],
        random_state=seed,
    )

    # Second split: train vs val (val_frac relative to train+val)
    relative_val = val_frac / (train_frac + val_frac)
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=relative_val,
        stratify=trainval_df["label"],
        random_state=seed,
    )

    _log_split("random", train_df, val_df, test_df)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def temporal_split_2017(df: pd.DataFrame) -> SplitTriple:
    """Temporal holdout split for CIC2017 using the 'day' column.

    Partitioning (determined by the dataset's natural temporal structure,
    not optimised for favourable results):
        Training:   Monday + Tuesday   (stable benign baseline + SSH brute force)
        Validation: Wednesday          (DoS variants; used for hyperparameter tuning only)
        Test:       Thursday + Friday  (Web Attacks, Botnet, PortScan, DDoS — unseen categories)

    This is the only partition that provides three non-overlapping windows that
    are (a) large enough for stable training, (b) distinct in attack distribution,
    and (c) strictly causally ordered.

    Args:
        df: Cleaned CIC2017 DataFrame with a 'day' column.

    Returns:
        (train_df, val_df, test_df)

    Raises:
        ValueError: If any required day is missing from the DataFrame.
    """
    from config import TEMPORAL_SPLIT_2017

    _validate_day_col(df, "day", TEMPORAL_SPLIT_2017)

    train_df = df[df["day"].isin(TEMPORAL_SPLIT_2017["train"])].copy()
    val_df   = df[df["day"].isin(TEMPORAL_SPLIT_2017["val"])].copy()
    test_df  = df[df["day"].isin(TEMPORAL_SPLIT_2017["test"])].copy()

    _assert_no_temporal_overlap(train_df, val_df, test_df, "day")
    _log_split("temporal-2017", train_df, val_df, test_df)
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def temporal_split_2018(df: pd.DataFrame) -> SplitTriple:
    """Temporal holdout split for CIC2018 using the 'session' column.

    Cutoff logic:
        The dataset contains 10 date files in chronological order:
        02-14, 02-15, 02-16, 02-20, 02-21, 02-22, 02-23, 02-28, 03-01, 03-02.

        Split: first 6 files → training (60 %), next 2 → validation (20 %),
               last 2 → test (20 %).  This preserves temporal ordering so that
               the model never sees future traffic during training or tuning.

    Args:
        df: Cleaned CIC2018 DataFrame with a 'session' column.

    Returns:
        (train_df, val_df, test_df)

    Raises:
        ValueError: If the 'session' column is missing or required sessions absent.
    """
    from config import TEMPORAL_SPLIT_2018

    _validate_day_col(df, "session", TEMPORAL_SPLIT_2018)

    train_df = df[df["session"].isin(TEMPORAL_SPLIT_2018["train"])].copy()
    val_df   = df[df["session"].isin(TEMPORAL_SPLIT_2018["val"])].copy()
    test_df  = df[df["session"].isin(TEMPORAL_SPLIT_2018["test"])].copy()

    _assert_no_temporal_overlap(train_df, val_df, test_df, "session")
    _log_split("temporal-2018", train_df, val_df, test_df)
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def get_temporal_split(df: pd.DataFrame, dataset_name: str) -> SplitTriple:
    """Dispatch to the correct temporal split function for the given dataset.

    Args:
        df:           Cleaned DataFrame.
        dataset_name: "CIC2017" or "CIC2018".

    Returns:
        (train_df, val_df, test_df)
    """
    if dataset_name == "CIC2017":
        return temporal_split_2017(df)
    if dataset_name == "CIC2018":
        return temporal_split_2018(df)
    raise ValueError(f"No temporal split defined for dataset '{dataset_name}'")


# ── Private helpers ───────────────────────────────────────────────────────

def _validate_day_col(df: pd.DataFrame, col: str, split_cfg: dict) -> None:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    required = set(split_cfg["train"]) | set(split_cfg["val"]) | set(split_cfg["test"])
    present  = set(df[col].unique())
    missing  = required - present
    if missing:
        raise ValueError(
            f"Required {col} values absent from data: {sorted(missing)}. "
            f"Present: {sorted(present)}"
        )
    # Rows whose period is not assigned to any split are silently discarded by
    # the .isin() filters in the callers.  Warn so the omission is visible.
    extra = present - required
    if extra:
        n_extra = int(df[col].isin(extra).sum())
        logger.warning(
            "%d row(s) with unrecognised %s value(s) %s will be discarded "
            "(not assigned to train, val, or test).",
            n_extra, col, sorted(extra),
        )


def _assert_no_temporal_overlap(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    col:      str,
) -> None:
    """Assert no shared day/session values between any pair of splits."""
    train_vals = set(train_df[col].unique())
    val_vals   = set(val_df[col].unique())
    test_vals  = set(test_df[col].unique())
    tv = train_vals & val_vals
    tt = train_vals & test_vals
    vt = val_vals   & test_vals
    if tv or tt or vt:
        raise AssertionError(
            f"Temporal overlap in '{col}': "
            f"train∩val={tv}, train∩test={tt}, val∩test={vt}"
        )


def _log_split(name: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    def _pct(sub: pd.DataFrame) -> str:
        return f"{sub['label'].mean() * 100:.1f}% attack"
    logger.info(
        "Split [%s]: train=%d (%s) | val=%d (%s) | test=%d (%s)",
        name,
        len(train), _pct(train),
        len(val),   _pct(val),
        len(test),  _pct(test),
    )
