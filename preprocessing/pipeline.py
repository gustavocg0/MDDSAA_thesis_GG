"""Preprocessing pipeline: train-only column filtering → imputation → scaling → feature selection → clipping.

LEAKAGE INVARIANT
─────────────────
Every data-dependent decision in this module is computed from the TRAINING
split only and then applied identically to validation and test splits.

Step 1 — _drop_high_missing_train_only()
    Computes the missing-value fraction of each feature column using the
    TRAINING data. Columns above MISSING_THRESHOLD are dropped from ALL three
    splits using the same column list derived from training. This prevents the
    test set's missingness pattern from influencing which columns survive.

Step 2 — sklearn Pipeline (imputer → scaler → selector → clipper)
    Fitted exclusively on X_train, then transform() is called on X_val and
    X_test without refitting. SelectKBest uses mutual information with y_train.
    ClipTransformer clips output to [-FEATURE_CLIP_VALUE, +FEATURE_CLIP_VALUE]:
    network flow features after RobustScaler can have extreme outliers (IQR ≈ 0
    but absolute values in the millions); clipping prevents the autoencoder
    from being dominated by a tiny fraction of extreme values.
"""

import logging
from functools import partial
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


# ── ClipTransformer ───────────────────────────────────────────────────────

class ClipTransformer(BaseEstimator, TransformerMixin):
    """Clip feature values to a fixed symmetric range after scaling."""

    def __init__(self, clip_value: float = 10.0):
        self.clip_value = clip_value

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(X, -self.clip_value, self.clip_value)


# ── Internal helpers ──────────────────────────────────────────────────────

def _drop_high_missing_train_only(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop high-missing feature columns using the TRAINING split only.

    The missing-value fraction is computed on train_df. Any feature column
    whose fraction exceeds MISSING_THRESHOLD is removed from all three splits
    using that same training-derived list.

    Args:
        train_df:     Training split.
        val_df:       Validation split.
        test_df:      Test split.
        feature_cols: Candidate feature column names (already excludes META_COLS).

    Returns:
        (train_df, val_df, test_df, kept_feature_cols) — DataFrames with
        high-missing columns removed; kept_feature_cols is the surviving list.
    """
    from config import MISSING_THRESHOLD

    missing_frac = train_df[feature_cols].isnull().mean()
    high_missing = missing_frac[missing_frac > MISSING_THRESHOLD].index.tolist()

    if high_missing:
        logger.debug(
            "Dropping %d high-missing cols (train fraction > %.2f): %s",
            len(high_missing), MISSING_THRESHOLD, high_missing,
        )
        train_df = train_df.drop(columns=high_missing)
        val_df   = val_df.drop(columns=high_missing, errors="ignore")
        test_df  = test_df.drop(columns=high_missing, errors="ignore")

    kept = [c for c in feature_cols if c not in high_missing]
    return train_df, val_df, test_df, kept


# ── Public API ────────────────────────────────────────────────────────────

def build_pipeline(seed: int = 0) -> Pipeline:
    """Return an unfitted sklearn Pipeline.

    Stages:
        1. SimpleImputer(strategy='median')   — handles NaN left after inf replacement.
        2. RobustScaler()                     — median/IQR normalisation, robust to
                                               the heavy-tailed flow features in CIC data.
        3. SelectKBest(mutual_info_classif)   — supervised feature selection;
           fitted on (X_train, y_train) only; k = config.N_FEATURES_SELECT.
           random_state=seed ensures deterministic MI estimation.

    Args:
        seed: Random state for mutual_info_classif.

    Returns:
        Unfitted sklearn Pipeline.
    """
    from config import CLIP_VALUE, N_FEATURES_SELECT

    mi = partial(mutual_info_classif, random_state=seed)
    return Pipeline([
        ("imputer",  SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler",   RobustScaler()),
        ("selector", SelectKBest(score_func=mi, k=N_FEATURES_SELECT)),
        ("clipper",  ClipTransformer(clip_value=CLIP_VALUE)),
    ])


def fit_transform_splits(
    train_df:   pd.DataFrame,
    val_df:     pd.DataFrame,
    test_df:    pd.DataFrame,
    target_col: str = "label",
    seed:       int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pipeline, Dict[str, Any]]:
    """Fit all preprocessing steps on train_df ONLY, then transform all splits.

    Execution order:
        1. Identify feature columns (all non-META_COLS).
        2. Drop high-missing columns using the FULL training split
           (calls _drop_high_missing_train_only — never the subsampled version).
        3. Optionally subsample for pipeline fit (stratified, seed-controlled).
        4. Fit sklearn Pipeline on (subsampled) X_train; transform all splits.

    Args:
        train_df:   Training split DataFrame (output of splitter).
        val_df:     Validation split DataFrame.
        test_df:    Test split DataFrame.
        target_col: Binary label column name (default "label").
        seed:       Controls MI random state and stratified subsampling seed.

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test, fitted_pipeline, audit)
        X arrays: float32 numpy arrays of shape (n_samples, N_FEATURES_SELECT).
        y arrays: int numpy arrays of shape (n_samples,).
        audit:    Dict with preprocessing metadata for result logging.

    Raises:
        ValueError: If target_col is missing from any split.
    """
    from config import META_COLS, PIPELINE_FIT_MAX_SAMPLES

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if target_col not in df.columns:
            raise ValueError(f"'{target_col}' column missing from {name} split.")

    # Step 1 — identify candidates (all non-meta columns)
    feature_cols = [c for c in train_df.columns if c not in META_COLS]
    n_raw = len(feature_cols)

    # Step 2 — drop high-missing columns using the FULL training split
    # (decision is made before any subsampling so it uses all train rows)
    train_df, val_df, test_df, feature_cols = _drop_high_missing_train_only(
        train_df, val_df, test_df, feature_cols
    )

    # Step 3 — extract arrays
    X_tr_raw  = train_df[feature_cols].values
    y_train   = train_df[target_col].values.astype(int)

    X_val_raw = val_df[feature_cols].values
    y_val     = val_df[target_col].values.astype(int)

    X_te_raw  = test_df[feature_cols].values
    y_test    = test_df[target_col].values.astype(int)

    # Step 4 — fit pipeline (subsample if training set exceeds cap)
    pipe = build_pipeline(seed=seed)
    n_all = len(y_train)
    if n_all > PIPELINE_FIT_MAX_SAMPLES:
        from sklearn.model_selection import train_test_split as _tts
        X_fit, _, y_fit, _ = _tts(
            X_tr_raw, y_train,
            train_size=PIPELINE_FIT_MAX_SAMPLES,
            stratify=y_train,
            random_state=seed,
        )
        logger.info(
            "Pipeline fit subsampled: %d → %d rows (stratified, seed=%d)",
            n_all, PIPELINE_FIT_MAX_SAMPLES, seed,
        )
        pipe.fit(X_fit, y_fit)
        fit_subsampled = True
        fit_n = PIPELINE_FIT_MAX_SAMPLES
    else:
        pipe.fit(X_tr_raw, y_train)
        fit_subsampled = False
        fit_n = n_all

    X_train = pipe.transform(X_tr_raw).astype("float32")
    X_val   = pipe.transform(X_val_raw).astype("float32")
    X_test  = pipe.transform(X_te_raw).astype("float32")

    logger.info(
        "Post-preprocessing range: train min=%.3f max=%.3f mean_abs=%.3f",
        float(np.min(X_train)),
        float(np.max(X_train)),
        float(np.mean(np.abs(X_train))),
    )

    selected  = get_selected_feature_names(pipe, feature_cols)
    clip_val  = float(pipe.named_steps["clipper"].clip_value)
    abs_train = np.abs(X_train)

    audit: Dict[str, Any] = {
        "n_features_raw":         n_raw,
        "n_features_kept":        len(feature_cols),
        "n_features_selected":    int(X_train.shape[1]),
        "selected_features":      selected,
        "fit_n_samples":          fit_n,
        "fit_subsampled":         fit_subsampled,
        "clip_value":             clip_val,
        "post_clip_max_abs":      float(abs_train.max()),
        "post_clip_p99":          float(np.percentile(abs_train, 99)),
        "post_clip_frac_at_clip": float((abs_train >= clip_val).mean()),
    }

    logger.info(
        "Preprocessing: %d raw → %d kept → %d selected | "
        "train=%d val=%d test=%d | fit_on=%d%s",
        n_raw, len(feature_cols), X_train.shape[1],
        len(y_train), len(y_val), len(y_test),
        fit_n, " (subsampled)" if fit_subsampled else "",
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, pipe, audit


def get_selected_feature_names(
    pipe:         Pipeline,
    feature_cols: list[str],
) -> list[str]:
    """Return names of the features surviving SelectKBest.

    Args:
        pipe:         Fitted pipeline from fit_transform_splits.
        feature_cols: Column list passed into the pipeline (post-high-missing
                      filter). Obtain from the 'feature_cols' local variable
                      inside fit_transform_splits, or re-derive as
                      [c for c in train_df.columns if c not in META_COLS].

    Returns:
        List of selected feature names in pipeline output order.
    """
    mask = pipe.named_steps["selector"].get_support()
    return [c for c, sel in zip(feature_cols, mask) if sel]
