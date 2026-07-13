"""Outer experiment loop: datasets × seeds × protocols.

run_experiment(dataset_name, seed) runs one full trial.
run_all() iterates over all datasets and seeds, saving each result
incrementally so a crash does not lose previously completed runs.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _set_global_seed(seed: int) -> None:
    """Set numpy, random, and torch seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark     = False
    except ImportError:
        pass


def _dataset_sanity_audit(df, dataset_name: str) -> Dict[str, Any]:
    """Collect per-day/session row counts and class distribution for audit log."""
    from config import DATASETS

    col = "day" if DATASETS[dataset_name]["type"] == "cic2017" else "session"
    audit: Dict[str, Any] = {
        "total_rows":    int(len(df)),
        "n_benign":      int((df["label"] == 0).sum()),
        "n_attack":      int((df["label"] == 1).sum()),
        "attack_ratio":  float(df["label"].mean()),
        "temporal_col":  col,
        "counts_by_period": {},
    }
    for period, group in df.groupby(col, sort=True):
        audit["counts_by_period"][str(period)] = {
            "n_rows":   int(len(group)),
            "n_benign": int((group["label"] == 0).sum()),
            "n_attack": int((group["label"] == 1).sum()),
        }
    logger.info(
        "Dataset audit [%s]: %d rows | benign=%d attack=%d (%.1f%%)",
        dataset_name,
        audit["total_rows"],
        audit["n_benign"],
        audit["n_attack"],
        audit["attack_ratio"] * 100,
    )
    return audit


def _split_audit(split_name: str, train_df, val_df, test_df) -> Dict[str, Any]:
    """Collect row counts and class distributions for each split."""
    def _stats(df) -> Dict:
        return {
            "n_rows":   int(len(df)),
            "n_benign": int((df["label"] == 0).sum()),
            "n_attack": int((df["label"] == 1).sum()),
            "attack_ratio": float(df["label"].mean()) if len(df) > 0 else 0.0,
        }
    return {
        "protocol": split_name,
        "train": _stats(train_df),
        "val":   _stats(val_df),
        "test":  _stats(test_df),
    }


def run_experiment(
    dataset_name: str,
    seed:         int,
    skip_drift:   bool = False,
) -> Dict[str, Any]:
    """Run one complete experiment for a given dataset and seed.

    Steps:
        1.  Load and clean the raw dataset.
        1b. Dataset sanity audit (rows, class counts by period).
        2.  Construct random split (Protocol 1) and temporal split (Protocol 2).
        2b. Split audit (verify temporal protocol has no overlap).
        3.  Preprocess each split independently (fit only on training data).
        4.  Train XGBoost on each protocol's training split.
        5.  Evaluate on each protocol's test split.
        6.  (Unless skip_drift) Train autoencoder on temporal training data,
            encode temporal train and test, compute drift metrics.
        7.  Return all results in a structured dict.

    Args:
        dataset_name: "CIC2017" or "CIC2018".
        seed:         Random seed used for all stochastic operations.
        skip_drift:   If True, skip the autoencoder and drift computation.

    Returns:
        Nested dict with keys:
            dataset, seed,
            random:   {metrics, n_train, n_val, n_test, preprocessing_audit, split_audit},
            temporal: {metrics, n_train, n_val, n_test, preprocessing_audit, split_audit},
            drift:    dict from drift.metrics.summarise_drift (or None),
            dataset_audit: dict with per-period row/class counts,
            runtime_s: float.
    """
    from config import MODEL_PARAMS

    from data.cleaner import clean_global
    from data.loader import load_dataset
    from data.splitter import get_temporal_split, random_split
    from drift.autoencoder import encode, train_autoencoder
    from drift.metrics import summarise_drift
    from models.classifier import (
        compute_proba_percentiles,
        evaluate_classifier,
        find_optimal_threshold,
        train_classifier,
    )
    from preprocessing.pipeline import fit_transform_splits

    t0 = time.perf_counter()
    _set_global_seed(seed)
    logger.info("=" * 60)
    logger.info("START  dataset=%s  seed=%d", dataset_name, seed)

    # ── 1. Load and global-clean ───────────────────────────────────────────
    df = load_dataset(dataset_name)
    df = clean_global(df, dataset_name)

    # ── 1b. Dataset sanity audit ──────────────────────────────────────────
    ds_audit = _dataset_sanity_audit(df, dataset_name)

    # ── 2. Splits ─────────────────────────────────────────────────────────
    rand_train, rand_val, rand_test   = random_split(df, seed=seed)
    temp_train, temp_val, temp_test   = get_temporal_split(df, dataset_name)

    # ── 2b. Split audits ──────────────────────────────────────────────────
    rand_split_audit = _split_audit("random",   rand_train, rand_val, rand_test)
    temp_split_audit = _split_audit("temporal", temp_train, temp_val, temp_test)

    # ── 3. Preprocess (seed-controlled, fit-on-train invariant) ──────────
    X_tr_r, y_tr_r, X_val_r, y_val_r, X_te_r, y_te_r, _, pp_audit_r = (
        fit_transform_splits(rand_train, rand_val, rand_test, seed=seed)
    )
    X_tr_t, y_tr_t, X_val_t, y_val_t, X_te_t, y_te_t, _, pp_audit_t = (
        fit_transform_splits(temp_train, temp_val, temp_test, seed=seed)
    )

    # ── 4. Train classifiers ──────────────────────────────────────────────
    model_r = train_classifier(X_tr_r, y_tr_r, X_val_r, y_val_r, MODEL_PARAMS, seed)
    model_t = train_classifier(X_tr_t, y_tr_t, X_val_t, y_val_t, MODEL_PARAMS, seed)

    # ── 5. Evaluate — default threshold (0.5) and val-tuned threshold ────────
    metrics_r = evaluate_classifier(model_r, X_te_r, y_te_r, threshold=0.5)
    metrics_t = evaluate_classifier(model_t, X_te_t, y_te_t, threshold=0.5)

    threshold_r, val_score_r = find_optimal_threshold(model_r, X_val_r, y_val_r)
    threshold_t, val_score_t = find_optimal_threshold(model_t, X_val_t, y_val_t)
    metrics_r_tuned = evaluate_classifier(model_r, X_te_r, y_te_r, threshold=threshold_r)
    metrics_t_tuned = evaluate_classifier(model_t, X_te_t, y_te_t, threshold=threshold_t)

    # Probability percentiles for train and val splits (test percentiles are
    # already embedded in metrics via evaluate_classifier → compute_proba_percentiles)
    proba_splits_r = {
        "train": compute_proba_percentiles(model_r, X_tr_r, y_tr_r),
        "val":   compute_proba_percentiles(model_r, X_val_r, y_val_r),
        "test":  metrics_r["proba_percentiles"],
    }
    proba_splits_t = {
        "train": compute_proba_percentiles(model_t, X_tr_t, y_tr_t),
        "val":   compute_proba_percentiles(model_t, X_val_t, y_val_t),
        "test":  metrics_t["proba_percentiles"],
    }

    # ── 6. Drift (temporal train → temporal test) ─────────────────────────
    drift_summary: Optional[Dict] = None
    ae_audit: Optional[Dict] = None
    if not skip_drift:
        try:
            ae, ae_audit = train_autoencoder(X_tr_t, seed=seed)
            Z_train = encode(ae, X_tr_t)
            Z_test  = encode(ae, X_te_t)
            drift_summary = summarise_drift(Z_train, Z_test)
        except Exception as exc:
            logger.error(
                "Drift computation failed for %s seed=%d: %s",
                dataset_name, seed, exc,
            )
            drift_summary = {"error": str(exc)}

    runtime = time.perf_counter() - t0
    logger.info("END    dataset=%s  seed=%d  runtime=%.1fs", dataset_name, seed, runtime)

    return {
        "dataset": dataset_name,
        "seed":    seed,
        "random": {
            "metrics":             metrics_r,
            "metrics_tuned":       metrics_r_tuned,
            "threshold_default":   0.5,
            "threshold_tuned":     threshold_r,
            "threshold_val_score": val_score_r,
            "proba_splits":        proba_splits_r,
            "n_train":             int(len(y_tr_r)),
            "n_val":               int(len(y_val_r)),
            "n_test":              int(len(y_te_r)),
            "preprocessing_audit": pp_audit_r,
            "split_audit":         rand_split_audit,
        },
        "temporal": {
            "metrics":             metrics_t,
            "metrics_tuned":       metrics_t_tuned,
            "threshold_default":   0.5,
            "threshold_tuned":     threshold_t,
            "threshold_val_score": val_score_t,
            "proba_splits":        proba_splits_t,
            "n_train":             int(len(y_tr_t)),
            "n_val":               int(len(y_val_t)),
            "n_test":              int(len(y_te_t)),
            "preprocessing_audit": pp_audit_t,
            "split_audit":         temp_split_audit,
        },
        "drift":         drift_summary,
        "ae_audit":      ae_audit,
        "dataset_audit": ds_audit,
        "runtime_s":     round(runtime, 2),
    }


def run_all(
    skip_drift:     bool = False,
    dataset_filter: Optional[List[str]] = None,
    seed_filter:    Optional[List[int]]  = None,
) -> List[Dict[str, Any]]:
    """Outer loop over all datasets and seeds.

    Results are saved incrementally after each completed experiment.
    If a result file already exists for a (dataset, seed) pair, that run
    is skipped (resumable behaviour).

    Args:
        skip_drift:      If True, skip autoencoder + drift for all runs.
        dataset_filter:  If set, only run these datasets (subset of config.DATASETS).
        seed_filter:     If set, only run these seeds (subset of config.SEEDS).

    Returns:
        List of all result dicts (both new and previously cached).
    """
    from config import DATASETS, OUTPUT_DIR, SEEDS

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = dataset_filter or list(DATASETS.keys())
    seeds    = seed_filter    or SEEDS

    all_results: List[Dict[str, Any]] = []

    for dataset_name in datasets:
        for seed in seeds:
            result_path = output_dir / f"results_{dataset_name}_{seed}.json"

            # Resume: skip if already done
            if result_path.exists():
                with open(result_path) as f:
                    cached = json.load(f)
                drift_err = (cached.get("drift") or {}).get("error")
                if drift_err:
                    logger.warning(
                        "Cached result for %s seed=%d has a drift error: %s. "
                        "Delete %s to re-run drift.",
                        dataset_name, seed, drift_err, result_path,
                    )
                else:
                    logger.info(
                        "Skipping %s seed=%d (cached at %s)", dataset_name, seed, result_path
                    )
                all_results.append(cached)
                continue

            try:
                result = run_experiment(dataset_name, seed, skip_drift=skip_drift)
                _save_result(result, result_path)
                all_results.append(result)
            except Exception as exc:
                logger.error(
                    "Experiment FAILED: dataset=%s seed=%d — %s",
                    dataset_name, seed, exc, exc_info=True,
                )
                failure = {
                    "dataset": dataset_name,
                    "seed":    seed,
                    "error":   str(exc),
                }
                _save_result(failure, result_path.with_suffix(".error.json"))

    return all_results


def _save_result(result: Dict[str, Any], path: Path) -> None:
    """Serialise result dict to JSON, converting numpy types to Python natives."""
    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(path, "w") as f:
        json.dump(_convert(result), f, indent=2)
    logger.debug("Saved result to %s", path)
