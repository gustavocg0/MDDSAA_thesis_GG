"""Statistical comparison between random and temporal protocol results.

Tests the hypothesis that random splitting inflates reported IDS performance
relative to the temporally-correct evaluation condition.

Pair definition (FROZEN): (random_seed_i, temporal_seed_i)
  — same seed, same model, same preprocessing, only the split method differs.

Primary test: one-sided Wilcoxon signed-rank (H1: random > temporal).
Effect size: bootstrap CI on the mean delta (random − temporal).
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

_METRIC_KEYS = ["macro_f1", "pr_auc", "roc_auc", "balanced_accuracy"]


def compute_deltas(
    random_metrics_list:   List[Dict],
    temporal_metrics_list: List[Dict],
) -> Dict[str, np.ndarray]:
    """Compute per-seed delta = random_i − temporal_i for each metric.

    The two lists must be aligned: index i corresponds to the same seed.

    Args:
        random_metrics_list:   List of metric dicts, one per seed, from random protocol.
        temporal_metrics_list: List of metric dicts, one per seed, from temporal protocol.

    Returns:
        Dict mapping metric name → np.ndarray of shape (n_seeds,) with deltas.

    Raises:
        ValueError: If the two lists have different lengths.
    """
    if len(random_metrics_list) != len(temporal_metrics_list):
        raise ValueError(
            f"Metric lists must have the same length; got "
            f"{len(random_metrics_list)} vs {len(temporal_metrics_list)}"
        )

    deltas: Dict[str, np.ndarray] = {}
    for metric in _METRIC_KEYS:
        # None values (undefined metrics, e.g. roc_auc with single test class) → NaN
        r_vals = np.array(
            [m[metric] if m[metric] is not None else float("nan") for m in random_metrics_list],
            dtype=float,
        )
        t_vals = np.array(
            [m[metric] if m[metric] is not None else float("nan") for m in temporal_metrics_list],
            dtype=float,
        )
        deltas[metric] = r_vals - t_vals

    return deltas


def wilcoxon_test(
    deltas: np.ndarray,
    alpha:  float = 0.05,
) -> Tuple[float, float]:
    """One-sided Wilcoxon signed-rank test on paired deltas.

    H0: median(delta) = 0
    H1: median(delta) > 0  (random protocol inflates performance)

    With n=5 pairs and no zero differences or ties, scipy's default
    method='auto' selects the exact signed-rank null distribution, whose
    one-sided floor is p = 1/2**5 = 0.03125.

    Args:
        deltas: Per-seed delta array (random_i − temporal_i), shape (n_seeds,).
        alpha:  Significance level (default config.ALPHA).

    Returns:
        (statistic, p_value)
    """
    # Drop NaN values (arise when a metric is undefined for some seeds, e.g.
    # roc_auc on a single-class test split).  Running scipy on NaN-containing
    # arrays raises ValueError in scipy >= 1.7 or silently corrupts the result.
    valid     = deltas[~np.isnan(deltas)]
    n_dropped = len(deltas) - len(valid)
    if n_dropped:
        logger.warning(
            "wilcoxon_test: %d NaN delta(s) dropped (metric undefined for some "
            "seeds). Test runs on %d of %d pairs.",
            n_dropped, len(valid), len(deltas),
        )
    if len(valid) < 3:
        logger.warning(
            "Wilcoxon test requires n>=3 valid pairs; got %d. Returning NaN.",
            len(valid),
        )
        return float("nan"), float("nan")

    stat, p_value = stats.wilcoxon(valid, alternative="greater")
    significant = p_value < alpha
    logger.debug(
        "Wilcoxon: stat=%.4f | p=%.4f | significant=%s (alpha=%.3f)",
        stat, p_value, significant, alpha,
    )
    return float(stat), float(p_value)


def bootstrap_ci(
    deltas:      np.ndarray,
    n_bootstrap: int   = 10_000,
    alpha:       float = 0.05,
    seed:        int   = 42,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI on the mean delta.

    Resamples delta values with replacement n_bootstrap times and returns
    the (alpha/2, 1 − alpha/2) percentile bounds.

    Args:
        deltas:      Per-seed delta array, shape (n_seeds,).
        n_bootstrap: Number of bootstrap iterations (default 10,000).
        alpha:       CI level: returns (1 − alpha) × 100 % CI (default 0.05 → 95 %).
        seed:        RNG seed for reproducibility.

    Returns:
        (ci_lower, ci_upper, mean_delta)
    """
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(deltas, size=len(deltas), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    mean_delta = float(deltas.mean())
    return ci_lower, ci_upper, mean_delta


def summarise_statistical_tests(
    random_metrics_list:   List[Dict],
    temporal_metrics_list: List[Dict],
) -> pd.DataFrame:
    """Run all statistical tests and return a summary DataFrame.

    One row per metric. Columns:
        metric, mean_delta, std_delta, ci_lower, ci_upper,
        wilcoxon_stat, p_value, significant.

    Args:
        random_metrics_list:   List of metric dicts (one per seed) from random protocol.
        temporal_metrics_list: List of metric dicts (one per seed) from temporal protocol.

    Returns:
        pd.DataFrame with one row per metric.
    """
    from config import ALPHA

    deltas_by_metric = compute_deltas(random_metrics_list, temporal_metrics_list)
    rows = []

    for metric, deltas in deltas_by_metric.items():
        stat, p_val       = wilcoxon_test(deltas, alpha=ALPHA)
        ci_lower, ci_upper, mean_delta = bootstrap_ci(deltas, alpha=ALPHA)

        test_skipped = bool(np.isnan(p_val))
        rows.append({
            "metric":        metric,
            "mean_delta":    mean_delta,
            "std_delta":     float(deltas.std()),
            "ci_lower":      ci_lower,
            "ci_upper":      ci_upper,
            "wilcoxon_stat": stat,
            "p_value":       p_val,
            "significant":   bool(p_val < ALPHA) if not test_skipped else False,
            # True when n < 3 or all NaN — distinguishes "not significant" from
            # "test not run".  Downstream tables should propagate this flag.
            "test_skipped":  test_skipped,
        })

    df = pd.DataFrame(rows).set_index("metric")
    logger.info(
        "Statistical summary:\n%s",
        df[["mean_delta", "p_value", "significant"]].to_string(),
    )
    return df
