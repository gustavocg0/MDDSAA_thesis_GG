"""Aggregate experiment results into analysis-ready DataFrames.

All functions accept the list of result dicts returned by experiment.runner.run_all.

Table inventory
───────────────
build_dataset_summary_table  — one row per dataset (rows, class counts, features)
build_split_table            — one row per (dataset, protocol, split)
build_performance_table      — one row per (dataset, seed, protocol) with all metrics
build_results_table          — alias of build_performance_table (backward compat)
build_delta_table            — C1: inflation gap with Wilcoxon p-value per metric
build_drift_table            — C2: PSI/KS drift summary per seed
build_hypothesis_table       — one row per contribution (C1/C2) with verdict
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ── Dataset-level summary ─────────────────────────────────────────────────

def build_dataset_summary_table(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """One row per dataset: total rows after cleaning, class distribution, features.

    Uses dataset_audit (populated before splitting) and the temporal protocol's
    preprocessing_audit for feature counts (both protocols select the same k).

    Columns:
        dataset, total_rows, n_benign, n_attack, attack_pct,
        n_features_raw, n_features_selected.
    """
    rows = []
    seen: set = set()
    for r in results_list:
        if "error" in r or r["dataset"] in seen:
            continue
        seen.add(r["dataset"])
        ds  = r.get("dataset_audit", {})
        pp  = r.get("temporal", {}).get("preprocessing_audit", {})
        rows.append({
            "dataset":             r["dataset"],
            "total_rows":          ds.get("total_rows"),
            "n_benign":            ds.get("n_benign"),
            "n_attack":            ds.get("n_attack"),
            "attack_pct":          round(ds.get("attack_ratio", 0) * 100, 1),
            "n_features_raw":      pp.get("n_features_raw"),
            "n_features_selected": pp.get("n_features_selected"),
        })
    df = pd.DataFrame(rows)
    logger.info("Dataset summary table: %d rows", len(df))
    return df


# ── Split-level summary ───────────────────────────────────────────────────

def build_split_table(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """One row per (dataset, protocol, split): rows and class distribution.

    For the temporal protocol the split is deterministic (same across seeds).
    For the random protocol the first available seed is used as a representative.

    Columns:
        dataset, protocol, split, n_rows, n_benign, n_attack, attack_pct.
    """
    rows = []
    seen: set = set()
    for r in results_list:
        if "error" in r:
            continue
        dataset = r["dataset"]
        for protocol in ("random", "temporal"):
            key = (dataset, protocol)
            if key in seen:
                continue
            seen.add(key)
            sa = r.get(protocol, {}).get("split_audit", {})
            for split in ("train", "val", "test"):
                s = sa.get(split, {})
                rows.append({
                    "dataset":    dataset,
                    "protocol":   protocol,
                    "split":      split,
                    "n_rows":     s.get("n_rows"),
                    "n_benign":   s.get("n_benign"),
                    "n_attack":   s.get("n_attack"),
                    "attack_pct": round(s.get("attack_ratio", 0) * 100, 1),
                })
    df = pd.DataFrame(rows)
    logger.info("Split table: %d rows", len(df))
    return df


# ── Per-seed performance ──────────────────────────────────────────────────

def build_performance_table(
    results_list: List[Dict[str, Any]],
    dataset_name: Optional[str] = None,
) -> pd.DataFrame:
    """Full per-seed metrics table (one row per dataset × seed × protocol).

    Includes default-threshold (0.5) metrics, val-tuned threshold metrics,
    probability diagnostics, and split/preprocessing metadata.

    Args:
        results_list:  Output of experiment.runner.run_all.
        dataset_name:  If given, filter to this dataset only.

    Returns:
        pd.DataFrame sorted by (dataset, protocol, seed).
    """
    rows = []
    for r in results_list:
        if "error" in r:
            logger.warning(
                "Skipping failed result: dataset=%s seed=%s", r.get("dataset"), r.get("seed")
            )
            continue
        if dataset_name and r["dataset"] != dataset_name:
            continue
        for protocol in ("random", "temporal"):
            pd_     = r.get(protocol, {})
            m       = pd_.get("metrics", {})
            mt      = pd_.get("metrics_tuned", {})
            pcr     = m.get("per_class_recall", {})
            pcr_t   = mt.get("per_class_recall", {})
            pp      = pd_.get("preprocessing_audit", {})
            rows.append({
                "dataset":                  r["dataset"],
                "seed":                     r["seed"],
                "protocol":                 protocol,
                # default threshold (0.5)
                "macro_f1":                 m.get("macro_f1"),
                "pr_auc":                   m.get("pr_auc"),
                "roc_auc":                  m.get("roc_auc"),
                "balanced_accuracy":        m.get("balanced_accuracy"),
                "recall_class_0":           pcr.get(0, pcr.get("0")),
                "recall_class_1":           pcr.get(1, pcr.get("1")),
                "predicted_positive_rate":  m.get("predicted_positive_rate"),
                "true_positive_rate":       m.get("true_positive_rate"),
                "best_iteration":           m.get("best_iteration"),
                "threshold_default":        pd_.get("threshold_default", 0.5),
                # val-tuned threshold
                "threshold_tuned":          pd_.get("threshold_tuned"),
                "threshold_val_score":      pd_.get("threshold_val_score"),
                "macro_f1_tuned":           mt.get("macro_f1"),
                "balanced_accuracy_tuned":  mt.get("balanced_accuracy"),
                "recall_class_1_tuned":     pcr_t.get(1, pcr_t.get("1")),
                "ppr_tuned":                mt.get("predicted_positive_rate"),
                # split / preprocessing
                "n_test_classes":           m.get("n_test_classes"),
                "n_train":                  pd_.get("n_train"),
                "n_val":                    pd_.get("n_val"),
                "n_test":                   pd_.get("n_test"),
                "n_features_selected":      pp.get("n_features_selected"),
                "fit_subsampled":           pp.get("fit_subsampled"),
                "fit_n_samples":            pp.get("fit_n_samples"),
            })

    df = (
        pd.DataFrame(rows)
        .sort_values(["dataset", "protocol", "seed"])
        .reset_index(drop=True)
    )
    logger.info("Performance table: %d rows × %d cols", len(df), df.shape[1])
    return df


# backward-compat alias used by main.py and build_delta_table
build_results_table = build_performance_table


# ── C1: Inflation gap ─────────────────────────────────────────────────────

def build_delta_table(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Per-dataset statistical comparison of random vs temporal protocol (C1).

    One row per (dataset, metric). Wilcoxon test is paired by seed index.

    Columns:
        dataset, n_seeds, metric, mean_delta, std_delta, ci_lower, ci_upper,
        wilcoxon_stat, p_value, significant.
    """
    from evaluation.statistical import summarise_statistical_tests

    results_df = build_performance_table(results_list)
    all_rows: List[pd.DataFrame] = []

    for dataset, group in results_df.groupby("dataset"):
        rand_rows = group[group["protocol"] == "random"].set_index("seed")
        temp_rows = group[group["protocol"] == "temporal"].set_index("seed")

        # Pair by matching seed values, not by positional order.  Positional
        # pairing silently computes cross-seed deltas when different seeds fail
        # in each protocol, corrupting every Wilcoxon test.
        common_seeds = rand_rows.index.intersection(temp_rows.index).sort_values()
        only_rand = rand_rows.index.difference(temp_rows.index)
        only_temp = temp_rows.index.difference(rand_rows.index)
        if only_rand.size or only_temp.size:
            logger.warning(
                "Seed mismatch for %s — random-only: %s, temporal-only: %s. "
                "Wilcoxon uses %d matched pairs.",
                dataset, sorted(only_rand), sorted(only_temp), len(common_seeds),
            )
        if len(common_seeds) == 0:
            logger.warning("No paired seeds for %s — skipping delta table", dataset)
            continue

        rand_rows = rand_rows.loc[common_seeds]
        temp_rows = temp_rows.loc[common_seeds]

        metric_cols  = ["macro_f1", "pr_auc", "roc_auc", "balanced_accuracy"]
        rand_metrics = rand_rows[metric_cols].to_dict("records")
        temp_metrics = temp_rows[metric_cols].to_dict("records")

        stat_df = summarise_statistical_tests(rand_metrics, temp_metrics)
        stat_df.insert(0, "dataset", dataset)
        stat_df.insert(1, "n_seeds", len(common_seeds))
        all_rows.append(stat_df.reset_index())

    if not all_rows:
        return pd.DataFrame()

    df = pd.concat(all_rows, ignore_index=True)
    logger.info("Delta table: %d rows", len(df))
    return df


# ── C2: Drift summary ─────────────────────────────────────────────────────

def build_drift_table(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Per-seed drift metrics (C2).

    One row per (dataset, seed) — only includes runs with drift data.

    Columns:
        dataset, seed, mean_psi, max_psi, n_drifted_dims, prop_drifted_dims,
        n_ks_significant, ae_n_train_orig, ae_n_train_used, ae_subsampled,
        ae_final_loss.
    """
    rows = []
    for r in results_list:
        if "error" in r or r.get("drift") is None:
            continue
        drift    = r["drift"]
        ae_audit = r.get("ae_audit") or {}
        if "error" in drift:
            logger.warning(
                "Drift error for dataset=%s seed=%s — excluded from C2 table: %s",
                r.get("dataset"), r.get("seed"), drift["error"],
            )
            continue
        bonf = drift.get("bonferroni_significant", [])
        rows.append({
            "dataset":           r["dataset"],
            "seed":              r["seed"],
            "mean_psi":          drift.get("mean_psi"),
            "max_psi":           drift.get("max_psi"),
            "n_drifted_dims":    drift.get("n_drifted_dims"),
            "prop_drifted_dims": drift.get("prop_drifted_dims"),
            "n_ks_significant":  int(sum(bonf)) if bonf else None,
            "ae_n_train_orig":   ae_audit.get("n_train_orig"),
            "ae_n_train_used":   ae_audit.get("n_train_used"),
            "ae_subsampled":     ae_audit.get("ae_subsampled"),
            "ae_final_loss":     ae_audit.get("final_loss"),
        })

    df = pd.DataFrame(rows)
    logger.info("Drift table: %d rows", len(df))
    return df


# ── Hypothesis conclusion ─────────────────────────────────────────────────

def build_hypothesis_table(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """One row per (contribution, dataset) with a plain-language verdict.

    C1 verdict: any metric shows a significant inflation gap (Wilcoxon p < α).
    C2 verdict: mean PSI > 0.10 (low-shift threshold) or any KS dimension significant.

    Columns:
        contribution, dataset, hypothesis,
        best_metric (C1) / drift_indicator (C2),
        mean_delta / mean_psi, ci_lower, ci_upper, p_value, conclusion.
    """
    from config import ALPHA, PSI_THRESHOLDS

    delta_df = build_delta_table(results_list)
    drift_df = build_drift_table(results_list)

    rows = []

    # C1 — one conclusion row per dataset, based on the metric with smallest p-value
    if not delta_df.empty:
        for dataset, grp in delta_df.groupby("dataset"):
            # prefer significant rows; among ties pick largest mean_delta
            sig  = grp[grp["significant"] == True]
            best = sig.iloc[0] if len(sig) else grp.loc[grp["mean_delta"].idxmax()]
            rows.append({
                "contribution": "C1",
                "dataset":      dataset,
                "hypothesis":   "Random-split protocol inflates reported performance vs temporal holdout",
                "indicator":    best["metric"],
                "value":        round(best["mean_delta"], 4) if pd.notna(best["mean_delta"]) else None,
                "ci_lower":     round(best["ci_lower"], 4)   if pd.notna(best.get("ci_lower")) else None,
                "ci_upper":     round(best["ci_upper"], 4)   if pd.notna(best.get("ci_upper")) else None,
                "p_value":      round(best["p_value"], 4)    if pd.notna(best["p_value"]) else None,
                "n_seeds":      int(best["n_seeds"])         if pd.notna(best.get("n_seeds")) else None,
                "conclusion":   "Supported" if len(sig) > 0 else "Not significant (n seeds too low for test)",
            })

    # C2 — one conclusion row per dataset
    if not drift_df.empty:
        lo_thr = PSI_THRESHOLDS["low"]
        for dataset, grp in drift_df.groupby("dataset"):
            mean_psi   = grp["mean_psi"].mean()
            mean_ks    = grp["n_ks_significant"].mean()
            drift_flag = (mean_psi > lo_thr) or (mean_ks > 0)
            rows.append({
                "contribution": "C2",
                "dataset":      dataset,
                "hypothesis":   "Latent-space distribution shift is detectable (temporal train → test)",
                "indicator":    "mean_PSI + n_KS_significant",
                "value":        round(mean_psi, 4),
                "ci_lower":     None,
                "ci_upper":     None,
                "p_value":      None,
                "n_seeds":      int(len(grp)),
                "conclusion":   "Drift detected" if drift_flag else "No significant drift",
            })

    df = pd.DataFrame(rows)
    logger.info("Hypothesis table: %d rows", len(df))
    return df
