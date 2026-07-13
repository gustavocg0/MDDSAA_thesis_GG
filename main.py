"""Entry point for the IDS evaluation protocol thesis experiments.

Usage:
    python main.py
    python main.py --datasets CIC2017 --seeds 42 43
    python main.py --skip-drift --output-dir /tmp/results
    python main.py --datasets CIC2017 CIC2018 --seeds 42 43 44 45 46
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python main.py` from within ids_thesis/
sys.path.insert(0, str(Path(__file__).parent))


def _configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def _parse_args() -> argparse.Namespace:
    from config import DATASETS, OUTPUT_DIR, SEEDS

    parser = argparse.ArgumentParser(
        description="IDS evaluation protocol thesis — run all experiments and generate reports"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASETS.keys()),
        help=f"Datasets to run (default: {list(DATASETS.keys())})",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help=f"Seeds to run (default: {SEEDS})",
    )
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR),
        help="Directory for result JSON files and figures",
    )
    parser.add_argument(
        "--skip-drift", action="store_true",
        help="Skip autoencoder training and drift computation",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("IDS Thesis — Evaluation Protocol Bias Study")
    logger.info("Datasets : %s", args.datasets)
    logger.info("Seeds    : %s", args.seeds)
    logger.info("Skip drift: %s", args.skip_drift)

    # Override OUTPUT_DIR from args so all modules use the same path
    import config as cfg
    cfg.OUTPUT_DIR = Path(args.output_dir)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Run all experiments ────────────────────────────────────────────────
    from experiment.runner import run_all

    results = run_all(
        skip_drift=args.skip_drift,
        dataset_filter=args.datasets,
        seed_filter=args.seeds,
    )

    if not results:
        logger.error("No results produced — check dataset paths in config.py")
        sys.exit(1)

    # ── Reporting ──────────────────────────────────────────────────────────
    import pandas as pd
    from reporting.tables import (
        build_dataset_summary_table,
        build_delta_table,
        build_drift_table,
        build_hypothesis_table,
        build_results_table,
        build_split_table,
    )

    results_df    = build_results_table(results)
    delta_df      = build_delta_table(results)
    drift_df      = build_drift_table(results)
    summary_df    = build_dataset_summary_table(results)
    split_df      = build_split_table(results)
    hypothesis_df = build_hypothesis_table(results)

    output_dir  = cfg.OUTPUT_DIR
    tables_dir  = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    # Root-level CSVs (backward compat)
    results_df.to_csv(output_dir / "results_raw.csv",        index=False)
    delta_df.to_csv(  output_dir / "c1_inflation_gap.csv",   index=False)
    drift_df.to_csv(  output_dir / "c2_drift_summary.csv",   index=False)

    # Full table set in tables/
    results_df.to_csv(   tables_dir / "results_raw.csv",        index=False)
    delta_df.to_csv(     tables_dir / "c1_inflation_gap.csv",   index=False)
    drift_df.to_csv(     tables_dir / "c2_drift_summary.csv",   index=False)
    summary_df.to_csv(   tables_dir / "dataset_summary.csv",    index=False)
    split_df.to_csv(     tables_dir / "split_table.csv",        index=False)
    hypothesis_df.to_csv(tables_dir / "hypothesis_table.csv",   index=False)

    # Per-dataset aggregated performance tables (mean ± std over seeds)
    _metric_cols = ["macro_f1", "pr_auc", "roc_auc", "balanced_accuracy"]
    _dataset_map = {"CIC2017": "perf_BCCC.csv", "CIC2018": "perf_CSE.csv"}
    for _ds, _fname in _dataset_map.items():
        _sub = results_df[results_df["dataset"] == _ds]
        if _sub.empty:
            continue
        _agg = (
            _sub.groupby("protocol")[_metric_cols]
            .agg(["mean", "std"])
            .round(4)
        )
        _agg.to_csv(tables_dir / _fname)

    # PSI per-dimension wide table for heatmap regeneration (no JSON needed)
    _psi_rows = []
    for r in results:
        drift = r.get("drift") or {}
        if "psi_per_dim" not in drift:
            continue
        _psi_rows.append({
            "dataset": r["dataset"],
            "seed":    r["seed"],
            **{f"dim_{i}": v for i, v in enumerate(drift["psi_per_dim"])},
        })
    if _psi_rows:
        pd.DataFrame(_psi_rows).to_csv(output_dir / "c2_psi_per_dim.csv", index=False)

    logger.info("Tables saved to %s", output_dir)

    # Plots
    from evaluation.statistical import compute_deltas
    from reporting.plots import (
        plot_delta_distributions,
        plot_inflation_gap_ci,
        plot_metric_comparison,
        plot_performance_boxplot,
        plot_psi_heatmap,
    )

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    plot_metric_comparison(results_df, figures_dir / "metric_comparison.png")

    for dataset in args.datasets:
        plot_performance_boxplot(
            results_df, dataset, figures_dir / f"performance_boxplot_{dataset}.png"
        )
        plot_inflation_gap_ci(
            delta_df, dataset, figures_dir / f"inflation_gap_ci_{dataset}.png"
        )
        plot_psi_heatmap(results, dataset, figures_dir / f"psi_heatmap_{dataset}.png")

        # Delta distributions (internal diagnostic)
        rand_rows = results_df[
            (results_df["dataset"] == dataset) & (results_df["protocol"] == "random")
        ].sort_values("seed")
        temp_rows = results_df[
            (results_df["dataset"] == dataset) & (results_df["protocol"] == "temporal")
        ].sort_values("seed")

        if len(rand_rows) == len(temp_rows) > 0:
            metric_cols = ["macro_f1", "pr_auc", "roc_auc", "balanced_accuracy"]
            rand_dicts = rand_rows[metric_cols].to_dict("records")
            temp_dicts = temp_rows[metric_cols].to_dict("records")
            deltas = compute_deltas(rand_dicts, temp_dicts)
            plot_delta_distributions(deltas, dataset, figures_dir / f"deltas_{dataset}.png")

    # ── Console summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (mean across seeds per protocol)")
    print("=" * 70)
    summary = (
        results_df.groupby(["dataset", "protocol"])
        [["macro_f1", "pr_auc", "roc_auc", "balanced_accuracy"]]
        .mean()
        .round(4)
    )
    print(summary.to_string())

    if not delta_df.empty:
        print("\n" + "=" * 70)
        print("C1 — INFLATION GAP (random − temporal) WITH WILCOXON p-VALUE")
        print("=" * 70)
        cols = ["dataset", "metric", "mean_delta", "ci_lower", "ci_upper", "p_value", "significant"]
        available = [c for c in cols if c in delta_df.columns]
        print(delta_df[available].round(4).to_string(index=False))

    if not drift_df.empty:
        print("\n" + "=" * 70)
        print("C2 — LATENT-SPACE DRIFT (temporal train → temporal test)")
        print("=" * 70)
        print(drift_df.round(4).to_string(index=False))

    print("\nAll outputs saved to:", output_dir)


if __name__ == "__main__":
    main()
