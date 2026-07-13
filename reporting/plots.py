"""Visualisations for thesis results.

Figure inventory
────────────────
plot_performance_boxplot   — random vs temporal boxplot per dataset      (one per dataset)
plot_inflation_gap_ci      — mean delta ± 95 % CI per metric             (one per dataset)
plot_psi_heatmap           — PSI per latent dim × seed                   (one per dataset)
plot_delta_distributions   — per-seed delta boxplots (internal helper)
plot_latent_scatter        — PCA/UMAP latent space projection
plot_metric_comparison     — legacy scatter comparison across datasets

All functions save to disk and return the figure.  No global matplotlib state.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

_PALETTE    = {"random": "#4c72b0", "temporal": "#dd8452"}
_METRICS    = ["macro_f1", "pr_auc", "roc_auc", "balanced_accuracy"]
_METRIC_LABELS = {
    "macro_f1":          "Macro F1",
    "pr_auc":            "PR-AUC",
    "roc_auc":           "ROC-AUC",
    "balanced_accuracy": "Balanced Accuracy",
}

_DATASET_LABELS = {
    "CIC2017": "BCCC-CIC-IDS-2017",
    "CIC2018": "CSE-CIC-IDS2018",
}


def _label(dataset: str) -> str:
    return _DATASET_LABELS.get(dataset, dataset)


def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)


# ── 1. Performance boxplot ────────────────────────────────────────────────

def plot_performance_boxplot(
    results_df:  pd.DataFrame,
    dataset_name: str,
    save_path:   Path,
) -> Optional[plt.Figure]:
    """Side-by-side boxplot: random vs temporal for each metric.

    Each box spans 5 seeds (or however many are present).  Individual seed
    values are overlaid as jittered points.

    Args:
        results_df:   Output of build_performance_table (all datasets).
        dataset_name: Dataset key ("CIC2017" or "CIC2018").
        save_path:    Destination PNG path.
    """
    sub = results_df[results_df["dataset"] == dataset_name]
    if sub.empty:
        logger.warning("No data for %s — skipping performance boxplot", dataset_name)
        return None

    n_metrics = len(_METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 3.5, 4.5), squeeze=False)
    axes = axes[0]

    for ax, metric in zip(axes, _METRICS):
        data_r = sub[sub["protocol"] == "random"][metric].dropna().values.astype(float)
        data_t = sub[sub["protocol"] == "temporal"][metric].dropna().values.astype(float)

        bp = ax.boxplot(
            [data_r, data_t],
            positions=[1, 2],
            widths=0.45,
            patch_artist=True,
            boxprops=dict(linewidth=1.2),
            medianprops=dict(linewidth=2, color="black"),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="x", markersize=5),
        )
        for patch, color in zip(bp["boxes"], [_PALETTE["random"], _PALETTE["temporal"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        rng = np.random.default_rng(0)
        for i, (data, color) in enumerate(
            [(data_r, _PALETTE["random"]), (data_t, _PALETTE["temporal"])], start=1
        ):
            jitter = rng.uniform(-0.08, 0.08, size=len(data))
            ax.scatter(i + jitter, data, color=color, zorder=4, s=35, alpha=0.9, edgecolors="white", linewidths=0.5)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Random", "Temporal"], fontsize=9)
        ax.set_title(_METRIC_LABELS[metric], fontsize=10)
        ax.set_ylim(-0.03, 1.06)
        ax.axhline(0.5, color="grey", linewidth=0.6, linestyle=":", alpha=0.6)
        ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Random vs Temporal Protocol — {_label(dataset_name)}",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ── 2. Inflation-gap CI plot ──────────────────────────────────────────────

def plot_inflation_gap_ci(
    delta_df:    pd.DataFrame,
    dataset_name: str,
    save_path:   Path,
) -> Optional[plt.Figure]:
    """Mean delta (random − temporal) with 95 % bootstrap CI per metric.

    Significant metrics are coloured in orange; non-significant in grey.
    A dashed line at Δ = 0 marks the no-inflation boundary.

    Args:
        delta_df:     Output of build_delta_table (all datasets).
        dataset_name: Dataset key.
        save_path:    Destination PNG path.
    """
    sub = delta_df[delta_df["dataset"] == dataset_name].copy()
    if sub.empty:
        logger.warning("No delta data for %s — skipping CI plot", dataset_name)
        return None

    sub = sub.set_index("metric").reindex(_METRICS).reset_index()
    sub = sub.dropna(subset=["mean_delta"])

    fig, ax = plt.subplots(figsize=(7, 4))

    # Use a contiguous positional counter (pos) for x-coordinates.
    # iterrows() yields the DataFrame index as the first element; after dropna
    # that index is gapped (e.g. 0,1,3), so using it directly as the x-position
    # misaligns error-bar markers with the set_xticks(range(len(sub))) labels.
    for pos, (_, row) in enumerate(sub.iterrows()):
        sig   = bool(row.get("significant", False))
        color = _PALETTE["temporal"] if sig else "#aaaaaa"
        mean  = row["mean_delta"]
        lo    = row.get("ci_lower", mean)
        hi    = row.get("ci_upper", mean)
        ax.errorbar(
            pos, mean,
            yerr=[[mean - lo], [hi - mean]],
            fmt="o", color=color,
            capsize=6, capthick=1.5,
            markersize=8, linewidth=1.5,
            label="significant" if sig else "n.s.",
            zorder=3,
        )
        ax.annotate(
            f"Δ={mean:.3f}",
            xy=(pos, hi),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", fontsize=7.5, color=color,
        )

    ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels([_METRIC_LABELS.get(m, m) for m in sub["metric"]], fontsize=9)
    ax.set_ylabel("Δ (random − temporal)", fontsize=10)
    ax.set_title(
        f"C1 — Inflation Gap with 95 % Bootstrap CI\n{_label(dataset_name)}",
        fontsize=11,
    )
    ax.tick_params(labelsize=9)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_PALETTE["temporal"],
               markersize=9, label="significant (Wilcoxon p < α)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#aaaaaa",
               markersize=9, label="not significant"),
    ]
    ax.legend(handles=handles, fontsize=8, framealpha=0.7)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ── 3. PSI heatmap ────────────────────────────────────────────────────────

def plot_psi_heatmap(
    results_list: List[Dict],
    dataset_name: str,
    save_path:    Path,
) -> Optional[plt.Figure]:
    """Heatmap of PSI per latent dimension × seed.

    Rows = seeds, Columns = latent dimensions (dim_0 … dim_15).
    Colour encodes PSI value; cells with PSI > moderate threshold are annotated.

    Args:
        results_list: Output of experiment.runner.run_all.
        dataset_name: Used to filter and for the title.
        save_path:    Destination PNG path.
    """
    from config import PSI_THRESHOLDS

    rows = []
    for r in results_list:
        if r.get("dataset") != dataset_name:
            continue
        drift = r.get("drift") or {}
        if "psi_per_dim" not in drift:
            continue
        rows.append({
            "seed": r["seed"],
            **{f"dim_{i}": v for i, v in enumerate(drift["psi_per_dim"])},
        })

    if not rows:
        logger.warning("No drift data for %s — skipping PSI heatmap", dataset_name)
        return None

    psi_df = pd.DataFrame(rows).set_index("seed").sort_index()
    vmax   = max(PSI_THRESHOLDS["moderate"], float(psi_df.values.max()))

    fig, ax = plt.subplots(figsize=(max(8, psi_df.shape[1] * 0.6), max(3, len(psi_df) * 0.7 + 1)))

    sns.heatmap(
        psi_df,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        linewidths=0.4,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        cbar_kws={"label": "PSI"},
    )
    ax.set_title(f"C2 — PSI per Latent Dimension — {_label(dataset_name)}", fontsize=11)
    ax.set_xlabel("Latent Dimension", fontsize=9)
    ax.set_ylabel("Seed", fontsize=9)

    # Threshold reference line on colour bar
    fig.text(
        0.91, 0.5,
        f"← moderate-drift\n  threshold\n  ({PSI_THRESHOLDS['low']})",
        va="center", ha="left", rotation=0, fontsize=7, color="grey",
    )

    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ── 3b. PSI heatmap from saved wide-format CSV ───────────────────────────

def plot_psi_heatmap_from_df(
    psi_wide_df:  pd.DataFrame,
    dataset_name: str,
    save_path:    Path,
) -> Optional[plt.Figure]:
    """PSI heatmap from a wide-format DataFrame (dataset, seed, dim_0 … dim_N).

    Mirrors plot_psi_heatmap but reads from the saved c2_psi_per_dim CSV
    instead of the raw results list.  Use this when JSON result files are
    no longer available.

    Args:
        psi_wide_df:  DataFrame with columns: dataset, seed, dim_0, …, dim_N.
        dataset_name: Dataset key used to filter the DataFrame and for the title.
        save_path:    Destination PNG path.
    """
    from config import PSI_THRESHOLDS

    sub = psi_wide_df[psi_wide_df["dataset"] == dataset_name].drop(columns="dataset")
    if sub.empty:
        logger.warning("No PSI data for %s — skipping heatmap", dataset_name)
        return None

    psi_df = sub.set_index("seed").sort_index()
    vmax   = max(PSI_THRESHOLDS["moderate"], float(psi_df.values.max()))

    fig, ax = plt.subplots(
        figsize=(max(8, psi_df.shape[1] * 0.6), max(3, len(psi_df) * 0.7 + 1))
    )
    sns.heatmap(
        psi_df, ax=ax, cmap="YlOrRd", vmin=0, vmax=vmax,
        linewidths=0.4, annot=True, fmt=".2f",
        annot_kws={"size": 7}, cbar_kws={"label": "PSI"},
    )
    ax.set_title(f"C2 — PSI per Latent Dimension — {_label(dataset_name)}", fontsize=11)
    ax.set_xlabel("Latent Dimension", fontsize=9)
    ax.set_ylabel("Seed", fontsize=9)
    fig.text(
        0.91, 0.5,
        f"← moderate-drift\n  threshold\n  ({PSI_THRESHOLDS['low']})",
        va="center", ha="left", rotation=0, fontsize=7, color="grey",
    )
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ── Legacy / internal helpers ─────────────────────────────────────────────

def plot_delta_distributions(
    delta_dict:   Dict[str, np.ndarray],
    dataset_name: str,
    save_path:    Path,
) -> plt.Figure:
    """Boxplots + jitter of per-seed deltas (random − temporal) per metric."""
    metrics = list(delta_dict.keys())
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.5, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        deltas = delta_dict[metric]
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.boxplot(
            deltas, widths=0.4, patch_artist=True,
            boxprops=dict(facecolor="#aec6cf", color="black"),
            medianprops=dict(color="navy"),
        )
        ax.scatter([1] * len(deltas), deltas, color="navy", alpha=0.7, zorder=3, s=40)
        ax.set_title(_METRIC_LABELS.get(metric, metric), fontsize=9)
        ax.set_xticks([])
        ax.set_ylabel("Δ (random − temporal)")

    fig.suptitle(f"Protocol Inflation Gap — {_label(dataset_name)}", fontsize=12, y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_latent_scatter(
    train_latent: np.ndarray,
    test_latent:  np.ndarray,
    dataset_name: str,
    save_path:    Path,
    method:       str = "pca",
) -> plt.Figure:
    """2D PCA or UMAP projection of train vs test latent representations."""
    combined = np.vstack([train_latent, test_latent])
    labels   = np.array(["train"] * len(train_latent) + ["test"] * len(test_latent))

    if method == "umap":
        try:
            import umap
            embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(combined)
        except ImportError:
            logger.warning("umap-learn not installed — falling back to PCA")
            method = "pca"

    if method == "pca":
        from sklearn.decomposition import PCA
        embedding = PCA(n_components=2, random_state=42).fit_transform(combined)

    fig, ax = plt.subplots(figsize=(7, 5))
    for split, color in [("train", "#4c72b0"), ("test", "#dd8452")]:
        mask = labels == split
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=color, label=split, alpha=0.4, s=10, linewidths=0,
        )
    ax.set_title(f"Latent Space — {_label(dataset_name)} ({method.upper()})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(markerscale=3)
    _save(fig, save_path)
    return fig


def plot_metric_comparison(
    results_df: pd.DataFrame,
    save_path:  Path,
) -> plt.Figure:
    """Side-by-side scatter comparison across all datasets and metrics (legacy)."""
    datasets  = results_df["dataset"].unique()
    n_metrics = len(_METRICS)

    fig, axes = plt.subplots(
        len(datasets), n_metrics,
        figsize=(n_metrics * 3.5, len(datasets) * 3.5),
        squeeze=False,
    )

    for row_idx, dataset in enumerate(datasets):
        sub = results_df[results_df["dataset"] == dataset]
        for col_idx, metric in enumerate(_METRICS):
            ax = axes[row_idx][col_idx]
            for protocol, color in _PALETTE.items():
                vals = sub[sub["protocol"] == protocol][metric].dropna().values
                ax.scatter([protocol] * len(vals), vals, color=color, alpha=0.8, s=50, zorder=3)
                if len(vals):
                    ax.plot(
                        [protocol] * 2, [vals.min(), vals.max()],
                        color=color, linewidth=1.5, alpha=0.5,
                    )
            ax.set_title(f"{_label(dataset)}\n{_METRIC_LABELS[metric]}", fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.tick_params(labelsize=8)

    fig.suptitle("Random vs Temporal Protocol — Metric Comparison", fontsize=12, y=1.01)
    plt.tight_layout()
    _save(fig, save_path)
    return fig
