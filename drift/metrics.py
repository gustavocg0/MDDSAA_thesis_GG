"""Distribution drift metrics: PSI per latent dimension, KS test, Bonferroni correction.

All statistics are computed dimension-by-dimension on the 16-dimensional latent
space produced by the autoencoder. Operating in latent space (rather than the 30
raw features) has two advantages:
  1. Fewer multiple comparisons: Bonferroni threshold = 0.05/16 = 0.003125
     vs 0.05/30 = 0.0017 for raw features.
  2. The autoencoder's bottleneck encourages each dimension to capture a
     coherent mode of traffic variation, though dimensions are not orthogonal
     principal components and lack explicit semantic labels.

PSI BINNING: Bin edges are set to the quantiles of the TRAINING (reference)
distribution. This is methodologically preferable to uniform bins because it
ensures each bin contains a similar proportion of reference samples, making the
contribution of sparsely-occupied regions to the PSI sum comparable across bins.
Test (comparison) samples may fall outside these edges; edge bins are extended
to cover the full test range.

EPISTEMIC LIMITATION: Latent dimensions are not semantically transparent by
construction. Their interpretation depends on association with input features.
This module reports distributional shift, not causal explanation.
"""

import logging

import numpy as np
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


def compute_psi(
    train_latent: np.ndarray,
    test_latent:  np.ndarray,
    n_bins:       int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PSI for each latent dimension independently.

    PSI = Σ_bins (comp_i - ref_i) × ln(comp_i / ref_i)

    Interpretation thresholds (config.PSI_THRESHOLDS):
        PSI < 0.10:  stable (negligible shift)
        PSI 0.10-0.25: moderate shift
        PSI > 0.25:  significant shift

    When a dimension has many identical values (e.g. ReLU saturation), quantile
    edges collapse after deduplication, reducing the effective bin count below
    n_bins.  PSI computed over fewer bins is not numerically comparable to PSI
    over the full n_bins, so the actual bin count per dimension is returned as a
    second array and recorded in the audit.  Dimensions with fewer than n_bins
    are flagged with a WARNING.

    Args:
        train_latent: Latent matrix from training split (n_train, latent_dim).
        test_latent:  Latent matrix from test split (n_test, latent_dim).
        n_bins:       Requested number of histogram bins per dimension (default 10).

    Returns:
        (psi_values, n_bins_used) both of shape (latent_dim,).
        n_bins_used records the actual bin count used for each dimension.
    """
    n_dims      = train_latent.shape[1]
    psi_values  = np.zeros(n_dims, dtype="float64")
    n_bins_used = np.zeros(n_dims, dtype="int32")

    for d in range(n_dims):
        ref  = train_latent[:, d]
        comp = test_latent[:, d]

        # Bin edges from TRAINING quantiles (reference distribution).
        # This gives equal-count bins in the reference, which is the standard
        # PSI methodology. Edges are extended to cover any test values that
        # fall outside the training range.
        quantile_pts = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(ref, quantile_pts)
        edges[0]  = min(edges[0],  comp.min()) - 1e-10
        edges[-1] = max(edges[-1], comp.max()) + 1e-10
        edges = np.unique(edges)   # deduplicate (can happen with discrete/ReLU values)
        if len(edges) < 2:
            psi_values[d]  = 0.0
            n_bins_used[d] = 0
            continue

        n_actual_bins  = len(edges) - 1
        n_bins_used[d] = n_actual_bins
        if n_actual_bins < n_bins:
            logger.warning(
                "PSI dim %d: edge collapse reduced bins from %d to %d "
                "(likely ReLU saturation). PSI value is not directly comparable "
                "to dimensions with %d bins.",
                d, n_bins, n_actual_bins, n_bins,
            )

        ref_hist, _  = np.histogram(ref,  bins=edges)
        comp_hist, _ = np.histogram(comp, bins=edges)

        eps = 1e-10
        ref_dist  = (ref_hist  + eps) / (ref_hist.sum()  + n_actual_bins * eps)
        comp_dist = (comp_hist + eps) / (comp_hist.sum() + n_actual_bins * eps)

        psi_values[d] = np.sum((comp_dist - ref_dist) * np.log(comp_dist / ref_dist))

    return psi_values, n_bins_used


def compute_ks_per_dim(
    train_latent: np.ndarray,
    test_latent:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """KS two-sample test for each latent dimension.

    Args:
        train_latent: Latent matrix from training split (n_train, latent_dim).
        test_latent:  Latent matrix from test split (n_test, latent_dim).

    Returns:
        (ks_stats, ks_pvals) each of shape (latent_dim,).
    """
    n_dims = train_latent.shape[1]
    ks_stats = np.zeros(n_dims)
    ks_pvals = np.zeros(n_dims)

    for d in range(n_dims):
        stat, pval = ks_2samp(train_latent[:, d], test_latent[:, d])
        ks_stats[d] = stat
        ks_pvals[d] = pval

    return ks_stats, ks_pvals


def apply_bonferroni(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Bonferroni-corrected significance: threshold = alpha / n_tests.

    Args:
        pvals: p-value array (one per latent dimension).
        alpha: Family-wise error rate (default config.ALPHA = 0.05).

    Returns:
        Boolean array: True where the corrected null hypothesis is rejected.
    """
    threshold = alpha / len(pvals)
    return pvals < threshold


def summarise_drift(
    train_latent: np.ndarray,
    test_latent:  np.ndarray,
) -> dict:
    """Compute all drift metrics and return a structured summary dict.

    Args:
        train_latent: Latent matrix from temporal training split (n_train, latent_dim).
        test_latent:  Latent matrix from temporal test split (n_test, latent_dim).

    Returns:
        Dict with keys:
            mean_psi, max_psi, psi_per_dim (list),
            n_drifted_dims, prop_drifted_dims,
            ks_stats (list), ks_pvals (list),
            bonferroni_significant (list of bool).
    """
    from config import ALPHA, PSI_N_BINS, PSI_THRESHOLDS

    psi_per_dim, n_bins_per_dim = compute_psi(
        train_latent, test_latent, n_bins=PSI_N_BINS
    )
    ks_stats, ks_pvals = compute_ks_per_dim(train_latent, test_latent)
    bonferroni_sig = apply_bonferroni(ks_pvals, alpha=ALPHA)

    low_threshold = PSI_THRESHOLDS["low"]
    n_drifted = int((psi_per_dim > low_threshold).sum())

    summary = {
        "mean_psi":              float(psi_per_dim.mean()),
        "max_psi":               float(psi_per_dim.max()),
        "psi_per_dim":           psi_per_dim.tolist(),
        # Actual bin counts per dimension — may be < PSI_N_BINS when ReLU
        # saturation collapses quantile edges.  Disclosed for audit purposes.
        "n_bins_per_dim":        n_bins_per_dim.tolist(),
        "n_drifted_dims":        n_drifted,
        "prop_drifted_dims":     n_drifted / len(psi_per_dim),
        "ks_stats":              ks_stats.tolist(),
        "ks_pvals":              ks_pvals.tolist(),
        "bonferroni_significant": bonferroni_sig.tolist(),
    }

    logger.info(
        "Drift summary: mean_psi=%.4f | max_psi=%.4f | "
        "drifted_dims=%d/%d | ks_sig=%d dims",
        summary["mean_psi"],
        summary["max_psi"],
        n_drifted,
        len(psi_per_dim),
        int(bonferroni_sig.sum()),
    )
    return summary
