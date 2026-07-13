"""Thin wrapper around sklearn classification metrics.

Exists to keep the models/classifier.py evaluate_classifier function
focused on training logic and provide a standalone metrics function
that can be used independently (e.g. for unit tests or ablations).
"""

from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, Any]:
    """Compute the standard binary classification metrics.

    Multiclass PR-AUC is computed via macro averaging using one-vs-rest
    decomposition (average_precision_score with average='macro').
    For binary classification, y_proba should be the positive-class column.

    Args:
        y_true:  True binary labels (n_samples,).
        y_pred:  Predicted binary labels (n_samples,).
        y_proba: Predicted probabilities for the positive class (n_samples,)
                 OR full probability matrix (n_samples, n_classes).

    Returns:
        Dict with keys: macro_f1, pr_auc, roc_auc, balanced_accuracy,
        per_class_recall (dict mapping int class -> float recall).
    """
    # Handle both (n,) and (n, 2) proba shapes
    if y_proba.ndim == 2:
        y_proba_pos = y_proba[:, 1]
    else:
        y_proba_pos = y_proba

    macro_f1          = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc           = balanced_accuracy_score(y_true, y_pred)
    per_class_recalls = recall_score(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )

    # roc_auc and pr_auc are undefined when y_true has only one class.
    # Return None for those metrics rather than raising ValueError.
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        pr_auc  = None
        roc_auc = None
    else:
        pr_auc  = float(average_precision_score(y_true, y_proba_pos))
        roc_auc = float(roc_auc_score(y_true, y_proba_pos))

    return {
        "macro_f1":          float(macro_f1),
        "pr_auc":            pr_auc,
        "roc_auc":           roc_auc,
        "balanced_accuracy": float(bal_acc),
        "per_class_recall": {
            int(c): float(r) for c, r in enumerate(per_class_recalls)
        },
    }
