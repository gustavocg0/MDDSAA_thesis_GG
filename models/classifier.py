"""XGBoost binary classifier with inverse-frequency class weighting.

Hyperparameters are FROZEN via config.MODEL_PARAMS and must not be changed
between experiments; the only per-run variable is the random seed.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import xgboost as xgb

logger = logging.getLogger(__name__)


def _compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency sample weights: w_c = n / (n_classes * count_c)."""
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    k = len(classes)
    weight_map = {cls: n / (k * cnt) for cls, cnt in zip(classes, counts)}
    return np.array([weight_map[label] for label in y], dtype="float32")


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    params:  Dict[str, Any],
    seed:    int,
) -> xgb.XGBClassifier:
    """Train an XGBoost binary classifier with early stopping on the validation set.

    Class imbalance is handled via inverse-frequency sample weights applied to
    the training split only.  The validation set is used solely for early stopping
    and never influences feature selection or preprocessing decisions.

    Args:
        X_train: Training feature matrix (n_samples, n_features).
        y_train: Training binary labels (n_samples,).
        X_val:   Validation feature matrix.
        y_val:   Validation binary labels.
        params:  XGBoost hyperparameter dict (from config.MODEL_PARAMS).
        seed:    Random seed for XGBoost.

    Returns:
        Fitted xgb.XGBClassifier.
    """
    from config import EARLY_STOPPING_ROUNDS

    sample_weights = _compute_sample_weights(y_train)

    model = xgb.XGBClassifier(
        **params,
        random_state=seed,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    best_iter = getattr(model, "best_iteration", None)
    logger.info(
        "Classifier trained: seed=%d | best_iteration=%s | train_n=%d val_n=%d",
        seed, best_iter, len(y_train), len(y_val),
    )
    return model


def find_optimal_threshold(
    model:       xgb.XGBClassifier,
    X_val:       np.ndarray,
    y_val:       np.ndarray,
    metric:      str = "balanced_accuracy",
    n_thresholds: int = 99,
) -> tuple:
    """Find the probability threshold maximising a metric on the validation set.

    Sweeps n_thresholds evenly-spaced candidates in (0, 1) exclusive, then
    picks the one with the highest metric score.  Ties broken by choosing the
    lower threshold (more aggressive positive prediction).

    Args:
        model:        Fitted XGBClassifier.
        X_val:        Validation feature matrix.
        y_val:        Validation binary labels.
        metric:       Metric to optimise — "balanced_accuracy" or "macro_f1".
        n_thresholds: Number of candidate thresholds to evaluate.

    Returns:
        (optimal_threshold, val_score) — the chosen threshold and the metric
        value achieved at that threshold on the validation set.
    """
    from sklearn.metrics import balanced_accuracy_score, f1_score

    y_proba = model.predict_proba(X_val)[:, 1]
    candidates = np.linspace(0.0, 1.0, n_thresholds + 2)[1:-1]  # exclude 0 and 1

    best_t     = 0.5
    best_score = -np.inf

    for t in candidates:
        y_pred = (y_proba >= t).astype(int)
        if metric == "balanced_accuracy":
            score = float(balanced_accuracy_score(y_val, y_pred))
        elif metric == "macro_f1":
            score = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        else:
            raise ValueError(f"Unknown threshold-tuning metric: {metric!r}")

        if score > best_score:
            best_score = score
            best_t     = float(t)

    logger.info(
        "Threshold search (val, metric=%s): optimal=%.3f → val_score=%.4f",
        metric, best_t, best_score,
    )
    return best_t, best_score


def compute_proba_percentiles(
    model:  xgb.XGBClassifier,
    X:      np.ndarray,
    y:      np.ndarray,
) -> Dict[str, Any]:
    """Percentiles of predicted attack probability, stratified by true class.

    Useful for diagnosing threshold and calibration issues: if the attack
    distribution is entirely below 0.5, the default threshold will miss all attacks
    regardless of ranking quality.

    Returns a dict with keys "all", "benign", "attack" (when both classes present),
    each containing p1/p5/p10/p25/p50/p75/p90/p95/p99.
    """
    percs   = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    y_proba = model.predict_proba(X)[:, 1]

    def _pct(arr: np.ndarray) -> Dict[str, float]:
        return {f"p{p}": float(np.percentile(arr, p)) for p in percs}

    result: Dict[str, Any] = {"all": _pct(y_proba)}
    unique = np.unique(y)
    if 0 in unique:
        result["benign"] = _pct(y_proba[y == 0])
    if 1 in unique:
        result["attack"] = _pct(y_proba[y == 1])
    return result


def evaluate_classifier(
    model:     xgb.XGBClassifier,
    X_test:    np.ndarray,
    y_test:    np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluate a trained classifier at a given decision threshold.

    Metrics:
        macro_f1, pr_auc, roc_auc, balanced_accuracy,
        per_class_recall, confusion_matrix,
        predicted_positive_rate, true_positive_rate,
        threshold, best_iteration, n_test_classes.

    pr_auc and roc_auc are set to None when only one class is present in y_test
    (both metrics are undefined in that case).

    Args:
        model:     Fitted XGBClassifier.
        X_test:    Test feature matrix.
        y_test:    True binary labels.
        threshold: Decision boundary for converting probabilities to class labels
                   (default 0.5). Use find_optimal_threshold() to obtain a
                   validation-tuned value.

    Returns:
        Dict with metric names as keys.
    """
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        recall_score,
        roc_auc_score,
    )

    unique_classes = np.unique(y_test)
    class_counts   = {int(c): int((y_test == c).sum()) for c in unique_classes}
    logger.info("Test class distribution: %s | threshold=%.3f", class_counts, threshold)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    macro_f1          = f1_score(y_test, y_pred, average="macro", zero_division=0)
    bal_acc           = balanced_accuracy_score(y_test, y_pred)
    per_class_recalls = recall_score(
        y_test, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    cm                = confusion_matrix(y_test, y_pred, labels=[0, 1])

    if len(unique_classes) < 2:
        logger.warning(
            "Only one class in test set (%s) — roc_auc and pr_auc set to None",
            class_counts,
        )
        roc_auc: Optional[float] = None
        pr_auc:  Optional[float] = None
    else:
        roc_auc = float(roc_auc_score(y_test, y_proba))
        pr_auc  = float(average_precision_score(y_test, y_proba))

    per_class_recall = {int(c): float(r) for c, r in enumerate(per_class_recalls)}
    tpr = per_class_recall.get(1, 0.0)   # attack recall == TPR
    ppr = float(y_pred.mean())            # predicted positive rate

    result: Dict[str, Any] = {
        "macro_f1":                float(macro_f1),
        "pr_auc":                  pr_auc,
        "roc_auc":                 roc_auc,
        "balanced_accuracy":       float(bal_acc),
        "per_class_recall":        per_class_recall,
        "confusion_matrix":        cm.tolist(),
        "predicted_positive_rate": ppr,
        "true_positive_rate":      tpr,
        "threshold":               float(threshold),
        "best_iteration":          getattr(model, "best_iteration", None),
        "n_test_classes":          int(len(unique_classes)),
        "proba_percentiles":       compute_proba_percentiles(model, X_test, y_test),
    }

    logger.info(
        "Eval@%.3f: macro_f1=%.4f | pr_auc=%s | roc_auc=%s | "
        "bal_acc=%.4f | TPR=%.4f | PPR=%.4f",
        threshold, macro_f1,
        f"{pr_auc:.4f}" if pr_auc is not None else "None",
        f"{roc_auc:.4f}" if roc_auc is not None else "None",
        bal_acc, tpr, ppr,
    )
    return result
