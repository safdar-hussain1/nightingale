"""Evaluation helpers: metrics at a threshold, and threshold tuning."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_probabilities(y_true, probs, threshold: float = 0.5) -> dict:
    """Compute ranking metrics plus thresholded classification metrics."""
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def tune_threshold(y_true, probs) -> float:
    """Pick the probability threshold that maximises F1 on the given data.

    With ~5% positives the default 0.5 threshold is rarely optimal; tuning on
    a held-out validation split (never on training folds) trades a little
    precision for substantially better recall.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, (probs >= t).astype(int), zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])
