"""Model selection, threshold tuning, and final-model training.

Run from the repo root:
    python -m disease_prediction.train

Protocol
--------
1. Split off a stratified 20% validation set. It is never touched by
   GridSearchCV, SMOTE, or scaling fits.
2. For each candidate pipeline, grid-search hyperparameters with
   stratified 5-fold CV (ROC-AUC) on the training split only.
3. Compare candidates on the validation set (PR-AUC first, ROC-AUC as
   tie-breaker) and tune the decision threshold there.
4. Refit the winning pipeline on all labelled data and save it together
   with a metrics report.
"""

from __future__ import annotations

import json

import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from . import config
from .data import load_train
from .evaluate import evaluate_probabilities, tune_threshold
from .pipeline import build_candidates


def select_model(X_train, y_train, X_val, y_val, verbose: bool = True) -> dict:
    """Grid-search every candidate and score it on the validation split."""
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    results = {}
    for name, spec in build_candidates().items():
        search = GridSearchCV(
            spec["pipeline"], spec["params"], scoring="roc_auc", cv=cv, n_jobs=-1, refit=True
        )
        search.fit(X_train, y_train)
        val_probs = search.predict_proba(X_val)[:, 1]
        metrics = evaluate_probabilities(y_val, val_probs)
        results[name] = {
            "search": search,
            "cv_roc_auc": float(search.best_score_),
            "val_metrics": metrics,
            "best_params": search.best_params_,
        }
        if verbose:
            print(
                f"{name:20s} cv_roc_auc={search.best_score_:.4f} "
                f"val_roc_auc={metrics['roc_auc']:.4f} val_pr_auc={metrics['pr_auc']:.4f}"
            )
    return results


def train(train_path=None, verbose: bool = True) -> dict:
    """Full training run; returns the report dict written to models/metrics.json."""
    X, y = load_train(train_path or config.TRAIN_PATH)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config.VALIDATION_SIZE,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )

    results = select_model(X_train, y_train, X_val, y_val, verbose=verbose)
    best_name = max(
        results, key=lambda n: (results[n]["val_metrics"]["pr_auc"], results[n]["val_metrics"]["roc_auc"])
    )
    best = results[best_name]

    val_probs = best["search"].predict_proba(X_val)[:, 1]
    threshold = tune_threshold(y_val, val_probs)
    tuned_metrics = evaluate_probabilities(y_val, val_probs, threshold)

    # Refit the winning configuration on ALL labelled data for deployment.
    final_model = best["search"].best_estimator_
    final_model.fit(X, y)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "threshold": threshold}, config.MODEL_PATH)

    report = {
        "best_model": best_name,
        "best_params": best["best_params"],
        "cv_roc_auc": best["cv_roc_auc"],
        "validation": tuned_metrics,
        "all_candidates": {
            name: {
                "cv_roc_auc": r["cv_roc_auc"],
                "val_roc_auc": r["val_metrics"]["roc_auc"],
                "val_pr_auc": r["val_metrics"]["pr_auc"],
                "best_params": r["best_params"],
            }
            for name, r in results.items()
        },
    }
    config.METRICS_PATH.write_text(json.dumps(report, indent=2))

    if verbose:
        print(f"\nBest model: {best_name} (threshold={threshold:.2f})")
        print(json.dumps(tuned_metrics, indent=2))
        print(f"Saved model -> {config.MODEL_PATH}")
    return report


if __name__ == "__main__":
    train()
