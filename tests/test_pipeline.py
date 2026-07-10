"""Fast sanity tests: schema validation, threshold tuning, and an end-to-end
training smoke test on a small synthetic sample."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score

from disease_prediction import config
from disease_prediction.data import validate
from disease_prediction.evaluate import evaluate_probabilities, tune_threshold
from disease_prediction.pipeline import build_candidates


def make_frame(n=200, with_target=True, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.uniform(size=(n, 10)), columns=config.FEATURES)
    df[config.ID_COLUMN] = np.arange(n)
    if with_target:
        # Signal on feature_1 so models can beat chance.
        df[config.TARGET] = (df["feature_1"] > 0.9).astype(int)
    return df


def test_validate_accepts_good_frame():
    validate(make_frame(), require_target=True)


def test_validate_rejects_missing_columns():
    df = make_frame().drop(columns=["feature_3"])
    with pytest.raises(ValueError, match="feature_3"):
        validate(df, require_target=True)


def test_validate_rejects_nan_features():
    df = make_frame()
    df.loc[0, "feature_1"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        validate(df, require_target=True)


def test_evaluate_probabilities_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = evaluate_probabilities(y, probs)
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"]["fp"] == 0


def test_tune_threshold_beats_default_on_imbalanced_probs():
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=2000) < 0.05).astype(int)
    # Informative but low-scale probabilities (max well under 0.5).
    probs = np.clip(y * 0.3 + rng.uniform(0, 0.2, size=2000), 0, 1)
    t = tune_threshold(y, probs)
    assert t < 0.5
    tuned = evaluate_probabilities(y, probs, t)
    default = evaluate_probabilities(y, probs, 0.5)
    assert tuned["f1"] > default["f1"]


def test_candidate_pipelines_fit_in_cv():
    """Every candidate must run inside stratified CV without leaking/erroring."""
    df = make_frame(n=300, seed=2)
    X, y = df[config.FEATURES], df[config.TARGET]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    for name, spec in build_candidates().items():
        scores = cross_val_score(spec["pipeline"], X, y, cv=cv, scoring="roc_auc")
        assert scores.mean() > 0.7, f"{name} failed to learn an easy signal"
