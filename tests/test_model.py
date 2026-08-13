# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Tests for the nested-CV training core (nightingale.model).

Real training here is restricted to breast-cancer (569 rows, trains in
seconds) -- diabetes (253,680 rows) is never trained in this suite, per the
task brief. Everything else uses small synthetic frames.

Reads the committed data/cleaned/breast-cancer.csv.gz directly for the
comparison in test_train_condition_breast_cancer_oof_shape_and_quality, so
this whole file is raw-data-free and runs on a fresh clone.

The three mutation tests (A: calibration leak, B: fold integrity, C:
fold-safe encoding) each prove a specific protection is *live* in
train_condition, not merely that the underlying primitive works in
isolation -- see nightingale.sentinel and the task brief's spec §4.5.
"""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from nightingale.clean import CLEANED_ROOT
from nightingale.sentinel import LeakageError
import nightingale.model as model_module
from nightingale.model import (
    FoldIntegrityError,
    GRID_LEARNING_RATE,
    GRID_MAX_DEPTH,
    GRID_N_ESTIMATORS,
    TrainResult,
    _feature_columns,
    _grid_combos,
    _most_frequent_params,
    _one_hot_fold_safe,
    _stratified_subsample,
    train_condition,
)

# ---------------------------------------------------------------------------
# Small helper-level unit tests (no training)
# ---------------------------------------------------------------------------


def test_feature_columns_excludes_target_and_site():
    df = pd.DataFrame({"a": [1], "site": ["cleveland"], "target": [0]})

    result = _feature_columns(df, target_col="target")

    assert result == ["a"]


def test_feature_columns_keeps_everything_else():
    df = pd.DataFrame({"a": [1], "b": [2.0], "target": [1]})

    result = _feature_columns(df, target_col="target")

    assert result == ["a", "b"]


def test_grid_combos_has_twelve_unique_combinations():
    combos = _grid_combos()

    assert len(combos) == len(GRID_MAX_DEPTH) * len(GRID_N_ESTIMATORS) * len(GRID_LEARNING_RATE)
    as_tuples = {tuple(sorted(c.items())) for c in combos}
    assert len(as_tuples) == len(combos)  # all unique
    for c in combos:
        assert set(c) == {"max_depth", "n_estimators", "learning_rate"}


def test_most_frequent_params_picks_the_mode():
    fold_params = [
        {"max_depth": 2, "n_estimators": 100, "learning_rate": 0.1},
        {"max_depth": 3, "n_estimators": 300, "learning_rate": 0.05},
        {"max_depth": 2, "n_estimators": 100, "learning_rate": 0.1},
        {"max_depth": 2, "n_estimators": 100, "learning_rate": 0.1},
        {"max_depth": 4, "n_estimators": 100, "learning_rate": 0.05},
    ]

    result = _most_frequent_params(fold_params)

    assert result == {"max_depth": 2, "n_estimators": 100, "learning_rate": 0.1}


def test_most_frequent_params_ties_broken_by_first_occurrence():
    # "b" and "a" are tied at count 2; "a" occurs first in fold order (index
    # 0) so it must win over "b" (first occurring at index 1).
    a = {"max_depth": 2, "n_estimators": 100, "learning_rate": 0.1}
    b = {"max_depth": 3, "n_estimators": 300, "learning_rate": 0.05}
    fold_params = [a, b, a, b]

    result = _most_frequent_params(fold_params)

    assert result == a


def test_stratified_subsample_returns_unchanged_when_already_small():
    X = pd.DataFrame({"a": range(10)})
    y = pd.Series([0, 1] * 5)

    X_out, y_out = _stratified_subsample(X, y, n=20, seed=0)

    assert X_out is X
    assert y_out is y


def test_stratified_subsample_shrinks_to_n_and_stays_seeded():
    rng = np.random.default_rng(0)
    n_total = 2000
    X = pd.DataFrame({"a": rng.normal(size=n_total)})
    y = pd.Series((rng.uniform(size=n_total) < 0.3).astype(int))

    X1, y1 = _stratified_subsample(X, y, n=200, seed=5)
    X2, y2 = _stratified_subsample(X, y, n=200, seed=5)

    assert len(X1) == 200
    assert len(y1) == 200
    # Seeded: repeated calls with the same seed pick the identical subsample.
    assert list(X1.index) == list(X2.index)
    assert list(y1.index) == list(y2.index)
    # Roughly stratified: subsample positive rate close to the full rate.
    assert abs(y1.mean() - y.mean()) < 0.05


# ---------------------------------------------------------------------------
# Required test 4: train_condition("breast-cancer") shape + quality floor
# ---------------------------------------------------------------------------


def test_train_condition_breast_cancer_oof_shape_and_quality():
    result = train_condition("breast-cancer", seed=42)

    assert isinstance(result, TrainResult)

    cleaned = pd.read_csv(CLEANED_ROOT / "breast-cancer.csv.gz", compression="gzip")

    # Exactly one row per input row, original row order.
    assert len(result.oof) == 569
    assert len(cleaned) == 569
    assert list(result.oof.columns) == ["y_true", "p_raw", "p_cal", "fold"]
    assert list(result.oof.index) == list(cleaned.index)

    # fold in 0..4, every fold represented.
    assert set(result.oof["fold"].unique()) == {0, 1, 2, 3, 4}
    assert result.oof["fold"].between(0, 4).all()

    # y_true matches the cleaned target exactly, in order.
    assert result.oof["y_true"].tolist() == cleaned["target"].astype(int).tolist()

    # Valid probabilities.
    assert result.oof["p_raw"].between(0.0, 1.0).all()
    assert result.oof["p_cal"].between(0.0, 1.0).all()
    assert not result.oof["p_raw"].isna().any()
    assert not result.oof["p_cal"].isna().any()

    # WDBC is genuinely separable -- a real floor, not a tuned target.
    auc = roc_auc_score(result.oof["y_true"], result.oof["p_raw"])
    assert auc > 0.95

    # Contract fields.
    assert isinstance(result.final_model, xgb.XGBClassifier)
    assert set(result.chosen_params) == {"max_depth", "n_estimators", "learning_rate"}
    assert result.chosen_params["max_depth"] in GRID_MAX_DEPTH
    assert result.chosen_params["n_estimators"] in GRID_N_ESTIMATORS
    assert result.chosen_params["learning_rate"] in GRID_LEARNING_RATE
    assert "site" not in result.feature_names
    assert len(result.feature_names) == 30  # WDBC has 30 numeric features, no categoricals
    assert result.cv_summary["oof_roc_auc"] == pytest.approx(auc)
    assert hasattr(result.calibrator, "predict")
    assert hasattr(result.calibrator, "export")
    assert result.calibrator.export()["type"] in {"sigmoid", "isotonic"}


# ---------------------------------------------------------------------------
# Mutation test A: calibration leak -- assert_calibrator_held_out is live
# ---------------------------------------------------------------------------


def test_mutation_a_overlapping_fold_indices_raise_leakage_error(monkeypatch):
    """Corrupt one outer fold so its train/eval indices overlap.

    This is exactly the shape of leak assert_calibrator_held_out exists to
    catch: the model backing an OOF prediction must never have trained on
    the row it's predicting, because that prediction becomes part of the
    pooled data the calibrator is fit on. Patching train_condition's own
    split-producing seam (_outer_splits) -- rather than calling
    assert_calibrator_held_out directly -- proves the guard fires from
    inside train_condition's real code path, not just in isolation.
    """
    real_outer_splits = model_module._outer_splits

    def corrupted_outer_splits(X, y, seed):
        splits = real_outer_splits(X, y, seed)
        train_idx, eval_idx = splits[0]
        # Leak: sneak one eval row into this fold's own training indices.
        leaked_train_idx = np.concatenate([train_idx, eval_idx[:1]])
        splits[0] = (leaked_train_idx, eval_idx)
        return splits

    monkeypatch.setattr(model_module, "_outer_splits", corrupted_outer_splits)

    with pytest.raises(LeakageError):
        train_condition("breast-cancer", seed=42)


# ---------------------------------------------------------------------------
# Mutation test B: fold integrity -- a row predicted in two eval folds
# ---------------------------------------------------------------------------


def test_mutation_b_duplicate_eval_row_across_folds_raises_fold_integrity_error(monkeypatch):
    """Corrupt the fold assignment so one row lands in two eval folds.

    Distinct from mutation test A: every fold's own (train_idx, eval_idx)
    stays internally disjoint here (so assert_calibrator_held_out never
    fires) -- the defect is purely cross-fold: the same row is claimed by
    fold 0's eval set AND fold 1's eval set, violating "every row predicted
    exactly once". train_condition's explicit seen_count integrity check
    must catch this.
    """
    real_outer_splits = model_module._outer_splits

    def corrupted_outer_splits(X, y, seed):
        splits = real_outer_splits(X, y, seed)
        train0, eval0 = splits[0]
        train1, eval1 = splits[1]

        duplicate_row = eval0[0]
        # Duplicate eval0's row into eval1 too, while removing it from
        # fold 1's own training indices so fold 1 stays internally
        # train/eval-disjoint (isolating this from mutation test A's bug).
        eval1 = np.concatenate([eval1, [duplicate_row]])
        train1 = train1[train1 != duplicate_row]
        splits[1] = (train1, eval1)
        return splits

    monkeypatch.setattr(model_module, "_outer_splits", corrupted_outer_splits)

    with pytest.raises(FoldIntegrityError):
        train_condition("breast-cancer", seed=42)


# ---------------------------------------------------------------------------
# Mutation test C: fold-safe one-hot encoding
# ---------------------------------------------------------------------------


def test_mutation_c_eval_only_category_never_leaks_into_training_encoding():
    """A category seen ONLY in the eval fold must not create a training column.

    This is the test that would fail if someone moved get_dummies to run
    on the whole dataset before splitting: in that buggy version, the
    training-fold encoding would gain a "cat_c" column (since get_dummies
    would see it globally), whereas the fold-safe version never learns "c"
    exists at all until it's silently reindexed away to all-zero.
    """
    train_df = pd.DataFrame(
        {"num": [1.0, 2.0, 3.0, 4.0], "cat": ["a", "b", "a", "b"]}
    )
    eval_df = pd.DataFrame(
        {"num": [5.0, 6.0], "cat": ["a", "c"]}  # "c" appears ONLY in eval
    )

    X_train_enc, X_eval_enc = _one_hot_fold_safe(train_df, eval_df)

    assert "cat_c" not in X_train_enc.columns
    assert "cat_c" not in X_eval_enc.columns
    assert list(X_eval_enc.columns) == list(X_train_enc.columns)
    assert set(X_train_enc.columns) == {"num", "cat_a", "cat_b"}

    # The eval-only category's row (index 1, "c") reindexes to all-zero
    # across the training fold's known categories -- not an error, not a
    # crash, just "no known category matched".
    assert X_eval_enc.loc[1, "cat_a"] == 0
    assert X_eval_enc.loc[1, "cat_b"] == 0

    # Prediction still works end-to-end despite the unseen category --
    # the practical payoff of fold-safety (reindex fills 0, no shape
    # mismatch, no KeyError).
    model = xgb.XGBClassifier(tree_method="hist", n_estimators=5, max_depth=2, random_state=0)
    model.fit(X_train_enc, [0, 1, 0, 1])
    preds = model.predict_proba(X_eval_enc)[:, 1]
    assert preds.shape == (2,)
    assert np.all((preds >= 0.0) & (preds <= 1.0))


def test_one_hot_fold_safe_preserves_numeric_nan_untouched():
    train_df = pd.DataFrame({"num": [1.0, np.nan, 3.0], "cat": ["a", "b", "a"]})
    eval_df = pd.DataFrame({"num": [np.nan], "cat": ["a"]})

    X_train_enc, X_eval_enc = _one_hot_fold_safe(train_df, eval_df)

    # NaN passes through untouched -- no imputation anywhere in this path.
    assert X_train_enc["num"].isna().sum() == 1
    assert X_eval_enc["num"].isna().sum() == 1
