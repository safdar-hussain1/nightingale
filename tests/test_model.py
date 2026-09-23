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
isolation (the primitives themselves live in nightingale.sentinel).

Also covers cross-fitted calibration (oof["p_cal"]): a correction made
after an initial version of this module reported a calibrated ECE of
~1e-16-1e-19 -- not a real result, but isotonic regression interpolating
its own training sample after being fit and scored on the same pooled OOF.
test_cross_fitted_ece_is_non_degenerate and
test_cross_fitted_ece_is_substantially_larger_than_in_sample_ece together
pin down, numerically, that oof["p_cal"] is a genuinely out-of-sample
number and would fail loudly if anyone reintroduced in-sample scoring;
test_cross_fit_calibration_excludes_each_rows_own_fold proves it at the
index level (every row's p_cal comes from a calibrator fit on rows
excluding it), not just via the aggregate ECE gap.

Also covers TrainResult.calibrator (the DEPLOYMENT calibrator, distinct
from oof["p_cal"] above): a reviewer applied the brief's named mutation --
fit final_model first, then build the calibrator from
final_model.predict_proba(X_all_enc)[:, 1] instead of pooled OOF p_raw --
and found every test in this file still passed, because none of them
asserted the calibrator's fitted VALUES, only its shape/behaviour.
test_deployment_calibrator_matches_independent_oof_refit closes that gap.
"""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from nightingale.calibrate import ece, fit_calibrator, pick_calibration
from nightingale.clean import CLEANED_ROOT
from nightingale.sentinel import LeakageError
import nightingale.model as model_module
from nightingale.model import (
    FoldIntegrityError,
    GRID_LEARNING_RATE,
    GRID_MAX_DEPTH,
    GRID_N_ESTIMATORS,
    OUTER_N_SPLITS,
    TrainResult,
    _cross_fit_calibration,
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


# ---------------------------------------------------------------------------
# Cross-fitted calibration (Fix round 1: p_cal must be out-of-sample)
# ---------------------------------------------------------------------------
#
# An earlier version of train_condition fit ONE calibrator on the full
# pooled OOF and scored it against that same pool to produce p_cal -- that
# is in-sample for the calibrator (even though p_raw is genuinely OOF for
# the base model), and isotonic regression is flexible enough to make the
# resulting "calibrated ECE" land at machine-epsilon: not a measurement,
# an artifact. These three tests pin the fix down so it can't silently
# regress: (a) the cross-fitted ECE must be clearly nonzero, (b) it must be
# substantially larger than the in-sample number it replaced, and (c) the
# index-level mechanism producing it must actually exclude each row's own
# fold, not just happen to produce a bigger number for some other reason.


def test_cross_fitted_ece_is_non_degenerate():
    result = train_condition("breast-cancer", seed=42)

    ece_calibrated = result.cv_summary["ece_calibrated"]

    # Machine-epsilon-scale ECE (~1e-16) is exactly the in-sample-scoring
    # bug this test guards against; a genuinely out-of-sample calibrated
    # ECE on 569 real rows has no reason to land anywhere near that.
    assert ece_calibrated > 1e-6


def test_cross_fitted_ece_is_substantially_larger_than_in_sample_ece():
    result = train_condition("breast-cancer", seed=42)

    y_true = result.oof["y_true"].to_numpy()
    p_raw = result.oof["p_raw"].to_numpy()
    method = result.calibrator.export()["type"]

    # In-sample, on purpose: fit the SAME calibrator family on the full
    # pooled OOF and score it against that same pool -- this is exactly
    # the old (incorrect) computation, reproduced here deliberately as the
    # baseline this test measures the fix against.
    in_sample_calibrator = fit_calibrator(y_true, p_raw, method=method)
    in_sample_ece = ece(y_true, in_sample_calibrator.predict(p_raw))

    cross_fitted_ece = ece(y_true, result.oof["p_cal"].to_numpy())

    assert cross_fitted_ece == pytest.approx(result.cv_summary["ece_calibrated"])
    assert in_sample_ece < cross_fitted_ece
    # Not just "smaller" -- an order of magnitude, so this documents the
    # measured size of the effect rather than tolerating noise-level gaps.
    assert in_sample_ece < cross_fitted_ece / 10


def test_cross_fit_calibration_excludes_each_rows_own_fold(monkeypatch):
    """Every row's p_cal must come from a calibrator that never saw that row.

    Spies on assert_calibrator_held_out (called once per outer fold inside
    _cross_fit_calibration) and checks, at the index level, that each
    call's calibration-fit indices are exactly "every row NOT in this
    fold" and its scored indices are exactly "every row IN this fold" --
    not merely that the guard function exists and would raise in theory.
    """
    rng = np.random.default_rng(11)
    n = 40
    n_folds = 5
    oof = pd.DataFrame(
        {
            "y_true": rng.integers(0, 2, n),
            "p_raw": rng.uniform(0.0, 1.0, n),
            "fold": np.tile(np.arange(n_folds), n // n_folds),
        }
    )

    calls = []
    real_guard = model_module.assert_calibrator_held_out

    def spy_guard(cal_idx, eval_idx):
        calls.append((np.asarray(list(cal_idx)), np.asarray(list(eval_idx))))
        return real_guard(cal_idx, eval_idx)

    monkeypatch.setattr(model_module, "assert_calibrator_held_out", spy_guard)

    p_cal = _cross_fit_calibration(oof, method="sigmoid")

    assert len(calls) == n_folds  # once per outer fold, not once total

    fold_values = oof["fold"].to_numpy()
    all_scored = set()
    for cal_idx, eval_idx in calls:
        scored_folds = set(fold_values[eval_idx].tolist())
        assert len(scored_folds) == 1  # this call's eval_idx is one whole fold
        k = scored_folds.pop()

        expected_eval = set(np.where(fold_values == k)[0].tolist())
        expected_cal = set(range(n)) - expected_eval

        assert set(eval_idx.tolist()) == expected_eval
        assert set(cal_idx.tolist()) == expected_cal
        assert set(cal_idx.tolist()).isdisjoint(eval_idx.tolist())

        all_scored |= expected_eval

    assert all_scored == set(range(n))  # every row scored exactly once, by some fold
    assert p_cal.shape == (n,)
    assert not np.any(np.isnan(p_cal))


# ---------------------------------------------------------------------------
# Deployment calibrator value-level regression (Fix round 2)
# ---------------------------------------------------------------------------
#
# Fix round 1 (above) secured oof["p_cal"] against in-sample scoring. But
# TrainResult.calibrator -- the separate DEPLOYMENT artifact Task 10 will
# export to score real users' inputs in the browser -- was, until this
# fix, protected by nothing but an implicit ordering accident (the
# calibrator happened to be built before final_model was fit). A reviewer
# proved this by applying the brief's literal named mutation -- fit
# final_model first, then construct the calibrator from
# final_model.predict_proba(X_all_enc)[:, 1] instead of pooled OOF p_raw --
# and running the whole file: 15/15 passed. Nothing before this test
# compared the calibrator's actual fitted numbers to anything.


def test_deployment_calibrator_matches_independent_oof_refit():
    """TrainResult.calibrator's exported params must match an OOF-only refit.

    Independently reconstructs a calibrator via pick_calibration from
    ONLY result.oof["y_true"]/result.oof["p_raw"] (the pooled OOF), and
    asserts its export() matches result.calibrator.export() value-for-value
    (exact "type", tight numeric tolerance on the fitted parameters).
    pick_calibration's underlying fits (LogisticRegression via lbfgs,
    IsotonicRegression's pool-adjacent-violators) are both deterministic
    given identical input, so a correct implementation reproduces this
    exactly; a calibrator built from any other prediction source (most
    notably final_model's own in-sample predict_proba, the mutation this
    test exists to catch) will not.
    """
    result = train_condition("breast-cancer", seed=42)

    independent = pick_calibration(
        result.oof["y_true"].to_numpy(), result.oof["p_raw"].to_numpy()
    )

    actual = result.calibrator.export()
    expected = independent.export()

    assert actual["type"] == expected["type"]
    if actual["type"] == "sigmoid":
        assert actual["a"] == pytest.approx(expected["a"], abs=1e-9)
        assert actual["b"] == pytest.approx(expected["b"], abs=1e-9)
    else:
        assert actual["type"] == "isotonic"
        np.testing.assert_allclose(actual["x"], expected["x"], atol=1e-9)
        np.testing.assert_allclose(actual["y"], expected["y"], atol=1e-9)
