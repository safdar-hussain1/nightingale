# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Tests for the four-hospital external validation study (nightingale.external).

Task 9's centrepiece: a model trained on Cleveland ONLY (303 rows) is scored,
naively and then with intercept-only recalibration, on the three sites it
never saw (hungarian, switzerland, va). This file covers, in order:

1. calibration_in_the_large recovers a PLANTED logit-scale intercept shift on
   synthetic data (the diagnostic primitive -- both intercept and slope free).
2. _fit_intercept_only recovers a planted shift under its OWN constraint
   (slope fixed at 1) -- the repair primitive, distinct from #1.
3. _calibration_eval_split's 30/70 index arrays are disjoint for every site,
   and assert_calibrator_held_out (imported straight from nightingale.sentinel,
   not reimplemented) both passes on the real split and raises on a
   deliberately corrupted one -- required test 2.
4. The AUC-invariance guarantee: intercept-only recalibration is a strictly
   monotone transform of p, so ROC-AUC before/after must be identical to
   ~1e-9 -- both a pure-synthetic version of this claim and a check against
   the real transfer_study(seed=42) output -- required test 3.
5. transfer_study(seed=42)'s live return value: all three target sites
   present, Cleveland absent (required test 5), naive/recalibrated blocks
   shaped correctly, and the whole call is deterministic across two runs.
6. The COMMITTED models/heart-disease/external.json artifact is well-formed
   (required test 4) -- read directly from disk, mirroring
   tests/test_train_all_artifacts.py's convention for the Task 8 artifacts,
   so this file runs unmodified on a fresh clone once external.json is
   committed.

transfer_study(seed=42) trains one nested-CV model on 303 Cleveland rows --
comparable wall-clock to a single train_condition("breast-cancer") call, not
diabetes-scale -- so the module-scoped fixture below pays that cost at most
twice for the whole file (once for the live-value tests, once more for the
explicit determinism check).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from nightingale.conditions import CONDITIONS
from nightingale.sentinel import LeakageError, assert_calibrator_held_out
from nightingale.external import (
    CALIBRATION_FRACTION,
    LOGIT_CLIP_EPS,
    TARGET_SITES,
    _calibration_eval_split,
    _fit_intercept_only,
    _logit,
    _sigmoid,
    calibration_in_the_large,
    transfer_study,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_JSON_PATH = REPO_ROOT / "models" / "heart-disease" / "external.json"
FIGURE_PATH = REPO_ROOT / "reports" / "figures" / "external-validation.png"

FOUR_METRICS = ("roc_auc", "pr_auc", "brier", "ece")


# ---------------------------------------------------------------------------
# _logit / _sigmoid: clipping guard
# ---------------------------------------------------------------------------


def test_logit_clips_exact_zero_and_one_instead_of_producing_inf():
    z = _logit(np.array([0.0, 1.0, 0.5]))
    assert np.all(np.isfinite(z))
    # Clipped to LOGIT_CLIP_EPS / 1 - LOGIT_CLIP_EPS, so the magnitude is
    # large but bounded -- not literally +/-inf.
    assert z[0] < -10
    assert z[1] > 10
    assert z[2] == pytest.approx(0.0, abs=1e-9)


def test_logit_sigmoid_are_inverses_away_from_the_clip_boundary():
    p = np.array([0.05, 0.2, 0.5, 0.8, 0.95])
    assert np.allclose(_sigmoid(_logit(p)), p, atol=1e-9)


def test_logit_clip_epsilon_is_documented_and_small():
    # A sanity floor on the constant itself, not a tuned value: it must be
    # small enough that clipping is a boundary-only guard, not something
    # that perturbs ordinary mid-range probabilities.
    assert 0 < LOGIT_CLIP_EPS < 1e-3


# ---------------------------------------------------------------------------
# calibration_in_the_large: required test 1 (planted intercept)
# ---------------------------------------------------------------------------


def _synthetic_shifted_prevalence(n: int, seed: int, shift: float):
    """A well-calibrated base model, then a naive-probability report shifted
    by a known logit-scale offset -- exactly the "prevalence mismatch"
    calibration_in_the_large is meant to diagnose.

    y is drawn from the TRUE (unshifted) probabilities, p_reported is the
    shifted probabilities a naively-transferred model would output. Fitting
    calibration_in_the_large(y, p_reported) with slope pinned near 1 should
    recover intercept ~= -shift (see the module docstring's derivation).
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    true_logit = 1.4 * x - 0.2
    p_true = _sigmoid(true_logit)
    y = (rng.uniform(size=n) < p_true).astype(int)
    p_reported = _sigmoid(true_logit + shift)
    return y, p_reported


def test_calibration_in_the_large_recovers_planted_intercept_shift():
    y, p = _synthetic_shifted_prevalence(6000, seed=0, shift=0.9)

    intercept, slope = calibration_in_the_large(y, p)

    assert abs(intercept - (-0.9)) < 0.15
    assert abs(slope - 1.0) < 0.15


def test_calibration_in_the_large_recovers_a_different_planted_shift():
    # A second, differently-signed shift -- guards against a fix that only
    # happens to work for one sign (e.g. an accidental abs()).
    y, p = _synthetic_shifted_prevalence(6000, seed=1, shift=-1.3)

    intercept, slope = calibration_in_the_large(y, p)

    assert abs(intercept - 1.3) < 0.15
    assert abs(slope - 1.0) < 0.15


def test_calibration_in_the_large_near_zero_intercept_for_well_calibrated_p():
    rng = np.random.default_rng(2)
    n = 6000
    logit_p = rng.normal(0, 1.2, n)
    p = _sigmoid(logit_p)
    y = (rng.uniform(size=n) < p).astype(int)

    intercept, slope = calibration_in_the_large(y, p)

    assert abs(intercept) < 0.1
    assert abs(slope - 1.0) < 0.1


# ---------------------------------------------------------------------------
# _fit_intercept_only: the repair primitive (slope fixed at 1)
# ---------------------------------------------------------------------------


def test_fit_intercept_only_recovers_planted_shift():
    y, p = _synthetic_shifted_prevalence(6000, seed=3, shift=1.1)

    c = _fit_intercept_only(y, p)

    assert abs(c - (-1.1)) < 0.15


def test_fit_intercept_only_near_zero_for_already_calibrated_p():
    rng = np.random.default_rng(4)
    n = 6000
    logit_p = rng.normal(0, 1.0, n)
    p = _sigmoid(logit_p)
    y = (rng.uniform(size=n) < p).astype(int)

    c = _fit_intercept_only(y, p)

    assert abs(c) < 0.1


# ---------------------------------------------------------------------------
# AUC invariance under intercept-only recalibration (required test 3, pure)
# ---------------------------------------------------------------------------


def test_intercept_only_recalibration_is_auc_invariant_synthetic():
    """A monotone transform of p can never change ROC-AUC (a rank statistic).

    This is the pure-mechanism version of required test 3: fits an
    intercept on one synthetic sample, applies it, and checks ROC-AUC is
    unchanged to ~1e-9 with NO training pipeline involved -- isolating the
    claim from transfer_study's plumbing so a failure here always points at
    _fit_intercept_only/_logit/_sigmoid, never at site scoring or splitting.
    """
    rng = np.random.default_rng(5)
    n = 500
    logit_p = rng.normal(0, 1.5, n)
    p = _sigmoid(logit_p)
    y = (rng.uniform(size=n) < _sigmoid(0.7 * logit_p + 0.5)).astype(int)

    c = _fit_intercept_only(y, p)
    p_recal = _sigmoid(_logit(p) + c)

    auc_before = roc_auc_score(y, p)
    auc_after = roc_auc_score(y, p_recal)

    assert auc_after == pytest.approx(auc_before, abs=1e-9)


def test_intercept_only_recalibration_changes_brier_when_shift_is_planted():
    # Companion to the AUC-invariance test: recalibration is NOT a no-op in
    # general -- it must actually change Brier (a magnitude-sensitive
    # metric) when there is a real shift to correct, or the "repair" would
    # be doing nothing.
    from sklearn.metrics import brier_score_loss

    y, p = _synthetic_shifted_prevalence(4000, seed=6, shift=1.2)
    c = _fit_intercept_only(y, p)
    p_recal = _sigmoid(_logit(p) + c)

    assert brier_score_loss(y, p_recal) < brier_score_loss(y, p)


# ---------------------------------------------------------------------------
# _calibration_eval_split: disjointness (required test 2, pure)
# ---------------------------------------------------------------------------


def test_calibration_eval_split_indices_are_disjoint():
    rng = np.random.default_rng(7)
    y = (rng.uniform(size=250) < 0.4).astype(int)

    cal_idx, eval_idx = _calibration_eval_split(y, seed=42)

    assert set(cal_idx.tolist()).isdisjoint(set(eval_idx.tolist()))
    # assert_calibrator_held_out is the SAME guard transfer_study calls live
    # -- proving it passes here is proving the real split satisfies it.
    assert assert_calibrator_held_out(cal_idx, eval_idx) is None


def test_calibration_eval_split_covers_every_row_exactly_once():
    rng = np.random.default_rng(8)
    n = 180
    y = (rng.uniform(size=n) < 0.6).astype(int)

    cal_idx, eval_idx = _calibration_eval_split(y, seed=1)

    assert sorted(cal_idx.tolist() + eval_idx.tolist()) == list(range(n))


def test_calibration_eval_split_is_approximately_30_70():
    rng = np.random.default_rng(9)
    n = 1000
    y = (rng.uniform(size=n) < 0.5).astype(int)

    cal_idx, eval_idx = _calibration_eval_split(y, seed=2, cal_fraction=CALIBRATION_FRACTION)

    assert abs(len(cal_idx) / n - CALIBRATION_FRACTION) < 0.02
    assert abs(len(eval_idx) / n - (1 - CALIBRATION_FRACTION)) < 0.02


def test_calibration_eval_split_is_seed_deterministic():
    rng = np.random.default_rng(10)
    y = (rng.uniform(size=200) < 0.4).astype(int)

    cal_a, eval_a = _calibration_eval_split(y, seed=99)
    cal_b, eval_b = _calibration_eval_split(y, seed=99)

    assert np.array_equal(cal_a, cal_b)
    assert np.array_equal(eval_a, eval_b)


def test_deliberately_overlapped_calibration_eval_indices_raise():
    """The guard actually fires on a corrupted split -- required test 2's
    second half ("assert the guard raises if you deliberately overlap
    them"). Takes a REAL split from _calibration_eval_split and corrupts it
    by leaking one calibration row into the evaluation set, rather than
    using two arbitrary hand-written arrays -- so this is specifically
    exercising the leak this module's split could in principle produce.
    """
    rng = np.random.default_rng(11)
    y = (rng.uniform(size=150) < 0.5).astype(int)
    cal_idx, eval_idx = _calibration_eval_split(y, seed=3)

    corrupted_eval_idx = np.concatenate([eval_idx, cal_idx[:1]])

    with pytest.raises(LeakageError):
        assert_calibrator_held_out(cal_idx, corrupted_eval_idx)


# ---------------------------------------------------------------------------
# transfer_study(seed=42): live return value
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def transfer_result():
    return transfer_study(seed=42)


def test_target_sites_constant_excludes_cleveland():
    assert set(TARGET_SITES) == {"hungarian", "switzerland", "va"}
    assert "cleveland" not in TARGET_SITES


def test_transfer_study_evaluates_exactly_the_three_non_cleveland_sites(transfer_result):
    # Required test 5: Cleveland is trained on, never evaluated as a target.
    assert set(transfer_result["sites"]) == {"hungarian", "switzerland", "va"}
    assert "cleveland" not in transfer_result["sites"]


def test_transfer_study_site_row_counts_match_the_cleaned_data(transfer_result):
    cleaned_path = REPO_ROOT / "data" / "cleaned" / "heart-disease.csv.gz"
    heart = pd.read_csv(cleaned_path, compression="gzip")
    for site in TARGET_SITES:
        expected_n = int((heart["site"] == site).sum())
        assert transfer_result["sites"][site]["n"] == expected_n


def test_transfer_study_every_site_has_naive_and_recalibrated_blocks(transfer_result):
    for site in TARGET_SITES:
        block = transfer_result["sites"][site]
        assert "naive" in block
        assert "recalibrated" in block
        for metric in FOUR_METRICS:
            point, lo, hi = block["naive"][metric]
            assert lo <= point <= hi
            point, lo, hi = block["recalibrated"][metric]
            assert lo <= point <= hi
            point, lo, hi = block["recalibrated"]["before"][metric]
            assert lo <= point <= hi


def test_transfer_study_calibration_in_the_large_present_for_naive_and_recalibrated(
    transfer_result,
):
    for site in TARGET_SITES:
        block = transfer_result["sites"][site]
        for sub in (block["naive"], block["recalibrated"], block["recalibrated"]["before"]):
            assert "calibration_intercept" in sub
            assert "calibration_slope" in sub
            assert np.isfinite(sub["calibration_intercept"])
            assert np.isfinite(sub["calibration_slope"])


def test_transfer_study_recalibration_calibration_and_evaluation_rows_are_disjoint(monkeypatch):
    """Integration-level version of required test 2: spies on the REAL guard
    call inside transfer_study's own code path (not a hand-constructed
    stand-in) and asserts disjointness directly on the captured index
    arrays, once per target site.
    """
    import nightingale.external as external_module

    calls = []
    real_guard = external_module.assert_calibrator_held_out

    def spy_guard(cal_idx, eval_idx):
        calls.append((np.array(list(cal_idx)), np.array(list(eval_idx))))
        return real_guard(cal_idx, eval_idx)

    monkeypatch.setattr(external_module, "assert_calibrator_held_out", spy_guard)

    result = transfer_study(seed=42)

    # At least one held-out call per target site's recalibration split (the
    # Cleveland-training loop also calls the same guard per outer fold, so
    # this asserts a lower bound, not an exact count).
    assert len(calls) >= len(TARGET_SITES)
    for cal_idx, eval_idx in calls:
        assert set(cal_idx.tolist()).isdisjoint(set(eval_idx.tolist()))

    assert set(result["sites"]) == {"hungarian", "switzerland", "va"}


def test_transfer_study_auc_identical_before_and_after_recalibration(transfer_result):
    """Required test 3, against the real pipeline output: for every target
    site, ROC-AUC on the 70% evaluation rows must be identical before and
    after intercept-only recalibration to ~1e-9. This fails if someone
    accidentally refits the slope, shuffles rows, or evaluates before/after
    on different subsets.
    """
    for site in TARGET_SITES:
        recal = transfer_result["sites"][site]["recalibrated"]
        auc_before = recal["before"]["roc_auc"][0]
        auc_after = recal["roc_auc"][0]
        assert auc_after == pytest.approx(auc_before, abs=1e-9)


def test_transfer_study_is_deterministic_across_two_independent_calls():
    result_a = transfer_study(seed=42)
    result_b = transfer_study(seed=42)

    # Byte-reproducible via JSON round-trip, matching Task 8's own
    # determinism discipline -- no wall-clock or unseeded randomness inside.
    assert json.dumps(result_a, sort_keys=True) == json.dumps(result_b, sort_keys=True)


def test_transfer_study_recalibration_split_sizes_are_documented_and_consistent(transfer_result):
    for site in TARGET_SITES:
        recal = transfer_result["sites"][site]["recalibrated"]
        n_site = transfer_result["sites"][site]["n"]
        assert recal["n_calibration"] + recal["n_evaluation"] == n_site
        assert recal["n_calibration"] > 0
        assert recal["n_evaluation"] > 0


def test_cleveland_training_block_is_present_and_well_formed(transfer_result):
    cleveland = transfer_result["cleveland_training"]
    assert cleveland["n"] == CONDITIONS["heart-disease"].n_rows - sum(
        transfer_result["sites"][s]["n"] for s in TARGET_SITES
    )
    assert set(cleveland["chosen_params"]) == {"max_depth", "n_estimators", "learning_rate"}
    assert cleveland["calibration_method"] in {"sigmoid", "isotonic"}
    assert "site" not in cleveland["feature_names"]
    assert "target" not in cleveland["feature_names"]


# ---------------------------------------------------------------------------
# Committed models/heart-disease/external.json (required test 4)
# ---------------------------------------------------------------------------
#
# Reads the file straight off disk -- never calls transfer_study() -- so
# this section runs on a fresh clone once external.json is committed, the
# same convention tests/test_train_all_artifacts.py uses for Task 8's
# artifacts.


def _load_external_json() -> dict:
    return json.loads(EXTERNAL_JSON_PATH.read_text())


def test_external_json_exists():
    assert EXTERNAL_JSON_PATH.is_file()


def test_external_json_all_three_target_sites_present_cleveland_absent():
    data = _load_external_json()
    assert set(data["sites"]) == {"hungarian", "switzerland", "va"}
    assert "cleveland" not in data["sites"]


@pytest.mark.parametrize("site", ["hungarian", "switzerland", "va"])
def test_external_json_site_has_naive_and_recalibrated_metric_triples(site):
    data = _load_external_json()
    block = data["sites"][site]
    assert "naive" in block
    assert "recalibrated" in block

    for metric in FOUR_METRICS:
        for sub in (block["naive"], block["recalibrated"], block["recalibrated"]["before"]):
            value = sub[metric]
            assert isinstance(value, list)
            assert len(value) == 3
            point, lo, hi = value
            for v in (point, lo, hi):
                assert isinstance(v, (int, float))
            assert lo <= point <= hi


@pytest.mark.parametrize("site", ["hungarian", "switzerland", "va"])
def test_external_json_auc_identical_before_and_after_recalibration(site):
    data = _load_external_json()
    recal = data["sites"][site]["recalibrated"]
    assert recal["before"]["roc_auc"][0] == pytest.approx(recal["roc_auc"][0], abs=1e-9)


def test_external_json_no_generated_utc_field_anywhere():
    # Same reproducibility discipline as metrics.json (Task 8): no
    # wall-clock timestamp baked into a byte-reproducible artifact.
    raw_text = EXTERNAL_JSON_PATH.read_text()
    assert "generated_utc" not in raw_text


def test_external_json_seed_and_epsilon_are_documented():
    data = _load_external_json()
    assert data["seed"] == 42
    assert data["clipping_epsilon"] == pytest.approx(LOGIT_CLIP_EPS)
    assert data["calibration_split_fraction"] == pytest.approx(CALIBRATION_FRACTION)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def test_external_validation_figure_exists_and_is_a_real_png():
    assert FIGURE_PATH.is_file()
    assert FIGURE_PATH.stat().st_size > 1000
    with open(FIGURE_PATH, "rb") as f:
        assert f.read(8) == _PNG_MAGIC
