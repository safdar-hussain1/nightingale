# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Tests for evaluation (nightingale.evaluate): bootstrap CIs, DCA, subgroup audit.

Covers: metric_ci's interval containing the point estimate, shrinking with
n, and seed determinism; net_benefit against a hand-computed 6-row example
(arithmetic worked out in comments, never via the function under test) plus
the treat_none/treat_all closed-form identities; subgroup_audit refusing to
report a numeric AUC for a group under min_n (NaN, not zero, not omitted);
and evaluate_oof on a real trained condition (breast-cancer).

Nothing here needs data/raw/ -- train_condition reads only the committed
data/cleaned/breast-cancer.csv.gz, so this whole file runs on a fresh
clone with no fetch/clean step required first.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

import nightingale.evaluate as evaluate_module
from nightingale.evaluate import evaluate_oof, metric_ci, net_benefit, subgroup_audit
from nightingale.model import train_condition

# ---------------------------------------------------------------------------
# metric_ci
# ---------------------------------------------------------------------------


def _synthetic_calibrated(n: int, seed: int):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(size=n) < p).astype(int)
    return y, p


def test_metric_ci_interval_contains_point_estimate():
    y, p = _synthetic_calibrated(300, seed=0)

    point, lo, hi = metric_ci(y, p, roc_auc_score, n_boot=500, seed=1)

    assert lo <= point <= hi


def test_metric_ci_interval_width_shrinks_with_n():
    y200, p200 = _synthetic_calibrated(200, seed=10)
    y2000, p2000 = _synthetic_calibrated(2000, seed=10)

    _, lo200, hi200 = metric_ci(y200, p200, roc_auc_score, n_boot=1000, seed=0)
    _, lo2000, hi2000 = metric_ci(y2000, p2000, roc_auc_score, n_boot=1000, seed=0)

    assert (hi2000 - lo2000) < (hi200 - lo200)


def test_metric_ci_same_seed_gives_identical_bounds():
    y, p = _synthetic_calibrated(200, seed=2)

    _, lo_a, hi_a = metric_ci(y, p, roc_auc_score, n_boot=500, seed=42)
    _, lo_b, hi_b = metric_ci(y, p, roc_auc_score, n_boot=500, seed=42)

    assert lo_a == lo_b
    assert hi_a == hi_b


# ---------------------------------------------------------------------------
# net_benefit: hand-computed 6-row example
# ---------------------------------------------------------------------------


def test_net_benefit_matches_hand_computed_six_row_example_exactly():
    # 6 labelled points:
    #   idx:  0    1    2    3    4    5
    #   y:    1    1    1    0    0    0
    #   p:  0.9  0.7  0.4  0.6  0.3  0.1
    # prevalence = 3/6 = 0.5
    #
    # threshold pt=0.5: predicted positive (p >= 0.5) = idx {0, 1, 3}
    #   TP = y==1 among {0,1,3} = {0,1} -> TP=2
    #   FP = y==0 among {0,1,3} = {3}   -> FP=1
    #   odds = 0.5/(1-0.5) = 1.0
    #   NB_model = TP/n - FP/n*odds = 2/6 - 1/6*1.0 = 0.16666666666666666
    #   NB_treat_all = prevalence - (1-prevalence)*odds = 0.5 - 0.5*1.0 = 0.0
    #
    # threshold pt=0.2: predicted positive (p >= 0.2) = idx {0, 1, 2, 3, 4}
    #   TP = y==1 among these = {0,1,2} -> TP=3
    #   FP = y==0 among these = {3,4}   -> FP=2
    #   odds = 0.2/(1-0.2) = 0.25
    #   NB_model = 3/6 - 2/6*0.25 = 0.5 - 0.08333333333333333 = 0.41666666666666663
    #   NB_treat_all = 0.5 - 0.5*0.25 = 0.375
    y = [1, 1, 1, 0, 0, 0]
    p = [0.9, 0.7, 0.4, 0.6, 0.3, 0.1]
    thresholds = np.array([0.5, 0.2])

    result = net_benefit(y, p, thresholds)

    assert list(result.columns) == ["threshold", "model", "treat_all", "treat_none"]
    assert len(result) == 2

    row_05 = result[result["threshold"] == 0.5].iloc[0]
    assert row_05["model"] == pytest.approx(0.16666666666666666, abs=1e-12)
    assert row_05["treat_all"] == pytest.approx(0.0, abs=1e-12)
    assert row_05["treat_none"] == pytest.approx(0.0, abs=1e-12)

    row_02 = result[result["threshold"] == 0.2].iloc[0]
    assert row_02["model"] == pytest.approx(0.41666666666666663, abs=1e-12)
    assert row_02["treat_all"] == pytest.approx(0.375, abs=1e-12)
    assert row_02["treat_none"] == pytest.approx(0.0, abs=1e-12)


def test_net_benefit_treat_none_zero_and_treat_all_matches_closed_form():
    rng = np.random.default_rng(5)
    n = 500
    y = (rng.uniform(size=n) < 0.3).astype(int)
    p = rng.uniform(0.0, 1.0, n)
    thresholds = np.array([0.05, 0.1, 0.3, 0.5, 0.7, 0.9])
    prevalence = y.mean()

    result = net_benefit(y, p, thresholds)

    assert (result["treat_none"] == 0.0).all()

    # Independent computation of the closed-form treat_all identity --
    # never calls net_benefit() to derive the expected values.
    expected_treat_all = prevalence - (1.0 - prevalence) * (thresholds / (1.0 - thresholds))
    np.testing.assert_allclose(result["treat_all"].to_numpy(), expected_treat_all)


# ---------------------------------------------------------------------------
# net_benefit: threshold-domain guard (fix round 1, finding 1)
# ---------------------------------------------------------------------------


def test_net_benefit_rejects_thresholds_outside_open_unit_interval():
    # np.linspace(0, 1, 11) is the obvious way to build a DCA grid, and it
    # always includes both endpoints -- pt=1.0 makes pt/(1-pt) a division
    # by zero (model=NaN, treat_all=-inf on that row); pt=0.0 is likewise
    # outside the open interval (0, 1) the function's contract requires.
    y = [1, 0, 1, 0]
    p = [0.9, 0.1, 0.6, 0.4]

    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        net_benefit(y, p, np.linspace(0.0, 1.0, 11))


def test_net_benefit_rejects_threshold_above_one():
    y = [1, 0, 1, 0]
    p = [0.9, 0.1, 0.6, 0.4]

    with pytest.raises(ValueError, match=r"1\.5"):
        net_benefit(y, p, [0.2, 1.5])


def test_net_benefit_valid_grid_never_produces_nan_or_inf():
    rng = np.random.default_rng(3)
    n = 200
    y = (rng.uniform(size=n) < 0.4).astype(int)
    p = rng.uniform(0.0, 1.0, n)
    # Deliberately close to, but strictly inside, the open interval.
    thresholds = np.linspace(0.01, 0.99, 25)

    result = net_benefit(y, p, thresholds)

    values = result[["threshold", "model", "treat_all", "treat_none"]].to_numpy()
    assert np.isfinite(values).all()


# ---------------------------------------------------------------------------
# subgroup_audit
# ---------------------------------------------------------------------------


def _synthetic_oof(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame({"y_true": y, "p_raw": p, "p_cal": p, "fold": 0})


def test_subgroup_audit_small_group_reports_nan_not_a_number():
    oof = _synthetic_oof(140, seed=0)
    # Group "B" gets exactly 10 rows (< min_n=40); "A" gets the rest (130).
    groups = pd.Series(["B"] * 10 + ["A"] * 130, index=oof.index)

    result = subgroup_audit(oof, groups, min_n=40)

    b_row = result[result["group"] == "B"].iloc[0]
    assert b_row["n"] == 10
    assert bool(b_row["sufficient"]) is False
    assert np.isnan(b_row["auc"])
    assert np.isnan(b_row["auc_lo"])
    assert np.isnan(b_row["auc_hi"])
    assert np.isnan(b_row["mean_cal_error"])


def test_subgroup_audit_sufficient_group_gets_real_bounded_auc():
    oof = _synthetic_oof(140, seed=0)
    groups = pd.Series(["B"] * 10 + ["A"] * 130, index=oof.index)

    result = subgroup_audit(oof, groups, min_n=40, n_boot=500)

    a_row = result[result["group"] == "A"].iloc[0]
    assert a_row["n"] == 130
    assert bool(a_row["sufficient"]) is True
    assert not np.isnan(a_row["auc"])
    assert a_row["auc_lo"] <= a_row["auc"] <= a_row["auc_hi"]


def test_subgroup_audit_raises_on_misaligned_groups():
    # Only 100 of 140 oof rows have a matching groups entry. Without an
    # explicit precondition check, reindex(oof.index) would silently turn
    # the other 40 into a NaN "group" that groupby(dropna=False) treats as
    # ordinary -- a spurious 40-row bucket clears min_n=40 and would
    # otherwise report a real-looking, "sufficient" numeric AUC for rows
    # that were never actually assigned a group.
    oof = _synthetic_oof(140, seed=0)
    groups = pd.Series(["A"] * 100, index=oof.index[:100])

    with pytest.raises(ValueError, match=r"40"):
        subgroup_audit(oof, groups, min_n=40)


# ---------------------------------------------------------------------------
# evaluate_oof: real trained condition (breast-cancer)
# ---------------------------------------------------------------------------


def test_evaluate_oof_breast_cancer_real_condition():
    result = train_condition("breast-cancer", seed=42)
    oof = result.oof

    metrics = evaluate_oof(oof, n_boot=500, seed=0)

    for key in ["roc_auc", "pr_auc", "brier", "ece"]:
        assert key in metrics
        point, lo, hi = metrics[key]
        assert lo <= point <= hi

    n = len(oof)
    assert metrics["n"] == n
    assert metrics["n_positive"] == int(oof["y_true"].sum())
    assert metrics["prevalence"] == pytest.approx(oof["y_true"].mean())

    reliability = metrics["reliability"]
    assert sum(b["n"] for b in reliability) == n
    for b in reliability:
        assert set(b) == {"bin_lo", "bin_hi", "n", "mean_pred", "frac_pos"}


# ---------------------------------------------------------------------------
# evaluate_oof: distinct per-metric bootstrap seed (fix round 1, finding 3)
# ---------------------------------------------------------------------------


def test_evaluate_oof_uses_a_distinct_bootstrap_seed_per_metric(monkeypatch):
    seen_seeds = []
    real_metric_ci = evaluate_module.metric_ci

    def spy_metric_ci(y, p, metric_fn, n_boot=2000, seed=0):
        seen_seeds.append(seed)
        return real_metric_ci(y, p, metric_fn, n_boot=n_boot, seed=seed)

    monkeypatch.setattr(evaluate_module, "metric_ci", spy_metric_ci)

    oof = _synthetic_oof(200, seed=0)
    evaluate_module.evaluate_oof(oof, n_boot=50, seed=7)

    # Four metrics (roc_auc, pr_auc, brier, ece) -> four metric_ci calls,
    # each with a DIFFERENT seed -- sharing one seed across all four would
    # make every metric resample byte-identical bootstrap indices.
    assert len(seen_seeds) == 4
    assert len(set(seen_seeds)) == 4
