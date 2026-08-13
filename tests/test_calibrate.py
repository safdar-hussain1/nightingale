# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Tests for probability calibration (nightingale.calibrate).

Covers: ece() against a hand-computed 3-bin example plus perfectly-/anti-
calibrated synthetic extremes; SigmoidCalibrator recovering a known planted
(a, b) logistic distortion; IsotonicCalibrator's export() round-tripping
through an independent, hand-written linear interpolation to prove the
JS-port contract (Task 10) is trustworthy; and pick_calibration choosing
the lower-ECE candidate.

Nothing here touches data/raw/ or data/cleaned/ -- calibration operates on
plain (y, p) arrays, synthetic or hand-built, so this whole file is
data-independent and always runs.
"""

import numpy as np
import pytest

from nightingale.calibrate import (
    Calibrator,
    IsotonicCalibrator,
    SigmoidCalibrator,
    ece,
    fit_calibrator,
    pick_calibration,
)

# ---------------------------------------------------------------------------
# ece()
# ---------------------------------------------------------------------------


def test_ece_hand_computed_three_bin_example_exact():
    # 3 bins over [0,1]: [0, 1/3), [1/3, 2/3), [2/3, 1]. Points chosen well
    # away from the bin boundaries so membership is unambiguous.
    y = [0, 1, 1, 0, 1, 1]
    p = [0.1, 0.2, 0.4, 0.5, 0.8, 0.9]
    # Bin 1 (p in [0, 1/3)):   p={0.1,0.2} y={0,1} -> conf=0.15, acc=0.5
    # Bin 2 (p in [1/3, 2/3)): p={0.4,0.5} y={1,0} -> conf=0.45, acc=0.5
    # Bin 3 (p in [2/3, 1]):   p={0.8,0.9} y={1,1} -> conf=0.85, acc=1.0
    expected = (2 / 6) * abs(0.5 - 0.15) + (2 / 6) * abs(0.5 - 0.45) + (2 / 6) * abs(1.0 - 0.85)

    result = ece(y, p, n_bins=3)

    assert result == pytest.approx(expected, abs=1e-12)


def test_ece_perfectly_calibrated_synthetic_preds_near_zero():
    rng = np.random.default_rng(0)
    n = 20000
    p = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(0.0, 1.0, n) < p).astype(int)  # y ~ Bernoulli(p) exactly

    result = ece(y, p, n_bins=10)

    assert result < 0.02


def test_ece_anti_calibrated_preds_is_large():
    rng = np.random.default_rng(1)
    n = 2000
    # Confidently wrong in both directions: predicts ~0.95 when true rate is
    # ~0.05 and vice versa.
    p = np.concatenate([np.full(n // 2, 0.95), np.full(n // 2, 0.05)])
    y = np.concatenate([np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)])

    result = ece(y, p, n_bins=10)

    assert result > 0.8


def test_ece_requires_matching_shapes():
    with pytest.raises(ValueError):
        ece([0, 1, 1], [0.1, 0.2])


# ---------------------------------------------------------------------------
# SigmoidCalibrator / fit_calibrator("sigmoid")
# ---------------------------------------------------------------------------


def test_sigmoid_calibrator_predict_matches_documented_formula():
    cal = SigmoidCalibrator(a=2.0, b=-1.0)
    p = np.array([0.0, 0.5, 1.0])

    result = cal.predict(p)

    expected = 1.0 / (1.0 + np.exp(2.0 * p - 1.0))
    np.testing.assert_allclose(result, expected)


def test_sigmoid_calibrator_recovers_planted_distortion():
    rng = np.random.default_rng(42)
    n = 6000
    a_true, b_true = 3.0, -1.5

    p_raw = rng.uniform(0.02, 0.98, n)
    p_cal_true = 1.0 / (1.0 + np.exp(a_true * p_raw + b_true))
    y = (rng.uniform(0.0, 1.0, n) < p_cal_true).astype(int)

    cal = fit_calibrator(y, p_raw, method="sigmoid")

    assert isinstance(cal, SigmoidCalibrator)
    assert cal.a == pytest.approx(a_true, abs=0.3)
    assert cal.b == pytest.approx(b_true, abs=0.3)


def test_sigmoid_calibrator_export_shape():
    cal = SigmoidCalibrator(a=1.5, b=-0.5)

    exported = cal.export()

    assert exported == {"type": "sigmoid", "a": pytest.approx(1.5), "b": pytest.approx(-0.5)}


# ---------------------------------------------------------------------------
# IsotonicCalibrator / fit_calibrator("isotonic")
# ---------------------------------------------------------------------------


def _hand_written_interp(p, x, y):
    """Independent linear-interpolation-with-clamping re-implementation.

    Deliberately not reusing numpy.interp (that's what IsotonicCalibrator
    itself uses internally) -- this proves the exported breakpoints alone
    are enough to reproduce .predict(), the property Task 10's from-scratch
    JS port depends on.
    """
    x = list(x)
    y = list(y)
    out = []
    for value in np.atleast_1d(p):
        if value <= x[0]:
            out.append(y[0])
            continue
        if value >= x[-1]:
            out.append(y[-1])
            continue
        for i in range(len(x) - 1):
            if x[i] <= value <= x[i + 1]:
                lo_x, hi_x = x[i], x[i + 1]
                lo_y, hi_y = y[i], y[i + 1]
                if hi_x == lo_x:
                    out.append(lo_y)
                else:
                    t = (value - lo_x) / (hi_x - lo_x)
                    out.append(lo_y + t * (hi_y - lo_y))
                break
    return np.array(out)


def test_isotonic_export_round_trips_through_hand_written_interpolation():
    rng = np.random.default_rng(7)
    n = 500
    p_raw = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(0.0, 1.0, n) < p_raw).astype(int)

    cal = fit_calibrator(y, p_raw, method="isotonic")
    assert isinstance(cal, IsotonicCalibrator)

    exported = cal.export()
    assert exported["type"] == "isotonic"
    assert exported["x"] == sorted(exported["x"])  # ascending breakpoints

    probe = np.linspace(-0.5, 1.5, 200)  # includes out-of-range probes
    via_predict = cal.predict(probe)
    via_hand_interp = _hand_written_interp(probe, exported["x"], exported["y"])

    np.testing.assert_allclose(via_predict, via_hand_interp, atol=1e-12)


def test_isotonic_calibrator_export_breakpoints_are_lists():
    cal = IsotonicCalibrator(x=[0.1, 0.5, 0.9], y=[0.05, 0.5, 0.95])

    exported = cal.export()

    assert exported == {"type": "isotonic", "x": [0.1, 0.5, 0.9], "y": [0.05, 0.5, 0.95]}


# ---------------------------------------------------------------------------
# fit_calibrator() dispatch / pick_calibration()
# ---------------------------------------------------------------------------


def test_fit_calibrator_rejects_unknown_method():
    with pytest.raises(ValueError):
        fit_calibrator([0, 1], [0.1, 0.9], method="quadratic")


def test_pick_calibration_returns_a_calibrator():
    rng = np.random.default_rng(3)
    n = 500
    p_raw = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(0.0, 1.0, n) < p_raw).astype(int)

    cal = pick_calibration(y, p_raw)

    assert isinstance(cal, Calibrator)
    preds = cal.predict(p_raw)
    assert np.all((preds >= 0.0) & (preds <= 1.0))


def test_pick_calibration_prefers_lower_ece_candidate():
    # A step-shaped truth (isotonic can represent, sigmoid can't) should
    # make isotonic win on ECE.
    rng = np.random.default_rng(9)
    n = 4000
    p_raw = rng.uniform(0.0, 1.0, n)
    p_true = np.where(p_raw < 0.5, 0.05, 0.95)  # sharp step, non-monotone-smooth
    y = (rng.uniform(0.0, 1.0, n) < p_true).astype(int)

    sigmoid = fit_calibrator(y, p_raw, method="sigmoid")
    isotonic = fit_calibrator(y, p_raw, method="isotonic")
    sigmoid_ece = ece(y, sigmoid.predict(p_raw))
    isotonic_ece = ece(y, isotonic.predict(p_raw))

    chosen = pick_calibration(y, p_raw)

    assert isotonic_ece < sigmoid_ece
    assert chosen.export()["type"] == "isotonic"
