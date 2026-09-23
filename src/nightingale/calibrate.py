# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Probability calibration: sigmoid/isotonic calibrators, ECE, and selection.

Two calibrator families are supported, both fit on a pooled out-of-fold
``(y_true, p_raw)`` sample (never on a model's own in-sample training
predictions — see :mod:`nightingale.model`, which owns that discipline and
calls :func:`nightingale.sentinel.assert_calibrator_held_out` in its own
code path):

- :class:`SigmoidCalibrator` (Platt-style): ``p_cal = 1 / (1 + exp(a *
  p_raw + b))``. This is the exact formula the browser walker
  (``docs/assets/walker.js``) re-implements verbatim, so ``a``/``b`` are
  usable directly against ``p_raw`` with no intermediate logit transform.
- :class:`IsotonicCalibrator`: piecewise-linear interpolation between
  ascending ``(x, y)`` breakpoints, clamped at the ends — exactly
  ``numpy.interp``'s semantics, again chosen so the JS port is a direct,
  dependency-free re-implementation.

:func:`ece` computes expected calibration error (equal-width binning).
:func:`pick_calibration` fits both families on the same OOF sample and
returns whichever has the lower ECE.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class Calibrator:
    """Common interface every fitted calibrator implements."""

    def predict(self, p) -> np.ndarray:
        """Map raw model probabilities ``p`` to calibrated probabilities."""
        raise NotImplementedError

    def export(self) -> dict:
        """Serialise this calibrator to its JSON-able ``model.json`` block."""
        raise NotImplementedError


class SigmoidCalibrator(Calibrator):
    """Platt-style sigmoid calibrator.

    ``p_cal = 1 / (1 + exp(a * p_raw + b))`` -- note this applies directly
    to the raw probability, not its logit. The JS re-implementation in
    ``docs/assets/walker.js`` must use this exact formula for parity.
    """

    def __init__(self, a: float, b: float):
        self.a = float(a)
        self.b = float(b)

    def predict(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return 1.0 / (1.0 + np.exp(self.a * p + self.b))

    def export(self) -> dict:
        return {"type": "sigmoid", "a": self.a, "b": self.b}


class IsotonicCalibrator(Calibrator):
    """Isotonic-regression calibrator: piecewise-linear, clamped at the ends.

    ``export()`` returns ascending ``x``/``y`` breakpoint lists. ``predict``
    at any ``p`` is linear interpolation between the two bracketing
    breakpoints, clamped to the first/last ``y`` value for ``p`` outside
    ``[x[0], x[-1]]`` -- exactly what ``numpy.interp`` does by default, which
    is what both this class and the JS port in ``docs/assets/walker.js``
    implement.
    """

    def __init__(self, x, y):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)

    def predict(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        return np.interp(p, self.x, self.y)

    def export(self) -> dict:
        return {"type": "isotonic", "x": self.x.tolist(), "y": self.y.tolist()}


def _fit_sigmoid(y, p_raw) -> SigmoidCalibrator:
    """Fit (a, b) by unregularised maximum-likelihood logistic regression.

    ``LogisticRegression`` models ``P(y=1|x) = 1 / (1 + exp(-(w*x + c)))``.
    Matching coefficients against ``p_cal = 1 / (1 + exp(a*p_raw + b))``
    gives ``a = -w``, ``b = -c``. ``C=np.inf`` fits an unregularised MLE
    (the modern spelling of the deprecated ``penalty=None``), matching
    classic Platt scaling.
    """
    X = np.asarray(p_raw, dtype=float).reshape(-1, 1)
    y_arr = np.asarray(y)
    clf = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=10000)
    clf.fit(X, y_arr)
    w = float(clf.coef_[0][0])
    c = float(clf.intercept_[0])
    return SigmoidCalibrator(a=-w, b=-c)


def _fit_isotonic(y, p_raw) -> IsotonicCalibrator:
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(np.asarray(p_raw, dtype=float), np.asarray(y))
    return IsotonicCalibrator(x=iso.X_thresholds_, y=iso.y_thresholds_)


_METHODS = {"sigmoid": _fit_sigmoid, "isotonic": _fit_isotonic}


def fit_calibrator(y, p_raw, method: str = "sigmoid") -> Calibrator:
    """Fit a calibrator of the given family on ``(y, p_raw)``.

    ``method`` is ``"sigmoid"`` or ``"isotonic"``.
    """
    if method not in _METHODS:
        raise ValueError(
            f"unknown calibration method {method!r}; choose one of {sorted(_METHODS)}"
        )
    return _METHODS[method](y, p_raw)


def ece(y, p, n_bins: int = 10) -> float:
    """Expected Calibration Error: equal-width-bin gap between confidence and accuracy.

    Rows are assigned to ``n_bins`` equal-width bins over ``[0, 1]`` by their
    predicted probability ``p`` (the last bin is closed on both ends so a
    prediction of exactly 1.0 is counted). ECE is the bin-size-weighted mean
    absolute gap between each bin's mean prediction (confidence) and mean
    true label (accuracy):

        ECE = sum_b (n_b / n) * |mean(y in bin b) - mean(p in bin b)|

    Empty bins contribute nothing. A perfectly calibrated set of
    predictions (mean p == mean y in every bin) scores 0; an
    anti-calibrated one (predictions confidently wrong) scores high.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if y.shape != p.shape:
        raise ValueError(f"y and p must have the same shape, got {y.shape} vs {p.shape}")

    n = len(y)
    if n == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        bin_n = int(mask.sum())
        if bin_n == 0:
            continue
        bin_confidence = p[mask].mean()
        bin_accuracy = y[mask].mean()
        total += (bin_n / n) * abs(bin_accuracy - bin_confidence)
    return float(total)


def pick_calibration(y, p_raw) -> Calibrator:
    """Fit both calibrator families on the same OOF sample; return the lower-ECE one.

    Both candidates are fit and scored on the identical ``(y, p_raw)`` pair
    passed in (typically the pooled OOF sample) -- this function makes no
    train/eval split of its own; that responsibility (never fitting a
    calibrator on a model's own in-sample predictions) belongs to the
    caller, see :mod:`nightingale.model`.
    """
    y = np.asarray(y)
    p_raw = np.asarray(p_raw, dtype=float)

    sigmoid = fit_calibrator(y, p_raw, method="sigmoid")
    isotonic = fit_calibrator(y, p_raw, method="isotonic")

    sigmoid_ece = ece(y, sigmoid.predict(p_raw))
    isotonic_ece = ece(y, isotonic.predict(p_raw))

    return sigmoid if sigmoid_ece <= isotonic_ece else isotonic
