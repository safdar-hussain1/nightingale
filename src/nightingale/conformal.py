# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Split-conformal prediction sets: calibrated "uncertain" verdicts.

Standard split-conformal classification (Angelopoulos & Bates, 2021, "A
Gentle Introduction to Conformal Prediction"), specialised to binary
labels. Given a HELD-OUT calibration sample's true labels and predicted
probabilities of the positive class, :func:`conformal_qhat` computes a
single threshold ``q_hat`` such that, for a fresh exchangeable point, the
set of labels built by :func:`prediction_set` contains the true label with
probability at least ``1 - alpha`` (the marginal coverage guarantee).

Nonconformity score for a labelled point ``(y, p)``:

    s = 1 - p(true class)      where p(true class) = p if y == 1 else 1 - p

A high score means the model assigned little probability mass to the
label that turned out to be correct -- exactly the notion of "how
surprising was this label" the conformal framework needs.

``conformal_qhat`` is fit ONCE, on a calibration sample disjoint from
whatever it is later applied to. In this project that calibration sample
is the pooled out-of-fold frame (``TrainResult.oof``, using ``p_cal``) --
"cross-conformal over OOF": because every ``p_cal`` value already comes
from a calibrator that never saw its own row (see
:mod:`nightingale.model`'s module docstring), the pooled OOF frame is a
legitimate stand-in for an independent calibration set, and the resulting
``q_hat`` can be applied at deployment time to score genuinely new rows.
"""

from __future__ import annotations

import numpy as np


def conformal_qhat(y, p_cal, alpha: float = 0.1) -> float:
    """The split-conformal threshold from a calibration sample.

    Nonconformity scores ``s_i = 1 - p(true class)_i`` (see module
    docstring) are sorted ascending; ``q_hat`` is the
    ``k = ceil((n + 1) * (1 - alpha))``-th smallest score (1-indexed) --
    equivalently the ``k / n`` empirical quantile taken with the
    "round up" (order-statistic) convention rather than linear
    interpolation, which is what the standard split-conformal formula
    requires.

    If ``k`` exceeds ``n`` (possible only when ``n`` is small relative to
    ``1 / alpha``), ``q_hat`` is capped at the maximum observed
    nonconformity score rather than treated as unbounded/infinite. The
    formally "correct" unbounded value would make every prediction set
    trivially contain both labels (infinitely conservative, i.e. always
    "uncertain"); capping instead keeps ``q_hat`` a genuine, finite number
    drawn from the data, at the cost of the marginal guarantee no longer
    being airtight in that rare, small-n regime. Every caller in this
    project uses OOF samples of at least a few hundred rows, well away
    from this edge.
    """
    y = np.asarray(y)
    p_cal = np.asarray(p_cal, dtype=float)
    n = len(y)

    p_true_class = np.where(y == 1, p_cal, 1.0 - p_cal)
    scores = np.sort(1.0 - p_true_class)

    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(k, n)
    return float(scores[k - 1])


def prediction_set(p: float, q_hat: float) -> str:
    """Collapse a conformal prediction set to one of "positive"/"negative"/"uncertain".

    A label is IN the set iff the probability assigned to it is at least
    ``1 - q_hat``: for the positive label that is ``p >= 1 - q_hat``; for
    the negative label (probability ``1 - p``) that is
    ``1 - p >= 1 - q_hat``, i.e. ``p <= q_hat``.

    - Both labels in the set -> ``"uncertain"`` (the honest answer: at
      this ``q_hat``, the model's confidence is not enough to rule either
      label out).
    - Exactly one label in the set -> that label.
    - NEITHER label in the set (empty set -- possible when ``q_hat < 0.5``
      and ``p`` falls strictly between ``q_hat`` and ``1 - q_hat``):
      rather than surface an empty set (which has no natural single-word
      verdict), this falls back to the plain argmax rule -- ``"positive"``
      if ``p >= 0.5`` else ``"negative"`` -- and is deliberately NOT
      reported as ``"uncertain"``. This is a documented design choice, not
      an oversight: an empty conformal set says the calibration data found
      NEITHER label plausible at this confidence level for a point like
      this one, which is a different situation from "both labels are
      plausible" (the ``"uncertain"`` case) -- collapsing the two into one
      bucket would blur two distinct failure modes together, and an
      argmax fallback is the more useful verdict for a caller that must
      still produce SOME answer.
    """
    positive_in = p >= 1.0 - q_hat
    negative_in = p <= q_hat

    if positive_in and negative_in:
        return "uncertain"
    if positive_in:
        return "positive"
    if negative_in:
        return "negative"
    return "positive" if p >= 0.5 else "negative"
