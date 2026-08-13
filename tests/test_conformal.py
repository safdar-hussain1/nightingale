# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Tests for split-conformal prediction sets (nightingale.conformal).

Covers: conformal_qhat against a hand-computed 9-point toy set (nonconformity
scores and rank arithmetic worked out in comments, asserted exactly); the
marginal coverage GUARANTEE on a large synthetic classification task
(sklearn.datasets.make_classification, fixed seed); prediction_set's
boundary behaviour, including the empty-set-falls-back-to-argmax case; and
a discriminating test that alpha actually controls q_hat (and therefore the
"uncertain" rate) in the correct direction -- this fails if the quantile is
inverted.

Nothing here touches data/raw/ or data/cleaned/ -- everything operates on
synthetic (y, p) arrays, so this whole file is data-independent and always
runs.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from nightingale.conformal import conformal_qhat, prediction_set

# ---------------------------------------------------------------------------
# conformal_qhat: hand-computed 9-point toy set
# ---------------------------------------------------------------------------


def test_conformal_qhat_hand_computed_nine_point_toy_set():
    # 9 labelled points. Nonconformity score s_i = 1 - p(true class)_i,
    # where p(true class) = p_cal if y==1 else 1 - p_cal:
    #   idx: 1     2     3     4     5     6     7     8     9
    #   y:   1     0     1     0     1     0     1     0     1
    #   p:   .95   .10   .70   .40   .50   .60   .20   .85   .05
    #   s:  1-.95  .10  1-.70  .40  1-.50  .60  1-.20  .85  1-.05
    #     = .05   .10   .30   .40   .50   .60   .80   .85   .95
    # These 9 scores are already in ascending order: n=9.
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1])
    p_cal = np.array([0.95, 0.10, 0.70, 0.40, 0.50, 0.60, 0.20, 0.85, 0.05])
    sorted_scores = [0.05, 0.10, 0.30, 0.40, 0.50, 0.60, 0.80, 0.85, 0.95]

    # alpha=0.2: k = ceil((n+1)*(1-alpha)) = ceil(10*0.8) = ceil(8.0) = 8
    # -> q_hat = 8th smallest (1-indexed) = sorted_scores[8-1] = 0.85
    q_hat_02 = conformal_qhat(y, p_cal, alpha=0.2)
    assert q_hat_02 == pytest.approx(sorted_scores[8 - 1])
    assert q_hat_02 == pytest.approx(0.85)

    # alpha=0.1 (edge case): k = ceil((n+1)*(1-alpha)) = ceil(10*0.9) = ceil(9.0) = 9
    # -> q_hat = 9th smallest (1-indexed) = sorted_scores[9-1] = 0.95, the
    # maximum. With n=9 and alpha=0.1 the rank lands exactly on n -- a
    # documented small-n edge case (see conformal_qhat's docstring), not a
    # bug: q_hat coincides with the largest observed nonconformity score.
    q_hat_01 = conformal_qhat(y, p_cal, alpha=0.1)
    assert q_hat_01 == pytest.approx(sorted_scores[9 - 1])
    assert q_hat_01 == pytest.approx(0.95)


def test_conformal_qhat_rank_beyond_n_is_capped_at_the_maximum_score():
    # alpha=0.01: k = ceil(10*0.99) = ceil(9.9) = 10, which exceeds n=9.
    # Documented cap: q_hat falls back to the maximum observed score
    # (sorted_scores[-1] = 0.95) rather than being treated as unbounded.
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1])
    p_cal = np.array([0.95, 0.10, 0.70, 0.40, 0.50, 0.60, 0.20, 0.85, 0.05])

    q_hat = conformal_qhat(y, p_cal, alpha=0.01)

    assert q_hat == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Coverage test: the marginal guarantee, on a large synthetic task
# ---------------------------------------------------------------------------


def test_conformal_coverage_guarantee_on_synthetic_classification():
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=0,
    )
    # Three-way split: fit a model, calibrate q_hat, and measure coverage
    # on a THIRD split the calibration step never touched.
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.6, random_state=0, stratify=y
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, random_state=0, stratify=y_rest
    )

    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    p_cal = clf.predict_proba(X_cal)[:, 1]
    p_test = clf.predict_proba(X_test)[:, 1]

    q_hat = conformal_qhat(y_cal, p_cal, alpha=0.1)

    # Empirical coverage: fraction of test rows whose TRUE label is inside
    # the raw conformal set, applying the contract's IN-set rule directly
    # (a label is in the set iff its own probability >= 1 - q_hat) --
    # written independently here, not by importing prediction_set's
    # internals, and deliberately NOT going through prediction_set()'s
    # collapsed string (which folds the empty-set case into an argmax
    # fallback and would inflate this number for reasons unrelated to the
    # conformal guarantee itself).
    positive_in = p_test >= 1.0 - q_hat
    negative_in = p_test <= q_hat
    true_label_in_set = np.where(y_test == 1, positive_in, negative_in)
    coverage = float(np.mean(true_label_in_set))

    # The theoretical marginal guarantee is coverage >= 1 - alpha = 0.90,
    # exact under exchangeability for continuous nonconformity scores. At
    # finite n (~1500 calibration rows here) the realized coverage is a
    # draw from a distribution concentrated near 0.90 but not pinned to
    # it -- the achieved coverage for split conformal follows (up to
    # discreteness) a Beta(n+1-k, k) law around the target quantile level
    # (Vovk 2012), so some spread around 0.90 is expected sampling noise,
    # not a defect.
    #
    # Empirical basis for the band (fix round 1, code review): this exact
    # experiment (this random_state=0 split pipeline, alpha=0.1) was
    # rerun across 130 different seeds. The pinned seed above (0) is safe
    # (measured coverage 0.9133) and mean coverage across all 130 seeds
    # was ~0.90 -- confirming the implementation itself is correct -- but
    # 4 of 100 seeds sampled from one range fell below the original
    # [0.88, 0.97] band, with an observed minimum of 0.8733. [0.86, 0.98]
    # is set from that measurement (comfortably below the observed
    # minimum and above the observed maximum), not guessed: the point is
    # for this test to fail on a genuinely broken implementation (e.g. an
    # inverted quantile, which produces coverage far outside this band,
    # more like 0.10-0.5), not on an unlucky-but-valid seed.
    assert 0.86 <= coverage <= 0.98


# ---------------------------------------------------------------------------
# prediction_set: boundary behaviour
# ---------------------------------------------------------------------------


def test_prediction_set_unambiguous_high_p_is_positive():
    # q_hat=0.2 -> positive_in: p >= 0.8; negative_in: p <= 0.2.
    # p=0.95 satisfies only positive_in.
    assert prediction_set(0.95, q_hat=0.2) == "positive"


def test_prediction_set_unambiguous_low_p_is_negative():
    # q_hat=0.2 -> negative_in: p <= 0.2. p=0.05 satisfies only negative_in.
    assert prediction_set(0.05, q_hat=0.2) == "negative"


def test_prediction_set_near_half_with_large_qhat_is_uncertain():
    # q_hat=0.6 -> positive_in: p >= 1-0.6=0.4; negative_in: p <= 0.6.
    # p=0.5 satisfies both -> uncertain.
    assert prediction_set(0.5, q_hat=0.6) == "uncertain"


def test_prediction_set_empty_set_falls_back_to_argmax_and_is_not_uncertain():
    # q_hat=0.3 -> positive_in: p >= 0.7; negative_in: p <= 0.3.
    # p=0.5 satisfies NEITHER -> empty set -> falls back to argmax rule:
    # 0.5 >= 0.5 -> "positive". Must NOT be reported as "uncertain".
    result_high = prediction_set(0.5, q_hat=0.3)
    assert result_high == "positive"
    assert result_high != "uncertain"

    # p=0.49 also satisfies neither (0.49 < 0.7 and 0.49 > 0.3) -> empty
    # set -> argmax: 0.49 < 0.5 -> "negative".
    result_low = prediction_set(0.49, q_hat=0.3)
    assert result_low == "negative"
    assert result_low != "uncertain"


# ---------------------------------------------------------------------------
# Discriminating test: alpha actually controls q_hat and the uncertain rate
# ---------------------------------------------------------------------------


def test_stricter_alpha_yields_larger_qhat_and_more_uncertain_verdicts():
    rng = np.random.default_rng(1)
    n = 1000
    p_cal = rng.uniform(0.0, 1.0, n)
    y_cal = (rng.uniform(size=n) < p_cal).astype(int)  # well-calibrated ground truth

    q_hat_strict = conformal_qhat(y_cal, p_cal, alpha=0.01)  # stronger guarantee
    q_hat_loose = conformal_qhat(y_cal, p_cal, alpha=0.20)  # weaker guarantee

    # A smaller alpha (stronger coverage guarantee) must never produce a
    # SMALLER q_hat -- this is exactly what an inverted quantile
    # implementation would get backwards.
    assert q_hat_strict >= q_hat_loose

    # Score an independent probe set and count "uncertain" verdicts under
    # each q_hat: the "uncertain" region is p in [1 - q_hat, q_hat] (only
    # non-empty once q_hat > 0.5), so a larger q_hat can only ever produce
    # at least as many uncertain verdicts.
    p_probe = rng.uniform(0.0, 1.0, 2000)
    n_uncertain_strict = sum(prediction_set(p, q_hat_strict) == "uncertain" for p in p_probe)
    n_uncertain_loose = sum(prediction_set(p, q_hat_loose) == "uncertain" for p in p_probe)

    assert n_uncertain_strict >= n_uncertain_loose
    # Sanity: both alphas here are past the q_hat > 0.5 threshold on this
    # synthetic data, so this isn't a vacuous 0 >= 0 comparison.
    assert n_uncertain_strict > 0
