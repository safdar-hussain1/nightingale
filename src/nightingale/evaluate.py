# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Evaluation: bootstrap CIs, decision-curve analysis, subgroup audit.

Every function here is meant to operate on the pooled out-of-fold
predictions produced by :func:`nightingale.model.train_condition` --
specifically ``oof["p_cal"]`` (cross-fitted calibrated probabilities),
never ``oof["p_raw"]``. See :mod:`nightingale.model`'s module docstring
for why that distinction matters: ``p_cal`` is the only column where each
row's probability comes from a calibrator that never saw that row.

:func:`metric_ci` is a general-purpose STRATIFIED bootstrap CI: each
bootstrap replicate resamples WITHIN each class label (same per-class
count as the original data, sampled with replacement), rather than
resampling the whole pooled dataset at once. This fixes class prevalence
across every replicate, so a metric like ROC-AUC is never computed on a
degenerate single-class resample, and the reported spread reflects
sampling uncertainty in the score itself rather than uncertainty in
prevalence (which the original data already fixes exactly). The interval
is a 95% percentile interval (the [2.5, 97.5] percentiles of the replicate
scores), not a normal-approximation interval.

:func:`evaluate_oof` bundles four headline metrics (ROC-AUC, PR-AUC,
Brier, ECE) each with a bootstrap CI from :func:`metric_ci`, plus a
reliability diagram's worth of equal-width bins (matching
:func:`nightingale.calibrate.ece`'s own binning) for plotting.

:func:`net_benefit` implements standard decision-curve analysis (Vickers &
Elkin, 2006): at each candidate treatment threshold ``pt``, net benefit
trades true positives against false positives, weighted by the odds
``pt / (1 - pt)`` -- the exchange rate a rational decision-maker implies
by choosing to treat at ``pt`` in the first place.

:func:`subgroup_audit` reports per-group discrimination (AUC + CI) and a
scalar calibration-in-the-large error for every distinct value of a
grouping series (e.g. heart-disease's ``site`` column), refusing to report
a numeric AUC for any group with fewer than ``min_n`` rows (default 40):
small-n subgroup AUCs are exactly the kind of number that looks precise
and isn't -- NaN is the honest value here, not a suppressed warning and
not a silently dropped row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from nightingale.calibrate import ece

N_RELIABILITY_BINS = 10


def metric_ci(
    y, p, metric_fn, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Stratified-bootstrap 95% percentile CI for ``metric_fn(y, p)``.

    Each of ``n_boot`` replicates resamples WITH replacement independently
    within each class of ``y`` (same per-class size as the original
    sample), then concatenates the per-class resamples -- prevalence is
    therefore identical, by construction, in every replicate and in the
    original data; only within-class composition varies replicate to
    replicate. ``metric_fn`` must accept ``(y, p)`` positionally and
    return a float (e.g. ``sklearn.metrics.roc_auc_score``).

    Returns ``(point, lo, hi)``: ``point`` is ``metric_fn`` evaluated on
    the ORIGINAL (unresampled) data, ``lo``/``hi`` are the [2.5, 97.5]
    percentiles of the ``n_boot`` replicate scores.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    point = float(metric_fn(y, p))

    rng = np.random.default_rng(seed)
    class_indices = [np.where(y == c)[0] for c in np.unique(y)]

    replicate_scores = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        resampled = np.concatenate(
            [rng.choice(idx, size=len(idx), replace=True) for idx in class_indices]
        )
        replicate_scores[b] = metric_fn(y[resampled], p[resampled])

    lo, hi = np.percentile(replicate_scores, [2.5, 97.5])
    return point, float(lo), float(hi)


def _reliability_bins(
    y: np.ndarray, p: np.ndarray, n_bins: int = N_RELIABILITY_BINS
) -> list[dict]:
    """Equal-width reliability bins over [0, 1], matching :func:`nightingale.calibrate.ece`'s binning.

    Every bin in ``[0, n_bins)`` is included even when empty (``n=0``,
    ``mean_pred``/``frac_pos`` NaN), so bin counts always sum to
    ``len(y)`` and callers can plot a full-width reliability diagram
    without gaps in the bin edges.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        n = int(mask.sum())
        bins.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n": n,
                "mean_pred": float(p[mask].mean()) if n > 0 else float("nan"),
                "frac_pos": float(y[mask].mean()) if n > 0 else float("nan"),
            }
        )
    return bins


def evaluate_oof(oof: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """Headline metrics (with bootstrap CIs) + reliability bins for one condition's OOF frame.

    Computed from ``oof["p_cal"]`` (cross-fitted calibrated probabilities)
    against ``oof["y_true"]`` -- never ``oof["p_raw"]``, per the module
    docstring. Returns a dict with keys ``roc_auc``/``pr_auc``/``brier``/
    ``ece`` (each a ``[point, lo, hi]`` list), ``reliability`` (list of bin
    dicts, see :func:`_reliability_bins`), and ``n``/``n_positive``/
    ``prevalence``.
    """
    y = oof["y_true"].to_numpy()
    p = oof["p_cal"].to_numpy()
    n = len(oof)
    n_positive = int(y.sum())

    return {
        "roc_auc": list(metric_ci(y, p, roc_auc_score, n_boot=n_boot, seed=seed)),
        "pr_auc": list(metric_ci(y, p, average_precision_score, n_boot=n_boot, seed=seed)),
        "brier": list(metric_ci(y, p, brier_score_loss, n_boot=n_boot, seed=seed)),
        "ece": list(metric_ci(y, p, ece, n_boot=n_boot, seed=seed)),
        "reliability": _reliability_bins(y, p),
        "n": n,
        "n_positive": n_positive,
        "prevalence": n_positive / n,
    }


def net_benefit(y, p, thresholds: np.ndarray) -> pd.DataFrame:
    """Decision-curve net benefit for the model vs. treat-all vs. treat-none.

    At threshold ``pt`` a row is "predicted positive" iff ``p >= pt``.
    Net benefit (Vickers & Elkin, 2006, "Decision curve analysis"):

        NB_model(pt)     = TP/n - FP/n * (pt / (1 - pt))
        NB_treat_all(pt) = prevalence - (1 - prevalence) * (pt / (1 - pt))
        NB_treat_none    = 0   (nobody is treated -- no TP, no FP, by definition)

    ``pt`` must be strictly inside ``(0, 1)``: ``pt / (1 - pt)`` is
    undefined at ``pt == 1`` (and ``pt == 0`` makes "predicted positive"
    trivially everyone, which is a degenerate but not undefined case).
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    n = len(y)
    prevalence = float(y.sum()) / n

    rows = []
    for pt in np.asarray(thresholds, dtype=float):
        predicted_positive = p >= pt
        tp = int(np.sum(predicted_positive & (y == 1)))
        fp = int(np.sum(predicted_positive & (y == 0)))
        odds = pt / (1.0 - pt)
        rows.append(
            {
                "threshold": float(pt),
                "model": tp / n - (fp / n) * odds,
                "treat_all": prevalence - (1.0 - prevalence) * odds,
                "treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows, columns=["threshold", "model", "treat_all", "treat_none"])


def subgroup_audit(
    oof: pd.DataFrame, groups, min_n: int = 40, n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Per-group discrimination (AUC + CI) and calibration error, honest about small n.

    ``groups`` is a value per row, index-aligned to ``oof`` (e.g.
    heart-disease's ``site`` column). A group is "sufficient" iff it has
    at least ``min_n`` rows AND both classes are present (ROC-AUC is
    undefined for a single-class group) -- insufficient groups get
    ``sufficient=False`` and NaN for every numeric metric (never a
    computed number, never a silently dropped row): reporting "insufficient
    n" beats reporting a number that looks precise but is one draw of a
    handful of points.

    "Mean calibration error" here is the scalar
    ``abs(mean(p_cal) - mean(y_true))`` for the group -- calibration-in-
    the-large -- rather than a binned ECE: subgroup sizes can sit close to
    ``min_n``, where 10 ECE bins would mostly be near-empty and noisy; a
    single signed gap is the more honest number at that scale.
    """
    df = oof.copy()
    df["_group"] = pd.Series(groups).reindex(df.index)

    rows = []
    for group_value, sub in df.groupby("_group", dropna=False):
        y = sub["y_true"].to_numpy()
        p = sub["p_cal"].to_numpy()
        n = len(sub)
        n_positive = int(y.sum())
        has_both_classes = 0 < n_positive < n
        sufficient = bool(n >= min_n and has_both_classes)

        if sufficient:
            auc_point, auc_lo, auc_hi = metric_ci(
                y, p, roc_auc_score, n_boot=n_boot, seed=seed
            )
            mean_cal_error = float(abs(p.mean() - y.mean()))
        else:
            auc_point = auc_lo = auc_hi = float("nan")
            mean_cal_error = float("nan")

        rows.append(
            {
                "group": group_value,
                "n": n,
                "n_positive": n_positive,
                "auc": auc_point,
                "auc_lo": auc_lo,
                "auc_hi": auc_hi,
                "mean_cal_error": mean_cal_error,
                "sufficient": sufficient,
            }
        )

    return pd.DataFrame(
        rows,
        columns=["group", "n", "n_positive", "auc", "auc_lo", "auc_hi", "mean_cal_error", "sufficient"],
    )
