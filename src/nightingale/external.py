# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Four-hospital external validation: train on Cleveland, transfer elsewhere.

This is the project's honesty centrepiece. Every other trained model in
this repo (Task 5/8) is fit on data POOLED across whichever sites a
condition has -- heart-disease's pooled model, in particular, is trained on
all four hospitals at once. That pooled setup can never show what happens
when a model built at one hospital is deployed, unmodified, at a
DIFFERENT hospital it never saw during training -- which is the ordinary
real-world situation for a model that ships as software rather than being
refit locally everywhere it's used.

:func:`transfer_study` trains a model on Cleveland's 303 rows ONLY (the
richest-quality, most-complete site -- Phase 0's profile: 1% missing
`ca`/`thal`, no cholesterol sentinel corruption), using the exact nested-CV
protocol :func:`nightingale.model.train_condition` uses internally --
reusing that module's private helpers (:func:`_outer_splits`,
:func:`_select_inner_params`, :func:`_one_hot_fold_safe`,
:func:`_most_frequent_params`, :func:`_fit_deployment_calibrator`) directly
rather than reimplementing fold-safe encoding or the grid search a second
time. ``train_condition`` itself has no site-filtering hook (it always
reads a whole condition's cleaned CSV), which is why this module
orchestrates those helpers over a site-filtered frame instead of calling
``train_condition`` directly.

For each of the three sites Cleveland's model never saw --
``hungarian``/``switzerland``/``va`` -- :func:`transfer_study` reports THREE
blocks per site, and their names are deliberately unambiguous about which
row set each one is computed on (fix round 1 -- see :func:`transfer_study`'s
own docstring for the full story of why the schema is shaped this way):

1. **``naive_all_rows``**: the Cleveland-trained-and-calibrated pipeline
   (``final_model.predict_proba`` run through the Cleveland-fitted
   deployment calibrator) scored as-is on the WHOLE target site, with no
   site-specific adjustment at all -- informative on its own ("what does
   zero-effort deployment look like"), but NEVER compared directly against
   ``recalibrated`` (different row count, different rows).
2. **``naive``** and **``recalibrated``**: a seeded 30/70 site-LOCAL split
   (:func:`_calibration_eval_split`) fits a single intercept correction on
   the 30% calibration rows (:func:`_fit_intercept_only` -- slope pinned at
   1, only the intercept is free); ``naive`` is the SAME 70% evaluation
   rows scored WITHOUT that correction, ``recalibrated`` is those IDENTICAL
   rows scored WITH it. These two are the like-for-like pair: same ``n``,
   same rows, only the probabilities differ. The 30% calibration rows never
   appear in the 70% evaluation rows -- :func:`nightingale.sentinel.assert_calibrator_held_out`
   is called live on that split, exactly as :mod:`nightingale.model` calls
   it on every outer CV fold.

All three blocks report ROC-AUC/PR-AUC/Brier/ECE (each a bootstrap
``[point, lo, hi]`` triple via :func:`nightingale.evaluate.metric_ci`) plus
:func:`calibration_in_the_large`'s intercept/slope diagnostic.

Two calibration-diagnostic concepts appear here and are NOT the same thing,
despite both operating on logit(p):

- :func:`calibration_in_the_large` is a DIAGNOSTIC: an unconstrained
  logistic regression of y on logit(p) -- both intercept and slope are
  free. It measures HOW miscalibrated a set of probabilities is (intercept
  ~ prevalence mismatch, slope ~ over/under-dispersion), and is computed
  for ``naive_all_rows``, ``naive``, and ``recalibrated`` alike, purely to
  report the number.
- :func:`_fit_intercept_only` is the REPAIR: slope is PINNED at 1 by
  construction (only the intercept is fit), because that is what "intercept
  recalibration" means -- a pure prevalence-shift correction, not a full
  refit of the model's discrimination.

Because :func:`_fit_intercept_only`'s correction is
``p_recal = sigmoid(logit(p) + c)`` for a FIXED scalar ``c``, it is a
strictly monotone increasing function of ``p``. ROC-AUC is a rank
statistic, so it is provably invariant under any strictly monotone
transform of the scores it ranks -- recalibration can repair calibration,
never discrimination. ``tests/test_external.py``'s AUC-invariance tests pin
this down to ~1e-9 both in isolation and against the real pipeline output;
if a future change makes ROC-AUC differ before/after recalibration, that is
a bug (the slope got refit, rows got shuffled, or before/after are scored
on different subsets), not a modelling result.

``LOGIT_CLIP_EPS`` (1e-6) clips probabilities away from exactly 0/1 before
taking a logit anywhere in this module -- isotonic calibration in
particular can output values pinned exactly at the [0, 1] boundary
(:mod:`nightingale.calibrate`'s ``IsotonicCalibrator``, ``y_min=0.0,
y_max=1.0``), and an unclipped logit there is +/-inf. 1e-6 is small enough
to be invisible against the probability spread these models actually
produce, while keeping every logit finite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from nightingale.calibrate import ece
from nightingale.clean import CLEANED_ROOT
from nightingale.evaluate import metric_ci
from nightingale.model import (
    FoldIntegrityError,
    _feature_columns,
    _fit_deployment_calibrator,
    _most_frequent_params,
    _one_hot_fold_safe,
    _outer_splits,
    _select_inner_params,
)
from nightingale.sentinel import assert_calibrator_held_out

TARGET_SITES = ("hungarian", "switzerland", "va")
TRAINING_SITE = "cleveland"

CALIBRATION_FRACTION = 0.3
N_BOOT = 2000

# See the module docstring's last paragraph for the rationale.
LOGIT_CLIP_EPS = 1e-6

_INTERCEPT_MAX_ITER = 100
_INTERCEPT_TOL = 1e-10

_METRIC_FNS = {
    "roc_auc": roc_auc_score,
    "pr_auc": average_precision_score,
    "brier": brier_score_loss,
    "ece": ece,
}
_METRIC_ORDER = ("roc_auc", "pr_auc", "brier", "ece")


# ---------------------------------------------------------------------------
# Logit-space primitives
# ---------------------------------------------------------------------------


def _logit(p, eps: float = LOGIT_CLIP_EPS) -> np.ndarray:
    """logit(clip(p, eps, 1 - eps)) -- see the module docstring for why the clip exists."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(z) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))


def calibration_in_the_large(y, p) -> tuple[float, float]:
    """Unconstrained logistic recalibration of y on logit(p): (intercept, slope).

    Fits ``LogisticRegression(C=np.inf, ...)`` (unregularised MLE, matching
    :mod:`nightingale.calibrate`'s own ``_fit_sigmoid``) on a single feature
    -- ``logit(p)`` -- so ``intercept``/``slope`` are exactly this model's
    ``intercept_``/``coef_``. A perfectly calibrated ``p`` recovers
    intercept ~= 0, slope ~= 1. Intercept != 0 signals a prevalence
    mismatch (the reported probabilities are systematically too high/low in
    aggregate); slope != 1 signals over-confidence (< 1, extreme
    probabilities are too extreme) or under-confidence (> 1). This is a
    DIAGNOSTIC only -- it never adjusts ``p``, it just measures how far
    from calibrated it is. Probabilities are clipped via :func:`_logit`
    before the logit transform.
    """
    y = np.asarray(y)
    z = _logit(p).reshape(-1, 1)

    clf = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=10000)
    clf.fit(z, y)

    intercept = float(clf.intercept_[0])
    slope = float(clf.coef_[0][0])
    return intercept, slope


def _fit_intercept_only(y, p, eps: float = LOGIT_CLIP_EPS) -> float:
    """Fit ``c`` in ``sigmoid(logit(p) + c)`` by 1-D Newton-Raphson MLE (slope fixed at 1).

    Unlike :func:`calibration_in_the_large`, the slope is NOT free here --
    by construction this only ever corrects a prevalence-style shift, never
    the model's discrimination (that's what makes the recalibrated ROC-AUC
    identical to the pre-recalibration ROC-AUC: see the module docstring).

    The 1-parameter binomial log-likelihood in ``c`` is concave (its second
    derivative, ``-sum(p_c * (1 - p_c))``, is <= 0 everywhere), so Newton's
    method converges in a handful of iterations from ``c=0`` regardless of
    how far off the starting point is -- no external optimiser dependency
    needed for a 1-D problem this well-behaved.
    """
    y = np.asarray(y, dtype=float)
    z = _logit(p, eps=eps)

    c = 0.0
    for _ in range(_INTERCEPT_MAX_ITER):
        p_c = _sigmoid(z + c)
        grad = float(np.sum(y - p_c))
        hess = float(np.sum(p_c * (1.0 - p_c)))
        if hess < 1e-12:
            break
        step = grad / hess
        c += step
        if abs(step) < _INTERCEPT_TOL:
            break
    return float(c)


# ---------------------------------------------------------------------------
# 30/70 site-local calibration/evaluation split
# ---------------------------------------------------------------------------


def _calibration_eval_split(
    y, seed: int, cal_fraction: float = CALIBRATION_FRACTION
) -> tuple[np.ndarray, np.ndarray]:
    """A seeded, stratified 30/70 split of POSITIONAL indices into (cal_idx, eval_idx).

    Positional (0..len(y)-1), not pandas labels -- matching
    :mod:`nightingale.model`'s own convention for
    :class:`~nightingale.sentinel.assert_calibrator_held_out`'s contract
    (disjoint index collections; see :func:`nightingale.model._outer_splits`,
    which returns exactly this shape for the outer CV folds). Stratified on
    ``y`` so the 30% calibration slice doesn't starve the minority class at
    a small site (Switzerland's 123 rows are 93.5% positive -- only ~8
    negative rows in total).
    """
    y = np.asarray(y)
    positions = np.arange(len(y))
    cal_idx, eval_idx = train_test_split(
        positions, train_size=cal_fraction, random_state=seed, stratify=y
    )
    return cal_idx, eval_idx


# ---------------------------------------------------------------------------
# Bootstrap metric bundle
# ---------------------------------------------------------------------------


def _bootstrap_metrics(y, p, seed: int, n_boot: int = N_BOOT) -> dict:
    """roc_auc/pr_auc/brier/ece, each as a bootstrap ``[point, lo, hi]`` triple.

    Mirrors :func:`nightingale.evaluate.evaluate_oof`'s "distinct
    per-metric seed" discipline (``seed + k`` for the k-th metric, fixed
    order roc_auc/pr_auc/brier/ece) -- sharing one seed across all four
    would resample the exact same bootstrap indices for every metric,
    correlating four otherwise-independent intervals for no reason.
    :func:`nightingale.evaluate.metric_ci` is the actual resampling
    primitive, imported and reused rather than reimplemented here.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    return {
        name: list(metric_ci(y, p, _METRIC_FNS[name], n_boot=n_boot, seed=seed + k))
        for k, name in enumerate(_METRIC_ORDER)
    }


# ---------------------------------------------------------------------------
# Cleveland-only training (train_condition's protocol, site-filtered)
# ---------------------------------------------------------------------------


def _load_heart_disease() -> pd.DataFrame:
    """Read the committed data/cleaned/heart-disease.csv.gz directly (fresh-clone-safe)."""
    return pd.read_csv(CLEANED_ROOT / "heart-disease.csv.gz", compression="gzip")


def _train_cleveland(heart_df: pd.DataFrame, seed: int) -> dict:
    """Nested-CV training restricted to Cleveland's 303 rows.

    Exactly :func:`nightingale.model.train_condition`'s protocol -- outer
    5-fold stratified CV for parameter selection, an inner 3-fold grid
    search per outer fold, fold-safe one-hot encoding, and a deployment
    calibrator fit on the pooled Cleveland OOF ``p_raw`` -- just
    orchestrated over a site-filtered frame instead of a whole condition's
    cleaned CSV (``train_condition`` has no site-filtering hook). Every
    step below calls the SAME private helper ``train_condition`` itself
    calls, rather than reimplementing fold-safe encoding or the grid search
    a second time.
    """
    cleveland = heart_df[heart_df["site"] == TRAINING_SITE].reset_index(drop=True)
    y = cleveland["target"].astype(int)
    feature_cols = _feature_columns(cleveland, target_col="target")
    X = cleveland[feature_cols]
    n = len(cleveland)

    splits = _outer_splits(X, y, seed)

    oof_p_raw = np.full(n, np.nan)
    seen_count = np.zeros(n, dtype=int)
    fold_params: list[dict] = []

    for train_idx, eval_idx in splits:
        # Live leak guard -- identical role to train_condition's own call:
        # this fold's model must not have trained on the rows it is about
        # to predict, since those predictions feed the pooled OOF sample
        # the deployment calibrator is fit on below.
        assert_calibrator_held_out(train_idx, eval_idx)

        train_df, eval_df = X.iloc[train_idx], X.iloc[eval_idx]
        y_train = y.iloc[train_idx]

        X_train_enc, X_eval_enc = _one_hot_fold_safe(train_df, eval_df)

        params = _select_inner_params(X_train_enc, y_train, seed)
        fold_params.append(params)

        model = xgb.XGBClassifier(tree_method="hist", random_state=seed, **params)
        model.fit(X_train_enc, y_train)
        p = model.predict_proba(X_eval_enc)[:, 1]

        oof_p_raw[eval_idx] = p
        seen_count[eval_idx] += 1

    if not np.all(seen_count == 1):
        n_missing = int(np.sum(seen_count == 0))
        n_duplicated = int(np.sum(seen_count > 1))
        raise FoldIntegrityError(
            f"cleveland: outer fold assignment broken -- {n_missing} row(s) "
            f"never predicted, {n_duplicated} row(s) predicted more than once"
        )

    oof_for_calibrator = pd.DataFrame(
        {"y_true": y.to_numpy(dtype=int), "p_raw": oof_p_raw}
    )
    calibrator = _fit_deployment_calibrator(oof_for_calibrator)
    calibration_method = calibrator.export()["type"]

    chosen_params = _most_frequent_params(fold_params)

    X_all_enc = pd.get_dummies(X)
    final_model = xgb.XGBClassifier(tree_method="hist", random_state=seed, **chosen_params)
    final_model.fit(X_all_enc, y)

    return {
        "final_model": final_model,
        "X_all_enc_columns": list(X_all_enc.columns),
        "X_cleveland_features": X,
        "calibrator": calibrator,
        "calibration_method": calibration_method,
        "chosen_params": chosen_params,
        "fold_params": fold_params,
        "feature_cols": feature_cols,
        "n": n,
    }


def _score_site(cleveland_train: dict, site_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Score the Cleveland-trained-and-calibrated pipeline on a target site.

    Returns ``(y_site, p_naive)`` -- ``p_naive`` is ``final_model``'s raw
    prediction run through the Cleveland-fitted deployment calibrator: what
    a "train at one hospital, deploy elsewhere unmodified" pipeline would
    actually output. Encoding reuses :func:`nightingale.model._one_hot_fold_safe`
    directly, called with Cleveland's own feature frame as the "training
    fold" and the target site as the "eval fold" -- a category that exists
    only at the target site can never create a column final_model wasn't
    fit on, by the same construction :mod:`nightingale.model` relies on for
    ordinary CV folds.
    """
    X_cleveland = cleveland_train["X_cleveland_features"]
    feature_cols = cleveland_train["feature_cols"]

    y_site = site_df["target"].astype(int).to_numpy()
    X_site = site_df[feature_cols]

    _, X_site_enc = _one_hot_fold_safe(X_cleveland, X_site)

    p_raw = cleveland_train["final_model"].predict_proba(X_site_enc)[:, 1]
    p_naive = cleveland_train["calibrator"].predict(p_raw)
    return y_site, np.asarray(p_naive, dtype=float)


# ---------------------------------------------------------------------------
# JSON safety net (mirrors scripts/train_all.py's own _json_safe, applied
# here so transfer_study's return value is guaranteed JSON-safe regardless
# of what writes it to disk -- see the CONTRACT's "byte-reproducible" note).
# ---------------------------------------------------------------------------


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if (np.isnan(value) or np.isinf(value)) else value
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


# ---------------------------------------------------------------------------
# The study
# ---------------------------------------------------------------------------


def transfer_study(seed: int = 42) -> dict:
    """Train on Cleveland only; report naive and intercept-recalibrated transfer to the other three sites.

    See the module docstring for the full protocol. Deterministic given
    ``seed`` -- no wall-clock timestamp anywhere in the returned dict, so
    two calls with the same seed produce byte-identical JSON (pinned by
    ``tests/test_external.py::test_transfer_study_is_deterministic_across_two_independent_calls``).

    **Schema, and why it's shaped this way (fix round 1):** each site's
    ``"naive"`` and ``"recalibrated"`` blocks are DIRECT SIBLINGS, and both
    are computed on the IDENTICAL 70% evaluation row set -- ``"naive"`` is
    ``p_naive`` scored on ``eval_idx`` (no recalibration applied),
    ``"recalibrated"`` is the intercept-corrected ``p_recal`` scored on the
    SAME ``eval_idx``. This is deliberate: an earlier version of this
    function put the recalibrated-AFTER metrics directly under
    ``"recalibrated"`` and the pre-recalibration-on-the-same-70%-rows
    metrics one level deeper, under ``"recalibrated"["before"]`` -- with the
    WHOLE-SITE naive numbers occupying the sibling ``"naive"`` slot instead.
    That let a reader compare the two top-level siblings ``"naive"``
    (whole site) against ``"recalibrated"`` (70% subset, after) and observe
    ROC-AUC apparently CHANGE under recalibration -- not because
    recalibration changed anything (it provably can't: see the module
    docstring's monotone-transform argument), but purely because the two
    numbers being compared came from two DIFFERENT row sets. The actual
    before/after pair (both on the 70% subset) WAS bit-identical even in
    that version; the defect was that the schema made it trivial to compare
    the wrong two numbers and get a wrong, alarming-looking answer. Now
    ``site["naive"]["roc_auc"][0] == site["recalibrated"]["roc_auc"][0]``
    holds directly, for the two fields anyone would naturally compare.

    The whole-site "what if there's no site-local calibration data at all"
    numbers are still reported -- genuinely informative on their own -- but
    live in a clearly, unambiguously separate ``"naive_all_rows"`` block
    with its own ``n``, so nothing can mistake it for the like-for-like pair.
    """
    heart_df = _load_heart_disease()
    cleveland_train = _train_cleveland(heart_df, seed)

    sites_out: dict[str, dict] = {}

    for site_offset, site in enumerate(TARGET_SITES):
        site_df = heart_df[heart_df["site"] == site].reset_index(drop=True)
        y_site, p_naive_site = _score_site(cleveland_train, site_df)

        n = len(y_site)
        n_positive = int(y_site.sum())

        # Each site gets its own 1000-wide seed band, and each block within
        # a site (naive_all_rows / naive / recalibrated) its own 100-wide
        # sub-band, so every one of this study's ~36 bootstrap resampling
        # streams (3 sites x 3 blocks x 4 metrics) is distinct -- see
        # _bootstrap_metrics' docstring for why correlated bootstrap seeds
        # would be a defect.
        site_seed_base = seed + site_offset * 1000

        # Whole-site reference numbers -- informative (what transfer looks
        # like with zero site-local data), but NEVER the like-for-like
        # before/after pair. Kept clearly separate; see the docstring above.
        naive_all_rows_block = _bootstrap_metrics(y_site, p_naive_site, seed=site_seed_base)
        all_rows_intercept, all_rows_slope = calibration_in_the_large(y_site, p_naive_site)
        naive_all_rows_block["calibration_intercept"] = all_rows_intercept
        naive_all_rows_block["calibration_slope"] = all_rows_slope

        # Seeded 30/70 site-local split. assert_calibrator_held_out is
        # called live here, exactly as nightingale.model calls it on every
        # outer CV fold -- the 30% calibration rows must never leak into
        # the 70% evaluation rows both "naive" and "recalibrated" below
        # are scored on.
        cal_idx, eval_idx = _calibration_eval_split(y_site, seed=seed)
        assert_calibrator_held_out(cal_idx, eval_idx)

        y_cal, p_cal_set = y_site[cal_idx], p_naive_site[cal_idx]
        y_eval, p_eval = y_site[eval_idx], p_naive_site[eval_idx]

        fitted_intercept = _fit_intercept_only(y_cal, p_cal_set)
        p_recal_eval = _sigmoid(_logit(p_eval) + fitted_intercept)

        # "naive" and "recalibrated" are the like-for-like pair: SAME
        # eval_idx rows, only the probabilities differ (p_eval vs
        # p_recal_eval). ROC-AUC/PR-AUC must therefore match exactly
        # between them (monotone-transform invariance) -- verified by
        # tests/test_external.py against this exact published path.
        naive_block = _bootstrap_metrics(y_eval, p_eval, seed=site_seed_base + 100)
        naive_intercept, naive_slope = calibration_in_the_large(y_eval, p_eval)
        naive_block["calibration_intercept"] = naive_intercept
        naive_block["calibration_slope"] = naive_slope

        recalibrated_block = _bootstrap_metrics(y_eval, p_recal_eval, seed=site_seed_base + 200)
        recal_intercept, recal_slope = calibration_in_the_large(y_eval, p_recal_eval)
        recalibrated_block["calibration_intercept"] = recal_intercept
        recalibrated_block["calibration_slope"] = recal_slope

        sites_out[site] = {
            "n": n,
            "n_positive": n_positive,
            "prevalence": n_positive / n,
            "n_calibration": int(len(cal_idx)),
            "n_evaluation": int(len(eval_idx)),
            "fitted_intercept": fitted_intercept,
            "naive_all_rows": naive_all_rows_block,
            "naive": naive_block,
            "recalibrated": recalibrated_block,
        }

    result = {
        "seed": seed,
        "training_site": TRAINING_SITE,
        "target_sites": list(TARGET_SITES),
        "clipping_epsilon": LOGIT_CLIP_EPS,
        "calibration_split_fraction": CALIBRATION_FRACTION,
        "n_boot": N_BOOT,
        "cleveland_training": {
            "n": cleveland_train["n"],
            "chosen_params": cleveland_train["chosen_params"],
            "fold_params": cleveland_train["fold_params"],
            "calibration_method": cleveland_train["calibration_method"],
            "feature_names": cleveland_train["X_all_enc_columns"],
        },
        "sites": sites_out,
    }
    return _json_safe(result)
