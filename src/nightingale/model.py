# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Nested-CV training with held-out calibration -- the modelling core.

:func:`train_condition` is the public entry point. For one condition it
runs an outer 5-fold stratified CV to produce out-of-fold (OOF) raw
predictions (the pooled evaluation artifact every published metric derives
from -- see :mod:`nightingale.evaluate`), refits a final model on all rows,
and fits a probability calibrator on the pooled OOF predictions.

Protocol, exactly:

- Outer ``StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)``
  produces the OOF predictions: p_raw for a row always comes from a model
  that never saw that row during training.
- For each outer fold, an inner 3-fold ``StratifiedKFold`` grid search over
  ``max_depth x n_estimators x learning_rate`` (see :data:`GRID_MAX_DEPTH`
  etc.) picks that fold's params, fit ONLY on that outer fold's training
  rows (:func:`_select_inner_params`).
- ``XGBClassifier(tree_method="hist")``; XGBoost's own missing-value split
  logic handles NaN -- no imputation happens anywhere in this module.
- Categorical/object columns are one-hot encoded fold-locally
  (:func:`_one_hot_fold_safe`): the dummy-column set is derived from the
  training fold only, and the eval fold is reindexed onto it (missing
  columns filled 0) -- an eval-only category can never create a column,
  by construction, not by convention.
- ``site`` (heart-disease's contributing-hospital column) is dropped: it is
  provenance metadata, never a model feature.
- diabetes only: the *inner* grid search subsamples each outer training
  fold to :data:`DIABETES_INNER_SUBSAMPLE_N` rows (stratified, seeded) to
  keep the 12-combo x 3-inner-fold search affordable at 253,680 rows; the
  *outer* folds always fit/evaluate on the full, unsampled data.
- ``final_model`` is refit on ALL rows using the most-frequently-chosen
  params across the 5 outer folds (:func:`_most_frequent_params` documents
  the tie-break rule).
- ``TrainResult.calibrator`` is fit once, on the pooled OOF ``p_raw`` --
  never on ``final_model``'s own in-sample training predictions.
  :func:`_fit_deployment_calibrator` takes the OOF frame itself (not bare
  ``y``/``p_raw`` arrays) specifically so that substitution can't be made
  by quietly swapping one argument at the call site -- making exactly that
  swap on purpose (fit ``final_model`` first, then calibrate on its own
  ``predict_proba``) went undetected by every other test in this suite,
  because none of them asserted the calibrator's fitted VALUES;
  ``tests/test_model.py::test_deployment_calibrator_matches_independent_oof_refit``
  now does. This is the DEPLOYMENT calibrator: the artifact
  :mod:`nightingale.export` writes into ``model.json`` for scoring
  genuinely new rows, where "fit on all the OOF signal we have" is correct
  and desirable.
- ``oof["p_cal"]``, by contrast, is produced by CROSS-FITTED calibration
  (:func:`_cross_fit_calibration`): for each outer fold k, a fresh
  calibrator is fit on every OOF row NOT in fold k and applied only to fold
  k's rows. A single calibrator fit on the full pooled OOF and then scored
  on that same pool would be in-sample for the calibrator itself (even
  though the underlying ``p_raw`` values are genuinely out-of-fold for the
  base model) -- isotonic regression in particular is flexible enough to
  reproduce its own training sample's empirical bin frequencies almost
  exactly, which is what made an earlier version of this module report a
  calibrated ECE of ~1e-16-1e-19: not a result, an artifact of scoring a
  calibrator on its own training data. Every published metric MUST be
  computed from ``oof["p_cal"]``, never from re-scoring
  ``TrainResult.calibrator`` against the pooled OOF it was fit on.
  ``assert_calibrator_held_out`` is called live here too, once per
  cross-fit, on that fold's real held-out/scored index split.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from nightingale.calibrate import Calibrator, ece, fit_calibrator, pick_calibration
from nightingale.clean import CLEANED_ROOT
from nightingale.conditions import CONDITIONS
from nightingale.sentinel import assert_calibrator_held_out

OUTER_N_SPLITS = 5
INNER_N_SPLITS = 3

GRID_MAX_DEPTH = (2, 3, 4)
GRID_N_ESTIMATORS = (100, 300)
GRID_LEARNING_RATE = (0.05, 0.1)

# Non-feature columns dropped before training regardless of condition. Only
# heart-disease's cleaned frame actually carries "site"; excluding it
# unconditionally is harmless for the other five (a no-op `in` check).
_NON_FEATURE_COLUMNS = ("site",)

DIABETES_SLUG = "diabetes"
# For the inner grid search only: 253,680 rows x 12 combos x 3
# inner folds is not worth the wall-clock; a 20k stratified seeded subsample
# keeps the search affordable while the outer folds still see every row.
DIABETES_INNER_SUBSAMPLE_N = 20_000


class FoldIntegrityError(Exception):
    """Raised when the outer CV fold assignment breaks "predicted exactly once".

    OOF predictions are only meaningful if every row is covered by exactly
    one outer eval fold. A row missing from every eval fold, or claimed by
    more than one, means the pooled OOF array silently mixes rows that were
    never scored with rows scored more than once -- this is checked
    explicitly rather than trusted to fall out of the CV splitter.
    """


@dataclass(frozen=True)
class TrainResult:
    """Everything :func:`train_condition` produces for one condition.

    ``oof`` has exactly one row per input row (index-aligned to the cleaned
    frame, original row order), columns ``y_true`` (int), ``p_raw`` (float),
    ``p_cal`` (float), ``fold`` (int, 0..``OUTER_N_SPLITS - 1``).
    ``feature_names`` lists the post-one-hot-encoding feature columns of
    ``final_model``, in its column order.
    """

    final_model: xgb.XGBClassifier
    oof: pd.DataFrame
    chosen_params: dict
    cv_summary: dict
    feature_names: list[str]
    calibrator: Calibrator


def _load_cleaned(slug: str) -> pd.DataFrame:
    """Read the committed ``data/cleaned/<slug>.csv.gz`` artifact directly.

    Never calls :func:`nightingale.clean.clean` (which needs ``data/raw/``,
    gitignored) -- the cleaned CSVs are committed, so training works on a
    fresh clone with no fetch/clean step required first.
    """
    path = CLEANED_ROOT / f"{slug}.csv.gz"
    return pd.read_csv(path, compression="gzip")


def _feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """Every column except the target and non-feature columns (``site``)."""
    excluded = {target_col, *_NON_FEATURE_COLUMNS}
    return [c for c in df.columns if c not in excluded]


def _one_hot_fold_safe(
    train_df: pd.DataFrame, eval_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One-hot encode object/categorical columns using ONLY the training fold.

    ``pd.get_dummies`` is called separately on ``train_df`` and ``eval_df``;
    ``eval_df``'s result is then reindexed onto ``train_df``'s dummy-column
    set, with any column ``train_df`` doesn't have filled 0. A category that
    appears only in ``eval_df`` therefore never creates a dummy column at
    all -- it silently reindexes away, mapping to all-zero across the
    training fold's known categories. This is fold-safe by construction:
    nothing about which columns exist can depend on the eval fold's values.
    Numeric columns (including NaN) pass through untouched either way --
    XGBoost's ``tree_method="hist"`` handles NaN natively.
    """
    X_train = pd.get_dummies(train_df)
    X_eval = pd.get_dummies(eval_df).reindex(columns=X_train.columns, fill_value=0)
    return X_train, X_eval


def _outer_splits(X: pd.DataFrame, y: pd.Series, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """The outer 5-fold stratified split that produces the OOF predictions.

    Factored out (rather than inlined in :func:`train_condition`) so tests
    can monkeypatch it to inject a corrupted split and prove the live
    integrity/leakage guards downstream actually fire.
    """
    cv = StratifiedKFold(n_splits=OUTER_N_SPLITS, shuffle=True, random_state=seed)
    return list(cv.split(X, y))


def _grid_combos() -> list[dict]:
    """The 12-combination grid, in a fixed, deterministic enumeration order.

    Order matters for :func:`_select_inner_params`'s tie-break rule: ties
    are won by whichever combination is reached first in this order
    (max_depth outermost, then n_estimators, then learning_rate, each
    ascending).
    """
    return [
        {"max_depth": max_depth, "n_estimators": n_estimators, "learning_rate": learning_rate}
        for max_depth in GRID_MAX_DEPTH
        for n_estimators in GRID_N_ESTIMATORS
        for learning_rate in GRID_LEARNING_RATE
    ]


def _stratified_subsample(
    X: pd.DataFrame, y: pd.Series, n: int, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """A seeded stratified subsample of at most ``n`` rows of ``(X, y)``.

    Returns ``(X, y)`` unchanged when ``len(X) <= n`` -- subsampling only
    ever shrinks, never pads or resamples.
    """
    if len(X) <= n:
        return X, y
    sub_idx, _ = train_test_split(
        np.arange(len(X)), train_size=n, stratify=y, random_state=seed
    )
    return X.iloc[sub_idx], y.iloc[sub_idx]


def _select_inner_params(
    X_train_enc: pd.DataFrame,
    y_train: pd.Series,
    seed: int,
    subsample_n: int | None = None,
) -> dict:
    """Inner 3-fold grid search; returns the params with the best mean inner ROC-AUC.

    ``subsample_n`` (diabetes only) subsamples ``X_train_enc``/``y_train``
    (stratified, seeded) before the search -- the search sees at most that
    many rows, keeping the 12 x 3 = 36 fits per outer fold affordable at
    diabetes's scale. The outer fold's own eval predictions are unaffected:
    this function only ever touches the grid-search input, never what the
    winning params are subsequently fit or evaluated on.

    Tie-break rule: :func:`_grid_combos` enumerates the grid in a fixed
    order; each candidate's mean inner-fold ROC-AUC only replaces the
    running best on a strict ``>``, so the first combination (in that fixed
    order) to reach the maximum score wins any tie.

    Each candidate fit uses ``n_jobs=1``: this loop does up to 12 combos x 3
    inner folds = 36 small/medium fits, where per-fit thread-pool spin-up
    overhead dominates actual compute -- empirically ~3x faster overall than
    letting each tiny fit claim every core. The outer-fold model and the
    final refit (far fewer, larger fits) are unaffected by this and keep
    XGBoost's default multi-threading.
    """
    if subsample_n is not None:
        X_search, y_search = _stratified_subsample(X_train_enc, y_train, subsample_n, seed)
    else:
        X_search, y_search = X_train_enc, y_train

    inner_cv = StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=seed)
    best_params: dict | None = None
    best_score = -np.inf

    for params in _grid_combos():
        scores = []
        for tr_idx, va_idx in inner_cv.split(X_search, y_search):
            model = xgb.XGBClassifier(tree_method="hist", random_state=seed, n_jobs=1, **params)
            model.fit(X_search.iloc[tr_idx], y_search.iloc[tr_idx])
            p = model.predict_proba(X_search.iloc[va_idx])[:, 1]
            scores.append(roc_auc_score(y_search.iloc[va_idx], p))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    assert best_params is not None  # grid is non-empty by construction
    return best_params


def _most_frequent_params(fold_params: list[dict]) -> dict:
    """The mode of the per-outer-fold chosen params.

    Tie-break rule: among params tied for the highest occurrence count, the
    one that appears earliest by outer-fold order (fold 0, then fold 1, ...)
    wins -- deterministic, and cheap to audit by eye against
    ``cv_summary["fold_params"]``.
    """
    keyed = [tuple(sorted(p.items())) for p in fold_params]
    counts = Counter(keyed)
    max_count = max(counts.values())
    for k in keyed:
        if counts[k] == max_count:
            return dict(k)
    raise AssertionError("unreachable: fold_params must be non-empty")  # pragma: no cover


def _cross_fit_calibration(oof: pd.DataFrame, method: str) -> np.ndarray:
    """Out-of-sample ``p_cal`` via cross-fitted calibration over the outer folds.

    For each outer fold k, fits a fresh calibrator of ``method`` on every
    OOF row NOT in fold k (their own ``p_raw``/``y_true``), then applies it
    to fold k's rows only. Every returned ``p_cal`` value therefore comes
    from a calibrator that never saw that row -- unlike scoring a single
    calibrator fit on the *full* pooled OOF against that same pool, which
    is in-sample for the calibrator (see the module docstring for why that
    distinction is load-bearing, not pedantic).

    ``method`` is fixed across every cross-fit rather than re-selected via
    :func:`nightingale.calibrate.pick_calibration` per fold. Two reasons:
    (1) it keeps the reported calibrated-ECE an honest out-of-sample
    estimate of the SAME family ``TrainResult.calibrator`` actually
    deploys, rather than a metric for a method that might differ fold to
    fold; (2) re-running method selection on ~4/5 of an already modest OOF
    sample, 5 times, adds selection variance the metric doesn't need.
    Callers pick ``method`` once, from :func:`pick_calibration` on the full
    pooled OOF (the same selection ``TrainResult.calibrator`` used).

    ``assert_calibrator_held_out`` is called once per fold: ``cal_idx``
    (the rows fitting this fold's calibrator) and ``eval_idx`` (fold k's
    own rows, being scored) are disjoint by construction of the boolean
    mask below, and this call proves that live rather than trusting it.
    """
    y_true = oof["y_true"].to_numpy()
    p_raw = oof["p_raw"].to_numpy()
    fold = oof["fold"].to_numpy()
    n = len(oof)
    positions = np.arange(n)

    p_cal = np.full(n, np.nan)
    for k in sorted(set(fold.tolist())):
        eval_mask = fold == k
        cal_mask = ~eval_mask

        assert_calibrator_held_out(positions[cal_mask], positions[eval_mask])

        fold_calibrator = fit_calibrator(y_true[cal_mask], p_raw[cal_mask], method=method)
        p_cal[eval_mask] = fold_calibrator.predict(p_raw[eval_mask])

    assert not np.any(np.isnan(p_cal))  # every row's fold is in `fold`'s own value set
    return p_cal


def _fit_deployment_calibrator(oof: pd.DataFrame) -> Calibrator:
    """Fit the calibrator ``TrainResult.calibrator`` exports for scoring new rows.

    Takes the pooled OOF frame -- and nothing else -- on purpose. This is
    the ONLY out-of-fold signal this module has about the base model's
    calibration quality, and the deployment calibrator MUST be fit from
    ``oof["p_raw"]``/``oof["y_true"]``, never from ``final_model``'s own
    (in-sample) training predictions: that substitution is precisely the
    leak this function's signature is built to make hard to commit by
    accident. A bare ``pick_calibration(y, p_raw)`` call at the
    ``train_condition`` call site could have ``p_raw`` swapped for
    ``final_model.predict_proba(X_all_enc)[:, 1]`` in a quiet one-line
    edit -- and making exactly that edit on purpose showed the rest of the
    test suite (correctly) noticed nothing, because nothing before that
    check asserted the calibrator's actual fitted VALUES.
    Requiring the OOF DataFrame itself means reproducing that mutation
    needs a visibly fabricated stand-in DataFrame, not a one-token swap --
    and test_model.py's
    test_deployment_calibrator_matches_independent_oof_refit asserts the
    exported values directly, so even that visible fabrication is caught.
    """
    return pick_calibration(oof["y_true"].to_numpy(), oof["p_raw"].to_numpy())


def train_condition(slug: str, seed: int = 42) -> TrainResult:
    """Nested-CV training with held-out calibration for one condition.

    See the module docstring for the full protocol. Raises
    :class:`FoldIntegrityError` if the outer fold assignment doesn't predict
    every row exactly once, and :class:`nightingale.sentinel.LeakageError`
    if any outer fold's (train, eval) indices overlap (both defensive: a
    correct :class:`~sklearn.model_selection.StratifiedKFold` split never
    trips either check).
    """
    condition = CONDITIONS[slug]
    df = _load_cleaned(slug)
    target_col = condition.target

    y = df[target_col].astype(int)
    feature_cols = _feature_columns(df, target_col)
    X = df[feature_cols]
    n = len(df)

    subsample_n = DIABETES_INNER_SUBSAMPLE_N if slug == DIABETES_SLUG else None

    splits = _outer_splits(X, y, seed)

    oof_p_raw = np.full(n, np.nan)
    oof_fold = np.full(n, -1, dtype=int)
    seen_count = np.zeros(n, dtype=int)
    fold_params: list[dict] = []

    for fold_idx, (train_idx, eval_idx) in enumerate(splits):
        # Live leak guard, not decoration: this fold's model must not have
        # trained on the rows it is about to predict, because those
        # predictions become part of the pooled OOF data the calibrator is
        # fit on below. A correct StratifiedKFold split always passes this;
        # tests/test_model.py's mutation test A corrupts the split to prove
        # the guard actually fires from inside train_condition.
        assert_calibrator_held_out(train_idx, eval_idx)

        train_df, eval_df = X.iloc[train_idx], X.iloc[eval_idx]
        y_train, y_eval = y.iloc[train_idx], y.iloc[eval_idx]

        X_train_enc, X_eval_enc = _one_hot_fold_safe(train_df, eval_df)

        params = _select_inner_params(X_train_enc, y_train, seed, subsample_n=subsample_n)
        fold_params.append(params)

        model = xgb.XGBClassifier(tree_method="hist", random_state=seed, **params)
        model.fit(X_train_enc, y_train)
        p = model.predict_proba(X_eval_enc)[:, 1]

        oof_p_raw[eval_idx] = p
        oof_fold[eval_idx] = fold_idx
        seen_count[eval_idx] += 1
        del y_eval  # not needed beyond feeding eval_idx into the OOF arrays above

    if not np.all(seen_count == 1):
        n_missing = int(np.sum(seen_count == 0))
        n_duplicated = int(np.sum(seen_count > 1))
        raise FoldIntegrityError(
            f"{slug}: outer fold assignment broken -- {n_missing} row(s) never "
            f"predicted, {n_duplicated} row(s) predicted more than once "
            f"(every row must be predicted exactly once across the "
            f"{OUTER_N_SPLITS} outer folds)"
        )

    oof = pd.DataFrame(
        {
            "y_true": y.to_numpy(dtype=int),
            "p_raw": oof_p_raw,
            "fold": oof_fold,
        },
        index=df.index,
    )

    # Deployment calibrator: fit once on the pooled OOF p_raw -- never on
    # final_model's own (in-sample, below) training predictions. This is
    # the artifact TrainResult.calibrator exports (via nightingale.export)
    # for scoring genuinely new rows; it is NOT what oof["p_cal"] is
    # computed from below.
    # _fit_deployment_calibrator's signature (takes `oof`, not bare arrays)
    # is deliberately shaped so this line can't be quietly rewired to
    # final_model's in-sample predictions -- see its docstring.
    calibrator = _fit_deployment_calibrator(oof)
    chosen_calibration_method = calibrator.export()["type"]

    # oof["p_cal"]: cross-fitted, out-of-sample calibration -- see
    # _cross_fit_calibration and the module docstring for why scoring the
    # single deployment `calibrator` against its own pooled-OOF training
    # sample (the previous, incorrect approach) is not a valid metric.
    oof["p_cal"] = _cross_fit_calibration(oof, method=chosen_calibration_method)
    oof = oof[["y_true", "p_raw", "p_cal", "fold"]]

    chosen_params = _most_frequent_params(fold_params)

    X_all_enc = pd.get_dummies(X)
    final_model = xgb.XGBClassifier(tree_method="hist", random_state=seed, **chosen_params)
    final_model.fit(X_all_enc, y)
    feature_names = list(X_all_enc.columns)

    oof_roc_auc = float(roc_auc_score(oof["y_true"], oof["p_raw"]))
    oof_brier = float(brier_score_loss(oof["y_true"], oof["p_raw"]))
    ece_uncalibrated = ece(oof["y_true"].to_numpy(), oof["p_raw"].to_numpy())
    ece_calibrated = ece(oof["y_true"].to_numpy(), oof["p_cal"].to_numpy())

    cv_summary = {
        "seed": seed,
        "outer_n_splits": OUTER_N_SPLITS,
        "inner_n_splits": INNER_N_SPLITS,
        "grid": {
            "max_depth": list(GRID_MAX_DEPTH),
            "n_estimators": list(GRID_N_ESTIMATORS),
            "learning_rate": list(GRID_LEARNING_RATE),
        },
        "fold_params": fold_params,
        "chosen_params": chosen_params,
        "oof_roc_auc": oof_roc_auc,
        "oof_brier": oof_brier,
        "ece_uncalibrated": ece_uncalibrated,
        "ece_calibrated": ece_calibrated,
        "calibration_method": chosen_calibration_method,
        "diabetes_inner_subsample_n": subsample_n,
    }

    return TrainResult(
        final_model=final_model,
        oof=oof,
        chosen_params=chosen_params,
        cv_summary=cv_summary,
        feature_names=feature_names,
        calibrator=calibrator,
    )
