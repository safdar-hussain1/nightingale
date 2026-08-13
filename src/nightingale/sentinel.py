# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Leakage and sentinel-value integrity detectors.

Three independent checks that catch the ways a clinical-data pipeline can
quietly leak information or fool itself into an over-optimistic result:

- :func:`scan_sentinel_zeros` flags numeric columns where an exact-``0``
  spike is implausible given the column's own nonzero distribution — the
  bug class behind Switzerland's heart-disease ``chol`` column (see
  :mod:`nightingale.clean`), where missing serum cholesterol was encoded as
  a literal ``0`` instead of ``NaN``.
- :func:`check_split_integrity` counts train/test row duplication — the
  fingerprint left behind when resampling (SMOTE/ADASYN) is applied
  *before* the train/test split instead of after, which plants
  near-identical synthetic copies of a training row on both sides of the
  split and inflates test-set performance.
- :func:`assert_calibrator_held_out` raises if the indices used to fit a
  probability calibrator overlap the indices used to evaluate it, which
  would let the calibrator's reported quality reflect memorisation rather
  than generalisation.

Dependency-light by design: only pandas is used, so this module can sit
ahead of every later pipeline stage without pulling in heavier dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


class LeakageError(Exception):
    """Raised when calibration and evaluation index sets are not disjoint."""


@dataclass(frozen=True)
class SplitReport:
    """Result of :func:`check_split_integrity`."""

    n_train: int
    n_test: int
    n_overlap: int
    overlap_fraction: float  # n_overlap / n_test; 0.0 when X_test is empty

    @property
    def clean(self) -> bool:
        """True iff no test row appears identically in the training set."""
        return self.n_overlap == 0


def scan_sentinel_zeros(
    df: pd.DataFrame,
    cols: Iterable[str] | None = None,
    min_frac: float = 0.05,
) -> dict[str, float]:
    """Flag numeric columns carrying an implausible spike of exact-0 values.

    A column is flagged — and included in the returned ``{column: fraction}``
    dict, with the fraction of its non-missing values that are exactly 0 —
    when both of the following hold:

    - at least ``min_frac`` of the column's non-missing values are exactly
      ``0``, and
    - ``0`` would be a low outlier relative to the column's own *nonzero*
      values: ``0 < Q1(nonzero) - 1.5 * IQR(nonzero)``, i.e. zero lies below
      the low fence a genuine reading of this column would plausibly reach.

    That second condition is what separates a real missing-value sentinel
    (Switzerland heart-disease ``chol == 0``: cholesterol clusters around
    180-300, so an exact 0 is nowhere near the real distribution) from a
    column whose genuine low end legitimately touches zero (e.g. ST-segment
    depression ``oldpeak``, bounded below by 0, with plenty of real 0
    readings) — the latter is never flagged because 0 sits inside its
    nonzero values' expected low range rather than below it.

    Columns holding only ``{0, 1}`` are always skipped, regardless of their
    zero fraction: a binary indicator's zeros are a meaningful category
    ("absent" / "no"), not a stand-in for a missing measurement, so judging
    them for outlier-ness is a category error rather than a real check.

    ``cols`` restricts the scan to a subset of columns (default: every
    numeric column in ``df``). Non-numeric columns are always skipped.
    """
    if cols is None:
        candidate_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    else:
        candidate_cols = list(cols)

    flagged: dict[str, float] = {}
    for col in candidate_cols:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            continue

        observed = series.dropna()
        if observed.empty:
            continue

        zero_frac = float((observed == 0).sum()) / len(observed)
        if zero_frac < min_frac:
            continue

        unique_values = set(observed.unique().tolist())
        if unique_values == {0, 1}:
            # Genuine binary 0/1 indicator (both values actually occur): the
            # zeros are meaningful, not sentinels. Note this is an exact
            # match, not a subset test — a column holding only {0} (no 1s
            # at all) is not a real indicator, it's degenerate, and falls
            # through to the checks below instead of being excused here.
            continue

        nonzero = observed[observed != 0]
        if nonzero.empty:
            # Every observed value is exactly 0, and (per the check above)
            # this isn't a real 0/1 indicator column — a column where not
            # even one reading ever escapes zero is maximally implausible
            # for anything but a boolean flag, so flag it outright rather
            # than skip for lack of a nonzero distribution to compare to.
            flagged[col] = zero_frac
            continue

        q1 = nonzero.quantile(0.25)
        q3 = nonzero.quantile(0.75)
        iqr = q3 - q1
        low_fence = q1 - 1.5 * iqr
        if low_fence > 0:
            # Zero sits below where a genuine low reading of this column
            # would plausibly fall — implausible, i.e. a sentinel.
            flagged[col] = zero_frac

    return flagged


# Sentinel object used when hashing row tuples so that two rows with NaN in
# the same column position compare equal. Plain float NaN is never equal to
# itself (`nan != nan`), which would otherwise make any row containing a
# missing value impossible to match — silently undercounting train/test
# duplication for the (extremely common, in this clinical data) case where
# the duplicated rows carry missing values.
_NAN_MARKER = object()


def _row_key(row: tuple) -> tuple:
    return tuple(_NAN_MARKER if pd.isna(v) else v for v in row)


def check_split_integrity(X_train: pd.DataFrame, X_test: pd.DataFrame) -> SplitReport:
    """Count X_test rows that appear identically (on shared columns) in X_train.

    Rows are compared via a hash of their (NaN-normalised) value tuples
    rather than a pairwise O(n_train * n_test) comparison, so this stays
    fast even for large splits.

    This is the resample-before-split fingerprint: applying SMOTE/ADASYN
    before splitting plants synthetic near-copies of training rows that can
    land on both sides of the split, so a nonzero overlap here is evidence
    the split happened after resampling instead of before it.
    """
    shared_cols = [c for c in X_train.columns if c in X_test.columns]

    train_keys = {
        _row_key(row) for row in X_train[shared_cols].itertuples(index=False, name=None)
    }
    n_overlap = sum(
        1
        for row in X_test[shared_cols].itertuples(index=False, name=None)
        if _row_key(row) in train_keys
    )

    n_train = len(X_train)
    n_test = len(X_test)
    overlap_fraction = (n_overlap / n_test) if n_test else 0.0

    return SplitReport(
        n_train=n_train,
        n_test=n_test,
        n_overlap=n_overlap,
        overlap_fraction=overlap_fraction,
    )


def assert_calibrator_held_out(cal_idx: Iterable, eval_idx: Iterable) -> None:
    """Raise :class:`LeakageError` if the calibration and eval indices overlap.

    A probability calibrator (Platt scaling, isotonic regression, ...) must
    be fit on data disjoint from whatever it's later evaluated against, or
    its reported calibration quality is measuring memorisation rather than
    generalisation. Passes silently when the two index collections are
    disjoint.
    """
    overlap = set(cal_idx) & set(eval_idx)
    if overlap:
        raise LeakageError(
            f"calibrator not held out: {len(overlap)} index/indices appear in "
            f"both the calibration set and the evaluation set: {sorted(overlap)}"
        )
