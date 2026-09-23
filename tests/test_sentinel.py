# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Tests for the leakage/integrity sentinel module (nightingale.sentinel).

Covers the three detectors: scan_sentinel_zeros (implausible exact-0 spikes),
check_split_integrity (train/test row duplication — the resample-before-split
fingerprint), and assert_calibrator_held_out (calibration/eval index overlap).

The real-data tests at the bottom are the point of the module: they prove
scan_sentinel_zeros flags `chol` on the raw Switzerland heart-disease file —
the wild example that motivated writing this detector in the first place —
and that the cleaner's sentinel-to-NaN conversion (nightingale.clean) makes
the flag go away, i.e. the cleaner and the detector agree.

The raw-file test needs data/raw/ (gitignored, populated by `fetch`, not by
cloning the repo) and is skipped with an actionable reason when it's absent
-- see conftest.skip_if_raw_missing. The cleaned-data test reads the
committed data/cleaned/heart-disease.csv.gz directly rather than calling
clean(), so it stays raw-data-free and always runs, including on a fresh
clone.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import skip_if_raw_missing
from nightingale.clean import CLEANED_ROOT
from nightingale.fetch import RAW_ROOT
from nightingale.sentinel import (
    LeakageError,
    SplitReport,
    assert_calibrator_held_out,
    check_split_integrity,
    scan_sentinel_zeros,
)

# ---------------------------------------------------------------------------
# scan_sentinel_zeros
# ---------------------------------------------------------------------------

# One synthetic frame, four columns, each built deterministically (no RNG) so
# the exact-fraction assertions below are stable:
#
# - chol_like:    far-from-zero distribution (~180-280) with 15% of rows
#                 forced to exactly 0 — an injected sentinel-zero spike.
# - age_like:     far-from-zero distribution, no zeros at all — a clean
#                 column that must not be flagged.
# - flag:         binary {0, 1} indicator, 90% zero — zeros are meaningful
#                 category membership here, not a missing-value sentinel.
# - oldpeak_like: bounded-below-by-zero distribution whose real low end
#                 legitimately reaches 0 (10% exact zeros are genuine
#                 readings, not sentinels) — zero is within the nonzero
#                 values' expected low range, not an outlier.


@pytest.fixture
def synthetic_frame():
    chol_like = np.concatenate([np.zeros(30), np.linspace(180, 280, 170)])
    age_like = np.linspace(30, 80, 200)
    flag = np.array([0] * 180 + [1] * 20)
    oldpeak_like = np.concatenate([np.zeros(20), np.linspace(0.1, 6.0, 180)])
    return pd.DataFrame(
        {
            "chol_like": chol_like,
            "age_like": age_like,
            "flag": flag,
            "oldpeak_like": oldpeak_like,
        }
    )


def test_injected_zero_spike_is_flagged(synthetic_frame):
    result = scan_sentinel_zeros(synthetic_frame)
    assert "chol_like" in result
    assert result["chol_like"] == pytest.approx(0.15)


def test_clean_far_from_zero_column_not_flagged(synthetic_frame):
    result = scan_sentinel_zeros(synthetic_frame)
    assert "age_like" not in result


def test_binary_indicator_column_not_flagged(synthetic_frame):
    result = scan_sentinel_zeros(synthetic_frame)
    assert "flag" not in result


def test_legitimate_low_end_zero_not_flagged(synthetic_frame):
    result = scan_sentinel_zeros(synthetic_frame)
    assert "oldpeak_like" not in result


def test_min_frac_threshold_excludes_small_spikes():
    # A handful of zeros (well under the 5% default threshold) in an
    # otherwise far-from-zero column should not be flagged, even though
    # those zeros would be implausible outliers if there were enough of them.
    df = pd.DataFrame({"x": np.concatenate([np.zeros(2), np.linspace(180, 280, 198)])})
    result = scan_sentinel_zeros(df)
    assert "x" not in result


# ---------------------------------------------------------------------------
# check_split_integrity
# ---------------------------------------------------------------------------


def test_duplicated_rows_across_split_are_counted_exactly():
    X_train = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
    # Two of these three test rows are exact duplicates of train rows.
    X_test = pd.DataFrame({"a": [1, 3, 99], "b": [10, 30, 999]})

    report = check_split_integrity(X_train, X_test)

    assert isinstance(report, SplitReport)
    assert report.n_train == 5
    assert report.n_test == 3
    assert report.n_overlap == 2
    assert report.overlap_fraction == pytest.approx(2 / 3)
    assert report.clean is False


def test_disjoint_split_reports_zero_overlap_and_is_clean():
    X_train = pd.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30]})
    X_test = pd.DataFrame({"a": [4, 5, 6], "b": [40, 50, 60]})

    report = check_split_integrity(X_train, X_test)

    assert report.n_overlap == 0
    assert report.overlap_fraction == 0.0
    assert report.clean is True


def test_matching_nan_rows_are_treated_as_overlap():
    # NaN handled consistently: a test row that is NaN in the same position
    # as a matching train row must count as a duplicate, not be silently
    # excluded because NaN != NaN under naive equality.
    X_train = pd.DataFrame({"a": [1.0, float("nan"), 3.0], "b": [10, 20, 30]})
    X_test = pd.DataFrame({"a": [float("nan")], "b": [20]})

    report = check_split_integrity(X_train, X_test)

    assert report.n_overlap == 1


# ---------------------------------------------------------------------------
# assert_calibrator_held_out
# ---------------------------------------------------------------------------


def test_raises_leakage_error_naming_overlap_count_on_overlap():
    cal_idx = [1, 2, 3, 4, 5]
    eval_idx = [4, 5, 6, 7]  # overlap: {4, 5} -> 2

    with pytest.raises(LeakageError) as exc_info:
        assert_calibrator_held_out(cal_idx, eval_idx)

    assert "2" in str(exc_info.value)


def test_passes_silently_on_disjoint_indices():
    assert assert_calibrator_held_out([1, 2, 3], [4, 5, 6]) is None


# ---------------------------------------------------------------------------
# Real-data proof: the detector on the wild example that motivated it
# ---------------------------------------------------------------------------

# The processed.*.data files are headerless; this is the documented UCI 45
# column order (same as nightingale.clean._HEART_COLUMNS).
_HEART_COLUMNS = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
)


def test_raw_switzerland_chol_is_flagged():
    skip_if_raw_missing("heart-disease")

    path = RAW_ROOT / "heart-disease" / "processed.switzerland.data"
    df = pd.read_csv(path, header=None, names=list(_HEART_COLUMNS), na_values="?")

    result = scan_sentinel_zeros(df)

    assert "chol" in result


def test_cleaned_switzerland_chol_is_not_flagged():
    # Proves the cleaner and the detector agree: nightingale.clean's
    # site-scoped chol == 0 -> NaN conversion for Switzerland/VA is exactly
    # what makes this scan stop flagging chol on the cleaned data. Reads the
    # committed data/cleaned/heart-disease.csv.gz artifact directly (rather
    # than calling clean(), which would need data/raw/) so this test needs
    # no raw data and always runs, including on a fresh clone.
    cleaned_path = CLEANED_ROOT / "heart-disease.csv.gz"
    df = pd.read_csv(cleaned_path, compression="gzip")
    swiss = df[df["site"] == "switzerland"]

    result = scan_sentinel_zeros(swiss)

    assert "chol" not in result
