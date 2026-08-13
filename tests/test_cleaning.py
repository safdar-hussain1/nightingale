# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Tests for the six per-condition data cleaners (nightingale.clean).

Uses the real, already-fetched raw data under data/raw/ (populated by Task 2's
`fetch`). No test here is network-marked: `clean()` calls `fetch()`, but since
the cache is already populated `fetch()` only verifies checksums locally.

`clean()` is run once per condition for the whole test session (see the
`cleaned_frames` fixture) and every quirk/contract test reads back the actual
written `data/cleaned/<slug>.csv.gz` artifact, not an in-memory object, so the
tests exercise the real on-disk contract every downstream task depends on.
"""

import pandas as pd
import pytest

from nightingale.clean import PIPELINES, CleaningError, _validate, clean
from nightingale.conditions import CONDITIONS

ALL_SLUGS = sorted(CONDITIONS.keys())


@pytest.fixture(scope="session")
def cleaned_frames():
    frames = {}
    for slug in ALL_SLUGS:
        path = clean(slug)
        frames[slug] = pd.read_csv(path, compression="gzip")
    return frames


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_pipelines_registered_for_all_six_slugs():
    assert set(PIPELINES.keys()) == set(CONDITIONS.keys())
    assert len(PIPELINES) == 6


# ---------------------------------------------------------------------------
# breast-cancer
# ---------------------------------------------------------------------------


def test_breast_cancer_target_sum(cleaned_frames):
    df = cleaned_frames["breast-cancer"]
    assert df["target"].sum() == 212


# ---------------------------------------------------------------------------
# cervical-cancer
# ---------------------------------------------------------------------------


def test_cervical_screening_outcome_columns_absent_from_features(cleaned_frames):
    df = cleaned_frames["cervical-cancer"]
    feature_cols = set(df.columns) - {"target"}
    # Hinselmann/Schiller/Citology are alternative outcomes of the same
    # screening episode as Biopsy (the chosen target) — leakage if kept as
    # predictors. Biopsy itself becomes `target`, not a feature column.
    for leaky in ("Hinselmann", "Schiller", "Citology", "Biopsy"):
        assert leaky not in feature_cols


def test_cervical_sparse_stds_time_columns_dropped(cleaned_frames):
    df = cleaned_frames["cervical-cancer"]
    assert "STDs: Time since first diagnosis" not in df.columns
    assert "STDs: Time since last diagnosis" not in df.columns


def test_cervical_target_sum(cleaned_frames):
    df = cleaned_frames["cervical-cancer"]
    assert df["target"].sum() == 55


# ---------------------------------------------------------------------------
# heart-disease
# ---------------------------------------------------------------------------


def test_heart_row_count(cleaned_frames):
    assert len(cleaned_frames["heart-disease"]) == 920


def test_heart_switzerland_chol_entirely_nan(cleaned_frames):
    df = cleaned_frames["heart-disease"]
    swiss = df[df["site"] == "switzerland"]
    assert len(swiss) == 123
    assert swiss["chol"].isna().all()


def test_heart_cleveland_and_hungary_chol_not_all_nan(cleaned_frames):
    # Proves the sentinel rule is site-scoped, not applied globally: these
    # two sites genuinely have no chol==0 readings, so their chol column
    # must retain real (non-all-NaN) values.
    df = cleaned_frames["heart-disease"]
    for site in ("cleveland", "hungarian"):
        subset = df[df["site"] == site]
        assert not subset["chol"].isna().all()


def test_heart_no_zero_chol_or_trestbps_anywhere(cleaned_frames):
    df = cleaned_frames["heart-disease"]
    assert (df["chol"] == 0).sum() == 0
    assert (df["trestbps"] == 0).sum() == 0


def test_heart_site_column_has_all_four_values(cleaned_frames):
    df = cleaned_frames["heart-disease"]
    assert set(df["site"].unique()) == {"cleveland", "hungarian", "switzerland", "va"}


# ---------------------------------------------------------------------------
# kidney-disease
# ---------------------------------------------------------------------------


def test_kidney_target_two_classes_and_exact_counts(cleaned_frames):
    df = cleaned_frames["kidney-disease"]
    assert df["target"].nunique() == 2
    counts = df["target"].value_counts().to_dict()
    # The 2 whitespace-dirty 'ckd\t' rows must land in the 250 positives,
    # not silently form a third class.
    assert counts[1] == 250
    assert counts[0] == 150


# ---------------------------------------------------------------------------
# liver-disease
# ---------------------------------------------------------------------------


def test_liver_target_sum(cleaned_frames):
    df = cleaned_frames["liver-disease"]
    assert df["target"].sum() == 416


def test_liver_sex_male_is_binary(cleaned_frames):
    df = cleaned_frames["liver-disease"]
    assert set(df["sex_male"].unique()) == {0, 1}


# ---------------------------------------------------------------------------
# diabetes
# ---------------------------------------------------------------------------


def test_diabetes_shape(cleaned_frames):
    assert cleaned_frames["diabetes"].shape == (253680, 22)


def test_diabetes_target_sum(cleaned_frames):
    assert cleaned_frames["diabetes"]["target"].sum() == 35346


# ---------------------------------------------------------------------------
# Contract tests over all six conditions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_row_count_matches_condition(cleaned_frames, slug):
    assert len(cleaned_frames[slug]) == CONDITIONS[slug].n_rows


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_target_dtype_is_integer_and_values_are_binary(cleaned_frames, slug):
    target = cleaned_frames[slug]["target"]
    assert pd.api.types.is_integer_dtype(target)
    assert set(target.unique()) == {0, 1}


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_no_all_nan_columns(cleaned_frames, slug):
    df = cleaned_frames[slug]
    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    assert all_nan_cols == []


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_site_column_present_only_for_heart_disease(cleaned_frames, slug):
    df = cleaned_frames[slug]
    if slug == "heart-disease":
        assert "site" in df.columns
    else:
        assert "site" not in df.columns


# ---------------------------------------------------------------------------
# CleaningError / fail-fast validation
# ---------------------------------------------------------------------------


def test_validate_raises_cleaning_error_for_short_frame():
    # Deliberately far too few rows for any real condition.
    bad_df = pd.DataFrame({"target": [0, 1]})
    with pytest.raises(CleaningError):
        _validate("breast-cancer", bad_df)


def test_validate_raises_cleaning_error_for_non_binary_target():
    n = CONDITIONS["breast-cancer"].n_rows
    bad_df = pd.DataFrame({"target": [2] * n})
    with pytest.raises(CleaningError):
        _validate("breast-cancer", bad_df)


def test_validate_raises_cleaning_error_for_all_nan_column():
    n = CONDITIONS["breast-cancer"].n_rows
    bad_df = pd.DataFrame(
        {
            "feature": [float("nan")] * n,
            "target": ([0, 1] * ((n // 2) + 1))[:n],
        }
    )
    with pytest.raises(CleaningError):
        _validate("breast-cancer", bad_df)


def test_clean_raises_cleaning_error_when_pipeline_produces_bad_shape(monkeypatch):
    def bad_pipeline(raw_dir):
        return pd.DataFrame({"target": [0, 1]})

    monkeypatch.setitem(PIPELINES, "breast-cancer", bad_pipeline)
    with pytest.raises(CleaningError):
        clean("breast-cancer")
