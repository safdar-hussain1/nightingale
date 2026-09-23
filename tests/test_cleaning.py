# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Tests for the six per-condition data cleaners (nightingale.clean).

Most tests here read the already-committed `data/cleaned/<slug>.csv.gz`
artifacts directly (see the `cleaned_frames` fixture) rather than calling
`clean()` -- those files are checked into the repo, so this keeps the bulk
of the suite raw-data-free and green on a fresh clone with no `data/raw/`.

A couple of tests genuinely need to exercise `clean()` (and therefore
`fetch()`) end-to-end against the real, already-fetched raw data under
`data/raw/` (gitignored, populated by running `fetch`, not by cloning the
repo). No test here is network-marked -- if the raw cache is populated,
`fetch()` only verifies checksums locally -- but on a fresh clone the cache
is absent, so those tests call `conftest.skip_if_raw_missing` first and
skip with an actionable reason instead of failing or reaching for the
network.
"""

import struct

import pandas as pd
import pytest

import nightingale.clean as clean_module
from conftest import skip_if_raw_missing
from nightingale.clean import (
    CLEANED_ROOT,
    PIPELINES,
    CleaningError,
    _apply_heart_sentinels,
    _validate,
    clean,
)
from nightingale.conditions import CONDITIONS

ALL_SLUGS = sorted(CONDITIONS.keys())


@pytest.fixture(scope="session")
def cleaned_frames():
    # Reads the committed artifact directly (not clean()) so every test
    # using this fixture is raw-data-free and runs on a fresh clone.
    frames = {}
    for slug in ALL_SLUGS:
        path = CLEANED_ROOT / f"{slug}.csv.gz"
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


def test_apply_heart_sentinels_chol_rule_is_site_scoped():
    # Genuinely discriminates the site-scoping: a synthetic frame with a
    # cleveland row and a switzerland row that BOTH have chol == 0. If the
    # sentinel rule were applied globally instead of scoped to
    # switzerland/va, the cleveland zero would also be wiped to NaN. Real
    # cleveland/hungary raw data happens to have zero chol==0 rows at all,
    # which is why testing against the real cleaned frames (as the old
    # version of this test did) can't tell scoped from global — this
    # synthetic frame can.
    df = pd.DataFrame(
        {
            "chol": [0, 0],
            "trestbps": [120, 130],
            "site": ["cleveland", "switzerland"],
        }
    )

    result = _apply_heart_sentinels(df)

    cleveland_chol = result.loc[result["site"] == "cleveland", "chol"].iloc[0]
    switzerland_chol = result.loc[result["site"] == "switzerland", "chol"].iloc[0]
    assert cleveland_chol == 0  # site-scoped rule: cleveland zero survives
    assert pd.isna(switzerland_chol)  # switzerland zero becomes NaN


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
    # clean() calls fetch() before it ever reaches the (monkeypatched)
    # pipeline, so this genuinely needs breast-cancer's raw data cached.
    skip_if_raw_missing("breast-cancer")

    def bad_pipeline(raw_dir):
        return pd.DataFrame({"target": [0, 1]})

    monkeypatch.setitem(PIPELINES, "breast-cancer", bad_pipeline)
    with pytest.raises(CleaningError):
        clean("breast-cancer")


# ---------------------------------------------------------------------------
# Deterministic output (a requirement of the signed provenance manifest)
# ---------------------------------------------------------------------------


def test_cleaned_output_is_byte_reproducible(tmp_path, monkeypatch):
    # This is the one test that genuinely exercises clean() (and therefore
    # fetch()) end-to-end -- it needs breast-cancer's raw data cached.
    skip_if_raw_missing("breast-cancer")

    # gzip embeds a Unix mtime in its header by default, which makes two
    # writes of the identical DataFrame produce two different files. The
    # signed Ed25519 provenance manifest needs `clean()`'s output to be
    # byte-for-byte reproducible so an honest, untouched, correctly
    # regenerated repo verifies as OK rather than TAMPERED. Redirect
    # CLEANED_ROOT to a scratch dir (rather than asserting against the real
    # data/cleaned/ artifact) so this test doesn't depend on run order or
    # clobber anything, and reads the file back between the two clean()
    # calls since both write to the same fixed path.
    monkeypatch.setattr(clean_module, "CLEANED_ROOT", tmp_path)

    path = clean("breast-cancer")
    first_write = path.read_bytes()

    # Belt-and-braces: check the gzip header's mtime field (bytes 4-7,
    # little-endian uint32) is pinned to 0 directly, not just that two
    # back-to-back writes happen to match. A real Unix timestamp is never 0
    # for any date this project runs on, so this catches a regression to
    # the default (wall-clock) mtime deterministically -- unlike a bare
    # write-write comparison, which could accidentally pass even without
    # the fix if both calls land within the same wall-clock second (gzip's
    # mtime has 1-second resolution).
    mtime_field = struct.unpack_from("<I", first_write, 4)[0]
    assert mtime_field == 0

    path_again = clean("breast-cancer")
    second_write = path_again.read_bytes()

    assert path == path_again
    assert first_write == second_write
