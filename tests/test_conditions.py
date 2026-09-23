# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Tests for the condition registry (nightingale.conditions)."""

import dataclasses

import pytest

from nightingale.conditions import CONDITIONS, Condition

EXPECTED_SLUGS = {
    "breast-cancer",
    "cervical-cancer",
    "heart-disease",
    "kidney-disease",
    "liver-disease",
    "diabetes",
}

EXPECTED_N_ROWS = {
    "breast-cancer": 569,
    "cervical-cancer": 858,
    "heart-disease": 920,
    "kidney-disease": 400,
    "liver-disease": 583,
    "diabetes": 253680,
}

# ucimlrepo returns broken/partial copies of these two, so both must use the zip fetch.
EXPECTED_ZIP_ONLY = {"cervical-cancer", "heart-disease"}


def test_registry_has_exactly_six_slugs():
    assert set(CONDITIONS.keys()) == EXPECTED_SLUGS
    assert len(CONDITIONS) == 6


@pytest.mark.parametrize("slug", sorted(EXPECTED_SLUGS))
def test_condition_is_frozen_dataclass_instance(slug):
    condition = CONDITIONS[slug]
    assert isinstance(condition, Condition)
    assert dataclasses.is_dataclass(condition)
    with pytest.raises(dataclasses.FrozenInstanceError):
        condition.slug = "mutated"


@pytest.mark.parametrize("slug", sorted(EXPECTED_SLUGS))
def test_condition_slug_matches_registry_key(slug):
    assert CONDITIONS[slug].slug == slug


@pytest.mark.parametrize("slug,expected_rows", sorted(EXPECTED_N_ROWS.items()))
def test_n_rows_matches_spec_table(slug, expected_rows):
    assert CONDITIONS[slug].n_rows == expected_rows


@pytest.mark.parametrize("slug", sorted(EXPECTED_ZIP_ONLY))
def test_fetch_mode_is_zip_for_broken_ucimlrepo_sources(slug):
    assert CONDITIONS[slug].fetch_mode == "zip"


@pytest.mark.parametrize("slug", sorted(EXPECTED_SLUGS - EXPECTED_ZIP_ONLY))
def test_fetch_mode_is_ucimlrepo_for_remaining_sources(slug):
    assert CONDITIONS[slug].fetch_mode == "ucimlrepo"


@pytest.mark.parametrize("slug", sorted(EXPECTED_SLUGS))
def test_every_field_is_non_empty(slug):
    condition = CONDITIONS[slug]
    assert condition.slug
    assert condition.display
    assert condition.source
    assert condition.fetch_mode
    assert condition.fetch_ref
    assert isinstance(condition.raw_files, tuple)
    assert len(condition.raw_files) > 0
    assert all(condition.raw_files)
    assert condition.target
    assert condition.positive_meaning
    assert isinstance(condition.n_rows, int)
    assert condition.n_rows > 0
    assert condition.citation
    assert condition.license_note


@pytest.mark.parametrize("slug", sorted(EXPECTED_SLUGS))
def test_fetch_mode_is_valid_choice(slug):
    assert CONDITIONS[slug].fetch_mode in ("ucimlrepo", "zip")


def test_target_column_is_canonical_target_after_cleaning():
    for condition in CONDITIONS.values():
        assert condition.target == "target"
