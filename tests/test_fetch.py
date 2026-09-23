# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Tests for the checksum-pinned dataset fetcher (nightingale.fetch).

Default (non-network) suite: uses tmp_path fixtures and monkeypatching so no
test in the default run touches the network. Real end-to-end downloads are
marked ``@pytest.mark.network`` and excluded by default (see pyproject.toml
``addopts``); run them explicitly with ``pytest -m network``.
"""

import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import nightingale.fetch as fetch
from conftest import skip_if_raw_missing
from nightingale.conditions import CONDITIONS


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# (a) unknown slug raises KeyError
# ---------------------------------------------------------------------------


def test_unknown_slug_raises_key_error(tmp_path):
    with pytest.raises(KeyError):
        fetch.fetch("not-a-real-condition", dest_root=tmp_path)


# ---------------------------------------------------------------------------
# (b) checksum mismatch on a pre-populated dir raises FetchIntegrityError,
#     without touching the network.
# ---------------------------------------------------------------------------


def test_checksum_mismatch_raises_fetch_integrity_error(tmp_path, monkeypatch):
    slug = "breast-cancer"
    condition = CONDITIONS[slug]
    slug_dir = tmp_path / slug
    slug_dir.mkdir(parents=True)

    # Populate with content that does NOT match the pinned digests below.
    for filename in condition.raw_files:
        (slug_dir / filename).write_bytes(b"not the real content")

    wrong_pins = {
        filename: _sha256(b"this is definitely different")
        for filename in condition.raw_files
    }
    monkeypatch.setitem(fetch.PINNED_SHA256, slug, wrong_pins)

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("network must not be touched on integrity mismatch")

    monkeypatch.setattr(fetch, "fetch_ucirepo", _raise_if_called)
    monkeypatch.setattr(fetch.urllib.request, "urlretrieve", _raise_if_called)

    with pytest.raises(fetch.FetchIntegrityError):
        fetch.fetch(slug, dest_root=tmp_path)


# ---------------------------------------------------------------------------
# (c) fetch() on an already-populated, valid dir does not re-download.
# ---------------------------------------------------------------------------


def test_populated_valid_dir_skips_redownload(tmp_path, monkeypatch):
    slug = "breast-cancer"
    condition = CONDITIONS[slug]
    slug_dir = tmp_path / slug
    slug_dir.mkdir(parents=True)

    contents = {
        filename: f"contents of {filename}".encode()
        for filename in condition.raw_files
    }
    pins = {}
    mtimes_before = {}
    for filename, data in contents.items():
        path = slug_dir / filename
        path.write_bytes(data)
        pins[filename] = _sha256(data)
        mtimes_before[filename] = path.stat().st_mtime_ns

    monkeypatch.setitem(fetch.PINNED_SHA256, slug, pins)

    def _raise_if_called(*args, **kwargs):
        raise AssertionError("network must not be touched when cache is valid")

    monkeypatch.setattr(fetch, "fetch_ucirepo", _raise_if_called)
    monkeypatch.setattr(fetch.urllib.request, "urlretrieve", _raise_if_called)

    result = fetch.fetch(slug, dest_root=tmp_path)

    assert result == slug_dir
    for filename in condition.raw_files:
        path = slug_dir / filename
        assert path.read_bytes() == contents[filename]
        assert path.stat().st_mtime_ns == mtimes_before[filename]


# ---------------------------------------------------------------------------
# ucimlrepo mode: mocked fetch_ucirepo, no network.
# ---------------------------------------------------------------------------


def test_ucimlrepo_mode_writes_features_and_targets_csv(tmp_path, monkeypatch):
    slug = "kidney-disease"
    condition = CONDITIONS[slug]
    assert condition.fetch_mode == "ucimlrepo"

    class _FakeData:
        features = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        targets = pd.DataFrame({"class": ["ckd", "notckd"]})

    class _FakeResult:
        data = _FakeData()

    calls = []

    def _fake_fetch_ucirepo(id=None):
        calls.append(id)
        return _FakeResult()

    monkeypatch.setattr(fetch, "fetch_ucirepo", _fake_fetch_ucirepo)
    monkeypatch.setitem(fetch.PINNED_SHA256, slug, {})

    result = fetch.fetch(slug, dest_root=tmp_path)

    assert calls == [int(condition.fetch_ref)]
    assert result == tmp_path / slug
    features = pd.read_csv(result / "features.csv")
    targets = pd.read_csv(result / "targets.csv")
    assert list(features.columns) == ["a", "b"]
    assert list(targets.columns) == ["class"]


# ---------------------------------------------------------------------------
# zip mode: locally built zip fixture, no network.
# ---------------------------------------------------------------------------


def test_zip_mode_extracts_only_raw_files(tmp_path, monkeypatch):
    slug = "cervical-cancer"
    condition = CONDITIONS[slug]
    assert condition.fetch_mode == "zip"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("risk_factors_cervical_cancer.csv", "age,target\n30,0\n")
        zf.writestr("NOTICE.txt", "irrelevant archive member")
        zf.writestr("subdir/extra.csv", "should also be ignored")
    zip_bytes = buf.getvalue()

    def _fake_urlretrieve(url, filename):
        Path(filename).write_bytes(zip_bytes)
        return filename, None

    monkeypatch.setattr(fetch.urllib.request, "urlretrieve", _fake_urlretrieve)
    monkeypatch.setitem(fetch.PINNED_SHA256, slug, {})

    result = fetch.fetch(slug, dest_root=tmp_path)

    assert result == tmp_path / slug
    landed = sorted(p.name for p in result.iterdir())
    assert landed == list(condition.raw_files)
    assert (result / "risk_factors_cervical_cancer.csv").read_text() == "age,target\n30,0\n"


def test_zip_mode_missing_expected_file_raises_integrity_error(tmp_path, monkeypatch):
    slug = "cervical-cancer"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("unrelated.csv", "nothing useful\n")
    zip_bytes = buf.getvalue()

    def _fake_urlretrieve(url, filename):
        Path(filename).write_bytes(zip_bytes)
        return filename, None

    monkeypatch.setattr(fetch.urllib.request, "urlretrieve", _fake_urlretrieve)
    monkeypatch.setitem(fetch.PINNED_SHA256, slug, {})

    with pytest.raises(fetch.FetchIntegrityError):
        fetch.fetch(slug, dest_root=tmp_path)


# ---------------------------------------------------------------------------
# RAW_ROOT sanity
# ---------------------------------------------------------------------------


def test_raw_root_resolves_under_repo_root():
    assert fetch.RAW_ROOT.name == "raw"
    assert fetch.RAW_ROOT.parent.name == "data"


# ---------------------------------------------------------------------------
# The real PINNED_SHA256 dict against the real, already-cached data/raw/ --
# no monkeypatching. Every other test in this file substitutes its own pins
# before calling fetch(), so none of them ever check that the committed
# PINNED_SHA256 entries actually match the committed(-by-fetch) cache on
# disk. A tampered or wrong pin for an already-cached slug would be
# invisible without this test.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slug", sorted(CONDITIONS.keys()))
def test_real_pinned_sha256_matches_cached_raw_files(slug):
    skip_if_raw_missing(slug)
    condition = CONDITIONS[slug]
    slug_dir = fetch.RAW_ROOT / slug
    pins = fetch.PINNED_SHA256[slug]
    for filename in condition.raw_files:
        actual = _sha256((slug_dir / filename).read_bytes())
        assert actual == pins[filename], (
            f"{slug}/{filename}: cached file's sha256 does not match the "
            f"real PINNED_SHA256 entry"
        )


# ---------------------------------------------------------------------------
# Real network end-to-end fetch — excluded from the default suite.
# ---------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.parametrize("slug", sorted(CONDITIONS.keys()))
def test_real_fetch_all_slugs(slug, tmp_path):
    result = fetch.fetch(slug, dest_root=tmp_path, force=True)
    condition = CONDITIONS[slug]
    for filename in condition.raw_files:
        assert (result / filename).exists()


@pytest.mark.network
def test_real_fetch_heart_disease_site_row_counts(tmp_path):
    result = fetch.fetch("heart-disease", dest_root=tmp_path, force=True)
    expected_lines = {
        "processed.cleveland.data": 303,
        "processed.hungarian.data": 294,
        "processed.switzerland.data": 123,
        "processed.va.data": 200,
    }
    for filename, n_lines in expected_lines.items():
        text = (result / filename).read_text()
        lines = [line for line in text.splitlines() if line.strip()]
        assert len(lines) == n_lines


@pytest.mark.network
def test_real_fetch_cervical_cancer_has_expected_file(tmp_path):
    result = fetch.fetch("cervical-cancer", dest_root=tmp_path, force=True)
    assert (result / "risk_factors_cervical_cancer.csv").exists()
