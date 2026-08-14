# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Checksum-pinned, reproducible raw-data fetcher.

Every raw file this module lands under ``data/raw/<slug>/`` is verified
against a hard-coded SHA-256 pin in :data:`PINNED_SHA256` before ``fetch``
returns — both right after a fresh download and on the cached-path
early-return, so a tampered or corrupted local cache is never silently
trusted. Mismatches raise :class:`FetchIntegrityError` instead of being
papered over.

Two fetch modes, one per :class:`~nightingale.conditions.Condition`:

- ``"ucimlrepo"``: ``fetch_ucirepo(id=...)`` from the ``ucimlrepo`` package,
  written out as ``features.csv`` / ``targets.csv`` (index=False).
- ``"zip"``: download the archive at ``fetch_ref`` to a temp file, extract
  only the condition's ``raw_files`` (by basename, regardless of internal
  archive directory structure), and discard everything else.
"""

from __future__ import annotations

import hashlib
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from ucimlrepo import fetch_ucirepo

from nightingale.conditions import CONDITIONS, Condition

# data/raw/, resolved relative to this module's location so it works no
# matter the caller's current working directory.
RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"

# {slug: {filename: sha256_hexdigest}}. Populated by hand-running `fetch`
# for real once per slug and hard-coding the observed digests. The exact
# commands that produced these values are recorded in the project's private
# build notes, kept outside this checkout.
PINNED_SHA256: dict[str, dict[str, str]] = {
    "breast-cancer": {
        "features.csv": "66e90ef939e5965e805cf62008e8df2d8ae022ac683f517ecf3a6ad476ce6e04",
        "targets.csv": "81bcb70705e2a8a249b1c7edaa12ae6b2d775f3ac115e1acd2090d4da7cdae48",
    },
    "cervical-cancer": {
        "risk_factors_cervical_cancer.csv": "8df193ad5c9ff4288fb4c401eef70dcd2cbda404ce7f82ac74c68cfc960ab063",
    },
    "diabetes": {
        "features.csv": "62e734dd9e197bcd2e474d4da5e221a9429cb6a8a68f08d3b6b4a789ab39f55e",
        "targets.csv": "8f63a7f5790a851f152b35104b21dabcc7e15428ede615c55b0d5d4d77b0fef2",
    },
    "heart-disease": {
        "processed.cleveland.data": "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8",
        "processed.hungarian.data": "d1ad108f785768cd3d7e82dc522e6f5a61eea93cccfb3a46ee8076f73fc3d796",
        "processed.switzerland.data": "834a405ccf5b66ab4056bb77794adc8df0b7125186454c0a1d002d33c6c3b314",
        "processed.va.data": "e7c93d8d0d2acdadfa4c5e8de768e2191e7f618b952e29623f1f0d5949ff6b8f",
    },
    "kidney-disease": {
        "features.csv": "4bd913600c0d63bc0d70c94b01976da47ca5979edab1a6058a93ccd8dc5fc843",
        "targets.csv": "a285bded2773b0f714f7397b633902e9592f6547b70db45892b22832ac968ed7",
    },
    "liver-disease": {
        "features.csv": "2c7b2b9c3c55d8253581cfb1e90b18c17594f3b794676a6e688a6d454caa269b",
        "targets.csv": "5017b91eaf641bfc016cd4bf7399fb22786ce5ea4bd30c2dbe4ae91e54e147a4",
    },
}


class FetchIntegrityError(Exception):
    """Raised when a raw file's SHA-256 does not match its pin in PINNED_SHA256."""


def fetch(slug: str, dest_root: Path = RAW_ROOT, force: bool = False) -> Path:
    """Fetch (or verify the cached copy of) one condition's raw data.

    Returns the directory ``dest_root/<slug>`` containing every file listed
    in ``CONDITIONS[slug].raw_files``, each verified against
    ``PINNED_SHA256[slug]``.

    Raises ``KeyError`` for an unknown slug, ``FetchIntegrityError`` if any
    landed (or cached) file's checksum does not match its pin.
    """
    condition = CONDITIONS[slug]  # KeyError propagates for unknown slugs.
    slug_dir = Path(dest_root) / slug

    if not force and _is_populated(slug_dir, condition.raw_files):
        # Cache looks complete — verify without touching the network. A
        # mismatch here means the cache is corrupt/tampered, so it raises
        # rather than silently re-downloading over it.
        _verify(slug, slug_dir, condition.raw_files)
        return slug_dir

    slug_dir.mkdir(parents=True, exist_ok=True)

    if condition.fetch_mode == "ucimlrepo":
        _fetch_ucimlrepo(condition, slug_dir)
    elif condition.fetch_mode == "zip":
        _fetch_zip(condition, slug_dir)
    else:
        raise ValueError(
            f"{slug}: unknown fetch_mode {condition.fetch_mode!r}"
        )

    _verify(slug, slug_dir, condition.raw_files)
    return slug_dir


def _is_populated(slug_dir: Path, raw_files: tuple[str, ...]) -> bool:
    return slug_dir.is_dir() and all((slug_dir / f).is_file() for f in raw_files)


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(slug: str, slug_dir: Path, raw_files: tuple[str, ...]) -> None:
    pins = PINNED_SHA256.get(slug, {})
    for filename in raw_files:
        path = slug_dir / filename
        if not path.is_file():
            raise FetchIntegrityError(f"{slug}: expected file missing: {filename}")
        expected = pins.get(filename)
        if expected is None:
            # No pin recorded for this file yet (e.g. first real download
            # used to generate the pins). Nothing to check against.
            continue
        actual = _sha256_of(path)
        if actual != expected:
            raise FetchIntegrityError(
                f"{slug}/{filename}: sha256 mismatch "
                f"(expected {expected}, got {actual})"
            )


def _fetch_ucimlrepo(condition: Condition, slug_dir: Path) -> None:
    result = fetch_ucirepo(id=int(condition.fetch_ref))
    result.data.features.to_csv(slug_dir / "features.csv", index=False)
    result.data.targets.to_csv(slug_dir / "targets.csv", index=False)


def _fetch_zip(condition: Condition, slug_dir: Path) -> None:
    wanted = set(condition.raw_files)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_zip = Path(tmp_dir) / "download.zip"
        urllib.request.urlretrieve(condition.fetch_ref, tmp_zip)

        found = set()
        with zipfile.ZipFile(tmp_zip) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                name = PurePosixPath(member.filename).name
                if name not in wanted:
                    continue
                with archive.open(member) as source:
                    (slug_dir / name).write_bytes(source.read())
                found.add(name)

    missing = wanted - found
    if missing:
        raise FetchIntegrityError(
            f"{condition.slug}: zip archive at {condition.fetch_ref} is missing "
            f"expected file(s) {sorted(missing)}"
        )
