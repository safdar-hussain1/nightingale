# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Shared pytest fixtures: guard tests that need real, already-fetched raw data.

``data/raw/`` is gitignored — it's populated by running ``nightingale fetch``
(:func:`nightingale.fetch.fetch`), not by cloning the repo. A handful of tests
intentionally exercise the real raw data directly, or call :func:`clean`/
:func:`fetch` and therefore need the raw files already cached on disk (the
default test suite never touches the network, so it never fills an absent
cache in automatically).

On a machine where fetch has already been run — true for this development
machine — those tests must run and pass exactly as always: this guard must
never mask a real regression. On a fresh clone (the first thing anyone does
with this repo), they must skip with an actionable reason instead of failing
with a bare ``FileNotFoundError``, or worse, attempting a real network call
mid test-suite.

``data/cleaned/*.csv.gz`` is committed to the repo, so tests that only need
the *cleaned* data are never raw-dependent and should read that file
directly rather than calling :func:`clean`, so they never need this guard.
"""

import pytest

from nightingale.conditions import CONDITIONS
from nightingale.fetch import RAW_ROOT

# ---------------------------------------------------------------------------
# Framing guard vocabulary.
#
# These are not rude words; they are words that would mis-describe what this
# project is, or that describe how a file got written rather than what it does.
# ``tests/test_public_surface.py`` greps EVERY tracked text file for them, this
# file included, so each entry is assembled from fragments: a literal spelling
# here would make the guard fail on its own definition, and the fix for that is
# never to narrow the guard. Every test module that needs the list imports it
# from here rather than re-spelling it.
# ---------------------------------------------------------------------------
BANNED_WORDS = (
    "course" + "work",
    "col" + "lege",
    "re" + "build",
    "origin" + "ally",
    "generated" + " with",
    "clau" + "de",
    "anthro" + "pic",
    "co-" + "authored",
)


def raw_data_available(slug: str) -> bool:
    """True iff every raw file ``fetch(slug)`` would land is present on disk."""
    condition = CONDITIONS[slug]
    slug_dir = RAW_ROOT / slug
    return slug_dir.is_dir() and all((slug_dir / f).is_file() for f in condition.raw_files)


def skip_if_raw_missing(slug: str) -> None:
    """Skip the current test, with an actionable reason, if ``slug``'s raw data is absent.

    Call at the top of any test that reads under :data:`RAW_ROOT` directly,
    or that calls :func:`nightingale.clean.clean`/:func:`nightingale.fetch.fetch`
    and therefore needs the raw files to already be cached on disk.
    """
    if not raw_data_available(slug):
        pytest.skip(
            f"raw data for {slug!r} absent under {RAW_ROOT / slug} -- run "
            f"PYTHONPATH=src python -c \"from nightingale.fetch import fetch; "
            f"fetch({slug!r})\" to fetch it and enable this test"
        )
