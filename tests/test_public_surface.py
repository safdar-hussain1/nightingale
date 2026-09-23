# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Repo-wide guards over everything git tracks.

Three earlier gates checked the framing word list, but each was scoped to a
handful of named files: the three public documents, the generated dashboard,
the two notebooks. Everything else -- source comments, test data, generated
metadata, anything added later -- was unguarded, and that gap is exactly
where a stray word could reach a tracked file and survive. So the unit of the
guard here is not "the files we remembered to list"; it is ``git ls-files``.
A file is in scope the moment it is tracked.

Two properties:

1. No tracked text file contains any framing word from
   :data:`conftest.BANNED_WORDS`, case-insensitively, as a substring.
2. No tracked text file contains an absolute home-directory path. Those are
   machine-specific, meaningless to a reader, and leak the author's directory
   layout; anything that needs a path should print it relative to the repo.

Binary files are skipped by extension and, as a backstop, by a decode check --
a ``.csv.gz`` cannot meaningfully "contain a word", and forcing it through a
text search would only produce noise. Everything else, including every JSON
artifact under ``models/``, is read as UTF-8 text and searched.

Both needles are assembled from fragments rather than written out, because
this file is itself tracked and therefore in its own scope. That is the
deliberate design: the guard must be inescapable, so the test that defines it
is subject to it too. If a future edit makes this file trip the guard, the fix
is to spell the offending word in fragments -- never to exclude a file.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from conftest import BANNED_WORDS

REPO_ROOT = Path(__file__).resolve().parents[1]

# Extensions whose contents are not text. Kept explicit rather than sniffed, so
# that adding a binary format to the repo is a deliberate, reviewable edit here.
BINARY_SUFFIXES = frozenset({".png", ".gz", ".pem", ".pdf", ".jpg", ".jpeg", ".ico", ".woff2"})

BANNED_RE = re.compile("|".join(re.escape(word) for word in BANNED_WORDS), re.IGNORECASE)

# The home-directory prefix of this platform, in fragments (see module docstring).
HOME_PREFIX = "/" + "Users" + "/"


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [name for name in result.stdout.split("\0") if name]


def _tracked_text_files() -> list[Path]:
    """Every tracked file that can be read as UTF-8 text."""
    paths = []
    for name in _tracked_files():
        path = REPO_ROOT / name
        if path.suffix.lower() in BINARY_SUFFIXES or not path.is_file():
            continue
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        paths.append(path)
    return paths


TEXT_FILES = _tracked_text_files()


def test_the_guard_actually_has_files_to_guard():
    """A broken ``git ls-files`` would make both guards below vacuously green."""
    assert len(TEXT_FILES) > 50, f"only {len(TEXT_FILES)} tracked text files found"
    names = {path.relative_to(REPO_ROOT).as_posix() for path in TEXT_FILES}
    for expected in ("README.md", "docs/index.html", "models/run_meta.json", __file__.split("/")[-1]):
        assert any(name.endswith(expected) for name in names), f"{expected} not in scope"


def test_no_tracked_file_contains_a_framing_word():
    hits = []
    for path in TEXT_FILES:
        text = path.read_text(encoding="utf-8")
        for match in BANNED_RE.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            hits.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{line}: {match.group(0)!r}")
    assert hits == [], "framing words in tracked files:\n" + "\n".join(hits)


def test_no_tracked_file_contains_an_absolute_home_path():
    hits = []
    for path in TEXT_FILES:
        text = path.read_text(encoding="utf-8")
        start = 0
        while (index := text.find(HOME_PREFIX, start)) != -1:
            line = text.count("\n", 0, index) + 1
            hits.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{line}")
            start = index + 1
    assert hits == [], "absolute home-directory paths in tracked files:\n" + "\n".join(hits)


@pytest.mark.parametrize("word", BANNED_WORDS)
def test_the_word_guard_catches_a_planted_violation(word: str):
    """Positive control: each word is really detectable, in any casing, mid-sentence."""
    assert BANNED_RE.search(f"a sentence that {word.upper()} sits inside of")


def test_the_path_guard_catches_a_planted_violation():
    """Positive control, same idea, for the absolute-path needle."""
    planted = HOME_PREFIX + "someone/Desktop/x.pem"
    assert HOME_PREFIX in planted
