# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Tests for the three public documents: README, model card, data dictionary.

These are the surfaces a reader meets first, so they get the same treatment as
the artifacts: the honesty banner has to be present verbatim (not paraphrased,
not softened), the framing has to stay clear of a fixed banned-word list, the
headline ROC-AUC figures printed in the README's six-condition table have to
match ``models/<slug>/metrics.json`` to the precision they are printed at, and
the dashboard screenshot the README embeds has to exist.

The ROC-AUC check parses the README table rather than hard-coding the values a
second time -- a copy of the numbers in the test would drift alongside a copy in
the doc, which is exactly the failure this is meant to catch.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

README = REPO_ROOT / "README.md"
MODEL_CARD = REPO_ROOT / "MODEL_CARD.md"
DATA_DICTIONARY = REPO_ROOT / "data" / "DATA_DICTIONARY.md"

DOCS = (README, MODEL_CARD, DATA_DICTIONARY)

# Framing guard. This is a standalone product; none of these words belong on any
# public surface of it. Matched case-insensitively, as substrings, mirroring the
# `grep -riE` gate the task's own checklist runs.
BANNED_WORDS = (
    "coursework",
    "college",
    "rebuild",
    "originally",
    "generated with",
    "claude",
    "anthropic",
)

HONESTY_BANNER = (
    "These are screening-triage risk models trained on small public research "
    "datasets. They are not diagnostic devices and must not be used for medical "
    "decisions. The diabetes labels are self-reported survey responses. The "
    "cervical-cancer cohort has 55 positive biopsies."
)

# The README's six-condition table names each dataset by its UCI id, which is
# the one column that is unambiguous per condition.
UCI_ID_TO_SLUG = {
    "UCI 17": "breast-cancer",
    "UCI 383": "cervical-cancer",
    "UCI 45": "heart-disease",
    "UCI 336": "kidney-disease",
    "UCI 225": "liver-disease",
    "UCI 891": "diabetes",
}


def _doc_id(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


@pytest.mark.parametrize("path", DOCS, ids=_doc_id)
def test_document_exists_and_is_not_empty(path: Path):
    assert path.is_file(), f"{path} is missing"
    assert path.read_text(encoding="utf-8").strip()


@pytest.mark.parametrize("path", DOCS, ids=_doc_id)
@pytest.mark.parametrize("word", BANNED_WORDS)
def test_banned_words_have_zero_hits(path: Path, word: str):
    text = path.read_text(encoding="utf-8").lower()
    assert word not in text, f"banned word {word!r} appears in {_doc_id(path)}"


@pytest.mark.parametrize("path", (README, MODEL_CARD), ids=_doc_id)
def test_honesty_banner_is_present_verbatim(path: Path):
    text = path.read_text(encoding="utf-8")
    assert HONESTY_BANNER in text, (
        f"{_doc_id(path)} does not carry the honesty banner verbatim -- it must "
        "appear as one unbroken line, not reflowed or reworded"
    )


def _readme_condition_table() -> dict[str, str]:
    """Parse the README's six-condition table -> {slug: printed ROC-AUC point}."""
    printed: dict[str, str] = {}
    for line in README.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 5:
            continue
        slug = UCI_ID_TO_SLUG.get(cells[1])
        if slug is None:
            continue
        match = re.fullmatch(r"(\d\.\d+) \[\d\.\d+, \d\.\d+\]", cells[4])
        assert match, f"malformed ROC-AUC cell for {slug}: {cells[4]!r}"
        printed[slug] = match.group(1)
    return printed


def test_readme_condition_table_covers_all_six_conditions():
    assert set(_readme_condition_table()) == set(UCI_ID_TO_SLUG.values())


@pytest.mark.parametrize("slug", sorted(UCI_ID_TO_SLUG.values()))
def test_readme_roc_auc_matches_metrics_json_to_printed_precision(slug: str):
    printed = _readme_condition_table()[slug]
    metrics = json.loads(
        (REPO_ROOT / "models" / slug / "metrics.json").read_text(encoding="utf-8")
    )
    actual = metrics["roc_auc"][0]
    places = len(printed.split(".")[1])
    rounded = f"{actual:.{places}f}"
    assert Decimal(printed) == Decimal(rounded), (
        f"README prints ROC-AUC {printed} for {slug}, but metrics.json holds "
        f"{actual!r} (which rounds to {rounded})"
    )


def test_readme_embeds_the_dashboard_screenshot_and_it_exists():
    text = README.read_text(encoding="utf-8")
    matches = re.findall(r"!\[dashboard\]\(([^)]+)\)", text)
    assert matches, "README does not embed ![dashboard](...)"
    for relpath in matches:
        target = (REPO_ROOT / relpath).resolve()
        assert target.is_file(), f"README references {relpath}, which does not exist"
        assert target.stat().st_size > 0
