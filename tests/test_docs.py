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


# The README's transfer-study table names each unseen hospital in prose; these
# are the keys the same site carries in models/heart-disease/external.json.
SITE_LABEL_TO_KEY = {
    "Hungary": "hungarian",
    "Switzerland": "switzerland",
    "VA Long Beach": "va",
}

# `point [lo, hi]` as the README prints every interval. Bold markers around a
# cell are stripped before matching, so **0.0466 [0.0282, 0.1028]** parses too.
METRIC_RE = re.compile(r"(\d+\.\d+) \[(\d+\.\d+), (\d+\.\d+)\]")


def _table_rows(n_cells: int) -> list[list[str]]:
    """Every README table row with exactly ``n_cells`` cells, bold stripped."""
    rows = []
    for line in README.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]
        if len(cells) == n_cells:
            rows.append(cells)
    return rows


def _assert_printed_matches(printed: str, actual: float, what: str) -> None:
    """The printed string must equal ``actual`` rounded to its own precision."""
    places = len(printed.split(".")[1])
    rounded = f"{actual:.{places}f}"
    assert Decimal(printed) == Decimal(rounded), (
        f"README prints {what} = {printed}, but the artifact holds {actual!r} "
        f"(which rounds to {rounded} at {places} places)"
    )


def _readme_condition_table() -> dict[str, tuple[str, str, str]]:
    """Parse the six-condition table -> {slug: (point, lo, hi) as printed}."""
    printed: dict[str, tuple[str, str, str]] = {}
    for cells in _table_rows(5):
        slug = UCI_ID_TO_SLUG.get(cells[1])
        if slug is None:
            continue
        match = METRIC_RE.fullmatch(cells[4])
        assert match, f"malformed ROC-AUC cell for {slug}: {cells[4]!r}"
        printed[slug] = match.groups()
    return printed


def _readme_transfer_table() -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Parse the recalibration table -> {site key: (naive ECE, recal ECE)}."""
    printed: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for cells in _table_rows(5):
        key = SITE_LABEL_TO_KEY.get(cells[0])
        if key is None or "→" not in cells[3]:
            continue
        found = METRIC_RE.findall(cells[3])
        assert len(found) == 2, (
            f"the ECE naive -> recalibrated cell for {key} must print two "
            f"intervals, got {cells[3]!r}"
        )
        printed[key] = (found[0], found[1])
    return printed


def test_readme_condition_table_covers_all_six_conditions():
    assert set(_readme_condition_table()) == set(UCI_ID_TO_SLUG.values())


@pytest.mark.parametrize("slug", sorted(UCI_ID_TO_SLUG.values()))
def test_readme_roc_auc_matches_metrics_json_to_printed_precision(slug: str):
    point, lo, hi = _readme_condition_table()[slug]
    metrics = json.loads(
        (REPO_ROOT / "models" / slug / "metrics.json").read_text(encoding="utf-8")
    )
    actual_point, actual_lo, actual_hi = metrics["roc_auc"]
    # Both bounds, not only the point -- an interval quoted from the wrong run
    # is exactly as wrong as a point estimate quoted from one.
    _assert_printed_matches(point, actual_point, f"{slug} ROC-AUC point")
    _assert_printed_matches(lo, actual_lo, f"{slug} ROC-AUC CI lower bound")
    _assert_printed_matches(hi, actual_hi, f"{slug} ROC-AUC CI upper bound")


def test_readme_transfer_table_covers_all_three_unseen_sites():
    assert set(_readme_transfer_table()) == set(SITE_LABEL_TO_KEY.values())


@pytest.mark.parametrize("site", sorted(SITE_LABEL_TO_KEY.values()))
@pytest.mark.parametrize("block", ("naive", "recalibrated"))
def test_readme_transfer_ece_matches_external_json(site: str, block: str):
    external = json.loads(
        (REPO_ROOT / "models" / "heart-disease" / "external.json").read_text(
            encoding="utf-8"
        )
    )
    actual = external["sites"][site][block]["ece"]
    naive_printed, recal_printed = _readme_transfer_table()[site]
    printed = naive_printed if block == "naive" else recal_printed
    for label, printed_value, actual_value in zip(
        ("point", "CI lower bound", "CI upper bound"), printed, actual
    ):
        _assert_printed_matches(
            printed_value, actual_value, f"{site} {block} ECE {label}"
        )


def test_readme_embeds_the_dashboard_screenshot_and_it_exists():
    text = README.read_text(encoding="utf-8")
    matches = re.findall(r"!\[dashboard\]\(([^)]+)\)", text)
    assert matches, "README does not embed ![dashboard](...)"
    for relpath in matches:
        target = (REPO_ROOT / relpath).resolve()
        assert target.is_file(), f"README references {relpath}, which does not exist"
        assert target.stat().st_size > 0
