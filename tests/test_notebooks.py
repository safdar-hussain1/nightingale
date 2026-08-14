# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Tests for the two committed, executed analysis notebooks.

Both `notebooks/01_six_conditions.ipynb` and `notebooks/02_external_validation.ipynb`
are executed top-to-bottom and committed with their outputs -- these tests inspect the
committed JSON directly (never re-executing the notebook, which would be slow and would
retrain models) to pin three properties an executed, publishable notebook must have:

1. Every code cell actually ran, in order, exactly once, from a single top-to-bottom
   execution -- checked via `execution_count` being the contiguous sequence
   ``1..len(code_cells)``. A notebook edited and re-run cell-by-cell out of order, or
   with a stale unexecuted cell left in the middle, fails this even though it might
   "look" executed at a glance.
2. No cell's output contains an error (``output_type == "error"``) -- an executed
   notebook that raised partway through and was saved anyway is not a valid artifact.
3. No banned word appears anywhere in the notebook (source or output), case-insensitive
   -- notebooks are a public surface, and this project's own no-AI-attribution rule
   applies to them exactly as it does to the README and the dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_ROOT = REPO_ROOT / "notebooks"

NOTEBOOK_PATHS = [
    NOTEBOOKS_ROOT / "01_six_conditions.ipynb",
    NOTEBOOKS_ROOT / "02_external_validation.ipynb",
]

# Same discipline as the rest of this project's public surfaces (README, dashboard,
# LinkedIn copy): no AI attribution, no trace of how this project's writing got made.
BANNED_WORDS = (
    "coursework",
    "college",
    "rebuild",
    "originally",
    "generated with",
    "claude",
    "anthropic",
)


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text())


def _code_cells(nb: dict) -> list[dict]:
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _cell_text(cell: dict) -> str:
    """Every string an executed cell carries: its source and every text-bearing output."""
    parts = ["".join(cell.get("source", []))]
    for out in cell.get("outputs", []):
        if "text" in out:
            parts.append("".join(out["text"]))
        data = out.get("data", {})
        for mime in ("text/plain", "text/html"):
            if mime in data:
                value = data[mime]
                parts.append(value if isinstance(value, str) else "".join(value))
    return "\n".join(parts)


def _notebook_text(nb: dict) -> str:
    parts = []
    for cell in nb["cells"]:
        parts.append("".join(cell.get("source", [])))
        for out in cell.get("outputs", []):
            if "text" in out:
                parts.append("".join(out["text"]))
            data = out.get("data", {})
            for mime in ("text/plain", "text/html"):
                if mime in data:
                    value = data[mime]
                    parts.append(value if isinstance(value, str) else "".join(value))
    return "\n".join(parts)


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=lambda p: p.name)
def test_notebook_exists(path: Path) -> None:
    assert path.is_file(), f"expected committed notebook at {path}"


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=lambda p: p.name)
def test_notebook_has_at_least_one_code_cell(path: Path) -> None:
    nb = _load_notebook(path)
    assert len(_code_cells(nb)) > 0, f"{path.name}: no code cells found"


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=lambda p: p.name)
def test_notebook_execution_counts_are_contiguous_from_one(path: Path) -> None:
    """Every code cell ran, in order, exactly once, from one top-to-bottom execution.

    ``execution_count`` for the N code cells (in document order) must be exactly
    ``[1, 2, ..., N]`` -- not merely "all present" or "all distinct". A cell skipped,
    re-run out of order, or left over from a partial/interactive editing session would
    still show *some* execution_count, but not this exact contiguous sequence.
    """
    nb = _load_notebook(path)
    code_cells = _code_cells(nb)
    execution_counts = [c.get("execution_count") for c in code_cells]
    expected = list(range(1, len(code_cells) + 1))
    assert execution_counts == expected, (
        f"{path.name}: execution_count sequence {execution_counts} is not the "
        f"contiguous {expected} a single top-to-bottom execution produces"
    )


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=lambda p: p.name)
def test_notebook_has_zero_error_outputs(path: Path) -> None:
    nb = _load_notebook(path)
    errors = []
    for i, cell in enumerate(_code_cells(nb)):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                errors.append((i, out.get("ename"), out.get("evalue")))
    assert errors == [], f"{path.name}: cell(s) with error output(s): {errors}"


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=lambda p: p.name)
def test_notebook_has_no_banned_words(path: Path) -> None:
    nb = _load_notebook(path)
    text = _notebook_text(nb).lower()
    hits = [word for word in BANNED_WORDS if word in text]
    assert hits == [], f"{path.name}: banned word(s) found: {hits}"


@pytest.mark.parametrize("path", NOTEBOOK_PATHS, ids=lambda p: p.name)
def test_notebook_uses_the_registered_ghvenv_kernel(path: Path) -> None:
    nb = _load_notebook(path)
    kernel_name = nb.get("metadata", {}).get("kernelspec", {}).get("name")
    assert kernel_name == "ghvenv", (
        f"{path.name}: kernelspec.name is {kernel_name!r}, expected 'ghvenv' "
        f"(python -m ipykernel install --user --name ghvenv)"
    )
