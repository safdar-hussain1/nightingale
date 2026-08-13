# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Python <-> JavaScript parity: the test that makes "the browser runs the real model" a fact.

For every condition, 200 seeded cases (a quarter of them with 1-4 features
blanked) are scored twice: once by :func:`nightingale.export.predict`, the
reference walker in this package, and once by ``docs/assets/walker.js``
running under real ``node``. The two must agree on ``p_cal`` to
:data:`~nightingale.export.PARITY_TOLERANCE` (1e-9) and on the conformal
verdict EXACTLY.

Both walkers read the COMMITTED ``models/<slug>/model.json`` -- no training
happens in this module. That is deliberate on two counts: it makes the
whole file run in about a second for all six conditions, and it tests the
artifact that actually ships rather than a fresh one that merely resembles
it. A drift between the committed bundle and the live model is a different
claim, asserted in ``tests/test_export.py``.

If ``node`` is absent the tests SKIP LOUDLY -- the skip reason says the JS
half was not verified, so a green run on a machine without node cannot be
mistaken for a run that proved parity.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from nightingale.conditions import CONDITIONS
from nightingale.export import MODELS_ROOT, PARITY_TOLERANCE, load_model, predict

REPO_ROOT = Path(__file__).resolve().parents[1]
WALKER_JS = REPO_ROOT / "docs" / "assets" / "walker.js"

PARITY_SEED = 77
N_CASES = 200
# A quarter of the cases carry missing values -- the branch a browser hits
# constantly, because a real person filling in a risk form leaves fields
# blank.
MISSING_FRACTION = 0.25
MAX_MISSING_PER_CASE = 4

ALL_SLUGS = sorted(CONDITIONS)


def node_executable() -> str:
    """Path to ``node``, or skip the test with an unmistakable reason."""
    found = shutil.which("node")
    if found is None:
        pytest.skip("node not found -- JS parity NOT verified")
    return found


def build_cases(model: dict, seed: int = PARITY_SEED, n: int = N_CASES) -> list[list]:
    """``n`` seeded cases for one model, a quarter of them with blanks.

    Numeric features are drawn uniformly from their declared ``[min, max]``
    and binary features from Bernoulli(0.5) -- the same construction the
    canary block uses, at 25x the volume. Blanked features become ``None``,
    which serialises to JSON ``null`` and is what both walkers treat as
    "not measured".
    """
    rng = np.random.default_rng(seed)
    features = model["features"]
    n_features = len(features)

    cases: list[list] = []
    for _ in range(n):
        row: list[float | None] = []
        for feature in features:
            if feature["type"] == "binary":
                row.append(float(rng.random() < 0.5))
            else:
                row.append(float(rng.uniform(feature["min"], feature["max"])))
        if rng.random() < MISSING_FRACTION:
            k = int(rng.integers(1, min(MAX_MISSING_PER_CASE, n_features) + 1))
            for position in rng.choice(n_features, size=k, replace=False):
                row[int(position)] = None
        cases.append(row)
    return cases


def run_walker_js(node: str, model_path: Path, cases: list[list], tmp_path: Path) -> list[dict]:
    """Score ``cases`` through ``node docs/assets/walker.js`` and parse its stdout."""
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases))
    completed = subprocess.run(
        [node, str(WALKER_JS), str(model_path), str(cases_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"walker.js exited {completed.returncode}\n"
            f"stdout: {completed.stdout}\nstderr: {completed.stderr}"
        )
    return json.loads(completed.stdout)


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_python_and_js_walkers_agree(slug, tmp_path):
    """200 seeded cases per condition: p_cal to 1e-9, verdict exactly."""
    node = node_executable()
    model_path = MODELS_ROOT / slug / "model.json"
    assert model_path.is_file(), f"models/{slug}/model.json is missing"
    model = load_model(model_path)

    cases = build_cases(model)
    assert len(cases) == N_CASES
    n_with_missing = sum(1 for row in cases if any(v is None for v in row))
    assert n_with_missing > 0, "seeded cases must exercise the missing-value path"

    ours = [predict(model, row) for row in cases]
    theirs = run_walker_js(node, model_path, cases, tmp_path)
    assert len(theirs) == N_CASES

    max_diff_cal = max(abs(a["p_cal"] - b["p_cal"]) for a, b in zip(ours, theirs))
    max_diff_raw = max(abs(a["p_raw"] - b["p_raw"]) for a, b in zip(ours, theirs))
    assert max_diff_raw <= PARITY_TOLERANCE, f"{slug}: p_raw max abs diff {max_diff_raw:g}"
    assert max_diff_cal <= PARITY_TOLERANCE, f"{slug}: p_cal max abs diff {max_diff_cal:g}"

    disagreements = [
        (i, a["set"], b["set"]) for i, (a, b) in enumerate(zip(ours, theirs)) if a["set"] != b["set"]
    ]
    assert not disagreements, f"{slug}: conformal verdict disagreements {disagreements[:5]}"


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_js_walker_reproduces_the_stored_canaries(slug, tmp_path):
    """walker.js, independently, gives the canary block's own stored answers."""
    node = node_executable()
    model_path = MODELS_ROOT / slug / "model.json"
    model = load_model(model_path)

    results = run_walker_js(node, model_path, model["canaries"]["inputs"], tmp_path)
    worst = max(
        abs(result["p_cal"] - stored)
        for result, stored in zip(results, model["canaries"]["p_cal"])
    )
    assert worst <= model["canaries"]["tolerance"], f"{slug}: canary max abs diff {worst:g}"


def test_js_walker_self_check_reports_canaries_ok(tmp_path):
    """``checkCanaries`` is what a browser calls on load; it must agree it is fine."""
    node = node_executable()
    script = tmp_path / "check.js"
    reports = {}
    for slug in ALL_SLUGS:
        model_path = MODELS_ROOT / slug / "model.json"
        script.write_text(
            "const w = require({walker});\n"
            "const m = JSON.parse(require('fs').readFileSync({model}, 'utf8'));\n"
            "process.stdout.write(JSON.stringify(w.checkCanaries(m)));\n".format(
                walker=json.dumps(str(WALKER_JS)), model=json.dumps(str(model_path))
            )
        )
        completed = subprocess.run(
            [node, str(script)], capture_output=True, text=True, check=False, timeout=120
        )
        assert completed.returncode == 0, completed.stderr
        reports[slug] = json.loads(completed.stdout)

    for slug, report in reports.items():
        assert report["ok"], f"{slug}: canary self-check failed, {report}"


def test_walker_js_cli_rejects_wrong_argument_counts():
    """The CLI guard must be a real guard, not a crash."""
    node = node_executable()
    completed = subprocess.run(
        [node, str(WALKER_JS)], capture_output=True, text=True, check=False, timeout=60
    )
    assert completed.returncode == 2
    assert "usage: node walker.js" in completed.stderr


def test_walker_js_can_be_required_without_running_the_cli(tmp_path):
    """``require()``ing the walker must not try to read argv -- the browser reuse path."""
    node = node_executable()
    script = tmp_path / "lib.js"
    script.write_text(
        "const w = require({walker});\n"
        "const nodes = [{{f:0,t:5,l:1,r:2,m:2,v:0}},"
        "{{f:-1,t:0,l:-1,r:-1,m:-1,v:-1}},{{f:-1,t:0,l:-1,r:-1,m:-1,v:1}}];\n"
        "process.stdout.write(JSON.stringify(["
        "w.walkTree(nodes,[1]),w.walkTree(nodes,[9]),w.walkTree(nodes,[null]),"
        "w.walkTree(nodes,[NaN])]));\n".format(walker=json.dumps(str(WALKER_JS)))
    )
    completed = subprocess.run(
        [node, str(script)], capture_output=True, text=True, check=False, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == [-1, 1, 1, 1]
