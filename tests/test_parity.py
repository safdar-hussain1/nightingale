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
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from nightingale.conditions import CONDITIONS
from nightingale.export import (
    MODELS_ROOT,
    PARITY_TOLERANCE,
    encoded_frame,
    load_model,
    predict,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
WALKER_JS = REPO_ROOT / "docs" / "assets" / "walker.js"

PARITY_SEED = 77
N_CASES = 200
# HALF the cases are real rows sampled from data/cleaned/<slug>.csv.gz, and
# that split is load-bearing rather than decorative. Synthetic cases drawn
# uniformly from each feature's [min, max] essentially never land exactly on
# a split threshold, so they cannot detect a walker that compares in float64
# instead of float32 -- a reviewer deleted Math.fround from walker.js and all
# 1,200 synthetic parity cases still passed. Under tree_method="hist" the
# split points ARE observed data values, so real rows sit on thresholds
# constantly: the same mutation moves p_cal on 21 of 2,400 real rows, worst
# case by 0.316. See test_float32_rounding_is_load_bearing_in_js for the
# explicit mutation test that pins this.
N_REAL_CASES = 100
N_SYNTHETIC_CASES = N_CASES - N_REAL_CASES
# A quarter of the cases carry missing values -- the branch a browser hits
# constantly, because a real person filling in a risk form leaves fields
# blank.
MISSING_FRACTION = 0.25
MAX_MISSING_PER_CASE = 4

# Real cleaned rows scored by the float64-mutation test. Five of six
# conditions are smaller than this and are checked in full; diabetes is
# sampled down.
REAL_ROW_SAMPLE = 2_000
REAL_ROW_SEED = 202

ALL_SLUGS = sorted(CONDITIONS)


def node_executable() -> str:
    """Path to ``node``, or skip the test with an unmistakable reason."""
    found = shutil.which("node")
    if found is None:
        pytest.skip("node not found -- JS parity NOT verified")
    return found


def _json_row(values) -> list:
    """One numeric row as JSON-safe values: NaN becomes ``None`` (JSON ``null``)."""
    return [None if value != value else float(value) for value in values]


def build_cases(
    model: dict, slug: str, seed: int = PARITY_SEED, n: int = N_CASES
) -> list[list]:
    """``n`` seeded cases for one model: half synthetic, half real, a quarter blanked.

    The synthetic half draws numeric features uniformly from their declared
    ``[min, max]`` and binary features from Bernoulli(0.5) -- the same
    construction the canary block uses. It covers the input space broadly,
    including combinations no patient in the dataset has.

    The real half is a seeded sample of rows from
    ``data/cleaned/<slug>.csv.gz``, encoded exactly as the model consumes
    them. This half is what actually exercises the float32 comparison:
    ``tree_method="hist"`` picks split points FROM observed values, so real
    rows land exactly on thresholds routinely while uniform draws never do.
    Real rows also carry their own genuine NaNs, on the columns where these
    datasets really are missing data rather than wherever a random mask
    fell.

    Blanking is then applied across both halves alike: a quarter of all
    cases get 1-4 features set to ``None``.
    """
    rng = np.random.default_rng(seed)
    features = model["features"]
    n_features = len(features)

    cases: list[list] = []
    for _ in range(n - N_REAL_CASES):
        row: list[float | None] = []
        for feature in features:
            if feature["type"] == "binary":
                row.append(float(rng.random() < 0.5))
            else:
                row.append(float(rng.uniform(feature["min"], feature["max"])))
        cases.append(row)

    encoded = encoded_frame(slug).to_numpy(dtype=float)
    take = min(N_REAL_CASES, len(encoded))
    chosen = rng.choice(len(encoded), size=take, replace=False)
    cases.extend(_json_row(encoded[index]) for index in chosen)

    for row in cases:
        if rng.random() < MISSING_FRACTION:
            k = int(rng.integers(1, min(MAX_MISSING_PER_CASE, n_features) + 1))
            for position in rng.choice(n_features, size=k, replace=False):
                row[int(position)] = None
    return cases


def thresholds_by_feature(model: dict) -> dict[int, set[float]]:
    """``{feature index: {split thresholds used on it}}`` over the whole forest."""
    by_feature: dict[int, set[float]] = {}
    for tree in model["trees"]:
        for node in tree:
            if node["f"] != -1:
                by_feature.setdefault(node["f"], set()).add(node["t"])
    return by_feature


def count_threshold_ties(model: dict, cases: list[list]) -> int:
    """How many (case, feature) values sit EXACTLY on a split point for that feature.

    This is the population that a float64-comparing walker gets wrong, so
    it is asserted to be non-zero rather than hoped for: it is the whole
    reason real rows are mixed into :func:`build_cases`.
    """
    by_feature = thresholds_by_feature(model)
    return sum(
        1
        for row in cases
        for index, value in enumerate(row)
        if value is not None and float(np.float32(value)) in by_feature.get(index, ())
    )


def count_straddle_rows(model: dict, cases: list[list]) -> int:
    """How many cases hold a value where float64 and float32 comparison DISAGREE.

    Landing on a split point is not by itself enough to expose a float64
    walker. If the value is exactly representable in float32 (any small
    integer is), then ``float32(x) == x`` and both comparisons agree -- the
    row goes the same way either way. Divergence needs a value that is NOT
    float32-exact and that rounds ONTO a threshold.

    This distinction is why diabetes is structurally immune: every one of
    its BRFSS survey features is a small integer, so no diabetes row can
    ever tell the two walkers apart, no matter how many are scored. The
    other five conditions carry real-valued measurements and produce
    straddles readily.

    Note the implication runs one way only: a straddling value diverges only
    if its node is actually REACHED on that row's path, so
    ``count_straddle_rows > 0`` does not guarantee a difference, while
    ``count_straddle_rows == 0`` does guarantee no difference.
    """
    by_feature = thresholds_by_feature(model)
    total = 0
    for row in cases:
        for index, value in enumerate(row):
            if value is None:
                continue
            rounded = float(np.float32(value))
            if any((value < t) != (rounded < t) for t in by_feature.get(index, ())):
                total += 1
                break
    return total


def run_walker_js(
    node: str,
    model_path: Path,
    cases: list[list],
    tmp_path: Path,
    walker: Path | None = None,
) -> list[dict]:
    """Score ``cases`` through ``node docs/assets/walker.js`` and parse its stdout.

    ``walker`` overrides which script is run, so the mutation tests below can
    point it at a deliberately-broken copy.
    """
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases))
    completed = subprocess.run(
        [node, str(walker or WALKER_JS), str(model_path), str(cases_path)],
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

    cases = build_cases(model, slug)
    assert len(cases) == N_CASES
    n_with_missing = sum(1 for row in cases if any(v is None for v in row))
    assert n_with_missing > 0, "seeded cases must exercise the missing-value path"
    # The half-real case mix must actually put values ON split points --
    # otherwise the float32 comparison is untested by this file and a
    # float64 walker would pass it. See build_cases' docstring.
    assert count_threshold_ties(model, cases) > 0, (
        f"{slug}: no case value lands on a split threshold, so these cases "
        f"cannot detect a float64 split comparison"
    )

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


# --------------------------------------------------------------------------
# Math.fround is load-bearing: prove it by deleting it
# --------------------------------------------------------------------------


def strip_fround(tmp_path: Path) -> Path:
    """A copy of walker.js with ``Math.fround`` replaced by an identity.

    ``Number(x)`` is the identity on a JavaScript number, so the mutant is
    walker.js comparing in float64 -- the exact bug this project has to be
    unable to ship. The occurrence count is asserted, so a refactor that
    renames or duplicates the call makes this test fail loudly instead of
    quietly mutating nothing and reporting green.
    """
    source = WALKER_JS.read_text()
    call = "Math.fround(value)"
    # The bare name also appears in walker.js's prose comments, so the call
    # site is matched exactly rather than the identifier.
    assert source.count(call) == 1, (
        f"expected exactly one {call} call site in walker.js, found "
        f"{source.count(call)}; update strip_fround if the walker is refactored, "
        f"or this mutation test silently tests nothing"
    )
    mutant = tmp_path / "walker_float64.js"
    mutant.write_text(source.replace(call, "Number(value)"))
    return mutant


def straddle_cases(model: dict) -> list[list]:
    """Cases built to sit in the gap between float64 and float32 comparison.

    For each tree's ROOT node (always visited, so the branch always
    matters), one case is built where that root's feature is set to the
    largest float64 STRICTLY BELOW the split threshold. Every other feature
    is fixed at the midpoint of its declared range so the two walkers see
    identical values everywhere else.

    Such a value satisfies ``x < t`` in float64 but ``float32(x) == t`` --
    so the intact walker goes right and a float64 walker goes left. This is
    a construction, not a sampling: the straddle is guaranteed by
    ``nextafter``, which is why this test does not depend on a real row
    happening to land on a threshold.
    """
    baseline = [
        0.0 if f["type"] == "binary" else (f["min"] + f["max"]) / 2.0
        for f in model["features"]
    ]
    cases = []
    for tree in model["trees"]:
        root = tree[0]
        if root["f"] == -1:
            continue
        row = list(baseline)
        row[root["f"]] = math.nextafter(root["t"], -math.inf)
        cases.append(row)
    return cases


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_float32_rounding_is_load_bearing_in_js(slug, tmp_path):
    """Deleting ``Math.fround`` from walker.js MUST change its answers.

    A reviewer removed ``Math.fround`` and every parity test still passed,
    because uniformly-drawn cases never land on a split point. Two fixes:
    :func:`build_cases` now mixes in real rows (which do), and this test
    removes the guesswork entirely by constructing inputs that straddle the
    float32/float64 boundary. Both the mutant's disagreement with the intact
    walker AND the intact walker's agreement with Python are asserted, so
    this cannot pass by both implementations being wrong.

    The comparison is on **p_raw**, not p_cal, and that matters. Two of the
    six isotonic calibrators (breast-cancer, kidney-disease) are saturated
    enough that they map a wrongly-routed margin back onto the SAME
    calibrated probability -- for those conditions all 300 straddle cases
    differ in p_raw and none differ in p_cal. A routing gate written at the
    p_cal level would therefore be blind on exactly the conditions whose
    calibrators hide the most, so it is written where routing is observable.
    """
    node = node_executable()
    model_path = MODELS_ROOT / slug / "model.json"
    model = load_model(model_path)

    cases = straddle_cases(model)
    assert cases, f"{slug}: no internal root nodes to straddle"

    intact = run_walker_js(node, model_path, cases, tmp_path)
    mutant = run_walker_js(node, model_path, cases, tmp_path, walker=strip_fround(tmp_path))

    differing = sum(1 for a, b in zip(intact, mutant) if a["p_raw"] != b["p_raw"])
    assert differing == len(cases), (
        f"{slug}: stripping Math.fround changed only {differing} of {len(cases)} "
        f"straddle cases -- each one is constructed to flip a root branch, so "
        f"anything short of all of them means the construction has stopped working"
    )

    # ...and the INTACT walker is the one that agrees with Python (which
    # rounds through numpy.float32), so the mutant is the wrong one.
    ours = [predict(model, row) for row in cases]
    worst = max(abs(a["p_cal"] - b["p_cal"]) for a, b in zip(ours, intact))
    assert worst <= PARITY_TOLERANCE


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_real_cleaned_rows_detect_a_float64_walker(slug, tmp_path):
    """Real rows -- not just constructed ones -- must expose the mutation.

    The straddle test above is a guarantee by construction. This one asks
    the more practical question: would ORDINARY data have caught it? It
    scores real rows from ``data/cleaned/<slug>.csv.gz`` through the intact
    and fround-stripped walkers and requires them to disagree.

    The assertion is conditional on :func:`count_straddle_rows`, and that is
    an exact criterion rather than a hedge. A float64 walker can only
    diverge on a value that is NOT float32-exact and rounds onto a split
    threshold. **diabetes has zero such values by construction** -- every
    BRFSS feature is a small integer, exactly representable in float32 -- so
    no number of diabetes rows can ever separate the two walkers, and the
    test asserts that immunity is total rather than pretending to cover it.
    The other five conditions carry real-valued measurements and are
    required to diverge. (Measured: breast-cancer 51/569, cervical-cancer
    15/858, heart-disease 18/920, kidney-disease 18/400, liver-disease
    84/583, diabetes 0/2000.)
    """
    node = node_executable()
    model_path = MODELS_ROOT / slug / "model.json"
    model = load_model(model_path)

    encoded = encoded_frame(slug).to_numpy(dtype=float)
    rng = np.random.default_rng(REAL_ROW_SEED)
    take = min(REAL_ROW_SAMPLE, len(encoded))
    chosen = np.sort(rng.choice(len(encoded), size=take, replace=False))
    cases = [_json_row(encoded[index]) for index in chosen]

    intact = run_walker_js(node, model_path, cases, tmp_path)
    mutant = run_walker_js(node, model_path, cases, tmp_path, walker=strip_fround(tmp_path))
    differing = sum(1 for a, b in zip(intact, mutant) if a["p_raw"] != b["p_raw"])

    straddles = count_straddle_rows(model, cases)
    if straddles == 0:
        assert differing == 0, (
            f"{slug}: no row holds a value where float32 and float64 comparison "
            f"disagree, so no row can possibly route differently -- yet "
            f"{differing} did"
        )
    else:
        assert differing > 0, (
            f"{slug}: {straddles} of {len(cases)} real rows straddle a split "
            f"threshold, but a float64 walker scored every one of them "
            f"identically -- real data would not catch the bug"
        )


# --------------------------------------------------------------------------
# input coercion: a blank form field is missing, never zero
# --------------------------------------------------------------------------


def run_js(node: str, tmp_path: Path, body: str, name: str = "snippet.js"):
    """Run a small JS snippet that prints JSON to stdout, and parse it."""
    script = tmp_path / name
    script.write_text(
        "const walker = require(" + json.dumps(str(WALKER_JS)) + ");\n"
        "const fs = require('fs');\n" + body
    )
    completed = subprocess.run(
        [node, str(script)], capture_output=True, text=True, check=False, timeout=120
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_js_treats_blank_and_unparseable_input_as_missing(tmp_path):
    """``""`` must NOT become 0. ``Number('') === 0`` is the whole hazard."""
    node = node_executable()
    model_path = MODELS_ROOT / "heart-disease" / "model.json"
    model = load_model(model_path)
    n = len(model["features"])

    # A realistic partly-filled form: feature 0 answered, the rest blank in
    # four different ways a browser can produce.
    variants = {
        "empty_string": [45.0] + [""] * (n - 1),
        "whitespace": [45.0] + ["   "] * (n - 1),
        "non_numeric": [45.0] + ["abc"] * (n - 1),
        "null": [45.0] + [None] * (n - 1),
        "nan_number": [45.0] + [None] * (n - 1),  # NaN cannot survive JSON; see below
    }
    results = run_js(
        node,
        tmp_path,
        "const m = JSON.parse(fs.readFileSync(" + json.dumps(str(model_path)) + ", 'utf8'));\n"
        "const v = " + json.dumps(variants) + ";\n"
        "v.nan_number = v.nan_number.map((x, i) => (i === 0 ? x : NaN));\n"
        "const zeros = [45.0].concat(new Array(m.features.length - 1).fill(0));\n"
        "const out = {zeros: walker.predict(m, zeros)};\n"
        "for (const k of Object.keys(v)) { out[k] = walker.predict(m, v[k]); }\n"
        "process.stdout.write(JSON.stringify(out));\n",
    )

    missing = results["null"]
    for key in ("empty_string", "whitespace", "non_numeric", "nan_number"):
        assert results[key] == missing, f"{key} must route as missing, like null"

    # The hazard itself: if '' silently became 0, the blank form would score
    # identically to a form filled in with zeros. It must not.
    assert results["zeros"]["p_cal"] != missing["p_cal"], (
        "a form of zeros and a form of blanks must not produce the same risk"
    )

    # And the Python walker agrees with the JS "all blank" answer.
    row = [45.0] + [None] * (n - 1)
    assert abs(predict(model, row)["p_cal"] - missing["p_cal"]) <= PARITY_TOLERANCE


def test_js_coerces_scalars_the_way_the_comment_claims(tmp_path):
    """``toFeatureValue`` unit cases, including the ones Number() gets wrong."""
    node = node_executable()
    results = run_js(
        node,
        tmp_path,
        "const cases = ['', '  ', 'abc', '12px', 'NaN', null, undefined, true, false,\n"
        "               '3.5', ' 42 ', 0, -1.5, [], {}];\n"
        "process.stdout.write(JSON.stringify(cases.map(function (c) {\n"
        "  const v = walker.toFeatureValue(c);\n"
        "  return (typeof v === 'number' && isNaN(v)) ? 'MISSING' : v;\n"
        "})));\n"
        "",
        name="coerce.js",
    )
    assert results == [
        "MISSING",  # ''      -- Number('') === 0, the hazard
        "MISSING",  # '  '    -- Number('  ') === 0 too
        "MISSING",  # 'abc'
        "MISSING",  # '12px'
        "MISSING",  # 'NaN'
        "MISSING",  # null    -- Number(null) === 0
        "MISSING",  # undefined
        1,  # true  -- a ticked checkbox is a real 1
        0,  # false -- an unticked checkbox is a real 0, not "unanswered"
        3.5,  # '3.5'  -- a genuinely filled-in field still parses
        42,  # ' 42 ' -- with surrounding whitespace
        0,  # 0     -- a real zero the user typed is kept
        -1.5,
        "MISSING",  # []  -- Number([]) === 0
        "MISSING",  # {}
    ]


def test_js_rejects_a_wrong_length_row(tmp_path):
    """Structural damage throws on BOTH sides -- it is a caller bug, not a blank."""
    node = node_executable()
    model_path = MODELS_ROOT / "heart-disease" / "model.json"
    model = load_model(model_path)

    result = run_js(
        node,
        tmp_path,
        "const m = JSON.parse(fs.readFileSync(" + json.dumps(str(model_path)) + ", 'utf8'));\n"
        "const out = {};\n"
        "for (const [name, row] of [['short', [1, 2, 3]], ['long', new Array(99).fill(1)],\n"
        "                           ['not_an_array', 'nope']]) {\n"
        "  try { walker.predict(m, row); out[name] = null; }\n"
        "  catch (e) { out[name] = e.message; }\n"
        "}\n"
        "process.stdout.write(JSON.stringify(out));\n",
        name="length.js",
    )
    for name in ("short", "long", "not_an_array"):
        assert result[name] is not None, f"{name} row must throw, not be guessed at"
        assert "expected" in result[name]

    # Python refuses the same input, with its own error type.
    with pytest.raises(ValueError, match="expected 13 feature values"):
        predict(model, [1.0, 2.0, 3.0])


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
