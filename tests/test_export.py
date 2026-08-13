# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Tests for :mod:`nightingale.export`: the bundle, and the walker that reads it.

The claim this file has to defend is "the browser runs the real model", so
the tests are layered from the smallest verifiable thing upwards:

1. A toy 2-tree booster, walked by hand in pure Python, reproduces
   ``predict_proba``. If this fails, nothing else in the export means
   anything.
2. The float32 split semantics and the base_score offset -- the two places
   where a plausible-looking implementation is silently wrong -- are pinned
   by tests that FAIL if you remove them.
3. The exported calibrator reproduces the fitted calibrator's own output.
4. The six committed ``models/<slug>/model.json`` files satisfy the schema,
   the gzip budget, and their own canary block.

The cross-language half of the claim lives in ``tests/test_parity.py``.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from nightingale.calibrate import IsotonicCalibrator, SigmoidCalibrator, fit_calibrator
from nightingale.conditions import CONDITIONS
from nightingale.conformal import prediction_set
from nightingale.export import (
    CANARY_MISSING_COUNTS,
    CANARY_TOLERANCE,
    GZIP_BUDGET_BYTES,
    MODELS_ROOT,
    N_CANARIES,
    PARITY_TOLERANCE,
    SCHEMA_VERSION,
    XGB_NATIVE_TOLERANCE,
    _sigmoid,
    _walk_tree,
    apply_calibrator,
    base_margin,
    build_features,
    encoded_frame,
    export_model,
    export_trees,
    load_model,
    parse_base_score,
    predict,
    train_result,
)

# breast-cancer trains in ~13s and is memoised for the process, so it is the
# condition used wherever a live fitted model is genuinely needed. Everything
# that can be asserted against the COMMITTED model.json is asserted there
# instead, for all six, at no training cost.
FAST_SLUG = "breast-cancer"

ALL_SLUGS = sorted(CONDITIONS)

# Rows scored per condition in the predict_proba gate. Every condition but
# diabetes is smaller than this and is checked in full; diabetes's 253,680
# rows are sampled down, which reduces how many rows are checked and not how
# tightly any of them is.
REAL_ROW_SAMPLE = 2_000
REAL_ROW_SEED = 202


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def toy_model(n_estimators: int = 2, max_depth: int = 2, seed: int = 0):
    """A 2-tree depth-2 booster on 8 synthetic rows, three of them carrying NaN.

    Deliberately tiny: with 8 rows and 3 features the whole forest can be
    read by eye out of ``get_dump``, so a failure here is debuggable rather
    than merely alarming.

    The labels are deliberately UNBALANCED (3 positives of 8). With a
    balanced 4/8 fixture, ``boost_from_average`` sets ``base_score`` to
    exactly 0.5, whose logit is 0 -- and a toy test that cannot tell a
    correct base_score offset from a missing one is worse than no test,
    since it reports green either way.
    """
    X = np.array(
        [
            [1.0, 10.0, 0.0],
            [2.0, 20.0, 1.0],
            [3.0, math.nan, 0.0],
            [4.0, 40.0, 1.0],
            [5.0, 50.0, 0.0],
            [math.nan, 60.0, 1.0],
            [7.0, 70.0, 0.0],
            [8.0, math.nan, 1.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0])
    model = xgb.XGBClassifier(
        tree_method="hist",
        random_state=seed,
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=0.3,
        # 8 rows cannot satisfy the default min_child_weight=1 on both
        # sides of a split, so the default would fit a forest of bare
        # leaves -- green, and testing nothing about tree walking.
        min_child_weight=0,
    )
    model.fit(X, y)
    return model, X, y


def walk_forest(booster, feature_names: list[str], rows) -> np.ndarray:
    """p_raw for every row, using ONLY the exported trees and the export's walker.

    Deliberately assembles the margin here rather than calling
    :func:`~nightingale.export.predict`, because these tests run before any
    calibrator or conformal threshold exists -- the claim under test is
    trees + base_score == XGBoost, and nothing else.
    """
    trees = export_trees(booster, feature_names)
    offset = base_margin(parse_base_score(booster))
    return np.asarray(
        [_sigmoid(offset + sum(_walk_tree(tree, row) for tree in trees)) for row in rows]
    )


def committed_model(slug: str) -> dict:
    path = MODELS_ROOT / slug / "model.json"
    if not path.is_file():
        pytest.fail(
            f"models/{slug}/model.json is missing -- regenerate it with "
            f"PYTHONPATH=src python -m nightingale.export"
        )
    return load_model(path)


# --------------------------------------------------------------------------
# 1. the toy model: the export's own walker vs predict_proba
# --------------------------------------------------------------------------


def test_toy_two_tree_walk_reproduces_predict_proba():
    """The whole export rests on this: flat trees + base_score == XGBoost."""
    model, X, _ = toy_model()
    booster = model.get_booster()
    names = [f"f{i}" for i in range(X.shape[1])]

    ours = walk_forest(booster, names, X.tolist())
    theirs = model.predict_proba(X)[:, 1].astype(float)

    assert np.max(np.abs(ours - theirs)) <= XGB_NATIVE_TOLERANCE


def test_toy_walk_reproduces_predict_proba_on_rows_with_nan():
    """NaN rows specifically -- the ``m`` branch is the easiest thing to get wrong."""
    model, X, _ = toy_model()
    booster = model.get_booster()
    names = [f"f{i}" for i in range(X.shape[1])]

    nan_rows = [row for row in X.tolist() if any(math.isnan(v) for v in row)]
    assert len(nan_rows) == 3  # the fixture's NaN rows, asserted so it can't drift

    ours = walk_forest(booster, names, nan_rows)
    theirs = model.predict_proba(np.asarray(nan_rows))[:, 1].astype(float)
    assert np.max(np.abs(ours - theirs)) <= XGB_NATIVE_TOLERANCE


def test_none_and_nan_route_identically():
    """JSON has no NaN, so ``null`` must mean exactly what ``float('nan')`` means."""
    model, X, _ = toy_model()
    booster = model.get_booster()
    names = [f"f{i}" for i in range(X.shape[1])]

    with_nan = [3.0, math.nan, 0.0]
    with_none = [3.0, None, 0.0]
    assert walk_forest(booster, names, [with_nan])[0] == walk_forest(booster, names, [with_none])[0]


def test_larger_model_walk_reproduces_predict_proba_on_50_rows():
    """The base_score offset, verified where accumulated float32 error is visible.

    50 rows through a 300-tree depth-4 fit: this is the configuration that
    established :data:`XGB_NATIVE_TOLERANCE`. If the offset were dropped or
    applied as a raw probability instead of a logit, the error here would be
    ~0.1, not ~1e-7.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6))
    X[rng.random((400, 6)) < 0.1] = np.nan
    y = (X[:, 0] > 0).astype(int)
    y[rng.random(400) < 0.15] ^= 1

    model = xgb.XGBClassifier(
        tree_method="hist", random_state=42, max_depth=4, n_estimators=300, learning_rate=0.1
    )
    model.fit(X, y)
    booster = model.get_booster()
    names = [f"f{i}" for i in range(X.shape[1])]

    rows = X[:50]
    ours = walk_forest(booster, names, rows.tolist())
    theirs = model.predict_proba(rows)[:, 1].astype(float)
    assert np.max(np.abs(ours - theirs)) <= XGB_NATIVE_TOLERANCE


def test_dropping_the_base_score_offset_breaks_the_walk():
    """A guard on the guard: without the logit offset, the toy model is visibly wrong."""
    model, X, _ = toy_model()
    booster = model.get_booster()
    trees = export_trees(booster, [f"f{i}" for i in range(X.shape[1])])

    theirs = model.predict_proba(X)[:, 1].astype(float)
    without_offset = np.array(
        [1.0 / (1.0 + math.exp(-sum(_walk_tree(t, row) for t in trees))) for row in X.tolist()]
    )
    assert np.max(np.abs(without_offset - theirs)) > 1e-3


# --------------------------------------------------------------------------
# 2. float32 split semantics
# --------------------------------------------------------------------------


def test_split_comparison_happens_in_float32():
    """A row sitting exactly on a hist split point must branch XGBoost's way.

    ``tree_method="hist"`` picks split points from observed feature values,
    so a training row landing exactly on one is routine, not exotic. Under a
    float64 comparison such a row can take the opposite branch: this test
    feeds every training row back through the walker and requires agreement
    with ``predict_proba``, which a float64 comparison fails.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    X[rng.random((200, 5)) < 0.1] = np.nan
    y = (X[:, 0] > 0).astype(int)
    y[rng.random(200) < 0.1] ^= 1
    model = xgb.XGBClassifier(
        tree_method="hist", random_state=42, max_depth=2, n_estimators=3, learning_rate=0.3
    )
    model.fit(X, y)
    booster = model.get_booster()
    names = [f"f{i}" for i in range(X.shape[1])]

    ours = walk_forest(booster, names, X.tolist())
    theirs = model.predict_proba(X)[:, 1].astype(float)
    assert np.max(np.abs(ours - theirs)) <= XGB_NATIVE_TOLERANCE

    # And the same walk WITHOUT float32 rounding gets rows wrong -- proving
    # the tolerance above is not passing by luck.
    trees = export_trees(booster, names)
    offset = base_margin(parse_base_score(booster))

    def walk_float64(nodes, row):
        i = 0
        while nodes[i]["f"] != -1:
            node = nodes[i]
            v = row[node["f"]]
            if v is None or math.isnan(v):
                i = node["m"]
            elif v < node["t"]:
                i = node["l"]
            else:
                i = node["r"]
        return nodes[i]["v"]

    naive = np.array(
        [
            1.0 / (1.0 + math.exp(-(offset + sum(walk_float64(t, row) for t in trees))))
            for row in X.tolist()
        ]
    )
    assert np.max(np.abs(naive - theirs)) > 1e-3


def test_thresholds_and_leaves_are_float32_exact():
    """Stored numbers must BE float32 values, not float64 neighbours of them.

    ``get_dump`` prints each float32 as nine-digit decimal text; parsing that
    text gives a float64 a few 1e-12 off the real float32. Keeping the
    off-by-1e-12 version is what sends rows sitting exactly on a split point
    down the wrong branch. Fitted on continuous data on purpose -- a fixture
    whose thresholds are all small integers passes this test no matter what
    the exporter does.
    """
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    model = xgb.XGBClassifier(
        tree_method="hist", random_state=0, max_depth=3, n_estimators=5, learning_rate=0.3
    )
    model.fit(X, y)
    trees = export_trees(model.get_booster(), [f"f{i}" for i in range(X.shape[1])])

    fractional = 0
    for tree in trees:
        for node in tree:
            assert float(np.float32(node["t"])) == node["t"]
            assert float(np.float32(node["v"])) == node["v"]
            if node["f"] != -1 and node["t"] != int(node["t"]):
                fractional += 1
    assert fractional > 0, "fixture must produce non-integer thresholds to be meaningful"


# --------------------------------------------------------------------------
# 3. tree flattening
# --------------------------------------------------------------------------


def test_flat_nodes_have_the_documented_schema():
    model, X, _ = toy_model()
    trees = export_trees(model.get_booster(), [f"f{i}" for i in range(X.shape[1])])
    assert trees
    # The fixture must actually SPLIT, or every structural assertion below
    # is vacuous on a forest of bare leaves.
    assert any(node["f"] != -1 for tree in trees for node in tree)
    for tree in trees:
        for node in tree:
            assert set(node) == {"f", "t", "l", "r", "m", "v"}
            assert isinstance(node["f"], int)
            if node["f"] == -1:
                assert node["l"] == node["r"] == node["m"] == -1
            else:
                # Children are positions in this same list, and `missing`
                # always points at one of the two real children.
                assert 0 <= node["l"] < len(tree)
                assert 0 <= node["r"] < len(tree)
                assert node["m"] in (node["l"], node["r"])


def test_feature_indices_resolve_by_name_before_position():
    """A column literally called ``f2`` must not be read as position 2."""
    df = pd.DataFrame(
        {
            "f2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "other": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    model = xgb.XGBClassifier(
        tree_method="hist", random_state=0, max_depth=2, n_estimators=2, learning_rate=0.3
    )
    model.fit(df, y)
    trees = export_trees(model.get_booster(), ["f2", "other"])
    used = {node["f"] for tree in trees for node in tree if node["f"] != -1}
    assert used <= {0, 1}


def test_indicator_columns_export_with_full_nodes():
    """Bool (one-hot) columns must not lose their threshold or missing direction.

    XGBoost types a bool DataFrame column as an indicator and its JSON
    dumper then omits ``split_condition`` and ``missing`` from those nodes.
    kidney-disease has such columns, so this is not hypothetical: without
    the ``feature_types`` override in :func:`export_trees` the export either
    crashes or (worse, if the missing key were defaulted) ships trees that
    route blanks the wrong way.
    """
    rng = np.random.default_rng(11)
    n = 300
    flag = rng.random(n) < 0.5
    frame = pd.DataFrame({"value": rng.normal(size=n), "flag": flag})
    frame.loc[rng.random(n) < 0.15, "value"] = np.nan
    y = (flag.astype(int) + (frame["value"].fillna(0) > 0).astype(int) > 1).astype(int).to_numpy()

    model = xgb.XGBClassifier(
        tree_method="hist", random_state=0, max_depth=3, n_estimators=20, learning_rate=0.3
    )
    model.fit(frame, y)
    assert "i" in (model.get_booster().feature_types or []), "fixture must produce an indicator"

    names = list(frame.columns)
    trees = export_trees(model.get_booster(), names)
    flag_index = names.index("flag")
    assert any(node["f"] == flag_index for tree in trees for node in tree), (
        "fixture must actually split on the indicator column"
    )

    ours = walk_forest(model.get_booster(), names, frame.to_numpy(dtype=float).tolist())
    theirs = model.predict_proba(frame)[:, 1].astype(float)
    assert np.max(np.abs(ours - theirs)) <= XGB_NATIVE_TOLERANCE

    # The override must not leave the booster mutated.
    assert model.get_booster().feature_types == ["float", "i"]


def test_missing_branch_is_taken_for_missing_values():
    """Feeding a feature as missing must land on ``m``, not on ``l``/``r``."""
    nodes = [
        {"f": 0, "t": 5.0, "l": 1, "r": 2, "m": 2, "v": 0.0},
        {"f": -1, "t": 0.0, "l": -1, "r": -1, "m": -1, "v": -1.0},
        {"f": -1, "t": 0.0, "l": -1, "r": -1, "m": -1, "v": 1.0},
    ]
    assert _walk_tree(nodes, [1.0]) == -1.0
    assert _walk_tree(nodes, [9.0]) == 1.0
    assert _walk_tree(nodes, [None]) == 1.0
    assert _walk_tree(nodes, [math.nan]) == 1.0


# --------------------------------------------------------------------------
# 4. calibrator export round-trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_exported_calibrator_reproduces_the_fitted_one(method):
    rng = np.random.default_rng(7)
    p_raw = rng.uniform(0.001, 0.999, size=500)
    y = (rng.random(500) < p_raw**1.5).astype(int)

    calibrator = fit_calibrator(y, p_raw, method=method)
    exported = calibrator.export()
    assert exported["type"] == method

    probe = np.concatenate([p_raw, [0.0, 1.0, 0.5]])
    ours = np.array([apply_calibrator(exported, float(p)) for p in probe])
    theirs = np.asarray(calibrator.predict(probe), dtype=float)
    assert np.max(np.abs(ours - theirs)) <= 1e-12


def test_calibrator_export_is_json_serialisable():
    """Both families must survive a JSON round-trip with no precision loss."""
    sigmoid = SigmoidCalibrator(a=-3.25, b=0.5)
    isotonic = IsotonicCalibrator(x=[0.0, 0.4, 1.0], y=[0.0, 0.3, 1.0])
    for calibrator in (sigmoid, isotonic):
        restored = json.loads(json.dumps(calibrator.export()))
        for p in (0.0, 0.13, 0.4, 0.77, 1.0):
            assert apply_calibrator(restored, p) == apply_calibrator(calibrator.export(), p)


def test_isotonic_clamps_outside_its_breakpoints():
    exported = {"type": "isotonic", "x": [0.2, 0.5, 0.8], "y": [0.1, 0.4, 0.9]}
    assert apply_calibrator(exported, 0.0) == 0.1
    assert apply_calibrator(exported, 1.0) == 0.9
    assert apply_calibrator(exported, 0.35) == pytest.approx(0.25, abs=1e-12)


def test_unknown_calibrator_type_is_rejected():
    with pytest.raises(ValueError, match="unknown calibrator type"):
        apply_calibrator({"type": "beta"}, 0.5)


# --------------------------------------------------------------------------
# 5. base_score parsing
# --------------------------------------------------------------------------


def test_base_score_parses_the_bracketed_vector_form():
    """xgboost >= 2.0 serialises base_score as ``"[5.5326086E-1]"``."""
    model, _, y = toy_model()
    booster = model.get_booster()
    base_score = parse_base_score(booster)
    assert 0.0 < base_score < 1.0
    # boost_from_average: base_score is the training-set positive rate,
    # to float32 precision.
    assert base_score == pytest.approx(float(y.mean()), abs=1e-6)
    assert base_margin(base_score) == pytest.approx(
        math.log(base_score / (1 - base_score)), abs=0.0
    )


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
def test_base_margin_rejects_impossible_base_scores(bad):
    with pytest.raises(ValueError, match="base_score must be in"):
        base_margin(bad)


# --------------------------------------------------------------------------
# 6. the feature contract
# --------------------------------------------------------------------------


def test_features_follow_model_column_order_with_real_ranges():
    encoded = encoded_frame(FAST_SLUG)
    names = list(encoded.columns)
    features = build_features(encoded, names)

    assert [f["name"] for f in features] == names
    for feature in features:
        assert feature["allow_missing"] is True
        assert feature["type"] in {"number", "binary"}
        assert feature["label"]
        assert isinstance(feature["unit"], str)
        assert feature["min"] <= feature["max"]
        if feature["type"] == "number":
            column = pd.to_numeric(encoded[feature["name"]])
            assert feature["min"] == pytest.approx(float(column.min()), abs=0.0)
            assert feature["max"] == pytest.approx(float(column.max()), abs=0.0)


def test_dummy_columns_are_typed_binary_and_labelled():
    """No cleaned dataset currently has an object column, but the encoder path must work."""
    frame = pd.DataFrame(
        {
            "age": [40.0, 55.0, 61.0],
            "cp": ["typical", "atypical", "typical"],
        }
    )
    encoded = pd.get_dummies(frame)
    features = build_features(encoded, list(encoded.columns))

    by_name = {f["name"]: f for f in features}
    assert by_name["age"]["type"] == "number"
    assert by_name["age"]["unit"] == "years"
    for name in ("cp_typical", "cp_atypical"):
        assert by_name[name]["type"] == "binary"
        assert (by_name[name]["min"], by_name[name]["max"]) == (0.0, 1.0)
        assert by_name[name]["unit"] == ""
        assert by_name[name]["label"].startswith("Chest pain type: ")


def test_known_clinical_units_are_populated():
    model = committed_model("heart-disease")
    units = {f["name"]: f["unit"] for f in model["features"]}
    assert units["age"] == "years"
    assert units["chol"] == "mg/dL"
    assert units["trestbps"] == "mm Hg"
    assert units["cp"] == ""  # a code, not a measurement -- no invented unit


# --------------------------------------------------------------------------
# 7. the committed bundles
# --------------------------------------------------------------------------


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_committed_model_json_matches_the_schema(slug):
    model = committed_model(slug)
    assert model["schema_version"] == SCHEMA_VERSION
    assert model["condition"] == slug
    assert set(model) == {
        "schema_version",
        "condition",
        "trees",
        "base_score",
        "base_score_probability",
        "calibrator",
        "conformal",
        "features",
        "canaries",
        "provenance",
    }
    assert model["trees"] and all(tree for tree in model["trees"])
    assert model["calibrator"]["type"] in {"sigmoid", "isotonic"}
    assert model["conformal"]["alpha"] == 0.1
    assert 0.0 <= model["conformal"]["q_hat"] <= 1.0
    assert model["provenance"]["author"] == "Safdar Hussain"
    assert model["provenance"]["commit"] != "unknown"
    assert model["provenance"]["built_utc"].endswith("Z")
    # base_score is the margin offset; base_score_probability its source.
    assert model["base_score"] == pytest.approx(
        math.log(
            model["base_score_probability"] / (1 - model["base_score_probability"])
        ),
        abs=1e-15,
    )


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_committed_model_json_is_within_the_gzip_budget(slug):
    """The accuracy-per-KB gate: a model a browser can actually download."""
    raw = (MODELS_ROOT / slug / "model.json").read_bytes()
    compressed = len(gzip.compress(raw, compresslevel=9))
    assert compressed < GZIP_BUDGET_BYTES, (
        f"{slug}/model.json gzips to {compressed} bytes, over the "
        f"{GZIP_BUDGET_BYTES}-byte budget"
    )


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_committed_model_json_has_no_nan_or_infinity(slug):
    """A hand-rolled JSON parser in a browser must never meet a bare ``NaN``."""
    text = (MODELS_ROOT / slug / "model.json").read_text()
    assert "NaN" not in text
    assert "Infinity" not in text


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_committed_features_match_the_cleaned_data(slug):
    model = committed_model(slug)
    encoded = encoded_frame(slug)
    assert [f["name"] for f in model["features"]] == list(encoded.columns)
    n_features = len(model["features"])
    for tree in model["trees"]:
        for node in tree:
            assert -1 <= node["f"] < n_features


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_committed_canaries_reproduce_through_the_walker(slug):
    """Self-consistency: the serialised artifact still gives its own stored answers."""
    model = committed_model(slug)
    canaries = model["canaries"]
    assert len(canaries["inputs"]) == N_CANARIES
    assert len(canaries["p_cal"]) == N_CANARIES
    assert canaries["tolerance"] == PARITY_TOLERANCE

    worst = 0.0
    for row, stored in zip(canaries["inputs"], canaries["p_cal"]):
        assert len(row) == len(model["features"])
        worst = max(worst, abs(predict(model, row)["p_cal"] - stored))
    assert worst <= CANARY_TOLERANCE


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_committed_canaries_exercise_the_missing_path(slug):
    model = committed_model(slug)
    inputs = model["canaries"]["inputs"]
    for index, expected in CANARY_MISSING_COUNTS.items():
        assert sum(1 for v in inputs[index] if v is None) == expected
    # ...and the early canaries are fully populated, so both paths are covered.
    assert all(v is not None for v in inputs[0])


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_committed_canary_inputs_sit_inside_the_declared_ranges(slug):
    model = committed_model(slug)
    for row in model["canaries"]["inputs"]:
        for value, feature in zip(row, model["features"]):
            if value is None:
                continue
            assert feature["min"] <= value <= feature["max"]


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_committed_conformal_qhat_matches_the_published_oof(slug):
    """q_hat must be the cross-conformal threshold from the committed OOF frame."""
    from nightingale.conformal import conformal_qhat

    oof = pd.read_csv(MODELS_ROOT / slug / "oof_predictions.csv.gz", compression="gzip")
    expected = conformal_qhat(oof["y_true"].to_numpy(), oof["p_cal"].to_numpy(), alpha=0.1)
    model = committed_model(slug)
    assert model["conformal"]["q_hat"] == pytest.approx(expected, abs=1e-12)


# --------------------------------------------------------------------------
# 8. the bundle is the live model, not a stale copy
# --------------------------------------------------------------------------


def test_export_is_deterministic_and_matches_the_committed_bundle(tmp_path):
    """Re-exporting ``breast-cancer`` reproduces the committed file byte for byte.

    This is the test that keeps ``models/breast-cancer/model.json`` honest:
    if the trees, the calibrator, q_hat, the feature contract or the canary
    values drift from what ``train_condition`` actually produces, the bytes
    differ and this fails. It also pins the byte-reproducibility Task 11's
    signing depends on -- which is why ``provenance.built_utc`` is the HEAD
    commit's timestamp rather than the wall clock.
    """
    written = export_model(FAST_SLUG, out_dir=tmp_path)
    committed = MODELS_ROOT / FAST_SLUG / "model.json"
    fresh = json.loads(written.read_text())
    stored = json.loads(committed.read_text())

    # provenance moves with HEAD, so compare it structurally and everything
    # else exactly.
    assert set(fresh["provenance"]) == set(stored["provenance"])
    fresh.pop("provenance")
    stored.pop("provenance")
    assert fresh == stored


@pytest.mark.parametrize("slug", ALL_SLUGS)
def test_exported_walker_matches_predict_proba_on_real_data(slug):
    """Every shipped bundle scores real rows the way XGBoost does.

    All six conditions, not just the fast one: this is the gate that says
    the exported trees ARE the trained model, and a gate that only covers
    one of six conditions is not a gate. Real cleaned rows specifically --
    they are the ones that sit exactly on ``hist`` split points, so this
    also exercises the float32 comparison on every condition.

    Runtime: each condition has to be refit, because Task 8 persisted
    metrics and OOF predictions but no fitted estimator. ``train_result``
    memoises per process, and diabetes (~5 min to fit, 253,680 rows) is
    scored on a seeded 2,000-row sample rather than the whole frame --
    sampling only reduces how many rows are checked, never how tightly.
    """
    result = train_result(slug)
    encoded = encoded_frame(slug)
    model = committed_model(slug)

    if len(encoded) > REAL_ROW_SAMPLE:
        chosen = np.random.default_rng(REAL_ROW_SEED).choice(
            len(encoded), size=REAL_ROW_SAMPLE, replace=False
        )
        rows = encoded.iloc[np.sort(chosen)]
    else:
        rows = encoded

    theirs = result.final_model.predict_proba(rows)[:, 1].astype(float)
    ours = np.array([predict(model, row)["p_raw"] for row in rows.to_numpy(dtype=float).tolist()])
    worst = float(np.max(np.abs(ours - theirs)))
    assert worst <= XGB_NATIVE_TOLERANCE, f"{slug}: max abs diff {worst:g} over {len(rows)} rows"


def test_exported_calibrator_and_verdict_match_the_python_pipeline():
    """p_cal and the conformal verdict, end to end, against the fitted objects."""
    result = train_result(FAST_SLUG)
    encoded = encoded_frame(FAST_SLUG)
    model = committed_model(FAST_SLUG)
    q_hat = model["conformal"]["q_hat"]

    rows = encoded.head(100).to_numpy(dtype=float).tolist()
    walked = [predict(model, row) for row in rows]

    p_raw = np.array([w["p_raw"] for w in walked])
    expected_cal = np.asarray(result.calibrator.predict(p_raw), dtype=float)
    ours_cal = np.array([w["p_cal"] for w in walked])
    assert np.max(np.abs(ours_cal - expected_cal)) <= 1e-12

    for walk, p in zip(walked, expected_cal):
        assert walk["set"] == prediction_set(float(p), q_hat)


def test_python_predict_refuses_a_wrong_length_row():
    """Python's job is pipeline strictness: a malformed row is a bug, not a blank.

    walker.js deliberately differs on *unfillable values* (``""``/``"abc"``
    route as missing, because a person left a form field blank), but agrees
    on *structural* damage: a row of the wrong length reads every feature
    from the wrong column, so both sides refuse it. The JS half is asserted
    in ``tests/test_parity.py::test_js_rejects_a_wrong_length_row``.
    """
    model = committed_model("heart-disease")
    n = len(model["features"])
    with pytest.raises(ValueError, match=f"expected {n} feature values"):
        predict(model, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match=f"expected {n} feature values"):
        predict(model, [0.0] * (n + 1))


def test_export_refuses_a_feature_order_mismatch(monkeypatch, tmp_path):
    """A silently reordered encoding would make every feature index wrong."""
    import nightingale.export as export_module

    real = export_module.encoded_frame

    def shuffled(slug: str):
        frame = real(slug)
        return frame[list(frame.columns)[::-1]]

    monkeypatch.setattr(export_module, "encoded_frame", shuffled)
    with pytest.raises(ValueError, match="encoded column order does not match"):
        export_module.export_model(FAST_SLUG, out_dir=tmp_path)


# --------------------------------------------------------------------------
# 9. walker.js exists and is dependency-free
# --------------------------------------------------------------------------


def test_walker_js_is_dependency_free_and_carries_the_licence_header():
    source = (Path(__file__).resolve().parents[1] / "docs" / "assets" / "walker.js").read_text()
    assert source.startswith("// Nightingale")
    assert "SPDX-License-Identifier: MIT" in source
    # The only require() calls allowed are node's own built-ins, guarded by
    # the CLI block; nothing from npm.
    requires = set(re.findall(r"require\(['\"]([^'\"]+)['\"]\)", source))
    assert requires <= {"fs"}
