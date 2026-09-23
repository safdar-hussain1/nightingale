# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Model export to a dependency-free JSON bundle, plus the reference walker.

``export_model(slug)`` writes ``models/<slug>/model.json``: everything a
browser needs to reproduce this project's published risk numbers with no
runtime, no server, and no libraries -- the XGBoost forest as flat node
arrays, the deployment calibrator's parameters, the conformal threshold,
the feature contract, and a canary block that lets any reader prove the
file has not drifted.

This module also contains the REFERENCE WALKER (:func:`predict`) that
``docs/assets/walker.js`` mirrors statement for statement. Both are
deliberately written in the same shape -- same accumulation order, same
branch order, same numerically-stable sigmoid -- because
``tests/test_parity.py`` asserts the two agree to 1e-9 on 200 seeded cases
per condition, and the cheapest way to hold a cross-language invariant is
to make the two implementations trivially diffable by eye.

Three numerical decisions are load-bearing; each one was measured, not
assumed.

**1. Split comparisons are made in float32.** XGBoost stores feature
values and split thresholds as ``float`` (32-bit) and compares them there:
a row goes left iff ``float32(x) < float32(threshold)``. Comparing a
float64 input against the same threshold gives a DIFFERENT branch whenever
the input sits within one float32 ulp of the split point -- and with
``tree_method="hist"`` split points ARE observed data values, so real rows
land exactly on them routinely. Measured on a 3-tree toy fit: 2 of 200
rows took the wrong branch under float64 comparison, moving their margin
by up to 0.29 (a ~7-percentage-point probability error). Both walkers
therefore round the input to float32 before comparing (``numpy.float32``
in Python, ``Math.fround`` in JavaScript, which is the same IEEE-754
single-precision rounding), and thresholds are serialised as the exact
float64 value of XGBoost's float32 threshold, so no rounding is needed on
the stored side.

**2. Everything downstream of the comparison is float64, in a fixed
order.** Leaf values are summed onto ``base_score`` in tree order, in
double precision, in both languages -- so the two walkers agree bit for
bit up to the one ``exp`` call, where V8's and libm's last-ulp behaviour
can differ. That is a ~1e-16 disagreement; the 1e-9 parity gate has eight
orders of magnitude of headroom.

**3. Agreement with ``predict_proba`` is float32-limited, and 1e-9 is not
reachable there.** ``XGBClassifier.predict_proba`` returns a float32
array and accumulates the margin in float32 internally, so a float64
re-walk cannot match it more closely than single precision allows: the
error floor is ~6e-8 from the output cast alone, before any accumulation
error. Measured max abs difference on a 300-tree depth-4 fit: 1.9e-7.
:data:`XGB_NATIVE_TOLERANCE` is therefore 1e-6 -- comfortably above the
measured floor, and still ~5 orders of magnitude tighter than any
difference that could change a displayed risk percentage. The 1e-9 gate
lives where it is actually meaningful: between the two float64 walkers
(:mod:`tests.test_parity`), and at 1e-12 for canary self-consistency.

The exported calibrator is ``TrainResult.calibrator`` -- the DEPLOYMENT
calibrator, fit on the pooled OOF ``p_raw``. It is deliberately NOT the
cross-fitted ``oof["p_cal"]`` machinery, which exists to produce honest
published metrics and has no single fitted object to ship. See
:mod:`nightingale.model`'s docstring for why those are two different
artifacts. ``conformal.q_hat``, by contrast, IS computed from the
cross-fitted ``oof["p_cal"]`` column (cross-conformal over OOF -- see
:mod:`nightingale.conformal`'s module docstring), because that is the only
column whose values are out-of-sample with respect to the calibrator that
produced them.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nightingale.clean import CLEANED_ROOT
from nightingale.conditions import CONDITIONS
from nightingale.conformal import conformal_qhat, prediction_set
from nightingale.model import TrainResult, _feature_columns, train_condition

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "models"

SCHEMA_VERSION = 1
CONFORMAL_ALPHA = 0.1
AUTHOR = "Safdar Hussain"

# Canary block: 8 deterministic inputs per condition, drawn from this
# seed, whose p_cal is computed by this module's own walker and stored in
# the file. Any later edit to the trees, the calibrator, q_hat, or the
# feature order moves at least one of these numbers, so a reader can prove
# the artifact is internally consistent without retraining anything.
CANARY_SEED = 4242
N_CANARIES = 8
# Canaries 7 and 8 (0-indexed 6 and 7) carry missing values, so the NaN
# routing path is exercised by the stored block itself and not only by the
# test suite.
CANARY_MISSING_COUNTS = {6: 3, 7: 4}

# Self-consistency: re-walking a stored canary input against the stored
# model must reproduce the stored p_cal to here. Both sides are float64
# doing identical arithmetic, so the only slack is ulp-level.
CANARY_TOLERANCE = 1e-12
# Cross-language: Python walker vs docs/assets/walker.js, float64 both
# sides, differing only in libm-vs-V8 `exp`.
PARITY_TOLERANCE = 1e-9
# Cross-implementation: this walker vs XGBoost's own float32 predict_proba.
# See the module docstring -- this floor is set by single precision, not by
# any defect in the export.
XGB_NATIVE_TOLERANCE = 1e-6

# Every model.json must gzip below this. The accuracy-per-KB gate: a
# browser-side model that costs a megabyte to download is not a
# browser-side model, it is a download.
GZIP_BUDGET_BYTES = 300 * 1024

_LEAF = -1  # `f` sentinel marking a leaf node in the flat node schema

# Cache of fitted models, keyed by (slug, seed). scripts/train_all.py
# persists only metrics.json and oof_predictions.csv.gz -- no fitted
# estimator -- so an export has to retrain. Training is seeded and
# reproducible (see the reproducibility note in scripts/train_all.py), so
# caching per process is a pure speed-up with no effect on what is written.
_TRAIN_CACHE: dict[tuple[str, int], TrainResult] = {}


def train_result(slug: str, seed: int = 42) -> TrainResult:
    """``train_condition(slug, seed)``, memoised for the life of the process.

    ``scripts/train_all.py`` writes metrics and OOF predictions to
    ``models/<slug>/`` but not the fitted estimator, so there is nothing on
    disk to load: the export genuinely has to refit. ``train_condition`` is
    deterministic given the seed, so memoising it changes only wall-clock --
    a test module that exports the same condition four times pays for one
    fit.
    """
    key = (slug, seed)
    if key not in _TRAIN_CACHE:
        _TRAIN_CACHE[key] = train_condition(slug, seed=seed)
    return _TRAIN_CACHE[key]


# --------------------------------------------------------------------------
# Booster -> flat node arrays
# --------------------------------------------------------------------------


def parse_base_score(booster) -> float:
    """XGBoost's ``base_score`` for this booster, as a probability.

    Read from ``booster.save_config()``, where xgboost >= 2.0 serialises it
    as a bracketed vector string (``"[5.5326086E-1]"``) because the field is
    per-target. The brackets are stripped and the single element parsed; a
    multi-target booster (never produced by this project's binary
    classifiers) is rejected loudly rather than silently taking element 0.

    Eight significant figures is exactly enough to round-trip the float32
    XGBoost holds internally, so no precision is lost going through the
    string.
    """
    config = json.loads(booster.save_config())
    raw = config["learner"]["learner_model_param"]["base_score"]
    if isinstance(raw, str) and raw.startswith("["):
        parts = raw.strip("[]").split(",")
        if len(parts) != 1:
            raise ValueError(
                f"expected a single-target base_score, got {len(parts)} targets: {raw!r}"
            )
        raw = parts[0]
    return _as_float32(raw)


def base_margin(base_score: float) -> float:
    """The margin offset ``binary:logistic`` adds before the sigmoid.

    XGBoost's ``base_score`` is a PROBABILITY; the value actually added to
    the summed leaf values is its logit. The walkers add this precomputed
    float64 rather than calling ``log`` themselves, so Python and
    JavaScript cannot disagree by a ulp on the offset.
    """
    if not 0.0 < base_score < 1.0:
        raise ValueError(f"base_score must be in (0, 1) for binary:logistic, got {base_score}")
    return math.log(base_score / (1.0 - base_score))


def _as_float32(value) -> float:
    """The float64 that exactly equals ``value`` rounded to IEEE-754 single.

    Everything XGBoost stores in a tree is a float32; ``get_dump`` renders
    it as decimal text and ``json`` parses that text back to float64, which
    lands *near* the float32 rather than *on* it. Rounding back through
    float32 recovers the exact stored value, and because every float32 is
    representable as a float64 the result serialises to JSON and reloads
    with no further loss -- so neither walker ever has to round a stored
    number.
    """
    return float(np.float32(value))


def _feature_index(split, feature_names: list[str]) -> int:
    """Resolve a dump node's ``split`` field to an index into ``feature_names``.

    ``get_dump`` emits the real column name when the booster was fit from a
    DataFrame (``"cp"``, ``"chol"``), and a positional ``"f12"`` when it was
    fit from a bare array. Name lookup is tried FIRST so that a dataset with
    a column genuinely called ``f3`` resolves to that column rather than to
    position 3.
    """
    if isinstance(split, str):
        by_name = {name: i for i, name in enumerate(feature_names)}
        if split in by_name:
            return by_name[split]
        match = re.fullmatch(r"f(\d+)", split)
        if match:
            index = int(match.group(1))
            if index >= len(feature_names):
                raise ValueError(
                    f"dump references feature index {index} but the model has "
                    f"{len(feature_names)} feature(s)"
                )
            return index
        raise ValueError(f"cannot resolve dump split {split!r} to a feature index")
    return int(split)


def _flatten_tree(root: dict, feature_names: list[str]) -> list[dict]:
    """One dump tree -> the flat node list the walkers index into.

    Nodes are emitted depth-first from the root and referenced by POSITION
    in the returned list, not by the dump's own ``nodeid``: the two happen
    to coincide for a complete tree, but xgboost makes no such promise for
    a pruned one, and a walker that trusted ``nodeid`` as an index would
    fail silently (wrong leaf, plausible probability) rather than loudly.

    Internal node: ``f`` = feature index, ``t`` = split threshold, ``l`` /
    ``r`` = positions taken when ``float32(x) < t`` / otherwise, ``m`` =
    position taken when the feature is missing, ``v`` = 0. Leaf: ``f`` =
    -1, ``v`` = leaf value, all positions -1.

    Thresholds and leaf values go through :func:`_as_float32` before being
    stored, and that is not cosmetic. ``get_dump`` prints a float32 to nine
    significant decimal digits: enough to round-trip, but the float64 you
    get by parsing that decimal is NOT the float32 -- it sits a few 1e-12
    away. Compare ``float32(x)`` against that slightly-off float64 and every
    row whose value equals the split point exactly (routine under
    ``tree_method="hist"``, where split points ARE observed values) falls to
    the LEFT where XGBoost sends it right. Measured cost of skipping this
    step on a 3-tree fit: one row in 200 off by 0.39 in probability, with
    the other 199 agreeing to 1e-8 -- the exact profile of a bug that
    survives a spot check.
    """
    nodes: list[dict] = []

    def emit(node: dict) -> int:
        position = len(nodes)
        if "leaf" in node:
            nodes.append(
                {"f": _LEAF, "t": 0.0, "l": -1, "r": -1, "m": -1, "v": _as_float32(node["leaf"])}
            )
            return position

        record = {
            "f": _feature_index(node["split"], feature_names),
            "t": _as_float32(node["split_condition"]),
            "l": -1,
            "r": -1,
            "m": -1,
            "v": 0.0,
        }
        nodes.append(record)

        children = {child["nodeid"]: child for child in node["children"]}
        record["l"] = emit(children[node["yes"]])
        record["r"] = emit(children[node["no"]])
        # `missing` always points at one of `yes`/`no`, both already
        # emitted, so this is a lookup rather than a third recursion.
        record["m"] = record["l"] if node["missing"] == node["yes"] else record["r"]
        return position

    emit(root)
    return nodes


def export_trees(booster, feature_names: list[str]) -> list[list[dict]]:
    """Every tree in the booster, flattened, in boosting order.

    ``feature_types`` is forced to all-``"q"`` (quantitative) for the
    duration of the dump, then restored. Fitting on a DataFrame with
    one-hot ``bool`` columns makes XGBoost type those features ``"i"``
    (indicator), and its JSON dumper then omits BOTH ``split_condition``
    and ``missing`` from indicator nodes -- the threshold and the missing
    direction are meant to be inferred from "it's an indicator". Under
    ``"q"`` the same tree dumps with every field present and identical
    values (verified against ``booster.save_raw("json")``'s own
    ``split_conditions``/``default_left`` arrays), so the flattening below
    stays a single uniform path rather than a uniform path plus an
    easily-wrong special case. kidney-disease is the condition that has
    these columns; it would have exported silently-wrong trees otherwise.
    """
    original_types = booster.feature_types
    try:
        if original_types is not None:
            booster.feature_types = ["q"] * len(original_types)
        dumped_trees = booster.get_dump(dump_format="json")
    finally:
        booster.feature_types = original_types
    return [_flatten_tree(json.loads(dumped), feature_names) for dumped in dumped_trees]


# --------------------------------------------------------------------------
# The reference walker -- docs/assets/walker.js mirrors this exactly
# --------------------------------------------------------------------------


def _is_missing(value) -> bool:
    """``None`` (JSON ``null``) and NaN both mean "this feature was not measured".

    ``value != value`` rather than ``math.isnan`` so the check is
    type-agnostic (Python float, numpy scalar, int) and reads the same as
    walker.js's ``value !== value``.
    """
    return value is None or value != value


def _walk_tree(nodes: list[dict], x) -> float:
    """The leaf value one flat tree assigns to ``x``.

    ``numpy.float32(value)`` reproduces XGBoost's own float32 comparison
    (``Math.fround`` in walker.js does the same rounding). See the module
    docstring: dropping this makes rows that sit exactly on a split point
    take the wrong branch.
    """
    index = 0
    while nodes[index]["f"] != _LEAF:
        node = nodes[index]
        value = x[node["f"]]
        if _is_missing(value):
            index = node["m"]
        elif float(np.float32(value)) < node["t"]:
            index = node["l"]
        else:
            index = node["r"]
    return nodes[index]["v"]


def _sigmoid(z: float) -> float:
    """``1 / (1 + exp(-z))``, in the branch that cannot overflow.

    For large positive ``z``, ``exp(-z)`` underflows to 0 (harmless); for
    large negative ``z``, ``exp(-z)`` would overflow to inf, so the
    algebraically identical ``exp(z) / (1 + exp(z))`` is used instead.
    walker.js branches the same way, so the two agree bit for bit apart
    from ``exp`` itself.
    """
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def apply_calibrator(calibrator: dict, p: float) -> float:
    """Apply an exported calibrator block to a raw probability.

    ``sigmoid``: ``1 / (1 + exp(a * p + b))`` -- applied to the probability
    directly, not its logit, matching
    :class:`nightingale.calibrate.SigmoidCalibrator` exactly.

    ``isotonic``: linear interpolation between ascending ``(x, y)``
    breakpoints, clamped to the end values outside ``[x[0], x[-1]]`` --
    ``numpy.interp``'s semantics, which is what
    :class:`nightingale.calibrate.IsotonicCalibrator` uses. The
    ``slope * (p - x0) + y0`` form (rather than the algebraically identical
    ``y0 + (y1 - y0) * (p - x0) / (x1 - x0)``) is numpy's own, so this
    function and the fitted calibrator agree to the last ulp.
    """
    kind = calibrator["type"]
    if kind == "sigmoid":
        return _sigmoid(-(calibrator["a"] * p + calibrator["b"]))
    if kind == "isotonic":
        xs, ys = calibrator["x"], calibrator["y"]
        if p <= xs[0]:
            return ys[0]
        if p >= xs[-1]:
            return ys[-1]
        lo, hi = 0, len(xs) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if xs[mid] <= p:
                lo = mid
            else:
                hi = mid
        if xs[lo + 1] == xs[lo]:
            return ys[lo + 1]
        slope = (ys[lo + 1] - ys[lo]) / (xs[lo + 1] - xs[lo])
        return slope * (p - xs[lo]) + ys[lo]
    raise ValueError(f"unknown calibrator type {kind!r}")


def predict(model: dict, x) -> dict:
    """Score one row against an exported model: ``{p_raw, p_cal, set}``.

    ``x`` is a sequence positionally aligned to ``model["features"]``, with
    ``None``/NaN for anything not measured. Leaf values are summed onto
    ``model["base_score"]`` in tree order -- walker.js accumulates in the
    same order, so the running float64 sum is bit-identical between the
    two.

    ``set`` is the conformal verdict from
    :func:`nightingale.conformal.prediction_set`, reused rather than
    reimplemented so the exported artifact and the published metrics can
    never disagree about what "uncertain" means.

    **Deliberate asymmetry with walker.js.** This function RAISES on a row
    of the wrong length, and lets a non-numeric value raise from
    ``numpy.float32``. walker.js instead coerces anything unusable (``""``,
    ``"abc"``, ``Infinity``) to NaN and routes it as missing. That is not
    an inconsistency, it is the two call sites having different jobs: this
    walker runs inside a pipeline, where a malformed row means a bug
    upstream and silently guessing would hide it; walker.js runs behind a
    form, where a blank field is a normal thing for a person to do and
    refusing to score would be useless. Neither ever invents a value --
    JavaScript's ``Number('') === 0`` is precisely what walker.js's
    ``toFeatureValue`` exists to stop. Structural damage (wrong row length)
    throws on both sides.
    """
    expected = len(model["features"])
    if len(x) != expected:
        raise ValueError(
            f"predict: expected {expected} feature values, got {len(x)} -- a row of "
            f"the wrong length would read every feature from the wrong column"
        )
    total = model["base_score"]
    for tree in model["trees"]:
        total += _walk_tree(tree, x)
    p_raw = _sigmoid(total)
    p_cal = apply_calibrator(model["calibrator"], p_raw)
    return {
        "p_raw": p_raw,
        "p_cal": p_cal,
        "set": prediction_set(p_cal, model["conformal"]["q_hat"]),
    }


# --------------------------------------------------------------------------
# The feature contract
# --------------------------------------------------------------------------

# Units for columns where the source datasets document one. Empty string
# where the column is a code, a count, an index, or a unitless ratio --
# inventing a unit for those would be worse than admitting there isn't one.
_UNITS: dict[str, str] = {
    "age": "years",
    "Age": "years",
    "trestbps": "mm Hg",
    "chol": "mg/dL",
    "thalach": "bpm",
    "oldpeak": "mm ST",
    "bp": "mm Hg",
    "bgr": "mg/dL",
    "bu": "mg/dL",
    "sc": "mg/dL",
    "sod": "mEq/L",
    "pot": "mEq/L",
    "hemo": "g/dL",
    "pcv": "%",
    "wbcc": "cells/cmm",
    "rbcc": "millions/cmm",
    "BMI": "kg/m^2",
    "MentHlth": "days",
    "PhysHlth": "days",
    "TB": "mg/dL",
    "DB": "mg/dL",
    "Alkphos": "IU/L",
    "Sgpt": "IU/L",
    "Sgot": "IU/L",
    "TP": "g/dL",
    "ALB": "g/dL",
    "Smokes (years)": "years",
    "Hormonal Contraceptives (years)": "years",
    "IUD (years)": "years",
    "First sexual intercourse": "years",
}

# Human labels for the clinical abbreviations. Anything not listed falls
# back to :func:`_humanise`, which is fine for columns that are already
# words ("Smoker", "Number of sexual partners").
_LABELS: dict[str, str] = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Serum cholesterol",
    "fbs": "Fasting blood sugar > 120 mg/dL",
    "restecg": "Resting ECG result",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (exercise vs rest)",
    "slope": "Slope of peak exercise ST segment",
    "ca": "Major vessels coloured by fluoroscopy",
    "thal": "Thalassemia test result",
    "bp": "Blood pressure",
    "sg": "Urine specific gravity",
    "al": "Albumin (urine)",
    "su": "Sugar (urine)",
    "rbc": "Red blood cells (urine)",
    "pc": "Pus cells",
    "pcc": "Pus cell clumps",
    "ba": "Bacteria",
    "bgr": "Blood glucose (random)",
    "bu": "Blood urea",
    "sc": "Serum creatinine",
    "sod": "Sodium",
    "pot": "Potassium",
    "hemo": "Haemoglobin",
    "pcv": "Packed cell volume",
    "wbcc": "White blood cell count",
    "rbcc": "Red blood cell count",
    "htn": "Hypertension",
    "dm": "Diabetes mellitus",
    "cad": "Coronary artery disease",
    "appet": "Appetite",
    "pe": "Pedal oedema",
    "ane": "Anaemia",
    "TB": "Total bilirubin",
    "DB": "Direct bilirubin",
    "Alkphos": "Alkaline phosphatase",
    "Sgpt": "Alanine aminotransferase (SGPT)",
    "Sgot": "Aspartate aminotransferase (SGOT)",
    "TP": "Total proteins",
    "ALB": "Albumin",
    "A/G Ratio": "Albumin / globulin ratio",
    "sex_male": "Sex is male",
    "BMI": "Body mass index",
    "HighBP": "High blood pressure",
    "HighChol": "High cholesterol",
    "CholCheck": "Cholesterol checked in last 5 years",
    "HeartDiseaseorAttack": "Coronary heart disease or myocardial infarction",
    "PhysActivity": "Physical activity in past 30 days",
    "HvyAlcoholConsump": "Heavy alcohol consumption",
    "AnyHealthcare": "Has any health coverage",
    "NoDocbcCost": "Could not see a doctor because of cost",
    "GenHlth": "Self-rated general health (1 best - 5 worst)",
    "MentHlth": "Poor mental health days (last 30)",
    "PhysHlth": "Poor physical health days (last 30)",
    "DiffWalk": "Serious difficulty walking or climbing stairs",
    "Dx": "Any prior diagnosis",
    "Dx:Cancer": "Prior cancer diagnosis",
    "Dx:CIN": "Prior CIN diagnosis",
    "Dx:HPV": "Prior HPV diagnosis",
}

# Breast-cancer's columns are <measurement><1|2|3>, where the suffix is the
# statistic taken over the nuclei in the image (UCI dataset 17's own
# ordering: mean, standard error, worst).
_WDBC_SUFFIX = {"1": "mean", "2": "std. error", "3": "worst"}
_WDBC_BASE = {
    "radius": "radius",
    "texture": "texture",
    "perimeter": "perimeter",
    "area": "area",
    "smoothness": "smoothness",
    "compactness": "compactness",
    "concavity": "concavity",
    "concave_points": "concave points",
    "symmetry": "symmetry",
    "fractal_dimension": "fractal dimension",
}


def _humanise(name: str) -> str:
    """Fallback label: a readable sentence-case rendering of a column name."""
    match = re.fullmatch(r"([a-z_]+)([123])", name)
    if match and match.group(1) in _WDBC_BASE:
        base = _WDBC_BASE[match.group(1)]
        return f"Nucleus {base} ({_WDBC_SUFFIX[match.group(2)]})"
    text = name.replace("_", " ").strip()
    return text[:1].upper() + text[1:] if text else name


def _label_for(name: str) -> str:
    return _LABELS.get(name, _humanise(name))


def build_features(encoded: pd.DataFrame, feature_names: list[str]) -> list[dict]:
    """The feature contract: one entry per POST-ENCODING model input, in model order.

    ``encoded`` is the one-hot-encoded frame the final model was fit on, so
    ``feature_names`` is exactly its column order and the ranges below are
    the ranges the model was actually trained over.

    ``min``/``max`` are the cleaned dataset's OBSERVED extremes -- real
    floats, not quantiles. They exist so a UI can build a sensible slider
    and warn on out-of-range input; they are NOT a validation rule, and the
    walker never consults them.

    A one-hot dummy column (bool dtype out of ``pandas.get_dummies``) is
    typed ``"binary"`` with range [0, 1]; everything else is ``"number"``.
    ``allow_missing`` is ``true`` for every feature without exception:
    XGBoost learned an explicit missing direction at every split, so "not
    measured" is a first-class input here rather than something to impute
    around.
    """
    features = []
    for name in feature_names:
        column = encoded[name]
        is_dummy = column.dtype == bool
        if is_dummy:
            low, high = 0.0, 1.0
        else:
            numeric = pd.to_numeric(column, errors="coerce")
            low = float(numeric.min())
            high = float(numeric.max())
        features.append(
            {
                "name": name,
                "label": _dummy_label(name) if is_dummy else _label_for(name),
                "type": "binary" if is_dummy else "number",
                "min": low,
                "max": high,
                "unit": "" if is_dummy else _UNITS.get(name, ""),
                "allow_missing": True,
            }
        )
    return features


def _dummy_label(name: str) -> str:
    """Humanise a ``pandas.get_dummies`` column, e.g. ``cp_typical`` -> "Chest pain type: typical".

    Splits on the LAST underscore, since the source column may itself
    contain one (``concave_points``); if the prefix isn't a column this
    project knows, the whole name is humanised instead.
    """
    if "_" in name:
        source, _, category = name.rpartition("_")
        if source in _LABELS:
            return f"{_LABELS[source]}: {category}"
    return _humanise(name)


# --------------------------------------------------------------------------
# Canaries
# --------------------------------------------------------------------------


def build_canaries(model: dict, seed: int = CANARY_SEED) -> dict:
    """Eight deterministic inputs and the p_cal this module's walker gives them.

    Numeric features are drawn uniformly from their own ``[min, max]``;
    binary features from Bernoulli(0.5). Canaries 7 and 8 additionally have
    3 and 4 numeric features blanked to ``null``, so the missing-value path
    is covered by the stored block itself.

    ``p_cal`` is computed by :func:`predict` -- the exporter's own walker --
    NOT by ``predict_proba``. That is the point: the canary block certifies
    that the SERIALISED artifact reproduces these numbers, which is the
    property a browser depends on. Agreement between the walker and XGBoost
    is a separate claim, asserted separately in ``tests/test_export.py``.

    The stored ``tolerance`` is :data:`PARITY_TOLERANCE` (1e-9): the bar an
    INDEPENDENT re-implementation should clear, and the number a third
    party checking this file should use. The test suite holds the two
    implementations in this repo to :data:`CANARY_TOLERANCE` (1e-12), which
    is strictly tighter -- publishing the looser figure is the honest one,
    since 1e-12 is a property of two walkers written to match, not a
    guarantee anyone else's code can be held to.
    """
    rng = np.random.default_rng(seed)
    features = model["features"]
    numeric_positions = [i for i, f in enumerate(features) if f["type"] == "number"]

    inputs: list[list[float | None]] = []
    for _ in range(N_CANARIES):
        row: list[float | None] = []
        for feature in features:
            if feature["type"] == "binary":
                row.append(float(rng.random() < 0.5))
            else:
                row.append(float(rng.uniform(feature["min"], feature["max"])))
        inputs.append(row)

    for index, n_missing in CANARY_MISSING_COUNTS.items():
        if index >= len(inputs):
            continue
        take = min(n_missing, len(numeric_positions))
        if take == 0:  # pragma: no cover - no condition is all-binary
            continue
        for position in rng.choice(numeric_positions, size=take, replace=False):
            inputs[index][int(position)] = None

    return {
        "inputs": inputs,
        "p_cal": [predict(model, row)["p_cal"] for row in inputs],
        "tolerance": PARITY_TOLERANCE,
    }


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def _git(*args: str) -> str | None:
    """Run a read-only git command in the repo; ``None`` if git or the repo is absent."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git installed
        return None
    if result.returncode != 0:
        return None  # pragma: no cover - not a git checkout
    return result.stdout.strip() or None


def build_provenance() -> dict:
    """``built_utc`` / ``commit`` / ``author`` for the exported bundle.

    ``built_utc`` is the HEAD commit's committer timestamp, NOT the wall
    clock. The signed provenance manifest (``nightingale sign``) covers
    every ``model.json`` and ``metrics.json``, and ``scripts/train_all.py``
    already established that a signed artifact has to reproduce byte for
    byte on an honest rerun or ``nightingale verify`` reports a false
    TAMPERED -- which is why ``metrics.json`` carries no timestamp at all.
    ``model.json``'s schema requires one, so it gets the one timestamp that
    is a property of the source rather than of the run: re-exporting at the
    same commit produces the same bytes, and the field still answers "how
    old is this model" honestly.

    The recorded commit is therefore the commit the model was built FROM,
    which is the parent of the commit that lands the model.json itself.
    That is the useful direction: it names the source state that produced
    these trees.
    """
    commit = _git("rev-parse", "HEAD")
    committed_at = _git("show", "-s", "--format=%cI", "HEAD")
    if committed_at:
        built_utc = (
            datetime.fromisoformat(committed_at)
            .astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    else:  # pragma: no cover - only outside a git checkout
        built_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {"built_utc": built_utc, "commit": commit or "unknown", "author": AUTHOR}


# --------------------------------------------------------------------------
# The export itself
# --------------------------------------------------------------------------


def build_model_dict(result: TrainResult, slug: str, encoded: pd.DataFrame) -> dict:
    """Assemble the full ``model.json`` payload (canaries included) for one condition."""
    booster = result.final_model.get_booster()
    base_score = parse_base_score(booster)

    model = {
        "schema_version": SCHEMA_VERSION,
        "condition": slug,
        "trees": export_trees(booster, result.feature_names),
        # The MARGIN offset the walkers add, per the walker contract
        # ("sum leaf values + base_score -> sigmoid"). XGBoost's own
        # base_score is a probability and is kept alongside for
        # provenance; storing the logit precomputed keeps Python and
        # JavaScript from disagreeing by a ulp on `log`.
        "base_score": base_margin(base_score),
        "base_score_probability": base_score,
        "calibrator": result.calibrator.export(),
        # Cross-conformal over OOF: q_hat comes from the CROSS-FITTED
        # p_cal column, the only one whose values are out-of-sample with
        # respect to the calibrator that produced them. See
        # nightingale/conformal.py's module docstring.
        "conformal": {
            "q_hat": conformal_qhat(
                result.oof["y_true"].to_numpy(),
                result.oof["p_cal"].to_numpy(),
                alpha=CONFORMAL_ALPHA,
            ),
            "alpha": CONFORMAL_ALPHA,
        },
        "features": build_features(encoded, result.feature_names),
        "provenance": build_provenance(),
    }
    model["canaries"] = build_canaries(model)
    return model


def encoded_frame(slug: str) -> pd.DataFrame:
    """The one-hot-encoded feature frame ``train_condition`` fits its final model on.

    Mirrors :func:`nightingale.model.train_condition`'s final-refit encoding
    exactly: drop the target and ``site``, then ``pandas.get_dummies``.
    :func:`export_model` checks the resulting column order against
    ``TrainResult.feature_names`` and refuses to write anything if they
    differ, rather than trusting the mirror to stay in step -- a silently
    reordered encoding would make every feature index in every tree point
    at the wrong column.
    """
    df = pd.read_csv(CLEANED_ROOT / f"{slug}.csv.gz", compression="gzip")
    target = CONDITIONS[slug].target
    return pd.get_dummies(df[_feature_columns(df, target)])


def export_model(slug: str, out_dir: Path | None = None, seed: int = 42) -> Path:
    """Train (or reuse) ``slug``'s model and write ``models/<slug>/model.json``.

    Returns the path written. The file is compact JSON (no indentation):
    it is machine-read by walker.js and the dashboard, and every byte is
    downloaded by a browser.
    """
    result = train_result(slug, seed=seed)
    encoded = encoded_frame(slug)
    if list(encoded.columns) != result.feature_names:
        raise ValueError(
            f"{slug}: encoded column order does not match the fitted model's "
            f"feature order -- refusing to export a bundle whose feature "
            f"indices would be wrong"
        )

    model = build_model_dict(result, slug, encoded)

    destination = (out_dir or MODELS_ROOT) / slug
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "model.json"
    # allow_nan=False: NaN/Infinity are not JSON, and a walker fed
    # `NaN` from a hand-rolled parser would be a silent wrong answer.
    # Missing canary values are serialised as `null` by construction.
    path.write_text(json.dumps(model, separators=(",", ":"), allow_nan=False) + "\n")
    return path


def export_all(out_dir: Path | None = None, seed: int = 42) -> dict[str, Path]:
    """Export every condition in the registry; returns ``{slug: path}``."""
    return {slug: export_model(slug, out_dir=out_dir, seed=seed) for slug in CONDITIONS}


def load_model(path: Path | str) -> dict:
    """Read an exported ``model.json`` back into a dict the walker can score."""
    return json.loads(Path(path).read_text())


if __name__ == "__main__":  # pragma: no cover - operator entry point
    for exported_slug, exported_path in export_all().items():
        print(f"{exported_slug}: {exported_path}")
