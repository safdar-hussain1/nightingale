#!/usr/bin/env python3
# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT

"""Assemble ``docs/index.html`` from the committed, signed artifacts.

The dashboard is one self-contained file: the six trained models, their
metrics, the four-hospital transfer study, the inference engine, two real
example records per condition and the page's own typefaces are all baked
into it at build time, so the page runs every model in the visitor's browser
with no server and no runtime download of any kind.

The shape is deliberate. ``scripts/dashboard_template.html`` holds all the
markup, style and behaviour and exactly one ``/*__DATA__*/`` placeholder;
this script produces the JSON that replaces it. The template writes no
numbers, so a number can only reach the page by being read out of an
artifact that ``nightingale verify`` covers. The words the form needs --
option names for coded fields, the groups the fields are shown in, the
plain-English verdict for each condition -- live in :data:`PRESENTATION`
below, and field labels are read from ``data/DATA_DICTIONARY.md`` so the
page and the documentation cannot describe a column two different ways.

Two properties are load-bearing and are tested in
``tests/test_build_dashboard.py``:

*   **Byte-determinism.** No wall-clock time, no set iteration, no unsorted
    dict; the out-of-fold sample is drawn from a fixed seed with the stdlib
    Mersenne Twister, and the example records are chosen by a rule with a
    fixed tie-break. Two builds of the same artifacts are the same bytes, so
    a diff on ``docs/index.html`` means the models changed.
*   **A byte-identical walker.** ``docs/assets/walker.js`` is inlined
    verbatim rather than adapted, because it is the file
    ``tests/test_parity.py`` scores against the Python reference. An adapted
    copy would look right and predict differently.

Usage::

    PYTHONPATH=src python scripts/build_dashboard.py
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPO_ROOT / "scripts" / "dashboard_template.html"
WALKER_PATH = REPO_ROOT / "docs" / "assets" / "walker.js"
OUTPUT_PATH = REPO_ROOT / "docs" / "index.html"
MODELS_DIR = REPO_ROOT / "models"
PUBKEY_PATH = REPO_ROOT / "provenance" / "pubkey.pem"
CLEANED_DIR = REPO_ROOT / "data" / "cleaned"
DICTIONARY_PATH = REPO_ROOT / "data" / "DATA_DICTIONARY.md"
FONTS_DIR = REPO_ROOT / "scripts" / "fonts"

DATA_PLACEHOLDER = "/*__DATA__*/"
WALKER_PLACEHOLDER = "/*__WALKER__*/"
FONTS_PLACEHOLDER = "/*__FONTS__*/"

# The page's two typefaces, inlined as data URIs so the file still works
# opened straight off a disk and makes no request a visitor did not ask for.
# Both are SIL Open Font License 1.1; the licence text sits beside each file.
FONT_FACES = (
    ("Instrument Serif", "instrument-serif-latin-400-normal.woff2", "400"),
    ("Figtree", "figtree-latin-wght-normal.woff2", "300 900"),
)

# Out-of-fold predictions drive the threshold slider's live confusion counts.
# Diabetes alone has 253,680 of them; shipping all six in full would put tens
# of megabytes on the wire to move a slider. A seeded sample keeps the page
# publishable, and the page states the sample size beside the counts rather
# than presenting them as the whole cohort.
OOF_SAMPLE_CAP = 2000
OOF_SAMPLE_SEED = 20260813

# A feature whose observed range is exactly [0, 1] is a yes/no answer, and a
# spin box asking for "0 to 1" is a worse way to ask it than two buttons.
# This is read off the exported schema, never listed by hand, so a retrain
# that widens a range re-renders that field as a number without a code change.
BINARY_RANGE = (0.0, 1.0)

# How many records the search for a "close call" example may score with the
# reference walker before giving up. The candidates are ordered by how close
# their out-of-fold estimate sits to 0.5, so the first few nearly always
# qualify; the cap only bounds the build time on a cohort where none do.
CLOSE_CALL_SEARCH_CAP = 400

# The finest step a slider offers. A few cohorts carry values with long float
# tails (a fraction of a year written to eight places); a slider stepping at
# that precision is unusable, and typing a value keeps full precision anyway.
MAX_DECIMALS = 4

# Units the data dictionary prints that are not units a person would type a
# value in: codes, counts, flags and ratios describe a field, they do not
# measure it.
NOT_A_UNIT = frozenset({"", "—", "0/1", "code", "grade", "count", "ratio"})

# One-hot category names as a form should say them, in the order a form
# should offer them: the unremarkable answer first.
CATEGORY_WORDS = {
    "no": "No",
    "normal": "Normal",
    "notpresent": "Not present",
    "good": "Good",
    "yes": "Yes",
    "abnormal": "Abnormal",
    "present": "Present",
    "poor": "Poor",
}
CATEGORY_ORDER = tuple(CATEGORY_WORDS)

# --------------------------------------------------------------------------
# What the form says about each condition
#
# ``groups`` decides which fields sit together and in what order; a field the
# groups do not name still appears, under "Other measurements", so a retrain
# that adds a column can never silently drop it from the form. ``choices``
# turns a coded column into named options -- the codes are the dataset's own,
# as documented in data/DATA_DICTIONARY.md. ``labels`` overrides the
# dictionary label only where that label spells out the codes, which the
# options now do instead.
# --------------------------------------------------------------------------

SEX = ((0, "Female"), (1, "Male"))

PRESENTATION: dict[str, dict[str, Any]] = {
    "heart-disease": {
        "order": 1,
        "name": "Heart disease",
        "icon": "heart",
        "noun": "patient records",
        "intro": "Routine measurements from four hospitals, from age and chest pain to an exercise ECG.",
        "positive": "Likely heart disease",
        "negative": "Likely no heart disease",
        "had": "Had heart disease",
        "had_not": "No heart disease",
        "groups": (
            ("The patient", ("age", "sex", "cp", "trestbps", "chol", "fbs", "restecg")),
            ("Exercise test", ("thalach", "exang", "oldpeak", "slope")),
            ("Imaging", ("ca", "thal")),
        ),
        "choices": {
            "sex": SEX,
            "cp": ((1, "Typical angina"), (2, "Atypical angina"), (3, "Non-anginal pain"), (4, "Asymptomatic")),
            "restecg": ((0, "Normal"), (1, "ST-T abnormality"), (2, "LV hypertrophy")),
            "slope": ((1, "Up"), (2, "Flat"), (3, "Down")),
            "ca": ((0, "0"), (1, "1"), (2, "2"), (3, "3")),
            "thal": ((3, "Normal"), (6, "Fixed defect"), (7, "Reversible defect")),
        },
        "labels": {
            "sex": "Sex",
            "cp": "Chest pain type",
            "restecg": "Resting ECG",
            "slope": "Slope of the peak exercise ST segment",
            "thal": "Thalassemia test",
        },
    },
    "diabetes": {
        "order": 2,
        "name": "Diabetes",
        "icon": "drop",
        "noun": "survey responses",
        "intro": "Answers to a national telephone health survey. The outcome is what each person reported, not a lab result.",
        "positive": "Likely diabetes or prediabetes",
        "negative": "Likely no diabetes or prediabetes",
        "had": "Reported diabetes",
        "had_not": "Reported neither",
        "groups": (
            ("Health", ("GenHlth", "HighBP", "HighChol", "CholCheck", "BMI", "Stroke",
                        "HeartDiseaseorAttack", "DiffWalk", "MentHlth", "PhysHlth")),
            ("The person", ("Sex", "Age", "Education", "Income", "AnyHealthcare", "NoDocbcCost")),
            ("Habits", ("Smoker", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump")),
        ),
        "choices": {
            "Sex": SEX,
            "GenHlth": ((1, "Excellent"), (2, "Very good"), (3, "Good"), (4, "Fair"), (5, "Poor")),
            "Age": (
                (1, "18–24"), (2, "25–29"), (3, "30–34"), (4, "35–39"), (5, "40–44"),
                (6, "45–49"), (7, "50–54"), (8, "55–59"), (9, "60–64"), (10, "65–69"),
                (11, "70–74"), (12, "75–79"), (13, "80 or older"),
            ),
            "Education": tuple((level, f"Level {level}") for level in range(1, 7)),
            "Income": tuple((level, f"Bracket {level}") for level in range(1, 9)),
        },
        "labels": {
            "Sex": "Sex",
            "GenHlth": "General health, as they rate it",
            "Age": "Age group",
            "Education": "Education level (1 lowest, 6 highest)",
            "Income": "Income bracket (1 lowest, 8 highest)",
        },
    },
    "kidney-disease": {
        "order": 3,
        "name": "Kidney disease",
        "icon": "kidney",
        "noun": "patient records",
        "intro": "Blood and urine tests with a short medical history.",
        "positive": "Likely chronic kidney disease",
        "negative": "Likely no chronic kidney disease",
        "had": "Had kidney disease",
        "had_not": "No kidney disease",
        "groups": (
            ("The patient", ("age", "bp", "htn", "dm", "cad", "appet", "pe", "ane")),
            ("Blood tests", ("bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc")),
            ("Urine tests", ("sg", "al", "su", "rbc", "pc", "pcc", "ba")),
        ),
        "choices": {
            "sg": ((1.005, "1.005"), (1.01, "1.010"), (1.015, "1.015"), (1.02, "1.020"), (1.025, "1.025")),
            "al": tuple((grade, str(grade)) for grade in range(6)),
            "su": tuple((grade, str(grade)) for grade in range(6)),
        },
        "labels": {},
    },
    "liver-disease": {
        "order": 4,
        "name": "Liver disease",
        "icon": "liver",
        "noun": "patient records",
        "intro": "Age, sex and a panel of liver blood tests. A positive record is a liver-clinic patient.",
        "positive": "Likely liver patient",
        "negative": "Likely not a liver patient",
        "had": "Liver patient",
        "had_not": "Not a liver patient",
        "groups": (
            ("The patient", ("Age", "sex_male")),
            ("Liver blood tests", ("TB", "DB", "Alkphos", "Sgpt", "Sgot", "TP", "ALB", "A/G Ratio")),
        ),
        "choices": {"sex_male": SEX},
        "labels": {"sex_male": "Sex"},
    },
    "breast-cancer": {
        "order": 5,
        "name": "Breast cancer",
        "icon": "ribbon",
        "noun": "biopsy images",
        "intro": "Measurements of cell nuclei from a digitised needle-biopsy image. Load a real record: these are hard to type by hand.",
        "positive": "Likely malignant",
        "negative": "Likely benign",
        "had": "Malignant",
        "had_not": "Benign",
        # The ten nucleus measurements are each reported three ways; the
        # column suffix says which, and the group heading says it in words.
        "suffix_groups": (
            ("1", "Average across the cells", " (mean)"),
            ("2", "Spread across the cells", " (std. error)"),
            ("3", "The three most extreme cells", " (worst)"),
        ),
        "groups": (),
        "choices": {},
        "labels": {},
    },
    "cervical-cancer": {
        "order": 6,
        "name": "Cervical cancer",
        "icon": "cell",
        "noun": "patient records",
        "intro": "History and risk factors. The outcome is a biopsy result, and only a few dozen biopsies in the data were positive.",
        "positive": "Likely positive biopsy",
        "negative": "Likely negative biopsy",
        "had": "Positive biopsy",
        "had_not": "Negative biopsy",
        "groups": (
            ("The patient", ("Age", "Number of sexual partners", "First sexual intercourse", "Num of pregnancies")),
            ("Smoking", ("Smokes", "Smokes (years)", "Smokes (packs/year)")),
            ("Contraception", ("Hormonal Contraceptives", "Hormonal Contraceptives (years)", "IUD", "IUD (years)")),
            ("Sexually transmitted infections", (
                "STDs", "STDs (number)", "STDs:condylomatosis", "STDs:vaginal condylomatosis",
                "STDs:vulvo-perineal condylomatosis", "STDs:syphilis", "STDs:pelvic inflammatory disease",
                "STDs:genital herpes", "STDs:molluscum contagiosum", "STDs:HIV", "STDs:Hepatitis B",
                "STDs:HPV", "STDs: Number of diagnosis",
            )),
            ("Earlier diagnoses", ("Dx:Cancer", "Dx:CIN", "Dx:HPV", "Dx")),
        ),
        "choices": {},
        "labels": {},
    },
}


# --------------------------------------------------------------------------
# Reading artifacts
# --------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any) -> Any:
    """Replace non-finite floats with ``None`` so the payload is valid JSON.

    ``json.dumps`` will happily emit bare ``NaN``, which is not JSON and which
    ``JSON.parse`` rejects — the page would fail to boot with a syntax error
    pointing at a megabyte of data. Turning it into ``null`` here means a
    missing metric renders as an em dash instead.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def pubkey_fingerprint(pem_path: Path = PUBKEY_PATH) -> str:
    """SHA-256 of the raw 32-byte Ed25519 public key inside the PEM.

    Hashing the key material rather than the PEM file means the fingerprint
    survives a re-wrap of the armour (line endings, header comments) and can
    be reproduced with one ``openssl`` pipe, which the page prints beside it.
    """
    body = b"".join(
        line
        for line in pem_path.read_bytes().splitlines()
        if not line.startswith(b"-----")
    )
    der = base64.b64decode(body)
    return hashlib.sha256(der[-32:]).hexdigest()


def sample_oof(slug: str, cap: int = OOF_SAMPLE_CAP) -> dict[str, Any]:
    """A seeded, order-preserving sample of one condition's OOF predictions.

    ``random.Random`` is used rather than numpy because the stdlib Mersenne
    Twister is version-stable by documented guarantee, and this sample has to
    reproduce byte-for-byte on any machine that runs this script.
    """
    import pandas as pd

    frame = pd.read_csv(MODELS_DIR / slug / "oof_predictions.csv.gz")
    total = len(frame)
    if total > cap:
        picks = sorted(random.Random(OOF_SAMPLE_SEED).sample(range(total), cap))
        frame = frame.iloc[picks]
    return {
        "n_total": int(total),
        "n_shown": int(len(frame)),
        "sampled": bool(total > cap),
        "seed": OOF_SAMPLE_SEED,
        "y": [int(value) for value in frame["y_true"]],
        # Six significant figures is finer than any pixel on the reliability
        # plot and roughly halves the payload against full float repr.
        "p": [float(f"{value:.6g}") for value in frame["p_cal"]],
    }


def saturated_fraction(slug: str) -> float:
    """Share of a condition's calibrated OOF probabilities pinned to 0 or 1.

    This is the measurement behind the kidney-disease conformal note: an
    isotonic calibrator on a near-separable cohort maps almost everything to
    an endpoint, the 90th-percentile nonconformity score collapses to zero,
    and the conformal set can never contain both labels. Computed here from
    the committed predictions rather than quoted, so it cannot drift.
    """
    import pandas as pd

    column = pd.read_csv(MODELS_DIR / slug / "oof_predictions.csv.gz")["p_cal"]
    return float(((column == 0.0) | (column == 1.0)).mean())


def dictionary_entries(path: Path = DICTIONARY_PATH) -> dict[str, dict[str, dict[str, str]]]:
    """``{slug: {column: {"label", "unit"}}}`` read from the data dictionary.

    The dictionary is the reviewed, human-facing description of every
    column, so the form takes its labels from there instead of keeping a
    second copy that could drift. Each condition is a ``## <slug>`` section
    whose feature table rows start with a backticked column name.
    """
    entries: dict[str, dict[str, dict[str, str]]] = {}
    current: str | None = None
    row = re.compile(r"^\| `([^`]+)` \| ([^|]+?) \| [^|]+? \| ([^|]*?) \|")
    for line in path.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^## (\S+)\s*$", line)
        if heading:
            current = heading.group(1)
            entries.setdefault(current, {})
            continue
        match = row.match(line)
        if match and current is not None:
            entries[current][match.group(1)] = {
                "label": match.group(2).strip(),
                "unit": match.group(3).strip(),
            }
    return entries


# --------------------------------------------------------------------------
# The form: one question per thing a person would be asked
# --------------------------------------------------------------------------


def _feature_schema(feature: dict[str, Any]) -> dict[str, Any]:
    """One model input, described entirely by the exported schema."""
    low = float(feature["min"])
    high = float(feature["max"])
    return {
        "name": feature["name"],
        "label": feature["label"],
        "unit": feature.get("unit") or "",
        "min": low,
        "max": high,
        # A range of exactly [0, 1] is a yes/no; a range of zero width is a
        # column that never varied in the cohort and can only ever be told
        # the one value it took.
        "binary": (low, high) == BINARY_RANGE,
        "constant": low == high,
    }


def _decimals(values: Any) -> int:
    """The fewest decimal places that write every observed value exactly.

    The form's sliders step at this precision, so a field recorded in whole
    years moves a year at a time and a ratio recorded to two places moves by
    hundredths, without a hand-kept list of which column is which.
    """
    finite = [float(value) for value in values if value == value]
    for places in range(MAX_DECIMALS + 1):
        scale = 10.0**places
        if all(abs(value * scale - round(value * scale)) < 1e-6 for value in finite):
            return places
    return MAX_DECIMALS


def _dummy_source(name: str) -> tuple[str, str]:
    """``("rbc", "abnormal")`` for the one-hot column ``rbc_abnormal``."""
    source, _, category = name.rpartition("_")
    return source, category


def _typical(question: dict[str, Any], cleaned: Any, features: list[dict[str, Any]]) -> float | int | None:
    """The answer an ordinary record gives: a median, or the commonest answer.

    Blanks are left out of both, so "typical" is always an answer someone
    actually gave. Ties between equally common answers go to the smallest
    value, which ``Series.mode`` already returns first.
    """
    if question["kind"] == "onehot":
        source = _dummy_source(features[question["features"][0]]["name"])[0]
        category = str(cleaned[source].dropna().mode().iloc[0])
        names = {features[index]["name"]: index for index in question["features"]}
        return names[f"{source}_{category}"]
    column = cleaned[features[question["features"][0]]["name"]].dropna()
    if question["kind"] == "number":
        return float(column.median())
    return float(column.mode().iloc[0])


def build_questions(slug: str, model: dict[str, Any]) -> dict[str, Any]:
    """Turn a model's flat feature list into the questions the form asks.

    Four kinds, each answered differently but always mapped back onto the
    exact feature vector the model was trained on:

    ``number``  one numeric feature; blank means not measured (``null``).
    ``yesno``   one 0/1 feature; blank means not measured (``null``).
    ``choice``  one coded feature, offered by name; blank is ``null``.
    ``onehot``  several one-hot columns that are really one categorical
                answer. Picking a category sets its column to 1 and the
                others to 0; leaving it unanswered sets every column to 0,
                which is exactly how ``pandas.get_dummies`` encoded a missing
                category in training -- so "not recorded" here means what it
                meant to the model.

    Features with a single observed value carry no information and cannot be
    split on; they are listed as omitted instead of being asked.

    Every question also carries ``typical``: the median of a number, or the
    most common recorded answer otherwise. The page explains a result by
    swapping one answer at a time for this value -- "what if this patient
    were ordinary here?" -- which keeps meaning even where the calibrated
    risk is pinned at 0 or 1, unlike swapping it for a blank.
    """
    import pandas as pd

    presentation = PRESENTATION[slug]
    dictionary = dictionary_entries().get(slug, {})
    cleaned = pd.read_csv(CLEANED_DIR / f"{slug}.csv.gz")
    features = model["features"]
    index_of = {feature["name"]: position for position, feature in enumerate(features)}

    def describe(name: str) -> tuple[str, str]:
        entry = dictionary.get(name, {})
        label = presentation["labels"].get(name) or entry.get("label") or name
        unit = entry.get("unit", "")
        return label, ("" if unit in NOT_A_UNIT else unit)

    questions: dict[str, dict[str, Any]] = {}
    omitted: list[str] = []
    for position, feature in enumerate(features):
        name = feature["name"]
        schema = _feature_schema(feature)
        if schema["constant"]:
            omitted.append(describe(name)[0])
            continue
        if feature.get("type") == "binary":
            source, category = _dummy_source(name)
            question = questions.setdefault(source, {
                "key": source,
                "kind": "onehot",
                "label": feature["label"].split(":")[0].strip(),
                "unit": "",
                "features": [],
                "options": [],
            })
            question["features"].append(position)
            question["options"].append({
                "label": CATEGORY_WORDS.get(category, category.capitalize()),
                "feature": position,
                "rank": CATEGORY_ORDER.index(category) if category in CATEGORY_ORDER else len(CATEGORY_ORDER),
            })
            continue
        label, unit = describe(name)
        if name in presentation["choices"]:
            questions[name] = {
                "key": name,
                "kind": "choice",
                "label": label,
                "unit": "",
                "features": [position],
                "options": [{"value": value, "label": text} for value, text in presentation["choices"][name]],
            }
        elif schema["binary"]:
            questions[name] = {
                "key": name,
                "kind": "yesno",
                "label": label,
                "unit": "",
                "features": [position],
                "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}],
            }
        else:
            questions[name] = {
                "key": name,
                "kind": "number",
                "label": label,
                "unit": unit,
                "features": [position],
                "min": schema["min"],
                "max": schema["max"],
                "decimals": _decimals(cleaned[name]) if name in cleaned else 2,
            }

    for question in questions.values():
        if question["kind"] == "onehot":
            question["options"].sort(key=lambda option: (option["rank"], option["label"]))
            for option in question["options"]:
                del option["rank"]
        question["typical"] = _typical(question, cleaned, features)

    # Grouping. Either explicit groups by name, or (for the nucleus table) by
    # the column suffix, with the suffix's words stripped from each label
    # because the group heading now says them.
    grouped: list[dict[str, Any]] = []
    placed: set[str] = set()
    if presentation.get("suffix_groups"):
        for suffix, title, label_suffix in presentation["suffix_groups"]:
            keys = [key for key in questions if key.endswith(suffix) and key not in placed]
            for key in keys:
                if questions[key]["label"].endswith(label_suffix):
                    # The form shows the short label under its group heading;
                    # the driver list and tooltips have no heading, so they
                    # keep the full one.
                    questions[key]["long_label"] = questions[key]["label"]
                    questions[key]["label"] = questions[key]["label"][: -len(label_suffix)]
            grouped.append({"title": title, "questions": keys})
            placed.update(keys)
    for title, names in presentation["groups"]:
        keys = [name for name in names if name in questions and name not in placed]
        unknown = [name for name in names if name not in questions and name not in index_of]
        if unknown:
            raise SystemExit(f"{slug}: presentation groups name unknown fields {unknown}")
        grouped.append({"title": title, "questions": keys})
        placed.update(keys)
    leftover = [key for key in questions if key not in placed]
    if leftover:
        grouped.append({"title": "Other measurements", "questions": leftover})

    return {
        "questions": [questions[key] for group in grouped for key in group["questions"]],
        "groups": [{"title": group["title"], "keys": group["questions"]} for group in grouped],
        "omitted": omitted,
    }


# --------------------------------------------------------------------------
# Real example records
# --------------------------------------------------------------------------


def _row_values(encoded: Any, position: int) -> list[float | None]:
    """One encoded row as the walker takes it: numbers, with ``None`` for blank."""
    values: list[float | None] = []
    for value in encoded.iloc[position].tolist():
        if isinstance(value, bool) or type(value).__name__ == "bool_":
            values.append(1.0 if value else 0.0)
        elif value is None or value != value:
            values.append(None)
        else:
            values.append(float(value))
    return values


def select_examples(slug: str, model: dict[str, Any]) -> list[dict[str, Any]]:
    """Two or three real records the visitor can load with one click.

    *   ``had`` / ``had_not``: among the records of that outcome with the
        fewest blank fields, the one whose out-of-fold estimate sits closest
        to the median out-of-fold estimate of that outcome -- a typical case,
        not the most convincing one. Ties go to the earliest row.
    *   ``close_call``: among the records with the fewest blank fields, the
        one nearest 0.5 out-of-fold that the shipped model answers
        "uncertain" on. Only offered where the model can give that answer at
        all (three of the six cannot, and the page says so), and only when
        neither typical record is already a close call.

    A position in the out-of-fold file has to be the same patient as that
    position in the cleaned data, or every example would carry another
    record's estimate. That is checked here, not assumed: the two files must
    have the same length and the same outcome on every row.
    """
    import numpy as np
    import pandas as pd

    from nightingale.conditions import CONDITIONS
    from nightingale.export import encoded_frame, predict

    encoded = encoded_frame(slug)
    cleaned_target = pd.read_csv(CLEANED_DIR / f"{slug}.csv.gz")[CONDITIONS[slug].target].to_numpy()
    oof = pd.read_csv(MODELS_DIR / slug / "oof_predictions.csv.gz")
    if len(oof) != len(encoded) or not (oof["y_true"].to_numpy() == cleaned_target).all():
        raise SystemExit(f"{slug}: out-of-fold predictions are not row-aligned with the cleaned data")
    y = oof["y_true"].to_numpy()
    p = oof["p_cal"].to_numpy()
    blanks = encoded.isna().sum(axis=1).to_numpy()

    examples: list[dict[str, Any]] = []
    for kind, outcome in (("had", 1), ("had_not", 0)):
        rows = np.flatnonzero(y == outcome)
        fewest = blanks[rows].min()
        candidates = rows[blanks[rows] == fewest]
        median = float(np.median(p[rows]))
        distance = np.abs(p[candidates] - median)
        pick = int(candidates[np.lexsort((candidates, distance))[0]])
        examples.append({
            "kind": kind,
            "row": pick + 1,
            "outcome": outcome,
            "values": _row_values(encoded, pick),
        })

    # A close call is only worth a button when the model can make that call
    # and neither typical record already shows it.
    q_hat = float(model["conformal"]["q_hat"])
    already_uncertain = any(predict(model, example["values"])["set"] == "uncertain" for example in examples)
    if q_hat >= 0.5 and not already_uncertain:
        order = np.lexsort((np.arange(len(p)), np.abs(p - 0.5), blanks))
        for position in order[:CLOSE_CALL_SEARCH_CAP]:
            values = _row_values(encoded, int(position))
            if predict(model, values)["set"] == "uncertain":
                examples.append({
                    "kind": "close_call",
                    "row": int(position) + 1,
                    "outcome": int(y[position]),
                    "values": values,
                })
                break
    return examples


# --------------------------------------------------------------------------
# Per-condition payload
# --------------------------------------------------------------------------


def _condition_notes(slug: str, metrics: dict[str, Any], q_hat: float) -> list[str]:
    """Honest caveats that belong beside this condition's output, not in a footnote."""
    notes: list[str] = []
    if q_hat == 0.0:
        share = saturated_fraction(slug)
        notes.append(
            "This model can never answer “too close to call”. Its conformal "
            f"threshold is exactly 0, because {share:.2%} of its out-of-fold "
            "calibrated probabilities are pinned to 0 or 1: the cohort is "
            "close to separable and the isotonic calibrator saturates. Every "
            "verdict here comes from the plain 0.5 cut, and the absence of "
            "“too close to call” says nothing about how sure the model is."
        )
    if slug == "diabetes":
        notes.append(
            "Every field here is a survey answer, coded as an integer. Age is "
            "a 13-level band, not a count of years, and the outcome is what "
            "the respondent said about themselves, not a clinical diagnosis."
        )
    if metrics["n_positive"] < 100:
        notes.append(
            f"The whole cohort contains {metrics['n_positive']} positive cases. "
            "Read every interval on this condition as wide, and the subgroup "
            "numbers as indicative at best."
        )
    return notes


def build_condition(slug: str, display: str, meta: dict[str, Any]) -> dict[str, Any]:
    model = _read_json(MODELS_DIR / slug / "model.json")
    metrics = _read_json(MODELS_DIR / slug / "metrics.json")
    summary = metrics["cv_summary"]
    q_hat = float(model["conformal"]["q_hat"])
    presentation = PRESENTATION[slug]
    form = build_questions(slug, model)

    return {
        "slug": slug,
        "display": display,
        "source": meta["source"],
        "positive_meaning": meta["positive_meaning"],
        "citation": meta["citation"],
        "license_note": meta["license_note"],
        "n": int(metrics["n"]),
        "n_positive": int(metrics["n_positive"]),
        "prevalence": float(metrics["prevalence"]),
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "brier": metrics["brier"],
        "ece": metrics["ece"],
        "reliability": metrics["reliability"],
        "net_benefit": metrics["net_benefit"],
        "conformal": metrics["conformal"],
        "subgroups": metrics["subgroup_audits"],
        "cv": {
            "calibration_method": summary["calibration_method"],
            "outer_n_splits": summary["outer_n_splits"],
            "inner_n_splits": summary["inner_n_splits"],
            "seed": summary["seed"],
            "params": summary["chosen_params"],
            "ece_uncalibrated": summary["ece_uncalibrated"],
            "ece_calibrated": summary["ece_calibrated"],
            "subsample_n": summary.get("diabetes_inner_subsample_n"),
        },
        "features": [_feature_schema(f) for f in model["features"]],
        "ui": {
            "order": presentation["order"],
            "name": presentation["name"],
            "icon": presentation["icon"],
            "noun": presentation["noun"],
            "intro": presentation["intro"],
            "positive": presentation["positive"],
            "negative": presentation["negative"],
            "had": presentation["had"],
            "had_not": presentation["had_not"],
            "questions": form["questions"],
            "groups": form["groups"],
            "omitted": form["omitted"],
            "examples": select_examples(slug, model),
        },
        "notes": _condition_notes(slug, metrics, q_hat),
        "oof": sample_oof(slug),
        "sha256": _sha256_file(MODELS_DIR / slug / "model.json"),
        "model": model,
    }


# --------------------------------------------------------------------------
# The four-hospital transfer study
# --------------------------------------------------------------------------

# Columns whose availability is itself site-dependent — the exhibit the
# transfer panel is built around. Measured on the cleaned, signed cohort, so
# what the page shows is what the model actually trained and tested on.
TRANSFER_EXHIBIT_COLUMNS = ("chol", "ca", "thal")


def site_availability() -> dict[str, Any]:
    """Per-site share of the exhibit columns the model cannot use.

    Cleaning converted heart-disease's sentinel-coded zeros to missing, so a
    column reading 100% unusable at one site is the sentinel detector's catch
    made visible: Switzerland recorded every cholesterol as 0 mg/dL, and a
    model that believed those zeros would read an entire hospital as extreme
    hypocholesterolaemia.
    """
    import pandas as pd

    frame = pd.read_csv(CLEANED_DIR / "heart-disease.csv.gz")
    rows = []
    for site, group in frame.groupby("site", sort=True):
        rows.append(
            {
                "site": str(site),
                "n": int(len(group)),
                "unusable": {
                    column: float(group[column].isna().mean())
                    for column in TRANSFER_EXHIBIT_COLUMNS
                },
            }
        )
    return {"columns": list(TRANSFER_EXHIBIT_COLUMNS), "rows": rows}


def build_external() -> dict[str, Any]:
    study = _read_json(MODELS_DIR / "heart-disease" / "external.json")
    sites = {}
    for name, site in sorted(study["sites"].items()):
        sites[name] = {
            "n": site["n"],
            "n_positive": site["n_positive"],
            "n_calibration": site["n_calibration"],
            "n_evaluation": site["n_evaluation"],
            "prevalence": site["prevalence"],
            "fitted_intercept": site["fitted_intercept"],
            "naive": site["naive"],
            "recalibrated": site["recalibrated"],
            "naive_all_rows": site["naive_all_rows"],
        }
    return {
        "training_site": study["training_site"],
        "training_n": study["cleveland_training"]["n"],
        "calibration_split_fraction": study["calibration_split_fraction"],
        "n_boot": study["n_boot"],
        "seed": study["seed"],
        "sites": sites,
        "availability": site_availability(),
    }


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def build_payload() -> dict[str, Any]:
    from nightingale.conditions import CONDITIONS

    conditions = [
        build_condition(
            slug,
            entry.display,
            {
                "source": entry.source,
                "positive_meaning": entry.positive_meaning,
                "citation": entry.citation,
                "license_note": entry.license_note,
            },
        )
        for slug, entry in CONDITIONS.items()
    ]

    payload = {
        "conditions": conditions,
        "external": build_external(),
        # What the footer can honestly print.
        #
        # Not the manifest's own SHA-256. This page is itself a manifest
        # artifact, so signing it changes the manifest, which would change the
        # hash printed here, which would change the page, which would change
        # the manifest: the fixpoint does not exist. Baking a manifest hash
        # would either be stale the moment the page was signed or would make
        # the build non-deterministic, and both are worse than saying so.
        #
        # Everything below is stable under signing and is checkable by hand:
        # the key fingerprint comes from the committed public key, each model
        # digest is the value the manifest already carries for that bundle,
        # and the commit is the one the bundles were exported at.
        "provenance": {
            "commit": conditions[0]["model"]["provenance"]["commit"],
            "author": conditions[0]["model"]["provenance"]["author"],
            "model_count": len(conditions),
            "pubkey_fingerprint": pubkey_fingerprint(),
        },
        "settings": {
            "oof_sample_cap": OOF_SAMPLE_CAP,
            "oof_sample_seed": OOF_SAMPLE_SEED,
            "contribution_slots": 5,
        },
    }
    return _finite(payload)


def font_faces_css() -> str:
    """``@font-face`` rules with each typeface inlined as a base64 data URI."""
    rules = []
    for family, filename, weight in FONT_FACES:
        data = base64.b64encode((FONTS_DIR / filename).read_bytes()).decode("ascii")
        rules.append(
            "@font-face{font-family:'" + family + "';font-style:normal;font-weight:" + weight
            + ";font-display:swap;src:url(data:font/woff2;base64," + data + ") format('woff2')}"
        )
    return "\n".join(rules)


def render_page() -> str:
    """The finished page as a string, with nothing read from the clock."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    for placeholder in (DATA_PLACEHOLDER, WALKER_PLACEHOLDER, FONTS_PLACEHOLDER):
        if template.count(placeholder) != 1:
            raise SystemExit(f"template must contain exactly one {placeholder}")

    blob = json.dumps(build_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    # </script> inside a JSON string would close the tag early; escaping the
    # slash is invisible to JSON.parse and keeps the parser inside the block.
    blob = blob.replace("</", "<\\/")

    page = template.replace(WALKER_PLACEHOLDER, WALKER_PATH.read_text(encoding="utf-8"))
    page = page.replace(FONTS_PLACEHOLDER, font_faces_css())
    page = page.replace(DATA_PLACEHOLDER, blob)
    return page


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=OUTPUT_PATH, help="where to write the page"
    )
    args = parser.parse_args()

    page = render_page()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")

    raw = len(page.encode("utf-8"))
    packed = len(gzip.compress(page.encode("utf-8"), 9))
    print(f"{args.out}: {raw:,} bytes ({packed:,} gzipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
