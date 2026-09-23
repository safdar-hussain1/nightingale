# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Per-condition data cleaners producing the canonical cleaned-data contract.

Every cleaner takes the raw directory ``Path`` returned by
:func:`nightingale.fetch.fetch` and returns a :class:`pandas.DataFrame` with
feature columns plus a ``target`` column (``int64``, values exactly
``{0, 1}``). ``heart-disease`` additionally carries a non-feature ``site``
column (see :mod:`nightingale.conditions` for the full contract).

Positive-class convention: for all six conditions, ``target == 1`` always
means "has the condition / positive finding" — malignant tumour (breast
cancer), positive biopsy (cervical cancer), presence of heart disease,
chronic kidney disease, liver patient, and self-reported diabetes/prediabetes
respectively. Each cleaner below honours that convention explicitly.

:func:`clean` is the public entry point: it looks up a condition's cleaner in
:data:`PIPELINES`, runs it against the fetched raw data, fail-fast validates
the result against :class:`~nightingale.conditions.Condition`, writes
``data/cleaned/<slug>.csv.gz``, and returns its path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from nightingale.conditions import CONDITIONS
from nightingale.fetch import fetch

# data/cleaned/, resolved relative to this module's location so it works no
# matter the caller's current working directory.
CLEANED_ROOT = Path(__file__).resolve().parents[2] / "data" / "cleaned"


class CleaningError(Exception):
    """Raised when a cleaned DataFrame fails the fail-fast contract checks."""


# ---------------------------------------------------------------------------
# breast-cancer
# ---------------------------------------------------------------------------


def _clean_breast_cancer(raw_dir: Path) -> pd.DataFrame:
    features = pd.read_csv(raw_dir / "features.csv")
    targets = pd.read_csv(raw_dir / "targets.csv")

    # Defensive: this vintage of the ucimlrepo export doesn't carry an id
    # column, but drop it if a future re-fetch ever reintroduces one.
    features = features.drop(columns=[c for c in ("id", "ID") if c in features.columns])

    df = pd.concat([features.reset_index(drop=True), targets.reset_index(drop=True)], axis=1)
    df["target"] = (df["Diagnosis"] == "M").astype("int64")  # malignant = positive
    df = df.drop(columns=["Diagnosis"])
    return df


# ---------------------------------------------------------------------------
# cervical-cancer
# ---------------------------------------------------------------------------

# 91.7% missing (787/858) — no reasonable imputation recovers signal from
# this few real observations, so the columns are dropped outright rather
# than kept with a missingness indicator.
_CERVICAL_SPARSE_COLUMNS = (
    "STDs: Time since first diagnosis",
    "STDs: Time since last diagnosis",
)

# Hinselmann/Schiller/Citology are alternative screening tests performed in
# the same clinical episode as the chosen target, Biopsy — they are outcomes
# of the same underlying exam, not independent predictors. Keeping them as
# features would leak the label into the inputs.
_CERVICAL_LEAKAGE_COLUMNS = ("Hinselmann", "Schiller", "Citology")


def _clean_cervical_cancer(raw_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_dir / "risk_factors_cervical_cancer.csv")
    df = df.replace("?", pd.NA)
    df = df.apply(pd.to_numeric, errors="coerce")

    df = df.drop(columns=list(_CERVICAL_SPARSE_COLUMNS))

    df["target"] = df["Biopsy"].astype("int64")  # positive biopsy = positive
    df = df.drop(columns=[*_CERVICAL_LEAKAGE_COLUMNS, "Biopsy"])
    return df


# ---------------------------------------------------------------------------
# heart-disease
# ---------------------------------------------------------------------------

# The processed.*.data files are headerless; this is the documented UCI 45
# column order.
_HEART_COLUMNS = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
)

_HEART_SITES = {
    "processed.cleveland.data": "cleveland",
    "processed.hungarian.data": "hungarian",
    "processed.switzerland.data": "switzerland",
    "processed.va.data": "va",
}


def _apply_heart_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Convert heart-disease missing-value sentinels (chol/trestbps == 0) to NaN.

    A living patient cannot have zero serum cholesterol or zero resting
    blood pressure — both are missing-value sentinels, not real readings.

    ``chol == 0`` is scoped to the Switzerland and VA sites only: they are
    confirmed (raw-data profile) to encode missing cholesterol this way,
    while Cleveland and Hungary never emit a genuine ``chol == 0`` — applying
    the substitution globally would instead corrupt any real low-cholesterol
    reading those two sites happened to record.

    ``trestbps == 0`` is applied globally (no site scoping): it is
    physiologically impossible everywhere the file records it.

    Returns a new DataFrame; ``df`` is not mutated in place.
    """
    df = df.copy()
    swiss_or_va = df["site"].isin(["switzerland", "va"])
    df.loc[swiss_or_va & (df["chol"] == 0), "chol"] = pd.NA
    df.loc[df["trestbps"] == 0, "trestbps"] = pd.NA
    return df


def _clean_heart_disease(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for filename, site in _HEART_SITES.items():
        site_df = pd.read_csv(
            raw_dir / filename, header=None, names=list(_HEART_COLUMNS), na_values="?"
        )
        site_df["site"] = site
        frames.append(site_df)
    df = pd.concat(frames, ignore_index=True)

    df = _apply_heart_sentinels(df)

    df["target"] = (df["num"] > 0).astype("int64")  # presence of disease = positive
    df = df.drop(columns=["num"])
    return df


# ---------------------------------------------------------------------------
# kidney-disease
# ---------------------------------------------------------------------------

_KIDNEY_TARGET_COLUMN = "class"
_KIDNEY_TARGET_MAP = {"ckd": 1, "notckd": 0}  # ckd = positive

# Genuinely categorical columns (normal/abnormal, present/notpresent,
# yes/no, good/poor) — kept as strings for later one-hot encoding rather
# than coerced to numeric.
_KIDNEY_CATEGORICAL_COLUMNS = {
    "rbc",
    "pc",
    "pcc",
    "ba",
    "htn",
    "dm",
    "cad",
    "appet",
    "pe",
    "ane",
}


def _clean_kidney_disease(raw_dir: Path) -> pd.DataFrame:
    features = pd.read_csv(raw_dir / "features.csv")
    targets = pd.read_csv(raw_dir / "targets.csv")
    df = pd.concat([features.reset_index(drop=True), targets.reset_index(drop=True)], axis=1)

    # Strip whitespace/tabs from every string column BEFORE mapping labels —
    # the raw target column has a handful of 'ckd\t' rows (trailing tab)
    # that would otherwise fail to match 'ckd' and silently form a third
    # class instead of joining the 250 positives.
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()

    df["target"] = df[_KIDNEY_TARGET_COLUMN].map(_KIDNEY_TARGET_MAP).astype("int64")
    df = df.drop(columns=[_KIDNEY_TARGET_COLUMN])

    # Coerce numeric-looking string columns (anything not in the known
    # categorical set) to numeric; genuinely categorical columns stay as
    # strings for later one-hot encoding.
    for col in df.columns:
        if col == "target":
            continue
        if pd.api.types.is_string_dtype(df[col]) and col not in _KIDNEY_CATEGORICAL_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# liver-disease
# ---------------------------------------------------------------------------


def _clean_liver_disease(raw_dir: Path) -> pd.DataFrame:
    features = pd.read_csv(raw_dir / "features.csv")
    targets = pd.read_csv(raw_dir / "targets.csv")
    df = pd.concat([features.reset_index(drop=True), targets.reset_index(drop=True)], axis=1)

    # ILPD's "Selector" uses an inverted convention: 1 = liver patient,
    # 2 = non-patient. Remap to the canonical 1=positive/0=negative target.
    df["target"] = df["Selector"].map({1: 1, 2: 0}).astype("int64")
    df = df.drop(columns=["Selector"])

    # Name the column after what it means once mapped (male=1), not the raw
    # "Gender" label, so its meaning is self-evident downstream.
    df["sex_male"] = (df["Gender"] == "Male").astype("int64")
    df = df.drop(columns=["Gender"])

    return df


# ---------------------------------------------------------------------------
# diabetes
# ---------------------------------------------------------------------------


def _clean_diabetes(raw_dir: Path) -> pd.DataFrame:
    features = pd.read_csv(raw_dir / "features.csv")
    targets = pd.read_csv(raw_dir / "targets.csv")
    df = pd.concat([features.reset_index(drop=True), targets.reset_index(drop=True)], axis=1)

    # Features are already integer-coded survey responses; only the target
    # needs renaming to the canonical column name.
    df = df.rename(columns={"Diabetes_binary": "target"})
    df["target"] = df["target"].astype("int64")  # diabetes/prediabetes = positive
    return df


PIPELINES: dict[str, Callable[[Path], pd.DataFrame]] = {
    "breast-cancer": _clean_breast_cancer,
    "cervical-cancer": _clean_cervical_cancer,
    "heart-disease": _clean_heart_disease,
    "kidney-disease": _clean_kidney_disease,
    "liver-disease": _clean_liver_disease,
    "diabetes": _clean_diabetes,
}


def _validate(slug: str, df: pd.DataFrame) -> None:
    """Fail-fast contract checks shared by every condition.

    Raises :class:`CleaningError` on the first violation found: wrong row
    count, missing/non-binary/non-integer target, or any all-NaN column.
    """
    condition = CONDITIONS[slug]

    if len(df) != condition.n_rows:
        raise CleaningError(
            f"{slug}: expected {condition.n_rows} rows, got {len(df)}"
        )

    if "target" not in df.columns:
        raise CleaningError(f"{slug}: missing 'target' column")

    target = df["target"]
    if not pd.api.types.is_integer_dtype(target):
        raise CleaningError(
            f"{slug}: target dtype must be integer, got {target.dtype}"
        )

    target_values = set(target.unique().tolist())
    if target_values != {0, 1}:
        raise CleaningError(
            f"{slug}: target values must be exactly {{0, 1}}, got {sorted(target_values)}"
        )

    all_nan_cols = [c for c in df.columns if df[c].isna().all()]
    if all_nan_cols:
        raise CleaningError(f"{slug}: all-NaN column(s): {all_nan_cols}")


# gzip embeds a Unix mtime in its header by default, which makes
# `df.to_csv(..., compression="gzip")` produce a different file on every
# call even when the DataFrame content is byte-for-byte identical. That
# breaks the signed Ed25519 provenance manifest (`nightingale verify`),
# which must report an untouched, correctly-regenerated repo as OK rather
# than TAMPERED. Pinning mtime=0 makes the gzip container itself
# deterministic: two writes of the same DataFrame produce the same file, not
# just the same decompressed content.
_GZIP_COMPRESSION = {"method": "gzip", "mtime": 0}


def clean(slug: str) -> Path:
    """Clean one condition's raw data and write ``data/cleaned/<slug>.csv.gz``.

    Fetches (or verifies the cached copy of) the raw data, runs the
    registered cleaner from :data:`PIPELINES`, fail-fast validates the
    result, writes the cleaned CSV (gzip-compressed, ``index=False``,
    deterministic — see :data:`_GZIP_COMPRESSION`), and returns its path.
    """
    raw_dir = fetch(slug)
    pipeline = PIPELINES[slug]
    df = pipeline(raw_dir)

    _validate(slug, df)

    CLEANED_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = CLEANED_ROOT / f"{slug}.csv.gz"
    df.to_csv(out_path, index=False, compression=_GZIP_COMPRESSION)
    return out_path
