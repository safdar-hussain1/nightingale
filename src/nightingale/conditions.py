# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""The condition registry.

Each of the six clinical conditions Nightingale models is described by a single
:class:`Condition` entry below. The registry is the source of truth every later
stage (fetch, clean, train, export) reads from — nothing about a condition's
data source, expected shape, or target semantics should be hardcoded elsewhere.

Cleaned-data contract (produced by ``nightingale clean``, consumed by every
later stage): ``data/cleaned/<slug>.csv.gz`` has feature columns plus a
``target`` column (int, 0/1). ``heart-disease`` additionally carries a ``site``
column (one of ``cleveland``/``hungarian``/``switzerland``/``va``) recording
which of the four contributing hospitals a row came from; ``site`` is NOT a
model feature and must be excluded before training.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Condition:
    """Static description of one of the six modelled conditions."""

    slug: str  # e.g. "breast-cancer"
    display: str  # "Breast cancer (WDBC)"
    source: str  # "UCI 17" etc.
    fetch_mode: str  # "ucimlrepo" | "zip"
    fetch_ref: str  # ucimlrepo id as str, or full zip URL
    raw_files: tuple[str, ...]  # filenames expected under data/raw/<slug>/
    target: str  # canonical binary target column name after cleaning: "target"
    positive_meaning: str  # human sentence, e.g. "malignant tumour"
    n_rows: int  # expected cleaned row count (fail-fast check)
    citation: str
    license_note: str


CONDITIONS: dict[str, Condition] = {
    "breast-cancer": Condition(
        slug="breast-cancer",
        display="Breast cancer (WDBC)",
        source="UCI 17",
        # ucimlrepo works fine for this dataset.
        fetch_mode="ucimlrepo",
        fetch_ref="17",
        raw_files=("features.csv", "targets.csv"),
        target="target",
        positive_meaning="malignant tumour",
        n_rows=569,
        citation=(
            "Wolberg, W., Street, W., & Mangasarian, O. (1995). Breast Cancer "
            "Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. "
            "Dua, D. & Graff, C., UCI Machine Learning Repository, "
            "https://archive.ics.uci.edu/dataset/17."
        ),
        license_note="CC BY 4.0 (UCI Machine Learning Repository).",
    ),
    "cervical-cancer": Condition(
        slug="cervical-cancer",
        display="Cervical cancer (risk factors)",
        source="UCI 383",
        # ucimlrepo is broken/partial for this dataset (spec §2) — zip download
        # of the full archive is required instead. "?" marks missing values in
        # the raw CSV and must be treated as NaN during cleaning.
        fetch_mode="zip",
        fetch_ref="https://archive.ics.uci.edu/static/public/383/cervical+cancer+risk+factors.zip",
        raw_files=("risk_factors_cervical_cancer.csv",),
        target="target",
        positive_meaning="positive cervical cancer biopsy",
        n_rows=858,
        citation=(
            "Fernandes, K., Cardoso, J., & Fernandes, J. (2017). Cervical "
            "Cancer (Risk Factors) [Dataset]. UCI Machine Learning Repository. "
            "Dua, D. & Graff, C., UCI Machine Learning Repository, "
            "https://archive.ics.uci.edu/dataset/383."
        ),
        license_note="CC BY 4.0 (UCI Machine Learning Repository).",
    ),
    "heart-disease": Condition(
        slug="heart-disease",
        display="Heart disease (four-hospital cohort)",
        source="UCI 45",
        # ucimlrepo only serves the Cleveland site; the zip archive carries all
        # four processed site files (Cleveland, Hungary, Switzerland, VA Long
        # Beach), which is what the multi-site external-validation work needs.
        # Cleaned output additionally carries a non-feature "site" column
        # recording which of the four hospitals each row came from.
        fetch_mode="zip",
        fetch_ref="https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
        raw_files=(
            "processed.cleveland.data",
            "processed.hungarian.data",
            "processed.switzerland.data",
            "processed.va.data",
        ),
        target="target",
        positive_meaning="presence of heart disease (num > 0)",
        n_rows=920,
        citation=(
            "Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., "
            "Schmid, J., Sandhu, S., Guppy, K., Lee, S., & Froelicher, V. "
            "(1988). Heart Disease [Dataset]. UCI Machine Learning "
            "Repository. Dua, D. & Graff, C., UCI Machine Learning "
            "Repository, https://archive.ics.uci.edu/dataset/45."
        ),
        license_note="CC BY 4.0 (UCI Machine Learning Repository).",
    ),
    "kidney-disease": Condition(
        slug="kidney-disease",
        display="Chronic kidney disease",
        source="UCI 336",
        # ucimlrepo works, but raw labels are dirty (e.g. "ckd\t" with a
        # trailing tab) and must be stripped during cleaning.
        fetch_mode="ucimlrepo",
        fetch_ref="336",
        raw_files=("features.csv", "targets.csv"),
        target="target",
        positive_meaning="chronic kidney disease diagnosis (ckd)",
        n_rows=400,
        citation=(
            "Rubini, L., Soundarapandian, P., & Eswaran, P. (2015). Chronic "
            "Kidney Disease [Dataset]. UCI Machine Learning Repository. "
            "Dua, D. & Graff, C., UCI Machine Learning Repository, "
            "https://archive.ics.uci.edu/dataset/336."
        ),
        license_note="CC BY 4.0 (UCI Machine Learning Repository).",
    ),
    "liver-disease": Condition(
        slug="liver-disease",
        display="Liver disease (ILPD)",
        source="UCI 225",
        # ucimlrepo works. Raw "Selector" target is 1/2 with 1=patient and
        # 2=non-patient — inverted relative to the canonical 0/1 target and
        # must be remapped during cleaning. A/G Ratio has nulls to impute.
        fetch_mode="ucimlrepo",
        fetch_ref="225",
        raw_files=("features.csv", "targets.csv"),
        target="target",
        positive_meaning="liver patient (Selector == 1)",
        n_rows=583,
        citation=(
            "Ramana, B. & Venkateswarlu, N. (2012). ILPD (Indian Liver "
            "Patient Dataset) [Dataset]. UCI Machine Learning Repository. "
            "Dua, D. & Graff, C., UCI Machine Learning Repository, "
            "https://archive.ics.uci.edu/dataset/225."
        ),
        license_note="CC BY 4.0 (UCI Machine Learning Repository).",
    ),
    "diabetes": Condition(
        slug="diabetes",
        display="Diabetes (CDC BRFSS-2015)",
        source="UCI 891",
        # ucimlrepo works. Labels are self-reported survey responses (BRFSS
        # 2015), not clinical diagnoses — the honesty banner must say so.
        fetch_mode="ucimlrepo",
        fetch_ref="891",
        raw_files=("features.csv", "targets.csv"),
        target="target",
        positive_meaning="self-reported diabetes or prediabetes",
        n_rows=253680,
        citation=(
            "CDC Diabetes Health Indicators [Dataset] (2015). Derived from "
            "the CDC Behavioral Risk Factor Surveillance System (BRFSS) "
            "2015 survey. UCI Machine Learning Repository. "
            "https://archive.ics.uci.edu/dataset/891."
        ),
        license_note="Public domain (derived from CDC BRFSS 2015).",
    ),
}
