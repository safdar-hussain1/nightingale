# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Well-formedness tests for the published Task 8 artifacts.

These tests read the COMMITTED ``models/<slug>/metrics.json`` and
``models/<slug>/oof_predictions.csv.gz`` files (and the figures under
``reports/figures/``) directly -- they never call
:func:`nightingale.model.train_condition` or re-run
``scripts/train_all.py``. Diabetes alone takes several minutes to train
(253,680 rows); re-training six conditions inside the test suite on every
``pytest`` invocation would be both slow and pointless, since the whole
point of this file is to check the ARTIFACTS actually shipped, not to
reproduce them. This also means the file runs unmodified on a fresh clone
(the artifacts are committed, unlike ``data/raw/``).

Covers the well-formedness contract from the Task 8 brief: every metric is
a ``[point, lo, hi]`` triple with ``lo <= point <= hi``; every reliability
bin's counts sum to ``n``; OOF row counts match
``Condition.n_rows``; conformal verdict fractions sum to 1.0; subgroup
audit rows never report a numeric AUC for an "insufficient" group; and the
figures exist as real, non-trivial PNGs. The final section pins the task
brief's own published sanity gates (breast-cancer >= 0.95, diabetes in
[0.75, 0.85], cervical-cancer's PR-AUC CI genuinely wide) as a regression
guard against the real, already-run numbers -- not a target tuned after
the fact.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from nightingale.conditions import CONDITIONS

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = REPO_ROOT / "models"
FIGURES_ROOT = REPO_ROOT / "reports" / "figures"

SLUGS = sorted(CONDITIONS)

REQUIRED_TOP_LEVEL_KEYS = {
    "slug",
    "n",
    "n_positive",
    "prevalence",
    "roc_auc",
    "pr_auc",
    "brier",
    "ece",
    "reliability",
    "conformal",
    "chosen_params",
    "feature_names",
    "net_benefit",
    "subgroup_audits",
    "cv_summary",
}

FOUR_METRICS = ("roc_auc", "pr_auc", "brier", "ece")

# Every subgroup dimension actually available per condition (Task 8 brief
# §2): site is heart-disease-only (audit dimension, never a model
# feature); sex exists for heart-disease/liver-disease/diabetes; age_band
# exists everywhere except breast-cancer (WDBC has no age feature at all).
EXPECTED_SUBGROUP_DIMENSIONS = {
    "breast-cancer": set(),
    "cervical-cancer": {"age_band"},
    "heart-disease": {"site", "sex", "age_band"},
    "kidney-disease": {"age_band"},
    "liver-disease": {"sex", "age_band"},
    "diabetes": {"sex", "age_band"},
}


def _load_metrics(slug: str) -> dict:
    return json.loads((MODELS_ROOT / slug / "metrics.json").read_text())


def _load_oof(slug: str) -> pd.DataFrame:
    return pd.read_csv(MODELS_ROOT / slug / "oof_predictions.csv.gz", compression="gzip")


# ---------------------------------------------------------------------------
# metrics.json: presence, required keys, metric-triple shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slug", SLUGS)
def test_metrics_json_exists(slug):
    assert (MODELS_ROOT / slug / "metrics.json").is_file()


@pytest.mark.parametrize("slug", SLUGS)
def test_metrics_json_has_every_required_key(slug):
    data = _load_metrics(slug)
    missing = REQUIRED_TOP_LEVEL_KEYS - set(data)
    assert not missing, f"{slug}: metrics.json missing key(s) {missing}"


@pytest.mark.parametrize("slug", SLUGS)
def test_metrics_json_slug_field_matches_its_own_directory(slug):
    assert _load_metrics(slug)["slug"] == slug


@pytest.mark.parametrize("slug", SLUGS)
@pytest.mark.parametrize("metric", FOUR_METRICS)
def test_every_headline_metric_is_a_valid_point_lo_hi_triple(slug, metric):
    value = _load_metrics(slug)[metric]
    assert isinstance(value, list)
    assert len(value) == 3
    point, lo, hi = value
    for v in (point, lo, hi):
        assert isinstance(v, (int, float))
    assert lo <= point <= hi


@pytest.mark.parametrize("slug", SLUGS)
def test_n_and_prevalence_are_internally_consistent(slug):
    data = _load_metrics(slug)
    assert data["n"] == CONDITIONS[slug].n_rows
    assert 0 <= data["n_positive"] <= data["n"]
    assert data["prevalence"] == pytest.approx(data["n_positive"] / data["n"])


# ---------------------------------------------------------------------------
# Reliability bins
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slug", SLUGS)
def test_reliability_bin_counts_sum_to_n(slug):
    data = _load_metrics(slug)
    assert sum(b["n"] for b in data["reliability"]) == data["n"]
    assert len(data["reliability"]) == 10  # nightingale.evaluate.N_RELIABILITY_BINS


# ---------------------------------------------------------------------------
# Conformal: q_hat + verdict fractions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slug", SLUGS)
def test_conformal_block_is_well_formed(slug):
    data = _load_metrics(slug)
    conformal = data["conformal"]
    assert conformal["alpha"] == pytest.approx(0.1)
    assert 0.0 <= conformal["q_hat"] <= 1.0

    vf = conformal["verdict_fractions"]
    assert set(vf) == {"positive", "negative", "uncertain"}
    for v in vf.values():
        assert 0.0 <= v <= 1.0
    assert sum(vf.values()) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# chosen_params / feature_names / net_benefit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slug", SLUGS)
def test_chosen_params_are_from_the_documented_grid(slug):
    params = _load_metrics(slug)["chosen_params"]
    assert set(params) == {"max_depth", "n_estimators", "learning_rate"}
    assert params["max_depth"] in (2, 3, 4)
    assert params["n_estimators"] in (100, 300)
    assert params["learning_rate"] in (0.05, 0.1)


@pytest.mark.parametrize("slug", SLUGS)
def test_feature_names_present_and_never_include_site(slug):
    feature_names = _load_metrics(slug)["feature_names"]
    assert isinstance(feature_names, list)
    assert len(feature_names) > 0
    assert "site" not in feature_names  # non-feature provenance column, every condition


@pytest.mark.parametrize("slug", SLUGS)
def test_net_benefit_table_is_well_formed(slug):
    nb = _load_metrics(slug)["net_benefit"]
    assert len(nb) == 99  # np.linspace(0.01, 0.99, 99)
    for row in nb:
        assert set(row) == {"threshold", "model", "treat_all", "treat_none"}
        assert 0.0 < row["threshold"] < 1.0  # net_benefit's own open-interval contract
        assert row["treat_none"] == 0.0


# ---------------------------------------------------------------------------
# oof_predictions.csv.gz
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slug", SLUGS)
def test_oof_predictions_row_count_matches_condition_n_rows(slug):
    assert len(_load_oof(slug)) == CONDITIONS[slug].n_rows


@pytest.mark.parametrize("slug", SLUGS)
def test_oof_predictions_columns_ranges_and_no_missing_values(slug):
    oof = _load_oof(slug)
    assert list(oof.columns) == ["y_true", "p_raw", "p_cal", "fold", "verdict"]
    assert set(oof["y_true"].unique()) <= {0, 1}
    assert oof["p_raw"].between(0.0, 1.0).all()
    assert oof["p_cal"].between(0.0, 1.0).all()
    assert oof["fold"].between(0, 4).all()
    assert set(oof["verdict"].unique()) <= {"positive", "negative", "uncertain"}
    assert not oof.isna().any().any()


@pytest.mark.parametrize("slug", SLUGS)
def test_oof_verdict_column_matches_metrics_json_verdict_fractions(slug):
    """The per-row verdict column and the reported fractions must agree exactly."""
    oof = _load_oof(slug)
    data = _load_metrics(slug)
    n = len(oof)
    counts = oof["verdict"].value_counts()
    for label, fraction in data["conformal"]["verdict_fractions"].items():
        expected = counts.get(label, 0) / n
        assert fraction == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.mark.parametrize("slug", SLUGS)
@pytest.mark.parametrize("kind", ["reliability", "decision-curve"])
def test_figure_exists_and_is_a_real_png(slug, kind):
    path = FIGURES_ROOT / f"{slug}-{kind}.png"
    assert path.is_file(), f"missing figure {path}"
    assert path.stat().st_size > 1000  # not a truncated/empty write
    with open(path, "rb") as f:
        assert f.read(8) == _PNG_MAGIC


# ---------------------------------------------------------------------------
# run_meta.json (unsigned run metadata -- see scripts/train_all.py's
# module docstring for why generated_utc lives here and not in metrics.json)
# ---------------------------------------------------------------------------


def test_run_meta_json_covers_every_condition():
    data = json.loads((MODELS_ROOT / "run_meta.json").read_text())
    assert set(data["per_condition_wall_seconds"]) == set(CONDITIONS)
    assert data["total_wall_seconds"] > 0
    assert data["conformal_alpha"] == pytest.approx(0.1)
    assert data["seed"] == 42


def test_generated_utc_is_absent_from_every_metrics_json():
    """metrics.json must stay byte-reproducible across reruns -- see the
    module docstring in scripts/train_all.py. A wall-clock timestamp field
    would break that on the very next honest rerun.
    """
    for slug in SLUGS:
        assert "generated_utc" not in _load_metrics(slug)


# ---------------------------------------------------------------------------
# Subgroup audits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("slug", SLUGS)
def test_subgroup_audit_dimensions_match_available_demographics(slug):
    data = _load_metrics(slug)
    assert set(data["subgroup_audits"]) == EXPECTED_SUBGROUP_DIMENSIONS[slug]


@pytest.mark.parametrize(
    "slug", sorted(slug for slug, dims in EXPECTED_SUBGROUP_DIMENSIONS.items() if dims)
)
def test_subgroup_audit_rows_never_report_a_number_for_an_insufficient_group(slug):
    data = _load_metrics(slug)
    for dim_name, rows in data["subgroup_audits"].items():
        assert len(rows) > 0, f"{slug}/{dim_name}: no groups reported"
        for row in rows:
            assert row["n"] >= 0
            assert row["n_positive"] <= row["n"]
            if row["sufficient"]:
                assert row["n"] >= 40  # SUBGROUP_MIN_N
                assert row["auc"] is not None
                assert row["auc_lo"] <= row["auc"] <= row["auc_hi"]
                assert row["mean_cal_error"] is not None
            else:
                # Honest NaN-as-null -- never a computed-looking number for
                # a group too small to trust (nightingale.evaluate's own
                # contract; see also test_evaluate.py's small-group test).
                assert row["auc"] is None
                assert row["auc_lo"] is None
                assert row["auc_hi"] is None


def test_heart_disease_site_audit_covers_all_four_hospitals():
    rows = _load_metrics("heart-disease")["subgroup_audits"]["site"]
    groups = {r["group"] for r in rows}
    assert groups == {"cleveland", "hungarian", "switzerland", "va"}
    assert sum(r["n"] for r in rows) == 920


# ---------------------------------------------------------------------------
# Published sanity gates (Task 8 brief) -- pinned against the real,
# already-run artifacts as a regression guard. These are read, not tuned:
# the brief specifies the acceptance bars up front, and the run either
# clears them or the brief requires reporting the miss honestly (which the
# numbers file and task-8 report do for anything that doesn't clear).
# ---------------------------------------------------------------------------


def test_breast_cancer_meets_the_wdbc_separability_floor():
    assert _load_metrics("breast-cancer")["roc_auc"][0] >= 0.95


def test_diabetes_roc_auc_is_within_the_brfss_literature_range():
    point = _load_metrics("diabetes")["roc_auc"][0]
    assert 0.75 <= point <= 0.85


def test_cervical_cancer_pr_auc_is_reported_with_a_genuinely_wide_ci():
    # The brief predicts LOW PR-AUC with a WIDE CI at 6.4% prevalence / 55
    # positives -- this pins that the artifact reports that honestly
    # (interval actually wide) rather than silently narrowing it, without
    # asserting a specific point value the way the other two gates do.
    point, lo, hi = _load_metrics("cervical-cancer")["pr_auc"]
    assert lo <= point <= hi
    assert (hi - lo) > 0.05
