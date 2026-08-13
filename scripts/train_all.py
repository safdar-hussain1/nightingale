# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Train, evaluate, and report all six conditions -- the published numbers.

For each of the six condition slugs in :data:`nightingale.conditions.CONDITIONS`:

1. :func:`nightingale.model.train_condition` (nested CV + cross-fitted
   calibration) produces the pooled out-of-fold frame every metric below is
   computed from.
2. :func:`nightingale.evaluate.evaluate_oof` bootstraps the four headline
   metrics (ROC-AUC, PR-AUC, Brier, ECE) plus reliability bins, all from
   ``oof["p_cal"]``.
3. :func:`nightingale.conformal.conformal_qhat` (alpha=0.1) fits the
   split-conformal threshold on the pooled OOF frame ("cross-conformal over
   OOF", see ``nightingale/conformal.py``'s module docstring), and
   :func:`nightingale.conformal.prediction_set` tallies how many rows of
   that same frame would receive each of the three verdicts.
4. :func:`nightingale.evaluate.net_benefit` builds a decision-curve table
   on a grid strictly inside ``(0, 1)`` (``net_benefit`` raises otherwise).
5. :func:`nightingale.evaluate.subgroup_audit` runs once per applicable
   demographic dimension (site/sex/age-band -- see :data:`_SEX_COLUMN`,
   :data:`_AGE_COLUMN`, and heart-disease's extra ``site`` dimension below).
6. Two PNG figures (reliability curve, decision curve) are written to
   ``reports/figures/``.
7. ``models/<slug>/metrics.json`` and ``models/<slug>/oof_predictions.csv.gz``
   are written -- see :func:`_build_metrics` and :func:`_write_oof_csv` for
   the exact schema.

Reproducibility note (metrics.json does NOT carry a wall-clock timestamp):
Task 11 signs every file under ``models/``, and a signed artifact must
reproduce byte-for-byte on an honest rerun or ``nightingale verify`` reports
a false TAMPERED. A ``generated_utc`` field stamped with the actual run time
would make that impossible by construction -- every rerun would differ in
exactly one byte range even when nothing else changed. Decision made here:
``generated_utc`` is omitted from ``metrics.json`` entirely; the actual UTC
timestamp of this run is instead written to the separate, UNSIGNED
``models/run_meta.json`` (one file for the whole run, not per-condition),
alongside the exact command and per-condition wall-clock. See the task-8
report and the numbers file for the full rationale.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nightingale.clean import CLEANED_ROOT
from nightingale.conditions import CONDITIONS
from nightingale.conformal import conformal_qhat, prediction_set
from nightingale.evaluate import evaluate_oof, net_benefit, subgroup_audit
from nightingale.model import train_condition

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = REPO_ROOT / "models"
FIGURES_ROOT = REPO_ROOT / "reports" / "figures"

SEED = 42
N_BOOT = 2000
CONFORMAL_ALPHA = 0.1
SUBGROUP_MIN_N = 40

# Decision-curve grid: strictly inside the open interval (0, 1) --
# net_benefit raises ValueError on either endpoint (division by zero at
# pt=1, and pt=0 is meaningless as a "treat everyone above this risk"
# threshold). 99 points, one per integer percentage point from 1% to 99%.
DCA_THRESHOLDS = np.linspace(0.01, 0.99, 99)

# gzip embeds a Unix mtime in its header by default, which would make two
# writes of byte-identical DataFrame content produce different files --
# breaking Task 11's Ed25519 provenance manifest (see nightingale/clean.py's
# _GZIP_COMPRESSION, which this mirrors exactly for the same reason).
_GZIP_COMPRESSION = {"method": "gzip", "mtime": 0}

# ---------------------------------------------------------------------------
# Subgroup audit dimensions
# ---------------------------------------------------------------------------

# Sex/gender feature column per condition, where one exists (spec: heart
# disease `sex`, liver-disease `sex_male`, diabetes `Sex`; cervical-cancer,
# breast-cancer, and kidney-disease carry none).
_SEX_COLUMN = {
    "heart-disease": "sex",
    "liver-disease": "sex_male",
    "diabetes": "Sex",
}

# Age feature column per condition, where one exists (breast-cancer has
# none -- WDBC is purely image-derived measurements).
_AGE_COLUMN = {
    "cervical-cancer": "Age",
    "heart-disease": "age",
    "kidney-disease": "age",
    "liver-disease": "Age",
    "diabetes": "Age",
}

# diabetes's "Age" is NOT raw years -- it is the CDC BRFSS 13-level 5-year
# age-group code (1 = 18-24, 2 = 25-29, ..., 13 = 80+), the standard coding
# for this derived UCI 891 dataset. Mapped directly to the requested bands
# rather than treated as a continuous year value.
_BRFSS_AGE_CODE_TO_BAND = {
    1: "<45", 2: "<45", 3: "<45", 4: "<45", 5: "<45",
    6: "45-54", 7: "45-54",
    8: "55-64", 9: "55-64",
    10: "65+", 11: "65+", 12: "65+", 13: "65+",
}

AGE_BAND_LABELS = ["<45", "45-54", "55-64", "65+"]


def _age_band_from_years(age_years: pd.Series) -> pd.Series:
    """Bucket a raw-years age column into the four requested bands.

    ``right=False`` makes each bin half-open on the right (``[lo, hi)``),
    so integer ages land exactly where the band names say: 45-54 means
    ages 45 through 54 inclusive, not 45 through 55.
    """
    bins = [-np.inf, 45, 55, 65, np.inf]
    return pd.cut(age_years, bins=bins, labels=AGE_BAND_LABELS, right=False).astype(object)


def _subgroup_dimensions(slug: str, cleaned_df: pd.DataFrame) -> dict[str, pd.Series]:
    """Every applicable {dimension_name: groups_series} pair for one condition.

    ``groups_series`` is index-aligned to ``cleaned_df`` (and therefore to
    the OOF frame -- both are read with pandas' default RangeIndex over the
    same row order, per nightingale.model's "index-aligned to the cleaned
    frame, original row order" contract).
    """
    dims: dict[str, pd.Series] = {}

    if slug == "heart-disease":
        # Audit dimension ONLY -- site is explicitly excluded as a model
        # feature (nightingale.conditions, nightingale.model), but it is
        # exactly the kind of provenance grouping a subgroup audit exists
        # to check.
        dims["site"] = cleaned_df["site"]

    if slug in _SEX_COLUMN:
        dims["sex"] = cleaned_df[_SEX_COLUMN[slug]]

    if slug in _AGE_COLUMN:
        col = _AGE_COLUMN[slug]
        if slug == "diabetes":
            dims["age_band"] = cleaned_df[col].map(_BRFSS_AGE_CODE_TO_BAND)
        else:
            dims["age_band"] = _age_band_from_years(cleaned_df[col])

    return dims


# ---------------------------------------------------------------------------
# JSON-safety: NaN/Infinity are not valid JSON tokens (evaluate.py's
# net_benefit docstring flags exactly this failure mode for the dashboard;
# the same discipline applies here for subgroup_audit's NaN "insufficient
# group" cells and any empty reliability bin).
# ---------------------------------------------------------------------------


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if (np.isnan(value) or np.isinf(value)) else value
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


# ---------------------------------------------------------------------------
# Per-condition pipeline
# ---------------------------------------------------------------------------


def _verdict_fractions(oof: pd.DataFrame, q_hat: float) -> tuple[dict, list[str]]:
    """Verdict distribution over the pooled OOF frame at this q_hat.

    Every verdict category (positive/negative/uncertain) is always present
    in the returned dict, at 0.0 if a category never occurs -- so
    downstream consumers (and the well-formedness tests) can rely on all
    three keys existing without a membership check.
    """
    verdicts = [prediction_set(float(p), q_hat) for p in oof["p_cal"].to_numpy()]
    n = len(verdicts)
    counts = Counter(verdicts)
    fractions = {
        label: counts.get(label, 0) / n for label in ("positive", "negative", "uncertain")
    }
    return fractions, verdicts


def _build_metrics(slug: str, oof: pd.DataFrame, result, cleaned_df: pd.DataFrame) -> dict:
    metrics = evaluate_oof(oof, n_boot=N_BOOT, seed=0)

    q_hat = conformal_qhat(oof["y_true"].to_numpy(), oof["p_cal"].to_numpy(), alpha=CONFORMAL_ALPHA)
    verdict_fractions, verdicts = _verdict_fractions(oof, q_hat)

    nb_df = net_benefit(oof["y_true"].to_numpy(), oof["p_cal"].to_numpy(), DCA_THRESHOLDS)

    subgroup_audits = {}
    for dim_name, groups in _subgroup_dimensions(slug, cleaned_df).items():
        audit_df = subgroup_audit(oof, groups, min_n=SUBGROUP_MIN_N, n_boot=N_BOOT, seed=0)
        subgroup_audits[dim_name] = audit_df.to_dict(orient="records")

    payload = {
        "slug": slug,
        "n": metrics["n"],
        "n_positive": metrics["n_positive"],
        "prevalence": metrics["prevalence"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "brier": metrics["brier"],
        "ece": metrics["ece"],
        "reliability": metrics["reliability"],
        "conformal": {
            "alpha": CONFORMAL_ALPHA,
            "q_hat": q_hat,
            "verdict_fractions": verdict_fractions,
        },
        "chosen_params": result.chosen_params,
        "feature_names": result.feature_names,
        "net_benefit": nb_df.to_dict(orient="records"),
        "subgroup_audits": subgroup_audits,
        "cv_summary": result.cv_summary,
    }
    return _json_safe(payload), verdicts


def _write_oof_csv(slug: str, oof: pd.DataFrame, verdicts: list[str]) -> Path:
    out = oof.copy()
    out["verdict"] = verdicts
    out_path = MODELS_ROOT / slug / "oof_predictions.csv.gz"
    out.to_csv(out_path, index=False, compression=_GZIP_COMPRESSION)
    return out_path


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_COLOR_MODEL = "#2a78d6"  # dataviz reference palette, categorical slot 1 (blue)
_COLOR_SECONDARY = "#eb6834"  # categorical slot 2 (orange) -- treat_all
_COLOR_NEUTRAL = "#8a8a86"  # neutral gray -- reference line / treat_none
_COLOR_TEXT = "#2b2b28"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#c9c8c2",
        "axes.labelcolor": _COLOR_TEXT,
        "text.color": _COLOR_TEXT,
        "xtick.color": _COLOR_TEXT,
        "ytick.color": _COLOR_TEXT,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.7,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
    }
)


def _add_watermark(fig) -> None:
    fig.text(
        0.99,
        0.01,
        "Nightingale · Safdar Hussain",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#9a9a94",
        alpha=0.9,
    )


def _plot_reliability(slug: str, display: str, reliability: list[dict]) -> Path:
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=150)

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.3, color=_COLOR_NEUTRAL, label="Perfect calibration")

    non_empty = [b for b in reliability if b["n"] > 0]
    xs = [b["mean_pred"] for b in non_empty]
    ys = [b["frac_pos"] for b in non_empty]
    sizes = [15 + 40 * (b["n"] / max(1, max(bb["n"] for bb in non_empty))) for b in non_empty]
    ax.plot(xs, ys, color=_COLOR_MODEL, linewidth=1.8, zorder=2)
    ax.scatter(xs, ys, s=sizes, color=_COLOR_MODEL, zorder=3, label="Observed (calibrated p)")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability (calibrated)")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title(f"Reliability — {display}")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _add_watermark(fig)

    out_path = FIGURES_ROOT / f"{slug}-reliability.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _plot_decision_curve(slug: str, display: str, nb_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=150)

    ax.plot(nb_df["threshold"], nb_df["model"], color=_COLOR_MODEL, linewidth=2.0, label="Model")
    ax.plot(
        nb_df["threshold"], nb_df["treat_all"], color=_COLOR_SECONDARY, linewidth=1.6,
        linestyle="--", label="Treat all",
    )
    ax.plot(
        nb_df["threshold"], nb_df["treat_none"], color=_COLOR_NEUTRAL, linewidth=1.6,
        linestyle=":", label="Treat none",
    )

    model_max = float(nb_df["model"].max())
    treat_all_finite = nb_df["treat_all"].replace([np.inf, -np.inf], np.nan).dropna()
    upper = max(model_max, float(treat_all_finite.max()) if len(treat_all_finite) else 0.0, 0.02) * 1.3
    lower = -0.15 * upper if upper > 0 else -0.02

    ax.set_xlim(0, 1)
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(f"Decision curve — {display}")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)

    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _add_watermark(fig)

    out_path = FIGURES_ROOT / f"{slug}-decision-curve.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(slugs: list[str]) -> dict:
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    run_started = datetime.now(timezone.utc)
    per_condition_seconds: dict[str, float] = {}
    headline_rows: list[dict] = []

    for slug in slugs:
        condition = CONDITIONS[slug]
        print(f"=== {slug} ({condition.display}) ===", flush=True)
        t0 = time.time()

        result = train_condition(slug, seed=SEED)
        oof = result.oof
        cleaned_df = pd.read_csv(CLEANED_ROOT / f"{slug}.csv.gz", compression="gzip")

        metrics_payload, verdicts = _build_metrics(slug, oof, result, cleaned_df)

        slug_dir = MODELS_ROOT / slug
        slug_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = slug_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_payload, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")

        oof_path = _write_oof_csv(slug, oof, verdicts)

        reliability_fig = _plot_reliability(slug, condition.display, metrics_payload["reliability"])
        nb_df = pd.DataFrame(metrics_payload["net_benefit"])
        dca_fig = _plot_decision_curve(slug, condition.display, nb_df)

        elapsed = time.time() - t0
        per_condition_seconds[slug] = elapsed

        roc = metrics_payload["roc_auc"]
        pr = metrics_payload["pr_auc"]
        brier = metrics_payload["brier"]
        ece_ = metrics_payload["ece"]
        print(
            f"  n={metrics_payload['n']} prevalence={metrics_payload['prevalence']:.4f} "
            f"roc_auc={roc[0]:.4f} [{roc[1]:.4f}, {roc[2]:.4f}]  "
            f"pr_auc={pr[0]:.4f} [{pr[1]:.4f}, {pr[2]:.4f}]  "
            f"brier={brier[0]:.4f} [{brier[1]:.4f}, {brier[2]:.4f}]  "
            f"ece={ece_[0]:.4f} [{ece_[1]:.4f}, {ece_[2]:.4f}]  "
            f"q_hat={metrics_payload['conformal']['q_hat']:.4f} "
            f"uncertain_rate={metrics_payload['conformal']['verdict_fractions']['uncertain']:.4f}  "
            f"wall={elapsed:.1f}s",
            flush=True,
        )
        print(f"  wrote {metrics_path.relative_to(REPO_ROOT)}", flush=True)
        print(f"  wrote {oof_path.relative_to(REPO_ROOT)}", flush=True)
        print(f"  wrote {reliability_fig.relative_to(REPO_ROOT)}", flush=True)
        print(f"  wrote {dca_fig.relative_to(REPO_ROOT)}", flush=True)

        headline_rows.append(
            {
                "slug": slug,
                "n": metrics_payload["n"],
                "prevalence": metrics_payload["prevalence"],
                "roc_auc": roc,
                "pr_auc": pr,
                "brier": brier,
                "ece": ece_,
                "chosen_params": result.chosen_params,
                "q_hat": metrics_payload["conformal"]["q_hat"],
                "uncertain_rate": metrics_payload["conformal"]["verdict_fractions"]["uncertain"],
                "wall_seconds": elapsed,
            }
        )

    run_finished = datetime.now(timezone.utc)

    # Merge into any existing run_meta.json rather than overwrite it: a
    # partial run (``--slug X``, e.g. re-training just one condition to
    # check byte-reproducibility) must not erase the other conditions'
    # recorded wall-clock from a prior full run. Only the slugs actually
    # trained THIS invocation get their per-condition entry replaced.
    run_meta_path = MODELS_ROOT / "run_meta.json"
    if run_meta_path.exists():
        existing = json.loads(run_meta_path.read_text())
        merged_seconds = dict(existing.get("per_condition_wall_seconds", {}))
    else:
        merged_seconds = {}
    merged_seconds.update(per_condition_seconds)
    total_seconds = sum(merged_seconds.values())

    run_meta = {
        "generated_utc": run_finished.isoformat(),
        "command": "PYTHONPATH=src " + sys.executable + " scripts/train_all.py",
        "seed": SEED,
        "n_boot": N_BOOT,
        "conformal_alpha": CONFORMAL_ALPHA,
        "subgroup_min_n": SUBGROUP_MIN_N,
        "per_condition_wall_seconds": merged_seconds,
        "total_wall_seconds": total_seconds,
        "started_utc": run_started.isoformat(),
        "finished_utc": run_finished.isoformat(),
    }
    with open(run_meta_path, "w") as f:
        json.dump(_json_safe(run_meta), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")

    this_run_seconds = sum(per_condition_seconds.values())
    print(f"\nTotal wall-clock (this invocation): {this_run_seconds:.1f}s across {len(slugs)} conditions", flush=True)
    return {"headline_rows": headline_rows, "total_seconds": this_run_seconds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slug",
        action="append",
        dest="slugs",
        choices=sorted(CONDITIONS),
        help="Train only this condition (repeatable). Default: all six.",
    )
    args = parser.parse_args()
    slugs = args.slugs if args.slugs else list(CONDITIONS)
    run(slugs)


if __name__ == "__main__":
    main()
