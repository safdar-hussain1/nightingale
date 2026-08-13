# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Run the four-hospital external validation study and publish its artifacts.

Calls :func:`nightingale.external.transfer_study` (train on Cleveland only,
score naive + intercept-recalibrated transfer on hungarian/switzerland/va),
writes the deterministic result to ``models/heart-disease/external.json``,
and renders ``reports/figures/external-validation.png`` -- per-site AUC with
bootstrap CIs (left) and per-site ECE before/after intercept recalibration
(right), watermarked to match the Task 8 figures.

Reproducibility note, same discipline as ``scripts/train_all.py``: no
``generated_utc``/wall-clock field is written into ``external.json`` itself
-- ``transfer_study`` is a deterministic function of ``seed``, so a signed
artifact (Task 11) must reproduce byte-for-byte on an honest rerun. This
script's own stdout reports wall-clock for the human running it, but that
number never enters the JSON.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from nightingale.external import TARGET_SITES, transfer_study

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = REPO_ROOT / "models"
FIGURES_ROOT = REPO_ROOT / "reports" / "figures"

SEED = 42

# "va" is the VA Long Beach hospital (an abbreviation, not a capitalisable
# word) -- str.capitalize() would render it "Va", which reads as a typo.
_SITE_DISPLAY = {"hungarian": "Hungarian", "switzerland": "Switzerland", "va": "VA"}

# Same reference palette as scripts/train_all.py's figures, kept as a small
# local copy rather than importing from a script module.
_COLOR_MODEL = "#2a78d6"
_COLOR_SECONDARY = "#eb6834"
_COLOR_NEUTRAL = "#8a8a86"
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


def _plot_external_validation(result: dict) -> Path:
    sites = list(TARGET_SITES)
    x = np.arange(len(sites))

    fig, (ax_auc, ax_ece) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=150)

    # --- Left: naive transfer ROC-AUC per site, with bootstrap CI --------
    # Uses "naive_all_rows" (the whole site, zero site-local adjustment) --
    # the headline "what does zero-effort deployment look like" number.
    # AUC is identical on the 70% subset (see the right panel's before/after
    # invariance), so either would show the same shape; naive_all_rows uses
    # every row the site has, for the tightest CI this study can report.
    auc_point = [result["sites"][s]["naive_all_rows"]["roc_auc"][0] for s in sites]
    auc_lo = [result["sites"][s]["naive_all_rows"]["roc_auc"][1] for s in sites]
    auc_hi = [result["sites"][s]["naive_all_rows"]["roc_auc"][2] for s in sites]
    auc_err = np.array([[p - lo, hi - p] for p, lo, hi in zip(auc_point, auc_lo, auc_hi)]).T

    ax_auc.axhline(0.5, linestyle="--", linewidth=1.2, color=_COLOR_NEUTRAL, label="Chance (0.5)")
    ax_auc.errorbar(
        x, auc_point, yerr=auc_err, fmt="o", markersize=8, color=_COLOR_MODEL,
        ecolor=_COLOR_MODEL, elinewidth=1.8, capsize=5, label="Naive transfer AUC (all rows)",
    )
    ax_auc.set_xticks(x)
    ax_auc.set_xticklabels([_SITE_DISPLAY[s] for s in sites])
    ax_auc.set_ylim(0.3, 1.0)
    ax_auc.set_ylabel("ROC-AUC (95% CI)")
    ax_auc.set_title("Naive transfer discrimination\n(Cleveland-trained model, unseen site, all rows)")
    ax_auc.legend(loc="lower left", frameon=False, fontsize=8.5)

    # --- Right: ECE before vs. after intercept recalibration, same 70% --
    # "naive" and "recalibrated" are the like-for-like pair: identical
    # n_evaluation rows, only the probabilities differ (see
    # nightingale.external.transfer_study's docstring for why this schema
    # -- direct siblings, not one nested under the other -- is load-bearing).
    ece_before = [result["sites"][s]["naive"]["ece"][0] for s in sites]
    ece_before_lo = [result["sites"][s]["naive"]["ece"][1] for s in sites]
    ece_before_hi = [result["sites"][s]["naive"]["ece"][2] for s in sites]
    ece_after = [result["sites"][s]["recalibrated"]["ece"][0] for s in sites]
    ece_after_lo = [result["sites"][s]["recalibrated"]["ece"][1] for s in sites]
    ece_after_hi = [result["sites"][s]["recalibrated"]["ece"][2] for s in sites]

    before_err = np.array(
        [[p - lo, hi - p] for p, lo, hi in zip(ece_before, ece_before_lo, ece_before_hi)]
    ).T
    after_err = np.array(
        [[p - lo, hi - p] for p, lo, hi in zip(ece_after, ece_after_lo, ece_after_hi)]
    ).T

    width = 0.35
    ax_ece.bar(
        x - width / 2, ece_before, width, yerr=before_err, capsize=4,
        color=_COLOR_SECONDARY, label="Before recalibration",
    )
    ax_ece.bar(
        x + width / 2, ece_after, width, yerr=after_err, capsize=4,
        color=_COLOR_MODEL, label="After intercept recalibration",
    )
    ax_ece.set_xticks(x)
    ax_ece.set_xticklabels([_SITE_DISPLAY[s] for s in sites])
    ax_ece.set_ylabel("ECE on the held-out 70% (95% CI)")
    ax_ece.set_title("Calibration before / after\n(intercept-only, fit on the 30% site-local split)")
    ax_ece.legend(loc="upper left", frameon=False, fontsize=8.5)

    fig.suptitle(
        "External validation — heart disease, Cleveland-trained model on unseen hospitals",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    _add_watermark(fig)

    out_path = FIGURES_ROOT / "external-validation.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    MODELS_ROOT.joinpath("heart-disease").mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    print("=== external validation (heart-disease, cleveland -> {hungarian,switzerland,va}) ===", flush=True)
    t0 = time.time()

    result = transfer_study(seed=SEED)

    elapsed = time.time() - t0
    print(f"  cleveland_training: n={result['cleveland_training']['n']} "
          f"chosen_params={result['cleveland_training']['chosen_params']} "
          f"calibration_method={result['cleveland_training']['calibration_method']}", flush=True)

    for site in TARGET_SITES:
        block = result["sites"][site]
        all_rows = block["naive_all_rows"]
        naive = block["naive"]
        recal = block["recalibrated"]
        print(
            f"  {site}: n={block['n']} prevalence={block['prevalence']:.4f}  "
            f"naive_all_rows_roc_auc={all_rows['roc_auc'][0]:.4f} "
            f"[{all_rows['roc_auc'][1]:.4f}, {all_rows['roc_auc'][2]:.4f}]  "
            f"naive_all_rows_cal_intercept={all_rows['calibration_intercept']:.4f}  "
            f"naive_all_rows_cal_slope={all_rows['calibration_slope']:.4f}",
            flush=True,
        )
        print(
            f"    recalibration (like-for-like, n_cal={block['n_calibration']} "
            f"n_eval={block['n_evaluation']}): fitted_intercept={block['fitted_intercept']:.4f}  "
            f"ece_naive={naive['ece'][0]:.4f} ece_recalibrated={recal['ece'][0]:.4f}  "
            f"roc_auc_naive={naive['roc_auc'][0]:.4f} roc_auc_recalibrated={recal['roc_auc'][0]:.4f} "
            f"(must match -- AUC is recalibration-invariant)",
            flush=True,
        )

    out_path = MODELS_ROOT / "heart-disease" / "external.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(f"  wrote {out_path.relative_to(REPO_ROOT)}", flush=True)

    fig_path = _plot_external_validation(result)
    print(f"  wrote {fig_path.relative_to(REPO_ROOT)}", flush=True)

    print(f"\nTotal wall-clock: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
