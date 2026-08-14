#!/usr/bin/env python3
# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT

"""Assemble ``docs/index.html`` from the committed, signed artifacts.

The dashboard is one self-contained file: the six trained models, their
metrics, the four-hospital transfer study and the inference engine are all
baked into it at build time, so the page runs every model in the visitor's
browser with no server and no runtime download but Chart.js.

The shape is deliberate. ``scripts/dashboard_template.html`` holds all the
markup, style and behaviour and exactly one ``/*__DATA__*/`` placeholder;
this script produces the JSON that replaces it. Nothing here writes copy and
nothing in the template writes numbers, so a number can only reach the page
by being read out of an artifact that ``nightingale verify`` covers.

Two properties are load-bearing and are tested in
``tests/test_build_dashboard.py``:

*   **Byte-determinism.** No wall-clock time, no set iteration, no unsorted
    dict; the out-of-fold sample is drawn from a fixed seed with the stdlib
    Mersenne Twister. Two builds of the same artifacts are the same bytes, so
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
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPO_ROOT / "scripts" / "dashboard_template.html"
WALKER_PATH = REPO_ROOT / "docs" / "assets" / "walker.js"
OUTPUT_PATH = REPO_ROOT / "docs" / "index.html"
MODELS_DIR = REPO_ROOT / "models"
PUBKEY_PATH = REPO_ROOT / "provenance" / "pubkey.pem"
CLEANED_DIR = REPO_ROOT / "data" / "cleaned"

DATA_PLACEHOLDER = "/*__DATA__*/"
WALKER_PLACEHOLDER = "/*__WALKER__*/"

# Chart.js draws the two cartesian plates in the calibration lab. It is
# pinned to an exact version and guarded by a Subresource Integrity hash, so
# a compromised CDN cannot change what this page computes; the page also
# carries its own SVG renderer and uses it whenever Chart.js is absent, which
# is what makes the file work opened straight off a disk.
CHARTJS_URL = "https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"
CHARTJS_SRI = "sha384-JUh163oCRItcbPme8pYnROHQMC6fNKTBWtRG3I3I0erJkzNgL7uxKlNwcrcFKeqF"

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
    be reproduced with one ``openssl`` pipe, which is what the footer prints.
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
        # plate and roughly halves the payload against full float repr.
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


# --------------------------------------------------------------------------
# Per-condition payload
# --------------------------------------------------------------------------


def _feature_schema(feature: dict[str, Any]) -> dict[str, Any]:
    """One form field, described entirely by the exported schema."""
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


def _condition_notes(slug: str, metrics: dict[str, Any], q_hat: float) -> list[str]:
    """Honest caveats that belong beside this condition's output, not in a footnote."""
    notes: list[str] = []
    if q_hat == 0.0:
        share = saturated_fraction(slug)
        notes.append(
            "This model can never answer “uncertain”. Its conformal threshold "
            f"q̂ is exactly 0, because {share:.2%} of its out-of-fold "
            "calibrated probabilities are pinned to 0 or 1 — the cohort is "
            "close to separable and the isotonic calibrator saturates. Every "
            "verdict below therefore comes from the plain 0.5 cut, and the "
            "absence of “uncertain” says nothing about how sure the model is."
        )
    if slug == "diabetes":
        notes.append(
            "Every field here is a survey answer, coded as an integer. Age is "
            "a 13-level band, not a count of years, and the outcome is what "
            "the respondent said about themselves — not a clinical diagnosis."
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
            "contribution_slots": 6,
        },
    }
    return _finite(payload)


def render_page() -> str:
    """The finished page as a string, with nothing read from the clock."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if template.count(DATA_PLACEHOLDER) != 1:
        raise SystemExit(f"template must contain exactly one {DATA_PLACEHOLDER}")
    if template.count(WALKER_PLACEHOLDER) != 1:
        raise SystemExit(f"template must contain exactly one {WALKER_PLACEHOLDER}")

    blob = json.dumps(build_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    # </script> inside a JSON string would close the tag early; escaping the
    # slash is invisible to JSON.parse and keeps the parser inside the block.
    blob = blob.replace("</", "<\\/")

    page = template.replace(WALKER_PLACEHOLDER, WALKER_PATH.read_text(encoding="utf-8"))
    page = page.replace(DATA_PLACEHOLDER, blob)
    page = page.replace("__CHARTJS_URL__", CHARTJS_URL)
    page = page.replace("__CHARTJS_SRI__", CHARTJS_SRI)
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
