# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""``nightingale`` — one command-line entry point over the whole pipeline.

Nine subcommands, one per stage of this project's pipeline:
``fetch``, ``clean``, ``train``, ``evaluate``, ``external``, ``export``,
``sign``, ``verify``, ``predict``. Every subcommand owns its own flags —
there is deliberately NO top-level ``--condition`` (or any other shared
flag): argparse silently lets a subparser's flag shadow a same-named
top-level one (whichever is given LAST on the command line wins, with no
warning), so the only safe fix is never defining the same flag at both
levels. :func:`build_parser` is asserted, by an introspection test, to
carry no top-level ``--condition`` option for exactly this reason.

Each subcommand is a THIN wrapper: the actual work happens in the pipeline
modules, each already tested on its own (:mod:`nightingale.fetch`,
:mod:`nightingale.clean`, ``scripts/train_all.py``,
:mod:`nightingale.evaluate`, :mod:`nightingale.external`,
:mod:`nightingale.export`, :mod:`nightingale.provenance`). This module adds
no modelling logic of its own.

``train`` calls ``scripts/train_all.py``'s ``run(slugs)`` directly (loaded
via :func:`_load_train_all`, since ``scripts/`` is not an installed
package) rather than shelling out via ``subprocess`` — the script's own
artifact-writing, figure-rendering, ``run_meta.json``-merging behaviour is
reused byte for byte, not reimplemented.

Long-running commands (``train``, which can take minutes per condition —
diabetes alone trains on 253,680 rows — and ``external``, one nested-CV fit)
are never exercised end-to-end by this project's test suite; only their
argument routing is, via monkeypatched stand-ins for
:func:`_load_train_all` and :func:`nightingale.external.transfer_study`.

``predict`` is the one subcommand with no equivalent script: it loads a
condition's committed ``models/<slug>/model.json`` and scores one row
through :func:`nightingale.export.predict` — the same reference walker
``docs/assets/walker.js`` mirrors — never a re-fit model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

from nightingale.conditions import CONDITIONS
from nightingale.evaluate import evaluate_oof
from nightingale.export import export_model, load_model
from nightingale.export import predict as export_predict
from nightingale.external import transfer_study
from nightingale.provenance import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_PUBKEY_PATH,
    REPO_ROOT,
    build_manifest,
    discover_artifacts,
    fingerprint,
    sign_manifest,
)
from nightingale.provenance import verify as provenance_verify

SCRIPTS_DIR = REPO_ROOT / "scripts"

# The six condition slugs, alphabetically — the canonical "valid slugs" list
# printed on an unknown-slug error, and the default set of conditions a
# subcommand without an explicit --condition operates on.
VALID_SLUGS = sorted(CONDITIONS)


# ---------------------------------------------------------------------------
# scripts/train_all.py, loaded as a module (not a subprocess call)
# ---------------------------------------------------------------------------


def _load_script_module(name: str) -> ModuleType:
    """Load ``scripts/<name>.py`` as a module, without a ``scripts/__init__.py``.

    ``scripts/`` is deliberately not an installed package (it holds
    operator entry points, not library code), so it can't be imported with
    a plain ``import scripts.train_all``. This loads the file directly from
    its path instead.
    """
    path = SCRIPTS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"nightingale_scripts_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None  # narrows the type for mypy; always true for a file path
    spec.loader.exec_module(module)
    return module


def _load_train_all() -> ModuleType:
    """``scripts/train_all.py``, loaded fresh. A thin, monkeypatchable seam for tests."""
    return _load_script_module("train_all")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _json_safe(obj):
    """Replace NaN/Infinity (not valid JSON tokens) with ``None``, recursively.

    ``evaluate_oof``'s reliability bins carry NaN for empty bins, and
    :func:`nightingale.provenance.fingerprint` records ``inf`` for a
    structurally non-matching condition — both need to survive a trip
    through ``--json`` without ``json.dumps(..., allow_nan=False)`` raising.
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    return obj


def _print_json(payload: dict) -> None:
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False))


def _provenance_line(provenance: dict) -> str:
    return (
        f"provenance: commit={provenance.get('commit')} "
        f"built_utc={provenance.get('built_utc')} "
        f"author={provenance.get('author')}"
    )


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def _cmd_fetch(args: argparse.Namespace) -> int:
    from nightingale.fetch import fetch

    for slug in args.condition or VALID_SLUGS:
        path = fetch(slug, force=args.force)
        print(f"{slug}: fetched -> {path}")
    return 0


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------


def _cmd_clean(args: argparse.Namespace) -> int:
    from nightingale.clean import clean

    for slug in args.condition or VALID_SLUGS:
        path = clean(slug)
        print(f"{slug}: cleaned -> {path}")
    return 0


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def _cmd_train(args: argparse.Namespace) -> int:
    slugs = args.condition or list(CONDITIONS)
    train_all = _load_train_all()
    train_all.run(slugs)
    return 0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def _cmd_evaluate(args: argparse.Namespace) -> int:
    slugs = args.condition or VALID_SLUGS
    results = {}
    for slug in slugs:
        oof_path = REPO_ROOT / "models" / slug / "oof_predictions.csv.gz"
        oof = pd.read_csv(oof_path, compression="gzip")
        metrics = evaluate_oof(oof)
        results[slug] = metrics
        if not args.json:
            roc = metrics["roc_auc"]
            pr = metrics["pr_auc"]
            brier = metrics["brier"]
            ece = metrics["ece"]
            print(
                f"{slug}: n={metrics['n']} prevalence={metrics['prevalence']:.4f}  "
                f"roc_auc={roc[0]:.4f} [{roc[1]:.4f}, {roc[2]:.4f}]  "
                f"pr_auc={pr[0]:.4f} [{pr[1]:.4f}, {pr[2]:.4f}]  "
                f"brier={brier[0]:.4f} [{brier[1]:.4f}, {brier[2]:.4f}]  "
                f"ece={ece[0]:.4f} [{ece[1]:.4f}, {ece[2]:.4f}]"
            )
    if args.json:
        _print_json(results)
    return 0


# ---------------------------------------------------------------------------
# external
# ---------------------------------------------------------------------------


def _cmd_external(args: argparse.Namespace) -> int:
    result = transfer_study(seed=args.seed)

    out_dir = REPO_ROOT / "models" / "heart-disease"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "external.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")

    if args.json:
        _print_json(result)
    else:
        print(f"wrote {out_path}")
        for site, block in result["sites"].items():
            naive = block["naive_all_rows"]
            print(
                f"  {site}: n={block['n']} prevalence={block['prevalence']:.4f}  "
                f"naive_roc_auc={naive['roc_auc'][0]:.4f} "
                f"[{naive['roc_auc'][1]:.4f}, {naive['roc_auc'][2]:.4f}]"
            )
    return 0


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def _cmd_export(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir) if args.out_dir else None
    for slug in args.condition or VALID_SLUGS:
        path = export_model(slug, out_dir=out_dir, seed=args.seed)
        print(f"{slug}: exported -> {path}")
    return 0


# ---------------------------------------------------------------------------
# sign
# ---------------------------------------------------------------------------


def _cmd_sign(args: argparse.Namespace) -> int:
    out_path = Path(args.out) if args.out else DEFAULT_MANIFEST_PATH
    artifacts = discover_artifacts(REPO_ROOT)
    manifest = build_manifest(artifacts, repo_root=REPO_ROOT)
    written = sign_manifest(manifest, key_path=args.key, out_path=out_path)
    print(f"signed {len(manifest['artifacts'])} artifact(s) -> {written}")
    return 0


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def _cmd_verify(args: argparse.Namespace) -> int:
    pubkey_path = Path(args.key) if args.key else DEFAULT_PUBKEY_PATH

    if args.fingerprint:
        report = fingerprint(args.fingerprint)
        if args.json:
            _print_json(
                {
                    "condition": report.condition,
                    "matches": report.matches,
                    "max_abs_diff": report.max_abs_diff,
                }
            )
        elif report.matched:
            print(f"{args.fingerprint}: matches {report.condition}")
        elif report.matches:
            print(
                f"{args.fingerprint}: ambiguous -- agrees with {len(report.matches)} "
                f"registered conditions: {', '.join(report.matches)}"
            )
        else:
            print(f"{args.fingerprint}: no match against any registered condition")
        return 0 if report.matched else 1

    manifest_path = Path(args.manifest) if args.manifest else None
    report = provenance_verify(REPO_ROOT, pubkey_path, manifest_path=manifest_path)

    if args.json:
        _print_json(
            {
                "ok": report.ok,
                "signature_valid": report.signature_valid,
                "artifacts": report.artifacts,
            }
        )
    else:
        for relpath, status in sorted(report.artifacts.items()):
            print(f"{status}: {relpath}")
        print(f"signature: {'valid' if report.signature_valid else 'INVALID'}")
        print("OK" if report.ok else "TAMPERED")
    return 0 if report.ok else 1


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def _parse_set_args(pairs: list[str] | None, valid_names: list[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for pair in pairs or []:
        if "=" not in pair:
            print(
                f"error: --set expects feature=value, got {pair!r}",
                file=sys.stderr,
            )
            sys.exit(2)
        name, _, raw_value = pair.partition("=")
        if name not in valid_names:
            print(f"error: unknown feature {name!r} for this condition", file=sys.stderr)
            print(f"valid features: {', '.join(valid_names)}", file=sys.stderr)
            sys.exit(2)
        try:
            overrides[name] = float(raw_value)
        except ValueError:
            print(
                f"error: --set {pair!r}: {raw_value!r} is not a number",
                file=sys.stderr,
            )
            sys.exit(2)
    return overrides


def _cmd_predict(args: argparse.Namespace) -> int:
    model_path = REPO_ROOT / "models" / args.condition / "model.json"
    model = load_model(model_path)
    feature_names = [f["name"] for f in model["features"]]

    overrides = _parse_set_args(args.set, feature_names)
    row = [overrides.get(name, float("nan")) for name in feature_names]

    result = export_predict(model, row)

    if args.json:
        _print_json(
            {
                "condition": args.condition,
                "p_raw": result["p_raw"],
                "p_cal": result["p_cal"],
                "verdict": result["set"],
                "provenance": model["provenance"],
            }
        )
    else:
        print(f"condition: {args.condition}")
        print(f"p_raw: {result['p_raw']:.6f}")
        print(f"p_cal: {result['p_cal']:.6f}")
        print(f"verdict: {result['set']}")
        print(_provenance_line(model["provenance"]))
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser. NO top-level ``--condition`` — see module docstring."""
    parser = argparse.ArgumentParser(
        prog="nightingale",
        description="Calibrated clinical risk models for six conditions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_p = subparsers.add_parser("fetch", help="Fetch raw data for one or more conditions.")
    fetch_p.add_argument("--condition", action="append", choices=VALID_SLUGS)
    fetch_p.add_argument("--force", action="store_true", help="Re-fetch even if cached.")
    fetch_p.set_defaults(func=_cmd_fetch)

    clean_p = subparsers.add_parser("clean", help="Clean raw data for one or more conditions.")
    clean_p.add_argument("--condition", action="append", choices=VALID_SLUGS)
    clean_p.set_defaults(func=_cmd_clean)

    train_p = subparsers.add_parser(
        "train", help="Train, evaluate, and report one or more conditions (slow)."
    )
    train_p.add_argument("--condition", action="append", choices=VALID_SLUGS)
    train_p.set_defaults(func=_cmd_train)

    evaluate_p = subparsers.add_parser(
        "evaluate", help="Recompute headline metrics from committed OOF predictions."
    )
    evaluate_p.add_argument("--condition", action="append", choices=VALID_SLUGS)
    evaluate_p.add_argument("--json", action="store_true")
    evaluate_p.set_defaults(func=_cmd_evaluate)

    external_p = subparsers.add_parser(
        "external", help="Run the four-hospital external validation study (heart-disease only)."
    )
    external_p.add_argument("--seed", type=int, default=42)
    external_p.add_argument("--json", action="store_true")
    external_p.set_defaults(func=_cmd_external)

    export_p = subparsers.add_parser(
        "export", help="Export the browser-side model.json bundle for one or more conditions."
    )
    export_p.add_argument("--condition", action="append", choices=VALID_SLUGS)
    export_p.add_argument("--out-dir", default=None)
    export_p.add_argument("--seed", type=int, default=42)
    export_p.set_defaults(func=_cmd_export)

    sign_p = subparsers.add_parser("sign", help="Build and sign the provenance manifest.")
    sign_p.add_argument(
        "--key",
        default=None,
        help="Path to the Ed25519 private key PEM. Falls back to NIGHTINGALE_SIGNING_KEY.",
    )
    sign_p.add_argument("--out", default=None, help="Manifest output path.")
    sign_p.set_defaults(func=_cmd_sign)

    verify_p = subparsers.add_parser(
        "verify", help="Verify the signed manifest, or fingerprint a lone model.json."
    )
    verify_p.add_argument("--key", default=None, help="Path to the Ed25519 public key PEM.")
    verify_p.add_argument("--manifest", default=None, help="Manifest path.")
    verify_p.add_argument(
        "--fingerprint",
        default=None,
        metavar="PATH",
        help="Identify which condition a lone model.json belongs to, instead of verifying.",
    )
    verify_p.add_argument("--json", action="store_true")
    verify_p.set_defaults(func=_cmd_verify)

    predict_p = subparsers.add_parser("predict", help="Score one row against a condition's model.")
    predict_p.add_argument("--condition", required=True, choices=VALID_SLUGS)
    predict_p.add_argument(
        "--set",
        action="append",
        metavar="feature=value",
        help="Set one feature's value (repeatable). Unset features are NaN (missing).",
    )
    predict_p.add_argument("--json", action="store_true")
    predict_p.set_defaults(func=_cmd_predict)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    exit_code = args.func(args)
    sys.exit(exit_code)


if __name__ == "__main__":  # pragma: no cover - exercised via __main__.py / console script
    main()
