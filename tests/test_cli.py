# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Tests for the ``nightingale`` CLI (:mod:`nightingale.cli`).

Covers the binding requirements from the Task 12 brief: no top-level
``--condition`` (each subcommand owns its own flags — the argparse
parent-flag shadowing gotcha); ``predict`` on breast-cancer with no
``--set`` at all still returns a valid probability (an all-NaN row is a
legitimate input, XGBoost routes it via the learned missing direction);
unknown slugs exit 2 with the six valid slugs on stderr; ``verify`` exits 0
on the real, untampered tree and 1 after a byte is flipped in a temp copy;
and at least two true subprocess invocations proving the installed
``nightingale`` console-script entry point (and ``python -m nightingale``)
actually work, not just ``main()`` called in-process.

``train`` and ``external`` are long-running (diabetes alone is minutes of
wall-clock) and are never actually run here — only their argument routing
is checked, via a monkeypatched stand-in for
:func:`nightingale.cli._load_train_all` and for
:func:`nightingale.external.transfer_study` respectively.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import nightingale.cli as cli
from nightingale.cli import VALID_SLUGS, build_parser, main

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run_subprocess(args: list[str], cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    env = {"PYTHONPATH": "src"}
    import os

    full_env = dict(os.environ)
    full_env.update(env)
    return subprocess.run(
        [PYTHON, "-m", "nightingale", *args],
        cwd=cwd,
        env=full_env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _run_cli(monkeypatch, args: list[str]) -> int:
    """Call ``main`` in-process and capture its ``SystemExit`` code."""
    monkeypatch.setattr(sys, "argv", ["nightingale", *args])
    with pytest.raises(SystemExit) as excinfo:
        main()
    return excinfo.value.code


# ---------------------------------------------------------------------------
# No top-level --condition (argparse shadowing gotcha)
# ---------------------------------------------------------------------------


def test_no_top_level_condition_flag():
    parser = build_parser()
    top_level_options = {
        option for action in parser._actions for option in action.option_strings
    }
    assert "--condition" not in top_level_options


def test_every_subcommand_that_takes_a_condition_owns_its_own_flag():
    parser = build_parser()
    subparsers_action = next(
        a for a in parser._actions if isinstance(a, __import__("argparse")._SubParsersAction)
    )
    for name in ("fetch", "clean", "train", "evaluate", "export", "predict"):
        sub = subparsers_action.choices[name]
        options = {opt for action in sub._actions for opt in action.option_strings}
        assert "--condition" in options, f"{name} should own a --condition flag"


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def test_predict_breast_cancer_no_set_returns_valid_probability(monkeypatch, capsys):
    exit_code = _run_cli(monkeypatch, ["predict", "--condition", "breast-cancer"])
    assert exit_code == 0
    out = capsys.readouterr().out
    lines = dict(line.split(": ", 1) for line in out.splitlines() if ": " in line)
    # p_raw (pre-calibration) is strictly inside (0, 1) by construction (a
    # sigmoid output); p_cal can legitimately clamp to the isotonic
    # calibrator's saturated endpoint (exactly 0.0 or 1.0) for an all-NaN
    # row on breast-cancer, so it is checked on the closed interval.
    p_raw = float(lines["p_raw"])
    p_cal = float(lines["p_cal"])
    assert 0.0 < p_raw < 1.0
    assert 0.0 <= p_cal <= 1.0
    assert lines["verdict"] in ("positive", "negative", "uncertain")
    assert "provenance" in lines


def test_predict_with_set_overrides_a_feature(monkeypatch, capsys):
    model = json.loads((REPO_ROOT / "models" / "breast-cancer" / "model.json").read_text())
    feature_name = model["features"][0]["name"]
    exit_code = _run_cli(
        monkeypatch,
        ["predict", "--condition", "breast-cancer", "--set", f"{feature_name}=1.0"],
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "p_cal" in out


def test_predict_json_output_is_valid_json(monkeypatch, capsys):
    exit_code = _run_cli(monkeypatch, ["predict", "--condition", "breast-cancer", "--json"])
    assert exit_code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["condition"] == "breast-cancer"
    assert 0.0 < payload["p_raw"] < 1.0
    assert 0.0 <= payload["p_cal"] <= 1.0
    assert payload["verdict"] in ("positive", "negative", "uncertain")
    assert "commit" in payload["provenance"]


def test_predict_unknown_feature_name_exits_2(monkeypatch, capsys):
    exit_code = _run_cli(
        monkeypatch,
        ["predict", "--condition", "breast-cancer", "--set", "not_a_real_feature=1.0"],
    )
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "not_a_real_feature" in err
    assert "valid features" in err


def test_predict_unknown_slug_exits_2_with_six_valid_slugs_on_stderr(monkeypatch, capsys):
    with pytest.raises(SystemExit) as excinfo:
        monkeypatch.setattr(sys, "argv", ["nightingale", "predict", "--condition", "not-a-slug"])
        build_parser().parse_args(["predict", "--condition", "not-a-slug"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    for slug in VALID_SLUGS:
        assert slug in err
    assert len(VALID_SLUGS) == 6


@pytest.mark.parametrize("command", ["fetch", "clean", "train", "evaluate", "export"])
def test_unknown_slug_exits_2_with_six_valid_slugs_on_stderr(command, capsys):
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args([command, "--condition", "not-a-slug"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    for slug in VALID_SLUGS:
        assert slug in err


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def test_verify_exit_code_0_on_the_real_clean_tree(monkeypatch):
    exit_code = _run_cli(monkeypatch, ["verify"])
    assert exit_code == 0


def test_verify_exit_code_1_after_tampering_a_temp_copy(monkeypatch, tmp_path):
    tmp_repo = tmp_path / "nightingale_copy"
    shutil.copytree(
        REPO_ROOT,
        tmp_repo,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".venv"),
    )

    # Flip one byte of a manifest-listed artifact.
    manifest = json.loads((tmp_repo / "provenance" / "manifest.json").read_text())
    relpath = next(iter(manifest["artifacts"]))
    target = tmp_repo / relpath
    data = bytearray(target.read_bytes())
    data[0] ^= 0xFF
    target.write_bytes(bytes(data))

    monkeypatch.setattr(cli, "REPO_ROOT", tmp_repo)
    exit_code = _run_cli(monkeypatch, ["verify"])
    assert exit_code == 1


def test_verify_json_output(monkeypatch, capsys):
    exit_code = _run_cli(monkeypatch, ["verify", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["signature_valid"] is True


def test_verify_fingerprint_matches_own_condition(monkeypatch, capsys):
    model_path = REPO_ROOT / "models" / "breast-cancer" / "model.json"
    exit_code = _run_cli(monkeypatch, ["verify", "--fingerprint", str(model_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "breast-cancer" in out


# ---------------------------------------------------------------------------
# train / external — argument routing only, never a real (slow) run
# ---------------------------------------------------------------------------


def test_train_routes_condition_to_train_all_run(monkeypatch):
    calls = []
    fake_module = SimpleNamespace(run=lambda slugs: calls.append(list(slugs)))
    monkeypatch.setattr(cli, "_load_train_all", lambda: fake_module)

    exit_code = _run_cli(
        monkeypatch, ["train", "--condition", "breast-cancer", "--condition", "diabetes"]
    )
    assert exit_code == 0
    assert calls == [["breast-cancer", "diabetes"]]


def test_train_defaults_to_all_six_conditions(monkeypatch):
    calls = []
    fake_module = SimpleNamespace(run=lambda slugs: calls.append(list(slugs)))
    monkeypatch.setattr(cli, "_load_train_all", lambda: fake_module)

    exit_code = _run_cli(monkeypatch, ["train"])
    assert exit_code == 0
    assert set(calls[0]) == set(VALID_SLUGS)


def test_external_routes_seed_to_transfer_study(monkeypatch, tmp_path):
    calls = []

    def fake_transfer_study(seed):
        calls.append(seed)
        return {"sites": {"hungarian": {"n": 1, "prevalence": 0.5, "naive_all_rows": {"roc_auc": [0.6, 0.5, 0.7]}}}}

    monkeypatch.setattr(cli, "transfer_study", fake_transfer_study)
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)

    exit_code = _run_cli(monkeypatch, ["external", "--seed", "7"])
    assert exit_code == 0
    assert calls == [7]
    assert (tmp_path / "models" / "heart-disease" / "external.json").is_file()


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_breast_cancer_json(monkeypatch, capsys):
    exit_code = _run_cli(monkeypatch, ["evaluate", "--condition", "breast-cancer", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "breast-cancer" in payload
    assert 0.0 <= payload["breast-cancer"]["roc_auc"][0] <= 1.0


# ---------------------------------------------------------------------------
# sign — argument routing (no real signing key needed; a fresh one is used)
# ---------------------------------------------------------------------------


def test_sign_writes_a_manifest(monkeypatch, tmp_path):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

    key = Ed25519PrivateKey.generate()
    key_path = tmp_path / "key.pem"
    key_path.write_bytes(
        key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    )
    out_path = tmp_path / "manifest.json"

    exit_code = _run_cli(
        monkeypatch, ["sign", "--key", str(key_path), "--out", str(out_path)]
    )
    assert exit_code == 0
    assert out_path.is_file()
    manifest = json.loads(out_path.read_text())
    assert "signature_b64" in manifest


# ---------------------------------------------------------------------------
# export — argument routing
# ---------------------------------------------------------------------------


def test_export_writes_model_json_to_out_dir(monkeypatch, tmp_path):
    exit_code = _run_cli(
        monkeypatch,
        ["export", "--condition", "breast-cancer", "--out-dir", str(tmp_path)],
    )
    assert exit_code == 0
    assert (tmp_path / "breast-cancer" / "model.json").is_file()


# ---------------------------------------------------------------------------
# True subprocess tests — the installed console-script entry point and
# `python -m nightingale`, not just main() called in-process.
# ---------------------------------------------------------------------------


def test_subprocess_python_dash_m_nightingale_predict():
    result = _run_subprocess(["predict", "--condition", "breast-cancer", "--json"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert 0.0 < payload["p_raw"] < 1.0


def test_subprocess_python_dash_m_nightingale_unknown_slug_exits_2():
    result = _run_subprocess(["predict", "--condition", "not-a-real-slug"])
    assert result.returncode == 2
    for slug in VALID_SLUGS:
        assert slug in result.stderr


def test_subprocess_console_script_entry_point():
    nightingale_bin = shutil.which("nightingale")
    if nightingale_bin is None:
        pytest.skip("nightingale console script not installed on PATH")
    result = subprocess.run(
        [nightingale_bin, "verify"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
