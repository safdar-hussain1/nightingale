# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""Tests for :mod:`nightingale.provenance`: signed manifest + fingerprint.

Two separate claims, two separate mechanisms:

1. **The manifest** proves nothing in the working tree has been altered
   since it was signed -- a SHA-256 per artifact, Ed25519 over the sorted
   set of hashes. ``verify`` recomputes every hash from disk and checks the
   signature; it never regenerates an artifact to compare against, because
   regeneration is not byte-reproducible across commits (a bundle's
   ``provenance.built_utc`` is the HEAD commit's timestamp).
2. **The fingerprint** proves a lone ``model.json`` -- found anywhere, under
   any name, with its ``provenance`` block stripped -- is a copy of one of
   this project's six trees. For each registered condition it runs THAT
   CONDITION'S OWN committed canary inputs (never the candidate's) through
   both the reference's trees and the candidate's trees, and compares the
   two ``p_raw`` vectors. Pinning the inputs to the reference closes the
   forgery an earlier, candidate-inputs design allowed (pick 8 copies of
   one saturating input and forge a match against anything); comparing
   ``p_raw`` rather than ``p_cal`` closes the other one (an isotonic
   calibrator's saturated plateau hides a tampered tree's margin shift).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_pem_private_key,
)

from nightingale.export import REPO_ROOT, _LEAF
from nightingale.provenance import (
    ARTIFACT_GLOBS,
    DEFAULT_PUBKEY_PATH,
    FingerprintReport,
    VerifyReport,
    build_manifest,
    canonical_json_bytes,
    discover_artifacts,
    fingerprint,
    sign_manifest,
    verify,
)

# The private signing key has no default location. It is read from this
# environment variable and nothing else -- the same way ``nightingale sign``
# reads it (see nightingale.provenance._load_private_key) -- so the one test
# that needs the key skips, naming the variable, wherever it is not set.
SIGNING_KEY_ENV = "NIGHTINGALE_SIGNING_KEY"

CONDITIONS_SLUGS = [
    "breast-cancer",
    "cervical-cancer",
    "diabetes",
    "heart-disease",
    "kidney-disease",
    "liver-disease",
]


def _write_keypair(tmp_path: Path) -> tuple[Path, Path]:
    """A fresh Ed25519 keypair on disk under ``tmp_path``; returns (priv, pub)."""
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    key = Ed25519PrivateKey.generate()
    priv_path = tmp_path / "key.pem"
    pub_path = tmp_path / "key.pub.pem"
    priv_path.write_bytes(key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()))
    pub_path.write_bytes(
        key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    )
    return priv_path, pub_path


def _tiny_repo(tmp_path: Path) -> Path:
    """A minimal artifact tree under ``tmp_path`` matching the manifest globs."""
    root = tmp_path / "repo"
    (root / "models" / "breast-cancer").mkdir(parents=True)
    (root / "data" / "cleaned").mkdir(parents=True)
    (root / "models" / "breast-cancer" / "model.json").write_text('{"a": 1}')
    (root / "models" / "breast-cancer" / "metrics.json").write_text('{"b": 2}')
    (root / "data" / "cleaned" / "breast-cancer.csv.gz").write_bytes(b"\x1f\x8b\x00fake")
    return root


# --------------------------------------------------------------------------
# canonical_json_bytes / discover_artifacts / build_manifest
# --------------------------------------------------------------------------


def test_canonical_json_bytes_sorted_no_whitespace():
    payload = canonical_json_bytes({"b": 1, "a": {"z": 1, "y": 2}})
    assert payload == b'{"a":{"y":2,"z":1},"b":1}'


def test_discover_artifacts_only_lists_existing_files(tmp_path):
    root = _tiny_repo(tmp_path)
    found = discover_artifacts(root)
    relpaths = sorted(str(p.relative_to(root)) for p in found)
    assert relpaths == [
        "data/cleaned/breast-cancer.csv.gz",
        "models/breast-cancer/metrics.json",
        "models/breast-cancer/model.json",
    ]
    # docs/index.html and reports/figures/*.png are absent in this tree and
    # must not appear -- absent at signing time means not listed.
    assert not any("docs" in r or "figures" in r for r in relpaths)


def test_build_manifest_hashes_and_sorts(tmp_path):
    root = _tiny_repo(tmp_path)
    paths = discover_artifacts(root)
    manifest = build_manifest(paths, root)

    assert list(manifest["artifacts"].keys()) == sorted(manifest["artifacts"].keys())
    expected = hashlib.sha256(
        (root / "models" / "breast-cancer" / "model.json").read_bytes()
    ).hexdigest()
    assert manifest["artifacts"]["models/breast-cancer/model.json"] == expected
    assert set(manifest["meta"]) >= {"built_utc", "commit", "author"}
    assert manifest["meta"]["author"] == "Safdar Hussain"


# --------------------------------------------------------------------------
# sign / verify round trip
# --------------------------------------------------------------------------


def test_sign_and_verify_roundtrip_ok(tmp_path):
    root = _tiny_repo(tmp_path)
    priv, pub = _write_keypair(tmp_path)
    paths = discover_artifacts(root)
    manifest = build_manifest(paths, root)

    out_path = root / "provenance" / "manifest.json"
    written = sign_manifest(manifest, key_path=priv, out_path=out_path)
    assert written == out_path
    assert out_path.is_file()

    report = verify(root, pub, manifest_path=out_path)
    assert isinstance(report, VerifyReport)
    assert report.signature_valid is True
    assert set(report.artifacts.values()) == {"OK"}
    assert report.ok is True


def test_verify_flip_one_byte_marks_only_that_artifact_tampered(tmp_path):
    root = _tiny_repo(tmp_path)
    priv, pub = _write_keypair(tmp_path)
    manifest = build_manifest(discover_artifacts(root), root)
    out_path = sign_manifest(manifest, key_path=priv, out_path=root / "provenance" / "manifest.json")

    target = root / "models" / "breast-cancer" / "model.json"
    data = bytearray(target.read_bytes())
    data[0] ^= 0xFF
    target.write_bytes(bytes(data))

    report = verify(root, pub, manifest_path=out_path)
    assert report.artifacts["models/breast-cancer/model.json"] == "TAMPERED"
    assert report.artifacts["models/breast-cancer/metrics.json"] == "OK"
    assert report.artifacts["data/cleaned/breast-cancer.csv.gz"] == "OK"
    # The manifest bytes themselves were untouched, so the signature over
    # them is still valid -- it is the artifact hash that disagrees.
    assert report.signature_valid is True
    assert report.ok is False


def test_verify_missing_artifact(tmp_path):
    root = _tiny_repo(tmp_path)
    priv, pub = _write_keypair(tmp_path)
    manifest = build_manifest(discover_artifacts(root), root)
    out_path = sign_manifest(manifest, key_path=priv, out_path=root / "provenance" / "manifest.json")

    (root / "models" / "breast-cancer" / "metrics.json").unlink()

    report = verify(root, pub, manifest_path=out_path)
    assert report.artifacts["models/breast-cancer/metrics.json"] == "MISSING"
    assert report.ok is False


def test_verify_with_different_key_signature_invalid(tmp_path):
    root = _tiny_repo(tmp_path)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    priv_a, _pub_a = _write_keypair(tmp_path / "a")
    _priv_b, pub_b = _write_keypair(tmp_path / "b")
    manifest = build_manifest(discover_artifacts(root), root)
    out_path = sign_manifest(
        manifest, key_path=priv_a, out_path=root / "provenance" / "manifest.json"
    )

    report = verify(root, pub_b, manifest_path=out_path)
    assert report.signature_valid is False
    assert report.ok is False
    # verify() checks the signature FIRST and refuses to hash anything once
    # it fails -- artifacts are untouched on disk, but that is not reported
    # as "OK", since nothing was actually checked against them.
    assert set(report.artifacts.values()) == {"UNVERIFIED"}


def test_sign_manifest_reads_key_path_from_env(tmp_path, monkeypatch):
    root = _tiny_repo(tmp_path)
    priv, pub = _write_keypair(tmp_path)
    monkeypatch.setenv("NIGHTINGALE_SIGNING_KEY", str(priv))
    manifest = build_manifest(discover_artifacts(root), root)

    out_path = sign_manifest(manifest, out_path=root / "provenance" / "manifest.json")
    report = verify(root, pub, manifest_path=out_path)
    assert report.signature_valid is True


# --------------------------------------------------------------------------
# Path handling -- signature checked before any hashing, and every
# relpath must resolve under repo_root even when the signature is valid.
# --------------------------------------------------------------------------


def test_verify_rejects_traversal_and_absolute_paths_without_reading_them(tmp_path):
    root = _tiny_repo(tmp_path)
    priv, pub = _write_keypair(tmp_path)

    outside = tmp_path / "secret.txt"
    outside.write_text("outside the checkout")

    manifest = build_manifest(discover_artifacts(root), root)
    # Genuinely, validly signed manifest that ALSO lists a path outside the
    # repo two different ways -- this is what a signature check alone
    # cannot catch, since the signature is only over whatever content the
    # manifest actually has.
    manifest["artifacts"]["../secret.txt"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    manifest["artifacts"][str(outside)] = hashlib.sha256(outside.read_bytes()).hexdigest()
    out_path = sign_manifest(manifest, key_path=priv, out_path=root / "provenance" / "manifest.json")

    report = verify(root, pub, manifest_path=out_path)
    assert report.signature_valid is True  # the manifest itself is honestly signed
    assert report.artifacts["../secret.txt"] == "INVALID_PATH"
    assert report.artifacts[str(outside)] == "INVALID_PATH"
    assert report.ok is False


def test_verify_never_hashes_anything_when_signature_invalid(tmp_path):
    """An attacker who edits the manifest (e.g. to add a path-escaping entry)

    without the private key invalidates the signature; verify must refuse
    to hash ANY listed path in that case, including the ones that would
    have been legitimate, and must not touch the escaping path either.
    """
    root = _tiny_repo(tmp_path)
    priv, pub = _write_keypair(tmp_path)
    manifest = build_manifest(discover_artifacts(root), root)
    out_path = sign_manifest(manifest, key_path=priv, out_path=root / "provenance" / "manifest.json")

    outside = tmp_path / "secret.txt"
    outside.write_text("outside the checkout")
    raw = json.loads(out_path.read_text())
    raw["artifacts"]["../secret.txt"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    out_path.write_text(json.dumps(raw))  # NOT re-signed -- signature now stale

    report = verify(root, pub, manifest_path=out_path)
    assert report.signature_valid is False
    assert set(report.artifacts.values()) == {"UNVERIFIED"}
    assert report.ok is False


# --------------------------------------------------------------------------
# Fail closed on malformed signature material.
# --------------------------------------------------------------------------


def test_verify_garbage_base64_signature_is_invalid_not_a_crash(tmp_path):
    root = _tiny_repo(tmp_path)
    priv, pub = _write_keypair(tmp_path)
    manifest = build_manifest(discover_artifacts(root), root)
    out_path = sign_manifest(manifest, key_path=priv, out_path=root / "provenance" / "manifest.json")

    raw = json.loads(out_path.read_text())
    raw["signature_b64"] = "not valid base64 !!! ??"
    out_path.write_text(json.dumps(raw))

    report = verify(root, pub, manifest_path=out_path)
    assert report.signature_valid is False
    assert report.ok is False


def test_verify_missing_signature_key_is_invalid_not_a_crash(tmp_path):
    root = _tiny_repo(tmp_path)
    priv, pub = _write_keypair(tmp_path)
    manifest = build_manifest(discover_artifacts(root), root)
    out_path = sign_manifest(manifest, key_path=priv, out_path=root / "provenance" / "manifest.json")

    raw = json.loads(out_path.read_text())
    del raw["signature_b64"]
    out_path.write_text(json.dumps(raw))

    report = verify(root, pub, manifest_path=out_path)
    assert report.signature_valid is False
    assert report.ok is False


# --------------------------------------------------------------------------
# Editing the manifest's own content -- an artifact hash, or meta --
# must invalidate the signature. This is the exact attack signing exists
# to catch.
# --------------------------------------------------------------------------


def test_verify_edited_artifact_hash_invalidates_signature(tmp_path):
    root = _tiny_repo(tmp_path)
    priv, pub = _write_keypair(tmp_path)
    manifest = build_manifest(discover_artifacts(root), root)
    out_path = sign_manifest(manifest, key_path=priv, out_path=root / "provenance" / "manifest.json")

    # Tamper the artifact on disk AND rewrite the manifest's recorded hash
    # to match the tampered bytes -- the attack signing is meant to defeat.
    target = root / "models" / "breast-cancer" / "model.json"
    data = bytearray(target.read_bytes())
    data[0] ^= 0xFF
    target.write_bytes(bytes(data))

    raw = json.loads(out_path.read_text())
    raw["artifacts"]["models/breast-cancer/model.json"] = hashlib.sha256(bytes(data)).hexdigest()
    out_path.write_text(json.dumps(raw))  # NOT re-signed

    report = verify(root, pub, manifest_path=out_path)
    assert report.signature_valid is False
    assert report.ok is False


def test_verify_edited_meta_invalidates_signature(tmp_path):
    root = _tiny_repo(tmp_path)
    priv, pub = _write_keypair(tmp_path)
    manifest = build_manifest(discover_artifacts(root), root)
    out_path = sign_manifest(manifest, key_path=priv, out_path=root / "provenance" / "manifest.json")

    raw = json.loads(out_path.read_text())
    raw["meta"]["author"] = "Someone Else"
    out_path.write_text(json.dumps(raw))  # NOT re-signed

    report = verify(root, pub, manifest_path=out_path)
    assert report.signature_valid is False
    assert report.ok is False


# --------------------------------------------------------------------------
# The committed manifest, against the committed repo -- no synthetic
# fixtures. Every other verify() test above signs its own manifest for a
# tmp_path tree, so none of them ever check that the actual, committed
# provenance/manifest.json currently validates against the actual,
# committed models/ tree it claims to cover. Tampering with a real,
# committed artifact would be invisible without this test.
# --------------------------------------------------------------------------


def test_verify_real_committed_manifest_against_real_repo():
    report = verify(REPO_ROOT, DEFAULT_PUBKEY_PATH)
    assert report.signature_valid is True
    assert report.ok is True
    assert set(report.artifacts.values()) == {"OK"}
    # Pin the artifact count so a silently shrinking manifest (e.g. globs
    # quietly matching fewer files) also fails this test.
    assert len(report.artifacts) == 34


# --------------------------------------------------------------------------
# fingerprint
# --------------------------------------------------------------------------


def _load(slug: str) -> dict:
    return json.loads((REPO_ROOT / "models" / slug / "model.json").read_text())


@pytest.mark.parametrize("slug", CONDITIONS_SLUGS)
def test_fingerprint_matches_own_condition(tmp_path, slug):
    report = fingerprint(REPO_ROOT / "models" / slug / "model.json")
    assert isinstance(report, FingerprintReport)
    assert report.condition == slug
    assert report.matched is True


def test_fingerprint_renamed_and_stripped_copy_still_matches(tmp_path):
    data = _load("breast-cancer")
    del data["provenance"]
    data["condition"] = "totally-different-name"
    copy_path = tmp_path / "some" / "arbitrary" / "path" / "weights.json"
    copy_path.parent.mkdir(parents=True)
    copy_path.write_text(json.dumps(data))

    report = fingerprint(copy_path)
    assert report.condition == "breast-cancer"
    assert report.matched is True


def test_fingerprint_perturbed_leaves_no_match(tmp_path):
    data = _load("breast-cancer")
    for tree in data["trees"]:
        for node in tree:
            if node["f"] == _LEAF:
                node["v"] = node["v"] + 1e-3
    copy_path = tmp_path / "perturbed.json"
    copy_path.write_text(json.dumps(data))

    report = fingerprint(copy_path)
    assert report.condition is None
    assert report.matched is False
    assert "breast-cancer" not in report.matches


def test_fingerprint_cross_condition_collision_rejected(tmp_path):
    """A +1e-2 leaf perturbation saturates breast-cancer's canary ``p_cal``
    vector to all-``1.0`` -- which happens to equal kidney-disease's own
    fully-saturated canary ``p_cal`` vector. A p_cal-based fingerprint would
    have reported this tampered breast-cancer copy as an AUTHENTIC
    kidney-disease model. The p_raw-based, reference-pinned design must
    reject it outright, not merely reject it as breast-cancer.
    """
    data = _load("breast-cancer")
    for tree in data["trees"]:
        for node in tree:
            if node["f"] == _LEAF:
                node["v"] = node["v"] + 1e-2
    copy_path = tmp_path / "perturbed.json"
    copy_path.write_text(json.dumps(data))

    report = fingerprint(copy_path)
    assert report.condition != "kidney-disease"
    assert report.condition is None
    assert report.matches == []


# --------------------------------------------------------------------------
# Guard tests: the private key must never be inside the repo, and .gitignore
# must actually keep it (and private/) out.
# --------------------------------------------------------------------------


def test_gitignore_blocks_key_and_private_paths():
    for target in ("keys/x", "nested/dir/keys/x", "private/x"):
        result = subprocess.run(
            ["git", "check-ignore", target],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{target} is not gitignored"


def test_no_private_pem_or_private_dir_tracked_by_git():
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = result.stdout.splitlines()
    pem_files = [f for f in tracked if f.endswith(".pem")]
    assert pem_files == ["provenance/pubkey.pem"]
    assert not any(f.startswith("private/") for f in tracked)
    assert not any("keys" in f.split("/")[:-1] for f in tracked)


def test_signing_key_exists_matches_committed_pubkey_and_lives_outside_repo():
    """The real signing key: present, a valid Ed25519 private key, its public
    half byte-identical to the committed ``provenance/pubkey.pem``, and
    resolved to a location outside this checkout. (An earlier version of
    this test compared the same hardcoded path string to itself, which
    could never fail regardless of where the key actually was.)

    The key comes only from ``NIGHTINGALE_SIGNING_KEY``, read the way
    ``nightingale sign`` reads it (unset or empty means no key). Without it
    the test skips; with it, every assertion below applies in full.
    """
    configured = os.environ.get(SIGNING_KEY_ENV)
    if not configured:
        pytest.skip(
            f"{SIGNING_KEY_ENV} is not set -- set it to the private signing key's "
            "path to run this check. The committed manifest's signature is still "
            "verified against the committed public key by the verify tests above."
        )
    key_path = Path(configured)
    if not key_path.is_file():
        pytest.skip(
            f"{SIGNING_KEY_ENV} does not point at a file ({key_path}) -- set it to "
            "the private signing key's path to run this check."
        )

    key = load_pem_private_key(key_path.read_bytes(), password=None)
    assert isinstance(key, Ed25519PrivateKey)

    public_pem = key.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    committed_pem = (REPO_ROOT / "provenance" / "pubkey.pem").read_bytes()
    assert public_pem == committed_pem

    assert not str(key_path.resolve()).startswith(str(REPO_ROOT.resolve()) + os.sep)
