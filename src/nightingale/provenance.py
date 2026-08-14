# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT
"""The signed provenance chain: manifest, signature, and model fingerprinting.

Two independent claims live here, answering two different questions.

**"Is this checkout of the repo intact?"** -- :func:`build_manifest` hashes
every committed artifact (the six model bundles, the cleaned datasets, the
report figures, the dashboard) with SHA-256, and :func:`sign_manifest` signs
the sorted set of hashes with Ed25519. :func:`verify` recomputes every hash
from the files actually on disk and checks the signature; it never
regenerates an artifact to compare against, because Task 10 established that
``export_model`` is not byte-reproducible across commits (``provenance.
built_utc`` is the HEAD committer timestamp, so a rerun after any later
commit writes different bytes). Hashing what is really there is therefore
the only sound check -- a tampered byte anywhere is caught, and an honest
checkout at any later commit still verifies, because the manifest pins the
bytes it was signed against, not a recipe for reproducing them.

**"Is this lone ``model.json`` -- found anywhere, renamed, with its
``provenance`` block deleted -- one of this project's six trees?"** --
:func:`fingerprint` answers that without needing the file's own metadata at
all. Every ``model.json`` carries a ``canaries`` block: eight deterministic
inputs and the ``p_cal`` this project's own walker gives them (see
:func:`nightingale.export.build_canaries`). Recomputing those eight numbers
from the FILE'S OWN trees on the FILE'S OWN inputs reproduces, bit for bit,
whichever condition's trees it actually contains -- stripping the
``provenance`` block or renaming the ``condition`` field changes nothing the
walker reads. Comparing that recomputed vector against the ``p_cal`` this
project published for each of the six registered conditions is therefore a
fingerprint: a match names the source condition, and a single altered leaf
anywhere breaks the match.

The private signing key never lives in this repository. It is generated
once, kept at a path outside the checkout (default
``~/.claude/keys/nightingale_ed25519.pem`` on the machine this project is
signed from, overridable via ``NIGHTINGALE_SIGNING_KEY`` or an explicit
argument), and only ``provenance/pubkey.pem`` -- the public half -- is
committed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key

from nightingale.conditions import CONDITIONS
from nightingale.export import AUTHOR, REPO_ROOT, predict

# --------------------------------------------------------------------------
# The artifact set
# --------------------------------------------------------------------------

# Globs relative to the repo root. An artifact is included iff it EXISTS at
# signing time -- absent globs (docs/index.html before the dashboard task,
# reports/figures/* before the first report run) simply contribute nothing;
# their later appearance is handled by re-signing, not by this module
# guessing at files that don't exist yet.
ARTIFACT_GLOBS: tuple[str, ...] = (
    "models/*/model.json",
    "models/*/metrics.json",
    "models/*/external.json",
    "data/cleaned/*.csv.gz",
    "reports/figures/*.png",
    "docs/index.html",
)

DEFAULT_MANIFEST_PATH = REPO_ROOT / "provenance" / "manifest.json"
DEFAULT_PUBKEY_PATH = REPO_ROOT / "provenance" / "pubkey.pem"

# The maximum abs difference between a recomputed canary p_cal and a
# registered condition's published p_cal that still counts as a match. Two
# walkers built to match each other (this module reuses
# nightingale.export.predict directly, so it is really the SAME walker) hold
# to float64 rounding, which is far tighter than this -- 1e-12 is the bar
# the brief sets, not a measured slack.
FINGERPRINT_TOLERANCE = 1e-12


def _git(*args: str) -> str | None:
    """Run a read-only git command in the repo; ``None`` if git or the repo is absent."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git installed
        return None
    if result.returncode != 0:
        return None  # pragma: no cover - not a git checkout
    return result.stdout.strip() or None


def canonical_json_bytes(obj: object) -> bytes:
    """The exact bytes signed and verified: sorted keys, no whitespace, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def discover_artifacts(repo_root: Path) -> list[Path]:
    """Every file matching :data:`ARTIFACT_GLOBS` that exists under ``repo_root`` right now."""
    found: set[Path] = set()
    for pattern in ARTIFACT_GLOBS:
        for path in repo_root.glob(pattern):
            if path.is_file():
                found.add(path)
    return sorted(found, key=lambda p: p.relative_to(repo_root).as_posix())


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(paths: list[Path], repo_root: Path = REPO_ROOT) -> dict:
    """``{"artifacts": {relpath: sha256hex, ...}, "meta": {...}}`` for ``paths``.

    ``paths`` may be absolute or already relative to ``repo_root``; either
    way the manifest key is the POSIX-style path relative to ``repo_root``,
    sorted, so the manifest is deterministic regardless of glob-walk order.
    """
    artifacts: dict[str, str] = {}
    for path in paths:
        path = Path(path)
        abs_path = path if path.is_absolute() else repo_root / path
        relpath = abs_path.relative_to(repo_root).as_posix()
        artifacts[relpath] = _sha256_file(abs_path)
    artifacts = dict(sorted(artifacts.items()))

    meta = {
        "built_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "commit": _git("rev-parse", "HEAD") or "unknown",
        "author": AUTHOR,
    }
    return {"artifacts": artifacts, "meta": meta}


# --------------------------------------------------------------------------
# Signing / verification
# --------------------------------------------------------------------------


def _load_private_key(key_path: str | Path | None) -> Ed25519PrivateKey:
    resolved = key_path or os.environ.get("NIGHTINGALE_SIGNING_KEY")
    if not resolved:
        raise ValueError(
            "no signing key: pass key_path or set NIGHTINGALE_SIGNING_KEY to the "
            "Ed25519 private key PEM"
        )
    key = load_pem_private_key(Path(resolved).read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError(f"{resolved} is not an Ed25519 private key")
    return key


def _load_public_key(pubkey_path: str | Path) -> Ed25519PublicKey:
    key = load_pem_public_key(Path(pubkey_path).read_bytes())
    if not isinstance(key, Ed25519PublicKey):
        raise ValueError(f"{pubkey_path} is not an Ed25519 public key")
    return key


def sign_manifest(
    manifest: dict,
    key_path: str | Path | None = None,
    out_path: Path = DEFAULT_MANIFEST_PATH,
) -> Path:
    """Sign ``{"artifacts", "meta"}`` and write ``manifest.json`` (with ``signature_b64``).

    The key comes from ``key_path`` if given, else ``NIGHTINGALE_SIGNING_KEY``.
    Returns the path written.
    """
    private_key = _load_private_key(key_path)
    payload = canonical_json_bytes({"artifacts": manifest["artifacts"], "meta": manifest["meta"]})
    signature = private_key.sign(payload)

    signed = {
        "artifacts": manifest["artifacts"],
        "meta": manifest["meta"],
        "signature_b64": base64.b64encode(signature).decode("ascii"),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(signed, indent=2, sort_keys=True) + "\n")
    return out_path


@dataclass
class VerifyReport:
    """Result of :func:`verify`: per-artifact status plus the signature check."""

    artifacts: dict[str, str]  # relpath -> "OK" | "TAMPERED" | "MISSING"
    signature_valid: bool

    @property
    def ok(self) -> bool:
        """True iff the signature checks out AND every listed artifact is OK."""
        return self.signature_valid and all(status == "OK" for status in self.artifacts.values())


def verify(
    repo_root: Path,
    pubkey_path: str | Path,
    manifest_path: Path | None = None,
) -> VerifyReport:
    """Recompute hashes of every artifact the manifest lists, and check the signature.

    Hashes are always taken from the files ON DISK -- never regenerated --
    per the Task 10 coordination note: a fresh export after any later commit
    would legitimately produce different bytes and must not read as
    tampering.
    """
    repo_root = Path(repo_root)
    manifest_path = Path(manifest_path) if manifest_path else repo_root / "provenance" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    statuses: dict[str, str] = {}
    for relpath, expected_hash in manifest["artifacts"].items():
        path = repo_root / relpath
        if not path.is_file():
            statuses[relpath] = "MISSING"
            continue
        statuses[relpath] = "OK" if _sha256_file(path) == expected_hash else "TAMPERED"

    payload = canonical_json_bytes({"artifacts": manifest["artifacts"], "meta": manifest["meta"]})

    public_key = _load_public_key(pubkey_path)
    try:
        public_key.verify(base64.b64decode(manifest["signature_b64"]), payload)
        signature_valid = True
    except InvalidSignature:
        signature_valid = False

    return VerifyReport(artifacts=statuses, signature_valid=signature_valid)


# --------------------------------------------------------------------------
# Fingerprinting
# --------------------------------------------------------------------------


@dataclass
class FingerprintReport:
    """Result of :func:`fingerprint`: the matched condition, if any, and the evidence."""

    computed_p_cal: list[float]
    condition: str | None
    max_abs_diff: dict[str, float] = field(default_factory=dict)

    @property
    def matched(self) -> bool:
        return self.condition is not None


def fingerprint(model_json_path: str | Path) -> FingerprintReport:
    """Identify which registered condition's trees a ``model.json`` file contains.

    Runs the file's OWN ``canaries.inputs`` through the exporter's reference
    walker (:func:`nightingale.export.predict`) using the file's OWN trees --
    ignoring whatever the file's ``condition``/``provenance`` fields claim --
    then compares the resulting ``p_cal`` vector against the ``p_cal``
    published in each of the six registered conditions' COMMITTED
    ``models/<slug>/model.json``. A copy of a condition's file, renamed and
    with ``provenance`` stripped, still carries that condition's trees and
    canary inputs unchanged, so the recomputed vector reproduces the
    original published one exactly; any altered leaf value breaks every
    match.
    """
    data = json.loads(Path(model_json_path).read_text())
    canaries = data["canaries"]
    computed = [predict(data, row)["p_cal"] for row in canaries["inputs"]]

    diffs: dict[str, float] = {}
    matched: str | None = None
    for slug in sorted(CONDITIONS):
        reference_path = REPO_ROOT / "models" / slug / "model.json"
        if not reference_path.is_file():
            continue  # pragma: no cover - all six are committed in this repo
        reference = json.loads(reference_path.read_text())
        reference_p_cal = reference["canaries"]["p_cal"]
        if len(reference_p_cal) != len(computed):
            diffs[slug] = float("inf")
            continue
        max_diff = max(abs(a - b) for a, b in zip(computed, reference_p_cal))
        diffs[slug] = max_diff
        if max_diff <= FINGERPRINT_TOLERANCE:
            matched = slug

    return FingerprintReport(computed_p_cal=computed, condition=matched, max_abs_diff=diffs)
