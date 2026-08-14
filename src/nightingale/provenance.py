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
all, and WITHOUT trusting anything the candidate file supplies about which
inputs to test it on.

An earlier version of this function ran the CANDIDATE's own
``canaries.inputs`` through the candidate's own trees and compared the
result to each registered condition's STORED ``canaries.p_cal``. That is
forgeable two different ways: (1) many canary rows land on a calibrator's
saturated plateau (``p_cal`` pinned to 0 or 1), so a tampered tree can drift
substantially in ``p_raw`` while every affected ``p_cal`` stays put --
measured at up to 100% false-authentic acceptance of tampered leaves on
5 of 6 conditions; (2) because the candidate supplies its OWN inputs, an
attacker can pick 8 copies of one input the target's calibrator saturates
and forge a match against ANY registered condition, including ones with a
completely different tree structure.

The fix removes both trust anchors. For each registered condition, this
module loads THAT CONDITION'S OWN COMMITTED ``model.json`` -- never the
candidate's -- and takes its ``canaries.inputs`` from there. Those
reference-pinned inputs are run through both the reference's own trees and
the CANDIDATE's trees, and the two ``p_raw`` (pre-calibration) vectors are
compared: ``p_raw`` is a continuous sigmoid, not a step function, so it
carries a genuine tree-identity signal that a saturated ``p_cal`` erases.
A condition "matches" only if it is the SOLE registered condition whose
reference inputs produce agreement within :data:`FINGERPRINT_TOLERANCE`;
zero or more-than-one agreeing conditions is reported as no match rather
than picking the last one seen. The candidate never gets to choose what it
is tested on, and the comparison happens in the space where tampering is
actually visible.

The private signing key never lives in this repository. It is generated
once, kept in a key directory outside the checkout on the machine this
project is signed from, and supplied at signing time via
``NIGHTINGALE_SIGNING_KEY`` or an explicit ``key_path``
argument; only ``provenance/pubkey.pem`` -- the public half -- is
committed.
"""

from __future__ import annotations

import base64
import binascii
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

# Relative to the repo root; the single source of truth for where the
# signed manifest and public key live, so the default in every function
# that needs one is computed from here instead of a repeated literal.
MANIFEST_RELPATH = Path("provenance/manifest.json")
PUBKEY_RELPATH = Path("provenance/pubkey.pem")
DEFAULT_MANIFEST_PATH = REPO_ROOT / MANIFEST_RELPATH
DEFAULT_PUBKEY_PATH = REPO_ROOT / PUBKEY_RELPATH

# The maximum abs difference between two p_raw vectors that still counts as
# a match. Two walkers built to match each other (this module reuses
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
    """SHA-256 hex digest of ``path``, read in fixed-size chunks.

    Chunked rather than ``path.read_bytes()`` so hashing a large artifact
    (a cleaned dataset, a figure) never pulls the whole file into memory at
    once. Any I/O failure -- the file disappearing between an existence
    check and this call, permission trouble -- surfaces as ``OSError`` (its
    usual subclasses, e.g. ``FileNotFoundError``), which callers are
    expected to catch rather than let propagate as a crash.
    """
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _resolve_artifact_path(repo_root: Path, relpath: str) -> Path | None:
    """``repo_root / relpath`` if that stays under ``repo_root``, else ``None``.

    Rejects an absolute ``relpath`` outright, and rejects any ``relpath``
    (e.g. ``"../../etc/passwd"``) whose resolved location escapes
    ``repo_root`` -- a manifest is a list of relative paths INTO the
    checkout, never an instruction to read anything else on disk.
    """
    candidate = Path(relpath)
    if candidate.is_absolute():
        return None
    resolved = (repo_root / candidate).resolve()
    try:
        resolved.relative_to(repo_root.resolve())
    except ValueError:
        return None
    return resolved


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

    # relpath -> "OK" | "TAMPERED" | "MISSING" | "INVALID_PATH" | "UNVERIFIED"
    artifacts: dict[str, str]
    signature_valid: bool

    @property
    def ok(self) -> bool:
        """True iff the signature checks out AND every listed artifact is OK."""
        return self.signature_valid and all(status == "OK" for status in self.artifacts.values())


def _check_signature(manifest: dict, pubkey_path: str | Path) -> bool:
    """Verify ``manifest``'s ``signature_b64`` over its own ``artifacts``/``meta``.

    Fails closed: a missing ``signature_b64``/``artifacts``/``meta`` key
    (``KeyError``), non-base64 garbage in ``signature_b64``
    (``binascii.Error``, which ``base64.b64decode`` raises with
    ``validate=True``), or any other malformed input is treated the same as
    a cryptographically invalid signature -- ``False``, never a raised
    exception.
    """
    public_key = _load_public_key(pubkey_path)
    try:
        payload = canonical_json_bytes({"artifacts": manifest["artifacts"], "meta": manifest["meta"]})
        signature = base64.b64decode(manifest["signature_b64"], validate=True)
        public_key.verify(signature, payload)
        return True
    except (InvalidSignature, binascii.Error, KeyError, ValueError, TypeError):
        return False


def verify(
    repo_root: Path,
    pubkey_path: str | Path,
    manifest_path: Path | None = None,
) -> VerifyReport:
    """Check the signature, THEN recompute hashes of every artifact the manifest lists.

    The signature is checked FIRST and hashing never happens if it fails --
    an attacker who edits the manifest (adding artifact entries, changing
    hashes, pointing a path outside the checkout) invalidates the signature
    over ``{"artifacts", "meta"}``, and this function refuses to touch the
    filesystem on their behalf when that happens. Every listed relpath is
    additionally checked to resolve to somewhere under ``repo_root`` before
    being hashed -- a defence-in-depth measure independent of the signature
    check, since a manifest is a list of relative paths INTO the checkout,
    never an instruction to read arbitrary locations.

    Hashes are always taken from the files ON DISK -- never regenerated --
    per the Task 10 coordination note: a fresh export after any later commit
    would legitimately produce different bytes and must not read as
    tampering.
    """
    repo_root = Path(repo_root).resolve()
    manifest_path = Path(manifest_path) if manifest_path else repo_root / MANIFEST_RELPATH
    manifest = json.loads(manifest_path.read_text())
    artifact_hashes: dict[str, str] = manifest.get("artifacts", {}) or {}

    signature_valid = _check_signature(manifest, pubkey_path)
    if not signature_valid:
        # Refuse to hash anything an unverified manifest points at.
        return VerifyReport(
            artifacts={relpath: "UNVERIFIED" for relpath in artifact_hashes},
            signature_valid=False,
        )

    statuses: dict[str, str] = {}
    for relpath, expected_hash in artifact_hashes.items():
        resolved = _resolve_artifact_path(repo_root, relpath)
        if resolved is None:
            statuses[relpath] = "INVALID_PATH"
            continue
        try:
            if not resolved.is_file():
                statuses[relpath] = "MISSING"
                continue
            statuses[relpath] = "OK" if _sha256_file(resolved) == expected_hash else "TAMPERED"
        except OSError:
            # e.g. the file vanished between the is_file() check and the
            # read (a TOCTOU race), or a permission error mid-read.
            statuses[relpath] = "MISSING"

    return VerifyReport(artifacts=statuses, signature_valid=True)


# --------------------------------------------------------------------------
# Fingerprinting
# --------------------------------------------------------------------------


@dataclass
class FingerprintReport:
    """Result of :func:`fingerprint`: the matched condition, if any, and the evidence.

    ``condition`` is set only when exactly one registered condition agrees
    within :data:`FINGERPRINT_TOLERANCE` -- zero agreeing conditions and
    more than one agreeing condition both report ``None`` (see ``matches``
    for which, and how many, actually agreed).
    """

    condition: str | None
    matches: list[str] = field(default_factory=list)
    max_abs_diff: dict[str, float] = field(default_factory=dict)

    @property
    def matched(self) -> bool:
        return self.condition is not None


def fingerprint(model_json_path: str | Path) -> FingerprintReport:
    """Identify which registered condition's trees a ``model.json`` file contains.

    For each of the six registered conditions, this loads THAT CONDITION'S
    OWN COMMITTED ``models/<slug>/model.json`` and takes its
    ``canaries.inputs`` -- never the candidate's -- then runs those
    reference-pinned inputs through both the reference's trees and the
    candidate's trees, comparing the two ``p_raw`` (pre-calibration)
    vectors. A condition counts as agreeing iff every row's ``p_raw`` is
    within :data:`FINGERPRINT_TOLERANCE`; ``condition`` is set only if
    EXACTLY ONE registered condition agrees.

    Using ``p_raw`` rather than ``p_cal`` matters: an isotonic calibrator
    can saturate several canary rows to the same boundary value, hiding a
    tampered tree's margin shift behind an unchanged calibrated output.
    Pinning the inputs to the REFERENCE rather than letting the candidate
    supply its own matters too: a candidate that chooses its own inputs can
    pick ones a target condition's calibrator saturates and forge a match
    against a condition with entirely different trees. Neither hazard is
    reachable here -- the candidate never gets to choose what it is tested
    on, in either sense.

    A candidate whose own trees expect a different number of features than
    a given reference (almost always the case for a NON-matching condition)
    simply fails to agree with that reference; :func:`nightingale.export.
    predict` raising on the row-length mismatch is caught and recorded as
    non-agreement, not propagated.
    """
    data = json.loads(Path(model_json_path).read_text())

    diffs: dict[str, float] = {}
    matches: list[str] = []
    for slug in sorted(CONDITIONS):
        reference_path = REPO_ROOT / "models" / slug / "model.json"
        if not reference_path.is_file():
            continue  # pragma: no cover - all six are committed in this repo
        reference = json.loads(reference_path.read_text())
        reference_inputs = reference["canaries"]["inputs"]
        reference_p_raw = [predict(reference, row)["p_raw"] for row in reference_inputs]
        try:
            candidate_p_raw = [predict(data, row)["p_raw"] for row in reference_inputs]
        except (ValueError, KeyError, IndexError, TypeError):
            # Structurally can't be this condition (wrong feature count, or
            # not even a well-formed model dict) -- not a match, not a crash.
            diffs[slug] = float("inf")
            continue
        max_diff = max(abs(a - b) for a, b in zip(candidate_p_raw, reference_p_raw))
        diffs[slug] = max_diff
        if max_diff <= FINGERPRINT_TOLERANCE:
            matches.append(slug)

    condition = matches[0] if len(matches) == 1 else None
    return FingerprintReport(condition=condition, matches=matches, max_abs_diff=diffs)
