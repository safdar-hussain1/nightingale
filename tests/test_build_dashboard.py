# Nightingale — calibrated clinical risk models
# Copyright (c) 2026 Safdar Hussain · https://github.com/safdar-hussain1/nightingale
# SPDX-License-Identifier: MIT

"""Gates on the built dashboard.

The dashboard is a single self-contained ``docs/index.html`` assembled by
``scripts/build_dashboard.py`` from committed, signed artifacts. Every test
here guards one property that a human reviewer cannot re-check by eye on a
1.5 MB file: that no placeholder survived, that the inlined inference engine
is byte-identical to the one the parity test exercises, that the page carries
its own disclaimer verbatim, that the build is reproducible, and that the
committed file is the one the current artifacts produce.
"""

from __future__ import annotations

import gzip
import importlib.util
import re
import sys
from pathlib import Path

import pytest

# Framing words that must never reach a public surface: not rude words, but words
# that would mis-describe what this project is. Spelled once, in fragments, in
# conftest -- see the note there.
from conftest import BANNED_WORDS

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_dashboard.py"
TEMPLATE = REPO_ROOT / "scripts" / "dashboard_template.html"
WALKER = REPO_ROOT / "docs" / "assets" / "walker.js"
OUTPUT = REPO_ROOT / "docs" / "index.html"

SLUGS = (
    "breast-cancer",
    "cervical-cancer",
    "heart-disease",
    "kidney-disease",
    "liver-disease",
    "diabetes",
)

# The verbatim disclaimer. It appears on the README, the model card and the
# dashboard with identical wording on purpose: a reader who meets the project
# twice must not meet two different sets of caveats.
HONESTY_BANNER = (
    "These are screening-triage risk models trained on small public research "
    "datasets. They are not diagnostic devices and must not be used for medical "
    "decisions. The diabetes labels are self-reported survey responses. The "
    "cervical-cancer cohort has 55 positive biopsies."
)

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _load_builder():
    """Import the build script by path (``scripts/`` is not a package)."""
    spec = importlib.util.spec_from_file_location("build_dashboard", BUILD_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_dashboard"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _load_builder()


@pytest.fixture(scope="module")
def html(builder) -> str:
    """The page as the current artifacts produce it, built in memory."""
    return builder.render_page()


# --------------------------------------------------------------------------
# The placeholder and the payload
# --------------------------------------------------------------------------


def test_template_carries_exactly_one_placeholder():
    assert TEMPLATE.read_text(encoding="utf-8").count("/*__DATA__*/") == 1


def test_no_placeholder_survives_the_build(html):
    assert "/*__DATA__*/" not in html


def test_every_slug_is_embedded(html):
    for slug in SLUGS:
        assert slug in html, f"{slug} missing from the built page"


def test_every_model_ships_its_canaries(builder):
    payload = builder.build_payload()
    slugs = {c["slug"] for c in payload["conditions"]}
    assert slugs == set(SLUGS)
    for condition in payload["conditions"]:
        canaries = condition["model"]["canaries"]
        assert len(canaries["inputs"]) == len(canaries["p_cal"]) > 0
        assert condition["model"]["trees"], "a model with no trees cannot predict"


def test_oof_sample_is_capped_and_labelled(builder):
    payload = builder.build_payload()
    for condition in payload["conditions"]:
        oof = condition["oof"]
        assert len(oof["y"]) == len(oof["p"]) == oof["n_shown"]
        assert oof["n_shown"] <= builder.OOF_SAMPLE_CAP
        assert oof["n_total"] >= oof["n_shown"]


# --------------------------------------------------------------------------
# The inlined inference engine
# --------------------------------------------------------------------------


def test_walker_is_inlined_byte_identical(html):
    """The page must run the exact walker the parity test scores.

    A dashboard that inlines a lightly-edited copy of the walker would pass
    every visual check while quietly disagreeing with the Python reference,
    so the committed source is compared byte-for-byte rather than by feature.
    """
    source = WALKER.read_text(encoding="utf-8")
    assert source in html


# --------------------------------------------------------------------------
# What the page says
# --------------------------------------------------------------------------


def test_honesty_banner_is_verbatim(html):
    stripped = re.sub(r"<[^>]+>", "", html)
    collapsed = re.sub(r"\s+", " ", stripped)
    assert HONESTY_BANNER in collapsed


def test_title_is_set(html):
    match = re.search(r"<title>([^<]+)</title>", html)
    assert match is not None
    assert match.group(1).strip()


@pytest.mark.parametrize("word", BANNED_WORDS)
def test_banned_framing_word_absent(html, word):
    assert word not in html.lower(), f"{word!r} reached a public surface"


def test_no_email_address_anywhere(html):
    assert EMAIL_RE.search(html) is None


def test_author_and_repo_credited(html):
    assert "Safdar Hussain" in html
    assert "github.com/safdar-hussain1/nightingale" in html


# The four authorship marks, and why each one is separate: the ``<meta>`` tag
# and the JSON-LD block are for machines, the footer is for a reader, and the
# two below survive the two ways this page actually gets detached from its
# origin — saved to disk and reopened (the source comment), or embedded in a
# frame with the chrome hidden (the console line). A build that dropped either
# would still look right, so they are asserted rather than eyeballed.


def test_source_comment_carries_the_notice(html):
    """A ``view-source`` reader meets the notice before anything else."""
    assert (
        "<!-- Nightingale — © 2026 Safdar Hussain — "
        "github.com/safdar-hussain1/nightingale — MIT: this notice must be retained. -->"
    ) in html
    # Near the top: before the opening <html> tag, so it cannot be scrolled past.
    assert html.index("<!-- Nightingale —") < html.index("<html lang=")


def test_console_signature_is_present_and_unobtrusive(html):
    """Exactly one console line on load — a signature, not a log stream."""
    assert (
        'console.log("Nightingale — © 2026 Safdar Hussain — '
        'github.com/safdar-hussain1/nightingale")'
    ) in html
    assert html.count("console.log(") == 1


def _score_body(html: str) -> str:
    """The body of the page's ``score()`` function, braces balanced."""
    start = html.index("function score()")
    open_brace = html.index("{", start)
    depth = 0
    for offset in range(open_brace, len(html)):
        if html[offset] == "{":
            depth += 1
        elif html[offset] == "}":
            depth -= 1
            if depth == 0:
                return html[open_brace + 1 : offset]
    raise AssertionError("score() is not brace-balanced")


def _zero_supplied_branch(html: str) -> tuple[str, str]:
    """Split ``score()`` into its zero-supplied branch and everything after it."""
    body = _score_body(html)
    guard = body.index("if (supplied === 0) {")
    depth = 0
    for offset in range(body.index("{", guard), len(body)):
        if body[offset] == "{":
            depth += 1
        elif body[offset] == "}":
            depth -= 1
            if depth == 0:
                return body[guard:offset], body[offset:]
    raise AssertionError("the zero-supplied branch is not brace-balanced")


# Markup the calculator must not emit while nothing has been measured: the
# numeral, the risk bar, and the conformal verdict chip.
VERDICT_MARKUP = ("'risk num'", "'risk-bar'", "'chip '")


def test_untouched_calculator_withholds_the_verdict(html):
    """An empty form must not be answered with a risk, a bar or a verdict.

    Left alone, every feature routes down its missing branch and several of
    these models saturate — one of them to a full bar and a positive conformal
    set on the very condition whose note says it can never answer "uncertain".
    That reads as a finding about a patient when it is a property of
    missing-value routing, so the whole verdict apparatus is withheld until at
    least one field is supplied.

    The guarantee is proved statically: the zero-supplied branch is an early
    ``return``, it emits the explanatory note, and every piece of verdict
    markup lives after it and is therefore unreachable from an empty form.
    """
    branch, tail = _zero_supplied_branch(html)

    assert branch.rstrip().endswith("return;"), (
        "the zero-supplied branch must return, or the verdict below it still runs"
    )
    assert "'risk-blank'" in branch, "the blank state must explain itself where the numeral was"
    for markup in VERDICT_MARKUP:
        assert markup not in branch, f"{markup} is emitted on an untouched form"
        assert markup in tail, f"{markup} is not emitted at all — the calculator is broken"


def test_blank_state_names_missing_branch_routing(html):
    """The note has to say *why* the number is withheld, not merely that it is."""
    branch, _ = _zero_supplied_branch(html)
    assert "missing value" in branch
    assert "saturates" in branch


# --------------------------------------------------------------------------
# How the page loads
# --------------------------------------------------------------------------


def test_chartjs_is_pinned_with_subresource_integrity(html):
    match = re.search(r"<script[^>]*chart[^>]*></script>", html, re.IGNORECASE)
    assert match is not None, "no Chart.js script tag"
    tag = match.group(0)
    assert "integrity=" in tag
    assert "crossorigin=" in tag
    assert re.search(r"chart\.js@\d+\.\d+\.\d+", tag), "Chart.js is not version-pinned"


def test_chartjs_is_the_only_remote_subresource(html):
    """Everything but Chart.js is inline, so the page works from ``file://``."""
    # rel="canonical" is crawler metadata, never fetched — drop it before the
    # scan so only tags that actually pull bytes are held to the rule.
    scannable = re.sub(r'<link rel="canonical"[^>]*>', "", html)
    remote = re.findall(r'(?:src|href)="(https?://[^"]+)"', scannable)
    subresources = [url for url in remote if not url.startswith("https://github.com")]
    assert all("chart.js" in url for url in subresources), subresources


def test_page_never_fetches_at_runtime(html):
    assert "fetch(" not in html
    assert "XMLHttpRequest" not in html


def test_selftest_hook_is_present(html):
    assert "selftest=1" in html
    assert "NIGHTINGALE SELFTEST PASS" in html
    assert "NIGHTINGALE SELFTEST FAIL" in html


def test_theme_toggle_persists_before_reload(html):
    """A toggle that reloads before writing storage forgets the click."""
    assert "localStorage" in html
    assert "location.reload" not in html


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def test_build_is_byte_deterministic(builder):
    assert builder.render_page() == builder.render_page()


def test_committed_page_matches_a_fresh_build(html):
    assert OUTPUT.exists(), "docs/index.html has not been built"
    assert OUTPUT.read_text(encoding="utf-8") == html, (
        "docs/index.html is stale — rerun scripts/build_dashboard.py"
    )


def test_page_weight_stays_publishable(html):
    """A GitHub Pages visitor downloads this over whatever link they have."""
    gzipped = len(gzip.compress(html.encode("utf-8"), 9))
    assert gzipped < 2_000_000, f"{gzipped} bytes gzipped is too heavy to ship"
