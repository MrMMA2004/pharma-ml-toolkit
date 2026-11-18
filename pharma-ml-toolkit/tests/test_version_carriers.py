"""The four-carrier version pin (round 263 — the r262 lesson made structural).

Round 262 shipped v1.2.169 twice.  The first build trued only ``VERSION``
and the web's ``SHIPPED_TOOLKIT_VERSION``; the other two carriers kept
saying 1.2.168, and each stale carrier failed a DIFFERENT downstream
gate — the web's ``readToolkitVersion()`` prefers ``pyproject`` (the
zip-size endpoint answered ``exact:false source:stream`` off the stale
lookup) and the notebooks print ``pharmaml.__version__`` (the
notebook-freshness guard ``test_executed_outputs_print_shipped_version``
refused the tree).  The release has FOUR version carriers, and every one
of them must move in the SAME bump commit:

1. ``VERSION`` — the release basename (``pharma-ml-toolkit-vX.Y.Z.zip``)
   and the zip builder's dynamic input;
2. ``pyproject.toml`` ``[project] version`` — the carrier the web reads
   first;
3. ``src/pharmaml/__init__.py`` ``__version__`` — the carrier the
   notebooks print;
4. ``<repo>/src/lib/toolkit-version.ts`` ``SHIPPED_TOOLKIT_VERSION`` —
   the web badge's SSR fallback.

This module fails the gate the instant any carrier drifts, so the next
half-trued bump is caught at ``pytest`` time — not by an endpoint
symptom after the zip is built.  Two adjacent pins ride along: the
CHANGELOG must lead with the version being shipped, and the VERSION
file must stay a clean ``X.Y.Z`` release tag (the zip basename, the
endpoint lookup and the notebook guard all parse it).

The web-side carrier skips honestly in the zip-only layout (the archive
ships the toolkit tree alone) — the same skip contract as
``test_web_parity``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import tomllib

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TOOLKIT_ROOT.parent
VERSION_FILE = TOOLKIT_ROOT / "VERSION"
PYPROJECT_FILE = TOOLKIT_ROOT / "pyproject.toml"
INIT_FILE = TOOLKIT_ROOT / "src" / "pharmaml" / "__init__.py"
CHANGELOG_FILE = TOOLKIT_ROOT / "CHANGELOG.md"
WEB_VERSION_TS = REPO_ROOT / "src" / "lib" / "toolkit-version.ts"

#: ``## [X.Y.Z] — date`` — the newest release heading the CHANGELOG leads with.
CHANGELOG_HEADING_RE = re.compile(r"^## \[(\d+\.\d+\.\d+)\]", re.MULTILINE)

#: The web carrier line: ``export const SHIPPED_TOOLKIT_VERSION = "X.Y.Z";``
WEB_CARRIER_RE = re.compile(
    r'export const SHIPPED_TOOLKIT_VERSION = "(\d+\.\d+\.\d+)"'
)

needs_web_side = pytest.mark.skipif(
    not WEB_VERSION_TS.is_file(),
    reason="web checkout (<repo>/src/lib/toolkit-version.ts) required; "
    "the zip-only tree skips — the web carrier pin lives in the repo",
)


def _version_file() -> str:
    return VERSION_FILE.read_text(encoding="utf-8").strip()


def _pyproject_version() -> str:
    with PYPROJECT_FILE.open("rb") as handle:
        data = tomllib.load(handle)
    return str(data["project"]["version"])


def _dunder_version() -> str:
    import pharmaml

    return str(pharmaml.__version__)


def _web_carrier() -> str:
    match = WEB_CARRIER_RE.search(WEB_VERSION_TS.read_text(encoding="utf-8"))
    assert match, (
        "SHIPPED_TOOLKIT_VERSION not found (or not a plain X.Y.Z literal) "
        f"in {WEB_VERSION_TS}"
    )
    return match.group(1)


def _changelog_lead() -> str:
    text = CHANGELOG_FILE.read_text(encoding="utf-8")
    match = CHANGELOG_HEADING_RE.search(text)
    assert match, "CHANGELOG.md has no '## [X.Y.Z]' release heading"
    return match.group(1)


class TestVersionCarriers:
    """The r262 four-carrier contract — one bump, four files, zero drift."""

    def test_version_file_is_a_clean_release_tag(self) -> None:
        version = _version_file()
        assert re.fullmatch(r"\d+\.\d+\.\d+", version), (
            f"VERSION must be a bare X.Y.Z tag (zip basename + endpoint "
            f"lookup parse it); got {version!r}"
        )

    def test_pyproject_carrier_matches_version_file(self) -> None:
        expected = _version_file()
        actual = _pyproject_version()
        assert actual == expected, (
            "pyproject [project] version drifted from VERSION — the web's "
            "readToolkitVersion() PREFERS pyproject, so a stale value here "
            "demotes the zip-size endpoint to exact:false source:stream "
            f"(expected {expected}, pyproject says {actual})"
        )

    def test_dunder_version_carrier_matches_version_file(self) -> None:
        expected = _version_file()
        actual = _dunder_version()
        assert actual == expected, (
            "pharmaml.__version__ drifted from VERSION — the notebooks "
            "print it and the freshness guard refuses a stale tree "
            f"(expected {expected}, __init__.py says {actual})"
        )

    @needs_web_side
    def test_web_shipped_carrier_matches_version_file(self) -> None:
        expected = _version_file()
        actual = _web_carrier()
        assert actual == expected, (
            "SHIPPED_TOOLKIT_VERSION drifted from VERSION — the web badge "
            "falls back to it pre-hydration "
            f"(expected {expected}, toolkit-version.ts says {actual})"
        )

    @needs_web_side
    def test_all_four_carriers_agree_at_once(self) -> None:
        """The headline pin: the r262 failure mode as a single assertion."""
        carriers = {
            "VERSION": _version_file(),
            "pyproject [project] version": _pyproject_version(),
            "pharmaml.__version__": _dunder_version(),
            "SHIPPED_TOOLKIT_VERSION": _web_carrier(),
        }
        unique = set(carriers.values())
        assert len(unique) == 1, (
            "the four version carriers disagree — true ALL of them in the "
            f"bump commit before running the gate: {carriers}"
        )

    def test_changelog_leads_with_the_version_being_shipped(self) -> None:
        expected = _version_file()
        actual = _changelog_lead()
        assert actual == expected, (
            "the CHANGELOG's newest release heading must equal VERSION — "
            "a release without its changelog entry is the same drift one "
            f"layer out (expected {expected}, CHANGELOG leads with {actual})"
        )

    def test_web_carrier_is_the_only_exported_pin(self) -> None:
        """No second hard-coded version literal can hide in the TS module.

        The r262 lesson generalises: a carrier is wherever a version
        literal lives.  ``toolkit-version.ts`` must carry exactly one
        exported version constant — imports of it are free, duplicates
        are drift waiting to happen.
        """
        if not WEB_VERSION_TS.is_file():  # zip-only layout: nothing to pin
            pytest.skip("web checkout absent — zip-only tree")
        text = WEB_VERSION_TS.read_text(encoding="utf-8")
        literals = re.findall(r'"(\d+\.\d+\.\d+)"', text)
        assert len(literals) == 1, (
            "toolkit-version.ts must carry exactly one version literal "
            f"(found {literals}) — a second hard-coded version is a fifth "
            "carrier"
        )
