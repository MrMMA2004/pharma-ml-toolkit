"""Phase 6 (part6 §16): the release-readiness evidence pins.

These tests pin the readiness MECHANISM, not a verdict: the dependency
matrix must probe honestly, the capability counts must match the live
registry, the suite inventory must cover every test file, and the
generated report must contain each §16 section with real numbers (or
an honest "not gathered"). A regression here means the report could
silently lie — the one thing part6 forbids.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TK = Path(__file__).resolve().parents[1]
if str(_TK) not in sys.path:
    sys.path.insert(0, str(_TK))
if str(_TK / "tools") not in sys.path:
    sys.path.insert(0, str(_TK / "tools"))


def test_dependency_matrix_probes_every_optional_family() -> None:
    import release_readiness as rr

    rows = rr.probe_dependencies()
    names = {name for name, _status, _consumer in rows}
    # every optional family the toolkit documents must be probed
    for required in ("rdkit", "torch", "xgboost", "shap", "matplotlib",
                     "PySide6", "pyqtgraph"):
        assert required in names, required
    # statuses are only ever the honest vocabulary
    for _name, status, _consumer in rows:
        assert status == "installed" or status == "not installed" \
            or status.startswith("broken ("), status


def test_capability_counts_match_the_live_registry() -> None:
    import release_readiness as rr

    counts = rr.probe_capabilities()
    from pharmaml.studio.capability_registry import CAPABILITIES

    assert counts["total"] == len(CAPABILITIES)
    assert counts["total"] == sum(
        v for k, v in counts.items() if k in ("integrated", "api"))
    assert counts["without_tests"] == 0  # §16: no test gaps


def test_suite_inventory_covers_every_test_file() -> None:
    import release_readiness as rr

    counts = rr.count_suites()
    on_disk = {str(p.relative_to(_TK)) for p in
               (_TK / "tests").glob("test_*.py")}
    on_disk |= {str(p.relative_to(_TK)) for p in
                (_TK / "legacy" / "tests_ui").glob("test_*.py")}
    assert on_disk == set(counts), (on_disk ^ set(counts))
    assert sum(counts.values()) > 5000  # the real inventory, not a stub


def test_version_carriers_agree() -> None:
    import release_readiness as rr

    versions = rr.probe_versions()
    assert versions["agreement"] == "yes"
    assert versions["VERSION"] == versions["pyproject.toml"]
    assert versions["VERSION"] == versions["src/pharmaml/__init__.py"]
    assert versions["CHANGELOG.md leads with"] == versions["VERSION"]


def test_report_renders_every_section_with_honest_numbers() -> None:
    import release_readiness as rr

    r = rr.Readiness()
    r.dependency_rows = rr.probe_dependencies()
    r.capability_counts = rr.probe_capabilities()
    r.suite_counts = {"tests/test_pk.py": 49, "legacy/tests_ui/test_phase5.py": 7}
    r.version_agreement = rr.probe_versions()
    r.recorded_runs = {"chunk-a": "10 passed"}
    r.archive_audit = {"status": "passed", "checks": "8",
                       "how": "audit_release"}
    text = rr.render(r)

    for section in ("Dependency matrix", "Capability coverage",
                    "Test evidence", "Version carriers",
                    "Release-archive verification"):
        assert section in text, section
    assert "never as broken" in text          # the honesty sentence
    assert "| chunk-a | 10 passed |" in text.replace("`", "")
    assert "**yes**" in text                   # carrier agreement
    assert "Status: **passed** (8 checks)" in text


def test_report_says_not_gathered_when_evidence_is_absent() -> None:
    import release_readiness as rr

    r = rr.Readiness()
    r.dependency_rows = rr.probe_dependencies()
    r.capability_counts = rr.probe_capabilities()
    r.suite_counts = rr.count_suites() if False else {"tests/test_pk.py": 49}
    r.version_agreement = rr.probe_versions()
    # no recorded runs, no audit → both must say so honestly
    text = rr.render(r)
    assert "Not gathered" in text
    assert "Status: **not gathered**" in text
    # and the not-gathered text names the gathering command
    assert "audit_release" in text or "build_zip" in text
