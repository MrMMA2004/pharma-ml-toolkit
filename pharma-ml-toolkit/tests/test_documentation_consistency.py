"""part13 §18's automation lane — documentation consistency guards.

§18's own rule: "Keep the following synchronized ... Add automated
documentation consistency checks where possible. Prevent stale documents
from causing repeated implementation of already completed features."

This file turns three of those synchronisation promises into CI-enforced
pins instead of prose claims:

1. the version-carrier lockstep — every carrier the master prompt §5
   enumerates (VERSION, pyproject, ``pharmaml.__version__``, the frontend
   package, the Tauri config, the Windows version_info quadruple, the
   CHANGELOG's lead entry, the FEATURES freeze stamp, the web app's TS
   carrier) must agree with the VERSION file at every commit;
2. the §21 deliverable set's existence — the twelve audit documents the
   master prompt requires must actually be on disk;
3. the ledger agreement — the worklog's newest round entry and the
   100-runs counter must describe the same round and run number, so a
   stale counter can never again masquerade as the state of the world
   (the exact failure mode rounds 315/316/321 documented).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pharmaml

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO.parent


def _version_file() -> str:
    return (REPO / "VERSION").read_text(encoding="utf-8").strip()


class TestVersionCarrierLockstep:
    def test_pyproject_matches(self):
        text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        assert match, "pyproject.toml lost its project version"
        assert match.group(1) == _version_file()

    def test_runtime_carrier_matches(self):
        assert pharmaml.__version__ == _version_file()

    def test_frontend_package_matches(self):
        data = json.loads(
            (REPO / "frontend" / "package.json").read_text(encoding="utf-8")
        )
        assert data["version"] == _version_file()

    def test_tauri_conf_matches(self):
        data = json.loads(
            (REPO / "src-tauri" / "tauri.conf.json").read_text(
                encoding="utf-8"
            )
        )
        assert data["version"] == _version_file()

    def test_windows_version_info_matches(self):
        text = (
            REPO / "packaging" / "windows" / "version_info.txt"
        ).read_text(encoding="utf-8")
        match = re.search(
            r"filevers=\((\d+), (\d+), (\d+), (\d+)\)", text
        )
        assert match, "version_info.txt lost its filevers quadruple"
        assert ".".join(match.groups()[:3]) == _version_file()

    def test_changelog_leads_with_the_current_version(self):
        text = (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
        match = re.search(r"^## \[([^\]]+)\]", text, re.MULTILINE)
        assert match, "CHANGELOG has no versioned lead entry"
        assert match.group(1) == _version_file()

    def test_features_freeze_stamp_matches(self):
        text = (REPO / "docs" / "FEATURES.md").read_text(encoding="utf-8")
        assert f"v{_version_file()}" in text

    def test_web_carrier_matches(self):
        text = (
            ROOT / "src" / "lib" / "toolkit-version.ts"
        ).read_text(encoding="utf-8")
        match = re.search(r'SHIPPED_TOOLKIT_VERSION\s*=\s*"([^"]+)"', text)
        assert match, "toolkit-version.ts lost its shipped-version export"
        assert match.group(1) == _version_file()


class TestDeliverableSet:
    DELIVERABLES = (
        "docs/part12_audit/COMPREHENSIVE_PROJECT_AUDIT.md",
        "docs/part12_audit/BACKEND_GUI_INTEGRATION_MATRIX.md",
        "docs/part12_audit/ARCHITECTURE_REVIEW.md",
        "docs/ARCHITECTURE_REVIEW.md",
        "docs/part12_audit/BUG_REPORT.md",
        "docs/part12_audit/SCIENTIFIC_VALIDATION_REPORT.md",
        "docs/part12_audit/PERFORMANCE_REPORT.md",
        "docs/part12_audit/FEATURE_GAP_AND_GITHUB_REFERENCES.md",
        "docs/part12_audit/SECURITY_AND_PACKAGING_AUDIT.md",
        "docs/part12_audit/WINDOWS_CLEAN_INSTALL_REPORT.md",
        "docs/RELEASE_READINESS.md",
        "docs/MIGRATION_TAURI2.md",
    )

    def test_the_master_prompt_21_deliverables_exist(self):
        for relative in self.DELIVERABLES:
            assert (REPO / relative).is_file(), f"missing: {relative}"

    def test_the_authoritative_status_document_names_the_queue(self):
        text = (
            REPO / "docs" / "PROJECT_STATUS.md"
        ).read_text(encoding="utf-8")
        assert "## 6. Next-campaign priorities" in text
        assert "Phase 6" in text

    def test_the_module_index_stays_generated(self):
        index = (REPO / "docs" / "INDEX.md")
        assert index.is_file(), "docs/INDEX.md is gone"
        text = index.read_text(encoding="utf-8")
        assert "## `pharmaml`" in text
        assert (REPO / "tools" / "build_docs_index.py").is_file()


class TestLedgerAgreement:
    def test_worklog_newest_round_matches_the_counter(self):
        counter = json.loads(
            (ROOT / "scratch" / "100runs-counter.json").read_text(
                encoding="utf-8"
            )
        )
        round_match = re.search(r"round (\d+)", counter["last_run_note"])
        assert round_match, "counter lost its round annotation"
        round_no = int(round_match.group(1))
        runs = int(counter["runs_completed"])

        worklog = (ROOT / "worklog.md").read_text(encoding="utf-8")
        entries = re.findall(
            r"\*\*R(\d+) — the (\d+)th recorded run", worklog
        )
        assert entries, "worklog carries no bold-format round entries"
        last_round, last_run = entries[-1]
        assert int(last_round) == round_no, (
            f"worklog's newest entry is R{last_round} but the counter "
            f"says round {round_no} — one ledger is stale"
        )
        assert int(last_run) == runs, (
            f"worklog's newest entry says run {last_run} but the counter "
            f"says {runs} — one ledger is stale"
        )
