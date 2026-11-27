"""part18 §10 — the legacy studio.py retirement guards.

The owner's mandatory directive, pinned by tests: ``studio.py`` is NOT
an active supported application entry point.  This file asserts the
retirement's mechanics so a rollback or a careless restore cannot
silently resurrect the legacy GUI as the product's face:

* the launcher is ABSENT from the toolkit root and PRESENT only in the
  quarantined ``legacy/`` area, banner-marked RETIRED at its head;
* ``legacy/README.md`` states the unsupported status and the supported
  surfaces;
* the production sidecar spec EXCLUDES the whole Qt stack and its
  entry is the FastAPI backend, NOT the legacy GUI;
* ``requirements.txt`` carries no active PySide6 pin (the GUI deps are
  commented-out optional extras at most);
* the supported surfaces remain present and importable: the Tauri app
  config, the CLI entry module, the FastAPI app factory;
* the README no longer presents ``python studio.py`` as a launch
  command (only the quarantined ``legacy/studio.py`` path appears);
* the retired Tkinter ``playground.py`` stays retired (no resurrection).
"""

from __future__ import annotations

import re
from pathlib import Path

TOOLKIT = Path(__file__).resolve().parents[1]


class TestLauncherQuarantined:
    def test_no_active_launcher_at_the_toolkit_root(self) -> None:
        assert not (TOOLKIT / "studio.py").exists(), (
            "studio.py must not live at the toolkit root (part18 §10 "
            "mandatory retirement) -- it belongs in legacy/"
        )

    def test_legacy_launcher_exists_with_retirement_banner(self) -> None:
        launcher = TOOLKIT / "legacy" / "studio.py"
        assert launcher.is_file()
        head = launcher.read_text(encoding="utf-8")[:2000]
        assert "RETIRED" in head and "NO LONGER an active supported" in head
        assert "React/Tauri 2 desktop app" in head

    def test_legacy_readme_declares_unsupported(self) -> None:
        note = TOOLKIT / "legacy" / "README.md"
        assert note.is_file()
        text = note.read_text(encoding="utf-8")
        assert "UNSUPPORTED" in text
        assert "React/Tauri 2" in text
        assert "backend_entry.py" in text

    def test_playground_stays_retired(self) -> None:
        assert not (TOOLKIT / "playground.py").exists()
        assert not (TOOLKIT / "tests" / "test_playground.py").exists()


class TestPackagingRefusesTheLegacyGui:
    def test_sidecar_spec_excludes_the_qt_stack(self) -> None:
        spec = (
            TOOLKIT / "packaging" / "windows" / "pharmaml_backend.spec"
        ).read_text(encoding="utf-8")
        for banned in ("PySide6", "shiboken6", "pyqtgraph", "qt_material", "qdarkstyle"):
            assert banned in spec, f"the sidecar excludes list lost {banned!r}"
        excludes = re.search(r"excludes = \[(.*?)\]", spec, re.DOTALL)
        assert excludes, "the sidecar spec's excludes list not found"
        block = excludes.group(1)
        for banned in ("PySide6", "shiboken6", "pyqtgraph"):
            assert f'"{banned}"' in block, f"{banned} not in the excludes list"

    def test_sidecar_entry_is_the_backend_not_the_gui(self) -> None:
        spec = (
            TOOLKIT / "packaging" / "windows" / "pharmaml_backend.spec"
        ).read_text(encoding="utf-8")
        assert "backend_entry.py" in spec
        # the ACTUAL entry list (the Analysis scripts) must not carry the
        # legacy launcher — the docstring's historical mentions are prose
        scripts = re.search(r"excludes = \[", spec)
        assert scripts is not None
        analysis = re.findall(r"\[str\((.*?)\)\]", spec)
        assert analysis, "the spec's entry list not found"
        joined = ", ".join(analysis)
        assert "studio.py" not in joined, (
            f"the sidecar entry must not be the legacy GUI: {joined}"
        )

    def test_legacy_app_spec_points_at_the_quarantine(self) -> None:
        spec = (
            TOOLKIT / "packaging" / "windows" / "pharmaml.spec"
        ).read_text(encoding="utf-8")
        assert 'TOOLKIT / "legacy" / "studio.py"' in spec
        assert 'TOOLKIT / "studio.py"' not in spec

    def test_requirements_carry_no_active_pyside6_pin(self) -> None:
        text = (TOOLKIT / "requirements.txt").read_text(encoding="utf-8")
        active = [
            line
            for line in text.splitlines()
            if line.strip().startswith(("PySide6", "pyqtgraph", "qt-material", "qdarkstyle"))
            and not line.strip().startswith("#")
        ]
        assert active == [], f"production requirements must not pin Qt GUI deps: {active}"

    def test_packaging_validator_carries_the_quarantine_not_the_root(
        self,
    ) -> None:
        """Round 365, run 45856387513 step-20 fix: the --static gate's
        must-carry list still demanded the retired root ``studio.py``
        and the Windows Release build died at the artifact validation
        AFTER a fully green freeze+Tauri+NSIS+MSI ladder.  The gate
        itself must never demand what part18 §10 forbids."""
        validator = (
            TOOLKIT / "packaging" / "windows" / "packaging_validation.py"
        ).read_text(encoding="utf-8")
        must_carry = re.search(
            r"for rel in \(\s*(.*?)\):", validator, re.DOTALL
        )
        assert must_carry, "the validator's must-carry tuple not found"
        block = must_carry.group(1)
        assert '"studio.py"' not in block, (
            "the packaging validator must not demand the retired root "
            "studio.py (part18 §10); the quarantine path is the contract"
        )
        assert '"legacy/studio.py"' in block
        assert '"packaging/windows/backend_entry.py"' in block


class TestSupportedSurfacesRemain:
    def test_tauri_app_config_present(self) -> None:
        conf = TOOLKIT / "src-tauri" / "tauri.conf.json"
        assert conf.is_file()
        assert "frontend" in conf.read_text(encoding="utf-8")

    def test_cli_entry_module_present(self) -> None:
        entry = TOOLKIT / "src" / "pharmaml" / "__main__.py"
        assert entry.is_file()

    def test_fastapi_app_factory_importable(self) -> None:
        from pharmaml.api import create_app  # noqa: F401 -- the import IS the test

    def test_readme_launches_no_root_studio(self) -> None:
        text = (TOOLKIT / "README.md").read_text(encoding="utf-8")
        # r368 refinement: check EVERY textual occurrence of the launcher
        # command (the old first-match-only read could miss a root command
        # hiding after one sanctioned mention). Each occurrence must be
        # either the quarantined legacy/studio.py path or sit on a line
        # that PROHIBITS the command ("Do not use ..." / "must not" /
        # "never ..."); presenting it as a launch instruction fails.
        violations: list[str] = []
        for match in re.finditer(r"`?python (legacy/)?studio\.py", text):
            line = text[max(0, match.start() - 60) : match.end() + 10]
            if match.group(1) == "legacy/":
                continue  # the quarantined archival path, sanctioned
            lowered = line.lower()
            if any(phrase in lowered for phrase in ("do not use", "must not", "never ")):
                continue  # a prohibition, not a presentation
            violations.append(line.strip())
        assert violations == [], (
            "README must not present a root-level studio.py launch command "
            f"(prohibition lines are fine): {violations}"
        )
        assert "RETIRED" in text
        # r368: accept either phrasing of the supported-surface callout
        # ("React/Tauri Studio" or the expanded "React + TypeScript ...
        # Tauri 2" form) -- the substance is the supported-surface pointer
        assert "React/Tauri" in text or "React + Tauri 2" in text


class TestGuiTreeQuarantined:
    """part18 §10's quarantine COMPLETION (the ui/-pages + workspace.py
    increment the r365 addendum recorded as open): the legacy PySide6
    GUI tree itself — ``ui/`` and its ``tests/ui/`` family — moved to
    ``legacy/ui/`` and ``legacy/tests_ui/``, while the Qt-free
    ``pharmaml.studio`` persistence modules stay in ``src/`` where they
    legitimately serve the FastAPI backend.  These guards pin the move
    so a rollback or a careless restore cannot resurrect the Qt pages
    as in-tree product source."""

    def test_no_ui_package_at_the_toolkit_root(self) -> None:
        assert not (TOOLKIT / "ui").exists(), (
            "the legacy PySide6 package must not live at the toolkit root "
            "(part18 §10 quarantine completion) -- it belongs in legacy/"
        )
        assert not (TOOLKIT / "ui" / "__init__.py").exists()

    def test_no_ui_test_family_under_tests(self) -> None:
        assert not (TOOLKIT / "tests" / "ui").exists(), (
            "the archival GUI tests live at legacy/tests_ui (out of the "
            "default testpaths gate), never under tests/"
        )

    def test_legacy_ui_tree_present_with_the_egl_stub(self) -> None:
        pkg = TOOLKIT / "legacy" / "ui"
        assert (pkg / "__init__.py").is_file()
        assert (pkg / "pages").is_dir()
        assert (pkg / "native" / "libEGL.so.1").is_file(), (
            "the committed EGL stub rides the quarantined tree (the "
            "headless fallback every pytest invocation preloads)"
        )

    def test_legacy_tests_ui_present_with_their_own_conftest(self) -> None:
        family = TOOLKIT / "legacy" / "tests_ui"
        assert (family / "conftest.py").is_file()
        assert (family / "__init__.py").is_file()
        assert len(list(family.glob("test_*.py"))) > 40, (
            "the archival GUI suite (74 files at the move) must be whole"
        )

    def test_root_conftest_carries_the_legacy_path(self) -> None:
        text = (TOOLKIT / "conftest.py").read_text(encoding="utf-8")
        assert 'str(_LEGACY)' in text or '"legacy"' in text, (
            "the root conftest must put legacy/ on sys.path so the "
            "quarantined ui.* imports (the stress lane + the archival "
            "suite) resolve without resurrecting the root package"
        )

    def test_no_product_source_imports_the_qt_tree(self) -> None:
        """src/pharmaml (the toolkit + the FastAPI backend) never imports
        the quarantined ``ui.*`` package: the Qt-free
        ``pharmaml.studio`` persistence modules are the only studio
        surface the backend touches."""
        violations: list[str] = []
        for path in sorted((TOOLKIT / "src" / "pharmaml").rglob("*.py")):
            text = path.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped.startswith(("#", '"', "'")):
                    continue
                if re.search(r"^\s*(from|import)\s+ui\b", line):
                    violations.append(f"{path.relative_to(TOOLKIT)}: {line.strip()}")
        assert violations == [], (
            "product source must not import the quarantined Qt tree: "
            f"{violations[:5]}"
        )

    def test_qt_free_studio_persistence_still_serves_the_backend(self) -> None:
        """The FastAPI backend's own consumption of the persistence
        modules survives the move: the capability route imports the
        registry; the workflow contracts import the workspace."""
        from pharmaml.api import create_app  # noqa: F401 - importability IS the test
        from pharmaml.studio import (  # noqa: F401
            capability_registry,
            comparison,
            evidence,
            reports,
            runs,
            workspace,
        )
        registry_text = (
            TOOLKIT / "src" / "pharmaml" / "studio" / "capability_registry.py"
        ).read_text(encoding="utf-8")
        assert "legacy/ui/" in registry_text or "legacy" in registry_text, (
            "the capability registry's module strings must point at the "
            "quarantined tree, never a root ui/ that no longer exists"
        )

    def test_venv_repair_points_at_the_quarantined_stub(self) -> None:
        """The rollback-repair script must copy the EGL stub from its
        quarantined home (a stale root path would break the next
        rollback repair)."""
        script = (
            TOOLKIT.parent / "scratch" / "venv_repair.sh"
        ).read_text(encoding="utf-8")
        assert "pharma-ml-toolkit/legacy/ui/native/libEGL.so.1" in script
        assert "pharma-ml-toolkit/ui/native" not in script


class TestProductionWorkflowFreeOfLegacyPackaging:
    """r368 (part19 Phase 2) additions: the production Windows release
    workflow must freeze ONLY the Qt-free sidecar spec, the retired
    Inno-Setup packaging lane must stay retired and banner-marked, and
    the CI's legacy-GUI lane must remain an explicitly-listed
    ``legacy/tests_ui`` lane outside the default testpaths gate -- so
    neither a rollback nor a careless edit can put the Qt GUI back into
    the shipped installer or the default test gate."""

    REPO_ROOT = TOOLKIT.parent
    RELEASE_WF = REPO_ROOT / ".github" / "workflows" / "windows-release.yml"
    CI_WF = REPO_ROOT / ".github" / "workflows" / "ci.yml"

    def test_release_workflow_freezes_the_qt_free_sidecar(self) -> None:
        text = self.RELEASE_WF.read_text(encoding="utf-8")
        assert "pharmaml_backend.spec" in text, (
            "the production release must freeze the Qt-free sidecar spec"
        )
        for banned in (
            "pharmaml.spec",
            "installer.iss",
            "build_windows.ps1",
            "Inno-Setup",
        ):
            assert banned not in text, (
                f"the production release workflow must not reference the "
                f"retired legacy packaging lane: {banned}"
            )

    def test_inno_setup_lane_stays_retired_and_banner_marked(self) -> None:
        script = TOOLKIT / "packaging" / "windows" / "build_windows.ps1"
        iss = TOOLKIT / "packaging" / "windows" / "installer.iss"
        for path in (script, iss):
            text = path.read_text(encoding="utf-8")
            assert "RETIRED PACKAGING LANE" in text[:800], (
                f"{path.name} must open with the retired-lane banner so no "
                "one mistakes it for the supported installer path"
            )
        # the quarantine completion re-aimed the legacy GUI freeze at the
        # quarantined entry, never a resurrected root launcher
        iss_text = iss.read_text(encoding="utf-8")
        assert "legacy" in iss_text.lower()

    def test_ci_legacy_gui_lane_is_explicit(self) -> None:
        """The CI may run the archival GUI suite only as an explicitly
        listed legacy/tests_ui lane; the default gate (testpaths) must
        never readmit it."""
        text = self.CI_WF.read_text(encoding="utf-8")
        assert "legacy/tests_ui/" in text, (
            "the CI's GUI lane must point at the quarantined suite home"
        )
        # the CI lane never runs a root-level tests/ui family
        assert "tests/ui/" not in text, (
            "the retired root tests/ui family must not re-enter CI"
        )

    def test_pyproject_testpaths_exclude_legacy(self) -> None:
        text = (TOOLKIT / "pyproject.toml").read_text(encoding="utf-8")
        m = re.search(r"testpaths\s*=\s*\[(.*?)\]", text, re.DOTALL)
        assert m, "pyproject testpaths not found"
        paths = re.findall(r'"([^"]+)"', m.group(1))
        assert "legacy" not in paths and "legacy/tests_ui" not in paths, (
            "the quarantined GUI suite must stay OUT of the default "
            f"testpaths gate (current: {paths})"
        )

    def test_legacy_gui_freeze_only_from_the_quarantine(self) -> None:
        """The legacy pharmaml.spec (the frozen-equivalent test tier) must
        keep pointing at the quarantined launcher; a root studio.py
        entry would mean the launcher got resurrected."""
        spec = (
            TOOLKIT / "packaging" / "windows" / "pharmaml.spec"
        ).read_text(encoding="utf-8")
        assert 'TOOLKIT / "legacy" / "studio.py"' in spec
        assert 'TOOLKIT / "studio.py"' not in spec
