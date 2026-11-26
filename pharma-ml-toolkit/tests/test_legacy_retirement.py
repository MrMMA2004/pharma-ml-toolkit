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
        # the only sanctioned textual references are the quarantined path
        root_launch = re.search(r"(?<!legacy/)`?python (legacy/)?studio\.py", text)
        assert root_launch is not None and root_launch.group(1) == "legacy/", (
            "README must not present a root-level studio.py launch command"
        )
        assert "RETIRED" in text
        assert "React/Tauri 2" in text
