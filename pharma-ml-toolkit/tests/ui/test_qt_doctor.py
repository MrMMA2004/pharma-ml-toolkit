"""Round 237 -- the Qt import guard and the ``studio.py --doctor`` self-check.

The v1.2.147 Windows report behind these features (Miniconda py312)::

    ImportError: DLL load failed while importing Shiboken: The specified
    module could not be found.

...dies inside the binding's own import chain, before any Studio code.
The guard must turn that into exit code 2 with a fix recipe; the
doctor must survive (and correctly diagnose) a broken binding; the
healthy sandbox must answer ``--doctor`` with exit 0.

The two subprocess tests shadow ``PySide6`` through ``PYTHONPATH``
(a fake package whose import raises the exact user error) -- Python
resolves ``PYTHONPATH`` entries ahead of ``site-packages``, so the
shadow is a faithful stand-in for a broken install without touching
the real wheel set.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ui.doctor import (
    FAIL,
    OK,
    WARN,
    Check,
    collect_checks,
    doctor_exit_code,
    format_report,
)
from ui.qt_import_guard import QT_ENV_EXIT_CODE, qt_import_guidance

TK = Path(__file__).resolve().parents[2]

_USER_ERROR = (
    "DLL load failed while importing Shiboken: "
    "The specified module could not be found."
)


def _run_studio(args: list[str], extra_env: dict[str, str] | None = None):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    # part18 §10: the launcher lives in the quarantined legacy/ area now
    return subprocess.run(
        [sys.executable, str(TK / "legacy" / "studio.py"), *args],
        cwd=TK,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )


def _shadow_dir(tmp_path: Path) -> Path:
    """A PYTHONPATH dir whose PySide6 raises the user's exact error."""
    fake = tmp_path / "PySide6"
    fake.mkdir()
    (fake / "__init__.py").write_text(f"raise ImportError({_USER_ERROR!r})\n")
    return tmp_path


# -- guidance formatter (pure) -------------------------------------------


def test_guidance_dll_failure_family() -> None:
    """A broken Qt stack is identified as retired, not current Studio."""
    text = qt_import_guidance(ImportError(_USER_ERROR))
    assert "retired PySide6 interface" in text
    assert "archived and unsupported" in text
    assert "React + TypeScript hosted by Tauri 2" in text
    assert "Next.js console is a separate" in text
    assert "No PySide6 reinstall recipe" in text
    assert "--doctor" not in text


def test_guidance_missing_module_family() -> None:
    """Only archival reproduction is offered a local-extra install command."""
    text = qt_import_guidance(ModuleNotFoundError("No module named 'PySide6'"))
    assert "For archival reproduction only" in text
    assert "-m pip install -e \".[studio]\"" in text
    assert sys.executable in text  # command targets the launching interpreter
    assert "supported React/Tauri application" in text
    assert "uninstall" not in text


# -- doctor formatting / exit code (pure) ---------------------------------


def test_doctor_report_format_and_exit_codes() -> None:
    """Fail lines carry their fix; warns never fail the exit status."""
    checks = [
        Check("python", OK, "3.12.4"),
        Check("rdkit", WARN, "not installed", "install rdkit"),
        Check("PySide6", FAIL, "wheel 6.8.3 present but import fails", "reinstall"),
    ]
    report = format_report(checks, version="X.Y.Z")
    for needle in ("python", "PySide6", "WARN", "FAIL", "fix: reinstall"):
        assert needle in report
    assert "1 OK / 1 warn / 1 failed" in report
    assert doctor_exit_code(checks) == 1
    assert doctor_exit_code(checks[:2]) == 0  # OK + WARN only


def test_doctor_collect_live() -> None:
    """On the healthy sandbox every check answers, the stack is OK.

    18 checks since the Mineral round: the 16 part7 checks plus the
    two font-status checks (§5: the interface/technical font's
    availability is reported through diagnostics, WARN not FAIL)."""
    checks = collect_checks()
    assert len(checks) == 18
    assert all(c.status in (OK, WARN, FAIL) for c in checks)
    by_name = {c.name: c for c in checks}
    for name in ("python", "PySide6", "shiboken6", "QtWidgets",
                 "pyqtgraph", "qt-material", "numpy", "rdkit",
                 "font: IBM Plex Sans", "font: JetBrains Mono"):
        assert by_name[name].status == OK, (name, by_name[name])
    assert "matches PySide6" in by_name["shiboken6"].detail
    assert doctor_exit_code(checks) == 0  # warnings do not fail the doctor


# -- end to end: the real studio.py entry points --------------------------


def test_doctor_cli_subprocess() -> None:
    """``python studio.py --doctor`` answers exit 0 with the full report."""
    proc = _run_studio(["--doctor"])
    assert proc.returncode == 0, proc.stderr[-2000:]
    for needle in ("environment doctor", "PySide6", "shiboken6",
                   "QtWidgets", "OK", "requirements.txt"):
        assert needle in proc.stdout, (needle, proc.stdout[-2000:])


def test_guarded_qt_import_exit_code(tmp_path: Path) -> None:
    """A broken PySide6 -> exit 2 + the fix recipe, no traceback wall."""
    proc = _run_studio(
        [], extra_env={"PYTHONPATH": str(_shadow_dir(tmp_path))}
    )
    assert proc.returncode == QT_ENV_EXIT_CODE
    assert "could not load PySide6" in proc.stderr
    assert "uninstall -y PySide6 PySide6-Addons PySide6-Essentials shiboken6" in proc.stderr
    assert "Traceback" not in proc.stderr  # the point of the guard


def test_doctor_survives_broken_binding(tmp_path: Path) -> None:
    """``--doctor`` on a broken PySide6 diagnoses it and exits 1 (not crash)."""
    proc = _run_studio(
        ["--doctor"], extra_env={"PYTHONPATH": str(_shadow_dir(tmp_path))}
    )
    assert proc.returncode == 1
    assert "PySide6" in proc.stdout
    assert "FAIL" in proc.stdout
    assert "fix:" in proc.stdout  # every failure carries its remedy
    assert "python" in proc.stdout  # the other checks still reported
    assert "Traceback" not in proc.stdout + proc.stderr
