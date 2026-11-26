"""``python studio.py --doctor`` -- the Studio environment self-check.

Round 237's answer to the Windows support pattern (v1.2.147: a user's
Miniconda ``py312`` stack died at ``import PySide6`` with ``DLL load
failed while importing Shiboken`` -- an environment break the raw
traceback gave them nothing to act on).  One command now reports the
whole launch environment: the interpreter, the Qt binding family, the
GUI providers and the science stack, with a suggested fix printed
under every failing check.

What the doctor deliberately does NOT do: mutate the environment.  It
only reads (imports, wheel metadata) and reports -- fixes stay the
user's decision, printed as copy-pasteable commands built from
``sys.executable`` so they target the interpreter that ran the doctor.

Design notes:

* The PySide6/shiboken6 checks read wheel metadata via
  ``importlib.metadata`` FIRST and attempt the import second -- the
  metadata survives exactly the DLL-level breakage that kills the
  import, so the doctor can still say "wheel 6.8.3 present but import
  fails" and detect a PySide6/shiboken6 version MISMATCH (the classic
  partial-upgrade signature) even when neither imports.
* Exit status: ``0`` when no check failed (warnings allowed --
  optional feature sets degrade, they do not block), ``1`` when any
  required check failed.  Scriptable, CI-friendly.
* Check levels mirror ``requirements.txt``'s own contract: the Core
  and Studio-GUI sections are required (FAIL when broken); the
  optional feature sets sampled here (rdkit, matplotlib, qdarkstyle)
  warn -- the app launches, the corresponding pages/features degrade.

Run it with::

    python studio.py --doctor
"""

from __future__ import annotations

import importlib
import importlib.metadata
import sys
from dataclasses import dataclass
from pathlib import Path

from .headless import ensure_headless

__all__ = [
    "FAIL",
    "OK",
    "WARN",
    "Check",
    "collect_checks",
    "doctor_exit_code",
    "format_report",
    "run_doctor",
]

OK, WARN, FAIL = "OK", "WARN", "FAIL"

#: The Qt wheel family -- one version, four distributions, must move
#: as a set (the partial-upgrade lesson behind the round-237 guard).
QT_FAMILY = "PySide6 PySide6-Addons PySide6-Essentials shiboken6"

_RULE = "=" * 72


@dataclass
class Check:
    """One doctor line: name, status, human detail, optional fix."""

    name: str
    status: str
    detail: str
    fix: str | None = None


def _py() -> str:
    return sys.executable or "python"


def _pip(*args: str) -> str:
    return f'"{_py()}" -m pip ' + " ".join(args)


def _meta(dist: str) -> str | None:
    """Wheel metadata version, or None when the distribution is absent."""
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _toolkit_version() -> str:
    version_file = Path(__file__).resolve().parents[2] / "VERSION"
    try:
        return version_file.read_text().strip()
    except OSError:  # pragma: no cover - VERSION always ships
        return "?"


# -- individual checks ---------------------------------------------------


def check_python() -> Check:
    """Interpreter floor: 3.10+ required (requirements.txt), 3.12 verified."""
    vi = sys.version_info
    detail = (
        f"{vi.major}.{vi.minor}.{vi.micro}"
        f" ({sys.implementation.name}, {sys.platform})"
    )
    if vi < (3, 10):
        return Check(
            "python", FAIL, f"{detail} -- 3.10+ required",
            "install Python 3.10+ (3.12 is the verified pin)",
        )
    if vi < (3, 12):
        return Check("python", WARN, f"{detail} -- works, 3.12 is the verified pin")
    return Check("python", OK, detail)


def check_pyside6() -> Check:
    """The Qt binding -- metadata first, import second (see module notes)."""
    installed = _meta("PySide6")
    try:
        import PySide6
    except Exception as exc:  # noqa: BLE001 - the exact user failure is an ImportError
        if installed is None:
            return Check("PySide6", FAIL, f"not installed ({exc})", _pip("install PySide6"))
        return Check(
            "PySide6", FAIL,
            f"wheel {installed} present but import fails: {exc}",
            _pip(f"uninstall -y {QT_FAMILY}") + " then " + _pip("install --upgrade PySide6"),
        )
    version = getattr(PySide6, "__version__", None) or installed or "?"
    return Check("PySide6", OK, str(version))


def check_shiboken6() -> Check:
    """shiboken6 -- imported, and version-matched to the PySide6 wheel."""
    installed = _meta("shiboken6")
    try:
        import shiboken6  # noqa: F401 - presence check
    except Exception as exc:  # noqa: BLE001 - DLL failures land here
        if installed is None:
            return Check(
                "shiboken6", FAIL, f"not installed ({exc})",
                _pip(f"uninstall -y {QT_FAMILY}") + " then " + _pip("install --upgrade PySide6"),
            )
        return Check(
            "shiboken6", FAIL,
            f"wheel {installed} present but import fails: {exc}",
            _pip(f"uninstall -y {QT_FAMILY}") + " then " + _pip("install --upgrade PySide6"),
        )
    pyside = _meta("PySide6")
    if installed and pyside and installed != pyside:
        return Check(
            "shiboken6", FAIL,
            f"{installed} but PySide6 is {pyside} -- version mismatch",
            _pip(f"uninstall -y {QT_FAMILY}") + " then " + _pip("install --upgrade PySide6"),
        )
    match = " (matches PySide6)" if pyside and installed == pyside else ""
    return Check("shiboken6", OK, f"{installed or '?'}{match}")


def check_qtwidgets(applied: dict[str, str | None]) -> Check:
    """The exact import ``ui.run`` performs before anything else can load."""
    try:
        from PySide6.QtWidgets import QApplication  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return Check(
            "QtWidgets", FAIL, f"QApplication import failed: {exc}",
            _pip(f"uninstall -y {QT_FAMILY}") + " then " + _pip("install --upgrade PySide6"),
        )
    plat = applied.get("platform") or "preset"
    return Check("QtWidgets", OK, f"QApplication importable (platform: {plat})")


def _font_probe() -> tuple[bool, str | None]:
    """Register the bundle and report whether Qt is usable for it.

    QFontDatabase requires a QGuiApplication to exist (its statics
    touch platform font databases); the doctor runs before any window
    does, so a minimal offscreen app instance is created ONLY when
    none exists yet.  Returns (ok, detail) — ok=False means the probe
    itself could not run (never a Studio failure).
    """
    try:
        # QApplication (not the lighter QGuiApplication): it satisfies
        # the font-database statics the same way, and — when the probe
        # is the first to build an app instance in this interpreter —
        # the surviving instance can still parent QWidgets later
        # (pytest-qt reuses whatever instance exists; a bare
        # QGuiApplication would abort every later QWidget test under
        # PySide6 6.12).
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance()
        if app is None:
            QApplication(
                ["pharmaml-doctor", "-platform", "offscreen"])
        from .font_loader import register_fonts

        register_fonts()
    except Exception as exc:  # noqa: BLE001 - diagnostics never raise
        return False, f"probe failed: {exc}"
    return True, None


def check_font_plex() -> Check:
    """Mineral §5: report the interface font's status through diagnostics.

    A missing IBM Plex Sans is NOT a failure — the fallback chain
    (Segoe UI → Arial) carries the UI — so this check warns instead of
    failing, and the fix names the bundled file rather than a pip line.
    """
    ok, detail = _font_probe()
    if not ok:
        return Check("font: IBM Plex Sans", WARN, detail)
    try:
        from .font_loader import has_plex

        if has_plex():
            return Check("font: IBM Plex Sans", OK, "bundled and registered")
        return Check(
            "font: IBM Plex Sans", WARN,
            "not available -- fallback chain (Segoe UI, Arial) in use",
            "restore fonts/IBMPlexSans-*.ttf from the toolkit archive",
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics never raise
        return Check("font: IBM Plex Sans", WARN, f"probe failed: {exc}")


def check_font_mono() -> Check:
    """Mineral §5: the technical font's diagnostic status (SMILES/logs)."""
    ok, detail = _font_probe()
    if not ok:
        return Check("font: JetBrains Mono", WARN, detail)
    try:
        from .font_loader import has_mono

        if has_mono():
            return Check("font: JetBrains Mono", OK, "bundled and registered")
        return Check(
            "font: JetBrains Mono", WARN,
            "not available -- fallback chain (Cascadia Code, Consolas) in use",
            "restore fonts/JetBrainsMono-Regular.ttf from the toolkit archive",
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics never raise
        return Check("font: JetBrains Mono", WARN, f"probe failed: {exc}")


def _import_check(
    name: str,
    module: str,
    *,
    fix: str | None = None,
    warn: bool = False,
    note: str = "",
) -> Check:
    """Generic import probe used for the providers and the science stack."""
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - report whatever broke
        detail = f"{type(exc).__name__}: {exc}"
        if note:
            detail = f"{detail} -- {note}"
        return Check(name, WARN if warn else FAIL, detail, fix)
    version = getattr(mod, "__version__", None) or _meta(module) or "?"
    return Check(name, OK, f"{version}{f' -- {note}' if note else ''}")


# -- assembly, formatting, exit -------------------------------------------


def collect_checks() -> list[Check]:
    """Run every check; returns the report rows (never raises)."""
    applied = ensure_headless()
    return [
        check_python(),
        check_pyside6(),
        check_shiboken6(),
        check_qtwidgets(applied),
        check_font_plex(),
        check_font_mono(),
        _import_check("pyqtgraph", "pyqtgraph", fix=_pip("install pyqtgraph==0.13.7")),
        _import_check(
            "qt-material", "qt_material",
            fix=_pip("install qt-material==2.17"),
            note="the default theme provider -- Studio needs it to start",
        ),
        _import_check("numpy", "numpy", fix=_pip("install numpy")),
        _import_check("pandas", "pandas", fix=_pip("install pandas")),
        _import_check("scikit-learn", "sklearn", fix=_pip("install scikit-learn")),
        _import_check("scipy", "scipy", fix=_pip("install scipy")),
        _import_check("joblib", "joblib", fix=_pip("install joblib")),
        _import_check("openpyxl", "openpyxl", fix=_pip("install openpyxl")),
        _import_check("tqdm", "tqdm", fix=_pip("install tqdm")),
        _import_check(
            "rdkit", "rdkit", warn=True, fix=_pip("install rdkit"),
            note="optional -- chemistry pages need it",
        ),
        _import_check(
            "matplotlib", "matplotlib", warn=True, fix=_pip("install matplotlib"),
            note="optional -- report/export figures need it",
        ),
        _import_check(
            "qdarkstyle", "qdarkstyle", warn=True, fix=_pip("install qdarkstyle==3.2.3"),
            note="optional -- the alternative dark theme provider",
        ),
    ]


def doctor_exit_code(checks: list[Check]) -> int:
    """0 when no check failed (warnings allowed), 1 otherwise."""
    return 1 if any(c.status == FAIL for c in checks) else 0


def format_report(checks: list[Check], *, version: str) -> str:
    """The full console report (pure; trailing newline included)."""
    name_w = max(len(c.name) for c in checks) + 2
    counts = {OK: 0, WARN: 0, FAIL: 0}
    lines = [
        _RULE,
        f"PharmaML Studio -- environment doctor (toolkit {version})",
        _RULE,
    ]
    for c in checks:
        counts[c.status] = counts.get(c.status, 0) + 1
        lines.append(f"{c.name:<{name_w}}{c.status:<7}{c.detail}")
        if c.fix:
            lines.append(f'{"":<{name_w}}{"":<7}fix: {c.fix}')
    lines += [
        _RULE,
        (
            f"{counts[OK]} OK / {counts[WARN]} warn / {counts[FAIL]} failed"
            f" -> exit {doctor_exit_code(checks)}"
        ),
        f'Verified dependency pins: requirements.txt ("{_py()}" -m pip install -r requirements.txt)',
        f'Re-run after fixing: "{_py()}" studio.py --doctor',
        _RULE,
        "",
    ]
    return "\n".join(lines)


def run_doctor(stream=None) -> int:
    """Collect, report, return the exit status (see module docstring)."""
    checks = collect_checks()
    report = format_report(checks, version=_toolkit_version())
    (stream or sys.stdout).write(report)
    return doctor_exit_code(checks)
