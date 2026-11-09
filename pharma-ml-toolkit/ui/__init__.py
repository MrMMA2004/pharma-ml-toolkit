"""PharmaML Studio -- the PySide6 front end to the pharmaml toolkit.

Directive #16 (GUI redesign): PySide6 application architecture,
PyQtGraph scientific visualization, Qt-Material primary theming and
QDarkStyleSheet as the verified alternative, all behind one
centralized theme manager.  The legacy Tkinter playground was
retired in round 228 -- this application is THE front end.

Import order matters on headless boxes: call :func:`ensure_headless`
before touching PySide6 (it is also called automatically by
:func:`run` and by the test suite's conftest).

Round 237 hardens the launch path for broken desktop environments:
the Qt import is guarded (:mod:`ui.qt_import_guard`) so a broken
PySide6/shiboken6 install exits with an actionable fix recipe instead
of a raw traceback, and ``--doctor`` (:mod:`ui.doctor`) runs the full
environment self-check without starting the GUI.
"""

from __future__ import annotations

import sys

__all__ = ["__version__", "ensure_headless", "run"]

__version__ = "1.0"

from .headless import ensure_headless


def run(argv: list[str] | None = None) -> int:
    """Create the QApplication and run the Studio main window.

    Returns the Qt exit code. ``ensure_headless()`` runs first so the
    offscreen platform and EGL fallback apply before any Qt import;
    ``install_qt_message_filter()`` (round 231) then drops the two
    known-benign Qt startup notices (the Windows fonts-dir advice and
    the Qt 6.8 QStyleHints connect chatter) BEFORE the application
    object is constructed, because that is when they fire.

    Round 237: ``--doctor`` in the arguments runs the environment
    self-check (:mod:`ui.doctor`) and returns its exit status without
    starting the GUI; a failing Qt import returns
    :data:`ui.qt_import_guard.QT_ENV_EXIT_CODE` (2) with the fix
    recipe on stderr instead of raising -- the v1.2.147 Windows
    report (``DLL load failed while importing Shiboken``) dies inside
    the binding's own import chain, before any Studio code, so the
    user deserves the diagnosis rather than the traceback.
    """
    args = sys.argv[1:] if argv is None else argv[1:]
    if "--doctor" in args:
        from .doctor import run_doctor

        return run_doctor()

    ensure_headless()
    from .qt_noise_filter import install_qt_message_filter

    install_qt_message_filter()
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        from .qt_import_guard import QT_ENV_EXIT_CODE, qt_import_guidance

        sys.stderr.write(qt_import_guidance(exc))
        return QT_ENV_EXIT_CODE

    from .app_settings import AppSettings
    from .main_window import MainWindow
    from .theme_manager import ThemeManager

    app = QApplication(argv if argv is not None else [])
    app.setApplicationName("PharmaML Studio")
    app.setOrganizationName("PharmaML")
    settings = AppSettings()
    theme = ThemeManager(app, settings.settings())
    window = MainWindow(theme)
    theme.restore_state()
    window.show()
    return int(app.exec())
