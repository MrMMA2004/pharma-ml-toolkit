"""PharmaML Studio -- the PySide6 front end to the pharmaml toolkit.

Directive #16 (GUI redesign): PySide6 application architecture,
PyQtGraph scientific visualization, Qt-Material primary theming and
QDarkStyleSheet as the verified alternative, all behind one
centralized theme manager. The Tkinter playground stays fully
functional alongside until every section is migrated and verified.

Import order matters on headless boxes: call :func:`ensure_headless`
before touching PySide6 (it is also called automatically by
:func:`run` and by the test suite's conftest).
"""

from __future__ import annotations

__all__ = ["__version__", "ensure_headless", "run"]

__version__ = "1.0"

from .headless import ensure_headless


def run(argv: list[str] | None = None) -> int:
    """Create the QApplication and run the Studio main window.

    Returns the Qt exit code. ``ensure_headless()`` runs first so the
    offscreen platform and EGL fallback apply before any Qt import.
    """
    ensure_headless()
    from PySide6.QtWidgets import QApplication

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
