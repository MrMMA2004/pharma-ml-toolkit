"""Archived PySide6 interface source; not the supported PharmaML Studio.

The supported desktop application is React + TypeScript hosted by Tauri 2.
This module and ``legacy/studio.py`` remain for historical reproducibility;
they are unsupported and excluded from the production desktop application.
The ``.[studio]`` extra is retained only for reproducing this archived source.
"""

from __future__ import annotations

import sys

__all__ = ["__version__", "ensure_headless", "run"]

__version__ = "1.0"

from .headless import ensure_headless


def run(argv: list[str] | None = None) -> int:
    """Run the retired PySide6 window for archival reproduction only.

    This is not the supported React/Tauri desktop application. The
    legacy entry point calls ``ensure_headless()`` before Qt imports and
    returns a distinct exit code if the optional Qt binding is unavailable.
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
    # Mineral §5: register the bundled IBM Plex Sans/JetBrains Mono
    # before any widget exists; the fallback chains cover a missing bundle
    from .font_loader import register_fonts

    register_fonts()
    settings = AppSettings()
    theme = ThemeManager(app, settings.settings())
    window = MainWindow(theme)
    theme.restore_state()
    window.show()
    return int(app.exec())
