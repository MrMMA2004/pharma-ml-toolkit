"""QSettings persistence wrapper for PharmaML Studio (directive 16.15).

Window geometry/position, the active theme, and the last navigation
section survive restarts through QSettings (INI-backed on Linux). No
credentials or secrets are ever stored here -- only UI state.
"""

from __future__ import annotations

from PySide6.QtCore import QByteArray, QSettings

ORG = "PharmaML"
APP = "Studio"


class AppSettings:
    """Thin typed facade over QSettings for UI-state persistence."""

    def __init__(self) -> None:
        self._s = QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope, ORG, APP)

    # -- window geometry --------------------------------------------------
    def save_geometry(self, geometry: QByteArray, state: QByteArray | None = None) -> None:
        self._s.setValue("window/geometry", geometry)
        if state is not None:
            self._s.setValue("window/state", state)
        self._s.sync()  # same-process round-trips need an explicit flush

    def geometry(self) -> QByteArray | None:
        raw = self._s.value("window/geometry")
        return raw if isinstance(raw, QByteArray) else None

    def state(self) -> QByteArray | None:
        raw = self._s.value("window/state")
        return raw if isinstance(raw, QByteArray) else None

    # -- navigation --------------------------------------------------------
    def save_nav(self, section: str) -> None:
        self._s.setValue("nav/section", section)
        self._s.sync()

    def nav(self) -> str | None:
        raw = self._s.value("nav/section")
        return raw if isinstance(raw, str) and raw else None

    # -- passthrough (for the theme manager and pages) ----------------------
    def settings(self) -> QSettings:
        return self._s

    def clear(self) -> None:
        """Drop every stored key (the settings page's 'reset' action)."""
        self._s.clear()
