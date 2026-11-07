"""Centralized theme management for PharmaML Studio.

Directive #16 (GUI redesign, 16.9/16.10/16.11): one ACTIVE global theme
provider at a time -- Qt-Material (primary Material Design provider),
QDarkStyleSheet (verified alternative), or native Qt styling (fallback).
Switching providers always clears the previous stylesheet first so the
two engines never compete, and every switch re-emits a
:class:`ChartPalette` so PyQtGraph plots re-style to match (a Qt
stylesheet never reaches inside pyqtgraph's own rendering).

Usage::

    tm = ThemeManager(app)
    tm.apply(provider="qt-material", theme="dark_teal.xml")
    ...
    tm.apply(provider="qdarkstyle")          # dark, single theme
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtWidgets import QApplication

PROVIDERS: tuple[str, ...] = ("qt-material", "qdarkstyle", "native")

# Curated, toolkit-branded subset of qt-material's theme list (teal is
# the toolkit accent; blue/indigo families deliberately not offered).
MATERIAL_THEMES: tuple[str, ...] = (
    "dark_teal.xml",
    "light_teal.xml",
    "dark_cyan.xml",
    "light_cyan.xml",
    "dark_amber.xml",
    "light_amber.xml",
)
MATERIAL_DEFAULT = "dark_teal.xml"


@dataclass(frozen=True)
class ChartPalette:
    """Colors PyQtGraph widgets should use under the active theme.

    ``background``/``foreground``/``grid`` are color strings pyqtgraph
    understands; ``series`` cycles through plot pens; ``accent`` marks
    selections/highlights. Derived per provider+theme (16.11: "Do not
    assume that changing the application's stylesheet automatically
    updates every PyQtGraph plot element").
    """

    background: str = "w"
    foreground: str = "k"
    grid: tuple[int, int, int] = (200, 200, 200)
    accent: str = "#009688"
    series: tuple[str, ...] = field(
        default_factory=lambda: (
            "#009688",
            "#E65100",
            "#5D4037",
            "#7B1FA2",
            "#33691E",
            "#0277BD",
        )
    )
    dark: bool = False


_DARK_PALETTES: dict[str, ChartPalette] = {
    "qt-material": ChartPalette(
        background="#1d1f21",
        foreground="#e0e0e0",
        grid=(70, 74, 77),
        accent="#009688",
        series=("#80CBC4", "#FFB74D", "#A1887F", "#CE93D8", "#AED581", "#81D4FA"),
        dark=True,
    ),
    "qdarkstyle": ChartPalette(
        background="#323232",
        foreground="#dddddd",
        grid=(78, 78, 78),
        accent="#4DA1FF",
        series=("#6FB2E0", "#D6956C", "#9CCC65", "#CE93D8", "#FFD54F", "#90A4AE"),
        dark=True,
    ),
}
_LIGHT_PALETTES: dict[str, ChartPalette] = {
    "qt-material": ChartPalette(background="w", foreground="k", grid=(200, 200, 200), accent="#00796B"),
    "qdarkstyle": ChartPalette(background="w", foreground="k", grid=(210, 210, 210), accent="#37474F"),
    "native": ChartPalette(background="w", foreground="k", grid=(200, 200, 200), accent="#00796B"),
}


def _palette_for(provider: str, theme: str) -> ChartPalette:
    """Map provider+theme to a chart palette (dark themes get dark plots)."""
    if provider == "qt-material":
        dark = theme.startswith("dark_")
        return _DARK_PALETTES["qt-material"] if dark else _LIGHT_PALETTES["qt-material"]
    if provider == "qdarkstyle":
        return _DARK_PALETTES["qdarkstyle"]
    return _LIGHT_PALETTES["native"]


class ThemeManager(QObject):
    """Owns the active provider/theme and applies it to the application.

    Signals
    -------
    theme_applied(str, str, ChartPalette)
        Emitted after every successful application with the provider,
        theme name and the chart palette plots should adopt.
    """

    theme_applied = Signal(str, str, object)

    def __init__(self, app: QApplication, settings: QSettings | None = None) -> None:
        super().__init__()
        self._app = app
        self._settings = settings
        self._provider = "qt-material"
        self._theme = MATERIAL_DEFAULT
        self._density_scale = 0

    # -- queries ---------------------------------------------------------
    @property
    def provider(self) -> str:
        return self._provider

    @property
    def theme(self) -> str:
        return self._theme

    @property
    def density_scale(self) -> int:
        return self._density_scale

    def chart_palette(self) -> ChartPalette:
        return _palette_for(self._provider, self._theme)

    @staticmethod
    def available_themes(provider: str) -> tuple[str, ...]:
        """Theme names offered for a provider (16.9.1 curated list)."""
        if provider == "qt-material":
            return MATERIAL_THEMES
        if provider == "qdarkstyle":
            return ("dark",)
        return ("native",)

    # -- application -----------------------------------------------------
    def apply(self, provider: str, theme: str | None = None, density_scale: int = 0) -> ChartPalette:
        """Switch the active stylesheet cleanly (one provider at a time).

        Raises ``ValueError`` for an unknown provider; a bad theme name
        falls back to the provider default with the fallback recorded.
        """
        if provider not in PROVIDERS:
            raise ValueError(f"unknown theme provider {provider!r} (expected one of {PROVIDERS})")
        themes = self.available_themes(provider)
        if theme is None or theme not in themes:
            theme = themes[0]

        # 16.10.4 step 1: remove the previously active stylesheet BEFORE
        # applying the next one so providers never compete.
        self._app.setStyleSheet("")

        if provider == "qt-material":
            from qt_material import apply_stylesheet

            # qt-material 2.17 carries density through the template `extra`
            # dict (its apply_stylesheet has no density_scale kwarg).
            apply_stylesheet(
                self._app,
                theme=theme,
                extra={"density_scale": str(int(density_scale))},
            )
        elif provider == "qdarkstyle":
            import qdarkstyle

            self._app.setStyleSheet(qdarkstyle.load_stylesheet())

        self._provider = provider
        self._theme = theme
        self._density_scale = density_scale
        palette = self.chart_palette()
        self.theme_applied.emit(provider, theme, palette)
        return palette

    # -- persistence -----------------------------------------------------
    def save_state(self) -> None:
        if self._settings is None:
            return
        self._settings.setValue("theme/provider", self._provider)
        self._settings.setValue("theme/name", self._theme)
        self._settings.setValue("theme/density", self._density_scale)

    def restore_state(self) -> ChartPalette:
        """Apply the persisted theme (or the default when unset)."""
        provider = "qt-material"
        theme: str | None = MATERIAL_DEFAULT
        density = 0
        if self._settings is not None:
            stored_provider = self._settings.value("theme/provider", "")
            if stored_provider in PROVIDERS:
                provider = stored_provider
            stored_theme = self._settings.value("theme/name", "")
            themes = self.available_themes(provider)
            theme = stored_theme if stored_theme in themes else themes[0]
            density = int(str(self._settings.value("theme/density", 0)))
        return self.apply(provider, theme, density_scale=density)
