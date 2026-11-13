"""Theme manager tests (directive 16.9-16.11): clean switching, palettes."""

from __future__ import annotations

import pytest

from ui.theme_manager import PROVIDERS, ChartPalette, ThemeManager


def test_providers_registry_is_exact() -> None:
    # part7: the studio provider (token-built QSS identity) leads;
    # the directive-16 providers remain available
    assert PROVIDERS == ("studio", "qt-material", "qdarkstyle", "native")


def test_apply_material_dark_and_light(theme_manager: ThemeManager) -> None:
    dark = theme_manager.apply("qt-material", "dark_teal.xml")
    assert isinstance(dark, ChartPalette) and dark.dark
    assert theme_manager.provider == "qt-material"
    assert theme_manager.theme == "dark_teal.xml"

    light = theme_manager.apply("qt-material", "light_teal.xml")
    assert not light.dark
    assert theme_manager.chart_palette().background == "w"


def test_apply_clears_previous_stylesheet(theme_manager: ThemeManager, studio_app) -> None:
    theme_manager.apply("qt-material", "dark_teal.xml")
    material_sheet = studio_app.styleSheet()
    assert material_sheet.strip()

    theme_manager.apply("native")
    assert studio_app.styleSheet() == ""  # 16.10.4: no competing stylesheets


def test_qdarkstyle_alternative(theme_manager: ThemeManager, studio_app) -> None:
    palette = theme_manager.apply("qdarkstyle")
    assert theme_manager.provider == "qdarkstyle"
    assert studio_app.styleSheet().strip()  # its stylesheet is active
    assert palette.dark


def test_unknown_provider_rejected(theme_manager: ThemeManager) -> None:
    with pytest.raises(ValueError):
        theme_manager.apply("pyqt5")


def test_bad_theme_falls_back_to_provider_default(theme_manager: ThemeManager) -> None:
    palette = theme_manager.apply("qt-material", "no_such_theme.xml")
    assert theme_manager.theme == "dark_teal.xml"  # curated default
    assert palette.dark


def test_theme_applied_signal_carries_palette(theme_manager: ThemeManager) -> None:
    seen: list[tuple[str, str, ChartPalette]] = []
    theme_manager.theme_applied.connect(lambda p, t, c: seen.append((p, t, c)))
    theme_manager.apply("qdarkstyle")
    assert seen and seen[-1][0] == "qdarkstyle"
    assert isinstance(seen[-1][2], ChartPalette)


def test_available_themes_curated(theme_manager: ThemeManager) -> None:
    material = ThemeManager.available_themes("qt-material")
    assert "dark_teal.xml" in material and "light_teal.xml" in material
    assert ThemeManager.available_themes("qdarkstyle") == ("dark",)
    assert ThemeManager.available_themes("native") == ("native",)


def test_persistence_roundtrip(theme_manager: ThemeManager, studio_app, tmp_path) -> None:
    from PySide6.QtCore import QSettings

    ini = tmp_path / "settings.ini"
    settings = QSettings(str(ini), QSettings.Format.IniFormat)
    managed = ThemeManager(studio_app, settings)
    managed.apply("qt-material", "light_teal.xml", density_scale=1)
    managed.save_state()

    second = ThemeManager(studio_app, QSettings(str(ini), QSettings.Format.IniFormat))
    palette = second.restore_state()
    assert second.provider == "qt-material"
    assert second.theme == "light_teal.xml"
    assert second.density_scale == 1
    assert not palette.dark


# -- part7: the studio provider (token-built visual identity) -------------
def test_studio_provider_themes_and_default(theme_manager: ThemeManager) -> None:
    assert ThemeManager.available_themes("studio") == ("light", "dark")
    palette = theme_manager.apply("studio", "light")
    assert theme_manager.provider == "studio"
    assert theme_manager.theme == "light"
    assert not palette.dark
    assert palette.accent == "#C6E84B"
    # §5: Okabe-Ito scientific series, NOT the interface accent
    assert palette.series[0] == "#0072B2"
    assert "#C6E84B" not in palette.series


def test_studio_dark_palette_matches_canvas(theme_manager: ThemeManager) -> None:
    palette = theme_manager.apply("studio", "dark")
    assert palette.dark
    assert palette.background == "#101B24"
    assert palette.series[-1] == "#CCCCCC"  # the dark-variant gray swap


def test_studio_qss_carries_the_identity(studio_app) -> None:
    from ui.studio_theme import studio_qss

    light = studio_qss("light")
    assert "#F3F5F1" in light      # Warm Ivory canvas
    assert "#172F43" in light      # Deep Navy structure
    assert "#C6E84B" in light      # Electric Lime accent
    assert "Inter" in light and "JetBrains Mono" in light
    dark = studio_qss("dark")
    assert "#101B24" in dark and "#263D4D" in dark
    # one identity, two themes: no token bleeds across
    assert "#101B24" not in light


def test_studio_switch_round_trip(theme_manager: ThemeManager,
                                  studio_app) -> None:
    theme_manager.apply("studio", "light")
    first = studio_app.styleSheet()
    assert first
    theme_manager.apply("studio", "dark")
    second = studio_app.styleSheet()
    assert second != first
    theme_manager.apply("studio", "light")
    assert studio_app.styleSheet() == first


def test_studio_is_the_fresh_install_default(
        studio_app, monkeypatch, tmp_path) -> None:
    """No saved preference → studio/light (part7 identity first)."""
    import os

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    from PySide6.QtCore import QSettings

    tm = ThemeManager(studio_app, QSettings("pharmaml-test", "fresh"))
    palette = tm.restore_state()
    assert tm.provider == "studio"
    assert tm.theme == "light"
    assert not palette.dark


def test_studio_bad_theme_falls_back_to_light(
        theme_manager: ThemeManager) -> None:
    palette = theme_manager.apply("studio", "neon")
    assert theme_manager.theme == "light"
    assert not palette.dark
