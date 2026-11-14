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


# -- Mineral + alternates: the studio provider (token-built identity) ----
def test_studio_provider_themes_and_default(theme_manager: ThemeManager) -> None:
    assert ThemeManager.available_themes("studio") == (
        "light", "dark",                       # 01 / MINERAL
        "spectral", "spectral-dark",           # 02 / SPECTRAL
        "catalyst", "catalyst-dark",           # 03 / CATALYST
    )
    palette = theme_manager.apply("studio", "light")
    assert theme_manager.provider == "studio"
    assert theme_manager.theme == "light"
    assert not palette.dark
    assert palette.accent == "#076F73"
    # §4.2: the Mineral scientific data series, NOT the interface accent
    assert palette.series[0] == "#0072B2"
    assert "#076F73" not in palette.series


def test_alternative_directions_apply_and_persist(
        theme_manager: ThemeManager, studio_app, tmp_path) -> None:
    """Spectral/Catalyst switch cleanly through the SAME provider path
    (QSS swap, palette signal) and persist like any studio theme."""
    from PySide6.QtCore import QSettings

    seen: list[tuple[str, str]] = []
    theme_manager.theme_applied.connect(
        lambda p, t, _c: seen.append((p, t)))

    spectral = theme_manager.apply("studio", "spectral")
    assert theme_manager.theme == "spectral"
    assert not spectral.dark and spectral.accent == "#5048A5"
    assert "#F5F2EC" in studio_app.styleSheet()   # warm paper canvas

    catalyst = theme_manager.apply("studio", "catalyst")
    assert theme_manager.theme == "catalyst"
    assert not catalyst.dark and catalyst.accent == "#2E5F7A"
    assert "#172F43" in studio_app.styleSheet()   # the navy frame shell

    catalyst_dark = theme_manager.apply("studio", "catalyst-dark")
    assert catalyst_dark.dark and catalyst_dark.accent == "#C6E84B"
    spectral_dark = theme_manager.apply("studio", "spectral-dark")
    assert spectral_dark.dark and spectral_dark.accent == "#9D96E8"
    # both dark variants carry the shared dark scientific series
    assert spectral_dark.series == catalyst_dark.series

    ini = tmp_path / "settings.ini"
    settings = QSettings(str(ini), QSettings.Format.IniFormat)
    managed = ThemeManager(studio_app, settings)
    managed.apply("studio", "spectral")
    managed.save_state()
    second = ThemeManager(studio_app,
                          QSettings(str(ini), QSettings.Format.IniFormat))
    second.restore_state()
    assert second.provider == "studio" and second.theme == "spectral"
    assert ("studio", "spectral") in seen


def test_studio_dark_palette_matches_canvas(theme_manager: ThemeManager) -> None:
    palette = theme_manager.apply("studio", "dark")
    assert palette.dark
    assert palette.background == "#0D1B23"
    assert palette.accent == "#79D9CF"
    assert palette.series[-1] == "#E1E7EB"  # the ink companion
    # the grid is the dark border/subtle token, not a generic gray
    assert palette.grid == (0x26, 0x3A, 0x45)


def test_studio_qss_carries_the_identity(studio_app) -> None:
    from ui.studio_theme import studio_qss

    light = studio_qss("light")
    assert "#F4F7F8" in light      # Mineral canvas
    assert "#076F73" in light      # teal action color
    assert "#E3F3F1" in light      # action/selected
    assert "IBM Plex Sans" in light and "JetBrains Mono" in light
    dark = studio_qss("dark")
    assert "#0D1B23" in dark and "#14232C" in dark
    assert "#79D9CF" in dark       # bright teal action color
    # one identity, two themes: no token bleeds across
    assert "#0D1B23" not in light
    # the retired part7 identity is gone from the MINERAL sheets (the
    # same hues return only as the opt-in Catalyst direction)
    for retired in ("#C6E84B", "#F3F5F1", "#172F43", "#101B24"):
        assert retired not in light and retired not in dark, retired


def test_alternative_qss_carries_each_identity(studio_app) -> None:
    """Each alternate direction's QSS carries its spec anchors — and
    only its own (no cross-direction token bleed)."""
    from ui.studio_theme import studio_qss

    spectral = studio_qss("spectral")
    assert "#F5F2EC" in spectral and "#5048A5" in spectral
    assert "#172F43" not in spectral and "#C6E84B" not in spectral
    spectral_dark = studio_qss("spectral-dark")
    assert "#151321" in spectral_dark and "#9D96E8" in spectral_dark
    catalyst = studio_qss("catalyst")
    assert "#172F43" in catalyst and "#F3F5F1" in catalyst
    assert "#5048A5" not in catalyst
    catalyst_dark = studio_qss("catalyst-dark")
    assert "#0E1D2B" in catalyst_dark and "#C6E84B" in catalyst_dark
    assert "#F5F2EC" not in catalyst_dark


def test_alternative_direction_geometry(studio_app) -> None:
    """Spectral reads 'rounded panels', Catalyst 'squared cards' — the
    per-direction radius ramp, delivered through the one shared QSS."""
    from ui.studio_theme import _theme_radius

    assert _theme_radius("light") == (4, 6, 8, 12)      # Mineral §6
    assert _theme_radius("dark") == (4, 6, 8, 12)
    assert _theme_radius("spectral") == (6, 8, 10, 14)  # one step rounder
    assert _theme_radius("spectral-dark") == (6, 8, 10, 14)
    assert _theme_radius("catalyst") == (2, 4, 6, 8)    # squared
    assert _theme_radius("catalyst-dark") == (2, 4, 6, 8)


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
    """No saved preference → studio/light (Mineral identity first)."""
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
