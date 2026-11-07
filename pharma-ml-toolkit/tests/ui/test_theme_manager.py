"""Theme manager tests (directive 16.9-16.11): clean switching, palettes."""

from __future__ import annotations

import pytest

from ui.theme_manager import PROVIDERS, ChartPalette, ThemeManager


def test_providers_registry_is_exact() -> None:
    assert PROVIDERS == ("qt-material", "qdarkstyle", "native")


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
