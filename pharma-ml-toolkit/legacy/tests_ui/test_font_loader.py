"""Font loading (Mineral §5): IBM Plex Sans + JetBrains Mono with fallbacks."""

from __future__ import annotations

from PySide6.QtGui import QFont
from ui.font_loader import (
    _BUNDLED,
    _registered,
    has_mono,
    has_plex,
    mono_font,
    register_fonts,
    ui_font,
)


def test_bundled_fonts_register_and_resolve(studio_app) -> None:
    families = register_fonts()
    # the bundled files exist in the repo and Qt gained them
    assert "IBM Plex Sans" in families
    assert "JetBrains Mono" in families
    assert has_plex() and has_mono()


def test_ui_font_carries_the_full_chain(studio_app) -> None:
    font = ui_font(12)
    assert font.families() == [
        "IBM Plex Sans", "Segoe UI", "Arial", "sans-serif"]
    assert font.pointSize() == 12


def test_mono_font_for_technical_content(studio_app) -> None:
    font = mono_font(9)
    assert font.families()[0] == "JetBrains Mono"
    assert font.families()[-1] == "monospace"


def test_weights_map_to_qt_enum(studio_app) -> None:
    from ui.font_loader import _qt_weight

    assert _qt_weight(400) == QFont.Weight.Normal
    assert _qt_weight(500) == QFont.Weight.Medium
    assert _qt_weight(600) == QFont.Weight.DemiBold
    assert _qt_weight(700) == QFont.Weight.Bold
    assert _qt_weight(999) == QFont.Weight.Normal


def test_registration_is_idempotent(studio_app) -> None:
    first = register_fonts()
    second = register_fonts()
    assert first == second


def test_bundled_list_matches_repo(studio_app) -> None:
    """Every claimed bundle file exists next to its OFL license."""
    from ui.font_loader import _FONT_DIR

    for name in _BUNDLED:
        assert (_FONT_DIR / name).is_file(), name
    assert (_FONT_DIR / "IBMPlexSans-OFL.txt").is_file()
    assert (_FONT_DIR / "JetBrainsMono-OFL.txt").is_file()
    # the retired Inter bundle is gone (license files included)
    assert not (_FONT_DIR / "Inter-Regular.ttf").exists()
    assert not (_FONT_DIR / "Inter-OFL.txt").exists()


def test_missing_bundle_degrades_to_chains(studio_app, monkeypatch,
                                            tmp_path) -> None:
    """§5: never assume the fonts are installed — with an empty
    bundle dir the loader registers nothing new and never raises;
    the fallback chain still constructs a usable font.  (Families
    registered earlier in the process stay available app-wide —
    Qt's font database is global — so only the missing-bundle
    behavior is asserted, not the database state.)"""
    import ui.font_loader as fl

    monkeypatch.setattr(fl, "_FONT_DIR", tmp_path)  # empty dir
    monkeypatch.setattr(fl, "_registered", False)
    assert fl.register_fonts() == []  # nothing NEW from the empty dir
    # the fallback chain still produces a usable font
    font = fl.ui_font(10)
    assert font.families() == [
        "IBM Plex Sans", "Segoe UI", "Arial", "sans-serif"]


def test_registered_flag_module_state() -> None:
    # sanity: the module-level flag is a bool (no import-time Qt)
    assert isinstance(_registered, bool)
