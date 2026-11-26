"""Mono-font adoption (part7 §3.2, R29-a).

Technical-content widgets — SMILES inputs, serialized configs,
numerical logs — carry an EXPLICIT mono QFont (beats QSS), so they
stay mono under qt-material/qdarkstyle too, where the studio QSS
does not apply.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QPlainTextEdit

from ui.font_loader import mono_font, set_mono
from ui.pages.data_explorer import DataExplorerPage
from ui.pages.docking import DockingWorkbenchPage
from ui.pages.generative import GenerativeWorkbenchPage


def _families(widget) -> list[str]:
    return widget.font().families()


def test_set_mono_sets_the_chain(studio_app) -> None:
    pane = QPlainTextEdit()
    set_mono(pane, 9)
    assert pane.font().families()[0] == "JetBrains Mono"
    assert pane.font().families()[-1] == "monospace"
    assert pane.font().pointSize() == 9


def test_set_mono_tolerates_none(studio_app) -> None:
    set_mono(None)  # no-op, never raises
    set_mono(object())  # non-widget: swallowed


def test_generative_smiles_pane_is_mono(studio_app) -> None:
    page = GenerativeWorkbenchPage()
    families = _families(page._sa_molecules)
    assert families[0] == "JetBrains Mono"


def test_docking_box_config_is_mono(studio_app) -> None:
    page = DockingWorkbenchPage()
    families = _families(page._box_edit)
    assert families[0] == "JetBrains Mono"


def test_data_explorer_log_panes_are_mono(studio_app) -> None:
    page = DataExplorerPage()
    for pane in (page._audit_output, page._split_audit_output,
                 page._sar_output, page._split_audit_ranges):
        assert _families(pane)[0] == "JetBrains Mono", pane.objectName()


def test_mono_across_providers(studio_app) -> None:
    """The three-provider font contract (§16.10 one-ACTIVE-provider +
    part7 §3.2): studio styles QPlainTextEdit mono via QSS; native
    keeps the explicit ctor-time QFont (no stylesheet font rules);
    a FOREIGN provider (qt-material) owns its fonts at polish time —
    Roboto replaces the ctor font, and an explicit re-set sticks
    until the next polish.  That is provider semantics, not a leak."""
    from ui.theme_manager import ThemeManager

    page = DataExplorerPage()
    tm = ThemeManager(studio_app)

    tm.apply("native")
    assert _families(page._audit_output)[0] == "JetBrains Mono"

    tm.apply("studio", "dark")
    assert _families(page._audit_output)[0] == "JetBrains Mono"

    tm.apply("qt-material", "dark_teal.xml")
    assert _families(page._audit_output)[0] == "Roboto"
    # an explicit re-set under a foreign provider wins until re-polish
    set_mono(page._audit_output)
    assert _families(page._audit_output)[0] == "JetBrains Mono"


def test_prose_labels_stay_ui_font(studio_app) -> None:
    """Discipline cuts both ways: status/prose labels are NOT mono."""
    page = GenerativeWorkbenchPage()
    assert _families(page._report)[0] != "JetBrains Mono"


def test_mono_font_helper_shape(studio_app) -> None:
    font = mono_font()
    assert font.families() == [
        "JetBrains Mono", "Cascadia Code", "Consolas", "monospace"]
    label = QLabel()
    set_mono(label)
    assert label.font().families() == font.families()
