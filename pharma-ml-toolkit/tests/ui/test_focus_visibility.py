"""Focus-visibility audit (R30-b, part7 §8 accessibility).

Keyboard users must see WHERE they are in every workspace: the
studio QSS carries :focus rules for every widget CLASS the nine
pages actually instantiate (buttons, inputs, combos were already
covered; tables, lists, trees, tab bars, checkboxes, radios,
sliders and group boxes joined in R30-b), and the interactive
widgets keep a keyboard-reachable focus policy (nothing critical
is NoFocus).

The audit walks each page the same way the smoke suite does and
censuses the focusable widget classes it finds.
"""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTabBar,
    QTableWidget,
    QTreeWidget,
)

from ui.studio_theme import studio_qss

#: The nine workspaces (the nine-card home's destinations).
PAGES = [
    "ui.pages.data_explorer:DataExplorerPage",
    "ui.pages.chemistry:ChemistryWorkbenchPage",
    "ui.pages.pkpd:PKPDWorkbenchPage",
    "ui.pages.biology:BiologyWorkbenchPage",
    "ui.pages.docking:DockingWorkbenchPage",
    "ui.pages.model_analysis:ModelAnalysisPage",
    "ui.pages.ml_workbench:MLWorkbenchPage",
    "ui.pages.generative:GenerativeWorkbenchPage",
    "ui.pages.image_lab:ImageLabPage",
]


def _load(page_spec: str):
    module_name, class_name = page_spec.split(":")
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _focusable_classes(page) -> set[str]:
    """Widget classes on this page that can own keyboard focus."""
    classes = (
        QPushButton, QLineEdit, QPlainTextEdit, QComboBox, QSpinBox,
        QDoubleSpinBox, QCheckBox, QRadioButton, QSlider, QTableWidget,
        QListWidget, QTreeWidget, QTabBar, QGroupBox,
    )
    found: set[str] = set()
    for cls in classes:  # PySide6 findChildren takes one type at a time
        for widget in page.findChildren(cls):
            policy = widget.focusPolicy()
            if policy in (Qt.FocusPolicy.StrongFocus, Qt.FocusPolicy.TabFocus,
                          Qt.FocusPolicy.WheelFocus):
                found.add(type(widget).__name__)
    return found


#: Base classes whose :focus selectors cover any subclass (Qt style
#: matching walks the class hierarchy, so a DataTableView matches the
#: QTableView rule — the audit asserts BASE coverage, never a
#: subclass-name coincidence).
_BASE_OF = {
    "DataTableView": "QTableView",
    "PandasTableModel": "QTableView",
}


def _rule_name(cls: str) -> str:
    """The QSS class name whose :focus rule covers this widget class."""
    return _BASE_OF.get(cls, cls)


@pytest.mark.parametrize("page_spec", PAGES)
def test_every_focusable_class_has_a_qss_focus_rule(studio_app, page_spec) -> None:
    """No keyboard-focusable widget class on any workspace renders
    without a :focus rule in the studio theme (both variants)."""
    page = _load(page_spec)()
    try:
        classes = _focusable_classes(page)
        assert classes, f"{page_spec}: no focusable widgets found?"
        for theme in ("light", "dark"):
            qss = studio_qss(theme)
            missing = [
                cls for cls in classes
                if f"{_rule_name(cls)}:focus" not in qss
            ]
            assert not missing, (
                f"{page_spec} [{theme}]: no :focus rule for {missing}"
            )
    finally:
        page.close()


def test_the_focus_rule_set_covers_the_core_classes() -> None:
    """The classes the audit walks each have a rule in both themes."""
    core = [
        "QPushButton", "QLineEdit", "QPlainTextEdit", "QComboBox",
        "QSpinBox", "QDoubleSpinBox",
        # R30-b additions
        "QTableWidget", "QListWidget", "QTreeWidget", "QTabBar",
        "QCheckBox", "QRadioButton", "QSlider", "QGroupBox",
    ]
    for theme in ("light", "dark"):
        qss = studio_qss(theme)
        for cls in core:
            assert f"{cls}:focus" in qss, f"{theme}: {cls} lacks :focus"


def test_focus_rules_use_the_theme_focus_color() -> None:
    """Focus indicators draw with the THEME'S FOCUS RING TOKEN —
    visibility is the point.  Light uses the teal focus/ring
    (#076F73, 5.95:1 on white); dark uses the bright teal focus/ring
    (#79D9CF, 9.67:1 on its surfaces) (WCAG 1.4.11)."""
    for theme in ("light", "dark"):
        qss = studio_qss(theme)
        focus_rules = [
            line for line in qss.split("\n")
            if ":focus" in line and "solid" in line
        ]
        assert focus_rules
        expected = "#076F73" if theme == "light" else "#79D9CF"
        for line in focus_rules:
            # every solid-border focus rule paints the theme focus token
            assert expected in line, (theme, line)


def test_selected_tab_underline_uses_the_focus_color() -> None:
    """The selected-tab underline is a state indicator on the surface,
    so it takes the focus/ring token in both themes."""
    for theme in ("light", "dark"):
        qss = studio_qss(theme)
        block = qss.split("QTabBar::tab:selected {{\n")[1] if \
            "QTabBar::tab:selected {{\n" in qss else qss.split("QTabBar::tab:selected {")[1]
        body = block.split("}")[0]
        expected = "#076F73" if theme == "light" else "#79D9CF"
        assert expected in body, (theme, body)


def test_selection_is_not_the_only_cue_in_tables() -> None:
    """The QSS says it explicitly: QTableWidget:focus exists AND the
    selection rule is separate (selection != focus)."""
    qss = studio_qss("light")
    assert "QTableWidget:focus" in qss
    assert "selection-background-color" in qss


def test_no_page_disables_tab_focus_everywhere(studio_app) -> None:
    """A page where NOTHING is keyboard-reachable would trap or
    strand keyboard users; every workspace keeps at least its
    buttons tab-reachable."""
    for page_spec in PAGES:
        page = _load(page_spec)()
        try:
            reachable = [
                w for w in page.findChildren(QPushButton)
                if w.focusPolicy() != Qt.FocusPolicy.NoFocus and w.isEnabled()
            ]
            assert reachable, f"{page_spec}: no keyboard-reachable buttons"
        finally:
            page.close()
