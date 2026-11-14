"""The 'studio' theme provider — part7's visual identity as QSS.

Generates the application stylesheet from :mod:`ui.design_tokens`
(the single source of every color, font and spacing decision), so
the Deep Navy + Warm Ivory + Electric Lime identity lives in ONE
place and the existing provider machinery delivers it: registering
here means theme switching, persistence and the ChartPalette signal
path all keep working unchanged (§6.1: "retain the existing
technology stack"; §8 step 10: "migrate reusable components
incrementally").

The QSS is deliberately CONSERVATIVE: it styles the shared widget
classes and the nine-card objectNames, not every page's custom
layout — pages keep their own structure and inherit the identity.
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

from .design_tokens import (
    LAYOUT,
    SCALE,
    SCIENTIFIC_CATEGORICAL,
    SCIENTIFIC_CATEGORICAL_DARK,
    TYPEFACES,
    ColorTokens,
    tokens,
)

STUDIO_THEMES: tuple[str, ...] = ("light", "dark")


def studio_qss(theme: str) -> str:
    """The full application stylesheet for a studio theme name."""
    t = tokens(theme)
    L = LAYOUT
    f_ui = TYPEFACES.ui_css()
    f_mono = TYPEFACES.mono_css()
    on_accent = t.primary  # dark foreground on lime fills (§2.3)
    canvas_meta = t.text  # canvas metadata promoted to primary (4.34:1 fix)

    return f"""
/* ---- base ---------------------------------------------------------- */
QWidget {{
    background: {t.background};
    color: {t.text};
    font-family: {f_ui};
    font-size: {SCALE.body_pt}pt;
    selection-background-color: {t.selected_surface};
    selection-color: {t.text};
}}
QToolTip {{
    background: {t.surface};
    color: {t.text};
    border: {L.border}px solid {t.border};
    padding: {L.space_xs}px;
    font-size: 9pt;
}}

/* ---- surfaces ------------------------------------------------------ */
QFrame#nineCard {{
    background: {t.surface};
    border: {L.border}px solid {t.border};
    border-radius: {L.radius_md}px;
}}
QLabel {{ background: transparent; }}
QLabel#studioCardTitle {{
    /* review 4.1: dark's t.primary (#203645) is a FILL token (panels,
       status bar) — as title text on t.surface it measures 1.11:1.
       Titles read t.text in dark (9.84:1); light keeps t.primary
       (13.78:1). The WCAG guard test pins both pairs. */
    color: {t.text if theme == 'dark' else t.primary};
    font-size: 15pt;
    font-weight: 600;
}}
QLabel#studioCardDetail, QLabel#nineCardDesc {{
    color: {t.text_secondary};
}}
QLabel#nineCardBadge {{
    background: {t.selected_surface};
    color: {t.text};
    border-radius: {L.radius_sm}px;
    padding: {L.space_xs}px {L.space_sm}px;
    font-weight: 600;
}}
QLabel#nineCardChip {{
    font-family: {f_mono};
    color: {canvas_meta};
    font-size: 8.5pt;
}}

/* ---- buttons: primary vs secondary ---------------------------------- */
QPushButton {{
    background: {t.surface};
    color: {t.text};
    border: {L.border}px solid {t.border};
    border-radius: {L.radius_sm}px;
    padding: {L.space_sm}px {L.space_md}px;
    font-weight: 500;
}}
QPushButton:hover {{
    background: {t.selected_surface};
    border-color: {t.accent_hover};
}}
QPushButton:pressed {{ background: {t.selected_surface}; }}
QPushButton:focus {{ border: {L.border + 1}px solid {t.focus_color()}; }}
QPushButton:disabled {{ color: {t.disabled or t.text_secondary}; border-color: {t.border}; }}

QPushButton#nineCardEntry-primary {{
    background: {t.primary};
    color: {t.accent};
    border: none;
    font-weight: 600;
}}
QPushButton#nineCardEntry-primary:hover {{
    background: {t.accent};
    color: {on_accent};
}}
QPushButton#nineCardEntry-primary:pressed {{ background: {t.accent_hover}; color: {on_accent}; }}

/* ---- inputs --------------------------------------------------------- */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox {{
    background: {t.surface};
    color: {t.text};
    border: {L.border}px solid {t.border};
    border-radius: {L.radius_sm}px;
    padding: {L.space_xs}px {L.space_sm}px;
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus {{
    border: {L.border + 1}px solid {t.focus_color()};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {t.disabled or t.text_secondary};
}}
QPlainTextEdit, QTextEdit, QAbstractScrollArea#mono {{
    font-family: {f_mono};
    font-size: 9pt;
}}

/* ---- combos ---------------------------------------------------------- */
QComboBox {{
    background: {t.surface};
    color: {t.text};
    border: {L.border}px solid {t.border};
    border-radius: {L.radius_sm}px;
    padding: {L.space_xs}px {L.space_sm}px;
}}
QComboBox:hover {{ border-color: {t.accent_hover}; }}
QComboBox:focus {{ border: {L.border + 1}px solid {t.focus_color()}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {t.surface};
    color: {t.text};
    border: {L.border}px solid {t.border};
    selection-background-color: {t.selected_surface};
    selection-color: {t.text};
}}

/* ---- tables ---------------------------------------------------------- */
QHeaderView::section {{
    background: {t.primary};
    color: {t.text if theme == 'dark' else '#EAF0F4'};
    border: none;
    padding: {L.space_xs}px {L.space_sm}px;
    font-weight: 600;
}}
QTableWidget, QTableView {{
    background: {t.surface};
    color: {t.text};
    gridline-color: {t.border};
    border: {L.border}px solid {t.border};
    selection-background-color: {t.selected_surface};
    selection-color: {t.text};
    alternate-background-color: {t.selected_surface if theme == 'dark' else t.background};
}}
/* (R30-b) keyboard focus is visible on data widgets too — selection
   and focus are DIFFERENT states; a selected row in an unfocused
   table must not be the only visual cue.  The accent border marks
   the widget that owns the keyboard. */
QTableWidget:focus, QTableView:focus, QListWidget:focus, QTreeWidget:focus {{
    border: {L.border + 1}px solid {t.focus_color()};
}}

/* ---- tabs ------------------------------------------------------------ */
QTabWidget::pane {{
    border: {L.border}px solid {t.border};
    border-radius: {L.radius_sm}px;
    background: {t.surface};
}}
QTabBar::tab {{
    background: transparent;
    color: {t.text_secondary};
    padding: {L.space_sm}px {L.space_md}px;
    border-bottom: 2px solid transparent;
    font-weight: 500;
}}
QTabBar::tab:hover {{ color: {t.text}; }}
QTabBar::tab:selected {{
    color: {t.text};
    border-bottom: 2px solid {t.focus_color()};
    font-weight: 600;
}}
/* (R30-b) the tab bar owning the keyboard gets the accent underline
   on the focused tab as well (selected ≠ focused: arrow keys move
   focus inside a tab bar even when another tab stays selected). */
QTabBar:focus {{ border-bottom: 2px solid {t.focus_color()}; }}

/* ---- checkboxes, radios, sliders, group boxes (R30-b) ---------------- */
QCheckBox:focus, QRadioButton:focus {{
    border: none;
    color: {t.focus_color()};
    font-weight: 600;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {t.border};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {t.text_secondary};
    width: 12px;
    margin: -5px 0;
    border-radius: 6px;
}}
/* (R30-b) the slider WIDGET rule (the handle pseudo-element above
   cannot carry the widget-level :focus state).  (P-3) the ring now
   hugs the groove (padding 0 + tight margins) and the HANDLE grows
   and takes the focus color, so keyboard focus reads as "this
   control" on the thin slider instead of a box around dead space. */
QSlider:focus {{
    border: {L.border + 1}px solid {t.focus_color()};
    border-radius: {L.radius_sm}px;
    padding: 0px;
}}
QSlider::handle:horizontal:focus {{
    background: {t.focus_color()};
    width: 16px;
    margin: -7px 0;
    border-radius: 8px;
}}
QGroupBox {{
    border: {L.border}px solid {t.border};
    border-radius: {L.radius_sm}px;
    margin-top: {L.space_md}px;
    font-weight: 600;
}}
QGroupBox:focus {{
    border: {L.border + 1}px solid {t.focus_color()};
}}

/* ---- dialogs, progress, status --------------------------------------- */
QDialog {{ background: {t.surface}; }}
QProgressBar {{
    background: {t.surface};
    border: {L.border}px solid {t.border};
    border-radius: {L.radius_sm}px;
    text-align: center;
    color: {t.text};
}}
QProgressBar::chunk {{
    background: {t.accent};
    border-radius: {L.radius_sm}px;
}}
QStatusBar {{ background: {t.primary}; color: {'#EAF0F4'}; }}
/* review P-1: a hairline separates the bar from the canvas in BOTH
   themes (dark nearly merged with Midnight before) */
QStatusBar {{ border-top: 1px solid {t.focus_color()}; }}
QMenuBar {{ background: {t.primary}; color: {'#EAF0F4'}; }}
QMenuBar::item:selected {{ background: {t.selected_surface}; color: {t.text}; }}
QMenu {{
    background: {t.surface};
    color: {t.text};
    border: {L.border}px solid {t.border};
}}
QMenu::item:selected {{ background: {t.selected_surface}; }}

/* ---- scrollbars ------------------------------------------------------ */
QScrollBar:vertical, QScrollBar:horizontal {{
    background: {t.background};
    width: 10px; height: 10px;
}}
QScrollBar::handle {{
    background: {t.border};
    border-radius: 4px;
}}
QScrollBar::handle:hover {{ background: {t.accent_hover}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
"""


def apply_studio(app: QApplication, theme: str) -> None:
    """Clear any previous stylesheet, then apply the studio QSS.

    The same discipline ThemeManager.apply already enforces (16.10.4
    step 1 — one ACTIVE provider, no engine competition): called
    from ThemeManager after it clears, and safe standalone.
    """
    app.setStyleSheet("")
    app.setStyleSheet(studio_qss(theme))


def studio_chart_series(theme: str) -> tuple[str, ...]:
    """The Okabe-Ito cycle for a studio theme (§5)."""
    return (SCIENTIFIC_CATEGORICAL_DARK if theme == "dark"
            else SCIENTIFIC_CATEGORICAL)


__all__ = ["STUDIO_THEMES", "apply_studio", "studio_chart_series",
           "studio_qss"]
