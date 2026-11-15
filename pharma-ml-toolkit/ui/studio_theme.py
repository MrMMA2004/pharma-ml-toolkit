"""The 'studio' theme provider — the Mineral identity as QSS.

Generates the application stylesheet from :mod:`ui.design_tokens`
(the single source of every color, font and spacing decision), so the
Mineral identity ("A clearer path from molecules to evidence") lives
in ONE place and the existing provider machinery delivers it:
registering here means theme switching, persistence and the
ChartPalette signal path all keep working unchanged.

The studio provider offers Mineral's light/dark (the canonical §2
identity) PLUS the Design System's two alternative directions as
switchable themes: 02/SPECTRAL (warm paper + indigo, rounded panels)
and 03/CATALYST (navy frame + ivory surfaces + lime selection, squared
cards), each in light and dark variants.  §2 pinned Mineral as the
default; the alternates are opt-in via the Settings page.

The QSS is deliberately CONSERVATIVE: it styles the shared widget
classes and the studio objectNames, not every page's custom layout —
pages keep their own structure and inherit the identity.  Flat,
restrained surfaces: 1 px separators, no gradients, elevation reserved
for floating dialogs.  Catalyst's direction-specific geometry (squared
cards) and Spectral's (rounder panels) are delivered through the
per-theme radius ramp below — still one QSS, token-driven.
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication

from .design_tokens import (
    LAYOUT,
    SCALE,
    SCIENTIFIC_CATEGORICAL,
    SCIENTIFIC_CATEGORICAL_DARK,
    TYPEFACES,
    is_dark_theme,
    tokens,
)

STUDIO_THEMES: tuple[str, ...] = (
    "light", "dark",                       # 01 / MINERAL (canonical)
    "spectral", "spectral-dark",           # 02 / SPECTRAL (alternative)
    "catalyst", "catalyst-dark",           # 03 / CATALYST (alternative)
)


def _theme_radius(theme: str) -> tuple[int, int, int, int]:
    """Per-direction corner radii (sm, md, lg, xl), token-driven.

    Mineral ships the spec §6 scale verbatim (4/6/8/12).  Spectral's
    preview card reads "rounded panels" — one step rounder.  Catalyst's
    reads "squared cards" — one step squarer, squared cards outright.
    """
    if theme.startswith("spectral"):
        return (6, 8, 10, 14)
    if theme.startswith("catalyst"):
        return (2, 4, 6, 8)
    return (LAYOUT.radius_sm, LAYOUT.radius_md, LAYOUT.radius_lg,
            LAYOUT.radius_xl)


def studio_qss(theme: str) -> str:
    """The full application stylesheet for a studio theme name."""
    t = tokens(theme)
    L = LAYOUT
    r_sm, r_md, r_lg, r_xl = _theme_radius(theme)
    f_ui = TYPEFACES.ui_css()
    f_mono = TYPEFACES.mono_css()
    on_primary = t.on_primary  # white on teal (light), ink on bright teal (dark)
    shell = t.shell_fill()
    shell_text = t.shell_foreground()

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
    background: {t.elevated};
    color: {t.text};
    border: {L.border}px solid {t.border};
    padding: {L.space_xs}px;
    font-size: {SCALE.metadata_pt}pt;
}}

/* ---- surfaces ------------------------------------------------------ */
QFrame#nineCard {{
    background: {t.surface};
    border: {L.border}px solid {t.border_subtle};
    border-radius: {r_lg}px;
}}
/* ---- app shell (Catalyst's navy frame; other themes: the canvas) ---- */
QWidget#studioTopBar {{
    background: {shell};
    color: {shell_text};
    border-bottom: 1px solid {t.border};
}}
QLabel#studioTopBarTitle, QLabel#studioTopBarLocation,
QLabel#studioTopBarJobs {{
    background: transparent;
    color: {shell_text};
}}
QListWidget#studioNav {{
    background: {shell};
    color: {shell_text};
    border: none;
}}
QListWidget#studioNav::item {{ color: {shell_text}; }}
QListWidget#studioNav::item:selected {{
    background: {t.selected_surface};
    color: {t.shell_text or t.text};
}}
QLabel {{ background: transparent; }}
QLabel#studioCardTitle {{
    color: {t.text};
    font-size: {SCALE.card_title_pt}pt;
    font-weight: 600;
}}
/* ---- (part8 §12) the horizontal nine-workspace navigation ---------- */
QWidget#studioWorkspaceNav {{
    background: {shell};
    border-bottom: 1px solid {t.border};
}}
QWidget#studioWorkspaceNav QPushButton {{
    background: transparent;
    color: {t.shell_text or t.text_secondary};
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    padding: {L.space_sm}px {L.space_md}px;
    font-weight: 500;
}}
QWidget#studioWorkspaceNav QPushButton:hover {{
    color: {t.shell_text or t.text};
    background: {t.selected_surface};
}}
QWidget#studioWorkspaceNav QPushButton:checked {{
    color: {t.shell_text or t.text};
    border-bottom: 2px solid {t.focus_color()};
    font-weight: 600;
}}
QLabel#studioSectionHeading {{
    color: {t.text};
    font-size: {SCALE.card_title_pt}pt;
    font-weight: 600;
}}
QFrame#introductionWorkspaceRow {{
    background: {t.surface};
    border: {L.border}px solid {t.border_subtle};
    border-radius: {r_sm}px;
}}
QLabel#studioCardDetail, QLabel#nineCardDesc {{
    color: {t.text_secondary};
}}
QLabel#nineCardBadge {{
    background: {t.selected_surface};
    color: {t.text};
    border-radius: {r_sm}px;
    padding: {L.space_xs}px {L.space_sm}px;
    font-weight: 600;
}}
QLabel#nineCardChip {{
    font-family: {f_mono};
    color: {t.text_secondary};
    font-size: {SCALE.metadata_pt}pt;
}}

/* ---- empty states (tokenized: re-color with every theme switch) ----- */
QWidget#studioEmptyState {{ background: transparent; }}
QLabel#studioEmptyGlyph {{
    color: {t.text_muted};
    background: transparent;
    font-size: 34px;
}}
QLabel#studioEmptyHead {{
    color: {t.text};
    background: transparent;
    font-size: {SCALE.card_title_pt}pt;
    font-weight: 600;
}}
QLabel#studioEmptyGuide {{
    color: {t.text_secondary};
    background: transparent;
}}

/* ---- buttons: primary vs secondary ---------------------------------- */
QPushButton {{
    background: {t.surface};
    color: {t.text};
    border: {L.border}px solid {t.border};
    border-radius: {r_md}px;
    padding: {L.space_sm}px {L.space_md}px;
    font-weight: 500;
}}
QPushButton:hover {{
    background: {t.subtle};
    border-color: {t.accent_hover};
}}
QPushButton:pressed {{ background: {t.subtle}; border-color: {t.accent_pressed}; }}
QPushButton:focus {{ border: {L.border_emph}px solid {t.focus_color()}; }}
QPushButton:disabled {{ color: {t.text_muted}; border-color: {t.border_subtle}; background: {t.disabled}; }}

/* the primary action button: the Mineral teal fill (semantic action
   color — the ONLY place the identity teal appears as a fill) */
QPushButton#nineCardEntry-primary {{
    background: {t.primary};
    color: {on_primary};
    border: none;
    font-weight: 600;
}}
QPushButton#nineCardEntry-primary:hover {{
    background: {t.accent_hover};
    color: {on_primary};
}}
QPushButton#nineCardEntry-primary:pressed {{ background: {t.accent_pressed}; color: {on_primary}; }}
QPushButton#nineCardEntry-primary:focus {{ border: {L.border_emph}px solid {t.focus_color()}; }}

/* ---- inputs --------------------------------------------------------- */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox {{
    background: {t.surface};
    color: {t.text};
    border: {L.border}px solid {t.border};
    border-radius: {r_sm}px;
    padding: {L.space_xs}px {L.space_sm}px;
    selection-background-color: {t.selected_surface};
    selection-color: {t.text};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus {{
    border: {L.border_emph}px solid {t.focus_color()};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    color: {t.text_muted};
    background: {t.disabled};
}}
QPlainTextEdit, QTextEdit, QAbstractScrollArea#mono {{
    font-family: {f_mono};
    font-size: {SCALE.code_pt}pt;
}}

/* ---- combos ---------------------------------------------------------- */
QComboBox {{
    background: {t.surface};
    color: {t.text};
    border: {L.border}px solid {t.border};
    border-radius: {r_sm}px;
    padding: {L.space_xs}px {L.space_sm}px;
}}
QComboBox:hover {{ border-color: {t.accent_hover}; }}
QComboBox:focus {{ border: {L.border_emph}px solid {t.focus_color()}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {t.elevated};
    color: {t.text};
    border: {L.border}px solid {t.border};
    selection-background-color: {t.selected_surface};
    selection-color: {t.text};
}}

/* ---- tables ---------------------------------------------------------- */
QHeaderView::section {{
    background: {t.subtle};
    color: {t.text_secondary};
    border: none;
    border-bottom: {L.border}px solid {t.border};
    padding: {L.space_xs}px {L.space_sm}px;
    font-weight: 600;
}}
QTableWidget, QTableView {{
    background: {t.surface};
    color: {t.text};
    gridline-color: {t.border_subtle};
    border: {L.border}px solid {t.border};
    selection-background-color: {t.selected_surface};
    selection-color: {t.text};
    alternate-background-color: {t.subtle};
}}
/* (R30-b, carried into Mineral) keyboard focus is visible on data
   widgets too — selection and focus are DIFFERENT states; a selected
   row in an unfocused table must not be the only visual cue.  The
   focus ring marks the widget that owns the keyboard. */
QTableWidget:focus, QTableView:focus, QListWidget:focus, QTreeWidget:focus {{
    border: {L.border_emph}px solid {t.focus_color()};
}}

/* ---- tabs ------------------------------------------------------------ */
QTabWidget::pane {{
    border: {L.border}px solid {t.border_subtle};
    border-radius: {r_sm}px;
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
QTabBar:focus {{ border-bottom: 2px solid {t.focus_color()}; }}

/* ---- checkboxes, radios, sliders, group boxes ------------------------ */
QCheckBox:focus, QRadioButton:focus {{
    border: none;
    color: {t.focus_color()};
    font-weight: 600;
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px; height: 16px;
    border: {L.border}px solid {t.border};
    border-radius: 3px;
    background: {t.surface};
}}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background: {t.primary};
    border-color: {t.primary};
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
QSlider:focus {{
    border: {L.border_emph}px solid {t.focus_color()};
    border-radius: {r_sm}px;
    padding: 0px;
}}
QSlider::handle:horizontal:focus {{
    background: {t.focus_color()};
    width: 16px;
    margin: -7px 0;
    border-radius: 8px;
}}
QGroupBox {{
    border: {L.border}px solid {t.border_subtle};
    border-radius: {r_sm}px;
    margin-top: {L.space_md}px;
    font-weight: 600;
}}
QGroupBox:focus {{
    border: {L.border_emph}px solid {t.focus_color()};
}}

/* ---- dialogs, progress, status --------------------------------------- */
QDialog {{ background: {t.surface}; }}
QProgressBar {{
    background: {t.surface};
    border: {L.border}px solid {t.border};
    border-radius: {r_sm}px;
    text-align: center;
    color: {t.text};
}}
QProgressBar::chunk {{
    background: {t.primary};
    border-radius: {r_sm}px;
}}
/* the shell strip: bg/subtle surface with a hairline separator and
   secondary text — flat Mineral chrome, not a colored bar (Catalyst:
   the navy frame strip with ivory text) */
QStatusBar {{
    background: {shell};
    color: {shell_text};
    border-top: 1px solid {t.border};
}}
QMenuBar {{ background: {shell}; color: {shell_text}; }}
QMenuBar::item:selected {{ background: {t.selected_surface}; color: {t.text}; }}
QMenu {{
    background: {t.elevated};
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

    The same discipline ThemeManager.apply already enforces (one ACTIVE
    provider, no engine competition): called from ThemeManager after
    it clears, and safe standalone.
    """
    app.setStyleSheet("")
    app.setStyleSheet(studio_qss(theme))


def studio_chart_series(theme: str) -> tuple[str, ...]:
    """The scientific data cycle for a studio theme (§4.2 — shared by
    all three directions: "One scientific foundation")."""
    return (SCIENTIFIC_CATEGORICAL_DARK if is_dark_theme(theme)
            else SCIENTIFIC_CATEGORICAL)


__all__ = ["STUDIO_THEMES", "apply_studio", "studio_chart_series",
           "studio_qss"]
