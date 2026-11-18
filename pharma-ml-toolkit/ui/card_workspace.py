"""Generic card workspace (part6 §5, Phase 1).

When a card is opened from the home grid, the Studio shows a real
workspace for it — title, brief explanation, the card's workflow
entries as navigation buttons, and an in-card filter box.  This is the
Phase-1 structural layer: the deep per-card integration (dependency
status, recent activity, results areas) is Phases 2-5; the entries
already route to the genuine pages that own each workflow, so nothing
is a dead end and nothing is duplicated.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pharmaml.studio.capability_registry import capabilities_for_card

from .nine_cards import NineCard

#: Section keys whose workflows the Advanced sidebar and Ctrl+1..9 cover;
#: the workspace itself always shows every card entry.
_WORKSPACE_SECTION = {
    "introduction": "dashboard",
    "datasets": "data",
    "machine-learning": "models",
    "deep-learning": "deep_learning",
    "generative-ai": "generative",
    "molecular-docking": "docking",
    "pkpd": "pkpd",
    "molecular-dynamics-analysis": "molecular_dynamics",
    "visualization": "imagelab",
}


class CardWorkspace(QWidget):
    """One card's workspace: what it is, and how to enter its workflows."""

    open_section = Signal(str)

    def __init__(self, card: NineCard, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.card = card
        self.setObjectName(f"cardWorkspace-{card.card_id}")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        head = QHBoxLayout()
        badge = QLabel(card.badge)
        badge.setObjectName("nineCardBadge")
        title = QLabel(card.title)
        title.setObjectName("studioPageTitle")
        head.addWidget(badge)
        head.addWidget(title, stretch=1)
        lay.addLayout(head)

        desc = QLabel(card.description)
        desc.setObjectName("studioPageSubtitle")
        desc.setWordWrap(True)
        lay.addWidget(desc, alignment=Qt.AlignmentFlag.AlignLeft)

        caps = QLabel(card.capabilities)
        caps.setObjectName("studioSectionTitle")
        caps.setWordWrap(True)
        lay.addWidget(caps, alignment=Qt.AlignmentFlag.AlignLeft)

        # -- the registry's capability list for this card (part6 §6) ------
        # Status is honest: integrated rows are live today; api rows are
        # the Phase-3 targets (real backend + tests, GUI form pending).
        card_caps = capabilities_for_card(card.card_id)
        if card_caps:
            status_words = {
                "integrated": "ready",
                "api": "Phase 3",
                "cli": "Phase 3",
                "blocked": "unavailable",
            }
            listing = " · ".join(
                f"{cap.name} ({status_words[cap.status]})" for cap in card_caps
            )
            cap_list = QLabel(listing)
            cap_list.setObjectName("nineCardLivePreview")
            cap_list.setWordWrap(True)
            lay.addWidget(cap_list)

        # -- workflow entries -----------------------------------------------
        entries_header = QLabel("Workflows in this card")
        entries_header.setObjectName("studioSectionTitle")
        lay.addWidget(entries_header)

        self._entry_buttons: list[QPushButton] = []
        self._entry_sections: list[str] = []
        for entry in card.entries:
            btn = QPushButton(entry.label)
            btn.setObjectName(f"cardEntry-{entry.section}")
            btn.setToolTip(f"Open the {entry.section} page that owns this workflow")
            btn.clicked.connect(
                lambda _checked=False, section=entry.section: self.open_section.emit(section)
            )
            self._entry_buttons.append(btn)
            self._entry_sections.append(entry.section)
            lay.addWidget(btn)

        # -- in-card filter (Phase-1 search: filters the entry labels) ------
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter workflows in this card…")
        self._filter.textChanged.connect(self._apply_filter)
        lay.addWidget(self._filter)

        lay.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll)

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().lower()
        from .main_window import SECTIONS

        titles = dict(SECTIONS)
        for btn, section in zip(self._entry_buttons, self._entry_sections):
            title = titles.get(section, section)
            visible = (
                not needle
                or needle in btn.text().lower()
                or needle in title.lower()
                or needle in self.card.title.lower()
            )
            btn.setVisible(visible)

    def entry_buttons(self) -> list[QPushButton]:
        return list(self._entry_buttons)
