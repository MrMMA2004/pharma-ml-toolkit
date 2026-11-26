"""The nine-card home screen (part6 §2.1, Phase 1).

Promotes the §5 grid out of the Dashboard page into the application's
default landing screen.  The grid widget is the very same
:class:`~ui.nine_cards.NineCardGrid` the Dashboard hosted since
v1.2.169 — same registry, same evidence previews, same routing — the
shell around it is new.  The Dashboard page keeps its live census and
remains reachable as a section (its grid section is now the home).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .nine_cards import build_nine_card_grid


class NineCardHome(QWidget):
    """The default Studio screen: title strip + the promoted §5 grid.

    ``card_activated`` re-emits the grid's signal so MainWindow can open
    the card's workspace (part6 §5) without importing the shell here.
    """

    card_activated = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("nineCardHome")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        title = QLabel("PharmaML Studio")
        title.setObjectName("studioPageTitle")
        subtitle = QLabel(
            "Nine-card home — every workflow enters through a card; "
            "Advanced navigation (top right) re-shows the classic sidebar."
        )
        subtitle.setObjectName("studioPageSubtitle")
        subtitle.setWordWrap(True)
        lay.addWidget(title, alignment=Qt.AlignmentFlag.AlignLeft)
        lay.addWidget(subtitle, alignment=Qt.AlignmentFlag.AlignLeft)

        cards_host, _frames = build_nine_card_grid()
        self._cards_host = cards_host
        cards_host.card_activated.connect(self.card_activated.emit)
        lay.addWidget(cards_host)
        lay.addStretch(1)

        scroll.setWidget(body)
        outer.addWidget(scroll)

    # -- the §11 evidence pass-through (MainWindow wires the provider) -----
    def set_card_context(self, provider) -> None:
        self._cards_host.set_card_context(provider)

    @property
    def cards_host(self):
        return self._cards_host

    def showEvent(self, event) -> None:
        """Every visit re-reads the §11 evidence (same labels, fresh text)."""
        super().showEvent(event)
        self._cards_host.refresh_evidence()
