"""Shared empty-state placeholder (review C-1/P-7 fix).

First-run pages measured 90–99% background canvas: bare forms and
white plot panels with one hint line at best.  This widget gives
every sparse surface the same honest pattern — a large glyph, a
one-line headline, a concrete next step, and (when useful) an action
button — styled from the design tokens so it reads as part of the
identity, not as an error.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class EmptyState(QWidget):
    """Glyph + headline + guidance (+ optional action) for empty panels.

    Colors come from the studio QSS via objectName rules
    (studioEmptyState / studioEmptyGlyph / studioEmptyHead) — no inline
    hex, so the placeholder re-colors with every theme switch instead
    of keeping its construction-time palette.
    """

    def __init__(
        self,
        headline: str,
        guidance: str = "",
        glyph: str = "◎",
        action_text: str = "",
        on_action=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("studioEmptyState")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(8)
        lay.addStretch(1)

        self._glyph = QLabel(glyph)
        self._glyph.setObjectName("studioEmptyGlyph")
        self._glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._glyph)

        self._head = QLabel(headline)
        self._head.setObjectName("studioEmptyHead")
        self._head.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._head.setWordWrap(True)
        lay.addWidget(self._head)

        if guidance:
            self._guide = QLabel(guidance)
            self._guide.setObjectName("studioEmptyGuide")
            self._guide.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._guide.setWordWrap(True)
            lay.addWidget(self._guide)

        if action_text and on_action is not None:
            self._action = QPushButton(action_text)
            self._action.clicked.connect(on_action)
            wrap = QVBoxLayout()
            wrap.setAlignment(Qt.AlignmentFlag.AlignCenter)
            wrap.addWidget(self._action)
            lay.addLayout(wrap)

        lay.addStretch(1)
