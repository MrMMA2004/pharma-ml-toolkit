"""The horizontal nine-workspace navigation bar (part8 Phase C, §12.1).

The audit directive replaced the 18-item vertical sidebar as the
dominant default with ONE horizontal bar over the nine primary
workspaces, in the exact commission order:

    Introduction · Datasets · Machine Learning · Deep Learning ·
    Generative AI · Molecular Docking · PK/PD ·
    Molecular Dynamics Analysis · Visualization

The bar is a :class:`QToolBar`-free plain widget: a horizontally
scrolling row of checkable QPushButton segments (one exclusive answer
at any time = the active workspace), Mineral-consistent through the
``studioWorkspaceNav`` objectName QSS, keyboard-accessible via the
main window's Ctrl+1..9 actions, and overflow-safe on narrow windows
(a QScrollArea scrolls instead of reordering).
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

#: The nine workspaces in the §12.1 exact left-to-right order.
#: (key, label) — keys match the nine-card registry ids.
WORKSPACE_ORDER: tuple[tuple[str, str], ...] = (
    ("introduction", "Introduction"),
    ("datasets", "Datasets"),
    ("machine-learning", "Machine Learning"),
    ("deep-learning", "Deep Learning"),
    ("generative-ai", "Generative AI"),
    ("molecular-docking", "Molecular Docking"),
    ("pkpd", "PK/PD"),
    ("molecular-dynamics-analysis", "Molecular Dynamics Analysis"),
    ("visualization", "Visualization"),
)


class WorkspaceNav(QWidget):
    """Horizontal, ordered, keyboard-reachable workspace switcher.

    Signals
    -------
    workspace_selected(str)
        The chosen workspace key (one of :data:`WORKSPACE_ORDER`).
    """

    workspace_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("studioWorkspaceNav")
        # the shell strip paints its own background (the navy Catalyst
        # frame) — the bar must not draw a flat box over it.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # §12.1: "appropriate overflow or scrolling behavior on smaller
        # screens without changing the required section order" — a
        # horizontal scroll area keeps the order and never wraps.
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents
        )

        row = QWidget()
        self._row_lay = QHBoxLayout(row)
        self._row_lay.setContentsMargins(12, 4, 12, 4)
        self._row_lay.setSpacing(4)
        self._scroll.setWidget(row)
        outer.addWidget(self._scroll)

        self._buttons: dict[str, QPushButton] = {}
        for key, label in WORKSPACE_ORDER:
            btn = QPushButton(label)
            btn.setObjectName(f"workspaceNav-{key}")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            # the full label is the accessible name even when elided
            # visually by a narrow window (the tooltip carries it too)
            btn.setAccessibleName(label)
            btn.setToolTip(f"Switch to the {label} workspace")
            btn.clicked.connect(lambda _checked=False, k=key: self._pick(k))
            self._row_lay.addWidget(btn)
            self._buttons[key] = btn
        # the trailing stretch keeps the segments left-packed (order
        # reads left-to-right; empty space trails on wide windows)
        self._row_lay.addStretch(1)

    # -- API ------------------------------------------------------------------
    def _pick(self, key: str) -> None:
        self.select(key)
        self.workspace_selected.emit(key)

    def select(self, key: str) -> None:
        """Mark ``key`` active WITHOUT emitting (programmatic routing)."""
        for other, btn in self._buttons.items():
            btn.setChecked(other == key)

    def current(self) -> str | None:
        """The active workspace key (or None before the first pick)."""
        for key, btn in self._buttons.items():
            if btn.isChecked():
                return key
        return None

    def keys(self) -> list[str]:
        """The nine keys in the §12.1 exact order."""
        return [key for key, _label in WORKSPACE_ORDER]

    def button(self, key: str) -> QPushButton | None:
        return self._buttons.get(key)

    def labels(self) -> dict[str, str]:
        return dict(WORKSPACE_ORDER)

    # -- keyboard §12.1 ---------------------------------------------------------
    def keyPressEvent(self, event) -> None:  # Qt signature
        """Left/Right move the active segment; Enter/Space activates it."""
        keys = self.keys()
        idx = keys.index(self.current()) if self.current() in keys else 0
        if event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            step = -1 if event.key() == Qt.Key.Key_Left else 1
            nxt = keys[(idx + step) % len(keys)]
            self._pick(nxt)
            self._buttons[nxt].setFocus()
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return):
            current = self.current()
            if current is not None:
                self.workspace_selected.emit(current)
            event.accept()
            return
        super().keyPressEvent(event)

    def sizeHint(self) -> QSize:  # Qt signature
        hint = super().sizeHint()
        # a single compact row; the scroll area handles the rest
        return QSize(max(hint.width(), 640), 44)
