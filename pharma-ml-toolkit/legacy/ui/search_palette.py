"""The command-palette dialog (part6 §7, Phase 4) over :mod:`ui.search_index`.

``Ctrl+K`` opens this from anywhere in the Studio: type a query, the
ranked results show the capability, its parent card and destination
(§7), and Enter navigates via the main window's public ``go_to``.
Dependent-module honesty rides through from the index rows: a
capability whose optional dependency is missing stays listed with a
"needs <dep>" tag.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .search_index import SearchRow, build_search_index, search


class SearchPalette(QDialog):
    """A modal command palette; ``navigate`` carries the chosen route out."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Search capabilities (Ctrl+K)")
        self.setModal(True)
        self.resize(640, 420)
        self._index = build_search_index()
        self._rows: tuple[SearchRow, ...] = ()
        self._route: str | None = None

        lay = QVBoxLayout(self)
        self._box = QLineEdit()
        self._box.setPlaceholderText(
            "Search capabilities, cards and sections — e.g. \"morgan "
            "fingerprint\", \"conformal prediction\", \"rmsd\"…"
        )
        self._box.textChanged.connect(self._refresh)
        self._box.returnPressed.connect(self._accept_current)
        lay.addWidget(self._box)

        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(lambda _item: self._accept_current())
        lay.addWidget(self._list, stretch=1)

        hint = (
            "↑/↓ move · Enter opens · Esc closes.  A \"needs <dep>\" tag "
            "means the capability's optional dependency is not installed "
            "in this environment — the workflow itself exists."
        )
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        self._hint = hint
        self.setStatusTip(hint)

        self._refresh("")
        self._box.setFocus()

    # -- behaviour ------------------------------------------------------------
    def _refresh(self, text: str) -> None:
        self._rows = search(self._index, text)
        self._list.clear()
        for row in self._rows:
            item = QListWidgetItem(f"{row.status_line()}\n    {row.detail}")
            item.setData(Qt.ItemDataRole.UserRole, row.route)
            self._list.addItem(item)
        if self._list.count():
            self._list.setCurrentRow(0)

    def _accept_current(self) -> None:
        current = self._list.currentItem()
        if current is None:
            return
        self._route = current.data(Qt.ItemDataRole.UserRole)
        self.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Down:
            self._list.setFocus()
            self._list.setCurrentRow(min(self._list.currentRow() + 1,
                                         self._list.count() - 1))
            return
        if event.key() == Qt.Key.Key_Up:
            self._list.setFocus()
            self._list.setCurrentRow(max(self._list.currentRow() - 1, 0))
            return
        super().keyPressEvent(event)

    # -- result ------------------------------------------------------------------
    @property
    def chosen_route(self) -> str | None:
        """The section key the user picked (None = dismissed)."""
        return self._route
