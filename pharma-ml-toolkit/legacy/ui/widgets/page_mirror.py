"""A parentless pages-stack mirror (part8 §12 compatibility shim).

The Phase C restructure parents the 18 detail pages inside their
workspace tab widgets.  The historical ``main_window.pages`` QStackedWidget
API is pinned by dozens of tests and cross-page callers (``count()``,
``widget(i)``, ``currentWidget()``, ``currentIndex()``), so this mirror
keeps that contract alive WITHOUT owning the widgets (Qt's
single-parent rule): it records the 18 pages in the historical
SECTIONS order and answers index lookups; ``go_to()`` drives
``setCurrentIndex`` on it purely as state.
"""

from __future__ import annotations

from PySide6.QtWidgets import QStackedWidget, QWidget


class PageMirror(QStackedWidget):
    """Answers the pages-stack API over pages parented elsewhere."""

    def __init__(self, pages: list[QWidget], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mirrored: list[QWidget] = list(pages)
        self._current = 0
        self.hide()  # never part of the visible layout

    # -- the pinned contract ---------------------------------------------------
    def count(self) -> int:  # Qt override
        return len(self._mirrored)

    def widget(self, index: int) -> QWidget | None:  # Qt override
        if 0 <= index < len(self._mirrored):
            return self._mirrored[index]
        return None

    def currentWidget(self) -> QWidget | None:  # Qt override
        return self.widget(self._current)

    def currentIndex(self) -> int:  # Qt override
        return self._current

    def setCurrentIndex(self, index: int) -> None:  # Qt override
        if 0 <= index < len(self._mirrored):
            self._current = index

    def setCurrentWidget(self, page: QWidget) -> None:  # Qt override
        for index, candidate in enumerate(self._mirrored):
            if candidate is page:
                self._current = index
                return

    def indexOf(self, page: QWidget) -> int:  # Qt override
        for index, candidate in enumerate(self._mirrored):
            if candidate is page:
                return index
        return -1
