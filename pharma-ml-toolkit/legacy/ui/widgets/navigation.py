"""Sidebar navigation for PharmaML Studio (directive 16.7).

Every entry corresponds to a real application capability -- the same
functional sections the Tkinter playground exposes as notebook tabs.
Keyboard: Up/Down move the selection, Enter activates, and the main
window binds Ctrl+1..9 for direct jumps.
"""

from __future__ import annotations

from typing import ClassVar

from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QKeyEvent, QPainter, QPixmap, QPolygonF
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QWidget


def _glyph(color: str, kind: str) -> QIcon:
    """Draw a tiny flat icon so the sidebar ships no asset files."""
    icon = QIcon()
    for size in (16, 24):
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QColor(color)
        p.setPen(c)
        p.setBrush(c)
        m = max(2, size // 8)
        if kind == "grid":
            step = (size - 2 * m) // 2
            p.drawRect(m, m, step, step)
            p.drawRect(m + step + 2, m, step, step)
            p.drawRect(m, m + step + 2, step, step)
            p.drawRect(m + step + 2, m + step + 2, step, step)
        elif kind == "table":
            p.drawRect(m, m, size - 2 * m, size - 2 * m)
            p.drawLine(m, size // 3, size - m, size // 3)
            p.drawLine(size // 2, size // 3, size // 2, size - m)
        elif kind == "chart":
            poly = QPolygonF()
            base = size - m
            poly.append(QPointF(m, base))
            poly.append(QPointF(size // 3, m * 2))
            poly.append(QPointF(size // 2, base - m))
            poly.append(QPointF(size - m, m))
            p.drawPolyline(poly)
        elif kind == "gear":
            p.drawEllipse(size // 4, size // 4, size // 2, size // 2)
            p.setBrush(Qt.GlobalColor.transparent)
            p.drawEllipse(size // 3, size // 3, size // 3, size // 3)
        p.end()
        icon.addPixmap(pm)
    return icon


class Navigation(QListWidget):
    """The app's sidebar: one row per real section, activation navigates."""

    section_selected = Signal(str)

    GLYPHS: ClassVar[dict[str, str]] = {
        "dashboard": "grid",
        "data": "table",
        "models": "chart",
        "settings": "gear",
    }
    # The Mineral action/primary teal — the same value as the design
    # token accent; kept inline because the sidebar icons are painted,
    # not styled (QSS never reaches inside a QIcon).  The stray-hex
    # audit test pins it against the token so it cannot drift.
    # ``set_theme_accent`` re-paints the glyphs when the studio theme
    # changes (Spectral indigo / Catalyst navy or lime), because a
    # painted QIcon cannot be recolored by a stylesheet.
    ACCENT = "#076F73"

    def __init__(self, sections: list[tuple[str, str]], parent: QWidget | None = None) -> None:
        """``sections`` is a list of ``(key, title)`` pairs in display order."""
        super().__init__(parent)
        self.setMinimumWidth(190)
        self.setMaximumWidth(230)
        self.setObjectName("studioNav")
        self.setIconSize(QSize(18, 18))
        self._sections = list(sections)
        for key, title in self._sections:
            item = QListWidgetItem(_glyph(self.ACCENT, self.GLYPHS.get(key, "grid")), title)
            item.setData(Qt.ItemDataRole.UserRole, key)
            self.addItem(item)
        self.currentRowChanged.connect(self._on_row)
        self.setStyleSheet("QListWidget#studioNav { border: none; background: transparent; }")

    def set_theme_accent(self, accent: str) -> None:
        """Re-paint the sidebar glyphs in a new theme's action color."""
        if accent == self.ACCENT:
            return
        self.ACCENT = accent
        for i, (key, _title) in enumerate(self._sections):
            item = self.item(i)
            if item is not None:
                item.setIcon(_glyph(accent, self.GLYPHS.get(key, "grid")))

    def _on_row(self, row: int) -> None:
        item = self.item(row) if row >= 0 else None
        if item is not None:
            self.section_selected.emit(str(item.data(Qt.ItemDataRole.UserRole)))

    def select(self, key: str) -> None:
        """Programmatically select a section (no duplicate emission)."""
        for i in range(self.count()):
            if str(self.item(i).data(Qt.ItemDataRole.UserRole)) == key:
                if self.currentRow() != i:
                    self.setCurrentRow(i)
                return

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            item = self.currentItem()
            if item is not None:
                self.section_selected.emit(str(item.data(Qt.ItemDataRole.UserRole)))
            return
        if key == Qt.Key.Key_Down:
            self.setCurrentRow(min(self.currentRow() + 1, self.count() - 1))
            return
        if key == Qt.Key.Key_Up:
            self.setCurrentRow(max(self.currentRow() - 1, 0))
            return
        super().keyPressEvent(event)
