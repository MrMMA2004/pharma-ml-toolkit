"""Model/view data tables for PharmaML Studio (directive 16.12).

``PandasTableModel`` adapts a pandas DataFrame to QAbstractTableModel
(sortable, resizable, copyable); ``DataTableView`` renders it with a
header context menu (copy selection / all), honest empty state, and
an incremental-loading guard: tables past ``SOFT_ROW_LIMIT`` rows show
the first slice only with a status note, keeping the GUI thread free
(the directive's "do not load unnecessarily large datasets directly").
"""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, QPersistentModelIndex, Qt
from PySide6.QtGui import QGuiApplication, QPaintEvent
from PySide6.QtWidgets import QTableView, QWidget

SOFT_ROW_LIMIT = 5_000


class PandasTableModel(QAbstractTableModel):
    """A read-only Qt table model over a pandas DataFrame."""

    def __init__(self, frame: pd.DataFrame, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._frame = frame
        self._shown = frame.head(SOFT_ROW_LIMIT)
        self.truncated = len(frame) > SOFT_ROW_LIMIT

    # -- Qt model API -----------------------------------------------------
    def rowCount(self, parent: QModelIndex | QPersistentModelIndex | None = None) -> int:
        return 0 if parent and parent.isValid() else len(self._shown)

    def columnCount(self, parent: QModelIndex | QPersistentModelIndex | None = None) -> int:
        return 0 if parent and parent.isValid() else len(self._shown.columns)

    def data(self, index: QModelIndex | QPersistentModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> object:
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            value = self._shown.iat[index.row(), index.column()]
            if pd.isna(value):
                return "—"
            text = str(value)
            return text if len(text) <= 64 else text[:61] + "…"
        if role == Qt.ItemDataRole.TextAlignmentRole:
            import numpy as np

            value = self._shown.iat[index.row(), index.column()]
            numeric = isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value)
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter) if numeric else int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole) -> object:
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._shown.columns[section])
        return str(section)

    def sort(self, column: int, order: Qt.SortOrder = Qt.SortOrder.AscendingOrder) -> None:
        """Sort the underlying frame (stable, NaN-safe)."""
        self.layoutAboutToBeChanged.emit()
        name = self._shown.columns[column]
        self._shown = self._shown.sort_values(name, ascending=bool(order == Qt.SortOrder.AscendingOrder), kind="stable", na_position="last").reset_index(drop=True)
        self.layoutChanged.emit()

    # -- conveniences ------------------------------------------------------
    @property
    def frame(self) -> pd.DataFrame:
        return self._frame

    @property
    def shown(self) -> pd.DataFrame:
        return self._shown

    def column_series(self, column: int) -> pd.Series:
        return self._frame[self._frame.columns[column]]

    def to_clipboard_text(self, rows: slice | None = None) -> str:
        """TSV of the shown slice (or a sub-range) for clipboard export."""
        block = self._shown if rows is None else self._shown.iloc[rows]
        return block.to_csv(sep="\t", index=False)


class DataTableView(QTableView):
    """QTableView with copy support and an honest empty state."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self.setSortingEnabled(True)
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self._empty_note = ""
        self.statusHint = ""

    def bind_frame(self, frame: pd.DataFrame | None, note: str = "") -> None:
        """Swap the displayed frame; ``None``/empty shows the empty state."""
        self._empty_note = note
        if frame is None or frame.empty:
            self.setModel(PandasTableModel(pd.DataFrame()))
            self.statusHint = note or "No data loaded"
        else:
            model = PandasTableModel(frame)
            self.setModel(model)
            self.resizeColumnsToContents()
            if model.truncated:
                self.statusHint = f"showing first {SOFT_ROW_LIMIT:,} of {len(frame):,} rows"
            else:
                self.statusHint = f"{len(frame):,} rows × {len(frame.columns)} columns"
        self.viewport().update()

    def copy_selection(self) -> str:
        """Copy the selected range (or everything) as TSV; returns it."""
        model = self.model()
        if not isinstance(model, PandasTableModel):
            return ""
        selection = self.selectionModel().selectedIndexes() if self.selectionModel() else []
        if not selection:
            text = model.to_clipboard_text()
        else:
            rows = sorted({i.row() for i in selection})
            cols = sorted({i.column() for i in selection})
            block = model.shown.iloc[rows, cols]
            text = block.to_csv(sep="\t", index=False)
        QGuiApplication.clipboard().setText(text)
        return text

    # -- empty state (16.12: "clear empty-state messages") -------------------
    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        if self._empty_note and (self.model() is None or self.model().rowCount() == 0):
            from PySide6.QtGui import QColor, QFont, QPainter

            painter = QPainter(self.viewport())
            painter.setPen(QColor(128, 128, 128))
            font = QFont(self.font())
            font.setItalic(True)
            painter.setFont(font)
            painter.drawText(self.viewport().rect(), int(Qt.AlignmentFlag.AlignCenter), self._empty_note)
            painter.end()
