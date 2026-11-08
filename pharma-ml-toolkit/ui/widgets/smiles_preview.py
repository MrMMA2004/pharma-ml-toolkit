"""SMILES-to-molecule interaction for Studio tables (Part-3 §22, round 231).

§22.1: "When the user clicks a SMILES entry or holds the mouse pointer
over it, display the corresponding two-dimensional molecular structure"
-- no copy-pasting into a separate drawing tool.

Three pieces:

* :class:`MoleculeRenderer` -- SMILES → ``QPixmap`` via RDKit's
  ``MolDraw2DCairo`` (the same drawer ``chem.bit_highlights`` uses),
  with a size-keyed FIFO cache (§22.3 "cache generated molecular
  images", "avoid repeatedly redrawing identical molecules") and an
  honest ``(None, reason)`` return for invalid SMILES (§22.4 -- never
  a misleading image, always the original string preserved).
* :class:`MoleculePopup` -- the hover preview: a small always-on-top
  frame that appears after a configurable dwell interval and hides on
  leave / scroll / click.
* :class:`SmilesDetailDialog` -- the click target (§22.5): the large
  structure plus original SMILES, canonical SMILES, molecular formula,
  molecular weight and the SA score -- each either calculated right
  here or shown as an explicit "not calculated" line (never invented).

:func:`attach_smiles_previews` wires the behaviour onto any
:class:`~ui.widgets.data_table.DataTableView` for a set of *column
names* -- the controller resolves cells at hover time, so it keeps
working as the table rebinds frames (the Generative tab's shared
table serves the SA board, the evolution mutants, and the topo/front
children this way).

Nothing renders until a preview is requested (§22.3: "do not perform
unnecessary molecular rendering"): the renderer is lazy and the dwell
timer is the only trigger.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import pandas as pd
from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSize, Qt, QTimer
from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .data_table import DataTableView, PandasTableModel

if TYPE_CHECKING:
    from PySide6.QtGui import QEnterEvent

__all__ = [
    "CACHE_LIMIT",
    "DEFAULT_DWELL_MS",
    "DETAIL_IMAGE_SIZE",
    "HOVER_PREVIEW_SIZE",
    "MoleculePopup",
    "MoleculeRenderer",
    "SmilesDetailDialog",
    "SmilesPreviewController",
    "attach_smiles_previews",
]

#: Hover preview canvas (logical pixels; rendered at 2x for crispness).
HOVER_PREVIEW_SIZE = QSize(260, 200)
#: Detail-dialog structure canvas (logical pixels; rendered at 2x).
DETAIL_IMAGE_SIZE = QSize(460, 340)
#: §22.2 "configurable short interval" default (ms).
DEFAULT_DWELL_MS = 350
#: §22.3 cache bound: distinct (smiles, w, h) renders kept.
CACHE_LIMIT = 128

_INVALID_NOTE = (
    "Invalid SMILES — no structure drawn.\n"
    "The original value is preserved exactly as given."
)


def _cursor_pos() -> QPoint:
    """Cursor position (inline import keeps the module import light)."""
    from PySide6.QtGui import QCursor

    return QCursor.pos()


class MoleculeRenderer:
    """Render SMILES to ``QPixmap`` with a bounded FIFO cache.

    Returns ``(pixmap, None)`` on success and ``(None, reason)``
    for invalid input -- the two failure modes never mix (§22.4).
    """

    def __init__(self, cache_limit: int = CACHE_LIMIT) -> None:
        self._cache: OrderedDict[tuple[str, int, int], QPixmap] = OrderedDict()
        self._cache_limit = cache_limit
        self.render_count = 0
        self.cache_hits = 0

    # -- API ----------------------------------------------------------------
    def render(self, smiles: str, size: QSize) -> tuple[QPixmap | None, str | None]:
        """Return the structure pixmap for ``smiles`` at ``size``.

        ``size`` is the *logical* size; the PNG is drawn at 2x and the
        pixmap carries ``devicePixelRatio=2`` so high-DPI screens show
        crisp bonds (§22.3 "support larger, higher-resolution
        previews").
        """
        key = (smiles, size.width(), size.height())
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached, None
        pixmap, error = self._draw(smiles, size)
        if pixmap is not None:
            self._cache[key] = pixmap
            while len(self._cache) > self._cache_limit:
                self._cache.popitem(last=False)
        return pixmap, error

    # -- internals ------------------------------------------------------------
    def _draw(self, smiles: str, size: QSize) -> tuple[QPixmap | None, str | None]:
        if QGuiApplication.instance() is None:  # pragma: no cover - GUI guard
            return None, "No QApplication — cannot create pixmaps."
        if not isinstance(smiles, str) or not smiles.strip():
            return None, "Empty SMILES cell — nothing to preview."
        try:
            from rdkit import Chem
            from rdkit.Chem.Draw import rdMolDraw2D
        except Exception:  # noqa: BLE001 - pragma: no cover; optional binding, honest refusal
            return None, "RDKit is not installed — structure preview unavailable."

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # §22.4: identify the record, keep the original value, and
            # never draw a made-up structure for a failed parse.
            return None, f"Invalid SMILES — RDKit cannot parse {smiles!r}."
        if mol.GetNumAtoms() == 0:  # pragma: no cover - defensive
            return None, "Empty molecule (0 atoms) — nothing to draw."

        scale = 2
        try:
            drawer = rdMolDraw2D.MolDraw2DCairo(
                size.width() * scale, size.height() * scale
            )
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
            drawer.FinishDrawing()
            png = bytes(drawer.GetDrawingText())  # exact PNG payload
        except Exception as exc:  # noqa: BLE001 - pragma: no cover; drawer failure returns its reason
            return None, f"Structure drawing failed: {exc}"

        pixmap = QPixmap()
        # PySide6 6.8.x: the format must be a keyword arg (positional
        # b"PNG" trips the shiboken signature checker); PNG is also
        # auto-detected from the magic bytes, but the explicit pin
        # documents the contract.
        if not pixmap.loadFromData(png, format=b"PNG"):  # pragma: no cover - corrupt bytes
            return None, "Structure drawing produced an unreadable image."
        pixmap.setDevicePixelRatio(scale)
        self.render_count += 1
        return pixmap, None


class MoleculePopup(QWidget):
    """The small hover preview frame (structure + caption).

    Shown by the controller at the cursor; hides itself when the mouse
    enters it (the user is moving on) or on any explicit hide.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.ToolTip)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        frame = QFrame(self)
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(frame)
        inner = QVBoxLayout(frame)
        inner.setContentsMargins(8, 8, 8, 8)
        self._image = QLabel(frame)
        self._image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(self._image)
        self._caption = QLabel(frame)
        self._caption.setWordWrap(True)
        self._caption.setMaximumWidth(HOVER_PREVIEW_SIZE.width())
        inner.addWidget(self._caption)

    def show_structure(self, pixmap: QPixmap, caption: str) -> None:
        self._image.clear()
        self._image.setPixmap(pixmap)
        self._caption.setText(caption)
        self._show_clamped()

    def show_message(self, message: str) -> None:
        """§22.4: the honest invalid-SMILES notice instead of an image."""
        self._image.clear()
        self._image.setText(_INVALID_NOTE)
        self._caption.setText(message)
        self._show_clamped()

    def _show_clamped(self) -> None:
        self.adjustSize()
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        cursor = _cursor_pos()
        screen = QGuiApplication.primaryScreen()
        available = screen.availableGeometry() if screen else QRect()
        x = min(
            max(cursor.x() + 16, available.left()),
            max(available.left(), available.right() - self.width()),
        )
        y = cursor.y() + 18
        if available.height() and y + self.height() > available.bottom():
            y = max(available.top(), cursor.y() - self.height() - 12)
        self.move(QPoint(x, y))
        self.show()
        self.raise_()

    def enterEvent(self, event: QEnterEvent) -> None:  # pragma: no cover - visual
        self.hide()
        super().enterEvent(event)


class SmilesDetailDialog(QDialog):
    """§22.5 detail view: large structure + honest calculated properties.

    Everything shown is computed in this dialog (formula, weight, SA
    score) or quoted verbatim (the SMILES); there is no invented
    descriptor.  Invalid input renders the validation message with the
    original value preserved.
    """

    def __init__(self, smiles: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Molecule detail")
        self.setModal(False)
        self._smiles = smiles
        self._renderer = _SHARED_RENDERER
        self._rows: dict[str, str] = {}

        layout = QVBoxLayout(self)
        self._image_label = QLabel(self)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(DETAIL_IMAGE_SIZE)
        layout.addWidget(self._image_label)

        self._props = QLabel(self)
        self._props.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._props.setWordWrap(True)
        layout.addWidget(self._props)

        self._copy_btn = QPushButton("Copy canonical SMILES")
        self._copy_btn.clicked.connect(self._copy_canonical)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self._copy_btn)
        layout.addLayout(row)

        self._note = QLabel(self)
        self._note.setWordWrap(True)
        layout.addWidget(self._note)
        self._populate()

    # -- population -------------------------------------------------------
    def _populate(self) -> None:
        pixmap, error = self._renderer.render(self._smiles, DETAIL_IMAGE_SIZE)
        if pixmap is not None:
            self._image_label.setPixmap(pixmap)
        else:
            self._image_label.setText(_INVALID_NOTE)
        rows = self._properties(self._smiles)
        self._rows = rows
        self._props.setText(
            "<br>".join(f"<b>{name}:</b> {value}" for name, value in rows.items())
        )
        if error:
            self._note.setText(error)
        canonical = rows.get("canonical SMILES", "")
        self._copy_btn.setEnabled(bool(canonical))
        if not canonical:
            self._copy_btn.setText("Copy original SMILES")
            self._copy_btn.setEnabled(bool(self._smiles))

    def _properties(self, smiles: str) -> dict[str, str]:
        """Calculate (never invent) the §22.5 property rows."""
        rows: dict[str, str] = {"original SMILES": smiles}
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors
        except Exception:  # noqa: BLE001 - pragma: no cover; venv ships rdkit
            rows["canonical SMILES"] = "not calculated (RDKit unavailable)"
            return rows
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            rows["parse status"] = "invalid — no properties calculated"
            return rows
        rows["canonical SMILES"] = Chem.MolToSmiles(mol)
        rows["molecular formula"] = rdMolDescriptors.CalcMolFormula(mol)
        rows["molecular weight"] = f"{Descriptors.MolWt(mol):.2f} g/mol"  # type: ignore[attr-defined]
        try:
            from pharmaml.chem.sa_scoring import SA_SCORE_METHOD, sa_score

            result = sa_score(smiles)
            rows["synthetic accessibility"] = (
                f"{result.score:.2f} (Ertl SA, [1 easy – 10 hard])"
                if result.valid and result.score is not None
                else "not scored (invalid SMILES)"
            )
            rows["SA method"] = SA_SCORE_METHOD
        except Exception:  # noqa: BLE001 - pragma: no cover; engine failure stays honest
            rows["synthetic accessibility"] = "not calculated"
        return rows

    def _copy_canonical(self) -> None:
        text = self._rows.get("canonical SMILES") or self._smiles
        if text:
            QApplication.clipboard().setText(text)

    # -- test accessors -------------------------------------------------------
    def property_rows(self) -> dict[str, str]:
        return dict(self._rows)

    def image_pixmap(self) -> QPixmap | None:
        return self._image_label.pixmap()


class SmilesPreviewController(QObject):
    """Hover-dwell + click wiring for one ``DataTableView``.

    The event filter lives on the view's viewport; the dwell timer
    restarts when the hovered cell changes and cancels on leave/scroll.
    Columns are matched *by name*, so rebound frames keep the feature
    wherever a configured SMILES column appears.
    """

    def __init__(
        self,
        view: DataTableView,
        columns: tuple[str, ...] = ("smiles", "canonical"),
        dwell_ms: int = DEFAULT_DWELL_MS,
        renderer: MoleculeRenderer | None = None,
    ) -> None:
        super().__init__(view)
        self._view = view
        self._columns = tuple(columns)
        self._renderer = renderer or MoleculeRenderer()
        self._dwell = QTimer(self)
        self._dwell.setSingleShot(True)
        self._dwell.setInterval(int(dwell_ms))
        self._dwell.timeout.connect(self._show_popup)
        self._current: tuple[int, int] | None = None
        self._pending: str | None = None
        self._popup = MoleculePopup()
        view.viewport().installEventFilter(self)
        # hide the preview whenever the table scrolls under the cursor
        view.horizontalScrollBar().valueChanged.connect(self._hide_popup)
        view.verticalScrollBar().valueChanged.connect(self._hide_popup)

    # -- event filter -------------------------------------------------------
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        kind = event.type()
        if kind == QEvent.Type.HoverMove:
            self._hover(_event_pos(event))
        elif kind in (QEvent.Type.HoverLeave, QEvent.Type.Leave):
            self._cancel()
        elif kind == QEvent.Type.Wheel:
            self._hide_popup()
        elif kind == QEvent.Type.MouseButtonRelease:
            self._click(_event_pos(event))
        return super().eventFilter(watched, event)

    # -- internals -------------------------------------------------------
    def _cell_value(self, pos: QPoint) -> tuple[str, str] | None:
        index = self._view.indexAt(pos)
        if not index.isValid():
            return None
        model = self._view.model()
        if not isinstance(model, PandasTableModel):
            return None
        header = model.headerData(index.column(), Qt.Orientation.Horizontal)
        name = str(header) if header is not None else ""
        if name not in self._columns:
            return None
        try:
            value = model.shown.iat[index.row(), index.column()]
        except Exception:  # noqa: BLE001 - pragma: no cover; model raced a rebind
            return None
        if pd.isna(value):
            return None
        return name, str(value)

    def _hover(self, pos: QPoint) -> None:
        index = self._view.indexAt(pos)
        cell = self._cell_value(pos)
        if cell is None:
            self._cancel()
            return
        key = (index.row(), index.column())
        if self._current != key:
            self._current = key
            self._pending = cell[1]
            self._hide_popup()
            self._dwell.start()
        elif not self._popup.isVisible():
            self._dwell.start()

    def _click(self, pos: QPoint) -> None:
        cell = self._cell_value(pos)
        if cell is None:
            self._hide_popup()
            return
        self._hide_popup()
        dialog = SmilesDetailDialog(cell[1], self._view)
        dialog.show()
        dialog.raise_()

    def _show_popup(self) -> None:
        if self._pending is None:
            return
        pixmap, error = self._renderer.render(self._pending, HOVER_PREVIEW_SIZE)
        if pixmap is not None:
            self._popup.show_structure(pixmap, self._pending)
        else:
            self._popup.show_message(error or "preview unavailable")

    def _cancel(self) -> None:
        self._dwell.stop()
        self._current = None
        self._pending = None
        self._hide_popup()

    def _hide_popup(self, *args: object) -> None:
        self._popup.hide()

    # -- test accessors -------------------------------------------------------
    @property
    def renderer(self) -> MoleculeRenderer:
        return self._renderer

    def open_detail(self, smiles: str) -> SmilesDetailDialog:
        """Programmatic §22.5 entry (tests + keyboard paths)."""
        dialog = SmilesDetailDialog(smiles, self._view)
        dialog.show()
        return dialog


def _event_pos(event: QEvent) -> QPoint:
    """Local position from a hover/mouse event (Qt6 API)."""
    position = getattr(event, "position", None)
    if position is not None:
        return position().toPoint()
    return QPoint()  # pragma: no cover - Qt5-shaped events


#: One renderer per process: the §22.3 cache is shared by every table.
_SHARED_RENDERER = MoleculeRenderer()


def attach_smiles_previews(
    view: DataTableView,
    columns: tuple[str, ...] = ("smiles", "canonical"),
    dwell_ms: int = DEFAULT_DWELL_MS,
) -> SmilesPreviewController:
    """Enable §22 hover/click molecule previews on ``view``.

    The controller parents itself to the view, so Qt ownership keeps it
    alive; the returned reference documents intent for the caller.
    """
    return SmilesPreviewController(view, columns=columns, dwell_ms=dwell_ms)
