"""Round 232 -- Part-3 §21 slice 1: hover data inspection for every chart.

``ChartInspector`` attaches to a :class:`~ui.widgets.plot_widget.PlotPanel`
and gives every plot drawn through it (lines, scatters, bars/histograms):

* a crosshair cursor (§21.2) that follows the mouse inside the plot,
* nearest-data-point snapping with a highlight marker,
* a readout line with the REAL values (§21.3: exact x/y, series name,
  and any metadata the page attached -- never fabricated numbers),
* cached metadata (§21.6: expensive per-point providers run once per
  point, mouse-move storms only rebuild text when the snapped point
  actually changes),
* a context menu with per-series visibility toggles, a hover-inspection
  on/off switch and reset-to-default-view (§21.2),
* click-to-select that emits ``pointSelected(label, index)`` for pages
  that want a deeper drill-down (§21.5 wiring point).

Distance work is normalised to the visible view range, so the snap
tolerance is resolution-independent (5% of the view diagonal in each
axis), and numpy does the point search in one vectorised pass.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPen

from ..theme_manager import ChartPalette

_SNAP_TOLERANCE = 0.08  # fraction of the visible range per axis (~32 px on a 400 px plot)
_READOUT_EMPTY = "—"


class _SeriesRecord:
    """One plottable series registered for inspection."""

    __slots__ = ("_meta_cache", "bar_width", "item", "kind", "label", "meta", "x", "y")

    def __init__(
        self,
        label: str,
        x: np.ndarray,
        y: np.ndarray,
        item: pg.GraphicsObject,
        kind: str,
        meta: Callable[[int], dict[str, Any]] | None,
        bar_width: float | None = None,
    ) -> None:
        self.label = label
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.item = item
        self.kind = kind
        self.meta = meta
        self.bar_width = bar_width
        self._meta_cache: dict[int, dict[str, Any]] = {}

    def metadata(self, index: int) -> dict[str, Any]:
        """Per-point metadata, computed once and cached (§21.6)."""
        if self.meta is None:
            return {}
        if index not in self._meta_cache:
            try:
                self._meta_cache[index] = dict(self.meta(index))
            except Exception:  # noqa: BLE001 - metadata must never kill hover
                self._meta_cache[index] = {}
        return self._meta_cache[index]


class ChartInspector(QObject):
    """Hover/crosshair/series-toggle layer for one :class:`PlotPanel`."""

    pointSelected = Signal(str, int)  # (series label, data index)

    def __init__(self, plot: pg.PlotWidget, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._plot = plot
        self._palette = ChartPalette()
        self._series: OrderedDict[str, _SeriesRecord] = OrderedDict()
        self._enabled = True
        self._last_snap: tuple[str, int] | None = None
        self._pinned: tuple[str, int] | None = None

        # Crosshair + snap marker are both LAZY decorations: they join the
        # plot only on the first hover and leave it on every clear(), so
        # (a) page tests that read listDataItems()[0] see exactly the real
        # curves (a ScatterPlotItem implements 'plotData' and would pollute
        # that list), and (b) the pkpd pin "no InfiniteLine survives clear()"
        # stays true — plot.clear() empties the plot and nothing is re-added
        # until the mouse returns.
        self._vline = pg.InfiniteLine(angle=90, movable=False)
        self._hline = pg.InfiniteLine(angle=0, movable=False)
        self._marker = pg.ScatterPlotItem(
            size=13,
            symbol="o",
            pen=pg.mkPen(None),
            brush=pg.mkBrush(0, 0, 0, 0),
        )
        for deco in (self._vline, self._hline, self._marker):
            deco.hide()

        # Mouse wiring: pyqtgraph's scene signals (§21.6 mechanisms).
        plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)
        # Context menu replaces the dev-flavoured pyqtgraph default.
        plot.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        plot.customContextMenuRequested.connect(self._show_context_menu)

        # The readout label is owned by PlotPanel (it sits in its layout).
        self._readout: Any = None  # QLabel, injected by PlotPanel

        self.apply_palette(self._palette)

    # -- wiring ---------------------------------------------------------------
    def set_readout(self, label: Any) -> None:
        """Inject the QLabel that renders the hover text."""
        self._readout = label
        self._render_readout(_READOUT_EMPTY)

    def register_series(
        self,
        label: str,
        x: np.ndarray,
        y: np.ndarray,
        item: pg.GraphicsObject,
        *,
        kind: str = "line",
        meta: Callable[[int], dict[str, Any]] | None = None,
        bar_width: float | None = None,
    ) -> None:
        """Track one plotted series (called by PlotPanel's plot methods)."""
        if not label:
            label = f"series {len(self._series) + 1}"
        self._series[label] = _SeriesRecord(label, x, y, item, kind, meta, bar_width)
        self._last_snap = None

    def attach_metadata(self, label: str, meta: Callable[[int], dict[str, Any]]) -> bool:
        """Attach/replace a metadata provider for an already-plotted series."""
        record = self._series.get(label)
        if record is None:
            return False
        record.meta = meta
        record._meta_cache.clear()
        return True

    def reset(self) -> None:
        """Drop all series (PlotPanel.clear passes through here)."""
        self._series.clear()
        self._last_snap = None
        self._pinned = None
        self._hide_extras()

    # -- inspection control ----------------------------------------------------
    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        if not enabled:
            self._hide_extras()
            self._render_readout(_READOUT_EMPTY)

    @property
    def enabled(self) -> bool:
        return self._enabled

    # -- theming ---------------------------------------------------------------
    def apply_palette(self, palette: ChartPalette) -> None:
        self._palette = palette
        hair = QColor(palette.foreground)
        hair.setAlpha(110)
        hair_pen = QPen(hair)
        hair_pen.setStyle(Qt.PenStyle.DotLine)
        for line in (self._vline, self._hline):
            line.setPen(hair_pen)
        self._marker.setPen(pg.mkPen(palette.accent, width=2))

    # -- hover engine ------------------------------------------------------------
    def _ensure_decorations_added(self) -> None:
        """Add any hover decoration not currently in the scene (first use,
        or the first hover after a ``plot.clear()`` removed them)."""
        plot_item = self._plot.getPlotItem()
        for deco in (self._vline, self._hline, self._marker):
            if deco.scene() is None:
                plot_item.addItem(deco, ignoreBounds=True)

    def _on_mouse_moved(self, scene_pos: QPointF) -> None:
        vb = self._plot.getPlotItem().vb
        if not self._enabled or not vb.sceneBoundingRect().contains(scene_pos):
            if self._pinned is None:
                self._hide_extras()
                self._render_readout(_READOUT_EMPTY)
            return
        self._ensure_decorations_added()
        view_pos = vb.mapSceneToView(scene_pos)
        self._vline.setPos(view_pos.x())
        self._hline.setPos(view_pos.y())
        self._vline.show()
        self._hline.show()
        self.inspect_at((view_pos.x(), view_pos.y()))

    def inspect_at(self, view_xy: tuple[float, float]) -> tuple[str, int] | None:
        """Snap to the nearest visible real data point and update the UI.

        Returns the ``(label, index)`` snapped to, or ``None`` when the
        mouse is too far from every point (§21.6: tooltips only on a
        meaningful target).  Public for tests and for programmatic use.
        """
        vx, vy = float(view_xy[0]), float(view_xy[1])
        (x0, x1), (y0, y1) = self._view_range()
        xspan = max(x1 - x0, 1e-12)
        yspan = max(y1 - y0, 1e-12)

        best: tuple[str, int] | None = None
        best_d = float("inf")
        for record in self._series.values():
            if hasattr(record.item, "isVisible") and not record.item.isVisible():
                continue
            if record.x.size == 0:
                continue
            if record.kind == "bars" and record.bar_width:
                # §21.3 bar charts: the whole bar BODY is the hover target
                # (plus a tolerance cap above it), not just the top point.
                half = record.bar_width / 2.0
                dx_abs = np.maximum(np.abs(record.x - vx) - half, 0.0)
                dy_abs = np.maximum(vy - record.y, 0.0)
                dx = dx_abs / (xspan * _SNAP_TOLERANCE)
                dy = dy_abs / (yspan * _SNAP_TOLERANCE)
                d2 = dx * dx + dy * dy
            else:
                # Vectorised normalised-distance search (one pass per series).
                dx = (record.x - vx) / (xspan * _SNAP_TOLERANCE)
                dy = (record.y - vy) / (yspan * _SNAP_TOLERANCE)
                d2 = dx * dx + dy * dy
            i = int(np.argmin(d2))
            if d2[i] < best_d and d2[i] <= 1.0:
                best_d = float(d2[i])
                best = (record.label, i)

        if best is None:
            self._marker.hide()
            if self._pinned is None:
                self._render_readout(_READOUT_EMPTY)
            self._last_snap = None
            return None

        label, index = best
        record = self._series[label]
        self._ensure_decorations_added()
        self._marker.setData([float(record.x[index])], [float(record.y[index])])
        self._marker.show()
        if (label, index) != self._last_snap:  # §21.6 debounced text rebuild
            self._render_readout(self._format(record, index))
            self._last_snap = (label, index)
        return (label, index)

    def _view_range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        try:
            (x0, x1), (y0, y1) = self._plot.getPlotItem().vb.viewRange()
        except ValueError:  # pragma: no cover - empty view before first plot
            return (0.0, 1.0), (0.0, 1.0)
        return (x0, x1), (y0, y1)

    def _format(self, record: _SeriesRecord, index: int) -> str:
        """Readout text: the REAL values plus any cached metadata."""
        x = float(record.x[index])
        y = float(record.y[index])
        parts = [
            f"<b>{record.label}</b>",
            f"x = {x:.6g}",
            f"y = {y:.6g}",
        ]
        for key, value in record.metadata(index).items():
            parts.append(f"{key} = {_fmt_value(value)}")
        if self._pinned == (record.label, index):
            parts.append("pinned (click empty space to unpin)")
        return " · ".join(parts)

    # -- click-to-select (§21.5 wiring point) -----------------------------------
    def _on_mouse_clicked(self, event: Any) -> None:
        if not self._enabled:
            return
        vb = self._plot.getPlotItem().vb
        pos = event.scenePos()
        if not vb.sceneBoundingRect().contains(pos):
            return
        view_pos = vb.mapSceneToView(pos)
        snapped = self.inspect_at((view_pos.x(), view_pos.y()))
        if snapped is not None:
            self._pinned = snapped
            self.pointSelected.emit(snapped[0], snapped[1])
            record = self._series[snapped[0]]
            self._render_readout(self._format(record, snapped[1]))
        else:
            self._pinned = None
            self._render_readout(_READOUT_EMPTY)

    # -- context menu: series toggles + inspection switch + reset view ----------
    def _show_context_menu(self, pos: Any) -> None:
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self._plot)
        if self._series:
            header = menu.addAction("Series")
            header.setEnabled(False)
            for label, record in self._series.items():
                action = menu.addAction(label)
                action.setCheckable(True)
                action.setChecked(
                    not hasattr(record.item, "isVisible") or record.item.isVisible()
                )
                action.setData(label)
            menu.addSeparator()
        inspection = menu.addAction("Hover inspection")
        inspection.setCheckable(True)
        inspection.setChecked(self._enabled)
        menu.addSeparator()
        reset = menu.addAction("Reset view")

        chosen = menu.exec(self._plot.mapToGlobal(pos))
        if chosen is None or not chosen.isCheckable():
            if chosen is reset:
                self._plot.getPlotItem().vb.autoRange()
            return
        if chosen is inspection:
            self.set_enabled(not self._enabled)
            return
        label = chosen.data()
        picked = self._series.get(label)
        if picked is not None:
            visible = not picked.item.isVisible()
            picked.item.setVisible(visible)
            if not visible and self._last_snap and self._last_snap[0] == label:
                self._last_snap = None

    # -- helpers -----------------------------------------------------------------
    def _hide_extras(self) -> None:
        self._vline.hide()
        self._hline.hide()
        self._marker.hide()
        self._last_snap = None

    def _render_readout(self, text: str) -> None:
        if self._readout is not None:
            self._readout.setText(text)


def _fmt_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
