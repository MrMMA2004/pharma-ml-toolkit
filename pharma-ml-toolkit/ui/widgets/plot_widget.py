"""Theme-aware PyQtGraph plot panel (directive 16.8).

``PlotPanel`` wraps ``pg.PlotWidget`` and re-styles itself whenever the
theme manager emits a new :class:`~pharmaml.ui.theme_manager.ChartPalette`
-- dark themes get dark backgrounds, readable axis labels and visible
grids (16.11: a Qt stylesheet never reaches pyqtgraph's own rendering).
Scientific data goes in as numpy arrays with real labels/units.

Round 232 (Part-3 §21 slice 1): every panel ships with a
:class:`~ui.widgets.chart_inspector.ChartInspector` -- crosshair
cursor, nearest-point snapping, a real-value hover readout line under
the plot, click-to-select, per-series visibility toggles and a
reset-view entry in the context menu.  Pages opt out per chart with
``set_inspection_enabled(False)``; per-point hover metadata is attached
through the ``meta=`` callable or ``attach_metadata()``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..theme_manager import ChartPalette
from .chart_inspector import ChartInspector
from .image_export import export_widget_grab

# §23.6 (round 238) adaptive tick-label density: a category axis renders
# EVERY tick it is handed, so 40 feature names on a narrow panel collide
# into an unreadable smear.  The policy below thins the RENDERED subset
# (the §21.3 hover meta keeps the full list) and ellipsizes what remains.
_TICK_PAD_PX = 16   # breathing room around a label when sizing a tick
_MIN_AVAIL_PX = 80  # narrower than this = not laid out yet: keep all ticks
_MIN_SHARE_PX = 24  # a kept label never gets less than this much room


class PlotPanel(QWidget):
    """A single themed plot with labelled axes and a callable API."""

    #: §21.5 (round 236): forwarded from the inspector's click-to-select
    #: so pages can open drill-down views (the molecule detail dialog)
    #: without reaching into the inspector -- ``(series label, data
    #: index)`` of the clicked point.
    point_selected = Signal(str, int)

    def __init__(self, parent: QWidget | None = None, title: str = "") -> None:
        super().__init__(parent)
        self._palette = ChartPalette()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._plot = pg.PlotWidget()
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        if title:
            self._plot.setTitle(title)
        layout.addWidget(self._plot)
        # §21 hover readout: a floating overlay on the plot's bottom edge,
        # NOT a layout row -- the plot keeps the panel's full geometry (the
        # §17 export pins grab exactly the panel size), the readout costs
        # no vertical space (§23: no layout jump when it fills/empties),
        # and mouse events pass straight through it to the chart beneath.
        self._readout = QLabel("—", self)
        self._readout.setTextFormat(Qt.TextFormat.RichText)
        self._readout.setWordWrap(True)
        self._readout.setMinimumHeight(18)
        self._readout.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._readout.hide()
        self._inspector = ChartInspector(self._plot)
        self._inspector.set_readout(self._readout)
        self._inspector.pointSelected.connect(self.point_selected)
        # §23.6: the full category tick list behind the bottom axis (or
        # None); the rendered subset is re-derived on every resize.
        self._cat_ticks: list[tuple[float, str]] | None = None
        self._readout.show()
        self.apply_palette(self._palette)

    def resizeEvent(self, event) -> None:  # Qt signature
        """Pin the readout and re-fit the §23.6 tick density to the width."""
        super().resizeEvent(event)
        self._position_readout()
        self._apply_tick_density()

    # -- §23.6 adaptive tick-label density -----------------------------------
    def _bottom_axis_available_px(self) -> int:
        """Pixel width the bottom axis can draw its tick labels into.

        Measured from the PANEL's own width, not the plot widget's:
        the two are equal once the §23.2 zero-margin layout is live,
        but the panel's geometry updates immediately even in the
        pre-layout state where the layout has not propagated to the
        child yet (a hidden widget defers that to show()).
        """
        left = self._plot.getAxis("left")
        return max(0, self.width() - int(left.width()) - _TICK_PAD_PX)

    def _apply_tick_density(self) -> None:
        """Re-fit the category tick labels to the axis width (§23.6).

        Two-stage policy, both deterministic and width-driven:

        1. **stride thinning** -- the stride is chosen so that even the
           widest label (plus padding) fits between consecutive kept
           ticks, so no two rendered labels can overlap regardless of
           their text;
        2. **pixel-budget ellipsization** -- each kept label is elided
           to its per-tick share of the axis width, so one very long
           category name cannot spill into its neighbour either.

        The FULL list always stays in ``_cat_ticks`` -- the §21.3 bar
        hover meta and any explicit ``attach_metadata()`` continue to
        resolve the complete, untruncated category names; only the
        axis rendering thins.  A panel narrower than
        :data:`_MIN_AVAIL_PX` keeps all ticks -- that is the
        pre-layout state, and real layouts never get that narrow
        (the §23.2 splitter policy floors are 320 px).
        """
        axis = self._plot.getAxis("bottom")
        if not self._cat_ticks:
            return
        avail = self._bottom_axis_available_px()
        if avail < _MIN_AVAIL_PX:
            axis.setTicks([list(self._cat_ticks)])
            return
        font = axis.style.get("tickFont") or self.font()
        fm = QFontMetrics(font)
        worst = max(fm.horizontalAdvance(text) for _, text in self._cat_ticks) + _TICK_PAD_PX
        max_visible = max(1, int(avail // worst))
        stride = max(1, math.ceil(len(self._cat_ticks) / max_visible))
        kept = self._cat_ticks[::stride]
        share = int(max(_MIN_SHARE_PX, avail / len(kept) - _TICK_PAD_PX))
        rendered = [
            (pos, fm.elidedText(text, Qt.TextElideMode.ElideRight, share) or text)
            for pos, text in kept
        ]
        axis.setTicks([rendered])

    def _position_readout(self) -> None:
        if self._plot.height() <= 0:
            return
        self._readout.move(6, max(0, self.height() - self._readout.height() - 6))
        self._readout.setMaximumWidth(max(1, self.width() - 12))

    # -- theming -----------------------------------------------------------
    def apply_palette(self, palette: ChartPalette) -> None:
        """Re-style background, axes, grid, pen cycle and the §21 readout."""
        self._palette = palette
        self._plot.setBackground(palette.background)
        for axis_name in ("left", "bottom"):
            axis = self._plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(palette.foreground, width=1))
            axis.setTextPen(pg.mkPen(palette.foreground))
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._inspector.apply_palette(palette)
        self._readout.setStyleSheet(
            f"color: {palette.foreground}; background: transparent;"
            " font-size: 11px; padding: 1px 2px;"
        )

    # -- plotting (real data only) ------------------------------------------
    def scatter(self, x: np.ndarray, y: np.ndarray, *, label: str = "", symbol: str = "o", size: int = 6, index: int = 0, meta: Callable[[int], dict[str, Any]] | None = None) -> None:
        pen = pg.mkPen(None)
        brush = pg.mkBrush(self._palette.series[index % len(self._palette.series)])
        item = self._plot.plot(x, y, pen=pen, symbol=symbol, symbolSize=size, symbolBrush=brush, name=label or None)
        self._inspector.register_series(label, x, y, item, kind="scatter", meta=meta)

    def line(self, x: np.ndarray, y: np.ndarray, *, label: str = "", width: int = 2, index: int = 0, style: str = "solid", fill: bool = False, meta: Callable[[int], dict[str, Any]] | None = None) -> None:
        pen = pg.mkPen(self._palette.series[index % len(self._palette.series)], width=width)
        if style == "dash":
            pen.setStyle(Qt.PenStyle.DashLine)
        kwargs: dict[str, object] = {}
        if fill:  # area-under-the-curve shading (NCA profiles)
            fill_color = pg.mkColor(self._palette.series[index % len(self._palette.series)])
            fill_color.setAlpha(70)
            kwargs["fillLevel"] = 0.0
            kwargs["brush"] = pg.mkBrush(fill_color)
        item = self._plot.plot(x, y, pen=pen, name=label or None, **kwargs)
        self._inspector.register_series(label, x, y, item, kind="line", meta=meta)

    def vlines(self, xs, *, label: str = "") -> None:
        """Vertical marker lines (dose times, event boundaries).

        Palette-aware dotted lines; removed by the next ``clear()``.
        """
        for x in xs:
            pen = pg.mkPen(self._palette.foreground, width=1, style=Qt.PenStyle.DotLine)
            self._plot.addLine(x=float(x), pen=pen)

    def bars(self, heights: np.ndarray, *, labels: list[str] | None = None, index: int = 0, width: float = 0.6, x: np.ndarray | None = None, meta: Callable[[int], dict[str, Any]] | None = None) -> None:
        """Vertical bar chart (feature importance and histograms).

        ``x`` (round 215, the Foundations sampler) places the bars at
        VALUE-space positions -- the histogram centers -- instead of the
        default ``0..n-1`` slot grid, so a theoretical density curve can
        be drawn over the sample in the same coordinates.
        """
        if x is None:
            x = np.arange(len(heights), dtype=float)
        brush = pg.mkBrush(self._palette.series[index % len(self._palette.series)])
        bg = pg.BarGraphItem(x=x, height=heights, width=width, brush=brush)
        self._plot.addItem(bg)
        if labels:
            # §23.6: store the full list, render the width-fitted subset.
            self._cat_ticks = [(float(i), str(labels[i])) for i in range(len(labels))]
            self._apply_tick_density()
            if meta is None:
                # §21.3 bar charts: category + exact value hover for free.
                cat_labels = list(labels)

                def meta(i: int, _labels: list[str] = cat_labels) -> dict[str, Any]:
                    return {"category": _labels[i]}

        self._inspector.register_series(
            "" if labels is None else "bars",
            np.asarray(x, dtype=float),
            np.asarray(heights, dtype=float),
            bg,
            kind="bars",
            meta=meta,
            bar_width=float(width),
        )

    def histogram(self, values: np.ndarray, *, bins: int | np.ndarray = 30, index: int = 0, range: tuple[float, float] | None = None) -> None:
        counts, edges = np.histogram(values, bins=bins, range=range)
        centers = (edges[:-1] + edges[1:]) / 2.0

        def bin_meta(i: int) -> dict[str, Any]:
            # §21.3 histograms: exact bin boundaries + frequency.
            return {
                "bin": f"[{edges[i]:.6g}, {edges[i + 1]:.6g})",
                "count": int(counts[i]),
            }

        self.bars(counts, labels=None, index=index, width=float(np.median(np.diff(centers))), x=centers, meta=bin_meta)

    def clear(self) -> None:
        self._plot.clear()
        self._cat_ticks = None
        self._plot.getAxis("bottom").setTicks(None)
        # The inspector's hover decorations are lazy: plot.clear()
        # removed them along with the data, and nothing is re-added
        # until the next hover (see ChartInspector._ensure_decorations_added)
        # -- so the data-item list and the InfiniteLine census of a freshly
        # cleared plot stay exactly what the page's tests pin.
        self._inspector.reset()

    # -- hover inspection (Part-3 §21, round 232) ------------------------------
    def set_inspection_enabled(self, enabled: bool) -> None:
        """Enable/disable the crosshair + hover readout on this chart."""
        self._inspector.set_enabled(enabled)

    def inspection_enabled(self) -> bool:
        return self._inspector.enabled

    def attach_metadata(self, label: str, meta: Callable[[int], dict[str, Any]]) -> bool:
        """Attach a hover-metadata provider to a plotted series (§21.3)."""
        return self._inspector.attach_metadata(label, meta)

    # -- labels --------------------------------------------------------------
    def set_labels(self, x: str, y: str, title: str | None = None) -> None:
        self._plot.setLabel("bottom", x)
        self._plot.setLabel("left", y)
        if title is not None:
            self._plot.setTitle(title)

    def add_legend(self) -> None:
        self._plot.addLegend(size=(10, 10))

    # -- export (Part-3 §17) --------------------------------------------------
    def export_image(
        self,
        path: str | Path,
        *,
        jpeg_quality: int | None = None,
        compression: str | None = None,
        dpi: float | None = None,
        min_size: QSize | None = None,
    ):
        """Export the rendered plot to PNG/TIFF/JPEG (§17 engine).

        The extension picks the format; every write is reopened,
        magic-byte-sniffed and dimension-checked before success is
        reported.  ``min_size`` is the §23.3 legibility floor (a
        squeezed-live-geometry export re-renders from the vector scene
        at floor scale, aspect preserved); ``None`` keeps the
        widget-owns-its-size contract.  Returns the core
        :class:`~pharmaml.viz.export.ExportResult`.
        """
        return export_widget_grab(
            self._plot,
            path,
            jpeg_quality=jpeg_quality,
            compression=compression,
            dpi=dpi,
            min_size=min_size,
        )

    def widget(self) -> pg.PlotWidget:
        return self._plot
