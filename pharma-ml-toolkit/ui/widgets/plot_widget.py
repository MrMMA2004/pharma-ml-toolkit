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

# §23.7 (round 239) legend hygiene: a legend is also unbounded -- long
# series names spill sideways over the data and dozens of entries stack
# past the plot's bottom edge.  The policy below ellipsizes each legend
# label to a pixel budget and escalates the column count when the
# entries would stack past the vertical capacity.
_LEGEND_ROW_PX = 22        # empirical per-row height incl. spacing
_LEGEND_MAX_COLS = 3       # the legend never widens past 3 columns
_LEGEND_MIN_ROWS = 2       # even a tiny plot gets 2 rows per column
_LEGEND_MAX_LABEL_PX = 220 # absolute ceiling for one legend label


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
        # §23.7: id(LabelItem) -> the full, unelided legend text, so the
        # label budget can follow the width BOTH ways; pruned against the
        # live legend items on every fit (clear()/replot never leak ids).
        self._legend_full: dict[int, str] = {}
        self._readout.show()
        self.apply_palette(self._palette)

    def resizeEvent(self, event) -> None:  # Qt signature
        """Pin the readout and re-fit the §23.6/§23.7 density policies."""
        super().resizeEvent(event)
        self._position_readout()
        self._apply_tick_density()
        self._fit_legend()

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

    # -- §23.7 legend hygiene -------------------------------------------------
    def _fit_legend(self) -> None:
        """Re-fit the legend to the panel geometry (§23.7).

        A pyqtgraph legend is unbounded: long series names spill
        sideways over the data and dozens of entries stack past the
        plot's bottom edge.  Two-stage policy, re-run after every
        data-adding call and on every geometry change:

        1. **label budget** -- each legend label wider than
           ``min(_LEGEND_MAX_LABEL_PX, 45% of the panel width)`` is
           elided; the full text is kept behind the panel, so the
           budget follows the width BOTH ways -- a label elided on a
           narrow panel is restored verbatim when the panel widens
           again;
        2. **column escalation** -- when the entry count exceeds the
           vertical capacity (the height the bottom axis leaves, at
           :data:`_LEGEND_ROW_PX` per row), the legend splits into
           columns, capped at :data:`_LEGEND_MAX_COLS`.

        Display-only: the legend relabels never touch the §21
        inspector's series keys -- hover metadata, click-to-select and
        ``attach_metadata()`` keep resolving the full names.
        """
        legend = self._plot.plotItem.legend
        if legend is None:
            return
        if not legend.items:
            self._legend_full.clear()
            return
        live_ids = {id(label) for _sample, label in legend.items}
        for stale in [k for k in self._legend_full if k not in live_ids]:
            del self._legend_full[stale]
        fm = QFontMetrics(self.font())
        budget = min(_LEGEND_MAX_LABEL_PX, max(80, int(self.width() * 0.45)))
        for _sample, label in legend.items:
            full = self._legend_full.setdefault(id(label), label.text)
            elided = fm.elidedText(full, Qt.TextElideMode.ElideRight, budget)
            if elided != label.text:
                label.setText(elided or full)
        avail_h = self.height() - int(self._plot.getAxis("bottom").height()) - 24
        rows = max(_LEGEND_MIN_ROWS, avail_h // _LEGEND_ROW_PX)
        cols = min(_LEGEND_MAX_COLS, max(1, math.ceil(len(legend.items) / rows)))
        if getattr(legend, "columnCount", 1) != cols:
            legend.setColumnCount(cols)
            legend.updateSize()

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
        self._fit_legend()

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
        self._fit_legend()

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
        self._fit_legend()

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
