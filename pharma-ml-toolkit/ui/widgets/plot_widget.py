"""Theme-aware PyQtGraph plot panel (directive 16.8).

``PlotPanel`` wraps ``pg.PlotWidget`` and re-styles itself whenever the
theme manager emits a new :class:`~pharmaml.ui.theme_manager.ChartPalette`
-- dark themes get dark backgrounds, readable axis labels and visible
grids (16.11: a Qt stylesheet never reaches pyqtgraph's own rendering).
Scientific data goes in as numpy arrays with real labels/units.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..theme_manager import ChartPalette
from .image_export import export_widget_grab


class PlotPanel(QWidget):
    """A single themed plot with labelled axes and a callable API."""

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
        self.apply_palette(self._palette)

    # -- theming -----------------------------------------------------------
    def apply_palette(self, palette: ChartPalette) -> None:
        """Re-style background, axes, grid and the default pen cycle."""
        self._palette = palette
        self._plot.setBackground(palette.background)
        for axis_name in ("left", "bottom"):
            axis = self._plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(palette.foreground, width=1))
            axis.setTextPen(pg.mkPen(palette.foreground))
        self._plot.showGrid(x=True, y=True, alpha=0.25)

    # -- plotting (real data only) ------------------------------------------
    def scatter(self, x: np.ndarray, y: np.ndarray, *, label: str = "", symbol: str = "o", size: int = 6, index: int = 0) -> None:
        pen = pg.mkPen(None)
        brush = pg.mkBrush(self._palette.series[index % len(self._palette.series)])
        self._plot.plot(x, y, pen=pen, symbol=symbol, symbolSize=size, symbolBrush=brush, name=label or None)

    def line(self, x: np.ndarray, y: np.ndarray, *, label: str = "", width: int = 2, index: int = 0, style: str = "solid", fill: bool = False) -> None:
        pen = pg.mkPen(self._palette.series[index % len(self._palette.series)], width=width)
        if style == "dash":
            pen.setStyle(Qt.PenStyle.DashLine)
        kwargs: dict[str, object] = {}
        if fill:  # area-under-the-curve shading (NCA profiles)
            fill_color = pg.mkColor(self._palette.series[index % len(self._palette.series)])
            fill_color.setAlpha(70)
            kwargs["fillLevel"] = 0.0
            kwargs["brush"] = pg.mkBrush(fill_color)
        self._plot.plot(x, y, pen=pen, name=label or None, **kwargs)

    def vlines(self, xs, *, label: str = "") -> None:
        """Vertical marker lines (dose times, event boundaries).

        Palette-aware dotted lines; removed by the next ``clear()``.
        """
        for x in xs:
            pen = pg.mkPen(self._palette.foreground, width=1, style=Qt.PenStyle.DotLine)
            self._plot.addLine(x=float(x), pen=pen)

    def bars(self, heights: np.ndarray, *, labels: list[str] | None = None, index: int = 0, width: float = 0.6, x: np.ndarray | None = None) -> None:
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
            ticks = [(float(i), labels[i]) for i in range(len(labels))]
            self._plot.getAxis("bottom").setTicks([ticks])

    def histogram(self, values: np.ndarray, *, bins: int | np.ndarray = 30, index: int = 0, range: tuple[float, float] | None = None) -> None:
        counts, edges = np.histogram(values, bins=bins, range=range)
        centers = (edges[:-1] + edges[1:]) / 2.0
        self.bars(counts, labels=None, index=index, width=float(np.median(np.diff(centers))), x=centers)

    def clear(self) -> None:
        self._plot.clear()
        self._plot.getAxis("bottom").setTicks(None)

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
    ):
        """Export the rendered plot to PNG/TIFF/JPEG (§17 engine).

        The extension picks the format; every write is reopened,
        magic-byte-sniffed and dimension-checked before success is
        reported.  Returns the core
        :class:`~pharmaml.viz.export.ExportResult`.
        """
        return export_widget_grab(
            self._plot,
            path,
            jpeg_quality=jpeg_quality,
            compression=compression,
            dpi=dpi,
        )

    def widget(self) -> pg.PlotWidget:
        return self._plot
