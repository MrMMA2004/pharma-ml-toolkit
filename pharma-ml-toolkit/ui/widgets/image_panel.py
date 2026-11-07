"""Theme-aware PyQtGraph image panel (directive 16.8.2-16.8.4).

``ImagePanel`` is the plot widget's sibling for MATRIX data: a
``pg.GraphicsLayoutWidget`` carrying an ``ImageItem`` (16.8.2), a
``HistogramLUTItem`` (16.8.4 -- the histogram + colormap control,
axis levels draggable in real time) and a ``RectROI`` (16.8.3 -- the
interactive region whose sub-matrix statistics are reported through
:attr:`roi_changed`).  ``export_png`` (16.8.7 / 16.19's export
criterion) grabs the rendered widget to a PNG file.

Like :class:`~ui.widgets.plot_widget.PlotPanel` the chrome re-styles
itself from the theme manager's :class:`ChartPalette` (background,
axis pens, ROI pen); the colormap itself stays a data-driven gradient
(presets viridis/magma/inferno/plasma/thermal/turbo/grey), never a
theme guess, so contrast is readable under every theme (16.11's
rule: a Qt stylesheet never reaches pyqtgraph's own rendering).

The ROI statistics use the probe's own arithmetic -- a bounding-rect
slice ``values[y0:y1, x0:x1]`` clipped to the matrix, then
count/mean/min/max over the finite entries -- so the page's tests pin
exactly what ``scratch/probe_imagelab_221.py`` printed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..theme_manager import ChartPalette

#: The colormap presets the HistogramLUT control can switch between
#: (16.8.4).  Deterministic, print-safe gradients -- viridis is the
#: default because its perceptual uniformity survives both dark and
#: light chrome.
GRADIENT_PRESETS: tuple[str, ...] = (
    "viridis",
    "magma",
    "inferno",
    "plasma",
    "thermal",
    "turbo",
    "grey",
)


class ImagePanel(QWidget):
    """A themed matrix view: image + histogram/LUT colormap + RectROI."""

    #: Emitted after the ROI moved/resized and the stats were recomputed.
    roi_changed = Signal()

    def __init__(self, parent: QWidget | None = None, title: str = "") -> None:
        super().__init__(parent)
        self._palette = ChartPalette()
        self._values: np.ndarray | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._graphics = pg.GraphicsLayoutWidget()
        self._view = self._graphics.addPlot(row=0, col=0)
        if title:
            self._view.setTitle(title)
        self._view.showGrid(x=True, y=True, alpha=0.2)
        self._image = pg.ImageItem(axisOrder="row-major")
        self._view.addItem(self._image)

        # 16.8.4: the histogram + colormap control beside the image.
        self._hist = pg.HistogramLUTItem()
        self._hist.setImageItem(self._image)
        self._graphics.addItem(self._hist, row=0, col=1)

        # 16.8.3: the interactive region, dragged by its handles.
        self._roi = pg.RectROI(
            [0.0, 0.0],
            [4.0, 4.0],
            pen=pg.mkPen(self._palette.series[0], width=2),
            removable=False,
        )
        self._roi.sigRegionChanged.connect(self._on_roi_changed)
        self._view.addItem(self._roi)

        layout.addWidget(self._graphics)
        self.apply_palette(self._palette)
        self.set_gradient("viridis")

    # -- theming -----------------------------------------------------------
    def apply_palette(self, palette: ChartPalette) -> None:
        """Re-style chrome (background, axes, ROI pen); colormap untouched."""
        self._palette = palette
        self._graphics.setBackground(palette.background)
        for axis_name in ("left", "bottom"):
            axis = self._view.getAxis(axis_name)
            axis.setPen(pg.mkPen(palette.foreground, width=1))
            axis.setTextPen(pg.mkPen(palette.foreground))
        self._roi.setPen(pg.mkPen(palette.series[0], width=2))

    # -- matrix input (real data only) --------------------------------------
    def set_matrix(
        self,
        values: np.ndarray,
        *,
        x_labels: list[str] | None = None,
        y_labels: list[str] | None = None,
    ) -> None:
        """Display a 2-D matrix; NaN entries are masked out honestly."""
        data = np.ma.masked_invalid(np.asarray(values, dtype=float))
        self._values = np.asarray(values, dtype=float)
        self._image.setImage(data)
        if x_labels is not None:
            ticks = [(float(i), str(label)) for i, label in enumerate(x_labels)]
            self._view.getAxis("bottom").setTicks([ticks])
        else:
            self._view.getAxis("bottom").setTicks(None)
        if y_labels is not None:
            ticks = [(float(i), str(label)) for i, label in enumerate(y_labels)]
            self._view.getAxis("left").setTicks([ticks])
        else:
            self._view.getAxis("left").setTicks(None)
        self._view.autoRange()
        self._on_roi_changed()

    def set_gradient(self, preset: str) -> None:
        """Switch the colormap preset (16.8.4); unknown names refuse."""
        if preset not in GRADIENT_PRESETS:
            raise ValueError(
                f"unknown gradient preset {preset!r} -- "
                f"choose one of {', '.join(GRADIENT_PRESETS)}"
            )
        self._hist.gradient.loadPreset(preset)

    # -- ROI statistics (16.8.3) --------------------------------------------
    def roi_stats(self) -> dict[str, float] | None:
        """count/mean/min/max over the ROI's finite sub-matrix, or None."""
        if self._values is None:
            return None
        height, width = self._values.shape
        x0, y0 = self._roi.pos()
        w, h = self._roi.size()
        col0 = int(np.floor(max(x0, 0.0)))
        col1 = int(np.ceil(min(x0 + w, float(width))))
        row0 = int(np.floor(max(y0, 0.0)))
        row1 = int(np.ceil(min(y0 + h, float(height))))
        col0, col1 = max(col0, 0), min(col1, width)
        row0, row1 = max(row0, 0), min(row1, height)
        if col1 <= col0 or row1 <= row0:
            return None
        region = self._values[row0:row1, col0:col1]
        finite = region[np.isfinite(region)]
        if finite.size == 0:
            return None
        return {
            "rows": float(row1 - row0),
            "cols": float(col1 - col0),
            "count": float(finite.size),
            "mean": float(finite.mean()),
            "min": float(finite.min()),
            "max": float(finite.max()),
        }

    def _on_roi_changed(self) -> None:
        self.roi_changed.emit()

    # -- export (16.8.7 / 16.19) ---------------------------------------------
    def export_png(self, path: str | Path) -> bool:
        """Grab the rendered panel to a PNG file; True on success."""
        pixmap = self._graphics.grab()
        ok = pixmap.save(str(path))
        return bool(ok)

    def clear(self) -> None:
        self._values = None
        self._image.clear()
        self._view.getAxis("bottom").setTicks(None)
        self._view.getAxis("left").setTicks(None)

    # -- accessors for tests --------------------------------------------------
    def widget(self) -> pg.GraphicsLayoutWidget:
        return self._graphics

    def image_item(self) -> pg.ImageItem:
        return self._image

    def roi(self) -> pg.RectROI:
        return self._roi

    def values(self) -> np.ndarray | None:
        return self._values


def roi_stats_from_values(
    values: np.ndarray, row0: int, row1: int, col0: int, col1: int
) -> dict[str, float] | None:
    """The probe's arithmetic, shared by tests: slice + finite stats.

    ``ImagePanel.roi_stats`` clips the ROI bounding rect to the matrix
    and reports count/mean/min/max over the finite entries; this pure
    helper is the same math without a widget, so the page tests can
    pin the probe numbers directly.
    """
    region = np.asarray(values, dtype=float)[row0:row1, col0:col1]
    finite = region[np.isfinite(region)]
    if finite.size == 0:
        return None
    return {
        "rows": float(row1 - row0),
        "cols": float(col1 - col0),
        "count": float(finite.size),
        "mean": float(finite.mean()),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }
