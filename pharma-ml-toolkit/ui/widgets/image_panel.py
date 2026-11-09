"""Theme-aware PyQtGraph image panel (directive 16.8.2-16.8.4, Part-3 §21.4).

``ImagePanel`` is the plot widget's sibling for MATRIX data: a
``pg.GraphicsLayoutWidget`` carrying an ``ImageItem`` (16.8.2), a
``HistogramLUTItem`` (16.8.4 -- the histogram + colormap control,
axis levels draggable in real time) and a ``RectROI`` (16.8.3 -- the
interactive region whose sub-matrix statistics are reported through
:attr:`roi_changed`).  ``export_png`` (16.8.7 / 16.19's export
criterion) grabs the rendered widget to a PNG file.

Part-3 §21.4 (round 240) -- the pixel inspection layer: every mouse
move over the scene maps through ``ImageItem.mapFromScene`` into data
coordinates and floors onto the pixel under the cursor; the payload
(row, column, exact cursor position, the pixel's value or per-channel
RGB values, NaN disclosure, axis labels when the matrix carried them)
rides :attr:`pixel_hovered` for the page's readout.  Emissions are
debounced per pixel (§21.6 -- a mouse-move storm over one cell emits
once), leaving the widget resets the readout, and
:meth:`image_metadata` exposes the at-rest facts (shape, dtype,
channels, finite/NaN census, value range).  :meth:`reset_view` is the
§21.2 reset-to-default control for images.  The probe behind the
mapping contract is ``scratch/probe_image_hover_240.py`` (round-trip
exactness + RGB feasibility, engine-printed first).

Like :class:`~ui.widgets.plot_widget.PlotPanel` the chrome re-styles
itself from the theme manager's :class:`ChartPalette` (background,
axis pens, ROI pen); the colormap itself stays a data-driven gradient
(presets viridis/magma/inferno/plasma/thermal/turbo/grey), never a
theme guess, so contrast is readable under every theme (16.11's
rule: a Qt stylesheet never reaches pyqtgraph's own rendering).

Part-3 §23.10 (round 245) -- the sub-pixel mapping disclosure under
extreme zoom: the §21.4 hover payload now carries
``image_px_screen``, the on-screen size (device px) of ONE image
pixel per axis, measured live from the view transform
(:meth:`view_pixel_scale`).  Zoomed out past ``ZOOMED_OUT_PX`` one
screen pixel can straddle several image pixels -- the readout then
discloses that the shown value is ONE of them (the ≤ceil coverage
note); zoomed in past ``ZOOMED_IN_PX`` one image pixel is a big
screen block -- the readout then explains why the value holds steady
inside it.  The note itself is :func:`format_scale_note`, a pure
function the Image Lab page rides.  The scale refresh rides the next
pixel transition (the §21.6 debounce keys on the pixel, so a zoom
with a stationary cursor re-discloses on the next move -- probed,
not guessed).

Part-3 §21.4 (round 246) -- the original-resolution view: the TRUE
1:1 toggle the stretch leg asked for, built on the §23.10 scale
measurement.  :meth:`set_original_resolution` sets the view range so
ONE image pixel renders at ONE screen pixel per axis (the
pixel-perfect "actual size" view): the target range is the view
box's own (width, height) in px -- recovered from the live
``viewRange()``/``viewPixelSize()`` pair (the circular arithmetic)
-- centred on the CURRENT viewport centre (the image-viewer
convention: the point under the eye stays under the eye), applied
with ``padding=0``.  The achieved scale is MEASURED back
(:meth:`view_pixel_scale`), not assumed -- the probe
(``scratch/probe_image_1to1_246.py``) pinned 1.0000 on both axes
from the fit / zoom-in / sub-pixel / non-square starts, exact centre
preservation, idempotence, and that a hidden widget answers a fake
default geometry (hence the ``isVisible()`` gate: the honest refusal
-- geometry is unknown until shown).  :meth:`is_original_resolution`
reports the ≈1:1 state within :data:`RESOLUTION_TOLERANCE` (integer
view quantisation keeps the achieved scale NEAR 1.0, not always at
it); the Image Lab page's 1:1 button toggles between this state and
the §21.2 fit, disclosing the achieved scale in its status line.

The ROI statistics use the probe's own arithmetic -- a bounding-rect
slice ``values[y0:y1, x0:x1]`` clipped to the matrix, then
count/mean/min/max over the finite entries -- so the page's tests pin
exactly what ``scratch/probe_imagelab_221.py`` printed.  For RGB(A)
input the statistics span every channel (documented; the count is
cells-times-channels).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QSize, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..theme_manager import ChartPalette
from .image_export import export_widget_grab

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

#: Part-3 §23.10 -- the on-screen size (device px) of ONE image pixel
#: below which the hover readout discloses sub-pixel coverage: a
#: screen pixel this small straddles more than one image pixel, so the
#: shown value is one of several under the cursor.
ZOOMED_OUT_PX = 0.9

#: Part-3 §23.10 -- the scale at/above which one image pixel is a
#: visible screen block, and the readout explains why the value holds
#: steady while the cursor moves inside it (the §21.6 debounce plus
#: a whole pixel under the cursor).
ZOOMED_IN_PX = 4.0

#: Part-3 §21.4 (round 246) -- how far each axis of the live scale
#: (screen px per image px) may sit from exactly 1.0 and still count
#: as the original-resolution view.  Integer view quantisation moves
#: the achieved scale by a fraction of a percent; a band this wide
#: tolerates that while still excluding every honest zoom state.
RESOLUTION_TOLERANCE = 0.05


class ImagePanel(QWidget):
    """A themed matrix view: image + histogram/LUT colormap + RectROI."""

    #: Emitted after the ROI moved/resized and the stats were recomputed.
    roi_changed = Signal()

    #: Emitted after every matrix/clear swap (§21.4 at-rest refresh).
    matrix_changed = Signal()

    #: Emitted with the §21.4 hover payload (a dict) for the pixel under
    #: the cursor, or ``None`` when the cursor leaves the image.  Keys:
    #: ``row``/``col`` (pixel indices), ``x``/``y`` (the exact cursor
    #: position in data units), ``value`` (the pixel value, ``None`` when
    #: NaN-masked), ``channels`` (1 for grayscale, 3/4 for RGB(A)),
    #: ``rgb`` (per-channel values for 3-D input, else ``None``),
    #: ``masked`` (NaN disclosure), ``x_label``/``y_label`` (the axis
    #: labels at that column/row when ``set_matrix`` carried them),
    #: ``shape`` (the displayed matrix shape), and ``image_px_screen``
    #: (§23.10 -- the (x, y) on-screen size of one image pixel in
    #: device px, measured at emission time; the sub-pixel disclosure
    #: refreshes on the next pixel transition after a zoom).
    pixel_hovered = Signal(object)

    def __init__(self, parent: QWidget | None = None, title: str = "") -> None:
        super().__init__(parent)
        self._palette = ChartPalette()
        self._values: np.ndarray | None = None
        self._x_labels: list[str] | None = None
        self._y_labels: list[str] | None = None
        self._last_pixel: tuple[int, int] | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._graphics = pg.GraphicsLayoutWidget()
        self._view = self._graphics.addPlot(row=0, col=0)
        # §23.4 "preserve image aspect ratios / avoid stretching":
        # matrix pixels stay square under every resize -- a 100×50
        # correlation matrix renders 2:1 on screen exactly as the data
        # is shaped, fit-to-view, never stretched to fill the panel.
        # (§23.5: display geometry only -- the values are untouched.)
        self._view.setAspectLocked(True)
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

        # §21.4/§21.6: the scene-wide mouse wiring -- pyqtgraph's own
        # mechanism (the same one ChartInspector rides).  Every move maps
        # through the ImageItem transform; emissions debounce per pixel.
        self._graphics.scene().sigMouseMoved.connect(self._on_scene_mouse_moved)

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
        """Display a 2-D matrix (NaN masked) or an RGB(A) (h,w,3|4) image.

        The 2-D path is the Lab's real data (similarity/response/explorer
        matrices): NaN entries are masked out honestly.  The 3-D path
        (§21.4's "RGB channel information") passes the array to
        ``ImageItem`` natively -- the values must be NaN-free (masked
        RGB arrays are refused with a clear error rather than rendered
        wrong), and the HistogramLUT control stops being meaningful
        (the colors come from the data itself).
        """
        array = np.asarray(values)
        if array.ndim == 3 and array.shape[2] in (3, 4):
            if not np.isfinite(array).all():
                raise ValueError(
                    "RGB(A) matrices must be NaN-free -- mask or impute "
                    "before display (the 2-D path supports honest NaN holes)"
                )
            self._values = array.astype(float)
            self._image.setImage(array)
            self._view.getAxis("bottom").setTicks(None)
            self._view.getAxis("left").setTicks(None)
        else:
            data = np.ma.masked_invalid(array.astype(float))
            self._values = array.astype(float)
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
        self._x_labels = list(x_labels) if x_labels is not None else None
        self._y_labels = list(y_labels) if y_labels is not None else None
        self._last_pixel = None  # the next hover must re-emit (§21.6)
        self._view.autoRange()
        self._on_roi_changed()
        self.matrix_changed.emit()

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

    # -- pixel inspection (Part-3 §21.4, round 240) -------------------------
    def pixel_at(self, x: float, y: float) -> dict | None:
        """The §21.4 payload for the pixel under data coordinates (x, y).

        ``x`` indexes columns, ``y`` rows (row-major, the axis order the
        item renders with); fractional positions floor onto the pixel.
        Coordinates outside the matrix -- including negatives -- return
        ``None`` (the honest refusal; the probe verified negatives stay
        negative through the item transform).  A NaN pixel reports
        ``value=None``/``masked=True`` instead of a number.
        """
        if self._values is None:
            return None
        rows, cols = self._values.shape[0], self._values.shape[1]
        col = int(np.floor(x))
        row = int(np.floor(y))
        if not (0 <= row < rows and 0 <= col < cols):
            return None
        payload: dict = {
            "row": row,
            "col": col,
            "x": float(x),
            "y": float(y),
            "channels": 1 if self._values.ndim == 2 else int(self._values.shape[2]),
            "rgb": None,
            "masked": False,
            "x_label": self._x_labels[col] if self._x_labels else None,
            "y_label": self._y_labels[row] if self._y_labels else None,
            "shape": tuple(int(s) for s in self._values.shape),
        }
        if self._values.ndim == 3:
            payload["rgb"] = tuple(
                float(v) for v in self._values[row, col]
            )
            payload["value"] = None  # no single scalar on the RGB path
        else:
            raw = float(self._values[row, col])
            if np.isnan(raw):
                payload["value"] = None
                payload["masked"] = True
            else:
                payload["value"] = raw
        return payload

    def image_metadata(self) -> dict | None:
        """The at-rest §21.4 facts: shape/dtype/channels/NaN census/range."""
        if self._values is None:
            return None
        finite = self._values[np.isfinite(self._values)]
        return {
            "rows": int(self._values.shape[0]),
            "cols": int(self._values.shape[1]),
            "shape": tuple(int(s) for s in self._values.shape),
            "dtype": str(self._values.dtype),
            "channels": 1 if self._values.ndim == 2 else int(self._values.shape[2]),
            "finite": int(finite.size),
            "masked": int(self._values.size - finite.size),
            "min": float(finite.min()) if finite.size else None,
            "max": float(finite.max()) if finite.size else None,
            "mean": float(finite.mean()) if finite.size else None,
        }

    def reset_view(self) -> None:
        """Reset to the default fit-to-view range (§21.2's reset control)."""
        self._view.autoRange()

    def set_original_resolution(self) -> tuple[float, float] | None:
        """§21.4's original-resolution view: 1 image px = 1 screen px.

        Sets the view range so ONE image pixel renders at ONE screen
        pixel per axis -- the pixel-perfect "actual size" view -- centred
        on the CURRENT viewport centre (the image-viewer convention:
        the point under the eye stays under the eye).  The target range
        is the view box's own (width, height) in px, recovered from the
        live ``viewRange()``/``viewPixelSize()`` pair (the circular
        arithmetic) and applied with ``padding=0``; the aspect lock
        leaves it alone because a range with equal data-units-per-px
        on both axes already satisfies it.

        Returns the ACHIEVED :meth:`view_pixel_scale` (≈(1.0, 1.0) --
        measured back, never assumed; integer view quantisation keeps
        it near, not always at, 1.0), or ``None`` when the view is
        unavailable: no matrix, the panel not shown (a hidden widget
        answers a fake default-geometry scale -- the probe pinned
        ~40 screen px per image px there, so the honest refusal gates
        on ``isVisible()``), or a degenerate view box.  The state is
        idempotent (re-apply stays ≈1:1) and is left by user zoom/pan
        or :meth:`reset_view`.
        """
        if self._values is None or not self.isVisible():
            return None
        vb = self._view.getViewBox()
        if vb is None:
            return None
        try:
            view_px = vb.viewPixelSize()
            (x0, x1), (y0, y1) = vb.viewRange()
        except Exception:  # noqa: BLE001 - degenerate pre-show geometry
            return None
        pw, ph = float(view_px[0]), float(view_px[1])
        if pw <= 0.0 or ph <= 0.0:
            return None
        box_w, box_h = (x1 - x0) / pw, (y1 - y0) / ph
        if box_w <= 0.0 or box_h <= 0.0:
            return None
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        vb.setRange(
            xRange=[cx - box_w / 2.0, cx + box_w / 2.0],
            yRange=[cy - box_h / 2.0, cy + box_h / 2.0],
            padding=0,
        )
        return self.view_pixel_scale()

    def is_original_resolution(self) -> bool:
        """Whether the live scale sits at ≈1:1 per axis.

        ``True`` only when BOTH axes of :meth:`view_pixel_scale` lie
        within :data:`RESOLUTION_TOLERANCE` of 1.0; ``False`` when the
        scale is missing (no data / degenerate view) or any honest
        zoom state sits outside the band.
        """
        scale = self.view_pixel_scale()
        if scale is None:
            return False
        sx, sy = scale
        return (
            abs(sx - 1.0) <= RESOLUTION_TOLERANCE
            and abs(sy - 1.0) <= RESOLUTION_TOLERANCE
        )

    def view_pixel_scale(self) -> tuple[float, float] | None:
        """The §23.10 live scale: one image pixel's on-screen size.

        Returns the (x, y) size in device pixels of ONE image pixel at
        the CURRENT view range -- ``1 / ViewBox.viewPixelSize()`` per
        axis, the transform the item renders through (aspect-locked,
        so the two axes agree except for stray rounding).  ``None``
        when no matrix is displayed, the view box is not yet wired, or
        the geometry is degenerate (a zero-sized view pre-show).
        Values ``< ZOOMED_OUT_PX`` mean a screen pixel straddles
        several image pixels; values ``>= ZOOMED_IN_PX`` mean one
        image pixel is a big screen block (see
        :func:`format_scale_note`).
        """
        if self._values is None:
            return None
        vb = self._view.getViewBox()
        if vb is None:
            return None
        try:
            view_px = vb.viewPixelSize()
        except Exception:  # noqa: BLE001 - degenerate pre-show geometry
            return None
        pw, ph = float(view_px[0]), float(view_px[1])
        if pw <= 0.0 or ph <= 0.0:
            return None
        return (1.0 / pw, 1.0 / ph)

    def _on_scene_mouse_moved(self, pos) -> None:  # QPointF from pyqtgraph
        if self._values is None:
            return
        mapped = self._image.mapFromScene(pos)
        payload = self.pixel_at(mapped.x(), mapped.y())
        key = None if payload is None else (payload["row"], payload["col"])
        if key == self._last_pixel:
            return  # §21.6 debounce: one emission per pixel transition
        self._last_pixel = key
        if payload is not None:
            # §23.10: the sub-pixel disclosure rides every real payload
            # -- measured at emission time, refreshed on the next pixel
            # transition after a zoom (the debounce keys on the pixel).
            payload["image_px_screen"] = self.view_pixel_scale()
        self.pixel_hovered.emit(payload)

    def _reset_hover(self) -> None:
        """Emit the cursor-left reset exactly once (leaveEvent path)."""
        if self._last_pixel is not None:
            self._last_pixel = None
            self.pixel_hovered.emit(None)

    def leaveEvent(self, event: QEvent) -> None:
        self._reset_hover()
        super().leaveEvent(event)

    def _on_roi_changed(self) -> None:
        self.roi_changed.emit()

    # -- export (16.8.7 / 16.19 / Part-3 §17) ------------------------------
    def export_image(
        self,
        path: str | Path,
        *,
        jpeg_quality: int | None = None,
        compression: str | None = None,
        dpi: float | None = None,
        min_size: QSize | None = None,
    ):
        """Export the rendered panel to PNG/TIFF/JPEG (§17 engine).

        The extension picks the format; TIFF carries the lossless
        compression choice (default ``tiff_lzw``), JPEG the quality
        setting, and the written file is reopened + validated before
        success is reported (§17.4).  ``min_size`` is the §23.3
        legibility floor (a squeezed-live-geometry export re-renders
        from the vector scene at floor scale, aspect preserved);
        ``None`` keeps the widget-owns-its-size contract.  Returns the
        core :class:`~pharmaml.viz.export.ExportResult`.
        """
        return export_widget_grab(
            self._graphics,
            path,
            jpeg_quality=jpeg_quality,
            compression=compression,
            dpi=dpi,
            min_size=min_size,
        )

    def export_png(self, path: str | Path) -> bool:
        """Grab the rendered panel to a PNG file; True on success.

        Compatibility shim over :meth:`export_image` (the pre-§17 API
        callers keep working, now validation-checked).
        """
        try:
            self.export_image(path)
            return True
        except Exception:  # noqa: BLE001 - boolean contract
            return False

    def clear(self) -> None:
        self._values = None
        self._x_labels = None
        self._y_labels = None
        self._last_pixel = None
        self._image.clear()
        self._view.getAxis("bottom").setTicks(None)
        self._view.getAxis("left").setTicks(None)
        self.matrix_changed.emit()

    # -- accessors for tests --------------------------------------------------
    def widget(self) -> pg.GraphicsLayoutWidget:
        return self._graphics

    def image_item(self) -> pg.ImageItem:
        return self._image

    def roi(self) -> pg.RectROI:
        return self._roi

    def values(self) -> np.ndarray | None:
        return self._values


def format_scale_note(scale: tuple[float, float] | None) -> str | None:
    """The §23.10 readout note for a live ``image_px_screen`` scale.

    Pure: ``None`` when the scale is missing, non-positive, near
    unity (a ~1:1 mapping needs no disclosure), or between the two
    thresholds.  Below :data:`ZOOMED_OUT_PX` the note discloses the
    ≤ceil coverage -- how many image pixels one screen pixel can
    straddle, with the honest "(one shown)"; at/above
    :data:`ZOOMED_IN_PX` it explains the steady value inside a
    zoomed-in block.  Per-axis ceil is the correct worst case: a
    screen pixel ``1/sx`` image units wide touches at most
    ``ceil(1/sx)`` pixel columns (boundary straddle included).
    """
    if not scale:
        return None
    sx, sy = float(scale[0]), float(scale[1])
    if sx <= 0.0 or sy <= 0.0:
        return None
    if min(sx, sy) < ZOOMED_OUT_PX:
        cover_x = int(np.ceil(1.0 / sx))
        cover_y = int(np.ceil(1.0 / sy))
        return f"1 screen px spans ≤{cover_x}×{cover_y} image px (one shown)"
    if min(sx, sy) >= ZOOMED_IN_PX:
        return f"1 image px ≈ {sx:.0f}×{sy:.0f} screen px (value steady in block)"
    return None


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
