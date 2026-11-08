"""Qt widget grab → §17 export bridge (Part-3 §17 for the GUI layer).

One helper shared by every panel that renders through Qt: grab the
widget (or scene) to PNG bytes, then run those bytes through
:func:`pharmaml.viz.export.export_widget_image` -- the same
integrity-checked TIFF/JPEG/PNG pipeline the core module gives numpy
arrays and matplotlib figures.  The GUI never re-implements format
policy; it renders, the engine decides.

Part-3 §23.3 ("generated plots remain legible when exported") adds the
optional *legibility floor*: ``min_size`` re-renders a cramped widget
from its vector scene at floor scale when the live geometry has been
squeezed below it -- text, pens and curves are re-rendered crisp at
the larger size (never bitmap-stretched), the aspect ratio is
preserved (§23.4), and the §23.5 rule holds: display geometry only,
the underlying data and plot ranges are untouched.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from pharmaml.viz.export import export_widget_image

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PySide6.QtCore import QSize
    from PySide6.QtWidgets import QWidget

    from pharmaml.viz.export import ExportResult

__all__ = ["export_widget_grab", "grab_png_bytes"]


def grab_png_bytes(
    widget: QWidget,
    *,
    min_size: QSize | None = None,
) -> bytes:
    """Render a widget offscreen to PNG bytes (the Qt-native grab).

    ``QWidget.grab`` works headless (the studio's offscreen mode
    included) and renders at the widget's current size -- the same
    pixels the user sees, alpha channel included.

    ``min_size`` (the §23.3 legibility floor, logical px) is honoured
    for ``QGraphicsView``-backed panels (every plot/image panel in the
    studio): when the live viewport is smaller than the floor, the
    scene is re-rendered into a floor-scaled pixmap -- same view, same
    aspect, vector-crisp text.  Widgets already at or above the floor,
    and plain non-scene widgets, take the plain grab path unchanged.
    """
    from PySide6.QtCore import QBuffer, QIODevice, QRectF, Qt
    from PySide6.QtGui import QPainter, QPixmap
    from PySide6.QtWidgets import QGraphicsView

    pixmap = widget.grab()
    if min_size is not None and isinstance(widget, QGraphicsView):
        floored = _scene_grab_at_floor(widget, min_size)
        if floored is not None:
            pixmap = floored

    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    saved = pixmap.save(buffer, "PNG")
    if not saved:  # pragma: no cover - Qt failure path
        raise RuntimeError("Qt pixmap grab failed -- widget not rendered?")
    return bytes(buffer.data().data())


def _scene_grab_at_floor(
    view: QGraphicsView,
    min_size: QSize,
) -> QPixmap | None:
    """Re-render the view's scene at floor scale, or ``None`` to skip.

    ``None`` (fall back to the plain grab) when the viewport is not
    laid out yet, already meets the floor, or the scene mapping is
    degenerate -- the floor is a render upgrade, never a failure mode.
    """
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QPainter, QPixmap

    viewport = view.viewport()
    vp_w, vp_h = viewport.width(), viewport.height()
    if vp_w <= 0 or vp_h <= 0:
        return None  # pragma: no cover - not shown/laid out yet
    if vp_w >= min_size.width() and vp_h >= min_size.height():
        return None  # already legible -- keep the exact-screen contract

    from PySide6.QtCore import Qt

    source = view.mapToScene(viewport.rect()).boundingRect()
    if source.width() <= 0 or source.height() <= 0:  # pragma: no cover - degenerate
        return None

    scale = max(
        min_size.width() / source.width(),
        min_size.height() / source.height(),
    )
    target_w = int(math.ceil(source.width() * scale))
    target_h = int(math.ceil(source.height() * scale))

    pixmap = QPixmap(target_w, target_h)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        # pyqtgraph paints its chrome background on the VIEW layer
        # (GraphicsView.drawBackground), which scene.render never
        # calls -- fill it ourselves so the floor render keeps the
        # on-screen background colour instead of exporting it
        # transparent by accident.
        brush = view.backgroundBrush()
        if brush is not None and brush.color().alpha() > 0:
            painter.fillRect(0, 0, target_w, target_h, brush)
        view.scene().render(
            painter,
            QRectF(0.0, 0.0, float(target_w), float(target_h)),
            source,
        )
    finally:
        painter.end()
    return pixmap


def export_widget_grab(
    widget: QWidget,
    path: str | Path,
    *,
    jpeg_quality: int | None = None,
    compression: str | None = None,
    dpi: float | None = None,
    min_size: QSize | None = None,
) -> ExportResult:
    """Grab ``widget`` and export it to any §17 raster format.

    The extension of ``path`` picks the format (``.png``/``.tif``/
    ``.tiff``/``.jpg``/``.jpeg``); TIFF writes carry the lossless
    compression choice, JPEG writes the quality setting, and every
    file is reopened + magic-sniffed + dimension-checked before
    success is reported (§17.4).

    ``min_size`` is the §23.3 legibility floor (see
    :func:`grab_png_bytes`): ``None`` keeps the documented
    widget-owns-its-size contract, user-facing export buttons pass
    :data:`ui.widgets.responsive.LEGIBILITY_FLOOR`.
    """
    return export_widget_image(
        grab_png_bytes(widget, min_size=min_size),
        path,
        jpeg_quality=jpeg_quality,
        compression=compression,
        dpi=dpi,
    )
