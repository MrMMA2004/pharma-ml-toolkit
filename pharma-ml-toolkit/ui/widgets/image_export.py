"""Qt widget grab → §17 export bridge (Part-3 §17 for the GUI layer).

One helper shared by every panel that renders through Qt: grab the
widget (or scene) to PNG bytes, then run those bytes through
:func:`pharmaml.viz.export.export_widget_image` -- the same
integrity-checked TIFF/JPEG/PNG pipeline the core module gives numpy
arrays and matplotlib figures.  The GUI never re-implements format
policy; it renders, the engine decides.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pharmaml.viz.export import export_widget_image

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PySide6.QtWidgets import QWidget

    from pharmaml.viz.export import ExportResult

__all__ = ["export_widget_grab", "grab_png_bytes"]


def grab_png_bytes(widget: QWidget) -> bytes:
    """Render a widget offscreen to PNG bytes (the Qt-native grab).

    ``QWidget.grab`` works headless (the studio's offscreen mode
    included) and renders at the widget's current size -- the same
    pixels the user sees, alpha channel included.
    """
    from PySide6.QtCore import QBuffer, QIODevice

    pixmap = widget.grab()
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    saved = pixmap.save(buffer, "PNG")
    if not saved:  # pragma: no cover - Qt failure path
        raise RuntimeError("Qt pixmap grab failed -- widget not rendered?")
    return bytes(buffer.data().data())


def export_widget_grab(
    widget: QWidget,
    path: str | Path,
    *,
    jpeg_quality: int | None = None,
    compression: str | None = None,
    dpi: float | None = None,
) -> ExportResult:
    """Grab ``widget`` and export it to any §17 raster format.

    The extension of ``path`` picks the format (``.png``/``.tif``/
    ``.tiff``/``.jpg``/``.jpeg``); TIFF writes carry the lossless
    compression choice, JPEG writes the quality setting, and every
    file is reopened + magic-sniffed + dimension-checked before
    success is reported (§17.4).
    """
    return export_widget_image(
        grab_png_bytes(widget),
        path,
        jpeg_quality=jpeg_quality,
        compression=compression,
        dpi=dpi,
    )
