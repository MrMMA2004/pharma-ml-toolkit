"""Per-page chart export affordance (R30-c, part7 §17 engine).

Every chart-bearing workspace can save its plot: one shared helper
attaches an "Export chart…" button to a page's layout, wired to the
:class:`~ui.widgets.plot_widget.PlotPanel`'s ``export()`` (the §17
raster engine — PNG/TIFF/JPEG with reopen + magic-sniff + dimension
checks) under the studio palette the panel already renders with.

The button is deliberately TINY and lives at the end of the plot's
column: exporting is a power-user action, never the page's primary
flow.  Status goes to the page's own report label when it has one
(the pages all expose ``_report``); failures surface as report text,
never dialogs.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QPushButton, QWidget

from .responsive import LEGIBILITY_FLOOR

__all__ = ["attach_chart_export"]


def attach_chart_export(
    page: QWidget,
    panel,
    name: str,
    *,
    layout=None,
    status_label=None,
) -> QPushButton:
    """Add the per-page chart-export button.

    Parameters
    ----------
    page:
        The workspace page (used as the file dialog's parent and the
        fallback status target).
    panel:
        A :class:`PlotPanel` (anything with ``export_image(path,
        ..., min_size=...)``).
    name:
        The default file stem, e.g. ``"chemistry-features"``.
    layout:
        Where the button goes — a ``QHBoxLayout`` the caller owns.
        When ``None`` a fresh row is appended to the page's own
        layout (must be a ``QVBoxLayout``-like ``addWidget``).
    status_label:
        A ``QLabel``-like target for the result line (the page's
        ``_report``).  Falls back to ``page._report`` when present,
        else nothing (the export still happens).

    Returns
    -------
    QPushButton
        The wired button (tests find it by ``objectName``).
    """
    button = QPushButton("Export chart…")
    button.setObjectName("chartExportButton")
    button.setToolTip(
        "Save this plot as PNG / TIFF / JPEG (the §17 raster engine, "
        "reopened and verified before success is reported)"
    )

    def _export() -> None:
        path, _selected = QFileDialog.getSaveFileName(
            page,
            "Export the chart",
            f"{name}.png",
            "PNG (*.png);;TIFF (*.tif *.tiff);;JPEG (*.jpg *.jpeg);;All files (*)",
        )
        if not path:
            return
        target = status_label or getattr(page, "_report", None)
        try:
            result = panel.export_image(Path(path), min_size=LEGIBILITY_FLOOR)
        except Exception as exc:  # noqa: BLE001 - surfaced, never fatal
            if target is not None:
                target.setText(f"Chart export FAILED — {exc}")
            return
        if target is not None:
            target.setText(
                f"Chart exported — {result.path} "
                f"({result.format}, {result.width}x{result.height}, "
                f"{result.size_bytes:,} bytes)"
            )

    button.clicked.connect(_export)
    if layout is not None:
        layout.addWidget(button)
    return button
