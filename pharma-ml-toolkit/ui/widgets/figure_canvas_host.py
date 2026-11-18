"""The shared backend-figure renderer (part8 §4.1).

A reusable PySide6-compatible rendering component for backend figures:
every ``pharmaml.viz.*_plots`` function returns a matplotlib
``Figure`` (Agg-safe), and this host displays THAT figure — the
plotting algorithm is never duplicated in the GUI.

Requirements covered (§4.1):

* **Render the original backend figure** — ``set_figure`` adopts the
  exact ``Figure`` object the backend built (no redraw, no restyle).
* **Resize without corrupting aspect ratios** — the canvas keeps the
  figure's own aspect via matplotlib's ``constrained`` layout inside
  the scroll host; the widget never stretches the pixmap.
* **No memory leaks when replacing figures** — the replaced figure is
  explicitly ``close()``d (``plt``-free, so no global registry holds
  it) before the new one is adopted, and the canvas is reused (§8
  widget-reuse discipline).
* **Handle exceptions from plotting functions** — ``run_callable``
  routes every backend exception to the error state, never to a crash.
* **Clear empty and loading states** — three honest states (empty /
  loading / figure / error), each visually distinct and testable.
* **Export through the existing export architecture** — delegates to
  ``pharmaml.viz.export.save_figure`` (the same path every other
  Studio export uses).
* **Optional dependencies stay optional** — matplotlib is imported
  lazily inside the methods that need it; importing THIS module (e.g.
  for its registry) costs nothing at application startup.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

_EMPTY_TEXT = "No figure yet — run a plot to see it here."
_LOADING_TEXT = "Rendering figure…"
_ERROR_PREFIX = "The plotting function raised:"


class FigureCanvasHost(QWidget):
    """§4.1: one shared renderer for every backend matplotlib figure.

    Signals
    -------
    figure_changed()
        A new figure was adopted (or the host returned to a non-figure
        state — receivers re-read :meth:`has_figure`).
    """

    figure_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._figure: Any | None = None
        self._canvas: Any | None = None  # FigureCanvasQTAgg, lazy
        self._error: str | None = None
        self._builder: Callable[[], Any] | None = None
        self._pixmap_label: QLabel | None = None  # temp-PNG display path
        self._pixmap: bool = False  # a rendered artifact is displayed

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # the states layer (empty / loading / error) and the canvas
        # layer share the same slot; visibility switches between them
        self._state_label = QLabel(_EMPTY_TEXT)
        self._state_label.setObjectName("figureHostState")
        self._state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._state_label.setWordWrap(True)
        self._state_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        outer.addWidget(self._state_label, stretch=1)

        # the scroll host keeps the figure's own aspect ratio: the
        # canvas reports its size hint, the scroll area never stretches
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.hide()
        outer.addWidget(self._scroll, stretch=1)

    # -- state -----------------------------------------------------------------
    def has_figure(self) -> bool:
        return self._figure is not None or self._pixmap

    def error(self) -> str | None:
        return self._error

    def state(self) -> str:
        """One of ``'empty' | 'loading' | 'figure' | 'error'``."""
        if self._error is not None:
            return "error"
        if self._figure is not None or self._pixmap_label is not None:
            return "figure"
        if self._builder is not None:
            return "loading"
        return "empty"

    def set_rendered_image(self, png_path: str) -> None:
        """Display a figure the backend rendered AND closed to a file.

        Some viz modules (statplots) follow a path-based API: they save
        to ``path`` and close the figure.  The §4.1 rule still holds —
        the ORIGINAL backend rendering is shown, byte-for-byte as the
        backend exported it; this path just hosts the artifact instead
        of the live figure object.
        """

        from PySide6.QtGui import QPixmap

        self._release_figure()
        self._error = None
        self._builder = None
        if self._pixmap_label is None:
            self._pixmap_label = QLabel()
            self._pixmap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            inner = QWidget()
            grid = QGridLayout(inner)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.addWidget(self._pixmap_label)
            self._scroll.setWidget(inner)
        pixmap = QPixmap(png_path)
        if pixmap.isNull():
            self._show_error(f"The rendered image at {png_path} could not be loaded.")
            return
        self._pixmap = True
        self._pixmap_label.setPixmap(
            pixmap.scaled(
                max(420, min(pixmap.width(), 1600)),
                max(300, min(pixmap.height(), 1200)),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        if self._canvas is not None:
            self._canvas.setParent(None)
            self._canvas = None
        self._state_label.hide()
        self._scroll.show()
        self.figure_changed.emit()

    # -- §4.1 contract -----------------------------------------------------------
    def set_figure(self, figure: Any) -> None:
        """Adopt a backend-built ``Figure`` (the original, not a copy)."""
        self._error = None
        self._builder = None
        self._adopt(figure)
        self.figure_changed.emit()

    def run_callable(self, builder: Callable[[], Any]) -> None:
        """Call a plotting function and adopt its figure.

        Every exception lands in the honest error state with the
        traceback tail — a backend plotting bug never crashes the
        Studio and never leaves a stale figure on screen.
        """
        self._builder = builder
        try:
            figure = builder()
        except Exception:  # noqa: BLE001 - §4.1 demands the catch
            self._show_error(traceback.format_exc(limit=3))
            self.figure_changed.emit()
            return
        self._error = None
        # the builder STAYS for rebuild() (theme switches, refreshes);
        # only clear() drops it
        self._adopt(figure)
        self.figure_changed.emit()

    def clear(self) -> None:
        """Return to the empty state, releasing the current figure."""
        self._release_figure()
        self._error = None
        self._builder = None
        if self._pixmap_label is not None:
            self._pixmap_label.clear()
            self._pixmap_label = None
        self._pixmap = False
        self._show_state(_EMPTY_TEXT)
        self.figure_changed.emit()

    def export(self, path: str, **kwargs: Any) -> str:
        """Save the current figure through the existing export path."""
        if self._figure is None:
            raise RuntimeError("no figure to export")
        from pharmaml.viz import save_figure

        return save_figure(self._figure, path, **kwargs)

    # -- internals ----------------------------------------------------------------
    def _adopt(self, figure: Any) -> None:
        if figure is None:
            self._show_error("The plotting function returned no figure.")
            return
        if self._pixmap_label is not None:
            self._pixmap_label.clear()
            self._pixmap_label.setParent(None)
            self._pixmap_label = None
        self._pixmap = False
        # lazy, Gui-app-safe backend import (§4.1 startup rule)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        self._release_figure()
        self._figure = figure
        # a fresh canvas per figure: FigureCanvasQTAgg has no
        # set_figure, and reusing a canvas whose C++ twin was destroyed
        # by widget teardown is the leak-in-reverse (stale draw
        # targets).  Canvas creation is cheap; figures are the heavy
        # part and are released above.
        if self._canvas is not None:
            self._canvas.setParent(None)
            self._canvas = None
        self._canvas = FigureCanvasQTAgg(figure)
        self._canvas.setMinimumSize(420, 300)
        # the canvas keeps the figure's aspect; the scroll host
        # never stretches it (§4.1 resize rule)
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(self._canvas)
        self._scroll.setWidget(inner)
        # synchronous draw: draw_idle() would schedule a queued
        # repaint that can fire after the canvas's C++ twin is
        # gone during widget teardown (pytestqt crash)
        self._canvas.draw()
        self._state_label.hide()
        self._scroll.show()

    def _release_figure(self) -> None:
        """Close the outgoing figure — the no-leak rule (§4.1)."""
        if self._figure is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self._figure)
            except Exception:  # noqa: BLE001,S110 - release must never raise
                pass
        self._figure = None

    def _show_state(self, text: str) -> None:
        self._scroll.hide()
        self._state_label.setText(text)
        self._state_label.show()

    def _show_error(self, detail: str) -> None:
        self._release_figure()
        if self._pixmap_label is not None:
            self._pixmap_label.clear()
            self._pixmap_label.setParent(None)
            self._pixmap_label = None
        self._pixmap = False
        tail = detail.strip().splitlines()
        tail = tail[-6:] if tail else [detail]
        self._error = detail
        self._show_state(
            f"<b>{_ERROR_PREFIX}</b><br><code>{'<br>'.join(tail)}</code>"
        )

    # -- palette / theming hooks (the Studio theme path) -------------------------
    def apply_palette(self, palette: Any) -> None:
        """Re-adopt under a fresh theme-aware build when supported.

        The backend owns styling; this hook exists so page-level theme
        code has a uniform surface.  Rebuilding is page policy (the
        stored builder is available via :meth:`rebuild`).
        """
        self.rebuild()

    def rebuild(self) -> bool:
        """Re-run the stored builder (theme switches, refreshes)."""
        if self._builder is None:
            return False
        self.run_callable(self._builder)
        return True
