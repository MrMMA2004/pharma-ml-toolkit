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

Round 243 (Part-3 §23.9): ``add_legend()`` docks the legend BELOW the
plot -- in the layout row under the bottom axis, in the view column --
instead of pyqtgraph's default float INSIDE the view box at a (30, 30)
offset over the data (probed on 0.13.7).  Hard ``maximumHeight`` /
``maximumWidth`` caps (re-set on every fit) keep the band from eating
the plot or widening the grid, and the §23.7 label-budget /
column-escalation policies re-target the docked row.  The §21 hover
readout re-anchors to the view box's bottom edge whenever a live
legend band sits under the plot.
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

from ..design_tokens import MARKER_DISCIPLINE_DARK, MARKER_DISCIPLINE_LIGHT
from ..theme_manager import ChartPalette
from .chart_inspector import ChartInspector
from .image_export import export_widget_grab

#: R29-b shapes for series whose color needs a non-color encoding.
#: pyqtgraph's recognised symbol names, cycled by series index.
_DISCIPLINE_SYMBOLS: tuple[str, ...] = ("t", "s", "d", "p", "x")


def _needs_marker(palette: ChartPalette, color: str) -> bool:
    """Whether this series must carry a non-color encoding (§12).

    The discipline set is theme-aware (ui.design_tokens): light flags
    the sub-3:1-on-white series colors; dark flags the green/sky
    companion pair, whose token-export values converge to
    near-identical mint tints (1.08:1, 14° apart).
    """
    discipline = MARKER_DISCIPLINE_DARK if palette.dark \
        else MARKER_DISCIPLINE_LIGHT
    return color in discipline


def _discipline_symbol(index: int) -> str:
    """A stable marker shape per series index for flagged series."""
    return _DISCIPLINE_SYMBOLS[index % len(_DISCIPLINE_SYMBOLS)]

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

# §23.8 (round 242) title budget: a plot title is the last unbounded
# string on the panel -- pages compose long titles (method + facts
# suffixes like "UMAP — trustworthiness 0.87 · a=1.200 b=0.300") and
# pyqtgraph renders them verbatim, overflowing a narrow panel.  The
# policy below ellipsizes the RENDERED title to the view column's
# pixel width (middle elide: titles are centered and BOTH ends carry
# meaning -- the method up front, the computed facts at the end),
# while the full string stays on the panel verbatim (attribute +
# tooltip) and is restored whenever the width lets it fit again.
_TITLE_PAD_PX = 12  # breathing room around the centered title
_TITLE_FALLBACK_PT = 9.0  # pyqtgraph 0.13.7's default title size (probed)

# §23.9 (round 243) legend placement: pyqtgraph's default legend floats
# INSIDE the view box at a (30, 30) offset -- over the data (probed on
# 0.13.7: the legend's scene rect intersects the view box's, always).
# The panel docks the legend in the layout row BELOW the bottom axis
# instead -- structurally incapable of covering a data point at any
# width -- with hard size caps so the docked band stays a guest of the
# plot, never its landlord.
_LEGEND_BAND_FRAC = 0.35    # the docked row may claim at most this fraction of the panel height
_LEGEND_SAMPLE_PX = 40      # per-entry overhead: the ItemSample swatch + spacings (probed)
_LEGEND_MIN_LABEL_PX = 24   # a label never gets less than this much room


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
        # R29-b: how many flagged series this panel has drawn — the
        # marker-shape cycle keys on it so co-plotted low-contrast
        # series always take DIFFERENT shapes (raw index 1 and 6
        # would collide on the 5-shape cycle).
        self._discipline_count = 0
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
        # §23.8: the full, verbatim title behind the rendered one (and
        # the last string the policy actually rendered, so setTitle is
        # only touched when the fit CHANGES -- pyqtgraph's setTitle
        # rebuilds the LabelItem each call).
        self._title_full = title or ""
        self._title_rendered: str | None = None
        if self._title_full:
            self.setToolTip(self._title_full)
        self._readout.show()
        # (C-1) first-run guidance: a hint painted INSIDE the plot until
        # the first series lands — an empty white panel explains nothing.
        self._empty_hint: str = ""
        self._hint_item: pg.TextItem | None = None
        self.apply_palette(self._palette)
        # review 4.4: the hint used a fixed #64748B — light-theme grey —
        # which sank below comfortable contrast on dark plot backgrounds.
        # apply_palette() now re-themes the hint text alongside the axes.
        if self._hint_item is not None and self._empty_hint:
            self._hint_item.setColor(self._palette.foreground)

    # -- (C-1) the empty-plot hint ------------------------------------------------
    def set_empty_hint(self, text: str) -> None:
        """Show ``text`` centered in the panel until data is drawn.

        Pages call this with their concrete first step ("Load a dataset
        and press Train…").  The hint hides itself on the first
        scatter/line/bars/vlines call and re-appears after clear() —
        the panel never fakes data, it explains the next action.
        """
        self._empty_hint = text
        if not text:
            self._hide_hint()
            return
        if self._hint_item is None:
            # review 4.4: themed, not fixed grey — dark backgrounds get
            # the palette's foreground (the axes' own color)
            self._hint_item = pg.TextItem(
                text, color=self._palette.foreground, anchor=(0.5, 0.5)
            )
            self._plot.addItem(self._hint_item, ignoreBounds=True)
        else:
            self._hint_item.setText(text)
        self._hint_item.setPos(
            self._plot.getViewBox().viewRect().center().x(),
            self._plot.getViewBox().viewRect().center().y(),
        )
        self._hint_item.setVisible(True)

    def _hide_hint(self) -> None:
        if self._hint_item is not None:
            self._hint_item.setVisible(False)

    def _drawn_data(self) -> None:
        """Called by every draw method: real data displaces the hint."""
        if self._hint_item is not None and self._hint_item.isVisible():
            self._hint_item.setVisible(False)

    def resizeEvent(self, event) -> None:  # Qt signature
        """Pin the readout and re-fit the §23.6/§23.7/§23.8/§23.9 policies."""
        super().resizeEvent(event)
        self._apply_tick_density()
        self._fit_legend()
        self._apply_title_budget()
        self._position_readout()  # last: reads the freshly capped legend band
        # (C-1) keep the empty-state hint centered through resizes
        if self._hint_item is not None and self._hint_item.isVisible():
            vr = self._plot.getViewBox().viewRect()
            self._hint_item.setPos(vr.center().x(), vr.center().y())

    def showEvent(self, event) -> None:  # Qt signature
        super().showEvent(event)
        # §23.9: a pre-show resize() delivers no resize event when the
        # panel is shown at the same size, so the readout would keep
        # its pre-show (legacy) anchor while the legend band is already
        # live -- re-anchor once actually visible.
        self._position_readout()

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

    # -- §23.7 legend hygiene + §23.9 legend placement ------------------------
    def _fit_legend(self) -> None:
        """Re-fit the docked legend to the panel geometry (§23.7 + §23.9).

        A pyqtgraph legend is unbounded: long series names spill
        sideways over the data and dozens of entries stack past the
        plot's bottom edge -- and floating INSIDE the view box (the
        0.13.7 default, probed) it covers data points outright.  The
        panel docks it below the bottom axis (§23.9) and re-fits it
        after every data-adding call and on every geometry change:

        1. **label budget** -- each legend label wider than its
           per-column share of the view column's width (capped by
           :data:`_LEGEND_MAX_LABEL_PX`) is elided; the full text is
           kept behind the panel, so the budget follows the width BOTH
           ways -- a label elided on a narrow panel is restored
           verbatim when the panel widens again;
        2. **column escalation** -- entries beyond the band's row
           capacity split into columns, capped at
           :data:`_LEGEND_MAX_COLS`; the band itself is capped at
           :data:`_LEGEND_BAND_FRAC` of the panel height, so the
           docked legend never eats the plot;
        3. **hard caps** -- ``maximumHeight`` (the band) and
           ``maximumWidth`` (the view column) are re-set on every fit:
           the grid row hugs the legend's contents up to the band, and
           a wide legend can never widen the grid past the view.

        An empty legend (the pre-data state, or a freshly ``clear()``ed
        panel) hides and claims no row at all.  Display-only: the
        legend relabels never touch the §21 inspector's series keys --
        hover metadata, click-to-select and ``attach_metadata()`` keep
        resolving the full names.
        """
        legend = self._plot.plotItem.legend
        if legend is None:
            return
        if not legend.items:
            self._legend_full.clear()
            # §23.9: an empty legend claims no band -- the plot keeps
            # the panel's full height until real series land.
            legend.setMaximumHeight(0)
            legend.hide()
            legend.updateSize()
            self._plot.plotItem.layout.activate()
            self._position_readout()  # restore the legacy bottom anchor
            return
        legend.show()
        live_ids = {id(label) for _sample, label in legend.items}
        for stale in [k for k in self._legend_full if k not in live_ids]:
            del self._legend_full[stale]
        # §23.9 band budget: the docked row may claim at most a fixed
        # fraction of the panel height, floored so a tiny panel still
        # shows the minimum two rows.
        band = max(
            _LEGEND_MIN_ROWS * _LEGEND_ROW_PX,
            int(self.height() * _LEGEND_BAND_FRAC),
        )
        rows = max(_LEGEND_MIN_ROWS, band // _LEGEND_ROW_PX)
        cols = min(_LEGEND_MAX_COLS, max(1, math.ceil(len(legend.items) / rows)))
        if getattr(legend, "columnCount", 1) != cols:
            legend.setColumnCount(cols)
        # §23.9 label budget: the per-column share of the view column's
        # width (the docked legend no longer trades width against the
        # data -- it must simply fit the row it sits in), still capped
        # by the absolute ceiling.
        avail_w = (
            self.width()
            - int(self._plot.getAxis("left").width())
            - int(self._plot.getAxis("right").width())
        )
        fm = QFontMetrics(self.font())
        share = int(max(_LEGEND_MIN_LABEL_PX, avail_w // max(1, cols) - _LEGEND_SAMPLE_PX))
        budget = min(_LEGEND_MAX_LABEL_PX, share)
        for _sample, label in legend.items:
            full = self._legend_full.setdefault(id(label), label.text)
            elided = fm.elidedText(full, Qt.TextElideMode.ElideRight, budget)
            if elided != label.text:
                label.setText(elided or full)
        legend.updateSize()
        # §23.9 hard caps: the band never eats the plot, and the row
        # never widens the grid past the view column.
        legend.setMaximumHeight(band)
        legend.setMaximumWidth(max(80, avail_w))
        self._plot.plotItem.layout.activate()
        self._position_readout()  # the band moved: re-anchor above it

    # -- §23.8 title budget ----------------------------------------------------
    def _apply_title_budget(self) -> None:
        """Elide the rendered title to the panel's pixel width (§23.8).

        Pages compose long titles (method up front, computed facts at
        the end) and pyqtgraph renders them verbatim -- on a narrow
        panel they overflow.  The policy measures the title with the
        LIVE LabelItem font (probed: pyqtgraph 0.13.7 renders the
        default title at 9pt Sans Serif and never elides on its own),
        middle-elides to the view column's width (titles are centered
        and BOTH ends carry meaning), and keeps the full string
        verbatim on the panel ("_title_full" + the widget tooltip).
        The budget follows the width BOTH ways: a title elided narrow
        is restored verbatim once the full string fits again.  Only
        pyqtgraph's setTitle is touched when the rendered fit CHANGES
        (it rebuilds the LabelItem on every call).
        """
        title = self._title_full
        if not title:
            return
        title_item = self._plot.getPlotItem().layout.itemAt(0, 1)
        font = title_item.item.font() if title_item is not None else None
        if font is None or font.pointSizeF() <= 0:
            font = self.font()
            font.setPointSizeF(_TITLE_FALLBACK_PT)
        metrics = QFontMetrics(font)
        left = self._plot.getAxis("left")
        right = self._plot.getAxis("right")
        avail = self.width() - int(left.width()) - int(right.width()) - _TITLE_PAD_PX
        if avail <= 0 or metrics.horizontalAdvance(title) <= avail:
            rendered = title  # fits (or nothing laid out yet): verbatim
        else:
            rendered = metrics.elidedText(title, Qt.TextElideMode.ElideMiddle, avail) or title
        if rendered != self._title_rendered:
            self._title_rendered = rendered
            self._plot.setTitle(rendered)
        if self.toolTip() != title:
            self.setToolTip(title)

    def _position_readout(self) -> None:
        if self._plot.height() <= 0:
            return
        y = self.height() - self._readout.height() - 6
        # §23.9: with a live legend band under the plot the readout
        # floats at the view box's bottom edge instead of the panel's --
        # it must never cover legend entries (scene == widget coords:
        # the view is frameless and scrollless, probed).
        legend = self._plot.plotItem.legend
        if legend is not None and legend.isVisible() and legend.items:
            vb = self._plot.plotItem.vb
            bottom = self._plot.mapFromScene(vb.sceneBoundingRect().bottomLeft()).y()
            if bottom > 0:
                y = min(y, int(bottom) - self._readout.height() - 6)
        self._readout.move(6, max(0, y))
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
        # review 4.4: keep the empty-state hint readable after theme
        # switches (same color the axes use on this palette)
        if self._hint_item is not None:
            self._hint_item.setColor(palette.foreground)

    # -- plotting (real data only) ------------------------------------------
    def scatter(self, x: np.ndarray, y: np.ndarray, *, label: str = "", symbol: str = "o", size: int = 6, index: int = 0, meta: Callable[[int], dict[str, Any]] | None = None) -> None:
        pen = pg.mkPen(None)
        brush = pg.mkBrush(self._palette.series[index % len(self._palette.series)])
        # R29-b: a flagged (low-contrast) light-theme scatter takes a
        # distinct SHAPE per flagged series, not just the caller's
        # default — color never carries class identity alone (§5).
        if _needs_marker(self._palette,
                         self._palette.series[index % len(self._palette.series)]):
            symbol = _discipline_symbol(self._discipline_count)
            self._discipline_count += 1
        self._drawn_data()
        item = self._plot.plot(x, y, pen=pen, symbol=symbol, symbolSize=size, symbolBrush=brush, name=label or None)
        self._inspector.register_series(label, x, y, item, kind="scatter", meta=meta)
        self._fit_legend()

    def line(self, x: np.ndarray, y: np.ndarray, *, label: str = "", width: int = 2, index: int = 0, style: str = "solid", fill: bool = False, meta: Callable[[int], dict[str, Any]] | None = None) -> None:
        color = self._palette.series[index % len(self._palette.series)]
        pen = pg.mkPen(color, width=width)
        if style == "dash":
            pen.setStyle(Qt.PenStyle.DashLine)
        kwargs: dict[str, object] = {}
        if fill:  # area-under-the-curve shading (NCA profiles)
            fill_color = pg.mkColor(color)
            fill_color.setAlpha(70)
            kwargs["fillLevel"] = 0.0
            kwargs["brush"] = pg.mkBrush(fill_color)
        # part7 §5 / R29-b marker discipline: in LIGHT theme a series
        # whose color sits below 3:1 on white must carry a non-color
        # encoding — the line gains a marker at fixed intervals.
        if _needs_marker(self._palette, color):
            kwargs["symbol"] = _discipline_symbol(self._discipline_count)
            kwargs["symbolSize"] = 7
            kwargs["symbolBrush"] = pg.mkBrush(color)
            self._discipline_count += 1
        item = self._plot.plot(x, y, pen=pen, name=label or None, **kwargs)
        self._drawn_data()
        self._inspector.register_series(label, x, y, item, kind="line", meta=meta)
        self._fit_legend()

    def vlines(self, xs, *, label: str = "") -> None:
        """Vertical marker lines (dose times, event boundaries).

        Palette-aware dotted lines; removed by the next ``clear()``.
        """
        for x in xs:
            pen = pg.mkPen(self._palette.foreground, width=1, style=Qt.PenStyle.DotLine)
            self._plot.addLine(x=float(x), pen=pen)
        self._drawn_data()

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
        self._drawn_data()
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
        self._discipline_count = 0  # shape cycle restarts with the data
        self._plot.getAxis("bottom").setTicks(None)
        # (C-1) cleared panels re-explain themselves
        if self._empty_hint and self._hint_item is not None:
            self._hint_item.setVisible(True)
        # The inspector's hover decorations are lazy: plot.clear()
        # removed them along with the data, and nothing is re-added
        # until the next hover (see ChartInspector._ensure_decorations_added)
        # -- so the data-item list and the InfiniteLine census of a freshly
        # cleared plot stay exactly what the page's tests pin.
        self._inspector.reset()
        # §23.9: release the legend band for the freshly emptied legend
        # (pyqtgraph 0.13.7's PlotItem.clear() empties legend.items --
        # probed and pinned in the §23.7 suite).
        self._fit_legend()

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
            self._title_full = title
            self._apply_title_budget()

    def add_legend(self) -> None:
        """Dock a legend BELOW the plot (§23.9).

        pyqtgraph's default ``addLegend()`` floats the legend INSIDE the
        view box at a (30, 30) offset -- over the data (probed on
        0.13.7: the legend's scene rect always intersects the view
        box's).  The panel instead constructs the ``LegendItem``
        directly -- no ``size`` pin (a pinned size freezes
        ``updateSize()`` into a no-op, probed: the legacy
        ``size=(10, 10)`` call left the frame at 10×10 px with the
        entries rendering outside it) and no offset anchor -- and docks
        it in the layout row below the bottom axis, in the view column,
        hugging its contents (``AlignLeft | AlignTop``).  Hard
        ``maximumHeight``/``maximumWidth`` caps are re-set by
        ``_fit_legend()`` on every data-adding call and geometry change.
        """
        pi = self._plot.plotItem
        legend = pi.legend
        if legend is None:
            legend = pg.LegendItem(offset=None)
            pi.legend = legend
            # row 4 = the free grid row under the bottom axis; column 1
            # = the view column.  An explicit alignment keeps the grid
            # from stretching the legend to fill its cell (probed:
            # without it a stretch feedback loop hands the whole panel
            # to the legend row).
            pi.layout.addItem(
                legend, 4, 1, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
            )
            legend.setMaximumHeight(0)
            legend.hide()  # empty: the band starts collapsed
        self._fit_legend()

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
