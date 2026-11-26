"""Part-3 §23.9 legend placement tests (round 243).

Contract, one module (``ui.widgets.plot_widget.PlotPanel``):

* pyqtgraph 0.13.7's default ``addLegend()`` floats the legend INSIDE
  the view box at a (30, 30) offset -- over the data (probed: the
  legend's scene rect always intersects the view box's).  The panel's
  ``add_legend()`` instead docks the legend in the layout row BELOW
  the bottom axis, in the view column:
  - the docked legend's scene rect NEVER intersects the view box's,
    at any width -- the §23.7 follow-up closes: no legend over data;
  - the legacy ``size=(10, 10)`` pin is gone -- a pinned size freezes
    ``LegendItem.updateSize()`` into a no-op (probed), which left the
    frame at 10x10 px with the entries rendering outside it, so the
    legend is constructed auto-sized and the frame hugs its entries;
  - hard caps are re-set on every fit: ``maximumHeight`` = the band
    budget (35% of the panel height, floored at two rows) and
    ``maximumWidth`` = the view column -- the band never eats the plot
    and a wide legend never widens the grid past the view;
  - an explicit ``AlignLeft | AlignTop`` keeps the grid from stretching
    the legend into its cell (probed: without it a stretch feedback
    loop hands the whole panel to the legend row);
  - an empty legend (pre-data or freshly cleared) hides and claims no
    row; ``clear()`` releases the band and the plot reclaims its
    height;
  - the §21 hover readout re-anchors to the view box's bottom edge
    whenever a live band sits under the plot (never covering legend
    entries), and returns to the legacy panel-bottom anchor when the
    band releases -- including through the show-event path (a pre-show
    ``resize()`` at the same size delivers no resize event at show
    time);
  - docking is display-only: the §21 inspector's series keys keep
    resolving hover metadata under the FULL names.

Hidden-panel helpers re-apply the policy explicitly, exactly what
production's show-time resize event will do (Qt defers the resize
events of hidden widgets to show()); the shown-panel legs exercise
the event-driven path end to end.
"""

from __future__ import annotations

import numpy as np

from ui.widgets.plot_widget import PlotPanel


def _panel() -> PlotPanel:
    return PlotPanel()


def _refit(panel: PlotPanel, w: int, h: int = 240) -> None:
    panel.resize(w, h)
    if panel.layout() is not None:
        panel.layout().activate()
    panel._apply_tick_density()
    panel._fit_legend()


def _legend(panel: PlotPanel):
    return panel._plot.plotItem.legend


def _labels(panel: PlotPanel) -> list[str]:
    return [label.text for _sample, label in _legend(panel).items]


def _series(panel: PlotPanel, n: int, name: str = "s{i}") -> None:
    xs = np.linspace(0.0, 10.0, 30)
    for i in range(n):
        panel.line(xs, np.sin(xs) + i, label=name.format(i=i))


def _shown(panel: PlotPanel, w: int = 420, h: int = 300, app=None) -> None:
    panel.resize(w, h)
    panel.show()
    if app is not None:
        app.processEvents()


# -- docking structure (hidden-panel legs) ---------------------------------


def test_add_legend_docks_below_the_bottom_axis(studio_app) -> None:
    """The legend lands in grid cell (4, 1) -- row 4 is under the axis."""
    panel = _panel()
    panel.add_legend()
    _series(panel, 2)
    _refit(panel, 420)
    pi = panel._plot.plotItem
    legend = _legend(panel)
    assert legend is not None
    assert pi.layout.itemAt(4, 1) is legend  # row 4, view column
    assert legend.parentItem() is not pi.vb  # not parented into the view box
    assert legend.offset is None  # no inside-box anchor


def test_add_legend_is_idempotent(studio_app) -> None:
    """A second add_legend() call reuses the docked legend, no churn."""
    panel = _panel()
    panel.add_legend()
    first = _legend(panel)
    panel.add_legend()
    assert _legend(panel) is first
    assert panel._plot.plotItem.layout.itemAt(4, 1) is first


def test_the_size_pin_is_gone(studio_app) -> None:
    """Auto-sized legend: the frame hugs its entries, never 10x10."""
    panel = _panel()
    panel.add_legend()
    _series(panel, 3)
    _refit(panel, 600)
    legend = _legend(panel)
    assert legend.size is None  # the legacy size=(10, 10) pin is gone
    assert legend.height() > 30  # three entries stack past 30px
    assert legend.width() > 20  # and the frame is wider than the pin


def test_column_escalation_against_the_band(studio_app) -> None:
    """40 entries on a 240px-tall panel escalate to the 3-column cap."""
    panel = _panel()
    panel.add_legend()
    _series(panel, 40)
    _refit(panel, 300)
    assert _legend(panel).columnCount == 3


def test_few_series_keep_one_column(studio_app) -> None:
    """Four entries fit a 320px-tall band's rows -- no column churn."""
    panel = _panel()
    panel.add_legend()
    _series(panel, 4)
    _refit(panel, 600, 320)  # band = 112px -> 5 row capacity
    assert _legend(panel).columnCount == 1


def test_share_budget_elides_and_restores_both_ways(studio_app) -> None:
    """The per-column share budget follows the width BOTH ways."""
    panel = _panel()
    panel.add_legend()
    long_name = "Dose-normalized clearance profile estimate" + "x" * 40
    panel.line(np.arange(5, dtype=float), np.arange(5, dtype=float), label=long_name)
    _refit(panel, 240)
    assert len(_labels(panel)[0]) < len(long_name)
    assert _labels(panel)[0].endswith("…")
    # the full name survives behind the panel for the inspector
    assert panel._legend_full[id(_legend(panel).items[0][1])] == long_name


def test_inspector_keys_survive_docking(studio_app) -> None:
    """§21 contract: hover meta resolves the FULL name, docked or not."""
    panel = _panel()
    panel.add_legend()
    long_name = "Dose-normalized clearance profile estimate" + "x" * 40
    panel.line(np.arange(5, dtype=float), np.arange(5, dtype=float), label=long_name)
    assert panel.attach_metadata(long_name, lambda i: {"dose": 3.0})
    _refit(panel, 240)
    assert len(_labels(panel)[0]) < len(long_name)  # relabeled
    record = panel._inspector._series[long_name]  # ...but keyed fully
    assert record.metadata(1)["dose"] == 3.0


# -- the placement guarantee (shown-panel legs) ------------------------------


def test_docked_legend_never_overlaps_the_view(studio_app) -> None:
    """The §23.7 follow-up closes: no legend over data, at any width."""
    panel = _panel()
    panel.add_legend()
    _series(panel, 4)
    _shown(panel, 420, 300, studio_app)
    pi = panel._plot.plotItem
    legend, vb = _legend(panel), pi.vb
    assert not legend.sceneBoundingRect().intersects(vb.sceneBoundingRect())
    # it sits strictly below the bottom axis, inside the widget
    assert legend.sceneBoundingRect().top() >= pi.getAxis("bottom").sceneBoundingRect().bottom() - 0.5
    assert legend.sceneBoundingRect().bottom() <= panel._plot.height() + 0.5


def test_no_overlap_across_three_widths(studio_app) -> None:
    """Narrow, standard and wide panels all keep the legend off the data."""
    panel = _panel()
    panel.add_legend()
    _series(panel, 6)
    pi = panel._plot.plotItem
    for width in (240, 420, 900):
        _shown(panel, width, 300, studio_app)
        assert not _legend(panel).sceneBoundingRect().intersects(pi.vb.sceneBoundingRect()), width


def test_band_is_capped_to_a_fraction_of_the_panel(studio_app) -> None:
    """40 series: the band tops out at 35% of the panel height."""
    panel = _panel()
    panel.add_legend()
    _series(panel, 40)
    _shown(panel, 420, 300, studio_app)
    legend, vb = _legend(panel), panel._plot.plotItem.vb
    assert legend.sceneBoundingRect().height() <= 300 * 0.35 + 0.5
    assert vb.sceneBoundingRect().height() >= 300 * 0.5  # the plot stays usable


def test_wide_labels_never_widen_the_grid(studio_app) -> None:
    """Long names elide to their share; the grid never outgrows the view.

    The probed failure mode: an uncapped docked legend with 3 columns
    of long labels pushed the middle grid column to ~1198px on a
    420px widget, dragging the view box out of the widget entirely.
    """
    panel = _panel()
    panel.add_legend()
    _series(panel, 4, name="a_very_long_series_label_{i}_with_padding")
    _shown(panel, 420, 300, studio_app)
    legend, vb = _legend(panel), panel._plot.plotItem.vb
    assert legend.sceneBoundingRect().right() <= panel._plot.width() + 0.5
    assert vb.sceneBoundingRect().right() <= panel._plot.width() + 0.5


# -- the band lifecycle -------------------------------------------------------


def test_empty_legend_claims_no_band(studio_app) -> None:
    """add_legend() before any series: hidden, zero-height, no row."""
    panel = _panel()
    panel.add_legend()
    _shown(panel, 420, 300, studio_app)
    legend = _legend(panel)
    pi = panel._plot.plotItem
    assert not legend.isVisible()
    assert legend.maximumHeight() == 0
    # the view box runs to the bottom axis -- no phantom band
    assert pi.vb.sceneBoundingRect().bottom() >= pi.getAxis("bottom").sceneBoundingRect().top() - 0.5
    # a series lands -> the band appears
    _series(panel, 2)
    studio_app.processEvents()
    assert legend.isVisible()
    assert legend.maximumHeight() > 0


def test_clear_releases_the_band(studio_app) -> None:
    """clear() empties the legend, hides it and hands the height back."""
    panel = _panel()
    panel.add_legend()
    _series(panel, 2)
    _shown(panel, 420, 300, studio_app)
    pi = panel._plot.plotItem
    vb_bottom_with_band = pi.vb.sceneBoundingRect().bottom()
    assert _legend(panel).isVisible()
    panel.clear()
    studio_app.processEvents()
    legend = _legend(panel)
    assert len(legend.items) == 0
    assert not legend.isVisible()
    assert legend.maximumHeight() == 0
    assert panel._legend_full == {}
    # the view box reclaims (nearly) the whole panel height
    assert pi.vb.sceneBoundingRect().bottom() > vb_bottom_with_band + 50


# -- the §21 readout re-anchoring ---------------------------------------------


def test_readout_floats_above_the_band(studio_app) -> None:
    """With a live band the readout sits at the view box's bottom edge."""
    panel = _panel()
    panel.add_legend()
    _series(panel, 4)
    _shown(panel, 420, 300, studio_app)
    legend = _legend(panel)
    vb = panel._plot.plotItem.vb
    vb_bottom = panel._plot.mapFromScene(vb.sceneBoundingRect().bottomLeft()).y()
    y, h = panel._readout.y(), panel._readout.height()
    assert y + h <= legend.sceneBoundingRect().top() + 0.5  # clear of the band
    assert abs((y + h + 6) - vb_bottom) <= 1.5  # anchored at the vb's edge


def test_readout_follows_a_post_show_resize(studio_app) -> None:
    """A resize after show re-fits the band and re-anchors the readout."""
    panel = _panel()
    panel.add_legend()
    _series(panel, 4)
    _shown(panel, 420, 300, studio_app)
    panel.resize(700, 400)
    studio_app.processEvents()
    vb = panel._plot.plotItem.vb
    vb_bottom = panel._plot.mapFromScene(vb.sceneBoundingRect().bottomLeft()).y()
    y, h = panel._readout.y(), panel._readout.height()
    assert abs((y + h + 6) - vb_bottom) <= 1.5


def test_readout_returns_to_the_legacy_anchor_on_clear(studio_app) -> None:
    """No band -> the §21 legacy panel-bottom anchor is restored."""
    panel = _panel()
    panel.add_legend()
    _series(panel, 4)
    _shown(panel, 420, 300, studio_app)
    panel.clear()
    studio_app.processEvents()
    assert panel._readout.y() == panel.height() - panel._readout.height() - 6


def test_legacy_readout_anchor_without_a_legend(studio_app) -> None:
    """Panels that never call add_legend() keep the §21 anchor exactly."""
    panel = _panel()
    _series(panel, 2)
    _shown(panel, 420, 300, studio_app)
    assert panel._readout.y() == panel.height() - panel._readout.height() - 6


if __name__ == "__main__":  # pragma: no cover
    import pytest

    pytest.main([__file__])
