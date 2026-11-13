"""Marker discipline (part7 §5, R29-b).

In LIGHT theme a series whose color sits below 3:1 on white must
carry a non-color encoding: lines gain symbols, scatters take a
distinct shape per index.  Dark themes flag nothing.  The set of
flagged colors is ui.design_tokens.MARKER_DISCIPLINE_LIGHT — the
same data the token tests pin.
"""

from __future__ import annotations

import numpy as np

from ui.design_tokens import MARKER_DISCIPLINE_LIGHT
from ui.theme_manager import ChartPalette
from ui.widgets.plot_widget import (
    _DISCIPLINE_SYMBOLS,
    _discipline_symbol,
    _needs_marker,
    PlotPanel,
)


def _studio_light() -> ChartPalette:
    # exactly what ThemeManager.apply("studio", "light") emits
    from ui.theme_manager import _palette_for

    return _palette_for("studio", "light")


def _studio_dark() -> ChartPalette:
    from ui.theme_manager import _palette_for

    return _palette_for("studio", "dark")


def test_flagged_colors_are_exactly_the_light_set() -> None:
    palette = _studio_light()
    flagged = {c for c in palette.series if _needs_marker(palette, c)}
    assert flagged == set(MARKER_DISCIPLINE_LIGHT)
    assert "#0072B2" not in flagged      # Blue holds on white
    assert "#F0E442" in flagged          # Yellow does not


def test_dark_theme_flags_nothing() -> None:
    palette = _studio_dark()
    for color in palette.series:
        assert not _needs_marker(palette, color), color


def test_foreign_light_palette_has_no_discipline() -> None:
    """qt-material light / native light use the OLD default palette —
    discipline only guards the studio (Okabe-Ito) series."""
    from ui.theme_manager import _palette_for

    palette = _palette_for("qt-material", "light_teal.xml")
    for color in palette.series:
        assert not _needs_marker(palette, color)


def test_line_gains_symbol_exactly_when_flagged(studio_app) -> None:
    panel = PlotPanel()
    panel.apply_palette(_studio_light())
    x = np.linspace(0.0, 1.0, 12)

    # index 0 → Okabe-Ito Blue (#0072B2): not flagged, no symbol
    panel.line(x, x, label="blue", index=0)
    blue = panel._plot.listDataItems()[-1]
    assert blue.opts.get("symbol") is None

    # index 6 → Okabe-Ito Yellow (#F0E442): flagged → first shape
    panel.line(x, x, label="yellow", index=6)
    yellow = panel._plot.listDataItems()[-1]
    assert yellow.opts.get("symbol") == _DISCIPLINE_SYMBOLS[0]
    assert yellow.opts.get("symbolBrush") is not None


def test_dark_line_never_gains_discipline_symbol(studio_app) -> None:
    panel = PlotPanel()
    panel.apply_palette(_studio_dark())
    x = np.linspace(0.0, 1.0, 12)
    for index in range(8):
        panel.line(x, x, label=f"s{index}", index=index)
        item = panel._plot.listDataItems()[-1]
        assert item.opts.get("symbol") is None, index


def test_scatter_takes_shape_per_flagged_series(studio_app) -> None:
    panel = PlotPanel()
    panel.apply_palette(_studio_light())
    x = np.arange(5, dtype=float)

    panel.scatter(x, x, label="blue", index=0)     # not flagged
    assert panel._plot.listDataItems()[-1].opts["symbol"] == "o"

    panel.scatter(x, x, label="yellow", index=6)   # flagged → shape 1
    item = panel._plot.listDataItems()[-1]
    assert item.opts["symbol"] in _DISCIPLINE_SYMBOLS
    assert item.opts["symbol"] != "o"

    panel.scatter(x, x, label="orange", index=1)   # flagged → shape 2
    other = panel._plot.listDataItems()[-1]
    assert other.opts["symbol"] in _DISCIPLINE_SYMBOLS
    # the counter (not the raw index) keys the cycle: indices 6 and 1
    # are DIFFERENT flagged series, so their shapes must differ
    assert other.opts["symbol"] != item.opts["symbol"]


def test_discipline_counter_resets_on_clear(studio_app) -> None:
    panel = PlotPanel()
    panel.apply_palette(_studio_light())
    x = np.arange(5, dtype=float)
    panel.scatter(x, x, label="yellow", index=6)
    first = panel._plot.listDataItems()[-1].opts["symbol"]
    panel.clear()
    panel.scatter(x, x, label="orange", index=1)
    second = panel._plot.listDataItems()[-1].opts["symbol"]
    # after clear() the shape cycle restarts — same slot, fresh plot
    assert second == first


def test_symbol_cycle_is_stable() -> None:
    assert _discipline_symbol(0) == _discipline_symbol(5)
    assert _discipline_symbol(0) != _discipline_symbol(1)


def test_discipline_symbols_are_pyqtgraph_names() -> None:
    # t triangle, s square, d diamond, p pentagon, x cross
    assert set(_DISCIPLINE_SYMBOLS) == {"t", "s", "d", "p", "x"}
