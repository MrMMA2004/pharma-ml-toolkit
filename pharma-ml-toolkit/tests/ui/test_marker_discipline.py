"""Marker discipline (Mineral §12, R29-b).

A series whose color cannot carry its identity alone — below 3:1 on
the plot surface (light) or converging with another series (dark's
green/sky mint pair) — must carry a non-color encoding: lines gain
symbols, scatters take a distinct shape per index.  The set of
flagged colors is ui.design_tokens.MARKER_DISCIPLINE_* — the same
data the token tests pin.
"""

from __future__ import annotations

import numpy as np

from ui.design_tokens import (
    MARKER_DISCIPLINE_DARK,
    MARKER_DISCIPLINE_LIGHT,
)
from ui.theme_manager import ChartPalette
from ui.widgets.plot_widget import (
    _DISCIPLINE_SYMBOLS,
    PlotPanel,
    _discipline_symbol,
    _needs_marker,
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
    assert "#E69F00" in flagged          # Yellow does not
    assert "#56B4E9" in flagged          # Sky does not


def test_dark_theme_flags_the_mint_pair() -> None:
    """The token-export companions for green (#63D3AC) and sky
    (#8DD6AC) converge to near-identical mint tints — the non-color
    encoding tells that pair apart, exactly as §12 demands."""
    palette = _studio_dark()
    flagged = {c for c in palette.series if _needs_marker(palette, c)}
    assert flagged == set(MARKER_DISCIPLINE_DARK)
    assert "#63D3AC" in flagged and "#8DD6AC" in flagged
    assert "#69BDE8" not in flagged      # blue companion holds


def test_foreign_light_palette_has_no_discipline() -> None:
    """qt-material light / native light use their own palettes —
    discipline only guards the studio (Mineral) series."""
    from ui.theme_manager import _palette_for

    palette = _palette_for("qt-material", "light_teal.xml")
    for color in palette.series:
        assert not _needs_marker(palette, color)


def test_line_gains_symbol_exactly_when_flagged(studio_app) -> None:
    panel = PlotPanel()
    panel.apply_palette(_studio_light())
    x = np.linspace(0.0, 1.0, 12)

    # index 0 → Mineral Blue (#0072B2): not flagged, no symbol
    panel.line(x, x, label="blue", index=0)
    blue = panel._plot.listDataItems()[-1]
    assert blue.opts.get("symbol") is None

    # index 4 → Mineral Yellow (#E69F00): flagged → first shape
    panel.line(x, x, label="yellow", index=4)
    yellow = panel._plot.listDataItems()[-1]
    assert yellow.opts.get("symbol") == _DISCIPLINE_SYMBOLS[0]
    assert yellow.opts.get("symbolBrush") is not None


def test_dark_line_gains_symbol_only_for_the_mint_pair(studio_app) -> None:
    panel = PlotPanel()
    panel.apply_palette(_studio_dark())
    x = np.linspace(0.0, 1.0, 12)
    for index in range(7):
        panel.line(x, x, label=f"s{index}", index=index)
        item = panel._plot.listDataItems()[-1]
        if index in (2, 5):  # green/sky companions: flagged
            assert item.opts.get("symbol") is not None, index
        else:
            assert item.opts.get("symbol") is None, index


def test_scatter_takes_shape_per_flagged_series(studio_app) -> None:
    panel = PlotPanel()
    panel.apply_palette(_studio_light())
    x = np.arange(5, dtype=float)

    panel.scatter(x, x, label="blue", index=0)     # not flagged
    assert panel._plot.listDataItems()[-1].opts["symbol"] == "o"

    panel.scatter(x, x, label="yellow", index=4)   # flagged → shape 1
    item = panel._plot.listDataItems()[-1]
    assert item.opts["symbol"] in _DISCIPLINE_SYMBOLS
    assert item.opts["symbol"] != "o"

    panel.scatter(x, x, label="sky", index=5)      # flagged → shape 2
    other = panel._plot.listDataItems()[-1]
    assert other.opts["symbol"] in _DISCIPLINE_SYMBOLS
    # the counter (not the raw index) keys the cycle: indices 4 and 5
    # are DIFFERENT flagged series, so their shapes must differ
    assert other.opts["symbol"] != item.opts["symbol"]


def test_discipline_counter_resets_on_clear(studio_app) -> None:
    panel = PlotPanel()
    panel.apply_palette(_studio_light())
    x = np.arange(5, dtype=float)
    panel.scatter(x, x, label="yellow", index=4)
    first = panel._plot.listDataItems()[-1].opts["symbol"]
    panel.clear()
    panel.scatter(x, x, label="sky", index=5)
    second = panel._plot.listDataItems()[-1].opts["symbol"]
    # after clear() the shape cycle restarts — same slot, fresh plot
    assert second == first


def test_symbol_cycle_is_stable() -> None:
    assert _discipline_symbol(0) == _discipline_symbol(5)
    assert _discipline_symbol(0) != _discipline_symbol(1)


def test_discipline_symbols_are_pyqtgraph_names() -> None:
    # t triangle, s square, d diamond, p pentagon, x cross
    assert set(_DISCIPLINE_SYMBOLS) == {"t", "s", "d", "p", "x"}
