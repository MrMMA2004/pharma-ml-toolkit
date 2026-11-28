"""Part-3 §23.7 legend hygiene tests (round 239; the budget and
escalation policies were re-targeted to the §23.9 docked row by
round 243 -- the placement itself is pinned in the §23.9 suite,
``test_plot_legend_placement_243.py``).

Contract, one module (``ui.widgets.plot_widget.PlotPanel``):

* a pyqtgraph legend is unbounded -- long series names spill sideways
  over the data and dozens of entries stack past the plot's bottom
  edge -- so the panel re-fits the legend after every data-adding
  call and on every geometry change:
  - **label budget**: each legend label wider than its per-column
    share of the view column's width (capped by
    :data:`_LEGEND_MAX_LABEL_PX`) is elided;
    the full text is kept behind the panel, so the budget follows the
    width BOTH ways (a label elided on a narrow panel is restored
    verbatim when the panel widens -- unless the full text itself
    exceeds the absolute :data:`_LEGEND_MAX_LABEL_PX` ceiling, which
    no panel width can lift);
  - **column escalation**: entries beyond the vertical capacity split
    into columns, capped at three;
* the id-keyed full-text map is pruned against the live legend items
  on every fit, so ``clear()``/replot cycles never leak stale ids;
* the relabeling is display-only -- the §21 inspector's series keys
  keep resolving hover metadata under the FULL names;
* pyqtgraph 0.13.7's ``PlotItem.clear()`` empties the legend (probed
  before implementing) -- pinned so a future pyqtgraph bump that
  starts leaking stale entries fails loudly here.

Hidden-panel helpers re-apply the policy explicitly, exactly what
production's show-time resize event will do (Qt defers the resize
events of hidden widgets to show() -- the event-driven path is
covered by the shown-panel leg in the §23.6 suite).
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


def test_long_labels_elided_on_a_narrow_panel(studio_app) -> None:
    """An 80-char series name cannot spill sideways at 280px."""
    panel = _panel()
    panel.add_legend()
    long_name = "Dose-normalized clearance profile estimate" + "x" * 40
    panel.line(np.arange(5, dtype=float), np.arange(5, dtype=float), label=long_name)
    _refit(panel, 280)
    texts = _labels(panel)
    assert len(texts) == 1
    assert len(texts[0]) < len(long_name)
    assert texts[0].endswith("…")


def test_labels_restore_verbatim_when_widened(studio_app) -> None:
    """The budget follows the width both ways for sub-ceiling names."""
    panel = _panel()
    panel.add_legend()
    mid_name = "Dose-normalized CL estimate"  # fits the 220px ceiling
    panel.line(np.arange(5, dtype=float), np.arange(5, dtype=float), label=mid_name)
    # 200px: the per-column share (~124px) elides the name; the §23.7
    # era's 45%-of-panel budget already elided it at 280px, but the
    # docked-row share is wider, so the both-ways leg narrows further.
    _refit(panel, 200)
    assert len(_labels(panel)[0]) < len(mid_name)
    _refit(panel, 1000)
    assert _labels(panel)[0] == mid_name


def test_labels_over_the_ceiling_stay_elided(studio_app) -> None:
    """No panel width lifts the absolute label cap for huge names."""
    panel = _panel()
    panel.add_legend()
    huge_name = "Dose-normalized clearance profile estimate" + "x" * 40
    panel.line(np.arange(5, dtype=float), np.arange(5, dtype=float), label=huge_name)
    _refit(panel, 1000)
    assert len(_labels(panel)[0]) < len(huge_name)
    assert _panel_legend_full(panel)[id(_legend(panel).items[0][1])] == huge_name


def _panel_legend_full(panel: PlotPanel) -> dict[int, str]:
    return panel._legend_full


def test_short_labels_untouched(studio_app) -> None:
    """Short names on a wide-enough panel stay verbatim."""
    panel = _panel()
    panel.add_legend()
    for name in ("CL", "Vd", "ka"):
        panel.scatter(np.arange(5, dtype=float), np.arange(5, dtype=float), label=name)
    _refit(panel, 600)
    assert _labels(panel) == ["CL", "Vd", "ka"]


def test_column_escalation_on_many_series(studio_app) -> None:
    """40 entries on a 240px-tall panel stack past the edge -> columns."""
    panel = _panel()
    panel.add_legend()
    for i in range(40):
        panel.scatter(np.arange(3, dtype=float), np.arange(3, dtype=float), label=f"s{i}")
    _refit(panel, 300)
    assert _legend(panel).columnCount == 3  # the policy cap


def test_few_series_keep_one_column(studio_app) -> None:
    """Three entries fit vertically -- no column churn."""
    panel = _panel()
    panel.add_legend()
    for i in range(3):
        panel.scatter(np.arange(3, dtype=float), np.arange(3, dtype=float), label=f"s{i}")
    _refit(panel, 600)
    assert _legend(panel).columnCount == 1


def test_no_legend_panel_is_a_noop(studio_app) -> None:
    """Panels without add_legend() never grow a legend nor crash."""
    panel = _panel()
    panel.scatter(np.arange(5, dtype=float), np.arange(5, dtype=float), label="orphan")
    _refit(panel, 300)
    assert _legend(panel) is None
    assert panel._legend_full == {}


def test_clear_empties_the_legend_and_the_map(studio_app) -> None:
    """pyqtgraph 0.13.7 clear() empties the legend -- pinned."""
    panel = _panel()
    panel.add_legend()
    panel.scatter(np.arange(5, dtype=float), np.arange(5, dtype=float), label="A")
    panel.line(np.arange(5, dtype=float), np.arange(5, dtype=float) * 2, label="B")
    assert len(_legend(panel).items) == 2
    panel.clear()
    panel._fit_legend()
    assert len(_legend(panel).items) == 0
    assert panel._legend_full == {}


def test_replot_prunes_stale_ids(studio_app) -> None:
    """A clear/replot cycle never leaks stale id keys into the map."""
    panel = _panel()
    panel.add_legend()
    panel.scatter(np.arange(5, dtype=float), np.arange(5, dtype=float), label="A")
    panel.clear()
    panel.scatter(np.arange(5, dtype=float), np.arange(5, dtype=float), label="B")
    panel._fit_legend()
    assert list(panel._legend_full.values()) == ["B"]


def test_inspector_keys_survive_the_relabel(studio_app) -> None:
    """§21 contract: hover meta resolves the FULL name after elision."""
    panel = _panel()
    panel.add_legend()
    long_name = "Dose-normalized clearance profile estimate" + "x" * 40
    panel.line(np.arange(5, dtype=float), np.arange(5, dtype=float), label=long_name)
    assert panel.attach_metadata(long_name, lambda i: {"dose": 3.0})
    _refit(panel, 280)
    assert len(_labels(panel)[0]) < len(long_name)  # relabeled
    record = panel._inspector._series[long_name]  # ...but keyed by the full name
    assert record.metadata(1)["dose"] == 3.0


def test_fit_is_idempotent(studio_app) -> None:
    """A second fit changes nothing -- no churn, no re-elision drift."""
    panel = _panel()
    panel.add_legend()
    for i in range(40):
        panel.scatter(np.arange(3, dtype=float), np.arange(3, dtype=float), label=f"s{i}")
    _refit(panel, 300)
    before = (list(_labels(panel)), _legend(panel).columnCount)
    panel._fit_legend()
    after = (list(_labels(panel)), _legend(panel).columnCount)
    assert before == after


if __name__ == "__main__":  # pragma: no cover
    import pytest

    pytest.main([__file__])
