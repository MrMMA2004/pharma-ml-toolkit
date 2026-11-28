"""Round 242 -- Part-3 §23.8: the plot-title pixel budget.

A plot title is the last unbounded string on a ``PlotPanel``: pages
compose long titles (method up front, computed facts at the end) and
pyqtgraph renders them verbatim -- on a narrow panel they overflow.
The policy measures the title with the LIVE LabelItem font (probed:
pyqtgraph 0.13.7 renders the default title at 9pt Sans Serif and
never elides on its own), middle-elides to the view column's width
(titles are centered and BOTH ends carry meaning), and keeps the full
string verbatim on the panel (``_title_full`` + the widget tooltip).
The budget follows the width BOTH ways: a title elided narrow is
restored verbatim once the full string fits again, and setTitle is
only touched when the rendered fit actually changes.
"""

from __future__ import annotations

import pytest
from ui.widgets.plot_widget import PlotPanel

LONG_TITLE = (
    "UMAP — trustworthiness 0.87 · a=1.200 b=0.300 · seed 42 · "
    "500 rows embedded in 3.2s · trustworthiness recheck pass two"
)
SHORT_TITLE = "parity"


@pytest.fixture()
def panel(studio_app):
    panel = PlotPanel(title=LONG_TITLE)
    yield panel
    panel.close()


def _title_item(panel: PlotPanel):
    return panel._plot.getPlotItem().layout.itemAt(0, 1)


class TestTitleBudget:
    def test_narrow_panel_elides_the_rendered_title(self, panel):
        panel.resize(300, 200)
        panel._apply_title_budget()
        rendered = panel._title_rendered
        assert rendered is not None
        assert rendered != LONG_TITLE  # actually elided
        assert "…" in rendered
        item = _title_item(panel)
        assert item.text == rendered  # pyqtgraph shows exactly the policy output

    def test_full_title_survives_verbatim(self, panel):
        panel.resize(300, 200)
        panel._apply_title_budget()
        assert panel._title_full == LONG_TITLE
        assert panel.toolTip() == LONG_TITLE  # hover discloses the full string

    def test_middle_elide_keeps_both_ends(self, panel):
        panel.resize(300, 200)
        panel._apply_title_budget()
        rendered = panel._title_rendered
        # titles are centered and both ends carry meaning: the method
        # up front, the computed facts at the end
        assert rendered.startswith(LONG_TITLE[:5])
        assert rendered.endswith(LONG_TITLE[-5:])

    def test_wide_panel_renders_verbatim(self, panel):
        panel.resize(900, 300)
        panel._apply_title_budget()
        assert panel._title_rendered == LONG_TITLE
        assert _title_item(panel).text == LONG_TITLE

    def test_budget_follows_the_width_both_ways(self, panel):
        panel.resize(300, 200)
        panel._apply_title_budget()
        assert panel._title_rendered != LONG_TITLE
        panel.resize(900, 300)  # widen -> restore
        panel._apply_title_budget()
        assert panel._title_rendered == LONG_TITLE
        panel.resize(300, 200)  # narrow again -> elide again
        panel._apply_title_budget()
        assert panel._title_rendered != LONG_TITLE

    def test_settitle_skipped_when_the_fit_is_unchanged(self, panel):
        panel.resize(900, 300)
        panel._apply_title_budget()
        item = _title_item(panel)
        panel._apply_title_budget()
        panel._apply_title_budget()
        assert _title_item(panel) is item  # same LabelItem: setTitle not rebuilt

    def test_set_labels_updates_the_full_title_and_refits(self, panel):
        panel.resize(300, 200)
        panel.set_labels(x="x", y="y", title=LONG_TITLE)
        assert panel._title_full == LONG_TITLE
        assert panel.toolTip() == LONG_TITLE
        assert panel._title_rendered != LONG_TITLE  # narrow: elided
        panel.resize(900, 300)
        panel._apply_title_budget()
        assert panel._title_rendered == LONG_TITLE

    def test_titleless_panel_adopts_one_via_set_labels(self, studio_app):
        panel = PlotPanel()
        try:
            assert (_title_item(panel).text or "") == ""  # no title text yet
            panel.resize(300, 200)
            panel.set_labels(x="x", y="y", title=LONG_TITLE)
            assert panel._title_rendered != LONG_TITLE
            assert _title_item(panel).text == panel._title_rendered
        finally:
            panel.close()

    def test_clear_keeps_the_title(self, panel):
        panel.resize(300, 200)
        panel._apply_title_budget()
        rendered_before = panel._title_rendered
        panel.clear()
        panel._apply_title_budget()
        assert panel._title_full == LONG_TITLE  # the title is the panel's, not the data's
        assert panel._title_rendered == rendered_before

    def test_empty_title_is_a_noop(self, studio_app):
        panel = PlotPanel()
        try:
            panel._apply_title_budget()  # must not raise
            assert panel._title_rendered is None
            assert (_title_item(panel).text or "") == ""  # setTitle never called
            assert panel.toolTip() == ""
        finally:
            panel.close()

    def test_short_title_fits_verbatim_even_narrow(self, studio_app):
        panel = PlotPanel(title=SHORT_TITLE)
        try:
            panel.resize(300, 200)
            panel._apply_title_budget()
            assert panel._title_rendered == SHORT_TITLE
        finally:
            panel.close()

    def test_resize_event_drives_the_refit(self, panel, studio_app):
        # the shown-panel end-to-end leg: a real resize event, not the
        # helper called by hand
        panel.resize(900, 300)
        panel.show()
        studio_app.processEvents()
        assert panel._title_rendered == LONG_TITLE
        panel.resize(240, 200)
        studio_app.processEvents()
        assert panel._title_rendered != LONG_TITLE
        assert "…" in panel._title_rendered
        panel.close()
