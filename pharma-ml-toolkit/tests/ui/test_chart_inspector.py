"""Round 232 -- Part-3 §21 slice 1: the ChartInspector hover layer.

Every :class:`~ui.widgets.plot_widget.PlotPanel` now carries a
:class:`~ui.widgets.chart_inspector.ChartInspector`: crosshair,
nearest-real-point snapping, a live readout with EXACT values (§21.3),
cached hover metadata (§21.6), per-series visibility toggles and
click-to-select.  These tests drive the engine programmatically
(``inspect_at`` and stubbed scene geometry) so they run offscreen.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QPointF, QRectF

from ui.widgets.plot_widget import PlotPanel


@pytest.fixture()
def panel(studio_app):
    panel = PlotPanel()
    yield panel
    panel.close()


def _stub_scene_to_view(panel, monkeypatch):
    """Make scene<->view mapping identity and the scene rect infinite,
    so programmatic mouse events land wherever we say they do."""
    vb = panel.widget().getPlotItem().vb
    monkeypatch.setattr(vb, "sceneBoundingRect", lambda: QRectF(-1e9, -1e9, 2e9, 2e9))
    monkeypatch.setattr(vb, "mapSceneToView", lambda p: QPointF(p.x(), p.y()))
    return vb


class TestHoverReadout:
    def test_readout_shows_real_values(self, panel):
        xs = np.array([0.0, 1.0, 2.0, 3.0])
        ys = np.array([0.0, 10.0, 40.0, 90.0])
        panel.line(xs, ys, label="parabola")
        snapped = panel._inspector.inspect_at((2.0, 40.0))
        assert snapped == ("parabola", 2)
        text = panel._readout.text()
        assert "parabola" in text
        assert "x = 2" in text
        assert "y = 40" in text  # exact data, never estimated (§21.3)

    def test_snap_picks_nearest_of_two_series(self, panel):
        panel.line(np.array([0.0, 1.0]), np.array([0.0, 0.0]), label="flat")
        panel.scatter(np.array([0.0, 1.0]), np.array([5.0, 5.0]), label="high")
        # View range is x:[0,1] y:[0,5]; the 8%-of-range snap tolerance
        # accepts 0.3 in y -- pick the series the hover is closest to.
        snapped = panel._inspector.inspect_at((0.0, 4.7))
        assert snapped is not None and snapped[0] == "high"
        snapped = panel._inspector.inspect_at((0.0, 0.3))
        assert snapped is not None and snapped[0] == "flat"

    def test_far_hover_snaps_nothing(self, panel):
        panel.line(np.array([0.0, 1.0]), np.array([0.0, 1.0]), label="only")
        assert panel._inspector.inspect_at((500.0, 500.0)) is None
        assert panel._readout.text() == "—"
        assert not panel._inspector._marker.isVisible()


class TestMetadata:
    def test_metadata_cached_once(self, panel):
        calls = []

        def meta(i: int) -> dict:
            calls.append(i)
            return {"MW": 330.4 + i}

        panel.scatter(np.array([1.0, 2.0]), np.array([1.0, 2.0]), label="mols", meta=meta)
        panel._inspector.inspect_at((1.0, 1.0))
        panel._inspector.inspect_at((1.0, 1.0))
        panel._inspector.inspect_at((1.05, 1.0))  # still the same point
        assert calls == [0]  # §21.6: one computation, then cache
        assert "MW = 330.4" in panel._readout.text()

    def test_failing_metadata_degrades_gracefully(self, panel):
        def broken(i: int) -> dict:
            raise RuntimeError("no metadata for you")

        panel.line(np.array([1.0, 2.0]), np.array([1.0, 2.0]), label="frag", meta=broken)
        snapped = panel._inspector.inspect_at((1.0, 1.0))
        assert snapped == ("frag", 0)
        assert "x = 1" in panel._readout.text()  # values survive

    def test_attach_metadata_after_plot(self, panel):
        panel.line(np.array([1.0, 2.0]), np.array([1.0, 2.0]), label="late")
        assert panel.attach_metadata("late", lambda i: {"fold": i + 1})
        panel._inspector.inspect_at((2.0, 2.0))
        assert "fold = 2" in panel._readout.text()
        assert not panel.attach_metadata("missing", lambda i: {})


class TestChartKinds:
    def test_histogram_hover_shows_bin_boundaries(self, panel):
        rng = np.random.default_rng(7)
        panel.histogram(rng.normal(size=500), bins=10)
        record = next(iter(panel._inspector._series.values()))
        center = float(record.x[3])
        snapped = panel._inspector.inspect_at((center, float(record.y[3]) * 0.5))
        assert snapped is not None
        text = panel._readout.text()
        assert "bin = [" in text  # §21.3: exact boundaries
        assert "count = " in text

    def test_bars_hover_shows_category(self, panel):
        panel.bars(
            np.array([3.0, 5.0, 7.0]),
            labels=["Morgan", "MACCS", "RDKit"],
        )
        # Hover mid-body of the x=2 bar: the whole bar is the target.
        snapped = panel._inspector.inspect_at((2.0, 3.5))
        assert snapped == ("bars", 2)
        assert "category = RDKit" in panel._readout.text()


class TestMouseWiring:
    def test_mouse_move_drives_crosshair(self, panel, monkeypatch):
        _stub_scene_to_view(panel, monkeypatch)
        panel.line(np.array([0.0, 1.0]), np.array([0.0, 1.0]), label="a")
        panel._inspector._on_mouse_moved(QPointF(1.0, 1.0))
        assert panel._inspector._vline.isVisible()
        assert panel._inspector._hline.isVisible()
        # §12 dense-plot coalescing (round 4): the search runs once the
        # ~15 ms settle elapses; the readout catches up then, not on the
        # move itself.  Drain the timer synchronously (tests don't spin
        # the event loop) and the readout must be live.
        assert panel._inspector._settle_timer.isActive()
        panel._inspector._run_pending_inspect()
        assert panel._inspector._pending_view is None
        assert not panel._inspector._settle_timer.isActive()
        assert panel._readout.text() != "—"
        # Leaving the plot hides the decorations again AND cancels the
        # queued search (an outside move can never run a stale inspect).
        panel._inspector._on_mouse_moved(QPointF(1e12, 1e12))
        assert not panel._inspector._vline.isVisible()
        assert not panel._inspector._settle_timer.isActive()
        assert panel._inspector._pending_view is None
        assert panel._readout.text() == "—"

    def test_mouse_move_burst_runs_one_search(self, panel, monkeypatch):
        """§12 dense-plot debounce: a burst of moves coalesces to ONE
        deferred ``inspect_at`` -- only the last position matters."""
        _stub_scene_to_view(panel, monkeypatch)
        panel.line(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 4.0]), label="a")
        calls: list[tuple[float, float]] = []
        original = panel._inspector.inspect_at

        def counting(view_xy):
            calls.append((float(view_xy[0]), float(view_xy[1])))
            return original(view_xy)

        monkeypatch.setattr(panel._inspector, "inspect_at", counting)
        for x in (0.95, 0.96, 0.97, 0.98, 0.99):
            panel._inspector._on_mouse_moved(QPointF(x, 1.0))
        # Nothing ran yet -- the whole burst is one pending target.
        assert calls == []
        assert panel._inspector._pending_view == (0.99, 1.0)
        panel._inspector._run_pending_inspect()
        assert calls == [(0.99, 1.0)]  # exactly one search, last position
        # snapped to the nearest real point inside the tolerance window
        assert "x = 1" in panel._readout.text()

    def test_click_emits_point_selected(self, panel, monkeypatch):
        _stub_scene_to_view(panel, monkeypatch)
        panel.scatter(np.array([1.0, 2.0]), np.array([1.0, 2.0]), label="pts")
        events: list[tuple[str, int]] = []
        panel._inspector.pointSelected.connect(lambda *a: events.append(a))
        click = SimpleNamespace(scenePos=lambda: QPointF(2.0, 2.0))
        panel._inspector._on_mouse_clicked(click)
        assert events == [("pts", 1)]  # §21.5 wiring point for pages

    def test_disabled_inspection_ignores_mouse(self, panel, monkeypatch):
        _stub_scene_to_view(panel, monkeypatch)
        panel.line(np.array([0.0, 1.0]), np.array([0.0, 1.0]), label="a")
        panel.set_inspection_enabled(False)
        assert not panel.inspection_enabled()
        panel._inspector._on_mouse_moved(QPointF(1.0, 1.0))
        assert not panel._inspector._vline.isVisible()
        assert panel._readout.text() == "—"

    def test_dense_series_move_storm_runs_bounded_searches(self, panel, monkeypatch):
        """§12 (round 4): a 100k-point series under a 200-move storm
        runs ONE nearest-point search (the settle window), not 200 --
        the metadata calculation behind the readout is debounced, so
        dense plots stay interactive."""
        _stub_scene_to_view(panel, monkeypatch)
        xs = np.linspace(0.0, 1.0, 100_000)
        panel.line(xs, np.sin(xs * 50.0), label="dense")
        panel.scatter(xs[::100], np.cos(xs[::100] * 50.0), label="sparse")
        calls: list[tuple[float, float]] = []
        original = panel._inspector.inspect_at

        def counting(view_xy):
            calls.append((float(view_xy[0]), float(view_xy[1])))
            return original(view_xy)

        monkeypatch.setattr(panel._inspector, "inspect_at", counting)
        for i in range(200):
            panel._inspector._on_mouse_moved(QPointF(0.5 + i * 1e-4, 0.5))
        assert calls == []  # nothing during the storm
        panel._inspector._run_pending_inspect()
        assert len(calls) == 1  # exactly one search after the settle


class TestSeriesTogglesAndClear:
    def test_marker_stays_out_of_dataItems_until_first_hover(self, panel):
        """The round-232 regression pins: page tests read
        ``listDataItems()[0]`` as the first curve and census the
        InfiniteLines (the pkpd dose markers) — the inspector's
        decorations must leave BOTH lists alone until the first hover,
        and a clear() must empty the plot completely."""
        from pyqtgraph import InfiniteLine

        panel.line(np.array([0.0, 1.0]), np.array([0.0, 1.0]), label="a")
        plot_item = panel.widget().getPlotItem()
        assert len(plot_item.listDataItems()) == 1
        assert len([it for it in plot_item.items if isinstance(it, InfiniteLine)]) == 0
        panel._inspector.inspect_at((0.0, 0.0))
        assert len(plot_item.listDataItems()) == 2  # the marker joined lazily
        # After a clear the decorations leave again -- lazy re-add on hover.
        panel.clear()
        panel.line(np.array([0.0, 1.0]), np.array([0.0, 1.0]), label="b")
        assert len(plot_item.listDataItems()) == 1
        assert len([it for it in plot_item.items if isinstance(it, InfiniteLine)]) == 0

    def test_hidden_series_not_snapped(self, panel):
        panel.line(np.array([0.0, 1.0]), np.array([0.0, 0.0]), label="near")
        panel.scatter(np.array([0.5, 0.5]), np.array([0.5, 0.6]), label="hidden")
        panel._inspector._series["hidden"].item.setVisible(False)
        snapped = panel._inspector.inspect_at((0.0, 0.01))
        assert snapped == ("near", 0)

    def test_clear_resets_inspector(self, panel):
        panel.line(np.array([0.0, 1.0]), np.array([0.0, 1.0]), label="gone")
        panel.clear()
        assert panel._inspector._series == {}
        assert panel._inspector.inspect_at((0.5, 0.5)) is None
        assert panel._readout.text() == "—"

    def test_unlabelled_series_gets_placeholder_name(self, panel):
        panel.line(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        assert list(panel._inspector._series) == ["series 1"]
