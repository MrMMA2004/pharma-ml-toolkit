"""Foundations Workbench page tests (increment 9): real engines, real numbers.

Everything runs offscreen against the same backend entry points the
playground cards call (``compare_groups`` / ``sample_distribution``,
``welch_anova`` + ``tukey_hsd`` + ``games_howell``, ``profile_dataset``);
the assertions check the REAL numbers those engines produce on the
card's own default inputs -- every number was engine-printed first by
``scratch/probe_foundations_215.py`` (the committed probe).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


def _drain(pool_wait: float = 5.0) -> None:
    from PySide6.QtCore import QThreadPool

    deadline = time.monotonic() + pool_wait
    while QThreadPool.globalInstance().activeThreadCount() > 0 and time.monotonic() < deadline:
        time.sleep(0.01)


def _settle(qtbot, qapp) -> None:
    """Wait out the pool, then flush the queued result slots."""
    _drain()
    qtbot.wait(30)
    qapp.processEvents()


def _page(data_explorer: object | None = None):
    from ui.pages.foundations import FoundationsWorkbenchPage

    return FoundationsWorkbenchPage(data_explorer=data_explorer)


# -- construction -----------------------------------------------------------------


def test_page_constructs_with_four_tabs(studio_app) -> None:
    page = _page()
    assert page._tabs.count() == 4
    labels = [page._tabs.tabText(i) for i in range(page._tabs.count())]
    assert labels == ["Compare groups", "Distribution sampler", "Inferential tests", "Table profiler"]
    assert page._stat_mode.currentText() == "synthetic demo (two groups)"
    assert page._stat_alpha.text() == "0.05"
    assert page._stat_level.text() == "0.95"
    assert page._dist_kind.currentText() == "normal"
    assert page._dist_n.text() == "500"
    assert "control: 2.1, 2.4, 2.2, 2.5, 2.3" in page._istat_text.toPlainText()
    assert page._profiler_text.toPlainText().startswith("dose,route,response,note")


def test_parse_helpers_surface_honest_errors(studio_app) -> None:
    from ui.pages.foundations import _parse_float, _parse_int

    assert _parse_float(" 0.05 ", "alpha") == 0.05
    try:
        _parse_float("abc", "alpha")
    except ValueError as exc:
        assert "alpha" in str(exc) and "abc" in str(exc)
    else:
        raise AssertionError("expected ValueError")
    try:
        _parse_float("0", "scale", minimum=1e-6)
    except ValueError as exc:
        assert "scale must be >=" in str(exc)
    else:
        raise AssertionError("expected ValueError (minimum floors at parse level)")
    try:
        _parse_int("9", "n", minimum=10, maximum=100000)
    except ValueError as exc:
        assert "n must be between" in str(exc)
    else:
        raise AssertionError("expected ValueError")


# -- compare groups -----------------------------------------------------------------


def test_compare_groups_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_stats()
    done.wait()
    _settle(qtbot, studio_app)

    text = page.report_text()
    assert "synthetic demo: N(1.2, 0.4) n=60 vs N(1.0, 0.45) n=50" in text
    # the probe-printed real numbers (rng 42, compare_groups seed 42):
    assert "both normal → Welch t (unequal variances)" in text
    assert "Welch t-test · statistic 5.106" in text
    assert "REJECT H₀ at α 0.05 -- the groups differ (a > b)" in text
    assert "A · n 60 (0 missing) · mean 1.226" in text
    assert "B · n 50 (0 missing) · mean 0.9014" in text

    frame = page.model_frame()
    assert frame is not None and len(frame) == 2
    assert list(frame["group"]) == ["A", "B"]
    assert round(float(frame["mean"].iloc[0]), 4) == 1.2261  # probe: 1.226114
    assert round(float(frame["mean"].iloc[1]), 4) == 0.9014  # probe: 0.901388

    # both histograms share the axis: the bar items carry value-space x
    from pyqtgraph import BarGraphItem

    bars = [
        item
        for item in page._plot.widget().getPlotItem().items
        if isinstance(item, BarGraphItem)
    ]
    assert len(bars) == 2
    xs = [np.asarray(bar.opts["x"], dtype=float) for bar in bars]
    assert len(xs[0]) == 30 and len(xs[1]) == 30  # bins=30 on the shared range
    assert np.allclose(xs[0], xs[1])  # the SAME edges -> comparable overlays


def test_compare_groups_columns_mode_reads_the_explorer(studio_app, qtbot) -> None:
    class _FakeExplorer:
        def frame(self) -> pd.DataFrame:
            rng = np.random.default_rng(7)
            return pd.DataFrame(
                {"name": list("abcdefgh"), "x": rng.normal(0, 1, 8), "y": rng.normal(5, 1, 8)}
            )

    page = _page(data_explorer=_FakeExplorer())
    page._stat_mode.setCurrentIndex(1)
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_stats()
    done.wait()
    _settle(qtbot, studio_app)

    assert page._stat_col_a.count() == 2  # only the numeric columns offered
    assert page._stat_col_a.currentText() == "x"
    text = page.report_text()
    assert "x vs y" in text
    assert "branch:" in text


def test_compare_groups_columns_mode_refuses_empty_explorer(studio_app) -> None:
    class _EmptyExplorer:
        def frame(self) -> pd.DataFrame | None:
            return None

    page = _page(data_explorer=_EmptyExplorer())
    page._stat_mode.setCurrentIndex(1)
    page._start_stats()
    assert "no table loaded" in page.report_text()


# -- distribution sampler -----------------------------------------------------------------


def test_sampler_normal_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_sampler()
    done.wait()
    _settle(qtbot, studio_app)

    text = page.report_text()
    # the probe-printed real numbers (seed 42):
    assert "normal · n 500 · sample mean -0.01313 (theory 0)" in text
    assert "sample std 0.959 (theory 1)" in text
    assert "N(0, 1²)" in text  # the theory label

    frame = page.model_frame()
    assert frame is not None and len(frame) == 1
    assert frame["kind"].iloc[0] == "normal"
    assert int(frame["n"].iloc[0]) == 500

    # the histogram bars sit at VALUE-space centers and the theory line
    # rides the same axis (the round-215 PlotPanel extension)
    items = page._plot.widget().getPlotItem().listDataItems()
    curves = [item for item in items if hasattr(item, "getData") and item not in ()]
    assert curves, "the theory curve must be drawn"
    x_data, _y = curves[-1].getData()
    assert float(np.min(x_data)) <= -2.0 and float(np.max(x_data)) >= 2.5
    assert len(x_data) == 400  # the linspace theory grid


def test_sampler_binomial_pmf_rides_the_counts(studio_app, qtbot) -> None:
    page = _page()
    page._dist_kind.setCurrentText("binomial")
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_sampler()
    done.wait()
    _settle(qtbot, studio_app)

    # the probe: sample mean 2.952 (theory 3), value range 0..8
    text = page.report_text()
    assert "binomial · n 500 · sample mean 2.952 (theory 3)" in text
    items = page._plot.widget().getPlotItem().listDataItems()
    assert items, "the pmf curve must be drawn over the integer histogram"
    x_data, _y = items[-1].getData()
    assert int(np.round(float(np.min(x_data)))) == 0
    assert int(np.round(float(np.max(x_data)))) == 8


def test_sampler_rejects_bad_inputs_at_parse_level(studio_app) -> None:
    page = _page()
    page._dist_scale.setText("0")
    page._start_sampler()
    assert "scale must be >=" in page.report_text()
    page._dist_scale.setText("1")
    page._dist_n.setText("5")  # below the minimum of 10
    page._start_sampler()
    assert "sample size must be between" in page.report_text()


# -- inferential tests -----------------------------------------------------------------


def test_inferential_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_inferential()
    done.wait()
    _settle(qtbot, studio_app)

    # the probe-printed real numbers on the card's default groups:
    text = page.report_text()
    assert "Welch F = 70.916 · p = 1.67e-05 · 3 groups" in text
    assert "Games-Howell (pairwise):" in text
    assert "g1 g2" in text  # the pairwise table columns

    frame = page.model_frame()
    assert frame is not None and len(frame) == 3  # 3 pairs for 3 groups
    assert list(frame.columns) == ["A", "B", "mean(A)", "mean(B)", "diff", "se", "T", "p-tukey"]
    assert round(float(frame["diff"].iloc[0]), 4) == -0.65  # probe: control vs low
    assert round(float(frame["T"].iloc[1]), 4) == -13.2568  # probe: control vs high

    # the F-statistic bar renders
    from pyqtgraph import BarGraphItem

    bars = [
        item
        for item in page._plot.widget().getPlotItem().items
        if isinstance(item, BarGraphItem)
    ]
    assert len(bars) == 1


def test_inferential_needs_two_groups(studio_app) -> None:
    page = _page()
    page._istat_text.setPlainText("only: 1, 2, 3")
    page._start_inferential()
    assert "at least two" in page.report_text()


# -- table profiler -----------------------------------------------------------------


def test_profiler_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_profiler()
    done.wait()
    _settle(qtbot, studio_app)

    # the probe-printed real numbers on the default pasted CSV:
    text = page.report_text()
    assert "6 rows x 4 columns" in text
    assert "4.2% cells missing" in text
    assert "0 duplicate rows" in text
    assert "3 categorical, 1 text" in text
    assert "response: missing" in text
    assert "response: unique" in text

    frame = page.model_frame()
    assert frame is not None and len(frame) == 4
    assert list(frame["column"]) == ["dose", "route", "response", "note"]
    assert frame["type"].iloc[0] == "categorical"
    assert frame["type"].iloc[3] == "text"
    assert round(float(frame["missing"].iloc[2]), 4) == 0.1667  # 1 of 6
    assert frame["top"].iloc[1] == "IV"


def test_profiler_refuses_empty_paste(studio_app) -> None:
    page = _page()
    page._profiler_text.setPlainText("")
    page._start_profiler()
    assert "Paste a CSV" in page.report_text()


# -- main-window wiring -----------------------------------------------------------------


def test_main_window_carries_the_foundations_section(studio_app) -> None:
    from ui.main_window import SECTIONS

    keys = [key for key, _title in SECTIONS]
    assert keys.index("foundations") == len(keys) - 6  # before Best-of ML / Response AI / Image Lab / Utilities / Settings
    assert len(keys) == 16
