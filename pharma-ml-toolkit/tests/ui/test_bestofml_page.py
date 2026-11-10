"""Best-of ML Workbench page tests (increment 10): real engines, real numbers.

Everything runs offscreen against the same backend entry points the
playground cards call (``pharmaml.ml.catalog`` / ``privacy`` /
``recommend`` / ``annsearch`` / ``adversarial`` / ``timeseries`` /
``apriori``); the assertions check the REAL numbers those engines
produce on the card's own default inputs -- every number was
engine-printed first by ``scratch/probe_bestofml_217.py`` (the
committed probe).
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
    from ui.pages.bestofml import BestOfMLWorkbenchPage

    return BestOfMLWorkbenchPage(data_explorer=data_explorer)


# -- construction -----------------------------------------------------------------


def test_page_constructs_with_eight_tabs(studio_app) -> None:
    page = _page()
    assert page._tabs.count() == 8
    labels = [page._tabs.tabText(i) for i in range(page._tabs.count())]
    assert labels == [
        "Catalogue",
        "Coverage audit",
        "Differential privacy",
        "Recommender",
        "Vector search (LSH)",
        "Adversarial",
        "Time series",
        "Association rules",
    ]
    assert page._cat_search.text() == "differential privacy"
    assert page._cat_category.currentText() == "all categories"
    assert page._cat_category.count() >= 34  # "all categories" + the catalogue's sections
    assert page._dp_eps.text() == "1.0"
    assert page._dp_bins.text() == "8"
    assert page._cf_kind.currentText() == "item"
    assert page._cf_k.text() == "8"
    assert page._cf_n.text() == "5"
    assert page._lsh_probes.currentText() == "2"
    assert page._ts_period.text() == "24"
    assert page._assoc_text.toPlainText().startswith("aspirin, metformin")
    assert page._assoc_support.text() == "0.30"
    assert page._assoc_confidence.text() == "0.40"


def test_parse_helpers_surface_honest_errors(studio_app) -> None:
    from ui.pages.bestofml import _parse_float, _parse_int

    assert _parse_float(" 1.0 ", "epsilon") == 1.0
    try:
        _parse_float("abc", "epsilon")
    except ValueError as exc:
        assert "epsilon" in str(exc) and "abc" in str(exc)
    else:
        raise AssertionError("expected ValueError")
    try:
        _parse_float("0.005", "epsilon", minimum=0.01)
    except ValueError as exc:
        assert "epsilon must be >=" in str(exc)
    else:
        raise AssertionError("expected ValueError (minimum floors at parse level)")
    try:
        _parse_int("25", "bins", minimum=0, maximum=20)
    except ValueError as exc:
        assert "bins must be between" in str(exc)
    else:
        raise AssertionError("expected ValueError")


# -- catalogue -----------------------------------------------------------------


def test_catalogue_search_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_catalog("search")
    done.wait()
    _settle(qtbot, studio_app)

    text = page.report_text()
    # the probe-printed real numbers: 1 hit, 924 projects / 35 sections
    assert "1 project(s) · search 'differential privacy'" in text
    assert "snapshot 924 projects / 35 sections / 5,121,472 stars" in text

    frame = page.model_frame()
    assert frame is not None and len(frame) == 1
    assert frame["project"].iloc[0] == "Opacus"
    assert frame["category"].iloc[0] == "Privacy Machine Learning"
    assert int(frame["stars"].iloc[0]) == 1872
    assert frame["license"].iloc[0] == "Apache-2.0"

    # the hit's star count draws as the single bar
    from pyqtgraph import BarGraphItem

    bars = [
        item
        for item in page._plot.widget().getPlotItem().items
        if isinstance(item, BarGraphItem)
    ]
    assert len(bars) == 1
    assert np.allclose(np.asarray(bars[0].opts["height"], dtype=float), [1872.0])


def test_catalogue_top10_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_catalog("top")
    done.wait()
    _settle(qtbot, studio_app)

    frame = page.model_frame()
    assert frame is not None and len(frame) == 10
    # the probe: the two giants lead in best-of score order
    assert frame["project"].iloc[0] == "Tensorflow"
    assert frame["project"].iloc[1] == "PyTorch"
    assert int(frame["stars"].iloc[0]) == 195075
    text = page.report_text()
    assert "10 project(s) · top 10 by best-of score" in text


def test_catalogue_search_refuses_empty_term(studio_app) -> None:
    page = _page()
    page._cat_search.setText("   ")
    page._start_catalog("search")
    assert "Type a search term" in page.report_text()


# -- coverage audit -----------------------------------------------------------------


def test_coverage_audit_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_coverage()
    done.wait()
    _settle(qtbot, studio_app)

    # the probe-printed verdict counts over the 35 entries:
    text = page.report_text()
    assert "12 native · 6 partial · 4 external · 13 out of scope" in text
    assert "privacy, recommend, annsearch, adversarial, timeseries" in text

    frame = page.model_frame()
    assert frame is not None and len(frame) == 35
    statuses = set(frame["status"])
    assert statuses == {"native", "partial", "external", "out-of-scope"}

    # the verdict-count bars: 4 bars with the native count the tallest
    from pyqtgraph import BarGraphItem

    bars = [
        item
        for item in page._plot.widget().getPlotItem().items
        if isinstance(item, BarGraphItem)
    ]
    assert len(bars) == 1
    heights = np.asarray(bars[0].opts["height"], dtype=float)
    assert np.allclose(heights, [12.0, 6.0, 4.0, 13.0])


# -- differential privacy -----------------------------------------------------------------


def test_dp_statistics_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_dp()
    done.wait()
    _settle(qtbot, studio_app)

    # the probe-printed real numbers (synthetic N(50, 5) n=400, eps 1.0,
    # bins 8, seed 42): dp mean 49.65164 vs true 49.974067, sigma 50.3294
    text = page.report_text()
    assert "synthetic N(50, 5²) column (seed 42) · n=400" in text
    assert "dp mean 49.6516 vs true 49.9741" in text
    assert "needs sigma=50.3" in text
    assert "production bounds must be public" in text  # the data-bounds disclosure

    frame = page.model_frame()
    assert frame is not None and len(frame) == 4  # count/sum/mean + the worst bin
    assert list(frame["statistic"]) == ["count", "sum", "mean", "bin [47.4, 50.9)"]
    assert round(float(frame["true"].iloc[0]), 4) == 400.0
    assert round(float(frame["dp"].iloc[0]), 4) == 402.3816
    assert round(float(frame["true"].iloc[2]), 6) == 49.974067
    assert round(float(frame["dp"].iloc[2]), 4) == 49.6516

    # the grouped true-vs-dp bin bars (two BarGraphItems over bin centres)
    from pyqtgraph import BarGraphItem

    bars = [
        item
        for item in page._plot.widget().getPlotItem().items
        if isinstance(item, BarGraphItem)
    ]
    assert len(bars) == 2
    true_heights = np.asarray(bars[0].opts["height"], dtype=float)
    dp_heights = np.asarray(bars[1].opts["height"], dtype=float)
    assert len(true_heights) == 8 and len(dp_heights) == 8
    assert float(true_heights.max()) == 110.0  # the probe's worst bin true count
    assert round(float(dp_heights[3]), 4) == 119.0628  # the same bin's DP count


def test_dp_rejects_bad_inputs_at_parse_level(studio_app) -> None:
    page = _page()
    page._dp_eps.setText("0.001")  # below the 0.01 floor
    page._start_dp()
    assert "ε must be >=" in page.report_text()
    page._dp_eps.setText("1.0")
    page._dp_bins.setText("25")  # above the 20 ceiling
    page._start_dp()
    assert "bins must be between" in page.report_text()


# -- recommender -----------------------------------------------------------------


def test_recommender_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_recommender()
    done.wait()
    _settle(qtbot, studio_app)

    # the probe-printed real numbers (item-based, k 8, top 5, seed 42):
    # hits 8 vs popularity 11 over 24 users, lift 0.7273 — popularity won
    text = page.report_text()
    assert "item-based CF (k=8, top 5) on 24 users x 20 items (219 ratings)" in text
    assert "hit-rate 8/24 vs popularity 11/24 (lift 0.73)" in text
    assert "popularity won — the report says so rather than burying it" in text

    frame = page.model_frame()
    assert frame is not None and len(frame) == 3  # user 0's top 3
    assert int(frame["item"].iloc[0]) == 7
    assert round(float(frame["predicted"].iloc[0]), 4) == 3.7318
    assert round(float(frame["weight"].iloc[0]), 4) == 7.2049

    # the CF-vs-popularity hit bars
    from pyqtgraph import BarGraphItem

    bars = [
        item
        for item in page._plot.widget().getPlotItem().items
        if isinstance(item, BarGraphItem)
    ]
    assert len(bars) == 1
    assert np.allclose(np.asarray(bars[0].opts["height"], dtype=float), [8.0, 11.0])


def test_recommender_rejects_bad_inputs(studio_app) -> None:
    page = _page()
    page._cf_k.setText("0")  # below the minimum of 1
    page._start_recommender()
    assert "neighbours k must be between" in page.report_text()


# -- LSH -----------------------------------------------------------------


def test_lsh_sweep_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_lsh()
    done.wait()
    _settle(qtbot, studio_app)

    # the probe-printed real numbers (360x8 three-blob, probes 2, k 5):
    text = page.report_text()
    assert "synthetic three-blob collection (seed 42) (360 vectors x 8 dims, probes=2)" in text
    assert "0 empty queries across the sweep" in text

    frame = page.model_frame()
    assert frame is not None and len(frame) == 5
    assert list(frame["planes"]) == [2, 4, 6, 8, 10]
    assert round(float(frame["recall@5"].iloc[0]), 4) == 0.985
    assert round(float(frame["recall@5"].iloc[4]), 4) == 0.575
    assert round(float(frame["cheaper"].iloc[4]), 2) == 13.6

    # the recall + touched trade-off curves (two lines over the plane grid)
    items = page._plot.widget().getPlotItem().listDataItems()
    assert len(items) >= 2
    x_data, _y = items[0].getData()
    assert np.allclose(x_data, [2.0, 4.0, 6.0, 8.0, 10.0])


def test_lsh_reads_the_explorer_numeric_block(studio_app, qtbot) -> None:
    class _FakeExplorer:
        def frame(self) -> pd.DataFrame:
            rng = np.random.default_rng(7)
            return pd.DataFrame(
                {"a": rng.normal(0, 1, 80), "b": rng.normal(5, 1, 80), "label": list("x" * 80)}
            )

    page = _page(data_explorer=_FakeExplorer())
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_lsh()
    done.wait()
    _settle(qtbot, studio_app)

    text = page.report_text()
    assert "the numeric block of the loaded table (80 vectors x 2 dims" in text


# -- adversarial -----------------------------------------------------------------


def test_adversarial_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=120_000)
    page._start_adversarial()
    done.wait()
    _settle(qtbot, studio_app)

    # the probe-printed real numbers (two-blob logistic, radii .05-.5):
    text = page.report_text()
    assert "logistic regression on synthetic two-blob demo (seed 42) (160 rows x 3 features, classification)" in text
    assert "clean accuracy 0.988" in text
    assert "median adversarial margin 0.500 sd" in text

    frame = page.model_frame()
    assert frame is not None and len(frame) == 4
    assert [round(float(r), 2) for r in frame["radius"]] == [0.05, 0.1, 0.25, 0.5]
    assert list(frame["broken"]) == [5, 5, 14, 44]
    assert round(float(frame["rate"].iloc[3]), 4) == 0.275

    # the broken-rate curve (a line + its scatter markers)
    items = page._plot.widget().getPlotItem().listDataItems()
    assert items
    x_data, y_data = items[0].getData()
    assert np.allclose(x_data, [0.05, 0.1, 0.25, 0.5])
    assert round(float(y_data[3]), 4) == 0.275


def test_adversarial_reads_the_explorer_frame(studio_app, qtbot) -> None:
    class _FakeExplorer:
        def frame(self) -> pd.DataFrame:
            rng = np.random.default_rng(3)
            x = rng.normal(0, 1, (120, 4))
            return pd.DataFrame(
                {"f1": x[:, 0], "f2": x[:, 1], "f3": x[:, 2], "f4": x[:, 3], "cls": (x[:, 0] > 0).astype(float)}
            )

    page = _page(data_explorer=_FakeExplorer())
    done = qtbot.waitSignal(page._runner.job_ended, timeout=120_000)
    page._start_adversarial()
    done.wait()
    _settle(qtbot, studio_app)

    text = page.report_text()
    assert "logistic regression on the loaded table (120 rows x 4 features" in text


# -- time series -----------------------------------------------------------------


def test_timeseries_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_timeseries()
    done.wait()
    _settle(qtbot, studio_app)

    # the probe-printed real numbers (sine 240 points, period 24):
    text = page.report_text()
    assert "synthetic period-24 signal (seed 42) (240 points, period 24)" in text
    assert "the seasonal profile explains 98% of the detrended variance" in text
    assert "mean window slope -0.0343/step" in text

    frame = page.model_frame()
    assert frame is not None and len(frame) == 7  # changepoints row says 0 -> no "first change"
    assert list(frame["metric"]) == [
        "ACF peak lag",
        "ACF significant lags",
        "PACF lag-1",
        "seasonal strength",
        "trend strength",
        "changepoints",
        "window rows",
    ]
    assert frame["value"].iloc[0] == "24"  # the period carries the memory
    assert frame["value"].iloc[1] == "28 of 30"
    assert frame["value"].iloc[2] == "0.866"
    assert frame["value"].iloc[3] == "0.982"
    assert frame["value"].iloc[5] == "0"  # no changepoints in the pure sine
    assert frame["value"].iloc[6] == "10 x 13"  # 240/24 windows x tsfresh-lite columns

    # the ACF bars sit at VALUE-space lags (peak at lag 24)
    from pyqtgraph import BarGraphItem

    bars = [
        item
        for item in page._plot.widget().getPlotItem().items
        if isinstance(item, BarGraphItem)
    ]
    assert len(bars) == 1
    xs = np.asarray(bars[0].opts["x"], dtype=float)
    heights = np.asarray(bars[0].opts["height"], dtype=float)
    assert round(float(heights[0]), 4) == 1.0  # ACF(0) = 1 by definition
    assert int(xs[1 + int(np.argmax(heights[1:]))]) == 24  # the period carries the memory
    assert len(xs) == 31  # lags 0..30


def test_timeseries_refuses_short_series(studio_app) -> None:
    page = _page()
    page._ts_period.setText("168")  # 240 < 2*168 -> honest refusal
    page._start_timeseries()
    assert "shorter than two full periods" in page.report_text()


# -- association rules -----------------------------------------------------------------


def test_assoc_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_assoc()
    done.wait()
    _settle(qtbot, studio_app)

    # the probe-printed real numbers (5 default transactions, s .30, c .40):
    text = page.report_text()
    assert "5 transactions · 3 items · 5 itemsets · 4 rules" in text
    assert "statin -> aspirin  supp 0.40 · conf 1.00 · lift 1.25" in text
    assert "aspirin, metformin -> aspirin" not in text  # honest: single items only

    frame = page.model_frame()
    assert frame is not None and len(frame) == 4
    assert list(frame.columns) == [
        "antecedent",
        "consequent",
        "antecedent_support",
        "consequent_support",
        "support",
        "confidence",
        "lift",
        "leverage",
        "conviction",
    ]
    assert round(float(frame["lift"].iloc[0]), 4) == 1.25
    assert round(float(frame["support"].iloc[2]), 4) == 0.6

    # the lift bars: one bar per rule, labelled antecedent -> consequent
    from pyqtgraph import BarGraphItem

    bars = [
        item
        for item in page._plot.widget().getPlotItem().items
        if isinstance(item, BarGraphItem)
    ]
    assert len(bars) == 1
    assert np.allclose(np.asarray(bars[0].opts["height"], dtype=float), [1.25, 1.25, 0.9375, 0.9375])


def test_assoc_rejects_single_transaction(studio_app) -> None:
    page = _page()
    page._assoc_text.setPlainText("aspirin, metformin")
    page._start_assoc()
    assert "at least two transaction lines" in page.report_text()
    page._assoc_text.setPlainText(_page()._assoc_text.toPlainText())
    page._assoc_support.setText("2")  # above 1.0
    page._start_assoc()
    assert "must be <= 1" in page.report_text()


# -- main-window wiring -----------------------------------------------------------------


def test_main_window_carries_the_bestofml_section(studio_app) -> None:
    from ui.main_window import SECTIONS

    keys = [key for key, _title in SECTIONS]
    assert keys.index("bestofml") == len(keys) - 6  # before Response AI / Image Lab / MD / Utilities / Settings
    assert len(keys) == 18
