"""Response AI Workbench page tests (increment 11): real engines, real numbers.

Everything runs offscreen against the same backend entry points the
page's tabs call (``pharmaml.ml.drugresponse`` — the module's FIRST
GUI surface, audit row 12); the assertions check the REAL numbers
those engines produce on the tab's own default inputs -- every number
was engine-printed first by ``scratch/probe_response_218.py`` (the
committed probe, round 219 re-seeded per section so each tab is a
standalone seeded job).
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
    from ui.pages.responseai import ResponseAIWorkbenchPage

    return ResponseAIWorkbenchPage(data_explorer=data_explorer)


# -- construction -----------------------------------------------------------------


def test_page_constructs_with_eight_tabs(studio_app) -> None:
    page = _page()
    assert page._tabs.count() == 8
    labels = [page._tabs.tabText(i) for i in range(page._tabs.count())]
    assert labels == [
        "Hill curve",
        "Response metrics",
        "Rank correlations",
        "Response model",
        "Cross-validation",
        "Informative regression",
        "Response VAE",
        "Multi-omics fusion",
    ]
    assert page._hill_noise.text() == "0.01"
    assert page._met_x.currentText() == "synthetic noisy pair (seed 42)"
    assert page._met_y.currentText() == "synthetic noisy pair (seed 42)"
    assert page._rank_x.currentText() == "synthetic curved pair (seed 42)"
    assert page._drm_epochs.text() == "120"
    assert page._drm_factors.text() == "2"
    assert page._cv_folds.text() == "4"


def test_parse_helpers_surface_honest_errors(studio_app) -> None:
    from ui.pages.responseai import _parse_float, _parse_int

    assert _parse_float(" 0.01 ", "noise") == 0.01
    try:
        _parse_float("abc", "noise")
    except ValueError as exc:
        assert "noise" in str(exc) and "abc" in str(exc)
    else:
        raise AssertionError("expected ValueError")
    try:
        _parse_float("-0.1", "noise", minimum=0.0)
    except ValueError as exc:
        assert "noise must be >=" in str(exc)
    else:
        raise AssertionError("expected ValueError (minimum floors at parse level)")
    try:
        _parse_int("401", "epochs", minimum=20, maximum=400)
    except ValueError as exc:
        assert "epochs must be between" in str(exc)
    else:
        raise AssertionError("expected ValueError")


# -- Hill curve -----------------------------------------------------------------


def test_hill_fit_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_hill()
    done.wait()
    _settle(qtbot, studio_app)

    # the probe-printed real numbers (seed 42, noise 0.01, 21 points):
    text = page.report_text()
    assert "seeded 4PL demo (seed 42) · noise 0.01 · r² 0.999669 · converged · 21 points" in text
    assert "fitted AUC 3.055882 vs closed-form 3.057985 (log-dose" in text

    frame = page.model_frame()
    assert frame is not None and len(frame) == 4
    assert list(frame["parameter"]) == ["top", "bottom", "ic50", "hill_coef"]
    assert round(float(frame["fitted"].iloc[0]), 4) == 0.9981
    assert round(float(frame["fitted"].iloc[1]), 4) == 0.049
    assert round(float(frame["fitted"].iloc[2]), 4) == 0.8123
    assert round(float(frame["fitted"].iloc[3]), 4) == 1.1679
    assert list(frame["true"]) == [1.0, 0.05, 0.8, 1.2]

    # the fitted curve (a line over the log10-dose grid) + the measured scatter
    items = page._plot.widget().getPlotItem().listDataItems()
    assert len(items) == 2  # curve line + points scatter
    x_line, y_line = items[0].getData()
    assert len(x_line) == 240
    # the curve approaches the fitted top asymptotically over the finite grid
    assert abs(float(y_line.max()) - 0.9981) < 0.001
    assert abs(float(y_line.min()) - 0.049) < 0.001  # ...and the fitted bottom


def test_hill_rejects_bad_inputs_at_parse_level(studio_app) -> None:
    page = _page()
    page._hill_noise.setText("abc")
    page._start_hill()
    assert "noise: not a number" in page.report_text()
    page._hill_noise.setText("-0.1")  # below the 0.0 floor
    page._start_hill()
    assert "noise must be >=" in page.report_text()
    page._hill_noise.setText("2.5")  # above the 1.0 ceiling
    page._start_hill()
    assert "noise must be <= 1" in page.report_text()


# -- response metrics -----------------------------------------------------------------


def test_metrics_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_metrics()
    done.wait()
    _settle(qtbot, studio_app)

    # the probe-printed real numbers (seeded noisy pair, seed 42):
    text = page.report_text()
    assert "seeded noisy pair (seed 42) · n=60 · pearson 0.9794 · spearman 0.9667" in text
    assert "rmse 0.0590 · mae 0.0485" in text

    frame = page.model_frame()
    assert frame is not None and len(frame) == 5
    assert list(frame["metric"]) == ["n", "pearson", "spearman", "rmse", "mae"]
    assert list(frame["value"]) == ["60", "0.979392", "0.966669", "0.058964", "0.048469"]

    # observed-vs-predicted scatter + the identity line
    items = page._plot.widget().getPlotItem().listDataItems()
    assert len(items) == 2
    x_scatter, _ = items[0].getData()
    assert len(x_scatter) == 60


def test_metrics_read_the_explorer_columns(studio_app, qtbot) -> None:
    class _FakeExplorer:
        def frame(self) -> pd.DataFrame:
            rng = np.random.default_rng(7)
            return pd.DataFrame(
                {
                    "obs": rng.normal(0.0, 1.0, 90),
                    "pred": rng.normal(0.0, 1.0, 90),
                    "label": list("x" * 90),
                }
            )

    page = _page(data_explorer=_FakeExplorer())
    frame = page._explorer_frame()
    assert frame is not None
    page._refresh_column_pickers(
        [str(c) for c in frame.select_dtypes(include=[np.number]).columns]
    )
    page._met_x.setCurrentText("obs")
    page._met_y.setCurrentText("pred")
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_metrics()
    done.wait()
    _settle(qtbot, studio_app)

    text = page.report_text()
    assert "columns 'obs' / 'pred' of the loaded table (90 aligned rows)" in text
    assert "n=90" in text


def test_metrics_refuse_short_columns(studio_app) -> None:
    class _FakeExplorer:
        def frame(self) -> pd.DataFrame:
            return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})

    page = _page(data_explorer=_FakeExplorer())
    frame = page._explorer_frame()
    assert frame is not None
    page._refresh_column_pickers(
        [str(c) for c in frame.select_dtypes(include=[np.number]).columns]
    )
    page._met_x.setCurrentText("a")
    page._met_y.setCurrentText("b")
    page._start_metrics()
    assert "have only 3 finite aligned rows" in page.report_text()
    assert "need at least 8" in page.report_text()


# -- rank correlations -----------------------------------------------------------------


def test_rank_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_rank()
    done.wait()
    _settle(qtbot, studio_app)

    # the probe-printed real numbers (seeded curved pair, seed 42):
    text = page.report_text()
    assert (
        "seeded curved pair (seed 42) · n=48 · pearson 0.9143 · spearman 0.9180 · "
        "kendall 0.7447" in text
    )

    frame = page.model_frame()
    assert frame is not None and len(frame) == 4
    assert list(frame["statistic"]) == ["n", "pearson", "spearman", "kendall"]
    assert list(frame["value"]) == ["48", "0.914276", "0.918042", "0.744681"]

    items = page._plot.widget().getPlotItem().listDataItems()
    assert len(items) == 1  # the pair scatter
    x_scatter, _ = items[0].getData()
    assert len(x_scatter) == 48


def test_rank_reads_the_explorer_columns(studio_app, qtbot) -> None:
    class _FakeExplorer:
        def frame(self) -> pd.DataFrame:
            rng = np.random.default_rng(11)
            return pd.DataFrame(
                {"x": rng.uniform(0.0, 10.0, 70), "y": rng.normal(0.0, 1.0, 70)}
            )

    page = _page(data_explorer=_FakeExplorer())
    frame = page._explorer_frame()
    assert frame is not None
    page._refresh_column_pickers(
        [str(c) for c in frame.select_dtypes(include=[np.number]).columns]
    )
    page._rank_x.setCurrentText("x")
    page._rank_y.setCurrentText("y")
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_rank()
    done.wait()
    _settle(qtbot, studio_app)

    text = page.report_text()
    assert "columns 'x' / 'y' of the loaded table (70 aligned rows)" in text


# -- response model -----------------------------------------------------------------


def test_model_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=120_000)
    page._start_model()
    done.wait()
    _settle(qtbot, studio_app)

    # the probe-printed real numbers (seeded 12x4 screen, seed 42):
    text = page.report_text()
    assert "seeded 12x4 screen (seed 42, 15% NaN mask) · 40 observed pairs" in text
    assert "final loss 0.029817 after 120 epochs" in text
    assert "in-sample pearson 0.164369 — honestly modest, the toy screen is tiny" in text

    frame = page.model_frame()
    assert frame is not None and len(frame) == 2  # the predict head
    assert list(frame.columns) == ["DRUG_A", "DRUG_B", "DRUG_C", "DRUG_D"]
    assert list(frame.index) == ["CELL_00", "CELL_07"]
    assert round(float(frame["DRUG_A"].iloc[0]), 4) == 0.5056
    assert round(float(frame["DRUG_B"].iloc[0]), 4) == 0.3383
    assert round(float(frame["DRUG_D"].iloc[1]), 4) == 0.4398

    # observed-vs-predicted over the 40 observed pairs + the identity line
    items = page._plot.widget().getPlotItem().listDataItems()
    assert len(items) == 2
    x_scatter, _ = items[0].getData()
    assert len(x_scatter) == 40


def test_model_rejects_bad_inputs(studio_app) -> None:
    page = _page()
    page._drm_epochs.setText("10")  # below the minimum of 20
    page._start_model()
    assert "epochs must be between" in page.report_text()
    page._drm_epochs.setText("120")
    page._drm_factors.setText("9")  # above the maximum of 4
    page._start_model()
    assert "latent factors must be between" in page.report_text()


# -- cross-validation -----------------------------------------------------------------


def test_cv_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=240_000)
    page._start_cv()
    done.wait()
    _settle(qtbot, studio_app)

    # the probe-printed real numbers (4 cell-line-grouped folds, seed 42):
    text = page.report_text()
    assert "4 folds grouped by cell line (no cell straddles a fold) · mean pearson 0.025198" in text
    assert "the honest read: the toy screen barely carries signal" in text

    frame = page.model_frame()
    assert frame is not None and len(frame) == 4
    assert list(frame.columns) == ["fold", "test cells", "observed", "pearson", "rmse"]
    assert [round(float(v), 6) for v in frame["pearson"]] == [
        -0.190352,
        -0.599944,
        0.178499,
        0.712589,
    ]
    assert [round(float(v), 6) for v in frame["rmse"]] == [
        0.191737,
        0.196455,
        0.281619,
        0.132244,
    ]
    assert list(frame["observed"]) == [10, 9, 11, 10]

    # the per-fold pearson bars (negative folds draw downward — honest)
    from pyqtgraph import BarGraphItem

    bars = [
        item
        for item in page._plot.widget().getPlotItem().items
        if isinstance(item, BarGraphItem)
    ]
    assert len(bars) == 1
    heights = np.asarray(bars[0].opts["height"], dtype=float)
    assert round(float(heights[0]), 6) == -0.190352
    assert round(float(heights[3]), 6) == 0.712589


def test_cv_rejects_bad_folds(studio_app) -> None:
    page = _page()
    page._cv_folds.setText("1")  # below the minimum of 2
    page._start_cv()
    assert "folds must be between" in page.report_text()
    page._cv_folds.setText("12")  # above the maximum of 8
    page._start_cv()
    assert "folds must be between" in page.report_text()


# -- informative regression -----------------------------------------------------------------


def test_informative_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_info()
    done.wait()
    _settle(qtbot, studio_app)

    # the probe-printed real numbers (seed 42, priors [100, 4, 4]):
    text = page.report_text()
    assert "prior variance [100, 4, 4] per feature · coefs [0.479, 1.123, -0.598]" in text
    assert "vs generating [0.5, 1.5, -0.8] · loss 2.4213 → 0.5502 over 300 epochs" in text
    assert "intercept 0.0048" in text

    frame = page.model_frame()
    assert frame is not None and len(frame) == 4
    assert list(frame["term"]) == ["bias column (x0)", "x1", "x2", "model intercept"]
    assert [round(float(v), 4) for v in frame["fitted"]] == [0.4789, 1.1233, -0.5979, 0.0048]
    assert list(frame["generating"]) == [0.5, 1.5, -0.8, 0.0]

    # the loss curve (one line over 300 epochs)
    items = page._plot.widget().getPlotItem().listDataItems()
    assert len(items) == 1
    x_line, y_line = items[0].getData()
    assert len(x_line) == 300
    assert round(float(y_line[0]), 4) == 2.4213
    assert round(float(y_line[-1]), 4) == 0.5502


# -- response VAE -----------------------------------------------------------------


def test_vae_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=120_000)
    page._start_vae()
    done.wait()
    _settle(qtbot, studio_app)

    # the probe-printed real numbers (seed 42, 60x10, 80 epochs):
    text = page.report_text()
    assert "60x10 seeded features with a response head · ELBO -20.220437 → -10.032681" in text
    assert "(the bound improving)" in text
    assert "reconstruction 7.748560 · KL 1.343065 · response MSE 0.941055 · 80 epochs" in text

    frame = page.model_frame()
    assert frame is not None and len(frame) == 6
    assert list(frame["ledger"]) == [
        "elbo first",
        "elbo last",
        "reconstruction (last)",
        "kl divergence (last)",
        "response loss (last)",
        "epochs",
    ]
    assert list(frame["value"]) == [
        "-20.220437",
        "-10.032681",
        "7.748560",
        "1.343065",
        "0.941055",
        "80",
    ]

    # the ELBO line over the 80 epochs
    items = page._plot.widget().getPlotItem().listDataItems()
    assert len(items) == 1
    x_line, y_line = items[0].getData()
    assert len(x_line) == 80
    assert round(float(y_line[0]), 6) == -20.220437
    assert round(float(y_line[-1]), 6) == -10.032681


# -- multi-omics fusion -----------------------------------------------------------------


def test_fusion_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_fusion()
    done.wait()
    _settle(qtbot, studio_app)

    # the probe-printed real numbers (seed 42, 30 x (12 + 8)):
    text = page.report_text()
    assert "2 blocks fused → latent 30x20 (expr (12 cols), mut (8 cols))" in text
    assert "per-block total variance equalised to 1.0" in text
    assert "the wide expression block cannot drown the narrow mutation block" in text

    frame = page.model_frame()
    assert frame is not None and len(frame) == 2
    assert list(frame["block"]) == ["expr", "mut"]
    assert list(frame["n_features"]) == [12, 8]
    assert list(frame["n_constant"]) == [0, 0]
    assert list(frame["total_variance"]) == [1.0, 1.0]
    assert round(float(frame["mean column std (raw)"].iloc[0]), 6) == 0.927576
    assert round(float(frame["mean column std (raw)"].iloc[1]), 6) == 0.488877

    # the equalisation bars (both 1.0 — flat on purpose, that IS the point)
    from pyqtgraph import BarGraphItem

    bars = [
        item
        for item in page._plot.widget().getPlotItem().items
        if isinstance(item, BarGraphItem)
    ]
    assert len(bars) == 1
    heights = np.asarray(bars[0].opts["height"], dtype=float)
    assert np.allclose(heights, [1.0, 1.0])


# -- main-window wiring -----------------------------------------------------------------


def test_main_window_carries_the_responseai_section(studio_app) -> None:
    from ui.main_window import SECTIONS

    keys = [key for key, _title in SECTIONS]
    # responseai now sits right before the Image Lab (increment 12) + Utilities + Settings
    assert keys.index("responseai") == len(keys) - 4
    assert keys[len(keys) - 3] == "imagelab"
    assert keys[len(keys) - 2] == "utilities"
    assert len(keys) == 17
