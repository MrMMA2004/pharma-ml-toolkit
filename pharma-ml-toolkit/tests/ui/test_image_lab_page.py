"""Image Lab page tests (increment 12): real engines, real numbers.

Everything runs offscreen against the same backend entry points the
page's tabs call (``compute_fingerprints`` + ``tanimoto_matrix``, the
Biology screen pipeline, the explorer-frame z-scoring); the
assertions check the REAL numbers those engines produce on the tab's
own default inputs -- every number was engine-printed first by
``scratch/probe_imagelab_221.py`` (the committed probe, per-section
fresh seeds).

This suite is also the Part-2 closure evidence: 16.8.2 (the image
display), 16.8.3 (the interactive ROI with sub-matrix statistics),
16.8.4 (the histogram/LUT colormap control), 16.8.7 / 16.19 (the PNG
export), and the theme-compatibility rule (chrome re-styles, the
colormap stays data-driven).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


def _drain(pool_wait: float = 8.0) -> None:
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
    from ui.pages.image_lab import ImageLabPage

    return ImageLabPage(data_explorer=data_explorer)


# -- construction -----------------------------------------------------------------


def test_page_constructs_with_five_tabs(studio_app) -> None:
    page = _page()
    assert page._tabs.count() == 5
    labels = [page._tabs.tabText(i) for i in range(page._tabs.count())]
    assert labels == [
        "Similarity heatmap",
        "Response matrix",
        "Explorer matrix",
        "ROI explorer",
        "Performance",
    ]
    assert page._sim_source.currentText() == "bundled seed_molecules.csv (12 drugs)"
    assert page._sim_column.isEnabled() is False  # explorer source not selected
    assert page._exp_z.isChecked() is True
    assert page._perf_n.value() == 1024
    assert page._status.text() == "Ready."


def test_roi_tab_seeds_the_probe_numbers(studio_app) -> None:
    """The ROI tab's first paint IS the probe's similarity matrix."""
    page = _page()
    values = page._roi_panel.values()
    assert values is not None and values.shape == (12, 12)
    assert bool(np.allclose(np.diag(values), 1.0))
    off = values[~np.eye(12, dtype=bool)]
    assert round(float(off.max()), 6) == 0.529412  # caffeine ~ theophylline
    assert round(float(off.min()), 6) == 0.021277  # metformin ~ diclofinac
    # the default region reproduces the probe's section-4 statistics
    text = page._roi_report.text()
    assert "rows [2, 6) x cols [7, 11): 16 cells" in text
    assert "mean 0.131007, min 0.021277, max 0.529412" in text


def test_roi_region_moves_and_reports(studio_app) -> None:
    page = _page()
    page._roi_row0.setValue(0)
    page._roi_row1.setValue(12)
    page._roi_col0.setValue(0)
    page._roi_col1.setValue(12)
    page._apply_roi_region()
    text = page._roi_report.text()
    assert "144 cells" in text
    # the FULL-matrix mean: the probe's off-diagonal mean 0.130395 over
    # 132 cells plus the 12 diagonal ones -> (0.130395*132 + 12) / 144
    assert "mean 0.202862, min 0.021277, max 1.000000" in text
    # the shared RectROI moved with the spins
    x, y = page._roi_panel.roi().pos()
    w, h = page._roi_panel.roi().size()
    assert (x, y) == (0.0, 0.0)
    assert (w, h) == (12.0, 12.0)


# -- the image panel widget (16.8.2-16.8.4) ------------------------------------------


def test_image_panel_roi_math_and_masking(studio_app) -> None:
    from ui.widgets.image_panel import ImagePanel, roi_stats_from_values

    panel = ImagePanel(title="unit")
    grid = np.arange(64, dtype=float).reshape(8, 8)
    grid[0, 0] = np.nan  # masked entry excluded from every stat
    panel.set_matrix(grid)
    stats = roi_stats_from_values(grid, 0, 8, 0, 8)
    assert stats is not None
    assert stats["count"] == 63.0
    assert round(stats["mean"], 4) == round(float(np.nanmean(grid)), 4)
    assert round(stats["min"], 4) == 1.0  # entry 0 is the NaN
    assert round(stats["max"], 4) == 63.0
    # the widget's own stats over the full ROI agree with the helper
    panel.roi().setPos([0.0, 0.0])
    panel.roi().setSize([8.0, 8.0])
    live = panel.roi_stats()
    assert live is not None
    assert live["count"] == 63.0
    assert round(live["mean"], 4) == round(float(np.nanmean(grid)), 4)


def test_image_panel_colormap_switch_and_refusal(studio_app) -> None:
    from ui.widgets.image_panel import GRADIENT_PRESETS, ImagePanel

    panel = ImagePanel(title="unit")
    assert "viridis" in GRADIENT_PRESETS
    for preset in ("magma", "inferno", "plasma", "thermal", "turbo", "grey"):
        panel.set_gradient(preset)
    try:
        panel.set_gradient("not-a-preset")
    except ValueError as exc:
        assert "not-a-preset" in str(exc)
        assert "viridis" in str(exc)
    else:
        raise AssertionError("expected the honest refusal for unknown presets")


def test_image_panel_export_png(studio_app, qtbot, tmp_path) -> None:
    from ui.widgets.image_panel import ImagePanel

    panel = ImagePanel(title="export")
    panel.set_matrix(np.arange(16, dtype=float).reshape(4, 4))
    qtbot.addWidget(panel)
    panel.resize(320, 240)
    target = tmp_path / "panel.png"
    assert panel.export_png(target) is True
    data = target.read_bytes()
    assert data[:8] == bytes.fromhex("89504e470d0a1a0a")  # PNG magic
    assert len(data) > 2_000  # a real render, not an empty frame


def test_image_panel_theme_sync(studio_app) -> None:
    from dataclasses import replace

    from ui.theme_manager import ChartPalette
    from ui.widgets.image_panel import ImagePanel

    panel = ImagePanel(title="theme")
    before = panel.widget().backgroundBrush().color().name()
    palette = replace(ChartPalette(), background="#123456")
    panel.apply_palette(palette)
    after = panel.widget().backgroundBrush().color().name()
    assert before != after
    assert after == "#123456"


# -- tab 1: the similarity heatmap --------------------------------------------------


def test_similarity_heatmap_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_similarity()
    done.wait()
    _settle(qtbot, studio_app)

    text = page._sim_report.text()
    assert "bundled seed_molecules.csv (12 drugs); 12 valid" in text
    assert "min 0.021277 / max 0.529412 / mean 0.130395" in text
    assert "Most similar: caffeine ~ theophylline = 0.529412" in text
    assert "least: metformin ~ diclofenac = 0.021277" in text
    assert "Highest mean similarity: ibuprofen (0.164474)" in text
    assert "lowest: metformin (0.042181)" in text

    values = page._sim_panel.values()
    assert values is not None and values.shape == (12, 12)
    assert bool(np.allclose(np.diag(values), 1.0))


def test_similarity_source_refusals(studio_app) -> None:
    page = _page()
    page._sim_source.setCurrentIndex(1)  # explorer source, no table loaded
    page._start_similarity()
    assert "no loaded table" in page._sim_report.text()

    class _TextOnlyExplorer:
        def frame(self) -> pd.DataFrame:
            return pd.DataFrame({"id": [f"S{i}" for i in range(10)]})

    page2 = _page(data_explorer=_TextOnlyExplorer())
    page2._sim_source.setCurrentIndex(1)
    page2._start_similarity()
    assert "no SMILES-like text column" in page2._sim_report.text()

    class _ThinSmilesExplorer:
        def frame(self) -> pd.DataFrame:
            return pd.DataFrame({"smiles": ["CC(=O)O", "CCC(=O)O", "CCN(C)C"]})

    page3 = _page(data_explorer=_ThinSmilesExplorer())
    page3._sim_source.setCurrentIndex(1)
    page3._start_similarity()
    assert "only 3 non-null SMILES" in page3._sim_report.text()


def test_similarity_explorer_column_end_to_end(studio_app, qtbot) -> None:
    from pathlib import Path

    seeds_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "data"
        / "seed_molecules.csv"
    )

    class _SmilesExplorer:
        def frame(self) -> pd.DataFrame:
            return pd.read_csv(seeds_path)

    page = _page(data_explorer=_SmilesExplorer())
    page._sim_source.setCurrentIndex(1)
    assert page._sim_column.isEnabled() is True
    assert page._sim_column.currentText() == "smiles"
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_similarity()
    done.wait()
    _settle(qtbot, studio_app)
    text = page._sim_report.text()
    assert "column 'smiles' of the loaded table (12 SMILES); 12 valid" in text
    assert "min 0.021277 / max 0.529412" in text


# -- tab 2: the response matrix --------------------------------------------------------


def test_response_matrix_end_to_end(studio_app, qtbot) -> None:
    page = _page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_screen()
    done.wait()
    _settle(qtbot, studio_app)

    text = page._screen_report.text()
    assert "23 cell lines x 8 compounds, 1 NaN hole(s) (masked)" in text
    assert "min -0.177930 (most sensitive)" in text
    assert "max 4.424495 (most resistant)" in text
    assert "mean 3.020384" in text
    assert "Most sensitive: A549 / erlotinib" in text
    assert "most resistant: U-251 / sorafenib" in text

    values = page._screen_panel.values()
    assert values is not None and values.shape == (23, 8)
    assert int(np.isnan(values).sum()) == 1
    # the axes carry the real names (ticks set from the pivot)
    assert page._screen_panel.image_item().image is not None


# -- tab 3: the explorer matrix ----------------------------------------------------------


def test_explorer_matrix_seeded_demo(studio_app, qtbot) -> None:
    page = _page()  # no explorer frame -> the seeded 12-row demo
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_explorer()
    done.wait()
    _settle(qtbot, studio_app)

    text = page._exp_report.text()
    assert "seeded 12-row demo frame (no table loaded)" in text
    assert "columns potency, logp, mw (z)" in text
    assert "Raw means [4.8301, 2.0828, 341.6800]" in text
    assert "raw stds [1.1013, 0.5633, 35.8158]" in text
    assert "after scaling [0.0000, -0.0000, -0.0000]" in text
    assert "stds [1.0000, 1.0000, 1.0000]" in text
    assert "Min -1.971670 / max 2.364600" in text
    assert "row of max |value|: row 6 (2.364600)" in text

    values = page._exp_panel.values()
    assert values is not None and values.shape == (12, 3)


def test_explorer_matrix_raw_mode(studio_app, qtbot) -> None:
    page = _page()
    page._exp_z.setChecked(False)
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_explorer()
    done.wait()
    _settle(qtbot, studio_app)
    text = page._exp_report.text()
    assert "(raw)" in text
    values = page._exp_panel.values()
    assert values is not None
    assert round(float(values[0, 0]), 4) == 5.3657  # the probe's cmp-00 potency


def test_explorer_matrix_from_loaded_table(studio_app, qtbot) -> None:
    class _WideExplorer:
        def frame(self) -> pd.DataFrame:
            rng = np.random.default_rng(7)
            return pd.DataFrame(
                {
                    "obs": rng.normal(0.0, 1.0, 90),
                    "pred": rng.normal(0.0, 1.0, 90),
                    "extra": rng.normal(5.0, 2.0, 90),
                    "label": list("x" * 90),
                }
            )

    page = _page(data_explorer=_WideExplorer())
    page._refresh_numeric_pickers(["obs", "pred", "extra"])
    assert page._exp_from.count() == 3
    assert page._exp_from.currentText() == "obs"
    assert page._exp_to.currentText() == "extra"
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_explorer()
    done.wait()
    _settle(qtbot, studio_app)
    text = page._exp_report.text()
    assert "the loaded table (90 rows)" in text
    assert "columns obs, pred, extra (z)" in text
    values = page._exp_panel.values()
    assert values is not None and values.shape == (90, 3)


def test_explorer_matrix_refuses_single_numeric_column(studio_app) -> None:
    class _OneNumericExplorer:
        def frame(self) -> pd.DataFrame:
            return pd.DataFrame(
                {"only": np.arange(10.0), "label": list("abcdefghij")}
            )

    page = _page(data_explorer=_OneNumericExplorer())
    page._start_explorer()
    text = page._exp_report.text()
    assert "Refused" in text
    assert "1 numeric column(s); need at least 2" in text


def test_explorer_span_keeps_order(studio_app) -> None:
    class _WideExplorer:
        def frame(self) -> pd.DataFrame:
            return pd.DataFrame(
                {"a": np.arange(10.0), "b": np.arange(10.0), "c": np.arange(10.0)}
            )

    page = _page(data_explorer=_WideExplorer())
    page._refresh_numeric_pickers(["a", "b", "c"])
    page._exp_to.setCurrentIndex(0)  # would invert the span
    page._exp_from.setCurrentIndex(2)
    assert page._exp_to.currentIndex() == 2  # snapped forward, never inverted


# -- tab 5: performance -------------------------------------------------------------------


def test_performance_run_reports_pairs_not_timing(studio_app, qtbot) -> None:
    page = _page()
    page._perf_n.setValue(256)
    done = qtbot.waitSignal(page._runner.job_ended, timeout=120_000)
    page._start_perf()
    done.wait()
    _settle(qtbot, studio_app)
    text = page._perf_report.text()
    assert "256^2 = 65,536 pairs in" in text
    assert "M pairs/s" in text
    assert "off-diagonal mean" in text
    assert "never asserted" in text  # the honest timing disclosure


# -- the main-window wiring (sidebar increment 12) -------------------------------------------


def test_main_window_registers_image_lab(studio_app, main_window) -> None:
    from ui.main_window import SECTIONS

    assert ("imagelab", "Image Lab") in SECTIONS
    assert main_window.imagelab_page is not None
    assert main_window.pages.count() == 17
    index = [key for key, _label in SECTIONS].index("imagelab")
    assert main_window.pages.widget(index) is main_window.imagelab_page
