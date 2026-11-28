"""The shared backend-figure renderer (part8 §4.1/§4.2) — tests.

Covers the FigureCanvasHost contract (original figure, states, errors,
leak-free replacement, export), the plot-coverage registry (every
public viz function recorded, wired names valid), and the Image Lab's
coverage browser end-to-end (a real dataset reaches a real rendered
backend figure).
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

# -- §4.1: the host -----------------------------------------------------------

class TestFigureCanvasHost:
    def test_starts_empty(self, qtbot) -> None:
        from ui.widgets.figure_canvas_host import FigureCanvasHost

        host = FigureCanvasHost()
        qtbot.addWidget(host)
        assert host.state() == "empty"
        assert host.has_figure() is False
        assert host.error() is None

    def test_adopts_the_original_backend_figure(self, qtbot) -> None:
        from ui.widgets.figure_canvas_host import FigureCanvasHost

        from pharmaml.viz.ml_plots import plot_cv_scores

        figure = plot_cv_scores([0.7, 0.75, 0.8], metric="R2")
        host = FigureCanvasHost()
        qtbot.addWidget(host)
        host.set_figure(figure)
        assert host.state() == "figure"
        assert host.has_figure() is True
        # §4.1: the ORIGINAL figure object, not a copy
        assert host._figure is figure

    def test_exception_lands_in_error_state(self, qtbot) -> None:
        from ui.widgets.figure_canvas_host import FigureCanvasHost

        host = FigureCanvasHost()
        qtbot.addWidget(host)

        def broken() -> object:
            raise ValueError("boom from the backend")

        host.run_callable(broken)
        assert host.state() == "error"
        assert "boom from the backend" in (host.error() or "")
        assert host.has_figure() is False

    def test_replacement_releases_the_old_figure(self, qtbot) -> None:
        from ui.widgets.figure_canvas_host import FigureCanvasHost

        from pharmaml.viz.ml_plots import plot_cv_scores

        first = plot_cv_scores([0.7, 0.75, 0.8], metric="R2")
        second = plot_cv_scores([0.6, 0.65], metric="R2")
        host = FigureCanvasHost()
        qtbot.addWidget(host)
        host.set_figure(first)
        host.set_figure(second)
        assert host._figure is second
        # the released figure was closed (no leak): a closed figure
        # raises on further use, so assert via matplotlib's registry
        import matplotlib.pyplot as plt

        assert not plt.fignum_exists(first.number)
        assert plt.fignum_exists(second.number)

    def test_clear_returns_to_empty(self, qtbot) -> None:
        from ui.widgets.figure_canvas_host import FigureCanvasHost

        from pharmaml.viz.ml_plots import plot_cv_scores

        host = FigureCanvasHost()
        qtbot.addWidget(host)
        host.set_figure(plot_cv_scores([0.7], metric="R2"))
        host.clear()
        assert host.state() == "empty"
        assert host.has_figure() is False

    def test_rebuild_reruns_the_stored_builder(self, qtbot) -> None:
        from ui.widgets.figure_canvas_host import FigureCanvasHost

        from pharmaml.viz.ml_plots import plot_cv_scores

        host = FigureCanvasHost()
        qtbot.addWidget(host)
        host.run_callable(lambda: plot_cv_scores([0.7, 0.8], metric="R2"))
        assert host.state() == "figure"
        assert host.rebuild() is True
        assert host.state() == "figure"

    def test_export_through_the_existing_path(self, qtbot, tmp_path) -> None:
        from ui.widgets.figure_canvas_host import FigureCanvasHost

        from pharmaml.viz.ml_plots import plot_cv_scores

        host = FigureCanvasHost()
        qtbot.addWidget(host)
        host.set_figure(plot_cv_scores([0.7, 0.8], metric="R2"))
        out = tmp_path / "cv.png"
        saved = host.export(str(out))
        assert out.exists() and out.stat().st_size > 0
        assert saved  # the resolved path comes back

    def test_temp_png_display_path(self, qtbot, tmp_path) -> None:
        # statplots closes its figures: render one to a real file first
        from ui.widgets.figure_canvas_host import FigureCanvasHost

        from pharmaml.viz import statplots

        png = tmp_path / "h.png"
        statplots.plot_histogram(pd.Series([1.0, 2.0, 2.5, 3.0, 7.0]), str(png))
        host = FigureCanvasHost()
        qtbot.addWidget(host)
        host.set_rendered_image(str(png))
        assert host.state() == "figure"
        assert host.has_figure() is True
        host.clear()
        assert host.state() == "empty"


# -- §4.2: the registry ---------------------------------------------------------

class TestPlotCoverageRegistry:
    def test_every_module_is_recorded(self) -> None:
        from pharmaml.studio.plot_coverage import PLOT_MODULES, plot_coverage_table

        table = plot_coverage_table()
        modules = {e.module for e in table}
        assert modules == set(PLOT_MODULES)

    def test_the_full_public_api_is_covered(self) -> None:
        from pharmaml.studio.plot_coverage import plot_coverage_table

        table = plot_coverage_table()
        seen = {e.function for e in table}
        for module_name in (
            "chem_plots", "docking_plots", "generative_plots", "md_plots",
            "ml_plots", "multiobjective_plots", "pharma_plots", "statplots",
        ):
            module = importlib.import_module(f"pharmaml.viz.{module_name}")
            for name in getattr(module, "__all__", ()):
                assert f"{module_name}.{name}" in seen, name

    def test_wired_names_are_real_exports(self) -> None:
        from pharmaml.studio.plot_coverage import _NATIVE, _WIRED

        for table in (_WIRED, _NATIVE):
            for module_name, names in table.items():
                module = importlib.import_module(f"pharmaml.viz.{module_name}")
                exports = set(module.__all__)
                for name in names:
                    assert name in exports, f"{module_name}.{name}"

    def test_counts_are_consistent(self) -> None:
        from pharmaml.studio.plot_coverage import (
            BACKEND_ONLY,
            NATIVE_EQUIVALENT,
            WIRED,
            coverage_counts,
            plot_coverage_table,
        )

        counts = coverage_counts()
        table = plot_coverage_table()
        assert counts["total"] == len(table)
        assert counts[WIRED] == sum(1 for e in table if e.status == WIRED)
        assert counts[NATIVE_EQUIVALENT] == sum(
            1 for e in table if e.status == NATIVE_EQUIVALENT
        )
        assert counts[BACKEND_ONLY] == sum(
            1 for e in table if e.status == BACKEND_ONLY
        )
        assert counts["total"] == counts[WIRED] + counts[NATIVE_EQUIVALENT] + counts[BACKEND_ONLY]

    def test_high_value_plots_are_wired(self) -> None:
        from pharmaml.studio.plot_coverage import plot_coverage_table

        wired = {e.function for e in plot_coverage_table() if e.status == "wired"}
        for expected in (
            "ml_plots.plot_confusion_matrix",
            "ml_plots.plot_roc_curves",  # wired status, artifact-gated in GUI
            "ml_plots.plot_applicability_domain",
            "statplots.plot_histogram",
            "pharma_plots.plot_concentration_profile",
        ):
            assert expected in wired, expected

    def test_entries_carry_titles_and_workspaces(self) -> None:
        from pharmaml.studio.plot_coverage import plot_coverage_table

        for entry in plot_coverage_table():
            assert entry.title and entry.title != "(undocumented)", entry.function
            assert entry.workspace, entry.function


# -- the Image Lab surface --------------------------------------------------------

class TestCoverageBrowser:
    @pytest.fixture()
    def loaded_lab(self, qtbot):
        from PySide6.QtCore import QEventLoop, QTimer
        from ui.pages.data_explorer import DataExplorerPage
        from ui.pages.image_lab import ImageLabPage

        explorer = DataExplorerPage()
        lab = ImageLabPage(data_explorer=explorer)
        qtbot.addWidget(lab)
        explorer._load_builtin("qsar_classification.csv")
        loop = QEventLoop()
        QTimer.singleShot(600, loop.quit)
        loop.exec()
        assert explorer.frame() is not None
        return lab

    def test_the_tab_exists_with_the_full_table(self, loaded_lab) -> None:
        labels = [loaded_lab._tabs.tabText(i) for i in range(loaded_lab._tabs.count())]
        assert "Backend figures" in labels
        table = loaded_lab._coverage_tab._table
        assert table.rowCount() >= 60  # the whole public viz API

    def test_a_wired_plot_renders_from_the_loaded_frame(self, loaded_lab) -> None:
        tab = loaded_lab._coverage_tab
        for i in range(tab._table.rowCount()):
            if tab._table.item(i, 0).text() == "ml_plots.plot_correlation_heatmap":
                tab._table.setCurrentCell(i, 0)
                break
        assert tab.launch_current() is True
        assert tab._host.state() == "figure"

    def test_statplots_render_via_the_artifact_path(self, loaded_lab) -> None:
        tab = loaded_lab._coverage_tab
        for i in range(tab._table.rowCount()):
            if tab._table.item(i, 0).text() == "statplots.plot_histogram":
                tab._table.setCurrentCell(i, 0)
                break
        assert tab.launch_current() is True
        assert tab._host.state() == "figure"

    def test_a_real_fit_renders_the_learning_curve(self, loaded_lab) -> None:
        tab = loaded_lab._coverage_tab
        for i in range(tab._table.rowCount()):
            if tab._table.item(i, 0).text() == "ml_plots.plot_learning_curve":
                tab._table.setCurrentCell(i, 0)
                break
        assert tab.launch_current() is True
        assert tab._host.state() == "figure"

    def test_applicability_domain_renders_the_real_analysis(self, loaded_lab) -> None:
        tab = loaded_lab._coverage_tab
        for i in range(tab._table.rowCount()):
            if tab._table.item(i, 0).text() == "ml_plots.plot_applicability_domain":
                tab._table.setCurrentCell(i, 0)
                break
        assert tab.launch_current() is True
        assert tab._host.state() == "figure"

    def test_artifact_gated_plots_refuse_honestly(self, loaded_lab) -> None:
        tab = loaded_lab._coverage_tab
        for i in range(tab._table.rowCount()):
            if tab._table.item(i, 0).text() == "ml_plots.plot_calibration_curve":
                tab._table.setCurrentCell(i, 0)
                break
        assert tab.launch_current() is False
        assert tab._host.state() == "empty"
        assert "cannot supply" in tab._status_line.text() or "no GUI" in tab._status_line.text()

    def test_backend_only_rows_explain_themselves(self, loaded_lab) -> None:
        tab = loaded_lab._coverage_tab
        # a known backend-only function
        for i in range(tab._table.rowCount()):
            if tab._table.item(i, 0).text() == "ml_plots.plot_edf":
                tab._table.setCurrentCell(i, 0)
                break
        assert tab.launch_current() is False
        assert "backend" in tab._status_line.text()

    def test_without_a_dataset_nothing_renders(self, qtbot) -> None:
        from ui.widgets.coverage_browser import CoverageBrowser

        browser = CoverageBrowser(frame_provider=lambda: None)
        qtbot.addWidget(browser)
        for i in range(browser._table.rowCount()):
            if browser._table.item(i, 0).text() == "statplots.plot_histogram":
                browser._table.setCurrentCell(i, 0)
                break
        assert browser.launch_current() is False
        assert browser._host.state() == "empty"
