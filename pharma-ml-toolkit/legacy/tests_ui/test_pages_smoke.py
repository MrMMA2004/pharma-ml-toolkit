"""Page smoke tests: every page renders with REAL backend data offscreen."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel

_TK = Path(__file__).resolve().parents[2]


def test_dashboard_shows_live_census(studio_app) -> None:
    from ui.pages.dashboard import DashboardPage

    page = DashboardPage()
    texts = " ".join(label.text() for label in page.findChildren(QLabel))
    import pharmaml

    assert pharmaml.__version__ in texts
    assert "modules" in texts
    assert "workflows" in texts
    assert "example" in texts.lower()


def test_dashboard_census_numbers_are_real() -> None:
    from ui.pages.dashboard import _example_datasets, _model_registry_count, _module_census

    modules, lines = _module_census()
    assert modules > 80  # the real package tree
    assert lines > 100_000
    assert _model_registry_count() == 33
    datasets = _example_datasets()
    assert any(name == "qsar_regression.csv" for name, _size in datasets)


def test_data_explorer_loads_real_csv(studio_app, qtbot) -> None:
    from ui.pages.data_explorer import DataExplorerPage

    page = DataExplorerPage()
    page.load_path(_TK / "examples" / "data" / "qsar_regression.csv")
    # (part8 Phase B) load_path reads on the worker; wait for the bind
    qtbot.waitUntil(lambda: page.frame() is not None, timeout=10_000)
    frame = page.frame()
    assert frame is not None and len(frame) == 260
    assert "qsar_regression.csv" in page.source()
    assert page._column_combo.count() > 0  # numeric columns offered
    assert page._stats_label.text().startswith("rows")


def test_data_explorer_reads_excel_too(studio_app, qtbot, tmp_path) -> None:
    import pandas as pd

    from ui.pages.data_explorer import DataExplorerPage

    frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "tiny.xlsx"
    frame.to_excel(path, index=False)

    page = DataExplorerPage()
    page.load_path(path)
    qtbot.waitUntil(lambda: page.frame() is not None, timeout=10_000)
    assert page.frame() is not None and len(page.frame()) == 3  # the v1.2.116 Excel guarantee, Studio-side


def test_model_page_dataset_defaults(studio_app) -> None:
    from ui.pages.model_analysis import ModelAnalysisPage

    page = ModelAnalysisPage()
    assert page._target_combo.currentText() == "pIC50"  # default target wired


def test_model_page_training_end_to_end(studio_app, qtbot) -> None:

    from ui.pages.model_analysis import ModelAnalysisPage

    page = ModelAnalysisPage()
    page._dataset_combo.setCurrentText("qsar_regression.csv")
    page._model_combo.setCurrentText("ridge")
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)
    page._start_training()
    done.wait()
    qtbot.wait(100)

    text = page._metric_label.text()
    assert "R² (held-out)" in text  # real regression metrics, real split
    assert "train 195 / test 65" in text
    # plots carry real held-out points
    items = page._pred_plot.widget().getPlotItem().listDataItems()
    assert items and any(len(it.getData()[0]) == 65 for it in items if it.getData()[0] is not None)


def test_settings_page_switches_provider_live(studio_app) -> None:
    from ui.pages.settings_page import SettingsPage
    from ui.theme_manager import ThemeManager

    theme = ThemeManager(studio_app)
    page = SettingsPage(theme)
    page._provider_combo.setCurrentText("native")
    assert theme.provider == "native"
    assert studio_app.styleSheet() == ""
    page._provider_combo.setCurrentText("qt-material")
    assert studio_app.styleSheet().strip()


def test_main_window_theme_sync_reaches_plots(studio_app) -> None:
    from ui.main_window import MainWindow
    from ui.theme_manager import ThemeManager

    window = MainWindow(ThemeManager(studio_app))
    window.theme.apply("qt-material", "dark_teal.xml")
    bg = window.data_page._plot.widget().backgroundBrush().color().name()
    assert bg.lower() == "#1d1f21"  # plot re-styled to the dark palette
    window.theme.apply("qt-material", "light_teal.xml")
    bg2 = window.data_page._plot.widget().backgroundBrush().color().name()
    assert bg2.lower() == "#ffffff"
    window.close()
