"""GUI tests for the Table Tools page (part8 r9 — the STRETCH-card wiring).

These pin the wiring, not the science (tests/test_reshape.py does
that): the page rides the Explorer's shared frame, every operation
submits through the JobRunner (never the GUI thread), the result
previews in the table, the report line is honest, "Bind as active
dataset" announces frameLoaded, and backend validation errors surface
verbatim in the status area.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from ui.pages.data_explorer import DataExplorerPage
from ui.pages.table_tools import TableToolsPage
from ui.workers.job_worker import JobRunner


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def runner(qapp) -> JobRunner:
    return JobRunner()


@pytest.fixture
def explorer(qapp, runner) -> DataExplorerPage:
    return DataExplorerPage(runner=runner)


@pytest.fixture
def page(qapp, runner, explorer) -> TableToolsPage:
    page = TableToolsPage(runner=runner, data_explorer=explorer)
    yield page
    if page._runner.busy:
        page._runner.cancel()


LONG_FRAME = pd.DataFrame({
    "compound": ["A", "A", "B", "B", "C", "C"],
    "batch": ["b1", "b2", "b1", "b2", "b1", "b2"],
    "yield": [10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
    "ic50": [1.5, 1.6, 2.5, 2.6, 3.5, 3.6],
})


def _settle(qapp, runner: JobRunner, ticks: int = 80) -> None:
    """Pump the event loop until the runner finishes (or ticks run out)."""
    import time

    for _ in range(ticks):
        qapp.processEvents()
        if not runner.busy:
            break
        time.sleep(0.01)
    qapp.processEvents()


def _load(page: TableToolsPage, frame: pd.DataFrame) -> None:
    page._on_explorer_frame(frame)


class TestWiring:
    def test_tab_set_carries_all_seven_operations(self, page):
        labels = [page._tabs.tabText(i) for i in range(page._tabs.count())]
        assert labels == [
            "Combine", "Merge", "GroupBy", "Pivot",
            "Clean text", "Time series", "Filter",
        ]

    def test_page_consumes_explorer_shared_frame(self, page, explorer):
        explorer.frameLoaded.emit(LONG_FRAME)
        assert page.frame() is LONG_FRAME
        assert "6 rows" in page._source_label.text()

    def test_column_combos_refresh_on_frame_load(self, page):
        _load(page, LONG_FRAME)
        items = [page._groupby_key.itemText(i) for i in range(page._groupby_key.count())]
        assert items == ["compound", "batch", "yield", "ic50"]

    def test_numeric_pickers_only_numeric(self, page):
        _load(page, LONG_FRAME)
        items = [page._groupby_value.itemText(i) for i in range(page._groupby_value.count())]
        assert items == ["yield", "ic50"]

    def test_text_picker_prefers_non_numeric(self, page):
        _load(page, LONG_FRAME)
        items = [page._strings_column.itemText(i) for i in range(page._strings_column.count())]
        assert "compound" in items and "ic50" not in items

    def test_no_frame_notes_honestly(self, page):
        page._run_groupby()
        assert "No dataset loaded" in page._status.toPlainText()


class TestOperations:
    def test_groupby_runs_through_worker_and_previews(self, qapp, page, runner):
        _load(page, LONG_FRAME)
        page._groupby_key.setCurrentText("compound")
        page._groupby_value.setCurrentText("yield")
        page._groupby_aggs.setText("mean, count")
        page._run_groupby()
        assert runner.busy or page._result is not None  # submitted
        _settle(qapp, runner)
        assert page._result is not None
        assert len(page._result) == 3  # one row per compound
        assert "groupby: 6×4 → 3×3" in page._status.toPlainText()
        assert page._bind_btn.isEnabled()

    def test_filter_runs_and_reports(self, qapp, page, runner):
        _load(page, LONG_FRAME)
        page._filter_expression.setText("ic50 > 2.0")
        page._run_filter()
        _settle(qapp, runner)
        assert len(page._result) == 4
        assert "kept 4 of 6 rows" in page._status.toPlainText()

    def test_filter_error_surfaces_backend_message(self, qapp, page, runner):
        _load(page, LONG_FRAME)
        page._filter_expression.setText("yield >")
        page._run_filter()
        _settle(qapp, runner)
        text = page._status.toPlainText()
        assert "filter failed:" in text and "invalid filter expression" in text
        assert not page._bind_btn.isEnabled()

    def test_pivot_previews_wide_result(self, qapp, page, runner):
        _load(page, LONG_FRAME)
        page._pivot_index.setCurrentText("compound")
        page._pivot_columns.setCurrentText("batch")
        page._pivot_values.setCurrentText("yield")
        page._run_pivot()
        _settle(qapp, runner)
        assert list(page._result.columns) == ["compound", "yield_b1", "yield_b2"]
        assert "pivot" in page._status.toPlainText()

    def test_strings_clean_reports_changed(self, qapp, page, runner):
        frame = LONG_FRAME.assign(compound=["  A ", " A", " B ", "B", "  C", "C"])
        _load(page, frame)
        page._strings_column.setCurrentText("compound")
        page._run_strings()
        _settle(qapp, runner)
        assert page._result["compound"].tolist() == ["A", "A", "B", "B", "C", "C"]
        # 4 of 6 values actually changed ("B" and "C" were already clean)
        assert "4 values changed" in page._status.toPlainText()

    def test_merge_needs_second_table_note(self, qapp, page, runner):
        _load(page, LONG_FRAME)
        page._run_merge()
        assert "needs a second table" in page._status.toPlainText()

    def test_combine_needs_second_table_note(self, qapp, page, runner):
        _load(page, LONG_FRAME)
        page._run_combine()
        assert "needs a second table" in page._status.toPlainText()

    def test_resample_invalid_rule_surfaces_error(self, qapp, page, runner):
        frame = pd.DataFrame({
            "time": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "conc": [1.0, 2.0],
        })
        _load(page, frame)
        page._timeseries_time.setCurrentText("time")
        page._timeseries_value.setCurrentText("conc")
        page._resample_rule.setEditable(True)  # already editable; set arbitrary text
        page._resample_rule.setCurrentText("5 minn")
        assert page._resample_rule.currentText() == "5 minn"
        page._run_resample()
        _settle(qapp, runner)
        assert "invalid resample rule" in page._status.toPlainText()


class TestSecondTable:
    def test_second_frame_merge_end_to_end(self, qapp, page, runner):
        _load(page, LONG_FRAME)
        second = pd.DataFrame({
            "compound": ["A", "B", "D"],
            "solubility": [1.1, 2.2, 3.3],
        })
        page._second_frame = second
        page._second_source = "second.csv"
        page._second_combo_refresh()
        page._merge_key.setCurrentText("compound")
        page._run_merge()
        _settle(qapp, runner)
        assert len(page._result) == 6  # left merge keeps all 6 left rows
        assert "solubility" in page._result.columns
        assert "shared key values" in page._status.toPlainText()

    def test_combine_stacks_with_source_column(self, qapp, page, runner):
        _load(page, LONG_FRAME)
        page._second_frame = LONG_FRAME.head(3)
        page._combine_keys.setText("lab_a, lab_b")
        page._run_combine()
        _settle(qapp, runner)
        assert len(page._result) == 9
        assert set(page._result["source_frame"]) == {"lab_a", "lab_b"}


class TestBind:
    def test_bind_announces_frameLoaded(self, qapp, page, runner):
        _load(page, LONG_FRAME)
        page._filter_expression.setText("compound == 'A'")
        page._run_filter()
        _settle(qapp, runner)
        assert page._result is not None

        announced: list[object] = []
        page.frameLoaded.connect(announced.append)
        page._bind_btn.click()
        assert announced == [page._result]
        assert page.frame() is page._result
        assert "Bound as the active dataset" in page._status.toPlainText()

    def test_bind_disabled_without_result(self, page):
        _load(page, LONG_FRAME)
        assert not page._bind_btn.isEnabled()

    def test_failed_operation_disables_bind(self, qapp, page, runner):
        _load(page, LONG_FRAME)
        page._filter_expression.setText("nope > 1")
        page._run_filter()
        _settle(qapp, runner)
        assert not page._bind_btn.isEnabled()
        assert page._result is None


class TestWorkspaceIntegration:
    def test_page_lives_in_datasets_workspace(self, qapp):
        from ui.main_window import MainWindow

        mw = MainWindow()
        try:
            host = mw.workspaces["datasets"]
            labels = [host.tabText(i) for i in range(host.count())]
            assert "Table Tools" in labels
            assert labels.index("Table Tools") == 1  # second tab
            # the shared runner wiring: same JobRunner as the window
            assert mw.table_tools_page._runner is mw.runner
        finally:
            mw.deleteLater()

    def test_explorer_frame_flows_into_page(self, qapp):
        from ui.main_window import MainWindow

        mw = MainWindow()
        try:
            mw.data_page._bind_loaded_frame(
                __import__("pathlib").Path("test.csv"), LONG_FRAME
            )
            assert mw.table_tools_page.frame() is LONG_FRAME
        finally:
            mw.deleteLater()
