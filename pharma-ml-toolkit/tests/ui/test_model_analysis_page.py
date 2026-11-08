"""Model Analysis page tests -- the round-226 §19 CV interface port.

The Part-3 §19 cross-validation interface migrated from the legacy
Tkinter playground to the Studio's Model Analysis page (the owner's
"everything must be in studio.py" directive).  Everything runs
offscreen against the real backend entry points: the honest
group-source detectors, the pre-flight refusals (scaffold CV without a
SMILES column, grouped strategies without a group column), the shared-
splitter leaderboard (``compare_models_cv`` over a fast two-model
panel on the bundled 260-row QSAR regression example) and the
double-click fold-by-fold report window (fold table, aggregates with
honest directions, notes, the PyQtGraph fold chart).
"""

from __future__ import annotations

import time


def _drain(pool_wait: float = 30.0) -> None:
    from PySide6.QtCore import QThreadPool

    deadline = time.monotonic() + pool_wait
    while QThreadPool.globalInstance().activeThreadCount() > 0 and time.monotonic() < deadline:
        time.sleep(0.01)


def _settle(qtbot, qapp) -> None:
    """Wait out the pool, then flush the queued result slots."""
    _drain()
    qtbot.wait(30)
    qapp.processEvents()


def _fresh_page():
    from ui.pages.model_analysis import ModelAnalysisPage

    return ModelAnalysisPage()


# -- construction ------------------------------------------------------------ #


def test_page_constructs_with_the_cv_controls(studio_app) -> None:
    page = _fresh_page()
    assert page._cv_folds.value() == 5
    assert page._cv_folds.minimum() == 2 and page._cv_folds.maximum() == 10
    assert [page._cv_strategy.itemText(i) for i in range(page._cv_strategy.count())] == [
        "auto",
        "kfold",
        "stratified",
        "group",
        "stratified_group",
        "scaffold",
    ]
    assert page._cv_strategy.currentText() == "auto"
    assert page._compare_btn.text() == "Compare models (CV)"
    assert page._board.columnCount() == 0
    page.deleteLater()


# -- the honest detectors ------------------------------------------------------ #


def test_cv_strategy_tokens_mirror_the_library() -> None:
    from pharmaml.ml.cv_results import CV_STRATEGIES
    from ui.pages.model_analysis import _CV_STRATEGIES

    assert set(_CV_STRATEGIES) - {"auto"} == set(CV_STRATEGIES)


def test_detect_group_column_is_honest_about_what_counts() -> None:
    import pandas as pd
    from ui.pages.model_analysis import detect_group_column, detect_smiles_column

    frame = pd.DataFrame(
        {
            "value": [1.0] * 8,
            "cluster_id": [1, 1, 2, 2, 3, 3, 4, 4],  # 4 groups of 2
            "row_id": range(8),  # unique-per-row: an ID, never a group
            "batch": ["x"] * 8,  # constant: never a group
        }
    )
    assert detect_group_column(frame) == "cluster_id"
    assert detect_group_column(pd.DataFrame({"value": [1.0, 2.0]})) is None
    # the SMILES detector: name-based, case-insensitive, honest None
    assert detect_smiles_column(pd.DataFrame({"SMILES": ["CCO"], "x": [1.0]})) == "SMILES"
    assert detect_smiles_column(pd.DataFrame({"value": [1.0]})) is None


# -- the pre-flight refusals (descriptors-only example) ------------------------ #


def test_scaffold_cv_without_smiles_refuses_before_the_job(studio_app) -> None:
    page = _fresh_page()  # qsar_regression.csv: descriptors-only
    page._cv_strategy.setCurrentText("scaffold")
    page._start_compare()
    assert not page._runner.busy  # the gate stopped before any job started
    assert "Bemis-Murcko" in page._board_status.text()
    page.deleteLater()


def test_group_cv_without_a_group_column_refuses_before_the_job(studio_app) -> None:
    page = _fresh_page()  # no group/cluster/series column either
    page._cv_strategy.setCurrentText("group")
    page._start_compare()
    assert not page._runner.busy
    assert "never straddle" in page._board_status.text()
    page.deleteLater()


# -- the functional path: board + double-click report -------------------------- #


def test_compare_board_carries_the_formatted_score_and_opens_the_report(
    qtbot, studio_app, monkeypatch
) -> None:
    import pharmaml.ml as ml_module
    from pharmaml.ml import compare_models_cv as real_compare
    from ui.pages.model_analysis import CVReportDialog
    from PySide6.QtWidgets import QTableWidget

    page = _fresh_page()

    def fast_compare(X, y, task, models=None, **kwargs):
        # the real engine on a fast two-model panel -- the GUI test
        # must not fan out over the whole registry.
        return real_compare(X, y, task, ["ridge", "lasso"], **kwargs)

    monkeypatch.setattr(ml_module, "compare_models_cv", fast_compare)
    page._cv_folds.setValue(3)
    page._start_compare()
    qtbot.waitSignal(page._runner.job_ended, timeout=300_000).wait()
    _settle(qtbot, studio_app)
    assert page._board_status.text() != ""

    # the board carries the §19.2 formatted column
    columns = [
        page._board.horizontalHeaderItem(c).text() for c in range(page._board.columnCount())
    ]
    score_col = columns.index("cv_score")
    model_col = columns.index("model")
    strategy_col = columns.index("strategy")
    assert page._board.rowCount() == 2
    for row in range(page._board.rowCount()):
        assert "±" in page._board.item(row, score_col).text()
        assert page._board.item(row, score_col).text().endswith("(3-fold CV)")
    assert {
        page._board.item(row, model_col).text() for row in range(page._board.rowCount())
    } == {"ridge", "lasso"}
    assert {page._board.item(row, strategy_col).text() for row in range(2)} == {"kfold"}
    assert set(page._cv_results) == {"ridge", "lasso"}

    # double-click the FIRST row (the better model) -> the report window
    opened: list[object] = []
    monkeypatch.setattr(
        CVReportDialog,
        "exec",
        lambda self, *a, **k: opened.append(self),
    )
    page._open_cv_report(page._board.item(0, model_col))
    assert len(opened) == 1
    dialog = opened[0]
    expected_model = page._board.item(0, model_col).text()
    assert dialog.windowTitle() == f"CV report — {expected_model}"
    # the fold table + the aggregates table live inside (3 fold rows)
    tables = dialog.findChildren(QTableWidget)
    fold_tables = [
        t
        for t in tables
        if t.columnCount() >= 3
        and t.horizontalHeaderItem(0) is not None
        and t.horizontalHeaderItem(0).text() == "fold"
    ]
    assert fold_tables and fold_tables[0].rowCount() == 3
    dialog.deleteLater()
    page.deleteLater()


def test_double_click_on_a_foreign_board_answers_with_the_status_hint(studio_app) -> None:
    from PySide6.QtWidgets import QTableWidgetItem

    page = _fresh_page()
    # a board row with NO §19 report behind it (e.g. a family
    # benchmark) must not open a window -- the honest status hint
    page._cv_results = {}  # nothing stored (as after a foreign filler)
    page._board.setRowCount(1)
    page._board.setColumnCount(1)
    foreign = QTableWidgetItem("ecfp")
    foreign.setData(0x0100, None)  # Qt.ItemDataRole.UserRole stays empty
    page._board.setItem(0, 0, foreign)
    page._open_cv_report(foreign)
    assert "no §19 CV report" in page._board_status.text()
    page.deleteLater()
