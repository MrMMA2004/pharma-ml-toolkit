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


# -- the round-228 surface: manual params, kept run, diagnostics ---------------- #


def test_manual_hyperparameters_table_follows_the_model_combo(studio_app) -> None:
    page = _fresh_page()
    keys = [page._params_table.item(row, 0).text() for row in range(page._params_table.rowCount())]
    assert keys  # random_forest's grid arrived on construction
    page._model_combo.setCurrentText("ridge")
    keys_ridge = [
        page._params_table.item(row, 0).text() for row in range(page._params_table.rowCount())
    ]
    assert keys != keys_ridge or "alpha" in keys_ridge
    # the reset button blanks every value cell
    page._params_table.item(0, 1).setText("17")
    page._clear_param_values()
    assert all(
        page._params_table.item(row, 1).text() == "" for row in range(page._params_table.rowCount())
    )
    # the parser: ints, floats, None, true/false, strings, blanks skipped
    page._model_combo.setCurrentText("random_forest")
    rows = page._params_table.rowCount()
    assert rows >= 3
    texts = ["42", "0.5", "None", "true", "rbf", ""]
    for row in range(rows):
        page._params_table.item(row, 1).setText(texts[row % len(texts)])
    params = page._collect_model_params()
    expected = {
        texts[row % len(texts)] for row in range(rows) if texts[row % len(texts)]
    }
    parsed_expectation = {
        "42": 42,
        "0.5": 0.5,
        "None": None,
        "true": True,
        "rbf": "rbf",
    }
    for token in expected:
        assert parsed_expectation[token] in params.values()
    assert "" not in params.values()  # blank cells never pin anything
    assert len(params) == len(expected)
    page.deleteLater()


def test_trained_run_is_kept_for_the_workbench_and_last_run_is_honest(studio_app, qtbot) -> None:
    page = _fresh_page()
    assert page.last_run() is None  # the honest pre-first-Train answer
    page._model_combo.setCurrentText("ridge")
    page._start_training()
    _settle(qtbot, studio_app)
    run = page.last_run()
    assert run is not None and run.estimator is not None
    assert run.X_train is not None and run.X_test is not None
    assert run.task == "regression"
    assert {"r2", "rmse", "mae"} <= set(run.metrics)
    assert run.feature_names  # the bundle's feature contract
    assert "Last run: ridge" in page._diag_status.text()
    page.deleteLater()


def test_apply_search_result_pins_registry_families_and_notes_the_rest(studio_app) -> None:
    from pharmaml.ml.models import MODEL_REGISTRY

    page = _fresh_page()
    note = page.apply_search_result("ridge", {"alpha": 2.5})
    assert note.startswith("applied: ridge")
    assert page._model_combo.currentText() == "ridge"
    pinned = {
        page._params_table.item(row, 0).text(): page._params_table.item(row, 1).text()
        for row in range(page._params_table.rowCount())
    }
    assert pinned.get("alpha") == "2.5"
    # a token the registry does not carry answers with the honest note
    ghost = page.apply_search_result("no_such_family", {})
    assert "Deep Learning page" in ghost or "not carry" in ghost
    assert "no_such_family" not in MODEL_REGISTRY
    page.deleteLater()


def test_learning_curve_uses_the_kept_run_and_renders_native(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._model_combo.setCurrentText("ridge")
    page._start_training()
    _settle(qtbot, studio_app)
    assert page.last_run() is not None
    page._start_learning_curve()
    _settle(qtbot, studio_app)
    assert "Learning curve over" in page._diag_status.text()
    assert "±" in page._diag_status.text()  # the band edges ride the honest note
    page.deleteLater()


def test_calibration_and_roc_refuse_regression_runs_honestly(studio_app, qtbot) -> None:
    page = _fresh_page()  # qsar_regression.csv + ridge: a regression run
    page._model_combo.setCurrentText("ridge")
    page._start_training()
    _settle(qtbot, studio_app)
    page._start_calibration()
    assert not page._runner.busy  # refused BEFORE any job
    assert "predict_proba" in page._diag_status.text()
    page._start_roc()
    assert not page._runner.busy
    assert "predict_proba" in page._diag_status.text()
    page._start_pr()
    assert not page._runner.busy
    page.deleteLater()


def test_classifier_diagnostics_render_calibration_roc_and_pr(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._dataset_combo.setCurrentText("qsar_classification.csv")
    page._model_combo.setCurrentText("logistic_regression")
    page._start_training()
    _settle(qtbot, studio_app)
    run = page.last_run()
    assert run is not None and run.task == "classification"
    assert "accuracy" in run.metrics
    for starter, needle in (
        (page._start_calibration, "Calibration over"),
        (page._start_roc, "macro AUC"),
        (page._start_pr, "macro average precision"),
    ):
        starter()
        _settle(qtbot, studio_app)
        assert needle in page._diag_status.text()
    page.deleteLater()
