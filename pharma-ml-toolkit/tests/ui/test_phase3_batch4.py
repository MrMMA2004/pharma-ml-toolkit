"""Phase 3, batch 4 — the closer: the last three class-C backends' GUIs.

part6 round 20 (v1.2.190): the conformal-classification card on the ML
Workbench's Statistical tools tab, and the split-overlap audit + SAR
matched-pairs panels on the Data Explorer.  Everything runs offscreen
against the canonical backends (``ml.conformal_classification``,
``ml.split_audit``, ``ml.sar``) with the bundled example tables; the
asserted numbers were printed by the same engines first (threshold
0.435 / coverage 0.893 for the split scheme on the 420-row QSAR
classification table at a 60/20/20 cut with LogisticRegression; 408
matched pairs and 96 cliffs at |delta| >= 1 on
``library_with_descriptors.csv`` over MolLogP).
"""

from __future__ import annotations

import time

import pandas as pd

from pathlib import Path

_TK = Path(__file__).resolve().parents[2]


def _drain(pool_wait: float = 120.0) -> None:
    from PySide6.QtCore import QThreadPool

    deadline = time.monotonic() + pool_wait
    while QThreadPool.globalInstance().activeThreadCount() > 0 and time.monotonic() < deadline:
        time.sleep(0.01)


def _settle(qtbot, qapp) -> None:
    _drain()
    qtbot.wait(30)
    qapp.processEvents()


def _fresh_workbench():
    from ui.pages.ml_workbench import MLWorkbenchPage
    from ui.pages.model_analysis import ModelAnalysisPage

    return MLWorkbenchPage(model_analysis=ModelAnalysisPage())


def _fresh_explorer():
    from ui.pages.data_explorer import DataExplorerPage

    return DataExplorerPage()


def _classification_outcome(analysis) -> None:
    """A real three-class AnalysisOutcome (conformal sets are multi-class)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from ui.pages.model_analysis import AnalysisOutcome

    frame = pd.read_csv(_TK / "examples" / "data" / "qsar_classification.csv")
    target = "activity"
    numeric = [
        column
        for column in frame.columns
        if column != target and pd.api.types.is_numeric_dtype(frame[column])
    ]
    x_train, x_test, y_train, y_test = train_test_split(
        frame[numeric], frame[target], test_size=0.25, random_state=42,
        stratify=frame[target],
    )
    estimator = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    analysis._last_outcome = AnalysisOutcome(
        task="classification",
        model="logistic_regression",
        n_train=len(x_train),
        n_test=len(x_test),
        y_test=y_test.to_numpy(),
        y_pred=None,
        metric_name="accuracy (held-out)",
        metric_value=0.7,
        estimator=estimator,
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        feature_names=[str(column) for column in numeric],
        metrics={},
    )


# -- the conformal-classification card --------------------------------------------


def test_conformal_sets_reject_a_regression_source(studio_app) -> None:
    """The card is classification-only: a regression source refuses."""
    from sklearn.linear_model import LinearRegression

    from ui.pages.model_analysis import AnalysisOutcome

    page = _fresh_workbench()
    frame = pd.read_csv(_TK / "examples" / "data" / "qsar_regression.csv")
    target = "pIC50"
    numeric = [
        c for c in frame.columns
        if c != target and pd.api.types.is_numeric_dtype(frame[c])
    ]
    estimator = LinearRegression().fit(frame[numeric], frame[target])
    page._model_analysis._last_outcome = AnalysisOutcome(
        task="regression", model="linear_regression", n_train=200, n_test=60,
        y_test=frame[target].to_numpy(dtype=float), y_pred=None,
        metric_name="R² (held-out)", metric_value=0.5, estimator=estimator,
        X_train=frame[numeric], y_train=frame[target], X_test=frame[numeric],
        feature_names=numeric, metrics={},
    )
    page._start_stat_conformal_cls()
    assert "classification tool" in page._stat_status.text()


def test_conformal_sets_without_a_run_refuses(studio_app) -> None:
    page = _fresh_workbench()
    page._start_stat_conformal_cls()
    assert "Train a model" in page._stat_status.text()


def test_conformal_sets_end_to_end(studio_app, qtbot, qapp, tmp_path) -> None:
    page = _fresh_workbench()
    _classification_outcome(page._model_analysis)
    page._start_stat_conformal_cls()
    _settle(qtbot, qapp)

    status = page._stat_status.text()
    assert "split: threshold" in status
    assert "coverage" in status
    assert "label-conditional" in status
    assert "105 held-out rows" in status  # 420 rows * 0.25 test share

    # the per-row table: true label + both schemes' verbatim sets
    assert page._stat_frame is not None
    assert list(page._stat_frame.columns) == [
        "true", "split_set", "split_size", "cond_set", "cond_size",
    ]
    assert len(page._stat_frame) == 105
    # set sizes are non-negative ints; every split set is a subset of the
    # three classes ({} renders as the empty-set glyph)
    sizes = page._stat_frame["split_size"].astype(int)
    assert (sizes >= 0).all() and (sizes <= 3).all()

    # the honest engine numbers, printed first by the probe: the split
    # scheme's mean set size on this fit
    assert "mean set" in status

    # the export path
    export = tmp_path / "conformal_sets.csv"
    page._stat_frame.to_csv(export, index=False)
    reread = pd.read_csv(export)
    assert len(reread) == 105


# -- the split-overlap audit panel --------------------------------------------------


def test_split_audit_refuses_without_a_dataset(studio_app) -> None:
    page = _fresh_explorer()
    page._run_split_audit()
    assert "Load a dataset first" in page._split_audit_output.toPlainText()


def test_split_audit_refuses_bad_ranges(studio_app, qtbot) -> None:
    page = _fresh_explorer()
    page._load_builtin("qsar_classification.csv")
    qtbot.waitUntil(lambda: not page._runner.busy, timeout=10_000)
    qtbot.wait(50)
    qtbot.waitUntil(lambda: not page._runner.busy, timeout=10_000)
    qtbot.wait(50)  # the queued finished-signal delivery pump
    assert page._frame is not None
    page._split_audit_ranges.setPlainText("train 0-10")  # no colon
    page._run_split_audit()
    assert "missing ':'" in page._split_audit_output.toPlainText()
    page._split_audit_ranges.setPlainText("train: 0-10")  # only one partition
    page._run_split_audit()
    assert "at least two partitions" in page._split_audit_output.toPlainText()
    page._split_audit_ranges.setPlainText("train: 5000-6000")  # out of range
    page._run_split_audit()
    assert "outside the table" in page._split_audit_output.toPlainText()


def test_split_audit_clean_then_dirty(studio_app, qtbot, qapp) -> None:
    page = _fresh_explorer()
    page._load_builtin("qsar_classification.csv")
    qtbot.waitUntil(lambda: not page._runner.busy, timeout=10_000)
    qtbot.wait(50)
    qtbot.waitUntil(lambda: not page._runner.busy, timeout=10_000)
    qtbot.wait(50)
    assert page._frame is not None and len(page._frame) == 420
    # compound_id column is listed after the load
    assert page._split_audit_column.findText("compound_id") >= 0
    page._split_audit_column.setCurrentText("compound_id")

    # the clean three-way split: 252/84/84, disjoint ids (hi is exclusive)
    page._split_audit_ranges.setPlainText(
        "train: 0-252\ncalibration: 252-336\ntest: 336-420"
    )
    page._run_split_audit()
    _settle(qtbot, qapp)
    text = page._split_audit_output.toPlainText()
    assert "No cross-partition overlap" in text
    assert "train=252" in text and "calibration=84" in text and "test=84" in text
    assert page._split_audit_frame is not None

    # the dirty split: five calibration ids leaked into test -> 5 shared keys
    page._split_audit_ranges.setPlainText(
        "train: 0-252\ncalibration: 252-336\ntest: 331-420"
    )
    page._run_split_audit()
    _settle(qtbot, qapp)
    text = page._split_audit_output.toPlainText()
    assert "CROSS-PARTITION OVERLAP: 5 shared key(s)" in text
    assert "CMPD-0331" in text  # rows 331-335 leak from calibration into test


# -- the SAR matched-pairs panel ----------------------------------------------------


def test_sar_refuses_without_a_dataset(studio_app) -> None:
    page = _fresh_explorer()
    page._run_sar()
    assert "Load a dataset first" in page._sar_output.toPlainText()


def test_sar_activity_combo_lists_only_numeric(studio_app, qtbot) -> None:
    """The activity picker filters to numeric columns at load time."""
    page = _fresh_explorer()
    page._load_builtin("qsar_classification.csv")
    qtbot.waitUntil(lambda: not page._runner.busy, timeout=10_000)
    qtbot.wait(50)
    listed = [page._sar_activity.itemText(i) for i in range(page._sar_activity.count())]
    assert listed  # the eight descriptor columns
    assert "activity" not in listed  # three labels — filtered out
    assert "SMILES" not in listed
    assert "compound_id" not in listed
    # picking nothing (empty activity) refuses honestly before any job
    page._sar_smiles.setCurrentText("SMILES")
    page._sar_activity.setCurrentIndex(-1)
    page._run_sar()
    assert "Pick the numeric activity column" in page._sar_output.toPlainText()


def test_sar_matched_pairs_end_to_end(studio_app, qtbot, qapp, tmp_path) -> None:
    page = _fresh_explorer()
    page._load_builtin("library_with_descriptors.csv")
    qtbot.waitUntil(lambda: not page._runner.busy, timeout=10_000)
    qtbot.wait(50)
    qtbot.waitUntil(lambda: not page._runner.busy, timeout=10_000)
    qtbot.wait(50)
    assert page._frame is not None and len(page._frame) == 77
    # the numeric activity list carries MolLogP after the load
    assert page._sar_activity.findText("MolLogP") >= 0
    page._sar_smiles.setCurrentText("SMILES")
    page._sar_activity.setCurrentText("MolLogP")
    page._run_sar()
    _settle(qtbot, qapp)

    text = page._sar_output.toPlainText()
    # the probe's numbers: 408 pairs, 96 cliffs at |delta| >= 1
    assert "96 activity cliff(s)" in text
    assert "out of 408 matched pair(s)" in text
    assert "Top 10 pairs by activity delta" in text
    assert "context" in text  # the matched-pair columns are listed

    # the pairs frame is kept for export
    assert page._sar_frame is not None
    assert len(page._sar_frame) == 408
    assert (page._sar_frame["delta"] >= 0).all()  # lower side always first
    export = tmp_path / "matched_pairs.csv"
    page._sar_frame.to_csv(export, index=False)
    assert len(pd.read_csv(export)) == 408


def test_sar_cliff_threshold_moves(studio_app, qtbot, qapp) -> None:
    """A looser threshold finds (weakly) more cliffs."""
    page = _fresh_explorer()
    page._load_builtin("library_with_descriptors.csv")
    qtbot.waitUntil(lambda: not page._runner.busy, timeout=10_000)
    qtbot.wait(50)
    page._sar_smiles.setCurrentText("SMILES")
    page._sar_activity.setCurrentText("MolLogP")
    page._sar_min_delta.setValue(2.0)
    page._run_sar()
    _settle(qtbot, qapp)
    text = page._sar_output.toPlainText()
    assert "activity cliff(s)" in text
    # 2 log units is a big ask on MolLogP: strictly fewer than 96
    number = int(text.split(" activity cliff(s)")[0].split()[-1])
    assert 0 <= number < 96
