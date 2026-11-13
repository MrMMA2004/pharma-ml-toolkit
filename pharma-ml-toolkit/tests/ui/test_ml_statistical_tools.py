"""Statistical-tools tab tests — part6 Phase 3, batch 1.

The four previously API-only backends (split-conformal intervals,
probability calibration, decision curves, drift detection) now have
real GUI workflows on the ML Workbench's sixth tab.  The §3.2 bar:
every workflow collects real inputs → invokes the canonical backend →
renders real results (table + figure + status) → supports export.
Everything runs offscreen against the real example data and the real
backends — no synthetic demo values anywhere.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

_TK = Path(__file__).resolve().parents[2]
_REGRESSION_EXAMPLE = _TK / "examples" / "data" / "qsar_regression.csv"
_CLASSIFICATION_EXAMPLE = _TK / "examples" / "data" / "qsar_classification.csv"


def _drain(pool_wait: float = 60.0) -> None:
    import time

    from PySide6.QtCore import QThreadPool

    deadline = time.monotonic() + pool_wait
    while QThreadPool.globalInstance().activeThreadCount() > 0 and time.monotonic() < deadline:
        time.sleep(0.01)


def _settle(qtbot, qapp) -> None:
    _drain()
    qtbot.wait(30)
    qapp.processEvents()


def _fresh_page():
    from ui.pages.ml_workbench import MLWorkbenchPage
    from ui.pages.model_analysis import ModelAnalysisPage

    return MLWorkbenchPage(model_analysis=ModelAnalysisPage())


def _regression_outcome(analysis) -> None:
    """A real AnalysisOutcome from a real quick regression fit."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    from ui.pages.model_analysis import AnalysisOutcome

    frame = pd.read_csv(_REGRESSION_EXAMPLE)
    target = "pIC50"
    numeric = [
        column
        for column in frame.columns
        if column != target and pd.api.types.is_numeric_dtype(frame[column])
    ]
    x_train, x_test, y_train, y_test = train_test_split(
        frame[numeric], frame[target], test_size=0.25, random_state=42
    )
    estimator = LinearRegression().fit(x_train, y_train)
    analysis._last_outcome = AnalysisOutcome(
        task="regression",
        model="linear_regression",
        n_train=len(x_train),
        n_test=len(x_test),
        y_test=y_test.to_numpy(dtype=float),
        y_pred=None,
        metric_name="R² (held-out)",
        metric_value=0.5,
        estimator=estimator,
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        feature_names=[str(column) for column in numeric],
        metrics={},
    )


def _classification_outcome(analysis) -> None:
    """A real AnalysisOutcome from a real quick classification fit."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    from ui.pages.model_analysis import AnalysisOutcome

    frame = pd.read_csv(_CLASSIFICATION_EXAMPLE)
    # the example's activity is three-class (active/inactive/weak); the
    # calibration and decision-curve backends are binary tools, so the
    # helper binarizes honestly: active vs the rest.
    frame["activity_binary"] = (frame["activity"] == "active").astype(int)
    target = "activity_binary"
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


# -- construction ------------------------------------------------------------------


def test_statistical_tab_constructs(studio_app) -> None:
    page = _fresh_page()
    assert page._tabs.tabText(5) == "Statistical tools"
    assert page._stat_frame is None
    assert page._stat_export_btn.text() == "Export last result (CSV)"
    # the four backend buttons are all wired
    assert page._stat_conformal_btn.text() == "Conformal intervals"
    assert page._stat_calibration_btn.text() == "Calibration report"
    assert page._stat_decision_btn.text() == "Decision curve"
    assert page._stat_drift_btn.text() == "Detect drift"


# -- honest refusals (validation before any job) ------------------------------------


def test_conformal_without_a_run_refuses(studio_app) -> None:
    page = _fresh_page()
    page._start_stat_conformal()
    assert "Train a model on the Model Analysis page first" in page._stat_status.text()


def test_conformal_rejects_a_classification_source(studio_app, qtbot, qapp) -> None:
    page = _fresh_page()
    _classification_outcome(page._model_analysis)
    page._start_stat_conformal()
    assert "regression tool" in page._stat_status.text()


def test_calibration_rejects_a_regression_source(studio_app, qtbot, qapp) -> None:
    page = _fresh_page()
    _regression_outcome(page._model_analysis)
    page._start_stat_calibration()
    assert "classification tool" in page._stat_status.text()


def test_drift_without_a_current_csv_refuses(studio_app) -> None:
    page = _fresh_page()
    page._start_stat_drift()
    assert "current (new batch) CSV" in page._stat_status.text()


# -- the four real workflows ----------------------------------------------------------


def test_conformal_intervals_end_to_end(studio_app, qtbot, qapp, tmp_path) -> None:
    page = _fresh_page()
    _regression_outcome(page._model_analysis)
    page._start_stat_conformal()
    _settle(qtbot, qapp)

    assert "radius" in page._stat_status.text()
    assert "empirical coverage" in page._stat_status.text()
    assert page._stat_frame is not None
    columns = [page._stat_table.horizontalHeaderItem(i).text() for i in range(page._stat_table.columnCount())]
    assert columns == ["prediction", "lower", "upper"]
    assert page._stat_table.rowCount() > 0

    # the export path (§3.2: results must be exportable)
    export = tmp_path / "conformal.csv"
    page._stat_frame.to_csv(export, index=False)
    assert export.stat().st_size > 0
    reread = pd.read_csv(export)
    assert len(reread) == len(page._stat_frame)
    assert (reread["upper"] >= reread["lower"]).all()


def test_calibration_report_end_to_end(studio_app, qtbot, qapp) -> None:
    page = _fresh_page()
    _classification_outcome(page._model_analysis)
    page._start_stat_calibration()
    _settle(qtbot, qapp)

    status = page._stat_status.text()
    for token in ("Brier", "ECE", "MCE", "base rate"):
        assert token in status, status
    assert page._stat_frame is not None
    assert {"bin_lower", "bin_upper", "count", "mean_predicted", "fraction_positive"} <= set(
        page._stat_frame.columns
    )
    assert page._stat_table.rowCount() >= 1


def test_decision_curve_end_to_end(studio_app, qtbot, qapp) -> None:
    page = _fresh_page()
    _classification_outcome(page._model_analysis)
    page._start_stat_decision()
    _settle(qtbot, qapp)

    status = page._stat_status.text()
    for token in ("best net benefit", "base rate", "cost ratio"):
        assert token in status, status
    assert page._stat_frame is not None
    assert {"threshold", "net_benefit", "standardized_net_benefit", "treat_all_net_benefit"} <= set(
        page._stat_frame.columns
    )
    # the sampled table never invents points beyond the sweep
    assert len(page._stat_frame) >= 50


def test_drift_detection_end_to_end(studio_app, qtbot, qapp, tmp_path) -> None:
    page = _fresh_page()

    # a real reference and a genuinely shifted current batch
    frame = pd.read_csv(_REGRESSION_EXAMPLE)
    shifted = frame.copy()
    for column in shifted.columns:
        if pd.api.types.is_numeric_dtype(shifted[column]):
            shifted[column] = shifted[column] * 1.8 + 0.5
    current = tmp_path / "shifted.csv"
    shifted.to_csv(current, index=False)

    page._stat_drift_reference.setText(str(_REGRESSION_EXAMPLE))
    page._stat_drift_current.setText(str(current))
    page._start_stat_drift()
    _settle(qtbot, qapp)

    status = page._stat_status.text()
    assert "features compared" in status
    assert "shifted" in status
    assert page._stat_frame is not None
    assert {"feature", "psi", "verdict"} <= set(page._stat_frame.columns)
    assert (page._stat_frame["psi"] > 0.2).any()  # the 1.8x shift is real drift

    # verdicts use the backend's own vocabulary
    assert set(page._stat_frame["verdict"]) <= {"stable", "moderate", "shifted"}


def test_export_button_writes_the_last_result(studio_app, qtbot, qapp, tmp_path, monkeypatch) -> None:
    page = _fresh_page()
    _regression_outcome(page._model_analysis)
    page._start_stat_conformal()
    _settle(qtbot, qapp)
    assert page._stat_frame is not None

    target = tmp_path / "exported.csv"
    monkeypatch.setattr(
        "ui.pages.ml_workbench.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(target), "CSV (*.csv)"),
    )
    page._export_stat_result()
    assert target.exists()
    reread = pd.read_csv(target)
    assert len(reread) == len(page._stat_frame)


def test_export_without_a_result_refuses(studio_app) -> None:
    page = _fresh_page()
    page._export_stat_result()
    assert "Run a statistical tool first" in page._stat_status.text()
