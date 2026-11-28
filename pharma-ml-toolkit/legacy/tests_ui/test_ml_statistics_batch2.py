"""ML statistics batch-2 tests — part6 Phase 3, batch 2.

The remaining ML statistics cluster (model comparison, multiplicity,
power, coverage diagnostics, subgroups, model cards) gets its §3.2
workflows: real inputs → canonical backend → rendered results →
export.  Everything runs offscreen against the real example data —
the comparison panel really cross-validates, the coverage audit
really follows a conformal run, and the model card really writes from
a real saved bundle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_TK = Path(__file__).resolve().parents[2]
_REGRESSION_EXAMPLE = _TK / "examples" / "data" / "qsar_regression.csv"


def _drain(pool_wait: float = 120.0) -> None:
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

    page = MLWorkbenchPage(model_analysis=ModelAnalysisPage())
    # the comparison/coverage/subgroup workflows read the CURRENT
    # dataset via the Analysis page's loader: point it at the
    # regression example the same way its own tests do.
    page._model_analysis._dataset_combo.setCurrentText("qsar_regression.csv")
    page._model_analysis._target_combo.setCurrentText("pIC50")
    return page


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


# -- construction ------------------------------------------------------------------


def test_batch2_form_constructs(studio_app) -> None:
    page = _fresh_page()
    assert page._stat_models.text() == "ridge, random_forest, gradient_boosting"
    assert page._stat_folds.value() == 5
    assert page._stat_correction.currentText() == "fdr_bh"
    assert page._stat_mingroup.value() == 10
    assert page._stat_effect.value() == 0.0
    assert page._stat_conformal_last is None
    # the grouping combo lists the loaded dataset's columns
    combo_texts = {page._stat_group_column.itemText(i) for i in range(page._stat_group_column.count())}
    assert "descr_000" in combo_texts
    assert page._persist_card_btn.text() == "Write model card…"


# -- honest refusals -----------------------------------------------------------------


def test_comparison_without_a_dataset_refuses(studio_app) -> None:
    page = _fresh_page()
    page._model_analysis._dataset_combo.setCurrentText("qsar_regression.csv")
    # simulate the no-dataset state the loader guards
    page._training_context = lambda: (None, "", "")  # type: ignore[assignment]
    page._start_stat_comparison()
    assert "Load a dataset" in page._stat_status.text()


def test_comparison_with_one_model_refuses(studio_app) -> None:
    page = _fresh_page()
    page._stat_models.setText("ridge")
    page._start_stat_comparison()
    assert "at least two models" in page._stat_status.text()


def test_multiplicity_without_values_refuses(studio_app) -> None:
    page = _fresh_page()
    page._start_stat_multiplicity()
    assert "Paste the raw p-values" in page._stat_status.text()
    page._stat_paste.setText("0.03, abc")
    page._start_stat_multiplicity()
    assert "not numbers" in page._stat_status.text()


def test_coverage_without_a_conformal_run_refuses(studio_app) -> None:
    page = _fresh_page()
    page._start_stat_coverage()
    assert "Run Conformal intervals first" in page._stat_status.text()


def test_subgroup_without_a_model_refuses(studio_app) -> None:
    page = _fresh_page()
    page._start_stat_subgroups()
    assert "Train a model" in page._stat_status.text()


def test_model_card_without_a_bundle_refuses(studio_app) -> None:
    page = _fresh_page()
    page._write_model_card()
    assert "Open a model bundle first" in page._persist_status.text()


# -- the real workflows ---------------------------------------------------------------


def test_multiplicity_end_to_end(studio_app, qtbot, qapp) -> None:
    page = _fresh_page()
    page._stat_paste.setText("0.01, 0.04, 0.03, 0.005")
    page._start_stat_multiplicity()
    _settle(qtbot, qapp)

    assert "4 p-values adjusted (bonferroni)" in page._stat_status.text() or \
           "p-values adjusted" in page._stat_status.text()
    assert page._stat_frame is not None
    assert list(page._stat_frame.columns) == ["raw_p", "adjusted_p"]
    # Bonferroni multiplies by the family size (capped at 1)
    page._stat_correction.setCurrentText("bonferroni")
    page._stat_paste.setText("0.01, 0.04, 0.03, 0.005")
    page._start_stat_multiplicity()
    _settle(qtbot, qapp)
    frame = page._stat_frame
    assert np.allclose(frame["adjusted_p"], np.minimum(frame["raw_p"] * 4, 1.0))


def test_power_report_end_to_end(studio_app, qtbot, qapp) -> None:
    page = _fresh_page()
    page._stat_folds.setValue(5)
    page._stat_effect.setValue(2.0)
    page._start_stat_power()
    _settle(qtbot, qapp)

    status = page._stat_status.text()
    assert "MDE" in status
    assert "observed effect 2" in status
    assert page._stat_frame is not None
    values = dict(zip(page._stat_frame["quantity"], page._stat_frame["value"]))
    assert values["folds"] == "5"
    # a 2.0 effect over 5 folds (MDE 1.682) is sufficient
    assert "sufficient" in values["verdict"]


def test_comparison_end_to_end(studio_app, qtbot, qapp) -> None:
    page = _fresh_page()
    page._stat_models.setText("ridge, random_forest")
    page._stat_folds.setValue(3)
    page._start_stat_comparison()
    _settle(qtbot, qapp)

    status = page._stat_status.text()
    assert "leader" in status
    assert "3 folds" in status
    assert "fdr_bh" in status
    assert page._stat_frame is not None
    assert {"model", "mean", "difference", "adjusted_p", "note"} <= set(page._stat_frame.columns)
    # the rows are the CHALLENGERS: the leader is the baseline each is
    # tested against, so a 2-model panel yields 1 row.
    assert len(page._stat_frame) == 1
    assert page._stat_table.rowCount() == 1


def test_coverage_end_to_end(studio_app, qtbot, qapp) -> None:
    page = _fresh_page()
    _regression_outcome(page._model_analysis)
    # batch-1 conformal run first (the coverage source)
    page._start_stat_conformal()
    _settle(qtbot, qapp)
    assert page._stat_conformal_last is not None

    # a grouping column the dataset actually has: bin the first
    # descriptor into quartiles through a real derived column
    frame = pd.read_csv(_REGRESSION_EXAMPLE)
    groups = pd.qcut(frame["descr_000"], 2, labels=["low", "high"])
    frame["dose_group"] = groups.astype(str)
    frame.to_csv("/tmp/coverage-grouped.csv", index=False)
    page._model_analysis._dataset_combo.setCurrentText("")

    # point the coverage workflow's grouping at the derived column via
    # the free-text combo
    page._stat_group_column.setCurrentText("dose_group")
    # the loader's frame is the un-grouped example; patch the context
    # to the grouped frame the same way the analysis page carries it
    original = page._training_context

    def grouped_context():
        return frame, "pIC50", "regression"

    page._training_context = grouped_context  # type: ignore[assignment]
    try:
        page._start_stat_coverage()
        _settle(qtbot, qapp)
    finally:
        page._training_context = original  # type: ignore[assignment]

    status = page._stat_status.text()
    assert "overall coverage" in status
    assert "nominal 90%" in status
    assert page._stat_frame is not None
    assert {"group", "n_test", "n_covered", "coverage"} <= set(page._stat_frame.columns)
    # overall row + the two groups
    assert len(page._stat_frame) == 3


def test_subgroup_audit_end_to_end(studio_app, qtbot, qapp) -> None:
    page = _fresh_page()
    _regression_outcome(page._model_analysis)

    frame = pd.read_csv(_REGRESSION_EXAMPLE)
    frame["dose_group"] = pd.qcut(frame["descr_000"], 2, labels=["low", "high"]).astype(str)
    original = page._training_context

    def grouped_context():
        return frame, "pIC50", "regression"

    page._training_context = grouped_context  # type: ignore[assignment]
    page._stat_group_column.setCurrentText("dose_group")
    try:
        page._start_stat_subgroups()
        _settle(qtbot, qapp)
    finally:
        page._training_context = original  # type: ignore[assignment]

    status = page._stat_status.text()
    assert "subgroups over" in status
    assert "gaps tested (fdr_bh)" in status
    assert page._stat_frame is not None
    assert "subgroup" in page._stat_frame.columns
    assert page._stat_table.rowCount() == 2  # low + high


def test_model_card_end_to_end(studio_app, qtbot, qapp, tmp_path, monkeypatch) -> None:
    page = _fresh_page()
    _regression_outcome(page._model_analysis)

    # a real bundle on disk via the persistence tab's own writer
    bundle = tmp_path / "model.joblib"
    monkeypatch.setattr(
        "ui.pages.ml_workbench.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(bundle), "joblib"),
    )
    page._save_bundle()
    _settle(qtbot, qapp)
    assert bundle.exists()

    # open it back (the card's source), then write the card
    monkeypatch.setattr(
        "ui.pages.ml_workbench.QFileDialog.getOpenFileName",
        lambda *a, **k: (str(bundle), "joblib"),
    )
    page._open_bundle()
    _settle(qtbot, qapp)

    card = tmp_path / "model-card.md"
    monkeypatch.setattr(
        "ui.pages.ml_workbench.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(card), "Markdown (*.md)"),
    )
    page._write_model_card()
    _settle(qtbot, qapp)

    assert card.exists()
    text = card.read_text(encoding="utf-8")
    assert "linear_regression" in text
    assert "Model Card" in text or "model card" in text.lower()
    # html variant too
    card_html = tmp_path / "model-card.html"
    monkeypatch.setattr(
        "ui.pages.ml_workbench.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(card_html), "HTML (*.html)"),
    )
    page._write_model_card()
    _settle(qtbot, qapp)
    assert card_html.exists()
    assert "<html" in card_html.read_text(encoding="utf-8").lower()


def test_coverage_uses_saved_evaluation_rows_and_original_alpha(studio_app, qtbot, qapp, monkeypatch):
    from pharmaml.ml import conformal

    page = _fresh_page()
    _regression_outcome(page._model_analysis)
    observed = {}
    real_calibrate = conformal.calibrate_split_conformal

    def calibrate(model, X, y, **kwargs):
        observed["calibration_index"] = list(X.index)
        inner = real_calibrate(model, X, y, **kwargs)

        class Recorder:
            def predict_interval(self, X, **kw):
                observed["evaluation_index"] = list(X.index)
                return inner.predict_interval(X, **kw)
        return Recorder()

    monkeypatch.setattr(conformal, "calibrate_split_conformal", calibrate)
    page._start_stat_conformal()
    _settle(qtbot, qapp)
    assert observed["calibration_index"] and observed["evaluation_index"]
    assert set(observed["calibration_index"]).isdisjoint(observed["evaluation_index"])
    assert "independence from fitting/selection not verified" in page._stat_status.text()

    frame = pd.read_csv(_REGRESSION_EXAMPLE)
    frame["audit_group"] = np.where(frame.index % 2, "odd", "even")
    page._training_context = lambda: (frame, "pIC50", "regression")
    page._stat_group_column.setCurrentText("audit_group")
    page._stat_alpha.setValue(0.2)  # Must not relabel the previous alpha=0.1 result.
    page._start_stat_coverage()
    _settle(qtbot, qapp)
    assert "nominal 90%" in page._stat_status.text()
    result = page._stat_frame
    assert result is not None and result.iloc[0]["n_test"] == len(observed["evaluation_index"])
    assert "mean_width" in result.columns
    expected = frame.loc[observed["evaluation_index"], "audit_group"].value_counts()
    for group, n in expected.items():
        assert result.loc[result["group"] == group, "n_test"].iloc[0] == n
    frame.loc[observed["evaluation_index"][0], "pIC50"] += 1
    page._start_stat_coverage()
    assert "cannot verify group alignment" in page._stat_status.text()
    assert page._stat_frame is None and page._stat_table.rowCount() == 0


def test_subgroup_uses_actual_stratified_indices(studio_app, qtbot, qapp, monkeypatch):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from ui.pages.model_analysis import AnalysisOutcome

    from pharmaml.ml import subgroups

    page = _fresh_page()
    frame = pd.DataFrame({"f": range(120), "target": [0] * 100 + [1] * 20,
                          "group": [f"g{i % 3}" for i in range(120)]})
    x_train, x_test, y_train, y_test = train_test_split(
        frame[["f"]], frame.target, test_size=0.25, random_state=42, stratify=frame.target)
    model = LogisticRegression().fit(x_train, y_train)
    page._model_analysis._last_outcome = AnalysisOutcome(
        task="classification", model="logistic", n_train=len(x_train), n_test=len(x_test),
        y_test=y_test.to_numpy(), y_pred=None, metric_name="accuracy", metric_value=0.5,
        estimator=model, X_train=x_train, y_train=y_train, X_test=x_test, feature_names=["f"], metrics={})
    page._training_context = lambda: (frame, "target", "classification")
    page._stat_group_column.setCurrentText("group")
    captured = {}
    original = subgroups.subgroup_report

    def report(model, X, y, groups, **kwargs):
        captured["groups"] = groups.copy()
        return original(model, X, y, groups, **kwargs)

    monkeypatch.setattr(subgroups, "subgroup_report", report)
    page._start_stat_subgroups()
    _settle(qtbot, qapp)
    np.testing.assert_array_equal(captured["groups"], frame.loc[x_test.index, "group"].to_numpy())
    assert "not authenticated" in page._stat_status.text()
