"""ML Workbench page tests -- the round-228 studio port.

The playground's remaining Data & ML cards (the tuner, the family
benchmark + OOF stacking leaderboards, the ensemble builder, the
persistence bundle, the applicability domain) migrated to the Studio's
ML Workbench page (the owner's "everything must be in studio.py"
directive, the retirement round).  Everything runs offscreen against
the real backend entry points: the honest pre-flight refusals (no
dataset, regression-only families, resource-less hyperband, no SMILES
column, classification targets, no run yet, too few rows), a REAL
RandomSearch over the ridge hypermodel on the bundled 260-row QSAR
regression example with its defaults baseline and held-out verdict, the
Apply-best hand-off to the Model Analysis page, the family benchmark
over the SMILES-bearing classification example, the OOF stacking
leaderboard with its native parity scatter, the voting ensemble on the
classification example, the persistence round-trip (save -> load ->
predict -> save predictions, plus the feature-contract audit refusal),
and the applicability-domain analysis over both the synthesized
ensemble source and a real Model Analysis training run.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

_TK = Path(__file__).resolve().parents[2]
_REGRESSION_EXAMPLE = _TK / "examples" / "data" / "qsar_regression.csv"


def _drain(pool_wait: float = 60.0) -> None:
    from PySide6.QtCore import QThreadPool

    deadline = time.monotonic() + pool_wait
    while QThreadPool.globalInstance().activeThreadCount() > 0 and time.monotonic() < deadline:
        time.sleep(0.01)


def _settle(qtbot, qapp) -> None:
    """Wait out the pool, then flush the queued result slots."""
    _drain()
    qtbot.wait(30)
    qapp.processEvents()


def _fresh_page(model_analysis=None):
    from ui.pages.ml_workbench import MLWorkbenchPage

    return MLWorkbenchPage(model_analysis=model_analysis)


def _fresh_analysis():
    from ui.pages.model_analysis import ModelAnalysisPage

    return ModelAnalysisPage()


def _load_classification(page) -> None:
    """Point the Model Analysis page at the SMILES-bearing example."""
    page._model_analysis._dataset_combo.setCurrentText("qsar_classification.csv")
    page._model_analysis._target_combo.setCurrentText("activity")


def _synth_ens_last(page) -> None:
    """A real quick fit stored as the page's ensemble run.

    The tuple's contents are all real (a fitted ridge, its report, the
    actual split of the bundled regression example) -- only the card's
    button was skipped, exactly like the playground's last-run handoff.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    from pharmaml.ml import evaluate_model

    frame = pd.read_csv(_REGRESSION_EXAMPLE)
    target = "pIC50"
    numeric = [column for column in frame.columns if column != target]
    x_train, x_test, y_train, y_test = train_test_split(
        frame[numeric], frame[target], test_size=0.25, random_state=42
    )
    estimator = LinearRegression().fit(x_train, y_train)
    report = evaluate_model(
        estimator, x_test, y_test, "regression",
        keep_predictions=True, X_train=x_train, y_train=y_train,
    )
    page._ens_last = (
        estimator, report, "regression", ["ridge"], "ridge (quick fit)",
        x_train, y_train, x_test, y_test, target,
    )


def _synth_last_outcome(analysis) -> None:
    """A real AnalysisOutcome from a real quick fit (the interop source)."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    from pharmaml.ml import evaluate_model
    from ui.pages.model_analysis import AnalysisOutcome

    frame = pd.read_csv(_REGRESSION_EXAMPLE)
    target = "pIC50"
    numeric = [column for column in frame.columns if column != target]
    x_train, x_test, y_train, y_test = train_test_split(
        frame[numeric], frame[target], test_size=0.25, random_state=42
    )
    estimator = LinearRegression().fit(x_train, y_train)
    report = evaluate_model(
        estimator, x_test, y_test, "regression",
        keep_predictions=False, X_train=x_train, y_train=y_train,
    )
    analysis._last_outcome = AnalysisOutcome(
        task="regression",
        model="linear_regression",
        n_train=len(x_train),
        n_test=len(x_test),
        y_test=y_test.to_numpy(dtype=float),
        y_pred=report.predictions if report.predictions is not None else None,
        metric_name="R² (held-out)",
        metric_value=float(report.r2),
        estimator=estimator,
        X_train=x_train,
        y_train=y_train,
        X_test=x_test,
        feature_names=[str(column) for column in numeric],
        metrics={"r2": float(report.r2), "rmse": float(report.rmse), "mae": float(report.mae)},
    )


# -- construction --------------------------------------------------------------- #


def test_page_constructs_with_the_six_tabs(studio_app) -> None:
    page = _fresh_page()
    assert [page._tabs.tabText(i) for i in range(page._tabs.count())] == [
        "Hyperparameter search",
        "Leaderboards",
        "Ensemble builder",
        "Persistence",
        "Applicability domain",
        "Statistical tools",
    ]
    # the Phase-3 statistical tab's defaults (part6 §3.3: forms expose
    # the canonical backends' own defaults)
    assert page._stat_source.currentText() == "Model Analysis run"
    assert page._stat_alpha.value() == 0.10
    assert page._stat_bins.value() == 10
    assert page._stat_cost.value() == 1.0
    assert page._stat_frame is None
    assert page._tune_estimator.currentText() == "ann"
    assert page._ens_kind.currentText() == "voting"
    assert page._ens_voting.currentText() == "soft"
    assert page._ad_source.currentText() == "Model Analysis run"
    assert page._bench_model.text() == "auto"
    assert page._stack_base.text() == "ridge, random_forest, gradient_boosting, extra_trees"
    assert page._stack_meta.text() == "ridge, random_forest, gradient_boosting"
    assert {token for token, box in page._ens_members.items() if box.isChecked()} == {
        "random_forest",
        "svm",
        "knn",
    }
    page.deleteLater()


def test_tuner_tokens_mirror_the_playground_and_the_registry() -> None:
    from pharmaml.ml.tuner import registry_hypermodels
    from ui.pages.ml_workbench import TUNE_PRUNERS, TUNE_SEARCHERS

    assert TUNE_SEARCHERS == ("random", "tpe", "hyperband")
    assert TUNE_PRUNERS == ("off", "median")
    page = _fresh_page()
    tokens = [page._tune_estimator.itemText(i) for i in range(page._tune_estimator.count())]
    assert tuple(tokens[:3]) == ("ann", "hsvr", "dnn")
    # the registry families ride along in the registry's own declared order
    expected = tuple(
        dict.fromkeys(token for token, _task in registry_hypermodels())
    )
    assert tuple(tokens[3:]) == expected
    assert "ridge" in tokens and "random_forest" in tokens and "lightgbm" in tokens
    assert [page._tune_searcher.itemText(i) for i in range(page._tune_searcher.count())] == [
        "random",
        "tpe",
        "hyperband",
    ]
    assert [page._tune_pruner.itemText(i) for i in range(page._tune_pruner.count())] == [
        "off",
        "median",
    ]
    # the spinbox defaults ride the fields verbatim
    assert page._tune_trials.value() == 8
    assert page._tune_eta.value() == 3
    assert page._tune_cv.value() == 3
    assert page._stack_folds.value() == 5
    assert page._stack_test.value() == 20
    assert page._ad_k.value() == 5
    assert page._ad_pct.value() == 95.0
    assert page._ad_boot.value() == 30
    page.deleteLater()


# -- pre-flight refusals (nothing doomed ever reaches the runner) ------------------ #


def test_refusals_without_a_dataset_or_target(studio_app) -> None:
    page = _fresh_page()  # no Model Analysis reference at all
    page._start_tune_search()
    assert "Model Analysis page" in page._tune_summary.text()
    page._start_benchmark()
    assert "Model Analysis page" in page._bench_status.text()
    page._start_stack()
    assert "Model Analysis page" in page._stack_status.text()
    page._start_ensemble()
    assert "Model Analysis page" in page._ens_summary.text()
    assert page._runner.busy is False  # every refusal fired BEFORE any job

    # a frame but no resolvable target: the honest target refusal
    analysis = _fresh_analysis()
    analysis._target_combo.clear()  # no target picked
    page_with_frame = _fresh_page(model_analysis=analysis)
    page_with_frame._start_tune_search()
    assert "target" in page_with_frame._tune_summary.text().lower()
    assert page_with_frame._runner.busy is False
    page.deleteLater()
    page_with_frame.deleteLater()
    analysis.deleteLater()


def test_tuner_refuses_hsvr_for_a_classification_target(studio_app) -> None:
    analysis = _fresh_analysis()
    page = _fresh_page(model_analysis=analysis)
    _load_classification(page)  # activity -> classification
    page._tune_estimator.setCurrentText("hsvr")
    page._start_tune_search()
    assert "regression-only" in page._tune_summary.text()
    assert page._runner.busy is False
    page.deleteLater()
    analysis.deleteLater()


def test_tuner_refuses_hyperband_over_a_resource_less_family(studio_app) -> None:
    analysis = _fresh_analysis()  # qsar_regression.csv -> regression
    page = _fresh_page(model_analysis=analysis)
    page._tune_searcher.setCurrentText("hyperband")
    page._tune_estimator.setCurrentText("ridge")
    page._start_tune_search()
    assert "Hyperband halves a resource slot" in page._tune_summary.text()
    assert page._runner.busy is False
    # hsvr declares no resource slot either
    page._tune_estimator.setCurrentText("hsvr")
    page._start_tune_search()
    assert "Hyperband halves a resource slot" in page._tune_summary.text()
    page.deleteLater()
    analysis.deleteLater()


def test_family_benchmark_refuses_without_a_smiles_column(studio_app) -> None:
    analysis = _fresh_analysis()  # qsar_regression.csv: descriptors only
    page = _fresh_page(model_analysis=analysis)
    page._bench_btn.click()
    assert "no SMILES-like column" in page._bench_status.text()
    assert page._runner.busy is False
    page.deleteLater()
    analysis.deleteLater()


def test_oof_stacking_refuses_a_classification_target(studio_app) -> None:
    analysis = _fresh_analysis()
    page = _fresh_page(model_analysis=analysis)
    _load_classification(page)
    page._stack_btn.click()
    assert "regressor" in page._stack_status.text()
    assert "classification" in page._stack_status.text()
    assert page._runner.busy is False
    page.deleteLater()
    analysis.deleteLater()


def test_persistence_save_refuses_without_any_run(studio_app) -> None:
    analysis = _fresh_analysis()  # dataset loaded, but no training run yet
    page = _fresh_page(model_analysis=analysis)
    assert analysis.last_run() is None
    page._save_bundle_to("/nowhere/model.joblib")
    assert "Train a model or an ensemble first" in page._persist_status.text()
    assert page._runner.busy is False
    # predict without a bundle: the honest note, not a crash
    page._predict_csv()
    assert "Open a model bundle first" in page._persist_status.text()
    page._save_predictions()
    assert "no prediction table" in page._persist_status.text()
    page.deleteLater()
    analysis.deleteLater()


def test_ad_refuses_without_a_run_and_with_too_few_rows(studio_app) -> None:
    analysis = _fresh_analysis()
    page = _fresh_page(model_analysis=analysis)
    assert analysis.last_run() is None
    page._ad_btn.click()  # default source: the Model Analysis run
    assert "Train a model on the Model Analysis page first" in page._ad_status.text()
    assert page._runner.busy is False
    # a synthesized ensemble with k training rows: the too-few-rows guard
    page._ad_source.setCurrentText("Workbench ensemble")
    page._ens_last = (
        object(), None, "regression", ["ridge"], "tiny",
        pd.DataFrame({"x": [1.0] * 5}), pd.Series([0.0] * 5),
        pd.DataFrame({"x": [2.0] * 3}), pd.Series([1.0] * 3), "y",
    )
    page._ad_k.setValue(5)
    page._ad_btn.click()
    assert "k=5 needs at least 6 training rows; only 5 available" in page._ad_status.text()
    assert page._runner.busy is False
    # the bundle source refuses without a loaded bundle
    page._ad_source.setCurrentText("loaded bundle")
    page._ad_btn.click()
    assert "Open a model bundle" in page._ad_status.text()
    page.deleteLater()
    analysis.deleteLater()


# -- the functional tuner ----------------------------------------------------------- #


def test_random_search_over_ridge_with_baseline_and_board(qtbot, studio_app) -> None:
    analysis = _fresh_analysis()  # qsar_regression.csv / pIC50 -> regression
    page = _fresh_page(model_analysis=analysis)
    page._tune_estimator.setCurrentText("ridge")
    page._tune_trials.setValue(4)
    page._tune_cv.setValue(2)
    page._tune_btn.click()
    qtbot.waitSignal(page._runner.job_ended, timeout=300_000).wait()
    _settle(qtbot, studio_app)

    summary = page._tune_summary.text()
    for token in ("best", "defaults", "gain", "held-out R²", "winner:", "alpha="):
        assert token in summary, summary
    # the board carries the 4 ranked trials (columns + rows)
    columns = [
        page._tune_board.horizontalHeaderItem(c).text()
        for c in range(page._tune_board.columnCount())
    ]
    assert columns == ["rank", "trial", "configuration", "score", "score_std"]
    assert page._tune_board.rowCount() == 4
    assert page._tune_board.item(0, 0).text() == "1"  # ranked by score desc
    assert page._tune_last is not None
    results, estimator, task, target = page._tune_last
    assert estimator == "ridge" and task == "regression" and target == "pIC50"
    assert len(results.trials) == 4
    page.deleteLater()
    analysis.deleteLater()


def test_apply_best_points_model_analysis_at_the_family(qtbot, studio_app) -> None:
    analysis = _fresh_analysis()
    page = _fresh_page(model_analysis=analysis)
    # no search yet: the honest refusal
    page._tune_apply_btn.click()
    assert "Run a search first" in page._tune_summary.text()
    # a real (tiny) search over the ridge family, then the hand-off
    page._tune_estimator.setCurrentText("ridge")
    page._tune_trials.setValue(4)
    page._tune_cv.setValue(2)
    page._tune_btn.click()
    qtbot.waitSignal(page._runner.job_ended, timeout=300_000).wait()
    _settle(qtbot, studio_app)
    assert page._tune_last is not None
    page._tune_apply_btn.click()
    assert analysis._model_combo.currentText() == "ridge"
    assert "applied: ridge" in page._tune_summary.text()
    # the winning alpha is pinned into the manual-hyperparameters table
    values = {}
    for row in range(analysis._params_table.rowCount()):
        key_item = analysis._params_table.item(row, 0)
        value_item = analysis._params_table.item(row, 1)
        if key_item is not None and value_item is not None:
            values[key_item.text()] = value_item.text()
    assert values.get("alpha", "") != ""  # the winner's alpha rides the table
    # a family the registry also carries switches the combo the same way
    # (the registry's own dnn entry -- the Deep Learning note only fires
    # for tokens the combo genuinely lacks)
    page._tune_last = (page._tune_last[0], "dnn", "regression", "pIC50")
    page._tune_apply_btn.click()
    assert analysis._model_combo.currentText() == "dnn"
    assert "applied: dnn" in page._tune_summary.text()
    page.deleteLater()
    analysis.deleteLater()


# -- the functional leaderboards ------------------------------------------------------ #


def test_family_benchmark_over_the_smiles_bearing_example(qtbot, studio_app) -> None:
    analysis = _fresh_analysis()
    page = _fresh_page(model_analysis=analysis)
    _load_classification(page)  # SMILES + activity -> classification
    page._bench_folds.setValue(2)
    page._bench_btn.click()
    qtbot.waitSignal(page._runner.job_ended, timeout=300_000).wait()
    _settle(qtbot, studio_app)

    assert page._bench_board.rowCount() > 5  # the 18 2D families
    columns = [
        page._bench_board.horizontalHeaderItem(c).text()
        for c in range(page._bench_board.columnCount())
    ]
    assert "family" in columns
    assert "family benchmark" in page._bench_status.text()
    assert "higher is better" in page._bench_status.text()
    families = {
        page._bench_board.item(row, columns.index("family")).text()
        for row in range(page._bench_board.rowCount())
    }
    assert "ecfp" in families
    page.deleteLater()
    analysis.deleteLater()


def test_oof_stacking_leaderboard_and_parity_scatter(qtbot, studio_app) -> None:
    analysis = _fresh_analysis()  # regression target
    page = _fresh_page(model_analysis=analysis)
    page._stack_base.setText("ridge")
    page._stack_meta.setText("ridge")
    page._stack_folds.setValue(2)
    page._stack_test.setValue(20)
    page._stack_btn.click()
    qtbot.waitSignal(page._runner.job_ended, timeout=300_000).wait()
    _settle(qtbot, studio_app)

    assert page._stack_board.rowCount() >= 1  # ridge + the residual row
    status = page._stack_status.text()
    assert "2-fold OOF stacking" in status
    assert "1 base × 1 meta" in status
    assert "best blender: ridge" in status
    assert "test R²" in status
    assert page._stack_last is not None
    # the parity scatter is native PyQtGraph (§16.8: no matplotlib)
    items = page._stack_plot.widget().getPlotItem().listDataItems()
    assert items, "the parity scatter must draw"
    page.deleteLater()
    analysis.deleteLater()


# -- the functional ensemble ----------------------------------------------------------- #


def test_voting_ensemble_trains_and_is_kept_for_downstream(qtbot, studio_app) -> None:
    analysis = _fresh_analysis()
    page = _fresh_page(model_analysis=analysis)
    _load_classification(page)  # activity -> classification
    # fast members for the suite: KNN + Logistic (both proba-capable)
    for token, box in page._ens_members.items():
        box.setChecked(token in ("knn", "logistic_regression"))
    page._ens_btn.click()
    qtbot.waitSignal(page._runner.job_ended, timeout=300_000).wait()
    _settle(qtbot, studio_app)

    summary = page._ens_summary.text()
    assert "voting (knn+logistic_regression) trained" in summary
    assert "held-out accuracy" in summary
    assert page._ens_last is not None
    estimator, _report, task, members, name, _xtr, _ytr, _xte, _yte, target = page._ens_last
    assert hasattr(estimator, "predict")  # a fitted estimator rides the tuple
    assert task == "classification" and target == "activity"
    assert set(members) == {"knn", "logistic_regression"}
    assert name.startswith("voting")
    # classification draws the class-count bars natively
    items = page._ens_plot.widget().getPlotItem().listDataItems()
    assert items, "the class-count bars must draw"
    page.deleteLater()
    analysis.deleteLater()


# -- the persistence round-trip --------------------------------------------------------- #


def test_persistence_round_trip_with_contract_audit(qtbot, studio_app, tmp_path: Path) -> None:
    analysis = _fresh_analysis()  # holds the dataset, carries NO run
    page = _fresh_page(model_analysis=analysis)
    _synth_ens_last(page)  # the page's own ensemble is the save source
    assert analysis.last_run() is None

    # save -> the bundle + the .json sidecar land on disk
    bundle = tmp_path / "quick_ridge.joblib"
    page._save_bundle_to(str(bundle))
    qtbot.waitSignal(page._runner.job_ended, timeout=120_000).wait()
    _settle(qtbot, studio_app)
    assert bundle.is_file() and (tmp_path / "quick_ridge.json").is_file()
    assert "bundle written: quick_ridge.joblib" in page._persist_status.text()
    assert "+ quick_ridge.json" in page._persist_status.text()

    # load -> the provenance line names the model, task, versions
    page._load_bundle_from(bundle)
    qtbot.waitSignal(page._runner.job_ended, timeout=120_000).wait()
    _settle(qtbot, studio_app)
    assert page._bundle is not None
    _model, metadata, path = page._bundle
    assert metadata["name"] == "ridge (quick fit)"
    assert metadata["task"] == "regression"
    assert "ridge (quick fit)" in page._persist_status.text()
    assert "regression" in page._persist_status.text()
    assert "pharmaml" in page._persist_status.text()
    assert path == bundle

    # predict -> the training table itself (target dropped by the audit)
    frame = pd.read_csv(_REGRESSION_EXAMPLE)
    scored = tmp_path / "training_table.csv"
    frame.to_csv(scored, index=False)
    page._predict_on(scored)
    qtbot.waitSignal(page._runner.job_ended, timeout=120_000).wait()
    _settle(qtbot, studio_app)
    assert page._predictions is not None and len(page._predictions) == len(frame)
    assert "prediction" in page._predictions.columns
    columns = [
        page._predict_table.horizontalHeaderItem(c).text()
        for c in range(page._predict_table.columnCount())
    ]
    assert "prediction" in columns
    assert page._predict_table.rowCount() == 12  # the capped preview
    assert f"{len(frame)} rows predicted" in page._persist_status.text()

    # a CSV missing the bundle's feature columns fails the audit honestly
    bad = tmp_path / "mismatched.csv"
    pd.DataFrame({"descr_000": [1.0, 2.0], "junk": ["a", "b"]}).to_csv(bad, index=False)
    page._predict_on(bad)
    qtbot.waitSignal(page._runner.job_ended, timeout=120_000).wait()
    _settle(qtbot, studio_app)
    assert page._persist_status.text().startswith("Job failed —")
    assert "feature contract" in page._persist_status.text()

    # save predictions -> the full table (never the capped preview)
    out = tmp_path / "predictions.csv"
    page._save_predictions_to(out)
    qtbot.waitSignal(page._runner.job_ended, timeout=120_000).wait()
    _settle(qtbot, studio_app)
    written = pd.read_csv(out)
    assert len(written) == len(frame)
    assert "prediction" in written.columns
    page.deleteLater()
    analysis.deleteLater()


# -- the functional applicability domain -------------------------------------------------- #


def test_ad_knn_only_pass_over_the_ensemble_source(qtbot, studio_app) -> None:
    analysis = _fresh_analysis()  # dataset present (no SMILES: generic names)
    page = _fresh_page(model_analysis=analysis)
    _synth_ens_last(page)
    page._ad_source.setCurrentText("Workbench ensemble")
    page._ad_boot.setValue(0)  # the kNN-only pass: fast, no bootstrap
    page._ad_btn.click()
    qtbot.waitSignal(page._runner.job_ended, timeout=300_000).wait()
    _settle(qtbot, studio_app)

    status = page._ad_status.text()
    assert "inside AD" in status
    assert "kNN-only pass: bootstrap models = 0" in status
    columns = [
        page._ad_table.horizontalHeaderItem(c).text()
        for c in range(page._ad_table.columnCount())
    ]
    assert "knn_distance" in columns
    assert "inside_ad" in columns
    assert page._ad_table.rowCount() > 0
    # the inside/outside figure: two scatter sets + the threshold line
    items = page._ad_plot.widget().getPlotItem().listDataItems()
    assert len(items) >= 3, "two scatters + the dashed threshold must draw"
    assert page._ad_frame is not None
    page.deleteLater()
    analysis.deleteLater()


def test_ad_resolves_the_model_analysis_run(qtbot, studio_app) -> None:
    analysis = _fresh_analysis()
    _synth_last_outcome(analysis)  # a real AnalysisOutcome from a real fit
    page = _fresh_page(model_analysis=analysis)
    assert analysis.last_run() is not None
    page._ad_boot.setValue(0)
    page._ad_btn.click()  # default source: the Model Analysis run
    qtbot.waitSignal(page._runner.job_ended, timeout=300_000).wait()
    _settle(qtbot, studio_app)

    status = page._ad_status.text()
    assert "inside AD" in status
    # the run's OWN split was reused: 260 rows * 0.25 = 65 held-out queries
    assert page._ad_table.rowCount() == 65
    assert page._ad_plot.widget().getPlotItem().listDataItems()
    page.deleteLater()
    analysis.deleteLater()


# -- theming ----------------------------------------------------------------------------- #


def test_apply_palette_recolours_every_plot_panel(studio_app) -> None:
    from ui.theme_manager import ChartPalette

    page = _fresh_page()
    dark = ChartPalette(background="#1d1f21", foreground="#e0e0e0", dark=True)
    page.apply_palette(dark)
    for plot in (page._stack_plot, page._ens_plot, page._ad_plot):
        assert plot._palette is dark
    page.deleteLater()


# -- the checkpointed family benchmark (Part-3 §20) ----------------------------------- #


def test_family_benchmark_checkpoints_and_resumes(
    qtbot, studio_app, tmp_path: Path
) -> None:
    """The §20 loop: run 1 fills the checkpoint, run 2 resumes from it.

    The second Start with unchanged settings reuses every stored family
    row (the resume note leads the status line) and still lands the
    full board -- Stop's progress is never thrown away.
    """
    from pharmaml.utils.checkpoints import CheckpointManager

    analysis = _fresh_analysis()
    page = _fresh_page(model_analysis=analysis)
    _load_classification(page)  # SMILES + activity -> classification
    page._bench_folds.setValue(2)
    page._bench_checkpoint_dir = tmp_path  # the resume home for this test

    page._bench_btn.click()  # run 1: computes and checkpoints per family
    qtbot.waitSignal(page._runner.job_ended, timeout=300_000).wait()
    _settle(qtbot, studio_app)
    assert "family benchmark" in page._bench_status.text()
    rows_first_run = page._bench_board.rowCount()
    assert rows_first_run > 5  # the 18 2D families

    info = CheckpointManager(tmp_path, "family-benchmark").latest()
    assert info is not None
    assert info.step == info.total  # every family scored before the exit
    assert info.step == rows_first_run

    page._bench_btn.click()  # run 2: unchanged settings -> pure reuse
    qtbot.waitSignal(page._runner.job_ended, timeout=300_000).wait()
    _settle(qtbot, studio_app)
    status = page._bench_status.text()
    assert status.startswith("checkpoint found:")
    assert "families already scored" in status
    assert page._bench_board.rowCount() == rows_first_run
    page.deleteLater()
    analysis.deleteLater()
