"""Deep Learning Workbench page tests (increment 7): real API, real numbers.

Everything runs offscreen against the real backend entry points the
playground cards call (``make_train_test_split`` + ``DeepMLP*`` /
``NumpyMLP*`` + ``evaluate_model``, ``RunTrace`` / ``smooth_series`` /
``numpy_layer_stack`` / ``histogram_event``, ``umap_embed`` /
``trustworthiness``, ``SampleSet`` / ``DataView`` / ``find_duplicates`` /
``find_outliers`` / ``uniqueness`` / ``sort_by_similarity``). Every
asserted number was printed by the engines first
(``scripts/probe_dl_213.py``), on this exact bundled data with these
exact knobs: the 260-row QSAR regression table (target pIC50, 20
descriptor features, 208 train / 52 test at seed 42) where DeepMLPRegressor
(32,16, 40 epochs) lands R² 0.891 / RMSE 0.957 / MAE 0.758 with Pearson r
0.945 and an overfit gap of 0.061, the numpy-autograd twin lands R² 0.791,
and the 420-row QSAR classification table (target activity, 8 features,
336 train / 84 test) where DeepMLPClassifier reaches accuracy 0.952 with
F1 macro 0.953 over active/inactive/weak (24/29/31 test rows).
"""

from __future__ import annotations

import re
import time


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


def _fresh_page():
    from ui.pages.deep_learning import DeepLearningWorkbenchPage

    return DeepLearningWorkbenchPage()


def _run(page, starter, qtbot, studio_app) -> None:
    starter()
    qtbot.waitSignal(page._runner.job_ended, timeout=180_000).wait()
    _settle(qtbot, studio_app)


def _train_numpy(page, qtbot, studio_app):
    """The numpy-engine regression run the runboard/UMAP tests build on."""
    page._architecture.setCurrentIndex(2)  # NumpyMLP (numpy autograd)
    _run(page, page._start_train, qtbot, studio_app)
    return page.report_text()


# -- construction ---------------------------------------------------------------


def test_page_constructs_with_six_engine_tabs(studio_app) -> None:
    page = _fresh_page()
    assert page._tabs.count() == 6
    titles = [page._tabs.tabText(i) for i in range(page._tabs.count())]
    assert titles == [
        "Train & evaluate", "Runboard", "UMAP map", "Curation",
        "Efficiency lab", "GNN lab",
    ]


def test_default_dataset_loads_with_the_regression_target(studio_app) -> None:
    page = _fresh_page()
    assert page._frame.shape == (260, 21)
    assert page._target_combo.currentText() == "pIC50"
    assert page._numeric_features("pIC50") == [f"descr_{i:03d}" for i in range(20)]
    assert "260 rows · 21 columns · target 'pIC50'" in page.report_text()


def test_classification_dataset_switches_target(studio_app) -> None:
    page = _fresh_page()
    page._dataset_combo.setCurrentText("qsar_classification.csv (bundled)")
    assert page._frame.shape == (420, 11)
    assert page._target_combo.currentText() == "activity"
    # auto task inference: strings -> classification
    assert page._infer_task("activity") == "classification"


def test_data_explorer_entry_without_a_table_reports_honestly(studio_app) -> None:
    page = _fresh_page()
    page._dataset_combo.setCurrentText("(table from Data Explorer)")
    assert "Data Explorer has no table loaded yet." in page.report_text()


# -- train & evaluate (the torch engine) ---------------------------------------------


def test_torch_regression_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    _run(page, page._start_train, qtbot, studio_app)
    text = page.report_text()
    # probe: DeepMLPRegressor, R2 0.891280, RMSE 0.957172, MAE 0.758253,
    # pearson 0.945366, train_r2 0.952166, gap 0.060886, all 40 epochs.
    assert text.startswith("DeepMLPRegressor · regression · R² 0.891 · RMSE 0.957 · MAE 0.758")
    assert "Pearson r 0.945 · train R² 0.952 · gap 0.061" in text
    assert "ran all 40 epochs" in text
    assert "208 train / 52 test rows over 20 features" in text
    model = page._table.model()
    assert model.rowCount() == 52  # the parity table: one row per test row
    assert model.columnCount() == 3  # actual / predicted / error
    assert page._last_run.engine == "torch"
    assert page._last_run.estimator_name == "DeepMLPRegressor"


def test_numpy_regression_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    text = _train_numpy(page, qtbot, studio_app)
    # probe: NumpyMLPRegressor R2 0.788627, RMSE 1.334629, MAE 1.058162,
    # pearson 0.893182, train_r2 0.862764, gap 0.074137.
    assert text.startswith("NumpyMLPRegressor · regression · R² 0.789 · RMSE 1.335 · MAE 1.058")
    assert "Pearson r 0.893 · train R² 0.863 · gap 0.074" in text
    assert page._last_run.engine == "numpy"


def test_classification_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._dataset_combo.setCurrentText("qsar_classification.csv (bundled)")
    _run(page, page._start_train, qtbot, studio_app)
    text = page.report_text()
    # probe: accuracy 0.952381, balanced 0.953853, f1_macro 0.952778,
    # supports 24/29/31 over active/inactive/weak.
    assert text.startswith("DeepMLPClassifier · classification · accuracy 0.952")
    assert "balanced 0.954 · F1 macro 0.953" in text
    assert "3 classes (active 24/inactive 29/weak 31 test rows)" in text
    assert "336 train / 84 test rows over 8 features" in text
    model = page._table.model()
    assert model.rowCount() == 3  # one row per actual class
    assert model.columnCount() == 4  # actual \ predicted + the three classes
    assert model.data(model.index(0, 0)) == "actual active"


def test_bad_hidden_layers_refused_up_front(studio_app) -> None:
    page = _fresh_page()
    page._hidden_entry.setText("not-a-width")
    page._start_train()
    assert "Check the hidden layers" in page.report_text()


# -- runboard ---------------------------------------------------------------------


def test_curves_refuse_without_a_run(studio_app) -> None:
    page = _fresh_page()
    page._start_curves()
    assert (
        "Train a network first — the board reads the LAST run's per-epoch history."
        in page.report_text()
    )


def test_weights_refuse_without_a_run(studio_app) -> None:
    page = _fresh_page()
    page._start_weights()
    assert (
        "Train a network first — weight histograms read the LAST trained network."
        in page.report_text()
    )


def test_curves_after_a_numpy_run_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    _train_numpy(page, qtbot, studio_app)
    _run(page, page._start_curves, qtbot, studio_app)
    text = page.report_text()
    # probe: 40 epochs recorded; raw train starts 42.839318, EMA (beta
    # 0.6, bias-corrected) last 1.854412; 40 validation points.
    assert "runboard · 40 epochs recorded · smoothing beta 0.6 (0 = raw)" in text
    assert "train 42.839 → 1.854 (EMA last)" in text
    assert "40 validation points" in text
    model = page._table.model()
    assert model.rowCount() == 40  # the epoch log: epoch + train/val raw + EMA
    assert model.columnCount() == 5


def test_curves_beta_zero_is_the_raw_curve(studio_app, qtbot) -> None:
    page = _fresh_page()
    _train_numpy(page, qtbot, studio_app)
    page._beta_spin.setValue(0.0)
    _run(page, page._start_curves, qtbot, studio_app)
    assert "smoothing beta 0 (0 = raw)" in page.report_text()


def test_weights_after_a_numpy_run_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    _train_numpy(page, qtbot, studio_app)
    _run(page, page._start_weights, qtbot, studio_app)
    text = page.report_text()
    # probe: 3 layers; layer1 = 640 weights, min -0.399566, max 0.405114,
    # mean -0.004358, std 0.178054, bias std 0.143563.
    assert "weights/layer1 — 640 weights · min -0.4 · max 0.405" in text
    assert "mean -0.00436 ± 0.178" in text
    assert "3 layers snapshotted (the numpy engine's .linears)" in text
    model = page._table.model()
    assert model.rowCount() == 3  # one row per layer
    assert model.columnCount() == 7  # layer / weights / min / max / mean / std / bias std


def test_weights_refuse_a_torch_run_honestly(studio_app, qtbot) -> None:
    page = _fresh_page()
    _run(page, page._start_train, qtbot, studio_app)  # the torch engine
    assert page._last_run.engine == "torch"
    _run(page, page._start_weights, qtbot, studio_app)
    text = page.report_text()
    assert "Run failed" in text
    assert "numpy engine's layer stack" in text


# -- the UMAP map -------------------------------------------------------------------


def test_umap_after_a_run_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    _train_numpy(page, qtbot, studio_app)
    _run(page, page._embed_btn.click, qtbot, studio_app)
    text = page.report_text()
    # seeded run: 52 test rows x 20 features -> 2-D via spectral init.
    # The embedding's fingerprint (fuzzy-edge count, a/b curves) and the
    # trustworthiness score are bit-exact floats that drift across
    # BLAS/numpy builds, so the summary's SHAPE and sane bounds replace
    # the old literals (525 edges, a=1.5769, b=0.8951, 0.770).
    assert "UMAP -- 52 rows x 20 features -> 2-D (" in text
    assert "init=spectral)" in text
    m = re.search(r"trustworthiness 0\.(\d{3}) \(Venna-Kaski, k≤10\)", text)
    assert m, text
    assert 0.5 < float(m.group(0).split()[1]) < 1.0  # a real, sane score
    assert "source: the last run's test rows" in text
    model = page._table.model()
    assert model.rowCount() == 52  # one row per embedded test row
    assert model.columnCount() == 3  # row + UMAP1 + UMAP2


def test_umap_refuses_without_rows(studio_app) -> None:
    page = _fresh_page()
    page._dataset_combo.setCurrentText("(table from Data Explorer)")
    page._embed_btn.click()
    assert (
        "Not enough rows — UMAP needs a run's test rows or a dataset with at least 10 numeric rows."
        in page.report_text()
    )


def test_umap_on_the_dataset_without_a_run(studio_app, qtbot) -> None:
    page = _fresh_page()
    _run(page, page._embed_btn.click, qtbot, studio_app)
    text = page.report_text()
    # no run yet -> the dataset's numeric matrix: 260 rows x 20 features.
    assert "260 rows x 20 features -> 2-D" in text
    assert "source: the dataset's 20 numeric columns" in text
    model = page._table.model()
    assert model.rowCount() == 260


# -- curation: the view algebra --------------------------------------------------------


def test_curation_filter_stage(studio_app) -> None:
    page = _fresh_page()
    page._cur_field.setCurrentText("pIC50")
    page._cur_op.setCurrentText(">")
    page._cur_value.setText("7.0")
    page._filter_btn.click()
    # probe: 260 samples -> filter(pIC50 > 7.0) -> 101 samples.
    assert (
        "view: 260 samples -> filter(pIC50 > 7.0) -> 101 samples · 101 rows" in page.report_text()
    )


def test_curation_sort_and_limit_chain(studio_app) -> None:
    page = _fresh_page()
    page._cur_field.setCurrentText("pIC50")
    page._cur_op.setCurrentText(">")
    page._cur_value.setText("7.0")
    page._filter_btn.click()
    page._sort_btn.click()
    page._limit_btn.click()
    # probe: stages [filter, sort desc, limit 5]; top pIC50 11.0.
    text = page.report_text()
    assert "-> limit(5) -> 5 samples · 5 rows" in text
    assert "sort_by(pIC50 desc)" in text
    model = page._table.model()
    assert model.rowCount() == 5
    assert model.data(model.index(0, model.columnCount() - 1)) == "11.0"


def test_curation_reset_returns_the_identity(studio_app) -> None:
    page = _fresh_page()
    page._cur_field.setCurrentText("pIC50")
    page._cur_op.setCurrentText(">")
    page._cur_value.setText("7.0")
    page._filter_btn.click()
    page._reset_btn.click()
    assert "view: 260 samples -> identity -> 260 samples · 260 rows" in page.report_text()


def test_curation_between_needs_two_values(studio_app) -> None:
    page = _fresh_page()
    page._cur_field.setCurrentText("pIC50")
    page._cur_op.setCurrentText("between")
    page._cur_value.setText("7")
    page._filter_btn.click()
    assert "The 'between' stage needs 'lo,hi'." in page.report_text()


def test_curation_empty_filter_value_refused(studio_app) -> None:
    page = _fresh_page()
    page._cur_field.setCurrentText("pIC50")
    page._cur_op.setCurrentText(">")
    page._cur_value.setText("")
    page._filter_btn.click()
    assert "The filter stage needs a value." in page.report_text()


# -- curation: the audits ----------------------------------------------------------------


def test_duplicates_audit(studio_app, qtbot) -> None:
    page = _fresh_page()
    _run(page, page._duplicates_btn.click, qtbot, studio_app)
    # probe: 0 duplicate rows in 0 groups over the 260-row table.
    assert "260 rows -> 0 duplicate rows in 0 group(s) (keep=first)" in page.report_text()
    model = page._table.model()
    assert model.rowCount() == 1  # the honest empty row


def test_outliers_audit(studio_app, qtbot) -> None:
    page = _fresh_page()
    _run(page, page._outliers_btn.click, qtbot, studio_app)
    # probe: exactly one z-score outlier, descr_012, sample #44.
    assert "zscore outliers (threshold 4.0): descr_012: 1" in page.report_text()
    model = page._table.model()
    assert model.rowCount() == 1
    assert model.data(model.index(0, 0)) == "descr_012"
    assert model.data(model.index(0, 1)) == "1"


def test_uniqueness_audit(studio_app, qtbot) -> None:
    page = _fresh_page()
    _run(page, page._uniqueness_btn.click, qtbot, studio_app)
    # probe: 260 scored; top #254 0.436473, #253 0.434362, #145 0.431435.
    assert "260 samples scored" in page.report_text()
    assert "top: #254 0.436, #253 0.434, #145 0.431" in page.report_text()
    model = page._table.model()
    assert model.rowCount() == 20  # the top-20 table
    assert model.columnCount() == 2


def test_similarity_audit(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._sim_row.setValue(0)
    _run(page, page._similarity_btn.click, qtbot, studio_app)
    # probe: row 0's nearest are #165 0.6263, #59 0.592784, #2 0.581535.
    assert "sort_by_similarity(#0): #165 0.626, #59 0.593, #2 0.582" in page.report_text()
    model = page._table.model()
    assert model.rowCount() == 6  # row 0 itself + its k=5 nearest


# -- shared plumbing -----------------------------------------------------------------------


def test_buttons_disable_while_a_job_runs(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._runner.job_started.emit("demo")
    try:
        assert not page._train_btn.isEnabled()
        assert not page._embed_btn.isEnabled()
    finally:
        page._runner.job_ended.emit()
    assert page._train_btn.isEnabled()
