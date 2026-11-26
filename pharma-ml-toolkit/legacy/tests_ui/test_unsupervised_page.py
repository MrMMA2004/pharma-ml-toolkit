"""Unsupervised Workbench page tests (increment 6): real API, real numbers.

Everything runs offscreen against the real backend entry points the
playground cards call (``pca_reduce`` / ``kmeans_cluster`` /
``gmm_cluster`` / ``hierarchical_cluster`` / ``kde_fit`` /
``manifold_embedding``). Every asserted number was printed by the
engines first (``scripts/probe_unsupervised_212.py``), on this exact
bundled data: the 77-row descriptor library whose 8 descriptor columns
are checked by default -- PCA fits 8 components with PC1+PC2 at 81.5%
and PC1 dominated by NumHAcceptors, k-means k=3 lands silhouette
0.438 with sizes 8/39/30, the GMM BIC sweep picks 4 components
(converged, weights 0.55/0.09/0.21/0.16), ward hierarchical gives
29/7/41 with two merge heights, KDE on MolWt chooses bandwidth 10 by
cross-validation with mean log-likelihood -395.58, and every manifold
embeds 77 x 8 -> 2-D with its own knobs echoed.
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
    from ui.pages.unsupervised import UnsupervisedWorkbenchPage

    return UnsupervisedWorkbenchPage()


def _run(page, starter, qtbot, studio_app) -> None:
    starter()
    qtbot.waitSignal(page._runner.job_ended, timeout=120_000).wait()
    _settle(qtbot, studio_app)


# -- construction ---------------------------------------------------------------


def test_page_constructs_with_four_engine_tabs(studio_app) -> None:
    page = _fresh_page()
    assert page._tabs.count() == 4
    titles = [page._tabs.tabText(i) for i in range(page._tabs.count())]
    assert titles == ["PCA", "Clustering", "Density (KDE)", "Manifold"]


def test_default_dataset_loads_with_all_numeric_columns_checked(studio_app) -> None:
    page = _fresh_page()
    # 77 rows x 10 cols; the 8 descriptor columns are numeric (SMILES
    # and activity are not) and checked by default.
    assert page._frame.shape == (77, 10)
    assert page._checked_columns() == [
        "MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
        "NumRotatableBonds", "LabuteASA", "RingCount",
    ]
    assert page._kde_column.currentText() == "MolWt"
    assert "77 rows · 8 numeric columns checked" in page.report_text()


def test_unchecked_columns_refuse_the_run(studio_app) -> None:
    page = _fresh_page()
    page._none_btn.click()
    page._start_pca()
    assert "Check at least one numeric column." in page.report_text()


def test_data_explorer_entry_without_a_table_reports_honestly(studio_app) -> None:
    page = _fresh_page()
    page._dataset_combo.setCurrentText("(table from Data Explorer)")
    assert "Data Explorer has no table loaded yet." in page.report_text()


# -- PCA -----------------------------------------------------------------------------


def test_pca_auto_components_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    _run(page, page._start_pca, qtbot, studio_app)
    text = page.report_text()
    # probe: 8 components over 8 features, 77 rows; PC1 45.2%, PC2
    # 36.3%, PC3 9.4%; cumulative at 8 = 100.0%; PC1 dominated by
    # NumHAcceptors (|loading| 0.447).
    assert "PCA · 8 components over 8 features, 77 rows" in text
    assert "PC1 45.2% · PC2 36.3% · PC3 9.4%" in text
    assert "cumulative at 8 = 100.0%" in text
    assert "PC1 dominated by NumHAcceptors (|loading| 0.447)" in text
    model = page._table.model()
    assert model.rowCount() == 8  # the loadings table: one row per feature
    assert model.columnCount() == 9  # feature + PC1..PC8


def test_pca_fixed_three_components(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._pca_components.setValue(3)
    _run(page, page._start_pca, qtbot, studio_app)
    # probe: n_components=3 -> cumulative 0.909255
    assert "3 components over 8 features, 77 rows" in page.report_text()
    assert "cumulative at 3 = 90.9%" in page.report_text()


def test_pca_on_the_qsar_table(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._dataset_combo.setCurrentText("qsar_regression.csv (bundled)")
    assert page._frame.shape == (260, 21)
    _run(page, page._start_pca, qtbot, studio_app)
    # probe: auto PCA over 21 numeric columns -> 10 components,
    # cumulative 0.624565.
    assert "10 components over 21 features, 260 rows" in page.report_text()
    assert "cumulative at 10 = 62.5%" in page.report_text()


# -- clustering --------------------------------------------------------------------------


def test_kmeans_manual_k_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._cluster_k.setValue(3)
    _run(page, page._start_cluster, qtbot, studio_app)
    text = page.report_text()
    # probe: k=3 Lloyd -> silhouette 0.438382, inertia 73753.5587,
    # sizes 8/39/30.
    assert "k-means (Lloyd) · k=3 · 77 rows" in text
    assert "silhouette 0.438" in text
    assert "inertia 7.375e+04" in text
    assert "sizes 8/39/30" in text
    assert "fitted in" in text
    model = page._table.model()
    assert model.rowCount() == 3  # the centroid table: one row per cluster
    assert model.columnCount() == 9  # cluster + the 8 descriptor columns


def test_kmeans_auto_uses_the_silhouette_scan(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._cluster_k.setValue(0)
    _run(page, page._start_cluster, qtbot, studio_app)
    text = page.report_text()
    # probe: the scan (k=2..10) picks k=2 at silhouette 0.465551.
    assert "k=2 · 77 rows" in text
    assert "silhouette 0.466" in text
    assert "k by silhouette scan" in text


def test_mini_batch_kmeans(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._cluster_method.setCurrentText("mini-batch k-means")
    page._cluster_k.setValue(3)
    _run(page, page._start_cluster, qtbot, studio_app)
    # probe: mini-batch k=3 -> silhouette 0.445717, sizes 30/40/7.
    assert "mini-batch k-means · k=3 · 77 rows" in page.report_text()
    assert "silhouette 0.446" in page.report_text()
    assert "sizes 30/40/7" in page.report_text()


def test_gmm_bic_sweep_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._cluster_method.setCurrentText("Gaussian mixture (BIC sweep)")
    _run(page, page._start_cluster, qtbot, studio_app)
    text = page.report_text()
    # probe: the sweep (1..8) picks 4 components, converged,
    # log-likelihood -688.2237, weights 0.5454/0.0909/0.2078/0.1559,
    # hard labels 42/7/16/12.
    assert "Gaussian mixture (BIC sweep) · k=4 · 77 rows" in text
    assert "chosen by bic · converged" in text
    assert "log-likelihood -688.22" in text
    assert "weights 0.55/0.09/0.21/0.16" in text
    assert "sizes 42/7/16/12" in text
    model = page._table.model()
    assert model.rowCount() == 77  # the soft responsibilities: one row per sample
    assert model.columnCount() == 5  # sample + the 4 responsibilities


def test_gmm_manual_components_diag(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._cluster_method.setCurrentText("Gaussian mixture (manual k)")
    page._cluster_k.setValue(3)
    page._gmm_covariance.setCurrentText("diag")
    _run(page, page._start_cluster, qtbot, studio_app)
    # probe: manual n=3 diag -> chosen_by manual, converged,
    # weights 0.1034/0.4246/0.472.
    assert "Gaussian mixture (manual k) · k=3 · 77 rows" in page.report_text()
    assert "chosen by manual · converged" in page.report_text()
    assert "weights 0.10/0.42/0.47" in page.report_text()


def test_hierarchical_ward_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._cluster_method.setCurrentText("hierarchical (ward)")
    page._cluster_k.setValue(3)
    _run(page, page._start_cluster, qtbot, studio_app)
    text = page.report_text()
    # probe: ward k=3 -> inertia 75634.5744, sizes 29/7/41, two merge
    # heights (60251.6711, 133275.5917).
    assert "hierarchical (ward) · k=3 · 77 rows" in text
    assert "inertia 7.563e+04" in text
    assert "merge heights 2" in text
    assert "sizes 29/7/41" in text
    model = page._table.model()
    assert model.rowCount() == 3  # the cluster sizes table
    assert model.columnCount() == 2  # cluster + size


def test_hierarchical_complete_linkage(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._cluster_method.setCurrentText("hierarchical (complete)")
    page._cluster_k.setValue(3)
    _run(page, page._start_cluster, qtbot, studio_app)
    # probe: complete k=3 -> sizes 31/7/39 (not ward's 29/7/41 -- the
    # linkage choice visibly changes the cut).
    assert "hierarchical (complete) · k=3 · 77 rows" in page.report_text()
    assert "sizes 31/7/39" in page.report_text()


# -- KDE ---------------------------------------------------------------------------


def test_kde_auto_bandwidth_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    _run(page, page._start_kde, qtbot, studio_app)
    text = page.report_text()
    # probe: MolWt, n=77 -> bandwidth 10.0 chosen by cross-validation,
    # mean log-likelihood -395.583906, 200-point grid.
    assert "KDE · column 'MolWt' · n=77" in text
    assert "bandwidth 10 (cv, kernel gaussian)" in text
    assert "mean log-likelihood -395.58" in text
    assert "grid of 200 points" in text
    model = page._table.model()
    assert model.rowCount() == 200
    assert model.columnCount() == 2  # x + density


def test_kde_manual_bandwidth(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._kde_bandwidth.setText("20")
    _run(page, page._start_kde, qtbot, studio_app)
    # probe: manual 20 -> bandwidth_source manual.
    assert "bandwidth 20 (manual, kernel gaussian)" in page.report_text()


def test_kde_rejects_a_non_positive_bandwidth(studio_app) -> None:
    page = _fresh_page()
    page._kde_bandwidth.setText("-5")
    page._start_kde()
    assert "Bandwidth must be positive." in page.report_text()
    page._kde_bandwidth.setText("abc")
    page._start_kde()
    assert "Bandwidth must be a number" in page.report_text()


# -- manifold -------------------------------------------------------------------------


def test_isomap_embedding_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._manifold_method.setCurrentText("isomap")
    _run(page, page._start_manifold, qtbot, studio_app)
    # probe: isomap -> "isomap embedding -- 77 rows x 8 features -> 2-D
    # (n_neighbors=10)".
    assert "isomap embedding -- 77 rows x 8 features -> 2-D (n_neighbors=10)" in page.report_text()
    model = page._table.model()
    assert model.rowCount() == 77
    assert model.columnCount() == 3  # sample + ISOMAP1 + ISOMAP2


def test_lle_modified_variant(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._manifold_method.setCurrentText("lle")
    page._lle_method.setCurrentText("modified")
    _run(page, page._start_manifold, qtbot, studio_app)
    # probe: modified LLE echoes lle_method=modified.
    assert "lle embedding -- 77 rows x 8 features -> 2-D" in page.report_text()
    assert "lle_method=modified" in page.report_text()


def test_tsne_embedding_reports_its_perplexity(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._manifold_method.setCurrentText("tsne")
    _run(page, page._start_manifold, qtbot, studio_app)
    # probe: tsne echoes perplexity=30.0.
    assert "tsne embedding -- 77 rows x 8 features -> 2-D (perplexity=30.0)" in page.report_text()


def test_tsne_perplexity_above_the_sample_count_fails_honestly(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._manifold_method.setCurrentText("tsne")
    page._tsne_perplexity.setValue(100)
    _run(page, page._start_manifold, qtbot, studio_app)
    # probe: the engine's exact words -- perplexity must sit strictly
    # below the sample count (77).
    assert "perplexity strictly below the sample count" in page.report_text()
    assert "Run failed — DataValidationError" in page.report_text()


def test_isomap_neighborhood_above_the_sample_count_fails_honestly(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._manifold_method.setCurrentText("isomap")
    page._manifold_neighbors.setValue(100)
    _run(page, page._start_manifold, qtbot, studio_app)
    # probe: the engine's exact words for the violated precondition.
    assert "1 <= n_neighbors < n_samples (77)" in page.report_text()


# -- the shared matrix guard -------------------------------------------------------------


def test_the_matrix_needs_a_dataset(studio_app) -> None:
    page = _fresh_page()
    page._dataset_combo.setCurrentText("(table from Data Explorer)")
    page._start_pca()
    assert "Load a dataset first." in page.report_text()
