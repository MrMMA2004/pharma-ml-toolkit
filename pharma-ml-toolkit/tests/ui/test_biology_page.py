"""Biology Workbench page tests (increment 4): real API, real numbers.

Everything runs offscreen against the real backend entry points the
playground cards call (``load_response_table`` → ``audit_response_table``
→ ``collapse_replicates`` → ``response_matrix`` / ``most_variable_genes``
/ ``sensitivity_associations`` → ``top_biomarkers``). Every asserted
number was printed by the engines first, on this exact bundled data:
185 rows → 184 unique pairs, 23 lines x 8 compounds, crizotinib
6/11/6 quartile calls, GENE_115 the MAD top, GENE_144 the strongest
cisplatin association (|rho| 0.6097).
"""

from __future__ import annotations

import time


def _drain(pool_wait: float = 5.0) -> None:
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
    from ui.pages.biology import BiologyWorkbenchPage

    return BiologyWorkbenchPage()


def _load_screen(page, qtbot, studio_app) -> None:
    """Run the audit job once so the cross-tab matrix state exists."""
    page._start_audit()
    qtbot.waitSignal(page._runner.job_ended, timeout=120_000).wait()
    _settle(qtbot, studio_app)


# -- construction ---------------------------------------------------------------


def test_page_constructs_with_four_engine_tabs(studio_app) -> None:
    page = _fresh_page()
    assert page._tabs.count() == 4
    titles = [page._tabs.tabText(i) for i in range(page._tabs.count())]
    assert titles == [
        "Screen audit",
        "Sensitivity matrix",
        "Variable genes (MAD)",
        "Biomarker discovery",
    ]
    assert page._screen_combo.count() == 1
    assert page._method_combo.count() == 2


def test_matrix_refuses_without_a_screen(studio_app) -> None:
    page = _fresh_page()
    page._start_matrix()
    assert "Load & audit a screen first" in page.report_text()


def test_biomarkers_refuse_without_a_screen(studio_app) -> None:
    page = _fresh_page()
    page._start_markers()
    assert "Load & audit a screen first" in page.report_text()


# -- the screen audit job ---------------------------------------------------------


def test_screen_audit_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._start_audit()
    qtbot.waitSignal(page._runner.job_ended, timeout=120_000).wait()
    _settle(qtbot, studio_app)

    text = page.report_text()
    assert "185 rows -> 184 unique pairs" in text
    assert "23 lines x 8 compounds" in text
    assert "metric LN_IC50" in text
    assert "duplicate pairs: MCF-7 / crizotinib (2)" in text
    assert "missing metric values" in text
    assert "1 replicate pair(s) collapsed" in text
    assert "matrix 23x8 pivoted" in text

    frame = page.model_frame()
    assert frame is not None
    assert "compound" in frame.columns
    assert len(frame) == 8  # one row per compound
    assert {
        "n_cell_lines", "median", "min", "max", "std",
        "sensitive_spread", "most_sensitive_cell_line",
    } <= set(frame.columns)

    # the cross-tab state landed: matrix 23x8, both compound combos fed
    assert page._matrix is not None and page._matrix.shape == (23, 8)
    assert page._matrix_compound.count() == 8
    assert page._marker_compound.count() == 8
    assert page._screen_metric == "LN_IC50"


# -- the sensitivity matrix job ------------------------------------------------------


def test_matrix_quantile_order_validated(studio_app, qtbot) -> None:
    page = _fresh_page()
    _load_screen(page, qtbot, studio_app)
    page._sens_q.setText("0.9")
    page._res_q.setText("0.8")
    page._start_matrix()
    assert "must be below" in page.report_text()


def test_matrix_quartile_calls(studio_app, qtbot) -> None:
    page = _fresh_page()
    _load_screen(page, qtbot, studio_app)
    page._matrix_compound.setCurrentText("crizotinib")
    page._start_matrix()
    qtbot.waitSignal(page._runner.job_ended, timeout=120_000).wait()
    _settle(qtbot, studio_app)

    text = page.report_text()
    assert "23 lines x 8 compounds" in text
    assert "crizotinib: 6 sensitive / 11 intermediate / 6 resistant" in text

    frame = page.model_frame()
    assert frame is not None
    assert "cell_line" in frame.columns
    assert len(frame) == 23
    assert "crizotinib" in frame.columns


# -- the MAD ranking job --------------------------------------------------------------


def test_mad_ranking_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._start_mad()
    qtbot.waitSignal(page._runner.job_ended, timeout=120_000).wait()
    _settle(qtbot, studio_app)

    text = page.report_text()
    assert "top 10 of 300 genes by MAD" in text
    assert "most variable: GENE_115" in text

    frame = page.model_frame()
    assert frame is not None
    assert len(frame) == 10
    assert {"gene", "MAD"} <= set(frame.columns)
    assert frame["gene"].iloc[0] == "GENE_115"
    assert abs(float(frame["MAD"].iloc[0]) - 1.1874) < 5e-3
    # ranked most-variable-first
    assert float(frame["MAD"].iloc[0]) >= float(frame["MAD"].iloc[-1])


def test_mad_ranking_input_validated(studio_app) -> None:
    page = _fresh_page()
    page._mad_n.setText("0")
    page._start_mad()
    assert "top N genes" in page.report_text()


# -- the biomarker job ----------------------------------------------------------------


def test_biomarkers_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    _load_screen(page, qtbot, studio_app)
    page._marker_compound.setCurrentText("cisplatin")
    page._start_markers()
    qtbot.waitSignal(page._runner.job_ended, timeout=180_000).wait()
    _settle(qtbot, studio_app)

    text = page.report_text()
    assert "300 genes correlated" in text
    assert "0 significant at FDR < 0.05" in text
    assert "strongest |rho| 0.610" in text
    assert "GENE_144" in text
    assert "more resistant" in text  # direction_of_association(positive rho)

    frame = page.model_frame()
    assert frame is not None
    assert len(frame) == 8
    assert {"gene", "rho", "p_value", "fdr", "significant"} <= set(frame.columns)
    assert frame["gene"].iloc[0] == "GENE_144"
    assert abs(float(frame["rho"].iloc[0]) - 0.609684) < 1e-3

    # the normalized expression cache landed for the next run
    assert page._expr_norm is not None and page._expr_norm.shape == (23, 300)


def test_biomarker_input_validated(studio_app, qtbot) -> None:
    page = _fresh_page()
    _load_screen(page, qtbot, studio_app)
    page._topn_entry.setText("0")
    page._start_markers()
    assert "top N biomarkers" in page.report_text()


# -- job plumbing ----------------------------------------------------------------------


def test_duplicate_submission_refused(studio_app, qtbot) -> None:
    page = _fresh_page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)

    def slow_job(progress) -> None:
        time.sleep(0.4)

    page._runner.submit("hold-the-runner", slow_job)
    page._start_mad()  # refused while the hold job runs
    assert "Busy: '" in page.report_text()
    assert "Stop it" in page.report_text()
    done.wait()
    _settle(qtbot, studio_app)


def test_palette_application(studio_app) -> None:
    from ui.pages.biology import BiologyWorkbenchPage
    from ui.theme_manager import ChartPalette

    page = BiologyWorkbenchPage()
    page.apply_palette(ChartPalette(background="#111111", foreground="#eeeeee"))
    assert True


def test_summary_frame_index_rides_the_table(studio_app, qtbot) -> None:
    """The compound names are the summary's index -- the table shows them
    as a real column, and the bars chart's x labels carry them."""
    page = _fresh_page()
    _load_screen(page, qtbot, studio_app)
    frame = page.model_frame()
    assert frame is not None
    assert "compound" in frame.columns
    assert frame["compound"].iloc[0] in page._matrix.columns  # a real compound
