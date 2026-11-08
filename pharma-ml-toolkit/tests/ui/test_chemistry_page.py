"""Chemistry Workbench page tests (increment 2): real API, real guards.

Everything runs offscreen against the real backend entry points the
playground cards use (``standardize_csv``, ``compute_feature_frame``);
the batch guard is exercised by shrinking ``MemAvailable``, never by
faking a refusal.
"""

from __future__ import annotations

import time

import pandas as pd

_SMILES = [
    "CCO",
    "c1ccccc1O",
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",  # caffeine
    "not-a-smiles",  # the drop row
]


def _drain(pool_wait: float = 5.0) -> None:
    from PySide6.QtCore import QThreadPool

    deadline = time.monotonic() + pool_wait
    while QThreadPool.globalInstance().activeThreadCount() > 0 and time.monotonic() < deadline:
        time.sleep(0.01)


def _settle(qtbot, qapp) -> None:
    """Wait out the pool, then flush the queued result slots.

    ``job_ended`` is emitted from JobRunner._finish BEFORE the page's
    finished/failed slot (connected later) runs — both are queued, so
    the GUI-thread handler needs one event-loop pump to land.
    """
    _drain()
    qtbot.wait(30)
    qapp.processEvents()


def _tiny_frame() -> pd.DataFrame:
    return pd.DataFrame({"smiles": _SMILES, "id": [f"m{i}" for i in range(len(_SMILES))]})


def _page_with_frame(frame: pd.DataFrame, column: str = "smiles"):
    """A page whose dataset is replaced by the given frame (the combo too)."""
    from ui.pages.chemistry import ChemistryWorkbenchPage

    page = ChemistryWorkbenchPage()
    page._frame = frame
    page._smiles_combo.blockSignals(True)
    page._smiles_combo.clear()
    page._smiles_combo.addItem(column)
    page._smiles_combo.setCurrentText(column)
    page._smiles_combo.blockSignals(False)
    return page


# -- construction / data -------------------------------------------------------


def test_page_constructs_with_builtin_datasets(studio_app) -> None:
    from ui.pages.chemistry import _BUILTIN, ChemistryWorkbenchPage

    page = ChemistryWorkbenchPage()
    assert page._dataset_combo.count() == len(_BUILTIN) + 1
    assert page._families_combo.count() == 4


def test_builtin_dataset_autodetects_smiles_column(studio_app) -> None:
    from ui.pages.chemistry import ChemistryWorkbenchPage

    page = ChemistryWorkbenchPage()
    page._dataset_combo.setCurrentText("ligand_library.csv")
    assert page._frame is not None and len(page._frame) > 0
    assert "SMILES" in [page._smiles_combo.itemText(i) for i in range(page._smiles_combo.count())]
    assert page._smiles_combo.currentText() == "SMILES"


def test_all_families_preset_maps_to_the_canonical_registry(studio_app) -> None:
    from pharmaml.chem import FEATURE_FAMILIES
    from ui.pages.chemistry import _FAMILY_PRESETS

    assert _FAMILY_PRESETS["All families (heavy)"] is None
    assert _FAMILY_PRESETS["descriptors (fast)"] == ("descriptors",)
    for families in _FAMILY_PRESETS.values():
        if families is not None:
            assert set(families) <= set(FEATURE_FAMILIES)


def test_no_smiles_column_message(studio_app) -> None:
    from ui.pages.chemistry import ChemistryWorkbenchPage

    page = ChemistryWorkbenchPage()
    page._frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    page._smiles_combo.blockSignals(True)
    page._smiles_combo.clear()
    page._smiles_combo.addItem("")
    page._smiles_combo.blockSignals(False)
    page._start_features()
    assert "Pick a SMILES column" in page.report_text()


# -- guard -----------------------------------------------------------------------


def test_batch_guard_refuses_with_arithmetic(studio_app, qtbot, monkeypatch) -> None:
    import ui.pages.chemistry as chem

    page = _page_with_frame(_tiny_frame())
    # 0.001 MB available: the five-molecule probe matrix (~5 KB) cannot fit
    # half of it — the guard refuses with its arithmetic before any compute.
    monkeypatch.setattr(chem, "_meminfo_mb", lambda name: 0.001 if name == "MemAvailable" else None)
    page._start_features()
    qtbot.waitSignal(page._runner.job_ended, timeout=120_000).wait()
    _settle(qtbot, studio_app)
    assert "batch refused by the memory guard" in page.report_text()
    assert "× 8 B × 3 copies" in page.report_text()
    assert "MB is available" in page.report_text()


def test_batch_guard_stands_down_off_linux(studio_app, qtbot, monkeypatch) -> None:
    import ui.pages.chemistry as chem

    page = _page_with_frame(_tiny_frame())
    monkeypatch.setattr(chem, "_meminfo_mb", lambda name: None)
    page._start_features()  # no refusal: off-Linux the guards stand down
    qtbot.waitSignal(page._runner.job_ended, timeout=120_000).wait()
    _settle(qtbot, studio_app)
    assert "refused" not in page.report_text()
    assert "4/5 molecules kept" in page.report_text()


def test_empty_smiles_column_refuses(studio_app, qtbot) -> None:
    page = _page_with_frame(pd.DataFrame({"smiles": ["", "  ", ""]}))
    page._start_features()
    qtbot.waitSignal(page._runner.job_ended, timeout=60_000).wait()
    _settle(qtbot, studio_app)
    assert "no parseable input" in page.report_text()


# -- the real jobs ----------------------------------------------------------------


def test_standardize_job_produces_the_real_report(studio_app, qtbot) -> None:
    page = _page_with_frame(_tiny_frame())
    page._start_standardize()
    qtbot.waitSignal(page._runner.job_ended, timeout=120_000).wait()
    _settle(qtbot, studio_app)
    frame = page.model_frame()
    assert frame is not None and len(frame) == len(_SMILES)
    assert {"identifier", "original", "canonical", "status"} <= set(frame.columns)
    text = page.report_text()
    assert "Standardization —" in text
    assert "rows 5" in text


def test_feature_job_descriptors_family(studio_app, qtbot) -> None:
    page = _page_with_frame(_tiny_frame())
    page._families_combo.setCurrentText("descriptors (fast)")
    page._start_features()
    qtbot.waitSignal(page._runner.job_ended, timeout=180_000).wait()
    _settle(qtbot, studio_app)
    frame = page.model_frame()
    assert frame is not None and len(frame) == 4  # the invalid SMILES dropped
    assert frame.shape[1] > 20  # the descriptor block (43 columns on this stack)
    text = page.report_text()
    assert "4/5 molecules kept" in text
    assert "1 dropped" in text
    assert "descriptors " in text
    # the histogram combo is populated with real feature columns
    assert page._column_combo.count() > 10


def test_all_families_preset_resolves_to_the_canonical_registry(studio_app, qtbot, monkeypatch) -> None:
    """The heavy preset maps None -> FEATURE_FAMILIES (mypy caught the
    UnboundLocalError class here — this run proves the resolution)."""
    import ui.pages.chemistry as chem

    page = _page_with_frame(pd.DataFrame({"smiles": ["CCO", "c1ccccc1O"]}))
    monkeypatch.setattr(chem, "_meminfo_mb", lambda name: None)  # guard stands down
    page._families_combo.setCurrentText("All families (heavy)")
    page._start_features()
    qtbot.waitSignal(page._runner.job_ended, timeout=300_000).wait()
    _settle(qtbot, studio_app)
    text = page.report_text()
    assert "2/2 molecules kept" in text
    assert "shape_3d" in text and "rism" in text  # every family ran


def test_feature_job_cooperative_stop(studio_app, qtbot) -> None:
    """Stop pressed mid-run lands as a clean cancellation, not a crash."""
    page = _page_with_frame(
        pd.DataFrame(
            {
                "smiles": [
                    "CCO",
                    "c1ccccc1O",
                    "CC(=O)Oc1ccccc1C(=O)O",
                    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
                    "CCCCCCCCCC(=O)O",
                ]
                * 400  # 2000 molecules: the run outlives the stop request
            }
        )
    )
    page._start_features()
    deadline = time.monotonic() + 10.0
    while not page._runner.busy and time.monotonic() < deadline:
        time.sleep(0.01)
    assert page._runner.busy  # the job started
    page._runner.request_stop()  # the factory's checkpoint honours it
    qtbot.waitSignal(page._runner.job_ended, timeout=180_000).wait()
    _settle(qtbot, studio_app)
    assert not page._runner.busy
    assert "Cancelled after" in page.report_text()


def test_duplicate_submission_refused(studio_app, qtbot) -> None:
    page = _page_with_frame(_tiny_frame())
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)

    def slow_job(progress) -> None:
        time.sleep(0.4)

    page._runner.submit("hold-the-runner", slow_job)
    page._start_features()  # refused while the hold job runs
    assert "already running" in page.report_text()
    done.wait()
    _settle(qtbot, studio_app)


def test_palette_application(studio_app) -> None:
    from ui.pages.chemistry import ChemistryWorkbenchPage
    from ui.theme_manager import ChartPalette

    page = ChemistryWorkbenchPage()
    page.apply_palette(ChartPalette(background="#111111", foreground="#eeeeee"))
    assert True
