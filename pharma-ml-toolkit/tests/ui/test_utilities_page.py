"""Utilities Workbench page tests -- the round-227 studio port.

The playground's Utilities tab (the rar's others/General scripts)
migrated to the Studio's Utilities Workbench page (the owner's
"everything must be in studio.py" directive, increment 2).  Everything
runs offscreen against the real backend entry points: the five card
families' pre-flight refusals, the merge over the shipped example
tables, the three-way reconciliation with its native bar chart, the
annotate-with-SMILES join, the approved-drugs filter (CSV + the
python-docx Word report), the campaign file tools over the shipped
14-ligand example tree (collect/list/delete dry-run), and the dense
reindex with its gap rows.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd


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
    from ui.pages.utilities import UtilitiesWorkbenchPage

    return UtilitiesWorkbenchPage()


# -- construction --------------------------------------------------------------- #


def test_page_constructs_with_the_five_cards(studio_app) -> None:
    page = _fresh_page()
    assert [page._tabs.tabText(i) for i in range(page._tabs.count())] == [
        "Merge & split",
        "Compare & annotate",
        "Approved drugs",
        "Campaign tools",
        "Reindex",
        # round 24 (part6 §17A): the workflow-templates browser tab
        "Workflow templates",
    ]
    page.deleteLater()


def test_mode_tokens_mirror_the_playground_cards(studio_app) -> None:
    from ui.pages.utilities import COMPARE_MODES, MERGE_MODES, TOOL_MODES

    assert MERGE_MODES == (
        "merge picked files",
        "merge whole folder",
        "split into chunks",
    )
    assert COMPARE_MODES == (
        "compare a column (A vs B)",
        "annotate names with SMILES",
    )
    assert TOOL_MODES == (
        "collect files (name/pattern)",
        "list file names",
        "distribute files into folders",
        "delete named files",
        "cleanup tree (keep list)",
    )
    page = _fresh_page()
    # the card defaults ride the fields verbatim
    assert page._merge_mode.currentText() == "merge picked files"
    assert page._merge_ext.text() == ".csv"
    assert page._merge_chunk.text() == "1500000"
    assert page._cmp_column.text() == "Ligand"
    assert page._cmp_ref_smiles.text() == "SMILES"
    assert page._appr_col.text() == "regulatory approval"
    assert page._appr_status.text() == "approved"
    assert page._tool_filename.text() == "r.dlg"
    assert page._tool_pattern.text() == r"\d+\.pdbqt"
    assert page._tool_keep.text() == "r.pdbqt,r.dlg,r.gpf,r.dpf"
    assert page._tool_dry.isChecked() is True  # destructive paths dry by default
    assert page._reindex_column.text() == "ligand"
    assert page._reindex_start.text() == "0"
    assert page._reindex_stop.text() == "300"
    assert page._reindex_pattern.text() == r"(\d+)"
    page.deleteLater()


# -- pre-flight refusals (nothing picked / named) --------------------------------- #


def test_refusals_before_any_job(studio_app) -> None:
    page = _fresh_page()
    page._start_merge()
    assert page.report_text() == "Pick the tables to merge first."
    page._start_compare()
    assert page.report_text() == "Pick table A and table B first."
    page._start_approved()
    assert page.report_text() == "Pick the approval export files first."
    page._start_tools()
    assert page.report_text() == "Pick the campaign folder first."
    page._start_reindex()
    assert page.report_text() == "Pick the CSV first."
    assert page._runner.busy is False  # every refusal fired BEFORE any job
    page.deleteLater()


def test_split_refuses_without_a_parts_folder(studio_app) -> None:
    page = _fresh_page()
    page._merge_mode.setCurrentText("split into chunks")
    page._merge_split_file = Path("whatever.csv")  # a source, but no parts folder
    page._start_merge()
    assert "parts folder" in page.report_text()
    assert page._runner.busy is False
    page.deleteLater()


# -- the example loaders ----------------------------------------------------------- #


def test_merge_example_loader_picks_real_tables(studio_app) -> None:
    page = _fresh_page()
    page._load_merge_example()
    assert len(page._merge_files) == 2
    assert all(p.is_file() for p in page._merge_files)
    assert page._merge_split_file is not None and page._merge_split_file.is_file()
    assert "example tables" in page.report_text()
    page.deleteLater()


def test_campaign_example_loader_points_at_the_shipped_tree(studio_app) -> None:
    page = _fresh_page()
    page._load_tool_example()
    assert page._tool_folder is not None and page._tool_folder.is_dir()
    sub_folders = [p for p in page._tool_folder.iterdir() if p.is_dir()]
    assert len(sub_folders) == 14  # the 14 ligand folders
    page.deleteLater()


# -- merge & split, functional ------------------------------------------------------ #


def test_merge_over_the_example_tables(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._load_merge_example()
    page._merge_btn.click()
    _settle(qtbot, studio_app)
    frame = page.bound_frame()
    assert frame is not None and len(frame) > 0
    assert page.report_text().startswith("Merged —")
    assert "rows" in page.report_text()
    # the merged frame keeps both source schemas' column union
    assert len(frame.columns) >= 4
    # the six-format export writes through the real engine
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "merged.csv"
        page._write_frame(frame, str(out), "merge")
        assert out.is_file() and out.stat().st_size > 0
        assert "Saved —" in page.report_text()
    page.deleteLater()


def test_split_streams_the_example_into_parts(studio_app, qtbot, tmp_path: Path) -> None:
    page = _fresh_page()
    page._merge_split_file = (
        Path(__file__).resolve().parents[2] / "examples" / "data" / "qsar_regression.csv"
    )
    page._split_out_dir = tmp_path
    page._merge_mode.setCurrentText("split into chunks")
    page._merge_chunk.setText("100")  # 260 rows -> 3 parts
    page._merge_btn.click()
    _settle(qtbot, studio_app)
    parts = sorted(tmp_path.glob("*.csv"))
    assert len(parts) >= 3
    rows = sum(len(pd.read_csv(p)) for p in parts)
    assert rows == 260  # every row streamed exactly once
    assert page.report_text().startswith("Split into")
    page.deleteLater()


# -- compare & annotate, functional -------------------------------------------------- #


def _write_pair(tmp_path: Path) -> tuple[Path, Path]:
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    pd.DataFrame({"Ligand": ["lig1.pdbqt", "LIG2", " lig3 "]}).to_csv(a, index=False)
    pd.DataFrame({"Ligand": ["lig1", "lig2", "lig4"]}).to_csv(b, index=False)
    return a, b


def test_compare_reconciles_and_draws_native_bars(studio_app, qtbot, tmp_path: Path) -> None:
    page = _fresh_page()
    page._cmp_a, page._cmp_b = _write_pair(tmp_path)
    page._cmp_btn.click()
    _settle(qtbot, studio_app)
    frame = page.bound_frame()
    assert frame is not None
    sections = frame["section"].tolist()
    assert sections.count("common") == 2  # lig1 (extension-normalised) + lig2 (case)
    assert sections.count("only in A") == 1  # ' lig3 ' trimmed
    assert sections.count("only in B") == 1  # lig4
    assert page.report_text() == "Compared — common 2 · only A 1 · only B 1"
    # the reconciliation figure is native PyQtGraph bars (§16.8: no matplotlib)
    items = page._plot.widget().getPlotItem().listDataItems()
    assert items, "the reconciliation bars must draw"
    page.deleteLater()


def test_annotate_joins_smiles_from_the_reference(studio_app, qtbot, tmp_path: Path) -> None:
    page = _fresh_page()
    a = tmp_path / "names.csv"
    b = tmp_path / "reference.csv"
    pd.DataFrame({"Ligand": ["aspirin", "ibuprofen", "unknown"]}).to_csv(a, index=False)
    pd.DataFrame(
        {"Ligand": ["aspirin", "ibuprofen"], "SMILES": ["CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O"]}
    ).to_csv(b, index=False)
    page._cmp_a, page._cmp_b = a, b
    page._cmp_mode.setCurrentText("annotate names with SMILES")
    page._cmp_btn.click()
    _settle(qtbot, studio_app)
    frame = page.bound_frame()
    assert frame is not None and len(frame) == 3
    assert "Ligand_SMILES" in frame.columns
    assert frame.loc[frame["Ligand"] == "aspirin", "Ligand_SMILES"].iloc[0] == "CC(=O)Oc1ccccc1C(=O)O"
    assert "Annotated —" in page.report_text() and "1 found SMILES" in page.report_text()
    page.deleteLater()


# -- approved drugs, functional -------------------------------------------------------- #


def test_approved_filter_and_both_reports(studio_app, qtbot, tmp_path: Path) -> None:
    page = _fresh_page()
    export = tmp_path / "approvals.tsv"
    pd.DataFrame(
        {
            "drug": ["aspirin", "ibuprofen", "aspirin", "withdrawn"],
            "regulatory approval": ["approved", "approved", "approved", "withdrawn"],
        }
    ).to_csv(export, sep="\t", index=False)
    page._appr_files = [export]
    page._appr_btn.click()
    _settle(qtbot, studio_app)
    frame = page.bound_frame()
    assert frame is not None
    assert frame["drug"].tolist() == ["aspirin", "ibuprofen"]  # unique + sorted
    assert page.report_text() == "Approved — 2 drug name(s) (status 'approved')"
    # CSV arm through the real writer
    out_csv = tmp_path / "approved.csv"
    page._write_frame(frame, str(out_csv), "approved")
    assert out_csv.is_file()
    # DOCX arm through the real python-docx engine
    from pharmaml.viz import export_docx

    out_docx = tmp_path / "approved.docx"
    written = export_docx(frame, out_docx)
    assert Path(written).is_file() and Path(written).stat().st_size > 0
    page.deleteLater()


# -- campaign file tools over the shipped tree ------------------------------------------- #


def test_collect_dry_run_over_the_example_campaign(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._load_tool_example()
    page._tool_out = None  # the default collected/ arm, dry-run: nothing written
    page._tool_btn.click()
    _settle(qtbot, studio_app)
    frame = page.bound_frame()
    assert frame is not None
    assert len(frame) == 13  # the 13 ligand folders that carry r.dlg (of 14)
    assert page.report_text() == "Collected — 13 match(es) · would copy 13 into collected"
    assert not (page._tool_folder / "collected").exists()  # nothing touched the disk
    page.deleteLater()


def test_list_mode_names_the_example_trees_csv(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._load_tool_example()
    page._tool_mode.setCurrentText("list file names")
    page._tool_btn.click()
    _settle(qtbot, studio_app)
    frame = page.bound_frame()
    assert frame is not None
    assert frame["name"].tolist() == ["ligands"]  # the Nameextracter contract: bare names
    assert page.report_text().startswith("Listed — 1 .csv name")
    page.deleteLater()


def test_delete_dry_run_reports_the_count(studio_app, qtbot, tmp_path: Path) -> None:
    page = _fresh_page()
    (tmp_path / "01_one").mkdir()
    (tmp_path / "02_two").mkdir()
    (tmp_path / "01_one" / "r.dlg").write_text("x", encoding="utf-8")
    (tmp_path / "02_two" / "r.dlg").write_text("x", encoding="utf-8")
    page._tool_folder = tmp_path
    page._tool_mode.setCurrentText("delete named files")
    page._tool_btn.click()  # dry-run checkbox defaults True
    _settle(qtbot, studio_app)
    assert "would delete 2" in page.report_text()
    assert (tmp_path / "01_one" / "r.dlg").exists()  # nothing deleted
    page.deleteLater()


def test_distribute_copies_into_every_subfolder(studio_app, qtbot, tmp_path: Path) -> None:
    page = _fresh_page()
    for name in ("01_one", "02_two"):
        (tmp_path / name).mkdir()
    shared = tmp_path / "receptor.pdbqt"
    shared.write_text("ROOT", encoding="utf-8")
    page._tool_folder = tmp_path
    page._tool_files = [shared]
    page._tool_mode.setCurrentText("distribute files into folders")
    page._tool_dry.setChecked(False)  # real run: the card asks once -- tests answer Yes
    from PySide6.QtWidgets import QMessageBox

    import ui.pages.utilities as utilities_mod

    original = utilities_mod.QMessageBox.question
    utilities_mod.QMessageBox.question = staticmethod(
        lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    try:
        page._tool_btn.click()
        _settle(qtbot, studio_app)
    finally:
        utilities_mod.QMessageBox.question = original
    assert (tmp_path / "01_one" / "receptor.pdbqt").read_text(encoding="utf-8") == "ROOT"
    assert (tmp_path / "02_two" / "receptor.pdbqt").exists()
    assert page.report_text().startswith("Distributed — 2 copy/copies")
    page.deleteLater()


# -- reindex, functional --------------------------------------------------------------- #


def test_reindex_builds_the_dense_table_with_gaps(studio_app, qtbot, tmp_path: Path) -> None:
    page = _fresh_page()
    sparse = tmp_path / "sparse.csv"
    pd.DataFrame({"ligand": ["1_result.dlg", "3_result.dlg", "5_result.dlg"]}).to_csv(
        sparse, index=False
    )
    page._reindex_path = sparse
    page._reindex_stop.setText("5")
    page._reindex_btn.click()
    _settle(qtbot, studio_app)
    frame = page.bound_frame()
    assert frame is not None and len(frame) == 6  # dense 0..5
    missing = int(frame.iloc[:, 0].isna().sum())
    assert missing == 3  # 0, 2, 4 are gap rows
    assert page.report_text() == "Reindexed — dense 0…5 · 6 rows · 3 gap row(s)"
    page.deleteLater()


def test_reindex_refuses_bad_ranges(studio_app) -> None:
    page = _fresh_page()
    page._reindex_path = Path("whatever.csv")
    page._reindex_stop.setText("2")
    page._reindex_start.setText("5")  # stop < start
    page._start_reindex()
    assert page.report_text() == "stop must be >= start"
    page._reindex_start.setText("0")
    page._reindex_column.setText("")  # no column named
    page._start_reindex()
    assert page.report_text() == "Name the number column."
    assert page._runner.busy is False
    page.deleteLater()


# -- main-window integration -------------------------------------------------------------- #


def test_main_window_carries_the_utilities_page(main_window) -> None:
    main_window.go_to("utilities")
    assert main_window.pages.currentIndex() == 16  # after Image Lab, before Settings (the ML Workbench moved everyone +1, the round-250 Molecular Dynamics page +1 more)
    page = main_window.utilities_page
    assert page._tabs.count() == 6  # round 24: + the §17A Workflow-templates tab
