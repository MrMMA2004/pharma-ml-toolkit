"""Docking Workbench page tests (increment 5): real API, real numbers.

Everything runs offscreen against the real backend entry points the
playground cards call (``check_docking_completion`` +
``scan_docking_directory`` + ``consensus_rescore`` /
``parse_dlg_cluster_table`` / ``load_ligand_csv`` / ``target_frame`` /
``parse_search_box``). Every asserted number was printed by the engines
first, on this exact bundled data: 14 folders -> 12 OK / 1 missing log
/ 1 incomplete, metformin the consensus top (0.7817), aspirin's five
clusters with rank 1 at -4.91 kcal/mol (15 runs), 58 DOCKSTRING
targets, DRD2's box (9.250, 6.167, -7.000) / (30, 30, 32), the library
at 8 valid + 2 invalid rows.
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
    from ui.pages.docking import DockingWorkbenchPage

    return DockingWorkbenchPage()


def _scan_campaign(page, qtbot, studio_app) -> None:
    """Run the campaign job once so the cross-tab scan state exists."""
    page._start_campaign()
    qtbot.waitSignal(page._runner.job_ended, timeout=120_000).wait()
    _settle(qtbot, studio_app)


# -- construction ---------------------------------------------------------------


def test_page_constructs_with_five_engine_tabs(studio_app) -> None:
    page = _fresh_page()
    assert page._tabs.count() == 5
    titles = [page._tabs.tabText(i) for i in range(page._tabs.count())]
    assert titles == [
        "Campaign summary",
        "Cluster analysis",
        "Ligand library",
        "DOCKSTRING lab",
        "Contacts & PLIF",
    ]


def test_cluster_refuses_without_a_campaign(studio_app) -> None:
    page = _fresh_page()
    page._start_cluster()
    assert "Scan a campaign first" in page.report_text()


def test_target_config_refuses_without_the_registry(studio_app) -> None:
    page = _fresh_page()
    page._show_target_config()
    assert "Load the target registry first" in page.report_text()


# -- the campaign scan & rescore job ------------------------------------------------


def test_campaign_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._start_campaign()
    qtbot.waitSignal(page._runner.job_ended, timeout=120_000).wait()
    _settle(qtbot, studio_app)

    text = page.report_text()
    assert "14 folders" in text
    assert "12 OK / 1 missing log / 1 incomplete" in text
    assert "consensus over 3 evidence columns" in text
    assert "top: metformin" in text

    frame = page.model_frame()
    assert frame is not None
    assert len(frame) == 12  # one row per OK folder
    assert {
        "folder", "ligand", "best_lowest_energy", "best_num_in_cluster",
        "num_heavy_atoms", "ligand_efficiency", "consensus",
    } <= set(frame.columns)
    # ranked best-first; the engine-printed top hit
    assert frame["ligand"].iloc[0] == "metformin"
    assert abs(float(frame["consensus"].iloc[0]) - 0.781674) < 1e-4
    assert float(frame["consensus"].iloc[0]) >= float(frame["consensus"].iloc[-1])
    # the RDKit heavy atoms rode the table (metformin C4H11N2+ dimer -> 9)
    assert int(frame.loc[frame["ligand"] == "metformin", "num_heavy_atoms"].iloc[0]) == 9

    # the cross-tab state landed: the OK folders feed the cluster combo
    assert page._scan is not None and len(page._scan) == 14
    assert page._ok_frame is not None
    assert page._cluster_ligand.count() == 12
    assert "01_aspirin" in [page._cluster_ligand.itemText(i) for i in range(12)]


def test_campaign_scan_status_rows_ride_the_scan(studio_app, qtbot) -> None:
    page = _fresh_page()
    _scan_campaign(page, qtbot, studio_app)
    scan = page._scan
    assert scan is not None
    assert (scan["status"] == "OK").sum() == 12
    assert (scan["status"] == "MISSING_LOG").sum() == 1  # 08_benzocaine
    # 10_nicotine: interrupted before clustering -> explicit parse error status
    nicotine = scan.loc[scan["folder"] == "10_nicotine", "status"].iloc[0]
    assert str(nicotine).startswith("PARSE_ERROR")


# -- the cluster table job ------------------------------------------------------------


def test_cluster_table_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    _scan_campaign(page, qtbot, studio_app)
    page._cluster_ligand.setCurrentText("01_aspirin")
    page._start_cluster()
    qtbot.waitSignal(page._runner.job_ended, timeout=120_000).wait()
    _settle(qtbot, studio_app)

    text = page.report_text()
    assert "01_aspirin — 5 clusters" in text
    assert "rank 1: -4.91 kcal/mol (15 runs)" in text
    assert "most populated: rank 1 (15 runs)" in text

    frame = page.model_frame()
    assert frame is not None
    assert len(frame) == 5
    assert {"rank", "lowest_energy", "mean_energy", "num_in_cluster"} <= set(frame.columns)
    assert int(frame["num_in_cluster"].sum()) == 55  # 15+13+8+13+6


def test_cluster_most_populated_is_not_always_rank_one(studio_app, qtbot) -> None:
    page = _fresh_page()
    _scan_campaign(page, qtbot, studio_app)
    page._cluster_ligand.setCurrentText("05_caffeine")
    page._start_cluster()
    qtbot.waitSignal(page._runner.job_ended, timeout=120_000).wait()
    _settle(qtbot, studio_app)

    text = page.report_text()
    assert "05_caffeine — 5 clusters" in text
    assert "rank 1: -4.56 kcal/mol (9 runs)" in text
    assert "most populated: rank 2 (15 runs)" in text


# -- the ligand library job ------------------------------------------------------------


def test_library_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._start_library()
    qtbot.waitSignal(page._runner.job_ended, timeout=120_000).wait()
    _settle(qtbot, studio_app)

    text = page.report_text()
    assert "10 rows · 8 valid · 2 invalid (broken_row, empty_row)" in text

    frame = page.model_frame()
    assert frame is not None
    assert len(frame) == 10
    assert {"name", "smiles", "status", "error", "num_heavy_atoms", "n_rotatable"} <= set(frame.columns)
    # the invalid rows ride the table with their reasons, properties NaN
    broken = frame.loc[frame["name"] == "broken_row"].iloc[0]
    assert broken["status"] == "invalid"
    assert broken["error"] == "RDKit could not parse this SMILES"
    assert pd_is_na(broken["num_heavy_atoms"])
    empty = frame.loc[frame["name"] == "empty_row"].iloc[0]
    assert empty["error"] == "empty or non-text SMILES cell"
    # the valid rows carry the RDKit properties the prep card reports
    naproxen = frame.loc[frame["name"] == "naproxen"].iloc[0]
    assert int(naproxen["num_heavy_atoms"]) == 17
    assert int(naproxen["n_rotatable"]) == 3
    caffeine = frame.loc[frame["name"] == "caffeine"].iloc[0]
    assert int(caffeine["n_rotatable"]) == 0  # rigid xanthine


def pd_is_na(value) -> bool:
    import pandas as pd

    return bool(pd.isna(value))


# -- the DOCKSTRING registry job ---------------------------------------------------------


def test_registry_real_numbers(studio_app, qtbot) -> None:
    page = _fresh_page()
    page._start_registry()
    qtbot.waitSignal(page._runner.job_ended, timeout=120_000).wait()
    _settle(qtbot, studio_app)

    text = page.report_text()
    assert "58 indexed targets" in text
    assert "3 objectives (F2, promiscuous_PPAR, selective_JAK2)" in text
    assert "first ABL1 · last THRB" in text

    frame = page.model_frame()
    assert frame is not None
    assert len(frame) == 58
    assert {"target", "center_x", "center_y", "center_z", "size_x", "size_y", "size_z"} <= set(frame.columns)
    assert page._target_combo.count() == 58


def test_target_config_preview(studio_app, qtbot) -> None:
    page = _fresh_page()
    _scan_campaign(page, qtbot, studio_app)  # runner warm-up; registry not needed
    page._start_registry()
    qtbot.waitSignal(page._runner.job_ended, timeout=120_000).wait()
    _settle(qtbot, studio_app)

    page._target_combo.setCurrentText("DRD2")
    page._show_target_config()
    text = page.report_text()
    assert "DRD2 config:" in text
    assert "center_x = 9.250" in text
    assert "center_z = -7.000" in text
    assert "size_z = 32.000" in text


def test_custom_box_validates(studio_app) -> None:
    from pharmaml.docking import get_target

    page = _fresh_page()
    page._box_edit.setPlainText(get_target("DRD2").config_text())
    page._validate_box()
    text = page.report_text()
    assert "Custom box — center (9.25, 6.167, -7)" in text
    assert "size (30, 30, 32)" in text


def test_custom_box_rejects_an_incomplete_config(studio_app) -> None:
    page = _fresh_page()
    page._box_edit.setPlainText("center_x = 1.0\nsize_y = 2.0\n")
    page._validate_box()
    assert "search-box config is missing" in page.report_text()


# -- job plumbing ------------------------------------------------------------------------


def test_duplicate_submission_refused(studio_app, qtbot) -> None:
    page = _fresh_page()
    done = qtbot.waitSignal(page._runner.job_ended, timeout=60_000)

    def slow_job(progress) -> None:
        time.sleep(0.4)

    page._runner.submit("hold-the-runner", slow_job)
    page._start_library()  # refused while the hold job runs
    assert "already running" in page.report_text()
    done.wait()
    _settle(qtbot, studio_app)


def test_palette_application(studio_app) -> None:
    from ui.pages.docking import DockingWorkbenchPage
    from ui.theme_manager import ChartPalette

    page = DockingWorkbenchPage()
    page.apply_palette(ChartPalette(background="#111111", foreground="#eeeeee"))
    assert True


# -- the contacts & PLIF twins (r264) ----------------------------------------------


def test_contact_map_real_numbers(studio_app, qtbot) -> None:
    """The shipped demo pair, engine-first: 91 contacts over 25 residues,
    the closest an ASN103 H-bond at 3.0 A -- every number the job prints
    was printed by compute_contacts on this exact bundled data first."""
    page = _fresh_page()
    page._start_contacts()
    qtbot.waitSignal(page._runner.job_ended, timeout=120_000).wait()
    _settle(qtbot, studio_app)

    text = page.report_text()
    assert "91 pair(s) within 5 A of 25 residue(s)" in text
    assert "closest: A:ASN103 OD1 -- O2 3.00 A (hbond)" in text

    frame = page.model_frame()
    assert frame is not None
    assert len(frame) == 91
    assert set(frame["interaction"]) <= {"salt_bridge", "hbond", "hydrophobic", "vdw"}
    assert int((frame["interaction"] == "salt_bridge").sum()) == 3
    assert int((frame["interaction"] == "hydrophobic").sum()) == 24
    # the table is closest-first (compute_contacts' sort contract)
    distances = frame["distance (A)"].to_numpy(dtype=float)
    assert distances[0] == distances.min()


def test_contact_map_cutoff_widens_the_shell(studio_app, qtbot) -> None:
    """The spin drives the shell honestly: 6 A admits strictly more
    contacts than 5 A over the same pair."""
    page = _fresh_page()
    page._cutoff_spin.setValue(6.0)
    page._start_contacts()
    qtbot.waitSignal(page._runner.job_ended, timeout=120_000).wait()
    _settle(qtbot, studio_app)
    assert "within 6 A" in page.report_text()
    frame = page.model_frame()
    assert frame is not None and len(frame) > 91


def test_plif_real_numbers(studio_app, qtbot) -> None:
    """The PLIF twin: 25 residue rows x 4 class columns, the class totals
    the counts frame prints -- 3/3/24/61 over 91 contacts."""
    page = _fresh_page()
    page._start_plif()
    qtbot.waitSignal(page._runner.job_ended, timeout=120_000).wait()
    _settle(qtbot, studio_app)

    text = page.report_text()
    assert "91 contact(s) classified over 25 residue(s)" in text
    assert "salt_bridge 3 / hbond 3 / hydrophobic 24 / vdw 61" in text

    frame = page.model_frame()
    assert frame is not None
    assert list(frame.columns) == ["salt_bridge", "hbond", "hydrophobic", "vdw"]
    assert len(frame) == 25
    assert int(frame["salt_bridge"].sum()) == 3
    assert int(frame["hydrophobic"].sum()) == 24
    # ASP101 keeps a salt bridge; TRP62/63/108 are the hydrophobic walls
    assert int(frame.loc["A:ASP101", "salt_bridge"]) == 1
    assert int(frame.loc["A:TRP108", "hydrophobic"]) == 8
