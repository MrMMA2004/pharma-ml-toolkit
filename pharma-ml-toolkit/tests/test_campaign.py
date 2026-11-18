"""Unit tests for the docking-campaign additions: DPF/GPF generation
(``others/dpfer.py``), campaign file management (``others/General/*``),
the torsion-consistency audit (``ligand generation/ligprep_match.py``) and
the config-driven Vina batch (``others/Vina/*``)."""

from __future__ import annotations

import logging
import struct
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

import pharmaml.docking.smiles_docking as smiles_mod
import pharmaml.docking.workflow as workflow_mod
from pharmaml.docking.campaign import (
    cleanup_tree,
    collect_files,
    delete_named_files,
    distribute_files,
    list_file_names,
    reindex_by_number,
    scan_gpf_ligand_types,
    write_config_batch,
)
from pharmaml.docking.ligands import (
    TorsionCheck,
    check_torsion_consistency,
    scan_torsion_consistency,
)
from pharmaml.docking.parsing import scan_vina_directory
from pharmaml.docking.preparation import (
    GridBox,
    read_gpf,
    write_dpf,
    write_gpf,
)
from pharmaml.docking.smiles_docking import dock_smiles_array
from pharmaml.docking.workflow import (
    parse_vina_config,
    run_autodock4,
    run_vina_batch,
)
from pharmaml.exceptions import DataValidationError, DockingError, DockingParseError


@contextmanager
def _pharmaml_logs_visible():
    """Re-enable propagation on the ``pharmaml`` logger for caplog.

    Same dance as ``test_biology.py`` / ``test_ml_pipeline.py``: the CLI
    tests call :func:`configure_logging`, which sets ``propagate=False``
    on the ``pharmaml`` root logger, so caplog records only reach the
    true root logger while propagation is on (restored on exit).
    """
    pharma_root = logging.getLogger("pharmaml")
    prev_propagate = pharma_root.propagate
    pharma_root.propagate = True
    try:
        yield
    finally:
        pharma_root.propagate = prev_propagate


class TestWriteDPF:
    def test_dpf_structure(self, tmp_path, sample_ligand_pdbqt) -> None:
        ligand = tmp_path / "l.pdbqt"
        ligand.write_text(sample_ligand_pdbqt, encoding="utf-8")
        dpf = write_dpf(ligand, receptor_file="r.pdbqt")
        assert dpf.is_file()
        lines = dpf.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "autodock_parameter_version 4.2"
        assert "outlev 1" in lines
        assert "intelec" in lines
        assert "seed pid time" in lines
        assert "ligand_types C HD N OA" in lines
        assert "fld r.maps.fld" in lines
        assert lines.count("map r.C.map") == 1
        assert lines.count("map r.HD.map") == 1
        assert lines.count("map r.N.map") == 1
        assert lines.count("map r.OA.map") == 1
        assert "elecmap r.e.map" in lines
        assert "desolvmap r.d.map" in lines
        assert "move l.pdbqt" in lines
        assert "tran0 random" in lines
        assert "quaternion0 random" in lines
        assert "dihe0 random" in lines
        assert "torsdof 2" in lines          # TORSDOF 2 in the fixture
        assert "rmstol 2.0" in lines
        assert "extnrg 1000.0" in lines
        assert "e0max 0.0 10000" in lines
        assert "ga_pop_size 150" in lines
        assert "ga_num_evals 2500000" in lines
        assert "ga_num_generations 27000" in lines
        assert "set_ga" in lines
        assert "sw_max_its 300" in lines
        assert "ls_search_freq 0.06" in lines
        assert "set_psw1" in lines
        assert "unbound_model bound" in lines
        assert "ga_run 100" in lines         # dpfer.py default
        assert lines[-1] == "analysis"

    def test_about_line_is_ligand_centroid(self, tmp_path, sample_ligand_pdbqt) -> None:
        ligand = tmp_path / "l.pdbqt"
        ligand.write_text(sample_ligand_pdbqt, encoding="utf-8")
        dpf = write_dpf(ligand)
        about = next(
            line for line in dpf.read_text(encoding="utf-8").splitlines()
            if line.startswith("about")
        )
        cx, cy, cz = (float(value) for value in about.split()[1:])
        # centroid of the four fixture atoms, rounded to 3 decimals
        assert (cx, cy, cz) == pytest.approx((2.006, -0.225, 0.186), abs=1e-6)

    def test_defaults_to_r_dpf_next_to_ligand(self, tmp_path, sample_ligand_pdbqt) -> None:
        ligand = tmp_path / "ligand_07" / "l.pdbqt"
        ligand.parent.mkdir()
        ligand.write_text(sample_ligand_pdbqt, encoding="utf-8")
        dpf = write_dpf(ligand)
        assert dpf == ligand.parent / "r.dpf"

    def test_missing_ligand_raises(self, tmp_path) -> None:
        with pytest.raises(DockingError, match="not found"):
            write_dpf(tmp_path / "ghost.pdbqt")

    def test_custom_ga_run_honoured(self, tmp_path, sample_ligand_pdbqt) -> None:
        ligand = tmp_path / "l.pdbqt"
        ligand.write_text(sample_ligand_pdbqt, encoding="utf-8")
        dpf = write_dpf(ligand, ga_run=25, ga_pop_size=300, ga_num_evals=1000)
        text = dpf.read_text(encoding="utf-8")
        assert "ga_run 25" in text
        assert "ga_pop_size 300" in text
        assert "ga_num_evals 1000" in text
        assert "ga_run 100" not in text

    def test_missing_torsdof_defaults_to_zero(self, tmp_path) -> None:
        ligand = tmp_path / "l.pdbqt"
        ligand.write_text(
            "REMARK no torsdof here\n"
            "ATOM      1  C   LIG A   1       1.000   2.000   3.000  1.00  0.00    0.000 C\n",
            encoding="utf-8",
        )
        dpf = write_dpf(ligand)
        assert "torsdof 0" in dpf.read_text(encoding="utf-8")


class TestReadGPF:
    def test_round_trip_with_write_gpf(self, tmp_path, sample_ligand_pdbqt) -> None:
        ligand = tmp_path / "l.pdbqt"
        ligand.write_text(sample_ligand_pdbqt, encoding="utf-8")
        box = GridBox(1.5, -2.5, 3.5, npts=(40, 50, 60), spacing=0.5)
        gpf = write_gpf(ligand, receptor_file="rec.pdbqt", box=box)
        parsed = read_gpf(gpf)
        assert parsed["ligand_types"] == ["C", "HD", "N", "OA"]
        assert parsed["receptor_types"] == ["A", "C", "HD", "N", "NA", "OA", "SA"]
        assert parsed["npts"] == (40, 50, 60)
        assert parsed["spacing"] == pytest.approx(0.5)
        assert parsed["gridcenter"] == pytest.approx((1.5, -2.5, 3.5))
        assert parsed["receptor"] == "rec.pdbqt"
        assert parsed["maps"] == ["r.C.map", "r.HD.map", "r.N.map", "r.OA.map"]
        assert parsed["smooth"] == pytest.approx(0.5)
        assert parsed["dielectric"] == pytest.approx(-0.1465)

    def test_gpf_without_ligand_types_raises(self, tmp_path) -> None:
        gpf = tmp_path / "r.gpf"
        gpf.write_text("npts 40 40 40 # no ligand types here\n", encoding="utf-8")
        with pytest.raises(DockingParseError, match="ligand_types"):
            read_gpf(gpf)

    def test_missing_gpf_raises(self, tmp_path) -> None:
        with pytest.raises(DockingParseError, match="not found"):
            read_gpf(tmp_path / "missing.gpf")


class TestCollectFiles:
    def _make_dialog_tree(self, base: Path) -> None:
        for name in ("ligand_01", "ligand_02"):
            folder = base / name
            folder.mkdir(parents=True)
            (folder / "r.dlg").write_text(f"dlg of {name}", encoding="utf-8")

    def test_collects_dialogs_renamed_by_folder(self, tmp_path) -> None:
        self._make_dialog_tree(tmp_path)
        report = collect_files(tmp_path)
        assert list(report.columns) == ["source", "destination", "folder", "copied"]
        assert len(report) == 2
        assert report["copied"].all()
        collected = tmp_path / "collected"
        assert sorted(p.name for p in collected.iterdir()) == [
            "ligand_01.dlg", "ligand_02.dlg",
        ]
        row = report.iloc[0]
        assert Path(row["destination"]).read_text(encoding="utf-8") == "dlg of ligand_01"
        assert row["folder"] == "ligand_01"

    def test_dry_run_copies_nothing(self, tmp_path) -> None:
        self._make_dialog_tree(tmp_path)
        report = collect_files(tmp_path, dry_run=True)
        assert len(report) == 2
        assert not report["copied"].any()
        assert not (tmp_path / "collected").exists()

    def test_pattern_mode_collects_numbered_pdbqts(self, tmp_path) -> None:
        one = tmp_path / "1"
        two = tmp_path / "2"
        one.mkdir()
        two.mkdir()
        (one / "1.pdbqt").write_text("ligand 1", encoding="utf-8")
        (two / "2.pdbqt").write_text("ligand 2", encoding="utf-8")
        (two / "r.pdbqt").write_text("receptor", encoding="utf-8")  # not numbered
        report = collect_files(tmp_path, pattern=r"^\d+\.pdbqt$")
        assert len(report) == 2
        collected = tmp_path / "collected"
        assert sorted(p.name for p in collected.iterdir()) == ["1.pdbqt", "2.pdbqt"]
        # the receptor PDBQT must not ride along
        assert "r.pdbqt" not in {p.name for p in collected.iterdir()}

    def test_dedup_suffix_for_same_folder_name(self, tmp_path) -> None:
        (tmp_path / "a" / "shared").mkdir(parents=True)
        (tmp_path / "b" / "shared").mkdir(parents=True)
        (tmp_path / "a" / "shared" / "r.dlg").write_text("a", encoding="utf-8")
        (tmp_path / "b" / "shared" / "r.dlg").write_text("b", encoding="utf-8")
        collect_files(tmp_path)
        names = sorted(p.name for p in (tmp_path / "collected").iterdir())
        assert names == ["shared.dlg", "shared_1.dlg"]

    def test_skips_output_dir(self, tmp_path) -> None:
        (tmp_path / "ligand_01").mkdir()
        (tmp_path / "ligand_01" / "r.dlg").write_text("real", encoding="utf-8")
        out = tmp_path / "out"
        out.mkdir()
        (out / "r.dlg").write_text("stale", encoding="utf-8")  # must not re-match
        report = collect_files(tmp_path, out_dir=out)
        assert len(report) == 1
        assert report.iloc[0]["folder"] == "ligand_01"
        assert sorted(p.name for p in out.iterdir()) == ["ligand_01.dlg", "r.dlg"]

    def test_rerun_never_collects_its_own_output(self, tmp_path) -> None:
        self._make_dialog_tree(tmp_path)
        collect_files(tmp_path)
        second = collect_files(tmp_path)
        # the originals match again (collect copies, never moves) but the
        # files inside collected/ themselves must never be re-collected
        assert "collected" not in set(second["folder"])


class TestListFileNames:
    def test_sorted_stems_and_csv_export(self, tmp_path) -> None:
        for name in ("bravo.csv", "alpha.csv", "charlie.csv", "ignored.txt"):
            (tmp_path / name).write_text("x", encoding="utf-8")
        out = tmp_path / "names.csv"
        names = list_file_names(tmp_path, extension=".csv", out_path=out)
        assert names == ["alpha", "bravo", "charlie"]
        written = pd.read_csv(out)
        assert list(written.columns) == ["name"]
        assert written["name"].tolist() == names

    def test_extension_without_dot_accepted(self, tmp_path) -> None:
        (tmp_path / "one.dlg").write_text("x", encoding="utf-8")
        (tmp_path / "two.dlg").write_text("x", encoding="utf-8")
        assert list_file_names(tmp_path, extension="dlg") == ["one", "two"]

    def test_missing_directory_raises(self, tmp_path) -> None:
        with pytest.raises(DockingError, match="not found"):
            list_file_names(tmp_path / "nowhere")


class TestDistributeFiles:
    def test_distributes_into_every_subdirectory(self, tmp_path) -> None:
        target = tmp_path / "campaign"
        (target / "ligand_01").mkdir(parents=True)
        (target / "ligand_02").mkdir(parents=True)
        (target / "stray.txt").write_text("not a folder", encoding="utf-8")
        source = tmp_path / "r.pdbqt"
        source.write_text("receptor", encoding="utf-8")
        written = distribute_files([source], target)
        assert sorted(str(w.relative_to(target)) for w in written) == [
            "ligand_01/r.pdbqt", "ligand_02/r.pdbqt",
        ]
        assert (target / "ligand_02" / "r.pdbqt").read_text(encoding="utf-8") == "receptor"

    def test_existing_destinations_skipped_unless_overwrite(self, tmp_path) -> None:
        target = tmp_path / "campaign"
        (target / "sub").mkdir(parents=True)
        source = tmp_path / "engine.exe"
        source.write_text("v1", encoding="utf-8")
        assert distribute_files([source], target) == [target / "sub" / "engine.exe"]
        source.write_text("v2", encoding="utf-8")
        assert distribute_files([source], target) == []  # no overwrite
        assert distribute_files([source], target, overwrite=True) == [
            target / "sub" / "engine.exe"
        ]
        assert (target / "sub" / "engine.exe").read_text(encoding="utf-8") == "v2"

    def test_no_subdirectories_returns_empty(self, tmp_path) -> None:
        target = tmp_path / "flat"
        target.mkdir()
        source = tmp_path / "f.txt"
        source.write_text("x", encoding="utf-8")
        assert distribute_files([source], target) == []

    def test_missing_source_raises(self, tmp_path) -> None:
        target = tmp_path / "campaign"
        (target / "sub").mkdir(parents=True)
        with pytest.raises(DockingError, match="not found"):
            distribute_files([tmp_path / "ghost.txt"], target)


class TestCleanupTree:
    def _make_tree(self, base: Path) -> None:
        ligand = base / "ligand_01"
        ligand.mkdir(parents=True)
        for name in ("r.pdbqt", "r.dpf", "r.gpf", "12.pdbqt", "notes.txt"):
            (ligand / name).write_text(name, encoding="utf-8")

    def test_keeps_allowlist_and_pattern(self, tmp_path) -> None:
        self._make_tree(tmp_path)
        report = cleanup_tree(
            tmp_path,
            keep={"r.pdbqt", "r.dpf", "autodock4.exe"},
            keep_pattern=r"^\d+\.pdbqt$",
        )
        assert list(report.columns) == ["path", "action"]
        actions = {Path(row["path"]).name: row["action"] for row in report.to_dict("records")}
        assert actions["r.pdbqt"] == "kept"
        assert actions["r.dpf"] == "kept"
        assert actions["12.pdbqt"] == "kept"
        assert actions["r.gpf"] == "deleted"
        assert actions["notes.txt"] == "deleted"
        surviving = {p.name for p in (tmp_path / "ligand_01").iterdir()}
        assert surviving == {"r.pdbqt", "r.dpf", "12.pdbqt"}

    def test_dry_run_deletes_nothing(self, tmp_path) -> None:
        self._make_tree(tmp_path)
        report = cleanup_tree(tmp_path, keep={"r.pdbqt"}, dry_run=True)
        assert (report["action"] == "deleted").sum() == 4
        still_there = {p.name for p in (tmp_path / "ligand_01").iterdir()}
        assert still_there == {"r.pdbqt", "r.dpf", "r.gpf", "12.pdbqt", "notes.txt"}

    def test_remove_empty_directories(self, tmp_path) -> None:
        emptied = tmp_path / "emptied"
        emptied.mkdir()
        (emptied / "junk.log").write_text("x", encoding="utf-8")
        kept = tmp_path / "kept"
        kept.mkdir()
        (kept / "r.dlg").write_text("x", encoding="utf-8")
        report = cleanup_tree(
            tmp_path, keep={"r.dlg"}, remove_empty=True,
        )
        actions = {row["path"]: row["action"] for row in report.to_dict("records")}
        assert actions[str(emptied / "junk.log")] == "deleted"
        assert actions[str(emptied)] == "removed_dir"
        assert not emptied.exists()
        assert kept.is_dir() and (kept / "r.dlg").is_file()
        # the root itself is never removed
        assert tmp_path.is_dir()

    def test_dry_run_previews_removed_dirs_without_touching_disk(self, tmp_path) -> None:
        """dry_run + remove_empty must *preview* dir removals, not skip them."""
        emptied = tmp_path / "emptied"          # becomes empty after junk.log dies
        emptied.mkdir()
        (emptied / "junk.log").write_text("x", encoding="utf-8")
        already = tmp_path / "already_empty"   # empty from the start
        already.mkdir()
        kept = tmp_path / "kept"               # keeps r.dlg -> survives
        kept.mkdir()
        (kept / "r.dlg").write_text("x", encoding="utf-8")
        nested = tmp_path / "nested"           # whole subtree dies -> dir goes too
        (nested / "inner").mkdir(parents=True)
        (nested / "inner" / "junk.log").write_text("x", encoding="utf-8")

        report = cleanup_tree(
            tmp_path, keep={"r.dlg"}, remove_empty=True, dry_run=True,
        )
        actions = {row["path"]: row["action"] for row in report.to_dict("records")}
        assert actions[str(emptied)] == "removed_dir"
        assert actions[str(already)] == "removed_dir"
        assert actions[str(nested)] == "removed_dir"
        assert actions[str(nested / "inner")] == "removed_dir"
        assert str(kept) not in actions or actions[str(kept)] != "removed_dir"
        assert actions[str(kept / "r.dlg")] == "kept"
        # nothing was actually deleted
        assert (emptied / "junk.log").is_file()
        assert (nested / "inner" / "junk.log").is_file()
        assert kept.is_dir() and (kept / "r.dlg").is_file()
        assert tmp_path.is_dir()

    def test_dry_run_preview_matches_real_run_outcome(self, tmp_path) -> None:
        """The previewed file/dir actions must equal what a real run would do."""
        for name in ("ligand_01", "ligand_02", "scratch"):
            (tmp_path / name).mkdir()
        (tmp_path / "ligand_01" / "r.dlg").write_text("x", encoding="utf-8")
        (tmp_path / "ligand_01" / "junk.txt").write_text("x", encoding="utf-8")
        (tmp_path / "ligand_02" / "junk.txt").write_text("x", encoding="utf-8")
        (tmp_path / "scratch" / "junk.txt").write_text("x", encoding="utf-8")

        preview = cleanup_tree(
            tmp_path, keep={"r.dlg"}, remove_empty=True, dry_run=True
        )
        preview_actions = {row["path"]: row["action"] for row in preview.to_dict("records")}
        real = cleanup_tree(tmp_path, keep={"r.dlg"}, remove_empty=True)
        real_actions = {row["path"]: row["action"] for row in real.to_dict("records")}
        assert preview_actions == real_actions

    def test_missing_directory_raises(self, tmp_path) -> None:
        with pytest.raises(DockingError, match="not found"):
            cleanup_tree(tmp_path / "nowhere")


class TestDeleteNamedFiles:
    def test_counts_and_deletes_across_tree(self, tmp_path) -> None:
        for folder in ("a", "b", "c"):
            (tmp_path / folder).mkdir()
            (tmp_path / folder / "r.gpf").write_text("gpf", encoding="utf-8")
        (tmp_path / "b" / "r.dlg").write_text("keep me", encoding="utf-8")
        deleted = delete_named_files(tmp_path, "r.gpf")
        assert deleted == 3
        assert not any(tmp_path.rglob("r.gpf"))
        assert (tmp_path / "b" / "r.dlg").is_file()

    def test_dry_run_deletes_nothing(self, tmp_path) -> None:
        (tmp_path / "r.gpf").write_text("gpf", encoding="utf-8")
        assert delete_named_files(tmp_path, "r.gpf", dry_run=True) == 1
        assert (tmp_path / "r.gpf").is_file()


class TestReindexByNumber:
    def test_reindexes_onto_full_range(self) -> None:
        frame = pd.DataFrame(
            {"DLG File": ["1_result.dlg", "5_result.dlg", "9_result.dlg"], "v": [10, 50, 90]}
        )
        result = reindex_by_number(frame, "DLG File", start=0, stop=10)
        assert len(result) == 11
        assert list(result.index) == list(range(11))
        assert result.loc[1, "DLG File"] == "1_result.dlg"
        assert result.loc[9, "v"] == 90
        assert pd.isna(result.loc[0, "v"])
        assert pd.isna(result.loc[2, "DLG File"])
        assert pd.isna(result.loc[10, "v"])
        # the original column stays intact for the matched rows
        assert result["DLG File"].notna().sum() == 3

    def test_rows_without_number_are_dropped(self) -> None:
        frame = pd.DataFrame({"f": ["2_a", "no number", "7_b"]})
        result = reindex_by_number(frame, "f", start=0, stop=10)
        assert result["f"].notna().sum() == 2
        assert result.loc[2, "f"] == "2_a"

    def test_dropped_rows_warn_pattern_miss_and_out_of_range(self, caplog) -> None:
        """No silent loss (27-A): rows the reindex drops must warn.

        Pre-fix, both drop paths were silent: a row whose ``column`` value
        does not match ``pattern`` at all vanished at the notna() filter,
        and a row whose extracted number lies outside ``[start, stop]``
        vanished in the ``reindex`` join.
        """
        frame = pd.DataFrame(
            {
                "f": ["5_result.dlg", "no_number_here", "999_result.dlg"],
                "v": [1.0, 2.0, 3.0],
            }
        )
        with _pharmaml_logs_visible(), caplog.at_level(
            logging.WARNING, logger="pharmaml.docking.campaign"
        ):
            result = reindex_by_number(frame, "f", start=0, stop=10)
        # the rows are still dropped -- but loudly, with actionable counts
        assert result["v"].notna().sum() == 1
        assert result.loc[5, "v"] == 1.0
        assert "1 row(s) in 'f' do not match pattern" in caplog.text
        assert "1 row(s) with ligand number(s) outside [0, 10]" in caplog.text

    def test_in_range_rows_produce_no_drop_warnings(self, caplog) -> None:
        frame = pd.DataFrame({"f": ["2_a", "7_b"]})
        with _pharmaml_logs_visible(), caplog.at_level(
            logging.WARNING, logger="pharmaml.docking.campaign"
        ):
            reindex_by_number(frame, "f", start=0, stop=10)
        assert "do not match pattern" not in caplog.text
        assert "outside" not in caplog.text

    def test_missing_column_raises(self) -> None:
        with pytest.raises(DataValidationError, match="not found"):
            reindex_by_number(pd.DataFrame({"x": [1]}), "missing")


class TestScanGPFLigandTypes:
    def test_types_and_not_found_sentinel(self, tmp_path, sample_ligand_pdbqt) -> None:
        ligand = tmp_path / "ligand_01" / "l.pdbqt"
        ligand.parent.mkdir(parents=True)
        ligand.write_text(sample_ligand_pdbqt, encoding="utf-8")
        write_gpf(ligand, receptor_file="r.pdbqt")
        (tmp_path / "ligand_02").mkdir()
        report = scan_gpf_ligand_types(tmp_path)
        assert list(report.columns) == ["folder", "ligand_types"]
        values = dict(zip(report["folder"], report["ligand_types"]))
        assert values["ligand_01"] == "C HD N OA"
        assert values["ligand_02"] == "not found"

    def test_missing_directory_raises(self, tmp_path) -> None:
        with pytest.raises(DockingError, match="not found"):
            scan_gpf_ligand_types(tmp_path / "nowhere")

    def test_foreign_equals_separator_is_stripped(self, tmp_path) -> None:
        # Foreign GPF writers emit "ligand_types = ..." (pharmaml's own
        # write_gpf never does); the '=' must not leak into the value.
        for folder, line in (
            ("ligand_01", "ligand_types = C HD OA   # comment\n"),
            ("ligand_02", "ligand_types=C HD\n"),
        ):
            sub = tmp_path / folder
            sub.mkdir()
            (sub / "r.gpf").write_text(line, encoding="utf-8")
        report = scan_gpf_ligand_types(tmp_path)
        assert report["ligand_types"].tolist() == ["C HD OA", "C HD"]

    def test_pharmaml_equals_free_line_parses_unchanged(self, tmp_path) -> None:
        # No-regression pin: write_gpf's own '='-free format is byte-identical.
        sub = tmp_path / "ligand_01"
        sub.mkdir()
        (sub / "r.gpf").write_text("ligand_types C HD OA   # comment\n", encoding="utf-8")
        report = scan_gpf_ligand_types(tmp_path)
        assert report["ligand_types"].tolist() == ["C HD OA"]


class TestTorsionConsistency:
    def _pdbqt(self, torsdof: int, remarks: int, branches: int) -> str:
        lines = ["REMARK  Name = synthetic"]
        lines += [f"REMARK Active torsions: A {i + 1}" for i in range(remarks)]
        lines += [f"BRANCH {i + 1} {i + 2}" for i in range(branches)]
        lines.append(f"TORSDOF {torsdof}")
        lines += [
            f"ATOM      {i + 1}  C   LIG A   1       1.000   2.000   3.000  1.00  0.00    0.000 C"
            for i in range(branches)
        ]
        return "\n".join(lines) + "\n"

    def test_consistent_ligand_is_ok(self, tmp_path) -> None:
        path = tmp_path / "l.pdbqt"
        path.write_text(self._pdbqt(torsdof=2, remarks=2, branches=2), encoding="utf-8")
        check = check_torsion_consistency(path)
        assert isinstance(check, TorsionCheck)
        assert (check.torsdof, check.active_remarks, check.branches) == (2, 2, 2)
        assert check.ok
        assert check.status() == "OK"

    def test_mismatch_detected(self, tmp_path) -> None:
        path = tmp_path / "l.pdbqt"
        path.write_text(self._pdbqt(torsdof=3, remarks=2, branches=2), encoding="utf-8")
        check = check_torsion_consistency(path)
        assert not check.ok
        assert check.status() == "MISMATCH"

    def test_missing_ligand_raises(self, tmp_path) -> None:
        with pytest.raises(DockingError, match="not found"):
            check_torsion_consistency(tmp_path / "ghost.pdbqt")

    def test_scan_over_tree(self, tmp_path) -> None:
        good = tmp_path / "ligand_01"
        good.mkdir()
        (good / "l.pdbqt").write_text(
            self._pdbqt(torsdof=2, remarks=2, branches=2), encoding="utf-8"
        )
        bad = tmp_path / "ligand_02"
        bad.mkdir()
        (bad / "l.pdbqt").write_text(
            self._pdbqt(torsdof=4, remarks=1, branches=1), encoding="utf-8"
        )
        report = scan_torsion_consistency(tmp_path)
        assert list(report.columns) == [
            "ligand", "torsdof", "active_remarks", "branches", "status",
        ]
        statuses = dict(zip(report["ligand"], report["status"]))
        assert statuses["ligand_01/l.pdbqt"] == "OK"
        assert statuses["ligand_02/l.pdbqt"] == "MISMATCH"

    def test_scan_missing_directory_raises(self, tmp_path) -> None:
        with pytest.raises(DockingError, match="not found"):
            scan_torsion_consistency(tmp_path / "nowhere")


class TestWriteConfigBatch:
    """The gpfer/dpfer/gpferreceptorni batch loop, upgraded (100x-111).

    One config pair per campaign sub-folder, honest statuses for the
    folders that cannot be configured, and the custom receptor-types
    override the Ni-bearing script carried.
    """

    def _campaign(self, root: Path, ligand_text: str) -> None:
        for name in ("01_aspirin", "02_ibuprofen"):
            folder = root / name
            folder.mkdir(parents=True)
            (folder / "l.pdbqt").write_text(ligand_text, encoding="utf-8")
        (root / "03_empty").mkdir()

    def test_writes_pair_per_folder_with_statuses(
        self, tmp_path, sample_ligand_pdbqt
    ) -> None:
        root = tmp_path / "campaign"
        self._campaign(root, sample_ligand_pdbqt)
        report = write_config_batch(root)
        assert list(report["folder"]) == [
            "01_aspirin", "02_ibuprofen", "03_empty",
        ]
        assert list(report.loc[:1, "status"]) == ["ok", "ok"]
        assert report.loc[2, "status"] == "no ligand PDBQT"
        for name in ("01_aspirin", "02_ibuprofen"):
            assert (root / name / "r.gpf").is_file()
            assert (root / name / "r.dpf").is_file()
            parsed = read_gpf(root / name / "r.gpf")
            # default receptor types, ligand-derived box
            assert parsed["receptor_types"] == [
                "A", "C", "HD", "N", "NA", "OA", "SA",
            ]
            assert parsed["receptor"] == "r.pdbqt"
        assert report.loc[0, "centre"] != ""

    def test_custom_receptor_types_land_in_every_gpf(
        self, tmp_path, sample_ligand_pdbqt
    ) -> None:
        root = tmp_path / "campaign"
        self._campaign(root, sample_ligand_pdbqt)
        report = write_config_batch(
            root, receptor_types="A C HD N Ni OA SA"
        )
        assert (report["status"] == "ok").sum() == 2
        for name in ("01_aspirin", "02_ibuprofen"):
            parsed = read_gpf(root / name / "r.gpf")
            assert parsed["receptor_types"] == [
                "A", "C", "HD", "N", "Ni", "OA", "SA",
            ]  # the Ni type gpferreceptorni.py hard-coded

    def test_skip_existing_leaves_pairs_untouched(
        self, tmp_path, sample_ligand_pdbqt
    ) -> None:
        root = tmp_path / "campaign"
        self._campaign(root, sample_ligand_pdbqt)
        first = write_config_batch(root)
        assert first.loc[0, "status"] == "ok"
        marker = root / "01_aspirin" / "r.gpf"
        original = marker.read_text(encoding="utf-8")
        second = write_config_batch(root, skip_existing=True)
        assert second.loc[0, "status"] == "skipped (r.gpf exists)"
        assert marker.read_text(encoding="utf-8") == original

    def test_receptor_file_excluded_from_ligand_search(
        self, tmp_path, sample_ligand_pdbqt
    ) -> None:
        root = tmp_path / "campaign"
        folder = root / "01_ligand"
        folder.mkdir(parents=True)
        (folder / "r.pdbqt").write_text(sample_ligand_pdbqt,
                                        encoding="utf-8")
        (folder / "l.pdbqt").write_text(sample_ligand_pdbqt,
                                        encoding="utf-8")
        report = write_config_batch(root)
        assert report.loc[0, "ligand"] == "l.pdbqt"  # not the receptor

    def test_negative_padding_raises(self, tmp_path) -> None:
        with pytest.raises(DataValidationError):
            write_config_batch(tmp_path, padding=-1.0)

    def test_missing_directory_raises(self, tmp_path) -> None:
        with pytest.raises(DockingError):
            write_config_batch(tmp_path / "nope")

    def test_failed_folder_is_reported_not_fatal(
        self, tmp_path, sample_ligand_pdbqt
    ) -> None:
        root = tmp_path / "campaign"
        good = root / "01_good"
        good.mkdir(parents=True)
        (good / "l.pdbqt").write_text(sample_ligand_pdbqt,
                                      encoding="utf-8")
        bad = root / "02_bad"
        bad.mkdir()
        # no ATOM records -> no coordinates -> the honest failed: row
        (bad / "l.pdbqt").write_text(
            "REMARK  Name = empty\nTORSDOF 0\n", encoding="utf-8"
        )
        report = write_config_batch(root)
        assert report.loc[0, "status"] == "ok"
        assert report.loc[1, "status"].startswith("failed:")
        assert report.loc[1, "ligand"] == "l.pdbqt"


class TestVinaConfig:
    def test_parse_config(self, tmp_path) -> None:
        conf = tmp_path / "conf.txt"
        conf.write_text(
            "# AutoDock Vina config\nreceptor = r.pdbqt\n\ncenter_x = 11.1\ncenter_y = 22.2   # box centre\ncenter_z = 33.3\nsize_x = 20\nsize_y = 20\nsize_z = 20\nexhaustiveness = 8\nnum_modes = 9",
            encoding="utf-8",
        )
        config = parse_vina_config(conf)
        assert config == {
            "receptor": "r.pdbqt",
            "center_x": "11.1",
            "center_y": "22.2",
            "center_z": "33.3",
            "size_x": "20",
            "size_y": "20",
            "size_z": "20",
            "exhaustiveness": "8",
            "num_modes": "9",
        }

    def test_missing_config_raises(self, tmp_path) -> None:
        with pytest.raises(DockingError, match="not found"):
            parse_vina_config(tmp_path / "missing.txt")


class TestRunVinaBatch:
    def _make_batch(self, tmp_path: Path) -> Path:
        conf = tmp_path / "conf.txt"
        conf.write_text(
            "receptor = r.pdbqt\ncenter_x = 1.0\ncenter_y = 2.0\ncenter_z = 3.0\n"
            "size_x = 20\nsize_y = 20\nsize_z = 20\n",
            encoding="utf-8",
        )
        (tmp_path / "r.pdbqt").write_text("receptor", encoding="utf-8")
        ligands = tmp_path / "ligands"
        (ligands / "sub").mkdir(parents=True)
        (ligands / "1.pdbqt").write_text("ligand 1", encoding="utf-8")
        (ligands / "sub" / "2.pdbqt").write_text("ligand 2", encoding="utf-8")
        return conf

    def _fake_vina(self, monkeypatch, commands: list[list[str]]) -> None:
        monkeypatch.setattr(
            workflow_mod, "_require_executable", lambda name: f"/fake/bin/{name}"
        )

        def fake_run(cmd, **_kwargs):
            commands.append(list(cmd))
            Path(cmd[cmd.index("--out") + 1]).write_text("poses", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, b"log", b"")

        monkeypatch.setattr(workflow_mod.subprocess, "run", fake_run)

    def test_batch_builds_config_driven_commands(self, tmp_path, monkeypatch) -> None:
        conf = self._make_batch(tmp_path)
        commands: list[list[str]] = []
        self._fake_vina(monkeypatch, commands)
        out_dir = tmp_path / "vina_results"
        results = run_vina_batch(conf, tmp_path / "ligands", out_dir=out_dir)
        assert len(results) == 2
        assert all(result.ok for result in results)
        assert out_dir.is_dir()
        assert len(commands) == 2
        for cmd in commands:
            assert "--config" in cmd and str(conf) in cmd
            assert "--ligand" in cmd
            assert "--out" in cmd
            assert "--log" not in cmd
        ligands_used = [cmd[cmd.index("--ligand") + 1] for cmd in commands]
        assert sorted(Path(p).name for p in ligands_used) == ["1.pdbqt", "2.pdbqt"]
        out_names = {Path(cmd[cmd.index("--out") + 1]).name for cmd in commands}
        assert out_names == {"1_out.pdbqt", "2_out.pdbqt"}
        assert {result.log_file.name for result in results} == {"1.log", "2.log"}
        assert all(result.log_file.read_bytes() == b"log" for result in results)

    def test_parallel_batch_produces_same_results(self, tmp_path, monkeypatch) -> None:
        conf = self._make_batch(tmp_path)
        commands: list[list[str]] = []
        self._fake_vina(monkeypatch, commands)
        results = run_vina_batch(
            conf, tmp_path / "ligands", out_dir=tmp_path / "par", max_parallel=2
        )
        assert len(results) == 2
        assert all(result.ok for result in results)
        assert len(commands) == 2

    def test_failing_subprocess_reports_error(self, tmp_path, monkeypatch) -> None:
        conf = self._make_batch(tmp_path)
        monkeypatch.setattr(
            workflow_mod, "_require_executable", lambda name: f"/fake/bin/{name}"
        )

        def failing_run(cmd, **_kwargs):
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(workflow_mod.subprocess, "run", failing_run)
        results = run_vina_batch(conf, tmp_path / "ligands", out_dir=tmp_path / "out")
        assert len(results) == 2
        assert not any(result.ok for result in results)
        assert all(result.error for result in results)

    def test_missing_receptor_key_raises(self, tmp_path, monkeypatch) -> None:
        conf = tmp_path / "conf.txt"
        conf.write_text("center_x = 1.0\n", encoding="utf-8")
        ligands = tmp_path / "ligands"
        ligands.mkdir()
        with pytest.raises(DockingError, match="receptor"):
            run_vina_batch(conf, ligands)

    def test_receptor_file_must_exist_next_to_config(self, tmp_path) -> None:
        conf = tmp_path / "conf.txt"
        conf.write_text("receptor = ghost.pdbqt\n", encoding="utf-8")
        ligands = tmp_path / "ligands"
        ligands.mkdir()
        with pytest.raises(DockingError, match="ghost.pdbqt"):
            run_vina_batch(conf, ligands)


class TestVinaBatchResume:
    """run_vina_batch(skip_existing=True): batch.py / batch_vina.py resume.

    The 2026-10 "new folder" upload's Vina_contol controllers: a ligand
    whose ``_out.pdbqt`` AND non-empty log already exist is not docked
    again; a zero-byte log marks a crashed job and is redone.
    """

    def _make_batch(self, tmp_path: Path) -> Path:
        conf = tmp_path / "conf.txt"
        conf.write_text(
            "receptor = r.pdbqt\ncenter_x = 1.0\ncenter_y = 2.0\ncenter_z = 3.0\n"
            "size_x = 20\nsize_y = 20\nsize_z = 20\n",
            encoding="utf-8",
        )
        (tmp_path / "r.pdbqt").write_text("receptor", encoding="utf-8")
        ligands = tmp_path / "ligands"
        ligands.mkdir()
        (ligands / "1.pdbqt").write_text("ligand 1", encoding="utf-8")
        (ligands / "2.pdbqt").write_text("ligand 2", encoding="utf-8")
        return conf

    def _fake_vina(self, monkeypatch, commands: list[list[str]]) -> None:
        monkeypatch.setattr(
            workflow_mod, "_require_executable", lambda name: f"/fake/bin/{name}"
        )

        def fake_run(cmd, **_kwargs):
            commands.append(list(cmd))
            out = Path(cmd[cmd.index("--out") + 1])
            out.write_text("poses", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, b"log", b"")

        monkeypatch.setattr(workflow_mod.subprocess, "run", fake_run)

    def test_already_docked_ligands_are_skipped(self, tmp_path, monkeypatch) -> None:
        conf = self._make_batch(tmp_path)
        commands: list[list[str]] = []
        self._fake_vina(monkeypatch, commands)
        out_dir = tmp_path / "results"
        out_dir.mkdir()
        # ligand 1 was already docked: pose + non-empty log
        (out_dir / "1_out.pdbqt").write_text("poses", encoding="utf-8")
        (out_dir / "1.log").write_text("old log", encoding="utf-8")
        results = run_vina_batch(
            conf, tmp_path / "ligands", out_dir=out_dir, skip_existing=True
        )
        assert len(results) == 2
        assert results[0].skipped and results[0].ok and results[0].seconds == 0.0
        assert not results[1].skipped and results[1].ok
        # only ligand 2 was actually started
        assert len(commands) == 1
        assert commands[0][commands[0].index("--ligand") + 1].endswith("2.pdbqt")

    def test_zero_byte_log_means_crashed_and_is_redone(self, tmp_path, monkeypatch) -> None:
        # batch_vina.py's empty-log cleanup rule: an existing but EMPTY
        # log marks a crashed job -- the ligand must be docked again.
        conf = self._make_batch(tmp_path)
        commands: list[list[str]] = []
        self._fake_vina(monkeypatch, commands)
        out_dir = tmp_path / "results"
        out_dir.mkdir()
        (out_dir / "1_out.pdbqt").write_text("poses", encoding="utf-8")
        (out_dir / "1.log").write_text("", encoding="utf-8")  # crashed marker
        results = run_vina_batch(
            conf, tmp_path / "ligands", out_dir=out_dir, skip_existing=True
        )
        assert not any(result.skipped for result in results)
        assert len(commands) == 2

    def test_missing_output_means_not_docked_yet(self, tmp_path, monkeypatch) -> None:
        conf = self._make_batch(tmp_path)
        commands: list[list[str]] = []
        self._fake_vina(monkeypatch, commands)
        out_dir = tmp_path / "results"
        out_dir.mkdir()
        (out_dir / "1.log").write_text("orphan log without a pose", encoding="utf-8")
        results = run_vina_batch(
            conf, tmp_path / "ligands", out_dir=out_dir, skip_existing=True
        )
        assert not any(result.skipped for result in results)
        assert len(commands) == 2

    def test_resume_off_docks_everything_again(self, tmp_path, monkeypatch) -> None:
        # the default stays batch.py-less: every ligand re-docked
        conf = self._make_batch(tmp_path)
        commands: list[list[str]] = []
        self._fake_vina(monkeypatch, commands)
        out_dir = tmp_path / "results"
        out_dir.mkdir()
        (out_dir / "1_out.pdbqt").write_text("poses", encoding="utf-8")
        (out_dir / "1.log").write_text("old log", encoding="utf-8")
        results = run_vina_batch(
            conf, tmp_path / "ligands", out_dir=out_dir, skip_existing=False
        )
        assert not any(result.skipped for result in results)
        assert len(commands) == 2


class TestScanVinaDirectory:
    def test_ranking_by_best_affinity(self, tmp_path, sample_vina_output) -> None:
        weaker = tmp_path / "ligand_a.log"
        weaker.write_text(sample_vina_output, encoding="utf-8")  # best -10.3, 4 poses
        stronger = tmp_path / "ligand_b.log"
        stronger.write_text(
            "mode | affinity | dist from best mode\n   1      -11.5          0          0\n   2      -10.9      1.000      2.000",
            encoding="utf-8",
        )
        report = scan_vina_directory(tmp_path)
        assert list(report.columns) == ["ligand", "log", "best_affinity", "n_poses"]
        assert list(report["ligand"]) == ["ligand_b", "ligand_a"]  # ascending
        assert report.iloc[0]["best_affinity"] == pytest.approx(-11.5)
        assert report.iloc[0]["n_poses"] == 2
        assert report.iloc[1]["best_affinity"] == pytest.approx(-10.3)
        assert report.iloc[1]["n_poses"] == 4
        assert report.iloc[1]["log"] == "ligand_a.log"

    def test_unparseable_log_sorts_last_with_none(self, tmp_path) -> None:
        (tmp_path / "good.log").write_text(
            "   1       -9.0          0          0\n", encoding="utf-8"
        )
        (tmp_path / "broken.log").write_text("Vina crashed mid-run\n", encoding="utf-8")
        report = scan_vina_directory(tmp_path)
        assert len(report) == 2
        assert report.iloc[0]["ligand"] == "good"
        assert report.iloc[0]["best_affinity"] == pytest.approx(-9.0)
        assert pd.isna(report.iloc[-1]["best_affinity"])
        assert report.iloc[-1]["n_poses"] == 0

    def test_missing_directory_raises(self, tmp_path) -> None:
        with pytest.raises(DockingParseError, match="not found"):
            scan_vina_directory(tmp_path / "nowhere")


class TestRound2Fixes:
    """Regression guards for the review-round engineering fixes."""

    def test_vina_batch_output_name_collision_gets_prefixed(
        self, tmp_path, monkeypatch
    ) -> None:
        """Two sub-folders each holding ``l.pdbqt`` must not overwrite
        each other's output -- the second gets a ``<parent>_l`` prefix."""
        conf = tmp_path / "conf.txt"
        conf.write_text("receptor = r.pdbqt\n", encoding="utf-8")
        (tmp_path / "r.pdbqt").write_text("receptor", encoding="utf-8")
        ligands = tmp_path / "ligands"
        (ligands / "alpha").mkdir(parents=True)
        (ligands / "beta").mkdir(parents=True)
        (ligands / "alpha" / "l.pdbqt").write_text("a", encoding="utf-8")
        (ligands / "beta" / "l.pdbqt").write_text("b", encoding="utf-8")

        commands: list[list[str]] = []
        monkeypatch.setattr(
            workflow_mod, "_require_executable", lambda name: f"/fake/bin/{name}"
        )

        def fake_run(cmd, **_kwargs):
            commands.append(list(cmd))
            Path(cmd[cmd.index("--out") + 1]).write_text("poses", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, b"log", b"")

        monkeypatch.setattr(workflow_mod.subprocess, "run", fake_run)

        results = run_vina_batch(conf, ligands, out_dir=tmp_path / "out")
        assert len(results) == 2 and all(r.ok for r in results)
        out_names = {Path(cmd[cmd.index("--out") + 1]).name for cmd in commands}
        assert out_names == {"l_out.pdbqt", "beta_l_out.pdbqt"}
        assert {result.log_file.name for result in results} == {"l.log", "beta_l.log"}

    def test_collect_files_invalid_pattern_raises_docking_error(
        self, tmp_path
    ) -> None:
        with pytest.raises(DockingError, match="invalid regex for pattern"):
            collect_files(tmp_path, pattern="([unclosed")

    def test_cleanup_tree_invalid_pattern_raises_docking_error(
        self, tmp_path
    ) -> None:
        with pytest.raises(DockingError, match="invalid regex for keep_pattern"):
            cleanup_tree(tmp_path, keep_pattern="*bad[")

    def test_reindex_invalid_pattern_raises_docking_error(self) -> None:
        frame = pd.DataFrame({"f": ["x1"]})
        with pytest.raises(DataValidationError, match="invalid regex"):
            reindex_by_number(frame, "f", pattern="([oops")

    def test_reindex_pattern_without_capture_group_raises(self) -> None:
        frame = pd.DataFrame({"f": ["x1", "x2"]})
        with pytest.raises(DataValidationError, match="no capture group"):
            reindex_by_number(frame, "f", pattern=r"\d+")

    def test_reindex_multi_group_pattern_uses_first_group(self) -> None:
        frame = pd.DataFrame({"f": ["run_07_a", "run_09_b"]})
        out = reindex_by_number(frame, "f", start=0, stop=10, pattern=r"run_(\d+)_(\w+)")
        assert len(out) == 11
        assert out.loc[7, "f"] == "run_07_a"
        assert out.loc[9, "f"] == "run_09_b"


class TestTorsionConventions:
    """check_torsion_consistency auto-detects the PDBQT writer lineage."""

    MGL_CONTENT = (
        "REMARK  1  A    between atoms: T1_2-CA   T3_4-CB\n"
        "REMARK  2  A    between atoms: T5_6-CG   T7_8-CD\n"
        "ROOT\n"
        "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C\n"
        "ENDROOT\n"
        "BRANCH    1    2\nBRANCH    2    3\n"
        "TORSDOF {torsdof}\n"
    )

    MEEKO_CONTENT = (
        "REMARK SMILES CC(=O)Oc1ccccc1C(=O)O\n"
        "ROOT\n"
        "ATOM      1  C   LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C\n"
        "ENDROOT\n"
        "BRANCH    1    2\nBRANCH    7    8\nBRANCH    6   11\nBRANCH   11   13\n"
        "TORSDOF {torsdof}\n"
    )

    def test_meeko_file_passes_with_auto_rule(self, tmp_path) -> None:
        ligand = tmp_path / "meeko.pdbqt"
        ligand.write_text(self.MEEKO_CONTENT.format(torsdof=4), encoding="utf-8")
        check = check_torsion_consistency(ligand)
        assert check.active_remarks == 0 and check.branches == 4
        assert check.ok and check.status() == "OK"

    def test_meeko_file_fails_when_torsdof_disagrees(self, tmp_path) -> None:
        ligand = tmp_path / "meeko_bad.pdbqt"
        ligand.write_text(self.MEEKO_CONTENT.format(torsdof=3), encoding="utf-8")
        assert check_torsion_consistency(ligand).status() == "MISMATCH"

    def test_mgltools_file_passes_three_way_rule(self, tmp_path) -> None:
        ligand = tmp_path / "mgl.pdbqt"
        ligand.write_text(self.MGL_CONTENT.format(torsdof=2), encoding="utf-8")
        check = check_torsion_consistency(ligand)
        assert check.ok and check.active_remarks == 2 and check.branches == 2

    def test_mgltools_mismatch_detected(self, tmp_path) -> None:
        ligand = tmp_path / "mgl_bad.pdbqt"
        ligand.write_text(self.MGL_CONTENT.format(torsdof=3), encoding="utf-8")
        assert check_torsion_consistency(ligand).status() == "MISMATCH"

    def test_explicit_convention_overrides_auto(self, tmp_path) -> None:
        ligand = tmp_path / "mgl.pdbqt"
        ligand.write_text(self.MGL_CONTENT.format(torsdof=2), encoding="utf-8")
        # meeko rule (2 == 2 branches) also passes here:
        assert check_torsion_consistency(ligand, convention="meeko").ok
        # mgltools rule on a meeko-style file (0 remarks != 2) fails:
        meeko = tmp_path / "meeko.pdbqt"
        meeko.write_text(self.MEEKO_CONTENT.format(torsdof=4), encoding="utf-8")
        assert not check_torsion_consistency(meeko, convention="mgltools").ok

    def test_unknown_convention_raises(self, tmp_path) -> None:
        ligand = tmp_path / "any.pdbqt"
        ligand.write_text("TORSDOF 0\n", encoding="utf-8")
        with pytest.raises(DockingError, match="convention must be"):
            check_torsion_consistency(ligand, convention="ad4")


class TestReadGPFMalformedNumericFields:
    """Malformed numeric GPF fields raise the typed DockingParseError.

    Regression: ``spacing``/``smooth``/``dielectric`` escaped as raw
    ``ValueError`` from ``float()``, while malformed ``npts``/
    ``gridcenter`` lines silently kept their defaults -- the docstring
    promises :class:`DockingParseError` for unparseable files.
    """

    @pytest.mark.parametrize(
        ("field_line", "field"),
        [
            ("npts sixty 60 60", "npts"),
            ("npts 60 60", "npts"),  # too few values
            ("npts 40.5 40 40", "npts"),  # non-integer point count
            ("spacing zero", "spacing"),
            ("gridcenter 1.0 2.0", "gridcenter"),
            ("gridcenter 1.0 two 3.0", "gridcenter"),
            ("smooth abc", "smooth"),
            ("dielectric xyz", "dielectric"),
        ],
    )
    def test_malformed_field_raises_typed_error(
        self, tmp_path: Path, field_line: str, field: str
    ) -> None:
        gpf = tmp_path / "r.gpf"
        gpf.write_text(f"ligand_types C\n{field_line}\n", encoding="utf-8")
        with pytest.raises(DockingParseError, match=field):
            read_gpf(gpf)

    def test_well_formed_file_still_parses(self, tmp_path: Path) -> None:
        gpf = tmp_path / "r.gpf"
        gpf.write_text(
            "ligand_types C\nnpts 40 40 40\nspacing 0.375\n"
            "gridcenter 1.0 2.0 3.0\nsmooth 0.5\ndielectric -0.1465\n",
            encoding="utf-8",
        )
        parsed = read_gpf(gpf)
        assert parsed["npts"] == (40, 40, 40)
        assert parsed["gridcenter"] == (1.0, 2.0, 3.0)
        assert parsed["spacing"] == pytest.approx(0.375)
        assert parsed["smooth"] == pytest.approx(0.5)
        assert parsed["dielectric"] == pytest.approx(-0.1465)


class TestRunAutodock4Completion:
    """``skip_existing`` trusts the completion sentinel, not file presence.

    An interrupted AutoDock4 run leaves a partial ``r.dlg`` behind;
    treating its mere presence as "done" reported the run as ok without
    ever producing poses.  The sentinel check mirrors
    :func:`~pharmaml.docking.parsing.check_docking_completion`.
    """

    def _run_dir(self, tmp_path: Path, dlg_text: str) -> Path:
        work = tmp_path / "ligand_01"
        work.mkdir()
        (work / "r.gpf").write_text("ligand_types C\n", encoding="utf-8")
        (work / "r.dpf").write_text("ligand_types C\n", encoding="utf-8")
        (work / "r.glg").write_text("autogrid4 log\n", encoding="utf-8")
        (work / "r.dlg").write_text(dlg_text, encoding="utf-8")
        return work

    def test_interrupted_dlg_is_rerun_not_silently_ok(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # Interrupted run left a partial r.dlg; the re-run finds no
        # autodock4 engine -> a clean, actionable not-ok result.
        work = self._run_dir(tmp_path, "DPF> interrupted mid-run\n")

        def no_engine(name: str) -> str:
            raise DockingError(f"executable {name!r} not found on PATH")

        monkeypatch.setattr(workflow_mod, "_require_executable", no_engine)
        monkeypatch.setattr(workflow_mod, "_bundled_engine_dirs", lambda: ())
        monkeypatch.setattr(workflow_mod.shutil, "which", lambda _name: None)
        result = run_autodock4(work, skip_existing=True)
        assert result.ok is False
        assert result.log_file is None
        assert "not found" in (result.error or "")

    def test_completed_run_is_skipped_without_touching_engines(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        work = self._run_dir(tmp_path, "AutoDock4 log\n\nSuccessful Completion\n")

        def must_not_run(*args, **kwargs):
            raise AssertionError("completed run must be skipped, not re-run")

        monkeypatch.setattr(workflow_mod, "_require_executable", must_not_run)
        monkeypatch.setattr(workflow_mod.subprocess, "run", must_not_run)
        result = run_autodock4(work, skip_existing=True)
        assert result.ok is True
        assert result.log_file == work / "r.dlg"


class TestVinaBatchRunsInConfigDirectory:
    """run_vina_batch launches vina with ``cwd=`` the config's directory."""

    def test_subprocess_cwd_is_config_parent(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # Regression: the receptor is validated relative to the config's
        # own directory, but the engine was launched without cwd= -- so
        # Vina resolved config-relative paths against the process CWD.
        conf_dir = tmp_path / "engines"
        conf_dir.mkdir()
        conf = conf_dir / "conf.txt"
        conf.write_text("receptor = r.pdbqt\n", encoding="utf-8")
        (conf_dir / "r.pdbqt").write_text("receptor", encoding="utf-8")
        ligands = tmp_path / "ligands"
        ligands.mkdir()
        (ligands / "l.pdbqt").write_text("ligand", encoding="utf-8")
        monkeypatch.setattr(
            workflow_mod, "_require_executable", lambda name: f"/fake/bin/{name}"
        )
        seen: list[object] = []

        def fake_run(cmd, **kwargs):
            seen.append(kwargs.get("cwd"))
            Path(cmd[cmd.index("--out") + 1]).write_text("poses", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, b"log", b"")

        monkeypatch.setattr(workflow_mod.subprocess, "run", fake_run)
        results = run_vina_batch(conf, ligands, out_dir=tmp_path / "out")
        assert all(result.ok for result in results)
        assert seen == [conf.parent]


class TestVinaBatchProgress:
    """vina_batch_progress (100x-142): the pre-flight resume census.

    "N ligands already docked, M to go" BEFORE the Dock button -- with
    the SAME skip rule and the SAME collision-disambiguated output
    names ``run_vina_batch(skip_existing=True)`` uses, so the census and
    the run can never disagree.
    """

    def _tree(self, tmp_path: Path) -> tuple[Path, Path]:
        ligands = tmp_path / "ligands"
        ligands.mkdir()
        (ligands / "a.pdbqt").write_text("ligand a", encoding="utf-8")
        (ligands / "b.pdbqt").write_text("ligand b", encoding="utf-8")
        out = tmp_path / "out"
        out.mkdir()
        return ligands, out

    def test_census_counts_the_skip_rule_rows(self, tmp_path) -> None:
        from pharmaml.docking.workflow import vina_batch_progress

        ligands, out = self._tree(tmp_path)
        # nothing docked yet
        progress = vina_batch_progress(ligands, out)
        assert (progress.total, progress.docked, progress.remaining) == (2, 0, 2)
        # 'a' completes: pose + non-empty log
        (out / "a_out.pdbqt").write_text("poses", encoding="utf-8")
        (out / "a.log").write_text("log", encoding="utf-8")
        progress = vina_batch_progress(ligands, out)
        assert (progress.total, progress.docked, progress.remaining) == (2, 1, 1)

    def test_zero_byte_log_is_not_docked(self, tmp_path) -> None:
        # batch_vina.py's crashed-job marker: the census must agree with
        # the run -- a zero-byte log means REDO, not skip.
        from pharmaml.docking.workflow import vina_batch_progress

        ligands, out = self._tree(tmp_path)
        (out / "a_out.pdbqt").write_text("poses", encoding="utf-8")
        (out / "a.log").write_text("", encoding="utf-8")
        progress = vina_batch_progress(ligands, out)
        assert progress.docked == 0
        assert progress.remaining == 2

    def test_census_agrees_with_the_run_itself(self, tmp_path, monkeypatch) -> None:
        # the promise: docked == the skipped rows a resumed run reports
        from pharmaml.docking.workflow import run_vina_batch, vina_batch_progress

        ligands, out = self._tree(tmp_path)
        # half the campaign already landed
        (out / "a_out.pdbqt").write_text("poses", encoding="utf-8")
        (out / "a.log").write_text("log", encoding="utf-8")
        conf = tmp_path / "conf.txt"
        conf.write_text("receptor = r.pdbqt\n", encoding="utf-8")
        (tmp_path / "r.pdbqt").write_text("receptor", encoding="utf-8")
        monkeypatch.setattr(workflow_mod, "_require_executable", lambda name: f"/fake/{name}")

        def fake_run(cmd, **_kwargs) -> None:
            out_pdbqt = Path(cmd[cmd.index("--out") + 1])
            out_pdbqt.write_text("poses", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, b"log", b"")

        monkeypatch.setattr(workflow_mod.subprocess, "run", fake_run)
        before = vina_batch_progress(ligands, out)
        results = run_vina_batch(conf, ligands, out_dir=out, skip_existing=True)
        after = vina_batch_progress(ligands, out)
        # the run skipped exactly what the census promised
        assert sum(1 for result in results if result.skipped) == before.docked
        # and the campaign is now complete by the census's own rule
        assert after.docked == after.total == 2

    def test_same_stem_ligands_count_as_collisions(self, tmp_path) -> None:
        # two 'a.pdbqt' in different sub-folders: the flat output names
        # collide, the disambiguation renames the second, and the census
        # confesses the count (the census names files EXACTLY like the
        # run -- the renamed 'a' is not confused with the plain one).
        from pharmaml.docking.workflow import vina_batch_progress

        ligands, out = self._tree(tmp_path)
        sub = ligands / "sub"
        sub.mkdir()
        (sub / "a.pdbqt").write_text("twin", encoding="utf-8")
        # the FIRST 'a' is docked; the sub-folder twin is not
        (out / "a_out.pdbqt").write_text("poses", encoding="utf-8")
        (out / "a.log").write_text("log", encoding="utf-8")
        progress = vina_batch_progress(ligands, out)
        assert progress.total == 3
        assert progress.docked == 1
        assert progress.remaining == 2
        assert progress.collisions == 1

    def test_missing_ligands_dir_raises_the_runs_own_error(self, tmp_path) -> None:
        from pharmaml.docking.workflow import vina_batch_progress

        with pytest.raises(DockingError, match="ligands directory not found"):
            vina_batch_progress(tmp_path / "absent", tmp_path / "out")

    def test_progress_dataclass_is_frozen_with_named_fields(self) -> None:
        import dataclasses

        from pharmaml.docking.workflow import VinaBatchProgress

        fields = {field.name for field in dataclasses.fields(VinaBatchProgress)}
        assert fields == {"total", "docked", "remaining", "collisions"}
        progress = VinaBatchProgress(total=3, docked=1, remaining=2, collisions=0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            progress.docked = 5  # type: ignore[misc]

    def test_census_is_exported_from_the_docking_package(self) -> None:
        from pharmaml.docking import VinaBatchProgress, vina_batch_progress

        assert callable(vina_batch_progress)
        assert VinaBatchProgress.__name__ == "VinaBatchProgress"


# --------------------------------------------------------------------- #
# (100x-171) the .exe engine folder: vina.exe / autogrid4.exe / autodock4.exe
# --------------------------------------------------------------------- #
class TestEngineFolderResolution:
    """locate_engine_binary / engine_argv: ONE folder of Windows builds.

    The three uploaded executables -- ``vina.exe`` (PE32+), ``autogrid4.exe``
    and ``autodock4.exe`` (PE32 i386) -- are the engines the campaign
    runner must drive on machines where nothing is on PATH.  The resolver
    checks the engine folder first, then PATH, then verified bundled engines;
    both bare and ``.exe`` spellings are searched. A Windows PE binary on a
    non-Windows host is routed through ``wine`` by its MZ magic, not by
    its file name (renamed copies still route correctly).
    """

    def test_locate_finds_the_exe_in_the_engine_folder(self, tmp_path) -> None:
        engines = tmp_path / "engines"
        engines.mkdir()
        (engines / "vina.exe").write_bytes(b"")
        assert workflow_mod.locate_engine_binary("vina", engines) == engines / "vina.exe"

    def test_locate_bare_name_wins_over_the_exe_spelling(self, tmp_path) -> None:
        engines = tmp_path / "engines"
        engines.mkdir()
        (engines / "vina").write_bytes(b"")
        (engines / "vina.exe").write_bytes(b"")
        assert workflow_mod.locate_engine_binary("vina", engines) == engines / "vina"

    def test_locate_falls_back_to_path_and_the_exe_spelling(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            workflow_mod.shutil,
            "which",
            lambda name: f"/usr/local/bin/{name}" if name == "vina.exe" else None,
        )
        assert workflow_mod.locate_engine_binary("vina", tmp_path) == Path(
            "/usr/local/bin/vina.exe"
        )

    def test_locate_missing_everywhere_returns_none(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(workflow_mod.shutil, "which", lambda _name: None)
        monkeypatch.setattr(workflow_mod, "_bundled_engine_dirs", lambda: ())
        assert workflow_mod.locate_engine_binary("vina", tmp_path) is None

    def test_engine_argv_routes_a_pe_binary_through_wine(
        self, tmp_path, monkeypatch
    ) -> None:
        engines = tmp_path / "engines"
        engines.mkdir()
        exe = engines / "vina.exe"
        exe.write_bytes(b"MZ" + b"\x00" * 58 + struct.pack("<I", 64) + b"PE\x00\x00" + struct.pack("<H", 0x8664))
        monkeypatch.setattr(workflow_mod.os, "name", "posix")
        monkeypatch.setattr(
            workflow_mod.shutil,
            "which",
            lambda name: "/usr/bin/wine" if name == "wine" else None,
        )
        assert workflow_mod.engine_argv("vina", engines) == ["/usr/bin/wine", str(exe)]

    def test_engine_argv_pe_without_wine_is_an_actionable_error(
        self, tmp_path, monkeypatch
    ) -> None:
        engines = tmp_path / "engines"
        engines.mkdir()
        (engines / "vina.exe").write_bytes(b"MZ" + b"\x00" * 58 + struct.pack("<I", 64) + b"PE\x00\x00" + struct.pack("<H", 0x8664))
        monkeypatch.setattr(workflow_mod.os, "name", "posix")
        monkeypatch.setattr(workflow_mod.shutil, "which", lambda _name: None)
        with pytest.raises(DockingError, match="wine"):
            workflow_mod.engine_argv("vina", engines)

    def test_engine_argv_missing_in_folder_names_both_spellings(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(workflow_mod.shutil, "which", lambda _name: None)
        monkeypatch.setattr(workflow_mod, "_bundled_engine_dirs", lambda: ())
        with pytest.raises(DockingError, match="engine folder"):
            workflow_mod.engine_argv("vina", tmp_path / "engines")

    def test_engine_argv_without_folder_keeps_the_path_only_gate(
        self, monkeypatch
    ) -> None:
        # Preserve the historical PATH resolver hook for existing callers.
        monkeypatch.setattr(
            workflow_mod, "_require_executable", lambda name: f"/fake/bin/{name}"
        )
        assert workflow_mod.engine_argv("autodock4") == ["/fake/bin/autodock4"]


class TestEngineFolderRuns:
    """The runners launch the folder's .exe binaries -- argv[0] and all."""

    def _record(self, monkeypatch) -> list[list[str]]:
        commands: list[list[str]] = []
        monkeypatch.setattr(
            workflow_mod.subprocess,
            "run",
            lambda cmd, **_kwargs: commands.append(list(cmd)),
        )
        return commands

    def test_run_vina_launches_the_folder_exe(self, tmp_path, monkeypatch) -> None:
        engines = tmp_path / "engines"
        engines.mkdir()
        (engines / "vina.exe").write_bytes(b"")  # not a PE -> direct argv
        receptor = tmp_path / "r.pdbqt"
        receptor.write_text("ROOT\n", encoding="utf-8")
        ligand = tmp_path / "l.pdbqt"
        ligand.write_text("ATOM\n", encoding="utf-8")
        commands = []
        def fake_success(cmd, **_kwargs):
            commands.append(list(cmd))
            Path(cmd[cmd.index("--out") + 1]).write_text("fake pose", encoding="utf-8")
        monkeypatch.setattr(workflow_mod.subprocess, "run", fake_success)
        result = workflow_mod.run_vina(
            receptor,
            ligand,
            tmp_path / "out.pdbqt",
            center=(0.0, 0.0, 0.0),
            size=(20.0, 20.0, 20.0),
            engine_dir=engines,
        )
        assert result.ok
        assert commands[0][0] == str(engines / "vina.exe")
        assert commands[0][commands[0].index("--receptor") + 1] == str(receptor)

    def test_run_vina_missing_folder_binary_raises_typed_error(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(workflow_mod.shutil, "which", lambda _name: None)
        monkeypatch.setattr(workflow_mod, "_bundled_engine_dirs", lambda: ())
        with pytest.raises(DockingError, match="engine folder"):
            workflow_mod.run_vina(
                tmp_path / "r.pdbqt",
                tmp_path / "l.pdbqt",
                tmp_path / "out.pdbqt",
                center=(0.0, 0.0, 0.0),
                size=(20.0, 20.0, 20.0),
                engine_dir=tmp_path / "engines",
            )

    def test_run_autodock4_launches_both_folder_exes(
        self, tmp_path, monkeypatch
    ) -> None:
        engines = tmp_path / "engines"
        engines.mkdir()
        (engines / "autogrid4.exe").write_bytes(b"")
        (engines / "autodock4.exe").write_bytes(b"")
        work = tmp_path / "01_lig"
        work.mkdir()
        (work / "r.gpf").write_text("ligand_types C\n", encoding="utf-8")
        (work / "r.dpf").write_text("ligand_types C\n", encoding="utf-8")
        commands: list[list[str]] = []

        def fake_run(cmd, **_kwargs):
            commands.append(list(cmd))
            if cmd[0] == str(engines / "autogrid4.exe"):
                (work / "r.glg").write_text("autogrid4 log\n", encoding="utf-8")
            else:
                (work / "r.dlg").write_text(
                    "AutoDock4 log\n\nSuccessful Completion\n", encoding="utf-8"
                )

        monkeypatch.setattr(workflow_mod.subprocess, "run", fake_run)
        result = run_autodock4(work, engine_dir=engines)
        assert result.ok, result.error
        assert [cmd[0] for cmd in commands] == [
            str(engines / "autogrid4.exe"),
            str(engines / "autodock4.exe"),
        ]
        assert commands[0][1:] == ["-p", "r.gpf", "-l", "r.glg"]
        assert commands[1][1:] == ["-p", "r.dpf", "-l", "r.dlg"]

    def test_run_vina_batch_launches_the_folder_exe(
        self, tmp_path, monkeypatch
    ) -> None:
        engines = tmp_path / "engines"
        engines.mkdir()
        (engines / "vina.exe").write_bytes(b"")
        conf = tmp_path / "vina.conf"
        conf.write_text("receptor = r.pdbqt\n", encoding="utf-8")
        (tmp_path / "r.pdbqt").write_text("ROOT\n", encoding="utf-8")
        ligands = tmp_path / "ligands"
        ligands.mkdir()
        (ligands / "l.pdbqt").write_text("ATOM\n", encoding="utf-8")
        commands = []
        def fake_success(cmd, **_kwargs):
            commands.append(list(cmd))
            Path(cmd[cmd.index("--out") + 1]).write_text("pose", encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, b"log", b"")
        monkeypatch.setattr(workflow_mod.subprocess, "run", fake_success)
        results = run_vina_batch(
            conf, ligands, out_dir=tmp_path / "out", engine_dir=engines
        )
        assert all(result.ok for result in results)
        assert commands[0][0] == str(engines / "vina.exe")

    def test_run_one_threads_the_engine_folder_through(self, monkeypatch) -> None:
        seen: dict[str, object] = {}

        def fake_run_autodock4(directory, engine_dir=None):
            seen["directory"] = directory
            seen["engine_dir"] = engine_dir
            return workflow_mod.DockingRunResult(Path(directory), True, None, 0.0)

        monkeypatch.setattr(workflow_mod, "run_autodock4", fake_run_autodock4)
        workflow_mod._run_one("/tmp/ligand_01", "/opt/engines")
        assert seen == {"directory": "/tmp/ligand_01", "engine_dir": "/opt/engines"}


class TestSmilesArrayEngineFolder:
    """dock_smiles_array(engine_dir=...): the folder pre-flight + pass-through."""

    def _receptor(self, tmp_path) -> Path:
        receptor = tmp_path / "r.pdbqt"
        receptor.write_text("ROOT\n", encoding="utf-8")
        return receptor

    def test_missing_everywhere_names_the_folder_option(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setattr(workflow_mod.shutil, "which", lambda _name: None)
        monkeypatch.setattr(workflow_mod, "_bundled_engine_dirs", lambda: ())
        with pytest.raises(DockingError, match="engine folder"):
            dock_smiles_array(
                ["CCO"],
                self._receptor(tmp_path),
                tmp_path / "out",
                engine="vina",
                engine_dir=tmp_path / "engines",
            )

    def test_folder_exes_pass_the_gate_and_reach_vina(
        self, tmp_path, monkeypatch
    ) -> None:
        engines = tmp_path / "engines"
        engines.mkdir()
        (engines / "vina.exe").write_bytes(b"")
        seen: dict[str, object] = {}

        def fake_smiles_to_pdbqt(smiles, folder, name="l", seed=0):
            target = Path(folder)
            target.mkdir(parents=True, exist_ok=True)
            ligand = target / f"{name}.pdbqt"
            ligand.write_text("ATOM  stub\n", encoding="utf-8")
            return ligand

        monkeypatch.setattr(smiles_mod, "smiles_to_pdbqt", fake_smiles_to_pdbqt)

        def fake_run_vina(*args, **kwargs):
            seen["engine_dir"] = kwargs.get("engine_dir")
            out = Path(kwargs["out"])
            out.write_text(
                "REMARK VINA RESULT:    -5.2      0.000      0.000\nEND\n",
                encoding="utf-8",
            )
            return workflow_mod.DockingRunResult(out.parent, True, out, 0.0)

        monkeypatch.setattr(smiles_mod, "run_vina", fake_run_vina)

        frame = dock_smiles_array(
            ["CCO"],
            self._receptor(tmp_path),
            tmp_path / "out",
            engine="vina",
            engine_dir=engines,
        )
        assert seen["engine_dir"] == engines
        assert len(frame) == 1
        row = frame.iloc[0]
        assert bool(row["ok"]) and float(row["best_score"]) == -5.2
