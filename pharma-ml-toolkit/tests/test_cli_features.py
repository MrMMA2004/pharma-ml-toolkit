"""CLI-level tests for the new commands and flags.

Exercises ``pharmaml.data split``, ``ml train --plots/--keep-categorical``,
``ml train --pdf-report`` / ``ml compare --pdf-report`` (the 100x-75 CLI
surface of the PDF document), ``ml compare --out/--plot``, model aliases
and the docking campaign subcommands (``dpf``/``collect``/``torsion``/
``gpf-scan``/``vina-scan``/``rescore``) through :func:`pharmaml.cli.main`
exactly the way a shell would invoke them.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pharmaml.cli import main

logging.disable(logging.CRITICAL)


@pytest.fixture()
def clf_csv(tmp_path) -> str:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.normal(size=(90, 4)), columns=[f"d{i}" for i in range(4)])
    frame["Target"] = (frame["d0"] + frame["d1"] > 0).astype(int)
    frame["assay"] = rng.choice(["A", "B"], size=90)
    path = tmp_path / "clf.csv"
    frame.to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def reg_csv(tmp_path) -> str:
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(rng.normal(size=(90, 3)), columns=[f"d{i}" for i in range(3)])
    frame["pIC50"] = 5 + frame["d0"] * 2 - frame["d1"] + rng.normal(scale=0.2, size=90)
    path = tmp_path / "reg.csv"
    frame.to_csv(path, index=False)
    return str(path)


class TestDataSplitCommand:
    def test_split_writes_stratified_files(self, clf_csv, tmp_path, capsys):
        out_dir = tmp_path / "splits"
        code = main([
            "data", "split", "--data", clf_csv,
            "--test-size", "0.25", "--val-size", "0.15",
            "--out-dir", str(out_dir), "--prefix", "qsar",
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "stratified" in out
        for side in ("train", "val", "test"):
            assert (out_dir / f"qsar_{side}.csv").is_file()
        assert "qsar_train.csv" in out

    def test_split_default_writes_alongside_source(self, clf_csv, capsys):
        code = main(["data", "split", "--data", clf_csv, "--seed", "7"])
        assert code == 0
        source = Path(clf_csv)
        assert (source.parent / f"{source.stem}_train.csv").is_file()
        assert (source.parent / f"{source.stem}_test.csv").is_file()

    def test_split_no_stratify_without_target(self, tmp_path, capsys):
        path = tmp_path / "no_target.csv"
        pd.DataFrame({"a": range(40), "b": range(40, 80)}).to_csv(path, index=False)
        code = main([
            "data", "split", "--data", str(path), "--no-stratify", "--test-size", "0.25",
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "unstratified" in out

    def test_split_errors_reported_cleanly(self, tmp_path, capsys):
        path = tmp_path / "tiny.csv"
        pd.DataFrame({"a": [1, 2]}).to_csv(path, index=False)
        code = main(["data", "split", "--data", str(path)])
        assert code == 1
        assert "error:" in capsys.readouterr().err

    def test_split_refuses_to_overwrite_its_own_input(self, tmp_path, capsys):
        # 26-R5 defect 1: `data split --data qsar_train.csv --prefix qsar`
        # used to write the TRAIN side straight into the input file
        # (20 rows -> 16, exit 0) -- the realistic re-split-your-train-set
        # workflow destroyed the source data.
        path = tmp_path / "qsar_train.csv"
        pd.DataFrame(
            {"d": range(20), "Target": ["x"] * 10 + ["y"] * 10}
        ).to_csv(path, index=False)
        before = path.read_bytes()
        code = main(["data", "split", "--data", str(path), "--prefix", "qsar"])
        assert code == 1
        err = capsys.readouterr().err
        assert "error:" in err
        assert "refusing to overwrite the input file" in err
        # the input is untouched and no partial split files were written
        assert path.read_bytes() == before
        assert not (tmp_path / "qsar_test.csv").exists()

    def test_split_non_colliding_prefix_writes_normally(self, tmp_path):
        # the guard is about *collision*, not about splitting per se:
        # a distinct prefix next to a <stem>_train.csv input still writes.
        path = tmp_path / "qsar_train.csv"
        pd.DataFrame(
            {"d": range(20), "Target": ["x"] * 10 + ["y"] * 10}
        ).to_csv(path, index=False)
        code = main(["data", "split", "--data", str(path), "--prefix", "final"])
        assert code == 0
        assert (tmp_path / "final_train.csv").is_file()
        assert (tmp_path / "final_test.csv").is_file()
        assert len(pd.read_csv(path)) == 20  # input still intact


class TestMLTrainCommand:
    def test_alias_resolved_in_output(self, clf_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        code = main([
            "ml", "train", "--data", clf_csv, "--model", "rf",
            "--no-tune", "--cv", "3", "--out", str(tmp_path / "art"),
        ])
        assert code == 0
        assert "model:    random_forest" in capsys.readouterr().out

    def test_plots_bundle_for_classification(self, clf_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out_dir = tmp_path / "art"
        code = main([
            "ml", "train", "--data", clf_csv, "--model", "logistic",
            "--no-tune", "--cv", "3", "--plots", "--out", str(out_dir),
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "plots:" in out and "report.html" in out
        bundle = out_dir / "report"
        assert (bundle / "report.html").is_file()
        for figure in ("confusion_matrix.png", "roc_curves.png",
                       "class_distribution.png", "feature_importance.png"):
            assert (bundle / figure).is_file(), figure
        assert (bundle / "held_out_metrics.csv").is_file()
        assert (bundle / "class_counts.csv").is_file()

    def test_plots_bundle_for_regression(self, reg_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        out_dir = tmp_path / "art"
        code = main([
            "ml", "train", "--data", reg_csv, "--model", "ridge",
            "--no-tune", "--cv", "3", "--plots", "--out", str(out_dir),
        ])
        assert code == 0
        bundle = out_dir / "report"
        assert (bundle / "report.html").is_file()
        assert (bundle / "predicted_vs_actual.png").is_file()
        assert (bundle / "residuals.png").is_file()
        # regression: no confusion matrix / class distribution
        assert not (bundle / "confusion_matrix.png").exists()
        # --plots alone never writes the PDF document (100x-75 default pin)
        assert not (bundle / "report.pdf").exists()

    def test_pdf_report_implies_plots_and_writes_pdf(
        self, reg_csv, tmp_path, capsys, monkeypatch
    ):
        # --pdf-report without --plots: the bundle still ships (the flag
        # implies it) and report.pdf lands next to report.html.
        pypdf = pytest.importorskip("pypdf")
        monkeypatch.chdir(tmp_path)
        out_dir = tmp_path / "art"
        code = main([
            "ml", "train", "--data", reg_csv,
            "--model", "ridge", "--no-tune", "--cv", "3",
            "--pdf-report", "--out", str(out_dir),
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "plots:" in out and "report.html" in out  # implied bundle
        assert "pdf:" in out and "report.pdf" in out
        bundle = out_dir / "report"
        assert (bundle / "report.html").is_file()
        assert (bundle / "report.pdf").is_file()
        # the document: title page (title + subtitle + metric lines) then
        # one page per table/figure — at least metrics + 2 figures here.
        reader = pypdf.PdfReader(str(bundle / "report.pdf"))
        assert len(reader.pages) >= 4
        first = reader.pages[0].extract_text()
        assert "pharmaml ml train" in first
        assert "ridge" in first
        assert "rmse" in first.lower()

    def test_keep_categorical_flow(self, clf_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        code = main([
            "ml", "train", "--data", clf_csv, "--model", "rf",
            "--no-tune", "--cv", "3", "--keep-categorical",
            "--out", str(tmp_path / "art"),
        ])
        assert code == 0
        # 4 numeric + 1 categorical kept (5 features, not 4)
        assert "5 features" in capsys.readouterr().out


class TestMLCompareCommand:
    def test_out_and_plot(self, clf_csv, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        code = main([
            "ml", "compare", "--data", clf_csv,
            "--models", "rf", "logistic", "--cv", "3",
            "--out", str(tmp_path / "board.csv"),
            "--plot", "--out-dir", str(tmp_path / "board_report"),
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "leaderboard ->" in out and "plot:" in out
        assert (tmp_path / "board.csv").is_file()
        report_dir = tmp_path / "board_report"
        assert (report_dir / "report.html").is_file()
        assert (report_dir / "model_comparison.png").is_file()
        assert (report_dir / "leaderboard.csv").is_file()
        # --plot alone never writes the PDF document (100x-75 default pin)
        assert not (report_dir / "report.pdf").exists()

    def test_pdf_report_implies_plot(self, clf_csv, tmp_path, capsys, monkeypatch):
        # --pdf-report without --plot: leaderboard bundle + report.pdf.
        pypdf = pytest.importorskip("pypdf")
        monkeypatch.chdir(tmp_path)
        out_dir = tmp_path / "board_report"
        code = main([
            "ml", "compare", "--data", clf_csv,
            "--models", "rf", "logistic", "--cv", "3",
            "--pdf-report", "--out-dir", str(out_dir),
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "plot:" in out and "report.html" in out  # implied bundle
        assert "pdf:" in out and "report.pdf" in out
        assert (out_dir / "report.html").is_file()
        assert (out_dir / "model_comparison.png").is_file()
        assert (out_dir / "leaderboard.csv").is_file()
        reader = pypdf.PdfReader(str(out_dir / "report.pdf"))
        # exactly: title page + leaderboard table page + figure page
        assert len(reader.pages) == 3
        assert "pharmaml ml compare" in reader.pages[0].extract_text()
        assert "random_forest" in reader.pages[1].extract_text()

    def test_keep_categorical_compare(self, clf_csv, capsys, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        code = main([
            "ml", "compare", "--data", clf_csv, "--models", "rf",
            "--cv", "3", "--keep-categorical",
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "random_forest" in out

    def test_bad_model_alias_error(self, clf_csv, capsys):
        code = main(["ml", "compare", "--data", clf_csv, "--models", "not_a_model"])
        assert code == 1
        assert "error:" in capsys.readouterr().err


class TestMLCheckCommand:
    """`pharmaml ml check` — the round-trip audit on the CLI (100x-77)."""

    @pytest.fixture()
    def saved_model(self, clf_csv, tmp_path, monkeypatch):
        """Train + save a bundle the way `ml train` does."""
        monkeypatch.chdir(tmp_path)
        code = main([
            "ml", "train", "--data", clf_csv, "--model", "logistic",
            "--no-tune", "--cv", "3", "--out", str(tmp_path / "art"),
        ])
        assert code == 0
        bundle = tmp_path / "artifacts" / "logistic_regression_classification.joblib"
        assert bundle.is_file()
        return str(bundle)

    def test_check_ok_with_target_dropped(
        self, reg_csv, capsys, monkeypatch, tmp_path
    ):
        # the natural workflow: re-audit the very CSV the model was
        # trained on -- --target drops the label column first.  (reg_csv:
        # all-numeric columns, so the bundle's features are exactly
        # the non-target columns.)
        monkeypatch.chdir(tmp_path)
        code = main([
            "ml", "train", "--data", reg_csv, "--model", "ridge",
            "--no-tune", "--cv", "3", "--out", str(tmp_path / "art"),
        ])
        assert code == 0
        bundle = str(tmp_path / "artifacts" / "ridge_regression.joblib")
        code = main([
            "ml", "check", "--model", bundle, "--data", reg_csv,
            "--target", "pIC50",
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "model:    ridge (regression)" in out
        assert "trained:  " in out and "pharmaml 1." in out
        assert "audit: OK" in out

    def test_check_fails_with_exit_one_and_hint(
        self, clf_csv, saved_model, capsys, monkeypatch, tmp_path
    ):
        # no --target: the label column is an extra the model does not
        # know -> FAIL, exit 1, and the --target hint points the way.
        monkeypatch.chdir(tmp_path)
        code = main([
            "ml", "check", "--model", saved_model, "--data", clf_csv,
        ])
        assert code == 1
        out = capsys.readouterr().out
        assert "audit: FAIL" in out
        assert "extra column" in out and "Target" in out
        assert "hint: if one of the extra columns is the training target" in out

    def test_check_unknown_target_errors(
        self, clf_csv, saved_model, capsys, monkeypatch, tmp_path
    ):
        monkeypatch.chdir(tmp_path)
        code = main([
            "ml", "check", "--model", saved_model, "--data", clf_csv,
            "--target", "nope",
        ])
        assert code == 1
        err = capsys.readouterr().err
        assert "error: --target 'nope' is not a column" in err

    def test_check_missing_bundle_errors(self, clf_csv, capsys):
        code = main([
            "ml", "check", "--model", "ghost.joblib", "--data", clf_csv,
        ])
        assert code == 1
        assert "error:" in capsys.readouterr().err


class TestInspectShowsCategorical:
    def test_inspect(self, clf_csv, capsys):
        code = main(["data", "inspect", "--data", clf_csv, "--head", "2"])
        assert code == 0
        out = capsys.readouterr().out
        assert "dropped cols" in out          # default view drops 'assay'
        assert "first 2 row(s)" in out


class TestDockingCampaignCommands:
    """Upgrades of the rar campaign scripts exposed as CLI verbs."""

    @staticmethod
    def _ligand_pdbqt(tmp_path: Path) -> Path:
        carbon = "ATOM      1  C   LIG A   1       1.000   2.000   3.000  1.00  0.00    0.000 C"
        oxygen = "ATOM      2  O   LIG A   1       4.000   2.000   3.000  1.00  0.00    0.000 OA"
        ligand = tmp_path / "l.pdbqt"
        ligand.write_text(
            f"REMARK  Name = cli_ligand\nROOT\n{carbon}\n{oxygen}\nENDROOT\nTORSDOF 1"
            + "\n",
            encoding="utf-8",
        )
        return ligand

    def test_dpf_writes_parameter_file(self, tmp_path, capsys):
        ligand = self._ligand_pdbqt(tmp_path)
        code = main(
            ["docking", "dpf", "--ligand", str(ligand), "--ga-runs", "5"]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "dpf ->" in out
        dpf = tmp_path / "r.dpf"
        assert dpf.is_file()
        text = dpf.read_text(encoding="utf-8")
        assert "ga_run 5" in text
        assert "ligand_types C OA" in text
        assert "move l.pdbqt" in text

    def test_dpf_missing_ligand_reports_error(self, tmp_path, capsys):
        code = main(["docking", "dpf", "--ligand", str(tmp_path / "ghost.pdbqt")])
        assert code == 1
        assert "error:" in capsys.readouterr().err

    def test_collect_prints_report(self, tmp_path, capsys):
        folder = tmp_path / "campaign" / "ligand_01"
        folder.mkdir(parents=True)
        (folder / "r.dlg").write_text("dlg", encoding="utf-8")
        code = main(["docking", "collect", "--dir", str(tmp_path / "campaign")])
        assert code == 0
        out = capsys.readouterr().out
        assert "ligand_01.dlg" in out
        assert "1 match(es), 1 copied" in out
        assert (tmp_path / "campaign" / "collected" / "ligand_01.dlg").is_file()

    def test_collect_dry_run_copies_nothing(self, tmp_path, capsys):
        folder = tmp_path / "campaign" / "ligand_01"
        folder.mkdir(parents=True)
        (folder / "r.dlg").write_text("dlg", encoding="utf-8")
        code = main(
            ["docking", "collect", "--dir", str(tmp_path / "campaign"), "--dry-run"]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "dry run" in out
        assert not (tmp_path / "campaign" / "collected").exists()

    def test_collect_pattern_mode(self, tmp_path, capsys):
        folder = tmp_path / "campaign" / "7"
        folder.mkdir(parents=True)
        (folder / "7.pdbqt").write_text("ligand", encoding="utf-8")
        code = main(
            [
                "docking", "collect", "--dir", str(tmp_path / "campaign"),
                "--pattern", r"^\d+\.pdbqt$",
            ]
        )
        assert code == 0
        assert (tmp_path / "campaign" / "collected" / "7.pdbqt").is_file()

    def test_torsion_prints_audit(self, tmp_path, capsys):
        ligand = tmp_path / "l.pdbqt"
        ligand.write_text(
            "TORSDOF 1\nREMARK torsion: A 1\nBRANCH 1 2\n", encoding="utf-8"
        )
        code = main(["docking", "torsion", "--dir", str(tmp_path)])
        assert code == 0
        out = capsys.readouterr().out
        assert "l.pdbqt" in out
        assert "OK" in out

    def test_gpf_scan_prints_ligand_types(self, tmp_path, sample_ligand_pdbqt, capsys):
        folder = tmp_path / "ligand_01"
        folder.mkdir()
        ligand = folder / "l.pdbqt"
        ligand.write_text(sample_ligand_pdbqt, encoding="utf-8")
        from pharmaml.docking import write_gpf

        write_gpf(ligand)
        code = main(["docking", "gpf-scan", "--dir", str(tmp_path)])
        assert code == 0
        out = capsys.readouterr().out
        assert "ligand_01" in out
        assert "C HD N OA" in out

    def test_vina_scan_prints_ranking(self, tmp_path, capsys):
        log = tmp_path / "ligand_x.log"
        log.write_text(
            "   1      -10.5          0          0\n"
            "   2       -9.5      1.000      2.000\n",
            encoding="utf-8",
        )
        code = main(["docking", "vina-scan", "--dir", str(tmp_path)])
        assert code == 0
        out = capsys.readouterr().out
        assert "ligand_x" in out
        assert "-10.5" in out

    def test_campaign_list_prints_stems_and_writes_csv(self, tmp_path, capsys):
        for name in ("lig01.csv", "lig02.csv", "notes.txt"):
            (tmp_path / name).write_text("x", encoding="utf-8")
        out_csv = tmp_path / "names.csv"
        code = main(
            ["docking", "campaign", "list", "--dir", str(tmp_path), "--out", str(out_csv)]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "lig01" in out and "lig02" in out
        assert "2 file stem(s)" in out
        content = out_csv.read_text(encoding="utf-8").splitlines()
        assert content[0] == "name"
        assert "lig01" in content[1:]

    def test_campaign_distribute_copies_into_subdirs(self, tmp_path, capsys):
        for name in ("ligand_01", "ligand_02"):
            (tmp_path / name).mkdir()
        receptor = tmp_path / "receptor.pdbqt"
        receptor.write_text("ATOM", encoding="utf-8")
        code = main(
            ["docking", "campaign", "distribute", "--dir", str(tmp_path),
             "--file", str(receptor)]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "2 cop(y/ies) written" in out
        assert (tmp_path / "ligand_01" / "receptor.pdbqt").is_file()
        assert (tmp_path / "ligand_02" / "receptor.pdbqt").is_file()
        # re-run without --overwrite skips existing copies
        code = main(
            ["docking", "campaign", "distribute", "--dir", str(tmp_path),
             "--file", str(receptor)]
        )
        assert code == 0
        assert "0 cop(y/ies) written" in capsys.readouterr().out

    def test_campaign_cleanup_dry_run_previews_dirs(self, tmp_path, capsys):
        ligand = tmp_path / "ligand_01"
        ligand.mkdir()
        (ligand / "r.dlg").write_text("x", encoding="utf-8")
        (ligand / "junk.txt").write_text("x", encoding="utf-8")
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "junk.txt").write_text("x", encoding="utf-8")
        code = main(
            ["docking", "campaign", "cleanup", "--dir", str(tmp_path),
             "--keep", "r.dlg", "--remove-empty", "--dry-run"]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "removed_dir" in out
        assert "(dry run)" in out
        # nothing was deleted
        assert (scratch / "junk.txt").is_file()
        assert (ligand / "junk.txt").is_file()

    def test_campaign_cleanup_real_run(self, tmp_path, capsys):
        ligand = tmp_path / "ligand_01"
        ligand.mkdir()
        (ligand / "r.dlg").write_text("x", encoding="utf-8")
        (ligand / "junk.txt").write_text("x", encoding="utf-8")
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / "junk.txt").write_text("x", encoding="utf-8")
        code = main(
            ["docking", "campaign", "cleanup", "--dir", str(tmp_path),
             "--keep", "r.dlg", "--remove-empty"]
        )
        assert code == 0
        assert (ligand / "r.dlg").is_file()
        assert not (ligand / "junk.txt").exists()
        assert not scratch.exists()

    def test_campaign_delete_named_counts_and_deletes(self, tmp_path, capsys):
        for name in ("ligand_01", "ligand_02"):
            folder = tmp_path / name
            folder.mkdir()
            (folder / "r.gpf").write_text("x", encoding="utf-8")
        code = main(
            ["docking", "campaign", "delete-named", "--dir", str(tmp_path),
             "--name", "r.gpf", "--dry-run"]
        )
        assert code == 0
        assert "2 file(s) named 'r.gpf' would be deleted (dry run)" in capsys.readouterr().out
        assert (tmp_path / "ligand_01" / "r.gpf").is_file()
        code = main(
            ["docking", "campaign", "delete-named", "--dir", str(tmp_path), "--name", "r.gpf"]
        )
        assert code == 0
        assert "2 file(s) named 'r.gpf' deleted" in capsys.readouterr().out
        assert not (tmp_path / "ligand_01" / "r.gpf").exists()

    def test_campaign_reindex_writes_table(self, tmp_path, capsys):
        import pandas as pd

        src = tmp_path / "results.csv"
        pd.DataFrame(
            {
                "name": ["1_result.dlg", "5_result.dlg", "7_result.dlg"],
                "best": [-9.1, -10.2, -8.7],
            }
        ).to_csv(src, index=False)
        out_csv = tmp_path / "reindexed.csv"
        code = main(
            [
                "docking", "campaign", "reindex",
                "--data", str(src), "--column", "name",
                "--start", "1", "--stop", "10", "--out", str(out_csv),
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "3/10 ligand number(s) present in range 1..10" in out
        assert "missing numbers: 2, 3, 4, 6, 8, 9, 10" in out
        assert f"reindexed table -> {out_csv}" in out
        content = out_csv.read_text(encoding="utf-8").splitlines()
        assert content[0] == "number,name,best"
        assert content[1] == "1,1_result.dlg,-9.1"
        assert content[2] == "2,,"
        assert content[4] == "4,,"
        assert content[5] == "5,5_result.dlg,-10.2"
        assert len(content) == 11  # header + full 1..10 range

    def test_campaign_reindex_stdout_preview_without_out(self, tmp_path, capsys):
        import pandas as pd

        src = tmp_path / "results.csv"
        pd.DataFrame({"name": ["2_run.dlg"], "best": [-7.5]}).to_csv(src, index=False)
        code = main(
            [
                "docking", "campaign", "reindex",
                "--data", str(src), "--column", "name",
                "--start", "1", "--stop", "3",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "1/3 ligand number(s) present in range 1..3" in out
        assert "missing numbers: 1, 3" in out
        assert "2_run.dlg" in out  # the frame preview itself
        assert "number" in out

    def test_campaign_reindex_bad_column_reports_error(self, tmp_path, capsys):
        import pandas as pd

        src = tmp_path / "results.csv"
        pd.DataFrame({"name": ["1_result.dlg"]}).to_csv(src, index=False)
        code = main(
            [
                "docking", "campaign", "reindex",
                "--data", str(src), "--column", "bogus",
            ]
        )
        assert code == 1
        err = capsys.readouterr().err
        assert "error:" in err
        assert "bogus" in err
        assert "available columns" in err


class TestDockingRescoreCommand:
    """``pharmaml docking rescore`` -- the consensus_breakdown CLI surface.

    The explainable ranking (100x-82): evidence columns of a summary
    CSV combine into one 0-1 consensus, the leaderboard and the written
    table keep the per-column ``_norm`` units, and the printed recipe
    line names every column's direction and weight.
    """

    @pytest.fixture()
    def summary(self, tmp_path: Path) -> Path:
        import pandas as pd

        src = tmp_path / "summary.csv"
        pd.DataFrame(
            {
                "ligand": ["L1", "L2", "L3", "L4"],
                "best_lowest_energy": [-10.0, -8.0, -6.0, -7.5],
                "cluster_size": [10, 30, 20, 20],
            }
        ).to_csv(src, index=False)
        return src

    def test_rescore_prints_leaderboard_and_writes_csv(
        self, summary: Path, capsys
    ) -> None:
        import pandas as pd

        out = summary.parent / "ranked.csv"
        code = main(
            [
                "docking", "rescore",
                "--data", str(summary),
                "--columns", "best_lowest_energy,cluster_size",
                "--weights", "1,3",
                "--out", str(out),
            ]
        )
        assert code == 0
        text = capsys.readouterr().out
        # the explainable leaderboard: the driving evidence rides along
        assert "best_lowest_energy_norm" in text and "cluster_size_norm" in text
        assert "consensus" in text
        assert "L2" in text.splitlines()[1]  # cluster-weighted winner first
        # the recipe line names every column's direction and weight
        assert "evidence: best_lowest_energy (lower=better, weight 1)" in text
        assert "cluster_size (higher=better, weight 3)" in text
        assert "4 ligand(s) rescored" in text
        # the written table: sorted by consensus desc, units included
        ranked = pd.read_csv(out)
        assert list(ranked["ligand"][:2]) == ["L2", "L4"]
        assert ranked["consensus"].is_monotonic_decreasing
        assert ranked.loc[0, "cluster_size_norm"] == pytest.approx(1.0)

    def test_rescore_direction_override_flips_energy(
        self, summary: Path, capsys
    ) -> None:
        # single evidence column + --higher: the ranking inverts cleanly
        # (L3's -6.0 is the highest value once "higher is better").
        code = main(
            [
                "docking", "rescore",
                "--data", str(summary),
                "--columns", "best_lowest_energy",
                "--higher", "best_lowest_energy",
            ]
        )
        assert code == 0
        text = capsys.readouterr().out
        assert "evidence: best_lowest_energy (higher=better)" in text
        assert "L3" in text.splitlines()[1]  # -6.0 tops the flipped ranking
        # the default output lands next to the input
        assert (summary.parent / "summary_rescored.csv").is_file()

        # --lower on the second column: L1's tiny cluster (worst by
        # default) becomes the best evidence -- the flip is effective.
        code = main(
            [
                "docking", "rescore",
                "--data", str(summary),
                "--columns", "best_lowest_energy,cluster_size",
                "--lower", "cluster_size",
                "--out", str(summary.parent / "flipped.csv"),
            ]
        )
        assert code == 0
        text = capsys.readouterr().out
        assert "cluster_size (lower=better)" in text
        assert "L1" in text.splitlines()[1]  # best energy + flipped cluster

    def test_rescore_top_limits_the_printed_leaderboard(
        self, summary: Path, capsys
    ) -> None:
        code = main(
            [
                "docking", "rescore",
                "--data", str(summary),
                "--columns", "best_lowest_energy,cluster_size",
                "--top", "2",
                "--out", str(summary.parent / "t.csv"),
            ]
        )
        assert code == 0
        text = capsys.readouterr().out
        data_rows = [
            line for line in text.splitlines()
            if line.strip().startswith(("L1", "L2", "L3", "L4"))
        ]
        assert len(data_rows) == 2  # the table itself still has all 4 rows
        assert "4 ligand(s) rescored" in text

    def test_rescore_errors_through_the_standard_wrapper(
        self, summary: Path, capsys
    ) -> None:
        code = main(
            [
                "docking", "rescore",
                "--data", str(summary),
                "--columns", "no_such_column",
            ]
        )
        assert code == 1
        err = capsys.readouterr().err
        assert "error: missing columns ['no_such_column']" in err

        code = main(
            [
                "docking", "rescore",
                "--data", str(summary),
                "--columns", "best_lowest_energy",
                "--weights", "one",
            ]
        )
        assert code == 1
        err = capsys.readouterr().err
        assert "--weights must be comma-separated numbers" in err


class TestChemTableCommands:
    """`pharmaml chem compare` / `chem merge` surface tables.py in the CLI."""

    def test_compare_prints_buckets_and_writes_report(self, tmp_path, capsys):
        import pandas as pd

        table_a = tmp_path / "summary.csv"
        table_b = tmp_path / "library.csv"
        pd.DataFrame({"Ligand": ["lig1.pdbqt", "LIG2.pdbqt", "lig3.pdbqt"]}).to_csv(
            table_a, index=False
        )
        pd.DataFrame({"Ligand": ["lig2", "lig3", "lig9"]}).to_csv(table_b, index=False)
        report = tmp_path / "diff.csv"
        code = main(
            ["chem", "compare", "--data-a", str(table_a), "--data-b", str(table_b),
             "--column", "Ligand", "--out", str(report)]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "common=2" in out and "only_in_a=1" in out and "only_in_b=1" in out
        assert "lig1" in out and "lig9" in out
        assert report.is_file()

    def test_compare_keep_case_flag(self, tmp_path, capsys):
        import pandas as pd

        table_a = tmp_path / "a.csv"
        table_b = tmp_path / "b.csv"
        pd.DataFrame({"Ligand": ["Aspirin"]}).to_csv(table_a, index=False)
        pd.DataFrame({"Ligand": ["aspirin"]}).to_csv(table_b, index=False)
        code = main(
            ["chem", "compare", "--data-a", str(table_a), "--data-b", str(table_b),
             "--column", "Ligand", "--keep-case", "--keep-extension"]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "common=0" in out

    def test_compare_figure_flag_writes_overlap_png(self, tmp_path, capsys):
        import pandas as pd

        table_a = tmp_path / "summary.csv"
        table_b = tmp_path / "library.csv"
        pd.DataFrame({"Ligand": ["lig1.pdbqt", "lig2.pdbqt", "lig3.pdbqt"]}).to_csv(
            table_a, index=False
        )
        pd.DataFrame({"Ligand": ["lig2", "lig3", "lig9"]}).to_csv(table_b, index=False)
        figure = tmp_path / "overlap.png"
        code = main(
            ["chem", "compare", "--data-a", str(table_a), "--data-b", str(table_b),
             "--column", "Ligand", "--figure", str(figure)]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "overlap figure ->" in out and "overlap.png" in out
        assert figure.is_file() and figure.stat().st_size > 1_000
        # matplotlib figures must not leak between CLI invocations
        import matplotlib.pyplot as plt

        assert plt.get_fignums() == []

    def test_merge_writes_merged_output(self, tmp_path, capsys):
        import pandas as pd

        pd.DataFrame({"SMILES": ["CCO"], "zeta": [1]}).to_csv(
            tmp_path / "one.csv", index=False
        )
        pd.DataFrame({"SMILES": ["CCC"], "alpha": [2]}).to_csv(
            tmp_path / "two.csv", index=False
        )
        out = tmp_path / "merged.csv"
        code = main(
            ["chem", "merge", "--dir", str(tmp_path), "--out", str(out),
             "--first-columns", "SMILES,alpha"]
        )
        assert code == 0
        out_text = capsys.readouterr().out
        assert "2 rows" in out_text and str(out) in out_text
        frame = pd.read_csv(out)
        assert list(frame.columns) == ["SMILES", "alpha", "zeta"]


class TestPharmaApprovalsCommand:
    """`pharmaml pharma approvals` surfaces filter_approved_drugs."""

    def test_approvals_prints_sorted_names(self, tmp_path, capsys):
        export = tmp_path / "interactions.tsv"
        export.write_text(
            "drug\tregulatory approval\nWarfarin\tApproved\nCaffeine\tWithdrawn\n"
            "Aspirin\tapproved\n",
            encoding="utf-8",
        )
        code = main(
            ["pharma", "approvals", "--data", str(export)]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "2 unique" in out
        assert out.index("Aspirin") < out.index("Warfarin")
        assert "Caffeine" not in out

    def test_approvals_multiple_files_and_out(self, tmp_path, capsys):
        import pandas as pd

        first = tmp_path / "1.tsv"
        second = tmp_path / "2.tsv"
        first.write_text("drug\tregulatory approval\nIbuprofen\tApproved\n",
                         encoding="utf-8")
        second.write_text("drug\tregulatory approval\nIbuprofen\tApproved\nMetformin\tApproved\n",
                          encoding="utf-8")
        out = tmp_path / "approved.csv"
        code = main(
            ["pharma", "approvals", "--data", str(first), "--data", str(second),
             "--out", str(out)]
        )
        assert code == 0
        out_text = capsys.readouterr().out
        assert "2 unique" in out_text
        frame = pd.read_csv(out)
        assert frame["drug"].tolist() == ["Ibuprofen", "Metformin"]


class TestPharmaPkCommand:
    """`pharmaml pharma pk` -- NCA summary straight from a CSV."""

    def test_pk_summary_json(self, tmp_path, capsys):
        profile = tmp_path / "profile.csv"
        profile.write_text(
            "time,conc\n0,10.0\n1,6.0\n2,3.6\n3,2.2\n", encoding="utf-8"
        )
        code = main(["pharma", "pk", "--data", str(profile)])
        assert code == 0
        payload = json.loads(capsys.readouterr().out)
        # linear trapezoid: 0.5*(10+6) + 0.5*(6+3.6) + 0.5*(3.6+2.2) = 15.7
        assert payload["AUC"] == pytest.approx(15.7)
        assert payload["Cmax"] == pytest.approx(10.0)
        assert payload["Tmax"] == pytest.approx(0.0)

    def test_pk_out_json_file(self, tmp_path, capsys):
        profile = tmp_path / "profile.csv"
        profile.write_text(
            "time,conc\n0,10.0\n1,6.0\n2,3.6\n3,2.2\n", encoding="utf-8"
        )
        out = tmp_path / "summary.json"
        code = main(
            ["pharma", "pk", "--data", str(profile), "--out", str(out)]
        )
        assert code == 0
        assert "saved ->" in capsys.readouterr().out
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["AUC"] == pytest.approx(15.7)

    def test_pk_non_numeric_concentration_clean_error(self, tmp_path, capsys):
        # Regression: a text cell ('blq') in the conc column used to surface
        # as a raw numpy conversion error; it must be a typed message that
        # names the offending values.
        messy = tmp_path / "messy.csv"
        messy.write_text("time,conc\n0,5.0\n1,blq\n2,1.0\n", encoding="utf-8")
        code = main(["pharma", "pk", "--data", str(messy)])
        assert code == 1
        err = capsys.readouterr().err
        assert "error:" in err
        assert "numeric" in err


class TestPharmaDoseCommand:
    """`pharmaml pharma dose` -- regimen design from CL/Vd/Css."""

    def test_dose_regimen_output(self, capsys):
        code = main(
            [
                "pharma", "dose",
                "--cl", "5.0", "--vd", "35.0", "--target-css", "15.0",
                "--tau", "12.0", "--f", "0.9",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        # MD = CL*Css*tau/F = 5*15*12/0.9 = 1000; LD = Vd*Css/F = 35*15/0.9.
        assert "maintenance dose : 1000.00 per 12 h" in out
        assert "loading dose     : 583.33" in out

    def test_dose_rejects_negative_clearance(self, capsys):
        code = main(
            [
                "pharma", "dose",
                "--cl", "-5.0", "--vd", "35.0", "--target-css", "15.0",
            ]
        )
        assert code == 1
        assert "error:" in capsys.readouterr().err


class TestChemDescriptorsCommand:
    """`pharmaml chem descriptors` output alignment (round-19 triage item)."""

    @staticmethod
    def _ligands_csv(tmp_path: Path) -> Path:
        csv = tmp_path / "ligands.csv"
        csv.write_text("smiles\nCCO\nCCN\n", encoding="utf-8")
        return csv

    def test_descriptors_aligned_with_default_index_csv(self, tmp_path):
        # PINNING: the plain CSV path (default RangeIndex) is correct.
        csv = self._ligands_csv(tmp_path)
        out = tmp_path / "out.csv"
        code = main(
            [
                "chem", "descriptors", "--data", str(csv),
                "--smiles-col", "smiles", "--out", str(out),
            ]
        )
        assert code == 0
        frame = pd.read_csv(out)
        weights = frame.set_index("smiles")["MolWt"]
        assert weights["CCO"] == pytest.approx(46.07, abs=0.05)
        assert weights["CCN"] == pytest.approx(45.08, abs=0.05)

    def test_missing_smiles_col_exits_1_like_every_other_cli_failure(self, tmp_path, capsys):
        # 26-R5 defect 2: this was the single CLI failure path that exited 2
        # while the identical failure class exits 1 everywhere else.
        csv = self._ligands_csv(tmp_path)
        code = main(
            ["chem", "descriptors", "--data", str(csv), "--smiles-col", "nope"]
        )
        assert code == 1
        err = capsys.readouterr().err
        assert "error:" in err
        assert "nope" in err

    def test_descriptors_positionally_aligned_for_indexed_frames(
        self, tmp_path, monkeypatch
    ):
        # Regression (round-19 triage): compute_descriptors indexes rows by
        # POSITION in the SMILES column, but the CLI handler joined on the
        # caller frame's INDEX LABELS -- any frame without a default
        # RangeIndex silently attached the wrong row's descriptors.
        csv = self._ligands_csv(tmp_path)
        indexed = pd.read_csv(csv)
        indexed.index = [1, 0]  # non-default label/position permutation

        import pharmaml.utils.io as io_module

        monkeypatch.setattr(io_module, "read_table", lambda path: indexed.copy())
        out = tmp_path / "out.csv"
        code = main(
            [
                "chem", "descriptors", "--data", str(csv),
                "--smiles-col", "smiles", "--out", str(out),
            ]
        )
        assert code == 0
        frame = pd.read_csv(out)
        weights = frame.set_index("smiles")["MolWt"]
        # CCO (46.07 g/mol) must keep its own descriptors, not CCN's.
        assert weights["CCO"] == pytest.approx(46.07, abs=0.05)
        assert weights["CCN"] == pytest.approx(45.08, abs=0.05)


class TestBiologyCommands:
    """`pharmaml biology` verbs: response screens and biomarker discovery."""

    @staticmethod
    def _screen_csv(tmp_path: Path, name: str = "screen.csv") -> Path:
        """6 cell lines x 4 compounds, one duplicate pair, one missing value."""
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(17)
        cells = ["A549", "HCT116", "HL60", "K562", "MCF7", "U87"]
        compounds = ["cisplatin", "doxorubicin", "erlotinib", "imatinib"]
        tissue = {"A549": "lung", "HCT116": "colon", "HL60": "haematopoietic",
                  "K562": "haematopoietic", "MCF7": "breast", "U87": "brain"}
        rows = [
            {
                "cell_line": cell,
                "compound": compound,
                "tissue": tissue[cell],
                "LN_IC50": round(float(rng.uniform(-2.0, 4.0)), 4),
                "AUC": round(float(rng.uniform(0.1, 1.0)), 4),
            }
            for cell in cells
            for compound in compounds
        ]
        frame = pd.DataFrame(rows)
        extra = frame.iloc[2].copy()  # duplicate (A549, erlotinib)
        extra["LN_IC50"] = 1.5
        frame = pd.concat([frame, pd.DataFrame([extra])], ignore_index=True)
        frame.loc[3, "LN_IC50"] = float("nan")  # one missing value
        path = tmp_path / name
        frame.to_csv(path, index=False)
        return path

    def test_response_audit_prints_summary_audit_and_compounds(
        self, tmp_path, capsys
    ):
        screen_csv = self._screen_csv(tmp_path)
        code = main(
            [
                "biology", "response-audit", "--data", str(screen_csv),
                "--tissue-column", "tissue", "--metric", "LN_IC50",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "6 cell line(s) x 4 compound(s)" in out
        assert "audit:" in out
        assert "duplicate pairs" in out and "A549 / erlotinib" in out
        assert "missing metric values" in out
        assert "per-compound summary (metric 'LN_IC50')" in out
        assert "sensitive_spread" in out
        assert "most_sensitive_cell_line" in out
        for compound in ("cisplatin", "doxorubicin", "erlotinib", "imatinib"):
            assert compound in out

    def test_response_audit_bad_column_reports_error(self, tmp_path, capsys):
        screen_csv = self._screen_csv(tmp_path)
        code = main(
            ["biology", "response-audit", "--data", str(screen_csv),
             "--cell-column", "bogus"]
        )
        assert code == 1
        err = capsys.readouterr().err
        assert "error:" in err
        assert "bogus" in err

    def test_response_audit_replicate_cv_flag_prints_variability(
        self, tmp_path, capsys
    ):
        screen_csv = self._screen_csv(tmp_path)
        code = main(
            ["biology", "response-audit", "--data", str(screen_csv),
             "--replicate-cv"]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "replicate variability of duplicated pairs" in out
        assert "cv_pct = std / |mean| * 100" in out
        # the planted duplicate pair gets one row per metric
        assert "erlotinib" in out
        assert out.count("A549 erlotinib") == 2  # LN_IC50 row + AUC row
        assert "n_replicates" in out
        assert "LN_IC50" in out and "AUC" in out
        # the AUC replicates are byte-identical copies -> cv 0.0
        assert "0.000000" in out

    def test_response_audit_without_flag_omits_variability(
        self, tmp_path, capsys
    ):
        screen_csv = self._screen_csv(tmp_path)
        code = main(
            ["biology", "response-audit", "--data", str(screen_csv)]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "replicate variability" not in out
        assert "cv_pct" not in out

    def test_response_audit_replicate_cv_clean_screen_says_nothing_to_collapse(
        self, tmp_path, capsys
    ):
        import pandas as pd

        clean = pd.DataFrame(
            {
                "cell_line": ["A549", "HCT116"],
                "compound": ["cisplatin", "cisplatin"],
                "LN_IC50": [1.0, 2.0],
                "AUC": [0.5, 0.6],
            }
        )
        path = tmp_path / "clean.csv"
        clean.to_csv(path, index=False)
        code = main(
            ["biology", "response-audit", "--data", str(path),
             "--replicate-cv"]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "replicate variability: no duplicated" in out
        assert "nothing to collapse" in out

    def test_top_biomarkers_end_to_end_with_planted_gene(
        self, tmp_path, capsys
    ):
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(21)
        cells = [f"CELL{i}" for i in range(10)]
        genes = ["KRAS", "BRAF", "TP53", "MYC", "ALK", "PTEN", "RB1", "CDKN2A"]
        expression = pd.DataFrame(
            rng.normal(50, 10, size=(len(cells), len(genes))), columns=genes
        )
        expression.insert(0, "cell_line", cells)

        # planted biomarker: high EGFR <-> high pIC50 (sensitive, pIC50 style)
        pic50 = pd.Series(rng.uniform(4.0, 8.0, size=len(cells)))
        expression["EGFR"] = 10.0 + 5.0 * pic50
        expression_csv = tmp_path / "expression.csv"
        expression.to_csv(expression_csv, index=False)

        response_rows = []
        for i, cell in enumerate(cells):
            response_rows.append(
                {"cell_line": cell, "compound": "erlotinib",
                 "pIC50": round(float(pic50.iloc[i]), 4)}
            )
            # a second compound that must be filtered out via --compound
            response_rows.append(
                {"cell_line": cell, "compound": "sorafenib",
                 "pIC50": round(float(rng.uniform(4.0, 8.0)), 4)}
            )
        response_csv = tmp_path / "response.csv"
        pd.DataFrame(response_rows).to_csv(response_csv, index=False)

        hits_csv = tmp_path / "hits.csv"
        code = main(
            [
                "biology", "top-biomarkers",
                "--expression", str(expression_csv),
                "--response", str(response_csv),
                "--response-column", "pIC50",
                "--compound", "erlotinib",
                "--higher-is-sensitive",
                "--top", "5",
                "--out", str(hits_csv),
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "compound : erlotinib (10 cell line(s))" in out
        # the planted biomarker ranks first and is called sensitive
        first = out.splitlines()
        egfr_lines = [line for line in first if line.startswith("EGFR")]
        assert egfr_lines, "planted EGFR row missing from the CLI output"
        assert "expression high -> more sensitive" in egfr_lines[0]
        assert hits_csv.is_file()
        frame = pd.read_csv(hits_csv)
        assert frame.iloc[0]["gene"] == "EGFR"
        assert bool(frame.iloc[0]["significant"])
        assert "direction" in frame.columns
        assert len(frame) == 5

    def test_response_audit_reports_missing_identifiers(self, tmp_path, capsys):
        screen_csv = self._screen_csv(tmp_path)
        import pandas as pd

        frame = pd.read_csv(screen_csv)
        frame.loc[0, "cell_line"] = ""   # one row loses its cell-line identifier
        frame.loc[1, "compound"] = ""    # one row loses its compound identifier
        missing_csv = tmp_path / "screen_missing_ids.csv"
        frame.to_csv(missing_csv, index=False)
        code = main(
            [
                "biology", "response-audit", "--data", str(missing_csv),
                "--tissue-column", "tissue",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "missing identifier" in out
        assert "column 'cell_line'" in out
        assert "column 'compound'" in out

    def test_top_biomarkers_keeps_missing_cell_line_rows_visible(
        self, tmp_path, capsys
    ):
        """A NaN cell-line row must not vanish from the reported panel size.

        The replicate-collapse groupby used pandas' default dropna=True,
        so rows without a cell-line identifier were dropped silently;
        they now survive as their own group and the correlation
        alignment reports them as unmatched.
        """
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(23)
        cells = ["A", "B", "C"]
        expression = pd.DataFrame(
            rng.normal(50, 5, size=(len(cells), 2)), columns=["G1", "G2"]
        )
        expression.insert(0, "cell_line", cells)
        expression_csv = tmp_path / "expression.csv"
        expression.to_csv(expression_csv, index=False)

        rows = [
            {"cell_line": "A", "compound": "d", "pIC50": 1.0},
            {"cell_line": "A", "compound": "d", "pIC50": 3.0},   # replicate
            {"cell_line": None, "compound": "d", "pIC50": 9.0},  # no identifier
            {"cell_line": "B", "compound": "d", "pIC50": 5.0},
            {"cell_line": "C", "compound": "d", "pIC50": 7.0},
        ]
        response_csv = tmp_path / "response.csv"
        pd.DataFrame(rows).to_csv(response_csv, index=False)

        code = main(
            [
                "biology", "top-biomarkers",
                "--expression", str(expression_csv),
                "--response", str(response_csv),
                "--response-column", "pIC50",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        # 4 response rows survive the collapse (A, NaN, B, C), not 3
        assert "compound : d (4 cell line(s))" in out

    def test_top_biomarkers_conflicting_direction_flags_error(
        self, tmp_path, capsys
    ):
        screen_csv = self._screen_csv(tmp_path)
        code = main(
            [
                "biology", "top-biomarkers",
                "--expression", str(screen_csv),
                "--response", str(screen_csv),
                "--response-column", "LN_IC50",
                "--lower-is-sensitive", "--higher-is-sensitive",
            ]
        )
        assert code == 1
        err = capsys.readouterr().err
        assert "error:" in err
        assert "at most one of --lower-is-sensitive / --higher-is-sensitive" in err


class TestCliStderrSingleCopy:
    """26-R5 defect 6: every CLI failure used to hit stderr TWICE.

    The ``pharmaml`` console logging handler echoed the ``_LOG.error``
    record AND ``main()`` printed its plain ``error: ...`` line.  A real
    subprocess run is required -- in-process tests silence logging, and
    the failure path is exercised exactly as a shell would see it.
    """

    def test_failure_message_appears_exactly_once_on_stderr(self, tmp_path):
        import subprocess
        import sys

        tiny = tmp_path / "tiny.csv"
        pd.DataFrame({"a": [1, 2]}).to_csv(tiny, index=False)
        log_file = tmp_path / "cli.log"
        proc = subprocess.run(
            [
                sys.executable, "-m", "pharmaml.cli",
                "--log-file", str(log_file),
                "data", "split", "--data", str(tiny),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,  # the failure's exit code is asserted below
        )
        assert proc.returncode == 1
        marker = "too few to split"
        assert proc.stderr.count(marker) == 1, (
            "CLI failure message must appear on stderr exactly once:\n"
            + proc.stderr
        )
        # the --log-file handler still records the failure (only the
        # console echo is suppressed)
        assert marker in log_file.read_text(encoding="utf-8")


class TestOutPathMessagesMatchWrittenFiles:
    """26-R5 defect 7: success messages must name the file write_table RETURNS.

    ``write_table`` falls back to CSV for unknown extensions (``.dat`` ->
    ``.csv``), so the CLI used to print paths that were never written.
    """

    @staticmethod
    def _activity_csv(tmp_path: Path) -> Path:
        csv = tmp_path / "act.csv"
        pd.DataFrame({"IC50 (nM)": [100.0, 500.0]}).to_csv(csv, index=False)
        return csv

    def test_pic50_out_dat_message_names_written_csv(self, tmp_path, capsys):
        csv = self._activity_csv(tmp_path)
        out = tmp_path / "res.dat"
        code = main(["chem", "pic50", "--data", str(csv), "--out", str(out)])
        assert code == 0
        stdout = capsys.readouterr().out
        assert "res.csv" in stdout
        assert "res.dat" not in stdout
        assert (tmp_path / "res.csv").is_file()
        assert not out.exists()

    def test_descriptors_out_dat_message_names_written_csv(self, tmp_path, capsys):
        csv = tmp_path / "lig.csv"
        csv.write_text("smiles\nCCO\nCCN\n", encoding="utf-8")
        out = tmp_path / "res.dat"
        code = main(
            [
                "chem", "descriptors", "--data", str(csv),
                "--smiles-col", "smiles", "--out", str(out),
            ]
        )
        assert code == 0
        stdout = capsys.readouterr().out
        assert "res.csv" in stdout
        assert "res.dat" not in stdout
        assert (tmp_path / "res.csv").is_file()

    def test_docking_scan_out_dat_message_names_written_csv(
        self, tmp_path, capsys, sample_dlg
    ):
        run = tmp_path / "run" / "ligand_01"
        run.mkdir(parents=True)
        (run / "r.dlg").write_text(sample_dlg, encoding="utf-8")
        out = tmp_path / "res.dat"
        code = main(
            ["docking", "scan", "--directory", str(tmp_path / "run"), "--out", str(out)]
        )
        assert code == 0
        stdout = capsys.readouterr().out
        assert "res.csv" in stdout
        assert "res.dat" not in stdout
        assert (tmp_path / "res.csv").is_file()

    def test_docking_check_out_dat_message_names_written_csv(
        self, tmp_path, capsys, sample_dlg
    ):
        run = tmp_path / "run" / "ligand_01"
        run.mkdir(parents=True)
        (run / "r.dlg").write_text(sample_dlg, encoding="utf-8")
        out = tmp_path / "res.dat"
        code = main(
            ["docking", "check", "--directory", str(tmp_path / "run"), "--out", str(out)]
        )
        assert code == 0  # 'Successful Completion' sentinel -> all OK
        stdout = capsys.readouterr().out
        assert "res.csv" in stdout
        assert "res.dat" not in stdout
        assert (tmp_path / "res.csv").is_file()


class TestMlTrainHelp:
    def test_out_help_is_honest_about_artifacts(self, capsys):
        # 26-R5 defect 8: --out claimed to be the "artifact output
        # directory" but only controls plots; the model always lands
        # under ./artifacts/ (PipelineConfig.output_path).
        with pytest.raises(SystemExit):
            main(["ml", "train", "--help"])
        out = capsys.readouterr().out
        assert "plots/report output directory" in out
        assert "always saved under ./artifacts/" in out


class TestGenerativeRunPlotFlag:
    """`generative run --plot <path>`: the adaptive run's trajectory figure."""

    def test_plot_on_adaptive_run_writes_png(self, tmp_path, capsys):
        # 100x-44-A: --adaptive-operators + --plot saves the run's
        # operator-trajectory figure (plot_operator_trajectory) via
        # save_figure -- the chem compare --figure idiom on the GA path.
        figure = tmp_path / "operators.png"
        code = main([
            "generative", "run",
            "--method", "ga", "--recipe", "qed", "--adaptive-operators",
            "--iterations", "4", "--population", "12", "--seed", "42",
            "--top", "3", "--plot", str(figure),
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "adaptive : on" in out
        assert "operators.png" in out  # the artifact path is echoed back
        assert figure.is_file() and figure.stat().st_size > 1_000
        assert figure.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic
        # matplotlib figures must not leak between CLI invocations
        import matplotlib.pyplot as plt

        assert plt.get_fignums() == []

    def test_plot_on_uniform_run_is_an_actionable_error(self, tmp_path, capsys):
        # a uniform run has no operator trajectory to draw: the flag must
        # fail loudly with the fix, not write an empty/vacuous figure.
        figure = tmp_path / "never.png"
        code = main([
            "generative", "run",
            "--method", "ga", "--recipe", "qed",
            "--iterations", "1", "--population", "8", "--plot", str(figure),
        ])
        assert code == 1
        error = capsys.readouterr().err
        assert "error:" in error
        assert "--plot" in error and "--adaptive-operators" in error
        assert not figure.exists()

    def test_plot_with_svg_extension_writes_svg(self, tmp_path, capsys):
        # save_figure infers the format from the extension; svg is
        # byte-reproducible since 100x-41-E but this pins structure only.
        figure = tmp_path / "nested" / "operators.svg"
        code = main([
            "generative", "run",
            "--method", "ga", "--recipe", "qed", "--adaptive-operators",
            "--iterations", "3", "--population", "10", "--seed", "42",
            "--top", "2", "--plot", str(figure),
        ])
        assert code == 0
        assert figure.is_file() and figure.stat().st_size > 1_000
        head = figure.read_bytes()[:512]
        assert head.lstrip().startswith(b"<?xml") and b"<svg" in head

    def test_plot_on_rl_method_writes_introspection_png(self, tmp_path, capsys):
        # 100x-66: --plot on an RL run saves the prior-drift introspection
        # figure (plot_rl_introspection) -- the GA-only restriction is
        # lifted the same round the diagnostics landed in history. The
        # old "ga only" typed error is retired with it.
        figure = tmp_path / "drift.png"
        code = main([
            "generative", "run",
            "--method", "rl", "--recipe", "qed", "--plot", str(figure),
            "--iterations", "2", "--batch-size", "8",
            "--prior-epochs", "2", "--seed", "42", "--top", "3",
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "drift.png" in out  # the artifact path is echoed back
        assert "prior-drift introspection" in out
        assert figure.is_file() and figure.stat().st_size > 1_000
        assert figure.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic
        # matplotlib figures must not leak between CLI invocations
        import matplotlib.pyplot as plt

        assert plt.get_fignums() == []


class TestGenerativeRunParetoFlag:
    """`generative run --pareto`: the multi-objective trade-off report.

    Fail-first (100x-57): the flags did not exist before this round --
    ``--pareto`` was an argparse "unrecognized arguments" SystemExit(2),
    and the handler printed no pareto block.  The CLI surface follows the
    --plot convention (100x-44-A): actionable typed errors for wrong
    recipes, the artifact path echoed back, figures never leaked.
    """

    def test_pareto_on_multi_recipe_prints_front_audit_and_table(
        self, tmp_path, capsys
    ):
        # seed-42 druglike run: 7 of 36 molecules non-dominated (pinned,
        # deterministic); the deepest-rank line is the round's story.
        code = main([
            "generative", "run",
            "--method", "ga", "--recipe", "druglike",
            "--iterations", "4", "--population", "12", "--seed", "42",
            "--top", "3", "--pareto",
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "pareto   : Pareto front over 3 objectives" in out
        assert "[qed, mw_band, tpsa_band]" in out
        assert "7 of 36 molecules non-dominated" in out
        assert "deepest front member sits at leaderboard rank" in out
        assert "dominates" in out  # the front table's trade-off column
        assert "(front truncated to 3 of 7 non-dominated molecules" in out

    def test_pareto_keep_prunes_to_most_isolated_with_honest_summary(
        self, tmp_path, capsys
    ):
        # the same pinned seed-42 druglike run (7 of 36 non-dominated),
        # pruned to the 3 most isolated: the summary reports the ORIGINAL
        # front size + the cut, and the table carries the crowding column
        # (inf = the trade-off corners that must survive)
        code = main([
            "generative", "run",
            "--method", "ga", "--recipe", "druglike",
            "--iterations", "4", "--population", "12", "--seed", "42",
            "--top", "3", "--pareto", "--pareto-keep", "3",
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "7 of 36 molecules non-dominated" in out  # original size
        assert "pruned to the 3 most isolated of 7 by crowding distance" in out
        assert "crowding" in out  # the front table's isolation column
        assert "inf" in out       # corner members survive as +inf
        # the rank-truncation warning must NOT fire: prune already selected
        assert "front truncated" not in out

    def test_pareto_keep_without_pareto_is_an_error(self, tmp_path, capsys):
        code = main([
            "generative", "run",
            "--method", "ga", "--recipe", "druglike",
            "--iterations", "1", "--population", "8",
            "--pareto-keep", "4",
        ])
        assert code == 1
        error = capsys.readouterr().err
        assert "--pareto-keep prunes the front --pareto extracts" in error
        assert "re-run with --pareto" in error

    def test_pareto_keep_below_one_is_an_error(self, tmp_path, capsys):
        code = main([
            "generative", "run",
            "--method", "ga", "--recipe", "druglike",
            "--iterations", "1", "--population", "8",
            "--pareto", "--pareto-keep", "0",
        ])
        assert code == 1
        error = capsys.readouterr().err
        assert "--pareto-keep must be at least 1, got 0" in error

    def test_pareto_plot_writes_png_and_echoes_the_objective_pair(
        self, tmp_path, capsys
    ):
        figure = tmp_path / "front.png"
        code = main([
            "generative", "run",
            "--method", "ga", "--recipe", "druglike",
            "--iterations", "4", "--population", "12", "--seed", "42",
            "--top", "3", "--pareto", "--pareto-plot", str(figure),
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert f"pareto plot: {figure}" in out
        assert "(front over qed x mw_band" in out
        assert figure.is_file() and figure.stat().st_size > 1_000
        assert figure.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic
        # matplotlib figures must not leak between CLI invocations
        import matplotlib.pyplot as plt

        assert plt.get_fignums() == []

    def test_pareto_on_single_objective_recipe_is_an_actionable_error(
        self, tmp_path, capsys
    ):
        code = main([
            "generative", "run",
            "--method", "ga", "--recipe", "qed",
            "--iterations", "1", "--population", "8", "--pareto",
        ])
        assert code == 1
        error = capsys.readouterr().err
        assert "error:" in error
        assert "'qed' recipe optimises a single objective" in error
        assert "druglike, novel or similar" in error

    def test_pareto_plot_without_pareto_is_an_error(self, tmp_path, capsys):
        figure = tmp_path / "never.png"
        code = main([
            "generative", "run",
            "--method", "ga", "--recipe", "druglike",
            "--iterations", "1", "--population", "8",
            "--pareto-plot", str(figure),
        ])
        assert code == 1
        error = capsys.readouterr().err
        assert "--pareto-plot" in error and "re-run with --pareto" in error
        assert not figure.exists()

    def test_pareto_works_on_the_rl_method_too(self, tmp_path, capsys):
        # RL fills the same per-objective breakdown (rl.py collects
        # score_with_breakdown like the GA) -- --pareto is method-neutral
        # by design, unlike the GA-only introspection flags.
        pytest.importorskip("torch")
        code = main([
            "generative", "run",
            "--method", "rl", "--recipe", "druglike",
            "--iterations", "2", "--batch-size", "8", "--seed", "7",
            "--top", "3", "--pareto",
        ])
        assert code == 0
        out = capsys.readouterr().out
        assert "pareto   : Pareto front over 3 objectives" in out
