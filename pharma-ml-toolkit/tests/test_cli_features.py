"""CLI-level tests for the new commands and flags.

Exercises ``pharmaml.data split``, ``ml train --plots/--keep-categorical``,
``ml compare --out/--plot``, model aliases and the docking campaign
subcommands (``dpf``/``collect``/``torsion``/``gpf-scan``/``vina-scan``)
through :func:`pharmaml.cli.main` exactly the way a shell would invoke
them.
"""

from __future__ import annotations

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
            "\n".join(
                [
                    "REMARK  Name = cli_ligand",
                    "ROOT",
                    carbon,
                    oxygen,
                    "ENDROOT",
                    "TORSDOF 1",
                ]
            )
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
