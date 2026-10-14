"""CLI-level tests for the new commands and flags.

Exercises ``pharmaml.data split``, ``ml train --plots/--keep-categorical``,
``ml compare --out/--plot`` and model aliases through :func:`pharmaml.cli.main`
exactly the way a shell would invoke them.
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
