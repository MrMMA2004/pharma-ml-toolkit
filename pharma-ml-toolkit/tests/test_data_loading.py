"""Tests for :mod:`pharmaml.data` — importing the user's own files.

Covers ``load_dataset`` (CSV auto-detection, multiclass string labels,
regression inference, Excel), ``infer_task`` rules, ``summarize_dataset``
and the ``pharmaml data inspect`` / ``pharmaml ml train --task auto``
CLI commands built on top of them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pharmaml.data import (
    TARGET_CANDIDATES,
    Dataset,
    DatasetSummary,
    infer_task,
    load_dataset,
    summarize_dataset,
)
from pharmaml.exceptions import DataValidationError


def _classification_frame(n: int = 90, n_features: int = 4, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"descr_{i:02d}" for i in range(n_features)],
    )
    frame["Target"] = rng.choice([0, 1], size=n)
    return frame


@pytest.fixture(scope="module")
def binary_csv(tmp_path_factory) -> Path:
    """User CSV with a numeric binary target plus string side-columns."""
    frame = _classification_frame()
    frame["SMILES"] = ["CCO"] * len(frame)
    frame["ID"] = [f"cmpd_{i:03d}" for i in range(len(frame))]
    path = tmp_path_factory.mktemp("data") / "binary.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def multiclass_csv(tmp_path_factory) -> Path:
    """User CSV with a 3-class *string* target column."""
    rng = np.random.default_rng(21)
    n = 120
    frame = pd.DataFrame(
        rng.normal(size=(n, 4)), columns=[f"descr_{i:02d}" for i in range(4)]
    )
    frame["Activity"] = rng.choice(["low", "medium", "high"], size=n, p=[0.5, 0.3, 0.2])
    frame["Name"] = [f"lig-{i}" for i in range(n)]
    path = tmp_path_factory.mktemp("data") / "multiclass.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def regression_csv(tmp_path_factory) -> Path:
    """User CSV with a continuous pIC50 target."""
    rng = np.random.default_rng(31)
    n = 80
    frame = pd.DataFrame(
        rng.normal(size=(n, 3)), columns=[f"descr_{i:02d}" for i in range(3)]
    )
    frame["pIC50"] = 6.0 + frame.sum(axis=1) * 0.4 + rng.normal(scale=0.1, size=n)
    path = tmp_path_factory.mktemp("data") / "regression.csv"
    frame.to_csv(path, index=False)
    return path


class TestInferTask:
    def test_string_labels_are_classification(self) -> None:
        assert infer_task(pd.Series(["a", "b", "a"])) == "classification"

    def test_boolean_target_is_classification(self) -> None:
        assert infer_task(pd.Series([True, False, True])) == "classification"

    def test_numeric_multiclass_labels(self) -> None:
        assert infer_task(pd.Series([0, 1, 2, 2, 1])) == "classification"

    def test_continuous_numeric_is_regression(self) -> None:
        values = np.linspace(4.0, 8.0, 60)  # 60 distinct measurements
        assert infer_task(pd.Series(values)) == "regression"

    def test_few_unique_floats_are_classes(self) -> None:
        assert infer_task(pd.Series([0.0, 1.0, 2.0] * 10)) == "classification"

    def test_max_class_count_boundary(self) -> None:
        values = list(range(21))  # 21 unique values
        assert infer_task(pd.Series(values), max_class_count=21) == "classification"
        assert infer_task(pd.Series(values), max_class_count=20) == "regression"

    def test_single_value_raises(self) -> None:
        with pytest.raises(DataValidationError, match="distinct"):
            infer_task(pd.Series([1, 1, 1]))

    def test_bad_max_class_count_raises(self) -> None:
        with pytest.raises(DataValidationError, match="max_class_count"):
            infer_task(pd.Series([0, 1]), max_class_count=1)


class TestLoadDataset:
    def test_binary_autodetects_target(self, binary_csv: Path) -> None:
        ds = load_dataset(binary_csv)  # target=None -> finds 'Target'
        assert ds.target == "Target"
        assert ds.task == "classification"
        assert ds.classes == [0, 1]
        assert sum(ds.class_counts.values()) == ds.n_samples

    def test_string_multiclass(self, multiclass_csv: Path) -> None:
        ds = load_dataset(multiclass_csv, target="Activity")
        assert ds.task == "classification"
        assert ds.classes == ["high", "low", "medium"]
        assert len(ds.classes) == 3
        assert ds.n_features == 4  # only numeric descriptors
        assert "Name" in ds.dropped_columns
        assert "Activity" not in ds.X.columns

    def test_regression_inference(self, regression_csv: Path) -> None:
        ds = load_dataset(regression_csv)  # autodetects pIC50
        assert ds.target == "pIC50"
        assert ds.task == "regression"
        assert ds.classes is None and ds.class_counts is None

    def test_case_insensitive_target_autodetect(self, tmp_path: Path) -> None:
        frame = _classification_frame(n=20).rename(columns={"Target": "TARGET"})
        path = tmp_path / "upper.csv"
        frame.to_csv(path, index=False)
        ds = load_dataset(path)
        assert ds.target == "TARGET"

    def test_explicit_target_missing_raises(self, binary_csv: Path) -> None:
        with pytest.raises(DataValidationError, match="not found.*descr_00"):
            load_dataset(binary_csv, target="Nonexistent")

    def test_no_target_found_raises(self, tmp_path: Path) -> None:
        frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "no_target.csv"
        frame.to_csv(path, index=False)
        with pytest.raises(DataValidationError, match="target=<column name>"):
            load_dataset(path)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataValidationError, match="not found"):
            load_dataset(tmp_path / "ghost.csv")

    def test_invalid_task_argument(self, binary_csv: Path) -> None:
        with pytest.raises(DataValidationError, match="task must be"):
            load_dataset(binary_csv, task="clustering")

    def test_single_class_target_raises(self, tmp_path: Path) -> None:
        frame = _classification_frame(n=12)
        frame["Target"] = 1
        path = tmp_path / "single.csv"
        frame.to_csv(path, index=False)
        with pytest.raises(DataValidationError, match="at least 2"):
            load_dataset(path, target="Target")

    def test_regression_with_string_target_raises(self, multiclass_csv: Path) -> None:
        with pytest.raises(DataValidationError, match="non-numeric"):
            load_dataset(multiclass_csv, target="Activity", task="regression")

    def test_rows_with_missing_target_dropped(self, tmp_path: Path) -> None:
        frame = _classification_frame(n=30)
        frame.loc[0, "Target"] = np.nan
        path = tmp_path / "gap.csv"
        frame.to_csv(path, index=False)
        ds = load_dataset(path, target="Target")
        assert ds.n_samples == 29
        assert ds.y.notna().all()

    def test_drop_cols_excluded(self, binary_csv: Path) -> None:
        ds = load_dataset(binary_csv, target="Target", drop_cols=["descr_00"])
        assert "descr_00" not in ds.X.columns
        assert ds.n_features == 3

    def test_excel_input(self, tmp_path: Path) -> None:
        pytest.importorskip("openpyxl")
        frame = _classification_frame(n=24)
        path = tmp_path / "table.xlsx"
        frame.to_excel(path, index=False)
        ds = load_dataset(path, target="Target")
        assert ds.task == "classification"
        assert ds.n_samples == 24

    def test_semicolon_delimiter(self, tmp_path: Path) -> None:
        frame = _classification_frame(n=18)
        path = tmp_path / "euro.csv"
        frame.to_csv(path, sep=";", index=False)
        ds = load_dataset(path, target="Target")
        assert ds.n_features == 4


class TestDatasetObject:
    def test_properties(self, multiclass_csv: Path) -> None:
        ds = load_dataset(multiclass_csv, target="Activity")
        assert isinstance(ds, Dataset)
        assert ds.n_samples == 120
        assert ds.feature_names == [f"descr_{i:02d}" for i in range(4)]
        assert len(ds.feature_names) == ds.n_features

    def test_summary_roundtrip(self, multiclass_csv: Path) -> None:
        ds = load_dataset(multiclass_csv, target="Activity")
        summary = ds.summary()
        assert isinstance(summary, DatasetSummary)
        assert summary.task == "classification"
        assert summary.n_rows == 120
        assert summary.classes == ["high", "low", "medium"]

    def test_summarize_dataset_function(self, binary_csv: Path) -> None:
        ds = load_dataset(binary_csv, target="Target")
        summary = summarize_dataset(ds)
        payload = summary.as_dict()
        assert payload["target"] == "Target"
        assert payload["n_features"] == 4
        assert "dropped_columns" in payload

    def test_summary_str_mentions_classes(self, multiclass_csv: Path) -> None:
        ds = load_dataset(multiclass_csv, target="Activity")
        text = str(ds.summary())
        assert "classification" in text
        assert "low" in text and "medium" in text
        assert "dropped cols" in text  # 'Name' was dropped

    def test_target_candidates_sane(self) -> None:
        assert "Target" in TARGET_CANDIDATES
        assert "pIC50" in TARGET_CANDIDATES


class TestCli:
    def test_data_inspect(self, multiclass_csv: Path, capsys) -> None:
        from pharmaml.cli import main

        code = main(["data", "inspect", "--data", str(multiclass_csv), "--head", "3"])
        assert code == 0
        out = capsys.readouterr().out
        assert "classification" in out
        assert "first 3 row(s)" in out
        assert "Activity" in out

    def test_ml_train_auto_task(
        self, multiclass_csv: Path, capsys, monkeypatch, tmp_path: Path
    ) -> None:
        from pharmaml.cli import main

        monkeypatch.chdir(tmp_path)  # artifacts/ written here
        code = main(
            [
                "ml", "train",
                "--data", str(multiclass_csv),
                "--target", "Activity",
                "--model", "logistic_regression",
                "--no-tune",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "task:     classification" in out
        assert "'high'" in out  # class balance echoed
        assert (tmp_path / "artifacts").is_dir()

    def test_ml_compare_auto_task(self, multiclass_csv: Path, capsys) -> None:
        from pharmaml.cli import main

        code = main(
            [
                "ml", "compare",
                "--data", str(multiclass_csv),
                "--target", "Activity",
                "--models", "logistic_regression", "knn",
                "--cv", "3",
            ]
        )
        assert code == 0
        out = capsys.readouterr().out
        assert "logistic_regression" in out
