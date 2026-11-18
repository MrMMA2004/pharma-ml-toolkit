"""Held-out classification prediction sets and safe split checks."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from pharmaml.cli import main
from pharmaml.exceptions import DataValidationError
from pharmaml.ml.uncertainty_workbench import (
    classification_uncertainty_from_files,
    classification_uncertainty_report,
)


def data():
    def partition(start, n):
        x = np.arange(start, start + n, dtype=float)
        return pd.DataFrame({"id": [f"id{i}" for i in range(start, start + n)],
                             "x": x, "target": ["A", "B"] * (n // 2),
                             "group": ["g1", "g2"] * (n // 2)})
    return [partition(0, 20), partition(20, 10), partition(30, 10)]


def test_train_only_fitting_and_coverage_summary(monkeypatch):
    calls = []
    original = Pipeline.fit

    def spy(self, X, y=None, **kwargs):
        calls.append((list(X.index), list(y)))
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", spy)
    train, cal, test = data()
    cal.index += 100
    test.index += 200
    report = classification_uncertainty_report(train, cal, test, features=["x"],
                                                target="target", key="id", group="group")
    assert len(calls) == 1 and calls[0][0] == train.index.tolist()
    assert report["task"] == "classification" and report["classes"] == ["A", "B"]
    assert sum(group["n_test"] for group in report["class_coverage"]) == 10
    assert sum(report[name] for name in ("n_empty_sets", "n_singleton_sets", "n_multiple_sets")) == 10
    assert "not conditional guarantees" in report["limitations"]


def test_small_calibration_uses_full_set_fallback():
    train, cal, test = data()
    result = classification_uncertainty_report(train, cal[:2], test, features=["x"],
                                                target="target", key="id", alpha=0.1)
    assert result["full_set_fallback"] and result["mean_set_size"] == 2
    assert result["coverage"] == 1 and result["n_multiple_sets"] == 10


def test_reject_leakage_unknown_labels_and_missing_group():
    train, cal, test = data()
    test.loc[0, "id"] = train.loc[0, "id"]
    with pytest.raises(DataValidationError, match="overlap"):
        classification_uncertainty_report(train, cal, test, features=["x"], target="target", key="id")
    train, cal, test = data()
    cal.loc[0, "target"] = "unknown"
    with pytest.raises(DataValidationError, match="absent from training"):
        classification_uncertainty_report(train, cal, test, features=["x"], target="target", key="id")
    train, cal, test = data()
    test.loc[0, "target"] = "unknown"
    with pytest.raises(DataValidationError, match="absent from training"):
        classification_uncertainty_report(train, cal, test, features=["x"], target="target", key="id")
    train, cal, test = data()
    test.loc[0, "group"] = ""
    with pytest.raises(DataValidationError, match="subgroup labels"):
        classification_uncertainty_report(train, cal, test, features=["x"], target="target", key="id", group="group")
    train, cal, test = data()
    train["target"] = "A"
    with pytest.raises(DataValidationError, match="at least two"):
        classification_uncertainty_report(train, cal, test, features=["x"], target="target", key="id")


def test_cli_files_and_run_provenance(tmp_path, capsys):
    paths = []
    for index, part in enumerate(data()):
        path = tmp_path / f"{index}.csv"
        part.to_csv(path, index=False)
        paths.append(str(path))
    report = classification_uncertainty_from_files(*paths, features=["x"], target="target", key="id")
    assert report["provenance"]["parameters"]["workflow"] == "split-conformal-logistic"
    assert report["provenance"]["parameters"]["logistic_parameters"]["max_iter"] == 1000
    assert main(["ml", "conformal-classification", "--train", paths[0], "--calibration", paths[1],
                 "--test", paths[2], "--features", "x", "--target", "target", "--key", "id",
                 "--group", "group"]) == 0
    assert json.loads(capsys.readouterr().out)["task"] == "classification"


def test_different_workflows_do_not_compare_incommensurate_metrics(tmp_path):
    from pharmaml.ml.provenance import compare_uncertainty_reports
    from pharmaml.ml.uncertainty_workbench import regression_uncertainty_from_files

    paths = []
    for index, part in enumerate(data()):
        part["value"] = part["x"] * 2
        path = tmp_path / f"{index}.csv"
        part.to_csv(path, index=False)
        paths.append(path)
    classification = classification_uncertainty_from_files(*paths, features=["x"], target="target", key="id")
    regression = regression_uncertainty_from_files(*paths, features=["x"], target="value", key="id")
    comparison = compare_uncertainty_reports(classification, regression)
    assert not comparison["same_workflow"] and comparison["metrics"] is None
    assert not comparison["same_parameters"]


def test_optional_diagnostics_holdout_and_cli(tmp_path, capsys):
    train, calibration, test = data()
    plain = classification_uncertainty_report(train, calibration, test, features=["x"],
                                               target="target", key="id")
    diagnostic = classification_uncertainty_report(train, calibration, test, features=["x"],
                                                    target="target", key="id", diagnostics=True)
    assert "probability_diagnostics" not in plain and "applicability" not in plain
    assert diagnostic["coverage"] == plain["coverage"] and diagnostic["threshold"] == plain["threshold"]
    probability = diagnostic["probability_diagnostics"]
    assert 0 <= probability["multiclass_brier_sum"] <= 2
    assert all(sum(row["n"] for row in group["bins"]) == len(test) for group in probability["per_class"])
    assert diagnostic["applicability"]["status"] == "computed"
    paths = []
    for index, frame in enumerate((train, calibration, test)):
        path = tmp_path / f"{index}.csv"
        frame.to_csv(path, index=False)
        paths.append(str(path))
    assert main(["ml", "conformal-classification", "--train", paths[0], "--calibration", paths[1],
                 "--test", paths[2], "--features", "x", "--target", "target", "--key", "id",
                 "--diagnostics"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["probability_diagnostics"]["n_test"] == 10
    assert output["provenance"]["parameters"]["diagnostics"] is True
