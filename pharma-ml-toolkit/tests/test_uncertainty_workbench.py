import numpy as np
import pandas as pd
import pytest

from pharmaml.cli import main
from pharmaml.exceptions import DataValidationError
from pharmaml.ml.uncertainty_workbench import regression_uncertainty_report


def datasets():
    def frame(start, n):
        x = np.arange(start, start+n, dtype=float)
        return pd.DataFrame({"id": [f"c{i}" for i in range(start, start+n)],
                             "x": x, "y": x*2+1, "group": ["A", "B"]*(n//2)})
    return [frame(0, 20), frame(20, 10), frame(30, 10)]


def test_separate_partitions_and_subgroups():
    report = regression_uncertainty_report(*datasets(), features=["x"], target="y", key="id", group="group")
    assert report["n_train"] == 20 and report["n_calibration"] == report["n_test"] == 10
    assert sum(group["n_test"] for group in report["subgroups"]) == 10
    assert "no conditional guarantee" in report["limitations"]


def test_overlap_and_reserved_features_refused():
    train, cal, test = datasets()
    test.loc[0, "id"] = train.loc[0, "id"]
    with pytest.raises(DataValidationError, match="overlap"):
        regression_uncertainty_report(train, cal, test, features=["x"], target="y", key="id")
    with pytest.raises(DataValidationError, match="exclude"):
        regression_uncertainty_report(*datasets(), features=["x", "y"], target="y", key="id")


def test_insufficient_calibration_and_missing_groups_refused():
    train, cal, test = datasets()
    with pytest.raises(DataValidationError, match="calibration samples"):
        regression_uncertainty_report(train, cal[:2], test, features=["x"], target="y", key="id")
    test.loc[0, "group"] = None
    with pytest.raises(DataValidationError):
        regression_uncertainty_report(train, cal, test, features=["x"], target="y", key="id", group="group")


def test_cli_report_and_overlap_error(tmp_path, capsys):
    import json
    paths = []
    for i, frame in enumerate(datasets()):
        path = tmp_path / f"{i}.csv"
        frame.to_csv(path, index=False)
        paths.append(str(path))
    args = ["ml", "conformal-report", "--train", paths[0], "--calibration", paths[1],
            "--test", paths[2], "--features", "x", "--target", "y", "--key", "id", "--group", "group"]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["n_test"] == 10
    args[args.index("--test") + 1] = paths[0]
    assert main(args) == 1
    assert "overlap" in capsys.readouterr().err
