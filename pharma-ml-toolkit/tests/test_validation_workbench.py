import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from pharmaml.cli import main
from pharmaml.exceptions import DataValidationError
from pharmaml.ml.validation_workbench import nested_ridge_validation


def frame():
    x = np.arange(36, dtype=float)
    return pd.DataFrame({"x": x, "y": x * 2 + 1, "group": [f"g{i//3}" for i in range(36)]})


def test_nested_groups_both_inner_outer(monkeypatch):
    original = GroupKFold.split
    seen = []

    def checked(self, X, y=None, groups=None):
        assert groups is not None
        groups = np.asarray(groups)
        for train, test in original(self, X, y, groups):
            assert set(groups[train]).isdisjoint(groups[test])
            seen.append((len(train), len(test)))
            yield train, test

    monkeypatch.setattr(GroupKFold, "split", checked)
    result = nested_ridge_validation(frame(), features=["x"], target="y", group="group")
    assert len(result["folds"]) == 3 and len(seen) == 12
    assert result["mean_mae"] >= 0


def test_invalid_group_and_columns():
    data = frame()
    data.loc[0, "group"] = None
    with pytest.raises(DataValidationError, match="groups"):
        nested_ridge_validation(data, features=["x"], target="y", group="group")
    with pytest.raises(DataValidationError, match="exclude"):
        nested_ridge_validation(frame(), features=["y"], target="y")


def test_cli_nested(tmp_path, capsys):
    path = tmp_path / "data.csv"
    frame().to_csv(path, index=False)
    assert main(["ml", "nested-ridge", "--data", str(path), "--features", "x",
                 "--target", "y", "--group", "group"]) == 0
    assert len(json.loads(capsys.readouterr().out)["folds"]) == 3
