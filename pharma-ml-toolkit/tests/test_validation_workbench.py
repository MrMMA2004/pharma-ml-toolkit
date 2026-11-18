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


def test_cancellation_occurs_only_at_outer_fold_boundaries(monkeypatch):
    from sklearn.model_selection import GridSearchCV

    from pharmaml.exceptions import JobCancelled

    fits = []
    original = GridSearchCV.fit

    def spy(self, *args, **kwargs):
        fits.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GridSearchCV, "fit", spy)
    progress = []

    def stop_after_first(done, total):
        progress.append((done, total))
        return done < 1

    with pytest.raises(JobCancelled, match="fold boundary"):
        nested_ridge_validation(frame(), features=["x"], target="y", group="group", on_fold=stop_after_first)
    assert progress == [(0, 3), (1, 3)] and len(fits) == 1


def test_file_workflow_tracks_hash_and_refuses_changed_input(tmp_path, monkeypatch):
    from pharmaml.ml.validation_workbench import nested_ridge_from_file
    from pharmaml.utils import identifier_table

    path = tmp_path / "input.csv"
    frame().to_csv(path, index=False)
    result = nested_ridge_from_file(path, features=["x"], target="y", group="group")
    assert len(result["provenance"]["input"]["sha256"]) == 64
    original = identifier_table.read_identifier_table

    def change(path, delimiter):
        parsed = original(path, delimiter)
        path.write_text(path.read_text() + "\n")
        return parsed

    monkeypatch.setattr(identifier_table, "read_identifier_table", change)
    with pytest.raises(DataValidationError, match="changed while reading"):
        nested_ridge_from_file(path, features=["x"], target="y", group="group")
