"""Nested classification split hygiene, reports, and file/CLI behavior."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold

from pharmaml.cli import main
from pharmaml.exceptions import DataValidationError, JobCancelled
from pharmaml.ml.nested_classification import nested_logistic_from_file, nested_logistic_validation


def classification_frame():
    # Each group contains both labels; class strings are intentionally distinct IDs.
    index = np.arange(72)
    return pd.DataFrame({"x": (index % 2) * 3 + index / 200, "x2": index / 20,
                         "label": np.where(index % 2, "01", "1"),
                         "group": [f"g{i // 6:02d}" for i in index]})


def test_stratified_nested_classification_and_reproducible_report():
    data = classification_frame()
    first = nested_logistic_validation(data, features=["x", "x2"], target="label")
    second = nested_logistic_validation(data, features=["x", "x2"], target="label")
    assert first == second and first["classes"] == ["01", "1"]
    assert first["split_strategy"] == "stratified_shuffled_rows"
    assert len(first["folds"]) == 3 and 0 <= first["mean_balanced_accuracy"] <= 1
    for fold in first["folds"]:
        assert fold["selected_C"] in [0.1, 1.0, 10.0]
        assert sum(map(sum, fold["confusion_matrix"])) == fold["test_rows"]
        assert 0 <= fold["balanced_accuracy"] <= 1


def test_group_disjoint_both_levels_and_classes(monkeypatch):
    original = GroupKFold.split
    sizes = []

    def check(self, X, y=None, groups=None):
        g = np.asarray(groups)
        labels = np.asarray(y)
        for train, test in original(self, X, y, groups):
            assert set(g[train]).isdisjoint(g[test])
            assert set(labels[train]) == set(labels[test]) == {"01", "1"}
            sizes.append((len(train), len(test)))
            yield train, test

    monkeypatch.setattr(GroupKFold, "split", check)
    result = nested_logistic_validation(classification_frame(), features=["x"], target="label", group="group")
    assert result["split_strategy"] == "group_kfold" and len(sizes) == 12


def test_fold_local_preprocessing_and_training(monkeypatch):
    seen = []

    # Patching at the sklearn pipeline class guarantees cloning preserves the spy.
    from sklearn.impute import SimpleImputer
    actual_fit = SimpleImputer.fit

    def spy(self, X, *args, **kwargs):
        seen.append(set(X.index))
        return actual_fit(self, X, *args, **kwargs)

    monkeypatch.setattr(SimpleImputer, "fit", spy)
    data = classification_frame()
    result = nested_logistic_validation(data, features=["x", "x2"], target="label")
    outer = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(data[["x", "x2"]], data.label))
    assert len(seen) == 3 * (3 * 3 + 1)
    for i, (train, test) in enumerate(outer):
        for fitted_indices in seen[i * 10:(i + 1) * 10]:
            assert fitted_indices <= set(train) and fitted_indices.isdisjoint(test)
        assert result["folds"][i]["train_rows"] == len(train)


def test_missing_class_in_group_fold_rejected_before_any_model_fit(monkeypatch):
    data = pd.DataFrame({"x": range(24), "label": ["a"] * 6 + ["b"] * 6 + ["a"] * 6 + ["b"] * 6,
                         "group": [f"g{i//6}" for i in range(24)]})
    fits = []
    monkeypatch.setattr(GridSearchCV, "fit", lambda *args, **kwargs: fits.append(1))
    with pytest.raises(DataValidationError, match="every outer training and evaluation fold"):
        nested_logistic_validation(data, features=["x"], target="label", group="group", outer_folds=2)
    assert not fits


@pytest.mark.parametrize("case", ["nan_label", "blank_label", "single_class", "nan_group", "blank_group",
                                  "feature_inf", "feature_text", "target_feature", "too_few_class",
                                  "missing_inner_feature", "bad_seed"])
def test_invalid_data_fails_closed(case):
    data = classification_frame()
    options = {"features": ["x"], "target": "label", "group": "group"}
    if case == "nan_label":
        data.loc[0, "label"] = None
    elif case == "blank_label":
        data.loc[0, "label"] = " "
    elif case == "single_class":
        data["label"] = "a"
    elif case == "nan_group":
        data.loc[0, "group"] = None
    elif case == "blank_group":
        data.loc[0, "group"] = " "
    elif case == "feature_inf":
        data.loc[0, "x"] = np.inf
    elif case == "feature_text":
        data["x"] = "not numeric"
    elif case == "target_feature":
        options["features"] = ["label"]
    elif case == "too_few_class":
        data["label"] = "a"
        data.loc[0, "label"] = "b"
        options["group"] = None
    elif case == "missing_inner_feature":
        data["x"] = np.nan
        data.loc[0, "x"] = 2
    elif case == "bad_seed":
        options["seed"] = -1
    with pytest.raises(DataValidationError):
        nested_logistic_validation(data, **options)


def test_cancel_before_start_or_after_first_fold(monkeypatch):
    original = GridSearchCV.fit
    fits = []

    def spy(self, *args, **kwargs):
        fits.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GridSearchCV, "fit", spy)
    with pytest.raises(JobCancelled):
        nested_logistic_validation(classification_frame(), features=["x"], target="label", on_fold=lambda *_: False)
    assert not fits
    progress = []

    def stop(done, total):
        progress.append((done, total))
        return done < 1

    with pytest.raises(JobCancelled):
        nested_logistic_validation(classification_frame(), features=["x"], target="label", on_fold=stop)
    assert len(fits) == 1 and progress == [(0, 3), (1, 3)]


def test_file_provenance_literal_labels_and_changed_input(tmp_path, monkeypatch):
    from pharmaml.utils import identifier_table

    path = tmp_path / "table.csv"
    classification_frame().to_csv(path, index=False)
    report = nested_logistic_from_file(path, features=["x"], target="label", group="group")
    assert report["classes"] == ["01", "1"] and len(report["provenance"]["input"]["sha256"]) == 64
    original = identifier_table.read_identifier_table

    def changed(path, delimiter):
        result = original(path, delimiter)
        path.write_text(path.read_text() + "\n")
        return result

    monkeypatch.setattr(identifier_table, "read_identifier_table", changed)
    with pytest.raises(DataValidationError, match="changed while reading"):
        nested_logistic_from_file(path, features=["x"], target="label")


def test_cli_nested_logistic(tmp_path, capsys):
    path = tmp_path / "table.csv"
    classification_frame().to_csv(path, index=False)
    assert main(["ml", "nested-logistic", "--data", str(path), "--features", "x", "x2",
                 "--target", "label", "--group", "group"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["classes"] == ["01", "1"] and len(result["folds"]) == 3


def test_inner_group_class_deficiency_fails_before_fit(monkeypatch):
    # Three outer folds each contain both classes, but their training subsets
    # contain only two class-a groups and cannot cover three inner test folds.
    data = pd.DataFrame({"x": range(36), "label": ["a", "b"] * 18,
                         "group": [f"g{i // 6}" for i in range(36)]})
    for group in ("g0", "g2", "g4"):
        data.loc[data.group == group, "label"] = "a"
    for group in ("g1", "g3", "g5"):
        data.loc[data.group == group, "label"] = "b"
    # For GroupKFold(3), group pairs are (g5,g2), (g4,g1), (g3,g0).
    fits = []
    monkeypatch.setattr(GridSearchCV, "fit", lambda *args, **kwargs: fits.append(1))
    with pytest.raises(DataValidationError, match="every inner training and evaluation fold"):
        nested_logistic_validation(data, features=["x"], target="label", group="group", inner_folds=3)
    assert not fits


def test_convergence_warning_is_a_failure(monkeypatch):
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    def no_convergence(*args, **kwargs):
        warnings.warn("not converged", ConvergenceWarning)

    monkeypatch.setattr(GridSearchCV, "fit", no_convergence)
    with pytest.raises(DataValidationError, match="failed to converge"):
        nested_logistic_validation(classification_frame(), features=["x"], target="label")


def test_multiclass_confusion_matrix():
    frame = classification_frame()
    frame["label"] = np.array(["red", "green", "blue"] * 24)
    result = nested_logistic_validation(frame, features=["x", "x2"], target="label")
    assert result["classes"] == ["blue", "green", "red"]
    for fold in result["folds"]:
        assert len(fold["confusion_matrix"]) == 3
        assert all(len(row) == 3 for row in fold["confusion_matrix"])
        assert sum(map(sum, fold["confusion_matrix"])) == fold["test_rows"]
