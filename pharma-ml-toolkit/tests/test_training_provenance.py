"""Observed, opt-in training provenance; real runs and safety boundaries."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from pharmaml.cli import main
from pharmaml.exceptions import DataValidationError
from pharmaml.ml.preprocessing import make_train_test_split
from pharmaml.ml.training_provenance import (
    _json_value,
    compare_training_records,
    prepare_record_path,
)


def make_data(tmp_path):
    data = tmp_path / "input.csv"
    x = np.arange(60, dtype=float)
    pd.DataFrame({"x": x, "y": x * 2 + np.sin(x), "note": ["a"] * 60}).to_csv(data, index=False)
    return data


def cli_args(data, record, *extra):
    return ["ml", "train", "--data", str(data), "--target", "y", "--task", "regression", "--model", "ridge",
            "--run-record", str(record), "--cv", "2", *extra]


def test_real_run_record_and_comparison(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    data, first, second = make_data(tmp_path), tmp_path / "first.json", tmp_path / "second.json"
    assert main(cli_args(data, first, "--no-tune")) == 0
    one = json.loads(first.read_text())
    assert one["dataset"]["sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
    assert one["source_status_after_run"] == "match"
    assert one["input_processing"]["features"] == ["x"]
    assert one["input_processing"]["dropped_columns"] == ["note"]
    assert not one["split"]["stratified"] and one["split"]["train_rows"] == 48
    assert one["training"]["grid_search_performed"] is False
    artifact = tmp_path / "artifacts/ridge_regression.joblib"
    assert one["model_artifact"]["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert one["training"]["fitted_pipeline"]["estimator_type"] == "sklearn.pipeline.Pipeline"
    assert main(cli_args(data, second, "--no-tune")) == 0
    two = json.loads(second.read_text())
    comparison = compare_training_records(one, two)
    assert comparison["same_dataset_sha256"] and comparison["same_split"]
    assert comparison["same_training"] and comparison["same_input_processing"]
    capsys.readouterr()
    assert main(["research", "compare-training", str(first), str(second)]) == 0
    assert json.loads(capsys.readouterr().out)["same_split"]


def test_tuned_run_records_cv_and_selected_actual_parameters(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data, record = make_data(tmp_path), tmp_path / "run.json"
    assert main(cli_args(data, record)) == 0
    payload = json.loads(record.read_text())
    training = payload["training"]
    assert training["grid_search_performed"] and training["cv_folds"] == 2
    assert training["cv_strategy"] == "sklearn k-fold (no shuffle)"
    assert training["cv_scorer"] == "neg_root_mean_squared_error"
    assert len(training["cv_scores"]) == 2 and "alpha" in training["selected_parameters"]
    assert training["selected_parameters"]["alpha"] in training["registry_grid"]["alpha"]
    assert training["cv_n_jobs"] == -1


def test_existing_record_refused_before_model_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = make_data(tmp_path)
    record = tmp_path / "keep.json"
    record.write_text("untouched")
    from pharmaml.ml import training

    calls = []
    monkeypatch.setattr(training, "train_model", lambda *a, **k: calls.append(True))
    assert main(cli_args(data, record, "--no-tune")) == 1
    assert record.read_text() == "untouched" and not calls and not (tmp_path / "artifacts").exists()


def test_exclusive_write_does_not_clobber_later_race(tmp_path):
    from pharmaml.ml.training_provenance import write_new_record

    target = prepare_record_path(tmp_path / "new.json")
    target.write_text("someone else")
    with pytest.raises(DataValidationError, match="could not write"):
        write_new_record(target, {"a": 1})
    assert target.read_text() == "someone else"


def test_dangling_symlink_and_invalid_parent_refused(tmp_path):
    link = tmp_path / "link.json"
    try:
        link.symlink_to(tmp_path / "missing")
    except OSError:
        pytest.skip("symbolic links unavailable")
    with pytest.raises(DataValidationError, match="new, non-existing"):
        prepare_record_path(link)
    with pytest.raises(DataValidationError, match="parent directory"):
        prepare_record_path(tmp_path / "missing-parent/file.json")
    with pytest.raises(DataValidationError, match=".json"):
        prepare_record_path(tmp_path / "file.txt")


def test_changed_during_parse_aborts_before_model_output(tmp_path, monkeypatch):
    from pharmaml import cli

    monkeypatch.chdir(tmp_path)
    data = make_data(tmp_path)
    original = cli._load_cli_dataset

    def changed(args):
        dataset = original(args)
        data.write_text(data.read_text() + "\n")
        return dataset

    monkeypatch.setattr(cli, "_load_cli_dataset", changed)
    assert main(cli_args(data, tmp_path / "record.json", "--no-tune")) == 1
    assert not (tmp_path / "record.json").exists() and not (tmp_path / "artifacts").exists()


def test_input_changes_after_load_are_flagged(tmp_path, monkeypatch, capsys):
    from pharmaml.ml import training

    monkeypatch.chdir(tmp_path)
    data, record = make_data(tmp_path), tmp_path / "run.json"
    original = training.train_model

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        data.write_text(data.read_text() + "\n")
        return result

    monkeypatch.setattr(training, "train_model", changed)
    assert main(cli_args(data, record, "--no-tune")) == 0
    assert json.loads(record.read_text())["source_status_after_run"] == "changed"
    assert "source changed" in capsys.readouterr().err


def test_split_callback_reports_actual_strategy_and_only_on_success():
    x = pd.DataFrame({"x": np.arange(20)})
    states = []
    make_train_test_split(x, pd.Series(["a", "b"] * 10), on_split=states.append)
    make_train_test_split(x, pd.Series(["a"] * 19 + ["b"]), on_split=states.append)
    make_train_test_split(x, pd.Series(np.arange(20, dtype=float)), on_split=states.append)
    assert states == [True, False, False]
    with pytest.raises(DataValidationError):
        make_train_test_split(x, pd.Series(["a", "b"] * 10), test_size=1, on_split=states.append)
    assert len(states) == 3


def test_parameter_json_unsupported_nonfinite_and_key_types():
    sample = _json_value({1: float("nan"), 2: object(), 3: np.int64(7)})
    assert sample["typed_mapping"][0]["value"] == {"nonfinite_float": "nan"}
    assert "omitted_type" in sample["typed_mapping"][1]["value"]
    json.dumps(sample, allow_nan=False)


def test_comparison_rejects_malformed_records():
    with pytest.raises(DataValidationError, match="invalid or unsupported"):
        compare_training_records({}, {})


def test_classification_records_actual_stratification_and_categorical_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    x = np.arange(80)
    data, record = tmp_path / "classification.csv", tmp_path / "classification.json"
    pd.DataFrame({"x": x / 20, "category": np.where(x % 3, "A", "B"),
                  "label": np.where(x % 2, "active", "inactive")}).to_csv(data, index=False)
    assert main(["ml", "train", "--data", str(data), "--target", "label", "--task", "classification",
                 "--model", "logistic", "--no-tune", "--keep-categorical", "--run-record", str(record)]) == 0
    payload = json.loads(record.read_text())
    assert payload["split"]["stratified"] is True
    assert payload["training"]["requested_model"] == "logistic"
    assert payload["training"]["resolved_model"] == "logistic_regression"
    assert payload["input_processing"]["categorical_columns"] == ["category"]
    assert payload["input_processing"]["features"] == ["x", "category"]


def test_missing_target_split_hash_refers_to_cleaned_rows(tmp_path, monkeypatch):
    from pharmaml.data import load_dataset
    from pharmaml.ml.training_provenance import _index_hash

    monkeypatch.chdir(tmp_path)
    data, record = make_data(tmp_path), tmp_path / "clean.json"
    df = pd.read_csv(data)
    df.loc[[2, 5], "y"] = np.nan
    df.to_csv(data, index=False)
    dataset = load_dataset(data, target="y", task="regression")
    train, test, _, _ = make_train_test_split(dataset.X, dataset.y)
    assert main(cli_args(data, record, "--no-tune")) == 0
    payload = json.loads(record.read_text())
    assert payload["input_processing"]["cleaned_rows"] == 58
    assert payload["split"]["train_ordered_clean_row_indices_sha256"] == _index_hash(train.index)
    assert payload["split"]["test_ordered_clean_row_indices_sha256"] == _index_hash(test.index)


def test_removed_source_after_loading_is_flagged(tmp_path, monkeypatch, capsys):
    from pharmaml.ml import training

    monkeypatch.chdir(tmp_path)
    data, record = make_data(tmp_path), tmp_path / "run.json"
    original = training.train_model

    def remove(*args, **kwargs):
        result = original(*args, **kwargs)
        data.unlink()
        return result

    monkeypatch.setattr(training, "train_model", remove)
    assert main(cli_args(data, record, "--no-tune")) == 0
    assert json.loads(record.read_text())["source_status_after_run"] == "unavailable"
    assert "unavailable" in capsys.readouterr().err
