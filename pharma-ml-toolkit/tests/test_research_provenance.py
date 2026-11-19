import json

import pytest

from pharmaml.cli import main
from pharmaml.exceptions import DataValidationError
from pharmaml.ml.provenance import compare_manifests, experiment_manifest


def test_manifest_comparison_and_changed_input(tmp_path):
    path = tmp_path / "data.csv"
    # r297: raw bytes -- write_text CRLF-translates on Windows and the
    # manifest's size_bytes pin (4 B for "a\n1\n") would read 6 there.
    path.write_bytes(b"a\n1\n")
    before = experiment_manifest(path, parameters={"model": "ridge"}, split_strategy="scaffold")
    path.write_bytes(b"a\n2\n")
    after = experiment_manifest(path, parameters={"model": "ridge"}, split_strategy="scaffold")
    changes = compare_manifests(before, after)
    assert not changes["same_dataset_sha256"]
    assert changes["same_parameters"] and changes["same_split_strategy"]
    assert before["dataset"]["size_bytes"] == 4
    assert "not proof" in before["notice"]


def test_invalid_manifest_and_blank_strategy(tmp_path):
    with pytest.raises(DataValidationError):
        compare_manifests({}, {})
    with pytest.raises(DataValidationError):
        experiment_manifest(tmp_path / "missing", parameters={}, split_strategy=" ")


def test_cli_read_only_json_and_comparison(tmp_path, capsys):
    path = tmp_path / "data.csv"
    path.write_text("a\n1\n")
    assert main(["research", "manifest", "--data", str(path), "--split-strategy", "random",
                 "--parameter", "model=ridge"]) == 0
    doc = json.loads(capsys.readouterr().out)
    saved = tmp_path / "manifest.json"
    saved.write_text(json.dumps(doc))
    assert main(["research", "compare", str(saved), str(saved)]) == 0
    assert json.loads(capsys.readouterr().out)["same_dataset_sha256"]
    assert main(["research", "environment"]) == 0
    assert "packages" in json.loads(capsys.readouterr().out)
    assert main(["research", "manifest", "--data", str(path), "--split-strategy", "random",
                 "--parameter", "a=1", "--parameter", "a=2"]) == 1
    assert "unique" in capsys.readouterr().err
