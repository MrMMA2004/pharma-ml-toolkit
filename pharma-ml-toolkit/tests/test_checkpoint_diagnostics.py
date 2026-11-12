import json

from pharmaml.cli import main
from pharmaml.ml.checkpoint_diagnostics import inspect_checkpoint
from pharmaml.utils.checkpoint_schema import OperationSpec
from pharmaml.utils.checkpoints import CheckpointManager


def test_checkpoint_read_only_fingerprints_and_changed_input(tmp_path, capsys):
    data = tmp_path / "data.csv"
    data.write_text("before")
    manager = CheckpointManager(tmp_path / "jobs", "job")
    saved = manager.save(step=2, total=5, state={}, inputs=[data],
                         operation=OperationSpec(operation_type="numpy_mlp_training", operation_id="demo"))
    before = saved.path.read_bytes()
    result = inspect_checkpoint(saved.path, inputs=[data])
    assert result["valid_envelope"] and result["input_status"] == "match"
    assert result["implementation_status"] == "match" and result["resume_unit"] == 2
    assert inspect_checkpoint(saved.path)["input_status"] == "not checked"
    data.write_text("after")
    assert main(["research", "checkpoint", str(saved.path), "--input", str(data)]) == 1
    assert json.loads(capsys.readouterr().out)["input_status"].startswith("changed")
    assert saved.path.read_bytes() == before


def test_legacy_not_silently_upgraded(tmp_path):
    saved = CheckpointManager(tmp_path, "legacy").save(step=1, state={})
    result = inspect_checkpoint(saved.path)
    assert result["legacy"] and not result["valid_envelope"]
