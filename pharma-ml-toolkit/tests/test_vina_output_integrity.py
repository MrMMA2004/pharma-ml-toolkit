"""A zero exit without a new Vina output must not be reported as docking."""
import subprocess
from pathlib import Path

import pytest

from pharmaml.docking import workflow


@pytest.fixture
def fake_engine(monkeypatch):
    monkeypatch.setattr(workflow, "engine_argv", lambda *a, **kw: ["fake-vina"])


@pytest.mark.parametrize("content", [None, b""])
def test_single_run_rejects_missing_or_empty_output(tmp_path, monkeypatch, fake_engine, content):
    output = tmp_path / "pose.pdbqt"
    output.write_bytes(b"previous pose")
    def fake_run(cmd, **kwargs):
        path = Path(cmd[cmd.index("--out") + 1])
        if content is not None:
            path.write_bytes(content)
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(workflow.subprocess, "run", fake_run)
    result = workflow.run_vina("r", "l", output, (0, 0, 0), (1, 1, 1))
    assert not result.ok and result.log_file is None
    assert output.read_bytes() == b"previous pose"


def test_single_run_preserves_existing_output_on_engine_failure(tmp_path, monkeypatch, fake_engine):
    output = tmp_path / "pose.pdbqt"
    output.write_bytes(b"previous pose")
    def fail(cmd, **kwargs):
        Path(cmd[cmd.index("--out") + 1]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, cmd)
    monkeypatch.setattr(workflow.subprocess, "run", fail)
    result = workflow.run_vina("r", "l", output, (0, 0, 0), (1, 1, 1))
    assert not result.ok and output.read_bytes() == b"previous pose"


def test_single_run_promotes_only_new_nonempty_output(tmp_path, monkeypatch, fake_engine):
    output = tmp_path / "pose.pdbqt"
    output.write_bytes(b"previous pose")
    def success(cmd, **kwargs):
        path = Path(cmd[cmd.index("--out") + 1])
        assert path != output
        path.write_bytes(b"new pose")
    monkeypatch.setattr(workflow.subprocess, "run", success)
    result = workflow.run_vina("r", "l", output, (0, 0, 0), (1, 1, 1))
    assert result.ok and result.log_file == output and output.read_bytes() == b"new pose"


def test_promotion_failure_preserves_existing_output(tmp_path, monkeypatch, fake_engine):
    output = tmp_path / "pose.pdbqt"
    output.write_bytes(b"previous pose")
    def success(cmd, **kwargs):
        Path(cmd[cmd.index("--out") + 1]).write_bytes(b"new pose")
    def fail_replace(src, dst):
        raise OSError("promotion denied")
    monkeypatch.setattr(workflow.subprocess, "run", success)
    monkeypatch.setattr(workflow.os, "replace", fail_replace)
    result = workflow.run_vina("r", "l", output, (0, 0, 0), (1, 1, 1))
    assert not result.ok and "promotion denied" in result.error
    assert output.read_bytes() == b"previous pose"


def test_batch_supports_vina_12_logs_stdout_and_requires_new_pose(tmp_path, monkeypatch, fake_engine):
    config = tmp_path / "config.txt"
    config.write_text("receptor = receptor.pdbqt\n")
    (tmp_path / "receptor.pdbqt").write_text("receptor")
    ligands = tmp_path / "ligands"
    ligands.mkdir()
    (ligands / "l.pdbqt").write_text("ligand")
    out = tmp_path / "out"
    out.mkdir()
    pose, log = out / "l_out.pdbqt", out / "l.log"
    pose.write_bytes(b"old pose")
    log.write_bytes(b"old log")
    def fake_run(cmd, **kwargs):
        assert "--log" not in cmd
        Path(cmd[cmd.index("--out") + 1]).write_bytes(b"new pose")
        return subprocess.CompletedProcess(cmd, 0, b"vina output\n", b"warning\n")
    monkeypatch.setattr(workflow.subprocess, "run", fake_run)
    result = workflow.run_vina_batch(config, ligands, out_dir=out)[0]
    assert result.ok and pose.read_bytes() == b"new pose"
    assert log.read_bytes() == b"vina output\nwarning\n"


def test_batch_zero_exit_without_output_preserves_old_files(tmp_path, monkeypatch, fake_engine):
    config = tmp_path / "config.txt"
    config.write_text("receptor = receptor.pdbqt\n")
    (tmp_path / "receptor.pdbqt").write_text("receptor")
    ligands = tmp_path / "ligands"
    ligands.mkdir()
    (ligands / "l.pdbqt").write_text("ligand")
    out = tmp_path / "out"
    out.mkdir()
    (out / "l_out.pdbqt").write_bytes(b"old pose")
    (out / "l.log").write_bytes(b"old log")
    monkeypatch.setattr(workflow.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, b"", b""))
    result = workflow.run_vina_batch(config, ligands, out_dir=out)[0]
    assert not result.ok
    assert (out / "l_out.pdbqt").read_bytes() == b"old pose"
    assert (out / "l.log").read_bytes() == b"old log"
