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
