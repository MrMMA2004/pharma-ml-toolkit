"""Engine discovery and PE validation; subprocesses are mocked, not engines."""
from __future__ import annotations

import hashlib
import os
import struct
import subprocess
from pathlib import Path

import pytest

from pharmaml.docking import workflow
from pharmaml.exceptions import DockingError


def pe(machine: int) -> bytes:
    data = bytearray(128)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 64)
    data[64:68] = b"PE\0\0"
    struct.pack_into("<H", data, 68, machine)
    return bytes(data)


@pytest.fixture
def bundled(tmp_path, monkeypatch):
    folder = tmp_path / "engines"
    folder.mkdir()
    body = pe(0x8664)
    (folder / "vina.exe").write_bytes(body)
    monkeypatch.setattr(workflow, "_bundled_engine_dirs", lambda: (folder,))
    monkeypatch.setitem(workflow._BUNDLED_ENGINE_SHA256, "vina.exe", hashlib.sha256(body).hexdigest())
    return folder


def test_explicit_directory_takes_priority_over_path_and_bundle(tmp_path, monkeypatch, bundled):
    chosen = tmp_path / "custom"
    chosen.mkdir()
    (chosen / "vina").write_bytes(b"native custom")
    monkeypatch.setattr(workflow.shutil, "which", lambda name: "/native/vina" if name == "vina" else None)
    assert workflow.locate_engine_binary("vina", chosen) == chosen / "vina"


def test_path_has_priority_over_bundled_copy(monkeypatch, bundled):
    monkeypatch.setattr(workflow.shutil, "which", lambda name: "/native/vina" if name == "vina" else None)
    assert workflow.locate_engine_binary("vina") == Path("/native/vina")
    assert workflow.engine_argv("vina") == ["/native/vina"]


def test_bundled_fallback_is_verified_and_uses_wine_on_unix(monkeypatch, bundled):
    monkeypatch.setattr(workflow, "_host_platform", lambda: ("posix", "x86_64"))
    monkeypatch.setattr(workflow.shutil, "which", lambda name: "/usr/bin/wine" if name == "wine" else None)
    assert workflow.locate_engine_binary("vina") == bundled / "vina.exe"
    assert workflow.engine_argv("vina") == ["/usr/bin/wine", str(bundled / "vina.exe")]


def test_bundle_is_not_a_runnable_unix_engine_without_wine(monkeypatch, bundled):
    monkeypatch.setattr(workflow, "_host_platform", lambda: ("posix", "x86_64"))
    monkeypatch.setattr(workflow.shutil, "which", lambda name: None)
    # Discovery alone means presence, not runtime availability.
    assert workflow.locate_engine_binary("vina") == bundled / "vina.exe"
    with pytest.raises(DockingError, match="wine"):
        workflow.engine_argv("vina")


def test_corrupt_bundled_binary_fails_instead_of_launching(monkeypatch, bundled):
    monkeypatch.setattr(workflow.shutil, "which", lambda name: None)
    (bundled / "vina.exe").write_bytes(pe(0x8664) + b"changed")
    with pytest.raises(DockingError, match="checksum"):
        workflow.engine_argv("vina")


def test_unknown_custom_engine_does_not_resolve_a_bundled_name(monkeypatch, bundled):
    monkeypatch.setattr(workflow.shutil, "which", lambda name: None)
    assert workflow.locate_engine_binary("custom") is None


@pytest.mark.parametrize("machine", [0x14C, 0x8664])
def test_pe_architecture_is_detected(machine, tmp_path):
    path = tmp_path / "renamed"
    path.write_bytes(pe(machine))
    assert workflow._pe_machine(path) == machine


def test_malformed_or_unknown_pe_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(workflow, "_host_platform", lambda: ("posix", "x86_64"))
    for name, body in [("truncated", b"MZ"), ("other", pe(0xAA64))]:
        path = tmp_path / name
        path.write_bytes(body)
        with pytest.raises(DockingError, match="PE|architecture"):
            workflow._pe_argv(path)


def test_windows_x86_rejects_x64_engine_before_invocation(monkeypatch, tmp_path):
    monkeypatch.setattr(workflow, "_host_platform", lambda: ("nt", "x86"))
    exe = tmp_path / "engine.exe"
    exe.write_bytes(pe(0x8664))
    with pytest.raises(DockingError, match="64-bit"):
        workflow._pe_argv(exe)
    exe.write_bytes(pe(0x14C))
    assert workflow._pe_argv(exe) == [str(exe)]


def test_arm_host_is_not_automatically_assumed_to_run_x86_pe(monkeypatch, tmp_path):
    monkeypatch.setattr(workflow, "_host_platform", lambda: ("nt", "arm64"))
    exe = tmp_path / "engine.exe"
    exe.write_bytes(pe(0x14C))
    with pytest.raises(DockingError, match="architecture"):
        workflow._pe_argv(exe)


def test_path_exe_is_used_without_engine_folder(tmp_path, monkeypatch):
    exe = tmp_path / "vina.exe"
    exe.write_bytes(pe(0x8664))
    monkeypatch.setattr(workflow, "_host_platform", lambda: ("posix", "x86_64"))
    monkeypatch.setattr(workflow.shutil, "which", lambda name: {
        "vina.exe": str(exe), "wine": "/usr/bin/wine"
    }.get(name))
    assert workflow.engine_argv("vina") == ["/usr/bin/wine", str(exe)]


def test_autodock_relative_engine_directory_survives_changed_subprocess_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engines = tmp_path / "engines"
    engines.mkdir()
    for name in ("autogrid4", "autodock4"):
        (engines / name).write_text("native stub")
    work = tmp_path / "work"
    work.mkdir()
    (work / "r.gpf").write_text("grid")
    (work / "r.dpf").write_text("dock")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        assert Path(cmd[0]).is_absolute()
        if "autodock4" in cmd[0]:
            (work / "r.dlg").write_text("Successful Completion\n")

    monkeypatch.setattr(workflow.subprocess, "run", fake_run)
    result = workflow.run_autodock4("work", engine_dir="engines")
    assert result.ok and len(calls) == 2
    assert all(call[1]["cwd"] == Path("work") for call in calls)


def test_relative_path_engine_is_resolved_before_changing_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "bin"
    folder.mkdir()
    # r295 Windows honesty: CPython's shutil.which() consults PATHEXT on
    # Windows and an extensionless stub is invisible with the default
    # X_OK mode (the direct match is only honored for names already
    # carrying a PATHEXT extension) -- so the fake engine carries the
    # platform's own spelling and the assertion follows it. POSIX keeps
    # the classic extensionless stub; no execution happens either way.
    exe = folder / ("vina.exe" if os.name == "nt" else "vina")
    exe.write_text("stub")
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", "bin")
    assert workflow.engine_argv("vina") == [str(exe.resolve())]


def test_batch_passes_absolute_paths_with_config_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "r.pdbqt").write_text("receptor")
    (tmp_path / "config" / "vina.conf").write_text("receptor = r.pdbqt\n")
    (tmp_path / "ligands").mkdir()
    (tmp_path / "ligands" / "a.pdbqt").write_text("ligand")
    monkeypatch.setattr(workflow, "_require_executable", lambda _: "/fake/vina")
    commands = []
    def fake_run(cmd, **kw):
        commands.append((cmd, kw))
        Path(cmd[cmd.index("--out") + 1]).write_text("pose")
        return subprocess.CompletedProcess(cmd, 0, b"log", b"")
    monkeypatch.setattr(workflow.subprocess, "run", fake_run)
    result = workflow.run_vina_batch("config/vina.conf", "ligands", out_dir="output")
    assert len(result) == len(commands) == 1
    cmd, kwargs = commands[0]
    assert kwargs["cwd"] == Path("config")
    assert result[0].ok and "--log" not in cmd
    for flag in ("--config", "--ligand", "--out"):
        assert Path(cmd[cmd.index(flag) + 1]).is_absolute()


def test_oserror_is_reported_as_failed_engine_run(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "_require_executable", lambda _: "/fake/vina")
    def fail(*args, **kwargs):
        raise PermissionError("engine launch denied")
    monkeypatch.setattr(workflow.subprocess, "run", fail)
    result = workflow.run_vina("r", "l", tmp_path / "out", (0, 0, 0), (1, 1, 1))
    assert not result.ok and "engine launch denied" in result.error


def test_smiles_preflight_reports_missing_wine_before_preparation(tmp_path, monkeypatch, bundled):
    from pharmaml.docking.smiles_docking import dock_smiles_array
    monkeypatch.setattr(workflow, "_host_platform", lambda: ("posix", "x86_64"))
    monkeypatch.setattr(workflow.shutil, "which", lambda _: None)
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("receptor")
    with pytest.raises(DockingError, match="wine"):
        dock_smiles_array(["CCO"], receptor, tmp_path / "out", engine="vina")
    assert not (tmp_path / "out").exists()


def test_shipped_binaries_match_hashes_and_machine_types():
    folders = workflow._bundled_engine_dirs()
    assert folders
    # Only test the actual source/standalone resource when it exists; wheels
    # need not ship these binaries.
    folder = next((item for item in folders if (item / "vina.exe").is_file()), None)
    if folder is None:
        pytest.skip("no bundled engine resources in this installation")
    for name, expected in workflow._BUNDLED_ENGINE_SHA256.items():
        path = folder / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
        assert workflow._pe_machine(path) == workflow._BUNDLED_ENGINE_MACHINE[name]


def test_explicit_binary_path_precedes_engine_folder(tmp_path, monkeypatch, bundled):
    binary = tmp_path / "chosen.exe"
    binary.write_bytes(pe(0x14C))
    monkeypatch.setattr(workflow, "_host_platform", lambda: ("nt", "amd64"))
    assert workflow.engine_argv(str(binary), bundled) == [str(binary)]


def test_package_resource_lookup_does_not_guess_site_packages_sibling(tmp_path, monkeypatch):
    module = tmp_path / "site-packages" / "pharmaml" / "docking" / "workflow.py"
    module.parent.mkdir(parents=True)
    module.write_text("")
    monkeypatch.setattr(workflow, "__file__", str(module))
    assert workflow._bundled_engine_dirs() == (module.parents[1] / "engines",)


def test_autodock_launch_oserror_returns_failure(tmp_path, monkeypatch):
    (tmp_path / "r.gpf").write_text("grid")
    monkeypatch.setattr(workflow, "_require_executable", lambda _: "/fake/autogrid4")
    def fail(*args, **kwargs):
        raise PermissionError("denied")
    monkeypatch.setattr(workflow.subprocess, "run", fail)
    result = workflow.run_autodock4(tmp_path)
    assert not result.ok and "denied" in result.error


@pytest.mark.parametrize("failure", [
    subprocess.CalledProcessError(4, "vina"),
    subprocess.TimeoutExpired("vina", 1),
])
def test_vina_nonzero_exit_and_timeout_are_failed_runs(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(workflow, "_require_executable", lambda _: "/fake/vina")
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(workflow.subprocess, "run", fail)
    result = workflow.run_vina("r", "l", tmp_path / "out", (0, 0, 0), (1, 1, 1))
    assert not result.ok and result.log_file is None and result.error
