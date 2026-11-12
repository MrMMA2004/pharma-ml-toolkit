"""Recovery tests are restricted to disposable synthetic archive homes."""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
try:
    spec = importlib.util.spec_from_file_location("resume_release_under_test", TOOLS / "resume_release.py")
    assert spec and spec.loader
    recovery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recovery)
finally:
    sys.path.remove(str(TOOLS))


@pytest.fixture
def setup(tmp_path):
    source = tmp_path / "candidate.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("VERSION", "9.9.9")
    backup_dir = tmp_path / "independent"
    backup_dir.mkdir()
    backup = backup_dir / "candidate.zip"
    shutil.copyfile(source, backup)
    name = "pharma-ml-toolkit-v9.9.9.zip"
    homes = [tmp_path / f"home-{i}" for i in range(4)]
    for home in homes:
        home.mkdir()
    destinations = [home / name for home in homes]
    shutil.copyfile(source, destinations[0])
    return source, backup, recovery.digest(source), destinations


def test_plan_is_read_only_and_resume_copies_missing_only(setup):
    candidate, backup, expected, destinations = setup
    before = destinations[0].read_bytes()
    report = recovery.resume(candidate, backup, expected, destinations)
    assert not report["executed"]
    assert [i["status"] for i in report["destinations"]] == ["matches", "missing", "missing", "missing"]
    assert all(not p.exists() for p in destinations[1:])
    report = recovery.resume(candidate, backup, expected, destinations, execute=True)
    assert [i["status"] for i in report["destinations"]] == ["matches", "published", "published", "published"]
    assert all(p.read_bytes() == before for p in destinations)
    assert backup.read_bytes() == before and candidate.read_bytes() == before
    assert all(not os.path.samefile(backup, p) for p in destinations)


def test_collision_blocks_all_writes(setup):
    candidate, backup, expected, destinations = setup
    destinations[2].write_bytes(b"unique conflicting release")
    with pytest.raises(ValueError, match="unsafe destination"):
        recovery.resume(candidate, backup, expected, destinations, execute=True)
    assert destinations[2].read_bytes() == b"unique conflicting release"
    assert not destinations[1].exists() and not destinations[3].exists()


@pytest.mark.parametrize("issue", ["wrong_digest", "wrong_backup", "same_backup", "backup_in_home", "duplicate"])
def test_unsafe_preflight_refuses_publication(setup, issue):
    candidate, backup, expected, destinations = setup
    if issue == "wrong_digest":
        expected = "0" * 64
    elif issue == "wrong_backup":
        backup.write_bytes(b"not identical")
    elif issue == "same_backup":
        backup = candidate
    elif issue == "backup_in_home":
        backup = destinations[0]
    else:
        destinations = [*destinations, destinations[0]]
    with pytest.raises(ValueError):
        recovery.resume(candidate, backup, expected, destinations, execute=True)
    assert not destinations[1].exists()


def test_hardlinked_backup_is_not_independent(setup):
    candidate, backup, expected, destinations = setup
    backup.unlink()
    os.link(candidate, backup)
    with pytest.raises(ValueError, match="distinct"):
        recovery.resume(candidate, backup, expected, destinations, execute=True)
    assert not destinations[1].exists()


def test_racing_file_is_not_overwritten_and_earlier_copy_survives(setup, monkeypatch):
    candidate, backup, expected, destinations = setup
    original_link = recovery.os.link
    def racing_link(src, dest):
        if Path(dest) == destinations[2]:
            Path(dest).write_bytes(b"another writer")
        original_link(src, dest)
    monkeypatch.setattr(recovery.os, "link", racing_link)
    with pytest.raises(OSError, match="already published copies retained"):
        recovery.resume(candidate, backup, expected, destinations, execute=True)
    assert destinations[1].read_bytes() == candidate.read_bytes()
    assert destinations[2].read_bytes() == b"another writer"
    assert not destinations[3].exists()
    assert backup.read_bytes() == candidate.read_bytes()


def test_cli_defaults_to_plan(setup):
    candidate, backup, expected, destinations = setup
    cmd = [sys.executable, str(TOOLS / "resume_release.py"), "--candidate", str(candidate),
           "--backup", str(backup), "--sha256", expected, *(str(p) for p in destinations)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0 and '"executed": false' in proc.stdout
    assert not destinations[1].exists()


def test_symlink_destination_and_home_alias_are_rejected(setup):
    candidate, backup, expected, destinations = setup
    try:
        destinations[2].symlink_to(candidate)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="unsafe destination"):
        recovery.resume(candidate, backup, expected, destinations, execute=True)
    assert destinations[2].is_symlink() and not destinations[1].exists()


def test_corrupted_staged_copy_never_published(setup, monkeypatch):
    candidate, backup, expected, destinations = setup
    original = recovery.shutil.copy2
    def corrupt(source, destination):
        original(source, destination)
        Path(destination).write_bytes(b"corrupt")
    monkeypatch.setattr(recovery.shutil, "copy2", corrupt)
    with pytest.raises(OSError, match="staged copy changed"):
        recovery.resume(candidate, backup, expected, destinations, execute=True)
    assert all(not p.exists() for p in destinations[1:])
    assert recovery.digest(candidate) == recovery.digest(backup) == expected


@pytest.mark.skipif(os.name == "nt", reason="SIGKILL rehearsal requires POSIX")
def test_sigkill_then_resume_with_preserved_candidate_and_backup(tmp_path):
    """End-to-end recovery after a killed publisher; no real release paths."""
    root = tmp_path / "toolkit"
    root.mkdir()
    (root / "VERSION").write_text("9.9.9\n")
    (root / "payload.txt").write_text("synthetic payload\n")
    homes = [tmp_path / "download", root / "dist-archive", root / "download", tmp_path / "upload"]
    for home in homes:
        home.mkdir()
    independent = tmp_path / "backup"
    independent.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    predecessor = homes[0] / "pharma-ml-toolkit-v1.0.0.zip"
    predecessor.write_bytes(b"historical data")
    (independent / predecessor.name).write_bytes(b"historical data")
    script = r'''
import importlib.util, os, signal, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("test_builder", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
base = Path(sys.argv[2]); m.ROOT = base / "toolkit"; m.REPO = base
m.VERSION = "9.9.9"; m.NAME = "pharma-ml-toolkit-v9.9.9.zip"
m.HOMES = (base / "download", m.ROOT / "dist-archive", m.ROOT / "download", base / "upload")
m.OFFSITE_DIR = base / "backup"
original = m._publish_copy
def killed(src, dst, digest):
    original(src, dst, digest); os.kill(os.getpid(), signal.SIGKILL)
m._publish_copy = killed
m.main()
'''
    proc = subprocess.run([sys.executable, "-c", script, str(TOOLS / "build_zip.py"), str(tmp_path)],
                          env=dict(os.environ, TMPDIR=str(scratch)), capture_output=True, check=False)
    assert proc.returncode == -9
    paths = [home / "pharma-ml-toolkit-v9.9.9.zip" for home in homes]
    assert paths[0].is_file() and all(not p.exists() for p in paths[1:])
    backup = independent / paths[0].name
    shutil.copyfile(paths[0], backup)
    expected = recovery.digest(paths[0])
    result = recovery.resume(paths[0], backup, expected, paths, execute=True)
    assert [p["status"] for p in result["destinations"]] == ["matches", "published", "published", "published"]
    assert all(recovery.digest(p) == expected for p in paths)
    assert predecessor.read_bytes() == b"historical data"
    assert (independent / predecessor.name).read_bytes() == b"historical data"


def test_resume_is_idempotent_when_all_destinations_match(setup):
    candidate, backup, expected, destinations = setup
    recovery.resume(candidate, backup, expected, destinations, execute=True)
    before = [path.stat().st_ino for path in destinations]
    result = recovery.resume(candidate, backup, expected, destinations, execute=True)
    assert all(entry["status"] == "matches" for entry in result["destinations"])
    assert [path.stat().st_ino for path in destinations] == before


def test_backup_change_after_first_copy_stops_future_publication(setup, monkeypatch):
    candidate, backup, expected, destinations = setup
    original = recovery.os.link
    def mutate_backup_after_first(src, dst):
        original(src, dst)
        if Path(dst) == destinations[1]:
            backup.write_bytes(b"backup changed")
    monkeypatch.setattr(recovery.os, "link", mutate_backup_after_first)
    with pytest.raises(OSError, match="backup or staged copy changed"):
        recovery.resume(candidate, backup, expected, destinations, execute=True)
    assert destinations[1].read_bytes() == candidate.read_bytes()
    assert not destinations[2].exists() and not destinations[3].exists()


def test_invalid_zip_is_never_published(setup):
    candidate, backup, _, destinations = setup
    candidate.write_bytes(b"not an archive")
    backup.write_bytes(b"not an archive")
    expected = recovery.digest(candidate)
    with pytest.raises(ValueError, match="ZIP"):
        recovery.resume(candidate, backup, expected, destinations, execute=True)
    assert not destinations[1].exists()
