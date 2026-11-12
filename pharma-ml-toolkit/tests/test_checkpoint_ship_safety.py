"""Isolated publication/retention tests; no real release or Git mutation."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def shipping(tmp_path, monkeypatch):
    tool = Path(__file__).resolve().parents[1] / "tools" / "checkpoint.py"
    spec = importlib.util.spec_from_file_location("ship_safety_fixture", tool)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    root = tmp_path / "toolkit"
    root.mkdir()
    pyproject = root / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "9.9.9"\n')
    init = root / "__init__.py"
    init.write_text('__version__ = "9.9.9"\n')
    download = tmp_path / "download"
    archive_dir = root / "dist-archive"
    offsite = tmp_path / "backup"
    download.mkdir()
    archive_dir.mkdir()
    offsite.mkdir()
    for name, value in (("ROOT", root), ("REPO", tmp_path), ("PYPROJECT", pyproject),
                        ("INIT_PY", init), ("DOWNLOAD", download),
                        ("ARCHIVE_DIR", archive_dir), ("OFFSITE_DIR", offsite)):
        monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(module, "doctor", lambda **_: 0)
    monkeypatch.setattr(module, "_version", lambda: "9.9.9")
    monkeypatch.setattr(module, "_notebook_freshness_problems", list)
    monkeypatch.setattr(module, "_ship_gate_commands", list)
    commands = []
    monkeypatch.setattr(module, "_git", lambda *args, **kwargs: commands.append(args))
    def fake_build(version):
        target = download / f"pharma-ml-toolkit-v{version}.zip"
        target.write_bytes(b"new archive")
        return target
    monkeypatch.setattr(module, "_build_zip", fake_build)
    return module, commands


def test_existing_safe_copy_never_overwritten(shipping):
    cp, commands = shipping
    name = "pharma-ml-toolkit-v9.9.9.zip"
    existing = cp.ARCHIVE_DIR / name
    existing.write_bytes(b"previous unique release")
    assert cp.ship("test") != 0
    assert existing.read_bytes() == b"previous unique release"
    assert not commands


def test_failed_backup_aborts_before_retention_or_commit(shipping, monkeypatch):
    cp, commands = shipping
    name = "pharma-ml-toolkit-v1.0.0.zip"
    old = cp.DOWNLOAD / name
    old.write_bytes(b"unique historical release")
    monkeypatch.setattr(cp, "KEEP_ZIPS", 1)
    def fail(*_args, **_kwargs):
        raise OSError("injected backup failure")
    monkeypatch.setattr(cp.shutil, "copy2", fail)
    assert cp.ship("test") != 0
    assert old.read_bytes() == b"unique historical release"
    assert not commands


def test_retention_preserves_only_copy(shipping, monkeypatch):
    cp, commands = shipping
    old = cp.DOWNLOAD / "pharma-ml-toolkit-v1.0.0.zip"
    old.write_bytes(b"unique historical release")
    monkeypatch.setattr(cp, "KEEP_ZIPS", 1)
    assert cp.ship("test") == 0
    assert old.read_bytes() == b"unique historical release"
    assert not any(args[:2] == ("rm", "-q") for args in commands)


def test_retention_requires_matching_independent_backup(shipping, monkeypatch):
    cp, commands = shipping
    name = "pharma-ml-toolkit-v1.0.0.zip"
    old = cp.DOWNLOAD / name
    old.write_bytes(b"release A")
    (cp.OFFSITE_DIR / name).write_bytes(b"different release")
    monkeypatch.setattr(cp, "KEEP_ZIPS", 1)
    assert cp.ship("test") == 0
    assert old.read_bytes() == b"release A"
    assert not any(args[:2] == ("rm", "-q") for args in commands)


def test_successful_copies_and_retention_with_verified_backup(shipping, monkeypatch):
    cp, commands = shipping
    name = "pharma-ml-toolkit-v1.0.0.zip"
    old = cp.DOWNLOAD / name
    old.write_bytes(b"historical bytes")
    saved = cp.OFFSITE_DIR / name
    saved.write_bytes(b"historical bytes")
    monkeypatch.setattr(cp, "KEEP_ZIPS", 1)
    assert cp.ship("test") == 0
    assert not old.exists()
    assert saved.read_bytes() == b"historical bytes"
    newname = "pharma-ml-toolkit-v9.9.9.zip"
    assert (cp.ARCHIVE_DIR / newname).read_bytes() == b"new archive"
    assert (cp.OFFSITE_DIR / newname).read_bytes() == b"new archive"
    assert any(args[:2] == ("rm", "-q") for args in commands)


@pytest.mark.parametrize("alias", ["hardlink", "symlink"])
def test_retention_does_not_accept_aliased_backup(shipping, monkeypatch, alias):
    cp, commands = shipping
    name = "pharma-ml-toolkit-v1.0.0.zip"
    old = cp.DOWNLOAD / name
    old.write_bytes(b"historical bytes")
    backup = cp.OFFSITE_DIR / name
    if alias == "hardlink":
        cp.os.link(old, backup)
    else:
        backup.symlink_to(old)
    monkeypatch.setattr(cp, "KEEP_ZIPS", 1)
    assert cp.ship("test") == 0
    assert old.read_bytes() == b"historical bytes"
    assert not any(args[:2] == ("rm", "-q") for args in commands)


def test_corrupted_backup_is_not_published(shipping, monkeypatch):
    cp, commands = shipping
    original = cp.shutil.copy2
    def corrupt(src, dst):
        original(src, dst)
        Path(dst).write_bytes(b"corrupted copy")
    monkeypatch.setattr(cp.shutil, "copy2", corrupt)
    assert cp.ship("test") != 0
    assert not (cp.ARCHIVE_DIR / "pharma-ml-toolkit-v9.9.9.zip").exists()
    assert list(cp.ARCHIVE_DIR.iterdir()) == []
    assert not commands


def test_racing_backup_creation_preserved(shipping, monkeypatch):
    cp, commands = shipping
    original_link = cp.os.link
    destination = cp.ARCHIVE_DIR / "pharma-ml-toolkit-v9.9.9.zip"
    def race(src, dst):
        destination.write_bytes(b"another writer")
        original_link(src, dst)
    monkeypatch.setattr(cp.os, "link", race)
    assert cp.ship("test") != 0
    assert destination.read_bytes() == b"another writer"
    assert not commands


def test_second_backup_failure_keeps_first_verified_copy(shipping, monkeypatch):
    cp, commands = shipping
    original_copy = cp._copy_release_exclusive
    def fail_second(source, destination):
        if destination.parent == cp.OFFSITE_DIR:
            raise OSError("backup disk unavailable")
        original_copy(source, destination)
    monkeypatch.setattr(cp, "_copy_release_exclusive", fail_second)
    assert cp.ship("test") != 0
    name = "pharma-ml-toolkit-v9.9.9.zip"
    assert (cp.DOWNLOAD / name).read_bytes() == b"new archive"
    assert (cp.ARCHIVE_DIR / name).read_bytes() == b"new archive"
    assert not (cp.OFFSITE_DIR / name).exists()
    assert not commands


def test_copy_changed_before_final_verification_stops_ship(shipping, monkeypatch):
    cp, commands = shipping
    original_copy = cp._copy_release_exclusive
    def alter_first_after_second(source, destination):
        original_copy(source, destination)
        if destination.parent == cp.OFFSITE_DIR:
            (cp.ARCHIVE_DIR / source.name).write_bytes(b"changed after copy")
    monkeypatch.setattr(cp, "_copy_release_exclusive", alter_first_after_second)
    assert cp.ship("test") != 0
    assert not commands


def test_existing_output_detected_before_version_changes_and_build(shipping, monkeypatch):
    cp, commands = shipping
    name = "pharma-ml-toolkit-v9.9.9.zip"
    (cp.OFFSITE_DIR / name).write_bytes(b"keep")
    cp.PYPROJECT.write_text('[project]\nversion = "9.9.8"\n')
    called = []
    monkeypatch.setattr(cp, "_build_zip", lambda version: called.append(version))
    assert cp.ship("test") != 0
    assert cp.PYPROJECT.read_text() == '[project]\nversion = "9.9.8"\n'
    assert not called and not commands


def test_invalid_version_is_rejected_before_any_writes(shipping, monkeypatch):
    cp, commands = shipping
    monkeypatch.setattr(cp, "_version", lambda: "../bad")
    before = cp.PYPROJECT.read_bytes()
    assert cp.ship("test") != 0
    assert cp.PYPROJECT.read_bytes() == before
    assert not list(cp.DOWNLOAD.iterdir())
    assert not commands


def test_safe_copy_archive_retention_requires_matching_backup(shipping, monkeypatch):
    cp, commands = shipping
    old = cp.ARCHIVE_DIR / "pharma-ml-toolkit-v1.0.0.zip"
    old.write_bytes(b"only safe copy")
    monkeypatch.setattr(cp, "KEEP_ARCHIVE", 1)
    assert cp.ship("test") == 0
    assert old.read_bytes() == b"only safe copy"
    assert not any(args[:2] == ("rm", "-q") for args in commands)
