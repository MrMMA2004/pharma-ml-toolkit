"""Four-home publication tests with synthetic files, never actual releases."""
from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def publisher(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "tools" / "build_zip.py"
    spec = importlib.util.spec_from_file_location("zip_publisher_under_test", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    root = tmp_path / "toolkit"
    root.mkdir()
    (root / "VERSION").write_text("9.9.9\n")
    (root / "data.txt").write_text("original\n")
    homes = (tmp_path / "download", root / "dist-archive", root / "download", tmp_path / "upload")
    for folder in homes:
        folder.mkdir()
    backup = tmp_path / "safe-backup"
    backup.mkdir()
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "VERSION", "9.9.9")
    monkeypatch.setattr(module, "NAME", "pharma-ml-toolkit-v9.9.9.zip")
    monkeypatch.setattr(module, "HOMES", homes)
    monkeypatch.setattr(module, "OFFSITE_DIR", backup, raising=False)
    return module


def test_existing_destination_is_preserved(publisher):
    target = publisher.HOMES[1] / publisher.NAME
    target.write_bytes(b"existing unique release")
    assert publisher.main() != 0
    assert target.read_bytes() == b"existing unique release"
    assert all(not (home / publisher.NAME).exists() for home in publisher.HOMES if home != target.parent)


def test_existing_old_root_build_path_is_not_deleted(publisher):
    target = publisher.REPO / publisher.NAME
    target.write_bytes(b"unrelated file")
    assert publisher.main() == 0
    assert target.read_bytes() == b"unrelated file"


def test_predecessor_only_copy_is_retained(publisher):
    old = publisher.HOMES[0] / "pharma-ml-toolkit-v1.0.0.zip"
    old.write_bytes(b"unique predecessor")
    assert publisher.main() == 0
    assert old.read_bytes() == b"unique predecessor"


def test_scrub_operates_on_staged_copy_not_source(publisher):
    tools = publisher.ROOT / "tools"
    tools.mkdir()
    (tools / "scrub_home_paths.py").write_text(
        'from pathlib import Path\n'
        'root = Path(__file__).resolve().parents[1]\n'
        '(root / "data.txt").write_text("scrubbed\\n")\n'
    )
    assert publisher.main() == 0
    assert (publisher.ROOT / "data.txt").read_text() == "original\n"
    with zipfile.ZipFile(publisher.HOMES[0] / publisher.NAME) as archive:
        assert archive.read("data.txt") == b"scrubbed\n"


def test_scrub_failure_does_not_publish(publisher):
    tools = publisher.ROOT / "tools"
    tools.mkdir()
    (tools / "scrub_home_paths.py").write_text("raise SystemExit(1)\n")
    assert publisher.main() != 0
    assert all(not (home / publisher.NAME).exists() for home in publisher.HOMES)


def test_successful_release_has_identical_verified_copies(publisher):
    (publisher.ROOT / "scratch").mkdir()
    (publisher.ROOT / "scratch" / "secret.txt").write_text("excluded")
    assert publisher.main() == 0
    copies = [(home / publisher.NAME).read_bytes() for home in publisher.HOMES]
    assert all(copy == copies[0] for copy in copies)
    with zipfile.ZipFile(publisher.HOMES[0] / publisher.NAME) as archive:
        assert archive.testzip() is None
        assert archive.read("data.txt") == b"original\n"
        assert not any(name.startswith("scratch/") for name in archive.namelist())


def test_matching_independent_backup_allows_retention(publisher):
    name = "pharma-ml-toolkit-v1.0.0.zip"
    for home in publisher.HOMES:
        (home / name).write_bytes(b"old release")
    (publisher.OFFSITE_DIR / name).write_bytes(b"old release")
    assert publisher.main() == 0
    assert all(not (home / name).exists() for home in publisher.HOMES)
    assert (publisher.OFFSITE_DIR / name).read_bytes() == b"old release"


def test_mismatched_backup_never_authorizes_removal(publisher):
    name = "pharma-ml-toolkit-v1.0.0.zip"
    old = publisher.HOMES[0] / name
    old.write_bytes(b"release A")
    (publisher.OFFSITE_DIR / name).write_bytes(b"release B")
    assert publisher.main() == 0
    assert old.read_bytes() == b"release A"


@pytest.mark.parametrize("alias", ["hardlink", "symlink", "home"])
def test_aliased_backup_cannot_authorize_removal(publisher, monkeypatch, alias):
    name = "pharma-ml-toolkit-v1.0.0.zip"
    old = publisher.HOMES[0] / name
    old.write_bytes(b"old release")
    if alias == "hardlink":
        publisher.os.link(old, publisher.OFFSITE_DIR / name)
    elif alias == "symlink":
        (publisher.OFFSITE_DIR / name).symlink_to(old)
    else:
        monkeypatch.setattr(publisher, "OFFSITE_DIR", publisher.HOMES[0])
    assert publisher.main() == 0
    assert old.read_bytes() == b"old release"


def test_later_copy_failure_keeps_partial_copies_and_predecessor(publisher, monkeypatch):
    old = publisher.HOMES[0] / "pharma-ml-toolkit-v1.0.0.zip"
    old.write_bytes(b"predecessor")
    original = publisher._publish_copy
    def fail_second(source, destination, digest):
        if destination.parent == publisher.HOMES[1]:
            raise OSError("injected copy failure")
        original(source, destination, digest)
    monkeypatch.setattr(publisher, "_publish_copy", fail_second)
    assert publisher.main() != 0
    assert (publisher.HOMES[0] / publisher.NAME).is_file()
    assert all(not (home / publisher.NAME).exists() for home in publisher.HOMES[1:])
    assert old.read_bytes() == b"predecessor"


def test_concurrent_destination_creation_is_not_replaced(publisher, monkeypatch):
    destination = publisher.HOMES[1] / publisher.NAME
    original_link = publisher.os.link
    def race(source, target):
        if target == destination:
            destination.write_bytes(b"other writer")
        original_link(source, target)
    monkeypatch.setattr(publisher.os, "link", race)
    assert publisher.main() != 0
    assert destination.read_bytes() == b"other writer"


def test_broken_destination_symlink_is_not_replaced(publisher):
    target = publisher.HOMES[0] / publisher.NAME
    target.symlink_to(target.parent / "missing")
    assert publisher.main() != 0
    assert target.is_symlink()


def test_corrupt_copy_is_not_published(publisher, monkeypatch):
    original = publisher.shutil.copy2
    def corrupt_copy(source, destination):
        original(source, destination)
        if Path(destination).name == publisher.NAME:
            Path(destination).write_bytes(b"corrupt")
    monkeypatch.setattr(publisher.shutil, "copy2", corrupt_copy)
    assert publisher.main() != 0
    assert all(not (home / publisher.NAME).exists() for home in publisher.HOMES)


def test_source_walk_error_does_not_publish(publisher, monkeypatch):
    def fail_walk(_root, *, onerror):
        onerror(PermissionError("source unreadable"))
        yield  # pragma: no cover
    monkeypatch.setattr(publisher.os, "walk", fail_walk)
    assert publisher.main() != 0
    assert all(not (home / publisher.NAME).exists() for home in publisher.HOMES)


def test_source_symlinks_are_rejected(publisher, tmp_path):
    (publisher.ROOT / "link.txt").symlink_to(tmp_path / "elsewhere")
    assert publisher.main() != 0
    assert all(not (home / publisher.NAME).exists() for home in publisher.HOMES)


def test_invalid_version_and_duplicate_homes_are_rejected(publisher, monkeypatch):
    monkeypatch.setattr(publisher, "VERSION", "../bad")
    assert publisher.main() != 0
    monkeypatch.setattr(publisher, "VERSION", "9.9.9")
    monkeypatch.setattr(publisher, "HOMES", (publisher.HOMES[0], publisher.HOMES[0]))
    assert publisher.main() != 0


def test_source_copy_corruption_is_not_published(publisher, monkeypatch):
    original = publisher.shutil.copy2
    def corrupt(source, destination):
        original(source, destination)
        if Path(destination).name == "data.txt":
            Path(destination).write_bytes(b"corrupt staged data")
    monkeypatch.setattr(publisher.shutil, "copy2", corrupt)
    assert publisher.main() != 0
    assert (publisher.ROOT / "data.txt").read_text() == "original\n"
    assert all(not (home / publisher.NAME).exists() for home in publisher.HOMES)


def test_archive_leak_failure_prevents_publication(publisher):
    (publisher.ROOT / "data.txt").write_text("/ho" + "me/z/private")
    assert publisher.main() != 0
    assert all(not (home / publisher.NAME).exists() for home in publisher.HOMES)


def test_corruption_after_publication_prevents_retention(publisher, monkeypatch):
    old = publisher.HOMES[0] / "pharma-ml-toolkit-v1.0.0.zip"
    old.write_bytes(b"old")
    (publisher.OFFSITE_DIR / old.name).write_bytes(b"old")
    original = publisher._publish_copy
    def corrupt_first_after_last(source, destination, digest):
        original(source, destination, digest)
        if destination.parent == publisher.HOMES[-1]:
            (publisher.HOMES[0] / publisher.NAME).write_bytes(b"changed")
    monkeypatch.setattr(publisher, "_publish_copy", corrupt_first_after_last)
    assert publisher.main() != 0
    assert old.read_bytes() == b"old"
    assert all((home / publisher.NAME).is_file() for home in publisher.HOMES)


@pytest.mark.skipif(__import__("os").name == "nt", reason="SIGKILL rehearsal requires POSIX")
def test_sigkill_after_first_publication_preserves_partial_and_predecessor(publisher, tmp_path):
    """Real subprocess death after publication, within disposable fake homes only."""
    import os
    import subprocess
    import sys

    name = "pharma-ml-toolkit-v1.0.0.zip"
    old = publisher.HOMES[0] / name
    old.write_bytes(b"historical release")
    (publisher.OFFSITE_DIR / name).write_bytes(b"historical release")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    script = r'''
import importlib.util, os, signal, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("publisher_rehearsal", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
base = Path(sys.argv[2]); m.ROOT = base / "toolkit"; m.REPO = base
m.VERSION = "9.9.9"; m.NAME = "pharma-ml-toolkit-v9.9.9.zip"
m.HOMES = (base / "download", m.ROOT / "dist-archive", m.ROOT / "download", base / "upload")
m.OFFSITE_DIR = base / "safe-backup"
if sys.argv[3] == "kill":
    original = m._publish_copy
    def kill_after_publish(src, dst, digest):
        original(src, dst, digest)
        os.kill(os.getpid(), signal.SIGKILL)
    m._publish_copy = kill_after_publish
raise SystemExit(m.main())
'''
    tool = Path(__file__).resolve().parents[1] / "tools/build_zip.py"
    env = dict(os.environ, TMPDIR=str(scratch))
    killed = subprocess.run([sys.executable, "-c", script, str(tool), str(tmp_path), "kill"],
                            env=env, capture_output=True, check=False)
    assert killed.returncode == -9
    partial = publisher.HOMES[0] / publisher.NAME
    assert partial.is_file()
    with zipfile.ZipFile(partial) as archive:
        assert archive.testzip() is None
        assert archive.read("data.txt") == b"original\n"
    assert all(not (home / publisher.NAME).exists() for home in publisher.HOMES[1:])
    assert old.read_bytes() == b"historical release"
    assert (publisher.OFFSITE_DIR / name).read_bytes() == b"historical release"
    before = partial.read_bytes()
    retry = subprocess.run([sys.executable, "-c", script, str(tool), str(tmp_path), "retry"],
                           env=env, capture_output=True, check=False)
    assert retry.returncode != 0 and b"refusing to overwrite" in retry.stdout
    assert partial.read_bytes() == before and old.read_bytes() == b"historical release"
    assert all(not (home / publisher.NAME).exists() for home in publisher.HOMES[1:])
