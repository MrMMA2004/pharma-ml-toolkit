"""No-clobber checkpoint ZIP staging; all files here are synthetic fixtures."""
from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    tool = Path(__file__).resolve().parents[1] / "tools" / "checkpoint.py"
    spec = importlib.util.spec_from_file_location("checkpoint_build_safety", tool)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    source = tmp_path / "toolkit"
    source.mkdir()
    (source / "VERSION").write_text("9.9.9\n")
    # r295 Windows honesty: byte-asserted fixture written as raw bytes
    # (text-mode write would CRLF-translate on the Windows runner while
    # the builder copies files verbatim into the ZIP).
    (source / "example.txt").write_bytes(b"payload\n")
    monkeypatch.setattr(module, "ROOT", source)
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "DOWNLOAD", tmp_path / "download")
    return module


def test_existing_release_is_not_overwritten(checkpoint, tmp_path):
    folder = tmp_path / "download"
    folder.mkdir()
    target = folder / "pharma-ml-toolkit-v9.9.9.zip"
    target.write_bytes(b"original release bytes")
    assert checkpoint._build_zip("9.9.9") is None
    assert target.read_bytes() == b"original release bytes"


def test_existing_staging_directory_is_not_deleted(checkpoint, tmp_path):
    stage = tmp_path / ".zip-staging-9.9.9"
    stage.mkdir()
    sentinel = stage / "user-file"
    sentinel.write_text("keep")
    result = checkpoint._build_zip("9.9.9")
    assert sentinel.read_text() == "keep"
    assert result is not None
    with zipfile.ZipFile(result) as archive:
        assert archive.read("example.txt") == b"payload\n"
        assert archive.testzip() is None


def test_broken_symlink_at_target_is_not_overwritten(checkpoint, tmp_path):
    folder = tmp_path / "download"
    folder.mkdir()
    target = folder / "pharma-ml-toolkit-v9.9.9.zip"
    target.symlink_to(folder / "missing")
    assert checkpoint._build_zip("9.9.9") is None
    assert target.is_symlink()


def test_invalid_version_does_not_create_files(checkpoint, tmp_path):
    assert checkpoint._build_zip("../other") is None
    assert sorted(p.name for p in tmp_path.iterdir()) == ["toolkit"]


def test_concurrent_target_creation_is_not_overwritten(checkpoint, tmp_path, monkeypatch):
    target = tmp_path / "download" / "pharma-ml-toolkit-v9.9.9.zip"
    original_link = checkpoint.os.link

    def racing_link(source, destination):
        target.write_bytes(b"concurrent creator")
        original_link(source, destination)

    monkeypatch.setattr(checkpoint.os, "link", racing_link)
    assert checkpoint._build_zip("9.9.9") is None
    assert target.read_bytes() == b"concurrent creator"
    assert sorted(p.name for p in target.parent.iterdir()) == [target.name]


def test_failed_integrity_check_does_not_publish(checkpoint, tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint.zipfile.ZipFile, "testzip", lambda self: "bad.txt")
    assert checkpoint._build_zip("9.9.9") is None
    assert not (tmp_path / "download" / "pharma-ml-toolkit-v9.9.9.zip").exists()
    assert list((tmp_path / "download").iterdir()) == []


def test_link_failure_does_not_publish_or_delete_other_files(checkpoint, tmp_path, monkeypatch):
    folder = tmp_path / "download"
    folder.mkdir()
    sentinel = folder / "other.txt"
    sentinel.write_text("keep")
    def fail(*_args):
        raise OSError("hard links unavailable")
    monkeypatch.setattr(checkpoint.os, "link", fail)
    assert checkpoint._build_zip("9.9.9") is None
    assert sentinel.read_text() == "keep"
    assert sorted(p.name for p in folder.iterdir()) == [sentinel.name]


def test_output_directory_inside_toolkit_is_not_nested_in_zip(checkpoint, tmp_path, monkeypatch):
    output = tmp_path / "toolkit" / "release-output"
    monkeypatch.setattr(checkpoint, "DOWNLOAD", output)
    out = checkpoint._build_zip("9.9.9")
    assert out is not None
    with zipfile.ZipFile(out) as archive:
        assert archive.read("example.txt") == b"payload\n"
        assert not any(".checkpoint-zip-" in name or name.endswith(".zip")
                       for name in archive.namelist())


def test_unreadable_source_tree_does_not_publish_partial_zip(checkpoint, tmp_path, monkeypatch):
    # r295 Windows honesty: full os.walk signature + source-root gating
    # (the mirror of the build_zip guard) -- the Windows rmtree cleanup
    # walk passes keyword arguments a keyword-only fake cannot accept,
    # and the staging dir lives under DOWNLOAD, so every non-source walk
    # delegates to the genuine os.walk and cleanup still succeeds.
    real_walk = checkpoint.os.walk

    def fail_walk(top, topdown=True, onerror=None, followlinks=False):
        if Path(top) == checkpoint.ROOT:
            assert onerror is not None  # the production call always passes one
            onerror(PermissionError("cannot read source"))
        else:
            yield from real_walk(
                top, topdown=topdown, onerror=onerror, followlinks=followlinks
            )

    monkeypatch.setattr(checkpoint.os, "walk", fail_walk)
    assert checkpoint._build_zip("9.9.9") is None
    assert list((tmp_path / "download").iterdir()) == []
