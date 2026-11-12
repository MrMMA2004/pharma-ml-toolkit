"""Interrupted-publication inventory is read-only and fails closed."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools/release_recovery_status.py"
spec = importlib.util.spec_from_file_location("recovery_status", TOOL)
assert spec and spec.loader
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def make_zip(path, content="release"):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("data.txt", content)


def test_partial_publication_is_inventoried_without_changes(tmp_path):
    candidate = tmp_path / "candidate.zip"
    match = tmp_path / "matching.zip"
    mismatch = tmp_path / "different.zip"
    missing = tmp_path / "missing.zip"
    make_zip(candidate)
    shutil.copyfile(candidate, match)
    make_zip(mismatch, "another release")
    before = {p: p.read_bytes() for p in (candidate, match, mismatch)}
    result = recovery.inspect(candidate, [match, mismatch, missing])
    assert [x["status"] for x in result["destinations"]] == [
        "matches", "differs: manual review", "missing"]
    assert result["candidate_crc"] == "passed"
    assert {p: p.read_bytes() for p in before} == before
    assert not missing.exists()
    run = subprocess.run([sys.executable, str(TOOL), str(candidate), str(match), str(missing)],
                         capture_output=True, text=True, check=False)
    assert run.returncode == 1
    assert '"missing"' in run.stdout


def test_invalid_candidate_and_duplicate_destinations_fail(tmp_path):
    candidate = tmp_path / "bad.zip"
    candidate.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="ZIP"):
        recovery.inspect(candidate, [tmp_path / "dest"])
    make_zip(candidate)
    with pytest.raises(ValueError, match="duplicate"):
        recovery.inspect(candidate, [tmp_path / "dest", tmp_path / "dest"])


def test_symlinks_are_not_treated_as_independent_matching_copies(tmp_path):
    candidate = tmp_path / "candidate.zip"
    make_zip(candidate)
    link = tmp_path / "link.zip"
    try:
        link.symlink_to(candidate)
    except OSError:
        pytest.skip("symlink creation unavailable")
    assert recovery.inspect(candidate, [link])["destinations"][0]["status"].startswith("symlink")
    with pytest.raises(ValueError, match="symlink"):
        recovery.inspect(link, [candidate])


def test_hardlink_is_identified(tmp_path):
    candidate = tmp_path / "candidate.zip"
    make_zip(candidate)
    link = tmp_path / "hardlink.zip"
    link.hardlink_to(candidate)
    info = recovery.inspect(candidate, [link])["destinations"][0]
    assert info["status"] == "matches" and info["same_file_as_candidate"]
