"""Regression tests for the read-only release archive evidence generator."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "audit_release.py"


def _zip(path: Path, *, prefix: str = "", extra: dict[str, bytes] | None = None,
         bad_checksum: bool = False) -> Path:
    engines = {"autodock4.exe": b"a", "autogrid4.exe": b"b", "vina.exe": b"c"}
    lines = [f"{hashlib.md5(data, usedforsecurity=False).hexdigest()}  {name}" for name, data in engines.items()]
    if bad_checksum:
        lines[0] = "0" * 32 + "  autodock4.exe"
    entries = {
        "VERSION": b"9.9.9\n",
        "pyproject.toml": b'[project]\nversion = "9.9.9"\n',
        "src/pharmaml/__init__.py": b'__version__ = "9.9.9"\n',
        "engines/README.md": "\n".join(lines).encode(),
        "examples/data/a.csv": b"a\n1\n",
        "examples/data/nested/b.csv": b"b\n2\n",
        **{f"engines/{name}": data for name, data in engines.items()},
        **(extra or {}),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in entries.items():
            archive.writestr(prefix + name, data)
    return path


def _run(archive: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), str(archive), "--output-dir", str(output)],
                          capture_output=True, text=True, check=False)


@pytest.mark.parametrize("prefix", ["", "pharma-ml-toolkit/"])
def test_report_checks_archive_and_preserves_input(tmp_path: Path, prefix: str) -> None:
    archive = _zip(tmp_path / "pharma-ml-toolkit-v9.9.9.zip", prefix=prefix)
    original = archive.read_bytes()
    result = _run(archive, tmp_path / "report")
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "report/report.json").read_text())
    assert archive.read_bytes() == original
    assert report["archive_checks_passed"] is True
    assert report["archive"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert report["archive"]["file_count"] == 9
    assert report["example_data"]["recursive_file_count"] == 2
    assert report["example_data"]["top_level_file_count"] == 1
    assert report["checks"]["isolated_restore"]["unrelated_file_preserved"] is True
    assert all(engine["reference_matches"] and engine["execution"] == "not executed" for engine in report["engines"])
    assert [item["path"] for item in report["manifest"]] == sorted(item["path"] for item in report["manifest"])
    assert report["test_profiles"]["status"] == "not executed by this audit"
    assert "not a release certification" in (tmp_path / "report/SUMMARY.md").read_text()


def test_refuses_colliding_paths_without_extracting(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "bad.zip", extra={"Case.txt": b"a", "case.txt": b"b"})
    result = _run(archive, tmp_path / "report")
    assert result.returncode == 1
    report = json.loads((tmp_path / "report/report.json").read_text())
    assert report["checks"]["path_and_bounds"]["status"] == "failed"
    assert "collision" in report["checks"]["path_and_bounds"]["error"]
    assert report["checks"]["isolated_restore"]["status"] == "not executed"


def test_reports_excluded_files_and_bad_engine_reference(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "bad.zip", extra={"scratch/old.py": b"x"}, bad_checksum=True)
    result = _run(archive, tmp_path / "report")
    assert result.returncode == 1
    report = json.loads((tmp_path / "report/report.json").read_text())
    assert report["checks"]["excluded_paths"] == {"status": "failed", "hits": ["scratch/old.py"]}
    assert report["checks"]["engine_checksums"]["status"] == "failed"
    assert report["checks"]["isolated_restore"]["status"] == "passed"


def test_crc_failure_does_not_run_extraction(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "bad.zip")
    contents = archive.read_bytes()
    assert b"a\n1\n" in contents
    archive.write_bytes(contents.replace(b"a\n1\n", b"a\n3\n", 1))
    result = _run(archive, tmp_path / "report")
    assert result.returncode == 1
    report = json.loads((tmp_path / "report/report.json").read_text())
    assert report["checks"]["crc"]["status"] == "failed"
    assert report["checks"]["isolated_restore"]["status"] == "not executed"


def test_existing_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "source.zip")
    output = tmp_path / "report"
    output.mkdir()
    sentinel = output / "report.json"
    sentinel.write_text("keep this", encoding="utf-8")
    result = _run(archive, output)
    assert result.returncode != 0
    assert sentinel.read_text(encoding="utf-8") == "keep this"
