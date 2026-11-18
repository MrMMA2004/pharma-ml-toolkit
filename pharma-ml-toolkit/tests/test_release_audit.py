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


# -- part5-continuation round 5 (§14): --profile-results evidence folding --------


def _valid_profile_results() -> dict:
    return {
        "schema": "pharmaml-profile-results/2",
        "generated_by": "test",
        "environment": {"platform": "test"},
        "profiles": [
            {
                "name": "gui-offscreen",
                "command": "pytest tests/ui -k offscreen",
                "outcome_counts": {"passed": 10, "failed": 1, "skipped": 2,
                                   "blocked": 0, "not_executed": 3},
                "collected": 16,
                "notes": "probe",
            },
            {
                "name": "engines",
                "command": "(none)",
                "outcome_counts": {"passed": 0, "failed": 0, "skipped": 0,
                                   "blocked": 0, "not_executed": 7},
            },
        ],
        "gates": [
            {"name": "ruff", "outcome": "passed", "detail": "clean"},
            {"name": "real-display", "outcome": "not run"},
        ],
        "backup_and_recovery": {"status": "verified", "evidence": "live runs"},
        "limitations": ["offscreen only", "engines not executable"],
    }


def _write_results(tmp_path: Path, data: object) -> Path:
    target = tmp_path / "profile-results.json"
    target.write_text(json.dumps(data), encoding="utf-8")
    return target


def _run_with_profiles(archive: Path, output: Path, results: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(archive), "--output-dir", str(output),
         "--profile-results", str(results)],
        capture_output=True, text=True, check=False,
    )


def test_profile_results_fold_into_report_and_summary(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "source.zip")
    results = _write_results(tmp_path, _valid_profile_results())
    outcome = _run_with_profiles(archive, tmp_path / "report", results)
    assert outcome.returncode == 0, outcome.stdout + outcome.stderr
    report = json.loads((tmp_path / "report/report.json").read_text())
    profiles = report["test_profiles"]
    assert profiles["schema"] == "pharmaml-profile-results/2"
    assert profiles["status"] == "recorded evidence; not executed by this audit"
    assert [p["name"] for p in profiles["profiles"]] == ["gui-offscreen", "engines"]
    # counts ride per profile, verbatim -- never merged into any total
    assert profiles["profiles"][0]["outcome_counts"]["failed"] == 1
    assert profiles["profiles"][1]["outcome_counts"]["not_executed"] == 7
    assert "total" not in json.dumps(profiles)
    # a failing profile does not fail the archive audit (recorded, not certified)
    assert report["archive_checks_passed"] is True
    assert report["gates"][1] == {"name": "real-display", "outcome": "not run"}
    assert report["backup_and_recovery"]["status"] == "verified"
    assert "engines not executable" in report["limitations"]
    summary = (tmp_path / "report/SUMMARY.md").read_text()
    assert "recorded evidence; not executed by this audit" in summary
    assert "10 passed, 1 failed, 2 skipped, 0 blocked, 3 not-executed" in summary
    assert "7 not-executed" in summary
    assert "**not run**" in summary
    assert "Backup and recovery" in summary


def test_without_profile_results_the_placeholders_stay(tmp_path: Path) -> None:
    """Backward compatibility: no input file, the honest defaults stand."""
    archive = _zip(tmp_path / "source.zip")
    result = _run(archive, tmp_path / "report")
    assert result.returncode == 0
    report = json.loads((tmp_path / "report/report.json").read_text())
    assert report["test_profiles"] == {"status": "not executed by this audit"}
    assert report["backup_and_recovery"] == {"status": "not verified by this audit"}
    summary = (tmp_path / "report/SUMMARY.md").read_text()
    assert "not executed by this audit" in summary


@pytest.mark.parametrize("mutate, reason", [
    (lambda d: d.update(schema="wrong/1"), "schema must be"),
    (lambda d: d.update(profiles=[]), "non-empty list"),
    (lambda d: d["profiles"][0].update(name=""), "unique non-empty"),
    (lambda d: d["profiles"][1].update(name="gui-offscreen"), "unique non-empty"),
    (lambda d: d["profiles"][0].pop("command"), "recorded command is required"),
    (lambda d: d["profiles"][0].update(command="  "), "recorded command is required"),
    (lambda d: d["profiles"][0]["outcome_counts"].pop("blocked"), "missing outcome keys"),
    (lambda d: d["profiles"][0]["outcome_counts"].update(flaked=1), "unknown outcome keys"),
    (lambda d: d["profiles"][0]["outcome_counts"].update(passed=-1), "int >= 0"),
    (lambda d: d["profiles"][0]["outcome_counts"].update(passed=True), "int >= 0"),
    (lambda d: d["profiles"][0].update(collected=99), "counts sum to"),
    (lambda d: d.update(totals={"passed": 12}), "unknown top-level keys"),
    (lambda d: d["profiles"][0].update(extra=1), "unknown keys"),
    (lambda d: d.update(gates=[{"name": "x", "outcome": "maybe"}]), "passed/failed/not run"),
    (lambda d: d.update(gates=[{"name": "x", "outcome": "passed", "why": "?"}]), "malformed gate"),
    (lambda d: d.update(backup_and_recovery="nope"), "must be an object"),
    (lambda d: d.update(limitations="nope"), "list of strings"),
    (lambda d: d.update(limitations=[1, 2]), "list of strings"),
])
def test_profile_results_validation_refuses_dishonest_files(
    tmp_path: Path, mutate, reason: str
) -> None:
    """Every dishonest or malformed shape fails the whole input file with the
    precise reason -- never a silent degrade, never partial acceptance."""
    archive = _zip(tmp_path / "source.zip")
    data = _valid_profile_results()
    mutate(data)
    results = _write_results(tmp_path, data)
    outcome = _run_with_profiles(archive, tmp_path / "report", results)
    assert outcome.returncode != 0
    assert reason in outcome.stderr
    # the audit refuses BEFORE writing any output (fail closed, no partial report)
    assert not (tmp_path / "report").exists()


def test_profile_results_unreadable_file_fails_closed(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "source.zip")
    results = tmp_path / "bad.json"
    results.write_text("{not json", encoding="utf-8")
    outcome = _run_with_profiles(archive, tmp_path / "report", results)
    assert outcome.returncode != 0
    assert "unreadable" in outcome.stderr
    assert not (tmp_path / "report").exists()


def test_profile_results_recorded_total_must_match_the_counts(tmp_path: Path) -> None:
    """The recorded ``collected`` is cross-checked against the sum of the
    five outcome buckets -- a drifted or inflated total refuses the file."""
    archive = _zip(tmp_path / "source.zip")
    data = _valid_profile_results()
    # engines profile: 7 not-executed rides in the counts; claim 9 collected
    data["profiles"][1]["collected"] = 9
    results = _write_results(tmp_path, data)
    outcome = _run_with_profiles(archive, tmp_path / "report", results)
    assert outcome.returncode != 0
    assert "counts sum to 7, not 9" in outcome.stderr
