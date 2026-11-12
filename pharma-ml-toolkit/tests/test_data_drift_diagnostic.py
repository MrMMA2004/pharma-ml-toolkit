"""The optional drift diagnostic must describe differences, never waive them."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def diagnostic(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("diagnose_data_drift", ROOT / "tools/diagnose_data_drift.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comparison_describes_numeric_and_categorical_changes(diagnostic):
    report = diagnostic.compare_csv(
        b"x,class\n1.0,active\n2.0,inactive\n",
        b"x,class\n1.000000000000001,inactive\n2.00,inactive\n",
    )
    assert report["same_header"] and report["same_shape"]
    assert report["numeric_cells_with_changed_text"] == 2
    assert report["changed_cells_by_column"] == {"0:x": 2, "1:class": 1}
    assert report["sample_changes"][0] == {
        "csv_row": 2, "column": "0:x", "stored": "1.0", "generated": "1.000000000000001"
    }
    assert report["non_numeric_cells_with_changed_text"] == 1
    assert 0 < report["max_absolute_numeric_difference"] < 1e-14


def test_comparison_reports_shape_header_nonfinite_and_bad_utf8(diagnostic):
    report = diagnostic.compare_csv(b"x\n1\n", b"other\nnan\n3\n")
    assert not report["same_shape"] and not report["same_header"]
    assert report["numeric_nonfinite_or_overflow"] == 1
    assert "parse_error" in diagnostic.compare_csv(b"\xff", b"x\n")


def test_diagnostic_captures_byte_drift_without_modifying_files(diagnostic, tmp_path, monkeypatch):
    target = tmp_path / "sample.csv"
    target.write_bytes(b"x\n1.0\n")
    generated = b"x\n1.000000000000001\n"
    calls = []

    def original_writer(path, data):
        calls.append((path, data))

    def fake_main(args):
        assert args == ["--check"]
        diagnostic.builder._write_if_changed(target, generated)
        diagnostic.builder.STATS["drift"].append("strict byte mismatch")
        raise SystemExit(1)

    monkeypatch.setattr(diagnostic.builder, "_write_if_changed", original_writer)
    monkeypatch.setattr(diagnostic.builder, "main", fake_main)
    monkeypatch.setattr(diagnostic.builder, "STATS", {"drift": []})
    report = diagnostic.diagnose()
    assert target.read_bytes() == b"x\n1.0\n"
    assert diagnostic.builder._write_if_changed is original_writer
    assert calls == [(target, generated)]
    assert report["strict_check_exit_code"] == 1
    assert report["strict_drift"] == ["strict byte mismatch"]
    assert report["generated_files_with_byte_drift"][0]["csv_comparison"]["numeric_cells_with_changed_text"] == 1