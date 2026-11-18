"""§17G — favorites, export presets, layout preferences.

All three ride the SAME workspace file as additive top-level keys;
an older payload (or a foreign-schema file) must keep reading as
empty state, and every setter must round-trip through the real file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pharmaml.studio import workspace as ws


@pytest.fixture()
def workspace_file(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "studio-workspace.json"
    monkeypatch.setenv("PHARML_STUDIO_WORKSPACE", str(path))
    return path


def test_favorite_toggle_round_trips(workspace_file) -> None:
    assert ws.note("dataset", "/tmp/a.csv")
    assert ws.note("dataset", "/tmp/b.csv")
    assert not ws.is_favorite("/tmp/a.csv")  # nothing starred initially
    assert ws.toggle_favorite("/tmp/a.csv") is True   # star
    assert ws.is_favorite("/tmp/a.csv")
    assert ws.favorite_entries() == [ws.entries()[1]]  # only a.csv
    assert ws.toggle_favorite("/tmp/a.csv") is False  # unstar
    assert ws.favorite_entries() == []
    # the payload stays additive: favorites is a new top-level key,
    # the entries themselves are untouched
    payload = json.loads(workspace_file.read_text())
    assert payload["schema"] == "studio-workspace/1"
    assert len(payload["entries"]) == 2


def test_export_folders_per_template(workspace_file) -> None:
    assert ws.last_export_folder("qsar_model_development") == ""
    assert ws.remember_export_folder("/exports/qsar",
                                     "qsar_model_development")
    assert ws.remember_export_folder("/exports/shared")
    assert ws.last_export_folder(
        "qsar_model_development") == "/exports/qsar"
    assert ws.last_export_folder("pk_pd_modeling") == ""  # not set
    assert ws.last_export_folder() == "/exports/shared"
    # a different template's folder never bleeds into another
    ws.remember_export_folder("/exports/pk", "pk_pd_modeling")
    assert ws.last_export_folder(
        "qsar_model_development") == "/exports/qsar"


def test_layout_preferences_round_trip(workspace_file) -> None:
    assert ws.layout_preference("utilities.tab") == ""
    assert ws.set_layout_preference("utilities.tab", "templates")
    assert ws.layout_preference("utilities.tab") == "templates"
    assert ws.layout_preference("other.key", "fallback") == "fallback"


def test_old_payload_reads_as_empty_state(
        workspace_file) -> None:
    # an older build's file: entries + project, no §17G keys
    workspace_file.write_text(json.dumps({
        "schema": "studio-workspace/1",
        "entries": [{"kind": "dataset", "path": "/tmp/x.csv",
                     "name": "x.csv", "recorded": "2026-01-01T00:00:00+00:00"}],
        "current_project": "demo",
    }))
    assert not ws.is_favorite("/tmp/x.csv")
    assert ws.favorite_entries() == []
    assert ws.last_export_folder() == ""
    assert ws.layout_preference("anything") == ""
    # and the setters extend that file without dropping the old keys
    assert ws.toggle_favorite("/tmp/x.csv")
    payload = json.loads(workspace_file.read_text())
    assert payload["current_project"] == "demo"
    assert len(payload["entries"]) == 1
    assert payload["favorites"] == ["/tmp/x.csv"]


def test_corrupt_payload_reads_as_empty_state(workspace_file) -> None:
    workspace_file.write_text("{not json")
    assert ws.favorite_entries() == []
    assert ws.last_export_folder() == ""
    assert ws.layout_preference("k") == ""
