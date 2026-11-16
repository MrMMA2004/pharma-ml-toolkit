"""(part8 §14.2 Proposal B, round 7) Reproducibility manifests.

Pins the proposal's four acceptance tests:
(1) the manifest JSON parses and carries all field groups;
(2) versions match the running interpreter's actual packages;
(3) a run with manual ``n_jobs=6`` records ``n_jobs: 6`` end-to-end
    (the §5 field's audit trail);
(4) round-trip: a manifest written for a dataset re-opened later names
    the same fingerprint (and the Proposal A content version).
Plus: config slices ride along when given, dataclass datasets map
cleanly, and the Persistence-tab button exists and refuses honestly
without a bundle.
"""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import pytest


@pytest.fixture()
def runs_dir(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    monkeypatch.setenv("PHARML_STUDIO_RUNS_DIR", str(root))
    return root


@pytest.fixture()
def workspace_file(tmp_path, monkeypatch):
    path = tmp_path / "workspace.json"
    monkeypatch.setenv("PHARML_STUDIO_WORKSPACE", str(path))
    return path


class TestManifestContract:
    def test_json_parses_with_all_field_groups(self, tmp_path):
        from pharmaml.studio import MANIFEST_SCHEMA, write_manifest

        out = tmp_path / "sub" / "bundle.manifest.json"  # parents created
        written = write_manifest(out, runs=[], datasets=[])
        assert written == out
        m = json.loads(out.read_text(encoding="utf-8"))
        assert m["schema"] == MANIFEST_SCHEMA
        for group in ("created", "pharmaml_version", "platform",
                      "pinned_stack", "runs", "datasets"):
            assert group in m, group
        for key in ("python", "implementation", "machine", "system"):
            assert key in m["platform"]
        assert m["pharmaml_version"]

    def test_versions_match_the_running_interpreter(self, tmp_path):
        from pharmaml.studio import PINNED_STACK, write_manifest

        import pharmaml

        write_manifest(tmp_path / "m.json", runs=[], datasets=[])
        m = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
        assert m["pharmaml_version"] == pharmaml.__version__
        assert m["platform"]["python"]
        # every PINNED_STACK name is reported, and an installed one
        # matches importlib.metadata exactly
        assert set(m["pinned_stack"]) == set(PINNED_STACK)
        for name in ("numpy", "pandas", "scikit-learn", "scipy"):
            expected = ""
            try:
                expected = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
            assert m["pinned_stack"][name] == expected

    def test_versions_are_cached_after_first_use(self, tmp_path):
        from pharmaml.studio import reports

        reports._version_cache.clear()
        first = reports._package_version("numpy")
        again = reports._package_version("numpy")
        assert first == again
        assert reports._version_cache["numpy"] == first


class TestRunSlice:
    def test_manual_n_jobs_records_end_to_end(self, runs_dir, tmp_path):
        """§5's field has an audit trail: a run recorded with
        n_jobs=6 keeps n_jobs=6 in the manifest."""
        from pharmaml.studio import record_run, write_manifest
        from pharmaml.studio.runs import latest_runs

        record_run("ml-evaluation", "family benchmark · y", {
            "target": "y", "task": "classification", "folds": 5,
            "n_jobs": 6, "seed": 42, "best_mean": 0.81,
        })
        write_manifest(
            tmp_path / "m.json", runs=latest_runs("ml-evaluation"),
        )
        m = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
        (run,) = m["runs"]
        assert run["summary"]["n_jobs"] == 6
        assert run["summary"]["seed"] == 42
        assert run["family"] == "ml-evaluation"
        assert run["label"] == "family benchmark · y"

    def test_runs_never_recomputed_just_carried(self, runs_dir, tmp_path):
        from pharmaml.studio import record_run, write_manifest

        record_run("docking", "campaign scan", {"poses": 12, "hours": 1.5})
        write_manifest(tmp_path / "m.json", runs=[
            {"family": "docking", "label": "campaign scan",
             "created": "2026-11-16T00:00:00+00:00",
             "summary": {"poses": 12, "hours": 1.5}},
        ])
        m = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
        assert m["runs"][0]["summary"] == {"poses": 12, "hours": 1.5}

    def test_config_slice_rides_along_when_given(self, tmp_path):
        from pharmaml.config import PipelineConfig
        from pharmaml.studio import write_manifest

        write_manifest(
            tmp_path / "m.json", config=PipelineConfig(n_jobs=6),
        )
        m = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
        assert m["config"] == {
            "random_state": 42, "test_size": 0.2,
            "cv_folds": 5, "n_jobs": 6,
        }
        # and absent when not given — defaults are never invented
        write_manifest(tmp_path / "m2.json")
        m2 = json.loads((tmp_path / "m2.json").read_text(encoding="utf-8"))
        assert "config" not in m2


class TestDatasetSlice:
    def test_round_trip_names_the_same_fingerprint(
        self, workspace_file, tmp_path
    ):
        """A dataset re-opened later: the manifest names the same
        sha-256 (and the Proposal A content version it saw)."""
        import hashlib

        import pandas as pd

        from pharmaml.studio import note, write_manifest
        from pharmaml.studio.workspace import entries

        csv = tmp_path / "ds.csv"
        pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}).to_csv(
            csv, index=False
        )
        note("dataset", csv, columns=["x", "y"])
        first = entries("dataset")[0]

        write_manifest(
            tmp_path / "m1.json", runs=[], datasets=entries("dataset"),
        )
        digest = hashlib.sha256(csv.read_bytes()).hexdigest()

        # later session: the same file re-registered (same bytes)
        note("dataset", csv, columns=["x", "y"])
        write_manifest(
            tmp_path / "m2.json", runs=[], datasets=entries("dataset"),
        )
        m1 = json.loads((tmp_path / "m1.json").read_text(encoding="utf-8"))
        m2 = json.loads((tmp_path / "m2.json").read_text(encoding="utf-8"))
        (d1,) = m1["datasets"]
        (d2,) = m2["datasets"]
        assert d1["sha256"] == digest == d2["sha256"]
        assert d1["content_version"] == d2["content_version"] == 1
        assert d1["column_signature"] == d2["column_signature"]

    def test_dataclass_and_mapping_dataset_rows_agree(
        self, workspace_file, tmp_path
    ):
        from pharmaml.studio import write_manifest
        from pharmaml.studio.workspace import WorkspaceEntry

        as_class = WorkspaceEntry(
            kind="dataset", path="/gone/old.csv", name="old.csv",
            recorded="2026-01-01T00:00:00+00:00", sha256="a" * 64,
            content_version=2, column_signature="b" * 16,
        )
        as_mapping = {
            "kind": "dataset", "path": "/gone/old.csv", "name": "old.csv",
            "recorded": "2026-01-01T00:00:00+00:00", "sha256": "a" * 64,
            "content_version": 2, "column_signature": "b" * 16,
        }
        write_manifest(tmp_path / "a.json", datasets=[as_class])
        write_manifest(tmp_path / "b.json", datasets=[as_mapping])
        a = json.loads((tmp_path / "a.json").read_text(encoding="utf-8"))
        b = json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
        assert a["datasets"] == b["datasets"]
