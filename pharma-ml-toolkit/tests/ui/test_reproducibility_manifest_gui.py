"""(part8 §14.2 Proposal B, round 7) The Persistence-tab GUI wiring.

The manifest button: exists on the Persistence tab, refuses honestly
without a bundle, and — with a bundle + recorded slice — writes the
sidecar beside the bundle through the real job path.
"""

from __future__ import annotations

import json

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


class TestPersistenceTabButton:
    def test_button_exists_with_tooltip(self, qtbot, studio_app):
        from ui.pages.ml_workbench import MLWorkbenchPage

        page = MLWorkbenchPage()
        qtbot.addWidget(page)
        btn = getattr(page, "_persist_manifest_btn", None)
        assert btn is not None, "the manifest button is missing"
        assert "reproducibility manifest" in btn.text().lower()
        assert "Descriptive, not executable" in btn.toolTip()

    def test_refuses_honestly_without_a_bundle(self, qtbot, studio_app):
        from ui.pages.ml_workbench import MLWorkbenchPage

        page = MLWorkbenchPage()
        qtbot.addWidget(page)
        assert page._bundle is None
        page._write_manifest()
        status = page._persist_status.text().lower()
        assert "open a model bundle first" in status

    def test_writes_sidecar_beside_the_bundle(
        self, qtbot, studio_app, runs_dir, workspace_file, tmp_path,
        monkeypatch,
    ):
        """End-to-end: a recorded run + workspace dataset land in the
        manifest the button's job writes, beside the bundle."""
        import joblib
        import pandas as pd
        from sklearn.dummy import DummyClassifier

        from pharmaml.studio import note, record_run

        # a recorded run with the §5 field
        record_run("ml-evaluation", "family benchmark · y", {
            "target": "y", "n_jobs": 6, "seed": 42,
        })
        # a workspace dataset with Proposal A provenance
        csv = tmp_path / "ds.csv"
        pd.DataFrame({"x": [1, 2], "y": [0, 1]}).to_csv(csv, index=False)
        note("dataset", csv, columns=["x", "y"])

        from ui.pages.ml_workbench import MLWorkbenchPage

        page = MLWorkbenchPage()
        qtbot.addWidget(page)

        bundle_path = tmp_path / "model.joblib"
        joblib.dump(
            {"model": DummyClassifier(), "metadata": {"task": "classification"}},
            bundle_path,
        )
        page._bundle = (None, None, bundle_path)

        out = tmp_path / "sidecar.manifest.json"
        monkeypatch.setattr(
            "ui.pages.ml_workbench.QFileDialog.getSaveFileName",
            staticmethod(lambda *a, **k: (str(out), "JSON (*.json)")),
        )

        with qtbot.waitSignal(
            page._runner.job_ended, timeout=30_000, raising=True
        ) if hasattr(page._runner, "job_ended") else _null():
            page._write_manifest()

        assert out.exists(), "the manifest sidecar was not written"
        m = json.loads(out.read_text(encoding="utf-8"))
        assert m["schema"] == "pharmaml-manifest/1"
        (run,) = m["runs"]
        assert run["summary"]["n_jobs"] == 6
        (ds,) = m["datasets"]
        assert ds["name"] == "ds.csv"
        assert ds["content_version"] == 1
        assert page._persist_status.text().startswith("Manifest written to")


class _null:
    """A no-op context manager for runners without job_ended."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
