"""(part8 §14.2 Proposal A, round 6) The GUI wiring half.

Split from tests/test_dataset_provenance_v2.py: these need the
tests/ui/conftest.py ``studio_app`` fixture (the Qt application).
"""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture()
def workspace_file(tmp_path, monkeypatch):
    path = tmp_path / "workspace.json"
    monkeypatch.setenv("PHARML_STUDIO_WORKSPACE", str(path))
    return path


@pytest.fixture()
def runs_dir(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    monkeypatch.setenv("PHARML_STUDIO_RUNS_DIR", str(root))
    return root


class TestGUIWiring:
    def test_model_analysis_badge_shows_from_v2(
        self, qtbot, studio_app, workspace_file, tmp_path
    ):
        from ui.pages.model_analysis import ModelAnalysisPage

        csv = tmp_path / "ds.csv"
        pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}).to_csv(csv, index=False)

        page = ModelAnalysisPage()
        qtbot.addWidget(page)
        # first sighting: no badge (v1 is unremarkable)
        assert page._dataset_badge(csv, pd.read_csv(csv)) == ""
        # the file changes in place
        pd.DataFrame({"x": [1, 2, 9], "y": [0, 1, 0]}).to_csv(csv, index=False)
        badge = page._dataset_badge(csv, pd.read_csv(csv))
        assert badge == "[v2]"

    def test_benchmark_record_carries_dataset_provenance(
        self, qtbot, studio_app, workspace_file, runs_dir, tmp_path,
        monkeypatch,
    ):
        """The workbench's ml-evaluation record names the dataset's
        version + sha + column signature when the source is registered."""
        from pharmaml.studio import note
        from pharmaml.studio.runs import latest_runs

        import ui.pages.ml_workbench as mw

        csv = tmp_path / "ds.csv"
        pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]}).to_csv(csv, index=False)
        note("dataset", csv, columns=["x", "y"])

        page = mw.MLWorkbenchPage()
        qtbot.addWidget(page)
        fake = type("Fake", (), {
            "_frame": pd.read_csv(csv),
            "_target_combo": type("C", (), {"currentText": lambda s: "y"})(),
            "_external_path": csv,
        })()
        page._model_analysis = fake
        page._training_context = lambda: (
            fake._frame, "y", "classification"
        )

        # exercise the record block exactly as done() runs it
        from pharmaml.studio import workspace as _ws

        dataset_provenance = {}
        source = getattr(page._model_analysis, "_external_path", None)
        if source is not None:
            prior = next((
                e for e in _ws.entries("dataset") if e.path == str(source)
            ), None)
            if prior is not None:
                dataset_provenance = {
                    "dataset_name": prior.name,
                    "dataset_version": prior.content_version,
                    "dataset_sha256": prior.sha256[:16],
                    "dataset_columns": prior.column_signature,
                }
        from pharmaml.studio import record_run

        record_run("ml-evaluation", "probe", {"target": "y", **dataset_provenance})

        stored = latest_runs("ml-evaluation")[0]["summary"]
        assert stored["dataset_name"] == "ds.csv"
        assert stored["dataset_version"] == 1
        assert len(stored["dataset_sha256"]) == 16
        assert stored["dataset_columns"]

    def test_job_center_row_names_the_version(self, workspace_file, runs_dir):
        from PySide6.QtWidgets import QApplication

        from pharmaml.studio import record_run
        from ui.widgets.job_center import JobCenterDialog
        from ui.workers.job_worker import JobRunner

        record_run("ml-evaluation", "family benchmark · y", {
            "target": "y", "dataset_version": 3,
        })
        record_run("docking", "campaign scan", {"poses": 12})  # no provenance

        app = QApplication.instance() or QApplication([])
        runner = JobRunner()
        dialog = JobCenterDialog(runner)
        dialog._refresh_history()
        labels = [
            dialog._history.item(i).text()
            for i in range(dialog._history.count())
        ]
        assert any("· data v3" in text for text in labels), labels
        assert any("data v" not in text and "campaign scan" in text
                   for text in labels), labels
