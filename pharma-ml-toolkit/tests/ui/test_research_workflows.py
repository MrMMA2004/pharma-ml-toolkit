"""Offscreen UI tests with real backends and isolated synthetic inputs."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ui.pages.campaign_wizard import CampaignReviewWizard
from ui.pages.uncertainty_dialog import UncertaintyDialog
from ui.workers.job_worker import JobRunner


def test_uncertainty_dialog_uses_three_files_and_clears_stale_report(qtbot, studio_app, tmp_path, monkeypatch):
    runner = JobRunner()
    dialog = UncertaintyDialog(runner)
    qtbot.addWidget(dialog)
    for index, (name, start, n) in enumerate((("train", 0, 20), ("calibration", 20, 10), ("test", 30, 10))):
        x = np.arange(start, start + n, dtype=float)
        frame = pd.DataFrame({"id": [f"{i:04d}" for i in range(start, start+n)], "x": x,
                              "y": 2*x+1, "group": ["A", "B"] * (n // 2)})
        path = tmp_path / f"{index}.csv"
        frame.to_csv(path, index=False)
        dialog._paths[name].setText(str(path))
    dialog._features.setPlainText("x")
    dialog._target.setText("y")
    dialog._key.setText("id")
    dialog._group.setText("group")
    dialog._start()
    qtbot.waitUntil(lambda: dialog._status.text().startswith("Evaluation complete"), timeout=10000)
    report = json.loads(dialog._result.toPlainText())
    assert report["n_train"] == 20 and report["n_calibration"] == 10 and report["n_test"] == 10
    assert len(report["subgroups"]) == 2 and "no conditional guarantee" in report["limitations"]
    assert report["provenance"]["parameters"]["workflow"] == "split-conformal-ridge"
    saved = tmp_path / "report.json"
    monkeypatch.setattr("ui.pages.uncertainty_dialog.QFileDialog.getSaveFileName",
                        lambda *args, **kwargs: (str(saved), "JSON"))
    dialog._save_report()
    saved_contents = saved.read_bytes()
    assert json.loads(saved_contents)["n_test"] == 10
    dialog._save_report()
    assert saved.read_bytes() == saved_contents and "never replaced" in dialog._status.text()
    monkeypatch.setattr("ui.pages.uncertainty_dialog.QFileDialog.getOpenFileName",
                        lambda *args, **kwargs: (str(saved), "JSON"))
    dialog._compare_reports()
    qtbot.waitUntil(lambda: dialog._status.text().startswith("Comparison complete"), timeout=10000)
    assert json.loads(dialog._result.toPlainText())["same_parameters"]
    dialog._paths["test"].setText(dialog._paths["train"].text())
    assert not dialog._result.toPlainText()
    dialog._start()
    qtbot.waitUntil(lambda: dialog._status.text().startswith("Evaluation failed"), timeout=10000)
    assert "overlap" in dialog._status.text()
    assert not dialog._result.toPlainText()


def test_uncertainty_missing_fields_and_busy_job(qtbot, studio_app):
    dialog = UncertaintyDialog(JobRunner())
    qtbot.addWidget(dialog)
    dialog._start()
    assert "Select all three" in dialog._status.text()
    assert not dialog._runner.busy


def test_campaign_review_wizard_no_side_effects_and_clears_report(qtbot, studio_app, tmp_path, monkeypatch):
    config = tmp_path / "box.conf"
    receptor = tmp_path / "r.pdbqt"
    receptor.write_text("receptor")
    ligands = tmp_path / "ligands"
    ligands.mkdir()
    (ligands / "l.pdbqt").write_text("ligand")
    config.write_text("receptor = r.pdbqt\ncenter_x = 0\ncenter_y = 1\ncenter_z = 2\n"
                      "size_x = 10\nsize_y = 11\nsize_z = 12\n")
    monkeypatch.setattr("pharmaml.docking.campaign_preview.engine_argv", lambda *args: ["/bin/sh"])
    wizard = CampaignReviewWizard(JobRunner())
    qtbot.addWidget(wizard)
    wizard._config.setText(str(config))
    wizard._ligands.setText(str(ligands))
    out = tmp_path / "output"
    wizard._output.setText(str(out))
    wizard._start()
    qtbot.waitUntil(lambda: wizard._status.text().startswith("Review only"), timeout=10000)
    result = json.loads(wizard._result.toPlainText())
    assert result["preview"]["size_angstrom"] == [10, 11, 12]
    assert result["preview"]["executed"] is False and result["preview"]["ligands"] == 1 and not out.exists()
    wizard._output.setText(str(ligands))
    assert not wizard._result.toPlainText()
    wizard._start()
    qtbot.waitUntil(lambda: wizard._status.text().startswith("Review failed"), timeout=10000)
    assert "must not overlap" in wizard._status.text()


def test_dialogs_are_reachable_from_studio_pages(qtbot, studio_app):
    from ui.pages.docking import DockingWorkbenchPage
    from ui.pages.model_analysis import ModelAnalysisPage
    from ui.pages.validation_dialog import ValidationDialog

    docking = DockingWorkbenchPage()
    model = ModelAnalysisPage()
    qtbot.addWidget(docking)
    qtbot.addWidget(model)
    docking._wizard_btn.click()
    model._uncertainty_btn.click()
    model._validation_btn.click()
    assert isinstance(docking._campaign_wizard, CampaignReviewWizard)
    assert isinstance(model._uncertainty_dialog, UncertaintyDialog)
    assert isinstance(model._validation_dialog, ValidationDialog)
    docking._campaign_wizard.close()
    model._uncertainty_dialog.close()
    model._validation_dialog.close()


def test_classification_uncertainty_dialog(qtbot, studio_app, tmp_path):
    dialog = UncertaintyDialog(JobRunner())
    qtbot.addWidget(dialog)
    dialog._task.setCurrentIndex(1)
    for index, (name, start, n) in enumerate((("train", 0, 20), ("calibration", 20, 10), ("test", 30, 10))):
        x = np.arange(start, start + n, dtype=float)
        frame = pd.DataFrame({"id": [f"id{i}" for i in range(start, start+n)], "x": x,
                              "label": ["A", "B"] * (n // 2), "group": ["g1", "g2"] * (n // 2)})
        path = tmp_path / f"{index}.csv"
        frame.to_csv(path, index=False)
        dialog._paths[name].setText(str(path))
    dialog._features.setPlainText("x")
    dialog._target.setText("label")
    dialog._key.setText("id")
    dialog._group.setText("group")
    dialog._diagnostics.setChecked(True)
    dialog._start()
    qtbot.waitUntil(lambda: dialog._status.text().startswith("Evaluation complete"), timeout=10000)
    report = json.loads(dialog._result.toPlainText())
    assert report["probability_diagnostics"]["n_test"] == 10
    assert report["applicability"]["status"] == "computed"
    assert report["task"] == "classification" and len(report["class_coverage"]) == 2
    assert report["provenance"]["parameters"]["workflow"] == "split-conformal-logistic"
    dialog._task.setCurrentIndex(0)
    assert not dialog._result.toPlainText() and not dialog._save.isEnabled()


def test_nested_validation_real_and_stop(qtbot, studio_app, tmp_path, monkeypatch):
    import threading

    from pharmaml.exceptions import JobCancelled
    from ui.pages.validation_dialog import ValidationDialog

    dialog = ValidationDialog(JobRunner())
    qtbot.addWidget(dialog)
    x = np.arange(36, dtype=float)
    path = tmp_path / "data.csv"
    pd.DataFrame({"x": x, "y": 2*x+1, "group": [f"g{i//3}" for i in range(36)]}).to_csv(path, index=False)
    dialog._data.setText(str(path))
    dialog._features.setPlainText("x")
    dialog._target.setText("y")
    dialog._group.setText("group")
    dialog._start()
    qtbot.waitUntil(lambda: dialog._status.text().startswith("Nested validation complete"), timeout=10000)
    report = json.loads(dialog._result.toPlainText())
    assert len(report["folds"]) == 3 and report["group"] == "group"
    assert report["provenance"]["input"]["sha256"] and dialog._progress.value() == 100
    started, release = threading.Event(), threading.Event()

    def delayed(*args, on_fold, **kwargs):
        started.set()
        if not release.wait(5):
            raise RuntimeError("timeout")
        if on_fold(1, 3) is False:
            raise JobCancelled("at boundary")
        return {}

    monkeypatch.setattr("pharmaml.ml.validation_workbench.nested_ridge_from_file", delayed)
    dialog._start()
    try:
        qtbot.waitUntil(started.is_set, timeout=5000)
        dialog._request_stop()
    finally:
        release.set()
    qtbot.waitUntil(lambda: dialog._status.text().startswith("Cancelled"), timeout=10000)
    assert not dialog._result.toPlainText() and "no resume" in dialog._status.text()


def test_reproducibility_dialog_environment_checkpoint_and_drift(qtbot, studio_app, tmp_path, monkeypatch):
    from pharmaml.utils.checkpoint_schema import OperationSpec
    from pharmaml.utils.checkpoints import CheckpointManager
    from ui.pages.reproducibility_dialog import ReproducibilityDialog

    dialog = ReproducibilityDialog(JobRunner())
    qtbot.addWidget(dialog)
    dialog._show_environment()
    qtbot.waitUntil(lambda: dialog._status.text().startswith("Diagnostic complete"), timeout=10000)
    assert "packages" in json.loads(dialog._result.toPlainText())
    data = tmp_path / "data.csv"
    data.write_text("first")
    checkpoint = CheckpointManager(tmp_path / "jobs", "job").save(
        step=2, state={}, inputs=[data],
        operation=OperationSpec(operation_type="numpy_mlp_training", operation_id="demo"))
    before = checkpoint.path.read_bytes()
    dialog._checkpoint.setText(str(checkpoint.path))
    dialog._inputs.setPlainText(str(data))
    dialog._show_checkpoint()
    qtbot.waitUntil(lambda: dialog._status.text().startswith("Diagnostic complete"), timeout=10000)
    assert json.loads(dialog._result.toPlainText())["input_status"] == "match"
    data.write_text("changed")
    dialog._show_checkpoint()
    qtbot.waitUntil(lambda: dialog._status.text().startswith("Checkpoint warning"), timeout=10000)
    assert json.loads(dialog._result.toPlainText())["input_status"].startswith("changed")
    assert checkpoint.path.read_bytes() == before
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "diagnose_data_drift.py").write_text(
        "import json, sys\nprint(json.dumps({'strict_check_exit_code': 1, 'strict_drift': ['drift']}))\nsys.exit(1)\n")
    monkeypatch.setattr("ui.pages.reproducibility_dialog._TOOLKIT", tmp_path)
    dialog._show_drift()
    qtbot.waitUntil(lambda: dialog._status.text().startswith("STRICT BYTE DRIFT"), timeout=10000)
    assert json.loads(dialog._result.toPlainText())["strict_check_exit_code"] == 1


def test_reproducibility_dashboard_reachable(qtbot, studio_app):
    from ui.pages.reproducibility_dialog import ReproducibilityDialog
    from ui.pages.utilities import UtilitiesWorkbenchPage

    page = UtilitiesWorkbenchPage()
    qtbot.addWidget(page)
    page._diagnostics_btn.click()
    assert isinstance(page._diagnostics_dialog, ReproducibilityDialog)
    page._diagnostics_dialog.close()


def test_campaign_launch_requires_confirmation_and_runs_snapshot(qtbot, studio_app, tmp_path, monkeypatch):
    import os
    import sys

    from PySide6.QtWidgets import QMessageBox

    if os.name == "nt":
        return  # Windows engine launch is exercised by the dedicated synthetic integration.
    engine = tmp_path / "vina"
    engine.write_text(f"#!{sys.executable}\n" + '''import pathlib, sys
args = sys.argv[1:]
pathlib.Path(args[args.index('--out') + 1]).write_text('synthetic pose')
print('synthetic vina log')
''')
    engine.chmod(0o755)
    receptor = tmp_path / "r.pdbqt"
    receptor.write_text("receptor")
    ligands = tmp_path / "ligands"
    ligands.mkdir()
    (ligands / "a.pdbqt").write_text("ligand")
    config = tmp_path / "vina.conf"
    config.write_text("receptor = r.pdbqt\ncenter_x = 0\ncenter_y = 0\ncenter_z = 0\n"
                      "size_x = 10\nsize_y = 10\nsize_z = 10\nseed = 42\n")
    output = tmp_path / "new-output"
    wizard = CampaignReviewWizard(JobRunner())
    qtbot.addWidget(wizard)
    wizard._config.setText(str(config))
    wizard._ligands.setText(str(ligands))
    wizard._output.setText(str(output))
    wizard._engine.setText(str(engine))
    wizard._start()
    qtbot.waitUntil(lambda: wizard._status.text().startswith("Review only"), timeout=10000)
    assert wizard._launch.isEnabled() and not output.exists()
    monkeypatch.setattr("ui.pages.campaign_wizard.QMessageBox.question",
                        lambda *args, **kwargs: QMessageBox.StandardButton.No)
    wizard._confirm_launch()
    assert not output.exists() and wizard._launch.isEnabled()
    monkeypatch.setattr("ui.pages.campaign_wizard.QMessageBox.question",
                        lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    wizard._confirm_launch()
    qtbot.waitUntil(lambda: wizard._status.text().startswith("Campaign completed"), timeout=10000)
    result = json.loads((output / "campaign.json").read_text())
    assert result["status"] == "completed" and len(result["results"]) == 1
    assert not wizard._launch.isEnabled() and (output / result["results"][0]["pose"]).is_file()


def test_nested_logistic_validation_in_studio(qtbot, studio_app, tmp_path):
    from ui.pages.validation_dialog import ValidationDialog

    dialog = ValidationDialog(JobRunner())
    qtbot.addWidget(dialog)
    x = np.arange(72)
    path = tmp_path / "classes.csv"
    pd.DataFrame({"x": (x % 2) * 3 + x / 200, "label": np.where(x % 2, "01", "1"),
                  "group": [f"g{i // 6}" for i in x]}).to_csv(path, index=False)
    dialog._data.setText(str(path))
    dialog._features.setPlainText("x")
    dialog._target.setText("label")
    dialog._group.setText("group")
    dialog._task.setCurrentIndex(1)
    dialog._start()
    qtbot.waitUntil(lambda: dialog._status.text().startswith("Nested validation complete"), timeout=15000)
    report = json.loads(dialog._result.toPlainText())
    assert report["task"] == "classification" and report["classes"] == ["01", "1"]
    assert len(report["folds"]) == 3 and report["provenance"]["input"]["sha256"]
    assert dialog._progress.value() == 100
    dialog._task.setCurrentIndex(0)
    assert not dialog._result.toPlainText()


def test_binary_calibration_dialog(qtbot, studio_app, tmp_path, monkeypatch):
    from ui.pages.calibration_dialog import CalibrationDialog

    dialog = CalibrationDialog(JobRunner())
    qtbot.addWidget(dialog)
    for name, n in (("train", 60), ("calibration", 40), ("test", 30)):
        x = np.arange(n, dtype=float)
        path = tmp_path / f"{name}.csv"
        pd.DataFrame({"id": [f"{name}_{i}" for i in range(n)], "x": x,
                      "y": np.where(x % 2, "01", "1")}).to_csv(path, index=False)
        dialog._paths[name].setText(str(path))
    dialog._features.setPlainText("x")
    dialog._target.setText("y")
    dialog._key.setText("id")
    dialog._positive.setText("01")
    dialog._start()
    qtbot.waitUntil(lambda: dialog._status.text().startswith("Evaluation complete"), timeout=10000)
    report = json.loads(dialog._result.toPlainText())
    assert report["positive_label"] == "01" and report["n_probability_calibration"] == 40
    assert report["recalibrated"]["binary_brier"] >= 0 and dialog._save.isEnabled()
    saved = tmp_path / "report.json"
    monkeypatch.setattr("ui.pages.calibration_dialog.QFileDialog.getSaveFileName",
                        lambda *args, **kwargs: (str(saved), "JSON"))
    dialog._save_report()
    before = saved.read_bytes()
    dialog._save_report()
    assert saved.read_bytes() == before and "never replaced" in dialog._status.text()
    dialog._positive.setText("unknown")
    assert not dialog._save.isEnabled() and not dialog._result.toPlainText()
    dialog._start()
    qtbot.waitUntil(lambda: dialog._status.text().startswith("Evaluation failed"), timeout=10000)
    assert "positive label" in dialog._status.text()


def test_binary_calibration_dialog_reachable(qtbot, studio_app):
    from ui.pages.calibration_dialog import CalibrationDialog
    from ui.pages.model_analysis import ModelAnalysisPage

    page = ModelAnalysisPage()
    qtbot.addWidget(page)
    page._calibration_btn.click()
    assert isinstance(page._calibration_dialog, CalibrationDialog)
    page._calibration_dialog.close()


def test_reproducibility_compares_training_records(qtbot, studio_app, tmp_path, monkeypatch):
    from ui.pages.reproducibility_dialog import ReproducibilityDialog

    record = {"schema": "pharmaml-cli-training-run/1", "dataset": {"sha256": "dataset"},
              "input_processing": {}, "split": {"train_ordered_clean_row_indices_sha256": "train",
                                                 "test_ordered_clean_row_indices_sha256": "test"},
              "training": {}, "model_artifact": {"sha256": "model"}, "environment": {}}
    first, second = tmp_path / "one.json", tmp_path / "two.json"
    first.write_text(json.dumps(record))
    record["dataset"]["sha256"] = "different"
    second.write_text(json.dumps(record))
    choices = iter([(str(first), "JSON"), (str(second), "JSON")])
    monkeypatch.setattr("ui.pages.reproducibility_dialog.QFileDialog.getOpenFileName", lambda *a, **k: next(choices))
    dialog = ReproducibilityDialog(JobRunner())
    qtbot.addWidget(dialog)
    dialog._show_training_comparison()
    qtbot.waitUntil(lambda: dialog._status.text().startswith("Diagnostic complete"), timeout=10000)
    result = json.loads(dialog._result.toPlainText())
    assert not result["same_dataset_sha256"] and result["same_split"]
    assert "not proof" in result["notice"]


def test_preparation_dialog_review_confirm_and_invalidation(qtbot, studio_app, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from ui.pages.preparation_dialog import PreparationDialog

    output = tmp_path / "new-output"
    plan = {"schema": "pharmaml-preparation-plan/1", "output_dir": str(output),
            "review": {"rows": 2, "invalid_smiles": 1}}
    called = []

    def review(receptor, table, out, **options):
        called.append(("review", receptor, table, out, options))
        return plan

    def prepare(reviewed):
        assert reviewed == plan
        called.append(("prepare",))
        output.mkdir()
        return {"status": "completed_with_failures", "ligands": {"prepared": 1, "failed": 1}}

    monkeypatch.setattr("pharmaml.docking.campaign_preparation.build_preparation_plan", review)
    monkeypatch.setattr("pharmaml.docking.campaign_preparation.prepare_reviewed_inputs", prepare)
    dialog = PreparationDialog(JobRunner())
    qtbot.addWidget(dialog)
    dialog._receptor.setText(str(tmp_path / "receptor.pdb"))
    dialog._table.setText(str(tmp_path / "ligands.csv"))
    dialog._output.setText(str(output))
    dialog._start_review()
    qtbot.waitUntil(lambda: dialog._status.text().startswith("Review complete"), timeout=10000)
    assert dialog._prepare.isEnabled() and not output.exists()
    monkeypatch.setattr("ui.pages.preparation_dialog.QMessageBox.question",
                        lambda *args, **kwargs: QMessageBox.StandardButton.No)
    dialog._confirm_prepare()
    assert not output.exists() and len(called) == 1
    monkeypatch.setattr("ui.pages.preparation_dialog.QMessageBox.question",
                        lambda *args, **kwargs: QMessageBox.StandardButton.Yes)
    dialog._confirm_prepare()
    qtbot.waitUntil(lambda: dialog._status.text().startswith("Preparation completed_with_failures"), timeout=10000)
    assert output.is_dir() and not dialog._prepare.isEnabled() and len(called) == 2
    dialog._chain.setText("A")
    assert not dialog._result.toPlainText() and not dialog._prepare.isEnabled()


def test_preparation_dialog_reachable(qtbot, studio_app):
    from ui.pages.docking import DockingWorkbenchPage
    from ui.pages.preparation_dialog import PreparationDialog

    page = DockingWorkbenchPage()
    qtbot.addWidget(page)
    page._preparation_btn.click()
    assert isinstance(page._preparation_dialog, PreparationDialog)
    page._preparation_dialog.close()
