"""§17D UI — the run-report dialog + Job Center wiring.

Offscreen against a REAL recorded run (no mocks): the Job Center
exposes the Report button, the under/over-selection note is honest,
the dialog previews the eight §17D sections in order with the
no-conclusions statement, and the save path writes the self-contained
HTML through the real writer (QFileDialog is pointed at a real path).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pharmaml.studio import runs as runs_mod


@pytest.fixture()
def runs_root(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "studio-runs"
    monkeypatch.setattr(runs_mod, "runs_root", lambda: root)
    assert runs_mod.record_run(
        "ml-evaluation", "family benchmark · activity (5-fold CV)",
        {"eval_kind": "cv", "metric": "roc_auc_ovo", "folds": 5,
         "task": "classification", "target": "activity",
         "families": 18, "best_family": "descriptors",
         "best_mean": 0.812345, "n_failed": 0, "seconds": 12.5},
    )
    return root


def _center(qtbot, studio_app):
    from ui.widgets.job_center import JobCenterDialog
    from ui.workers.job_worker import JobRunner

    dialog = JobCenterDialog(JobRunner())
    qtbot.addWidget(dialog)
    return dialog


def _select_first(dialog) -> None:
    dialog._history.clearSelection()
    dialog._history.item(0).setSelected(True)


def test_job_center_offers_the_report_button(qtbot, studio_app, runs_root) -> None:
    dialog = _center(qtbot, studio_app)
    assert dialog._history.count() == 1
    assert dialog._report_btn.text().startswith("Report selected run")

    # zero selected → the honest note, no modal dialog
    # (setCurrentRow(0) from _refresh_history also selects row 0 in
    # Qt, so the selection must be cleared explicitly first)
    dialog._history.clearSelection()
    dialog._report_selected()
    assert "exactly one run" in dialog._compare_note.text()


def test_report_dialog_previews_eight_sections_in_order(
        qtbot, studio_app, runs_root) -> None:
    record = runs_mod.latest_runs("ml-evaluation")[0]
    from ui.widgets.run_report import RunReportDialog

    dialog = RunReportDialog(record)
    qtbot.addWidget(dialog)
    text = dialog._preview.toPlainText()
    order = ["== Inputs", "== Methods", "== Parameters", "== Results",
             "== Plots", "== Statistical metrics", "== Provenance",
             "== Limitations"]
    positions = [text.index(marker) for marker in order]
    assert positions == sorted(positions)
    assert "target: activity" in text
    assert "0.81234" in text  # 5 significant figures of 0.812345
    assert "no conclusions" in dialog._note.text().lower()
    assert "does not interpret" in text


def test_report_save_writes_self_contained_html(
        qtbot, studio_app, runs_root, tmp_path, monkeypatch) -> None:
    record = runs_mod.latest_runs("ml-evaluation")[0]
    from ui.widgets import run_report as rr

    dialog = rr.RunReportDialog(record)
    qtbot.addWidget(dialog)
    out = tmp_path / "report.html"
    monkeypatch.setattr(
        rr.QFileDialog, "getSaveFileName",
        staticmethod(lambda *a, **k: (str(out), "HTML (*.html)")))
    dialog._save()
    assert dialog._status.text().startswith("saved:")
    content = out.read_text(encoding="utf-8")
    for section in ("Inputs", "Methods", "Parameters", "Results",
                    "Plots", "Statistical metrics", "Provenance",
                    "Limitations"):
        assert section in content
    assert "<link" not in content
    assert "Conclusion" not in content
