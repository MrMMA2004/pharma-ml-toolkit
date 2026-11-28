"""Studio's separate-file audit calls the real backend and protects summaries."""
from __future__ import annotations

import json
import threading

from ui.pages.data_explorer import DataExplorerPage
from ui.pages.file_split_audit_dialog import FileSplitAuditDialog
from ui.workers.job_worker import JobRunner


def test_page_opens_separate_file_audit(qtbot, studio_app):
    page = DataExplorerPage()
    qtbot.addWidget(page)
    page._file_split_audit_btn.click()
    dialog = page._file_split_audit_dialog
    assert dialog.isVisible() and dialog._runner is page._runner
    dialog.close()
    page._file_split_audit_btn.click()
    assert page._file_split_audit_dialog is dialog


def test_audit_literal_ids_summary_and_save_no_clobber(qtbot, studio_app, tmp_path, monkeypatch):
    dialog = FileSplitAuditDialog(JobRunner())
    qtbot.addWidget(dialog)
    train, test = tmp_path / "train.csv", tmp_path / "test.tsv"
    train.write_text("id,site\n001,A\nNA,B\nSECRET,C\nSECRET,C\n", encoding="utf-8")
    test.write_text("id\tsite\n1\tA\nNA\tB\nSECRET\tC\n", encoding="utf-8")
    dialog._partitions.setPlainText(f"train={train}\ntest={test}")
    dialog._keys.setPlainText("id\nsite")
    dialog._run.click()
    qtbot.waitUntil(lambda: dialog._status.text().startswith("Audit complete"), timeout=10000)
    text = dialog._result.toPlainText()
    result = json.loads(text)
    assert result["shared_key_count"] == 2
    assert result["partitions"][0]["duplicate_rows"] == 1
    assert result["pairwise_shared_unique_keys"]["train"]["test"] == 2
    assert "SECRET" not in text and str(tmp_path) not in text
    output = tmp_path / "summary.json"
    monkeypatch.setattr("ui.pages.file_split_audit_dialog.QFileDialog.getSaveFileName",
                        lambda *args, **kwargs: (str(output), "JSON"))
    dialog._save.click()
    before = output.read_bytes()
    assert json.loads(before)["shared_key_count"] == 2
    dialog._save.click()
    assert output.read_bytes() == before and "never replaced" in dialog._status.text()
    dialog._keys.setPlainText("missing")
    assert not dialog._result.toPlainText() and not dialog._save.isEnabled()
    dialog._run.click()
    qtbot.waitUntil(lambda: dialog._status.text().startswith("Audit failed"), timeout=10000)
    assert "missing key" in dialog._status.text() and not dialog._result.toPlainText()


def test_validation_busy_and_stale_result(qtbot, studio_app, tmp_path, monkeypatch):
    from pharmaml.ml import file_split_audit

    runner = JobRunner()
    dialog = FileSplitAuditDialog(runner)
    qtbot.addWidget(dialog)
    dialog._run.click()
    assert "at least two" in dialog._status.text() and not runner.busy
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("id\na\n", encoding="utf-8")
    b.write_text("id\nb\n", encoding="utf-8")
    dialog._partitions.setPlainText(f"train={a}\ntrain={b}")
    dialog._keys.setPlainText("id")
    dialog._run.click()
    assert "unique" in dialog._status.text() and not runner.busy
    dialog._partitions.setPlainText(f"train={a}\ntest={b}")
    started, release = threading.Event(), threading.Event()
    original = file_split_audit.audit_partition_files

    def delayed(*args, **kwargs):
        started.set()
        if not release.wait(5):
            raise RuntimeError("timed out")
        return original(*args, **kwargs)

    monkeypatch.setattr(file_split_audit, "audit_partition_files", delayed)
    dialog._run.click()
    try:
        qtbot.waitUntil(started.is_set, timeout=5000)
        # Programmatic edits must also invalidate the in-flight result.
        dialog._keys.setPlainText("other")
        assert not dialog._run.isEnabled()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not runner.busy and dialog._run.isEnabled(), timeout=5000)
    assert not dialog._result.toPlainText() and not dialog._save.isEnabled()


def test_compare_declared_manifests(qtbot, studio_app, tmp_path, monkeypatch):
    from ui.pages.reproducibility_dialog import ReproducibilityDialog

    from pharmaml.ml.provenance import experiment_manifest

    data = tmp_path / "data.csv"
    data.write_text("id\n001\n", encoding="utf-8")
    first, second = tmp_path / "one.json", tmp_path / "two.json"
    first.write_text(json.dumps(experiment_manifest(data, parameters={"seed": "1"}, split_strategy="group")))
    second.write_text(json.dumps(experiment_manifest(data, parameters={"seed": "2"}, split_strategy="group")))
    paths = iter([(str(first), "JSON"), (str(second), "JSON")])
    monkeypatch.setattr("ui.pages.reproducibility_dialog.QFileDialog.getOpenFileName",
                        lambda *args, **kwargs: next(paths))
    dialog = ReproducibilityDialog(JobRunner())
    qtbot.addWidget(dialog)
    dialog._compare_manifests.click()
    qtbot.waitUntil(lambda: dialog._status.text().startswith("Diagnostic complete"), timeout=10000)
    result = json.loads(dialog._result.toPlainText())
    assert result["same_dataset_sha256"] and not result["same_parameters"]
    assert "do not prove" in result["notice"]
    # Malformed records fail rather than retaining the previous success.
    second.write_text("{}")
    paths = iter([(str(first), "JSON"), (str(second), "JSON")])
    dialog._compare_manifests.click()
    qtbot.waitUntil(lambda: dialog._status.text().startswith("Diagnostic failed"), timeout=10000)
    assert not dialog._result.toPlainText()
