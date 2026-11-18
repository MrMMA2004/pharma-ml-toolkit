"""Declared manifests and GA Stop reach real backends through Studio's runner."""
from __future__ import annotations

import json
import threading

import pytest

from ui.pages.declared_manifest_dialog import DeclaredManifestDialog
from ui.pages.reproducibility_dialog import ReproducibilityDialog
from ui.workers.job_worker import JobRunner


def test_manifest_create_save_compare_and_no_clobber(qtbot, studio_app, tmp_path, monkeypatch):
    from pharmaml.ml.provenance import compare_manifests

    parent = ReproducibilityDialog(JobRunner())
    qtbot.addWidget(parent)
    parent._create_manifest.click()
    dialog = parent._manifest_dialog
    assert dialog.isVisible() and dialog._runner is parent._runner
    data = tmp_path / "data.csv"
    data.write_text("id\n001\n", encoding="utf-8")
    dialog._data.setText(str(data))
    dialog._split.setText("group-disjoint")
    dialog._parameters.setPlainText("model=ridge\nseed=42")
    dialog._create.click()
    qtbot.waitUntil(lambda: dialog._status.text().startswith("Manifest created"), timeout=10000)
    report = json.loads(dialog._result.toPlainText())
    assert report["schema"] == "pharmaml-experiment-manifest/1"
    assert report["parameters"] == {"model": "ridge", "seed": "42"}
    assert report["dataset"]["name"] == "data.csv"
    assert compare_manifests(report, report)["same_dataset_sha256"]
    output = tmp_path / "manifest.json"
    monkeypatch.setattr("ui.pages.declared_manifest_dialog.QFileDialog.getSaveFileName",
                        lambda *a, **k: (str(output), "JSON"))
    dialog._save.click()
    original = output.read_bytes()
    assert json.loads(original) == report
    dialog._save.click()
    assert output.read_bytes() == original and "never replaced" in dialog._status.text()
    dialog._parameters.setPlainText("seed=1\nseed=2")
    assert not dialog._result.toPlainText() and not dialog._save.isEnabled()
    dialog._create.click()
    assert "unique" in dialog._status.text() and not dialog._runner.busy
    dialog._parameters.setPlainText("seed=1")
    dialog._data.setText(str(tmp_path / "missing.csv"))
    dialog._create.click()
    qtbot.waitUntil(lambda: dialog._status.text().startswith("Manifest failed"), timeout=10000)
    assert not dialog._save.isEnabled() and not dialog._result.toPlainText()


def test_manifest_busy_and_stale_job(qtbot, studio_app, tmp_path, monkeypatch):
    from pharmaml.ml import provenance

    dialog = DeclaredManifestDialog(JobRunner())
    qtbot.addWidget(dialog)
    dialog._create.click()
    assert "Select a dataset" in dialog._status.text()
    path = tmp_path / "data.csv"
    path.write_text("x\n1\n", encoding="utf-8")
    dialog._data.setText(str(path))
    dialog._split.setText("group")
    started, release = threading.Event(), threading.Event()
    real = provenance.experiment_manifest

    def delayed(*args, **kwargs):
        started.set()
        if not release.wait(5):
            raise RuntimeError("timed out")
        return real(*args, **kwargs)

    monkeypatch.setattr(provenance, "experiment_manifest", delayed)
    dialog._create.click()
    try:
        qtbot.waitUntil(started.is_set, timeout=5000)
        dialog._split.setText("changed")
        assert not dialog._create.isEnabled()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not dialog._runner.busy and dialog._create.isEnabled(), timeout=5000)
    assert not dialog._save.isEnabled() and not dialog._result.toPlainText()


def test_ga_stop_surfaces_partial_without_checkpoint(qtbot, studio_app, monkeypatch):
    from pharmaml.exceptions import JobCancelled
    from pharmaml.generative.result import GenerationResult
    from ui.pages.generative import GenerativeWorkbenchPage

    started, proceed = threading.Event(), threading.Event()
    captured = {}
    partial = GenerationResult(smiles=["CCO"], scores=[0.5], breakdown=[{}],
                               history=[{"generation": 0, "best_score": 0.5, "mean_score": 0.5}],
                               metadata={"partial": True, "elapsed_s": 0.0})

    class FakeOptimizer:
        def __init__(self, *_args, **_kwargs):
            pass

        def optimize(self, **kwargs):
            captured.update(kwargs)
            started.set()
            if not proceed.wait(5):
                raise RuntimeError("timed out")
            captured["keep_going"] = kwargs["progress"](1, 4, "generative search")
            if captured["keep_going"] is False:
                raise JobCancelled("cancelled", partial_results=partial)
            return partial

    monkeypatch.setattr("pharmaml.generative.GeneticOptimizer", FakeOptimizer)
    page = GenerativeWorkbenchPage()
    qtbot.addWidget(page)
    page._generations.setValue(4)
    page._start_evolve()
    try:
        qtbot.waitUntil(started.is_set, timeout=5000)
        page._runner.request_stop()
    finally:
        proceed.set()
    qtbot.waitUntil(lambda: page.report_text().startswith("PARTIAL"), timeout=10000)
    assert captured["keep_going"] is False and "checkpoint_dir" not in captured
    assert page._last_result is partial and page._table.model().rowCount() == 1
    assert "1/4 generations" in page.report_text() and "no checkpoint or resume" in page.report_text()
    assert "0.0 s" not in page.report_text()
    page._start_front()
    assert "last run is partial" in page.report_text()


@pytest.mark.parametrize("use_builtin,seeded,expected", [
    (True, False, None), (False, False, "corpus"), (False, True, "seed_smiles"),
])
def test_ga_progress_passed_for_each_corpus_mode(qtbot, studio_app, monkeypatch, use_builtin, seeded, expected):
    from pharmaml.generative.result import GenerationResult
    from ui.pages.generative import GenerativeWorkbenchPage

    captured = {}

    class FakeOptimizer:
        def __init__(self, *_args, **_kwargs):
            pass

        def optimize(self, **kwargs):
            captured.update(kwargs)
            assert kwargs["progress"](1, 4, "generative search") is True
            return GenerationResult(smiles=["CCO"], scores=[0.5], history=[], metadata={})

    monkeypatch.setattr("pharmaml.generative.GeneticOptimizer", FakeOptimizer)
    page = GenerativeWorkbenchPage()
    qtbot.addWidget(page)
    if use_builtin:
        page._corpus_combo.setCurrentIndex(1)
    page._seeded_check.setChecked(seeded)
    page._start_evolve()
    qtbot.waitUntil(lambda: page._last_result is not None, timeout=10000)
    assert callable(captured["progress"]) and "checkpoint_dir" not in captured
    assert ("corpus" in captured) == (expected == "corpus")
    assert ("seed_smiles" in captured) == (expected == "seed_smiles")
