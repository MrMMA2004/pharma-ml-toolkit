"""§17C UI — the experiment-comparison dialog + Job Center wiring.

Offscreen, against REAL run records written through ``record_run``
into a monkeypatched runs root (no mocks): the Job Center exposes
multi-select + the Compare button, the refusal note appears for
under-selection, and the dialog itself renders the compatibility
groups, the deltas and the refused pairs exactly as the backend
computed them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pharmaml.studio import runs as runs_mod


@pytest.fixture()
def runs_root(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "studio-runs"
    monkeypatch.setattr(runs_mod, "runs_root", lambda: root)
    return root


def _record(runs_root: Path, family: str, label: str, **summary) -> None:
    assert runs_mod.record_run(family, label, summary) is not None


@pytest.fixture()
def seeded(runs_root) -> Path:
    _record(runs_root, "ml-evaluation", "descriptors board",
            eval_kind="cv", metric="roc_auc_ovo", folds=5,
            task="classification", target="activity",
            best_family="descriptors", best_mean=0.82)
    _record(runs_root, "ml-evaluation", "fingerprints board",
            eval_kind="cv", metric="roc_auc_ovo", folds=5,
            task="classification", target="activity",
            best_family="fingerprints", best_mean=0.75)
    _record(runs_root, "ml-evaluation", "reg run (r2)",
            eval_kind="cv", metric="r2", folds=5,
            task="regression", target="pIC50", best_mean=0.54)
    _record(runs_root, "pkpd", "regimen · oral",
            route="oral", n_doses=1, c_max=8.2, t_max=1.5, auc=55.0)
    return runs_root


def _dialog(runner) -> object:
    from ui.widgets.job_center import JobCenterDialog

    return JobCenterDialog(runner)


def test_job_center_offers_compare_with_multiselect(
        qtbot, studio_app, seeded, runs_root) -> None:
    from ui.workers.job_worker import JobRunner

    dialog = _dialog(JobRunner())
    qtbot.addWidget(dialog)
    # every recorded run is listed
    assert dialog._history.count() == 4
    # multi-select is on and the compare button exists
    assert (dialog._history.selectionMode()
            == dialog._history.SelectionMode.ExtendedSelection)
    assert dialog._compare_btn.text().startswith("Compare selected")

    # select the two compatible boards
    dialog._history.clearSelection()
    for i in range(dialog._history.count()):
        item = dialog._history.item(i)
        if item.text().endswith(("descriptors board", "fingerprints board")):
            item.setSelected(True)
    selected = dialog._history.selectedItems()
    assert len(selected) == 2

    # under-selection shows the honest note (0 selected)
    dialog._history.clearSelection()
    dialog._compare_selected()
    assert "select two or more" in dialog._compare_note.text()


def test_compare_dialog_renders_groups_and_refusals(
        qtbot, studio_app, seeded) -> None:
    from ui.widgets.experiment_compare import ExperimentCompareDialog

    from pharmaml.studio.comparison import _run_view, latest_runs

    records = []
    for family in ("ml-evaluation", "pkpd"):
        records.extend(latest_runs(family, limit=5))
    views = [_run_view(r) for r in records]
    dialog = ExperimentCompareDialog(views)
    qtbot.addWidget(dialog)
    text = dialog._text.toPlainText()
    # the two boards form one ranked group with a verdict + deltas
    assert "task=classification · metric=roc_auc_ovo" in text
    assert "best on roc_auc_ovo: descriptors board" in text
    assert "Δ roc_auc_ovo vs best" in text
    # the r2 run and the pkpd run are NOT merged into that ranking
    assert "task=regression" in text
    assert "not compared" in text
    # headline counts the groups and the refused pairs
    head = dialog._headline.text()
    assert "compatibility group" in head
    assert "not ranked" in head


def test_compare_dialog_two_compatible_runs_rank_cleanly(
        qtbot, studio_app, seeded) -> None:
    from ui.widgets.experiment_compare import ExperimentCompareDialog

    from pharmaml.studio.comparison import _run_view, latest_runs

    records = latest_runs("ml-evaluation", limit=5)
    views = [_run_view(r) for r in records]
    dialog = ExperimentCompareDialog(views)
    qtbot.addWidget(dialog)
    text = dialog._text.toPlainText()
    assert "best on roc_auc_ovo: descriptors board" in text
    # the regression run is present but never ranked against roc_auc
    assert "no ranking" in text or "not compared" in text


def test_job_center_detail_still_works_after_the_compare_wiring(
        qtbot, studio_app, seeded) -> None:
    from ui.workers.job_worker import JobRunner

    dialog = _dialog(JobRunner())
    qtbot.addWidget(dialog)
    dialog._history.setCurrentRow(0)
    detail = dialog._detail.toPlainText()
    assert detail  # provenance JSON still renders on select
