"""Phase 4 (part6 §7-10): global search, job center, shared workspace.

The round-21 pins, over the real backends:

* §7 — the search index is pure (no QApplication); all EIGHT example
  queries from the directive land their top hit on the right section;
  the palette dialog navigates via the main window's public API; the
  dependency-honesty tag distinguishes "needs <dep>" from missing
  functionality.
* §10 — the job center shows the live job from the real runner state
  (never fabricated), Stop is the cooperative one, and the history
  rows come from ``pharmaml.studio.runs`` with the provenance detail.
* §8 — the workspace backend records datasets/bundles/artifacts in a
  file-based JSON registry (env-overridable path), refuses unknown
  kinds, de-duplicates by path, and never requires a project.

The runs/workspace directories are env-isolated per test so the suite
never touches the developer's real ``~/.pharmaml``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_TK = Path(__file__).resolve().parents[2]


# -- §7: the pure search index -------------------------------------------------


def test_index_builds_without_qapplication() -> None:
    """The index is pure: builds with no QApplication instance."""
    from ui.search_index import build_search_index

    index = build_search_index()
    kinds = {row.kind for row in index}
    assert kinds == {"capability", "card", "section"}
    # every registry row + 9 cards + 18 sections
    from pharmaml.studio.capability_registry import CAPABILITIES

    caps = [row for row in index if row.kind == "capability"]
    assert len(caps) == len(CAPABILITIES)
    assert len([row for row in index if row.kind == "card"]) == 9
    assert len([row for row in index if row.kind == "section"]) == 18


def test_the_eight_directive_example_queries_land_right() -> None:
    """§7's examples: query without knowing the card, land on the section."""
    from ui.search_index import build_search_index, search

    index = build_search_index()
    examples = {
        "morgan fingerprint": "chemistry",
        "conformal prediction": "mlworkbench",
        "gnn": "deep_learning",
        "mcts": "generative",
        "vina": "docking",
        "pbpk": "pkpd",
        "rmsd": "molecular_dynamics",
        "tiff": "imagelab",
    }
    for query, wanted_route in examples.items():
        hits = search(index, query)
        assert hits, f"{query!r}: no hits at all"
        assert hits[0].route == wanted_route, (
            f"{query!r}: top hit routes to {hits[0].route!r}, "
            f"directive expects {wanted_route!r}"
        )
        # §7: every result displays the parent card and the destination
        assert hits[0].detail


def test_search_distinguishes_missing_deps_from_missing_features() -> None:
    """A capability with an absent optional dep stays listed, tagged."""
    from ui import search_index as si

    row = next(r for r in si.build_search_index()
               if r.capability_id == "data.fingerprints")
    # rdkit IS installed in the QA env: no tag
    assert si.dependency_available("rdkit") is True
    assert row.needs_dep == ""
    # the honesty path itself: a hypothetical missing dep tags the row
    tagged = si.SearchRow(
        kind="capability", label="X", detail="d", route="r", card="c",
        capability_id="x", needs_dep="torch",
    )
    assert "needs torch" in tagged.status_line()


def test_the_empty_query_is_a_browsable_launch_pad() -> None:
    from ui.search_index import build_search_index, search

    rows = search(build_search_index(), "")
    assert rows
    assert all(r.kind in ("card", "section") for r in rows)
    assert rows[0].kind == "card"  # the nine cards lead


def test_the_palette_dialog_navigates(studio_app, qtbot) -> None:
    """Ctrl+K's dialog: search, Enter, and the main window routes."""
    from PySide6.QtCore import Qt

    from ui.main_window import MainWindow
    from ui.search_palette import SearchPalette

    window = MainWindow()
    qtbot.addWidget(window)
    palette = SearchPalette(window)
    qtbot.addWidget(palette)
    palette.show()
    palette._box.setText("rmsd")
    assert palette._list.count() >= 1
    # the top hit is the MD analysis row; accepting carries its route out
    assert palette._list.currentItem().data(Qt.ItemDataRole.UserRole) == "molecular_dynamics"
    palette._accept_current()
    assert palette.chosen_route == "molecular_dynamics"
    # the main window honors the choice through its public API
    window.go_to(palette.chosen_route)
    from ui.main_window import SECTIONS

    assert window.pages.currentIndex() == [k for k, _t in SECTIONS].index("molecular_dynamics")


# -- §10: the job center ----------------------------------------------------------


def test_the_job_center_history_reads_real_runs(studio_app, qtbot, tmp_path,
                                                monkeypatch) -> None:
    monkeypatch.setenv("PHARML_STUDIO_RUNS_DIR", str(tmp_path))
    from pharmaml.studio import record_run

    written = record_run(
        "ml-evaluation", "family benchmark · pIC50",
        {"eval_kind": "cv", "folds": 5},
    )
    assert written is not None and written.is_file()

    from ui.widgets.job_center import JobCenterDialog
    from ui.workers.job_worker import JobRunner

    runner = JobRunner()
    dialog = JobCenterDialog(runner)
    qtbot.addWidget(dialog)
    assert dialog._history.count() == 1
    dialog._history.setCurrentRow(0)
    detail = dialog._detail.toPlainText()
    assert "family benchmark · pIC50" in detail
    assert "ml-evaluation" in detail
    assert "folds: 5" in detail
    assert "toolkit version:" in detail  # §9 provenance field
    # §10 honesty: the live card says idle, Stop disabled — nothing fabricated
    assert "No job running" in dialog._live_label.text()
    assert not dialog._stop_btn.isEnabled()


def test_the_job_center_tracks_a_live_job(studio_app, qtbot, monkeypatch,
                                          tmp_path) -> None:
    monkeypatch.setenv("PHARML_STUDIO_RUNS_DIR", str(tmp_path))
    from PySide6.QtCore import QThreadPool

    from ui.widgets.job_center import JobCenterDialog
    from ui.workers.job_worker import JobRunner

    runner = JobRunner(pool=QThreadPool())
    dialog = JobCenterDialog(runner)
    qtbot.addWidget(dialog)

    started = []
    runner.job_started.connect(lambda name: started.append(name))

    def slow_job(progress):
        import time as _t

        _t.sleep(0.4)
        return 42

    runner.submit("a real job", slow_job)
    assert started == ["a real job"]
    assert "a real job" in dialog._live_label.text()
    assert dialog._stop_btn.isEnabled()  # the cooperative Stop is armed
    # the honest one-job-at-a-time contract
    assert runner.submit("second job", slow_job) is None
    qtbot.waitUntil(lambda: not runner.busy, timeout=10_000)
    assert "No job running" in dialog._live_label.text()


# -- §8: the shared workspace --------------------------------------------------------


def test_the_workspace_records_and_lists(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PHARML_STUDIO_WORKSPACE", str(tmp_path / "ws.json"))
    import pandas as pd

    from pharmaml.studio import (
        current_project,
        entries,
        note,
        set_current_project,
        summary_counts,
    )

    csv = tmp_path / "qsar.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(csv, index=False)

    assert note("dataset", csv, note="3 rows") is True
    assert note("bundle", tmp_path / "model.joblib", with_hash=False) is True
    assert note("artifact", tmp_path / "figure.png", with_hash=False) is True

    counts = summary_counts()
    assert counts == {"dataset": 1, "bundle": 1, "artifact": 1}

    ds = entries("dataset")
    assert len(ds) == 1 and ds[0].name == "qsar.csv"
    assert len(ds[0].sha256) == 64  # the canonical sha-256 was recorded
    assert ds[0].note == "3 rows"

    # projects are OPTIONAL — empty by default, settable, never forced
    assert current_project() == ""
    assert set_current_project("campaign-2026-11") is True
    assert current_project() == "campaign-2026-11"


def test_the_workspace_refuses_unknown_kinds(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PHARML_STUDIO_WORKSPACE", str(tmp_path / "ws.json"))
    from pharmaml.studio import note

    with pytest.raises(ValueError):
        note("galaxy", tmp_path / "x.csv")


def test_the_workspace_deduplicates_by_path(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PHARML_STUDIO_WORKSPACE", str(tmp_path / "ws.json"))
    from pharmaml.studio import entries, note

    csv = tmp_path / "same.csv"
    csv.write_text("a\n1\n")
    note("dataset", csv)
    note("dataset", csv, note="second open")
    ds = entries("dataset")
    assert len(ds) == 1
    assert ds[0].note == "second open"  # the newest recording wins


def test_a_corrupt_workspace_file_is_skipped_not_guessed(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PHARML_STUDIO_WORKSPACE", str(tmp_path / "ws.json"))
    (tmp_path / "ws.json").write_text("{not json")
    from pharmaml.studio import entries

    assert entries() == []


def test_the_data_explorer_notes_loaded_datasets(studio_app, qtbot, monkeypatch,
                                                 tmp_path) -> None:
    monkeypatch.setenv("PHARML_STUDIO_WORKSPACE", str(tmp_path / "ws.json"))
    from ui.pages.data_explorer import DataExplorerPage

    page = DataExplorerPage()
    qtbot.addWidget(page)
    page._load_builtin("qsar_classification.csv")
    # (part8 B) the load is async on the JobRunner worker: wait for the
    # worker, then let the queued finished-signal deliver (one extra
    # pump after busy clears -- the signal is queued, not synchronous).
    qtbot.waitUntil(lambda: not page._runner.busy, timeout=10_000)
    qtbot.wait(50)
    assert page._frame is not None

    from pharmaml.studio import entries

    ds = entries("dataset")
    assert any(e.name == "qsar_classification.csv" for e in ds)
    recorded = next(e for e in ds if e.name == "qsar_classification.csv")
    assert recorded.path.endswith("qsar_classification.csv")


def test_the_dashboard_shows_workspace_recents(studio_app, qtbot, monkeypatch,
                                               tmp_path) -> None:
    monkeypatch.setenv("PHARML_STUDIO_WORKSPACE", str(tmp_path / "ws.json"))
    import pandas as pd

    from pharmaml.studio import note

    csv = tmp_path / "recents.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv, index=False)
    note("dataset", csv)

    from ui.pages.dashboard import DashboardPage

    page = DashboardPage()
    qtbot.addWidget(page)
    text = page._find_text_of_all_labels() if hasattr(page, "_find_text_of_all_labels") else _all_label_text(page)
    assert "Shared workspace" in text
    assert "recents.csv" in text


def _all_label_text(widget) -> str:
    from PySide6.QtWidgets import QLabel

    return "\n".join(l.text() for l in widget.findChildren(QLabel))


def test_workspace_star_toggle_on_the_dashboard(
        studio_app, qtbot, monkeypatch, tmp_path) -> None:
    """§17G: the ★ on a workspace row toggles a real favorite."""
    monkeypatch.setenv("PHARML_STUDIO_WORKSPACE", str(tmp_path / "ws.json"))
    import pandas as pd
    from PySide6.QtWidgets import QToolButton

    from pharmaml.studio import note
    from pharmaml.studio.workspace import is_favorite

    csv = tmp_path / "starrable.csv"
    pd.DataFrame({"a": [1]}).to_csv(csv, index=False)
    note("dataset", csv)

    from ui.pages.dashboard import DashboardPage

    page = DashboardPage()
    qtbot.addWidget(page)
    stars = [b for b in page.findChildren(QToolButton)]
    assert stars, "the workspace rows carry star toggles"
    star = stars[0]
    assert star.text() == "☆"  # unstarred
    star.click()
    assert star.text() == "★"
    assert is_favorite(str(csv))
    star.click()
    assert star.text() == "☆"
    assert not is_favorite(str(csv))
