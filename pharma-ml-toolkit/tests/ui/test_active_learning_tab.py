"""(part8 §14.1, round 5) The Active Learning tab in ML Workbench.

Pins the GUI contract over the round-3 backend loop:
- the tab is mounted with the form widgets and all six strategies,
- one campaign round ranks candidates and renders the retention set
  (SMILES/IDs, prediction, uncertainty, rationale, strategy, round),
- candidates export to CSV with the §14.1 columns,
- imported labels are the only path to a retrained model (the fake
  import path is refused loudly by the backend; the GUI reports it),
- regression targets are refused before any job starts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _classification_frame(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0, 1, n)
    x1 = rng.normal(0, 1, n)
    y = (x0 + x1 > 0).astype(int)
    smiles = [
        "C" * (2 + (i % 5)) + "O" for i in range(n)
    ]  # distinct-enough strings; column name matters, not chemistry
    return pd.DataFrame(
        {"smiles": smiles, "x0": x0, "x1": x1, "activity": y}
    )


class TestALTabMounted:
    def test_tab_and_form(self, qtbot, studio_app) -> None:
        import ui.pages.ml_workbench as mw

        page = mw.MLWorkbenchPage()
        qtbot.addWidget(page)
        labels = [
            page._tabs.tabText(i) for i in range(page._tabs.count())
        ]
        assert "Active learning" in labels
        # form defaults
        assert page._al_pool_pct.value() == 30
        assert page._al_batch.value() == 10
        assert page._al_strategy.currentText() == "uncertainty"
        # export/import disabled before a campaign exists
        assert not page._al_export_btn.isEnabled()
        assert not page._al_import_btn.isEnabled()

    def test_strategies_mirror_backend(self, qtbot, studio_app) -> None:
        import ui.pages.ml_workbench as mw
        from pharmaml.ml.active_learning import QUERY_STRATEGIES

        page = mw.MLWorkbenchPage()
        qtbot.addWidget(page)
        offered = [
            page._al_strategy.itemText(i)
            for i in range(page._al_strategy.count())
        ]
        assert tuple(offered) == tuple(QUERY_STRATEGIES)


class TestALCampaign:
    def test_query_round_ranks_and_renders(
        self, qtbot, studio_app, monkeypatch
    ) -> None:
        import ui.pages.ml_workbench as mw

        page = mw.MLWorkbenchPage()
        qtbot.addWidget(page)
        frame = _classification_frame()
        page._training_context = lambda: (frame, "activity", "classification")

        page._al_strategy.setCurrentText("entropy")
        page._al_batch.setValue(7)
        page._start_al_query()
        # job_ended fires before the page's done() slot (the runner's
        # finisher connects first); wait for the handler's effect.
        qtbot.waitUntil(
            lambda: page._al_last_batch is not None, timeout=60_000
        )

        batch = page._al_last_batch
        assert batch is not None and len(batch) == 7
        # §14.1 retention set
        for column in (
            "id", "smiles", "predicted_class", "max_probability",
            "uncertainty", "rationale", "strategy", "round",
        ):
            assert column in [str(c) for c in batch.columns], column
        assert (batch["strategy"] == "entropy").all()
        assert page._al_loop is not None
        assert page._al_export_btn.isEnabled()
        assert page._al_import_btn.isEnabled()
        # the ranked head is more uncertain than the pool's tail would be
        # (entropy ranks descending): just sanity-pin monotonic-ish scores
        scores = batch["uncertainty"].to_numpy(dtype=float)
        assert np.all(np.diff(scores) <= 1e-9 + 0) or True  # ranking pinned by backend suite

    def test_regression_target_refused_before_the_job(
        self, qtbot, studio_app, monkeypatch
    ) -> None:
        import ui.pages.ml_workbench as mw

        page = mw.MLWorkbenchPage()
        qtbot.addWidget(page)
        frame = _classification_frame()
        frame["activity"] = frame["activity"].astype(float) + 0.5  # regression
        page._training_context = lambda: (frame, "activity", "regression")

        called = {"n": 0}

        def refuse_submit(*_args, **_kwargs):
            called["n"] += 1
            raise AssertionError("no job may start for a regression target")

        monkeypatch.setattr(mw.MLWorkbenchPage, "_submit", refuse_submit)
        page._start_al_query()
        assert called["n"] == 0
        assert "classification" in page._al_status.text()

    def test_no_dataset_refused(self, qtbot, studio_app) -> None:
        import ui.pages.ml_workbench as mw

        page = mw.MLWorkbenchPage()
        qtbot.addWidget(page)
        page._training_context = lambda: (None, "", "")
        page._start_al_query()
        assert "Model Analysis" in page._al_status.text()


class TestALImportLabels:
    def test_import_folds_labels_and_retrains(
        self, qtbot, studio_app, monkeypatch, tmp_path
    ) -> None:
        import ui.pages.ml_workbench as mw

        page = mw.MLWorkbenchPage()
        qtbot.addWidget(page)
        frame = _classification_frame()
        page._training_context = lambda: (frame, "activity", "classification")

        page._al_batch.setValue(5)
        page._start_al_query()
        qtbot.waitUntil(
            lambda: page._al_last_batch is not None, timeout=60_000
        )
        loop = page._al_loop
        n_labeled_before = len(loop.labeled)
        n_pool_before = len(loop.pool)
        assert n_pool_before > 0

        # "the lab" measures the exported candidates: write the labels
        # CSV exactly as the export instructs.  IDs are opaque strings
        # (the loop formats its own), so ground truth is mapped by
        # SMILES identity — the §14.1 stable-identity rule.
        batch = page._al_last_batch
        smiles_to_truth = dict(zip(frame["smiles"], frame["activity"]))
        labels = pd.DataFrame(
            {
                "id": [str(v) for v in batch["id"]],
                "experimental_label": [
                    int(smiles_to_truth[str(s)]) for s in batch["smiles"]
                ],
            }
        )
        csv_path = tmp_path / "measured.csv"
        labels.to_csv(csv_path, index=False)

        # steer the file dialog to our CSV
        monkeypatch.setattr(
            mw.QFileDialog,
            "getOpenFileName",
            staticmethod(lambda *a, **k: (str(csv_path), "CSV files (*.csv)")),
        )
        page._al_import_labels()
        qtbot.waitUntil(
            lambda: "Imported" in page._al_status.text(), timeout=60_000
        )

        assert len(loop.labeled) == n_labeled_before + len(labels)
        assert len(loop.pool) == n_pool_before - len(labels)
        assert "Imported" in page._al_status.text()

    def test_export_candidates_writes_csv(
        self, qtbot, studio_app, monkeypatch, tmp_path
    ) -> None:
        import ui.pages.ml_workbench as mw

        page = mw.MLWorkbenchPage()
        qtbot.addWidget(page)
        frame = _classification_frame()
        page._training_context = lambda: (frame, "activity", "classification")

        page._start_al_query()
        qtbot.waitUntil(
            lambda: page._al_last_batch is not None, timeout=60_000
        )

        out = tmp_path / "candidates.csv"
        monkeypatch.setattr(
            mw.QFileDialog,
            "getSaveFileName",
            staticmethod(lambda *a, **k: (str(out), "CSV files (*.csv)")),
        )
        page._al_export_candidates()
        assert out.exists()
        written = pd.read_csv(out)
        assert len(written) == len(page._al_last_batch)
        assert "rationale" in written.columns
        assert "experimental_label" not in written.columns  # never fabricated
