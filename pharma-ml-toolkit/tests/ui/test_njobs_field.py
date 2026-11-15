"""(part8 Phase 6) The validated manual ``n_jobs`` field.

Pins the commission contract:
- any nonzero integer is accepted (including ``-1`` = all cores),
- ``0`` is rejected (invalid under joblib semantics),
- blanks / non-integers are rejected without raising,
- the Chemistry Lab passes the user's value through to the backend and
  refuses to start the job on an invalid entry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ui.widgets.njobs_field import NJobsField


class TestNJobsField:
    def test_presets_and_default(self, qtbot, studio_app) -> None:
        field = NJobsField()
        qtbot.addWidget(field)
        assert field.currentText() == "-1"  # all cores, the offered default
        assert [field.itemText(i) for i in range(field.count())] == [
            "-1", "1", "2", "4", "8",
        ]

    @pytest.mark.parametrize("raw,expected", [
        ("-1", -1), ("1", 1), ("2", 2), ("4", 4), ("16", 16), ("-4", -4),
        (" 8 ", 8),
    ])
    def test_valid_nonzero_integers(self, qtbot, studio_app, raw, expected) -> None:
        field = NJobsField()
        qtbot.addWidget(field)
        field.setCurrentText(raw)
        assert field.is_valid()
        assert field.value() == expected

    @pytest.mark.parametrize("raw", ["0", "-0", "0 ", "abc", "1.5", "", " ", "1e2"])
    def test_invalid_entries_answer_none(self, qtbot, studio_app, raw) -> None:
        field = NJobsField()
        qtbot.addWidget(field)
        field.setCurrentText(raw)
        assert not field.is_valid()
        assert field.value() is None

    def test_editable_text_is_not_silently_rewritten(self, qtbot, studio_app) -> None:
        field = NJobsField()
        qtbot.addWidget(field)
        field.setCurrentText("3")
        assert field.currentText() == "3"  # explicit user entry survives


class TestChemistryLabNJobs:
    def test_user_value_reaches_the_backend(self, qtbot, studio_app, monkeypatch) -> None:
        import pharmaml.chem.features as features_mod
        import ui.pages.chemistry as chem
        from pharmaml.chem.features import FeatureReport

        frame = pd.DataFrame({"smiles": ["CCO", "c1ccccc1O", "CCN"]})
        page = chem.ChemistryWorkbenchPage()
        qtbot.addWidget(page)
        page._frame = frame
        page._smiles_combo.blockSignals(True)
        page._smiles_combo.clear()
        page._smiles_combo.addItem("smiles")
        page._smiles_combo.setCurrentText("smiles")
        page._smiles_combo.blockSignals(False)

        seen: dict[str, object] = {}

        def fake_compute(smiles, families=None, n_jobs=1, progress=None, **kw):
            seen["n_jobs"] = n_jobs
            report = FeatureReport(
                smiles=list(smiles), dropped=[], families={"descriptors": 1}
            )
            return pd.DataFrame({"MolWt": [46.07, 108.14, 45.08]}), report

        monkeypatch.setattr(features_mod, "compute_feature_frame", fake_compute)
        monkeypatch.setattr(chem, "_meminfo_mb", lambda name: None)

        page._jobs_field.setCurrentText("6")  # a value the old combo never offered
        page._start_features()
        qtbot.waitSignal(page._runner.job_ended, timeout=30_000).wait()
        assert seen["n_jobs"] == 6  # the explicit user configuration arrived

    def test_invalid_n_jobs_refuses_before_the_job(self, qtbot, studio_app, monkeypatch) -> None:
        import ui.pages.chemistry as chem

        frame = pd.DataFrame({"smiles": ["CCO"]})
        page = chem.ChemistryWorkbenchPage()
        qtbot.addWidget(page)
        page._frame = frame
        page._smiles_combo.blockSignals(True)
        page._smiles_combo.clear()
        page._smiles_combo.addItem("smiles")
        page._smiles_combo.setCurrentText("smiles")
        page._smiles_combo.blockSignals(False)

        called = {"n": 0}

        import pharmaml.chem.features as features_mod

        def fake_compute(*a, **kw):  # must never run
            called["n"] += 1
            raise AssertionError("backend must not see an invalid n_jobs")

        monkeypatch.setattr(features_mod, "compute_feature_frame", fake_compute)

        page._jobs_field.setCurrentText("0")  # joblib-invalid
        page._start_features()
        assert called["n"] == 0
        assert not page._runner.busy
        assert "nonzero" in page.report_text()
