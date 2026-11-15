"""(part8 Phase D) the Chemistry Lab's structured feature configuration.

Pins the commission contract on the REAL page: the fingerprint radius /
bit-width / count controls and the exact descriptor selection reach the
backend through ONE validated FeatureConfig, and the generation config
persists with the outcome (§7.4 reproducibility).
"""

from __future__ import annotations

import pandas as pd
import pytest


def _page_with_frame(frame: pd.DataFrame, column: str = "smiles"):
    from ui.pages.chemistry import ChemistryWorkbenchPage

    page = ChemistryWorkbenchPage()
    page._frame = frame
    page._smiles_combo.blockSignals(True)
    page._smiles_combo.clear()
    page._smiles_combo.addItem(column)
    page._smiles_combo.setCurrentText(column)
    page._smiles_combo.blockSignals(False)
    return page


@pytest.fixture()
def guard_off(monkeypatch):
    import ui.pages.chemistry as chem

    monkeypatch.setattr(chem, "_meminfo_mb", lambda name: None)
    return chem


def test_descriptor_picker_lists_the_real_catalog(studio_app) -> None:
    page = _page_with_frame(pd.DataFrame({"smiles": ["CCO"]}))
    count = page._descriptor_picker.count()
    assert count > 40  # the real RDKit property catalog
    assert page._descriptor_picker.itemText(0).startswith("all properties (")
    # each non-all entry carries its exact name in the item data
    assert page._descriptor_picker.itemData(1) is not None


def test_config_reaches_backend_and_persists(qtbot, studio_app, guard_off, monkeypatch) -> None:
    import pharmaml.chem.features as features_mod
    from pharmaml.chem.features import FeatureReport

    page = _page_with_frame(pd.DataFrame({"smiles": ["CCO", "c1ccccc1O"]}))
    qtbot.addWidget(page)

    # the exact selections under test
    page._families_combo.setCurrentText("descriptors (fast)")
    page._fp_radius.setValue(3)
    page._fp_bits.setValue(512)
    page._fp_counts.setChecked(True)
    catalog = page._descriptor_picker.currentData()  # None = all, fine
    assert catalog is None  # scope 'all' to start

    seen: dict[str, object] = {}

    def fake_compute(smiles, families=None, n_jobs=1, progress=None, config=None, **kw):
        seen["config"] = config
        report = FeatureReport(
            smiles=list(smiles), dropped=[], families={"descriptors": 2}
        )
        return pd.DataFrame({"desc_a": [1.0, 2.0]}), report

    monkeypatch.setattr(features_mod, "compute_feature_frame", fake_compute)

    page._start_features()
    qtbot.waitSignal(page._runner.job_ended, timeout=30_000).wait()

    cfg = seen["config"]
    assert isinstance(cfg, features_mod.FeatureConfig)
    assert cfg.morgan_radius == 3
    assert cfg.morgan_bits == 512
    assert cfg.as_counts is True
    assert cfg.descriptors is None
    assert "descriptors" in cfg.families
    assert cfg.n_jobs == -1  # the NJobsField default flows through


def test_exact_descriptor_subset_flows_to_backend(qtbot, studio_app, guard_off, monkeypatch) -> None:
    import pharmaml.chem.features as features_mod
    from pharmaml.chem.features import FeatureReport

    page = _page_with_frame(pd.DataFrame({"smiles": ["CCO"]}))
    qtbot.addWidget(page)
    page._families_combo.setCurrentText("descriptors (fast)")
    # pick an exact property (index 1 = the first catalog name)
    page._descriptor_picker.setCurrentIndex(1)
    chosen = page._descriptor_picker.currentData()
    assert chosen is not None

    seen: dict[str, object] = {}

    def fake_compute(smiles, families=None, n_jobs=1, progress=None, config=None, **kw):
        seen["descriptors"] = config.descriptors
        report = FeatureReport(
            smiles=list(smiles), dropped=[], families={"descriptors": 1}
        )
        return pd.DataFrame({"desc_x": [1.0]}), report

    monkeypatch.setattr(features_mod, "compute_feature_frame", fake_compute)

    page._start_features()
    qtbot.waitSignal(page._runner.job_ended, timeout=30_000).wait()
    assert tuple(seen["descriptors"]) == tuple(chosen)


def test_invalid_refusal_before_worker(qtbot, studio_app, guard_off, monkeypatch) -> None:
    import pharmaml.chem.features as features_mod

    page = _page_with_frame(pd.DataFrame({"smiles": ["CCO"]}))
    qtbot.addWidget(page)

    def must_not_run(*a, **kw):  # pragma: no cover - the guard must fire
        raise AssertionError("backend must not see an invalid config")

    monkeypatch.setattr(features_mod, "compute_feature_frame", must_not_run)
    monkeypatch.setattr(
        "pharmaml.chem.features.FeatureConfig.__post_init__",
        lambda self: (_ for _ in ()).throw(ValueError("bad config (test)")),
    )
    page._start_features()
    assert not page._runner.busy
    assert "refused" in page.report_text().lower()
