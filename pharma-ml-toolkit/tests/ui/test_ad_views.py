"""(part8 Phase 10) the completed Applicability-Domain views.

Pins the commission contract: the four required analyses behind the view
selector, honest degradation when the bootstrap half did not run, and
the sample-level inspection dossier — all re-derived from the ONE
stored backend result, never recomputed in the GUI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ad_page(qtbot, n_models: int = 8):
    """A workbench whose Model Analysis page carries a trained-style frame.

    The AD source is 'Model Analysis run' — the training context needs a
    frame + target; the AD job itself runs the REAL backend against a
    Ridge fit inside the source resolution.  We stand in the smallest
    honest surface: run the REAL applicability_domain backend on a
    synthetic regression and hand the result to the view renderer.
    """
    from sklearn.linear_model import Ridge

    from pharmaml.ml import applicability_domain
    from ui.pages.ml_workbench import MLWorkbenchPage

    rng = np.random.default_rng(0)
    x = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120)})
    y = pd.Series(2.0 * x["a"] - x["b"] + rng.normal(scale=0.1, size=120))
    model = Ridge().fit(x, y)
    result = applicability_domain(
        model, x, y, x.iloc[:30], k=3, percentile=95.0,
        n_models=n_models, uncertainty=n_models > 0,
    )
    page = MLWorkbenchPage()
    qtbot.addWidget(page)
    page._ad_result = result
    page._ad_query_pred = np.asarray(model.predict(x.iloc[:30]), dtype=float).ravel()
    page._ad_frame = result.to_frame()
    return page, result


def test_default_view_inside_outside(qtbot, studio_app) -> None:
    page, _result = _ad_page(qtbot)
    page._render_ad_view("query distance (inside/outside)")
    keys = list(page._ad_plot._inspector._series.keys())
    assert "inside AD" in keys and "outside AD" in keys and "threshold" in keys


def test_training_distance_distribution_view(qtbot, studio_app) -> None:
    page, _result = _ad_page(qtbot)
    page._render_ad_view("training distance distribution")
    keys = list(page._ad_plot._inspector._series.keys())
    assert "training rows" in keys and "threshold" in keys
    assert page._ad_plot._inspector._series["threshold"].x.size >= 1


def test_distance_vs_predicted_view(qtbot, studio_app) -> None:
    page, _ = _ad_page(qtbot)
    page._render_ad_view("distance vs predicted value")
    keys = list(page._ad_plot._inspector._series.keys())
    assert "inside AD" in keys and "outside AD" in keys


def test_uncertainty_view_and_honest_degradation(qtbot, studio_app) -> None:
    # with the bootstrap half: two populations + threshold
    page, _ = _ad_page(qtbot, n_models=8)
    page._render_ad_view("uncertainty distribution")
    keys = list(page._ad_plot._inspector._series.keys())
    assert "low uncertainty" in keys and "high uncertainty" in keys

    # kNN-only pass: the view refuses honestly, nothing invented
    page2, _ = _ad_page(qtbot, n_models=0)
    page2._render_ad_view("uncertainty distribution")
    assert "kNN-only" in page2._ad_status.text()


def test_combined_reliability_view(qtbot, studio_app) -> None:
    page, _result = _ad_page(qtbot, n_models=8)
    page._render_ad_view("combined reliability")
    keys = list(page._ad_plot._inspector._series.keys())
    assert "reliable (in AD, low unc.)" in keys
    # the backend's own decision rule populates every quadrant label
    assert "in AD, high uncertainty" in keys
    assert "out of AD, low uncertainty" in keys
    assert "out of AD, high uncertainty" in keys


def test_sample_inspection_dossier(qtbot, studio_app, monkeypatch) -> None:
    from PySide6.QtWidgets import QDialog, QLabel

    page, _result = _ad_page(qtbot, n_models=8)
    shown: dict[str, object] = {}

    def fake_exec(self):
        shown["title"] = self.windowTitle()
        shown["text"] = " ".join(w.text() for w in self.findChildren(QLabel))
        return 0

    monkeypatch.setattr(QDialog, "exec", fake_exec)
    page._ad_inspect_row(0, 0)
    assert "sample inspection" in str(shown.get("title", ""))
    text = str(shown.get("text", ""))
    assert "predicted" in text and "kNN distance" in text
    assert "inside AD" in text
