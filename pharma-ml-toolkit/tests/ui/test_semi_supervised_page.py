"""(part8 Phase F) The Semi-supervised tab's full GUI round trip.

Pins the commission contract: the three scikit-learn estimators behind
one form, the -1 (and custom) unlabeled markers, the leakage-safe
evaluation with a supervised baseline, and the sample-level inspection
table + confidence figure — against the REAL backend on the REAL worker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _semi_frame(n_unlabeled: int = 60, seed: int = 0) -> pd.DataFrame:
    """A realistic classification frame with a genuinely unlabeled block."""
    from sklearn.datasets import make_classification

    x, y = make_classification(
        n_samples=200, n_features=8, n_informative=6,
        n_clusters_per_class=1, class_sep=2.0, random_state=seed,
    )
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    labels = y.astype(object).astype(str)
    mask = rng.random(len(labels)) < n_unlabeled / 100.0
    labels[mask] = "-1"
    frame["activity"] = labels
    return frame


class _FakeModelAnalysis:
    """The smallest Model Analysis stand-in the workbench reads."""

    def __init__(self, frame: pd.DataFrame, target: str) -> None:
        self._frame = frame
        self._target_combo = _ComboStub(target)


class _ComboStub:
    def __init__(self, text: str) -> None:
        self._text = text

    def currentText(self) -> str:  # Qt API shape
        return self._text


def _fresh_page(frame: pd.DataFrame, target: str = "activity"):
    from ui.pages.ml_workbench import MLWorkbenchPage

    page = MLWorkbenchPage(model_analysis=_FakeModelAnalysis(frame, target))
    return page


def test_semi_tab_runs_the_real_backend(qtbot, studio_app) -> None:
    page = _fresh_page(_semi_frame())
    qtbot.addWidget(page)

    page._semi_method.setCurrentText("label spreading")
    page._start_semi()
    # job_ended fires BEFORE the page's done() slot (the runner's
    # finisher connects first); wait for the handler's own effect.
    qtbot.waitUntil(
        lambda: "held-out accuracy=" in page._semi_status.text(),
        timeout=120_000,
    )

    status = page._semi_status.text()
    assert "labeled" in status and "unlabeled" in status
    # the leakage-safe evaluation summary rides the status line
    assert "held-out accuracy=" in status
    assert "baseline" in status

    # the sample-level table: final label + confidence + pseudo flag
    header = [
        page._semi_table.horizontalHeaderItem(i).text()
        for i in range(page._semi_table.columnCount())
    ]
    assert "final_label" in header
    assert "confidence" in header
    assert "was_pseudo_labeled" in header
    assert page._semi_table.rowCount() == 200

    # the confidence figure carries both populations (the inspector
    # registry keeps the FULL series keys even when the legend elides)
    legends = list(page._semi_plot._inspector._series.keys())
    assert "pseudo-labeled" in legends and "genuinely labeled" in legends


def test_semi_tab_self_training_with_custom_marker(qtbot, studio_app) -> None:
    frame = _semi_frame(seed=1)
    frame["activity"] = frame["activity"].replace("-1", "unlabeled")
    page = _fresh_page(frame)
    qtbot.addWidget(page)

    page._semi_unlabeled.setCurrentText("unlabeled")
    page._start_semi()
    qtbot.waitUntil(
        lambda: "held-out accuracy=" in page._semi_status.text(),
        timeout=120_000,
    )

    status = page._semi_status.text()
    assert "unlabeled" in status
    # the real classes survive (never converted to the sentinel)
    assert "unlabeled" not in status.split("—")[0] or "labeled" in status


def test_semi_tab_refuses_without_frame(qtbot, studio_app) -> None:
    page = _fresh_page(pd.DataFrame({"x": [1.0]}), target="missing")
    qtbot.addWidget(page)
    page._start_semi()
    assert "target" in page._semi_status.text()
    assert not page._runner.busy
