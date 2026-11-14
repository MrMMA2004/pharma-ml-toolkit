"""Per-page chart export (R30-c, part7 §17 engine).

Every chart-bearing workspace carries an "Export chart…" button
wired to its PlotPanel's ``export()`` — the §17 raster engine under
the live studio palette.  The buttons are findable by objectName,
one per plot on multi-plot pages, and a real export round-trips a
verified PNG at the §23.3 legibility floor.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QPushButton

from ui.widgets.chart_export import attach_chart_export
from ui.widgets.plot_widget import PlotPanel
from ui.widgets.responsive import LEGIBILITY_FLOOR

#: page class -> expected export-button count (one per PlotPanel).
EXPECT = [
    ("ui.pages.data_explorer:DataExplorerPage", 1),
    ("ui.pages.chemistry:ChemistryWorkbenchPage", 1),
    ("ui.pages.pkpd:PKPDWorkbenchPage", 1),
    ("ui.pages.biology:BiologyWorkbenchPage", 1),
    ("ui.pages.docking:DockingWorkbenchPage", 1),
    ("ui.pages.model_analysis:ModelAnalysisPage", 3),
    ("ui.pages.generative:GenerativeWorkbenchPage", 1),
    ("ui.pages.unsupervised:UnsupervisedWorkbenchPage", 1),
    ("ui.pages.deep_learning:DeepLearningWorkbenchPage", 1),
]


def _load(page_spec: str):
    import importlib

    module_name, class_name = page_spec.split(":")
    return getattr(importlib.import_module(module_name), class_name)


@pytest.mark.parametrize("page_spec,count", EXPECT)
def test_page_carries_its_export_buttons(studio_app, page_spec, count) -> None:
    page = _load(page_spec)()
    try:
        buttons = [
            b for b in page.findChildren(QPushButton)
            if b.objectName() == "chartExportButton"
        ]
        assert len(buttons) == count, (
            f"{page_spec}: expected {count} export button(s), found {len(buttons)}"
        )
        for button in buttons:
            # M-3: buttons may carry a distinguishing label
            # ("Export parity…") instead of the generic text
            assert button.text().startswith("Export "), button.text()
            assert button.text().endswith("…")
            assert button.isEnabled()
            assert button.toolTip()  # the §17 engine is explained
    finally:
        page.close()


def test_a_real_export_round_trips_a_verified_png(
    studio_app, tmp_path: Path
) -> None:
    """The wiring, not just the button: drive the handler's export
    path directly (no file dialog offscreen) and verify the §17
    engine's checks passed — the PNG exists, is non-trivial and meets
    the legibility floor."""
    import numpy as np

    panel = PlotPanel(title="export round-trip")
    x = np.linspace(0.0, 1.0, 24)
    panel.line(x, np.sin(x * 6.0), label="sine")
    target = tmp_path / "round-trip.png"
    result = panel.export_image(target, min_size=LEGIBILITY_FLOOR)
    assert result.path == target
    assert result.format == "PNG"
    assert result.width >= LEGIBILITY_FLOOR.width()
    assert result.height >= LEGIBILITY_FLOOR.height()
    assert result.size_bytes > 4_000  # a real curve, not a blank frame
    # the engine already reopened + magic-sniffed it; re-verify here
    head = target.read_bytes()[:8]
    assert head == b"\x89PNG\r\n\x1a\n"


def test_attach_reports_failure_to_the_status_label(
    studio_app, tmp_path: Path
) -> None:
    """A failed export surfaces as status text, never a crash."""
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

    page = QWidget()
    lay = QVBoxLayout(page)
    status = QLabel("idle")
    panel = PlotPanel()

    class _Broken:
        def export_image(self, *_args, **_kwargs):
            raise RuntimeError("disk on fire")

    attach_chart_export(page, _Broken(), "broken", layout=lay,
                        status_label=status)
    button = page.findChildren(QPushButton)[0]
    # invoke the handler body without the file dialog: call through
    # the click with a monkeypatched dialog returning a temp path
    from PySide6.QtWidgets import QFileDialog

    original = QFileDialog.getSaveFileName
    QFileDialog.getSaveFileName = staticmethod(
        lambda *a, **k: (str(tmp_path / "x.png"), "")
    )
    try:
        button.click()
    finally:
        QFileDialog.getSaveFileName = original
    assert status.text().startswith("Chart export FAILED")
    assert "disk on fire" in status.text()
    page.close()


def test_the_floor_is_the_documented_one() -> None:
    """The export rides §23.3's floor, not a magic number."""
    assert (LEGIBILITY_FLOOR.width(), LEGIBILITY_FLOOR.height()) == (640, 480)
