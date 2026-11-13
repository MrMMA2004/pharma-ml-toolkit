"""Phase 5 (part6 §16/§17 E-F) GUI pins: settings surfaces, glossary
tooltips on the four previously-thin pages, and the accessibility
labels on the shared chrome.

Runs under the tests/ui offscreen bootstrap (see tests/ui/conftest.py).
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel


# -- the Settings page: presets, probe, doctor (§17 E + §12) --------------------


def test_settings_page_shows_presets_probe_and_doctor(studio_app, qtbot) -> None:
    from ui.pages.settings_page import SettingsPage
    from ui.theme_manager import ThemeManager

    page = SettingsPage(ThemeManager(studio_app))
    qtbot.addWidget(page)

    # the preset picker carries both policies with their trade notes
    assert page._preset_combo.count() == 2
    labels = " ".join(page._preset_combo.itemText(i) for i in range(2))
    assert "Balanced" in labels and "Low-memory" in labels
    assert "chunk" in labels  # the honest trade is spelled out

    # switching persists through the session echo
    page._preset_combo.setCurrentIndex(1)
    from pharmaml.utils.resources import stored_preset

    assert stored_preset() == "eco"
    page._preset_combo.setCurrentIndex(0)
    assert stored_preset() == "balanced"

    # the live probe readout never claims a GPU (§17 E honesty)
    text = _all_label_text(page)
    assert "CPUs" in text
    assert "GPU" in text and "not probed" in text
    assert "source:" in text  # where the memory number came from

    # the doctor panel mirrors the CLI doctor's verdicts honestly
    assert "Environment doctor:" in text
    assert ("All launch-environment checks pass." in text
            or "✗" in text)  # failures listed, never silently hidden


# -- §17 F: glossary tooltips landed on the four thin pages ----------------------


def test_apply_help_sets_the_tooltip_and_reports_misses(studio_app) -> None:
    from PySide6.QtWidgets import QLabel

    from pharmaml.studio.glossary import apply_help, tooltip_text

    label = QLabel()
    assert apply_help(label, "morgan_fingerprint") is True
    assert label.toolTip() == tooltip_text("morgan_fingerprint")
    assert apply_help(label, "bogus") is False  # a bug to fix, not a no-op


def test_pkpd_page_controls_carry_glossary_help(studio_app, qtbot) -> None:
    from ui.pages.pkpd import PKPDWorkbenchPage

    page = PKPDWorkbenchPage()
    qtbot.addWidget(page)
    assert "AUC" in page._nca_concs.toolTip() or "exposure" in page._nca_concs.toolTip()
    assert page._nca_times.toolTip()  # the t½/terminal-fit caveat
    assert page._ev_dose.toolTip()    # the MABEL anchor
    assert page._regimen_entries["cl"].toolTip()  # CL units + meaning


def test_md_page_metrics_carry_glossary_help(studio_app, qtbot) -> None:
    from ui.pages.molecular_dynamics import MolecularDynamicsPage

    page = MolecularDynamicsPage()
    qtbot.addWidget(page)
    assert "RMSD" in page._metric.toolTip()
    assert "RMSF" in page._second_metric.toolTip()
    assert "interaction" in page._ligand.toolTip().lower()


def test_docking_and_generative_carry_glossary_help(studio_app, qtbot) -> None:
    from ui.pages.docking import DockingWorkbenchPage
    from ui.pages.generative import GenerativeWorkbenchPage

    dock = DockingWorkbenchPage()
    qtbot.addWidget(dock)
    assert "PLIF" in dock._plif_btn.toolTip()
    assert "pose" in dock._pose_combo.toolTip().lower()

    gen = GenerativeWorkbenchPage()
    qtbot.addWidget(gen)
    assert "Monte-Carlo" in gen._mcts_iterations.toolTip()
    assert "seed" in gen._seed_spin.toolTip().lower()


# -- accessibility: the shared chrome announces itself --------------------------


def test_top_bar_labels_have_accessible_names(studio_app, qtbot) -> None:
    from ui.widgets.top_bar import TopBar

    bar = TopBar()
    qtbot.addWidget(bar)
    bar.set_location("Nine-card home")
    assert bar.location.accessibleName() == "Current location: Nine-card home"
    bar.set_jobs("Running: benchmark")
    assert bar.jobs_label.accessibleName() == "Job status: Running: benchmark"
    bar.set_jobs("")
    assert bar.jobs_label.accessibleName() == "Job status: idle"


def test_notifications_announce_their_severity(studio_app, qtbot) -> None:
    from ui.widgets.notification_widget import NotificationBar

    bar = NotificationBar("training finished", kind="success")
    qtbot.addWidget(bar)
    assert bar.accessibleName() == "Success notification: training finished"
    warn = NotificationBar("engine missing", kind="error")
    qtbot.addWidget(warn)
    assert "Error notification" in warn.accessibleName()


def _all_label_text(widget) -> str:
    return "\n".join(l.text() for l in widget.findChildren(QLabel))
