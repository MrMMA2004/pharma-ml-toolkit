"""Offscreen tests for the Studio MD study and plot workflow."""

from __future__ import annotations


def _study_folder(root, *, misaligned=False):
    folder = root / "ligand_A"
    folder.mkdir()
    (folder / "ligand_rmsd.xvg").write_text("0 1\n1 2\n2 3\n")
    (folder / "ligand_sasa.xvg").write_text(
        "0 2\n2 4\n3 6\n" if misaligned else "0 2\n1 4\n2 6\n"
    )
    return root


def test_study_to_plots_and_export(studio_app, qtbot, tmp_path):
    from ui.pages.molecular_dynamics import MolecularDynamicsPage

    page = MolecularDynamicsPage()
    qtbot.addWidget(page)
    page.load_path(_study_folder(tmp_path))
    qtbot.waitUntil(lambda: page._study is not None, timeout=10000)
    assert page._ligand.currentText() == "ligand_A"
    assert page._metric.count() == 2
    page._x_label.setText("Time (ps)")
    page._y_label.setText("Value (nm)")
    page._plot_series()
    assert page._figure.axes[0].get_xlabel() == "Time (ps)"
    assert list(page._figure.axes[0].lines[0].get_ydata()) == [1, 2, 3]
    assert page._export_btn.isEnabled()
    for ext in ("png", "svg", "pdf"):
        target = tmp_path / f"plot.{ext}"
        page.export_path(target)
        assert target.stat().st_size > 0
    page._plot_correlation()
    assert page._figure.axes[0].images[0].get_array()[0, 1] > 0.99
    assert "not evidence of causation" in page._status.text()
    page._x_label.setText("new label")
    assert page._figure is None and not page._export_btn.isEnabled()


def test_mismatch_and_failed_load_clear_previous_figures(studio_app, qtbot, tmp_path):
    from ui.pages.molecular_dynamics import MolecularDynamicsPage

    page = MolecularDynamicsPage()
    qtbot.addWidget(page)
    page.load_path(_study_folder(tmp_path, misaligned=True))
    qtbot.waitUntil(lambda: page._study is not None, timeout=10000)
    page._plot_series()
    assert page._figure is not None
    page._plot_correlation()
    assert page._figure is None and not page._export_btn.isEnabled()
    assert "axes must match" in page._status.text()
    page.load_path(tmp_path / "missing")
    qtbot.waitUntil(lambda: "Could not load study" in page._status.text(), timeout=10000)
    assert page._study is None and not page._series_btn.isEnabled()


def test_navigation_registers_md(main_window):
    main_window.go_to("molecular_dynamics")
    assert main_window.pages.currentWidget() is main_window.molecular_dynamics_page
