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


# -- part5 round 4 (§12): export validation on the MD figure paths -------------


def _plotted_page(studio_app, qtbot, tmp_path):
    from ui.pages.molecular_dynamics import MolecularDynamicsPage

    page = MolecularDynamicsPage()
    qtbot.addWidget(page)
    page.load_path(_study_folder(tmp_path))
    qtbot.waitUntil(lambda: page._study is not None, timeout=10000)
    page._plot_series()
    assert page._figure is not None
    return page


def test_export_reports_validated_format_and_dimensions(studio_app, qtbot, tmp_path):
    """§12: every exported figure is reopened (magic-sniffed for raster
    formats, %PDF header for PDF) and the success line carries the real
    format, dimensions and byte size -- never a bare 'Saved'."""
    page = _plotted_page(studio_app, qtbot, tmp_path)
    for ext, fmt in (("png", "PNG"), ("svg", "SVG"), ("pdf", "PDF")):
        target = tmp_path / f"figure.{ext}"
        page.export_path(target)
        status = page._status.text()
        assert "validated" in status, status
        assert fmt in status, status
        assert str(target) in status
    # the PNG line carries real pixel dimensions
    page.export_path(tmp_path / "figure2.png")
    assert "x" in page._status.text() and "bytes" in page._status.text()


def test_export_validation_refuses_a_corrupt_file(studio_app, qtbot, tmp_path, monkeypatch):
    """A save that produces unreadable bytes is an honest FAILURE, not a
    'Saved' over a broken file -- the §12 reopen-check discipline."""
    page = _plotted_page(studio_app, qtbot, tmp_path)

    import ui.pages.molecular_dynamics as md_page_mod

    def broken_save(fig, path):
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"this is not an image at all")
        return target

    monkeypatch.setattr(md_page_mod, "save_figure", broken_save)
    page.export_path(tmp_path / "corrupt.png")
    status = page._status.text()
    assert "FAILED" in status, status
    assert "magic bytes" in status, status
    assert "Saved" not in status


def test_export_validation_catches_empty_file(studio_app, qtbot, tmp_path, monkeypatch):
    """An empty file (truncated save) is refused with its reason."""
    page = _plotted_page(studio_app, qtbot, tmp_path)

    import ui.pages.molecular_dynamics as md_page_mod

    def empty_save(fig, path):
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
        return target

    monkeypatch.setattr(md_page_mod, "save_figure", empty_save)
    page.export_path(tmp_path / "empty.png")
    assert "FAILED" in page._status.text()
    assert "empty" in page._status.text()


def test_export_records_real_evidence_for_the_card(studio_app, qtbot, tmp_path, monkeypatch):
    """The §11 viz-run record now carries format/width/height/bytes from
    the VALIDATED file (not just a name) -- the Visualization card's
    evidence is the checked artifact."""
    recorded = []
    monkeypatch.setattr(
        "pharmaml.studio.record_run",
        lambda family, label, summary=None: recorded.append((family, label, summary)),
    )
    page = _plotted_page(studio_app, qtbot, tmp_path)
    target = tmp_path / "evidence.png"
    page.export_path(target)
    assert recorded, "the export must reach record_run"
    family, _label, summary = recorded[-1]
    assert family == "viz-export"
    assert summary["format"] == "PNG"
    assert summary["width"] > 0 and summary["height"] > 0
    assert summary["size_bytes"] == target.stat().st_size
    assert summary["kind"] == "md figure"
