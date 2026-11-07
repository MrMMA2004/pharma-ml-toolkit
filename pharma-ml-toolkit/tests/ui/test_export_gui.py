"""GUI-side §17 export tests: the panels and the Image Lab controls.

The widgets route every export through the core engine's
integrity-checked pipeline (``ui/widgets/image_export.py`` grabs the
rendered widget, ``viz/export.py`` decides format policy), so these
tests pin the CONTRACT the GUI adds:

* ``ImagePanel.export_image``/``export_png`` and the new
  ``PlotPanel.export_image`` write validated files at the widget's
  own rendered size;
* the Image Lab's export row exposes the §17 controls (format combo
  with TIFF default, JPEG quality spin 1-95, TIFF compression choice)
  and its status line reports the engine's honest result card --
  including the extension-stamping rule (§17.4: a foreign suffix gets
  the chosen format's extension before the engine ever sees it).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pharmaml.viz.export import sniff_format


def _drain() -> None:
    from PySide6.QtWidgets import QApplication

    QApplication.processEvents()


# -- ImagePanel -------------------------------------------------------------------


def test_image_panel_export_tiff_jpeg_png(studio_app, qtbot, tmp_path: Path) -> None:
    from ui.widgets.image_panel import ImagePanel

    panel = ImagePanel(title="export probe")
    panel.resize(320, 240)
    panel.set_matrix(np.arange(64, dtype=np.uint16).reshape(8, 8))
    panel.show()
    qtbot.wait(20)
    _drain()

    tiff = panel.export_image(tmp_path / "panel.tiff", compression="tiff_lzw")
    assert tiff.format == "TIFF" and tiff.validated
    assert (tiff.width, tiff.height) == (320, 240)

    jpeg = panel.export_image(tmp_path / "panel.jpg", jpeg_quality=85)
    assert jpeg.format == "JPEG" and jpeg.validated

    png = panel.export_image(tmp_path / "panel.png")
    assert png.format == "PNG" and png.validated
    assert sniff_format(tmp_path / "panel.png") == "PNG"
    panel.close()


def test_image_panel_export_png_compat(studio_app, qtbot, tmp_path: Path) -> None:
    """The pre-§17 boolean API still works (now validation-backed)."""
    from ui.widgets.image_panel import ImagePanel

    panel = ImagePanel()
    panel.resize(120, 90)
    panel.set_matrix(np.zeros((4, 4), dtype=np.uint8))
    panel.show()
    qtbot.wait(10)
    assert panel.export_png(tmp_path / "compat.png")
    assert (tmp_path / "compat.png").stat().st_size > 0
    panel.close()


# -- PlotPanel ----------------------------------------------------------------------


def test_plot_panel_export_all_formats(studio_app, qtbot, tmp_path: Path) -> None:
    from ui.widgets.plot_widget import PlotPanel

    panel = PlotPanel(title="line probe")
    panel.resize(300, 220)
    rng = np.random.default_rng(7)
    panel.scatter(np.arange(20), rng.random(20) * 5, label="points")
    panel.line(np.arange(20), np.arange(20) / 4, label="fit")
    panel.set_labels("time (h)", "response")
    panel.add_legend()
    panel.show()
    qtbot.wait(20)
    _drain()

    for name, kwargs in (
        ("plot.tiff", {"compression": "tiff_deflate"}),
        ("plot.jpg", {"jpeg_quality": 90}),
        ("plot.png", {}),
    ):
        result = panel.export_image(tmp_path / name, **kwargs)
        assert result.validated, name
        assert (result.width, result.height) == (300, 220), name
    panel.close()


# -- the Image Lab export controls ---------------------------------------------------


def _find_export_widgets(page):
    from PySide6.QtWidgets import QComboBox, QPushButton, QSpinBox

    buttons = [
        b for b in page.findChildren(QPushButton) if b.text().startswith("Export image")
    ]
    assert buttons, "the §17 export button is missing from the Image Lab"
    combos = page.findChildren(QComboBox)
    fmt_box = next((c for c in combos if "TIFF" in c.currentText()), None)
    assert fmt_box is not None, "the §17 format combo is missing"
    quality = next(
        (s for s in page.findChildren(QSpinBox) if s.maximum() == 95), None
    )
    assert quality is not None, "the JPEG quality spin (1-95) is missing"
    compress = next(
        (c for c in combos if c.currentText() in ("LZW", "DEFLATE", "None")), None
    )
    assert compress is not None, "the TIFF compression combo is missing"
    return buttons[0], fmt_box, quality, compress


def test_image_lab_export_row_controls(studio_app) -> None:
    from ui.pages.image_lab import ImageLabPage

    page = ImageLabPage()
    _button, fmt_box, quality, compress = _find_export_widgets(page)
    assert fmt_box.count() == 3  # TIFF, PNG, JPEG -- §17.1's set
    assert fmt_box.currentText().startswith("TIFF")  # scientific default
    assert 1 <= quality.value() <= 95
    assert quality.maximum() == 95
    assert [compress.itemText(i) for i in range(compress.count())] == [
        "LZW",
        "DEFLATE",
        "None",
    ]


def test_image_lab_export_reports_result_card(
    studio_app, qtbot, tmp_path: Path, monkeypatch
) -> None:
    """The status line carries the engine's honest report (§17.4)."""
    from PySide6.QtWidgets import QFileDialog

    from ui.pages.image_lab import ImageLabPage

    page = ImageLabPage()
    page.show()
    qtbot.wait(30)
    _drain()
    target = tmp_path / "lab.tiff"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "")),
    )
    _button, _fmt, _quality, _compress = _find_export_widgets(page)
    _button.click()
    _drain()

    assert target.exists()
    assert sniff_format(target) == "TIFF"
    status = page._status.text()
    assert "Exported" in status
    assert "TIFF" in status
    assert "bytes" in status


def test_image_lab_export_stamps_foreign_extension(
    studio_app, qtbot, tmp_path: Path, monkeypatch
) -> None:
    """§17.4: a name with a foreign suffix gets the chosen format's ext."""
    from PySide6.QtWidgets import QFileDialog

    from ui.pages.image_lab import ImageLabPage

    page = ImageLabPage()
    page.show()
    qtbot.wait(30)
    _drain()
    # TIFF chosen (the default), user typed a .png name -> becomes .tiff.
    target = tmp_path / "oops.png"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(target), "")),
    )
    _button, _fmt, _quality, _compress = _find_export_widgets(page)
    _button.click()
    _drain()
    assert (tmp_path / "oops.tiff").exists()
    assert not target.exists()
    assert sniff_format(tmp_path / "oops.tiff") == "TIFF"


def test_image_lab_export_failure_is_honest(
    studio_app, qtbot, tmp_path: Path, monkeypatch
) -> None:
    """A failing export surfaces the engine's error, not a silent pass."""
    from PySide6.QtWidgets import QFileDialog

    from ui.pages.image_lab import ImageLabPage

    page = ImageLabPage()
    page.show()
    qtbot.wait(30)
    _drain()
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(tmp_path / "lab.jpg"), "")),
    )
    # JPEG chosen + a refusal from the engine: monkeypatch the panel
    # export to raise like a 16-bit refusal would.
    from ui.widgets import image_panel as image_panel_module

    def _boom(self, path, **kwargs):
        from pharmaml.exceptions import DataValidationError

        raise DataValidationError("16-bit source refuses lossy JPEG")

    monkeypatch.setattr(image_panel_module.ImagePanel, "export_image", _boom)
    _button, fmt_box, _q, _c = _find_export_widgets(page)
    fmt_box.setCurrentIndex(2)  # JPEG
    _button.click()
    _drain()
    status = page._status.text()
    assert "FAILED" in status
    assert "16-bit" in status
