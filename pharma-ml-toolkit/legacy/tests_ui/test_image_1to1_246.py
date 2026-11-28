"""Round 246 -- Part-3 §21.4: the original-resolution (1:1) view tests.

The TRUE pixel-perfect toggle the stretch leg asked for, pinned end
to end on the §23.10 scale foundation:

* ``set_original_resolution`` -- the honest refusals (no matrix /
  after clear / a HIDDEN panel, whose default-geometry scale the
  round-245 probe already showed is a lie); the achieved ≈1:1 scale
  from the fit, the 2x2 zoom-in (155.7 screen px per image px) and
  the sub-pixel squeeze (0.375) starts; the exact circular
  arithmetic (``viewPixelSize()`` itself lands at ~1.0 per axis);
  idempotence; and viewport-centre preservation (the probe's
  (11.0, 41.0) -> (11.0, 41.0));
* ``is_original_resolution`` -- the state band: True within
  ``RESOLUTION_TOLERANCE`` of 1.0 on BOTH axes, False for no data /
  ``reset_view`` / a user zoom away;
* the page -- every matrix tab carries the 1:1 button; the golden
  path clicks it (achieved scale disclosed in the status line) and
  clicks it again (fit restored); a hidden panel's toggle refuses
  out loud, never silently.

The engine mechanics (the circular target-range arithmetic, the
padding-0 setRange under aspect lock landing at 1.0000 on both axes
from every start, centre preservation, autoRange leaving unity, the
hidden default-geometry value) were engine-printed first by the
round-246 probe (``scratch/probe_image_1to1_246.py``).
"""

from __future__ import annotations

import numpy as np
import pytest


def _panel(studio_app):
    from ui.widgets.image_panel import ImagePanel

    return ImagePanel()


def _shown(studio_app, width: int, height: int):
    """A shown panel at an explicit size (geometry drives the scale)."""
    panel = _panel(studio_app)
    panel.resize(width, height)
    panel.show()
    return panel


def _set_range(panel, x_range, y_range) -> None:
    panel.image_item().getViewBox().setRange(xRange=x_range, yRange=y_range)


# -- set_original_resolution: the honest refusals --------------------------------


def test_1to1_refused_without_data(studio_app) -> None:
    panel = _shown(studio_app, 400, 400)
    assert panel.set_original_resolution() is None
    assert panel.is_original_resolution() is False


def test_1to1_refused_after_clear(studio_app) -> None:
    panel = _shown(studio_app, 400, 400)
    panel.set_matrix(np.zeros((20, 20)))
    panel.clear()
    assert panel.set_original_resolution() is None


def test_1to1_refused_when_hidden(studio_app) -> None:
    """A hidden panel answers a fake default-geometry scale -- refuse."""
    panel = _panel(studio_app)
    panel.set_matrix(np.zeros((20, 20)))
    # never shown: isVisible() is False, so the honest refusal wins
    # even though the raw view would happily measure ~40 px/image-px.
    assert panel.set_original_resolution() is None


# -- set_original_resolution: the achieved ≈1:1 state ----------------------------


def test_1to1_achieves_unity_from_fit(studio_app) -> None:
    from PySide6.QtWidgets import QApplication
    from ui.widgets.image_panel import RESOLUTION_TOLERANCE

    panel = _shown(studio_app, 400, 400)
    panel.set_matrix(np.arange(10000, dtype=float).reshape(100, 100))
    QApplication.processEvents()
    assert not panel.is_original_resolution()  # the fit is ~3.1 px/px
    scale = panel.set_original_resolution()
    assert scale is not None
    assert scale[0] == pytest.approx(1.0, abs=RESOLUTION_TOLERANCE)
    assert scale[1] == pytest.approx(1.0, abs=RESOLUTION_TOLERANCE)
    assert panel.is_original_resolution() is True


def test_1to1_matches_exact_arithmetic(studio_app) -> None:
    """viewPixelSize() itself lands at ~1.0 per axis (the circular math)."""
    from PySide6.QtWidgets import QApplication
    from ui.widgets.image_panel import RESOLUTION_TOLERANCE

    panel = _shown(studio_app, 400, 400)
    panel.set_matrix(np.arange(10000, dtype=float).reshape(100, 100))
    QApplication.processEvents()
    panel.set_original_resolution()
    vb = panel.image_item().getViewBox()
    pw, ph = vb.viewPixelSize()
    assert pw == pytest.approx(1.0, abs=RESOLUTION_TOLERANCE)
    assert ph == pytest.approx(1.0, abs=RESOLUTION_TOLERANCE)


def test_1to1_idempotent(studio_app) -> None:
    from PySide6.QtWidgets import QApplication

    panel = _shown(studio_app, 400, 400)
    panel.set_matrix(np.arange(2500, dtype=float).reshape(50, 50))
    QApplication.processEvents()
    first = panel.set_original_resolution()
    second = panel.set_original_resolution()
    assert first is not None and second is not None
    assert second[0] == pytest.approx(1.0, abs=0.05)
    assert second[1] == pytest.approx(1.0, abs=0.05)
    assert panel.is_original_resolution() is True


def test_1to1_from_zoomed_in(studio_app) -> None:
    """155.7 screen px per image px -> ≈1:1 in one press."""
    from PySide6.QtWidgets import QApplication

    panel = _shown(studio_app, 400, 400)
    panel.set_matrix(np.arange(2500, dtype=float).reshape(50, 50))
    QApplication.processEvents()
    _set_range(panel, (10, 12), (10, 12))
    QApplication.processEvents()
    assert panel.view_pixel_scale()[0] > 50.0  # decisively zoomed in
    scale = panel.set_original_resolution()
    assert scale is not None
    assert scale[0] == pytest.approx(1.0, abs=0.05)
    assert scale[1] == pytest.approx(1.0, abs=0.05)


def test_1to1_from_sub_pixel(studio_app) -> None:
    """0.375 screen px per image px -> ≈1:1 in one press."""
    from PySide6.QtWidgets import QApplication

    panel = _shown(studio_app, 100, 100)
    panel.set_matrix(np.arange(10000, dtype=float).reshape(100, 100))
    QApplication.processEvents()
    _set_range(panel, (0, 100), (0, 100))
    QApplication.processEvents()
    assert panel.view_pixel_scale()[0] < 1.0  # sub-pixel image px
    scale = panel.set_original_resolution()
    assert scale is not None
    assert scale[0] == pytest.approx(1.0, abs=0.05)
    assert scale[1] == pytest.approx(1.0, abs=0.05)


def test_1to1_preserves_viewport_center(studio_app) -> None:
    """The image-viewer convention: the eye's centre stays the centre."""
    from PySide6.QtWidgets import QApplication

    panel = _shown(studio_app, 400, 400)
    panel.set_matrix(np.arange(2500, dtype=float).reshape(50, 50))
    QApplication.processEvents()
    _set_range(panel, (10, 12), (40, 42))
    QApplication.processEvents()
    assert panel.set_original_resolution() is not None
    (x0, x1), (y0, y1) = panel.image_item().getViewBox().viewRange()
    assert (x0 + x1) / 2.0 == pytest.approx(11.0, abs=2.0)
    assert (y0 + y1) / 2.0 == pytest.approx(41.0, abs=2.0)
    # the span is the view box's own px size (the circular arithmetic)
    assert (x1 - x0) > 100.0  # a ~300px view box, not the 2-unit window


# -- is_original_resolution: the state band ---------------------------------------


def test_reset_view_leaves_unity(studio_app) -> None:
    from PySide6.QtWidgets import QApplication

    panel = _shown(studio_app, 400, 400)
    panel.set_matrix(np.arange(10000, dtype=float).reshape(100, 100))
    QApplication.processEvents()
    assert panel.set_original_resolution() is not None
    panel.reset_view()
    QApplication.processEvents()
    assert panel.is_original_resolution() is False  # back at the fit


def test_user_zoom_leaves_unity(studio_app) -> None:
    """The state tracks the LIVE view, not the last button press."""
    from PySide6.QtWidgets import QApplication

    panel = _shown(studio_app, 400, 400)
    panel.set_matrix(np.arange(2500, dtype=float).reshape(50, 50))
    QApplication.processEvents()
    assert panel.set_original_resolution() is not None
    _set_range(panel, (10, 12), (10, 12))
    QApplication.processEvents()
    assert panel.is_original_resolution() is False


def test_tolerance_band_constant() -> None:
    """The ≈1:1 band is pinned by the exported constant."""
    from ui.widgets.image_panel import RESOLUTION_TOLERANCE

    assert RESOLUTION_TOLERANCE == pytest.approx(0.05)


def test_is_original_resolution_band_edges(studio_app, monkeypatch) -> None:
    """Both axes must sit inside the band; the edges are inclusive."""
    panel = _panel(studio_app)
    monkeypatch.setattr(panel, "view_pixel_scale", lambda: None)
    assert panel.is_original_resolution() is False
    monkeypatch.setattr(panel, "view_pixel_scale", lambda: (1.0, 1.0))
    assert panel.is_original_resolution() is True
    monkeypatch.setattr(panel, "view_pixel_scale", lambda: (0.96, 0.99))
    assert panel.is_original_resolution() is True
    monkeypatch.setattr(panel, "view_pixel_scale", lambda: (0.94, 1.0))
    assert panel.is_original_resolution() is False
    monkeypatch.setattr(panel, "view_pixel_scale", lambda: (1.06, 1.0))
    assert panel.is_original_resolution() is False
    # one axis inside, one outside: NOT the original resolution
    monkeypatch.setattr(panel, "view_pixel_scale", lambda: (1.0, 3.0))
    assert panel.is_original_resolution() is False


# -- the page wiring ---------------------------------------------------------------


def _shown_page(studio_app):
    from PySide6.QtWidgets import QApplication
    from ui.pages.image_lab import ImageLabPage

    page = ImageLabPage()
    page.resize(1000, 800)
    page.show()
    QApplication.processEvents()
    QApplication.processEvents()
    return page


def _show_roi_tab(page) -> None:
    from PySide6.QtWidgets import QApplication
    from ui.widgets.image_panel import ImagePanel

    for i in range(page._tabs.count()):
        if page._roi_panel in page._tabs.widget(i).findChildren(ImagePanel):
            page._tabs.setCurrentIndex(i)
            break
    if page._roi_panel.values() is None:  # RDKit-less box: seed honestly
        page._roi_panel.set_matrix(np.arange(144, dtype=float).reshape(12, 12))
    QApplication.processEvents()
    QApplication.processEvents()


def _visible_1to1_buttons(page):
    from PySide6.QtWidgets import QPushButton

    return [
        b
        for b in page.findChildren(QPushButton)
        if b.text() == "1:1 original resolution" and b.isVisible()
    ]


def test_page_toggle_golden_path(studio_app) -> None:
    from PySide6.QtWidgets import QApplication

    page = _shown_page(studio_app)
    _show_roi_tab(page)
    buttons = _visible_1to1_buttons(page)
    assert len(buttons) == 1  # exactly the current tab's toggle is live
    buttons[0].click()
    QApplication.processEvents()
    assert page._roi_panel.is_original_resolution() is True
    assert "1:1 original resolution" in page._status.text()
    buttons = _visible_1to1_buttons(page)
    buttons[0].click()  # the second press restores the §21.2 fit
    QApplication.processEvents()
    assert page._roi_panel.is_original_resolution() is False
    assert "fit-to-view restored" in page._status.text()


def test_page_status_discloses_achieved_scale(studio_app) -> None:
    page = _shown_page(studio_app)
    _show_roi_tab(page)
    buttons = _visible_1to1_buttons(page)
    buttons[0].click()
    text = page._status.text()
    # the MEASURED-back scale is disclosed, not assumed
    assert "1 image px = " in text
    assert "screen px (§21.4)" in text


def test_page_hidden_panel_refuses_out_loud(studio_app) -> None:
    """A panel in a non-current tab refuses honestly in the status line."""
    page = _shown_page(studio_app)
    _show_roi_tab(page)
    # leave the ROI tab: its panel (and toggle target) goes hidden
    page._tabs.setCurrentIndex(0)
    from PySide6.QtWidgets import QApplication

    QApplication.processEvents()
    page._toggle_original_resolution(page._roi_panel)
    assert "1:1 unavailable" in page._status.text()
