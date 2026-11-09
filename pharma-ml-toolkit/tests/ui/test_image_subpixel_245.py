"""Round 245 -- Part-3 §23.10: sub-pixel mapping disclosure tests.

The ImagePanel follow-up to §21.4, pinned end to end: under extreme
zoom the hover readout must disclose what a screen pixel actually
covers.

* ``view_pixel_scale`` -- the live on-screen size (device px) of ONE
  image pixel (None without data / after clear; the exact
  ``1 / ViewBox.viewPixelSize()`` arithmetic on a shown panel; the
  zoom-in and zoom-out regimes from real view ranges);
* ``format_scale_note`` -- the pure disclosure: the ≤ceil coverage
  note below ``ZOOMED_OUT_PX`` (per-axis worst case, "(one shown)"),
  the steady-value note at/above ``ZOOMED_IN_PX``, silence near 1:1,
  None on missing/non-positive scales;
* the scene path -- ``_on_scene_mouse_moved`` attaches
  ``image_px_screen`` to every real payload, the §21.6 debounce still
  keys on the pixel alone, and a zoom between moves refreshes the
  scale on the next pixel transition (the documented contract);
* the page -- the Image Lab hover label appends the note for a
  zoomed-out payload and stays silent for a ~1:1 one.

The viewPixelSize arithmetic itself (fit / zoom-in / zoom-out /
hidden-widget behavior, cache invalidation across range changes) was
engine-printed first by the round-245 probe
(``scratch/probe_image_scale_245.py``).
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QPointF


def _panel(studio_app):
    from ui.widgets.image_panel import ImagePanel

    return ImagePanel()


def _capture(panel):
    captured: list = []
    panel.pixel_hovered.connect(lambda payload: captured.append(payload))
    return captured


def _identity_mapping(panel, monkeypatch) -> None:
    """The chart-inspector idiom: mapFromScene becomes the identity."""
    monkeypatch.setattr(
        panel.image_item(), "mapFromScene", lambda p: QPointF(p.x(), p.y())
    )


def _shown(studio_app, width: int, height: int):
    """A shown panel at an explicit size (geometry drives the scale)."""
    panel = _panel(studio_app)
    panel.resize(width, height)
    panel.show()
    return panel


def _set_range(panel, x_range, y_range) -> None:
    panel.image_item().getViewBox().setRange(xRange=x_range, yRange=y_range)


# -- view_pixel_scale: the live measurement -----------------------------------


def test_scale_none_without_data(studio_app) -> None:
    panel = _shown(studio_app, 400, 400)
    assert panel.view_pixel_scale() is None


def test_scale_none_after_clear(studio_app) -> None:
    panel = _shown(studio_app, 400, 400)
    panel.set_matrix(np.zeros((20, 20)))
    assert panel.view_pixel_scale() is not None
    panel.clear()
    assert panel.view_pixel_scale() is None


def test_scale_matches_view_pixel_size(studio_app) -> None:
    """The exact arithmetic: one image pixel = 1 / viewPixelSize per axis."""
    from PySide6.QtWidgets import QApplication

    panel = _shown(studio_app, 400, 400)
    panel.set_matrix(np.arange(10000, dtype=float).reshape(100, 100))
    QApplication.processEvents()
    scale = panel.view_pixel_scale()
    assert scale is not None
    sx, sy = scale
    assert sx > 0.0 and sy > 0.0
    vb = panel.image_item().getViewBox()
    pw, ph = vb.viewPixelSize()
    assert sx == pytest.approx(1.0 / pw)
    assert sy == pytest.approx(1.0 / ph)
    # aspect-locked: the axes agree to rounding
    assert sx == pytest.approx(sy, rel=0.15)


def test_scale_extreme_zoom_in(studio_app) -> None:
    """A 2x2 window over 50x50 on a 400px view: one image px is huge."""
    from PySide6.QtWidgets import QApplication

    panel = _shown(studio_app, 400, 400)
    panel.set_matrix(np.arange(2500, dtype=float).reshape(50, 50))
    QApplication.processEvents()
    _set_range(panel, (10, 12), (10, 12))
    QApplication.processEvents()
    scale = panel.view_pixel_scale()
    assert scale is not None
    # ~99 screen px: the LUT column + axes narrow the view inside the
    # 400px panel -- decisively past the ZOOMED_IN_PX block regime
    assert scale[0] > 50.0 and scale[1] > 50.0


def test_scale_zoomed_out_sub_pixel(studio_app) -> None:
    """A 100x100 matrix squeezed into a ~100px view: sub-pixel image px."""
    from PySide6.QtWidgets import QApplication

    panel = _shown(studio_app, 100, 100)
    panel.set_matrix(np.arange(10000, dtype=float).reshape(100, 100))
    QApplication.processEvents()
    _set_range(panel, (0, 100), (0, 100))
    QApplication.processEvents()
    scale = panel.view_pixel_scale()
    assert scale is not None
    assert scale[0] < 1.0 and scale[1] < 1.0  # one image px < one screen px


# -- format_scale_note: the pure disclosure ------------------------------------


def test_note_none_cases() -> None:
    from ui.widgets.image_panel import format_scale_note

    assert format_scale_note(None) is None
    assert format_scale_note((0.0, 0.0)) is None
    assert format_scale_note((-1.0, 1.0)) is None
    # near 1:1 and the quiet middle band stay silent
    assert format_scale_note((1.0, 1.0)) is None
    assert format_scale_note((2.0, 2.0)) is None
    assert format_scale_note((3.9, 1.0)) is None


def test_note_zoomed_out_coverage() -> None:
    from ui.widgets.image_panel import format_scale_note

    # 0.375 screen px per image px -> ceil(1/0.375) = 3 columns/rows
    assert (
        format_scale_note((0.375, 0.375))
        == "1 screen px spans ≤3×3 image px (one shown)"
    )
    # exactly at the floor of the straddle arithmetic
    assert format_scale_note((0.5, 0.5)) == "1 screen px spans ≤2×2 image px (one shown)"
    # per-axis honesty: x sub-pixel, y at 1:1
    assert format_scale_note((0.5, 2.0)) == "1 screen px spans ≤2×1 image px (one shown)"


def test_note_zoomed_in_block() -> None:
    from ui.widgets.image_panel import format_scale_note

    assert (
        format_scale_note((200.0, 200.0))
        == "1 image px ≈ 200×200 screen px (value steady in block)"
    )
    # rounding rides the :.0f format
    assert "≈ 156×156 screen px" in (format_scale_note((155.7, 156.0)) or "")


def test_note_thresholds_are_module_constants() -> None:
    """The disclosure band is pinned by the exported constants."""
    from ui.widgets.image_panel import ZOOMED_IN_PX, ZOOMED_OUT_PX, format_scale_note

    assert ZOOMED_OUT_PX == pytest.approx(0.9)
    assert ZOOMED_IN_PX == pytest.approx(4.0)
    # just below the zoomed-out threshold discloses; just above stays silent
    assert format_scale_note((ZOOMED_OUT_PX - 1e-6, 1.0)) is not None
    assert format_scale_note((ZOOMED_OUT_PX, 1.0)) is None
    # just below the zoomed-in threshold stays silent; at it discloses
    # (the block note needs BOTH axes past the threshold: min(sx, sy))
    assert format_scale_note((ZOOMED_IN_PX - 1e-6, ZOOMED_IN_PX)) is None
    assert format_scale_note((ZOOMED_IN_PX, ZOOMED_IN_PX)) is not None


# -- the scene path: the payload carries the scale ------------------------------


def test_scene_payload_carries_image_px_screen(studio_app, monkeypatch) -> None:
    panel = _shown(studio_app, 400, 400)
    panel.set_matrix(np.arange(35, dtype=float).reshape(5, 7))
    captured = _capture(panel)
    _identity_mapping(panel, monkeypatch)
    panel._on_scene_mouse_moved(QPointF(3.2, 2.7))
    assert len(captured) == 1
    payload = captured[0]
    assert payload is not None
    scale = payload["image_px_screen"]
    assert isinstance(scale, tuple) and len(scale) == 2
    assert scale[0] > 0.0 and scale[1] > 0.0


def test_scene_reset_payload_has_no_scale(studio_app, monkeypatch) -> None:
    """The cursor-left None emission never carries a scale."""
    panel = _shown(studio_app, 400, 400)
    panel.set_matrix(np.arange(12, dtype=float).reshape(3, 4))
    captured = _capture(panel)
    _identity_mapping(panel, monkeypatch)
    panel._on_scene_mouse_moved(QPointF(1.1, 1.1))
    panel._on_scene_mouse_moved(QPointF(1e6, 1e6))  # outside -> None
    assert captured[-1] is None


def test_debounce_keys_on_pixel_not_scale(studio_app, monkeypatch) -> None:
    """Same pixel -> one emission; the scale refreshes on transitions."""
    from PySide6.QtWidgets import QApplication

    panel = _shown(studio_app, 400, 400)
    panel.set_matrix(np.arange(2500, dtype=float).reshape(50, 50))
    captured = _capture(panel)
    _identity_mapping(panel, monkeypatch)
    panel._on_scene_mouse_moved(QPointF(10.4, 10.4))
    panel._on_scene_mouse_moved(QPointF(10.8, 10.8))  # same pixel
    assert len(captured) == 1
    # zoom in hard between moves; the next pixel transition carries it
    _set_range(panel, (10, 12), (10, 12))
    QApplication.processEvents()
    panel._on_scene_mouse_moved(QPointF(11.5, 11.5))  # a NEW pixel
    assert len(captured) == 2
    assert captured[1]["image_px_screen"][0] > 50.0
    # moving back to the first pixel re-discloses the SAME new scale
    panel._on_scene_mouse_moved(QPointF(10.4, 10.4))
    assert len(captured) == 3
    assert captured[2]["image_px_screen"][0] > 50.0


# -- the page wiring ------------------------------------------------------------


def _page(studio_app):
    from ui.pages.image_lab import ImageLabPage

    return ImageLabPage()


def _page_payload(**overrides) -> dict:
    """A full §21.4 payload shape with an overridable §23.10 scale."""
    payload: dict = {
        "row": 3,
        "col": 2,
        "x": 2.5,
        "y": 3.5,
        "value": 0.42,
        "channels": 1,
        "rgb": None,
        "masked": False,
        "x_label": None,
        "y_label": None,
        "shape": (12, 12),
        "image_px_screen": None,
    }
    payload.update(overrides)
    return payload


def test_page_label_appends_zoomed_out_note(studio_app) -> None:
    from PySide6.QtWidgets import QLabel

    page = _page(studio_app)
    payload = _page_payload(image_px_screen=(0.375, 0.375))
    page._roi_panel.pixel_hovered.emit(payload)
    texts = [t.text() for t in page.findChildren(QLabel) if t.text().startswith("Hover:")]
    assert any("1 screen px spans ≤3×3 image px (one shown)" in t for t in texts), texts


def test_page_label_appends_zoomed_in_note(studio_app) -> None:
    from PySide6.QtWidgets import QLabel

    page = _page(studio_app)
    payload = _page_payload(image_px_screen=(200.0, 200.0))
    page._roi_panel.pixel_hovered.emit(payload)
    texts = [t.text() for t in page.findChildren(QLabel) if t.text().startswith("Hover:")]
    assert any(
        "1 image px ≈ 200×200 screen px (value steady in block)" in t for t in texts
    ), texts


def test_page_label_silent_near_unity(studio_app) -> None:
    from PySide6.QtWidgets import QLabel

    page = _page(studio_app)
    payload = _page_payload(image_px_screen=(1.0, 1.0))
    page._roi_panel.pixel_hovered.emit(payload)
    texts = [t.text() for t in page.findChildren(QLabel) if t.text().startswith("Hover:")]
    assert any("Pixel [3, 2]" in t for t in texts), texts
    assert not any("screen px" in t for t in texts), texts


def test_page_label_tolerates_missing_scale(studio_app) -> None:
    """A pixel_at-built payload (no scale key) renders unchanged."""
    from PySide6.QtWidgets import QLabel

    page = _page(studio_app)
    payload = _page_payload()
    del payload["image_px_screen"]
    page._roi_panel.pixel_hovered.emit(payload)
    texts = [t.text() for t in page.findChildren(QLabel) if t.text().startswith("Hover:")]
    assert any("Pixel [3, 2] (x 2.5, y 3.5)" in t for t in texts), texts
