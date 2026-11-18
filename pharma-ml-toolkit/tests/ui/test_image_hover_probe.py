"""Round 240 -- Part-3 SS21.4: image pixel/channel inspection tests.

The §21.4 acceptance line ("image coordinates and pixel values can be
inspected") pinned end to end for :class:`ui.widgets.image_panel.ImagePanel`
and the Image Lab page wiring:

* ``pixel_at`` -- the pure data-coordinate mapping (exact values,
  fractional flooring, outside/negative refusal, NaN disclosure, axis
  labels riding the payload);
* the scene path -- ``_on_scene_mouse_moved`` through a monkeypatched
  identity ``mapFromScene`` (the chart-inspector idiom), including the
  §21.6 per-pixel debounce and the leaveEvent reset;
* ``image_metadata`` -- the at-rest facts (shape, dtype, channels,
  finite/NaN census, range) and the clear() reset;
* the RGB(A) path -- per-channel payloads, channel-count metadata, and
  the honest NaN refusal;
* the page -- every tab's hover label answers a real payload and
  restores the metadata line at rest (the ROI tab's real 12x12 engine
  matrix is the fixture).

The mapping contract itself (``ImageItem.mapFromScene`` round-trip
exactness) was engine-printed first by
``scratch/probe_image_hover_240.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF


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


# -- pixel_at: the pure mapping --------------------------------------------


def test_pixel_at_reads_exact_value(studio_app) -> None:
    panel = _panel(studio_app)
    matrix = np.arange(35, dtype=float).reshape(5, 7)
    panel.set_matrix(matrix)
    payload = panel.pixel_at(3.2, 2.7)
    assert payload is not None
    assert payload["row"] == 2 and payload["col"] == 3
    assert payload["value"] == pytest.approx(17.0)  # matrix[2, 3]
    assert payload["x"] == pytest.approx(3.2)
    assert payload["y"] == pytest.approx(2.7)
    assert payload["channels"] == 1
    assert payload["rgb"] is None
    assert payload["masked"] is False
    assert payload["shape"] == (5, 7)


def test_pixel_at_outside_refused(studio_app) -> None:
    panel = _panel(studio_app)
    panel.set_matrix(np.arange(12, dtype=float).reshape(3, 4))
    for x, y in [(-0.4, 1.0), (0.5, -0.1), (4.0, 1.0), (3.5, 3.0), (1e6, 1e6)]:
        assert panel.pixel_at(x, y) is None, (x, y)
    # the last inclusive pixel is still readable
    edge = panel.pixel_at(3.99, 2.99)
    assert edge is not None and edge["row"] == 2 and edge["col"] == 3


def test_pixel_at_without_matrix_returns_none(studio_app) -> None:
    panel = _panel(studio_app)
    assert panel.pixel_at(1.0, 1.0) is None
    assert panel.image_metadata() is None


def test_nan_pixel_disclosed_as_masked(studio_app) -> None:
    panel = _panel(studio_app)
    matrix = np.ones((4, 4))
    matrix[1, 2] = np.nan
    panel.set_matrix(matrix)
    payload = panel.pixel_at(2.5, 1.5)
    assert payload is not None
    assert payload["value"] is None
    assert payload["masked"] is True
    # a finite neighbour still reads its number
    clean = panel.pixel_at(1.5, 1.5)
    assert clean is not None and clean["value"] == pytest.approx(1.0)


def test_axis_labels_ride_the_payload(studio_app) -> None:
    panel = _panel(studio_app)
    panel.set_matrix(
        np.zeros((2, 3)),
        x_labels=["alpha", "beta", "gamma"],
        y_labels=["row0", "row1"],
    )
    payload = panel.pixel_at(1.5, 0.5)
    assert payload is not None
    assert payload["x_label"] == "beta"
    assert payload["y_label"] == "row0"
    # no labels -> honest Nones
    panel.set_matrix(np.zeros((2, 3)))
    bare = panel.pixel_at(1.5, 0.5)
    assert bare is not None
    assert bare["x_label"] is None and bare["y_label"] is None


# -- image_metadata ----------------------------------------------------------


def test_metadata_reports_shape_and_census(studio_app) -> None:
    panel = _panel(studio_app)
    matrix = np.arange(12, dtype=float).reshape(3, 4)
    matrix[0, 0] = np.nan
    panel.set_matrix(matrix)
    meta = panel.image_metadata()
    assert meta is not None
    assert meta["rows"] == 3 and meta["cols"] == 4
    assert meta["shape"] == (3, 4)
    assert meta["channels"] == 1
    assert meta["finite"] == 11 and meta["masked"] == 1
    assert meta["min"] == pytest.approx(1.0)
    assert meta["max"] == pytest.approx(11.0)
    assert meta["mean"] == pytest.approx(float(np.arange(1, 12).mean()))
    panel.clear()
    assert panel.image_metadata() is None


def test_clear_resets_hover_state(studio_app) -> None:
    panel = _panel(studio_app)
    panel.set_matrix(np.arange(6, dtype=float).reshape(2, 3))
    assert panel.pixel_at(1.5, 0.5) is not None
    panel.clear()
    assert panel.pixel_at(1.5, 0.5) is None


# -- the scene path (§21.4 wiring + §21.6 debounce) ---------------------------


def test_scene_mouse_move_emits_payload(studio_app, monkeypatch) -> None:
    panel = _panel(studio_app)
    panel.set_matrix(np.arange(35, dtype=float).reshape(5, 7))
    captured = _capture(panel)
    _identity_mapping(panel, monkeypatch)
    panel._on_scene_mouse_moved(QPointF(3.2, 2.7))
    assert len(captured) == 1
    payload = captured[0]
    assert payload is not None
    assert payload["row"] == 2 and payload["col"] == 3
    assert payload["value"] == pytest.approx(17.0)


def test_scene_move_without_matrix_is_silent(studio_app, monkeypatch) -> None:
    panel = _panel(studio_app)
    captured = _capture(panel)
    _identity_mapping(panel, monkeypatch)
    panel._on_scene_mouse_moved(QPointF(1.0, 1.0))
    assert captured == []


def test_same_pixel_debounces(studio_app, monkeypatch) -> None:
    panel = _panel(studio_app)
    panel.set_matrix(np.arange(35, dtype=float).reshape(5, 7))
    captured = _capture(panel)
    _identity_mapping(panel, monkeypatch)
    panel._on_scene_mouse_moved(QPointF(1.1, 2.2))
    panel._on_scene_mouse_moved(QPointF(1.5, 2.8))  # same pixel (col 1, row 2)
    assert len(captured) == 1
    panel._on_scene_mouse_moved(QPointF(2.5, 2.5))  # a new pixel
    assert len(captured) == 2


def test_outside_emits_none_once(studio_app, monkeypatch) -> None:
    panel = _panel(studio_app)
    panel.set_matrix(np.arange(12, dtype=float).reshape(3, 4))
    captured = _capture(panel)
    _identity_mapping(panel, monkeypatch)
    panel._on_scene_mouse_moved(QPointF(1.1, 1.1))  # inside
    panel._on_scene_mouse_moved(QPointF(1e6, 1e6))  # outside -> None
    panel._on_scene_mouse_moved(QPointF(2e6, 2e6))  # still outside -> silent
    assert len(captured) == 2
    assert captured[0] is not None
    assert captured[1] is None


def test_leave_event_resets_once(studio_app, monkeypatch) -> None:
    panel = _panel(studio_app)
    panel.set_matrix(np.arange(12, dtype=float).reshape(3, 4))
    captured = _capture(panel)
    _identity_mapping(panel, monkeypatch)
    panel._on_scene_mouse_moved(QPointF(1.1, 1.1))
    n = len(captured)
    leave = QEvent(QEvent.Type.Leave)
    panel.leaveEvent(leave)
    panel.leaveEvent(leave)  # a second leave never double-resets
    assert captured[n:] == [None]


def test_replot_rearms_the_hover(studio_app, monkeypatch) -> None:
    """A new matrix must re-open emission even on the same pixel."""
    panel = _panel(studio_app)
    panel.set_matrix(np.zeros((3, 4)))
    captured = _capture(panel)
    _identity_mapping(panel, monkeypatch)
    panel._on_scene_mouse_moved(QPointF(1.1, 1.1))
    assert len(captured) == 1
    panel.set_matrix(np.ones((3, 4)))
    panel._on_scene_mouse_moved(QPointF(1.1, 1.1))  # same coords, new matrix
    assert len(captured) == 2
    assert captured[1]["value"] == pytest.approx(1.0)


# -- the RGB(A) path (§21.4 channel information) -------------------------------


def test_rgb_payload_reports_channels(studio_app) -> None:
    panel = _panel(studio_app)
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[1, 2] = (200, 100, 50)
    panel.set_matrix(rgb)
    payload = panel.pixel_at(2.4, 1.6)
    assert payload is not None
    assert payload["row"] == 1 and payload["col"] == 2
    assert payload["channels"] == 3
    assert payload["rgb"] == (200.0, 100.0, 50.0)
    assert payload["value"] is None  # no single scalar on the RGB path
    meta = panel.image_metadata()
    assert meta is not None
    assert meta["channels"] == 3
    assert meta["shape"] == (4, 6, 3)
    assert meta["min"] == pytest.approx(0.0)
    assert meta["max"] == pytest.approx(200.0)
    assert meta["masked"] == 0


def test_rgba_four_channel_payload(studio_app) -> None:
    panel = _panel(studio_app)
    rgba = np.full((2, 2, 4), 250, dtype=np.uint8)
    panel.set_matrix(rgba)
    payload = panel.pixel_at(0.5, 0.5)
    assert payload is not None
    assert payload["channels"] == 4
    assert payload["rgb"] == (250.0, 250.0, 250.0, 250.0)


def test_nan_rgb_refused_honestly(studio_app) -> None:
    panel = _panel(studio_app)
    with pytest.raises(ValueError, match="NaN-free"):
        panel.set_matrix(np.full((2, 2, 3), np.nan))


# -- reset view ----------------------------------------------------------------


def test_reset_view_is_safe(studio_app) -> None:
    panel = _panel(studio_app)
    panel.set_matrix(np.arange(35, dtype=float).reshape(5, 7))
    panel.roi().setPos([1.0, 1.0])
    panel.reset_view()  # §21.2's reset control for images -- no crash
    assert panel.values() is not None


# -- the page wiring -----------------------------------------------------------


def _page(studio_app):
    from ui.pages.image_lab import ImageLabPage

    return ImageLabPage()


def test_every_matrix_tab_carries_a_hover_label(studio_app) -> None:
    from PySide6.QtWidgets import QLabel

    from ui.widgets.image_panel import ImagePanel

    page = _page(studio_app)
    panels = [page._sim_panel, page._screen_panel, page._exp_panel, page._roi_panel]
    for panel in panels:
        assert isinstance(panel, ImagePanel)
    # each panel's hover label sits in the page (4 readouts, word-wrapped)
    labels = [
        lab
        for lab in page.findChildren(QLabel)
        if lab.text().startswith("Hover:")
    ]
    assert len(labels) == 4
    assert all(lab.wordWrap() for lab in labels)


def test_page_hover_label_answers_real_payload(studio_app) -> None:
    from PySide6.QtWidgets import QLabel

    page = _page(studio_app)
    panel = page._roi_panel  # the real, engine-seeded 12x12 matrix
    assert panel.values() is not None and panel.values().shape == (12, 12)
    payload = panel.pixel_at(2.5, 3.5)  # col 2, row 3 -- real engine value
    assert payload is not None
    panel.pixel_hovered.emit(payload)
    texts = [t.text() for t in page.findChildren(QLabel) if t.text().startswith("Hover:")]
    assert any("Pixel [3, 2] (x 2.5, y 3.5)" in t for t in texts), texts
    assert any("= 0." in t or "= 1" in t for t in texts), texts


def test_page_hover_label_restores_metadata_at_rest(studio_app) -> None:
    from PySide6.QtWidgets import QLabel

    page = _page(studio_app)
    panel = page._roi_panel
    meta = panel.image_metadata()
    assert meta is not None
    assert meta["rows"] == 12 and meta["cols"] == 12
    assert meta["finite"] == 144 and meta["masked"] == 0
    panel.pixel_hovered.emit(None)
    texts = [t.text() for t in page.findChildren(QLabel) if t.text().startswith("Hover:")]
    assert any("Matrix 12x12" in t for t in texts), texts
    assert any("144 finite / 0 masked" in t for t in texts), texts
