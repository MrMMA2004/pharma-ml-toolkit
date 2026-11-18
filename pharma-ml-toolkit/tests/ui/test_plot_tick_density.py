"""Part-3 §23.6 adaptive tick-label density tests (round 238).

Contract, one module (``ui.widgets.plot_widget.PlotPanel``):

* a category axis (``bars(labels=...)``) renders EVERY tick it is
  handed, so 40 feature names on a narrow panel collide into an
  unreadable smear -- the density policy thins the RENDERED subset by
  a width-derived stride and ellipsizes the kept labels to their
  per-tick pixel share;
* the FULL category list stays behind the panel -- the §21.3
  auto-generated bar hover meta keeps resolving the complete,
  untruncated names for points whose tick was thinned away;
* a panel narrower than ``_MIN_AVAIL_PX`` (the pre-layout state)
  keeps all ticks;
* ``clear()`` resets the stored list along with the axis ticks;
* the resizeEvent wiring really re-fits: a SHOWN panel (Qt defers
  the resize events of hidden widgets to show time) re-derives the
  rendered subset on every geometry change.

pyqtgraph 0.13 stores the rendered levels in ``AxisItem._tickLevels``
(verified against the installed 0.13.7 ``setTicks`` source); the
tests read that back to assert on what the axis will actually draw.
"""

from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QApplication

from ui.widgets.plot_widget import PlotPanel


def _make_panel() -> PlotPanel:
    return PlotPanel()


def _resize(panel: PlotPanel, w: int, h: int = 240) -> None:
    """Resize a HIDDEN panel and re-fit.

    Qt queues the resize event of a hidden widget until show() -- the
    event-driven path is covered separately by the shown-wiring test
    -- so the hidden-panel helpers re-apply the policy explicitly,
    exactly what production's show-time event will do.
    """
    panel.resize(w, h)
    if panel.layout() is not None:
        panel.layout().activate()
    panel._apply_tick_density()
    QApplication.processEvents()


def _rendered(panel: PlotPanel) -> list[tuple[float, str]]:
    levels = panel.widget().getAxis("bottom")._tickLevels
    return list(levels[0]) if levels else []


def test_narrow_resize_thins_the_labels(studio_app) -> None:
    """40 categories on a ~280px panel: stride thinning, first at 0."""
    panel = _make_panel()
    labels = [f"feat_{i:02d}" for i in range(40)]
    panel.bars(np.arange(40, dtype=float), labels=labels)
    _resize(panel, 280)
    rendered = _rendered(panel)
    assert 0 < len(rendered) < 40
    assert rendered[0][0] == 0.0
    stride = rendered[1][0] - rendered[0][0]
    assert stride >= 2
    # every kept tick sits exactly on a stride multiple
    assert all(int(pos) % stride == 0 for pos, _ in rendered)


def test_wider_resize_shows_more_labels(studio_app) -> None:
    """The density follows the width: 280px < 900px rendered count."""
    panel = _make_panel()
    labels = [f"feat_{i:02d}" for i in range(40)]
    panel.bars(np.arange(40, dtype=float), labels=labels)
    _resize(panel, 280)
    narrow = len(_rendered(panel))
    _resize(panel, 900)
    wide = _rendered(panel)
    assert len(wide) > narrow
    assert all(wide[i][0] < wide[i + 1][0] for i in range(len(wide) - 1))


def test_long_labels_are_ellipsized_to_their_share(studio_app) -> None:
    """Very long names on a narrow panel: each fits its per-tick share."""
    panel = _make_panel()
    labels = ["lipophilicity_descriptor_" + "x" * 200 for _ in range(3)]
    panel.bars(np.arange(3, dtype=float), labels=labels)
    _resize(panel, 300)
    rendered = _rendered(panel)
    assert len(rendered) >= 1
    fm = QFontMetrics(panel.font())
    per_slot = panel._bottom_axis_available_px() / len(rendered)
    for _, text in rendered:
        # the policy contract: a rendered label never outgrows its slot
        assert fm.horizontalAdvance(text) + 2 <= per_slot
        assert len(text) < len(labels[0])


def test_hover_meta_keeps_the_full_labels(studio_app) -> None:
    """§21.3 contract: thinning touches the axis only, not the meta."""
    panel = _make_panel()
    labels = [f"feat_{i:02d}" for i in range(40)]
    panel.bars(np.arange(40, dtype=float), labels=labels)
    _resize(panel, 280)
    rendered_positions = {pos for pos, _ in _rendered(panel)}
    thinned = [i for i in range(40) if float(i) not in rendered_positions]
    assert thinned  # the narrow panel really thinned something
    record = panel._inspector._series["bars"]
    for i in (thinned[0], thinned[-1]):
        assert record.metadata(i)["category"] == f"feat_{i:02d}"


def test_below_the_floor_keeps_all_ticks(studio_app) -> None:
    """Pre-layout widths (< _MIN_AVAIL_PX): the full tick list."""
    panel = _make_panel()
    labels = [f"feat_{i:02d}" for i in range(30)]
    panel.bars(np.arange(30, dtype=float), labels=labels)
    _resize(panel, 60, 200)
    assert panel._bottom_axis_available_px() < 80
    rendered = _rendered(panel)
    assert len(rendered) == 30
    assert rendered[17][1] == "feat_17"


def test_clear_resets_the_stored_ticks(studio_app) -> None:
    """After clear() the panel forgets the categories completely."""
    panel = _make_panel()
    panel.bars(np.arange(5, dtype=float),
               labels=["a", "b", "c", "d", "e"])
    assert panel._cat_ticks is not None
    panel.clear()
    assert panel._cat_ticks is None
    assert panel.widget().getAxis("bottom")._tickLevels is None


def test_resize_before_bars_also_fits(studio_app) -> None:
    """The density applies at plot time too, not only on resizeEvent."""
    panel = _make_panel()
    _resize(panel, 280)
    labels = [f"feat_{i:02d}" for i in range(40)]
    panel.bars(np.arange(40, dtype=float), labels=labels)
    assert 0 < len(_rendered(panel)) < 40


def test_shown_panel_refits_on_resize(studio_app) -> None:
    """The event-driven path: a SHOWN panel re-fits on every resize.

    Qt defers the resize events of hidden widgets to show() time, so
    production panels get their first fit at show and a re-fit on
    every geometry change -- this leg pins that wiring end to end.
    """
    panel = _make_panel()
    labels = [f"feat_{i:02d}" for i in range(40)]
    panel.bars(np.arange(40, dtype=float), labels=labels)
    panel.resize(900, 240)
    panel.show()
    QApplication.processEvents()
    wide = len(_rendered(panel))
    panel.resize(280, 240)
    QApplication.processEvents()
    narrow = len(_rendered(panel))
    assert narrow < wide
    panel.close()


@pytest.mark.parametrize("count", [1, 5])
def test_tiny_category_sets_never_thinned(count: int, studio_app) -> None:
    """A handful of short labels on a wide-enough panel stay intact."""
    panel = _make_panel()
    labels = [f"c{i}" for i in range(count)]
    panel.bars(np.arange(count, dtype=float), labels=labels)
    _resize(panel, 600)
    rendered = _rendered(panel)
    assert [text for _, text in rendered] == labels


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
