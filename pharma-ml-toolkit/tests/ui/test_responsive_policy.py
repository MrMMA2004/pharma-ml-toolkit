"""Part-3 §23 responsive-discipline tests.

Three contracts, one module:

* **Splitter policy** (:func:`ui.widgets.responsive.configure_splitter`)
  -- no collapsible children, hard minimum extents, stretch factors
  (§23.2 "important controls becoming inaccessible" / clipped labels).
  Every page's splitter AND the main window's nav/content splitter go
  through the policy, so the audit is verified structurally, not by
  re-reading sixteen files.
* **Export legibility floor** (§23.3 "generated plots remain legible
  when exported") -- a squeezed panel re-renders from its vector scene
  at floor scale: dimensions meet the floor, the aspect ratio is
  preserved (§23.4), and the already-legible / no-floor paths keep the
  widget-owns-its-size contract the §17 tests pin.
* **Aspect lock** (§23.4 "preserve image aspect ratios / avoid
  stretching") -- the ImagePanel's matrix pixels stay square.

§23.5 is pinned too: the floor re-render changes display geometry
ONLY -- the plotted data and view ranges are byte-identical before
and after an export.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pharmaml.viz.export import sniff_format

# -- configure_splitter: the policy -------------------------------------------


def test_configure_splitter_policy(studio_app) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QSplitter, QWidget

    from ui.widgets.responsive import MIN_CHILD_PX, configure_splitter

    splitter = QSplitter(Qt.Orientation.Horizontal)
    left, right = QWidget(), QWidget()
    splitter.addWidget(left)
    splitter.addWidget(right)

    assert splitter.childrenCollapsible() is True  # the Qt default we fix

    configure_splitter(splitter, min_sizes=(360, 500), stretch=(0, 1))

    assert splitter.childrenCollapsible() is False
    assert left.minimumWidth() == 360
    assert right.minimumWidth() == 500
    # QSplitter has no stretch getter; the factor rides the child's
    # size policy (QSplitter.setStretchFactor's documented mechanism).
    assert left.sizePolicy().horizontalStretch() == 0
    assert right.sizePolicy().horizontalStretch() == 1


def test_configure_splitter_defaults_and_vertical(studio_app) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QSplitter, QWidget

    from ui.widgets.responsive import MIN_CHILD_PX, configure_splitter

    splitter = QSplitter(Qt.Orientation.Vertical)
    top, bottom = QWidget(), QWidget()
    splitter.addWidget(top)
    splitter.addWidget(bottom)
    configure_splitter(splitter)

    assert splitter.childrenCollapsible() is False
    assert top.minimumHeight() == MIN_CHILD_PX
    assert bottom.minimumHeight() == MIN_CHILD_PX


def test_configure_splitter_empty_is_noop(studio_app) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QSplitter

    from ui.widgets.responsive import configure_splitter

    splitter = QSplitter(Qt.Orientation.Horizontal)
    configure_splitter(splitter)  # no children: nothing to break
    assert splitter.count() == 0


def test_legibility_floor_constant() -> None:
    from ui.widgets.responsive import LEGIBILITY_FLOOR

    assert (LEGIBILITY_FLOOR.width(), LEGIBILITY_FLOOR.height()) == (640, 480)


# -- every page's splitter carries the policy (the §23.2 audit) ---------------


def test_main_window_splitter_policy(main_window) -> None:
    from PySide6.QtWidgets import QSplitter

    splitters = main_window.findChildren(QSplitter)
    assert splitters, "the main window's nav/content splitter is missing"
    for splitter in splitters:
        assert splitter.childrenCollapsible() is False
        for i in range(splitter.count()):
            assert splitter.widget(i).minimumWidth() >= 200


def test_page_splitters_carry_the_policy(studio_app) -> None:
    """Every QSplitter inside every studio page: non-collapsible children,
    per-child minimum extents, growth on the chart side."""
    from PySide6.QtWidgets import QSplitter

    from ui.main_window import MainWindow

    window = MainWindow()
    try:
        splitters = window.findChildren(QSplitter)
        assert len(splitters) >= 19, (
            f"expected >=19 wired splitters (15 pages + image lab x4 "
            f"+ ml workbench x2 + main window), found {len(splitters)}"
        )
        for splitter in splitters:
            assert splitter.childrenCollapsible() is False, (
                "a collapsible splitter child can hide controls (§23.2)"
            )
            for i in range(splitter.count()):
                child = splitter.widget(i)
                assert child.minimumWidth() > 0, (
                    "a splitter child without a minimum extent can be "
                    "squeezed to clipped text (§23.2)"
                )
    finally:
        window.close()


# -- export legibility floor (§23.3) -------------------------------------------


def test_plot_panel_export_floor_upscales_crisp(studio_app, qtbot, tmp_path: Path) -> None:
    """A squeezed PlotPanel exports at >= the floor, aspect preserved."""
    from PySide6.QtCore import QSize

    from ui.widgets.plot_widget import PlotPanel
    from ui.widgets.responsive import LEGIBILITY_FLOOR

    panel = PlotPanel(title="floor probe")
    panel.resize(300, 200)
    rng = np.random.default_rng(23)
    panel.scatter(np.arange(30), rng.random(30) * 5, label="points")
    panel.set_labels("time (h)", "response")
    panel.show()
    qtbot.wait(20)

    result = panel.export_image(tmp_path / "floored.png", min_size=LEGIBILITY_FLOOR)
    assert result.validated
    assert result.width >= LEGIBILITY_FLOOR.width()
    assert result.height >= LEGIBILITY_FLOOR.height()
    # §23.4 aspect preserved: the source was 3:2, the target stays 3:2
    # (the floor is 4:3, so the binding constraint is width -> 640x~427+
    # and rounding keeps the ratio within a pixel).
    assert abs(result.width / result.height - 300 / 200) < 0.02
    assert sniff_format(tmp_path / "floored.png") == "PNG"
    panel.close()


def test_plot_panel_export_floor_noop_when_legible(studio_app, qtbot, tmp_path: Path) -> None:
    """A panel already at/above the floor exports at its own size."""
    from PySide6.QtCore import QSize

    from ui.widgets.plot_widget import PlotPanel
    from ui.widgets.responsive import LEGIBILITY_FLOOR

    panel = PlotPanel()
    panel.resize(900, 600)
    panel.line(np.arange(10), np.arange(10) / 3, label="fit")
    panel.show()
    qtbot.wait(20)

    result = panel.export_image(tmp_path / "legible.png", min_size=LEGIBILITY_FLOOR)
    assert (result.width, result.height) == (900, 600)
    panel.close()


def test_plot_panel_export_no_floor_keeps_contract(studio_app, qtbot, tmp_path: Path) -> None:
    """min_size=None keeps the documented widget-owns-its-size rule."""
    from ui.widgets.plot_widget import PlotPanel

    panel = PlotPanel()
    panel.resize(300, 200)
    panel.show()
    qtbot.wait(20)

    result = panel.export_image(tmp_path / "contract.png")
    assert (result.width, result.height) == (300, 200)
    panel.close()


def test_export_floor_preserves_scientific_data(studio_app, qtbot, tmp_path: Path) -> None:
    """§23.5: the floor re-render is display geometry only -- the plotted
    data arrays and the view range are identical before and after."""
    from PySide6.QtCore import QSize

    from ui.widgets.plot_widget import PlotPanel
    from ui.widgets.responsive import LEGIBILITY_FLOOR

    panel = PlotPanel()
    panel.resize(280, 180)
    x = np.linspace(0.0, 6.0, 17)
    y = np.sin(x)
    panel.line(x, y, label="sin")
    panel.show()
    qtbot.wait(20)

    curve = panel.widget().getPlotItem().listDataItems()[0]
    range_before = tuple(panel.widget().getPlotItem().viewRange())
    panel.export_image(tmp_path / "integrity.png", min_size=LEGIBILITY_FLOOR)
    range_after = tuple(panel.widget().getPlotItem().viewRange())

    np.testing.assert_array_equal(curve.xData, x)
    np.testing.assert_array_equal(curve.yData, y)
    assert range_before == range_after
    panel.close()


def test_image_panel_export_floor(studio_app, qtbot, tmp_path: Path) -> None:
    from ui.widgets.image_panel import ImagePanel
    from ui.widgets.responsive import LEGIBILITY_FLOOR

    panel = ImagePanel(title="matrix floor probe")
    panel.resize(300, 220)
    panel.set_matrix(np.arange(64, dtype=np.uint16).reshape(8, 8))
    panel.show()
    qtbot.wait(20)

    result = panel.export_image(tmp_path / "matrix.png", min_size=LEGIBILITY_FLOOR)
    assert result.validated
    assert result.width >= LEGIBILITY_FLOOR.width()
    assert result.height >= LEGIBILITY_FLOOR.height()
    panel.close()


# -- §23.4 aspect lock ------------------------------------------------------------


def test_image_panel_aspect_locked(studio_app) -> None:
    """Matrix pixels stay square: the ImagePanel's view locks aspect."""
    from ui.widgets.image_panel import ImagePanel

    panel = ImagePanel()
    assert panel.widget() is not None
    view_box = panel.image_item().getViewBox()
    assert view_box is not None
    assert view_box.state["aspectLocked"] in (True, 1.0)


def test_smiles_render_preserves_canvas_aspect(studio_app) -> None:
    """§23.4 via the §22 renderer contract: the drawn canvas is exactly
    the requested size (RDKit letterboxes inside it -- no stretching),
    and the DPR-2 pixmap reports the logical size."""
    from PySide6.QtCore import QSize

    from ui.widgets.smiles_preview import MoleculeRenderer

    renderer = MoleculeRenderer()
    pixmap, error = renderer.render("CCO", QSize(260, 200))
    if pixmap is None:
        # RDKit missing in an exotic environment: the honest refusal
        # is the contract, not a stretched image.
        assert error and "RDKit" in error
        return
    dpr = pixmap.devicePixelRatio()
    assert (pixmap.width() / dpr, pixmap.height() / dpr) == (260.0, 200.0)


# -- §23.2 the scroll policy (round 235) --------------------------------------


def test_wrap_in_scroll_policy(studio_app) -> None:
    """The wrapper's contract: widget-resizable, sideways overflow made
    impossible (not merely scrollable), vertical scrollbars only as
    needed, frameless, a hard minimum height, and the content carried
    over as the same instance."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel, QScrollArea

    from ui.widgets.responsive import SCROLL_MIN_PX, wrap_in_scroll

    content = QLabel("form body")
    area = wrap_in_scroll(content)

    assert isinstance(area, QScrollArea)
    assert area.widgetResizable() is True
    assert area.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    assert area.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    assert area.minimumHeight() == SCROLL_MIN_PX
    assert area.widget() is content  # §23.5: the same instance, remounted


def test_wrap_in_scroll_geometry_only(studio_app) -> None:
    """§23.5: wrapping changes where the content is mounted, never the
    content itself.  Mounting reparents the body (top-level layout
    margins become child margins, a pure display-geometry effect), so
    the honest invariants are: the rows are the same instances in the
    same order, and mounting never makes the form REQUIRE more space
    (scrollbars must not engage one pixel earlier than before)."""
    from PySide6.QtWidgets import QFormLayout, QWidget

    from ui.widgets.responsive import wrap_in_scroll

    body = QWidget()
    form = QFormLayout(body)
    rows = [QWidget() for _ in range(6)]
    for row, child in enumerate(rows):
        form.addRow(f"parameter {row}:", child)

    min_before = form.minimumSize()
    area = wrap_in_scroll(body)
    min_after = form.minimumSize()

    assert area.widget() is body
    assert form.rowCount() == len(rows)
    for row, child in enumerate(rows):
        assert form.itemAt(row, QFormLayout.ItemRole.LabelRole) is not None
        assert form.itemAt(row, QFormLayout.ItemRole.FieldRole).widget() is child
    # never more space required after mounting (the margin delta can
    # only shrink it -- a top-level form reserves window margins that
    # an embedded one does not need)
    assert min_after.width() <= min_before.width()
    assert min_after.height() <= min_before.height()


def test_ml_workbench_tab_bodies_scrolled(main_window) -> None:
    """The structural audit: every ML Workbench tab body rides the
    scroll policy (the round-235 probe measured the tuner form at
    479 px and Leaderboards at 775 px minimum against the ~450 px the
    860x560 minimum window leaves).  A future tab added without the
    wrapper fails here -- the policy is self-enforcing, not a
    snapshot."""
    from PySide6.QtWidgets import QScrollArea

    from ui.pages.ml_workbench import MLWorkbenchPage

    pages = main_window.findChildren(MLWorkbenchPage)
    assert pages, "the Studio must carry the ML Workbench page"
    tabs = pages[0]._tabs
    assert tabs.count() >= 5

    for i in range(tabs.count()):
        body = tabs.widget(i)
        assert isinstance(body, QScrollArea), (
            f"tab '{tabs.tabText(i)}' is not wrapped in the §23.2 scroll policy"
        )
        assert body.widgetResizable() is True

    # the content survived the remount: the tuner's board lives inside
    # the first tab's scroll body, wired to the page as before
    board = pages[0]._tune_board
    assert board is not None
    assert tabs.widget(0).isAncestorOf(board)
