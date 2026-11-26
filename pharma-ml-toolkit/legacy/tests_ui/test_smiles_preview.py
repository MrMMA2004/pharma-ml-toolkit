"""SMILES-to-molecule interaction tests (Part-3 §22, round 231).

Real RDKit ``MolDraw2DCairo`` renders, the real shared table model, the
real SA engine in the detail dialog.  Asserted values were printed by
the engines first: aspirin (``CC(=O)Oc1ccccc1C(=O)O``) has formula
``C9H8O4``, weight 180.16 g/mol and Ertl SA 1.58 on the bundled
Contrib model; ethanol (``CCO``) canonicalises to itself.
"""

from __future__ import annotations

import pytest

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ETHANOL = "CCO"
INVALID = "not-a-smiles"


def _renderer(cache_limit: int | None = None):
    from ui.widgets.smiles_preview import CACHE_LIMIT, MoleculeRenderer

    return MoleculeRenderer(cache_limit=cache_limit or CACHE_LIMIT)


def _size(w: int, h: int):
    from PySide6.QtCore import QSize

    return QSize(w, h)


# -- the renderer (§22.3 / §22.4) -----------------------------------------------


def test_renderer_draws_aspirin_at_2x(studio_app) -> None:
    renderer = _renderer()
    pixmap, error = renderer.render(ASPIRIN, _size(260, 200))
    assert error is None
    assert pixmap is not None and not pixmap.isNull()
    # drawn at 2x physical pixels for crisp high-DPI display
    assert pixmap.width() == 520 and pixmap.height() == 400
    assert pixmap.devicePixelRatio() == 2.0


def test_renderer_caches_identical_requests(studio_app) -> None:
    renderer = _renderer()
    first, _ = renderer.render(ASPIRIN, _size(260, 200))
    second, _ = renderer.render(ASPIRIN, _size(260, 200))
    assert second is first  # §22.3: identical molecules are not redrawn
    assert renderer.render_count == 1
    assert renderer.cache_hits == 1
    other, _ = renderer.render(ASPIRIN, _size(460, 340))  # different size
    assert other is not first and renderer.render_count == 2


def test_renderer_invalid_smiles_is_honest(studio_app) -> None:
    renderer = _renderer()
    pixmap, error = renderer.render(INVALID, _size(260, 200))
    assert pixmap is None
    assert error is not None and "Invalid SMILES" in error
    assert INVALID in error  # §22.4: the original value is preserved
    pixmap, error = renderer.render("", _size(260, 200))
    assert pixmap is None and error is not None and "Empty" in error


def test_renderer_cache_evicts_fifo(studio_app) -> None:
    renderer = _renderer(cache_limit=2)
    size = _size(260, 200)
    renderer.render(ETHANOL, size)
    renderer.render(ASPIRIN, size)
    assert renderer.render_count == 2
    renderer.render("c1ccccc1", size)  # evicts ETHANOL (oldest)
    assert renderer.render_count == 3
    renderer.render(ETHANOL, size)  # was evicted -> a real redraw
    assert renderer.render_count == 4


# -- the hover popup ------------------------------------------------------------


def test_popup_structure_and_message_paths(studio_app) -> None:
    from ui.widgets.smiles_preview import MoleculePopup

    popup = MoleculePopup()
    renderer = _renderer()
    pixmap, _ = renderer.render(ASPIRIN, _size(260, 200))
    popup.show_structure(pixmap, ASPIRIN)
    assert popup._caption.text() == ASPIRIN
    assert popup._image.pixmap() is not None

    popup.show_message("Invalid SMILES — RDKit cannot parse 'x'.")
    assert popup._image.pixmap() is None or popup._image.pixmap().isNull()
    assert "no structure drawn" in popup._image.text()
    assert "Invalid SMILES" in popup._caption.text()
    popup.hide()


# -- the detail dialog (§22.5) --------------------------------------------------


def test_detail_dialog_aspirin_real_properties(studio_app) -> None:
    from ui.widgets.smiles_preview import SmilesDetailDialog

    dialog = SmilesDetailDialog(ASPIRIN)
    try:
        rows = dialog.property_rows()
        assert rows["original SMILES"] == ASPIRIN
        assert rows["canonical SMILES"] == ASPIRIN
        assert rows["molecular formula"] == "C9H8O4"
        assert rows["molecular weight"].startswith("180.16")
        sa = float(rows["synthetic accessibility"].split()[0])
        assert 1.0 <= sa <= 10.0
        assert sa == pytest.approx(1.58, abs=0.05)  # the pinned Ertl value
        assert "Ertl" in rows["SA method"]
        assert dialog.image_pixmap() is not None
        assert dialog._copy_btn.isEnabled()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_detail_dialog_invalid_smiles_never_invents(studio_app) -> None:
    from ui.widgets.smiles_preview import SmilesDetailDialog

    dialog = SmilesDetailDialog(INVALID)
    try:
        rows = dialog.property_rows()
        assert rows["original SMILES"] == INVALID  # preserved for debugging
        assert rows["parse status"] == "invalid — no properties calculated"
        assert "canonical SMILES" not in rows
        assert "molecular weight" not in rows
        assert dialog.image_pixmap() is None or dialog.image_pixmap().isNull()
        # the copy button degrades to copying the original string
        assert dialog._copy_btn.isEnabled()
        assert dialog._copy_btn.text() == "Copy original SMILES"
    finally:
        dialog.close()
        dialog.deleteLater()


# -- the controller on the shared table (§22.2) ----------------------------------


def _bound_view(studio_app, qtbot):
    import pandas as pd

    from ui.widgets.data_table import DataTableView

    view = DataTableView()
    view.resize(700, 300)
    view.bind_frame(
        pd.DataFrame(
            {
                "rank": [1, 2],
                "smiles": [ETHANOL, INVALID],
                "qed": [0.41, float("nan")],
            }
        )
    )
    view.show()
    qtbot.wait(20)
    studio_app.processEvents()
    studio_app.processEvents()
    return view


def test_controller_hover_dwell_and_popup(studio_app, qtbot) -> None:
    from ui.widgets.smiles_preview import SmilesPreviewController

    view = _bound_view(studio_app, qtbot)
    controller = SmilesPreviewController(view, columns=("smiles",), dwell_ms=50)
    index = view.model().index(0, 1)  # smiles column
    center = view.visualRect(index).center()
    controller._hover(center)
    assert controller._pending == ETHANOL
    assert controller._dwell.isActive()
    qtbot.wait(120)  # dwell fires -> popup renders the real structure
    assert controller._popup.isVisible()
    assert controller._popup._caption.text() == ETHANOL
    assert controller._popup._image.pixmap() is not None
    controller._cancel()
    assert not controller._popup.isVisible()
    view.close()


def test_controller_invalid_cell_shows_the_validation_message(studio_app, qtbot) -> None:
    from ui.widgets.smiles_preview import SmilesPreviewController

    view = _bound_view(studio_app, qtbot)
    controller = SmilesPreviewController(view, columns=("smiles",), dwell_ms=50)
    index = view.model().index(1, 1)  # the invalid row
    center = view.visualRect(index).center()
    controller._hover(center)
    controller._show_popup()
    assert controller._popup.isVisible()
    assert "no structure drawn" in controller._popup._image.text()
    assert INVALID in controller._popup._caption.text()
    controller._cancel()
    view.close()


def test_controller_ignores_non_smiles_columns(studio_app, qtbot) -> None:
    from ui.widgets.smiles_preview import SmilesPreviewController

    view = _bound_view(studio_app, qtbot)
    controller = SmilesPreviewController(view, columns=("smiles",))
    index = view.model().index(0, 2)  # the numeric qed column
    center = view.visualRect(index).center()
    controller._hover(center)
    assert controller._pending is None
    assert not controller._dwell.isActive()
    view.close()


def test_controller_open_detail_programmatic(studio_app, qtbot) -> None:
    from ui.widgets.smiles_preview import SmilesPreviewController

    view = _bound_view(studio_app, qtbot)
    controller = SmilesPreviewController(view, columns=("smiles",))
    dialog = controller.open_detail(ASPIRIN)
    try:
        assert dialog.property_rows()["molecular formula"] == "C9H8O4"
    finally:
        dialog.close()
        dialog.deleteLater()
    view.close()


def test_generative_page_table_carries_the_previews(studio_app) -> None:
    """The page's shared table (SA board / mutants / children) is wired."""
    from ui.pages.generative import GenerativeWorkbenchPage

    page = GenerativeWorkbenchPage()
    controller = page._smiles_previews
    assert controller._columns == ("smiles", "canonical", "mutant", "child")
    assert controller._view is page._table
    page.deleteLater()
