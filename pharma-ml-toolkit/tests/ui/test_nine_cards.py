"""§5 nine-card dashboard contract — the registry, the grid, the honesty.

The directive's mapping table and exact card order live in
``ui/nine_cards.py``; these pins keep them true:

* the registry is exactly the nine §5 cards, in the directive's order,
  each routed to the live ``SECTIONS`` keys of ``ui.main_window``;
* the mapping covers every specialized page (settings stays out by the
  directive's own table) and duplicates none;
* the optional-dependency chips state the real boundaries — docking's
  bundled Windows docking engines and MD's external trajectory data included;
* the rendered grid navigates the real MainWindow (entry buttons route
  through ``go_to``; the page stack stays 18 widgets — cards are entry
  points, never duplicated implementations).
"""

from __future__ import annotations

from PySide6.QtWidgets import QPushButton

from ui.main_window import SECTIONS
from ui.nine_cards import NINE_CARDS, build_nine_card_grid, live_preview, probe_available

SECTION_INDEX = {key: i for i, (key, _title) in enumerate(SECTIONS)}

#: Part-4 §5: the exact order, left to right.
DIRECTIVE_ORDER = [
    "Introduction",
    "Datasets",
    "Machine Learning",
    "Deep Learning",
    "Generative AI",
    "Molecular Docking",
    "PK/PD",
    "Molecular Dynamics Analysis",
    "Visualization",
]

#: Part-4 §5 mapping table: card title -> the existing implementation.
DIRECTIVE_MAPPING = {
    "Introduction": {"dashboard"},
    "Datasets": {"data", "chemistry", "utilities"},
    "Machine Learning": {"models", "mlworkbench", "unsupervised", "bestofml", "foundations"},
    "Deep Learning": {"deep_learning"},
    "Generative AI": {"generative"},
    "Molecular Docking": {"docking"},
    "PK/PD": {"pkpd", "biology", "responseai"},
    "Molecular Dynamics Analysis": {"molecular_dynamics"},
    "Visualization": {"imagelab"},
}


class TestRegistryContract:
    def test_exactly_nine_cards_in_directive_order(self) -> None:
        assert [c.title for c in NINE_CARDS] == DIRECTIVE_ORDER
        assert len({c.card_id for c in NINE_CARDS}) == 9
        assert [c.badge for c in NINE_CARDS] == [f"{i:02d}" for i in range(1, 10)]

    def test_every_entry_section_is_a_live_nav_key(self) -> None:
        live = dict(SECTIONS)
        for card in NINE_CARDS:
            assert card.entries, f"{card.title} has no entry point"
            for entry in card.entries:
                assert entry.section in live, f"{entry.section} is not a SECTIONS key"
                assert entry.label, f"{card.title}/{entry.section} has an empty label"

    def test_mapping_matches_the_directive_table(self) -> None:
        for card in NINE_CARDS:
            expected = DIRECTIVE_MAPPING[card.title]
            got = {e.section for e in card.entries}
            assert got == expected, f"{card.title}: {got} != the §5 mapping {expected}"

    def test_all_specialized_pages_still_covered(self) -> None:
        covered = {e.section for c in NINE_CARDS for e in c.entries}
        every_page_but_settings = {key for key, _ in SECTIONS} - {"settings"}
        assert covered == every_page_but_settings

    def test_titles_and_descriptions_are_substantive(self) -> None:
        for card in NINE_CARDS:
            assert len(card.description) >= 40, f"{card.title} description too thin"
            assert len(card.capabilities) >= 60, f"{card.title} capability summary too thin"


class TestHonesty:
    def test_probes_are_boolean_and_exception_safe(self) -> None:
        assert isinstance(probe_available("rdkit"), bool)
        assert probe_available("definitely-not-importable-xyz") is False
        assert probe_available("PySide6.QtWidgets") is True

    def test_docking_card_declares_engine_and_runtime_boundaries(self) -> None:
        docking = next(c for c in NINE_CARDS if c.title == "Molecular Docking")
        assert "Windows Vina" in docking.external
        assert "compatible runtime" in docking.external
        assert "user-supplied receptor" in docking.external
        assert "Studio focuses on preparation and analysis" in docking.external

    def test_md_card_declares_the_external_data_source(self) -> None:
        md = next(c for c in NINE_CARDS if c.title == "Molecular Dynamics Analysis")
        assert "GROMACS" in md.external
        assert "external" in md.external

    def test_core_only_cards_say_so(self) -> None:
        for title in ("Datasets", "PK/PD"):
            card = next(c for c in NINE_CARDS if c.title == title)
            assert card.optional == (), f"{title} claims optional deps it does not have"
            assert card.external == ""

    def test_live_previews_are_computed_strings(self) -> None:
        for card in NINE_CARDS:
            preview = live_preview(card.card_id)
            assert isinstance(preview, str) and preview, f"{card.title} preview empty"
        assert "pharmaml" in live_preview("introduction")
        registry_preview = live_preview("machine-learning")
        assert registry_preview.split()[0].isdigit(), "model-registry preview must be numeric"


class TestRenderedGrid:
    def test_grid_builds_nine_cards_row_major(self, studio_app) -> None:
        host, frames = build_nine_card_grid()
        grid = host.layout()
        assert len(frames) == 9
        positions = [grid.indexOf(frames[c.card_id]) for c in NINE_CARDS]
        assert positions == sorted(positions), "cards must render in the §5 order"
        for i, card_id in enumerate([c.card_id for c in NINE_CARDS]):
            row, col, _, _ = grid.getItemPosition(i)
            assert (row, col) == (i // 3, i % 3)

    def test_every_card_has_working_entry_buttons(self, studio_app) -> None:
        _host, frames = build_nine_card_grid()
        for card in NINE_CARDS:
            frame = frames[card.card_id]
            buttons = frame.findChildren(QPushButton)
            assert len(buttons) == len(card.entries)
            for btn in buttons:
                assert btn.objectName().startswith("nineCardEntry-")

    def test_entry_button_routes_to_the_real_page(self, main_window) -> None:
        host, _frames = build_nine_card_grid()
        btn = next(
            b
            for b in host.findChildren(QPushButton)
            if b.objectName() == "nineCardEntry-models"
        )
        main_window.setCentralWidget(host)  # give the button a live window chain
        main_window.go_to("dashboard")
        assert main_window.pages.currentIndex() != SECTION_INDEX["models"]
        btn.click()
        assert main_window.pages.currentIndex() == SECTION_INDEX["models"]

    def test_dashboard_page_stays_census_plus_cards(self, main_window) -> None:
        # the cards are entry points: the page stack must stay 18 widgets
        assert main_window.pages.count() == 18
        dashboard = main_window.dashboard_page
        from PySide6.QtWidgets import QLabel

        labels = " ".join(l.text() for l in dashboard.findChildren(QLabel))
        assert "Workflow cards" in labels
        assert "Example datasets" in labels  # the census sections survive


class TestPhase3ResponsiveGrid:
    """Part-4 §8 Phase-3: responsive layouts — 3/2/1 columns, order kept.

    The directive's canonical 3×3 is the default; narrower windows
    re-flow row-major to two columns (below 880 logical px) and one
    (below 560).  The re-flow must reuse the SAME card widgets — a
    width change can never rebuild state or reorder the §5 sequence.
    """

    def test_columns_for_width_thresholds(self) -> None:
        from ui.nine_cards import (
            COLUMNS_2_MIN_WIDTH,
            COLUMNS_3_MIN_WIDTH,
            columns_for_width,
        )

        assert columns_for_width(1920) == 3
        assert columns_for_width(COLUMNS_3_MIN_WIDTH) == 3
        assert columns_for_width(COLUMNS_3_MIN_WIDTH - 1) == 2
        assert columns_for_width(COLUMNS_2_MIN_WIDTH) == 2
        assert columns_for_width(COLUMNS_2_MIN_WIDTH - 1) == 1
        assert columns_for_width(100) == 1

    def test_grid_defaults_to_the_directive_three_columns(self, studio_app) -> None:
        host, _frames = build_nine_card_grid()
        assert host.current_columns() == 3

    def test_reflow_two_columns_preserves_row_major_order(self, studio_app) -> None:
        host, frames = build_nine_card_grid()
        host.reflow(2)
        grid = host.layout()
        for i, card in enumerate(NINE_CARDS):
            row, col, _, _ = grid.getItemPosition(grid.indexOf(frames[card.card_id]))
            assert (row, col) == (i // 2, i % 2)

    def test_reflow_one_column_preserves_order(self, studio_app) -> None:
        host, frames = build_nine_card_grid()
        host.reflow(1)
        grid = host.layout()
        for i, card in enumerate(NINE_CARDS):
            row, col, _, _ = grid.getItemPosition(grid.indexOf(frames[card.card_id]))
            assert (row, col) == (i, 0)

    def test_reflow_reuses_the_same_frame_objects(self, studio_app) -> None:
        # the r262 id-reuse lesson: hold the REFERENCES, compare identity
        host, frames = build_nine_card_grid()
        held = list(frames.values())
        host.reflow(2)
        host.reflow(1)
        host.reflow(3)
        after = host.frames()
        grid = host.layout()
        for card in NINE_CARDS:
            assert after[card.card_id] is frames[card.card_id]
            assert grid.indexOf(after[card.card_id]) >= 0
        assert all(grid.indexOf(frame) >= 0 for frame in held)

    def test_reflow_rejects_impossible_column_counts(self, studio_app) -> None:
        import pytest

        host, _frames = build_nine_card_grid()
        for bad in (0, 4, -1):
            with pytest.raises(ValueError):
                host.reflow(bad)

    def test_resize_event_reflows_by_width_class(self, studio_app) -> None:
        from PySide6.QtCore import QSize
        from PySide6.QtGui import QResizeEvent

        host, _frames = build_nine_card_grid()
        host.resizeEvent(QResizeEvent(QSize(1200, 900), QSize(100, 30)))
        assert host.current_columns() == 3
        host.resizeEvent(QResizeEvent(QSize(700, 900), QSize(1200, 900)))
        assert host.current_columns() == 2
        host.resizeEvent(QResizeEvent(QSize(400, 900), QSize(700, 900)))
        assert host.current_columns() == 1
        host.resizeEvent(QResizeEvent(QSize(1000, 900), QSize(400, 900)))
        assert host.current_columns() == 3

    def test_dashboard_uses_the_responsive_grid(self, main_window) -> None:
        from ui.nine_cards import NineCardGrid

        # part6 §2.1 (Phase 1): the grid is PROMOTED to the nine-card home;
        # the Dashboard page keeps its own grid (the page survives as a
        # section).  Two grids total, one per host, both the real widget.
        grids = main_window.findChildren(NineCardGrid)
        assert len(grids) == 2, (
            "the promoted home AND the Dashboard page each host one responsive grid"
        )
        assert main_window.mode.currentIndex() == 0
        home_grid = main_window.home.cards_host
        assert isinstance(home_grid, NineCardGrid)


class TestPhase3SmilesPreview:
    """Part-4 §8 Phase-3: SMILES previews on the cards, where meaningful.

    The Datasets card carries the REAL first molecule of the screening
    library (the same file the Data Explorer opens) through the §22
    renderer — cached, honest on refusal, never fabricated.  No other
    card gets a molecule: §5 says "where meaningful", not "everywhere".
    """

    def test_preview_source_is_pinned_to_the_real_file(self) -> None:
        from ui.nine_cards import datasets_preview_smiles

        found = datasets_preview_smiles()
        assert found is not None, "examples/data/screening_library.csv must stay shippable"
        smiles, name = found
        assert name == "aspirin"
        assert smiles == "CC(=O)Oc1ccccc1C(=O)O"

    def test_datasets_card_carries_the_real_smiles_preview(self, studio_app) -> None:
        from PySide6.QtWidgets import QLabel

        _host, frames = build_nine_card_grid()
        frame = frames["datasets"]
        images = [w for w in frame.findChildren(QLabel) if w.objectName() == "nineCardMolecule"]
        assert len(images) == 1
        image = images[0]
        # rdkit + QApplication are live in this environment: the thumbnail
        # must be a real rendered pixmap, never a placeholder
        assert image.pixmap() is not None and not image.pixmap().isNull()
        captions = " ".join(
            w.text() for w in frame.findChildren(QLabel) if w.objectName() == "nineCardPreview"
        )
        assert "screening_library.csv" in captions
        assert "aspirin" in captions

    def test_only_the_datasets_card_has_a_molecule_visual(self, studio_app) -> None:
        from PySide6.QtWidgets import QLabel

        _host, frames = build_nine_card_grid()
        for card_id, frame in frames.items():
            images = [w for w in frame.findChildren(QLabel) if w.objectName() == "nineCardMolecule"]
            if card_id == "datasets":
                assert images, "the Datasets card must carry the SMILES preview"
            else:
                assert not images, f"{card_id} must not render a gratuitous molecule"

    def test_render_preview_refuses_invalid_smiles_honestly(self, studio_app) -> None:
        from ui.nine_cards import _render_preview

        pixmap, error = _render_preview("not-a-smiles")
        assert pixmap is None
        assert error and "Invalid SMILES" in error
        empty_pixmap, empty_error = _render_preview("")
        assert empty_pixmap is None and empty_error
