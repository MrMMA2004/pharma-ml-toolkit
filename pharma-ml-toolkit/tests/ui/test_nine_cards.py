"""§5 nine-card dashboard contract — the registry, the grid, the honesty.

The directive's mapping table and exact card order live in
``ui/nine_cards.py``; these pins keep them true:

* the registry is exactly the nine §5 cards, in the directive's order,
  each routed to the live ``SECTIONS`` keys of ``ui.main_window``;
* the mapping covers every specialized page (settings stays out by the
  directive's own table) and duplicates none;
* the optional-dependency chips state the real boundaries — docking's
  external Vina engine and MD's external trajectory data included;
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

    def test_docking_card_declares_the_external_engine(self) -> None:
        docking = next(c for c in NINE_CARDS if c.title == "Molecular Docking")
        assert "Vina" in docking.external
        assert "never bundled" in docking.external

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
