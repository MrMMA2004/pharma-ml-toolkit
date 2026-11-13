"""Pins for ``docs/FEATURES.md`` — the §4.5 truthful feature inventory.

The inventory is a traceability matrix (feature → backend → UI → test)
whose rows were derived from the LIVE tree, not from prose.  These pins
keep it that way: every live registry the matrix claims to cover (the 18
Studio nav keys, the 33 web card ids, the 27 API routes, the backend-only
module list) is re-derived here and asserted present in the document —
so adding a page/card/route without truing the matrix fails the release,
and so does retiring one.  The class-C rows are additionally checked for
honesty in the other direction: the modules really must have no GUI/web
references for the "backend-only" classification to stand.
"""

from __future__ import annotations

import re
from pathlib import Path

TOOLKIT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TOOLKIT_ROOT.parent
DOC = TOOLKIT_ROOT / "docs" / "FEATURES.md"
MAIN_WINDOW = TOOLKIT_ROOT / "ui" / "main_window.py"
NINE_CARDS_SRC = TOOLKIT_ROOT / "ui" / "nine_cards.py"
CARD_DIRECTORY = REPO_ROOT / "src" / "components" / "explorer" / "playground" / "card-directory.ts"
API_ROOT = REPO_ROOT / "src" / "app" / "api"

BACKEND_ONLY_MODULES: tuple[str, ...] = ()

#: v1.2.190 (part6 Phase 3, batch 4 — the closer): the last three
#: class-C modules left the table — the classification conformal card
#: rides the ML Workbench's Statistical tools tab, and the split-overlap
#: audit + SAR matched-pairs panels ride the Data Explorer. Every
#: backend module now has a GUI workflow.
STATUS_COUNTS = {"A": 46, "B": 6, "C": 0, "D": 0, "E": 0, "F": 3}


def _doc() -> str:
    assert DOC.exists(), f"missing {DOC.relative_to(TOOLKIT_ROOT)}"
    return DOC.read_text(encoding="utf-8")


def _nav_keys() -> list[str]:
    src = MAIN_WINDOW.read_text(encoding="utf-8")
    block = src.split("SECTIONS", 1)[1].split("class MainWindow", 1)[0]
    return re.findall(r'\("([a-z_]+)",', block)


def _card_ids() -> list[str]:
    src = CARD_DIRECTORY.read_text(encoding="utf-8")
    return re.findall(r'^\s*id: "([a-z0-9-]+)",', src, flags=re.MULTILINE)


def _route_names() -> list[str]:
    routes: list[str] = []
    for route in sorted(API_ROOT.rglob("route.ts")):
        rel = route.relative_to(API_ROOT).parent
        routes.append("(root)" if str(rel) == "." else "/".join(rel.parts))
    return routes


def _nine_card_titles() -> list[str]:
    """The card titles, in registry order, re-derived from the live source."""
    return _card_constant_titles(NINE_CARDS_SRC.read_text(encoding="utf-8"))


def _card_constant_titles(src: str) -> list[str]:
    titles: list[str] = []
    for match in re.finditer(r"^CARD_[A-Z_]+ = NineCard\(", src, flags=re.MULTILINE):
        start = match.end()
        title = re.search(r'title="([^"]+)"', src[start : start + 400])
        assert title, "a NineCard constant is missing its title"
        titles.append(title.group(1))
    return titles


def _nine_card_entry_keys() -> list[str]:
    """Every nav key the nine cards route to, re-derived from the source."""
    src = NINE_CARDS_SRC.read_text(encoding="utf-8")
    return sorted(set(re.findall(r'CardEntry\("([a-z_]+)"', src)))


class TestInventoryStructure:
    def test_purpose_header_and_freeze_stamp(self) -> None:
        text = _doc()
        assert "§4.5" in text
        assert "traceability matrix" in text
        version = (TOOLKIT_ROOT / "VERSION").read_text(encoding="utf-8").strip()
        assert version in text, "the doc must name the version it ships with"

    def test_six_status_classes_in_legend(self) -> None:
        text = _doc()
        for letter, phrase in (
            ("A", "Implemented and tested"),
            ("B", "Implemented but incompletely tested"),
            ("C", "Backend available but not integrated into the GUI"),
            ("D", "Partially implemented"),
            ("E", "Not implemented"),
            ("F", "Blocked by an optional dependency or external engine"),
        ):
            assert f"**{letter}**" in text, f"legend missing class {letter}"
            assert phrase in text, f"legend missing the {letter} phrase"

    def test_three_honesty_rules_verbatim(self) -> None:
        text = " ".join(_doc().split())  # collapse line-wraps
        assert "A GUI button is not proof of backend functionality" in text
        assert "A backend function is not proof of GUI integration" in text
        assert (
            "A synthetic demonstration is not proof of real experimental or "
            "production data support" in text
        )


class TestRegistryCoverage:
    def test_all_studio_nav_keys_mapped(self) -> None:
        keys = _nav_keys()
        assert len(keys) == 18, f"SECTIONS drifted: {len(keys)} keys"
        text = _doc()
        missing = [k for k in keys if f"(`{k}`)" not in text]
        assert not missing, f"matrix missing nav keys: {missing}"

    def test_all_web_card_ids_mapped(self) -> None:
        ids = _card_ids()
        assert len(ids) == 33, f"card directory drifted: {len(ids)} cards"
        text = _doc()
        missing = [c for c in ids if f"`{c}`" not in text]
        assert not missing, f"matrix missing card ids: {missing}"

    def test_all_web_routes_mapped(self) -> None:
        routes = _route_names()
        assert len(routes) == 27, f"route set drifted: {len(routes)} routes"
        text = _doc()
        missing = [r for r in routes if r != "(root)" and f"`{r}`" not in text]
        assert not missing, f"matrix missing routes: {missing}"

    def test_backend_packages_listed(self) -> None:
        text = _doc()
        for pkg in (
            "biology", "chem", "docking", "generative", "md", "ml",
            "pharma", "pkpd", "utils", "viz",
        ):
            assert f"`{pkg}`" in text, f"package {pkg} missing from the matrix"


class TestBackendOnlyHonesty:
    def test_c_modules_listed(self) -> None:
        text = _doc()
        for mod in BACKEND_ONLY_MODULES:
            assert f"`ml/{mod}.py`" in text, f"class-C row missing: {mod}"

    def test_c_modules_really_have_no_gui(self) -> None:
        """The honesty check in the other direction: if a page or web file
        starts referencing one of these modules, the C classification (and
        this test) must be revisited.  Documentation prose (overview.tsx,
        api-reference.tsx) is excluded — the §4.5 rule is that a *GUI
        button/card/route* is the integration evidence, not a mention in
        the narrative tab."""
        pages = list((TOOLKIT_ROOT / "ui" / "pages").glob("*.py"))
        prose = {"overview.tsx", "api-reference.tsx"}
        web_files = [
            p for p in [
                *API_ROOT.rglob("*.ts"),
                *API_ROOT.rglob("*.tsx"),
                *(REPO_ROOT / "src" / "components").rglob("*.tsx"),
                *(REPO_ROOT / "src" / "lib").glob("*.ts"),
            ]
            if p.name not in prose
        ]
        import re as _re

        for mod in BACKEND_ONLY_MODULES:
            pattern = _re.compile(rf"\b{mod}\b")
            for page in pages:
                assert not pattern.search(page.read_text(encoding="utf-8")), (
                    f"{mod} now referenced by {page.name} — re-classify it"
                )
            for web in web_files:
                assert not pattern.search(web.read_text(encoding="utf-8")), (
                    f"{mod} now referenced by {web.name} — re-classify it"
                )


class TestStatusSummary:
    def test_summary_counts(self) -> None:
        text = _doc()
        for letter, count in STATUS_COUNTS.items():
            row = re.search(
                rf"^\| \*\*{letter}\*\* \| (\d+) \|", text, flags=re.MULTILINE
            )
            assert row, f"summary row {letter} missing"
            assert int(row.group(1)) == count, (
                f"summary {letter}={row.group(1)} != pinned {count}"
            )

    def test_row_total_consistency(self) -> None:
        """18 pages + 12 web areas + 11 packages = 52 A/B FEATURE rows; E/F
        add 3 more (3 external) — v1.2.187-190 moved the statistical
        tools, the ML statistics cluster, the Deep Learning cluster AND
        the last three class-C modules onto existing pages' A rows (no
        new pages, so the module count grows without the page count
        changing; the registry's 52 integrated rows carry the per-module
        flip). The nine-card organization rides the Dashboard page row
        since v1.2.169 (§10)."""
        assert STATUS_COUNTS["A"] + STATUS_COUNTS["B"] == 52
        assert sum(STATUS_COUNTS.values()) == 55

    def test_synthetic_demo_register(self) -> None:
        text = _doc()
        for topic in ("MD trajectories", "Cell-line response tables", "Docking scores"):
            assert f"**{topic}**" in text, f"synthetic register missing: {topic}"


class TestNineCardMapping:
    """The §5 nine-card dashboard pins (v1.2.169, FEATURES.md §10).

    The mapping is re-derived from the live `ui/nine_cards.py` source,
    exactly like the nav keys — the doc may not drift from the registry.
    """

    def test_registry_has_exactly_nine_cards(self) -> None:
        titles = _nine_card_titles()
        assert len(titles) == 9, f"expected 9 cards, found {len(titles)}: {titles}"

    def test_doc_has_the_mapping_section(self) -> None:
        assert "## 10. The nine-card dashboard mapping" in _doc()

    def test_every_card_title_in_the_doc(self) -> None:
        text = _doc()
        for title in _nine_card_titles():
            assert title in text, f"card '{title}' missing from FEATURES.md"

    def test_every_entry_key_is_live_and_in_the_doc(self) -> None:
        text = _doc()
        live = set(_nav_keys())
        for key in _nine_card_entry_keys():
            assert key in live, f"nine-card entry '{key}' is not a SECTIONS key"
            assert f"`{key}`" in text, f"entry '{key}' missing from the §10 table"

    def test_entries_cover_every_specialized_page(self) -> None:
        entries = set(_nine_card_entry_keys())
        every_page_but_settings = set(_nav_keys()) - {"settings"}
        assert entries == every_page_but_settings, (
            "the nine cards must route to every page (settings stays out "
            "by the directive's own mapping table)"
        )
