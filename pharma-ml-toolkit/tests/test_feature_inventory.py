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
CARD_DIRECTORY = REPO_ROOT / "src" / "components" / "explorer" / "playground" / "card-directory.ts"
API_ROOT = REPO_ROOT / "src" / "app" / "api"

BACKEND_ONLY_MODULES = ("gnn", "geometric", "pruning", "quantization", "harness")

STATUS_COUNTS = {"A": 34, "B": 7, "C": 5, "D": 1, "E": 1, "F": 3}


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
        for mod in BACKEND_ONLY_MODULES:
            for page in pages:
                assert mod not in page.read_text(encoding="utf-8"), (
                    f"{mod} now referenced by {page.name} — re-classify it"
                )
            for web in web_files:
                assert mod not in web.read_text(encoding="utf-8"), (
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
        """18 pages + 12 web areas + 11 packages = 41 A/B rows; C/D/E/F add
        10 more (5 backend-only, 1 dashboard, 1 conformal, 3 external)."""
        assert STATUS_COUNTS["A"] + STATUS_COUNTS["B"] == 41
        assert sum(STATUS_COUNTS.values()) == 51

    def test_synthetic_demo_register(self) -> None:
        text = _doc()
        for topic in ("MD trajectories", "Cell-line response tables", "Docking scores"):
            assert f"**{topic}**" in text, f"synthetic register missing: {topic}"
