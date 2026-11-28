"""The Visualization dataset-column bridge drift guard (VIZ-1, the
roadmap's row 9 — the third 3-item set's opener).

The lane spans three sources that must never drift apart:

* the established bridge client -- ``services/datasets/client.ts``'s
  ``column()`` (the §11.1 bounded lane every other workspace rides,
  route-pinned by ``test_frontend_services.py``);
* the page's bridge block -- ``features/visualization/
  VisualizationPage.tsx`` stage 3 (the dataset select, the column
  select, the pull button, the honest notes);
* the lane's own vitest file -- ``DatasetColumnBridge.test.tsx`` (the
  roadmap's three acceptance clauses + the null/bound guards).

Plus the roadmap's own honesty clauses:

* one code path, two entry points -- the bridge fills the SAME
  ``valuesText`` the paste lane types into, so the stat request rides
  ``parseNumberList`` unchanged (never a parallel parser);
* a ``null`` cell (``json_safe``'s NaN → null) is a row-labeled
  error, never ``Number(null) === 0``;
* a truncated pull names its 500-row bound (§17.3: the bounded view
  never masquerades as the whole column);
* the paste lane stays the free-form lane, unchanged (the
  regression clause);
* the roadmap's row 9 status line records this round honestly.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGE = (
    REPO / "frontend" / "src" / "features" / "visualization" / "VisualizationPage.tsx"
)
CLIENT = REPO / "frontend" / "src" / "services" / "datasets" / "client.ts"
VITEST = (
    REPO
    / "frontend"
    / "src"
    / "features"
    / "visualization"
    / "__tests__"
    / "DatasetColumnBridge.test.tsx"
)
ROADMAP = (
    REPO
    / "docs"
    / "QA"
    / "WORKFLOW_ENHANCEMENT_ROADMAP.md"
)


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_page_carries_the_bridge_card_and_the_select_pair() -> None:
    """Stage 3 offers the dataset select + the column select."""
    src = _page()
    assert "Registered dataset column bridge" in src, (
        "the Visualization stage-3 bridge card title was lost"
    )
    for label in (
        '"the registered dataset"',
        '"the dataset column"',
    ):
        assert f'aria-label={label}' in src, (
            f"the bridge lost its aria-labelled select {label}"
        )
    assert "Pull column" in src, "the bridge lost its pull button"
    # the button stays disarmed until the pair is complete — the
    # gating clause every honest action button carries
    m = re.search(r"disabled=\{[^}]*bridgeBusy", src, re.DOTALL)
    assert m, "the bridge pull button lost its busy/selection gating"


def test_page_pulls_one_column_through_the_client_bridge() -> None:
    """The stat family's values ride the SAME ``datasetsApi.column`` lane."""
    src = _page()
    assert 'from "@/services/datasets/client"' in src, (
        "the page no longer imports the datasets service client"
    )
    assert len(re.findall(r"await datasetsApi\s*\.\s*column\(", src)) == 1, (
        "the VIZ-1 bridge must pull exactly one column through the "
        "client lane (the stat family's own values lane)"
    )
    assert re.search(r"datasetsApi\s*\.\s*list\(", src), (
        "the bridge lost its registry mount fetch"
    )
    assert re.search(r"datasetsApi\s*\.\s*view\(", src), (
        "the bridge lost its dataset view fetch (the columns source)"
    )


def test_page_never_hand_rolls_the_http_lane() -> None:
    """The page itself never issues raw ``fetch`` calls."""
    src = _page()
    assert "fetch(" not in src, (
        "the page hand-rolled an HTTP call — the typed client is the "
        "only lane (§11.1's control-plane rule)"
    )


def test_one_code_path_two_entry_points() -> None:
    """The bridge fills the paste lane's own ``valuesText`` — the stat
    request keeps riding ``parseNumberList`` (no parallel parser)."""
    src = _page()
    assert re.search(r"setValuesText\(", src), (
        "the bridge no longer feeds the paste lane's values textarea"
    )
    assert "function parseNumberList(" in src, (
        "the paste lane's own parser was removed"
    )
    # the generate lane still parses valuesText through that parser
    m = re.search(
    r'if \(family === "stat"\) \{\s*'
    r"const values = parseNumberList\(valuesText",
        src,
    )
    assert m, "the stat generate lane no longer rides parseNumberList"


def test_numeric_cell_guards_null_and_non_numeric_cells() -> None:
    """``numericCell`` refuses null and non-numeric cells honestly."""
    src = _page()
    m = re.search(
        r"function numericCell\([^)]*\)[^{]*\{(.{0,1200}?)\n\}",
        src,
        re.DOTALL,
    )
    assert m, "the numericCell converter drifted or was removed"
    body = m.group(1)
    assert "never silently zero" in body, (
        "the null-cell guard lost its honesty message"
    )
    assert "Number.isFinite" in body, (
        "the non-numeric-cell guard lost its finite check"
    )
    assert "?? 0" not in body, (
        "a silent zero default appeared in the converter — "
        "Number(null) === 0 would fabricate a value the frame never "
        "measured"
    )


def test_truncation_is_named_honestly() -> None:
    """A truncated pull names its bound (§17.3)."""
    src = _page()
    assert ".truncated" in src, (
        "the bridge no longer reads the truncation flag"
    )
    assert "bounded at 500 rows" in src, (
        "the honest bound note (the chemistry engine's own cap) was lost"
    )


def test_paste_lane_is_unchanged() -> None:
    """The free-form paste lane keeps its own parser, button, samples."""
    src = _page()
    assert "Load &amp; validate" in src, (
        "the paste lanes' own load button was removed"
    )
    assert "Load samples" in src, (
        "the paste lanes' sample button was removed"
    )
    assert 'placeholder={"one value per line' in src, (
        "the values textarea's own placeholder was lost"
    )


def test_registry_lane_renders_the_honest_empty_state() -> None:
    """An empty registry is stated, not hidden."""
    src = _page()
    m = re.search(
        r"No datasets registered yet — import one in the Datasets\s+"
        r"workspace to plot its columns\.",
        src,
    )
    assert m, "the honest empty-state line was lost"


def test_stage_one_hint_names_the_bridge() -> None:
    """Stage 1's name becomes true — a registered dataset's own column."""
    src = _page()
    assert "a registered dataset's own column (the stage-3 bridge" in src, (
        "the stage-1 hint lost the VIZ-1 bridge lane"
    )


def test_vitest_file_pins_the_roadmap_acceptance_clauses() -> None:
    """The lane's own vitest carries the roadmap's clauses + guards."""
    assert VITEST.exists(), (
        "the VIZ-1 vitest file is gone — the acceptance clauses lost "
        "their fail-first pins"
    )
    src = VITEST.read_text(encoding="utf-8")
    for clause in (
        "produces the identical stat request the paste lane does",
        "surfaces the backend's 422 on a non-numeric column verbatim",
        "keeps the paste lane unchanged",
        "never Number(null) === 0",
        "names the bound honestly",
        "honest empty state",
    ):
        assert clause in src, f"the vitest lost its acceptance clause: {clause}"
    # all six tests carry the VIZ-1 mount lane
    assert src.count("render(<VisualizationPage />)") >= 6, (
        "the vitest no longer mounts the full page on every clause"
    )


def test_roadmap_row_9_status_records_this_round() -> None:
    """The roadmap's VIZ-1 row is marked Implemented for this round."""
    src = ROADMAP.read_text(encoding="utf-8")
    m = re.search(
        r"\| ID / Feature \| VIZ-1 — Plot any registered dataset's "
        r"numeric column without pasting \|\n\| Current status \| "
        r"([^|]+)\|",
        src,
    )
    assert m, "the roadmap's VIZ-1 row header drifted"
    status = m.group(1)
    assert "Implemented" in status and "382" in status, (
        f"the VIZ-1 row status must record round 382 honestly, got: {status!r}"
    )
