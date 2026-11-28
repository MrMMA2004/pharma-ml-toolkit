"""The PK/PD dataset-bridge drift guard (PK-1, the roadmap's row 8).

The lane spans three sources that must never drift apart:

* the established bridge client -- ``services/datasets/client.ts``'s
  ``column()`` (the §11.1 bounded lane every other workspace rides,
  route-pinned by ``test_frontend_services.py``);
* the page's bridge block -- ``features/pkpd/PKPDPage.tsx`` stage 1
  (the dataset select, the time/conc column pair, the import button,
  the honest notes);
* the lane's own vitest file -- ``DatasetBridgeLane.test.tsx`` (the
  roadmap's three acceptance clauses + the null/bound/alignment
  guards).

Plus the roadmap's own honesty clauses:

* a ``null`` cell (``json_safe``'s NaN → null) is a row-labeled
  error, never ``Number(null) === 0`` -- the zero would fabricate a
  concentration the frame never measured;
* a truncated pull names its 500-row bound (§17.3: the bounded view
  never masquerades as the whole column);
* the paste lane stays the free-form lane, unchanged (the
  regression clause);
* the roadmap's row 8 status line records this round honestly.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "frontend" / "src" / "features" / "pkpd" / "PKPDPage.tsx"
CLIENT = REPO / "frontend" / "src" / "services" / "datasets" / "client.ts"
VITEST = (
    REPO
    / "frontend"
    / "src"
    / "features"
    / "pkpd"
    / "__tests__"
    / "DatasetBridgeLane.test.tsx"
)
ROADMAP = (
    REPO
    / "docs"
    / "QA"
    / "WORKFLOW_ENHANCEMENT_ROADMAP.md"
)


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_page_carries_the_bridge_card_and_the_column_pair_selects() -> None:
    """Stage 1 offers the dataset select + the two column selects."""
    src = _page()
    assert "Registered dataset bridge" in src, (
        "the PK/PD Import stage lost its bridge card title"
    )
    for label in (
        '"the registered dataset"',
        '"the times column"',
        '"the concentration column"',
    ):
        assert f'aria-label={label}' in src, (
            f"the bridge lost its aria-labelled select {label}"
        )
    assert "Import from dataset" in src, (
        "the bridge lost its import button"
    )
    # the button stays disarmed until the pair is complete — the
    # gating clause every honest action button carries (the page's
    # own multi-line JSX form tolerated)
    m = re.search(r"disabled=\{[^}]*bridgeBusy", src, re.DOTALL)
    assert m, "the bridge import button lost its busy/selection gating"


def test_page_pulls_both_columns_through_the_client_bridge() -> None:
    """Both columns ride the SAME ``datasetsApi.column`` lane."""
    src = _page()
    assert 'from "@/services/datasets/client"' in src, (
        "the page no longer imports the datasets service client"
    )
    assert len(re.findall(r"datasetsApi\s*\.\s*column\(", src)) == 2, (
        "the bridge must pull exactly two columns (time + conc) "
        "through the client lane"
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
    # the conversion lane never defaults a null to zero (the scan is
    # scoped to the converter body — the page's own display lanes
    # elsewhere predate this row and are not this guard's clause)
    assert "?? 0" not in body, (
        "a silent zero default appeared in the converter — "
        "Number(null) === 0 would fabricate a concentration the "
        "frame never measured"
    )


def test_truncation_is_named_honestly() -> None:
    """A truncated pull names its bound (§17.3)."""
    src = _page()
    assert ".truncated" in src, "the bridge no longer reads the truncation flag"
    assert "bounded at 500 rows" in src, (
        "the honest bound note (the chemistry engine's own cap) was lost"
    )


def test_paste_lane_is_unchanged() -> None:
    """The free-form paste lane keeps its own parser, button, textarea."""
    src = _page()
    assert "function parseSeries(" in src, (
        "the paste lane's own parser was removed"
    )
    assert "Import series" in src, (
        "the paste lane's import button was removed"
    )
    assert 'placeholder={"time conc\\n0 8.0' in src or (
        'placeholder={"time conc' in src
    ), "the paste textarea's own placeholder was lost"
    assert "Load single-dose sample" in src, (
        "the paste lane's sample buttons were removed"
    )


def test_registry_lane_renders_the_honest_empty_state() -> None:
    """An empty registry is stated, not hidden."""
    src = _page()
    assert (
        "No datasets registered yet — import one in the Datasets workspace."
        in src
    ), "the honest empty-state line was lost"


def test_vitest_file_pins_the_roadmap_acceptance_clauses() -> None:
    """The lane's own vitest carries the roadmap's three clauses + guards."""
    assert VITEST.exists(), (
        "the PK-1 vitest file is gone — the acceptance clauses lost "
        "their fail-first pins"
    )
    src = VITEST.read_text(encoding="utf-8")
    for clause in (
        "row-count and t/C ranges asserted",
        "surfaces the backend's 422 on a non-numeric column verbatim",
        "keeps the paste lane unchanged",
        "never Number(null) === 0",
        "names the bound honestly",
        "row-misaligned columns",
    ):
        assert clause in src, f"the vitest lost its acceptance clause: {clause}"
    # all seven tests carry the PK-1 mount lane
    assert src.count("render(<PKPDPage />)") >= 7, (
        "the vitest no longer mounts the full page on every clause"
    )


def test_roadmap_row_8_status_records_this_round() -> None:
    """The roadmap's PK-1 row is marked Implemented for this round."""
    src = ROADMAP.read_text(encoding="utf-8")
    m = re.search(
        r"\| ID / Feature \| PK-1 — Import stage reads a registered "
        r"dataset's time/conc columns \|\n\| Current status \| "
        r"([^|]+)\|",
        src,
    )
    assert m, "the roadmap's PK-1 row header drifted"
    status = m.group(1)
    assert "Implemented" in status and "381" in status, (
        f"the PK-1 row status must record round 381 honestly, got: {status!r}"
    )


def test_stage_one_hint_and_doc_comment_carry_the_bridge() -> None:
    """The stage header and the page doc name both import lanes."""
    src = _page()
    assert (
        "or a registered dataset's own time/conc columns" in src
    ), "the stage-1 hint lost the bridge lane"
    assert "the bounded `column()` bridge every other workspace rides" in src, (
        "the page doc comment lost the PK-1 bridge description"
    )
