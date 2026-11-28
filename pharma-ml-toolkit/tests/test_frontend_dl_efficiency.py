"""The DL-3 efficiency-lab drift guard (the roadmap's row 15).

The lane spans three sources that must never drift apart:

* the route -- ``routes/dl.py``'s ``POST /api/v1/dl/efficiency`` over
  the REAL weight engines (``pharmaml.ml.quantization``'s own
  ``quantization_report`` + ``pharmaml.ml.pruning``'s own
  ``magnitude_mask`` / ``count_sparsity``), the weight matrices read
  off the estimator's own ``_weights`` dict (the pickle-portable
  source of truth), the report registered as a table artifact;
* the client -- ``services/dl/client.ts``'s ``efficiency`` operation
  with its strict parser (``parseDLEfficiencyResult``: the
  mode-specific totals and per-layer fields, an absent value never an
  implied zero);
* the page -- ``features/dl/DLPage.tsx``'s stage-11c efficiency card
  (the mode/level/scope controls + the per-layer table + the
  artifact's digest-verified download);

plus the lane's own vitest pair -- ``EfficiencyLane.test.tsx`` (the
armed lane + the report card + the pre-flight + the honest 422) and
the client pair ``dl-efficiency.test.ts``.

Plus the roadmap's own honesty clauses:

* the persisted estimator is NEVER mutated (the prune lane scores a
  pruned COPY -- the API test pins the re-run identity);
* the level knob rides the prune lane only (a 422 on the quantize
  lanes, never a silent skip).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "frontend" / "src" / "features" / "dl" / "DLPage.tsx"
CLIENT = REPO / "frontend" / "src" / "services" / "dl" / "client.ts"
VITEST_PAGE = (
    REPO / "frontend" / "src" / "features" / "dl" / "__tests__"
    / "EfficiencyLane.test.tsx"
)
VITEST_CLIENT = (
    REPO / "frontend" / "src" / "services" / "dl" / "__tests__"
    / "dl-efficiency.test.ts"
)
ROUTE = REPO / "src" / "pharmaml" / "api" / "routes" / "dl.py"
ROADMAP = REPO / "docs" / "QA" / "WORKFLOW_ENHANCEMENT_ROADMAP.md"


def _route() -> str:
    return ROUTE.read_text(encoding="utf-8")


def _client() -> str:
    return CLIENT.read_text(encoding="utf-8")


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_the_route_serves_the_real_engines() -> None:
    """The efficiency route over the real quantization/pruning engines."""
    src = _route()
    assert '@router.post("/efficiency")' in src, (
        "the efficiency route was removed from routes/dl.py"
    )
    assert "from pharmaml.ml.quantization import quantization_report" in src, (
        "the route stopped importing the REAL quantization engine"
    )
    assert (
        "from pharmaml.ml.pruning import count_sparsity, magnitude_mask"
        in src
    ), "the route stopped importing the REAL pruning engine"
    # the weight matrices ride the estimator's own NumPy weight dict
    # (the pickle-portable source of truth -- never a second authority)
    assert 'getattr(model, "_weights", None)' in src
    # the prune lane scores a pruned COPY (never the persisted estimator)
    assert "copy.deepcopy(model)" in src
    # the honest refusals: the level/scope knobs ride the prune lane only
    assert "the level knob rides the prune lane only" in src
    assert "the scope knob rides the prune lane only" in src
    # the report registers as a table artifact (the roadmap's own close)
    assert 'kind="table"' in src
    assert "dl-efficiency-{req.mode}.csv" in src


def test_the_client_carries_the_operation_and_parser() -> None:
    """The efficiency operation + the strict parser stay."""
    src = _client()
    assert "async efficiency(options: DLEfficiencyOptions)" in src, (
        "the efficiency operation was removed"
    )
    assert "export function parseDLEfficiencyResult(" in src, (
        "the efficiency answer parser was removed"
    )
    assert '"/api/v1/dl/efficiency"' in src, (
        "the efficiency route literal was lost"
    )
    # the mode-driven totals: null on the lanes they do not ride
    assert "quantTotals" in src and "pruneTotals" in src
    # the strict refusals (an absent value is never an implied zero)
    assert "a prune row lost its counts" in src
    assert "a quantize row lost the engine's own report fields" in src


def test_the_page_carries_the_efficiency_card() -> None:
    """The card: the controls + the per-layer table + the artifact."""
    src = _page()
    assert "Efficiency lab (DL-3" in src, (
        "the efficiency card's title was removed"
    )
    assert 'aria-label="the efficiency mode"' in src
    assert 'aria-label="the pruning scope"' in src
    # the prune-only controls render conditionally
    assert 'efficiencyMode === "prune"' in src
    # the per-layer table renders the mode's own fields
    assert "row.kind === \"prune\"" in src
    assert "row.bitsPerWeight.toFixed(2)" in src
    # the honest pre-flight mirrors the route's own guard
    assert "the prune lane's level must land in [0, 1)" in src
    # the artifact downloads through the digest-verified lane
    assert "handleEfficiencyDownload" in src


def test_the_vitest_pair_pins_the_acceptance() -> None:
    """The lane's own vitest pair: the page clause + the client clause."""
    page_src = VITEST_PAGE.read_text(encoding="utf-8")
    assert "renders the engine's own report" in page_src
    assert "no request fired" in page_src
    assert "surfaces the backend's own 422 verbatim" in page_src
    assert "stays unarmed before a model exists" in page_src

    client_src = VITEST_CLIENT.read_text(encoding="utf-8")
    assert "mirrors the int8 answer field-for-field" in client_src
    assert "the counts + the metric block + the null artifact" in client_src
    assert "refuses a prune row that lost its counts" in client_src


def test_the_api_test_file_carries_the_acceptance() -> None:
    """The roadmap's fail-first API clauses, verbatim."""
    test_src = (
        REPO / "tests" / "api" / "test_dl_efficiency.py"
    ).read_text(encoding="utf-8")
    assert (
        "test_int8_report_within_the_engines_own_documented_bound" in test_src
    )
    assert "test_prune_at_half_sparsity_returns_exactly_half_zeros" in test_src
    assert (
        "test_the_metric_delta_scores_a_pruned_copy_never_the_model"
        in test_src
    )
    assert "test_the_report_registers_as_a_downloadable_table_artifact" in (
        test_src
    )


def test_roadmap_row_fifteen_is_marked_implemented() -> None:
    """The roadmap's own row tracks the lane as Implemented."""
    src = ROADMAP.read_text(encoding="utf-8")
    match = re.search(
        r"\| Current status \| \*\*Implemented \(round 384[^\n]*"
        r"dl/efficiency[^\n]*pruned COPY",
        src,
    )
    assert match is not None, (
        "the DL-3 row's status was not trued to the r384 implementation"
    )
