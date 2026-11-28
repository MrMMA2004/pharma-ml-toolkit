"""The DL-2 GNN explainability lane drift guard (the roadmap's row 14).

The lane spans three sources that must never drift apart:

* the route -- ``routes/dl.py``'s ``POST /api/v1/dl/gnn-explain`` over
  the REAL ``pharmaml.ml.gnnxai`` engine (the four explainers, the
  fidelity_plus/minus/sparsity verdicts, the deterministic circular
  layout);
* the client -- ``services/dl/client.ts``'s ``gnnExplain`` operation
  with its strict parser (``parseGnnExplainResult``: the scores must
  align with the served edge_index, the layout must carry one node
  position per node);
* the page -- ``features/dl/DLPage.tsx``'s edge-attribution card (the
  method/top_k/seed/target/labels controls + the per-graph edge table
  with the top rows highlighted + the PlotlyFigure sized-edge scatter
  over the backend's own circular layout);
* the lane's own vitest pair -- ``GnnExplainLane.test.tsx`` (the
  armed lane + the edge table + the pre-flight + the honest 422) and
  the client pair ``gnn-explain.test.ts``.

Plus the roadmap's own honesty clauses:

* the explained subgraph never exceeds the full graph (top_edges is
  a subset of the edge columns -- the route constructs it that way);
* a non-graph model and the attention lane's GAT requirement are the
  engine's own honest refusals, surfaced verbatim, never swallowed.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "frontend" / "src" / "features" / "dl" / "DLPage.tsx"
CLIENT = REPO / "frontend" / "src" / "services" / "dl" / "client.ts"
VITEST_PAGE = (
    REPO / "frontend" / "src" / "features" / "dl" / "__tests__"
    / "GnnExplainLane.test.tsx"
)
VITEST_CLIENT = (
    REPO / "frontend" / "src" / "services" / "dl" / "__tests__"
    / "gnn-explain.test.ts"
)
ROUTE = REPO / "src" / "pharmaml" / "api" / "routes" / "dl.py"
ROADMAP = REPO / "docs" / "QA" / "WORKFLOW_ENHANCEMENT_ROADMAP.md"


def _route() -> str:
    return ROUTE.read_text(encoding="utf-8")


def _client() -> str:
    return CLIENT.read_text(encoding="utf-8")


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_the_route_serves_the_real_engine() -> None:
    """The gnn-explain route over the real gnnxai engine stays."""
    src = _route()
    assert '@router.post("/gnn-explain")' in src, (
        "the gnn-explain route was removed from routes/dl.py"
    )
    assert "from pharmaml.ml.gnnxai import" in src, (
        "the route stopped importing the REAL gnnxai engine"
    )
    for symbol in ("GNNExplainer", "PGExplainer", "AttentionExplainer",
                   "SubgraphX", "fidelity_plus", "fidelity_minus", "sparsity"):
        assert symbol in src, (
            f"the route lost its engine symbol {symbol!r}"
        )
    # the deterministic circular layout (the sized-edge scatter's canvas)
    assert "_circular_layout" in src
    # the honest model-class refusal
    assert "is not a graph model" in src
    # the target stamp: ONE class for every explainer + both fidelity passes
    assert "data.y = np.asarray(req.target, dtype=int)" in src


def test_the_client_carries_the_operation_and_parser() -> None:
    """The gnnExplain operation + the strict parser stay."""
    src = _client()
    assert "async gnnExplain(options: GnnExplainOptions)" in src, (
        "the gnnExplain operation was removed"
    )
    assert "export function parseGnnExplainResult(" in src, (
        "the gnn-explain answer parser was removed"
    )
    assert '"/api/v1/dl/gnn-explain"' in src, (
        "the gnn-explain route literal was lost"
    )
    # the alignment guard: the scores must match the served edge_index
    assert "disagree" in src and "fabricated" in src


def test_the_page_carries_the_edge_attribution_card() -> None:
    """The card: the controls + the edge table + the sized-edge scatter."""
    src = _page()
    assert "Edge attribution (GNN explainability)" in src, (
        "the edge-attribution card's title was removed"
    )
    assert 'aria-label="the explain method"' in src
    assert 'aria-label="the graphs to explain"' in src
    # the edge table's top rows highlight with the verdict attribute
    assert 'data-top={isTop ? "true" : undefined}' in src
    # the sized-edge scatter rides the backend's own layout positions
    assert "row.layout.nodes" in src
    assert "sizemode" in src
    # the pre-flight mirrors (labels ride the pgexplainer lane only)
    assert "labels ride the pgexplainer lane only" in src


def test_the_vitest_pair_pins_the_acceptance() -> None:
    """The lane's own vitest pair: the page clause + the client clause."""
    page_src = VITEST_PAGE.read_text(encoding="utf-8")
    assert "renders the edge table + the sized-edge scatter" in page_src
    assert 'toHaveAttribute("data-top", "true")' in page_src
    assert "refuses labels with a non-pgexplainer method pre-flight" in page_src
    assert "surfaces the backend's own honest 422 verbatim" in page_src

    client_src = VITEST_CLIENT.read_text(encoding="utf-8")
    assert "POSTs the graphs + the method controls verbatim" in client_src
    assert "drops the null target/labels/iters from the request" in client_src
    assert "fabricated geometry" in client_src
    assert "lost its node positions" in client_src


def test_the_api_test_file_carries_the_acceptance() -> None:
    """The roadmap's fail-first API clauses, verbatim."""
    test_src = (
        REPO / "tests" / "api" / "test_dl_gnn_explain.py"
    ).read_text(encoding="utf-8")
    assert "test_the_mask_explainer_computes_fidelities_on_the_two_graph_fixture" in (
        test_src
    )
    assert "test_the_pgexplainer_lane_trains_over_the_requests_own_labels" in (
        test_src
    )
    assert "test_the_honest_404_and_422_lanes" in test_src


def test_roadmap_row_fourteen_is_marked_implemented() -> None:
    """The roadmap's own row tracks the lane as Implemented."""
    src = ROADMAP.read_text(encoding="utf-8")
    assert "**Implemented (round 383" in src, (
        "the DL-2 row's status was not trued to the r383 implementation"
    )
    # the row's own status line names the gnnxai engine
    match = re.search(
        r"\| Current status \| \*\*Implemented \(round 383[^\n]*gnnxai",
        src,
    )
    assert match is not None, (
        "the DL-2 row's status line lost its gnnxai engine reference"
    )
