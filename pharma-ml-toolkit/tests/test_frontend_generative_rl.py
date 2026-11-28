"""The GEN-3 REINVENT-style RL-lane drift guard (the roadmap's row 17).

The lane spans three sources that must never drift apart:

* the route -- ``routes/generative.py``'s ``POST /api/v1/generative/
  rl-optimize`` over the REAL engine (``pharmaml.generative.rl.
  run_rl_optimization``): the policy-gradient fine-tuning with the
  prior-drift anchor + the experience-replay buffer, the objective
  spec over the SAME builder the GA lane rides, the session prior
  (prior_id) or the engine's own on-the-fly prior, the managed job
  with per-step telemetry and cancellation BETWEEN iterations (the
  engine's own ``on_iteration`` hook);
* the client -- ``services/generative/client.ts``'s ``rlOptimize``
  operation with its strict parser (``parseRlOptimizeView``: the
  history rows' absent diagnostics stay null, never implied zeros);
* the page -- ``features/generative/GenerativePage.tsx``'s THIRD
  method card + the RL execution block (the prior select over the
  session registry, the iterations/batch/seed controls, the verdict
  card with the leaderboard + the per-iteration step curve through
  PlotlyFigure);

plus the lane's own vitest pair -- ``RlLane.test.tsx`` and the client
pair ``rl-client.test.ts``.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "frontend" / "src" / "features" / "generative" / "GenerativePage.tsx"
CLIENT = REPO / "frontend" / "src" / "services" / "generative" / "client.ts"
VITEST_PAGE = (
    REPO / "frontend" / "src" / "features" / "generative" / "__tests__"
    / "RlLane.test.tsx"
)
VITEST_CLIENT = (
    REPO / "frontend" / "src" / "services" / "generative" / "__tests__"
    / "rl-client.test.ts"
)
ROUTE = REPO / "src" / "pharmaml" / "api" / "routes" / "generative.py"
ENGINE = REPO / "src" / "pharmaml" / "generative" / "rl.py"
ROADMAP = REPO / "docs" / "QA" / "WORKFLOW_ENHANCEMENT_ROADMAP.md"


def _route() -> str:
    return ROUTE.read_text(encoding="utf-8")


def _client() -> str:
    return CLIENT.read_text(encoding="utf-8")


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_the_route_serves_the_real_engine() -> None:
    """The rl-optimize route over the REAL run_rl_optimization engine."""
    src = _route()
    assert '@router.post("/rl-optimize")' in src, (
        "the rl-optimize route was removed from routes/generative.py"
    )
    assert "from pharmaml.generative.rl import run_rl_optimization" in src, (
        "the route stopped importing the REAL RL engine"
    )
    # the engine's own cancellation seam rides the route body
    assert "on_iteration" in src
    assert "Cancelled between RL iterations" in src
    # the per-step telemetry publishes through the prior-train lane's
    # own event (TRAINING_METRICS)
    assert "rl optimized" in src
    # the objective spec rides the SAME builder the GA lane serves
    assert "_build_objective(objective_spec)" in src
    # the honest refusals: prior_id + corpus both, the feature-prior
    # pointer, the too-small corpus minimum
    assert "not both" in src
    assert "MOLECULAR prior" in src
    assert "at least 5 SMILES" in src


def test_the_engine_keeps_the_cancellation_hook() -> None:
    """The engine's own on_iteration seam stays (GEN-3's own addition)."""
    src = ENGINE.read_text(encoding="utf-8")
    assert "on_iteration" in src, (
        "the engine lost its cancellation/telemetry hook"
    )
    assert "def optimize(" in src
    assert "run_rl_optimization" in src


def test_the_client_carries_the_operation_and_parser() -> None:
    """The rlOptimize operation + the strict parser stay."""
    src = _client()
    assert "async rlOptimize(" in src, (
        "the rlOptimize operation was removed"
    )
    assert "export function parseRlOptimizeView(" in src, (
        "the rl-optimize answer parser was removed"
    )
    assert '"/api/v1/generative/rl-optimize"' in src
    # the history rows' absent diagnostics stay null
    assert "RlIterationRow" in src


def test_the_page_carries_the_third_method_card() -> None:
    """The THIRD method card + the RL execution block stay."""
    src = _page()
    assert '"rl-optimize" as GenerationMethod' in src, (
        "the third method card was removed from the selector"
    )
    assert "RL optimization (REINVENT-style)" in src
    # the prior select over the session registry
    assert 'aria-label="the RL lane\'s prior"' in src
    # the verdict card + the step curve through PlotlyFigure
    assert "The RL run&apos;s verdict" in src
    assert "mean score" in src and "best score" in src
    assert "PlotlyFigure" in src


def test_the_vitest_pair_pins_the_acceptance() -> None:
    """The lane's own vitest pair: the page clause + the client clause."""
    page_src = VITEST_PAGE.read_text(encoding="utf-8")
    assert "renders the third method card" in page_src
    assert "renders the leaderboard + the step curve" in page_src
    assert "the third method card" in page_src

    client_src = VITEST_CLIENT.read_text(encoding="utf-8")
    assert "never implied zeros" in client_src
    assert "the prior view rides parsed" in client_src


def test_the_api_test_file_carries_the_acceptance() -> None:
    """The roadmap's fail-first API clauses, verbatim."""
    test_src = (
        REPO / "tests" / "api" / "test_generative_rl.py"
    ).read_text(encoding="utf-8")
    assert "test_the_score_trajectory_improves_over_the_seeded_steps" in (
        test_src
    )
    assert "test_cancellation_lands_between_iterations" in test_src
    assert "test_the_honest_refusal_lanes" in test_src
    assert "test_a_session_prior_rides_the_lane_with_its_own_view" in (
        test_src
    )


def test_roadmap_row_seventeen_is_marked_implemented() -> None:
    """The roadmap's own row tracks the lane as Implemented."""
    src = ROADMAP.read_text(encoding="utf-8")
    match = re.search(
        r"\| Current status \| \*\*Implemented \(round 384[^\n]*"
        r"rl-optimize[^\n]*on_iteration",
        src,
    )
    assert match is not None, (
        "the GEN-3 row's status was not trued to the r384 implementation"
    )
