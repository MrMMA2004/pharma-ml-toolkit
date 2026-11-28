"""The DL-1 hyperparameter search lane drift guard (the roadmap's row 13).

The lane spans three sources that must never drift apart:

* the route -- ``routes/dl.py``'s ``DLTrainRequest`` search fields
  (``tune``/``search``/``n_trials``/``sampler``/``seed``) and the
  ``_dl_tune_body`` search body through the REAL in-house engine
  (``pharmaml.ml.hpo.HPOStudy`` -- the same engine the classical ML
  lane rides), with cancellation BETWEEN trials;
* the client -- ``services/dl/client.ts``'s ``DLTrainOptions.tune``
  block, the ``DLTuneReport`` surface, and the strict parsers
  (``parseDLTuneReport`` / ``parseDLTuneResult``);
* the page -- ``features/dl/DLPage.tsx`` stage 7b's search panel (the
  search-block textarea + the n_trials/sampler/seed controls, the
  trial table with the best row highlighted);
* the lane's own vitest pair -- ``TuneLane.test.tsx`` (the armed
  search + the trial table + the pre-flight refusals + the honest
  422) and the client pair ``dl-tune.test.ts``.

Plus the roadmap's own honesty clauses:

* the search vocabulary stays the route's own constructor vocabulary
  (the ``DL_HYPERPARAM_KEYS`` mirror in the client -- the same
  one-source rule the fixed lane rides);
* the trial table's best row IS the study's argmin (the page computes
  it from the payload's own values, never a re-encoded best).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "frontend" / "src" / "features" / "dl" / "DLPage.tsx"
CLIENT = REPO / "frontend" / "src" / "services" / "dl" / "client.ts"
VITEST_PAGE = (
    REPO / "frontend" / "src" / "features" / "dl" / "__tests__"
    / "TuneLane.test.tsx"
)
VITEST_CLIENT = (
    REPO / "frontend" / "src" / "services" / "dl" / "__tests__"
    / "dl-tune.test.ts"
)
ROUTE = REPO / "src" / "pharmaml" / "api" / "routes" / "dl.py"
ROADMAP = REPO / "docs" / "QA" / "WORKFLOW_ENHANCEMENT_ROADMAP.md"


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _client() -> str:
    return CLIENT.read_text(encoding="utf-8")


def _route() -> str:
    return ROUTE.read_text(encoding="utf-8")


def test_the_route_serves_the_search_lane_over_the_real_engine() -> None:
    """tune/search/n_trials/sampler/seed ride DLTrainRequest."""
    src = _route()
    for field in ("tune: bool", "search: dict[str, dict[str, Any]] | None",
                  "n_trials: int", "sampler: Literal[\"tpe\", \"random\"]",
                  "seed: int"):
        assert field in src, (
            f"the DLTrainRequest lost its search field {field!r}"
        )
    # the REAL engine (never a bespoke sampler)
    assert "from pharmaml.ml.hpo import" in src, (
        "the route stopped importing the in-house HPO engine"
    )
    assert "study = HPOStudy(" in src, (
        "the search body stopped riding HPOStudy"
    )
    # cancellation between trials (the population-route precedent)
    assert "Cancelled between trials" in src, (
        "the between-trials cancellation clause was lost"
    )
    # the objective: fit on the train side, score on the val side,
    # negated (the optuna rule the engine documents)
    assert "estimator.score(X_val, y_val)" in src
    assert "return -float(estimator.score(X_val, y_val))" in src
    # the up-front validation (a doomed search is a client error)
    assert "_build_search_space(req.search, req.network)" in src


def test_the_client_carries_the_tune_block_and_parsers() -> None:
    """DLTrainOptions.tune + the strict report parsers stay."""
    src = _client()
    assert "readonly tune?:" in src, (
        "the DLTrainOptions tune block was removed"
    )
    assert "export type DLTrainOptionsTune" in src, (
        "the tune block's own alias was removed"
    )
    assert "export function parseDLTuneReport(" in src, (
        "the search report parser was removed"
    )
    assert "export function parseDLTuneResult(" in src, (
        "the full tune result parser was removed"
    )
    assert "export interface DLTuneReport" in src, (
        "the report surface was removed"
    )


def test_the_page_carries_the_search_panel() -> None:
    """Stage 7b's panel: the textarea + the controls + the trial table."""
    src = _page()
    assert "search the hyperparameters instead of one fixed configuration" in (
        src
    ), "the search lane's arming checkbox was removed"
    assert 'aria-label="the tune search block"' in src
    assert 'aria-label="the tune sampler"' in src
    assert 'aria-label="arm the hyperparameter search"' in src
    # the trial table rides the payload's own values (the argmin row
    # IS the best, computed from the payload -- never re-encoded)
    assert "data-best" in src
    assert "Math.min(" in src
    # the search pre-flight mirror (the route's own contract)
    assert "function parseTuneSearch(" in src
    assert "unknown search key" in src


def test_the_vitest_pair_pins_the_acceptance() -> None:
    """The lane's own vitest pair: the page clause + the client clause."""
    page_src = VITEST_PAGE.read_text(encoding="utf-8")
    assert "rides the SAME train route with tune: true" in page_src
    assert 'toHaveAttribute("data-best", "true")' in page_src
    assert "refuses a doomed search block pre-flight" in page_src
    assert "surfaces the backend's own 422 verbatim" in page_src

    client_src = VITEST_CLIENT.read_text(encoding="utf-8")
    assert "POSTs tune: true with the search block verbatim" in client_src
    assert "the tune absence honest" in client_src
    assert "lost its number/value pair" in client_src


def test_the_api_test_file_carries_the_acceptance() -> None:
    """The roadmap's fail-first API clauses, verbatim."""
    test_src = (
        REPO / "tests" / "api" / "test_dl_tune.py"
    ).read_text(encoding="utf-8")
    assert "test_the_search_returns_a_trial_table_with_the_best_among_the_candidates" in (
        test_src
    )
    assert "test_the_refit_metric_meets_the_fixed_config_baseline" in test_src
    assert "test_cancellation_between_trials_lands_cancelled" in test_src
    assert "Cancelled between trials" in test_src or (
        "population-route precedent" in test_src
    )


def test_roadmap_row_thirteen_is_marked_implemented() -> None:
    """The roadmap's own row tracks the lane as Implemented."""
    src = ROADMAP.read_text(encoding="utf-8")
    assert re.search(
        r"DL-1[^\n]*\n[^\n]*\| Current status \| \*\*Implemented \(round 383", src
    ) or (
        "**Implemented (round 383" in src
    ), "the DL-1 row's status was not trued to the r383 implementation"
