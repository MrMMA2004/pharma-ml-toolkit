"""The ML-2 AD sub-lane drift guard (the roadmap's row 10).

The lane spans three sources that must never drift apart:

* the client -- ``services/ml/client.ts``'s two new operations
  (``knnDomain`` over ``POST /applicability-domain/knn`` and
  ``domainBoxCheck`` over ``POST /applicability-domain/domain-box``)
  with their strict parsers (``parseKnnDomainResult`` /
  ``parseDomainBoxReport``);
* the page -- ``features/ml/MLPage.tsx`` stage 10's two sub-lane
  cards (the distance-only kNN card + the domain-box check card),
  rendered beside the analyze/Williams lanes and OUTSIDE the
  trained-model gate (the matrix-only user the roadmap names);
* the lane's own vitest pair -- ``AdDistanceBoxLane.test.tsx`` (the
  matrix-only verdict + the honest-422 + the pre-flight + the box
  clauses) and the client pair ``ad-distance-box.test.ts``.

Plus the roadmap's own honesty clauses:

* the pre-flight mirrors the route's own bounds (the 10-row minimum,
  the feature-count match, the 1..50 k, the 50..<100 percentile) --
  a doomed request is refused client-side, honestly;
* every backend 422 surfaces verbatim through ``describeError``,
  never swallowed;
* the box lane's unchecked answer carries the route's own reason,
  never an implied verdict;
* the consumption pin (the cross-check this roadmap opened with)
  rides ``tests/test_ml_workspace.py`` -- the two paths stay
  consumed by the ml client.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "frontend" / "src" / "features" / "ml" / "MLPage.tsx"
CLIENT = REPO / "frontend" / "src" / "services" / "ml" / "client.ts"
VITEST_PAGE = (
    REPO / "frontend" / "src" / "features" / "ml" / "__tests__"
    / "AdDistanceBoxLane.test.tsx"
)
VITEST_CLIENT = (
    REPO / "frontend" / "src" / "services" / "ml" / "__tests__"
    / "ad-distance-box.test.ts"
)
ROADMAP = REPO / "docs" / "QA" / "WORKFLOW_ENHANCEMENT_ROADMAP.md"


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _client() -> str:
    return CLIENT.read_text(encoding="utf-8")


def test_client_carries_the_two_operations_and_parsers() -> None:
    """The two ML-2 operations + their strict parsers stay in the client."""
    src = _client()
    assert "async knnDomain(" in src, "the knnDomain operation was removed"
    assert "async domainBoxCheck(" in src, (
        "the domainBoxCheck operation was removed"
    )
    assert "export function parseKnnDomainResult(" in src, (
        "the kNN answer parser was removed"
    )
    assert "export function parseDomainBoxReport(" in src, (
        "the domain-box report parser was removed"
    )
    assert (
        '"/api/v1/applicability-domain/knn"' in src
    ), "the knn route literal was lost"
    assert (
        '"/api/v1/applicability-domain/domain-box"' in src
    ), "the domain-box route literal was lost"


def test_page_carries_both_sub_lane_cards() -> None:
    """Stage 10 renders the two ML-2 sub-lanes beside the analyze lane."""
    src = _page()
    assert "Distance-only AD (kNN)" in src, (
        "the distance-only sub-lane title was lost"
    )
    assert "Domain-box check" in src, (
        "the domain-box sub-lane title was lost"
    )
    for label in (
        '"the x_train matrix"',
        '"the x_test matrix"',
        '"the kNN metric"',
        '"the persisted bundle\'s metadata JSON"',
        '"the rows to check"',
        '"the box-bridge dataset"',
    ):
        assert f'aria-label={label}' in src, (
            f"the ML-2 sub-lane lost its aria-labelled control {label}"
        )
    assert "Run the distance-only AD" in src, (
        "the kNN sub-lane lost its run button"
    )
    assert "Check the domain box" in src, (
        "the box sub-lane lost its check button"
    )
    assert "Assemble rows from dataset" in src, (
        "the box bridge lost its assemble button"
    )


def test_the_sub_lanes_render_without_a_trained_model() -> None:
    """The matrix-only clause: the sub-lanes live outside the gate."""
    src = _page()
    # the no-model chip the lane carries (the unsupervised case)
    assert "no trained model required" in src, (
        "the distance-only lane lost its no-model chip"
    )
    # the gate line the analyze lane rides sits above the sub-lanes --
    # the sub-lane JSX follows the gate's closing, not inside it
    gate = src.find("Select a trained model in Evaluate or Compare")
    knn_card = src.find("Distance-only AD (kNN)")
    analyze_button = src.find("Run the AD analysis")
    assert gate != -1 and knn_card != -1 and analyze_button != -1
    assert gate < analyze_button < knn_card, (
        "the ML-2 sub-lanes drifted inside the trained-model gate"
    )


def test_the_preflight_mirrors_the_routes_own_bounds() -> None:
    """The honest pre-flight: the 10-row minimum, the feature match,
    the k bound, the percentile bound."""
    src = _page()
    assert "the route's own minimum is 10" in src, (
        "the 10-row pre-flight clause was lost"
    )
    assert "(they must match)" in src, (
        "the feature-count pre-flight clause was lost"
    )
    assert "1…50 bound" in src, (
        "the k bound pre-flight clause was lost"
    )
    assert "50…<100 bound" in src, (
        "the percentile bound pre-flight clause was lost"
    )
    assert "self-match" in src, (
        "the self-exclusion pre-flight clause was lost"
    )


def test_the_matrix_parser_is_row_labeled() -> None:
    """``parseMatrixLines`` labels every malformed cell by row/column."""
    src = _page()
    m = re.search(
        r"function parseMatrixLines\([^)]*\)[^{]*\{(.{0,1400}?)\n\}",
        src,
        re.DOTALL,
    )
    assert m, "the matrix parser drifted or was removed"
    body = m.group(1)
    assert "is not a finite number" in body, (
        "the labeled cell error was lost"
    )
    assert "at least 2 feature columns" in body, (
        "the single-column refusal was lost"
    )


def test_the_box_lane_names_the_unchecked_reason() -> None:
    """The unchecked answer renders the route's own reason."""
    src = _page()
    assert "not checked — " in src, (
        "the unchecked rendering lane was lost"
    )
    # the parser's own honest fallback rides the client, not the page
    client = _client()
    assert "the metadata carries no box" in client, (
        "the parser's honest fallback reason was lost"
    )
    # the box bridge refuses honestly when the metadata carries no box
    assert "nothing to bridge the columns against" in src, (
        "the box-bridge no-box refusal was lost"
    )


def test_page_never_hand_rolls_the_http_lane() -> None:
    """The page itself never issues raw ``fetch`` calls."""
    src = _page()
    assert "fetch(" not in src, (
        "the page hand-rolled an HTTP call — the typed client is the "
        "only lane (§11.1's control-plane rule)"
    )


def test_vitest_pair_pins_the_acceptance_clauses() -> None:
    """Both vitest files carry the roadmap's acceptance clauses."""
    assert VITEST_PAGE.exists() and VITEST_CLIENT.exists(), (
        "the ML-2 vitest pair is gone — the acceptance clauses lost "
        "their fail-first pins"
    )
    page_src = VITEST_PAGE.read_text(encoding="utf-8")
    for clause in (
        "without a trained model",
        "surfaces the backend's own 422 verbatim",
        "no request fired",
        "never an implied verdict",
        "rows textarea fills verbatim",
    ):
        assert clause in page_src, (
            f"the page vitest lost its acceptance clause: {clause}"
        )
    assert page_src.count("render(<MLPage />)") >= 6, (
        "the page vitest no longer mounts the full page on every clause"
    )
    client_src = VITEST_CLIENT.read_text(encoding="utf-8")
    for clause in (
        "POSTs the two raw matrices verbatim",
        "payload narrowing",
        "malformed-throw",
        "surfaces the backend's own 422 verbatim",
    ):
        assert clause in client_src, (
            f"the client vitest lost its acceptance clause: {clause}"
        )


def test_roadmap_row_10_status_records_this_round() -> None:
    """The roadmap's ML-2 row is marked Implemented for this round."""
    src = ROADMAP.read_text(encoding="utf-8")
    m = re.search(
        r"\| ID / Feature \| ML-2 — Expose `POST /applicability-domain/"
        r"knn` and `/domain-box` in the AD stage \|\n\| Current status "
        r"\| ([^|]+)\|",
        src,
    )
    assert m, "the roadmap's ML-2 row header drifted"
    status = m.group(1)
    assert "Implemented" in status and "382" in status, (
        f"the ML-2 row status must record round 382 honestly, got: {status!r}"
    )
