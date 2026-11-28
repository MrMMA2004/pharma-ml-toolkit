"""The ML-5 data-drift lane drift guard (the roadmap's row 12).

The lane spans three sources that must never drift apart:

* the route -- ``routes/ml.py``'s ``POST /api/v1/ml/drift`` over the
  REAL engine (``pharmaml.ml.drift.detect_drift`` — the same engine
  the batch-training CLI treats as mandatory), sync by design;
* the client -- ``services/ml/client.ts``'s ``drift`` operation with
  its strict parser (``parseDriftReport``);
* the page -- ``features/ml/MLPage.tsx`` stage 10f's drift card (the
  PSI table with the drifted rows highlighted), rendered beside the
  kNN/box sub-lanes and OUTSIDE the trained-model gate (the dataset
  reference kind needs no model);
* the lane's own vitest pair -- ``DriftLane.test.tsx`` (the
  drifted-highlight + the honest-422 + the pre-flight + the
  empty-overlap clauses) and the client pair ``drift.test.ts``.

Plus the roadmap's own honesty clauses:

* the engine's own exported threshold constants are pinned here —
  0.1 warn / 0.2 shift — and the route relays them verbatim (the
  GUI never re-encodes a threshold);
* the consumption pin (the cross-check this roadmap opened with)
  rides ``tests/test_ml_workspace.py`` — the path stays consumed by
  the ml client.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "frontend" / "src" / "features" / "ml" / "MLPage.tsx"
CLIENT = REPO / "frontend" / "src" / "services" / "ml" / "client.ts"
VITEST_PAGE = (
    REPO / "frontend" / "src" / "features" / "ml" / "__tests__"
    / "DriftLane.test.tsx"
)
VITEST_CLIENT = (
    REPO / "frontend" / "src" / "services" / "ml" / "__tests__"
    / "drift.test.ts"
)
ROUTE = REPO / "src" / "pharmaml" / "api" / "routes" / "ml.py"
ROADMAP = REPO / "docs" / "QA" / "WORKFLOW_ENHANCEMENT_ROADMAP.md"


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _client() -> str:
    return CLIENT.read_text(encoding="utf-8")


def _route() -> str:
    return ROUTE.read_text(encoding="utf-8")


def test_the_engine_threshold_constants_are_pinned() -> None:
    """The roadmap's own acceptance: the constants the engine exports."""
    from pharmaml.ml.drift import DEFAULT_SHIFT_PSI, DEFAULT_WARN_PSI

    assert DEFAULT_WARN_PSI == 0.1, (
        "the engine's warn threshold moved — the GUI table and the "
        "verdict bands ride this constant"
    )
    assert DEFAULT_SHIFT_PSI == 0.2, (
        "the engine's shift threshold moved — the drifted list's own "
        "gate rides this constant"
    )


def test_route_serves_the_engine_and_relays_the_constants() -> None:
    """The route over detect_drift, the thresholds relayed verbatim."""
    src = _route()
    assert '@router.post("/drift")' in src, (
        "the drift route was removed from routes/ml.py"
    )
    assert "from pharmaml.ml.drift import detect_drift" in src, (
        "the route stopped importing the REAL engine"
    )
    assert "detect_drift(reference, current.X, features=features)" in src, (
        "the route stopped calling the engine over the stored frames"
    )
    # the engine's constants relay through the report, never
    # re-encoded as literals in the route
    assert 'float(report.warn_psi)' in src and 'float(report.shift_psi)' in src
    assert not re.search(r'"warn_psi":\s*0\.1', src), (
        "a literal threshold re-encoded in the route — the engine's "
        "own constants must ride the payload"
    )


def test_client_carries_the_drift_operation_and_parser() -> None:
    """The ML-5 operation + its strict parser stay in the client."""
    src = _client()
    assert "async drift(options: DriftOptions): Promise<DriftResult>" in src, (
        "the drift operation was removed"
    )
    assert "export function parseDriftReport(" in src, (
        "the drift answer parser was removed"
    )
    assert '"/api/v1/ml/drift"' in src, (
        "the drift route literal was lost"
    )
    # the exactly-one-reference pre-flight mirrors the route's own
    # contract client-side
    assert (
        "give either modelId or datasetId as the drift reference" in src
    )


def test_page_carries_the_drift_lane_card() -> None:
    """Stage 10f's drift card: the PSI table + the drifted highlight."""
    src = _page()
    assert "Data-drift check (PSI + KS)" in src, (
        "the drift card's title was removed"
    )
    assert "score drift before you trust the metrics" in src, (
        "the roadmap's own framing line was removed"
    )
    # the drifted rows highlight with the verdict attribute the lane's
    # own tests pin
    assert 'data-verdict={row.verdict}' in src
    assert 'row.verdict === "shifted"' in src
    # the reference kind select carries both lanes (model/dataset)
    assert 'setDriftRefKind' in src
    # the honest empty answer, never a fabricated verdict
    assert "never a verdict" in src


def test_the_vitest_pair_exists_and_pins_the_acceptance() -> None:
    """The lane's own vitest pair: the page clause + the client clause."""
    page_src = VITEST_PAGE.read_text(encoding="utf-8")
    # the roadmap's page-test acceptance, verbatim: the PSI table with
    # the drifted rows highlighted
    assert "renders the PSI table with the drifted rows highlighted" in (
        page_src
    )
    assert 'toHaveAttribute("data-verdict", "shifted")' in page_src
    # the honest 422 surfaces verbatim
    assert "surfaces the backend's own 422 verbatim" in page_src
    # the empty overlap is the honest zero answer
    assert "never a verdict" in page_src

    client_src = VITEST_CLIENT.read_text(encoding="utf-8")
    assert "refuses both/neither references pre-flight" in client_src
    assert "lost its psi/verdict pair" in client_src
    # the threshold-constant loss: the vitest regex carries the
    # escaped slash form
    assert "warn_psi\\/shift_psi are not numbers" in client_src


def test_the_api_test_file_carries_the_fail_first_acceptance() -> None:
    """The roadmap's fail-first API clause, verbatim."""
    test_src = (
        REPO / "tests" / "api" / "test_ml_drift.py"
    ).read_text(encoding="utf-8")
    assert "test_shifted_column_flags_and_identical_does_not" in test_src
    assert 'verdict"] == "shifted"' in test_src
    assert 'verdict"] == "stable"' in test_src


def test_roadmap_row_twelve_is_marked_implemented() -> None:
    """The roadmap's own row tracks the lane as Implemented."""
    src = ROADMAP.read_text(encoding="utf-8")
    assert re.search(
        r"ML-5[^\n]*\|\s*\*\*Implemented \(round 383", src
    ) or "Implemented (round 383" in src, (
        "the ML-5 row's status was not trued to the r383 implementation"
    )
