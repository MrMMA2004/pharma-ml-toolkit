"""The DL-4 D-MPNN molecular-lane drift guard (the roadmap's row 16).

The lane spans three sources that must never drift apart:

* the route -- ``routes/dl.py``'s ``POST /api/v1/dl/mpnn-train`` +
  ``POST /api/v1/dl/mpnn-predict`` over the REAL numpy engine
  (``pharmaml.ml.mpnn.DMPNN`` -- chemprop's directed message
  passing): the SMILES column re-read server-side off the dataset's
  own source frame, the honest rejected-rows accounting, the seeded
  held-out carve the roadmap's own validation step names, the
  shipped model as the FULL-corpus refit, and the molecular models'
  own prediction path over raw SMILES;
* the client -- ``services/dl/client.ts``'s ``mpnnTrain`` /
  ``mpnnPredict`` operations with their strict parsers
  (``parseMpnnTrainResult`` / ``parseMpnnPredictResult``: the
  carve's verdicts never implied, the counts never zeroed silently);
* the page -- ``features/dl/DLPage.tsx``'s D-MPNN lane section (the
  lane's own dataset pick + the SMILES column select + the held-out
  verdict card + the raw-SMILES predict table with the rejected rows
  rendered inline);

plus the lane's own vitest pair -- ``MpnnLane.test.tsx`` and the
client pair ``mpnn-client.test.ts``.

Plus the roadmap's own honesty clauses:

* the carve minimum is a CLIENT error (the pre-flight before the job
  submits -- the DL-1 rule), never an asynchronous job failure;
* a degenerate correlation stays null, never a serialized NaN.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "frontend" / "src" / "features" / "dl" / "DLPage.tsx"
CLIENT = REPO / "frontend" / "src" / "services" / "dl" / "client.ts"
VITEST_PAGE = (
    REPO / "frontend" / "src" / "features" / "dl" / "__tests__"
    / "MpnnLane.test.tsx"
)
VITEST_CLIENT = (
    REPO / "frontend" / "src" / "services" / "dl" / "__tests__"
    / "mpnn-client.test.ts"
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
    """The mpnn-train route over the REAL numpy DMPNN engine stays."""
    src = _route()
    assert '@router.post("/mpnn-train")' in src, (
        "the mpnn-train route was removed from routes/dl.py"
    )
    assert '@router.post("/mpnn-predict")' in src, (
        "the mpnn-predict route was removed from routes/dl.py"
    )
    assert "from pharmaml.ml.mpnn import DMPNN" in src, (
        "the route stopped importing the REAL D-MPNN engine"
    )
    # the seeded held-out carve (the roadmap's own validation step)
    assert "val_rmse" in src
    # the shipped model is the FULL-corpus refit (the tune lane's rule)
    assert "refitting on the full corpus" in src
    # the honest rejected-rows accounting
    assert "_parse_smiles_rows" in src
    assert "a null cell" in src
    # the carve minimum pre-flighted in the ROUTE (the DL-1 rule: a
    # doomed request is a client error, never an async job failure)
    assert "never an asynchronous job failure" in src
    # the model registers as a MOLECULAR model with its own path
    assert 'name="dmpnn_molecular_regressor"' in src
    assert 'view["molecular_model"] = True' in src


def test_the_client_carries_the_operations_and_parsers() -> None:
    """The mpnnTrain/mpnnPredict operations + the strict parsers stay."""
    src = _client()
    assert "async mpnnTrain(options: MpnnTrainOptions)" in src, (
        "the mpnnTrain operation was removed"
    )
    assert "async mpnnPredict(" in src, (
        "the mpnnPredict operation was removed"
    )
    assert "export function parseMpnnTrainResult(" in src, (
        "the mpnn-train answer parser was removed"
    )
    assert "export function parseMpnnPredictResult(" in src, (
        "the mpnn-predict answer parser was removed"
    )
    assert '"/api/v1/dl/mpnn-train"' in src
    assert '"/api/v1/dl/mpnn-predict"' in src
    # the degenerate correlation stays null, never an implied zero
    assert "val_correlation" in src
    # the count disagreement refusal (the split template literal)
    assert "disagree with the" in src
    assert "served n_valid" in src


def test_the_page_carries_the_molecular_lane() -> None:
    """The lane: the dataset pick + the verdict card + the predict table."""
    src = _page()
    assert 'id="mpnn-lane-title"' in src, (
        "the molecular lane section was removed"
    )
    assert 'aria-label="the molecular lane\'s dataset"' in src
    assert 'aria-label="the SMILES column"' in src
    # the held-out verdict card
    assert "MpnnModelCard" in src
    assert "held-out RMSE" in src
    # the molecular models' OWN prediction path over raw SMILES
    assert 'aria-label="the SMILES to predict"' in src
    assert "rejected:" in src
    # the honest task gate (the route's own refusal mirrored)
    assert "the lane needs a regression target" in src


def test_the_vitest_pair_pins_the_acceptance() -> None:
    """The lane's own vitest pair: the page clause + the client clause."""
    page_src = VITEST_PAGE.read_text(encoding="utf-8")
    assert "renders the held-out verdicts" in page_src
    assert "the rejected rows rendered inline" in page_src
    assert "refuses a classification dataset before any request fires" in (
        page_src
    )
    assert "surfaces the backend's own 422 verbatim" in page_src

    client_src = VITEST_CLIENT.read_text(encoding="utf-8")
    assert "the carve's verdicts" in client_src
    assert "never an implied zero" in client_src
    assert "disagree with the served n_valid" in client_src
    assert "refuses a rejected row that lost its error" in client_src


def test_the_api_test_file_carries_the_acceptance() -> None:
    """The roadmap's fail-first API clauses, verbatim."""
    test_src = (
        REPO / "tests" / "api" / "test_dl_mpnn.py"
    ).read_text(encoding="utf-8")
    assert "test_the_held_out_rmse_lands_within_the_pre_registered_tolerance" in (
        test_src
    )
    assert "test_the_rejected_rows_ride_their_own_accounting" in test_src
    assert "test_the_predict_lane_over_raw_smiles" in test_src
    assert "test_the_job_mode_completes_with_the_model" in test_src


def test_roadmap_row_sixteen_is_marked_implemented() -> None:
    """The roadmap's own row tracks the lane as Implemented."""
    src = ROADMAP.read_text(encoding="utf-8")
    match = re.search(
        r"\| Current status \| \*\*Implemented \(round 384[^\n]*"
        r"mpnn-train[^\n]*held-out",
        src,
    )
    assert match is not None, (
        "the DL-4 row's status was not trued to the r384 implementation"
    )
