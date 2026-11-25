"""part17 §3.2 (Round 360, the G1 lane) — the global Job Center
drift guards.

The Python gate cannot click the React card, but every contract the
lane rides can be pinned against the real frontend + backend source —
a smuggled second ledger client, a dropped two-step confirmation, a
result button on a running row, or a swallowed 404/409 turns a guard
red before the surface drifts:

* the integration rule: the card's ONLY ledger lane is the typed
  ``jobsApi.list`` (the API-tested GET /jobs route — no duplicate
  client, no inline fetch, no parallel ledger anywhere in the tree);
* the authoritative per-row actions ride the EXISTING typed
  ``jobsApi.result`` / ``jobsApi.cancel`` — exactly one call site each,
  and never from the arm/disarm handlers;
* the vocabulary discipline: only a COMPLETED row offers the result
  button, only the live states (QUEUED / RUNNING /
  CANCELLATION_REQUESTED) offer cancellation — a terminal row never
  does either;
* the two-step cancellation gate: the row action only ARMS, the
  confirm button fires, and the honest 404 already-gone lane
  reconciles the ledger instead of hiding;
* the armed gate can never outlive its row (the refresh
  reconciliation disarms a job that left the ledger);
* the honest split stated in the card: this is the LIVE in-session
  ledger; the persisted history that survives restarts is the studio
  database card's own table — the two views never pretend to be one;
* the backend pair stays the authority it already was (the 409s on
  the result route for anything but COMPLETED, the 404s on cancel);
* the vitest suite carries the mandated lanes.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
SRC = FRONTEND / "src"
CARD = SRC / "components" / "system" / "JobCenterCard.tsx"
SUITE = SRC / "components" / "system" / "__tests__" / "JobCenterCard.test.tsx"
JOBS_CLIENT = SRC / "services" / "jobs" / "client.ts"
PAGE = SRC / "features" / "introduction" / "IntroductionPage.tsx"
ROUTE = ROOT / "src" / "pharmaml" / "api" / "routes" / "jobs.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# the lane's files exist


def test_the_lane_files_exist() -> None:
    for path in (CARD, SUITE, JOBS_CLIENT, PAGE, ROUTE):
        assert path.is_file(), f"missing: {path}"


# ---------------------------------------------------------------------------
# the integration rule — the typed client is the only lane


def test_the_card_reads_the_typed_client_only() -> None:
    card = _read(CARD)
    # exactly one ledger read — the mount-time refresh
    assert card.count("jobsApi.list()") == 1
    # exactly one result fetch and one cancel — the authoritative routes
    assert card.count("jobsApi.result(") == 1
    assert card.count("jobsApi.cancel(") == 1
    # no inline REST, no raw fetch — the typed client is the lane
    assert "api.get(" not in card
    assert "api.post(" not in card
    assert re.search(r"\bfetch\(", card) is None


def test_no_parallel_ledger_client_anywhere_in_the_frontend() -> None:
    offenders: list[Path] = []
    for path in SRC.rglob("*.ts"):
        if path == JOBS_CLIENT:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if '"/api/v1/jobs' in text and ("api.get(" in text or "api.post(" in text):
            offenders.append(path)
    assert offenders == [], f"duplicate jobs-ledger clients: {offenders}"


# ---------------------------------------------------------------------------
# the vocabulary discipline — who may offer which action


def test_only_completed_rows_offer_the_result_lane() -> None:
    card = _read(CARD)
    assert 'job.state === "COMPLETED" ?' in card
    assert '"Hide result"' in card
    # the result fetch rides the openResult handler only
    opened = card.split("const openResult = useCallback(")[1].split("  }, []);", 1)[0]
    assert "await jobsApi.result(job.job_id);" in opened


def test_only_the_live_states_offer_cancellation() -> None:
    card = _read(CARD)
    assert '"QUEUED"' in card and '"RUNNING"' in card
    assert '"CANCELLATION_REQUESTED"' in card
    assert "CANCELLABLE_STATES.includes(job.state)" in card


# ---------------------------------------------------------------------------
# the two-step cancellation gate


def test_the_row_action_arms_and_never_fires() -> None:
    card = _read(CARD)
    arm = card.split("const requestCancel = useCallback(")[1].split("  }, []);", 1)[0]
    assert "setCancelPending(job);" in arm
    assert "jobsApi.cancel" not in arm
    disarm = card.split("const cancelRequest = useCallback(")[1].split("  }, []);", 1)[0]
    assert "jobsApi.cancel" not in disarm


def test_only_the_confirm_button_fires_the_cancel() -> None:
    card = _read(CARD)
    confirm = card.split("const confirmCancel = useCallback(")[1].split("\n  );", 1)[0]
    assert "await jobsApi.cancel(job.job_id);" in confirm
    assert confirm.count("jobsApi.cancel(") == 1
    # the confirmation states the cooperative-cancellation scope
    assert "cooperative cancellation is requested through" in card
    assert "/cancel" in card


def test_the_honest_404_lane_reconciles_the_ledger() -> None:
    card = _read(CARD)
    confirm = card.split("const confirmCancel = useCallback(")[1].split("\n  );", 1)[0]
    assert "err instanceof ApiError && err.status === 404" in confirm
    assert "was already gone (404) — ledger refreshed." in card


def test_the_armed_gate_cannot_outlive_its_row() -> None:
    card = _read(CARD)
    refresh = card.split("const refresh = useCallback(")[1].split("  }, []);", 1)[0]
    assert "setCancelPending((pending) =>" in refresh
    assert "?? null" in refresh


# ---------------------------------------------------------------------------
# the honest splits and the result lane's verbatim failures


def test_the_live_vs_persisted_split_is_stated() -> None:
    card = _read(CARD)
    # the in-session view never pretends to be the persisted history
    # (the prose wraps in the JSX — the pins stay line-safe)
    assert "the persisted history that survives" in card
    assert "the studio database card" in card


def test_a_409_or_404_result_failure_is_surfaced_verbatim() -> None:
    card = _read(CARD)
    opened = card.split("const openResult = useCallback(")[1].split("  }, []);", 1)[0]
    assert "setResultError(describeError(err));" in opened
    # never swallowed, never replaced by an invented summary
    assert "throw" not in opened


# ---------------------------------------------------------------------------
# the backend pair stays the authority it already was


def test_the_backend_route_pair_is_the_authority() -> None:
    route = _read(ROUTE)
    assert '@router.get("")' in route
    assert '@router.get("/{job_id}/result")' in route
    assert '@router.post("/{job_id}/cancel")' in route
    # the result lane's honest 409s — cancelled OR not completed
    assert '"Job was cancelled; no result."' in route
    assert "not completed." in route
    # the cancel lane's honest 404
    assert 'detail="Unknown job_id."' in route


# ---------------------------------------------------------------------------
# the mount + the vitest mandated lanes


def test_the_card_is_mounted_in_the_introduction_workspace() -> None:
    page = _read(PAGE)
    assert "<JobCenterCard />" in page
    assert 'aria-label="Job Center"' in _read(CARD)


def test_the_vitest_suite_carries_the_mandated_lanes() -> None:
    suite = _read(SUITE)
    assert 'describe("JobCenterCard"' in suite
    assert "jobsApi" in suite or "list: vi.fn()" in suite
    for lane in (
        "loads the ledger on mount and renders the cross-workspace rows",
        "renders the honest empty state for a session with no jobs",
        "surfaces a rejected ledger fetch through the error banner",
        "only a COMPLETED row offers Result",
        "the result lane surfaces the backend's own 409 verbatim",
        "the row's Cancel action only ARMS the gate",
        "only the Confirm button fires the typed cancel and acknowledges it",
        "an already-gone job (404) reconciles the ledger honestly",
        "a terminal state never offers cancellation",
    ):
        assert lane in suite, f"missing Job Center vitest lane: {lane}"
