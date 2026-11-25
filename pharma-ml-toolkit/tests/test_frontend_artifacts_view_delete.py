"""part17 §3.2 (Round 360, the G1 lane) — the Visualization
consumption hub's artifact view/delete drift guards.

The Python gate cannot click the React rows, but every contract the
lane rides can be pinned against the real frontend source — a
smuggled second delete client, a one-click delete, a dangling
reference to an evicted row, or a hidden 404 turns a guard red
before the surface drifts:

* the integration rule: the page's ONLY view/delete lanes are the
  EXISTING typed ``artifactsApi.view`` / ``artifactsApi.delete``
  (the API-tested GET/DELETE /artifacts/{id} routes — no duplicate
  client, no inline fetch, no parallel lane in the tree);
* the two-step gate: the row's Delete action only ARMS the
  confirmation; only the confirm button fires the typed delete;
* the never-dangling rule: every stage-local derivative that
  referenced the evicted row is cleared (the selection, the viewed
  record, the compare memory, the digest cache) — never a reference
  to an id the registry no longer carries;
* the honest 404 lane: an already-gone row reconciles the registry
  and says so;
* the view lane surfaces its 404 verbatim (a row can be evicted
  server-side between the list and the click);
* the vitest suite carries the mandated lanes (the metadata panel,
  the arm/confirm split, the 404 reconciliation, the stray-fetch
  discipline).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
SRC = FRONTEND / "src"
PAGE = SRC / "features" / "visualization" / "VisualizationPage.tsx"
SUITE = SRC / "features" / "visualization" / "__tests__" / "VisualizationPage.test.tsx"
ARTIFACTS_CLIENT = SRC / "services" / "artifacts" / "client.ts"
ROUTES = ROOT / "src" / "pharmaml" / "api" / "routes" / "artifacts.py"
API_TESTS = ROOT / "tests" / "api" / "test_datasets_ml_artifacts.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# the lane's files exist


def test_the_lane_files_exist() -> None:
    for path in (PAGE, SUITE, ARTIFACTS_CLIENT, ROUTES, API_TESTS):
        assert path.is_file(), f"missing: {path}"


# ---------------------------------------------------------------------------
# the integration rule — the typed client is the only lane


def test_the_page_view_deletes_through_the_typed_client_only() -> None:
    page = _read(PAGE)
    # exactly one view fetch and one delete call site
    assert page.count("artifactsApi.view(") == 1
    assert page.count("artifactsApi.delete(") == 1
    # no inline REST, no raw fetch — the client is the lane
    assert "api.get(" not in page and "api.delete(" not in page
    assert re.search(r"\bfetch\(", page) is None


def test_no_parallel_artifact_view_delete_client_anywhere() -> None:
    offenders: list[Path] = []
    for path in SRC.rglob("*.ts"):
        if path == ARTIFACTS_CLIENT:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "/api/v1/artifacts/" in text and (
            "api.get(" in text or "api.delete(" in text
        ):
            offenders.append(path)
    assert offenders == [], f"duplicate artifact view/delete clients: {offenders}"


def test_the_client_pins_the_routes_and_encoding() -> None:
    client = _read(ARTIFACTS_CLIENT)
    assert "async view(artifactId: string): Promise<ArtifactView>" in client
    assert "async delete(artifactId: string): Promise<ArtifactEvicted>" in client
    assert (
        "`/api/v1/artifacts/${encodeURIComponent(artifactId)}`" in client
    )


# ---------------------------------------------------------------------------
# the two-step delete gate


def test_the_row_action_arms_and_never_fires() -> None:
    page = _read(PAGE)
    arm = page.split("const requestArtifactDelete = useCallback(")[1]
    arm = arm.split("  }, []);", 1)[0]
    assert "setDeletePending(artifact);" in arm
    assert "artifactsApi.delete" not in arm
    disarm = page.split("const cancelArtifactDelete = useCallback(")[1]
    disarm = disarm.split("  }, []);", 1)[0]
    assert "artifactsApi.delete" not in disarm


def test_only_the_confirm_button_fires_the_typed_delete() -> None:
    page = _read(PAGE)
    confirm = page.split("const confirmArtifactDelete = useCallback(")[1]
    confirm = confirm.split("\n  );", 1)[0]
    assert "await artifactsApi.delete(artifact.artifact_id);" in confirm
    assert confirm.count("artifactsApi.delete(") == 1


def test_the_confirmation_states_the_scope() -> None:
    page = _read(PAGE)
    # the lane comment — the EXISTING typed client, no second client
    assert "EXISTING typed artifactsApi.delete" in page


# ---------------------------------------------------------------------------
# the never-dangling rule — the evicted id leaves every derivative


def test_the_deleted_row_cannot_dangle_anywhere() -> None:
    page = _read(PAGE)
    confirm = page.split("const confirmArtifactDelete = useCallback(")[1]
    confirm = confirm.split("\n  );", 1)[0]
    for reset in (
        "setSelectedArtifact(null)",
        "setSelectedDigest(null)",
        "setViewedArtifact(null)",
        "prev.filter((row) => row.artifact_id !== artifact.artifact_id)",
        "delete next[artifact.artifact_id]",
    ):
        assert reset in confirm, f"missing derivative reset: {reset}"


# ---------------------------------------------------------------------------
# the honest response lanes


def test_the_404_lane_reconciles_the_registry() -> None:
    page = _read(PAGE)
    confirm = page.split("const confirmArtifactDelete = useCallback(")[1]
    confirm = confirm.split("\n  );", 1)[0]
    assert "error instanceof ApiError && error.status === 404" in confirm
    assert "was already gone (404) — registry refreshed." in page


def test_success_and_404_both_refresh_the_registry() -> None:
    page = _read(PAGE)
    confirm = page.split("const confirmArtifactDelete = useCallback(")[1]
    confirm = confirm.split("\n  );", 1)[0]
    assert confirm.count("await refreshRegistry();") == 2
    assert 'Deleted "${artifact.filename}" from the artifact registry.' in page


def test_the_view_lane_surfaces_its_404_verbatim() -> None:
    page = _read(PAGE)
    opened = page.split("const handleArtifactView = useCallback(")[1]
    opened = opened.split("  }, []);", 1)[0]
    assert "await artifactsApi.view(artifact.artifact_id);" in opened
    assert "setViewError(describeError(error));" in opened
    assert "setViewedArtifact(null);" in opened


# ---------------------------------------------------------------------------
# the backend pair stays the evict lane it already was


def test_the_backend_route_and_api_suite_pins() -> None:
    routes = _read(ROUTES)
    assert '@router.get("/{artifact_id}")' in routes
    assert '@router.delete("/{artifact_id}")' in routes
    assert '"evicted": True' in routes
    api = _read(API_TESTS)
    assert "def test_delete_then_404" in api or "evicted" in api


# ---------------------------------------------------------------------------
# the vitest mandated lanes


def test_the_vitest_suite_carries_the_mandated_lanes() -> None:
    suite = _read(SUITE)
    # the suite's own header carries the round + lane reference
    assert "part17" in suite and "Round 360" in suite
    # the network boundary is the ONLY mocked layer — the real
    # parseArtifactView rides the page under test
    assert "view: vi.fn()" in suite
    assert "delete: vi.fn()" in suite
    assert "list: vi.fn()" in suite
    for lane in (
        "renders the registry rows with the View and Delete actions",
        "the View action fetches the full record and renders the metadata panel",
        "the View lane surfaces a 404 verbatim",
        "the Delete action only ARMS the confirmation",
        "Cancel disarms the gate without any request",
        "only the Confirm button fires the typed delete, refreshes the registry, and reports it",
        "an already-gone row (404) reconciles the registry honestly",
        "every other failure stays on the confirmation card verbatim",
    ):
        assert lane in suite, f"missing artifacts view/delete vitest lane: {lane}"
