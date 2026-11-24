"""part17 §3.1 (Round 356) — the dataset-deletion drift guards.

The Python gate cannot click the React confirmation, but every
contract the lane rides can be pinned against the real frontend +
backend source — a dropped confirmation gate, a smuggled second
delete client, a missing 404 lane, or a silent stage renumbering
turns a guard red before the surface drifts:

* the integration rule: the page's ONLY deletion lane is the typed
  ``datasetsApi.remove`` (the API-tested DELETE route — no duplicate
  client, no inline fetch, no parallel delete implementation
  anywhere in the frontend tree);
* the two-step gate: the row action arms a confirmation that renders
  the dataset's name + metadata; only the confirm button's handler
  contains the ``remove`` call;
* the honest 404 lane — already-gone reconciles the registry;
* the registry refresh + the selection-reset branch (a deleted
  selection can never dangle);
* the propagation guard — the row's delete action never selects;
* the backend route + store pair stays the evict lane it already
  was (the honest 404, the transaction-consistent ``_forget`` with
  ``persistence.delete_dataset``);
* the vitest suite carries the §3.1 flows and the stage order
  1..8 stays untouched (no renumbering, §11.1's fixed order).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
SRC = FRONTEND / "src"
PAGE = SRC / "features" / "datasets" / "DatasetsPage.tsx"
CLIENT = SRC / "services" / "datasets" / "client.ts"
SUITE = SRC / "features" / "datasets" / "__tests__" / "DatasetsPage.test.tsx"
ROUTES = ROOT / "src" / "pharmaml" / "api" / "routes" / "datasets.py"
STORES = ROOT / "src" / "pharmaml" / "api" / "stores.py"
API_TESTS = ROOT / "tests" / "api" / "test_datasets_ml_artifacts.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# the lane's files exist


def test_the_deletion_lane_files_exist() -> None:
    for path in (PAGE, CLIENT, SUITE, ROUTES, STORES, API_TESTS):
        assert path.is_file(), f"missing: {path}"


# ---------------------------------------------------------------------------
# the integration rule — the typed client is the only delete lane


def test_the_page_deletes_through_the_typed_client_only() -> None:
    page = _read(PAGE)
    # exactly one remove call site — inside the confirm handler
    assert page.count("datasetsApi.remove(") == 1
    # no inline REST delete, no raw fetch — the client is the lane
    assert "api.delete(" not in page
    assert re.search(r"\bfetch\(", page) is None


def test_no_parallel_delete_client_anywhere_in_the_frontend() -> None:
    offenders: list[Path] = []
    for path in (SRC).rglob("*.ts"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if path == CLIENT:
            continue
        if "api.delete(`/api/v1/datasets/" in text or (
            "api.delete(\"/api/v1/datasets/" in text
        ):
            offenders.append(path)
    assert offenders == [], f"duplicate dataset-delete clients: {offenders}"


def test_the_client_remove_pins_the_route_and_encoding() -> None:
    client = _read(CLIENT)
    assert "async remove(datasetId: string): Promise<void>" in client
    assert "api.delete(`/api/v1/datasets/${encodeURIComponent(datasetId)}`" in client


# ---------------------------------------------------------------------------
# the two-step confirmation gate


def test_the_row_action_arms_the_gate_and_never_selects() -> None:
    page = _read(PAGE)
    assert "const requestDelete = useCallback" in page
    assert "setDeletePending(entry);" in page
    # the propagation guard — the trash never selects the row
    assert "event.stopPropagation();" in page
    assert 'aria-label={`Delete dataset ${entry.name}`}' in page


def test_the_confirmation_renders_name_and_metadata() -> None:
    page = _read(PAGE)
    # the confirmation title carries the dataset's NAME
    assert "Delete dataset “{deletePending.name}”?" in page
    # the metadata line: rows, columns, target, task, created, id
    assert "{deletePending.shape[0]} rows × {deletePending.shape[1]} columns" in page
    assert "{deletePending.created_at}" in page
    assert "id {deletePending.dataset_id}" in page
    # the scope note — never touches another record
    assert "never touches another record" in page
    # the explicit Cancel + the destructive confirm button
    assert "onClick={cancelDelete}" in page
    assert 'variant="destructive"' in page


def test_only_the_confirm_button_fires_the_request() -> None:
    page = _read(PAGE)
    # the remove call site rides the confirmDelete handler
    confirm = page.split("const confirmDelete = useCallback(")[1]
    confirm = confirm.split("  );", 1)[0]
    assert "await datasetsApi.remove(entry.dataset_id);" in confirm
    assert confirm.count("datasetsApi.remove(") == 1
    # the arm handler never fires the request
    arm = page.split("const requestDelete = useCallback(")[1].split("  }, []);", 1)[0]
    assert "datasetsApi.remove" not in arm
    # the cancel handler never fires the request either
    cancel = page.split("const cancelDelete = useCallback(")[1].split("  }, []);", 1)[0]
    assert "datasetsApi.remove" not in cancel


# ---------------------------------------------------------------------------
# the honest response lanes


def test_the_404_lane_reconciles_the_registry() -> None:
    page = _read(PAGE)
    assert "error.status === 404" in page
    assert "was already gone (404) — registry refreshed." in page


def test_success_and_404_both_refresh_the_registry() -> None:
    page = _read(PAGE)
    confirm = page.split("const confirmDelete = useCallback(")[1]
    confirm = confirm.split("  );", 1)[0]
    # the success path and the 404 path each re-list the registry
    assert confirm.count("await refreshRegistry();") == 2


def test_a_deleted_selection_cannot_dangle() -> None:
    page = _read(PAGE)
    assert "if (entry.dataset_id === selectedId) {" in page
    # the branch clears the selection AND its stage-local derivatives
    branch = page.split("if (entry.dataset_id === selectedId) {", 1)[1]
    branch = branch.split("await refreshRegistry();", 1)[0]
    for reset in (
        "setSelectedId(null)",
        "setSelected(null)",
        "setPreview(null)",
        "setProfile(null)",
        "setSplitReport(null)",
        "setExportArtifact(null)",
    ):
        assert reset in branch, f"missing reset in the deleted-selection branch: {reset}"


# ---------------------------------------------------------------------------
# the backend pair — the route stays the evict lane it already was


def test_the_backend_route_and_store_pair() -> None:
    routes = _read(ROUTES)
    assert '@router.delete("/{dataset_id}")' in routes
    assert "if not store.evict(dataset_id):" in routes
    assert 'raise HTTPException(status_code=404, detail="Unknown dataset_id.")' in routes
    assert '"evicted": True' in routes

    stores = _read(STORES)
    # the DatasetStore evict — the transaction-consistent forget
    # (the session record AND the persisted copy, nothing else)
    evict = stores.split("def evict(self, dataset_id: str) -> bool:", 1)[1]
    evict = evict.split("\n    def ", 1)[0] if "\n    def " in evict else evict
    assert "self._forget(dataset_id)" in evict
    forget = stores.split("def _forget(self, dataset_id: str) -> None:", 1)[1]
    forget = forget.split("\n    def ", 1)[0]
    assert "self._items.pop(dataset_id, None)" in forget
    assert "self._persistence.delete_dataset(dataset_id)" in forget


def test_the_api_suite_still_carries_the_evict_contracts() -> None:
    api = _read(API_TESTS)
    assert "def test_delete_then_404" in api
    assert 'r.json()["evicted"] is True' in api


# ---------------------------------------------------------------------------
# the suites — the vitest flows + the untouched stage order


def test_the_vitest_suite_carries_the_deletion_flows() -> None:
    suite = _read(SUITE)
    assert "describe(\"the §3.1 deletion lane" in suite
    assert "remove: vi.fn()" in suite
    assert "datasetsMocks.remove.mockReset()" in suite
    for flow in (
        "no request yet",
        "never selects the row (the propagation guard)",
        "Cancel disarms the gate without any request",
        "confirm evicts through the typed client, refreshes the registry",
        "never a dangling record",
        "the honest 404 lane",
        "a hard failure stays on the confirmation card",
    ):
        assert flow in suite, f"missing §3.1 vitest flow: {flow}"


def test_the_stage_order_1_through_8_is_untouched() -> None:
    page = _read(PAGE)
    indices = re.findall(r'<StageHeader\s+index={(\d)}', page)
    assert indices == ["1", "2", "3", "4", "5", "6", "7", "8"], (
        f"the §11.1 fixed stage order drifted: {indices}"
    )
    # stage 2 stays the Inspect stage that carries the registry + the gate
    inspect = page.split('<StageHeader\n        index={2}', 1)[1].split("index={3}", 1)[0]
    assert 'title="Inspect"' in inspect
    assert "deletePending" in inspect
