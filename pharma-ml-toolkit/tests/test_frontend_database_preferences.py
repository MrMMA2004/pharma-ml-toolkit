"""part17 §3.2 (Round 360, the G1 lane) — the durable
user-preference manager drift guards.

The Python gate cannot type into the React card, but every contract
the lane rides can be pinned against the real frontend + backend
source — a smuggled second preference client, a dropped client-side
validation gate, a one-click delete, or a hidden 404/409 turns a
guard red before the surface drifts:

* the client quartet: the card reads/writes through the ONE typed
  ``databaseApi`` (preferences / setPreference / deletePreference) —
  no duplicate client, no inline fetch, no parallel lane in the tree;
  the key is encodeURIComponent'd on every write/delete;
* the client-side gates fire BEFORE any request leaves the browser:
  the key bound (1–128, the backend's own rule) and the strict-JSON
  value validation — an invalid payload never reaches the wire;
* the two-step delete gate: the row action only ARMS the
  confirmation; only the confirm button fires the typed delete;
* the honest response lanes: the 404 already-gone lane reconciles
  the list; the 409 non-durable lane renders the honest note (the
  shipped host always runs durable) instead of an error banner;
* the honest split stated in the card: this is the BACKEND's
  durable SQLite store; the app's session settings are a different
  (localStorage) surface — neither pretends to be the other;
* the backend pair stays the lane it already was (the quartet
  routes, the 1–128 bound with its 422, the persistence layer
  beneath);
* the vitest suite carries the mandated lanes.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
SRC = FRONTEND / "src"
CARD = SRC / "components" / "system" / "DatabasePreferences.tsx"
SUITE = SRC / "components" / "system" / "__tests__" / "DatabasePreferences.test.tsx"
DB_CLIENT = SRC / "services" / "database" / "client.ts"
PAGE = SRC / "features" / "introduction" / "IntroductionPage.tsx"
ROUTE = ROOT / "src" / "pharmaml" / "api" / "routes" / "database.py"
PERSISTENCE = ROOT / "src" / "pharmaml" / "api" / "persistence.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# the lane's files exist


def test_the_lane_files_exist() -> None:
    for path in (CARD, SUITE, DB_CLIENT, PAGE, ROUTE, PERSISTENCE):
        assert path.is_file(), f"missing: {path}"


# ---------------------------------------------------------------------------
# the client quartet — the typed lane, encoded keys, runtime narrowing


def test_the_client_carries_the_preference_quartet() -> None:
    client = _read(DB_CLIENT)
    assert "async preferences(): Promise<readonly PreferenceRow[]>" in client
    assert "async setPreference(key: string, value: unknown): Promise<PreferenceSaved>" in client
    assert "async deletePreference(key: string): Promise<PreferenceDeleted>" in client
    assert (
        "`/api/v1/database/preferences/${encodeURIComponent(key)}`" in client
    )


def test_the_client_narrows_at_runtime() -> None:
    client = _read(DB_CLIENT)
    assert "export function parsePreferenceRow(" in client
    assert "the backend did not confirm the save" in client
    assert "the backend did not confirm the delete" in client


def test_no_parallel_preference_client_anywhere_in_the_frontend() -> None:
    offenders: list[Path] = []
    for path in SRC.rglob("*.ts*"):
        if path == DB_CLIENT:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "/api/v1/database/preferences" in text and (
            "api.get(" in text or "api.put(" in text or "api.delete(" in text
        ):
            offenders.append(path)
    assert offenders == [], f"duplicate preference clients: {offenders}"


def test_the_card_reads_the_typed_client_only() -> None:
    card = _read(CARD)
    assert card.count("databaseApi.preferences()") == 1
    assert card.count("databaseApi.setPreference(") == 1
    assert card.count("databaseApi.deletePreference(") == 1
    assert "api.get(" not in card and "api.put(" not in card
    assert "api.delete(" not in card
    assert re.search(r"\bfetch\(", card) is None


# ---------------------------------------------------------------------------
# the client-side gates fire before any request


def test_the_key_bound_is_enforced_client_side() -> None:
    card = _read(CARD)
    gate = card.split("const handleSave = useCallback(")[1].split(
        "\n  const requestDelete", 1
    )[0]
    assert "key.length === 0 || key.length > 128" in gate
    assert "the preference key must be 1–128 characters (the backend's own bound)" in gate
    # the refusal fires BEFORE the busy flag or any request
    assert gate.index("key.length === 0 || key.length > 128") < gate.index(
        "setWriteBusy(true)"
    )


def test_the_value_is_parsed_as_strict_json_before_the_request() -> None:
    card = _read(CARD)
    gate = card.split("const handleSave = useCallback(")[1].split(
        "\n  const requestDelete", 1
    )[0]
    assert "JSON.parse(valueText)" in gate
    assert "the preference value must be strict JSON" in gate


# ---------------------------------------------------------------------------
# the two-step delete gate


def test_the_row_action_arms_and_never_fires() -> None:
    card = _read(CARD)
    arm = card.split("const requestDelete = useCallback(")[1].split("  }, []);", 1)[0]
    assert "setDeletePending(row);" in arm
    assert "databaseApi.deletePreference" not in arm
    disarm = card.split("const cancelDelete = useCallback(")[1].split("  }, []);", 1)[0]
    assert "databaseApi.deletePreference" not in disarm


def test_only_the_confirm_button_fires_the_typed_delete() -> None:
    card = _read(CARD)
    confirm = card.split("const confirmDelete = useCallback(")[1].split("\n  );", 1)[0]
    assert "await databaseApi.deletePreference(row.key);" in confirm
    assert confirm.count("databaseApi.deletePreference(") == 1
    # the confirmation states the scope — exactly one row, nothing else
    assert "This removes exactly this one row from the durable store." in card


# ---------------------------------------------------------------------------
# the honest response lanes


def test_the_404_lane_reconciles_the_list() -> None:
    card = _read(CARD)
    confirm = card.split("const confirmDelete = useCallback(")[1].split("\n  );", 1)[0]
    assert "err instanceof ApiError && err.status === 404" in confirm
    assert "was already gone (404) — the list refreshed." in card


def test_the_409_non_durable_lane_renders_the_honest_note() -> None:
    card = _read(CARD)
    refresh = card.split("const refresh = useCallback(")[1].split(
        "\n  useEffect(", 1
    )[0]
    assert "err instanceof ApiError && err.status === 409" in refresh
    assert (
        "Non-durable instance — the preference store is unavailable "
        "(the shipped host always runs durable)."
    ) in card


def test_the_durable_vs_session_split_is_stated() -> None:
    card = _read(CARD)
    # the prose wraps in the JSX — the pins stay line-safe
    assert "The backend" in card and "local SQLite database" in card
    assert "localStorage" in card
    assert "neither pretends to" in card and "be the other." in card


# ---------------------------------------------------------------------------
# the backend pair stays the lane it already was


def test_the_backend_quartet_and_the_key_bound() -> None:
    route = _read(ROUTE)
    assert '@router.get("/preferences")' in route
    assert '@router.get("/preferences/{key}")' in route
    assert '@router.put("/preferences/{key}")' in route
    assert '@router.delete("/preferences/{key}")' in route
    assert 'detail="preference key must be 1-128 chars"' in route
    assert '"saved": True' in route
    assert '"deleted": True' in route


def test_the_persistence_layer_beneath() -> None:
    persistence = _read(PERSISTENCE)
    for fn in (
        "def set_preference(self, key: str, value: Any) -> None:",
        "def get_preference(",
        "def list_preferences(self) -> list[dict]:",
        "def delete_preference(self, key: str) -> bool:",
    ):
        assert fn in persistence, f"missing persistence function: {fn}"
    assert "user_preferences" in persistence


# ---------------------------------------------------------------------------
# the mount + the vitest mandated lanes


def test_the_card_is_mounted_in_the_introduction_workspace() -> None:
    page = _read(PAGE)
    assert "<DatabasePreferences />" in page
    assert 'aria-label="Studio preferences"' in _read(CARD)


def test_the_vitest_suite_carries_the_mandated_lanes() -> None:
    suite = _read(SUITE)
    assert 'describe("DatabasePreferences"' in suite
    for lane in (
        "loads the durable store on mount and renders the rows",
        "renders the honest empty state",
        "surfaces a rejected fetch through the error banner",
        "a strict-JSON value over a valid key fires the typed write and refreshes",
        "an invalid JSON value never leaves the browser — no request fires",
        "an out-of-bound key is refused client-side — no request fires",
        "the row's Delete action only ARMS the gate — no request yet",
        "Cancel disarms the gate without any request",
        "only the Confirm button fires the typed delete, refreshes, and reports it",
        "an already-gone key (404) reconciles the list honestly",
        "a non-durable instance (409) renders the honest note",
    ):
        assert lane in suite, f"missing preferences vitest lane: {lane}"
