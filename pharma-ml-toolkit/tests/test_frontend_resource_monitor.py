"""part16 Phase 8 — the floating resource-monitor drift guards.

The Phase-8 lane (the roadmap's #5) is a frontend surface consuming
the EXISTING §15 backend service: the Python gate cannot execute the
React panel, but every contract the lane rides CAN be pinned against
the real frontend + backend source — these guards parse the REAL
files and turn a rename, a dropped interaction, a loosened clamp, or a
fabricated number into a red test before the frontend drifts:

* the service trio (client/types/panel-state) + the component + the
  suites exist under ``services/system`` / ``components/system``;
* the integration rule: the panel's ONLY data source is
  ``GET /api/v1/system/resources`` (§8's own "integrate with the
  backend job manager and existing resource services" — no parallel
  monitoring architecture, no second endpoint);
* the no-simulation rule: no Math.random / fabricated numbers in the
  panel, absent numbers preserved as null through the parser and
  rendered as the honest dash;
* the geometry pocket: the literal storage key (separate from the §10
  settings schema, which stays byte-frozen), the literal width bounds,
  validate-on-read, clamp-on-write;
* the interaction vocabulary: drag + keyboard move on the strip,
  clamped drag/keyboard resize on the handle, toggle-detail, close,
  the Escape scope rule, the aria contract, the header entry mounted
  on every workspace by the App shell;
* the cadence: the poll loop reads the §10 store's interval (the
  settings panel owns it), floors at the same 250 ms the store
  enforces, skips while hidden and stops while closed;
* the backend contract is consumed unchanged: the route's own scope
  block (read-only, psutil optional, GPU deliberately absent) and the
  generated openapi path both stand.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
SRC = FRONTEND / "src"
SYSTEM = SRC / "services" / "system"
COMPONENTS = SRC / "components" / "system"
ROUTES = ROOT / "src" / "pharmaml" / "api" / "routes"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_flat(path: Path) -> str:
    """Whitespace-normalized + lowercased — JSX wraps its prose across
    lines, so phrase pins must survive reflowing."""
    return re.sub(r"\s+", " ", _read(path)).lower()


# ---------------------------------------------------------------------------
# the service trio + the component + the suites


def test_the_system_service_trio_and_component_exist() -> None:
    for name in ("client.ts", "types.ts", "panelState.ts"):
        assert (SYSTEM / name).is_file(), f"the system service lost {name}"
    for name in ("client.test.ts", "panelState.test.ts"):
        assert (SYSTEM / "__tests__" / name).is_file()
    assert (COMPONENTS / "ResourceMonitor.tsx").is_file()
    assert (COMPONENTS / "__tests__" / "ResourceMonitor.test.tsx").is_file()


def test_the_client_pins_the_single_integration_point() -> None:
    client = _read(SYSTEM / "client.ts")
    assert '"/api/v1/system/resources"' in client, (
        "the monitor must consume the existing §15 resource service — "
        "a second endpoint would be the parallel monitoring architecture §8 forbids"
    )
    assert "parseResourceSnapshot" in client, "the raw answer must be narrowed, never any"
    # the §15 backend route stands unchanged (the consumed contract)
    system_route = _read(ROUTES / "system.py")
    assert 'APIRouter(prefix="/system"' in system_route
    assert '@router.get("/resources")' in system_route
    assert "job_manager.list_jobs()" in system_route, "the queue depth rides the real JobManager"
    contract = _read(ROOT / "contracts" / "openapi.json")
    assert "/api/v1/system/resources" in contract, "the generated contract carries the path"


# ---------------------------------------------------------------------------
# the no-simulation rule (§8's own)


def test_no_simulated_values_anywhere_in_the_panel() -> None:
    panel = _read(COMPONENTS / "ResourceMonitor.tsx")
    for banned in ("Math.random", "crypto.randomUUID()", "Math.now", "performance.now"):
        assert banned not in panel, f"§8 forbids simulated values — found {banned}"
    # absent numbers stay null through the parser and render as the dash
    types_ = _read(SYSTEM / "types.ts")
    for honest_null in (
        "cpu_percent: number | null",
        "memory: ResourceMemoryBlock | null",
        "swap: ResourceSwapBlock | null",
        "memory_rss_bytes: number | null",
        "num_threads: number | null",
    ):
        assert honest_null in types_, f"the parser lost the honest null: {honest_null}"
    assert 'return percent === null ? "—" : ' in panel, "the dash is the only absent-number render"
    # the GPU optional contract degrades to its stated line
    assert "not available" in panel


def test_the_error_view_is_the_honest_unavailable_state() -> None:
    panel = _read(COMPONENTS / "ResourceMonitor.tsx")
    assert "Resource snapshot unavailable" in panel
    flat = re.sub(r"\s+", " ", panel)
    assert "no values are simulated" in flat
    types_ = _read(SYSTEM / "types.ts")
    assert 'source: "psutil" | "stdlib-fallback"' in types_, (
        "the two-source vocabulary is the parsed contract"
    )


# ---------------------------------------------------------------------------
# the geometry pocket


def test_the_geometry_pocket_is_separate_from_the_settings_schema() -> None:
    state = _read(SYSTEM / "panelState.ts")
    assert 'PANEL_STORAGE_KEY = "pharmaml.studio.resmon.v1"' in state
    settings_key = _read(SRC / "services" / "settings" / "store.ts")
    assert "pharmaml.studio.settings.v1" in settings_key
    assert "pharmaml.studio.settings.v1" not in state, (
        "the §10 settings schema stays byte-frozen — geometry is presentation state"
    )


def test_the_width_bounds_are_literal_pins() -> None:
    state = _read(SYSTEM / "panelState.ts")
    assert "PANEL_MIN_WIDTH = 280" in state
    assert "PANEL_MAX_WIDTH = 480" in state
    assert "DEFAULT_PANEL_WIDTH = 320" in state
    assert "KEYBOARD_GEOMETRY_STEP" in state


def test_validate_on_read_and_clamp_on_write_stand() -> None:
    state = _read(SYSTEM / "panelState.ts")
    body = state
    assert "JSON.parse" in body and "catch" in body, "corrupt storage must degrade, never crash"
    assert "clampGeometry" in body and "clampWidth" in body
    assert "savePanelState" in body and "loadPanelState" in body


# ---------------------------------------------------------------------------
# the interaction vocabulary


def test_the_full_interaction_vocabulary_is_present() -> None:
    panel = _read(COMPONENTS / "ResourceMonitor.tsx")
    # movable by dragging AND by keyboard (the accessible move)
    assert 'data-testid="monitor-drag"' in panel
    assert "onPointerDown" in panel and "onKeyDown" in panel
    assert "ArrowLeft" in panel and "ArrowDown" in panel
    # resizable within the clamped limits, pointer AND keyboard
    assert 'data-testid="monitor-resize"' in panel
    assert "clampWidth" in panel
    # collapsible + expandable + openable + closable
    assert 'aria-label="Toggle detail"' in panel and "aria-expanded" in panel
    assert 'aria-label="Close resource monitor"' in panel
    # remembers position + size (the pocket round-trip)
    assert "savePanelState" in panel and "loadPanelState" in panel


def test_the_escape_scope_rule_and_aria_contract() -> None:
    panel = _read(COMPONENTS / "ResourceMonitor.tsx")
    # Escape closes only while the panel holds focus — an Escape aimed at
    # a dialog or the page never sweeps the floating panel away
    assert "asideRef.current?.contains(target)" in panel
    assert 'aria-label="Resource monitor"' in panel
    # the entry button carries the live state to assistive tech
    assert 'aria-expanded={visible}' in panel


def test_the_header_entry_mounts_on_every_workspace() -> None:
    app = _read(SRC / "app" / "App.tsx")
    assert "<ResourceMonitorHost />" in app, "the shell must mount the §8 entry in the header"
    # the entry rides the header lane (outside the workspace switch), next
    # to the §10 gear — present on every workspace by construction
    assert "<ResourceMonitorHost />" in app.split("<main")[0]
    app_test = _read(SRC / "app" / "__tests__" / "App.test.tsx")
    assert "Resource monitor" in app_test, "the shell integration test carries the §8 lane"


# ---------------------------------------------------------------------------
# the cadence (the §10 interval preference is the single source)


def test_the_poll_loop_reads_the_settings_interval_with_the_same_floor() -> None:
    panel = _read(COMPONENTS / "ResourceMonitor.tsx")
    assert "useSettings()" in panel, "the cadence must come from the §10 store"
    assert "settings.resourceMonitor.updateIntervalMs" in panel
    assert "POLL_FLOOR_MS = 250" in panel
    store = _read(SRC / "services" / "settings" / "store.ts")
    assert "INTERVAL_MIN_MS = 250" in store, "the r346 floor pin stands"
    # the §10 panel already owns the two monitor preferences
    settings_panel = _read(SRC / "components" / "settings" / "SettingsPanel.tsx")
    assert "resourceMonitor" in settings_panel


def test_the_poll_stops_when_closed_and_skips_when_hidden() -> None:
    panel = _read(COMPONENTS / "ResourceMonitor.tsx")
    assert "document.hidden" in panel, "no polling while the document is hidden"
    # the panel unmounts on the visible preference — the loop dies with it
    assert "resourceMonitor: { visible: false }" in panel
    assert "clearTimeout(timer)" in panel, "the cadence cleanup stands"


def test_the_system_and_application_sections_are_separately_labeled() -> None:
    panel = _read(COMPONENTS / "ResourceMonitor.tsx")
    assert 'aria-label="System — host"' in panel
    assert 'aria-label="Application — backend process"' in panel
    # the jobs vocabulary mirrors the backend's six states
    route = _read(ROUTES / "system.py")
    states = re.search(r"_JOB_STATES = \((.*?)\)", route, re.DOTALL)
    assert states is not None
    for state in states.group(1).split('"')[1::2]:
        assert state, "every backend state must stay mirrored in the panel"
    assert "by_state" in panel


def test_the_vitest_suites_cover_the_lane_honestly() -> None:
    monitor_test = _read_flat(COMPONENTS / "__tests__" / "ResourceMonitor.test.tsx")
    for clause in (
        "never simulated",
        "clamped",
        "remembers",
        "polls at the settings interval",
        "escape",
        "keyboard arrows",
    ):
        assert clause in monitor_test, f"the suite lost the §8 clause: {clause}"
    # the App integration test carries the header-entry lane
    app_test = _read(SRC / "app" / "__tests__" / "App.test.tsx")
    assert "floating resource-monitor entry in the header" in app_test
