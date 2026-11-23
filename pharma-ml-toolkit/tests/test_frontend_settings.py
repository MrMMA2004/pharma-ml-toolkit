"""part16 Phase 7 — the settings-lane drift guards.

The Phase-7 settings lane (the roadmap's #4) is a frontend surface:
the Python gate cannot execute the React code, but every contract
the lane rides CAN be pinned against the real frontend source —
these guards parse the REAL service/store/panel/shell files and
turn a rename, a dropped control, a loosened bound, or a lost DOM
application into a red test before the frontend drifts:

* the settings service trio + the hook + the version carrier exist
  with the §10 inventory (theme/density/charts incl. export
  defaults/notifications/execution defaults/resource-monitor
  prefs/diagnostics/reset);
* the validate-before-apply bounds as literal pins (the -1..64
  parallelism contract of the r344 Phase-2 fix, the >=250 ms monitor
  floor, the enum vocabularies, the unknown-key rejection that
  enforces §10's no-unsafe-values rule);
* the persistence layer (the schema-versioned localStorage key,
  validate-on-read so corrupt storage degrades to defaults);
* the compact surface: the header entry on every workspace (the App
  shell mounts it) + the popover dialog + the §10 section set + the
  inline validation alert + reset-to-default;
* the three DOM applications: the ``.dark`` theme class protocol, the
  ``data-density`` scale, and the ``.charts-cvd`` Okabe-Ito palette
  block in globals.css (all five chart tokens);
* the ML form's n_jobs prefill reads the settings store (the
  execution-default application path);
* the notification preference's consumer: the shell's toast watcher
  subscribes the REAL event stream and filters the vocabulary's own
  terminal three (job.completed / job.failed / job.cancelled);
* the version carrier lockstep (version.ts == frontend package.json
  == the toolkit VERSION file).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
SRC = FRONTEND / "src"
SETTINGS = SRC / "services" / "settings"
COMPONENTS = SRC / "components" / "settings"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# the service trio + the §10 inventory


def test_the_settings_service_trio_and_react_binding_exist() -> None:
    for name in ("types.ts", "store.ts"):
        assert (SETTINGS / name).is_file(), f"the settings service lost {name}"
    assert (SETTINGS / "__tests__" / "store.test.ts").is_file()
    hook = _read(SRC / "hooks" / "useSettings.ts")
    assert "useSyncExternalStore" in hook, "the hook left the house state pattern"
    assert "getSettingsStore" in hook


def test_the_defaults_carry_the_full_section10_inventory() -> None:
    defaults = _read(SETTINGS / "types.ts")
    for key in (
        "schemaVersion",
        "theme",
        "density",
        "colorblindSafe",
        "defaultExportFormat",
        "jobCompletionToasts",
        "defaultParallelism",
        "updateIntervalMs",
        "visible",
    ):
        assert key in defaults, f"the DEFAULT_SETTINGS lost the §10 key {key}"
    for enum in ('"light" | "dark" | "system"', '"comfortable" | "compact"', '"png" | "svg"'):
        assert enum in defaults, f"the §10 vocabulary lost {enum}"


def test_the_shipped_defaults_are_the_conservative_pins() -> None:
    defaults = _read(SETTINGS / "types.ts")
    # theme: system (prefers-color-scheme at first paint, §10)
    assert re.search(r'theme:\s*"system"', defaults) is not None
    # the r344 conservative n_jobs default rides the settings lane unchanged
    assert re.search(r"defaultParallelism:\s*1[,\n]", defaults) is not None
    assert re.search(r"updateIntervalMs:\s*2_000", defaults) is not None


# ---------------------------------------------------------------------------
# validate-before-apply (the bounds as literal pins)


def test_the_validation_bounds_are_pinned() -> None:
    store = _read(SETTINGS / "store.ts")
    assert "PARALLELISM_MIN = -1" in store
    assert "PARALLELISM_MAX = 64" in store
    assert "INTERVAL_MIN_MS = 250" in store
    assert "INTERVAL_MAX_MS = 60_000" in store
    # the r344 Phase-2 contract: the -1..64 passthrough preserved
    assert "Number.isInteger" in store, "non-integer parallelism must be refused"


def test_unknown_keys_are_rejected_the_no_unsafe_values_rule() -> None:
    store = _read(SETTINGS / "store.ts")
    assert 'unknown setting' in store, "the unknown-key rejection vanished"
    for forbidden in ("token", "secret", "password", "apiKey"):
        assert not re.search(
            rf"readonly {forbidden}\??\s*:", store
        ), f"an unsafe setting field ({forbidden}) appeared"


def test_the_storage_layer_is_schema_versioned_and_validated_on_read() -> None:
    store = _read(SETTINGS / "store.ts")
    assert 'SETTINGS_STORAGE_KEY = "pharmaml.studio.settings.v1"' in store
    assert "validateSettings(JSON.parse" in store, "validate-on-read vanished"
    # corrupt storage degrades to defaults (never a crash)
    assert "catch" in store
    assert "readPersisted() ?? DEFAULT_SETTINGS" in store


# ---------------------------------------------------------------------------
# the compact surface + the DOM applications


def test_the_header_entry_and_the_popover_dialog() -> None:
    entry = _read(COMPONENTS / "SettingsEntry.tsx")
    assert 'aria-label="Settings"' in entry
    assert 'aria-haspopup="dialog"' in entry
    assert 'role="dialog"' in entry
    assert 'aria-label="Application settings"' in entry
    assert "Escape" in entry, "the popover must close on Escape"
    assert "pointerdown" in entry, "the popover must close on outside click"
    # §10: compact — never a full-screen page
    assert "min(24rem" in entry or "w-[min(24rem" in entry


def test_the_panel_carries_the_section10_control_set() -> None:
    panel = _read(COMPONENTS / "SettingsPanel.tsx")
    for section in (
        "Appearance",
        "Charts",
        "Scientific execution defaults",
        "Notifications",
        "Resource monitor",
        "Diagnostics",
    ):
        assert section in panel, f"the panel lost the §10 section {section}"
    assert "Reset all settings to defaults" in panel
    assert 'role="alert"' in panel, "validation errors must surface inline"
    assert "Default parallelism (n_jobs)" in panel
    assert "Colorblind-safe chart palette" in panel
    assert "Job completion toasts" in panel
    assert "Monitor update interval (ms)" in panel


def test_the_app_shell_mounts_the_entry_on_every_workspace() -> None:
    app = _read(SRC / "app" / "App.tsx")
    assert "<SettingsEntry />" in app, "the header lost the settings entry"
    assert "<JobCompletionToasts />" in app, "the shell lost the toast lane"


def test_the_theme_class_protocol_and_the_density_and_cvd_palettes() -> None:
    store = _read(SETTINGS / "store.ts")
    assert 'classList.toggle(\n      "dark"' in store or 'classList.toggle("dark"' in store
    assert 'documentElement.dataset.density' in store
    assert '"charts-cvd"' in store
    css = _read(SRC / "styles" / "globals.css")
    cvd = css.split(".charts-cvd")[-1]
    for token in ("--chart-1", "--chart-2", "--chart-3", "--chart-4", "--chart-5"):
        assert token in cvd, f"the CVD palette lost {token}"
    for rule in (
        'html[data-density="comfortable"]',
        'html[data-density="compact"]',
    ):
        assert rule in css, f"the density scale lost {rule}"


def test_the_ml_form_prefill_reads_the_settings_store() -> None:
    ml = _read(SRC / "features" / "ml" / "MLPage.tsx")
    assert (
        "getSettingsStore().snapshot().execution.defaultParallelism" in ml
    ), "the execution default no longer pre-fills the ML form"


# ---------------------------------------------------------------------------
# the notification preference's consumer


def test_the_toast_lane_rides_the_real_terminal_event_types() -> None:
    toasts = _read(COMPONENTS / "JobCompletionToasts.tsx")
    events = _read(SRC / "services" / "websocket" / "events.ts")
    for terminal in ("job.completed", "job.failed", "job.cancelled"):
        assert f'"{terminal}"' in toasts, f"the toast lane lost {terminal}"
        assert f'"{terminal}"' in events, f"the backend vocabulary lost {terminal}"
    assert "getEventStream()" in toasts, "the watcher must subscribe the real stream"
    assert "jobCompletionToasts" in toasts, "the preference must gate the lane"
    assert "event_id" in toasts, "the dedup (reconnect replays) vanished"


# ---------------------------------------------------------------------------
# the version-carrier lockstep


def test_the_version_carrier_lockstep() -> None:
    carrier = _read(SRC / "lib" / "version.ts")
    match = re.search(r'APP_VERSION = "([^"]+)"', carrier)
    assert match is not None, "the version carrier lost its constant"
    frontend_version = match.group(1)
    package = json.loads((FRONTEND / "package.json").read_text(encoding="utf-8"))
    toolkit_version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    assert frontend_version == package["version"] == toolkit_version, (
        "the diagnostics version drifted from the carriers: "
        f"{frontend_version} / {package['version']} / {toolkit_version}"
    )
