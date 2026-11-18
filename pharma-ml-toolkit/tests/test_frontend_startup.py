"""part11 §14.1's startup bridge — the Phase Sixteen fix, drift-guarded.

The backend–GUI integration audit (``docs/CAPABILITY_INTEGRATION_MATRIX.md``,
part11_1 §21) found the browser app never wired its own boot sequence:
``configureBackend`` and ``stream.connect`` had no caller inside
``frontend/src`` — only the Node e2e harnesses (``scratch/e2e280+``)
called them, so a real browser session could never reach its own
backend without an injected shim.  The audit filed it as the one FAIL
row; this round closes it:

* ``services/api/startup.ts`` — the §14.1 steps 9–12 sequence:
  acquire the handshake endpoint (the Tauri ``backend_endpoint`` IPC
  bridge in production; a dev-only same-origin fallback through the
  Vite proxy), wire the typed REST client, verify ``/api/v1/health``
  BEFORE opening the event stream, then ``getEventStream().connect()``;
* ``app/StartupGate.tsx`` — the shell mounts the workspaces only in
  the ready state; the failed state renders the typed stage verbatim
  with a retry (§14.2's controlled restart); the connection banner is
  the first ``useEventStreamState`` consumer (the reconnect notice);
* ``vite.config.ts`` — the dev proxy now carries the WS lane
  (``ws: true``) so the dev fallback's event stream rides the same
  same-origin path;
* ``package.json`` — ``@tauri-apps/api`` (the official, offline-bundled
  ``invoke`` wrapper — no filesystem, no shell passthrough).

These guards keep that wiring honest: a rename, reorder, or a
production leak of the dev fallback turns this suite red before the
§14.1 claim drifts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
SRC = FRONTEND / "src"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_startup_module_wires_the_14_1_sequence() -> None:
    """acquire → configureBackend → health verify → stream connect — in that order."""
    src = _read(SRC / "services" / "api" / "startup.ts")

    # the Tauri IPC bridge is the production channel (§14.1 steps 4–8)
    assert 'invoke<' in src and '"backend_endpoint"' in src, (
        "the startup bridge must acquire the handshake through the Tauri "
        "backend_endpoint IPC command (src-tauri/src/commands.rs)"
    )
    # the Rust BackendEndpoint serializes snake_case — the bridge maps it
    assert "raw.base_url" in src and "base_url" in src, (
        "the Tauri handshake arrives snake_case (base_url/token/version) — "
        "the bridge must map it into the client's camelCase contract"
    )

    # §14.1 step 10: the typed client is wired with the handshake endpoint
    assert "configureBackend(" in src, "the bridge must call configureBackend"

    # §14.1 step 10: health is verified BEFORE the stream opens — the
    # /api/v1/health probe must precede the connect() call in the module
    health_pos = src.find('"/api/v1/health"')
    connect_pos = src.find("getEventStream().connect()")
    assert health_pos >= 0 and connect_pos >= 0, (
        "the startup bridge must probe /api/v1/health and open the event "
        "stream (getEventStream().connect())"
    )
    assert health_pos < connect_pos, (
        "§14.1 order violated: the health probe must run BEFORE the "
        "WebSocket stream opens (workspaces unlock after verified health)"
    )
    assert 'health.status !== "ok"' in src, (
        "the health verdict must be checked, not just requested"
    )

    # the failure modes are typed and staged (§14.2's clear messages)
    assert "StartupError" in src and '"acquire"' in src and '"health"' in src and '"stream"' in src, (
        "startup failures must carry their failing stage"
    )


def test_dev_fallback_is_dev_only() -> None:
    """The same-origin dev fallback never leaks into production builds."""
    src = _read(SRC / "services" / "api" / "startup.ts")
    assert "import.meta.env.DEV" in src, (
        "the dev fallback must be guarded by import.meta.env.DEV"
    )
    assert "VITE_PHARMAML_TOKEN" in src, (
        "the dev token rides the VITE_PHARMAML_TOKEN env channel"
    )
    # the production branch throws when neither Tauri nor DEV is present
    assert "no startup channel" in src, (
        "outside Tauri and outside dev builds, the bridge must fail loudly "
        "instead of silently fabricating an endpoint"
    )


def test_startup_gate_renders_honest_states() -> None:
    """connecting / failed-with-retry / ready — never a silent fake."""
    src = _read(SRC / "app" / "StartupGate.tsx")

    assert '"connecting"' in src and '"failed"' in src and '"ready"' in src, (
        "the gate must render all three honest startup states"
    )
    assert "initializeBackend" in src, "the gate runs the §14.1 sequence"
    # §14.2: the failed state surfaces the stage + message and offers a retry
    assert "failure.stage" in src and "failure.message" in src, (
        "the failed card must show the typed StartupError stage and message"
    )
    assert "Retry the startup handshake" in src, (
        "the failed card must offer the controlled retry"
    )
    # StrictMode double-mount must not double-run the boot
    assert "booted.current" in src or "useRef(false)" in src, (
        "the gate must guard the boot sequence against StrictMode double mounts"
    )
    # the connection banner is the first useEventStreamState consumer
    assert "useEventStreamState" in src, (
        "the ready state must consume useEventStreamState (the WS degrade banner)"
    )


def test_app_boots_through_the_gate() -> None:
    """The shell mounts inside the gate — no page mounts pre-handshake."""
    src = _read(SRC / "app" / "App.tsx")
    assert "StartupGate" in src and "<StartupGate>" in src, (
        "App must mount its shell inside the StartupGate"
    )
    # the gate wraps the whole shell (header included)
    open_pos = src.find("<StartupGate>")
    header_pos = src.find("<header")
    close_pos = src.find("</StartupGate>")
    assert 0 <= open_pos < header_pos < close_pos, (
        "the gate must wrap the entire shell (header + nav + main), not just a fragment"
    )


def test_dev_proxy_carries_the_ws_lane() -> None:
    """The dev proxy forwards the WS lane so the fallback's stream opens."""
    src = _read(FRONTEND / "vite.config.ts")
    assert "PHARMAML_DEV_BACKEND_PORT" in src, (
        "the dev proxy stays keyed on PHARMAML_DEV_BACKEND_PORT"
    )
    assert re.search(r"ws:\s*true", src), (
        "the /api proxy must forward the WebSocket lane (ws: true) — the "
        "dev fallback opens the event stream through this same-origin path"
    )
    assert "import.meta.env" not in src, (
        "vite.config.ts runs in Node — it must read process.env only"
    )


def test_tauri_api_dependency_is_pinned() -> None:
    """The official invoke wrapper is a runtime dependency (offline-bundled)."""
    pkg = json.loads(_read(FRONTEND / "package.json"))
    deps = pkg.get("dependencies", {})
    assert "@tauri-apps/api" in deps, (
        "the startup bridge's invoke must come from the official "
        "@tauri-apps/api package (bundled offline — never a CDN)"
    )


def test_vite_env_types_declared() -> None:
    """The dev token channel is typed for the strict tsc gate."""
    env = _read(SRC / "vite-env.d.ts")
    assert "vite/client" in env, "the vite/client types must be referenced"
    assert "VITE_PHARMAML_TOKEN" in env, (
        "VITE_PHARMAML_TOKEN must be declared on ImportMetaEnv"
    )


def test_commands_contract_still_matches_the_bridge() -> None:
    """The Rust command the bridge invokes still exists with the same name."""
    commands = _read(ROOT / "src-tauri" / "src" / "commands.rs")
    assert 'backend_endpoint' in commands and 'invoke("backend_endpoint")' in commands.replace(
        '`invoke("backend_endpoint")`', 'invoke("backend_endpoint")'
    ), "the Tauri backend_endpoint IPC command must keep its documented name"
    manager = _read(ROOT / "src-tauri" / "src" / "backend_manager.rs")
    assert "pub base_url" in manager and "pub token" in manager, (
        "the Rust BackendEndpoint must keep the snake_case fields the bridge maps"
    )
