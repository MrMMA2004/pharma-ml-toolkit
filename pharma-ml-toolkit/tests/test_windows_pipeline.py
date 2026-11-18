"""part11_2's Windows release pipeline — the deliverable drift guards.

The addendum's §13 required deliverables exist as REAL files, and
these guards keep them honest (a rename/move/regression anywhere in
the pipeline turns this suite red before the claim drifts):

* ``.github/workflows/windows-release.yml`` — the Windows GitHub
  Actions release workflow exists, runs on a Windows runner, walks
  the documented 25-step order, pins action versions, carries
  least-privilege permissions, and never treats a build as a
  clean-machine verification (§12's honest labels);
* ``src-tauri/`` — the Tauri 2 host skeleton: the conf (offline
  WebView2 installer mode, NSIS + MSI targets, the loopback CSP),
  the least-privilege capability, and the Rust modules (the sidecar
  supervisor implementing the TESTED PHARMAML-READY handshake
  contract, the security boundary, the single IPC command);
* ``packaging/windows/pharmaml_backend.spec`` — the backend-sidecar
  PyInstaller spec: the backend entry (NOT the legacy studio.py GUI
  entry) and the Qt excludes (part11_2 §5: no PySide6 in the
  sidecar);
* ``packaging/windows/backend_entry.py`` + ``verify_sidecar.py`` —
  the sidecar entry + its live verification driver;
* the version carriers stay in lockstep (VERSION ==
  tauri.conf.json.version == frontend package.json.version) — the
  workflow's step 2 enforces this at build time; the guard enforces
  it between builds too.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
SRC_TAURI = ROOT / "src-tauri"
PACKAGING = ROOT / "packaging" / "windows"
FRONTEND = ROOT / "frontend"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_windows_release_workflow_exists_and_is_windows_only() -> None:
    workflow = WORKFLOWS / "windows-release.yml"
    assert workflow.is_file(), (
        "part11_2 §3's named deliverable is missing: "
        ".github/workflows/windows-release.yml"
    )
    src = _read(workflow)
    assert "runs-on: windows-2022" in src, (
        "the release workflow must run on a supported Windows x64 runner "
        "(part11_2 §3)"
    )
    # manual dispatch + tag builds (the two honest triggers)
    assert "workflow_dispatch:" in src and 'tags:' in src


def test_windows_release_workflow_walks_the_documented_order() -> None:
    src = _read(WORKFLOWS / "windows-release.yml")
    # §4's numbered stages in FILE order (Node 4 precedes restore 3
    # by design: the setup-node cache config must exist before npm ci
    # can use it — annotated in the workflow itself)
    ordered = [
        "1. Checkout the exact source revision",
        "2. Validate the version across every carrier",
        "4. Set up Node.js",
        "3. Restore frontend dependencies",
        "5. Set up Python",
        "6. Set up the Rust toolchain",
        "7. Install the pinned backend stack",
        "8. Frontend type check",
        "9. Frontend contract tests",
        "10. Backend unit tests",
        "11. API + database tests",
        "12. Freeze the backend sidecar",
        "13. Verify the frozen sidecar",
        "14. Stage the sidecar into the Tauri resources",
        "15. Build the frontend",
        "16-19. Build the Tauri app",
        "20. Validate the installer artifacts",
        "21. Generate the release manifest",
        "22. Generate SHA256SUMS.txt",
        "23. Upload the release artifacts",
    ]
    positions = [src.find(step) for step in ordered]
    assert all(pos >= 0 for pos in positions), (
        f"a required stage name is missing: {ordered[positions.index(-1)] if -1 in positions else '?'}"
    )
    assert positions == sorted(positions), (
        "the workflow stages must appear in the documented §4 order"
    )
    # §4's hard rules
    assert "permissions:" in src and "contents: write" in src
    assert "persist-credentials: false" in src, (
        "least-privilege: the checkout must not persist its token"
    )
    for pinned in ("actions/checkout@v4", "actions/setup-node@v4",
                   "actions/setup-python@v5", "actions/upload-artifact@v4",
                   "dtolnay/rust-toolchain@", "Swatinem/rust-cache@v2",
                   "softprops/action-gh-release@v2"):
        assert pinned in src, f"the action version must be pinned: {pinned}"
    # §10/§12: the workflow never claims clean-machine verification
    assert "BLOCKED: clean-Windows-10/11 installation test" in src, (
        "the closing note must label the clean-machine stages BLOCKED — "
        "a runner build is not a clean-machine verification"
    )


def test_tauri_conf_ships_offline_webview2_and_the_loopback_csp() -> None:
    conf = json.loads(_read(SRC_TAURI / "tauri.conf.json"))
    assert conf["version"] == _read(ROOT / "VERSION").strip()
    assert conf["build"]["frontendDist"] == "../frontend/dist", (
        "the production frontend is the Vite-built static assets, no "
        "remote URL (part11 §4.1)"
    )
    assert conf["build"].get("devUrl"), "the dev server is dev-only config"
    bundle = conf["bundle"]
    assert bundle["targets"] == ["nsis", "msi"]
    # §7: the offline WebView2 deployment — no online bootstrapper
    assert bundle["windows"]["webviewInstallMode"] == {
        "type": "offlineInstaller"
    }
    # §15: the CSP scopes connect-src to self + loopback only
    csp = conf["app"]["security"]["csp"]
    assert "connect-src" in csp and "http://127.0.0.1:*" in csp
    assert "ws://127.0.0.1:*" in csp
    assert "default-src 'self'" in csp
    assert "http://" not in csp.replace("http://127.0.0.1", "").replace(
        "http://ipc.localhost", ""
    ), "the CSP must not name any non-loopback http endpoint"


def test_tauri_capability_is_least_privilege() -> None:
    caps = json.loads(_read(SRC_TAURI / "capabilities" / "default.json"))
    assert caps["permissions"] == ["core:default"], (
        "the window's ONLY capability is the core set — no shell, no "
        "arbitrary fs, no dialog passthrough (part11 §9.3)"
    )


def test_the_rust_host_implements_the_tested_handshake_contract() -> None:
    manager = _read(SRC_TAURI / "src" / "backend_manager.rs")
    # the TESTED wire contract (tests/api/test_backend_service.py)
    assert 'PHARMAML-READY ' in manager and "port" in manager
    # §14.1: the token is NOT passed through argv — no arguments
    assert "NO arguments" in manager
    # §14.2: the bounded startup timeout + clear failure paths
    assert "STARTUP_TIMEOUT" in manager
    assert "TIMED OUT" in manager and "FAILED during startup" in manager
    # §14.3: graceful window then forced kill
    assert "GRACEFUL_WINDOW" in manager
    # the loopback-only assertion (§15) lives in security.rs
    security = _read(SRC_TAURI / "src" / "security.rs")
    assert "assert_loopback" in security and "127.0.0.1" in security
    # §9.1: the ONE IPC command
    commands = _read(SRC_TAURI / "src" / "commands.rs")
    assert "backend_endpoint" in commands
    assert "tauri::command" in commands
    # the entry wires the supervisor + the single command
    lib = _read(SRC_TAURI / "src" / "lib.rs")
    assert "spawn_backend" in lib and "backend_endpoint" in lib
    main = _read(SRC_TAURI / "src" / "main.rs")
    assert "pharmaml_studio_lib::run()" in main
    # the lib name matches Cargo.toml (main.rs must link)
    cargo = _read(SRC_TAURI / "Cargo.toml")
    assert 'name = "pharmaml_studio_lib"' in cargo


def test_the_backend_sidecar_spec_excludes_the_qt_stack() -> None:
    spec = _read(PACKAGING / "pharmaml_backend.spec")
    # part11_2 §5: the sidecar entry is the BACKEND, not the GUI
    assert "backend_entry.py" in spec
    assert 'TOOLKIT / "studio.py"' not in spec
    # the Qt stack never rides the sidecar
    for banned in ("PySide6", "pyqtgraph", "qt_material", "qdarkstyle"):
        assert f'"{banned}"' in spec.split("excludes = [", 1)[1], (
            f"the sidecar spec must exclude {banned}"
        )
    # ...while the scientific stack stays (§5: preserve Qt-independent
    # scientific dependencies — matplotlib's non-Qt export backends)
    assert "matplotlib.backends.backend_agg" in spec
    assert "backend_qtagg" not in spec.split("hiddenimports", 1)[1].split("datas", 1)[0]
    # the sidecar is a CONSOLE subprocess (stdout carries the handshake)
    assert "console=True" in spec


def test_the_sidecar_entry_and_verifier_exist() -> None:
    entry = _read(PACKAGING / "backend_entry.py")
    assert "freeze_support" in entry, (
        "the Windows spawn context needs freeze_support first (part9 §13)"
    )
    assert "from pharmaml.api.server import main" in entry
    verifier = _read(PACKAGING / "verify_sidecar.py")
    assert "PHARMAML-READY " in verifier
    assert "/api/v1/health" in verifier
    assert "SIDECAR-VERIFIED" in verifier
    # §4: never print the session token in build logs
    assert "redacted" in verifier


def test_the_version_carriers_stay_in_lockstep() -> None:
    version = _read(ROOT / "VERSION").strip()
    package = json.loads(_read(FRONTEND / "package.json"))
    assert package["version"] == version, (
        "frontend/package.json drifted from VERSION — the workflow's "
        "step 2 will refuse to build; re-align the carriers"
    )
    conf = json.loads(_read(SRC_TAURI / "tauri.conf.json"))
    assert conf["version"] == version
