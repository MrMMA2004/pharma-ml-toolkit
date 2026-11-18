# The part11_2 Windows release pipeline — build documentation

**Status date**: round 281 (v1.2.224). **Master prompt part11_2** — the
Linux-host addendum: the Linux development environment must not
prevent the Windows release pipeline from existing, being validated,
and being honestly reported. This document is the §13 deliverable
("Build documentation") and carries the §12 status labels.

## The pipeline (three layers, one source repository)

| Layer | Artifact | Built by | Where it runs |
| --- | --- | --- | --- |
| Frontend | `frontend/dist` static SPA (React + TS + Vite) | `npm run build` | the GitHub Actions Windows runner (step 15) |
| Backend sidecar | `pharmaml-backend` onedir (PyInstaller 6.22.3, `pharmaml_backend.spec`) | `python -m PyInstaller` | the Windows runner (step 12) — **PyInstaller is not a cross-compiler** (part11_2 §5): the Linux host never fakes it |
| Desktop host | Tauri 2 + WebView2 (NSIS EXE + WiX MSI) | `cargo tauri build` | the Windows runner (steps 16–19) |

The workflow: **`.github/workflows/windows-release.yml`** — the §3
named deliverable. `workflow_dispatch` (manual) and `v*` tags trigger
it; it runs on `windows-2022`, validates the version carriers first
(VERSION == tauri.conf.json == frontend/package.json), restores the
frontend deps from the committed lockfile, runs the full test ladder
(typecheck, the frontend drift guards, backend unit tests, the
api/database suites), freezes + **live-verifies** the sidecar
(`verify_sidecar.py`: the `PHARMAML-READY` handshake, the
authenticated health probe, the unauthenticated-refused guard, the
graceful shutdown), stages the onedir into `src-tauri/resources/`,
builds the frontend and the Tauri app, and emits
`release/windows/{PharmaML_Studio_Setup_vX.Y.Z.exe, PharmaML_Studio_vX.Y.Z.msi,
SHA256SUMS.txt, RELEASE_MANIFEST.json}`.

Actions are pinned (`checkout@v4`, `setup-node@v4`, `setup-python@v5`,
`upload-artifact@v4`, `rust-cache@v2`, `action-gh-release@v2`), the
checkout does not persist credentials, and the only credential in
play is the built-in `GITHUB_TOKEN` (contents: write, required solely
by the tagged release upload — never echoed, §4's rule).

## The deliverables (part11_2 §13) — where each lives

| §13 deliverable | File | Status |
| --- | --- | --- |
| `.github/workflows/windows-release.yml` | `.github/workflows/windows-release.yml` | **PARTIAL** — exists, YAML-validated, all 23 stages named in the documented order; NOT yet executed on a real runner (no Windows host in this environment) |
| Windows backend build scripts | `packaging/windows/pharmaml_backend.spec` + `backend_entry.py` + `build_windows.ps1` (the part9 pipeline, retained for the legacy tier) | **PARTIAL** — the sidecar spec/entry are new and guarded; first real freeze happens on the runner |
| Tauri Windows build configuration | `src-tauri/{tauri.conf.json, Cargo.toml, capabilities/default.json}` | **PARTIAL** — the conf (offline WebView2, NSIS+MSI targets, loopback CSP) + the least-privilege capability are validated JSON; the Rust host compiles on the runner for the first time |
| PyInstaller Windows packaging config | `packaging/windows/pharmaml_backend.spec` | **PARTIAL** — excludes the Qt stack (§5's no-PySide6 rule), keeps the scientific stack; spec-validated by `tests/test_windows_pipeline.py` |
| WebView2 offline installation config | `tauri.conf.json` → `bundle.windows.webviewInstallMode = offlineInstaller` | **PARTIAL** — the offline installer mode is the selected strategy (§7); the runner downloads the WebView2 offline installer at BUILD time and embeds it — the end user never sees the online bootstrapper. The runtime update policy: Evergreen offline installer → the runtime self-updates once the machine is online; documented in the release manifest |
| Installer validation scripts | `packaging/windows/packaging_validation.py` (`--static`, any OS) + `verify_sidecar.py` (live, the runner's step 13) | `--static`: **PASS** (the Linux gate runs it green); the live sidecar probe: **PARTIAL** (first run happens on the runner) |
| Clean-machine smoke-test instructions | this document, section below | **PARTIAL** — the instructions exist; the test itself is **BLOCKED** (no Windows host / Windows Sandbox here) |
| Release manifest generation | workflow step 21 (`RELEASE_MANIFEST.json`) | **PARTIAL** — the generator is in the workflow; the artifact materializes on the first runner build |
| Checksum generation | workflow step 22 (`SHA256SUMS.txt`) | **PARTIAL** — same |
| Build documentation | this document | **PASS** (this file) |

The drift guards: **`tests/test_windows_pipeline.py`** (8 tests) —
they pin the workflow's stage order and action pins, the
offline-WebView2 + loopback-CSP conf, the least-privilege capability,
the Rust host's handshake contract, the sidecar spec's Qt excludes,
and the version-carrier lockstep. A regression in any of part11_2's
deliverables turns the suite red.

## The honest labels (part11_2 §12)

* **PASS** — executed and verified here: the YAML/JSON validation,
  the guard suite, the `--static` packaging validation, and the
  backend's own tested handshake contract the Rust supervisor
  implements (`tests/api/test_backend_service.py`).
* **PARTIAL** — implemented, validated as far as this Linux host
  allows, but not yet executed on a real Windows runner: the
  workflow file, the Tauri host skeleton, the sidecar spec, the
  live sidecar verifier, the manifest/checksum generators.
* **BLOCKED** — requires a Windows environment this host does not
  have: executing the workflow on a Windows runner, the PyInstaller
  Windows freeze, the Tauri/NSIS/MSI/WiX build, the clean-Windows
  installation test, the offline runtime flows, and the
  upgrade/uninstall behavior. **No Windows build is reported as
  successful merely because this workflow file exists.**

## The clean-machine smoke test (§10 — run on a REAL clean Windows box)

Prerequisites: Windows 10/11 x64 VM or Windows Sandbox, **no Python,
no Anaconda/Miniconda, no pip packages, no Node.js, no Qt, no RDKit**
installed; the network disconnected after downloading the installer.

1. **Install** — run `PharmaML_Studio_Setup_vX.Y.Z.exe` (the
   WebView2 offline installer rides the bundle; the machine must
   never hit the network for it). Verify the Start-Menu and optional
   Desktop shortcuts, and the installed tree under
   `%LOCALAPPDATA%\Programs\PharmaML Studio` (currentUser mode).
2. **Launch offline** — from the Start Menu, then from the shortcut,
   then by double-clicking the installed exe: the window opens with
   the Introduction workspace; the backend sidecar handshake completes
   (startup is tens of seconds the first time — the frozen torch/
   rdkit imports).
3. **The nine workspaces** — navigate all nine in order; the
   Datasets workspace's §11.1 chain: import a CSV, inspect the
   paginated preview, run the profile, pull a SMILES column,
   compute Morgan descriptors and a 512-bit fingerprint (a real
   background job with live progress + cancel), run the stratified
   split, generate a histogram stat, and export the CSV artifact
   (the digest-verified download prints `digest verified: …`).
4. **Persistence** — close and relaunch: the SQLite database and the
   artifact vault survive (the user-data dir is never the install
   dir).
5. **Graceful shutdown** — close the app while a job runs: the
   shutdown envelope reaches the WebSocket before close; no orphaned
   `pharmaml-backend.exe` remains (check the Task Manager).
6. **Uninstall** — through Windows Settings: the app leaves, the
   user data (datasets/SQLite) stays unless the user opts in to its
   removal (the documented retention rule).

Record every result with the PASS/PARTIAL/FAIL label and the exact
artifact SHA-256 from `SHA256SUMS.txt`; a stage that cannot run gets
BLOCKED with the reason — never a silent pass.
