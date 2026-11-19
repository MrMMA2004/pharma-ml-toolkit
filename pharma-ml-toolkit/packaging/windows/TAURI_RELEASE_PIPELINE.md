# The part11_2 Windows release pipeline — build documentation

**Status**: v1.2.230's Windows workflow completed on a hosted runner:
the EXE/MSI and verification files were generated and attached to a
**draft** GitHub release. The clean-machine installation ritual below
has not been completed. **Master prompt part11_2** — the Linux-host
addendum: the Linux development environment must not prevent the Windows
release pipeline from existing, being validated, and being honestly
reported. This document is the §13 deliverable ("Build documentation")
and carries the §12 status labels.

## The pipeline (three layers, one source repository)

| Layer | Artifact | Built by | Where it runs |
| --- | --- | --- | --- |
| Frontend | `frontend/dist` static SPA (React + TS + Vite) | `npm run build` | the GitHub Actions Windows runner (step 15) |
| Backend sidecar | `pharmaml-backend` onedir (PyInstaller 6.22.3, `pharmaml_backend.spec`) | `python -m PyInstaller` | the Windows runner (step 12) — **PyInstaller is not a cross-compiler** (part11_2 §5): the Linux host never fakes it |
| Desktop host | Tauri 2 + WebView2 (NSIS EXE + WiX MSI) | `cargo tauri build` | the Windows runner (steps 16–19) |

The workflow: **`.github/workflows/windows-release.yml` — at the
REPOSITORY root** (round 293's registration fix: the live repository's
checkout nests the toolkit under `pharma-ml-toolkit/`, and GitHub
Actions registers ONLY root-level workflow files — the copy
previously committed at `pharma-ml-toolkit/.github/workflows/` was
invisible to Actions on this repository, verified against the live
repo's `/actions/workflows` API, which listed no such workflow). The
job-level `defaults: run: working-directory: pharma-ml-toolkit` keeps
every `run:` step's toolkit-relative paths (VERSION, frontend/,
src-tauri/, packaging/windows/) exactly as authored, while every
`uses:` step that resolves paths against the workspace root
(setup-node's cache path, rust-cache's workspaces, upload-artifact,
action-gh-release) carries the explicit `pharma-ml-toolkit/` prefix —
the §3 named deliverable. `workflow_dispatch` (manual) and `v*` tags trigger
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
| `.github/workflows/windows-release.yml` | `.github/workflows/windows-release.yml` (repository root) | **PASS — build path executed** on a hosted Windows runner for v1.2.230; all 25 steps completed. The generated release record is currently a draft, not a public release. |
| Windows backend build scripts | `packaging/windows/pharmaml_backend.spec` + `backend_entry.py` + `build_windows.ps1` | **PASS — runner evidence**: the backend sidecar froze and passed its live handshake, health, auth-guard and shutdown probe. |
| Tauri Windows build configuration | `src-tauri/{tauri.conf.json, Cargo.toml, capabilities/default.json}` | **PASS — build evidence**: the Tauri host generated NSIS EXE and WiX MSI packages with offline-WebView2 configuration. Clean-machine launch is still unverified. |
| PyInstaller Windows packaging config | `packaging/windows/pharmaml_backend.spec` | **PASS — runner evidence**: the spec excludes the Qt stack and the frozen sidecar passed its probe. |
| WebView2 offline installation config | `tauri.conf.json` → `bundle.windows.webviewInstallMode = offlineInstaller` | **PASS — package configuration/build**: the offline installer is embedded in the generated EXE. Actual installation and runtime startup on a clean offline machine remain unverified. |
| Installer validation scripts | `packaging/windows/packaging_validation.py` (`--static`, any OS) + `verify_sidecar.py` (live, workflow step 13) | **PASS**: static checks, frozen-sidecar probe and runner-side artifact validation completed. This does not replace the clean-machine test. |
| Clean-machine smoke-test instructions | this document, section below | Instructions are present; execution is **BLOCKED / pending** on a genuinely clean Windows 10/11 machine. |
| Release manifest generation | workflow step 21 (`RELEASE_MANIFEST.json`) | **PASS — generated** and attached to the draft release. |
| Checksum generation | workflow step 22 (`SHA256SUMS.txt`) | **PASS — generated** and attached to the draft release. |
| Build documentation | this document | **PASS** (this file) |

The drift guards: **`tests/test_windows_pipeline.py`** (9 tests) —
they pin the workflow's stage order and action pins, its REPOSITORY-root
registration + the working-directory/prefix topology (the round-293
guard), the
offline-WebView2 + loopback-CSP conf, the least-privilege capability,
the Rust host's handshake contract, the sidecar spec's Qt excludes,
and the version-carrier lockstep. A regression in any of part11_2's
deliverables turns the suite red.

**v1.2.230 execution evidence (part11_2 §12):** the Windows Release
workflow completed on its hosted runner. It froze and live-probed the
backend sidecar, built the Vite frontend and Tauri host, generated the
NSIS EXE and WiX MSI, validated the artifacts, generated
`RELEASE_MANIFEST.json` and `SHA256SUMS.txt`, and attached them to the
v1.2.230 GitHub release record. GitHub currently marks that release
record **draft**; the latest public release is still older. The hosted
runner proves the build and package checks, not the clean-machine
ritual. The remaining required evidence is the offline install/start,
upgrade/data-retention and uninstall flow on a genuinely clean Windows
10/11 machine.

## The honest labels (part11_2 §12)

* **PASS** — executed and verified on the Windows runner: backend
  freeze + live sidecar probe, Vite build, Tauri/NSIS/MSI build,
  artifact validation, manifest/checksum generation and upload to the
  draft release. The local YAML/JSON, drift-guard and `--static`
  packaging checks also pass.
* **PARTIAL** — the binaries exist and the build pipeline succeeded,
  but the release record is still draft and no clean-machine
  install/runtime/upgrade test has been recorded.
* **BLOCKED / pending** — clean Windows 10/11 offline installation,
  first launch and WebView2 startup, persistence across upgrade, and
  uninstall/data-retention behavior. A hosted build runner is not the
  required clean-machine test; do not report these stages as passed.

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
