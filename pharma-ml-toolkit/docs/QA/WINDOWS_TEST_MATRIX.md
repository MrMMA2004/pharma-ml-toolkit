# WINDOWS TEST MATRIX — part14 §13 deliverable #4

The part14 §9 environment inventory and blocked-test matrix for the
Windows-side legs. Seeded round 329 (2026-11-22).

## 9.1 Environment inventory (actual, honest)

| Environment | Reality | Impact |
| --- | --- | --- |
| Development environment of record | Linux sandbox (z container; Python 3.12.16 venv; bun/node; NO cargo/rustc; NO Windows; NO WebView2) | every Windows-only leg below is BLOCKED here — never PASS |
| Target environment (per the part14 §9 claims) | Windows 10/11 physical box, WebView2 runtime, physical GPU | the legs run there, or per §9.5 in a clean VM (acceptable for most legs) |
| CI (GitHub Actions) | live — 19 jobs incl. the windows-release pipeline (r325: 9/9) | Windows pipeline compilation/bundling verified in CI; run-time behavior still needs the box |

## 9.2 Blocked tests (each with the lowest-cost solution)

| Test | Why blocked | Unlock |
| --- | --- | --- |
| Clean-machine Win10/11 smoke (install → launch → ready → nine workspaces) | no Windows in sandbox | §9.5 clean-VM procedure; VM acceptable per §9.5 |
| NSIS/MSI installers: install/upgrade/uninstall + cleanup verification | same | same |
| Offline WebView2 boot | same | same |
| Signed-EXE validation | signing cert lane (r325 BLOCKED lane #3) | certificate availability |
| `tauri dev` / `tauri build` dev cycle | no Rust toolchain in sandbox | the Windows dev box (or any cargo-enabled env for the unit legs) |
| Rust host unit tests (D-4/D-5 11 cases) | no cargo/rustc | ANY cargo-enabled environment (lowest-cost unlock — not Windows-specific) |
| Process lifecycle (X-button, Task-Manager kill, orphans, single-instance) | needs the real app | Windows box |
| Sleep/resume, scaling/DPI, multi-monitor | needs real Windows session | Windows box |
| 45–60-min stability + memory/CPU/GPU under scientific workloads | needs real machine + GPU | Windows box (§9.5) |
| Antivirus compatibility | needs real Windows + AV products | Windows box |
| Stage-5 PySide6 retirement gate | the r325 standing lane | owner approval + the box |

## 9.3/9.4 Tooling and PowerShell automation (proposed, not yet executed)

- `ps1` driver scripts for the smoke matrix (launch → probe → assert
  → collect) — QUEUED for the box; written when the environment
  exists, never before (§16.15).
- Playwright Windows-side for the packaged WebView2 — QUEUED (the
  Linux agent-browser legs cover the web-console surface only).

## 9.5 Clean Windows testing strategy (the procedure, ready to execute on the box)

1. Clean Win10/11 VM (snapshot taken pre-install).
2. Install via the release MSI/NSIS artifact (the v1.2.231 release
   assets stand; source/installer parity at asset level verified r325).
3. First-boot → ready-state timing + WebView2 offline behavior.
4. The §6.1 desktop checklist (FS, traversal, lifecycle,
   single-instance, sleep/resume, DPI).
5. The §10 performance legs on the physical box (VM numbers never
   become universal guarantees — §10 rule).
6. Results land in this matrix with evidence; nothing pre-filled.

— seeded round 329; every row flips status only with named evidence
from the actual environment.
