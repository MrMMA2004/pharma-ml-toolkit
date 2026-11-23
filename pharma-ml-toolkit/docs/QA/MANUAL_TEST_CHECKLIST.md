# MANUAL TEST CHECKLIST — part14 §13 deliverable #2

The part14 §6 checklist instantiated as a first-class QA document.
Statuses are honest at the seeding round; every BLOCKED row names its
environment. Unexecuted rows are NOT marked PASS (§16.14).

Seeded: round 329 (2026-11-22). Environment of record for executed
legs: the Linux sandbox live web console (Chromium via agent-browser,
1440px + 390px) — the Windows legs ride the Windows box.

## 6.1 Tauri and desktop application

| # | Check | Status | Evidence / lane |
| --- | --- | --- | --- |
| 1 | `tauri dev` starts without errors | **BLOCKED** | needs the Windows dev box (no Tauri toolchain here) |
| 2 | Dev frontend + backend start order | **BLOCKED** | same |
| 3 | `tauri build` valid packaged app | **BLOCKED** | same |
| 4 | Packaged app reaches ready state | **BLOCKED** | the D-1/D-2 fixes (r327) remove the origin/CORS wall the packaged WebView2 hit; packaged verification needs the box |
| 5–7 | Icon (taskbar/EXE/installer), window title + metadata | **BLOCKED** | packaging legs |
| 8–10 | Minimize/restore keeps WS; suspension reconnect reliable; no duplicate subscriptions | **BLOCKED** | WS runtime legs; the WS token handshake is source-verified (close 4401, r327 §3.1-item-8 audit) |
| 11–13 | FS restricted; traversal rejected; symlink/junction | **SOURCE-VERIFIED** | `security.py` path policy + tests/api legs; runtime legs BLOCKED |
| 14–18 | Runs from Program Files / Desktop / Downloads / spaces+non-ASCII / user paths | **BLOCKED** | Windows-only |
| 19–24 | Clean X-button exit; no orphan backend after Task-Manager kill; backend-death recovery; repeated cycles; single-instance policy | **BLOCKED** | process-lifecycle legs (§6.1); the Rust side's `validate_handshake` + redaction are source-written (D-4/D-5) |
| 25–27 | Sleep/resume; display scaling; DPI | **BLOCKED** | Windows-only |

## 6.2 Frontend ↔ Backend communication (the API legs — EXECUTED)

| # | Check | Status | Evidence |
| --- | --- | --- | --- |
| 1 | REST endpoints return/handle expected types | **PASS** | `tests/api/` 449+1s green at r329; contract 109 paths byte-identical (r325 evidence) |
| 2 | HTTP 200 handled correctly | **PASS** | same |
| 3 | HTTP 400 handled correctly | **PASS** | validation lanes in the suite |
| 4 | HTTP 401 (token missing/invalid) | **PASS** | P5 leg + the pinned 401 lane (34-leg matrix) |
| 5 | HTTP 403 (origin/host rejected) | **PASS** | P1/P2/P7/P8 legs + lookalikes (r327/r329) |
| 6 | HTTP 404 | **PASS** | suite lanes |
| 7 | Error classification (the D-3 fix) | **PASS** | `startup.test.ts` 9 tests: auth/network/http/protocol/handshake causes, honest headlines; vitest 119/119 |
| 8 | Preflight CORS for trusted origins | **PASS** | P3/P4 legs: 405→200 + exact-origin ACAO (r327 probe + r329 re-pin) |

The remaining §6.2 legs (WS reconnection behavior on the live
desktop) ride the box; the token-auth handshake itself is
source-verified.

## 6.3–6.8 (datasets / ML / DL / generative / docking / PK-PD / MD)

The API surface of every workspace is under `tests/api/` and the
suite total above; the interactive GUI legs of each workspace ride the
desktop box (BLOCKED). The legacy PySide6 surface is retired-era
(Stage-5 lane) and is NOT a manual-test target (per the r325
release-lane ruling).

## Web-console legs executed this round (r329)

- Landing render 200; title `PharmaML Toolkit — ML · Pharma · Docking`.
- Zero page errors; zero console errors (agent-browser over the LIVE
  console).
- ML-card interaction: the registry/training surface renders (33-entry
  registry, `train_model()` / `compare_models()` copy present).
- Footer geometry: single semantic `<footer>`, `mt-auto` +
  safe-area-bottom padding, footer bottom == document height exactly
  (natural push-down, never overlapping) — measured 3967 == 3967.
- Pins: `docs/gui/screenshots/p14r329-pins-{desktop,mobile}.png`
  (1440px / 390px).

— seeded round 329; statuses flip only with named evidence.

---

# PART15 EXTENSION — the §6 (Phase Six) checklist, A–H

Instantiated round r340 (the 132nd recorded run) per part15 §6's own
five-status vocabulary (PASS / FAIL / BLOCKED / NOT TESTED / NOT
APPLICABLE). Environment of record for executed legs: the Linux sandbox
(web console + API lanes, Chromium via agent-browser 1440px/390px; the
API test suites). Windows legs ride the owner's box via the SS11 script
set — a PASS below is always tied to named evidence; the packaged-app
interactive rows stay BLOCKED until that box runs them.

## A. Tauri and desktop integration

| Row class | Status | Evidence / carrier |
| --- | --- | --- |
| App launch, no unwanted terminal, backend auto-start, handshake, IPC endpoint, secure token delivery, health probe, WS connect, retry, graceful termination, crash detection, restart correctness, no orphan processes, WebView2 console clean, reopen after shutdown | **BLOCKED** (interactive packaged legs) | needs the Windows box; the handshake/termination/redaction semantics are source-pinned (D-4/D-5 + 11 Rust unit cases); `test-tauri.ps1` / `test-rust.ps1` are the carriers |

## B. Backend communication — EXECUTED at the API level

| Check | Status | Evidence |
| --- | --- | --- |
| REST GET/POST/PUT/DELETE where supported | **PASS** | full `tests/api` suite exit 0 (r339); quick chunk 108/108 (r340) |
| JSON validation / invalid payloads | **PASS** | validation lanes (400-class) in the suite |
| Authentication / authorization | **PASS** | 401/403 lanes; the 34-leg trusted-origins matrix |
| CORS preflight / trusted Origin / rejected Origin / Host validation | **PASS** | P3/P4 preflight legs + lookalike refusals (r327/r329) |
| WebSocket authentication | **PASS** | close-4401 lanes (events suite) |
| **WebSocket Origin validation** | **PASS** | `tests/api/test_ws_origin.py` 12/12 (r339, FAIL-FIRST) — evil origin WITH valid token refused 4403 pre-accept |
| Large payload / upload-download | **PASS** | upload/download lanes in the suite; the release-zip streaming verified byte-exact |
| WS reconnect / event replay / backend restart during active work (live desktop) | **BLOCKED** | interactive legs; the REST-recovery design is source-pinned (WS never the only source of truth) |

## C. SQLite and FileVault

| Check | Status | Evidence |
| --- | --- | --- |
| DB creation/migration, dataset/model/job-history persistence, artifact registration + retrieval, restart recovery, integrity validation, corrupted-file handling, concurrent access, orphan cleanup | **PASS** (API-level, isolated temp DBs) | persistence suites in the api lane (never user data — §4.5 rule) |
| Backup creation/restoration | **PASS** (API-level lanes) | persistence suite |
| Disk-full handling | **BLOCKED** | environmental leg — needs a controlled low-disk box; carried in RELEASE_BLOCKERS |

## D. Plotly

| Check | Status | Evidence |
| --- | --- | --- |
| Line/bar/scatter/heatmap/confusion/ROC-PR/feature-importance/SHAP charts; hover; responsive resize; empty datasets; correct scientific labels | **PASS** (component-level) | `PlotlyFigure.test.tsx` + viz lanes in vitest 119/119 |
| Zoom/pan, image export incl. TIFF/JPEG where supported, large-dataset interactive performance | **BLOCKED** (interactive) | the packaged WebView2 legs ride the Windows box |

## E. Three.js

| Check | Status | Evidence |
| --- | --- | --- |
| 3D scene initialization + correct rendering (component surface) | **PASS** (component-level) | the md/visualization component suites |
| Mouse interaction, camera controls, window resize, WebGL error handling, resource disposal, repeated navigation without leaks, large scenes | **BLOCKED** (interactive) | Windows box; the manual §E rows are the carrier |

## F. UI and components

| Check | Status | Evidence |
| --- | --- | --- |
| Navigation/tabs/dropdowns/dialogs/tooltips/forms/tables/progress/notifications; loading/empty/error states; light + dark themes; keyboard navigation; accessibility | **PASS** (component + console surfaces) | atoms + lane suites; the browser-E2E pairs (r337–r339: zero errors, theme legs) |
| 1366×768 / 1920×1080 desktop viewports | **BLOCKED** (packaged app) / **PASS** (console surface, 1440×900 pins) | the console pins pairs; the packaged legs ride the box |
| 1024px width / 390×844 mobile viewport / no horizontal overflow | **PASS** (console surface) | the r338/r339 browser passes (390×844, zero overflow, footer at page end) |

## G. Stability and performance

| Check | Status | Evidence |
| --- | --- | --- |
| Long-running training / multiple concurrent jobs / job cancellation (API lanes) | **PASS** | job-lifecycle lanes; the workflow cooperative-cancel tests (r322) |
| Resource visibility (CPU/memory/queue depth) | **PASS** | the `/api/v1/system/resources` read-only lane (r324) + its 9-test suite |
| Repeated workspace switching / large CSV import / large molecular datasets / memory-CPU-GPU profiles on the packaged app | **BLOCKED** | Windows box legs; PERFORMANCE_BASELINES.md carries the measured baselines |

## H. Windows release

| Check | Status | Evidence |
| --- | --- | --- |
| NSIS + MSI installation on Win10/11, non-admin install, offline install + WebView2, sidecar + Python deps + RDKit included, correct permissions, user-data dir, upgrade, uninstall, user-data preservation, no unexpected firewall/network dependency, clean-machine startup | **VERIFIED at release** | the ninth smoke dispatch 45086526069 GREEN end-to-end (all 16 steps incl. the installed sidecar's live contract, the cross-scope upgrade, zero orphans, uninstalls) — the standing lane re-runs on every published release |
| Antivirus false-positive investigation | **NOT TESTED** | Defender clean on both installers in the smoke lane; third-party AV panels ride the owner's environment |
| Installer signing | **BLOCKED** | certificate-gated (RELEASE_BLOCKERS) |
| Owner-side manual re-runs of the installer matrix | **NOT TESTED** | the SS11 `test-installers.ps1` digest-verification script + `WINDOWS_VALIDATION_GUIDE.md` are the carriers |

— part15 layer added round 340; statuses flip only with named evidence
(the part14-era sections above stand as the historical record).

