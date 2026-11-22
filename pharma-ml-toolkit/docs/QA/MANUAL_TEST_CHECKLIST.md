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
