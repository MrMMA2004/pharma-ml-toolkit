# TEST EXECUTION REPORT — part15 §13 deliverable #5

The part15 Phase-Five honest execution record. Executed-this-round
(r342, at the round terminal) vs carried-from-executed-rounds are stated
separately; BLOCKED legs name their missing prerequisite (§7 rule:
never a silent skip, never a blocked test marked passed).

## 1. Executed this round (r342 — the roadmap #3/#4 close-out round)

| Lane | Command (class) | Result |
| --- | --- | --- |
| Canonical collection | `pytest --collect-only -q` | **8,602 tests / 0 errors / 472 files** (unchanged from r341 — no test-count change this round; measured fresh in 33.236 s) |
| Quick regression chunk | `pytest tests/test_approvals.py tests/test_biology.py -q` | **108 passed, exit 0** (the 72+36 dot count verified exactly) |
| Lint | `ruff check .` (FULL tree) + `ruff check src/pharmaml` (the CI scope) | **both "All checks passed!" exit 0** — after the round's own find: 15 permission-bit realignments (the r340 class) + 7 pre-existing code findings in two out-of-CI-scope dev scripts fixed minimally (no semantic change, py_compile both) |
| Root console lint | `bun run lint` (eslint, the touched root project) | **clean, true exit 0** — the first CJS cut was rejected (3 no-require-imports errors, the fail-first signal) and re-landed as ESM `.mjs` |
| Types | `mypy src` | Success: no issues found in 296 source files |
| Security supply chain | `pytest tests/test_security_supply_chain.py -q` | **8/8 exit 0** |
| API regression suite (measurement lane) | `pytest tests/api -q` | **461 passed + 1 skip = 462 collected, exit 0, 117.495 s wall** (was 449+1 at r331 — the tree +13 tests, the r339/r341 WS lanes) |
| Trusted-origin matrix (measurement lane) | `pytest tests/api/test_trusted_origins.py -q` | **34/34, exit 0, 10.955 s** (the r331 pin 5.83 s — box load honestly named) |
| Console | `GET /` | **200, title pin** — the reaper struck between every window; the per-call node_modules dance ran at least five times (once mid-round between an install and its own lint) |
| Zip-size API (measurement lane) | `GET /api/toolkit/zip-size` (timed) | `{"zipBytes":23489390,"version":"1.2.240","exact":true,"source":"release"}` at **0.308 s total** |
| Backend perf lanes (the r324 instruments) | `scratch/p15_r342_backend_perf.py` (spawn → PHARMAML-READY → health → resources → SIGTERM) | **startup 4.138 s; health first 82.5 ms / warm 0.9 ms; idle RSS 228.4 MiB +0.00 drift / 6 threads; graceful shutdown 0.665 s exit 0** — sandbox-venv context honestly named |
| PowerShell-safe dev scripts (roadmap #3) | executed fake-target verification (`scratch/p15_r342_portable_scripts.txt`) | copy happy-path + missing-source refusal exit 1; launcher NODE_ENV pin + 7/0 exit passthrough + missing-target surfacing; `node --check` both |
| Census (live) | the console's own walk | **1,752 files / 570,378 LOC / 8,555 tests** — files == the r341 pin (the counted delta +70: the QA doc rewrites + the two dev-script edits; the .tsx lanes never count) |
| Browser verification | agent-browser desktop + 390×844 mobile | **zero page errors, zero console errors**, footer at page end both viewports (**3967 desktop / 7882 mobile**), mobile zero horizontal overflow; evidence `scratch/r342-web-verify.png` + `scratch/r342-mobile-verify.png` |
| CI verdict at HEAD | authenticated check-runs poll | **19/19 success** at `9ff53b63` (the r341 evidence terminal, closed on arrival pre-flight) |

## 2. Carried from executed rounds (latest full-run evidence)

- **r341** (the D-6 frontend-lane round): vitest **126/126 over 16 files**
  + tsc clean over the D-6/D-8 payload (unchanged-carried this round —
  no frontend source changed in r342); the collection 8,602/0/472;
  the frontend-services drift guards 17/17; the documentation-consistency
  12/12 post-ledger.

- **r340** (the part15 continuation round): the rollback recovery in
  full — venv/node_modules/mirrors/ECC rebuilt at the exact pins; the
  canonical collection 8,601/0/472 (the r339 tree + the D-7 guard
  fix); vitest 119/119 over 16 files; the 101-file EXE002 sweep; the
  offline census replica 1,744/569,595 == the r339 pin; CI 19/19 at
  `0132d409`; the SS13 QA doc set delivered.
- **r339** (the part15 opening round): the FULL `tests/api` suite exit 0;
  `tests/api/test_ws_origin.py` **12/12** (FAIL-FIRST: 6 red witnessed
  against the un-edited source); ruff + mypy + vitest + tsc clean over
  the D-6 payload; the frontend-services suite green.
- **r338/r337** (part14 close-out): the 108-test quick chunk exit 0;
  CI 19/19 at the terminals; the browser verification passes (desktop +
  390px mobile, zero page/console errors, footer geometry 3967==3967).
- **r336** (the release verdict): the ninth smoke dispatch
  `45086526069` green end-to-end — all 16 steps against the published
  v1.2.240 assets (install/tree/version, the installed sidecar's live
  contract, upgrade/uninstall, zero orphans); CI 20/20 at the round
  terminal (the standard 19 + the green dispatch).
- **r325** (the last full ten-chunk gate): the full suite over the
  shipped tree — chunks 1–9 green (the r275 OOM-split drill on chunk-1's
  heavy single, honestly recorded), doctest 107s, api 129s; collection
  8,555/0/470 at the v1.2.240 freeze.

## 3. Standard skips and blocked legs (honest register)

| Leg | Status | Prerequisite / reason |
| --- | --- | --- |
| `cargo fmt --check` / `cargo clippy` / `cargo test` (D-4/D-5) | **BLOCKED** | no Rust toolchain in this sandbox — carrier: `scripts/windows/test-rust.ps1`; converts to plain FIXED on the owner's first green run |
| The nine SS11 PowerShell scripts | **BLOCKED (execution)** | no PowerShell in this sandbox — source-delivered, PS 5.1-compatible, ride the owner's Windows box |
| Clean-machine Windows E/F gates (install/uninstall/upgrade smoke on Win10/11) | **VERIFIED at release** (the smoke lane) / **BLOCKED (owner-side re-runs)** | the standing lane auto-runs on published releases; owner dispatch at will |
| Desktop-app interactive E2E (nine workspaces in the packaged WebView2) | **BLOCKED** | needs the Windows box; the web-console browser-E2E is the executed proxy surface |
| vitest/tsc when the reaper strikes between windows | transient, restored per-call | the sandbox node_modules wipe class — `npm ci`/`bun install` inside the same tool-call window (documented in the worklog r340) |
| The 2 standard suite skips | standing, documented | the suite's own parametrized-skip lanes (environment-conditional), recorded by pytest at run time |

## 4. Regression requirements compliance (§7)

Every confirmed bug found in this campaign window received its
reproducible failing test, root cause, minimal fix, regression test and
re-pinned lanes: D-6 (r339, 12-test suite, 6 red pre-fix), D-7 (r340,
the guard's failing state as its own reproduction, minimal regex fix,
13/13 post) and D-8 (r341, the reconnect-storm client lane: 6 red
witnessed against the un-edited client — the 4403/4401 closes ran the
full backoff ladder with no surfaced cause; the minimal close-code
terminal fix, 126/126 post). No assertion was weakened, no failing test
deleted (§7 rule).

— reported round 342 (the 134th recorded run).
