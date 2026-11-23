# TEST EXECUTION REPORT — part15 §13 deliverable #5

The part15 Phase-Five honest execution record. Executed-this-round
(r340, at `0132d409`) vs carried-from-executed-rounds are stated
separately; BLOCKED legs name their missing prerequisite (§7 rule:
never a silent skip, never a blocked test marked passed).

## 1. Executed this round (r340 — the rollback-recovery + SS13 round)

| Lane | Command (class) | Result |
| --- | --- | --- |
| Canonical collection | `pytest --collect-only -q` | **8,601 tests / 0 errors / 472 files** |
| Quick regression chunk | `pytest tests/test_approvals.py tests/test_biology.py -q` | 108 passed, exit 0 |
| Lint | `ruff check src tests` | clean — after the 101-file EXE002 mode-bit sweep (the documented rollback exec-bit drift class, r300/r339 precedent) |
| Types | `mypy src` | Success: no issues found in 296 source files |
| Frontend unit | `npx vitest run` (nested frontend, post `npm ci` 210 pkgs) | **119/119 over 16 files**, exit 0 |
| Frontend types | `npx tsc --noEmit` | clean (silent, exit 0) |
| Documentation guards | `pytest tests/test_documentation_consistency.py -q` | 13/13 — incl. the **D-7 fix** (the ordinal-blind ledger guard, found failing this round, fixed, green) |
| Security supply chain | `pytest tests/test_security_supply_chain.py -q` | 8/8 |
| Console | `GET /` | 200, title pin |
| Zip-size API | `GET /api/toolkit/zip-size` | `{"zipBytes":23489390,"version":"1.2.240","exact":true,"source":"release"}` |
| Census | tree walk (offline replica of the route rules) | 1,744 files / 569,595 lines / v1.2.240 == the r339 live-walk pin |
| Zip integrity | six-home sha-256 + `testzip` | ALL OK `0fadb6e5…a63e2cb3` at 23,489,390 B (mirrors re-seeded after the rollback wipe, 18 zips each) |
| CI verdict at HEAD | authenticated check-runs poll | **19/19 success** at `0132d409` |
| ECC standing check | `scratch/ecc_check.sh` | GREEN (clone 4adeee67/2.2.3 + harness 14 modules/196 skills — re-installed after the rollback wipe) |

## 2. Carried from executed rounds (latest full-run evidence)

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
re-pinned lanes: D-6 (r339, 12-test suite, 6 red pre-fix) and D-7 (r340,
the guard's failing state as its own reproduction, minimal regex fix,
13/13 post). No assertion was weakened, no failing test deleted
(§7 rule).

— reported round 340 (the 132nd recorded run) at `0132d409`.
