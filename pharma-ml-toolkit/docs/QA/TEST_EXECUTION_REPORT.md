# TEST EXECUTION REPORT — part15 §13 deliverable #5 (the part16 era rides it forward)

The part15 Phase-Five honest execution record, carried forward as the
standing executed-vs-carried ledger. Executed-this-round (r344, the
part16 opening round over the seventh-rollback-recovered `8940bd42`
tree) vs carried-from-executed-rounds are stated separately; BLOCKED
legs name their missing prerequisite (§7 rule: never a silent skip,
never a blocked test marked passed).

## 1. Executed this round (r344 — the part16 opening round: the seventh rollback recovered + the Phase-2 TerminatedWorkerError fix)

| Lane | Command (class) | Result |
| --- | --- | --- |
| Canonical collection | `pytest --collect-only -q` (per-file sum) | **8,610 tests / 473 files** (8,602 + the 8 new Phase-2 regression tests; the shipped footer pin 8,555 = the frozen v1.2.240 release archive, the two-lane convention) |
| Quick regression chunk | `pytest tests/test_approvals.py tests/test_biology.py -q` | **108 passed (72+36), exit 0** — re-run fresh over the repaired venv |
| The Phase-2 suite | `pytest tests/api/test_ml_worker_termination.py -q` | **8/8 exit 0** — FAIL-FIRST honored: 6 red pre-fix (the -1 default, the raw worker-death propagation ×3 lanes, the absent advisory ×2), 8 green post-fix (the sequential/concurrent lanes were already green and pin the §5.4 regression class) |
| The affected ML + contract lanes | `pytest tests/api/test_datasets_ml_artifacts.py tests/api/test_contracts.py tests/api/test_ml_worker_termination.py -q` | **exit 0, zero failures** (83 dots) — the existing ML suite + the contract guard over the re-exported OpenAPI (the 1-line `default: 1` diff) |
| Lint | `ruff check .` (FULL tree) | **clean, exit 0** — after the round's own hygiene incident: the realignment tooling's first two cuts carried a decimal/octal comparison bug that set 2,997 files to mode 0o420; the corrected NUL/TAB-parsed pass realigned every file back to the git index modes (54 root-owned upload/ strays excepted, EPERM-class, cosmetic — the index says 644, the disk says 777, root-owned, not writable by this user) + the 7 EXE001 shebang +x restores (the r342-terminal disk parity) |
| Types | `mypy src/pharmaml` | **Success: no issues found in 296 source files** (the Phase-2 payload included) |
| Documentation consistency | `pytest tests/test_documentation_consistency.py -q` | **12/12 exit 0** with the new `docs/QA/COMPREHENSIVE_CURRENT_STATE_AUDIT.md` on disk (pre-ledger; re-run post-ledger with R344 == counter 136) |
| Frontend unit suite | `vitest run` | **126/126 over 16 files** — the GUI default change (`nJobs` 1) rides the same suite green |
| Frontend types | `tsc --noEmit` | **clean, exit 0** |
| Console | `GET /` | **200, title pin** (the one-window dance — the reaper wiped node_modules twice this round) |
| Zip-size API | `GET /api/toolkit/zip-size` | `{"zipBytes":23489390,"version":"1.2.240","exact":true,"source":"release"}` — exact pin |
| ECC standing check | `scratch/ecc_check.sh` | **GREEN** — clone `4adeee67`/2.2.3/328 skills; harness 14 modules/196 skills (the combined invocation, the r343 honest find re-applied) |
| Mirrors | the six-home zip parity | **18 zips × (root dist-archive + offsite) rebuilt, cmp ok=18/bad=0; v1.2.240 sha `0fadb6e5` across all six homes; testzip OK** |
| Recovery safety | the recovery-snap | `fae6c1c2` on the local branch `recovery-snap-r344` — captured BEFORE any destructive step (the r322 playbook) |
| Browser verification | agent-browser desktop + 390×844 mobile | the evidence pair rides the follow-up evidence commit (the pins below) |
| CI verdict at HEAD | authenticated check-runs poll | **19/19 success** at `8940bd42` (closed GREEN at r343; re-verified this round pre-push) |

## 2. Executed round r343 (the sixth-rollback recovery round — carried)

| Lane | Command (class) | Result |
| --- | --- | --- |
| Canonical collection | `pytest --collect-only -q` (per-file sum) | **8,602 tests / 472 files** (unchanged — the same tree as r342's terminal; exit 0) |
| Quick regression chunk | `pytest tests/test_approvals.py tests/test_biology.py -q` | **108 passed (72+36), exit 0**, 8.4 s |
| Lint | `ruff check .` (FULL tree) + `ruff check src/pharmaml` (the CI scope) | **both clean, exit 0** — after the round's own find: **1,498 permission-bit realignments** (the r340/r342 reaper class ×100 — the rollback restored old-era disk bits on files the reset never rewrote; `git ls-files -s` the source of truth) + **the 7 EXE001 shebang +x restores** (the r342-terminal disk parity — the tree records 100644 for the shebang set; the r342 close left them +x; the rollback's tree-conformant restore broke the pinned gate) — 116 findings → 0 |
| Root console lint | `bun run lint` (eslint, the root project) | **clean, true exit 0** — rode the per-call one-window dance (the reaper wiped node_modules ≥6 times this round, once between an install and its own lint) |
| Types | `mypy src/pharmaml` | **Success: no issues found in 296 source files** |
| Security supply chain | `pytest tests/test_security_supply_chain.py -q` | **8/8 exit 0** |
| Frontend-services drift guards | `pytest tests/test_frontend_services.py -q` | **17/17 exit 0** |
| Documentation consistency | `pytest tests/test_documentation_consistency.py -q` | **12/12 pre-ledger; re-run post-ledger with R343 == counter 135** |
| Frontend unit suite | `vitest run` (the rebuilt node_modules) | **126/126 over 16 files**, 19.08 s — fresh at the rebuilt frontend tree |
| Frontend types | `tsc --noEmit` | **clean, exit 0** |
| Console | `GET /` | **200, title pin** — via the one-window dance (the daemon re-danced ×4) |
| Zip-size API | `GET /api/toolkit/zip-size` | `{"zipBytes":23489390,"version":"1.2.240","exact":true,"source":"release"}` — exact pin |
| ECC standing check | `scratch/ecc_check.sh` (the owner's r335 directive) | **GREEN** — clone `4adeee67`/2.2.3/328 skills; harness 14 modules/196 skills (the honest find: the plain `--profile core` run resolves only 7 modules/54 skills — the pins require the combined `--profile core --modules machine-learning,research-apis,security,documentation-governance` invocation) |
| Census (live) | the console's own walk | **1,752 files / 570,433 LOC / 8,555 tests** — files + tests == the r342 pins; **LOC re-trued +55: the offline replica proves 570,433 is the TRUE walk of the 84b24f5c tree** (569,252 tracked raw newlines + 1,181 newline-terminated counted files of 1,242 viewable; untracked viewables 0 — the r342-recorded 570,378 was its pass-time undercount) — evidence `scratch/p15_r343_walkdiff.txt` |
| Browser verification | agent-browser desktop + 390×844 mobile | **zero page errors, zero console errors**, footer at page end both viewports (**3967 == 3967 desktop / 7882 == 7882 mobile**), mobile zero horizontal overflow (docW 390 == vw 390); evidence `scratch/r343-web-verify.png` + `scratch/r343-mobile-verify.png` — the blank mobile first frame (the daemon dead at the reload, caught by the geometry check) re-shot fresh |
| CI verdict at HEAD | authenticated check-runs poll | **19/19 success** at `84b24f5c` (the r342 evidence terminal — closed GREEN on arrival) |

## 3. Carried from executed rounds (latest full-run evidence)

- **r342** (the roadmap #3/#4 close-out round): the API suite
  **461+1 skip = 462 collected, exit 0, 117.495 s**; the
  trusted-origin matrix **34/34, 10.955 s**; the backend perf lanes
  (startup 4.138 s, health first 82.5 ms / warm 0.9 ms, idle RSS
  228.4 MiB zero-drift, graceful shutdown 0.665 s — the re-runnable
  driver `scratch/p15_r342_backend_perf.py`); the portable-scripts
  fake-target verification; the PowerShell-safe dev script set +
  the PERFORMANCE_BASELINES refresh (the roadmap deliverables).

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
