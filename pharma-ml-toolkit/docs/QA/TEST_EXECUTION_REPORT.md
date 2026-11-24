# TEST EXECUTION REPORT — part15 §13 deliverable #5 (the part16 era rides it forward)

The part15 Phase-Five honest execution record, carried forward as the
standing executed-vs-carried ledger. Executed-this-round (r349, the
part16 fifth round over the pushed `1ea3cc9a` tree — the r348
addendum terminal) vs carried-from-executed-rounds are stated
separately; BLOCKED legs name their missing prerequisite (§7 rule:
never a silent skip, never a blocked test marked passed).

## 1. Executed this round (r349 — the part16 fifth round: the Phase-12 experiment-tracking / run-lineage lane delivered)

| Lane | Command (class) | Result |
| --- | --- | --- |
| Canonical collection | `pytest --collect-only -q` (per-file sum) | **8,693 tests / 478 files** (r348's 8,664/476 + 29 the runs lane — the API suite 15 + the drift guard 14; the arithmetic verified programmatically; the shipped footer pin 8,555 = the frozen v1.2.240 release archive, the two-lane convention) |
| Quick regression chunk | `pytest tests/test_approvals.py tests/test_biology.py -q` | **108 passed (72+36), exit 0** |
| The Phase-12 runs suite | `pytest tests/api/test_runs.py -q` | **15/15 exit 0** — the summary/view projections (the lineage table + the full §15.2 record), the §15.2 block field-for-field truth (dataset lineage, feature configuration, preprocessing, algorithm + toolkit version, hyperparameters requested AND tuned, random seed, CV configuration with the parallelism advisory, evaluation metrics, environment snapshot, model-artifact reference, execution timestamps, the job-reference log pointer), the filter set + the 422-on-bad-status/limit + 404-on-unknown-run_id honesty, the MIGRATIONS-v3 SQLite durability + the JSON-only rehydration on startup, the failed/cancelled-run record with its typed error |
| FAIL-FIRST | the two vitest suites pre-implementation | **red** (the unresolved imports while the existing 178 stayed green) → **green post-implementation** (the discipline honored: 20/20 new) |
| The affected lanes | `pytest tests/test_frontend_settings.py tests/test_frontend_foundation.py tests/test_frontend_startup.py tests/test_frontend_services.py tests/test_frontend_resource_monitor.py tests/test_frontend_runs.py -q` + `pytest tests/api/test_contracts.py tests/test_documentation_consistency.py tests/test_security_supply_chain.py -q` | **exit 0, zero failures** (73 + 26 dots) — the contract CHANGED this round (the backend runs pair: the 112→**114-path**/72-schema contract re-exported, the generated `api.d.ts` re-grown — the r345 stale-types incident class avoided by construction, the contracts suite green over the new pair) |
| Lint | `ruff check .` (FULL tree) | **clean, exit 0** |
| Types | `mypy src` | **Success: no issues found in 299 source files** (+2 — the runs modules `api/runs.py` + `routes/runs.py`) |
| Frontend unit suite | `vitest run` (the Studio frontend) | **198/198 over 25 files** (+20: the runs client 13, the RunLineage panel 7 — the services mocked only at the module boundary; the MLPage lane test rides the real client's jsdom rejection) |
| Frontend types | `tsc --noEmit` | **clean, exit 0** (the pinned local 5.9.3) |
| Root console lint | `bun run lint` (the root project, `eslint .` sweeps the workspace) | **clean, true exit 0** |
| Console | `GET /` | **200, title pin** (`PharmaML Toolkit — ML · Pharma · Docking`, the 202,047-B body) — over the self-healing supervisor daemon; the first boot served 500s on the **corrupt-nested-package class** (`hastscript`'s nested `property-information` failing to resolve `property-information/normalize` — the reaper-damaged node_modules subtree) — recovered by the targeted subtree reinstall (hastscript/refractor/property-information removed + `bun install` restoring clean copies) + ONE supervisor boot after the `.next` purge |
| Ten-chunk gate | the chunks 1–8 + 9 + 10 + 11 + 12a/b legs (`scratch/run_suite_chunks_349.py`, daemonized, the dev server down) | **green with the one standing deviation decoded per the drills** — chunk-1 89s / chunk-2 196s / chunk-3 139s / chunk-4 195s / chunk-6 113s / chunk-7 61s / chunk-8 57s rc=0; chunk-5 SIGKILL → the OOM split (sub-batches 1–4 green — 39s / 19s / 8s / 67s; sub-batch 5 rc=-9 → singles: `test_main_window` 65s rc=0 / `test_smiles_preview` 8s rc=0 / `test_table_tools` 23s rc=0; the heavy `test_nine_card_first.py` whole-file rc=-9 at 369s — re-probed rc=137, the envelope reproduced) → **the six-class split — ALL SIX classes exit=0, 23/23, the sixth independent verdict** (TestWorkspaceDefaults 8 / TestWorkspaceSubpages 3 / TestHistoryAndNavigation 6 / TestAdvancedNavigation 3 / TestKeyboardAccess 2 / TestRestoreBehaviour 1); chunk-9 the nine-card envelope 22s / chunk-10 src doctests 81s / chunk-11 the api suite 109s / chunk-12a+b the github-page guards rc=0; the script's honest `failures=1` names exactly the decoded envelope leg |
| OpenAPI contract | the runs pair added | **114 paths / 72 schemas** (the 112→114 delta = `GET /api/v1/runs` + `GET /api/v1/runs/{run_id}`; the generated `api.d.ts` re-grown +113 lines over the runs DTOs) |
| ECC standing check | (carried — no ECC-facing change this round) | the r348 GREEN verdict carried (the clone 4adeee67/2.2.3/328 skills + the harness at the r335 pins) |
| Browser verification | agent-browser desktop + 390×844 mobile | **zero page errors, zero console errors**; the footer carrying the LIVE walk **1,787 files / 574,728 LOC** (the live post-ledger walk == the page render; files +9 = the lane's own nine — the four backend .py + the five frontend nodes; LOC +1,603 = the four .py + the contract pair + the QA-doc truthing); the footer geometry exact at page end; mobile zero horizontal overflow (docW 390 == vw 390); the zip-size API pin exact (**23,489,390 B / v1.2.240 / exact / release**); the evidence pair `scratch/r349-web-verify.png` + `scratch/r349-mobile-verify.png` riding THIS commit |
| CI verdict at HEAD | authenticated check-runs poll | **19/19 success** at `1ea3cc9a` (the r348 addendum terminal — re-verified pre-push) |

## 2. Prior round (r348 — the part16 fourth round: the Phase-8 floating resource-monitor lane delivered)

| Lane | Command (class) | Result |
| --- | --- | --- |
| Canonical collection | `pytest --collect-only -q tests --doctest-modules src/pharmaml` (per-file sum) | **8,664 tests / 476 files** (r347's 8,650/475 + the 14-test Phase-8 drift guard — the arithmetic verified programmatically; the shipped footer pin 8,555 = the frozen v1.2.240 release archive, the two-lane convention) |
| Quick regression chunk | `pytest tests/test_approvals.py tests/test_biology.py -q` | **108 passed (72+36), exit 0** |
| The Phase-8 suite | `pytest tests/test_frontend_resource_monitor.py -q` | **14/14 exit 0** — the service trio + component existence, the single-integration-point pin (the §15 route + the generated contract both carry `/api/v1/system/resources`), the no-simulation scan (banned constructs + the honest-null parser pins + the dash render), the geometry pocket (the literal key/bounds/step pins, validate-on-read, clamp-on-write), the full interaction vocabulary (drag/keyboard move, clamped resize, toggle/close, the Escape scope rule, the aria contract), the header entry on every workspace, the cadence (the §10 interval + the same 250ms floor, stops closed, skips hidden), the section separation + the six-state mirror, the suite-honesty clauses |
| FAIL-FIRST | the three vitest suites pre-implementation | **red** (the 3 new test files + the App §8 case failing on the missing entry — 4 files failed pre-implementation) → **green post-implementation** (the discipline honored: 20/20 new) |
| The affected lanes | `pytest tests/test_frontend_settings.py tests/test_frontend_foundation.py tests/test_frontend_startup.py tests/test_frontend_services.py tests/test_frontend_resource_monitor.py -q` + `pytest tests/test_documentation_consistency.py tests/test_security_supply_chain.py -q` | **exit 0, zero failures** (59 + 20 dots) — the contract UNCHANGED this round (a frontend-only lane: zero backend bytes, the 112-path/72-schema contract untouched, the one-DTO rule untouched — the r345 stale-types incident class avoided by construction) |
| Lint | `ruff check .` (FULL tree) | **clean, exit 0** — after the ninth rollback's exec-bit residue swept (101 EXE002 files realigned to the git-index modes — the r313/r342/r346 terminal parity set) |
| Types | `mypy src/pharmaml` | **Success: no issues found in 297 source files** (the count unchanged — a frontend-only lane) |
| Frontend unit suite | `vitest run` (the Studio frontend) | **178/178 over 23 files** (+20: the system client 4, the panel geometry pocket 4, the ResourceMonitor panel 11, the App header-entry integration 1 — the services mocked only at the module boundary) |
| Frontend types | `tsc --noEmit` | **clean, exit 0** (the first cut's two unsafe Record casts in the parser replaced with the house field-by-field validated construction) |
| Root console lint | `bun run lint` (the root project, `eslint .` sweeps the workspace) | **clean, true exit 0** (the first cut's `react-hooks/refs` error — the render-time ref mutation — fixed at source: the ref eliminated for state closures + functional updates; re-run clean over the reaper-danced node_modules) |
| Console | `GET /` | **200, title pin** (`PharmaML Toolkit — ML · Pharma · Docking`, the 202,044-B body) — over the self-healing supervisor loop; the boot rode the round's own disk-full decode (the eslint "vanishing" between installs named at its true cause — `NoSpaceLeft creating node-gyp tempdir`, the r347 causal chain taught again: the stale /tmp recovery snap 1.1G + the puppeteer 617M / pip 534M / bun caches cleared, the reinstall landed clean) |
| Ten-chunk gate | the chunks 1–8 + 9 + 10 + 11 + 12a/b legs | **green over the r348 terminal tree with both deviations decoded per the standing drills** — chunk-1 199s / chunk-2 143s / chunk-4 322s / chunk-6 189s / chunk-7 123s / chunk-8 190s rc=0; chunk-5 SIGKILL → the OOM split (sub-batches 1–4 green, sub-batch 5 rc=-9 → singles: `test_phase5` 21s / `test_qt_noise_filter` 11s / `test_table_tools` 47s rc=0, the heavy `test_nine_card_first.py` whole-file rc=-9 **decoded via the six-class split — ALL SIX classes exit=0, 23/23 green, the r301/r303/r304/r347 verdict a FIFTH time**); chunk-9 the nine-card envelope 27s / chunk-10 doctests 93s / chunk-11 the api suite 117s / chunk-12a+b the github-page guards rc=0; **chunk-3 rc=1 — the round's own find, decoded honestly**: two `TestALImportLabels` ERRORs under the round's own concurrent interactive verification legs (the Qt-environment collision, no source defect — the file re-verified **7/7 solo** + the pair **2/2 targeted**; the discipline note: no concurrent python legs while the gate daemon runs) — see the worklog R348 entry for the per-chunk timings |
| OpenAPI contract | unchanged by construction | the 112 paths / 72 schemas contract byte-identical; the generated types untouched |
| ECC standing check | `scratch/ecc_check.sh` | **GREEN** — both legs reinstalled this round at the exact r335 pins after the ninth rollback wiped them (the clone 4adeee67/2.2.3/328 skills; the harness via the local clone's `scripts/ecc.js` — profile core + the four capability components, 14 modules/196 skills; the sql.js dependency note) |
| Browser verification | agent-browser desktop + 390×844 mobile | **zero page errors, zero console errors**; the footer carrying the LIVE walk **1,778 files / 573,125 LOC** — the three-way true (the live fresh walk == the offline census replica == the page render; files +8 = the lane's own eight, LOC +308 = the drift guard's .py + the QA-doc truthing — the .tsx lanes never count); the footer geometry exact at page end (**3958 == 3958 desktop**); mobile zero horizontal overflow (docW 390 == vw 390); the zip-size API pin exact (**23,489,390 B / v1.2.240 / exact / release**); the evidence pair `scratch/r348-web-verify.png` + `scratch/r348-mobile-verify.png` riding THIS commit |
| CI verdict at HEAD | authenticated check-runs poll | **19/19 success** at `580da84c` (the r347 terminal — re-verified pre-push) |

## 3. Executed round r346 (the part16 third round: the Phase-7 settings lane delivered — carried)

| Lane | Command (class) | Result |
| --- | --- | --- |
| Canonical collection | `pytest --collect-only -q` (per-file sum) | **8,650 tests / 475 files** (r345's 8,637/474 + the 13-test settings drift guard — the arithmetic verified programmatically; the shipped footer pin 8,555 = the frozen v1.2.240 release archive, the two-lane convention) |
| Quick regression chunk | `pytest tests/test_approvals.py tests/test_biology.py -q` | **108 passed (72+36), exit 0** |
| The Phase-7 suite | `pytest tests/test_frontend_settings.py -q` | **13/13 exit 0** — the service trio + the React binding, the §10 inventory keys + the conservative pins, the validation bounds as literal pins (-1..64 parallelism, the 250 ms floor, the enum vocabularies), the unknown-key rejection (the no-unsafe-values rule), the schema-versioned storage + validate-on-read, the header entry + the popover dialog + the compact width, the panel control set + the inline alert, the shell mounts, the three DOM applications + the CSS palette/density rules, the ML form prefill, the toast vocabulary mirror, the version-carrier lockstep (version.ts == package.json == VERSION) |
| FAIL-FIRST | the three vitest suites pre-implementation | **red** (3 test files, the imports unresolvable) → **green post-implementation** (the discipline honored: 26/26) |
| The affected lanes | `pytest tests/test_frontend_settings.py tests/test_frontend_foundation.py tests/test_frontend_startup.py tests/test_frontend_services.py -q` + `pytest tests/api/test_contracts.py tests/test_documentation_consistency.py tests/test_security_supply_chain.py -q` | **exit 0, zero failures** (45 + 26 dots) — the contract UNCHANGED this round (a frontend-only lane: zero backend bytes, the 112-path/72-schema contract untouched, the one-DTO rule untouched — the r345 stale-types incident class avoided by construction) |
| Lint | `ruff check .` (FULL tree) | **clean, exit 0** — after the eighth rollback's exec-bit residue swept (109 EXE002 files realigned to the git-index modes + the 7 EXE001 shebang +x restores — the r313/r342 terminal parity set) |
| Types | `mypy src` | **Success: no issues found in 297 source files** (the count unchanged — a frontend-only lane) |
| Security supply chain | `pytest tests/test_security_supply_chain.py -q` | **8/8 exit 0** (inside the 26-dot run above) |
| Documentation consistency | `pytest tests/test_documentation_consistency.py -q` | **12/12 pre-ledger** (re-run post-ledger with R346 == counter 138) |
| Frontend unit suite | `vitest run` (the Studio frontend) | **158/158 over 20 files** (+27: the settings store suite 12, the panel suite 7, the toasts suite 6, the App header-entry integration test 1 — the services mocked only at the module boundary; the full suite re-run green twice over the reaper-danced node_modules) |
| Frontend types | `tsc --noEmit` | **clean, exit 0** (the three TS2349 narrowing finds in the store test fixed in-round via the handlers-array matchMedia stub) |
| Root console lint | `bun run lint` (the root project, `eslint .` sweeps the workspace) | **clean, true exit 0** — the first cut's two `react-hooks/set-state-in-effect` errors on the panel's draft-sync effects fixed at source (the keyed-remount pattern), re-run clean |
| Console | `GET /` | **200, title pin** (`PharmaML Toolkit — ML · Pharma · Docking`) — over the self-healing supervisor loop (the daemon killed ≥3 times this round; the reaper wiped node_modules ≥4 times, once between an install and its own lint) |
| Zip-size API | `GET /api/toolkit/zip-size` | `{"zipBytes":23489390,"version":"1.2.240","exact":true,"source":"release"}` — exact pin |
| Census (live) | `python scratch/census.py` (the offline walk replica) | **1,769 files / 572,739 lines pre-ledger** (+11 the lane's own files: the 7 frontend modules + the 3 vitest suites + the drift guard; −1 the swept rollback-resurrection `.github` duplicate the r345-era disk walk had counted; LOC +218 = the drift guard's own .py lines — the .ts/.tsx lanes never count per the census rule; the post-ledger walk re-trues in the evidence commit) |
| OpenAPI contract | unchanged by construction | the 112 paths / 72 schemas contract byte-identical; `test_contracts.py` 6/6 green over it |
| ECC standing check | `scratch/ecc_check.sh` | **GREEN** — reinstalled this round at the exact r335 pins after the eighth rollback wiped BOTH legs (the clone 4adeee67/2.2.3/328 skills re-cloned; the harness 14 modules/196 skills via the local clone's `scripts/ecc.js` — the npm CLI rejected the `documentation-governance` module id, the clone's manifest carries it; the sql.js dependency note) |
| Browser verification | agent-browser desktop + 390×844 mobile | the evidence pair rides the follow-up evidence commit (the pins below) |
| CI verdict at HEAD | authenticated check-runs poll | **19/19 success** at `0e92d790` (the r345 evidence terminal — re-verified this round pre-push) |

## 4. Executed round r345 (the part16 second round: the Phase-5 demo-data lane delivered — carried)

| Lane | Command (class) | Result |
| --- | --- | --- |
| Canonical collection | `pytest tests src --collect-only -q` (per-file sum) | **8,637 tests / 474 files** (r344's 8,610/473 + the 26-test samples suite + the 1 new drift guard — the arithmetic verified programmatically; the shipped footer pin 8,555 = the frozen v1.2.240 release archive, the two-lane convention) |
| Quick regression chunk | `pytest tests/test_approvals.py tests/test_biology.py -q` | **108 passed (72+36), exit 0** |
| The Phase-5 suite | `pytest tests/api/test_samples.py -q` | **26/26 exit 0** — the catalog-reality guard (every entry ↔ the real shipped file, sha256 verified), preview-before-load never registering, load-twice-never-replaces, the four payload kinds, the per-workflow usability lanes (ML trains on the loaded tables; the chemistry descriptors + the docking ligand-prep + the DLG parse over the delivered payloads; the NCA end-to-end over the series), the token guard |
| The NCA zero-row fix (fail-first) | the series lane + the regression check | **red pre-fix** (the route's blanket positive-only validation rejected the shipped pk_profile.csv's pre-dose t=0 row — the very file the demo lane ships) → **green post-fix** (`NcaRequest.require_positive_conc=False` mirroring the steady-state precedent + the library's own NCA core; **negatives still rejected** — the validator's non-negative floor stands; `auc_trapezoid_log: null` honestly over the zero row) + the whole pkpd surface regression-checked: `test_docking_pkpd_md_viz_ad.py` **66/66 exit 0** |
| The affected lanes | `pytest tests/api/test_samples.py tests/api/test_contracts.py tests/test_frontend_services.py tests/api/test_docking_pkpd_md_viz_ad.py -q` | **exit 0, zero failures** (26 + 6 + 18 + 66) — the new drift guard `test_sample_rest_surface_covered` pins the samples route surface against the backend module AND the client's path literals (the round's own honest correction: the first cut's redundant single-entry GET route was born-then-removed when this guard exposed the coverage gap; the final surface 3 routes, all carried) |
| Lint | `ruff check .` (FULL tree) | **clean, exit 0** (the round's own F401 on the first cut of the test file fixed in-round) |
| Types | `mypy src/pharmaml` | **Success: no issues found in 297 source files** (+1: the route module `api/routes/samples.py`) |
| Security supply chain | `pytest tests/test_security_supply_chain.py -q` | **8/8 exit 0** |
| Frontend-services drift guards | `pytest tests/test_frontend_services.py -q` | **18/18 exit 0** (+1: the samples guard) |
| Documentation consistency | `pytest tests/test_documentation_consistency.py -q` | **12/12 pre-ledger** (re-run post-ledger with R345 == counter 137) |
| Frontend unit suite | `vitest run` | **131/131 over 17 files** (+5: the `DemoDataSection` suite — the catalog render, preview-first never-registers, the confirm gate, the series payload delivery, the unavailable row's honest disabled state; services mocked only at the module boundary) |
| Frontend types | `tsc --noEmit` | **clean, exit 0** (the one TS2532 in the new test fixed in-round) |
| Root console lint | `bun run lint` (the root project) | **clean, true exit 0** — rode the per-call one-window dance (the reaper wiped node_modules ≥3 times this round, once between an install and its own lint) |
| Console | `GET /` | **200, title pin** (`PharmaML Toolkit — ML · Pharma · Docking`) |
| Zip-size API | `GET /api/toolkit/zip-size` | `{"zipBytes":23489390,"version":"1.2.240","exact":true,"source":"release"}` — exact pin |
| Census (live) | the console's own walk | **1,759 files / 572,461 LOC pre-ledger** (files +5 = the route + the API suite + the client + the component + the component suite — the arithmetic checks; the post-ledger walk re-trued in the evidence commit; the .tsx lanes never count per the census rule) |
| OpenAPI contract | `python tools/export_openapi.py` | re-exported twice in-round (113 paths mid-round; **112 paths / 72 schemas at the final surface** — +3 sample routes, +1 `LoadRequest` schema, the redundant single-entry route removed within the round); `test_contracts.py` 6/6 green over it |
| ECC standing check | `scratch/ecc_check.sh` | **GREEN** — clone `4adeee67`/2.2.3/328 skills; harness 14 modules/196 skills |
| Browser verification | agent-browser desktop + 390×844 mobile | the evidence pair rides the follow-up evidence commit (the pins below) |
| CI verdict at HEAD | authenticated check-runs poll | **19/19 success** at `969f7926` (the r344 evidence terminal — re-verified this round pre-push) |

## 5. Executed round r344 (the part16 opening round: the seventh rollback recovered + the Phase-2 TerminatedWorkerError fix — carried)

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

## 6. Executed round r343 (the sixth-rollback recovery round — carried)

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

## 7. Carried from executed rounds (latest full-run evidence)

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
