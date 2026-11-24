# COMPREHENSIVE CURRENT STATE AUDIT — part16 §3 (Phase 0) deliverable

The part16 Phase-Zero evidence report: the 9-class current-state
classification over the directive's sixteen phases plus the
previously-requested campaign ledger (parts 11–15), every row naming
its implementation location, test evidence, status, and remaining
defect. Audited commit **`8940bd42`** (main == origin/main, the r343
evidence terminal); audited round **r344** (the 136th recorded run —
the part16 opening round, executed over the seventh-rollback-recovered
tree). Nothing here is copied from prose alone: each row was re-walked
against the live source this round.

Status vocabulary (the directive's §3 classes): **IMPLEMENTED_AND_TESTED**
· **INSUFFICIENTLY_TESTED** · **PARTIAL** · **BACKEND_ONLY** ·
**FRONTEND_ONLY** · **BROKEN** · **MISSING** · **BLOCKED_ENV** ·
**DEPRECATED**.

## 1. Repository state (re-established live this round)

| Item | Value | Evidence |
| --- | --- | --- |
| HEAD | `8940bd42` == origin/main (r343 evidence terminal) | fetch via the askpass lane, post-recovery |
| Version | 1.2.240, byte-frozen (the r334 ruling — no re-ship without approval) | nine-carrier lockstep, consistency 12/12 |
| Release | `v1.2.240` published, five assets intact | r337 live verification, standing ruling |
| CI at HEAD | 19/19 check-runs success (closed at r343) | authenticated check-runs API |
| Tests | collection 8,602 / 0 errors / 472 files; shipped pin 8,555 | quick chunk 108/108 re-run fresh this round |
| Rollback | the SEVENTH wave recovered pre-audit (the r343 class: history reset to the r186 checkpoint `2eb0ede`; remote + PAT + venv + node_modules + ECC + both zip mirrors wiped; the r322 playbook re-run in full — the recovery-snap `fae6c1c2` captured before any destructive step) | worklog R344 |

## 2. The part16 phase-by-phase classification

| Phase (directive §) | Implementation location | Test evidence | Status | Remaining defect |
| --- | --- | --- | --- | --- |
| 0 — repository recovery + previous-work verification | the standing 2h-sync ritual (worklog/counter/PROJECT_STATUS); this document is the phase's own deliverable | every recovery leg re-verified live this round (gates below) | IMPLEMENTED_AND_TESTED | none (the ritual itself) |
| 1 — backend-to-frontend capability integration | `docs/CAPABILITY_INTEGRATION_MATRIX.md` (313 rows-class rows, the part11_1 §21 campaign) + the 13 domain clients in `frontend/src/services/` | `tests/api/` suites + the frontend drift guards 17/17 + vitest 126/126 | PARTIAL | 13 matrix rows PARTIAL + 2 BLOCKED (the Rust lanes) — the vocabulary mapping in §5 |
| 2 — ML training worker termination (TerminatedWorkerError) | **OPENED THIS ROUND**: the conservative default `n_jobs=1` (`api/routes/ml.py` TrainRequest + the GUI form `MLPage.tsx`), the typed worker-death translation (`ml/training.py` — TerminatedWorkerError/BrokenProcessPool → actionable ModelError, original chained), the resource-aware clamp (`_resolve_parallelism`, concurrency-reduced never science, reported in the `parallelism` payload block) | `tests/api/test_ml_worker_termination.py` 8/8 (FAIL-FIRST: 6 red pre-fix), re-run green + the ML suite 83-dots exit 0 | IMPLEMENTED_AND_TESTED (the sandbox lanes) | the Windows multiprocessing + frozen-backend legs of §5.4 ride the Windows-box lane (BLOCKED_ENV, documented in the suite docstring); the memory-copy elimination lane (§5.2 "prevent unnecessary copying") is the continuation |
| 3 — export + artifact integrity | the digest-verified download contract (`api/routes/artifacts.py`, ETag/sha256/Content-Length) | `test_datasets_ml_artifacts.py` download lanes (magic bytes, round-trips) | IMPLEMENTED_AND_TESTED | none found this audit |
| 4 — dataset lifecycle + deletion | `api/routes/datasets.py` (register/preview/columns/split/delete) | `test_datasets_ml_artifacts.py` (delete-then-404, guards) | IMPLEMENTED_AND_TESTED | none found this audit |
| 5 — demo data + sample workflows | **DELIVERED r345** (the Phase-0 audit's MISSING row closed): `api/routes/samples.py` — the curated catalog over the 17 REAL shipped `examples/data` files (runtime available/size/sha256 guards), `GET /api/v1/samples` (+ `?workflow=` filter), the raw-head preview-before-load, the one-click load with four honest payload kinds (tabular through the REAL loader as a FRESH record; series → the times/conc arrays; library → the name+SMILES columns; text_file → the sha256-pinned content); the shared `frontend/src/components/demo/DemoDataSection.tsx` wired into all nine workspaces; the round's own NCA zero-row fix at source (`NcaRequest` mirrors the steady-state precedent — the shipped pk_profile.csv's pre-dose t=0 row was wrongly rejected by the blanket positive-only validation) | `tests/api/test_samples.py` **26/26** (the catalog-reality guard — every entry ↔ the real file, sha256 verified; the per-workspace usability: ML trains on the loaded tables, the chemistry descriptors + the docking ligand-prep + the DLG parse over the delivered payloads, the NCA end-to-end over the series) + the `DemoDataSection` vitest 5/5 + the drift guard 18/18 | IMPLEMENTED_AND_TESTED (the sandbox lanes) | the desktop-app deep-dive (the loaded demo dataset in the packaged WebView2 runtime) rides the Windows-box lane (BLOCKED_ENV) |
| 6 — workflow-specific visualization architecture | the part11-era ownership lanes + `viz/` (backend figures) + the Plotly adoption (r291) | capability matrix §3.2 (the 11th-fact PARTIAL row) | PARTIAL | the residual PARTIAL rows (the Visualization workspace's general-purpose-only boundary) |
| 7 — settings + application configuration | **DELIVERED r346** (the Phase-0 audit's MISSING row closed): `frontend/src/services/settings/` (types + store + the schema-versioned localStorage persistence `pharmaml.studio.settings.v1`), the `useSettings` hook on the house useSyncExternalStore pattern, the compact popover `SettingsEntry`/`SettingsPanel` on the app header (every workspace), the shell's `JobCompletionToasts` (the notification preference's real consumer over the REAL event stream), the three DOM applications (the `.dark` theme class + the system lane, the `data-density` scale, the `.charts-cvd` Okabe-Ito palette), the ML form's n_jobs prefill from the execution default, the plotly export-default rider, the diagnostics block over the version carriers | **26 vitest** over the three suites (the store 12: defaults, validate-before-apply, persistence round-trip, corrupt-storage degradation, reset, the matchMedia system lane; the panel 7: the popover mechanics, the §10 section inventory, theme apply+persist, density+CVD+export, the inline refusal, the diagnostics, reset; the toasts 6: the terminal three, non-terminal silence, dedup, the off-preference silence, auto-dismiss, unsubscribe) + the App header integration test + the Python drift guard `tests/test_frontend_settings.py` **13/13** | IMPLEMENTED_AND_TESTED (the sandbox lanes) | the desktop-app persistence across packaged-app restarts (the WebView2 storage lane) rides the Windows-box lane (BLOCKED_ENV) |
| 8 — floating resource monitor | **DELIVERED r348** (the Phase-0 audit's MISSING row closed): `frontend/src/services/system/` (the typed `ResourceSnapshot` DTOs + the strict parser + the one-operation client over the EXISTING `GET /api/v1/system/resources` — the §15 route consumed unchanged, the JobManager queue depth included; the geometry pocket `pharmaml.studio.resmon.v1` with validate-on-read + clamp-on-write) + `frontend/src/components/system/ResourceMonitor.tsx` (the compact floating panel: the compact summary and the detailed view with the System—host vs Application—backend-process separation, drag + keyboard move, clamped pointer + keyboard resize, the persisted visible preference toggle, position/size remembered, GPU degrading to its stated line, the absent stdlib-fallback numbers as the honest dash, the cadence = the §10 interval with the 250ms floor, no polling while hidden/closed) + the header entry beside the gear (every workspace's reach) | **20 vitest** over the four new suites (the client contract 4, the geometry pocket 4, the panel's full interaction vocabulary 11, the App header-entry integration 1 — 178/178 total) + the Python drift guard `tests/test_frontend_resource_monitor.py` **14/14** | IMPLEMENTED_AND_TESTED (the sandbox lanes) | the packaged-WebView2 runtime's live polling behavior rides the Windows-box lane (BLOCKED_ENV) |
| 9 — manual docking grid box | the numeric form EXISTS: `DockingPage.tsx` stage 4 "binding site / grid box" + the "Manual grid box" card; the backend `compute_grid_box` (docking/preparation.py) | docking suites green (the campaign lanes) | PARTIAL | the directive's deeper ask — the synchronized protein+box 3D visualization with numeric-coordinate sync — is the continuation |
| 10 — design system + visual modernization | the part13-era design system (Tailwind tokens, the console + app surfaces) | browser passes r337–r343 (zero errors, geometry pins) | IMPLEMENTED_AND_TESTED | ongoing polish rides the roadmap |
| 11 — installer branding + Windows packaging | v1.2.240 EXE/MSI (the r334/r337 ship, smoke-gated ninth dispatch) | release assets verified byte-level (r337) | IMPLEMENTED_AND_TESTED | the E/F clean-machine gates — BLOCKED_ENV (the owner's Windows box) |
| 12 — Job Center / experiment tracking / model registry / data quality / diagnostics | job center: the PySide6 legacy widget (DEPRECATED-class); model registry: **DELIVERED r350** (the §15.3 GUI projection: `DELETE /api/v1/ml/trained/{model_id}` — the deletion lane riding the bounded store's own evict, the persisted row + vault payload gone, the honest 404, the scope rails pinned — the §15.2 run lineage and the exported joblib artifacts outlive the row; the ml client's `catalog(undefined)` full listing + `evictTrained` + `exportTrained`; `components/registry/ModelRegistry.tsx` — the joined card in the ML Compare stage: the catalog with the per-row availability + the honest `requires` line, the trained registry newest-first with the derived version column, the on-demand cached record, the digest-verified joblib export, the two-click evict confirm); experiment tracking: **DELIVERED r349** (the §15.2 first sub-lane: `api/runs.py` — the `ExperimentRunRecord` joining the full §15.2 checklist in one queryable view + the bounded `RunStore` with the MIGRATIONS-v3 SQLite durability and JSON rehydration, recorded at BOTH training exits in `routes/ml.py` — the completed run and the failed/cancelled run — from the very objects the route already holds; the `GET /api/v1/runs` lineage table + `GET /api/v1/runs/{run_id}` full record at contract 114 paths; the frontend projection `services/runs/` + `components/runs/RunLineage.tsx` mounted in the ML Compare stage); data quality: **DELIVERED r351** (the §15.4 surface: `pharmaml.data_quality` — the two-stage SMILES validation [grammar parse, then sanitization] + the canonical-form duplicate census + the missing-value/duplicate-row/non-finite descriptor reports + the leakage family [target leakage, duplicate columns, the ID-like advisory, the train/test overlap advisory] + the class-imbalance warning + the pinned unit-range vocabulary, every check read-only; `POST /api/v1/datasets/{id}/quality` [sync + job modes, the honest 404] at contract 115 paths + the regenerated api.d.ts; the frontend projection `services/quality/` + `components/quality/DataQualityCard.tsx` mounted in the Datasets Validate stage — the severity chips, the pinned-code findings, the executed/skipped §15.4 check list, the advisory line, NO mutation control by design); diagnostics/recovery: **DELIVERED r352** (the §15.5 surface: `GET /api/v1/system/diagnostics` — the dependency availability block [the 21-probe requirements.txt vocabulary, import + version + the required/optional tier, a broken probe a REPORTED state never a 500] + the scientific-tool functional probes [rdkit's §15.4 grammar parse reused, torch's tiny tensor op, the selfies round-trip; absent tools the honest null] + the storage health [the persistence status + the data-dir disk block + the database/vault sizes, or the honest ephemeral note] + the crash diagnostics [the ten most recent FAILED jobs with their errors, from the SQLite history when persistent] + the log configuration report [read-only] + the recoverable job state [the persisted history vs the in-session manager, honestly split — QUEUED/RUNNING rows from a dead process are NOT resumed, checkpointing stays queued scope]; `POST /api/v1/system/cache/cleanup` — the safe cleanup over the enumerated derived cache only [the samples stats cache via its own public clear function], every user-data surface named preserved, the §15.5 verbatim rule riding both sources; the contract re-exported at 117 paths + api.d.ts re-grown; the frontend projection `services/system/diagnostics.ts` [the typed DTOs + the strict parser] + the two operations on the existing systemApi + `components/system/DiagnosticsCard.tsx` mounted in the Introduction workspace beside the studio-database section it completes — the ONE mutation control is the safe cleanup, the diagnostics read-only by design) | the backup/restore + job-history half already shipped (the database route lane) | `tests/api/test_data_quality.py` **22/22** + `tests/test_frontend_quality.py` **9/9** + the quality vitest pair **16** (234/234 total); the registry lane's 9/9 + 11/11 + 20 and the runs lane's 15/15 + 14/14 + 20 carried green; the diagnostics lane's `tests/api/test_system_diagnostics.py` **19/19** (incl. the user-data-survives-the-cleanup fail-first proof) + `tests/test_frontend_diagnostics.py` **11/11** + the diagnostics vitest pair **20** (254/254 total over 31 files); `/datasets/{id}/quality` + the finding vocabulary pinned in the new suites | experiment tracking + the model registry + the data-quality surface + the diagnostics/recovery surface IMPLEMENTED_AND_TESTED; the Phase-9 grid-box lane remains (the #7 continuation) | the §15.6 prioritization rides the roadmap's #7 (the last open item) |
| 13 — security + desktop runtime verification | the D-1..D-8 campaign (trusted origins, scoped CORS, WS guard, token lane) | `test_trusted_origins.py` 34, `test_ws_origin.py` 12/12, security suites | IMPLEMENTED_AND_TESTED | the desktop-runtime legs ride the Windows box (BLOCKED_ENV) |
| 14 — complete test strategy | the 8,776-test collection + the ten-chunk gate + the drift guards | collection 8,776/0/485 (r352 pin: +30 tests over the r351 data-quality lane — the diagnostics API suite 19 + the drift guard 11) | IMPLEMENTED_AND_TESTED | the frozen-backend E2E legs — BLOCKED_ENV |
| 15 — legacy PySide6 + studio.py retirement | **NOT STARTED**: `studio.py` (45-line launcher) + `ui/` (the PySide6 GUI) present; the playground (Tkinter) precedent retired at r338 | the r228/r304/r338 retirement lineage in the worklog | PARTIAL (the precedent) + BLOCKED_ENV (the parity verification gates on the Rust/Tauri compile lane) | the parity proof + the retirement sweep — the campaign's large stone |
| 16 — final build + installer validation + GitHub release | v1.2.240 shipped (five assets, manifest, SBOM) | the ninth smoke dispatch green end-to-end | IMPLEMENTED_AND_TESTED | the v1.2.241+ ship lane is closed by the byte-frozen ruling (owner approval required) |

## 3. The previously-requested campaign ledger (parts 11–15)

The r340-era COMPREHENSIVE_REPOSITORY_AUDIT §2 matrix (15 rows, all
VERIFIED COMPLETE or honestly execution-blocked) was re-walked this
round and stands — no row regressed: the backend/frontend stacks, the
REST contract (109 paths/71 schemas, CI-guarded, re-exported this
round with the Phase-2 default change), the WS lanes, SQLite +
FileVault, the nine workspaces, security, the 8,555-shipped pin, the
browser E2E lanes, the Windows installers, offline operation, and the
documentation set (consistency 12/12). The two BLOCKED rows (Tauri
Rust compile/test, Windows-box E/F gates) are unchanged environmental
classes, not defects.

## 4. Findings this audit

1. **The Phase-2 mechanism was real and is now fixed at source**
   (fail-first: 6 red → 8 green): the API default was `n_jobs=-1`
   (all cores) — exactly the nested-parallelism + OS-kill mechanism
   the directive names; the raw loky TerminatedWorkerError penetrated
   with zero guidance. Now: default 1 (both the API schema and the
   GUI form), the typed translation with the three retry levers
   chained to the original, and the §5.3 memory-aware clamp that
   reduces concurrency — never the science — and reports the
   decision in the payload. The OpenAPI contract re-exported
   (1-line diff, the default value).
2. **The genuinely-missing lanes this directive adds** (Phase 7
   settings persistence, Phase 8 the new-app resource monitor, the
   Phase 12 P0/P1 sub-lanes) — recorded here as MISSING, each a
   concrete next-round candidate. **The Phase-5 demo-data lane was
   this class's first row and is now DELIVERED (r345)** — this
   audit's own MISSING row closed within the round that followed;
   **the Phase-7 settings lane is the second (DELIVERED r346)**.
3. **The Phase-15 retirement is the campaign's largest stone**: the
   parity verification gates on the Rust/Tauri compile lane which this
   sandbox cannot run (the D-4/D-5 class) — the honest sequencing is
   the missing-lane build-out first, the parity proof when the
   Windows-box lane opens.
4. The seventh rollback wave (pre-audit) left zero content defects:
   the recovery walked the r343 playbook in full; the round's own
   hygiene incident (the tooling-mode corruption, 2,997 files
   realigned back to the git index modes within the round) is
   recorded in the worklog — the tree is bit-exact to the index again
   (54 root-owned upload/ strays excepted, EPERM-class, cosmetic).

## 5. Status vocabulary mapping (the three doc generations)

The part11-era matrix labels (PASS/PARTIAL/FAIL/BLOCKED/NOTAPPLICABLE)
and the part15-era audit labels (VERIFIED COMPLETE / SOURCE
IMPLEMENTED — EXECUTION BLOCKED) predate part16's §4.3 vocabulary. The
mapping, pinned here so the three docs stay coherent:
PASS ↔ CONNECTED_AND_TESTED ↔ IMPLEMENTED_AND_TESTED · PARTIAL ↔
PARTIAL/CONNECTED_UNTESTED per dimension · FAIL ↔ MISSING/BROKEN ·
BLOCKED ↔ BLOCKED_ENV · NOTAPPLICABLE ↔ BACKEND_ONLY_INTERNAL
(documented internal). The matrix itself is not rewritten wholesale —
its §21 spec defined its own labels; this audit is the part16 layer
over the same evidence.

— audited round 344 (the 136th recorded run, the part16 opening;
the Phase-5 row re-trued round 345 — the 137th recorded run, the
lane delivered; rows 12 + 14 re-trued round 349 — the 141st recorded
run, the experiment-tracking sub-lane delivered); the companion docs: CAPABILITY_INTEGRATION_MATRIX
(part11 §21), COMPREHENSIVE_REPOSITORY_AUDIT (part15 §13),
TEST_EXECUTION_REPORT (the executed table), IMPROVEMENT_ROADMAP
(the campaign queue).
