# ROUND 372 PREVIOUS-TASK REVALIDATION (MASTER_PROMPT part21 / Round 372 — Phase-1)

Date: 2026-11-27, ~21:07 UTC (the revalidation session's own `date -u` reading).
Revalidated terminal state: `main` @ `e8b565f69472611fe9cb444961df188848b075e7`
(VERSION **1.2.244**), worktree clean, local `main` == `origin/main`; the remote
carries exactly one branch (`main`); zero open PRs.

**Scope** — the previous audit to revalidate is the **part20 / R368–R371 cycle**
(the prior repository-wide audit + remediation + the v1.2.244 release): the r368
part19 13-phase comprehensive audit, the r369 recovery/absorption, the r370 part20
repository audit (the six deliverables), and the r371 release cycle that closed
D-15. Its claims live in `docs/QA/FULL_REPOSITORY_AUDIT.md`,
`docs/QA/BACKEND_FRONTEND_TRACEABILITY.md`, `docs/QA/WINDOWS_EXE_MSI_AUDIT.md`,
`docs/QA/DEFECT_REGISTER.md` (D-9..D-15), `docs/QA/TEST_EXECUTION_REPORT.md`
(§§0–1), `docs/RELEASE_READINESS.md` (r370 §415–440 / r371 §442–465),
`CHANGELOG.md` (the `[1.2.244]` lead), and `worklog.md` R368–R372
(worklog.md:20223, :20225, :20229, :20235, :20241).

**Method** — every row below was re-derived live this session: the GitHub API
(authenticated; the token read from `.credentials/github_pat` into a shell
variable, never printed), the local tree, and **collection-only** re-derivations
(`pytest --collect-only`, an `create_app().openapi()` derivation). **No tests,
no builds, no commits, no pushes, no branches were made** (the part21
revalidation mandate); no historical record was modified — this file is a new
addendum only. Where a prior round's *executed* run is cited, it is labeled as
carried evidence, not re-executed.

## 0. Revalidated terminal state (re-derived live)

| Item | Evidence (this session) | State |
|---|---|---|
| HEAD | `git rev-parse HEAD` = e8b565f6…; `git status --porcelain` empty; log: `e8b565f6` (r372 dev-server addendum) ← `9e06ba33` (r372 close-out) ← `7057f791` (r371 close-out) ← `5a48d6cf` (the release build commit) | clean, synced |
| Branches (remote) | API `repos/…/branches` → exactly `main` @ e8b565f6 | one branch |
| Open PRs | API `pulls?state=open` → 0 | none |
| Version carriers | `VERSION`=1.2.244; `pyproject.toml:7`; `src/pharmaml/__init__.py:52`; `src-tauri/Cargo.toml:3` (with the D-10 alignment comment); `src-tauri/tauri.conf.json` version; `frontend/package.json` version | all 1.2.244 |
| Release v1.2.244 | API `releases/latest`: tag v1.2.244, `published_at 2026-11-27T18:23:04Z`, `draft:false`; five assets `state=uploaded`: EXE 558,700,797 B (id 801570375) / MSI 711,290,358 B (id 801570373) / RELEASE_MANIFEST.json 510 B / sbom.cdx.json 23,625 B / SHA256SUMS.txt 368 B | PUBLISHED, intact |
| Tag → commit | API `git/ref/tags/v1.2.244` → tag object `bb2a20ede7734f45ed4acfc2adc4301301f5f6c0` → deref `git/tags/bb2a20ed…` → commit **5a48d6cf6cf6ee796071a83620518a5e6eb57530** | consistent |
| Release manifest | fetched `RELEASE_MANIFEST.json`: `"commit": "5a48d6cf…"`, `"version": "1.2.244"`, artifacts EXE 558,700,797 B / MSI 711,290,358 B, `webview2: "offlineInstaller (bundled, no online bootstrapper)"` | manifest==tag==VERSION |
| CI at HEAD | check-runs API at e8b565f6: **total 19, all completed/success** (14 ci.yml jobs + 5 CodeQL `Analyze` checks); run 46066722332 (CI) and run 46066722216 (Push on main) both completed/success | green |
| Post-release source delta | `git log 5a48d6cf..e8b565f6` = 3 commits; `git diff --stat` = 4 files (`docs/QA/TEST_EXECUTION_REPORT.md`, `docs/RELEASE_READINESS.md`, `scratch/100runs-counter.json`, `worklog.md`), +15/−9 | **zero shipped-source movement** — the installers stay source-current |
| Six-home zip parity | sha256 recomputed this session on `download/`, `dist-archive/`, `upload/`, `pharma-ml-toolkit/dist-archive/`, `pharma-ml-toolkit/download/` + `~/pharma-ml-toolkit-zips/`: all `f8a10d681a94602948aa93a7c205e367c4f96a04ef35fd556a90c5ffc6294ef4` | byte-identical, v1.2.244 |
| Ledger | `scratch/100runs-counter.json`: `runs_completed: 165`, last_run round 372 | consistent with worklog R372 |

## 1. The revalidation matrix (the previous audit's requirement rows)

The part20 acceptance criteria are the r370 gate table
(`RELEASE_READINESS.md:415–440`: PASS/FAIL/BLOCKED vocabulary, "no PASS on code
inspection alone where runtime verification is required", no P0/P1 open) and
`FULL_REPOSITORY_AUDIT.md` §10 (the readiness table). Classifications per the
part21 vocabulary: VERIFIED / PARTIALLY VERIFIED / NOT COMPLETED / BLOCKED /
OUTDATED. **Counts: 12 VERIFIED, 2 PARTIALLY VERIFIED (rows 13 and 14), 0 NOT
COMPLETED, 0 BLOCKED-as-failure, 0 OUTDATED rows** (outdated *claims* inside
preserved historical docs are named in rows 1 and 14 and in §2).

### Row 1 — Full repository audit (the part20 scope) — **VERIFIED**

* Prior claim: FULL_REPOSITORY_AUDIT.md (the coverage matrix §3 + readiness §10),
  RELEASE_READINESS.md:415–440, worklog.md:20229 (R370).
* Re-derived: the six part20 deliverables exist and their gates were re-checked
  at the *current* terminal, not the r370 one — CI 19/19 check-runs green at
  e8b565f6 (the §0 table); D-9's chemistry lane green in CI run 46066722332 at
  e8b565f6 (the exact lane that failed at 46016289515 pre-fix); D-10 verified
  (`src-tauri/Cargo.toml:3` — version 1.2.244 with the alignment comment); D-14
  verified (the 142/149 census re-derived live this session, row 4); D-15 closed
  by the r371 cycle (rows 7/8 evidence); the audit's honest BLOCKED legs
  re-confirmed unchanged (§2 below). The audit's own claims hold at the terminal
  state, with the r370-era docs standing as round-scoped historical records per
  the repo's "status truing, not history rewriting" convention.

### Row 2 — Backend–frontend integration (runtime consumer paths real?) — **VERIFIED**

* Prior claim: BACKEND_FRONTEND_TRACEABILITY.md §2 ("every operation has a
  frontend consumer reference and a backend test lane"; the wired end-to-end
  ladder: React control → hook/service → typed client → token REST/WS → FastAPI
  route → science module → DTO → typed response → state → rendered result).
* Re-derived: `frontend/src/services/api/client.ts:57/:87/:115` — the
  `X-Session-Token` header on every REST leg (JSON/FormData/GET lanes verified by
  direct read); 20 service dirs under `frontend/src/services/` (api, websocket,
  ml, dl, docking, generative, …); the traceability doc's consumer citations
  spot-verified in source: `services/ml/client.ts` AD `/analyze` (~:2157) +
  `/leverage-williams` (~:2188) both called; `services/ml/unsupervised.ts:335/
  :384/:417` and `services/generative/client.ts:271` (the D-13 methods) exist as
  typed wrappers. Regression: the 16-file drift-guard family
  (`tests/test_frontend_*.py`, 202 tests at the r371/r372 records) + vitest
  357/357 executed green in CI at HEAD (the "Studio frontend" job of run
  46066722332; `ci.yml:34–66` runs `npm --prefix frontend run build` +
  `npm --prefix frontend test` + the contract guards). Runtime: install-smoke
  run 46049348450 steps 7–8 (the installed sidecar's live
  handshake/health/guard/shutdown + the host lifecycle legs) — success.
* Verdict: the consumer paths are real in source, pinned by tests, and exercised
  on the runner. VERIFIED.

### Row 3 — All nine workspaces coverage — **VERIFIED**

* Prior claim: BACKEND_FRONTEND_TRACEABILITY.md §3 (the nine-workspace
  coverage statement, zero placeholder controls).
* Re-derived: `frontend/src/features/` contains **exactly nine** workspaces
  (datasets, dl, docking, generative, introduction, md, ml, pkpd,
  visualization); the backend side: **32 route modules**
  (`src/pharmaml/api/routes/` = 33 files incl. `__init__.py`) mounted at
  `/api/v1`; every workspace family maps onto the route families (r368 phase4 +
  r370 re-checks, carried; the service-grep sweep re-cited in row 2). VERIFIED.

### Row 4 — API contract consistency (OpenAPI vs routes vs generated types vs consumers) — **VERIFIED**

* Prior claim: r368 phase3 byte-level reconciliation + D-14's corrected census
  (149 operations / 142 paths); `api.d.ts` regenerates identically.
* Re-derived **live this session** (read-only derivation, no test lane):
  `create_app().openapi()` → **142 paths / 149 operations** (54 GET / 89 POST /
  4 DELETE / 1 PUT / **1 HEAD**); vs committed `contracts/openapi.json` (142
  paths): path sets equal **True**, path→methods equal **True**, operationIds
  equal **True** (149) — the runtime==contract identity holds at the current
  tree (the r368 byte-level normalization identity stands as the deeper
  historical proof). Generated types: `frontend/src/types/api.d.ts` (9,959 lines
  / 277,497 B) is produced by `bun run generate:types` =
  `openapi-typescript ../contracts/openapi.json --output src/types/api.d.ts`
  (`frontend/package.json` scripts) — generation, never hand-edited (the r368
  regeneration-identity proof carried). Sync enforcement: `tests/api/
  test_contracts.py` (6 tests), the 16-file drift-guard family, and the CI
  contract guards (`ci.yml:73–78`). VERIFIED.

### Row 5 — Tauri lifecycle (source + available runtime evidence) — **VERIFIED**

* Source re-read: 5 Rust files, 598 lines total (`backend_manager.rs` 460 /
  `commands.rs` 20 / `lib.rs` 58 / `main.rs` 8 / `security.rs` 52);
  `lib.rs:52` — `if let RunEvent::Exit` → `backend_manager::terminate` (the only
  termination hook; the D-11 basis); `commands.rs` — exactly ONE IPC command
  `backend_endpoint` (no restart command exists; the D-12 basis);
  `backend_manager.rs:96` `PHARMAML-READY` prefix, `:113` `validate_handshake`
  (port≠0, token 16..=128 urlsafe, pid≠0, version shape), the loopback port-0
  bind doc at `:14`; `tauri.conf.json` CSP `connect-src ipc:
  http://ipc.localhost http://127.0.0.1:* ws://127.0.0.1:*` — loopback-only.
* Runtime evidence available: install-smoke run 46049348450 step 7 ("Verify the
  INSTALLED sidecar (handshake + health + guard + shutdown)") + step 8 ("Launch
  the installed host + the lifecycle legs") — both success over the published
  installers.
* Honest toolchain note: **no `cargo`/`rustc` in this sandbox** (`command -v` →
  empty, re-verified this session) — the D-11/D-12 fixes cannot be
  compile-verified here; the D-4/D-5 precedent (never mark fixed without
  `cargo test`) is honored. VERIFIED (source + runner evidence; the toolchain
  limitation is exactly the prior audit's own honest boundary).

### Row 6 — SQLite/FileVault persistence (compat + regression evidence) — **VERIFIED**

* Source: `src/pharmaml/api/persistence.py` — SQLite pragmas on every
  connection (:63), `class FileVault` (:225, atomic temp+`os.replace` writes
  referenced from SQLite), the SQLite+vault pair (:311); backend-owned, no
  destructive migration (r370 audit carried; no migration code added since — the
  post-release delta touches zero backend files).
* Tests: `tests/api/test_persistence.py` (30 collected) +
  `tests/api/test_persistence_database.py` (37) = **67 collected** this session
  — the exact census behind the executed "66 passed + 1 skipped" (the documented
  ephemeral-mode skip) of r370 (TEST_EXECUTION_REPORT.md:32) and the r368
  phase8 backup-restore round-trip legs. Not re-executed this session (the
  no-tests mandate); the collection identity + the carried executions stand.
  VERIFIED.

### Row 7 — EXE source consistency (release manifest vs source commit 5a48d6cf) — **VERIFIED**

* API evidence (all fetched this session): the tag chain `v1.2.244` →
  `bb2a20ed…` → commit **5a48d6cf…**; the release's
  `RELEASE_MANIFEST.json` `"commit": "5a48d6cf…"`, EXE artifact bytes
  558,700,797; the asset `PharmaML_Studio_Setup_v1.2.244.exe` = 558,700,797 B,
  `state: uploaded`; the windows-release run 46043111328 `head_sha: 5a48d6cf`,
  `conclusion: success` (event: push, branch: v1.2.244); the install-smoke run
  46049348450 `head_sha: 5a48d6cf`; `SHA256SUMS.txt` EXE digest
  `9e796a5e9713ed545022e974f46c280def5c6b51908ab6a9ce6d8444c2ee1fae`.
  (The release's `target_commitish` field records the string `main` — the
  created-at label; the authoritative provenance is the tag→commit chain + the
  manifest, which agree.)
* Post-release delta: 3 commits / 4 files / +15−9 — **QA docs + ledger only**
  (row 0) — no shipped source moved after the freeze, so the D-15 closure
  (r371) still holds: the installers carry the full r368→r371 delta and nothing
  newer exists to miss. VERIFIED.

### Row 8 — MSI source consistency — **VERIFIED**

Same lane as row 7: the manifest's MSI artifact 711,290,358 B == the API asset
`PharmaML_Studio_v1.2.244.msi` (id 801570373, 711,290,358 B, uploaded); the
same run 46043111328 built both from 5a48d6cf; `SHA256SUMS.txt` MSI digest
`f626ff1a7c69a2e3a6886fad419e80519832d7b2ddb66d044e399aee1b77b0e6`. VERIFIED.

### Row 9 — Windows installation smoke (run 46049348450) — **VERIFIED**

* API: run 46049348450 "Windows Install Smoke"
  (`.github/workflows/windows-install-smoke.yml`, workflow_dispatch,
  head_sha 5a48d6cf): `status: completed`, `conclusion: success`, created
  18:25:23Z → updated 18:33:30Z; its single job "Clean-runner install smoke (the
  BLK-1 CI lane)" completed/success with **15/15 steps success**: checkout of
  the release tag; SHA256SUMS verify; manifest cross-check (version, commit,
  bytes); Defender scans of both installers; MSI silent install + installed-tree
  verify; the INSTALLED sidecar live probes (handshake + health + guard +
  shutdown); MSI lifecycle + uninstall; NSIS install + lifecycle + uninstall;
  the upgrade leg; the evidence-bundle upload; the honest-report step. VERIFIED.

### Row 10 — Upgrade and data retention (1.2.243 → 1.2.244) — **VERIFIED**

* Step 13 of run 46049348450 — "The upgrade leg (previous release → current,
  user-data retention)" — **success**; the r371 record (RELEASE_READINESS.md:458,
  TEST_EXECUTION_REPORT.md:19): the upgrade **1.2.243 → 1.2.244** with the
  user-data marker retained, dispatched with `previous_tag v1.2.243`. The
  retention rules are pinned in the workflow itself
  (`windows-install-smoke.yml:495–503` MSI, `:762–790` NSIS — the user-data
  retention rule recorded in the evidence bundle, the lane cleanup explicitly
  distinct from app behavior).
* Honest bound: this is hosted-runner evidence (windows-2022); a
  physical-machine upgrade rides the clean-machine lane (row 13/§2). VERIFIED
  as far as the prior audit claimed it.

### Row 11 — Offline support (dependencies + startup behavior) — **VERIFIED**

* Dependencies: `packaging/windows/pharmaml_backend.spec` — Qt/PySide6 +
  pytest/dev tools excluded (`:98–110`), `datas` carry engines/, LICENSE,
  VERSION, meeko (`:82+`) — the full offline scientific stack frozen;
  `windows-release.yml:304–307` — `webviewInstallMode offlineInstaller` embeds
  the WebView2 Evergreen OFFLINE installer into the NSIS package (no network
  fetch at install time); the manifest's `webview2: "offlineInstaller (bundled,
  no online bootstrapper)"` (row 0).
* Startup behavior: the backend binds loopback-only on a port-0 socket it owns
  (`backend_manager.rs:14` doc; handshake via local stdout `PHARMAML-READY`);
  the CSP `connect-src` has zero remote origins (row 5). No undocumented remote
  dependency in the freeze spec or host code (r370 finding re-verified by
  direct read this session). VERIFIED (construction + runner evidence; a live
  offline desktop execution is impossible in this Linux sandbox — named, not
  fabricated).

### Row 12 — CI regression coverage (latest CI runs + job list at current HEAD) — **VERIFIED**

* At **e8b565f6**: run 46066722332 (CI, event push) — completed/success with 14
  jobs, all success: Python 3.10 / 3.11 / 3.12, Studio MD, **Chemistry and
  docking preparation** (the D-9 lane), GitHub page, scikit-learn 1.3.0 /
  1.5.2 / 1.8.0, Web console, **Studio frontend** (vitest + production build +
  contract guards), Windows bundled engine smoke, Generative SA, Windows
  docking integration (synthetic); plus run 46066722216 (Push on main) —
  success. The commit's check-runs: **19/19 success** (14 CI + 5 CodeQL
  `Analyze`).
* Also green at the two intermediate HEADs: 9e06ba33 (runs 46066374871 /
  46066379718) and 7057f791 (runs 46057537033 / 46057537017) — the whole
  post-release chain is green. The D-9 fix is proven end-to-end remotely (the
  lane that failed at run 46016289515 pre-fix now succeeds at every post-fix
  HEAD). VERIFIED.

### Row 13 — Security findings (the outstanding defect register) — **PARTIALLY VERIFIED**

* The register's states were re-verified accurate one by one (nothing silently
  closed — the honesty itself checks out): D-11/D-12 OPEN compile-blocked
  (source unchanged: no Job Object / PDEATHSIG anywhere in `src-tauri/src/`
  (grep zero hits), `commands.rs` still has no restart IPC); signing BLOCKED
  certificate-gated (the installers unsigned, SHA256SUMS/manifest/SBOM present —
  row 0); clean-machine NOT_RUN owner-side
  (`scripts/windows/validate-clean-machine.ps1` exists, 25,295 B, unexecuted);
  the local penetration API suite stands (tests/api/
  test_penetration_security.py: **67 collected** this session — the exact
  census behind the r370/r371/r372 green executions); the conformal GUI gap is
  still real (grep over `frontend/src/features/` + `frontend/src/app/`: zero
  conformal consumers; only the generated `api.d.ts` mentions it); the 10
  API-only routes stand (BACKEND_FRONTEND_TRACEABILITY.md §2).
* **The unresolved lane**: penetration run 46038496882 (head_sha a803508a,
  workflow_dispatch) is **`in_progress`, `conclusion: None`**, last API update
  **2026-11-27T17:01:51Z**; its job "ZAP API scan (isolated ephemeral target)"
  shows the ZAP scan step in flight with the severity gate / report / stop
  steps still pending. As of this session's 21:07:42Z clock that is **~4h06m
  with no conclusion** — an **in-limbo lane: never counted green, never closed**.
  The r371 close-out's "its conclusion records in the next ledger touch"
  promise is unfulfilled (r372's touch also recorded it still in_progress).
* Verdict: PARTIALLY VERIFIED — the register is accurate and honest, but the
  security campaign carries a live unresolved lane, so the row cannot be
  counted complete. The deep scan stays on the standing Monday cron
  (`penetration-test.yml:44` — `cron: "30 4 * * 1"`, i.e. 2026-11-30).

### Row 14 — Documentation current (reports describe current revision?) — **PARTIALLY VERIFIED**

* Current: `RELEASE_READINESS.md` r371 section (lines 442–465) — trued to the
  final outcomes by 7057f791 (D-15 CLOSED, the four-attempt ladder, the
  install-smoke result, the in-flight penetration lane honestly recorded);
  `TEST_EXECUTION_REPORT.md` §0 (r371, lines 11–21) — the executed-vs-carried
  ledger with the BLOCKED legs named; `CHANGELOG.md` `[1.2.244]` r371 entry at
  line 1 ("the v1.2.244 NEW-version ship REQUIRED by the r370 D-15 gate");
  the version carriers + the six-home zip pins (row 0) all describe the current
  revision 1.2.244 at e8b565f6.
* **The one genuine gap**: `DEFECT_REGISTER.md`'s D-15 row (line 24) still reads
  "**OPEN (release-gated)**" — the register's last update is the r370 part20
  deliverables commit `6b8c0ae3` (`git log -3 -- docs/QA/DEFECT_REGISTER.md`);
  the r371 cycle that closed D-15 (per CHANGELOG, RELEASE_READINESS, and the
  TEST_EXECUTION_REPORT ladder row) never converted the standing register's
  row. The register is the one ledger out of date — a doc-debt finding of this
  revalidation, not a release defect (the closure is triple-recorded
  elsewhere).
* Round-scoped docs (ROUND368_* , FULL_REPOSITORY_AUDIT.md,
  WINDOWS_EXE_MSI_AUDIT.md, BACKEND_FRONTEND_TRACEABILITY.md) describe the
  v1.2.243-era audit state **by design** (historical records preserved per the
  repo convention); their superseded specifics — the r368 matrix's 148-census
  (superseded by D-14's 149) and WINDOWS_EXE_MSI_AUDIT's D-15-OPEN sections
  (superseded by the r371 closure) — are OUTDATED-as-claims but correctly
  preserved as history; a reader chains to the r371 sections for the terminal
  state. Verdict: PARTIALLY VERIFIED (the standing register's D-15 row stale;
  everything else current or intentionally historical).

## 2. Reconciled outstanding findings (current honest status)

| Finding | Current status (evidence this session) |
|---|---|
| **D-11 — orphaned sidecar after a Tauri host crash** | **OPEN, compile-blocked.** Source unchanged: termination only on `RunEvent::Exit` (`src-tauri/src/lib.rs:52`); zero Job Object / PDEATHSIG references in `src-tauri/src/` (grep). No `cargo`/`rustc` in this sandbox (re-verified) — the fix cannot be compile-verified here; never marked fixed (D-4/D-5 precedent). The runner's graceful-shutdown evidence (install-smoke step 7) covers the graceful path only, not the host-crash path. |
| **D-12 — startup retry cannot recover a dead backend** | **OPEN, compile-blocked.** `frontend/src/app/StartupGate.tsx:86` `retry` / `:146–:149` "Retry the startup handshake" re-arms the connection against the stale endpoint; `commands.rs` still has exactly one IPC command (`backend_endpoint`) — no respawn path exists. Recovery remains an app relaunch. |
| **Authenticode signing** | **BLOCKED — certificate-gated** (unchanged, honestly). The v1.2.244 installers are unsigned; no certificate exists or was fabricated. Integrity stands independently: SHA256SUMS.txt (EXE `9e796a5e…`, MSI `f626ff1a…`, manifest `60e34baa…`, SBOM `f5452580…`), RELEASE_MANIFEST.json (commit 5a48d6cf), the SBOM (23,625 B). |
| **Physical clean-machine validation** | **NOT_RUN — owner-side.** `scripts/windows/validate-clean-machine.ps1` (25,295 B) delivered and unexecuted; runner evidence (install-smoke) is never conflated with clean-machine evidence. The Win10/Win11 physical matrix and paths-with-spaces installs ride this lane. |
| **The ZAP penetration lane** | **UNRESOLVED / in-limbo.** Run 46038496882 `in_progress`, `conclusion: None`, last updated 2026-11-27T17:01:51Z (~4h06m stale at this revalidation); the ZAP scan step in flight, severity gate + report + cleanup steps pending. No conclusion has ever been recorded — the lane is neither green nor closed and is **never counted as passed**. The weekly deep scan rides the standing Monday 04:30 UTC cron (`penetration-test.yml:44`, next 2026-11-30). The 67-test local penetration API suite is green in the r370/r371/r372 records (collected 67 this session). |
| **GUI coverage gaps in the API traceability matrix** | **Still real.** The conformal family (`POST /model-evaluation/conformal-regression`, `/conformal-classification`, `/coverage-diagnostics`) has **zero runtime GUI consumers** (grep this session: no hits in `frontend/src/features/` or `frontend/src/app/`; only the generated `api.d.ts` mentions conformal). Backend-complete (tests/api/test_model_evaluation.py, 23 conformal refs per the r370 audit); the recommendation (a Model Evaluation stage in the ML workspace) remains deferred — never silently claimed as covered. |
| **API-only capabilities lacking UI** | **Stands at 10 documented routes** (BACKEND_FRONTEND_TRACEABILITY.md §2: capabilities ×2, AD catalog/knn/domain-box, model-evaluation catalog, conformal ×3, workflow run-detail) — each with its documented reason; the census behind them re-derived live (142/149, row 4). D-13's four dead service methods remain documented-kept typed-surface parity (`unsupervised.ts:335/:384/:417`, `generative/client.ts:271` — verified present, unreferenced). |
| *(reconciled additionally)* **D-15 — installers lag main** | **CLOSED at r371** — v1.2.244 built from 5a48d6cf with the full r368→r371 delta (rows 7/8); the post-release delta is 3 commits of QA-docs/ledger only (zero shipped code), so no new staleness exists. *Doc debt:* the DEFECT_REGISTER row (line 24) still says OPEN — see row 14. |
| *(reconciled additionally)* **D-9 / D-10 / D-14** | All verified at the terminal: D-9 — the chemistry lane green in CI runs 46057537033 (@7057f791) and 46066722332 (@e8b565f6); the re-aimed assertion reads at `tests/test_docking_prep.py:1105–1113`. D-10 — `src-tauri/Cargo.toml:3` = 1.2.244. D-14 — the 142/149 census re-derived live (row 4). |

## 3. CI and release evidence (plain run IDs / URLs)

* **Release v1.2.244** — https://github.com/MrMMA2004/pharma-ml-toolkit/releases/tag/v1.2.244
  (published 2026-11-27T18:23:04Z, draft=false): EXE 558,700,797 B, MSI
  711,290,358 B, RELEASE_MANIFEST.json 510 B, sbom.cdx.json 23,625 B,
  SHA256SUMS.txt 368 B — all `state: uploaded`, sizes exact per the API poll.
* **Windows Release (attempt 4)** — run 46043111328, completed/success on
  5a48d6cf: https://github.com/MrMMA2004/pharma-ml-toolkit/actions/runs/46043111328
  (the r371 four-attempt ladder: 46035752426 fail / 46038529441 fail /
  46040893195 fail / 46043111328 success — recorded in TEST_EXECUTION_REPORT.md:19).
* **Windows Install Smoke** — run 46049348450, completed/success on 5a48d6cf,
  15/15 steps incl. the upgrade leg:
  https://github.com/MrMMA2004/pharma-ml-toolkit/actions/runs/46049348450
* **Penetration lane** — run 46038496882, **in_progress / no conclusion**:
  https://github.com/MrMMA2004/pharma-ml-toolkit/actions/runs/46038496882
* **CI at the current HEAD e8b565f6** — run 46066722332 (CI, 14/14 jobs)
  https://github.com/MrMMA2004/pharma-ml-toolkit/actions/runs/46066722332 and
  run 46066722216 (Push on main):
  https://github.com/MrMMA2004/pharma-ml-toolkit/actions/runs/46066722216 —
  both completed/success; the commit's check-runs: 19/19 success.
* **CI at the intermediate HEADs** — 9e06ba33: runs 46066374871 / 46066379718;
  7057f791: runs 46057537033 / 46057537017 (all completed/success).
* **Zip parity** — sha256 `f8a10d681a94602948aa93a7c205e367c4f96a04ef35fd556a90c5ffc6294ef4`
  recomputed this session across all six homes (root `download/`, root
  `dist-archive/`, `upload/`, `pharma-ml-toolkit/dist-archive/`,
  `pharma-ml-toolkit/download/`, offsite `~/pharma-ml-toolkit-zips/`) — the
  r371/r372 ledger's exact digest, v1.2.244 / 28,121,913 B.
* **Re-derived collections** (this session, `--collect-only`, no tests run):
  default suite **8,286 tests / 442 files** (== the r371 pin exactly);
  `tests/api` **753** (== the r368 family count); penetration **67**;
  persistence **67** (30+37).

## 4. Exact limitations

1. **No tests or builds were executed this session** (the part21 revalidation
   mandate): the executed-gate evidence above rides the r368–r372 records plus
   the live green CI at HEAD; only `pytest --collect-only` counts and the
   `create_app().openapi()` derivation were re-derived locally.
2. **No Rust toolchain in this sandbox** (`cargo`/`rustc` absent, re-verified)
   — D-11/D-12 stay compile-blocked; the fresh-build leg stays
   sandbox-impossible (no Windows infrastructure).
3. **The ZAP lane is unresolved** — run 46038496882 has produced no conclusion;
   it is classified in-limbo, never green, never closed. Its eventual outcome
   must be recorded in the next ledger touch (the standing promise, still
   unfulfilled).
4. **Signing stays BLOCKED (certificate-gated); the clean-machine lane stays
   NOT_RUN (owner-side); the physical Win10/11 matrix stays untested** — none
   conflated with the hosted-runner evidence.
5. **The web-console dev daemon was not up at revalidation time** (GET /
   localhost:3000 → 000 at ~21:07Z) — the render-lane evidence rides the r372
   close-out's recorded GET / = 200 and the green "Web console" CI job at HEAD
   (run 46066722332). This affects no toolkit row above (the desktop app's
   runtime evidence is the runner lane, not the sandbox console).
6. **A local-only branch ref** `feat/shared-ml-dl-evaluation-visualizations`
   exists at exactly `e8b565f6` (created 2026-11-27 20:53:58 from HEAD;
   zero commits ahead of main; no remote counterpart — the remote carries only
   `main`). Harmless (zero divergence), recorded here for the ledger; no action
   taken per the no-mutation rule.
7. **The DEFECT_REGISTER's D-15 row is stale** vs the r371 terminal state (row
   14) — a documentation debt named by this revalidation; not re-trued here
   (this file is an addendum; no historical record was modified).
8. **History preserved**: this revalidation wrote exactly one new file; no
   commit, push, branch, release, or asset was touched; the GitHub token was
   read from `.credentials/github_pat` into a shell variable and used only in
   curl headers, never printed.

## 5. Summary verdict

The part20/R368–R371 previous-audit cycle **revalidates cleanly at the current
terminal state**: 12 of the 14 requirement rows VERIFIED (including the
release-provenance rows 7–10, whose manifest/tag/run SHAs all agree on 5a48d6cf,
and the contract row 4, whose runtime==contract identity was re-derived live),
2 PARTIALLY VERIFIED — row 13 (the honest register carries one unresolved
in-limbo ZAP lane) and row 14 (one stale standing-register row: D-15). The
honesty claims of the prior audit all held under re-derivation: no blocked leg
was counted green, no compile-blocked fix was marked fixed, and the outstanding
findings (D-11, D-12, signing, clean-machine, the ZAP lane, the conformal GUI
gap, the 10 API-only routes) remain exactly as registered — with the ZAP run
46038496882 now the single live thread awaiting a conclusion.
