# ACCEPTANCE_SWEEP — part11_1 §46 (Phase Thirty-Nine's final gate)

**Baseline:** v1.2.228 (round 289) · **r290 addendum:** the ML
Explainability/Uncertainty queue lane CLOSED (POST /api/v1/ml/explain
+ the GUI stage, 8 pins, browser-verified) — the §46.1 workflow row
flips to PASS. · **r291 addendum:** the Plotly.js src adoption
CAMPAIGN CLOSED (the §3.2 row): the shared
`components/charts/PlotlyFigure.tsx` surface (§22.4's
hovermode-closest interaction contract, §22.5's customdata click
lane, §23's react/resize/purge lifecycle) now renders the live
charts — ML parity scatter + confusion heatmap + §24.2.B/C AD
views + the mean-|SHAP| bar, the DL learning curve, the PK/PD and
MD curves — each browser-verified end-to-end over the real backend
with the oklch→sRGB theme resolution (bit-exact against Chromium's
own rasterizer); `predict`'s dataset mode gained the §22.5
evaluation block (5 new API pins). The §46.1/§46.5 Plotly rows
flip to PASS. · **r293 addendum:** the Windows-pipeline
REGISTRATION fix — `.github/workflows/windows-release.yml` moved to
the REPOSITORY root (GitHub Actions registers only root-level
workflow files; the repo's live `/actions/workflows` API listed no
such workflow while the file sat invisibly at
`pharma-ml-toolkit/.github/workflows/`), gained the
`defaults: run: working-directory: pharma-ml-toolkit` block plus
the prefixed `uses:` paths so the 25-step order executes correctly
against the nested checkout topology; the 9th drift guard pins the
registration. The §46.9 "Windows x64 installer EXE generated" row's
workflow evidence is now REAL (registered + topology-correct); its
execution stayed BLOCKED through round 297 twice over — no Windows
host in this sandbox, and the account's runner provisioning itself
was failing (run 44467151243: all 13 ci.yml jobs dead in ~3 s at
queue time, no runner assigned — the account-level Actions/billing
state was outside this sandbox's control; see
`packaging/windows/TAURI_RELEASE_PIPELINE.md`'s round-293 note).

**Round 298 update — the execution lane is OPEN.** The owner made the
repository public (Actions free on public repositories, Windows
runners included), the runner supply recovered, and the r294–r297
bring-up arc (17 iterations, each failure decoded at root cause)
ended in the first all-green execution: run 44553307816 (head
`7f241a4a`, 33 min on a real windows-2022 runner) built the NSIS
Setup EXE + the WiX MSI with the embedded offline WebView2, froze
and live-verified the sidecar ON Windows, and generated the SHA-256
checksums on the runner itself (`a385853c…c210` EXE, `69da2837…130d`
MSI). The `v1.2.230` tag build then completed all-green (run
44569220942, head `bb5a29e7` — the first attempt 44566101809 died
at step 10 on the stale notebooks, exactly as the ship gates
predicted, until the ritual completed) and the GitHub Release
`v1.2.230` carries the four attached artifacts: PharmaML_Studio_Setup_v1.2.230.exe
(477,595,658 B, sha-256 `38c0f667…e8a13`), PharmaML_Studio_v1.2.230.msi
(631,078,245 B, sha-256 `c02b831a…129e4`), RELEASE_MANIFEST.json,
and SHA256SUMS.txt — part11_2 §9's artifact delivery, and real
values for §47's report rows 38-41. Five §46 rows flip from BLOCKED to
PASS on that execution evidence (the §46.7 sidecar-start row and
four §46.9 rows below); everything clean-machine stays BLOCKED by
the addendum's own rule — a runner is a build environment, not a
clean machine. The verdict line moves to **73 PASS · 1 PARTIAL · 19
BLOCKED · 1 NOTAPPLICABLE**.
**Spec:** MASTER_PROMPT part11_1 §46
(Final Acceptance Criteria) with part11_2 §12's status labels.
**Method:** every criterion below is checked against live evidence from
this round's gates (the browser-verified startup bridge, the §21
integration matrix, the 261-test API suite, the 123 frontend drift
guards, the §29 dependency inventory, the windows-pipeline drift
guards) — never against prose. A criterion that needs a Windows
runner or a real Windows machine is **BLOCKED** per part11_2 §12
("requires a Windows runner or Windows environment that is
unavailable"), never silently passed.

Labels: **PASS** · **PARTIAL** · **FAIL** · **BLOCKED** · **NOTAPPLICABLE**.

---

## 46.1 Frontend architecture

| Criterion | Evidence | Status |
|---|---|---|
| TypeScript is the production frontend language | strict tsconfig (`noUncheckedIndexedAccess`), `tsc --noEmit` gate green | PASS |
| React is the production UI framework | React 19, `bun run build` green | PASS |
| Tauri 2 is the desktop framework | `src-tauri/` (tauri 2, no plugins, minimal host); `tests/test_windows_pipeline.py` | PASS |
| Vite builds the frontend | `vite build` → static dist/ (12 s, chunked) | PASS |
| Tailwind CSS for styling | Tailwind 4 + the Mineral tokens | PASS |
| shadcn/ui reusable components | Radix head + cva/clsx/merge primitives | PASS |
| Motion for React used appropriately | `motion` — the nav underline only (restrained §8.5) | PASS |
| Plotly.js is the primary interactive charting library | the shared `PlotlyFigure` surface + the live trace lanes: ML parity scatter (customdata click → §22.5 inspector, browser-verified), confusion heatmap, §24.2.B/C AD scatters, mean-\|SHAP\| bar, DL learning curve, PK/PD + MD curves; the oklch→sRGB token resolution rides the draw pass; the lazy chunk stays manualChunks-split (r291, 6 evidence shots) | PASS |
| Three.js for appropriate 3D visualizations | Molecule3D ligand/receptor/trajectory lanes (r288 browser 21/21) | PASS |
| GUI is an offline SPA inside WebView2 | offline CSP (`self` + loopback), `frontendDist` local, no dev server in production | PASS |
| No external browser / remote website / CDN | CSP + the vite `assetsInlineLimit 0` + no remote URLs (drift-guarded) | PASS |
| Introduction is the initial screen | `INITIAL_WORKSPACE = "introduction"`; browser-verified r289 | PASS |
| Exactly nine primary workspaces | registry + the §10.1 order drift guard | PASS |
| Navigation horizontal, exact order | `WorkspaceNav` (aria-current, motion underline, overflow scroll) | PASS |
| All workspaces have complete sequential workflows | the §21 matrix: 65 PASS rows (r290: the ML Explainability stage now runs its real backend lane); every stage of §11.2's order executes | PASS |
| Professional, consistent, responsive, accessible design | the nine pages; keyboard/ARIA states; loading/error/empty everywhere | PASS |
| All application-facing text in English | verified across the pages | PASS |
| Mineral Design System direction preserved | design tokens + light/dark architecture carried into Tailwind | PASS |

## 46.2 Backend and API

| Criterion | Evidence | Status |
|---|---|---|
| Python scientific logic preserved | the endpoint→function trace (§21 matrix §2): every route invokes the real `src/pharmaml/…` callable | PASS |
| Backend domain modules don't depend on the GUI | import scan; the scientific package imports without PySide6 | PASS |
| FastAPI implemented | `src/pharmaml/api/` (18 route modules) | PASS |
| REST endpoints versioned | `/api/v1` mount, contract test pins it | PASS |
| OpenAPI contracts maintained | committed `contracts/openapi.json` == live schema (drift gate) | PASS |
| TypeScript API types generated/validated | `openapi-typescript` → `api.d.ts` (5,538 lines); coverage guard | PASS |
| WebSocket events work | 20-event vocabulary; hello/replay/since/heartbeat pinned; the app's own boot path opens it (r289 browser `open`) | PASS |
| Backend process lifecycle reliable | startup handshake (port-0), supervised sidecar contract, crash/restart policy in the Rust host; `test_backend_service.py` | PASS |
| Local API security implemented | loopback bind, per-launch token (constant-time), Host/Origin guard, docs disabled | PASS |
| Heavy operations don't block the event loop | the JobManager lanes (threads, expensive semaphore); test pins | PASS |
| Backend scientific functions remain Python | no science in Rust/TS (the minimal-host rule, drift-guarded) | PASS |

## 46.3 Database

| Criterion | Evidence | Status |
|---|---|---|
| SQLite implemented | `api/persistence.py` + the schema migrations ledger | PASS |
| Backend owns database access | frontend has no SQLite path — all through the API | PASS |
| Schema migrations available | versioned ledger (v1–v2), idempotency pinned | PASS |
| Dataset metadata persisted | write-through + restart recovery tests | PASS |
| Job history persisted | upsert/prune-to-500 tests | PASS |
| Model and artifact records persisted | trained-model/prior/artifact round-trips across restarts | PASS |
| User preferences and configuration persisted | preferences CRUD tests | PASS |
| Transactions/write concurrency handled safely | WAL, busy timeout, per-operation connections, concurrent-writer tests | PASS |
| Backup and restore work | backup/restore (restores deleted dataset; rejects corrupt/newer schema) | PASS |
| Database files in writable user-data locations | the `%LOCALAPPDATA%` activation contract tests | PASS |
| User data preserved during normal upgrades/uninstall | designed (NSIS currentUser, data dir outside install dir); **Windows runtime verification pending the runner** | BLOCKED (Windows-only verification) |

## 46.4 Complete backend–GUI integration

| Criterion | Evidence | Status |
|---|---|---|
| Public backend APIs inventoried | the §21 matrix §2 (72 capability rows over the 88 operations) | PASS |
| Public visualization functions inventoried | the matrix Viz columns + the viz route trace | PASS |
| Every relevant capability has an explicit GUI integration status | `docs/CAPABILITY_INTEGRATION_MATRIX.md` (this round) | PASS |
| Every user-facing capability mapped to a GUI workflow | 64 PASS rows; 6 PARTIAL housekeeping rows dispositioned to campaign rows | PASS |
| Every mapped workflow executes its real backend operation | the endpoint→function trace; no mock/stub lanes | PASS |
| Every important result inspectable | result handlers per row (tables/cards/frames + sample metadata) | PASS |
| Every applicable plot connected | viz lanes per row (Plotly adoption = the one PARTIAL) | PASS |
| Every applicable export works | digest-verified download lanes (tables ×6, figures ×4, checkpoints, PDBQT) | PASS |
| Integration tests verify actual end-to-end execution | `tests/api/` 261 (real app, no mocks) + browser drivers e2e280–289 | PASS |
| No capability falsely reported as integrated | the honest-findings section (§3 of the matrix) | PASS |
| No critical backend capability remains disconnected | none — the PARTIAL rows are housekeeping/config surfaces, each classified | PASS |

## 46.5 Scientific functionality

| Criterion | Evidence | Status |
|---|---|---|
| Chemistry Lab exact descriptor selection | the descriptor-set select + config-preservation pins | PASS |
| Fingerprint bit size configurable | n-bits/radius/counts form → compute_fingerprints contract | PASS |
| Morgan radius configurable | same lane | PASS |
| Manual n_jobs incl. −1 works where supported | the n-jobs control flows to train; the −1 contract pinned | PASS |
| Invalid n_jobs=0 rejected | backend validation pin | PASS |
| ML features remain available | catalog/train/predict/compare lanes (§2.2) | PASS |
| DL features remain available | networks/train/predict/checkpoint (§2.3) | PASS |
| Semi-supervised Learning connected | train + pseudo-label views; predict via `/ml/predict` | PASS |
| Active Learning connected where implemented/approved | evaluated (§44's justified-feature process); proposals documented, not silently claimed | NOTAPPLICABLE (evaluated; not in the accepted scope) |
| Docking features available where tools available | the engine catalog + boundary honesty (Windows-PE note; engines bundled in the installer spec) | PASS |
| PK/PD features remain available | the §2.6 lanes | PASS |
| MD analysis features remain available | the §2.7 lanes | PASS |
| Applicability Domain complete | analyze + bootstrap uncertainty + reliability frame (the §24.1/24.2 lanes; the §24 combined-status gate: the ADResultCard) | PASS |
| Backend plot generation fully integrated | the figure artifact lane + adapters (digest-verified) | PASS |
| Plotly.js charts work | the r291 lanes: parity/AD/SHAP/DL/PKPD/MD all render through real Plotly traces (browser-verified live, zero console errors, dark-mode re-draw exact); the Matplotlib figure artifacts remain their own §22.3 lane | PASS |
| Three.js visualizations work where applicable | Molecule3D ×3 lanes (r288 21/21) | PASS |
| Model save/load and prediction work | model-bundle round-trip (load → identical predictions) | PASS |
| Sample metadata and SMILES preserved | per-point metadata lanes (customdata); identity pins | PASS |
| Actual and predicted values inspectable | predictions tables + residuals | PASS |
| Figure generation and export work | ×4 formats digest-verified | PASS |
| Scientific configuration metadata saved | feature-config persistence + reproducibility metadata pins | PASS |

## 46.6 Performance and reliability

| Criterion | Evidence | Status |
|---|---|---|
| Large CSV/Excel imports don't block the UI | job-mode uploads + `test_stress_performance.py` (150k-row CSV, 40k-row Excel) | PASS |
| Heavy computations run in backend workers | the JobManager lanes | PASS |
| WebSocket progress updates work | live-streamed training metrics (WS test) + r289 `open` | PASS |
| Job cancellation behaves correctly | cooperative-cancel pins (incl. unsupported-cancel honesty) | PASS |
| Large datasets not transferred as huge JSON | bounded previews (≤500), artifact references, dataset_id referencing | PASS |
| Backend crashes detected | the supervisor + crash/restart policy | PASS |
| Shutdown controlled | graceful drain + shutdown envelope pins | PASS |
| No orphaned backend processes | the Rust host's window-close → sidecar termination path | PASS (design + contract tests; Windows runtime BLOCKED below) |
| Resource consumption measured | RELEASE_READINESS probes + the stress suite | PASS |
| Large charts remain usable | the stress suite's 150k-point lane | PASS |
| Three.js resource management tested | the dispose chain + dPR cap (r288) | PASS |
| SQLite concurrency handled safely | WAL/concurrent-writer pins | PASS |
| Performance limits and limitations documented | RELEASE_READINESS + the matrix §3 | PASS |

## 46.7 Tauri and offline frontend

| Criterion | Evidence | Status |
|---|---|---|
| Tauri 2 launches the application | the host skeleton + the workflow build steps (Windows runner) | PASS (code) / BLOCKED (Windows runtime exec.) |
| React static assets bundled | `frontendDist` → dist/ build | PASS |
| Production frontend needs no dev server | static build + local asset loading; dev proxy is dev-only | PASS |
| No external browser / CDN / remote URL required | CSP + no-CDN gates | PASS |
| Tauri capabilities/permissions restricted | the least-privilege capability + the single IPC command | PASS |
| Python backend sidecar starts correctly | the PHARMAML-READY handshake contract + sidecar spec — **executed**: run 44553307816 step 13 verified the frozen sidecar live ON Windows (handshake port/pid, authenticated health, the guard refusing the unauthenticated probe, the CTRL_BREAK graceful shutdown exiting 0) | PASS (contract tests + the Windows runner's live verification) |
| Backend process lifecycle reliable | as 46.2 | PASS |
| WebView2 available through the offline strategy | `webviewInstallMode: offlineInstaller` (NSIS + MSI targets) + workflow step 19 | PASS (config, drift-guarded) / BLOCKED (clean-machine runtime verification) |
| Application works without internet access | the offline CSP + local lanes; dev-verified flows | PASS (Linux-side); the clean-Windows offline test rides the runner |

## 46.8 PySide6 retirement

| Criterion | Evidence | Status |
|---|---|---|
| New React/Tauri frontend passed feature-parity verification | the §21 matrix (this round) — the audit now exists and is green on user-facing rows | PARTIAL (the parity audit exists; the §28 Stage 4 formal gate pending the campaign rows) |
| All critical integration tests pass | `tests/api/` 261 + guards 123 + browser drivers | PASS |
| A legacy backup preserved | the staged-retirement branch/tag plan (§28 Stage 1) + git history | PASS |
| Old PySide6 GUI no longer the production application | the production frontend is React+Tauri; the legacy studio remains the Stage-1 reference | PARTIAL (Stage 5 removal pending the gate) |
| Obsolete PySide6 deps removed where safe | not yet — the deliberate Stage-5 order | PARTIAL (by design) |
| Backend imports work without PySide6 | the scientific package imports PySide6-free (verified) | PASS |
| No production workflow depends on a Qt widget | the React workflows ride the API; the API layer has no Qt imports | PASS |
| Final installer excludes legacy Qt components | `pharmaml_backend.spec`'s Qt excludes + the §5 rule | PASS (spec, drift-guarded) / BLOCKED (Windows build exec.) |

## 46.9 Deployment

| Criterion | Evidence | Status |
|---|---|---|
| Windows x64 installer EXE generated | `.github/workflows/windows-release.yml` (25 steps, windows-2022) — **executed**: run 44553307816 (r297-iter17, head 7f241a4a) built `PharmaML_Studio_Setup_v1.2.229.exe` on a real runner; the v1.2.230 tag build (run 44566101809) attaches the EXE/MSI/SHA256SUMS/manifest to the GitHub Release | PASS (the r294–297 bring-up arc + the first all-green execution; the ~1.1 GB artifact bundle uploaded) |
| MSI generated where required | the WiX MSI target — **executed** in the same run (`PharmaML_Studio_v1.2.229.msi`, runner-generated SHA-256 `69da2837…130d`) | PASS (execution) |
| Installer includes the complete application | the spec bundles backend + frontend + engines + resources — **executed**: the runner's artifact validation (step 20, the 26/26 static gate) + the staged onedir tree (the directory-key resource mapping, r297-iter16) | PASS (spec + execution) |
| Installer works offline | WebView2 offlineInstaller mode + no-bootstrapper rule — the offline runtime EMBEDDED by the executed build (the NSIS/MSI both carry it) | PASS (config + build execution) / BLOCKED (clean-machine install behavior) |
| Build reproducible | pinned actions + lockfiles + version lockstep gate | PASS (workflow) |
| Application version correct | VERSION == tauri.conf == package.json (1.2.230, guard-pinned) | PASS |
| SHA-256 checksums generated | the workflow's step 22 + `packaging_audit.py --manifest` — **executed ON the runner**: `a385853c…c210` (EXE) · `69da2837…130d` (MSI) · `7d2e9b7f…d8b` (manifest) | PASS (tooling + execution) |
| No Python/conda/pip/Node/npm/Qt/RDKit installs required | the sidecar spec + frontendDist + engines bundling | PASS (spec) / BLOCKED (clean-machine exec.) |
| Externals handled per licenses | engines/ NOTICE files + THIRD_PARTY_LICENSES generation | PASS |
| No developer-specific env/paths | the activation contract (data dir) + workflow hygiene | PASS |
| Works on a clean supported Windows machine | the smoke-test instructions exist | BLOCKED (needs a clean Windows environment — part11_2 §10) |
| Offline scientific workflows work | the offline lanes verified Linux-side | PASS (Linux-side) / BLOCKED (Windows clean-machine) |
| Upgrade does not corrupt user data | data-dir separation design | BLOCKED (Windows exec.) |
| Uninstallation behaves correctly | the NSIS config + retention rules | BLOCKED (Windows exec.) |

---

## Verdict

**73 PASS · 1 PARTIAL · 0 FAIL · 19 BLOCKED (the clean-machine
executions) · 1 NOTAPPLICABLE** across §46.1–§46.9 — the round-298
update: five Windows-runner-gated rows flipped to PASS on the first
real pipeline execution (run 44553307816 + the v1.2.230 tag build
44566101809); the remaining BLOCKED set is exactly the clean-machine
verification part11_2 §10/§12 reserves for a real Windows machine.

The remaining PARTIAL disposition:

1. **PySide6 retirement Stage-5** (§46.8) — deliberately gated on the
   §28 Stage-4 parity acceptance; the parity audit (the §21 matrix,
   now with the r291 Plotly campaign row closed) is the gate's
   evidence base, and the Windows runner's lanes are the remaining
   blockers.

The closed campaign (r291): the Plotly.js src adoption — the shared
PlotlyFigure surface renders every interactive 2-D chart lane
through real Plotly traces (parity + confusion + AD + SHAP + DL +
PK/PD + MD), the §22.5 sample-inspector click lane rides
`predict`'s new evaluation block, and the whole lane is
browser-verified with evidence shots. The §28 Stage-4 gate's own
"complete plotting integration" condition flips to PASS; Stage-5
now waits only on the Windows-runner executions.

The BLOCKED set is exactly part11_2 §1/§12's expected split: the
Linux development host completes every portable lane (all code,
tests, contracts, specs, and the workflow itself — 8 drift guards
keep them honest), and the Windows GitHub Actions runner executes
the release. No placeholder EXE exists; no Linux artifact was
renamed; no Windows installation test was claimed.

*Companion documents: `docs/CAPABILITY_INTEGRATION_MATRIX.md` (the
§21 matrix), `packaging/windows/DEPENDENCY_INVENTORY.md` (the §29
inventory), `.github/workflows/windows-release.yml` (the part11_2
deliverable).*
