# ROUND356_COMPREHENSIVE_AUDIT — part17 §2 (the opening-round re-inspection)

**Spec:** `MASTER_PROMPT part17` (Round 356) · **Executed:** the
r356 opening round, 2026-11-25 · **Inspected tree:** `97551e76` (the
r355 terminal, re-verified at open this round — the mandate's stated
inspection SHA, confirmed still `origin/main` at the round's fetch)
· **Method:** fresh read-only audit first, then the round's own lane.

Part17's own rule — "Do not assume that this SHA is still HEAD.
Re-inspect the repository before starting" — was honored: the round
opened onto the SIXTEENTH sandbox rollback (HEAD reverted to the r186
checkpoint `2eb0ede`, the remote + `.credentials/` wiped, the venv
gutted, both ECC legs wiped), and the audit below was taken AFTER the
full recovery ritual (the PAT re-splice + the fetch + the
`reset --hard` onto the durable terminal), so every fact below is
the live state of the pushed tree, not a carried-forward claim.

Status vocabulary (part17 §2): **VERIFIED_COMPLETE** ·
**IMPLEMENTED_UNTESTED** · **PARTIAL** · **BACKEND_ONLY** ·
**FRONTEND_ONLY** · **REGRESSED** · **NOT_IMPLEMENTED** ·
**BLOCKED_ENVIRONMENT** · **NOT_APPLICABLE**. Every claimed
completion names the source file, the test, the execution evidence,
and the remaining limitation.

---

## 1. Repository state (re-inspected live this round)

| Fact | Value | Evidence (this round's own execution) |
| --- | --- | --- |
| HEAD / branch | `97551e76` on `main` | `git rev-parse HEAD` after the recovery reset; `git log -1` names it the r355 terminal |
| origin/main | `97551e76` | the authenticated `ls-remote` + `fetch` at the round's open (the PAT drill, token never in argv/URL) |
| GitHub Actions at HEAD | **19/19 check-runs ALL SUCCESS** | the authenticated in-process urllib poll (`api.github.com/.../commits/97551e76.../check-runs`, conclusions `{success: 19}`) |
| Latest release | `v1.2.240` — EXE 558,022,042 B · MSI 710,553,078 B · RELEASE_MANIFEST.json · sbom.cdx.json · SHA256SUMS.txt | the authenticated releases/latest poll this round |
| Project version | `1.2.240` (`VERSION`) | read from the reset tree |
| Tracked-tree state | clean, 0/0 divergence from origin/main | `git status --porcelain` → 0 lines after the reset |
| Six-home v1.2.240 zip parity | all six byte-identical `0fadb6e5` × 23,489,390 B | the two rollback-wiped homes (root `dist-archive/` + the offsite mirror) re-seeded this round from the verified `download/` home and sha-256-compared |
| venv | repaired at the exact pins (numpy 1.26.4 / torch 2.14.1+cpu / matplotlib 3.10.9 / openpyxl 3.1.5 / pharmaml 1.2.240 / fastapi 0.128.0 / uvicorn 0.44.0 / rdkit 2026.9.1 / PySide6 6.8.3 / pyqtgraph 0.13.7) | `scratch/venv_repair.sh` foreground exit 0 with both verification lines (VENV-REPAIR-DONE + STUDIO-REPAIR-DONE) |
| ECC (the owner's standing directive) | GREEN — clone `4adeee67` / 2.2.3 / 328 skills; harness 14 modules / 196 skills / 8/8 domains / 4/4 capabilities | `scratch/ecc_check.sh` exit 0 after the reinstall |
| Frontend deps | `npm ci` 210 packages | the repair log this round |

## 2. The previous-phase verification (part17 §3's objective 1)

| Phase (the claiming round) | Status | Source | Test / execution evidence | Remaining limitation |
| --- | --- | --- | --- | --- |
| The nine-workspace React console (part12/part13 eras) | **VERIFIED_COMPLETE** | `frontend/src/features/*` + the App shell | the r355 sweep re-executed `scratch/r354_verify.py` over the identical tree (footer LIVE walk 1,811 files / 578,267 LOC, zero page/console errors, mobile 390 clean); vitest 287/287 over 33 files at r353–r355 | none beyond the standing Windows-box set below |
| ML worker termination (part16 §3.4: `n_jobs=1` default, memory-aware clamp, typed worker-death) | **VERIFIED_COMPLETE** | `src/pharmaml/ml/…` worker lanes + `tests/api/test_ml_worker_termination.py` | the r344 FAIL-FIRST delivery + the standing suite | the Windows frozen-backend + memory-copy verification stays the documented environment-gated leg |
| Data-quality lane (r351) | **VERIFIED_COMPLETE** | `src/pharmaml/data_quality.py` + `routes/datasets.py` `/quality` | 22/22 API + the vitest pair + the 9/9 drift guard | none recorded |
| Diagnostics/recovery lane (r352) | **VERIFIED_COMPLETE** | `/api/v1/system/diagnostics` + `/cache/cleanup` | 30/30 lane suite + the vitest pair | none recorded |
| Grid-box visualization lane (r353) | **VERIFIED_COMPLETE** | `components/docking/GridBoxViewer.tsx` + `services/docking/gridBox.ts` | vitest 287/287; the 13/13 drift guard re-run this round | none recorded |
| Security lanes D-1..D-6 (part15 §13, r340) | **VERIFIED_COMPLETE** | `docs/QA/SECURITY_AUDIT.md` (the live doc) + the lanes | `test_trusted_origins.py` 34 legs, `test_ws_origin.py` 12/12, `test_security_supply_chain.py` 8/8 | see §3 below — the **penetration-testing program itself** is NOT_IMPLEMENTED (a different, larger surface) |
| PySide6 legacy availability (part17 §3.5) | **VERIFIED_COMPLETE** (retained, not retired) | `pharma-ml-toolkit/studio.py` + `ui/` | the venv repair's STUDIO-REPAIR-DONE (QApplication OK offscreen) this round | retirement stays gated on the migration gates (React parity + clean Windows install + representative workflows + data preservation) — none of which is owner-verified yet |
| Plotly adoption (r291) | **VERIFIED_COMPLETE** | `components/charts/PlotlyFigure.tsx` | 16 files reference it (ML page + lanes, PK/PD, 12 test suites) — re-grepped live this round; the matrix's contradictory part11-era §3.2 text was TRUED with a dated addendum this round (part17 §3.3) | the historical prose remains as record, clearly marked superseded |

## 3. The part17 gap audit — what is actually missing (objective 2)

The r356 source-level re-audit over the current tree, with the
worklog's zero-mention proof where it applies:

| § | Surface | Status | The live evidence |
| --- | --- | --- | --- |
| §3.1 | Dataset deletion UI consumer | **CLOSED THIS ROUND** (was the matrix's PARTIAL row) | the Datasets · Inspect per-row Delete action → the two-step confirmation → the existing `datasetsApi.remove` → registry refresh + deleted-selection clear + the honest 404 lane; vitest §3.1 flows (15 over the page suite) + `tests/test_frontend_datasets_delete.py` 14/14 + the API suite's `test_delete_then_404` — the capability-matrix row trued to PASS |
| §3.2 | Artifacts view/delete UI | **BACKEND_ONLY** (service lane `artifactsApi.view/delete` exists, API-tested, no UI consumer) | `routes/artifacts.py` + `services/artifacts/client.ts`; the matrix row + the §3.4 housekeeping list |
| §3.2 | Global Job Center / job ledger UI | **BACKEND_ONLY** (`jobsApi.list` exists; `JobRunCard`/`useJob` are per-workflow trackers, not a global ledger) | `routes/jobs.py` + `services/jobs/client.ts` |
| §3.2 | Database backup/restore + preferences UI | **BACKEND_ONLY** (67 tests over the `/database` surface; no React consumer) | `test_persistence.py` 30 + `test_persistence_database.py` 37 |
| §3.3 | Visualization ownership reconciliation | **VERIFIED_COMPLETE** (this round's truing) | the §3.2 dated addendum above; the owning-workflow grep (16 PlotlyFigure references, workflow pages own their charts) |
| §4 | The penetration-testing program | **NOT_IMPLEMENTED** | the worklog carries ZERO "penetration" mentions across 355 recorded rounds (grep this round); no `.github/workflows/penetration-test.yml`, no `docs/QA/PENETRATION_TEST_*.md`, `THREAT_MODEL.md`, `SECURITY_REMEDIATION_REGISTER.md` — genuinely new campaign work |
| §5 | Generic Multi-Response / Multi-Output ML | **NOT_IMPLEMENTED** (the existing `ml.drugresponse` / `ml.response_model` / `ml.multiobjective` are domain lanes, NOT the generic multi-target supervised pipeline) | zero worklog mentions of multiresponse/multi-output regression across 355 rounds; no strategy registry, no `/api/v1/multiresponse` route, no frontend section |
| §6 | Response Surface Methodology (DOE + RSM + desirability) | **NOT_IMPLEMENTED** | zero worklog mentions; no `pharmaml.rsm` module, no DOE route, no RSM frontend section |

## 4. The part17 campaign queue (the lanes for the coming rounds)

Recorded here as the honest queue this audit opens — in the
mandate's own order, smallest-integration-gap-first as the working
order:

1. **P1 — the penetration-testing program (§4):** the OWASP
   WSTG/API-Top-10 methodology, the scope matrix (FastAPI backend,
   React/Tauri desktop, Next.js console, the supply chain), the
   focused automated tests (auth bypass, IDOR/BOLA, CORS/Origin
   bypass, WS hijacking, session replay, path traversal, ZIP Slip,
   symlink, malformed/oversized/MIME-spoofed uploads, dangerous
   names, log secrets, resource exhaustion, injection/XSS, CSP,
   process handshake, Tauri permission overreach), the
   `penetration-test.yml` ZAP workflow against an ISOLATED seeded
   test service, and the four deliverable documents
   (`PENETRATION_TEST_PLAN/REPORT`, `THREAT_MODEL`,
   `SECURITY_REMEDIATION_REGISTER`). A workflow that has not run
   stays **NOT_EXECUTED** — never an invented scan result.
2. **M1 — the generic Multi-Response ML (§5):** the strategy
   registry (MultiOutputRegressor / RegressorChain / PLS2 /
   MultiTaskElasticNet + the natively-multi-output families), the
   input contract (features, ≥2 targets, weights, scaling,
   preprocessing, split/CV/group-aware), the per-target evaluation
   surface, prediction/persistence with the full training contract,
   the ML-workspace section (no tenth workspace), the demo dataset,
   and the mandated test families.
3. **R1 — the Response Surface Methodology (§6):** the DOE engine
   (CCD/Box-Behnken/full-factorial, coded/actual levels, replicates,
   randomization), the second-order model fitting with ANOVA +
   hierarchy/rank checks, the contour/3D/interaction/residual
   visualization set (in the ML workspace), the desirability
   multi-response optimization with factor constraints and Pareto
   alternatives, and the mandated test families.
4. **G1 — the remaining §3.2 UI consumers:** artifact view/delete,
   the global Job Center, the database backup/restore +
   preferences surfaces — each behind the same confirmation and
   ownership discipline the §3.1 lane established this round.

## 5. Environment blockers (unchanged from the standing register)

* The Windows clean-machine evidence set (C-1..C-7 in the R293
  blocker register) — **BLOCKED_ENVIRONMENT**, owner-action-gated;
  no Windows installation is claimed this round or any round since
  the register's own record.
* The sandbox rollback class (this round's sixteenth instance) —
  absorbed by the standing recovery drill; the durable-source rule
  (origin/main holds the truth; the working tree re-attaches to it)
  held again.

## 6. This round's own lane (the first executed against this audit)

The §3.1 dataset-deletion closure above — delivered, tested, and
drift-guarded this round; the capability-matrix row trued from
PARTIAL to PASS; the campaign queue's remaining lanes (P1, M1, R1,
G1) stand open and honestly **NOT_IMPLEMENTED** as of this terminal.

## 7. r357 dated addendum (the P1 lane's delivery — status truing, not history rewriting)

The §4 row above and the queue's P1 entry describe the round-356
truth and stand as the record. One round later (2026-11-25, the
interactive "Continue MASTER_PROMPT part17 / Push" round): **P1 the
penetration-testing program is DELIVERED** — the automated security
suite `tests/api/test_penetration_security.py` (67 tests over the
real `create_app()` stack, the §4.3 list, OWASP-mapped, complementing
the standing origin/WS/supply-chain suites), the four program
documents (`docs/QA/THREAT_MODEL.md`, `PENETRATION_TEST_PLAN.md`,
`PENETRATION_TEST_REPORT.md`, `SECURITY_REMEDIATION_REGISTER.md`),
the isolated ZAP lane (`.github/workflows/penetration-test.yml`
scheduled + dispatch, `tools/zap_target.py` the disposable seeded
loopback target, the Automation-Framework config, the
`tools/zap_gate.py` severity gate + the documented suppression
channel), and the round's real find F-356-01 (the
Content-Disposition injection via artifact filenames) FIXED at source
with the 3/3 retest. The ZAP workflow itself honestly remains
**NOT_EXECUTED** until its first runner firing. The remaining queue:
M1, R1, G1.

## 8. r358 dated addendum (the M1 lane's delivery — status truing, not history rewriting)

The §5 row above and the queue's M1 entry describe the round-356
truth and stand as the record. Two rounds later (2026-11-25, the
interactive "Continue MASTER_PROMPT part17 / Then push" round): **M1
the generic Multi-Response / Multi-Output ML is DELIVERED** —
`src/pharmaml/ml/multiresponse.py` (the §5.3 strategy registry with
each mechanism STATED — the native 2-D families, the
MultiOutputRegressor wrapper with its per-target clones, the
RegressorChain feed; the §5.4 input contract with the
complete_case/per_target missing lanes, the per-target-rows
`PerTargetMultiOutput` masked multi-task wrapper, group-aware split,
sample-weight capability stated per strategy; the §5.5 evaluation
surface — per-target R2/RMSE/MAE/MSE/explained variance with the
uniform/weighted aggregates reported alongside, the bounded residual
samples, the target correlation matrix, the k-fold CV over the same
full procedure; the §5.6 persistence contract riding every bundle),
`/api/v1/multiresponse` (registry/train/predict/models — the
contract 117→121 paths / 73→75 schemas, `api.d.ts` re-grown), the §5.7
frontend section (stage 20 inside the EXISTING Machine Learning
workspace — no tenth workspace; the five typed service methods, the
per-target metric cards, the correlation matrix, the
predicted-vs-actual Plotly lane, the prediction table, the models
comparison surface, the joblib export), and the §5.8 test families
(`tests/api/test_multiresponse.py` 38/38 — PLS2, MultiOutputRegressor,
RegressorChain, MultiTaskElasticNet, per-target missing, the
target-scaler leakage discipline at the contract level, group-aware
split, CV, aggregates, serialization/reload, prediction ordering,
artifact export, contract, e2e; the vitest `MultiresponseLane` 6/6;
the `tests/test_frontend_multiresponse.py` drift guard 12/12). The
remaining queue: R1 (the RSM engine), G1 (the remaining §3.2 UI
consumers).

## 9. r359 dated addendum (the R1 lane's delivery — status truing, not history rewriting)

The queue's R1 entry above describes the round-356 truth and stands as
the record. Three rounds later (2026-11-25, the interactive "Continue
MASTER_PROMPT part17 / Then push" round): **R1 the Response Surface
Methodology engine is DELIVERED** — `src/pharmaml/ml/rsm.py` (the §6.3
design generators — CCD with the rotatable alpha DISCLOSED as outside
the declared ranges, Box-Behnken, the capped full factorial, the
two-block CCD split, seeded within-block randomisation, the coded AND
actual run tables, the typed 512-run bound; the §6.4 second-order fit —
the coefficient tables, the ANOVA partition with pure error /
lack-of-fit ONLY where replicated design points exist (nulls
elsewhere, never invented; the noise-free F reported as an UNDEFINED
null), the hierarchy notes flag-never-remove, the aliasing
disclosure, the machine-precision diagnostics guard; the §6.5
prediction lane with the zone vocabulary (inside_ranges /
axial_region / outside_design — extrapolation flagged, never silent);
the §6.5 contour / desirability grids with the displayed fixed
factors; the §6.6 Derringer-Suich optimiser — the four goals with
weights and exponents, the observed-min/max bounds fallback stated,
the deterministic multi-start search confined to the declared ranges
so it CANNOT extrapolate by construction, the sensitivity sweep, the
curated Pareto alternatives, the feasible-region mask),
`/api/v1/rsm` (the seven-path septet: registry / design / fit /
predict / contour / optimize / models — the contract 121→128 paths /
75→83 schemas, `api.d.ts` re-grown, the three path-count drift pins
trued at source with the dated comments), the §6.7 frontend section
(stage 21 inside the EXISTING Machine Learning workspace — NO tenth
workspace; the factor editor, the design table, the fit's coefficient
tables + diagnostics, the prediction zone table, the Plotly contour
lane, the goal editor, the optimisation candidate + alternatives, the
joblib export; the visualisations stay in the ML workspace per the
§6.7 rule — pinned by the drift guard), and the §6.8 test families
(`tests/api/test_rsm.py` 38/38 — the CCD/BBD/FFD generation counts
and levels, the coded↔actual conversion, the seeded randomisation
determinism, the exact coefficient recovery over the KNOWN surfaces,
the ANOVA nulls, the significant lack-of-fit for a cubic surface,
the complete-case dropped-row report, the rank-deficiency
disclosure, the hierarchy flag, the run lineage, the joblib export +
reload, the prediction CI/PI + zones, the contour modes, the
desirability arithmetic (all four goals, exact), the fine-grid
optimality check, the constrained optimisation with the feasible
mask, the infeasible refusal, determinism, the e2e workflow; the
vitest `RsmLane` 9/9; the `tests/test_frontend_rsm.py` drift guard
16/16). The documentation debt closed the same round:
`docs/ML/RESPONSE_SURFACE_METHODOLOGY.md` + `docs/ML/MULTI_RESPONSE_ML.md`
(the part17 §8 pair, the M1 doc owed since r358). The remaining
queue: **G1 (the remaining §3.2 UI consumers)** alone.

## 10. r360 dated addendum (the G1 lane's delivery — the campaign queue EMPTIED — status truing, not history rewriting)

The queue's G1 entry above describes the round-356 truth and stands as
the record. Four rounds later (2026-11-25, the interactive "Continue
MASTER_PROMPT part17 / Then push" round): **G1 the remaining §3.2 UI
consumers is DELIVERED — and with it the part17 campaign queue is
FULLY EMPTY** (P1 r357, M1 r358, R1 r359, G1 r360):

* **the global Job Center** — `frontend/src/components/system/
  JobCenterCard.tsx` mounted in the Introduction workspace: the live
  cross-workspace session ledger over the EXISTING typed
  `jobsApi.list` (the list lane that had a typed, API-tested client
  and no consumer), with the authoritative per-row actions — the
  result lane (`jobsApi.result`, offered only for COMPLETED rows;
  the backend's own 409/404 answers surfaced verbatim, never
  swallowed), the cancellation lane (`jobsApi.cancel`, offered only
  for the live states QUEUED / RUNNING / CANCELLATION_REQUESTED,
  behind the two-step confirmation whose armed gate can never
  outlive its row), and the honest split stated in the card (the
  live in-session ledger vs the studio database card's persisted
  history — the two views never pretend to be one);
* **the durable preference manager** — `frontend/src/components/
  system/DatabasePreferences.tsx` + the client quartet in
  `services/database/client.ts` (`preferences` / `setPreference` /
  `deletePreference` with `parsePreferenceRow` narrowing): the
  `/database/preferences` quartet's FIRST consumer (the backend
  surface that had no client lane at all), with the client-side
  gates firing BEFORE any request (the 1–128 key bound = the
  backend's own rule; the strict-JSON value validation — an
  invalid payload never leaves the browser), the two-step delete,
  the honest 404 already-gone lane, the 409 non-durable note (the
  shipped host always runs durable), and the durable-SQLite-store-
  vs-localStorage-settings split stated;
* **the artifacts view/delete consumers** — the Visualization
  consumption hub: the per-row View over the EXISTING typed
  `artifactsApi.view` (the full-record panel, a 404 rendered
  verbatim — a row can be evicted server-side between the list and
  the click) and the two-step Delete over the EXISTING typed
  `artifactsApi.delete` (the r356 dataset-deletion discipline
  applied to the artifact registry: the row action only arms, the
  confirm button fires, the honest 404 reconciles, and every
  stage-local derivative that referenced the evicted row is
  cleared — the selection, the viewed record, the compare memory,
  the digest cache — never a dangling reference).

No new API surface (the OpenAPI contract stays 128 paths / 83
schemas); NO duplicate clients anywhere in the frontend tree (each
lane rides the EXISTING typed client — pinned by the guards). The
mandated test families: the vitest trio
(`JobCenterCard.test.tsx` 9 + `DatabasePreferences.test.tsx` 11 +
`VisualizationPage.test.tsx` 8 — the full frontend battery
337/337 over 38 files) and the drift-guard trio
(`tests/test_frontend_job_center.py` 14 +
`tests/test_frontend_database_preferences.py` 16 +
`tests/test_frontend_artifacts_view_delete.py` 13 = 43 new Python
pins). The QA surface trued the same round: the
CAPABILITY_INTEGRATION_MATRIX r360 addendum (the three §3.4
PARTIAL rows closed to PASS, the stale semi-supervised evaluate
count trued, the §4 verdict table 70 PASS / 1 PARTIAL, the §3.6
pre-r310 record trued) + the PROJECT_STATUS r360 clause +
TEST_EXECUTION_REPORT r360 table + the worklog R360 entry. The
campaign queue: **EMPTY** — the next lane is upkeep alone.
