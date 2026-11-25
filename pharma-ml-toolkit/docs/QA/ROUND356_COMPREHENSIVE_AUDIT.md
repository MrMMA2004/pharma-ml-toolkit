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
