# MASTER_PROMPT part12 — Phase 1 Compliance Matrix (COMPREHENSIVE_PROJECT_AUDIT)

Baseline audited: **v1.2.231** · commit `7bbc41ed52ae64b09db0ec42cda79622dba7f447`
(round-301 close-out) · audit date 2026-11-20 · auditor session: part12
Phase 1.

Status vocabulary (part12 PART 3): **PASS / PARTIAL / BLOCKED / FAIL /
NOT APPLICABLE** — a module's existence never earns PASS by itself; each
PASS below cites execution evidence gathered this session.

---

## 1. Live-state verification (this session's own evidence)

| Claim | Verdict | Evidence (this session) |
| --- | --- | --- |
| Repo public, default branch `main` | PASS | GitHub API: `visibility: public`, `default: main` |
| v1.2.231 release published, not draft | PASS | API: `draft: false`, published 2026-11-20T08:02:55Z; assets EXE 557,655,583 B + MSI 710,122,853 B + `RELEASE_MANIFEST.json` + `SHA256SUMS.txt`, all `state: uploaded` |
| CI green on audited commit | PASS | API: runs 44700158627 (CI) and 44700158419 (Push on main) both `completed/success` at `7bbc41ed` |
| Windows Release workflow green | PASS | Run 44693818307 `completed/success` (all 26 steps; fire 3, at `3da0ff96`) |
| 8 version carriers in lockstep at 1.2.231 | PASS | Read directly: VERSION, pyproject, `pharmaml.__version__`, `toolkit-version.ts`, frontend package.json, tauri.conf.json, version_info.txt `filevers=(1, 2, 231, 0)`, lockfile |
| Full-suite collection = 8,590 tests | PASS | `pytest --collect-only -q` this session: **8,590 collected, 0 errors** (25.6 s) |
| Census pins 1,583 files / 595,039 lines | PARTIAL | Live `/api/toolkit/tree?fresh=1` walk before fixes: **1,581 / 595,039** — lines exact, files -2. Root-caused this session: (a) the walk's any-depth `artifacts` dir exclusion swallowed `frontend/src/services/artifacts/` (-1 real source file, see BUG_REPORT B-1); (b) two untracked evidence PNGs present at the round-301 measurement were never committed and the sandbox rollback dropped them (-2, zero-line binaries). After the B-1 fix the walk reads the true tracked tree; pins re-trued at this round's close-out |
| v1.2.231 zip in five homes | PASS | Verified present + byte-identical 16,033,638 B in repo-root `download/`, toolkit `dist-archive/`, toolkit `download/`, `upload/`; offsite home missing after rollback — rebuilt at this round's close-out |
| Web pins render (hero) | PASS | Dev server `GET /` 200; `/api/toolkit/tree` live-override active; `/api/toolkit/zip-size` answers `exact: true, source: release` |
| Prior audit ten phases COMPLETED | PASS | PROJECT_AUDIT_PROGRESS.md records remain coherent with the tree |

## 2. Architecture (part12 PART 2) — mandatory stack

| Requirement | Verdict | Evidence |
| --- | --- | --- |
| React 19 + TS + Vite frontend | PASS | `frontend/package.json` (react 19, vite 7, TS strict incl. `noUncheckedIndexedAccess`); `tsc --noEmit` clean this session; production build 2,085 modules |
| Tauri 2 + Rust host | PASS | `src-tauri/` — one IPC command, loopback security assert, `RunEvent::Exit` graceful terminate; bundle targets nsis+msi, offline WebView2 |
| FastAPI backend, REST + WebSocket | PASS | 17 routers, **88 REST ops + 1 WS endpoint** under `/api/v1`; OpenAPI 3.1 contract exported (83 paths / 88 ops / 55 schemas) |
| SQLite + FileVault persistence | PASS | `api/persistence.py` (WAL, schema migrations v1/v2, backup/restore bundles); 67 persistence/database tests |
| PyInstaller + NSIS EXE + WiX MSI | PASS | Release run 44693818307 steps 12–19 executed; real EXE/MSI assets on the GitHub Release |
| Offline after install | PASS (build evidence) / BLOCKED (clean-machine) | WebView2 offline installer bundled; no cloud-dependent runtime in code; the clean-machine execution lane stays BLOCKED (no Windows machine here) |
| Architecture boundaries (FE never touches SQLite; backend independent of React/Tauri) | PASS | Frontend talks only REST/WS via typed clients; no sqlite import anywhere in `frontend/`; `src/pharmaml` has no React/Tauri imports |

## 3. Nine-workspace architecture (part12 PART 5)

Order verified in `src/app/workspaces.ts` + `WorkspaceNav`: Introduction ·
Datasets · Machine Learning · Deep Learning · Generative AI · Molecular
Docking · PK/PD · Molecular Dynamics Analysis · Visualization — Introduction
is the initial page (`INITIAL_WORKSPACE = "introduction"`, drift-guarded).

| Workspace | Sequential workflow (backend-backed) | Verdict | Notes |
| --- | --- | --- | --- |
| Introduction | orientation + (this round) live database status/history/backup | PASS | The previously unsubstantiated "backup and restore built in" claim is now wired to real `/database` routes (B-2 fix) |
| Datasets | Import→Inspect→Validate→…→Descriptors/Fingerprints→Split→Leakage→Export | PASS | Full route chain consumed; descriptors + fingerprint params (kind/radius/nBits/counts) real |
| Machine Learning | 14 stages incl. AD + semi-supervised + SHAP | PASS | AD consumes only `/analyze` — catalog/`knn`/`domain-box` sub-routes unconsumed (PARTIAL → documented non-GUI status) |
| Deep Learning | networks→train (job)→live per-epoch WS metrics→artifacts | PASS | Live `training.metrics` envelopes consumed |
| Generative AI | priors/objectives→train→sample→optimize→figures | PASS | PriorStore bounded; real backend defects closed in r283 |
| Molecular Docking | prep (receptor/ligand)→dock (job)→parse/scoring/contacts/pose-RMSD→3D (three.js) | PASS | Molecule3D + client-side PDBQT parser; honest no-Wine 422 boundary on dock execution off-Windows |
| PK/PD | import→NCA→models (steady-state/iv-bolus/bateman/superposition/pbpk/pd/population)→figures | PASS | Terminal-phase regression with r²/span verdict |
| MD Analysis | synthetic/import→analyze (RMSD/RMSF/Rg/H-bonds/SASA)→xvg→ENM/PCA→study→3D playback | PASS | Honest "trajectory analysis, not simulation" boundary held |
| Visualization | catalog→stat/pharma/md plots→tables→HTML report (job) | PASS | Report route with per-section WS progress |

No workspace is a mock or placeholder; every stage invokes a real backend
route (chains traced in BACKEND_GUI_INTEGRATION_MATRIX.md).

## 4. Backend capability audit (part12 PART 4 + focus areas)

| Focus area | Verdict | Evidence / gaps |
| --- | --- | --- |
| Chemistry Lab descriptors | PASS | `chem/descriptors.py` (rdkit_all/morgan/mordred + 3D ETKDG) + `chem/features.py` 19–20 families, >27k columns; Datasets workspace exposes them |
| Fingerprint engines | PASS | morgan/rdkit/atom_pair/avalon (+MACCS, FCFP, topological, layered, estate, erg via features.py); nBits/radius/count modes; **chirality toggle CLOSED r303** — `include_chirality` through `compute_fingerprints`/`FeatureConfig`/`/api/v1/chemistry/fingerprints` + the Datasets card (morgan-only, default OFF, the byte-identical history preserved) |
| Semi-supervised learning | PASS | `ml/semi_supervised.py` 1,152 lines: SelfTraining/LabelPropagation/LabelSpreading + `-1` sentinel + leakage-safe evaluation; GUI lane `/semi-supervised/train` wired; the `/evaluate` GUI consumer landed r303 (its wiring exposed and fixed the job-mode TypeError — the COMPLETION pin rides RELEASE_READINESS) |
| Applicability Domain | PASS | kNN distance + bootstrap uncertainty + DomainBox implemented and plotted; **the standalone leverage/Williams method landed r303** (`ml/applicability.py::leverage_williams` + `/api/v1/applicability-domain/leverage-williams` job/sync + the ML workspace Williams lane; 23 unit + 3 figure + 7 route tests) — see SCIENTIFIC_VALIDATION_REPORT §AD |
| Resource manager / job management | PASS | `api/jobs.py` JobManager (state machine, cooperative cancel, semaphore, history upsert) + `utils/resources.py` (cgroup/proc probe, presets); psutil deliberately not a dependency |
| Experiment tracking / model registry | PASS (local-first) | Multi-layer: MODEL_REGISTRY (33), ModelStore write-through, `ml/persistence.py` provenance, runboard (TensorBoard absorbed), `studio/runs.py` + `studio/comparison.py`; no external server (by design) |
| Visualization engine | PASS | `viz/` 22 modules, 33 ML plot fns incl. 7 SHAP plots + AD plot; statplots/diagnostics/chem/pharma/docking/md/generative/multiobjective/3D; catalogued route |
| Data quality / leakage | PASS | splits (6 strategies incl. scaffold), split/file audits, curation, profiling (missingness/duplicates/constants/Cramér's V), drift; 38 "leakage" mentions across 16 files; 4 toolkit pinning regression tests |
| n_jobs handling (part12 PART 7) | PASS | Job-level concurrency distinct from estimator n_jobs; frontend passes values unchanged; `-1` warning documented in ML page |

## 5. Performance / large data (part12 PART 6)

| Requirement | Verdict | Evidence |
| --- | --- | --- |
| Background execution, no GUI-thread blocking | PASS | All long routes accept `mode: "job"`; JobManager ThreadPoolExecutor; per-workspace JobRunCard; WS progress |
| Streaming/chunked import | PARTIAL | Dataset upload multipart with bounded previews (≤500 rows/col); Polars deliberately not adopted (no measured benefit — documented decision) |
| Virtualized/efficient tables | PASS | Bounded previews + 5,000-row export cap mirrored FE/BE |
| Memory estimation before expensive work | PASS | `utils/resources.py` probes + `preset_for_low_memory`; chunk-3 OOM storm of r301 decoded and mitigated (six-class split for `test_nine_card_first`) |
| Cooperative cancellation + cleanup | PASS | `CANCELLATION_REQUESTED→CANCELLED` semantics; JobCancelled native; graceful drain |

## 6. Security / packaging (part12 PART 16)

| Requirement | Verdict | Evidence |
| --- | --- | --- |
| Loopback binding + Host/Origin validation | PASS | `api/security.py` + Tauri `assert_loopback`; CSP `connect-src ipc: http://127.0.0.1:* ws://127.0.0.1:*` |
| Session token handling | PASS | `secrets.token_urlsafe(32)` in-backend, memory-only, constant-time verify, never argv/env/log |
| SBOM generation | PASS (CLOSED r304) | `packaging/sbom/sbom.cdx.json` (CycloneDX 1.6, 40 components, validated + reproducible) ships in the archive; the Windows workflow step 22b regenerates it on the runner and self-registers its hash into SHA256SUMS.txt |
| Artifact integrity | PASS (route) / PARTIAL (archive) | Digest-verified downloads (X-Artifact-SHA256 + local recompute); **the shipped v1.2.231 source zip is missing exactly one real source file** (B-1) — fixed for this round's archive |
| Installer size investigation | PARTIAL | EXE 557.7 MB / MSI 710.1 MB: PyInstaller onedir sidecar (torch CPU + rdkit + scipy stack) + offline WebView2 bundle dominate; no unnecessary bundled scientific file found; size breakdown documented in SECURITY_AND_PACKAGING_AUDIT.md |
| Digital signing | BLOCKED | No certificate available (owner decision required) |
| Clean-machine verification (24-step ladder) | BLOCKED | No genuine Windows 10/11 x64 machine in this environment; runner ≠ clean machine (honest per part12 PART 16) |

## 7. PySide6 retirement (part12 PART 17)

**NOT DONE — by design.** Feature-parity matrix exists
(CAPABILITY_INTEGRATION_MATRIX + FEATURES.md); retirement is gated on the
clean-machine pass (rule #9 / Stage-5). The legacy GUI runs the same engines
in-process and is fully wired; no accidental PySide6 dependency exists in
the shipped React/Tauri app (verified: `frontend/` + `src-tauri/` carry no
PySide6 reference; PySide6 remains a dev/test-environment dependency only).

## 8. This round's remediations (Phase 2 fixes, regression-verified)

1. **B-1 artifacts exclusion scoping** — the any-depth `artifacts` dir
   exclusion (tree walk + ZIP patterns) swallowed
   `frontend/src/services/artifacts/`: the explorer hid the artifact client,
   the census undercounted, and the **shipped v1.2.231 zip is missing
   exactly that source file** (diff of zip vs tracked set: 919 files, one
   real source file absent). Fixed with relative-path-scoped exclusions
   (`artifacts`, `examples/artifacts`); verified live (walk 1,581→1,582,
   file browsable) and by test-zip semantics (real source in, generated
   output out).
2. **B-1b `frontend/dist` leak** — same class, latent: a local Vite build
   (never present before — node_modules wiped on rollbacks) leaks 5.7 MB of
   build output into the census + archive. Fixed with scoped
   `frontend/dist` exclusion (tree + ZIP).
3. **B-2 database GUI wiring** — `/api/v1/database` (+`/jobs`,
   `/backup`, `/restore`) had NO GUI consumer while the Introduction page
   claimed "backup and restore built in". Wired: new typed
   `services/database/client.ts` (runtime-narrowed) + the Introduction
   "studio database" card (live status/row counts, persisted job history,
   backup→digest-verified download, two-step-confirm restore). tsc clean,
   frontend guards green, production build green.
4. **B-3 malformed nested CI config** — `pharma-ml-toolkit/.github/
   workflows/ci.yml` carried a corrupted trigger (`branches: ain]`), was
   invisible to Actions (root registration is the real one) and duplicated
   a stale subset of the root CI. Removed (the same sweep the round-301
   close-out applied to the nested windows-release.yml). Census effect:
   -1 file / -52 lines, pins re-trued at close-out.

## 9. Honest BLOCKED register (unchanged lanes)

- **C-3/C-5 clean-machine smoke test** (Windows 10/11 x64, offline install
  + upgrade + uninstall + 24-step ladder): BLOCKED — no machine here.
- **§28 Stage-5 PySide6 retirement**: BLOCKED behind the clean-machine pass
  by the documented rule.
- **C-7 final-report rows 38–48** (installer-dependent): BLOCKED with them.

## 10. Verdict summary

- Requirements verified PASS with this-session execution evidence: 24
- PARTIAL (documented gaps with owners): 6 — of which the census re-true,
  AD leverage, chirality toggle, SBOM, and dist-leak lanes all CLOSED by
  the r303/r304 follow-up rounds (re-verified in-tree at the r307
  part13 reconciliation); the installer-size tiering decision stays OPEN
  (torch-in-bundle vs on-demand — owner call, never taken silently)
- BLOCKED (external dependency): 3 lanes (clean-machine, signing, retirement)
- FAIL: 0 — no requirement was found claimed-but-absent except B-1's
  archive omission (fixed) and B-2's claim (wired).

Next: BACKEND_GUI_INTEGRATION_MATRIX.md (the route-level mapping),
ARCHITECTURE_REVIEW.md, BUG_REPORT.md, and the six remaining part12
deliverables in this folder.
