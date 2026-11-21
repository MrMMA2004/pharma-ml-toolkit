# PROJECT_STATUS — the single authoritative status document

Part13 §2 deliverable ("Create a single authoritative project-status
document with the actual current state"). Replaces no audit; it
summarises them and wins any status disagreement.

- **Audit date**: 2026-11-21 (round 307, the 98th recorded run;
  Phase-2 addendum round 309; Phase-3 addendum round 310 — the
  interactive part13 continuation, trace 1a1fb0a0b935472d; Phase-4
  addendum + the Phase-3 browser-E2E close round 311 — the interactive
  part13 continuation, trace 1a1fc9a07e081d59; the Phase-4 close +
  Phase-5 open addendum round 312 — the interactive part13
  continuation, trace 1a1fd0516a6c330d, over the PARTIAL sixth
  rollback; the Phase-5 GNN-GUI addendum round 313 — the interactive
  part13 continuation, trace 1a1fd63a737e7e29, over the seventh
  rollback; the Phase-5 ADMET addendum round 314 — the interactive
  part13 continuation, trace 1a1fd9525b9d5b89, over the post-session
  machinery ghost-zip commit reset away per the r312 precedent; the
  Phase-5 DTA addendum round 315 — the interactive part13
  continuation, trace 1a1fdc2da50cd529, over the post-session
  machinery ghost-zip commit reset away per the same precedent; the
  Phase-5 Drug Repurposing addendum round 316 — the interactive
  part13 continuation, trace 1a1fe5e3bc4ba0f5, over the ninth
  arrival's stale-summary reconciliation — the venv SURVIVED this
  rollback (zero repair), the machinery ghost-zip commit reset away
  unpushed per the same precedent, the r315 push's CI verdict closed
  GREEN on arrival; the Phase-5 Cancer Cell-Line Drug Response
  addendum round 317 — the interactive part13 continuation, trace
  1a1fe97d37b3b40d, over the tenth arrival's stale-summary
  reconciliation — the venv survived again (zero repair), the
  machinery ghost-zip commit `9e42c69e` reset away unpushed per the
  same precedent (five-home sha parity `6fc0d25e` restored), the
  r316 push-3 CI verdict closed GREEN on arrival: `74b81167`
  19/19 check-runs success; the Phase-5 Pharmacogenomics addendum
  round 318 — the interactive part13 continuation, trace
  1a1fed8994f477c1, over the eleventh arrival's stale-summary
  reconciliation — the venv survived again (zero repair, the third
  rollback in a row), the machinery ghost-zip commit `c07a8a12` reset
  away unpushed per the same precedent (five-home sha parity
  `455cf00b` restored), the r317 push-2 CI verdict closed GREEN on
  arrival: `7df1806d` 19/19 check-runs success).
- **Audited base**: `origin/main` = `5446e89b` (the r306 close-out);
  Phase-2 addendum on `19effec8` (the r308 close-out); the Phase-3
  addendum verifies `ca1accc9` (the r309 close-out) clean and lands
  with the round-310 commit; the Phase-4 addendum verifies `1bbe7316`
  (the r310 terminal) — NOT clean: three latent r310 close-out
  defects found and fixed at the source (the 18/19 Studio MD
  segfault, the stale FEATURES.md freeze stamp, the twin note-cell
  regression) — and lands with the round-311 commit (the fifth
  rollback recovered first: PAT/origin/reset/venv per the r306–r308
  playbook).
- **Scope**: whole repository — source, tests, docs, release
  artifacts, web pins, archive ritual.
- **Status labels** (part13 §22): PASS / PARTIAL / FAIL / BLOCKED /
  NOT APPLICABLE. Nothing here converts absence of evidence into a
  pass; every row names its evidence.

## 1. Where to look for what

- **This file** = the authoritative current-state summary. Where a
  per-audit doc (`docs/part12_audit/*`) disagrees on status, this file
  wins (their stale rows were trued at the r307 reconciliation; the
  audits' original findings are preserved as history in git).
- Release gates: `docs/RELEASE_READINESS.md` (generated + addenda).
- Integration tiers: `docs/CAPABILITY_INTEGRATION_MATRIX.md` +
  `docs/part12_audit/BACKEND_GUI_INTEGRATION_MATRIX.md`.
- Per-round engineering ledger: `CHANGELOG.md` + repo-root `worklog.md`
  + `PROJECT_AUDIT_PROGRESS.md`.

## 2. Verified current state (evidence gathered this round)

| Item | Value | Evidence (this session) |
| --- | --- | --- |
| Repository | public, default branch `main` | GitHub API, authenticated |
| HEAD | r307 audited at `5446e89b`; the Phase-2 addendum verified `19effec8`; the Phase-3 addendum verifies `ca1accc9`; the Phase-4 addendum verifies `1bbe7316`; the Phase-5 addenda verify the r312 push-2 terminal (`e28ea93c`), the r313 push-2 terminal (`ac5bcd6b`), the r315 push (`b8f2e1cf`), the r316 push-3 (`74b81167` — the r317 continuation opened on it, the post-session machinery ghost-zip commit `9e42c69e` reset away unpushed per the r312 precedent, the disk homes restored with five-home sha parity `6fc0d25e` exact), and this round's r317 push
  (the r318 continuation opened on its push-2 re-ship `7df1806d`, the
  post-session machinery ghost-zip commit `c07a8a12` reset away
  unpushed per the r312 precedent, the disk homes restored with
  five-home sha parity `455cf00b` exact), and this round's r318
  push | fetch + reset --hard + ghost sweep (r307–r318) |
| CI at HEAD | the r317 push-2 `7df1806d` came back **19/19 — all
  check-runs success**, closed GREEN on arrival this round (the
  same-version re-ship convention held through the superseding
  re-ship push); this round's push re-verifies all jobs | the
  authenticated check-runs poll (this round) |
| Published release | v1.2.231 (not draft): EXE 557,655,583 B · MSI 710,122,853 B · RELEASE_MANIFEST.json · SHA256SUMS.txt — the v1.2.240 SOURCE archive shipped (four homes + offsite, 18 zips) but the Windows release lane stays gated on the clean machine (§4); no silent release claims | `releases/tags/v1.2.231` |
| Version carriers | VERSION · pyproject · `pharmaml.__version__` · TS carrier · CHANGELOG · tauri.conf.json · version_info.txt — all 1.2.240 | read directly this session, lockstep guard green |
| Test surface | 8,484 collected / 0 errors (`tests/` + `src/pharmaml` `--doctest-modules`, 465 files) — +20 this round (the multiobjective route family the interrupted continuation left; its API lane gated green, its GUI leg rides the next continuation) — PLUS the component-test lane: 94 Vitest tests / 13 files over the Studio frontend (jsdom + React Testing Library) | fresh collection + vitest this round |
| Census | 1,675 files / 563,174 lines / v1.2.240 (the r319 housekeeping census, trued three-way: the walk replica re-aligned to the r302 scoped-exclusion rule — artifacts only at its two relative depths, the r302 find that had hidden the frontend's real artifacts client — so the walk = the live API = the page render agree byte-for-byte for the first time since r302; the multiobjective API lane's three source files ride, the r318 close-out tooling trio's leakage literals scrubbed, the hydration-restored ghost zips and build-theme shots swept) | this round's census walk |
| Source archive | v1.2.240 — a NEW version ship (the part13 Phase-5 Pharmacogenomics lane — the route family + its tests + the GUI stage + the regenerated API types + the re-executed notebooks ride; the per-round sha/bytes live in the worklog per §7 — the archive cannot carry its own hash) — re-shipped same-version at r319 (the r303/r304/r317 convention: the leakage scrub + this row's re-true ride the rebuilt archive) | sha256 across four homes + offsite |
| venv pins | numpy 1.26.4 · torch 2.14.1+cpu · rdkit 2026.09.1 · sklearn 1.5.2 · pandas 2.2.3 · pharmaml 1.2.240 — zero drift vs `requirements.txt` | import probes (r319: rdkit was found missing after the eleventh rollback's partial hydration — `venv_repair.sh` ran FOREGROUND and the full import surface + the Studio GUI stack verified) |

## 3. The six previously-open audit items — CLOSED (re-verified in-tree this round)

1. **Standalone Leverage/Williams AD** — CLOSED r303:
   `ml/applicability.py::leverage_williams` (hat-matrix leverage,
   h\* = 3(p+1)/n, standardized-residual band, regression-only guards)
   + `plot_williams` + `/api/v1/applicability-domain/leverage-williams`
   (job/sync + figure artifact + the four-entry catalog) + the ML
   workspace's Williams lane; 23 unit + 3 figure + 7 route tests.
2. **Fingerprint chirality configuration** — CLOSED r303:
   `include_chirality` through `compute_fingerprints` /
   `FeatureConfig` / `/api/v1/chemistry/fingerprints` + the Datasets
   card (morgan-only, default OFF — byte-identical history);
   8 descriptor + 8 factory + 3 route tests.
3. **Semi-supervised `/evaluate` GUI integration** — CLOSED r303: the
   GUI consumer of `POST /semi-supervised/evaluate` landed.
4. **Semi-supervised job-mode argument correction** — CLOSED r303: the
   route's job mode passed a spurious `manager` kwarg (every job
   submission raised TypeError while sync masked it); fixed and pinned
   (the COMPLETION pin rides RELEASE_READINESS).
5. **Tk playground retirement** — CLOSED r304: `playground.py`
   (30,047 lines) + `tests/test_playground.py` (16,370 lines) left the
   tree; the v1.2.141 declaration executed; the zip builder's NEGATIVE
   gate pins the pair's absence; collection re-trued 8,643 → 8,362.
6. **SBOM generation + release attachment** — CLOSED r304:
   `packaging/sbom/sbom.cdx.json` (CycloneDX 1.6, 40 components,
   validated + reproducible) ships in the archive; the Windows
   workflow's step 22b regenerates it on the runner and self-registers
   its hash into SHA256SUMS.txt.

## 4. The three honest BLOCKED lanes (unchanged gates; Phase 2 executed around them)

- **C-3/C-5 clean-machine smoke test** — a genuine Windows 10/11 x64
  machine (physical or fresh VM, no dev tools) is required; a GitHub
  Windows runner is not a clean machine. BLOCKED.
- **Stage-5 PySide6 retirement** — gated behind the clean-machine pass
  + demonstrated feature parity (the rule the campaign set). BLOCKED.
  Phase-2 decision record (round 309): retirement does NOT proceed this
  phase — the parity ledger (§3.5: 65/73 PASS React rows, 6 PARTIAL)
  plus the machine gate both stand; the decision is now explicit, not
  implicit.
- **Installer digital signing** — certificate-gated (Authenticode +
  timestamping; secrets in GitHub Actions, never in the repository).
  BLOCKED. Phase-2 deliverable: the exact requirement + staged lane now
  documented in `docs/security/INSTALLER_SIGNING_ASSESSMENT.md` (round
  309) — the gate opens the day the certificate exists in Actions
  secrets; nothing about the current unsigned state was embellished.

## 5. Release posture (part13 §5)

- The published v1.2.231 binaries were built from an earlier commit
  than `main` — do not assume they include later source changes.
- Source-tree changes re-ship the archive at the SAME version (the
  r303/r304 same-version convention; superseding shas recorded in the
  CHANGELOG/worklog per the r301 three-sha convention). The published
  Windows assets are never overwritten by source re-ships.
- A NEW version (v1.2.232+) ships only after the applicable release
  gates pass (part13 Phase 7) — which requires the BLOCKED lanes to
  clear first. No silent version bumps, ever.

## 6. Next-campaign priorities (part13 Phases 2–7, in order)

- **Phase 2** — EXECUTED (round 309, the interactive continuation):
  the six items each got their in-tree leg (the offline-runtime config
  audit — offlineInstaller WebView2 + frozen-own-Python sidecar +
  compiled host; the upgrade/uninstall + user-data design proof —
  `%LOCALAPPDATA%` data home outside the install dir + 67-test
  backup/restore surface; the PySide6 decision record; the signing
  assessment deliverable). The machine/certificate legs stay honestly
  BLOCKED — full evidence table in
  `docs/part12_audit/WINDOWS_CLEAN_INSTALL_REPORT.md` § "part13
  Phase 2".
- **Phase 3** — EXECUTED (round 310, the interactive continuation,
  trace 1a1fb0a0b935472d): the component-test lane is real — Vitest
  5.1 + jsdom 31 + React Testing Library 16.3 + user-event 14.6 +
  jest-dom 8 as pinned devDependencies (npm-locked for the CI's
  `npm ci`), `vitest.config.ts` + the setup module, and 7 test files /
  48 tests over the REAL component tree: the shell navigation, the
  §14.1 startup-gate states + §14.2 controlled restart, the §13
  event-stream reconnection engine (hello/replay/backoff/liveness/
  shutdown), the nine-workspace registry guard, the UI atoms'
  a11y contracts, the §22 chart-surface contracts (hovermode pin,
  point identity, purge), and the Datasets workflow's first stage
  end-to-end (file selection, upload submission, typed API errors,
  job progress + cancellation, the COMPLETED relay, the registry
  table + row limit, the registry failure state). The CI
  studio-frontend job runs the lane after the build. **CLOSED by the
  round-311 continuation** (trace 1a1fc9a07e081d59): the queued
  browser-E2E lanes executed over the vite proxy (the r290/r291
  daemon pattern — the double-forked FastAPI backend on 8017 + Vite
  on 5174 with the dev-same-origin token channel, both surviving the
  sandbox reaper): the §14.1 app boot, the Generative AI workspace's
  full §11.4 workflow (the genetic-optimization job + verdict, the
  descriptor-engine validation Valid—10/Invalid—0, Filter/Rank),
  the honest error-handling lane (the similarity objective's own 422
  verbatim), the Visualization workspace's stat/md-series/report
  lanes + the artifact registry with the backend's sha256 digests,
  and the visual-regression baselines (desktop + 390px mobile, zero
  overflow + dark) — zero page/console errors across the run, 10
  evidence screenshots under `docs/gui/screenshots/r311-*`.
- **Phase 4** — CLOSED (rounds 311 + 312): the r311 §9.3 leg (the
  scaffold-aware coverage bridge, `ml/scaffold_coverage.py`, 18
  tests/doctests) + the r312 API connection — the r311-queued
  Model-Evaluation/Applicability-Domain row EXECUTED: the
  `/api/v1/model-evaluation` family over the REAL backend —
  `conformal-regression` (`calibrate_split_conformal` +
  `predict_interval` + empirical coverage + optional
  groups/scaffold-aware diagnostics), `conformal-classification`
  (split | label-conditional + per-class coverage rows), and
  `coverage-diagnostics` (the standalone Wilson report, groups |
  scaffold-aware) — 17 route tests, the OpenAPI contract re-exported
  (90 paths). The remaining §9.1/§9.2/§9.4/Report-Builder/Model-Card
  queued remainders stay queued, each with its honest status:
  **§9.1 Active Learning** REAL in-tree (the chemical-space
  integration queued); **§9.2 Chemical Space Explorer** scattered-real
  (the integrated workspace queued); **§9.4 Automated Benchmarking**
  real (the shared-protocol comparison surface queued); **Report
  Builder** `viz/report.py` real (§12's checklist queued); **Model
  Cards** `ml/model_card.py` real (§13's field audit queued).
- **Phase 5** — OPENED (round 312, the first item's API leg): the
  GNN/MPNN API lane EXECUTED — `POST /api/v1/dl/gnn-train` (the REAL
  `GraphClassifier` over inline graph specs or the synthetic motif
  fixture, sync/job, honest stage-level telemetry — the torch GNN
  trainer has no live per-epoch callback, loss_history_ lands after
  the fit, no fabricated epoch stream) + `POST /api/v1/dl/gnn-predict`
  (the graph models' own prediction path; `/api/v1/ml/predict` and
  `/ml/explain` refuse graph models up front with the pointer), 13
  lane tests + the node-feature width guards. **Round 313 closed the
  first remainder item — the GNN GUI card in the Deep Learning
  workspace** (the frontend lane over the API): the graph-lane section
  with the live vocabulary selects off the catalog's `gnn` block, the
  synthetic/inline source switch with client-side validation before
  any request, the full trainer controls, the job-mode COMPLETED
  relay into the graph model card, and the predict lane on the
  graph models' own path — 7 component tests + the browser-E2E golden
  path (zero page/console errors, evidence screenshots r313-*).
  The REMAINDER rides the next continuation: **Round 314 EXECUTED the
  ADMET item — both legs in one round** (trace 1a1fd9525b9d5b89): the
  API lane (`GET /api/v1/admet/endpoints` — the vendored
  swansonk14/admet_ai 52-endpoint registry with the quality joins,
  the low-confidence warning set, the radial axes with directionality
  and the honest provenance block; `POST /api/v1/admet/score` — the
  per-molecule scorecard over a SMILES batch: the 11-property
  physchem panel with the PAINS/BRENK/NIH FilterCatalog counters, the
  five-axis radial summary on the DrugBank quantile grid, the
  low-confidence list — sync-only light compute, batch cap 100,
  invalid rows reported per-row, the §11.1 disclaimer riding every
  answer; 8 route tests) + the GUI lane (the "ADMET Profile" stage
  in the Machine Learning workspace — the registry vocabulary line,
  the SMILES textarea with client-side validation before any
  request, the per-molecule scorecard cards with the radial bars,
  the honest verdict + provenance + disclaimer lines; 7 component
  tests + the browser-E2E golden path). §11.1's own rules held: the
  41 TDC-trained ensembles are NOT shipped and never invented — the
  provenance block states the absence; predicted values are model
  estimates, never experimental facts. **Round 315 EXECUTED the DTA
  item — both legs in one round** (trace 1a1fdc2da50cd529): the API
  lane over the REAL in-tree backend `ml/dta.py` (`GET
  /api/v1/dta/registry` — DeepDTA's three alphabets verbatim, the AAC
  arithmetic, the honest 0-checkpoints provenance; `POST
  /api/v1/dta/features` — the paired featurization, per-row verdicts,
  the backend's own hard 422s) + the GUI lane (the "Drug–Target
  Interaction" stage, stage 12) + the browser-E2E golden path. **Round
  316 EXECUTED the Drug Repurposing item — both legs in one round**
  (trace 1a1fe5e3bc4ba0f5): the API lane over the REAL in-tree
  backend `ml/response_model.py` (`GET /api/v1/repurposing/registry`
  — DeepResponse's split ladder verbatim + the CaDRReS formulation +
  the ten-key metric contract + the honest 0-checkpoints provenance
  with the SYNTHETIC-screen honesty; `POST
  /api/v1/repurposing/screen` — the full real round trip: the seeded
  screen → the CaDRReS fit → the predicted IC50 matrix → the metric
  contract on the observed entries → the ranked candidates per
  profile, non-finite metrics as null) + the GUI lane (the "Drug
  Repurposing" stage, stage 13) + the browser-E2E golden path (zero
  page/console errors, the r316 evidence pair). **Round 317 EXECUTED
  the Cancer Cell-Line Drug Response item — both legs in one round**
  (trace 1a1fe97d37b3b40d): the API lane over the REAL in-tree
  backend `ml/drugresponse.py` (`GET /api/v1/drugresponse/registry`
  — the bilinear CaDRReS-style formulation with its sigmoid link +
  the five-key metric contract + the sensitivity-ranking and
  biomarker schemas + the cross-cell-line fold rule + the
  one-response-scale harmonisation note + the honest 0-checkpoints
  provenance with the SYNTHETIC-screen honesty; `POST
  /api/v1/drugresponse/analyze` — the full real round trip: the
  seeded screen → the bilinear fit → the predicted response-score
  matrix → the metric contract on the observed entries → the
  per-cell sensitivity ranking (most-responsive first) → the
  per-drug biomarker weights (the composed `W w_d` readout) → the
  cross-cell-line validation folds with their across-fold summary)
  + the GUI lane (the "Cancer Cell-Line Drug Response" stage, stage
  14, the later stages renumbered) + the browser-E2E golden path
  (zero page/console errors, the r317 evidence pair). **Round 318
  EXECUTED the Pharmacogenomics item — both legs in one round**
  (trace 1a1fed8994f477c1): the API lane over the REAL in-tree
  backend `ml/pharmacogenomics.py` (`GET
  /api/v1/pharmacogenomics/registry` — the module's own vocabulary
  verbatim: the classic GDSC ANOVA design matrix `[1 | tissue
  dummies | msi | feature]` with its sequential QR F-test + the four
  feasibility gates, the 18-key row schema, the effect-size contract,
  the λ-grid Storey-Tibshirani q-values + the BH FDR with gdsctools'
  percent convention, the volcano transforms + the default hit rule,
  the drevalpy degenerate-input guards + the additive mean-effects
  ruler + the normalized metric, the cBioPortal vocabularies (the
  CNA codes, the diploid z-scores, the co-expression closed form, the
  enrichment dispatchers), and the honest 0-tables/0-checkpoints
  provenance with the SYNTHETIC-screen honesty; `POST
  /api/v1/pharmacogenomics/analyze` — the full real round trip: the
  seeded screen → the real GDSC ANOVA over every feasible drug ×
  feature pair → the q-value/FDR columns → the volcano verdict census
  → the top associations table → the tissue-aware mean-effects ruler
  evaluated through the guarded family with the normalized score
  beyond the naive drug-mean ruler; 15 route tests) + the GUI lane
  (the "Pharmacogenomics" stage, stage 15, the later stages
  renumbered) with client-side validation + 8 component tests + the
  browser-E2E golden path over the REAL backend (zero page/console
  errors, the r318 evidence pair). The guards trued in the same
  commit (the ml client REST-surface module set + the §11.2 stage
  chain carry the pharmacogenomics rows; the CI guard lane 129 passed
  locally BEFORE the push); the OpenAPI contract re-exported (100
  paths / 68 schemas) + the types regen byte-identical; the 8
  notebooks re-executed at v1.2.240. The remaining domain
  (multi-objective molecular optimization `ml/multiobjective.py` —
  a real backend already in-tree) awaits its API + GUI lanes.
- **Phase 6** — architecture hardening: the lightweight workflow
  engine evaluation (JobManager/capability-registry first),
  resource monitoring, security scanning lanes, documentation
  automation.
- **Phase 7** — the next production release, from the exact release
  commit, with all gates recorded.

## 7. Known limitations of this document

- Census/test numbers are as-of this round's runs; each round's ledger
  entries re-true them (the pins live in the web carrier
  `toolkit-stats.ts` + the live `/api/toolkit/tree` walk).
- The r307 same-version re-ship's sha/bytes are recorded in the
  CHANGELOG + worklog, not here — those move with each re-ship; this
  file records lanes and their statuses, not per-round bytes.
