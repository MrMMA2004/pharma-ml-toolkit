/** Single source of truth for every shipped-census number the UI paints
 *  before (or instead of) the live ``/api/toolkit/tree`` census.
 *
 *  (r33 review C-3.7 / 4.12): the test count, file count and registry
 *  size used to be hardcoded in six places across three components —
 *  and disagreed with each other and with the shipped docs. Every
 *  surface now imports from here; the release process bumps these
 *  alongside the four version carriers (VERSION, pyproject,
 *  pharmaml.__version__, toolkit-version.ts).
 */

/** Collected pytest cases at the shipped revision (repo-wide
 *  ``pytest --collect-only -q`` sum, re-verified on the part12 Phase-3
 *  session's venv, 2026-11-20: 8,362 collected / 0 errors — the r304
 *  re-true over the r303 pin (8,643): −753 the retired Tk playground's
 *  suite (test_playground.py left the tree with playground.py — the
 *  v1.2.141 declaration completed) + 472 the new architecture-hygiene
 *  guard (tests/test_architecture_hygiene.py: the duplicate-symbol map
 *  pins, the library-surface import-graph census, the mabel/noael
 *  parity grid, the SBOM artifact pins). */
/*  r367 re-true (measurement-last, the part18 second increment): the
    ui/-pages quarantine completion MOVED the 972-test legacy GUI
    family out of the default gate (tests/ui -> legacy/tests_ui,
    run explicitly when PySide6 is installed), and the DL
    architecture expansion added 110 (83 backend + 27 API) -- the
    default-suite collection re-measured live: 8,274 over 441
    files, 0 errors (the legacy lane: 972 over 74 files, explicit).
    Verified pin = live collection. */
/*  r371 re-true (the v1.2.244 release ship, measurement-last): the
    default-suite collection re-measured live on the shipped tree --
    8,286 collected / 0 errors over 442 files (the +12 over the r367
    pin 8,274/441: the r368 test_ml_catalog suite (6) + the D-9-era
    chemistry-lane pins; the shipped zip now carries the working
    tree, so the pin EQUALS the live collection). */
/*  r373 re-true (the v1.2.245 part21 ship, measurement-last): the
    collection re-measured live on the shipped tree -- 8,293
    collected / 0 errors over 443 files (the +7/+1 over the r371
    pin: tests/api/test_ml_predict_evaluation_curves.py, part21's
    shared ML/DL evaluation-contract suite). */
/*  r374 re-true (the v1.2.246 DS-1 ship, measurement-last): the
    collection re-measured live on the shipped tree -- 8,313
    collected / 0 errors over 444 files (the +20/+1 over the r373
    pin: the DS-1 strategy lane -- 13 split-strategy API tests in
    tests/api/test_datasets_ml_artifacts.py + the 7-test drift guard
    tests/test_frontend_split_strategy.py). */
/*  r375 re-true (the v1.2.247 ML-3 ship, measurement-last): the
    collection re-measured live on the shipped tree -- 8,329
    collected / 0 errors over 444 files (the +16 over the r374 pin:
    the ML-3 train-lane split section -- 9 train-split API tests in
    tests/api/test_datasets_ml_artifacts.py + the drift guard
    extended from 7 to 14 tests). */
/*  r376 re-true (the v1.2.248 ML-1 ship, measurement-last): the
    collection re-measured live on the shipped tree -- 8,336
    collected / 0 errors over 445 files (the +7/+1 over the r375
    pin: the ML-1 model-evaluation consumer lane -- the new 7-test
    drift guard tests/test_frontend_model_evaluation.py). */
/*  r377 re-true (the v1.2.249 GEN-1 ship, measurement-last): the
    collection re-measured live on the shipped tree -- 8,340
    collected / 0 errors over 445 files (the +4 over the r376 pin:
    the GEN-1 operator-mode route tests in
    tests/api/test_dl_semi_generative.py). */
/*  r378 re-true (the v1.2.250 GEN-2 ship, measurement-last): the
    collection re-measured live on the shipped tree -- 8,347
    collected / 0 errors over 445 files (the +7 over the r377 pin:
    the six GEN-2 metrics route tests in
    tests/api/test_dl_semi_generative.py + the metrics mirror guard). */
/*  r380 re-true (the v1.2.252 DS-3 ship, measurement-last): the
    collection re-measured live on the shipped tree -- 8,388
    collected / 0 errors over 447 files (the +41 over the r378 pin:
    the r379 DS-2 lane (9 API tests + 10 drift-guard tests + the
    vitest pair) + the r380 DS-3 lane (10 API tests + 13 drift-guard
    tests + the 3 vitests) + the notebook/contract churn; the r379
    entry's claimed 8,366 re-true never rode its commit -- this pin
    carries the LIVE measurement, caught and recorded honestly at
    r380). */
/*  r381 re-true (the v1.2.253 PK-1 ship, measurement-last): the
    collection re-measured live on the shipped tree -- 8,398
    collected / 0 errors over 449 files (the +10 over the r380
    pin: the PK-1 drift-guard's ten tests). */
/*  r382 re-true (the v1.2.254 ML-4 ship, measurement-last): the
    collection re-measured live on the shipped tree -- 8,438
    collected / 0 errors over 451 files (the +40 over the r381
    pin: the three roadmap rows' guards and API lane -- the 11
    VIZ-1 drift-guard tests + the 9 ML-2 drift-guard tests + the
    9 ML-4 drift-guard tests + the 11 ML-4 API tests in
    tests/api/test_unsupervised_reinforce.py). */
export const SHIPPED_TEST_COUNT = 8438;

/*  r366 (the part18 all-phases verification pass, the owner-directed
    firing): the collection re-measured LIVE at the r365 terminal
    762f60a5 -- 9,123 / 0 errors / 503 files, double-measured
    identical (the +1 over the r365 narrative's 9,122 decoded: the
    3237c983 regression guard post-dates that census tree a92b51b6
    -- the claim was honest for that tree, this is the current
    truth). The r361-era pin 9,017 had drifted 106 tests behind the
    part18 lanes (the clustering families + Isolation Forest + the
    RL lane + the retirement guards) and the v1.2.242 release never
    re-trued the carriers -- the same silent-drift class this file's
    own history warns about. Measurement-last: re-taken after this
    round's ledger edits. */

/** r315 note: the canonical collection re-verified at 8,428 / 0
 *  errors / 461 files (+10 the DTA route family, − 1 re-trued A-6
 *  param: ml.admet left the doctest-only tier when its route consumer
 *  landed at r314). */

/** Collected Vitest component tests at the shipped revision — the
 *  part13 Phase-3 lane + the Phase-5 GNN lane (2026-11-21, `npm
 *  --prefix frontend test`: 8 files / 55 tests over jsdom + React
 *  Testing Library: the shell navigation, the §14.1 startup-gate
 *  states + controlled restart, the §13 event-stream reconnection
 *  engine, the registry guard, the UI atoms' a11y contracts, the
 *  §22 chart surface contracts, the Datasets workflow's lanes, and
 *  the Deep Learning workspace's GNN/MPNN lane (the live-vocabulary
 *  read, the honest request shape, the job relay, the predict table,
 *  the typed 422 verbatim, the client-side gating — the r313
 *  addition). This pin moves independently of the Python suite
 *  testing-library: 10 files / 70 tests (jsdom + React Testing
 *  Library — the vitest file-count true: the r314 row's own "10
 *  files" was off by one, the on-disk truth at 315fdddf was 9).
 *  This pin moves independently of the Python suite above —
 *  the CI's studio-frontend job runs both lanes. */
/*  r367 re-true: the vitest suite re-measured live at the r367
    tree -- 369/369 over 43 files (the +12 over the r371 pin: the
    part21 shared-evaluation suite (10) + the DL Evaluate-stage
    integration pair); 357/357 over 41 files was the r371 pin (the two new lanes: the CNN +
    autoencoder lane and the feature-generative lane, +11). */
/*  r374 re-true (the v1.2.246 DS-1 ship, measurement-last): the
    suite re-measured live on the shipped tree -- 376/376 over 44
    files (the +7/+1 over the r373 pin 369/43: the DS-1 lane -- the
    4-test split-parser suite + the 3-test Split-stage strategy
    suite). */
/*  r375 re-true (the v1.2.247 ML-3 ship, measurement-last): the
    suite re-measured live on the shipped tree -- 386/386 over 46
    files (the +10/+2 over the r374 pin: the ML-3 lane -- the
    7-test train-split parser suite + the 3-test Configure Split/CV
    strategy suite). */
/*  r376 re-true (the v1.2.248 ML-1 ship, measurement-last): the
    suite re-measured live on the shipped tree -- 403/403 over 48
    files (the +17/+2 over the r375 pin: the ML-1 lane -- the
    10-test model-evaluation parser suite + the 7-test conformal
    panel page suite). */
/*  r377 re-true (the v1.2.249 GEN-1 ship, measurement-last): the
    suite re-measured live on the shipped tree -- 405/405 over 49
    files (the +2/+1 over the r376 pin: the GEN-1 operator-mode
    select-passthrough page suite). */
/*  r378 re-true (the v1.2.250 GEN-2 ship, measurement-last): the
    suite re-measured live on the shipped tree -- 407/407 over 50
    files (the +2/+1 over the r377 pin: the GEN-2 MOSES metrics
    card page suite). */
/*  r380 re-true (the v1.2.252 DS-3 ship, measurement-last): the
    suite re-measured live on the shipped tree -- 412/412 over 50
    files (the +5 over the r378 pin: the r379 DS-2 page pair + the
    r380 DS-3 register-lane trio). */
/*  r381 re-true (the v1.2.253 PK-1 ship, measurement-last): the
    suite re-measured live on the shipped tree -- 419/419 over 51
    files (the +7/+1 over the r380 pin: the PK-1 bridge-lane page
    suite over the full PK/PD page mount). */
/*  r382 re-true (the v1.2.254 ML-4 ship, measurement-last): the
    suite re-measured live on the shipped tree -- 444/444 over 55
    files (the +25/+4 over the r381 pin: the three roadmap rows'
    vitest lanes -- the 6-test VIZ-1 column-bridge page suite + the
    12-test ML-2 pair (the 6 client-op tests + the 6 AD sub-lane
    page tests) + the 7-test ML-4 unsupervised-methods lane). */
export const SHIPPED_COMPONENT_TEST_COUNT = 444;

/*  r366 re-true: the vitest suite re-measured live at the r365
    terminal -- 346/346 over 39 files (54.36s). The r361-era pin 337
    (38 files) had drifted +9 tests behind the part18 GUI lanes
    (the Unsupervised + Reinforcement Learning workspace sections
    + their service/tests). The pin now EQUALS the live run. */

/*  r361 re-true (superseded by the r366 pin above): 337/337 over 38
    files was the r360 pin, measured on the repaired venv. */

/*  r322 (the 1.2.240 same-version re-ship, the §8 workflow lane):
    +39 the workflow engine + route suites (22 engine + 17 route —
    the canonical collection re-trued 8,523/0/467 files);
    verified pin = live run. */

/*  r323 (the 1.2.240 same-version re-ship, the §8 GUI leg): +2 the
    workflow-lane drift guards (the WorkflowRunRequest DTO parity +
    the mirrored route bounds); the canonical collection re-trued
    8,525/0/467 files; verified pin = live run. */

/*  r324 (the 1.2.240 same-version re-ship, the Phase-6 remainder): +30
    the §15/§17/§18 lanes (9 the resource-monitoring route suite + 8 the
    supply-chain guard + 13 the documentation-consistency guard); the
    canonical collection re-trued 8,555/0/470 files; verified pin =
    live run. */

/** File count of the shipped toolkit tree (src, tests, examples, ui,
 *  docs, tools — the same census /api/toolkit/tree walks). The round-302
 *  re-true: 1,591 — the part12 audit-round tree: the r301 pin (1,583)
 *  counted two untracked build/theme-shots PNGs that never committed
 *  (the rollback dropped them); this round the walk finally sees the
 *  true tracked tree — +1 the artifacts client the any-depth `artifacts`
 *  exclusion used to hide (B-1), +1 the new database client (B-2), +9
 *  the part12_audit deliverables, -1 the removed malformed nested
 *  ci.yml (B-3). The round-304 re-true: still 1,591 — the Tk playground
 *  and its suite left the tree (-2, B-6) offset by the
 *  architecture-hygiene guard (+1) and the SBOM artifact (+1). The
 *  round-307 re-true (part13 Phase 1): 1,592 — +1 the authoritative
 *  docs/PROJECT_STATUS.md. The round-309 re-true (part13 Phase 2):
 *  1,593 — +1 the installer-signing assessment deliverable
 *  (docs/security/INSTALLER_SIGNING_ASSESSMENT.md). The round-310
 *  re-true (part13 Phase 3, the component-test lane): 1,608 — +9
 *  the vitest lane itself (the 7 test files under the src
 *  __tests__ folders, the test setup module, the vitest config), +4
 *  the retired v1.2.229/230 zip pair in both homes the rollback
 *  restored to disk, +2 the v1.2.232 ship zip in its two in-tree
 *  homes, and the stale pre-r293 windows-release.yml ghost at the
 *  toolkit .github path — resurrected by the same rollback, never
 *  registered at GitHub and never shipped in any archive — nets to
 *  zero (caught, removed); verified pin = live walk = page render. */
/*  r367 re-true (measurement-last): the walk re-measured after the
    quarantine completion (152 paths MOVED under legacy/, not added)
    + the twelve r367 files (the five DL modules, the five backend
    suites, the two vitest lanes) + the two in-tree v1.2.243 zip homes
    -- 1,873 files. */
/*  r371 re-true (the v1.2.244 release ship, measurement-last, the
    offline census replica — the r366 precedent — because the dev
    server's tree route sat wedged on the documented Turbopack
    SST-persist class this round): the walk re-measured at the
    close-out fixed point (after ALL in-tree ledger edits — the QA
    r371 section + the RELEASE_READINESS r371 addendum + the
    CHANGELOG amendment + the committed collect_default_371.txt):
    1,891 files / 168 dirs / 734,337,825 B (the +18 files over the
    r367 pin 1,873: the r368..r370 QA deliverable set + the
    clean-machine suite + test_ml_catalog + the D-9/D-10 fixes +
    the r371 scratch gate-count + collection txt files + the two
    new in-tree v1.2.244 zip homes). */
/*  r373 re-true (the v1.2.245 part21 ship, measurement-last): the
    offline census replica re-walked the toolkit tree at the
    close-out fixed point -- 1,914 files / 171 dirs / 791,009,552 B
    (the +23 files over the r371 pin 1,891: the part21 shared
    evaluation layer's 13 component files + the training chart + the
    backend evaluation_curves module + the two new test files + the
    four part21 QA deliverables + the two in-tree v1.2.245 zip
    homes). */
/*  r374 re-true (the v1.2.246 DS-1 ship, measurement-last): the
    offline census replica re-walked the toolkit tree at the
    close-out fixed point -- 1,916 files / 172 dirs / 790,904,603 B
    (the +2 files / +1 dir over the r373 pin: the new drift guard
    tests/test_frontend_split_strategy.py + the new
    services/datasets/__tests__/client.test.ts + its directory; the
    byte total reflects the v1.2.245 → v1.2.246 zip-home swap —
    both in-tree homes swept by the ship tool — plus the notebook
    re-execution outputs). */
/*  r375 re-true (the v1.2.247 ML-3 ship, measurement-last): the
    offline census replica re-walked the toolkit tree at the
    close-out fixed point -- 1,920 files / 172 dirs / 847,453,727 B
    (the +4 files over the r374 pin: the two ML-3 vitest files +
    the two new in-tree v1.2.247 zip homes (the toolkit download/ +
    the toolkit dist-archive/ — the v1.2.246 pair stays, both homes
    accumulate per the ship convention; the +56.5 MB is exactly the
    two new 28,242,985-byte copies)). */
/*  r376 re-true (the v1.2.248 ML-1 ship, measurement-last):
    offline census replica re-walked the toolkit tree at the
    close-out fixed point -- 1,925 files / 172 dirs / 904,061,972 B
    (the +5 files over the r375 pin: the two ML-1 vitest files +
    the new drift guard tests/test_frontend_model_evaluation.py +
    the two new in-tree v1.2.248 zip homes (the toolkit download/ +
    the toolkit dist-archive/ -- both homes accumulate per the ship
    convention; the +56.6 MB is exactly the two new 28,262,278-byte
    copies)). */
/*  r377 re-true (the v1.2.249 GEN-1 ship, measurement-last):
    offline census replica re-walked the toolkit tree at the
    close-out fixed point -- 1,928 files / 172 dirs / 960,623,588 B
    (the +3 files over the r376 pin: the new OperatorModesLane vitest
    file + the two new in-tree v1.2.249 zip homes (the toolkit
    download/ + the toolkit dist-archive/ -- both homes accumulate
    per the ship convention; the +56.6 MB is exactly the two new
    28,269,061-byte copies)). */
/*  r378 re-true (the v1.2.250 GEN-2 ship, measurement-last):
    offline census replica re-walked the toolkit tree at the
    close-out fixed point -- 1,931 files / 172 dirs / 1,017,224,581 B
    (the +3 files over the r377 pin: the new MetricsLane vitest file
    + the two new in-tree v1.2.250 zip homes (the toolkit download/ +
    the toolkit dist-archive/ -- both homes accumulate per the ship
    convention; the +56.6 MB is exactly the two new 28,278,832-byte
    copies)). */
/*  r380 re-true (the v1.2.252 DS-3 ship, measurement-last): the
    offline census replica re-walked the toolkit tree at the FINAL
    close-out fixed point -- 1,931 files / 172 dirs / 960,704,980 B
    (the r380 frames-register drift-guard file + the two 252 zip
    homes, less the predecessor churn: the three-build ship's
    retention rule removed the 249/250/251 predecessors from the
    toolkit's two in-tree homes -- the first post-build measurement
    read 1,935/1,073,799,196 B before the churn finished; the final
    fixed point re-measured after the last build is the pin). */
/*  r381 re-true (the v1.2.253 PK-1 ship, measurement-last): the
    offline census replica re-walked the toolkit tree at the FINAL
    close-out fixed point -- 1,933 files / 173 dirs / 961,637,766 B
    (the +2/+1 over the r380 pin: the PK-1 vitest file + the PK-1
    drift guard + the two v1.2.253 zip homes in the toolkit's
    download/ + dist-archive/ against the retention-removed 252
    predecessors -- the net zip churn is +1 file); the r381 census
    re-walked AFTER the sandbox recycle's resurrection class was
    swept: the recycle restored six untracked r185-era leftovers
    into the toolkit tree (the 103/105/106 zips + the 30K-line
    playground pair + a corrupted truncated .github/workflows/ci.yml
    -- "branches: ain]") that the r380 fixed point never carried --
    the first post-sweep reading of 1,939 files / 657,355 lines /
    1,014,673,554 B was the pollution, the pinned numbers are the
    post-sweep live truth). */
/*  r382 re-true (the v1.2.254 ML-4 ship, measurement-last): the
    census taken over the FINAL close-out fixed point -- the tracked
    set after this round's commit (1,933 minus the retention-removed
    two toolkit 253-zip homes, plus this round's two 254-zip homes)
    plus the seven new test files -- 1,940 files / 961,015,202 B
    (the method that reproduces the r381 pin exactly: the r381
    commit re-counted under it reads 1,933 files / 961,637,766 B,
    byte-identical to the recorded pin; the byte delta is the zip
    churn -- four 253 homes out, four 254 homes in -- plus the
    notebook-output churn + the round's own lanes). */
export const SHIPPED_FILE_COUNT = 1940;

/*  r366 re-true (measurement-last): the offline census replica
    re-walked the toolkit tree at the r365 terminal AFTER this
    round's own ledger edits -- 1,860 files / 167 dirs (the +16
    files over the r361-era pin 1,844 are the part18 campaign's
    tree growth: the clustering/anomaly/reinforce module family +
    their two route families + the unsupervised.ts service client
    + the retirement guards + docs/DESIGN.md + the r365 ledger
    deltas). The pin now EQUALS the live walk (the exact
    toolkit-fs.ts walk semantics, the canonical replica). */

/*  r322 (the 1.2.240 same-version re-ship, the §8 workflow lane):
    +13 tree files (the pharmaml.workflow package x4, the route
    family, the two test suites, ARCHITECTURE_REVIEW.md, the regrown
    openapi.json/api.d.ts pair, + the four r322 evidence screenshots);
    verified pin = live walk = page render; .tsx lanes never count. */

/*  r323 (the 1.2.240 same-version re-ship, the §8 GUI leg):
    +5 tree files (the WorkflowBuilderLane component suite + the
    four r323 evidence screenshots); the .tsx lanes count as tree
    nodes but never as lines, by the long-standing census rule;
    verified pin = live walk = page render. */

/*  r324 (the 1.2.240 same-version re-ship, the Phase-6 remainder):
    +6 tree files (the §15 system route + its 9-test suite + the two
    guard suites + the r324 pins pair); the root .github workflow and
    the web pins carrier live outside the walked toolkit tree, by
    the same census rule; verified pin = live walk = page render. */

/*  r325 (the 1.2.240 same-version re-ship, the Phase-7 release-gate
    verification round): +2 tree files (the r325 pins pair) +98 the
    counted doc deltas (the CHANGELOG entry + the PROJECT_STATUS
    Phase-7 addenda); no src/tests change; verified pin = live
    walk = page render. */

/** Source lines of the shipped tree, formatted with thousands
 *  separators at render time (raw number kept for math). The round-304
 *  re-true: 552,786 — the part12 Phase-3 tree (the r303 pin's 597,658
 *  minus the 46,417 retired lines — the Tk playground's 30,047 plus its
 *  753-test suite's 16,370 — plus the round's +1,545: the
 *  architecture-hygiene guard, the SBOM artifact, the ten
 *  docstring-map sites, the windows-release step 22b, the
 *  README/COVERAGE/FEATURES/RELEASE_READINESS/CHANGELOG addenda; the
 *  file count stays 1,591 — the two retirements offset by the hygiene
 *  test and the SBOM). The same-version re-ship carries exactly this
 *  state. (Honest note: the round had two concurrent close-out passes
 *  — their walks measured 552,625 pre-CHANGELOG-entry and 552,780
 *  post-entry/pre-merge; the final post-merge walk 552,786 is the
 *  canonical pin.) The round-307 re-true (part13 Phase 1): 553,002 —
 *  the r306 base plus the doc-reconciliation delta (+216:
 *  docs/PROJECT_STATUS.md, the seven stale-statement trues, the
 *  CHANGELOG entry, the PROJECT_AUDIT_PROGRESS append); the doc-only
 *  round touches zero src/tests bytes (verified by git diff). The
 *  round-309 re-true (part13 Phase 2): 553,185 — the r307 live-walk
 *  base plus the Phase-2 doc delta (+183: the signing assessment,
 *  the clean-install report's Phase-2 section, the PROJECT_STATUS
 *  addenda + row trues, the CHANGELOG entry) MINUS the round-309
 *  empty-file true-up (-4: the JS carrier's countLines() has counted
 *  the four 0-byte __init__.py files as 0 lines since 2026-10-13
 *  while the Python census replica counted 1 each — a +4 bias in
 *  every census since r303; the r307-era pin 553,002 answered a live
 *  walk of 552,998; found and trued this round — the pin now EQUALS
 *  the live /api/toolkit/tree walk); the doc-only round again
 *  touches zero src/tests bytes. The round-310 re-true (part13
 *  Phase 3, taken after the notebook re-execution and the last doc
 *  edit): 552,641 — the r309 base plus +884 the frontend
 *  package-lock.json growth (the vitest/jsdom/testing-library
 *  dev-dependency trees the component-test lane added — .json is a
 *  counted viewable extension) and +100 the round's counted doc
 *  delta (the CHANGELOG entry incl. the CI-catch bullet + the
 *  PROJECT_STATUS Phase-3 addenda; the .ts/.tsx test files are tree
 *  nodes but never counted, by the long-standing census rule)
 *  minus 330 lines of the stale pre-r293 windows-release.yml ghost
 *  the rollback resurrected (caught and removed with the census) and
 *  minus 1,198 the eight-notebook re-execution at 1.2.232 (the CI's
 *  notebook-freshness gate caught the stale 1.2.231 prints; the
 *  re-executed outputs are leaner); verified pin = live walk = page
 *  render three-way.
 *
 *  r313 (the v1.2.235 ship): +1 file the GnnLane component suite
 *  (.tsx, a tree node never line-counted), +2 the r313 E2E evidence
 *  screenshots, +2 the v1.2.235 ship zip in its two in-tree homes,
 *  +7 the scratch-drift the seventh rollback restored into the
 *  toolkit tree (probe scripts from pre-r290 rounds, back on disk
 *  after the history reset — counted by the same walk semantics the
 *  live API applies, so the three-way true holds), the .md deltas of
 *  the round's own docs (CHANGELOG + PROJECT_STATUS + FEATURES);
 *  verified pin = live walk = page render.
 *
 *  r314 (the v1.2.236 ship, the ADMET lane): +2 the route family
 *  (routes/admet.py + tests/api/test_admet.py), +1 the AdmetLane
 *  component suite (.tsx, a tree node never line-counted), +2 the
 *  v1.2.236 ship zip in its two in-tree homes, +4 the round's
 *  evidence screenshots (2 ADMET cards + the pins pair), the .md
 *  deltas of the round's own docs (CHANGELOG + PROJECT_STATUS +
 *  FEATURES), the line deltas of the two frontend carriers + the
 *  re-executed notebooks/twins — and −7 the retired ancient zips
 *  (v103/105/106/229/230 across the toolkit homes) the post-session
 *  machinery ghost-zip commit had tracked before its reset (the
 *  r312 precedent sweep — they had been counted on disk by the
 *  r313 walk); verified pin = live walk = page render.
 *
 *  The push-1 correction (the CI's own catch, the r312 pattern): the
 *  ship's first push skipped the notebook re-execution step — the
 *  eight notebooks re-executed at v1.2.235 after the red (their
 *  outputs leaner by 11 lines: 557,111 -> 557,100) + the twins
 *  re-stripped, and the root download/ home's current zip TRAKED in
 *  git again (the r312 convention — the CI checkout must see it);
 *  re-trued pin = live walk = page render. */
/*  r319 (the 1.2.240 same-version re-ship, the 2h-sync housekeeping
 *  firing): the walk re-measured 561,930 — the +14 vs the r318 pin is
 *  that round's close-out commit's own tracked-line movement after its
 *  census moment (the tooling trio rode in post-census, and its
 *  `~/z/` literals were scrubbed at the source this round); the
 *  file pin stayed 1,672 (the hydration-restored stray zips + the
 *  orphan WIP route removed, the ritual's seven home-zips kept);
 *  verified pin = live walk = page render. */
/*  r362 (the 2h-sync housekeeping firing, the fifteenth-rollback
    recovery): the walk re-measured AFTER the round's own ledger
    close-out -- the r361 lesson: its final ledger edits (the QA
    section + the cascade renumber + the clause + the addendum +
    the bullet, all inside the walked root) landed after its walk
    measurement and the committed pin rode +132 lines stale;
    measurement-last discipline: pin = live walk = page render,
    re-verified over the rewritten ledger. */
/*  r363 (the 2h-sync housekeeping firing, the sixteenth-rollback
    recovery + the r362 gate debt decoded): the walk re-measured AFTER
    the round's own ledger close-out (the measurement-last discipline
    held) -- 590,114 lines / 528,234,787 B at 1,842 files / v1.2.240:
    +153 lines exactly = the round's five inside-tree ledger deltas
    (the QA r363 section + the cascade renumber + the PROJECT_STATUS
    clause 13 + the RELEASE_READINESS addendum + the CHANGELOG bullet
    + the studio_app conftest fix, 170 insertions - 17 deletions
    verified by git numstat); the file pin stayed 1,842; the +1 dir
    (166 -> 167) decoded as the gitignored `build/` theme-shot
    artifact the gate's theme tests create -- a test residue the
    served walk never sees, not a carrier. Verified pin = live walk =
    page render. */
/*  r364 (the owner-directed release round, the v1.2.241 NEW-version
    ship): the walk re-measured AFTER the round's ledger close-out +
    the zip ship (the measurement-last discipline held) -- 590,269
    lines / 590,301 / 584,028,421 B at 1,844 files / v1.2.241
    (re-walked at the round's close-out after ALL in-tree ledger
    edits + the zap-automation tuning comment; the walk history:
    584,023,791 B pre-fix / 584,023,793 B post-SBOM /
    584,024,404 B post-psutil / 584,028,421 B close-out): the file pin
    +2 = exactly the two new zip homes (the toolkit download/ + the
    toolkit dist-archive/ carrying v1.2.241.zip), the bytes +55.8 MB
    = exactly the two zip copies (27,887,437 B x 2 post-psutil-fix), the
    lines +155
    = the round's inside-tree ledger deltas (the CHANGELOG 1.2.241
    entry + the QA r364 section + the cascade renumber + the
    PROJECT_STATUS clause 14 + the RELEASE_READINESS addendum + the
    FEATURES freeze stamp) + the eight-notebook re-execution
    movement + the psutil importorskip comment (+4) + the zap-automation tuning comment (+7) + the close-out verdict truing (+21: the QA verdict rows split + the RR/PS bullet trues), re-measured at the close-out walk. The
    collection + vitest + registry pins re-measured reproducing:
    9,017 / 337 / 33. Verified pin = live walk = page render. */
/*  r367 re-true (measurement-last): the walk re-measured after the
    DL expansion + the quarantine doc/tool re-aims + the eight
    notebooks re-executed at 1.2.243 + this round's own ledger edits
    (the QA r367 section + the FEATURES stamp + the ship pins) + the
    two in-tree v1.2.243 zip homes -- re-walked fixed-point AFTER the
    pin edits themselves (the r362 lesson): 601,767 lines at 1,873
    files (the re-walk after the CI-catch row + the fixes + the
    fixed-tree re-ship ledger trues). */
/*  r371 re-true (the v1.2.244 release ship, measurement-last): the
    walk re-measured at the close-out fixed point -- 603,723 lines
    (the +1,956 over the r367 pin 601,767: the r368..r370 QA ledger
    growth + the r371 QA section + the RELEASE_READINESS r371
    addendum + the CHANGELOG 1.2.244 entry + its build-attempt
    amendment + the re-executed notebook outputs + the carriers +
    the committed collection txt). */
/*  r373 re-true (the v1.2.245 part21 ship, measurement-last): the
    replica walk re-measured at the close-out fixed point --
    606,292 lines (the +2,569 over the r371 pin 603,723: the shared
    evaluation layer + its tests + the four part21 QA deliverables +
    the CHANGELOG 1.2.245 entry + the re-executed 1.2.245 notebook
    outputs + the FEATURES api row). */
/*  r374 re-true (the v1.2.246 DS-1 ship, measurement-last): the
    replica walk re-measured at the close-out fixed point --
    606,978 lines (the +686 over the r373 pin: the DS-1 strategy
    lane (the route extension + the client parser + the Split-stage
    selector) + the 20 new tests + the CHANGELOG 1.2.246 entry +
    the RELEASE_READINESS r373 section + the re-executed 1.2.246
    notebook outputs). */
/*  r375 re-true (the v1.2.247 ML-3 ship, measurement-last): the
    replica walk re-measured at the close-out fixed point --
    607,683 lines (the +705 over the r374 pin: the ML-3 train-lane
    split section (the TrainSplitSpec route + the carve helper +
    the runs.py builders + the client parser/types + the MLPage
    selector/readout) + the 16 new backend tests + the drift-guard
    extension + the CHANGELOG 1.2.247 entry + the roadmap's row-2
    execution log + the re-executed 1.2.247 notebook outputs). */
/*  r380 re-true (the v1.2.252 DS-3 ship, measurement-last): the
    replica walk re-measured at the close-out fixed point --
    610,717 lines (the +3,034 over the r378 pin: the r379 DS-2 lane
    + the r380 DS-3 lane (the route + the drift guards + the page +
    the client + the re-exported contract) + the re-executed
    1.2.251/1.2.252 notebook outputs). */
/*  r381 re-true (the v1.2.253 PK-1 ship, measurement-last): the
    replica walk re-measured at the close-out fixed point --
    610,958 lines (the +241 over the r380 pin: the PK-1 lane (the
    page's bridge card + the 7-vitest suite + the 10-test drift
    guard + the CHANGELOG/roadmap entries) + the re-executed
    1.2.253 notebook outputs, the notebook output churn itself
    netting the round down to +241). */
/*  r382 re-true (the v1.2.254 ML-4 ship, measurement-last): the
    census re-measured live on the FINAL fixed point (after the
    notebook battery's output churn) under the committed
    census_210/toolkit-fs countLines semantics -- 612,419 lines
    (the same reading the committed census_210 replica walk and the
    live page render take). Honest convention note: the r381 line
    pin (610,958) carried the r381 session's own line convention;
    the SAME r381 commit re-counted under the census_210 semantics
    reads 609,687 lines -- the +1,271 there is the r381
    convention's extra, never tree growth, and the r382 pin rides
    the committed semantics from here (the growth over the
    like-for-like r381 reading is +2,732: the three roadmap rows'
    lanes + the contract re-export + the re-executed notebook
    outputs + the ledger carriers). */
export const SHIPPED_SOURCE_LINES = 612419;

/*  r366-continuation re-true (measurement-last): the firing's
    close-out correction (the drift-guard figure 202 -> 130 and the
    contracts figure 28 -> 6 corrected at source in the five ledger
    carriers, both failed independent reproduction at this
    continuation) added a net +7 counted lines inside the walked
    toolkit tree (git diff: 13 insertions / 6 deletions over the
    CHANGELOG + the PROJECT_STATUS clause + the QA report + the
    RELEASE_READINESS addendum) -- 594,843 lines at 1,860 files
    (the file pin unchanged: no in-tree file added or removed).
    Verified pin = live walk = page render, re-flown over the
    production build in the same commit. */

/*  r366 re-true (measurement-last, the fixed-point discipline
    held): the walk re-measured AFTER this round's own ledger
    close-out edits -- 594,836 lines at 1,860 files / v1.2.242
    (the +4,535 lines over the r361-era pin 590,301 are the part18
    campaign's growth: the three production module families with
    their full integration path + the retirement work + the r365
    ledger set + this round's own QA/clause/addendum/bullet deltas,
    +128 of which are this round's own). Verified pin = live walk. */

/*  r322 (the 1.2.240 same-version re-ship, the §8 workflow lane):
    +3,422 the counted delta (the workflow package + the route family
    + the two test suites + ARCHITECTURE_REVIEW/PROJECT_STATUS/
    CHANGELOG doc deltas + the contract pair regrowth); the .tsx
    lanes never count by the long-standing census rule; verified
    pin = live walk = page render. */

/*  r323 (the 1.2.240 same-version re-ship, the §8 GUI leg):
    +209 the counted doc delta (the CHANGELOG entry + the
    PROJECT_STATUS addenda + the ARCHITECTURE_REVIEW next-update);
    the .tsx lanes never count by the long-standing census rule;
    verified pin = live walk = page render. */

/*  r324 (the 1.2.240 same-version re-ship, the Phase-6 remainder):
    +797 the counted delta (the four lane files + the regrown
    openapi.json/api.d.ts contract pair + the CHANGELOG entry + the
    PROJECT_STATUS addenda); the .tsx lanes never count by the
    long-standing census rule; verified pin = live walk = page
    render. */

/*  r325 (the 1.2.240 same-version re-ship, the Phase-7 release-gate
    verification round): +94 the counted doc delta (the CHANGELOG
    entry + the PROJECT_STATUS Phase-7 addenda; live-walk semantics:
    the Python replica's +98 counts the four 0-byte __init__.py
    files as 1 line each — the r309-documented JS bias, the pin
    equals the live walk); zero src/tests change this round;
    verified pin = live walk = page render. */

/*  r361 re-true (both census carriers, the fixed-point discipline):
    the live /api/toolkit/tree?fresh=1 walk at the recovered terminal
    measured 1,842 files / 589,728 lines -- the +136 files / +21,928
    lines over the r325 pin 1,706/567,800 are the r326..r360 rounds'
    accumulated tree growth (the ML workspace lanes, the part16
    diagnostics lane, the part17 RSM/G1 lanes and their QA doc set),
    never re-trued since the r325 census moment. Verified pin = live
    walk = page render three-way. */

/** r315 (the v1.2.237 ship, the DTA lane): +2 the route family
 *  (routes/dta.py + tests/api/test_dta.py), +1 the DtaLane
 *  component suite (.tsx, a tree node never line-counted), +4 the
 *  r315 evidence screenshots (2 DTA cards + the pins pair), +2 the
 *  v1.2.237 ship zip in its two in-tree homes, the .md deltas of the
 *  round's own docs (CHANGELOG + PROJECT_STATUS + FEATURES), the
 *  line deltas of the re-executed notebooks/twins and the frontend
 *  carriers; the ghost sweep (the rollback-resurrected 229/230
 *  zips across three homes + the stale .github duplicate + the
 *  dormant toolkit-path counter) trued before the walk; verified
 *  pin = live walk = page render. */

/** r313 file-count note: the terminal census 1,637 = the walk above
 *  +2 the r313 pins evidence screenshots themselves (the fixed-point
 *  discipline: the archive is rebuilt AFTER the last tree edit, so
 *  the pins pair rides the terminal build — the r312 convention). */

/** Entries in pharmaml.ml.models.MODEL_REGISTRY at the shipped
 *  revision (verified by import, 2026-11-16). */
export const SHIPPED_MODEL_REGISTRY_SIZE = 33;

/** sklearn families within the registry (the rest: 3 boosting, 3 dnn,
 *  hsvr, voting, stacking). */
export const SHIPPED_SKLEARN_FAMILIES = 24;
