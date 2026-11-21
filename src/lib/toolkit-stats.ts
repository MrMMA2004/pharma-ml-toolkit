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
export const SHIPPED_TEST_COUNT = 8484;

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
export const SHIPPED_COMPONENT_TEST_COUNT = 102;

/*  r321 (the 1.2.240 same-version re-ship, the multiobjective GUI
 *  leg): +8 the MultiobjectiveLane suite (14 files / 102 tests — the
 *  registry line, the full round trip, the 3-objective null
 *  honesty, the 422 verbatim, the three client-side gates, the
 *  provenance carriage); verified pin = live run. */

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
export const SHIPPED_FILE_COUNT = 1680;

/*  r321 (the 1.2.240 same-version re-ship, the multiobjective GUI
 *  leg): +1 the MultiobjectiveLane component suite (.tsx, a tree
 *  node never line-counted), +2 the r321 E2E evidence screenshots
 *  (the card + the gate pair), +2 the r321 pins pair riding the
 *  terminal build; the backend-daemon theme-shots swept per the
 *  r319/r320 precedent; verified pin = live walk = page render. */

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
export const SHIPPED_SOURCE_LINES = 563278;

/*  r321 (the 1.2.240 same-version re-ship, the multiobjective GUI
 *  leg): +104 the round's counted doc delta (the CHANGELOG r321
 *  entry + the PROJECT_STATUS Phase-5 close addenda + the FEATURES
 *  r321 note + the stage-chain guard row) — the .tsx lanes never
 *  count by the long-standing census rule; verified pin = live
 *  walk = page render. */

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
