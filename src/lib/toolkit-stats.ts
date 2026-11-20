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
export const SHIPPED_TEST_COUNT = 8362;

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
 *  architecture-hygiene guard (+1) and the SBOM artifact (+1). */
export const SHIPPED_FILE_COUNT = 1591;

/** Source lines of the shipped tree, formatted with thousands
 *  separators at render time (raw number kept for math). The round-304
 *  re-true: 552,780 — the part12 Phase-3 tree (the r303 pin's 597,658
 *  minus the 46,417 retired lines — the Tk playground's 30,047 plus its
 *  753-test suite's 16,370 — plus the round's +1,539: the
 *  architecture-hygiene guard, the SBOM artifact, the ten
 *  docstring-map sites, the windows-release step 22b, the
 *  README/COVERAGE/FEATURES/RELEASE_READINESS/CHANGELOG addenda; the
 *  file count stays 1,591 — the two retirements offset by the hygiene
 *  test and the SBOM). The same-version re-ship carries exactly this
 *  state. (Honest note: a concurrent 2h-sync firing measured 552,625
 *  mid-round and wrote a provisional pin at 16:39; this is the final
 *  fresh `?fresh=1` walk after the round's own CHANGELOG entry
 *  landed.) */
export const SHIPPED_SOURCE_LINES = 552780;

/** Entries in pharmaml.ml.models.MODEL_REGISTRY at the shipped
 *  revision (verified by import, 2026-11-16). */
export const SHIPPED_MODEL_REGISTRY_SIZE = 33;

/** sklearn families within the registry (the rest: 3 boosting, 3 dnn,
 *  hsvr, voting, stacking). */
export const SHIPPED_SKLEARN_FAMILIES = 24;
