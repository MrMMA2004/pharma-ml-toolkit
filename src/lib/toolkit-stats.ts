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
 *  ``pytest --collect-only -q`` sum, 2026-11-20 the r300 firing walk:
 *  the default collection (testpaths tests+src, doctest-modules on)
 *  collects 8,590. The tree re-trued after the ff-only re-sync absorbed
 *  rounds 290-299 (the Windows-release arc) plus the arena/f5c3dd10
 *  merge (the landing-page redesign + Studio CI guards, whose
 *  test_ml_workspace / test_datasets_ml_artifacts deltas ride the
 *  count). The previous 8,576 pin was the r289 firing walk. */
export const SHIPPED_TEST_COUNT = 8590;

/** File count of the shipped toolkit tree (src, tests, examples, ui,
 *  docs, tools — the same census /api/toolkit/tree walks). The r300
 *  firing re-trued walk: 1,584 (post-re-sync + arena merge). The
 *  previous 1,569 pin was the r289 firing walk. */
export const SHIPPED_FILE_COUNT = 1584;

/** Source lines of the shipped tree, formatted with thousands
 *  separators at render time (raw number kept for math). The r300
 *  firing live census: 595,238. */
export const SHIPPED_SOURCE_LINES = 595238;

/** Entries in pharmaml.ml.models.MODEL_REGISTRY at the shipped
 *  revision (verified by import, 2026-11-16). */
export const SHIPPED_MODEL_REGISTRY_SIZE = 33;

/** sklearn families within the registry (the rest: 3 boosting, 3 dnn,
 *  hsvr, voting, stacking). */
export const SHIPPED_SKLEARN_FAMILIES = 24;
