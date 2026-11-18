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
 *  ``pytest --collect-only -q`` sum, 2026-11-19 the r289 firing walk:
 *  the default collection (testpaths tests+src, doctest-modules on)
 *  collects 8,576 = tests/ 7,661 (of which tests/ui 971, tests/api 261)
 *  + src doctests 915. The r288 molecular3d family (27 guards) and the
 *  r289 §14.1 startup-gate family (8 guards) ride the count. The
 *  previous 8,541 pin was the r287 close-out walk. */
export const SHIPPED_TEST_COUNT = 8576;

/** File count of the shipped toolkit tree (src, tests, examples, ui,
 *  docs, tools — the same census /api/toolkit/tree walks). The r289
 *  firing re-trued walk: 1,569 (the §14.1 startup trio + its drift
 *  guard + the browser-verify harness ride the live walk; the same
 *  convention the r287/288 rounds used for their in-flight rounds). */
export const SHIPPED_FILE_COUNT = 1569;

/** Source lines of the shipped tree, formatted with thousands
 *  separators at render time (raw number kept for math). The r289
 *  firing live census: 588,975. */
export const SHIPPED_SOURCE_LINES = 588975;

/** Entries in pharmaml.ml.models.MODEL_REGISTRY at the shipped
 *  revision (verified by import, 2026-11-16). */
export const SHIPPED_MODEL_REGISTRY_SIZE = 33;

/** sklearn families within the registry (the rest: 3 boosting, 3 dnn,
 *  hsvr, voting, stacking). */
export const SHIPPED_SKLEARN_FAMILIES = 24;
