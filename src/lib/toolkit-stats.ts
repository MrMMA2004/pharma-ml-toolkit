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
 *  ``pytest --collect-only -q`` sum, 2026-11-18 the r287 close-out walk:
 *  the default collection (testpaths tests+src, doctest-modules on)
 *  collects 8,541 = tests/ 7,626 (of which tests/ui 971, tests/api 261)
 *  + src doctests 915. The r285/286/287 workspace guard families
 *  (docking 16, pkpd 14, md+visualization 21) and the r287
 *  exclusion-drift guard ride the count. The previous 8,545 pin
 *  double-counted tests/ui ("7,574 incl. ui" + 971) — the honest
 *  default-collection total at r284 was 8,489; 8,489 + 52 = 8,541. */
export const SHIPPED_TEST_COUNT = 8541;

/** File count of the shipped toolkit tree (src, tests, examples, ui,
 *  docs, tools — the same census /api/toolkit/tree walks). The r287
 *  close-out re-trued walk: 1,546 (the MD + Visualization workspace
 *  pages, clients and guard suites ride the tree). */
export const SHIPPED_FILE_COUNT = 1546;

/** Source lines of the shipped tree, formatted with thousands
 *  separators at render time (raw number kept for math). The r287
 *  close-out live census: 582,735. */
export const SHIPPED_SOURCE_LINES = 582735;

/** Entries in pharmaml.ml.models.MODEL_REGISTRY at the shipped
 *  revision (verified by import, 2026-11-16). */
export const SHIPPED_MODEL_REGISTRY_SIZE = 33;

/** sklearn families within the registry (the rest: 3 boosting, 3 dnn,
 *  hsvr, voting, stacking). */
export const SHIPPED_SKLEARN_FAMILIES = 24;
