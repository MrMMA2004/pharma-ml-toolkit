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
 *  ``pytest --collect-only -q`` sum, 2026-11-18 post-release walk (r278: part11 Phase
 *  Eleven -- the SQLite persistence backend + the /api/v1/database
 *  surface -- adds tests/api/test_persistence.py,
 *  tests/api/test_persistence_database.py and the
 *  tests/api/test_contracts.py OpenAPI drift gate; the api suite
 *  moves to 251; then the arena/cfdb76b9 absorption (GA checkpoint
 *  identity, file-split audit, held-out partition + their UI
 *  dialogs) moves the count to 8,410 under the FULL pinned
 *  stack -- every optional dep installed, so the
 *  conditionally-defined suites collect). */
export const SHIPPED_TEST_COUNT = 8461;

/** File count of the shipped toolkit tree (src, tests, examples, ui,
 *  docs, tools — the same census /api/toolkit/tree walks). */
export const SHIPPED_FILE_COUNT = 1510;

/** Source lines of the shipped tree, formatted with thousands
 *  separators at render time (raw number kept for math). */
export const SHIPPED_SOURCE_LINES = 559611;

/** Entries in pharmaml.ml.models.MODEL_REGISTRY at the shipped
 *  revision (verified by import, 2026-11-16). */
export const SHIPPED_MODEL_REGISTRY_SIZE = 33;

/** sklearn families within the registry (the rest: 3 boosting, 3 dnn,
 *  hsvr, voting, stacking). */
export const SHIPPED_SKLEARN_FAMILIES = 24;
