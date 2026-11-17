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
 *  ``pytest --collect-only -q`` sum, 2026-11-17 (r275-cont: the part10
 *  datasets/ml/artifacts row adds tests/api/test_datasets_ml_artifacts.py
 *  — 47 tests over the upload/train/predict/artifact contracts; the count
 *  moves 7,416 -> 7,463), under the FULL pinned
 *  stack — every optional dep installed, so the conditionally-defined
 *  suites collect; the r269 6,934 pin had been measured under a
 *  degraded venv and undercounted by 386). */
export const SHIPPED_TEST_COUNT = 7463;

/** File count of the shipped toolkit tree (src, tests, examples, ui,
 *  docs, tools — the same census /api/toolkit/tree walks). */
export const SHIPPED_FILE_COUNT = 1473;

/** Source lines of the shipped tree, formatted with thousands
 *  separators at render time (raw number kept for math). */
export const SHIPPED_SOURCE_LINES = 481574;

/** Entries in pharmaml.ml.models.MODEL_REGISTRY at the shipped
 *  revision (verified by import, 2026-11-16). */
export const SHIPPED_MODEL_REGISTRY_SIZE = 33;

/** sklearn families within the registry (the rest: 3 boosting, 3 dnn,
 *  hsvr, voting, stacking). */
export const SHIPPED_SKLEARN_FAMILIES = 24;
