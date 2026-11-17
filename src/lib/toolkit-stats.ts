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
 *  ``pytest --collect-only -q`` sum, 2026-11-17 (r276: the part10
 *  dl/semi-supervised/generative row adds tests/api/test_dl_semi_generative.py
 *  — 30 tests over the torch-training-telemetry/semi-supervised/generative
 *  contracts, moving the api suite to 119; the count moves to 8,246 under
 *  the FULL pinned stack — every optional dep installed, so the
 *  conditionally-defined suites collect. The jump from 7,463 is larger
 *  than the round's additions because this close's census ran over the
 *  FULLY RESTORED tree: the session-opening blob-level reconciliation
 *  restored 42 tracked files the sandbox recycle had wiped (among them
 *  test files whose suites had silently left the collection at the
 *  r275-cont close), and the collection now honestly includes them —
 *  the same restore also fed this round's eleven-block gate, so the
 *  number and the green evidence cover the same surface). */
export const SHIPPED_TEST_COUNT = 8246;

/** File count of the shipped toolkit tree (src, tests, examples, ui,
 *  docs, tools — the same census /api/toolkit/tree walks). */
export const SHIPPED_FILE_COUNT = 1484;

/** Source lines of the shipped tree, formatted with thousands
 *  separators at render time (raw number kept for math). */
export const SHIPPED_SOURCE_LINES = 530272;

/** Entries in pharmaml.ml.models.MODEL_REGISTRY at the shipped
 *  revision (verified by import, 2026-11-16). */
export const SHIPPED_MODEL_REGISTRY_SIZE = 33;

/** sklearn families within the registry (the rest: 3 boosting, 3 dnn,
 *  hsvr, voting, stacking). */
export const SHIPPED_SKLEARN_FAMILIES = 24;
