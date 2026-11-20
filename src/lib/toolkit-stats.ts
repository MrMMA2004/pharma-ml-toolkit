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
 *  ``pytest --collect-only -q`` sum, re-verified on the part12 audit
 *  session's repaired venv, 2026-11-20: 8,590 collected / 0 errors.
 *  The tree is engine-identical to the r301 close-out (the part12
 *  round changed docs/frontend/web only), so the count stands. */
export const SHIPPED_TEST_COUNT = 8590;

/** File count of the shipped toolkit tree (src, tests, examples, ui,
 *  docs, tools — the same census /api/toolkit/tree walks). The round-302
 *  re-true: 1,591 — the part12 audit-round tree: the r301 pin (1,583)
 *  counted two untracked build/theme-shots PNGs that never committed
 *  (the rollback dropped them); this round the walk finally sees the
 *  true tracked tree — +1 the artifacts client the any-depth `artifacts`
 *  exclusion used to hide (B-1), +1 the new database client (B-2), +9
 *  the part12_audit deliverables, -1 the removed malformed nested
 *  ci.yml (B-3). */
export const SHIPPED_FILE_COUNT = 1591;

/** Source lines of the shipped tree, formatted with thousands
 *  separators at render time (raw number kept for math). The round-302
 *  re-true: 596,008 — the part12 audit-round close-out tree (the r301
 *  pin's 595,039 minus the removed nested ci.yml's 52 lines, plus the
 *  round-302 CHANGELOG entry, the nine part12_audit deliverables and
 *  the MIGRATION/READINESS addenda). The final re-ship carries exactly
 *  this state. */
export const SHIPPED_SOURCE_LINES = 596008;

/** Entries in pharmaml.ml.models.MODEL_REGISTRY at the shipped
 *  revision (verified by import, 2026-11-16). */
export const SHIPPED_MODEL_REGISTRY_SIZE = 33;

/** sklearn families within the registry (the rest: 3 boosting, 3 dnn,
 *  hsvr, voting, stacking). */
export const SHIPPED_SKLEARN_FAMILIES = 24;
