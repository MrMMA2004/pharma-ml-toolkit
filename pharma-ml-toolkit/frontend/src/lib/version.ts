/**
 * The application version carrier — part16 §10's diagnostics block
 * reads this single constant. It stays in lockstep with
 * `frontend/package.json` and the toolkit's `VERSION` file (the
 * same carrier discipline the version-consistency guard enforces
 * tree-wide; the r346 drift guard pins this file to both).
 */

export const APP_VERSION = "1.2.245";
