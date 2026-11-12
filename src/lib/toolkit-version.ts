/** The toolkit version the frontend ships/falls back to.

 * Single source of truth for every "vX.Y.Z" the UI paints before the
 * live census (``/api/toolkit/tree``) answers — the header badge and
 * the download panel both read this, so a version bump only touches
 * this file (plus the toolkit's own carriers).
 */
export const SHIPPED_TOOLKIT_VERSION = "1.2.179";
