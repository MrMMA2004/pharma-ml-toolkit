/**
 * part16 §12 (Phase 9) — the manual grid-box visualization's pure data
 * lane: the extents/volume math, the in-box atom census, the honest
 * validation warnings, and the two pinned honesty notes the viewer
 * renders verbatim.
 *
 * Phase 9's rules ride this module:
 *
 * * "The 3D view and numerical configuration must remain synchronized"
 *   — the viewer consumes the SAME six numbers the stage-6 dock
 *   request serializes (`center: [centerX, centerY, centerZ]`,
 *   `size: [sizeX, sizeY, sizeZ]`); the numeric inputs stay the ONE
 *   mutation surface, so the view can never drift from the request.
 * * "Ensure the grid box is aligned with the actual protein coordinate
 *   system used by the docking backend" + "Never silently shift,
 *   rescale, or rotate the protein independently of the docking
 *   coordinates" — the box is drawn in the receptor's own coordinate
 *   frame, untransformed; the frame note is pinned below and the
 *   drift guard pins it verbatim.
 * * "Grid spacing, where supported by the backend docking engine" —
 *   the backend `DockRequest` (routes/docking.py) carries center +
 *   size only, NO spacing field; the viewer states that honestly
 *   rather than rendering an imagined spacing control.
 * * "Show coordinate values, box dimensions, units" — Å everywhere,
 *   the unit note pinned below.
 * * "Validation warnings: out-of-range or invalid configurations" —
 *   the display-layer sanity checks below (they describe, they never
 *   gate the request — the backend route's own validation stays the
 *   authority).
 */

/** The six DockRequest-aligned numbers, in Å (the manual form's state). */
export interface GridBoxConfig {
  readonly centerX: number;
  readonly centerY: number;
  readonly centerZ: number;
  readonly sizeX: number;
  readonly sizeY: number;
  readonly sizeZ: number;
}

/** A display-layer validation finding (never a request gate). */
export interface GridBoxWarning {
  readonly code:
    | "non-finite"
    | "non-positive-size"
    | "extreme-size"
    | "far-origin"
    | "empty-box"
    | "ligand-outside";
  readonly severity: "error" | "warning";
  readonly message: string;
}

/** The axis-aligned extents of the box, or null when malformed. */
export interface GridBoxExtents {
  readonly minX: number;
  readonly maxX: number;
  readonly minY: number;
  readonly maxY: number;
  readonly minZ: number;
  readonly maxZ: number;
}

/**
 * The pinned reference-frame honesty note (Phase 9's alignment rule).
 * The drift guard pins this string verbatim in the viewer.
 */
export const REFERENCE_FRAME_NOTE =
  "The grid box is drawn in the receptor's own coordinate frame — the viewer never shifts, rescales, or rotates the protein.";

/**
 * The pinned spacing honesty note (Phase 9's "where supported" rule):
 * the backend DockRequest carries center + size only.
 */
export const SPACING_NOTE =
  "Grid spacing is not a DockRequest field — the docking engine's own grid spacing default applies, so no spacing control is rendered.";

/** The pinned unit note (Phase 9's "Show … units" rule). */
export const UNIT_NOTE = "All coordinates and dimensions are Ångström (Å).";

/** Beyond this dimension (Å) the box is flagged extreme for docking. */
export const EXTREME_SIZE_ANGSTROM = 150;

/** Beyond this |center| (Å) the box is flagged far from the origin. */
export const FAR_ORIGIN_ANGSTROM = 500;

/** Fixed two-decimal coordinate formatting for the readouts. */
export function formatCoord(value: number): string {
  return Number.isFinite(value) ? value.toFixed(2) : "—";
}

/**
 * The box extents, or null when any dimension is non-finite or
 * non-positive (the malformed box draws nothing and the validation
 * reports why).
 */
export function boxExtents(config: GridBoxConfig): GridBoxExtents | null {
  const { centerX, centerY, centerZ, sizeX, sizeY, sizeZ } = config;
  const sizes = [sizeX, sizeY, sizeZ];
  const centers = [centerX, centerY, centerZ];
  if (![...sizes, ...centers].every((v) => Number.isFinite(v))) {
    return null;
  }
  if (sizes.some((s) => s <= 0)) {
    return null;
  }
  return {
    minX: centerX - sizeX / 2,
    maxX: centerX + sizeX / 2,
    minY: centerY - sizeY / 2,
    maxY: centerY + sizeY / 2,
    minZ: centerZ - sizeZ / 2,
    maxZ: centerZ + sizeZ / 2,
  };
}

/** The box volume in Å³ (0 for a malformed box). */
export function boxVolume(config: GridBoxConfig): number {
  const { sizeX, sizeY, sizeZ } = config;
  if (![sizeX, sizeY, sizeZ].every((v) => Number.isFinite(v) && v > 0)) {
    return 0;
  }
  return sizeX * sizeY * sizeZ;
}

/**
 * The count of atoms inside the box (inclusive bounds per axis) —
 * the binding-site region census. Only atoms that actually arrived
 * are counted (the never-fabricate rule); an empty atom list counts 0.
 */
export function countAtomsInBox(
  atoms: readonly { readonly x: number; readonly y: number; readonly z: number }[],
  config: GridBoxConfig,
): number {
  const extents = boxExtents(config);
  if (extents === null) {
    return 0;
  }
  let count = 0;
  for (const atom of atoms) {
    if (
      atom.x >= extents.minX &&
      atom.x <= extents.maxX &&
      atom.y >= extents.minY &&
      atom.y <= extents.maxY &&
      atom.z >= extents.minZ &&
      atom.z <= extents.maxZ
    ) {
      count += 1;
    }
  }
  return count;
}

/**
 * The display-layer validation pass (Phase 9's "Validation warnings"
 * rule). Every check describes; none gates the dock request (the
 * backend route's own validation stays the authority). The in-box
 * censuses arrive precomputed so the warnings and the readouts share
 * one source of truth.
 */
export function validateGridBox(
  config: GridBoxConfig,
  options: {
    readonly receptorAtomCount: number;
    readonly receptorInBox: number;
    readonly ligandAtomCount: number | null;
    readonly ligandInBox: number | null;
  },
): readonly GridBoxWarning[] {
  const warnings: GridBoxWarning[] = [];
  const { centerX, centerY, centerZ, sizeX, sizeY, sizeZ } = config;
  const values = [centerX, centerY, centerZ, sizeX, sizeY, sizeZ];
  if (!values.every((v) => Number.isFinite(v))) {
    warnings.push({
      code: "non-finite",
      severity: "error",
      message:
        "A center or size value is not a finite number — the box cannot be drawn or submitted meaningfully.",
    });
    return warnings;
  }
  if ([sizeX, sizeY, sizeZ].some((s) => s <= 0)) {
    warnings.push({
      code: "non-positive-size",
      severity: "error",
      message:
        "Every box dimension must be positive (Å) — a zero or negative size draws no box.",
    });
  }
  if ([sizeX, sizeY, sizeZ].some((s) => s > EXTREME_SIZE_ANGSTROM)) {
    warnings.push({
      code: "extreme-size",
      severity: "warning",
      message: `A dimension exceeds ${EXTREME_SIZE_ANGSTROM} Å — docking boxes that large are usually a mistake.`,
    });
  }
  if (
    [centerX, centerY, centerZ].some(
      (c) => Math.abs(c) > FAR_ORIGIN_ANGSTROM,
    )
  ) {
    warnings.push({
      code: "far-origin",
      severity: "warning",
      message: `A center coordinate sits beyond ±${FAR_ORIGIN_ANGSTROM} Å from the origin — check the box against the protein's coordinate frame.`,
    });
  }
  if (options.receptorAtomCount > 0 && options.receptorInBox === 0) {
    warnings.push({
      code: "empty-box",
      severity: "warning",
      message:
        "No receptor atoms fall inside the box — the box may not overlap the binding site in the protein's coordinate frame.",
    });
  }
  if (
    options.ligandAtomCount !== null &&
    options.ligandAtomCount > 0 &&
    options.ligandInBox !== null &&
    options.ligandInBox === 0
  ) {
    warnings.push({
      code: "ligand-outside",
      severity: "warning",
      message:
        "No ligand atoms fall inside the box — the reference ligand sits outside the docking region.",
    });
  }
  return warnings;
}
