/**
 * part16 §12 (Phase 9) — the grid-box pure data lane's contract.
 *
 * The extents/volume/census math and the validation vocabulary the
 * viewer renders, exercised at the unit boundary: the malformed-box
 * nulls, the inclusive-bound census, and every warning code the
 * display layer can raise (each check describes; none gates).
 */
import { describe, expect, it } from "vitest";
import {
  boxExtents,
  boxVolume,
  countAtomsInBox,
  EXTREME_SIZE_ANGSTROM,
  FAR_ORIGIN_ANGSTROM,
  formatCoord,
  REFERENCE_FRAME_NOTE,
  SPACING_NOTE,
  UNIT_NOTE,
  validateGridBox,
  type GridBoxConfig,
} from "@/services/docking/gridBox";

const BASE: GridBoxConfig = {
  centerX: 10,
  centerY: 20,
  centerZ: 30,
  sizeX: 22.5,
  sizeY: 22.5,
  sizeZ: 22.5,
};

describe("boxExtents", () => {
  it("computes the min/max corners from center ± size/2", () => {
    const e = boxExtents(BASE);
    expect(e).not.toBeNull();
    expect(e!.minX).toBeCloseTo(-1.25);
    expect(e!.maxX).toBeCloseTo(21.25);
    expect(e!.minY).toBeCloseTo(8.75);
    expect(e!.maxY).toBeCloseTo(31.25);
    expect(e!.minZ).toBeCloseTo(18.75);
    expect(e!.maxZ).toBeCloseTo(41.25);
  });

  it("returns null for a zero size", () => {
    expect(boxExtents({ ...BASE, sizeZ: 0 })).toBeNull();
  });

  it("returns null for a negative size", () => {
    expect(boxExtents({ ...BASE, sizeX: -5 })).toBeNull();
  });

  it("returns null for a non-finite coordinate", () => {
    expect(boxExtents({ ...BASE, centerY: Number.NaN })).toBeNull();
    expect(boxExtents({ ...BASE, sizeY: Number.POSITIVE_INFINITY })).toBeNull();
  });
});

describe("boxVolume", () => {
  it("multiplies the three dimensions", () => {
    expect(boxVolume({ ...BASE, sizeX: 2, sizeY: 3, sizeZ: 4 })).toBe(24);
  });

  it("is 0 for a malformed box", () => {
    expect(boxVolume({ ...BASE, sizeX: 0 })).toBe(0);
    expect(boxVolume({ ...BASE, sizeZ: Number.NaN })).toBe(0);
  });
});

describe("countAtomsInBox", () => {
  it("counts atoms inside with inclusive bounds", () => {
    // the box spans x [-1.25, 21.25]: an atom exactly on a wall counts
    const atoms = [
      { x: 10, y: 20, z: 30 }, // the center
      { x: -1.25, y: 20, z: 30 }, // exactly on the min-x wall
      { x: 21.25, y: 20, z: 30 }, // exactly on the max-x wall
      { x: 100, y: 20, z: 30 }, // far outside
    ];
    expect(countAtomsInBox(atoms, BASE)).toBe(3);
  });

  it("counts 0 for an empty atom list (nothing arrived, nothing counted)", () => {
    expect(countAtomsInBox([], BASE)).toBe(0);
  });

  it("counts 0 for a malformed box rather than throwing", () => {
    expect(countAtomsInBox([{ x: 1, y: 2, z: 3 }], { ...BASE, sizeY: -1 })).toBe(0);
  });
});

describe("validateGridBox", () => {
  it("is silent for a sane box over a receptor", () => {
    const warnings = validateGridBox(BASE, {
      receptorAtomCount: 10,
      receptorInBox: 4,
      ligandAtomCount: null,
      ligandInBox: null,
    });
    expect(warnings).toEqual([]);
  });

  it("raises the non-finite error for NaN inputs", () => {
    const warnings = validateGridBox({ ...BASE, centerX: Number.NaN }, {
      receptorAtomCount: 0,
      receptorInBox: 0,
      ligandAtomCount: null,
      ligandInBox: null,
    });
    expect(warnings).toHaveLength(1);
    expect(warnings[0]!.code).toBe("non-finite");
    expect(warnings[0]!.severity).toBe("error");
  });

  it("raises the non-positive-size error", () => {
    const warnings = validateGridBox({ ...BASE, sizeY: 0 }, {
      receptorAtomCount: 0,
      receptorInBox: 0,
      ligandAtomCount: null,
      ligandInBox: null,
    });
    expect(warnings.some((w) => w.code === "non-positive-size" && w.severity === "error")).toBe(true);
  });

  it("raises the extreme-size warning above the pinned threshold", () => {
    const warnings = validateGridBox({ ...BASE, sizeX: EXTREME_SIZE_ANGSTROM + 0.5 }, {
      receptorAtomCount: 0,
      receptorInBox: 0,
      ligandAtomCount: null,
      ligandInBox: null,
    });
    expect(warnings.some((w) => w.code === "extreme-size" && w.severity === "warning")).toBe(true);
  });

  it("raises the far-origin warning beyond the pinned radius", () => {
    const warnings = validateGridBox({ ...BASE, centerZ: FAR_ORIGIN_ANGSTROM + 1 }, {
      receptorAtomCount: 0,
      receptorInBox: 0,
      ligandAtomCount: null,
      ligandInBox: null,
    });
    expect(warnings.some((w) => w.code === "far-origin")).toBe(true);
  });

  it("raises the empty-box warning when the box misses the protein", () => {
    const warnings = validateGridBox(BASE, {
      receptorAtomCount: 1200,
      receptorInBox: 0,
      ligandAtomCount: null,
      ligandInBox: null,
    });
    expect(warnings.some((w) => w.code === "empty-box")).toBe(true);
  });

  it("raises the ligand-outside warning when the reference ligand misses", () => {
    const warnings = validateGridBox(BASE, {
      receptorAtomCount: 10,
      receptorInBox: 4,
      ligandAtomCount: 25,
      ligandInBox: 0,
    });
    expect(warnings.some((w) => w.code === "ligand-outside")).toBe(true);
  });

  it("does not raise empty-box when no receptor arrived", () => {
    const warnings = validateGridBox(BASE, {
      receptorAtomCount: 0,
      receptorInBox: 0,
      ligandAtomCount: null,
      ligandInBox: null,
    });
    expect(warnings).toEqual([]);
  });
});

describe("formatting + pinned notes", () => {
  it("formats coordinates with two decimals and an honest dash", () => {
    expect(formatCoord(1.234)).toBe("1.23");
    expect(formatCoord(Number.NaN)).toBe("—");
  });

  it("pins the reference-frame note (never shift/rescale/rotate)", () => {
    expect(REFERENCE_FRAME_NOTE).toContain("never shifts, rescales, or rotates");
    expect(REFERENCE_FRAME_NOTE).toContain("receptor's own coordinate frame");
  });

  it("pins the spacing honesty (DockRequest carries no spacing field)", () => {
    expect(SPACING_NOTE).toContain("not a DockRequest field");
  });

  it("pins the unit note (Å)", () => {
    expect(UNIT_NOTE).toContain("Å");
  });
});
