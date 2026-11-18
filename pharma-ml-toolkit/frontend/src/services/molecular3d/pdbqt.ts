/**
 * The part11 Phase Eighteen molecular-3D data lane: the PDBQT text
 * parser + the element vocabulary the Three.js viewer renders.
 *
 * Phase Eighteen's honesty rules ride this module:
 *
 * * "Do not fabricate molecular or structural information" — the
 *   parser reads ONLY the ATOM/HETATM records the backend's own
 *   preparation lanes (`pharmaml.docking.preparation`) registered;
 *   every atom rendered is a record that actually arrived.
 * * "Audit 3D assets and coordinate conventions before
 *   visualization" — the fixed-column coordinate slices are the
 *   PDB/PDBQT convention (x 31-38, y 39-46, z 47-54, 1-based
 *   inclusive), the AutoDock atom type is the record's LAST token,
 *   and the bond inference is a DISPLAY heuristic (covalent-radius
 *   distance cutoff) stated as such — never scientific output.
 * * "Use Three.js where it provides meaningful scientific value" —
 *   molecular structures are genuine 3D scenes (the §11.5 ligand /
 *   receptor structures and the §11.7 trajectory frames); 2D lanes
 *   stay Plotly.
 */

/** One parsed ATOM/HETATM record — only fields that actually arrived. */
export interface PdbqtAtom {
  readonly index: number;
  /** The AutoDock atom type token verbatim (e.g. `A`, `HD`, `OA`). */
  readonly atomType: string;
  /** The display element the type resolves to (`A`→`C`, `OA`→`O`…). */
  readonly element: string;
  readonly x: number;
  readonly y: number;
  readonly z: number;
  /** The record's serial number (PDB columns 7-11), when present. */
  readonly serial: number | null;
  /** The atom name (columns 13-16), when present. */
  readonly name: string | null;
  /** The residue name (columns 18-20), when present. */
  readonly residue: string | null;
}

/** A display-only bond inferred by the distance heuristic. */
export interface InferredBond {
  readonly a: number;
  readonly b: number;
}

/**
 * The AutoDock atom-type → display-element map (the `prepare` tools'
 * own vocabulary: aromatic carbons are typed `A`, polar hydrogens
 * `HD`, H-bond-acceptor oxygens/nitrogens `OA`/`NA`, sulfur
 * acceptors `SA`, … — the decorations are charge-state annotations,
 * not different elements).
 */
const TYPE_TO_ELEMENT: Readonly<Record<string, string>> = {
  C: "C", A: "C", N: "N", NA: "N", NS: "N", O: "O", OA: "O", OS: "O",
  H: "H", HD: "H", HS: "H", S: "S", SA: "S", P: "P", F: "F",
  Cl: "Cl", CL: "Cl", Br: "Br", BR: "Br", I: "I", Zn: "Zn", ZN: "Zn",
  Mg: "Mg", MG: "Mg", Ca: "Ca", CA: "Ca", Fe: "Fe", FE: "Fe",
  Mn: "Mn", MN: "Mn", Cu: "Cu", CU: "Cu", Na: "Na",
  K: "K", Li: "Li", Se: "Se", Si: "Si", B: "B",
};

/** Resolve an AutoDock atom type token to its display element. */
export function displayElementForType(atomType: string): string {
  const direct = TYPE_TO_ELEMENT[atomType];
  if (direct !== undefined) {
    return direct;
  }
  // unknown types resolve through the element-name convention: the
  // leading alphabetic run, capped at two characters, first letter
  // upper + optional second lower — anything still unmatched renders
  // honestly unknown ("?") rather than guessed
  const leading = atomType.replace(/^[^A-Za-z]+/, "").slice(0, 2);
  if (leading.length === 0) {
    return "?";
  }
  const canonical =
    leading.length === 2
      ? leading.charAt(0).toUpperCase() + leading.charAt(1).toLowerCase()
      : leading.charAt(0).toUpperCase();
  return TYPE_TO_ELEMENT[canonical] !== undefined ? canonical : "?";
}

/**
 * The Jmol/CPK display palette — the scientific convention for clear
 * molecular representation (Phase Eighteen's own requirement). These
 * are data-driven render colors, not UI chrome.
 */
export const CPK_COLORS: Readonly<Record<string, string>> = {
  H: "#f5f5f5", C: "#4f4f4f", N: "#2f5fe0", O: "#e03a3a", S: "#e6d92e",
  P: "#e08a2e", F: "#6fd94f", Cl: "#3fc95a", Br: "#a2451f", I: "#8e2f8e",
  Zn: "#7d80b0", Mg: "#8fd42e", Ca: "#a9a9a9", Fe: "#b56a3f", Mn: "#8e4ac2",
  Cu: "#b87b34", Na: "#ab5cf5", K: "#8f40d4", Li: "#cc8080", Se: "#e0a15c",
  Si: "#c8a25a", B: "#d9b0d9", "?": "#ff69b4",
};

/** The display color for an element (CPK, unknown → pink). */
export function cpkColor(element: string): string {
  return CPK_COLORS[element] ?? CPK_COLORS["?"]!;
}

/**
 * Bondi/Slater display radii in Ångström for the ball-and-stick
 * spheres (scaled down — ball-and-stick, not space-filling).
 */
const DISPLAY_RADII: Readonly<Record<string, number>> = {
  H: 0.3, C: 0.46, N: 0.45, O: 0.44, S: 0.6, P: 0.62, F: 0.42,
  Cl: 0.58, Br: 0.66, I: 0.74, Zn: 0.58, Mg: 0.56, Ca: 0.72,
  Fe: 0.56, Mn: 0.54, Cu: 0.5, Na: 0.66, K: 0.72, Li: 0.5,
  Se: 0.58, Si: 0.56, B: 0.5,
};

/** The ball radius for an element (unknown → a mid-gray 0.45). */
export function displayRadius(element: string): number {
  return DISPLAY_RADII[element] ?? 0.45;
}

/**
 * Covalent radii in Ångström for the display bond heuristic.
 */
const COVALENT_RADII: Readonly<Record<string, number>> = {
  H: 0.31, C: 0.76, N: 0.71, O: 0.66, S: 1.05, P: 1.07, F: 0.57,
  Cl: 1.02, Br: 1.2, I: 1.39, Zn: 1.22, Mg: 1.36, Ca: 1.74,
  Fe: 1.32, Mn: 1.39, Cu: 1.32, Na: 1.66, K: 2.03, Li: 1.28,
  Se: 1.2, Si: 1.11, B: 0.84,
};

/** The covalent radius for the display-bond heuristic (unknown → 0.8). */
function covalentRadius(element: string): number {
  return COVALENT_RADII[element] ?? 0.8;
}

/** The parse outcome — the parsed atoms plus the honest diagnostics. */
export interface PdbqtParseResult {
  readonly atoms: readonly PdbqtAtom[];
  /** ATOM/HETATM records seen (including any skipped as malformed). */
  readonly recordCount: number;
  /** Records skipped as malformed (never silent — the count surfaces). */
  readonly skipped: number;
  /** Non-ATOM records present (ROOT/BRANCH/TORSDOF/REMARK…). */
  readonly auxiliaryLines: number;
  /** Records beyond the parse limit, counted not parsed (honest note). */
  readonly truncated: number;
}

/** Above this atom count the O(n²) display-bond pass is skipped. */
export const BOND_INFERENCE_ATOM_LIMIT = 2000;

/**
 * Parse the ATOM/HETATM records out of PDBQT text — the fixed-column
 * PDB coordinate convention with the AutoDock trailing type token.
 * Malformed records are counted and surfaced (never a silent drop);
 * the scientific authority for the text itself stays the backend.
 */
export function parsePdbqtAtoms(
  text: string,
  label: string,
  atomLimit = 4000,
): PdbqtParseResult {
  const atoms: PdbqtAtom[] = [];
  let recordCount = 0;
  let skipped = 0;
  let auxiliaryLines = 0;
  let truncated = 0;
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trimEnd();
    if (line.startsWith("ATOM  ") || line.startsWith("HETATM")) {
      recordCount += 1;
      if (atoms.length >= atomLimit) {
        truncated += 1;
        continue;
      }
      const atom = parseAtomRecord(line);
      if (atom === null) {
        skipped += 1;
        continue;
      }
      atoms.push({ ...atom, index: atoms.length });
    } else if (line.trim().length > 0) {
      auxiliaryLines += 1;
    }
  }
  if (atoms.length === 0) {
    throw new Error(
      `${label}: no ATOM/HETATM records parsed — the 3D view renders only structures that actually arrived`,
    );
  }
  return { atoms, recordCount, skipped, auxiliaryLines, truncated };
}

/** Parse one fixed-column ATOM/HETATM record (null → malformed). */
function parseAtomRecord(line: string): PdbqtAtom | null {
  const record = line.slice(0, 6).trim();
  if (record !== "ATOM" && record !== "HETATM") {
    return null;
  }
  const tokens = line.trim().split(/\s+/);
  if (tokens.length < 8) {
    return null;
  }
  const serialText = (line.slice(6, 11).trim() || tokens[1] || "").trim();
  const serial = /^\d+$/.test(serialText) ? Number.parseInt(serialText, 10) : null;
  const name = line.slice(12, 16).trim() || null;
  const residue = line.slice(17, 20).trim() || null;
  const xText = line.slice(30, 38).trim();
  const yText = line.slice(38, 46).trim();
  const zText = line.slice(46, 54).trim();
  let x: number;
  let y: number;
  let z: number;
  if (xText !== "" && yText !== "" && zText !== "") {
    x = Number.parseFloat(xText);
    y = Number.parseFloat(yText);
    z = Number.parseFloat(zText);
  } else {
    // the PDB token order fallback: ATOM serial name resName chain
    // resSeq x y z … — tokens[5..7] are the coordinates
    x = Number.parseFloat(tokens[5] ?? "NaN");
    y = Number.parseFloat(tokens[6] ?? "NaN");
    z = Number.parseFloat(tokens[7] ?? "NaN");
  }
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
    return null;
  }
  const atomType = tokens[tokens.length - 1] ?? "?";
  return {
    index: 0,
    atomType,
    element: displayElementForType(atomType),
    x,
    y,
    z,
    serial,
    name,
    residue,
  };
}

/**
 * The display-bond heuristic: a bond is DRAWN (display only, never
 * scientific output) when two atoms sit within
 * `(r_cov_a + r_cov_b) × tolerance` Ångström of each other. Skipped
 * entirely above {@link BOND_INFERENCE_ATOM_LIMIT} atoms (the O(n²)
 * pass would stall the UI; the honest no-bonds note surfaces then).
 */
export function inferDisplayBonds(
  atoms: readonly PdbqtAtom[],
  tolerance = 1.25,
): readonly InferredBond[] {
  if (atoms.length > BOND_INFERENCE_ATOM_LIMIT) {
    return [];
  }
  const bonds: InferredBond[] = [];
  const radii = atoms.map((a) => covalentRadius(a.element));
  for (let i = 0; i < atoms.length; i += 1) {
    const a = atoms[i]!;
    for (let j = i + 1; j < atoms.length; j += 1) {
      const b = atoms[j]!;
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dz = a.z - b.z;
      const dist2 = dx * dx + dy * dy + dz * dz;
      const limit = (radii[i]! + radii[j]!) * tolerance;
      if (dist2 <= limit * limit) {
        bonds.push({ a: i, b: j });
      }
    }
  }
  return bonds;
}

/** The element histogram for the structure metadata strip. */
export function elementHistogram(
  atoms: readonly PdbqtAtom[],
): readonly { element: string; count: number }[] {
  const counts = new Map<string, number>();
  for (const atom of atoms) {
    counts.set(atom.element, (counts.get(atom.element) ?? 0) + 1);
  }
  return [...counts.entries()]
    .map(([element, count]) => ({ element, count }))
    .sort((p, q) => q.count - p.count || p.element.localeCompare(q.element));
}

/**
 * The uniform element for coordinate-only sources (the §11.7 inline
 * stack carries positions, never element identities — the viewer
 * renders them honestly uniform instead of guessing).
 */
export const UNIFORM_COORDINATE_ELEMENT = "C";
