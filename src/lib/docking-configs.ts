/**
 * Browser-side docking helpers used by the separate Next.js web console.
 * Their file formats and algorithms correspond to functions in
 * `pharmaml.docking`:
 *
 * - `scan_vina_directory`  (parsing.py)  → {@link scanVinaLogs}
 * - `compute_grid_box`     (preparation.py) → {@link computeGridBox}
 * - `extract_pdbqt_atom_types` (preparation.py) → {@link extractPdbqtAtomTypes}
 * - `write_gpf` / `write_dpf` (preparation.py) → {@link buildGpfText} / {@link buildDpfText}
 * - `write_config_batch`   (campaign.py) → {@link writeConfigBatch}
 *
 * These helpers run in the browser on pasted or locally picked files,
 * without a server round-trip. Their output formatting follows the
 * Python writers (including Python-style float strings) for compatible
 * configuration text. This module does not define a desktop UI surface.
 */

import { parseVinaOutput } from "@/lib/docking-parse";

/* ------------------------------------------------------------------ */
/* Vina results-folder scan — mirror of scan_vina_directory            */
/* ------------------------------------------------------------------ */

/** One log file collected from a folder / multi-file pick. */
export interface VinaScanFile {
  /** Path relative to the scanned root ("sub/lig01.log") or bare name. */
  path: string;
  content: string;
}

/** One ranked row of the scan table (DataFrame columns ligand/log/
 * best_affinity/n_poses). */
export interface VinaScanRow {
  ligand: string;
  log: string;
  /** Minimum affinity in kcal/mol, or null when no pose table parsed. */
  bestAffinity: number | null;
  nPoses: number;
}

/**
 * Summarise a tree of Vina logs: every file is parsed with the same
 * pose-table parser the main card uses, the best (minimum) affinity and
 * pose count are reported per ligand, and rows sort best-binder-first
 * with crashed (unparseable) logs kept last — exactly the
 * `scan_vina_directory` contract, NaN rows included instead of dropped.
 */
export function scanVinaLogs(files: VinaScanFile[]): VinaScanRow[] {
  const rows: VinaScanRow[] = files.map((file) => {
    const name = file.path.split("/").pop() ?? file.path;
    const ligand = name.replace(/\.[^.]+$/, "");
    const poses = parseVinaOutput(file.content);
    const affinities = poses.map((p) => p.affinity).filter(Number.isFinite);
    return {
      ligand,
      log: file.path,
      bestAffinity: affinities.length > 0 ? Math.min(...affinities) : null,
      nPoses: poses.length,
    };
  });
  // ascending best affinity, null (crashed) rows last — pandas
  // sort_values(ascending=True, na_position="last")
  return rows.sort((a, b) => {
    if (a.bestAffinity === null && b.bestAffinity === null) return 0;
    if (a.bestAffinity === null) return 1;
    if (b.bestAffinity === null) return -1;
    return (a.bestAffinity as number) - (b.bestAffinity as number);
  });
}

/* ------------------------------------------------------------------ */
/* PDBQT readers — mirrors of the preparation.py helpers               */
/* ------------------------------------------------------------------ */

/**
 * Sorted unique AutoDock atom types of a ligand PDBQT: the last
 * whitespace token of every ATOM/HETATM record, filtered to purely
 * alphabetic tokens so partial charges never leak into the type set.
 */
export function extractPdbqtAtomTypes(content: string): string[] {
  const types = new Set<string>();
  for (const rawLine of content.split(/\r?\n/)) {
    if (!rawLine.startsWith("ATOM") && !rawLine.startsWith("HETATM")) continue;
    const parts = rawLine.trim().split(/\s+/);
    const atomType = parts[parts.length - 1] ?? "";
    if (/^[A-Za-z]+$/.test(atomType)) types.add(atomType);
  }
  return Array.from(types).sort();
}

/** ATOM/HETATM coordinates as [x, y, z] triples. */
export function extractPdbqtCoords(content: string): number[][] {
  const coords: number[][] = [];
  for (const rawLine of content.split(/\r?\n/)) {
    if (!rawLine.startsWith("ATOM") && !rawLine.startsWith("HETATM")) continue;
    let parsed = false;
    if (rawLine.length >= 54) {
      const x = Number.parseFloat(rawLine.slice(30, 38));
      const y = Number.parseFloat(rawLine.slice(38, 46));
      const z = Number.parseFloat(rawLine.slice(46, 54));
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
        coords.push([x, y, z]);
        parsed = true;
      }
    }
    if (!parsed) {
      // Token-order fallback: the last three tokens of the first run of
      // four consecutive float-parsable tokens (chain-less records
      // shift every field one token left).
      const parts = rawLine.trim().split(/\s+/);
      for (let start = 0; start + 4 <= parts.length; start += 1) {
        const window = parts.slice(start, start + 4);
        const values = window.map((t) => Number.parseFloat(t));
        if (values.every(Number.isFinite)) {
          coords.push(values.slice(1));
          break;
        }
      }
    }
  }
  return coords;
}

/** TORSDOF value of a ligand PDBQT (last well-formed record wins). */
export function readTorsdof(content: string): number {
  let torsdof = 0;
  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (line.startsWith("TORSDOF")) {
      const value = line.split(/\s+/)[1] ?? "";
      if (/^\d+$/.test(value)) torsdof = Number.parseInt(value, 10);
    }
  }
  return torsdof;
}

/* ------------------------------------------------------------------ */
/* Grid box — mirror of compute_grid_box                               */
/* ------------------------------------------------------------------ */

/** Docking search box geometry (GridBox + its `size` property). */
export interface GridBoxTs {
  centerX: number;
  centerY: number;
  centerZ: number;
  npts: [number, number, number];
  spacing: number;
  /** Edge lengths in Å — fencepost: (npts - 1) * spacing. */
  size: [number, number, number];
}

/**
 * Grid box enclosing a reference ligand plus symmetric padding. The
 * centre is the midpoint of the ligand bounding box; `npts` is a
 * per-axis *minimum* — each axis grows so the box size covers the
 * ligand extent plus `2 * padding` (ceil(required/spacing) + 1 grid
 * points), never shrinking below the requested minimum.
 */
export function computeGridBox(
  coords: number[][],
  padding: number,
  npts: [number, number, number],
  spacing: number
): GridBoxTs {
  if (coords.length === 0) {
    throw new Error("no atom coordinates found in the ligand PDBQT");
  }
  const mins: [number, number, number] = [Infinity, Infinity, Infinity];
  const maxs: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (const [x, y, z] of coords) {
    if (x < mins[0]) mins[0] = x;
    if (y < mins[1]) mins[1] = y;
    if (z < mins[2]) mins[2] = z;
    if (x > maxs[0]) maxs[0] = x;
    if (y > maxs[1]) maxs[1] = y;
    if (z > maxs[2]) maxs[2] = z;
  }
  const center: [number, number, number] = [
    (mins[0] + maxs[0]) / 2,
    (mins[1] + maxs[1]) / 2,
    (mins[2] + maxs[2]) / 2,
  ];
  const out: [number, number, number] = [0, 0, 0];
  for (let axis = 0; axis < 3; axis += 1) {
    const required = maxs[axis] - mins[axis] + 2 * padding;
    out[axis] = Math.max(
      npts[axis],
      Math.ceil(required / spacing) + 1
    );
  }
  return {
    centerX: center[0],
    centerY: center[1],
    centerZ: center[2],
    npts: out,
    spacing,
    size: [
      (out[0] - 1) * spacing,
      (out[1] - 1) * spacing,
      (out[2] - 1) * spacing,
    ],
  };
}

/* ------------------------------------------------------------------ */
/* Config builders — mirrors of write_gpf / write_dpf                  */
/* ------------------------------------------------------------------ */

/** Default AutoDock4 receptor atom types (RECEPTOR_TYPES). */
export const DEFAULT_RECEPTOR_TYPES = "A C HD N NA OA SA";

/**
 * Python str(float): keeps a trailing ".0" on integral values so the
 * generated lines match the Python writers byte-for-byte.
 */
function pyFloat(value: number): string {
  return Number.isInteger(value) ? `${value}.0` : String(value);
}

/** Python round(x, 3) followed by str() — used by the DPF `about` line. */
function pyRound3(value: number): string {
  return pyFloat(Math.round(value * 1000) / 1000);
}

export interface GpfOptions {
  /** Sorted ligand atom types (extractPdbqtAtomTypes output). */
  ligandTypes: string[];
  /** Receptor PDBQT filename referenced in the GPF. */
  receptorFile: string;
  box: GridBoxTs;
  /** Space-separated receptor types; null = library default. */
  receptorTypes?: string | null;
  smooth?: number;
  dielectric?: number;
}

/** Build the AutoDock4 grid parameter file text for one ligand. */
export function buildGpfText(options: GpfOptions): string {
  const { box, ligandTypes, receptorFile } = options;
  const receptorTypes = options.receptorTypes ?? DEFAULT_RECEPTOR_TYPES;
  const smooth = pyFloat(options.smooth ?? 0.5);
  const dielectric = pyFloat(options.dielectric ?? -0.1465);
  const lines: string[] = [
    `npts ${box.npts[0]} ${box.npts[1]} ${box.npts[2]}          # num.grid points in xyz`,
    "gridfld r.maps.fld                    # grid_data_file",
    `spacing ${pyFloat(box.spacing)}                        # spacing(A)`,
    `receptor_types ${receptorTypes}     # receptor atom types`,
    `ligand_types ${ligandTypes.join(" ")}            # ligand atom types`,
    `receptor ${receptorFile}                     # macromolecule`,
    `gridcenter ${pyFloat(box.centerX)} ${pyFloat(box.centerY)} ${pyFloat(box.centerZ)}      # xyz-coordinates`,
    `smooth ${smooth}                           # store minimum energy w/in rad(A)`,
  ];
  for (const t of ligandTypes) {
    lines.push(`map r.${t}.map                          # atom-specific affinity map`);
  }
  lines.push(
    "elecmap r.e.map                      # electrostatic potential map",
    "dsolvmap r.d.map              # desolvation potential map",
    `dielectric ${dielectric}                   # <0, AD4 distance-dep.diel;>0, constant`
  );
  return `${lines.join("\n")}\n`;
}

export interface DpfOptions {
  /** Ligand filename (the `move` record). */
  ligandName: string;
  ligandTypes: string[];
  /** Centroid of the ligand atoms — the `about` rotation centre. */
  about: [number, number, number];
  torsdof: number;
  gaRun: number;
  gaPopSize: number;
}

/** Build the AutoDock4 docking parameter file text for one ligand. */
export function buildDpfText(options: DpfOptions): string {
  const { ligandTypes, ligandName, about, torsdof } = options;
  const lines: string[] = [
    "autodock_parameter_version 4.2",
    "outlev 1",
    "intelec",
    "seed pid time",
    `ligand_types ${ligandTypes.join(" ")}`,
    "fld r.maps.fld",
  ];
  for (const t of ligandTypes) {
    lines.push(`map r.${t}.map`);
  }
  lines.push(
    "elecmap r.e.map",
    "desolvmap r.d.map",
    `move ${ligandName}`,
    `about ${pyRound3(about[0])} ${pyRound3(about[1])} ${pyRound3(about[2])}`,
    "tran0 random",
    "quaternion0 random",
    "dihe0 random",
    `torsdof ${torsdof}`,
    "rmstol 2.0",
    "extnrg 1000.0",
    "e0max 0.0 10000",
    `ga_pop_size ${options.gaPopSize}`,
    "ga_num_evals 2500000",
    "ga_num_generations 27000",
    "ga_elitism 1",
    "ga_mutation_rate 0.02",
    "ga_crossover_rate 0.8",
    "ga_window_size 10",
    "ga_cauchy_alpha 0.0",
    "ga_cauchy_beta 1.0",
    "set_ga",
    "sw_max_its 300",
    "sw_max_succ 4",
    "sw_max_fail 4",
    "sw_rho 1.0",
    "sw_lb_rho 0.01",
    "ls_search_freq 0.06",
    "set_psw1",
    "unbound_model bound",
    `ga_run ${options.gaRun}`,
    "analysis"
  );
  return `${lines.join("\n")}\n`;
}

/* ------------------------------------------------------------------ */
/* Batch configs — mirror of write_config_batch                        */
/* ------------------------------------------------------------------ */

/** One campaign sub-folder with the PDBQT files found inside it. */
export interface CampaignFolder {
  folder: string;
  files: Array<{ name: string; content: string }>;
}

export interface BatchConfigOptions {
  receptorFile: string;
  padding: number;
  npts: [number, number, number];
  spacing: number;
  /** Custom receptor types (e.g. "A C HD N Ni OA SA"); null = default. */
  receptorTypes?: string | null;
  gaRun: number;
  gaPopSize: number;
}

/** One report row — the write_config_batch DataFrame, config texts in memory. */
export interface BatchConfigRow {
  folder: string;
  /** Reference ligand filename (empty when none was found). */
  ligand: string;
  /** "ok" | "no ligand PDBQT" | "failed: <reason>" */
  status: string;
  /** Box centre as "x y z" (empty when no box). */
  centre: string;
  npts: [number, number, number] | null;
  ligandTypes: string[];
  gpfText: string | null;
  dpfText: string | null;
}

/**
 * Build the r.gpf + r.dpf pair for every campaign sub-folder: the first
 * `*.pdbqt` that is not the receptor file becomes the reference ligand,
 * the grid box is computed from it, and both config texts are generated
 * using the package's campaign-preparation conventions. Folders without
 * a candidate ligand are reported instead of aborting the batch. The
 * web console presents generated text through download buttons; this
 * browser helper is separate from the React/Tauri Studio.
 */
export function writeConfigBatch(
  folders: CampaignFolder[],
  options: BatchConfigOptions
): BatchConfigRow[] {
  return folders.map((entry) => {
    const row: BatchConfigRow = {
      folder: entry.folder,
      ligand: "",
      status: "",
      centre: "",
      npts: null,
      ligandTypes: [],
      gpfText: null,
      dpfText: null,
    };
    const candidates = entry.files
      .filter(
        (f) =>
          f.name.toLowerCase().endsWith(".pdbqt") &&
          f.name !== options.receptorFile
      )
      .sort((a, b) => a.name.localeCompare(b.name));
    const ligand = candidates[0];
    if (!ligand) {
      row.status = "no ligand PDBQT";
      return row;
    }
    row.ligand = ligand.name;
    const ligandTypes = extractPdbqtAtomTypes(ligand.content);
    if (ligandTypes.length === 0) {
      row.status = `failed: no atom types found in ${ligand.name}`;
      return row;
    }
    row.ligandTypes = ligandTypes;
    const coords = extractPdbqtCoords(ligand.content);
    try {
      const box = computeGridBox(coords, options.padding, options.npts, options.spacing);
      row.npts = box.npts;
      row.centre = [box.centerX, box.centerY, box.centerZ]
        .map((v) => v.toFixed(3))
        .join(" ");
      const centroid: [number, number, number] = [0, 0, 0];
      for (const [x, y, z] of coords) {
        centroid[0] += x;
        centroid[1] += y;
        centroid[2] += z;
      }
      const n = Math.max(coords.length, 1);
      row.gpfText = buildGpfText({
        ligandTypes,
        receptorFile: options.receptorFile,
        box,
        receptorTypes: options.receptorTypes ?? null,
      });
      row.dpfText = buildDpfText({
        ligandName: ligand.name,
        ligandTypes,
        about: [centroid[0] / n, centroid[1] / n, centroid[2] / n],
        torsdof: readTorsdof(ligand.content),
        gaRun: options.gaRun,
        gaPopSize: options.gaPopSize,
      });
      row.status = "ok";
    } catch (err) {
      row.status = `failed: ${err instanceof Error ? err.message : String(err)}`;
    }
    return row;
  });
}

/* ------------------------------------------------------------------ */
/* File-pick helpers (shared by the two cards)                         */
/* ------------------------------------------------------------------ */

/**
 * Group a `webkitdirectory` pick into campaign folders: every immediate
 * sub-directory of the picked root becomes one ligand folder holding
 * its PDBQT files. Files sitting directly in the root are not ligand
 * folders, matching the package's one-directory-per-ligand convention.
 */
export function groupCampaignFiles(
  files: Array<{ path: string; content: string }>
): CampaignFolder[] {
  const byFolder = new Map<string, Array<{ name: string; content: string }>>();
  for (const file of files) {
    const parts = file.path.split("/").filter((p) => p.length > 0);
    if (parts.length < 3) continue; // picked-root itself or deeper? root/<folder>/<file> = 3
    const folder = parts[parts.length - 2];
    if (!byFolder.has(folder)) byFolder.set(folder, []);
    byFolder.get(folder)?.push({ name: parts[parts.length - 1] ?? "", content: file.content });
  }
  return Array.from(byFolder.entries())
    .map(([folder, files]) => ({ folder, files }))
    .sort((a, b) => a.folder.localeCompare(b.folder));
}
