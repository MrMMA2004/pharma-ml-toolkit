/**
 * Docking output parsers (Vina / AutoDock4 DLG / generic CSV) plus
 * ligand-efficiency and consensus z-score scoring. Pure functions —
 * used by the /api/docking/parse route.
 */

import type { DockingChartDatum, DockingFormat, DockingPose } from "@/lib/toolkit-types";

export interface ParsedDocking {
  poses: DockingPose[];
  /** Score columns detected (CSV only; single column otherwise). */
  scoreColumns: string[];
}

/* ------------------------------------------------------------------ */
/* Vina output                                                         */
/* ------------------------------------------------------------------ */

/**
 * Parse classic AutoDock Vina output:
 *
 * ```
 * mode |   affinity | dist from best mode
 *      | (kcal/mol) | rmsd l.b. | rmsd u.b.
 * -----+------------+----------+----------
 *    1         -9.6      0.000      0.000
 * ```
 */
export function parseVinaOutput(text: string): DockingPose[] {
  const poses: DockingPose[] = [];
  let pastSeparator = false;
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    // Dashed separator such as -----+----------+----------
    if (/^-[-+]+$/.test(line)) {
      pastSeparator = true;
      continue;
    }
    if (!pastSeparator || line.length === 0) continue;
    const parts = line.split(/\s+/);
    if (parts.length < 2) continue;
    const mode = Number.parseInt(parts[0] ?? "", 10);
    const affinity = Number.parseFloat(parts[1] ?? "");
    if (!Number.isInteger(mode) || mode < 1 || !Number.isFinite(affinity)) continue;
    const rmsdLb = Number.parseFloat(parts[2] ?? "");
    const rmsdUb = Number.parseFloat(parts[3] ?? "");
    poses.push({
      rank: poses.length + 1,
      mode,
      affinity,
      rmsdLb: Number.isFinite(rmsdLb) ? rmsdLb : undefined,
      rmsdUb: Number.isFinite(rmsdUb) ? rmsdUb : undefined,
    });
  }
  return poses;
}

/* ------------------------------------------------------------------ */
/* AutoDock4 DLG cluster table                                         */
/* ------------------------------------------------------------------ */

/**
 * Parse AutoDock4 DLG clustering table rows:
 *
 * ```
 * | Cluster Rank |  Lowest Energy  |  Run  |  Mean Energy  |  # in cluster |
 * |      1       |     -10.38      |  47   |    -10.21      |       5       |
 * ```
 */
export function parseDlgClusterTable(text: string): DockingPose[] {
  const poses: DockingPose[] = [];
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line.includes("|")) continue;
    if (/^[_\-|]+$/.test(line.replace(/\s/g, ""))) continue; // separator row
    const parts = line
      .split("|")
      .map((p) => p.trim())
      .filter((p) => p.length > 0);
    if (parts.length < 4) continue;
    const rank = Number.parseInt(parts[0] ?? "", 10);
    if (!Number.isInteger(rank) || rank < 1) continue; // header row
    const affinity = Number.parseFloat(parts[1] ?? "");
    if (!Number.isFinite(affinity)) continue;
    const run = Number.parseInt(parts[2] ?? "", 10);
    const meanEnergy = Number.parseFloat(parts[3] ?? "");
    const clusterSize = parts.length > 4 ? Number.parseInt(parts[4] ?? "", 10) : NaN;
    poses.push({
      rank: poses.length + 1,
      mode: rank,
      affinity,
      run: Number.isInteger(run) ? run : undefined,
      meanEnergy: Number.isFinite(meanEnergy) ? meanEnergy : undefined,
      clusterSize: Number.isInteger(clusterSize) ? clusterSize : undefined,
    });
  }
  return poses;
}

/* ------------------------------------------------------------------ */
/* Generic CSV                                                         */
/* ------------------------------------------------------------------ */

const SCORE_HINTS = ["score", "affinity", "energy", "docking", "vina", "ad4"];
const MODE_COLUMN_NAMES = ["pose", "mode", "rank", "id"];

function isNumericColumn(values: string[]): boolean {
  const nonEmpty = values.filter((v) => v.trim().length > 0);
  if (nonEmpty.length === 0) return false;
  const numeric = nonEmpty.filter((v) => Number.isFinite(Number.parseFloat(v)));
  return numeric.length / nonEmpty.length >= 0.8;
}

/** Parse a header-defined CSV of docking poses. Ranks by ascending best score. */
export function parseDockingCsv(text: string): ParsedDocking {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  if (lines.length < 2) {
    return { poses: [], scoreColumns: [] };
  }

  const delimiter = detectDelimiter(lines[0] ?? "");
  const header = (lines[0] ?? "")
    .split(delimiter)
    .map((h) => h.trim().replace(/^"|"$/g, ""));
  if (header.length < 2 || header.every((h) => h.length === 0)) {
    return { poses: [], scoreColumns: [] };
  }

  const rows: string[][] = [];
  for (const line of lines.slice(1)) {
    const cells = line.split(delimiter).map((c) => c.trim().replace(/^"|"$/g, ""));
    if (cells.length !== header.length) continue;
    rows.push(cells);
  }

  // Identify numeric columns.
  const numericCols: number[] = [];
  header.forEach((_, ci) => {
    const col = rows.map((r) => r[ci] ?? "");
    if (isNumericColumn(col)) numericCols.push(ci);
  });
  if (numericCols.length === 0) {
    return { poses: [], scoreColumns: [] };
  }

  // Score columns: prefer named score-like columns, else the first numeric one.
  let scoreCols = numericCols.filter((ci) =>
    SCORE_HINTS.some((hint) => (header[ci] ?? "").toLowerCase().includes(hint))
  );
  if (scoreCols.length === 0) scoreCols = [numericCols[0] ?? 0];

  // Mode/pose id column: explicit name match, else the first numeric column
  // that is not a score column.
  let modeCol = header.findIndex((h) => MODE_COLUMN_NAMES.includes(h.toLowerCase()));
  if (modeCol < 0) {
    modeCol = numericCols.find((ci) => !scoreCols.includes(ci)) ?? -1;
  }

  // Ligand column: first non-numeric column (or a column named ligand/name).
  let ligandCol = header.findIndex((h) =>
    ["ligand", "name", "molecule", "compound"].includes(h.toLowerCase())
  );
  if (ligandCol < 0) {
    ligandCol = header.findIndex((_, ci) => !numericCols.includes(ci));
  }

  // Primary score = first score column.
  const primary = scoreCols[0] ?? numericCols[0] ?? 0;
  const structures: Array<{
    mode: number;
    ligand?: string;
    scores: Record<string, number>;
    extra: Record<string, number>;
  }> = rows.map((cells, ri) => {
    const modeVal = modeCol >= 0 ? Number.parseInt(cells[modeCol] ?? "", 10) : NaN;
    const scores: Record<string, number> = {};
    for (const ci of scoreCols) {
      const key = header[ci] ?? `score${ci}`;
      const v = Number.parseFloat(cells[ci] ?? "");
      scores[key] = Number.isFinite(v) ? v : Number.NaN;
    }
    const extra: Record<string, number> = {};
    for (const ci of numericCols) {
      if (ci === primary || ci === modeCol) continue;
      const key = header[ci] ?? `col${ci}`;
      const v = Number.parseFloat(cells[ci] ?? "");
      if (Number.isFinite(v)) extra[key] = v;
    }
    return {
      mode: Number.isInteger(modeVal) ? modeVal : ri + 1,
      ligand:
        ligandCol >= 0 && ligandCol !== primary && ligandCol !== modeCol
          ? cells[ligandCol] || undefined
          : undefined,
      scores,
      extra,
    };
  });

  // More negative = better for docking affinities → ascending sort.
  structures.sort((a, b) => (a.scores[header[primary] ?? ""] ?? 0) - (b.scores[header[primary] ?? ""] ?? 0));

  const primaryKey = header[primary] ?? "score";
  const poses: DockingPose[] = structures.map((s, i) => ({
    rank: i + 1,
    mode: s.mode,
    ligand: s.ligand,
    affinity: s.scores[primaryKey] ?? Number.NaN,
    extraScores: Object.keys(s.extra).length > 0 ? s.extra : undefined,
  }));

  return { poses, scoreColumns: scoreCols.map((ci) => header[ci] ?? `col${ci}`) };
}

function detectDelimiter(headerLine: string): string {
  const commas = (headerLine.match(/,/g) ?? []).length;
  const semis = (headerLine.match(/;/g) ?? []).length;
  const tabs = (headerLine.match(/\t/g) ?? []).length;
  if (tabs > commas && tabs > semis) return "\t";
  if (semis > commas) return ";";
  return ",";
}

/* ------------------------------------------------------------------ */
/* Scoring helpers                                                     */
/* ------------------------------------------------------------------ */

/** Ligand efficiency: affinity (kcal/mol) per heavy atom. */
export function ligandEfficiency(affinity: number, heavyAtoms: number): number {
  if (!Number.isFinite(affinity) || heavyAtoms <= 0) return Number.NaN;
  return affinity / heavyAtoms;
}

/**
 * Consensus z-score rescoring across multiple score columns.
 * Each column is z-scored across poses, the z values are summed, and
 * poses are re-ranked by ascending consensus (lower = better).
 */
export function consensusRescore(
  poses: DockingPose[],
  scoreColumns: string[],
  columnValues: Array<Map<number, Record<string, number>>>
): void {
  if (scoreColumns.length < 2 || poses.length < 2) return;

  // Build pose -> {column -> value} lookup using row index.
  const perPose: Array<Record<string, number>> = poses.map((_, i) => {
    const acc: Record<string, number> = {};
    columnValues.forEach((m, _ci) => {
      const entry = m.get(i);
      if (entry) Object.assign(acc, entry);
    });
    return acc;
  });

  // z-score per column
  for (const col of scoreColumns) {
    const values = perPose.map((p) => p[col]).filter((v) => Number.isFinite(v));
    if (values.length < 2) continue;
    const m = values.reduce((a, b) => a + b, 0) / values.length;
    const sd = Math.sqrt(values.reduce((a, b) => a + (b - m) ** 2, 0) / values.length);
    for (const p of perPose) {
      const v = p[col];
      p[col] = Number.isFinite(v) && sd > 1e-12 ? (v - m) / sd : 0;
    }
  }

  poses.forEach((pose, i) => {
    const zs = scoreColumns.map((c) => perPose[i]?.[c] ?? 0);
    pose.zScore = zs.reduce((a, b) => a + b, 0);
  });

  const ranked = [...poses].sort((a, b) => (a.zScore ?? 0) - (b.zScore ?? 0));
  ranked.forEach((p, i) => {
    p.consensusRank = i + 1;
  });
}

/* ------------------------------------------------------------------ */
/* Entry point                                                         */
/* ------------------------------------------------------------------ */

export interface ParseDockingResult {
  format: DockingFormat;
  poses: DockingPose[];
  scoreColumns: string[];
  chartData: DockingChartDatum[];
  stats: { best: number; worst: number; mean: number; n: number };
  consensus: { applied: boolean; columns: string[]; note: string } | undefined;
}

export function parseDockingOutput(
  format: DockingFormat,
  content: string,
  heavyAtoms: number
): ParseDockingResult {
  let parsed: ParsedDocking;
  switch (format) {
    case "vina":
      parsed = { poses: parseVinaOutput(content), scoreColumns: [] };
      break;
    case "dlg":
      parsed = { poses: parseDlgClusterTable(content), scoreColumns: [] };
      break;
    case "csv":
      parsed = parseDockingCsv(content);
      break;
  }

  const poses = parsed.poses;
  if (poses.length === 0) {
    throw new Error(
      `No pose rows could be parsed from the input as ${formatLabel(format)} output. ` +
        "Check the format selector, or load a sample to see the expected layout."
    );
  }

  // Ligand efficiency
  for (const p of poses) {
    p.ligandEfficiency = ligandEfficiency(p.affinity, heavyAtoms);
  }

  // Consensus z-score rescoring for CSV with ≥2 score columns.
  let consensus: ParseDockingResult["consensus"];
  if (format === "csv" && parsed.scoreColumns.length >= 2 && poses.length >= 2) {
    // Rebuild raw column values per pose rank for z-scoring.
    const columnValues = buildColumnValues(content, parsed.scoreColumns);
    consensusRescore(poses, parsed.scoreColumns, columnValues);
    consensus = {
      applied: true,
      columns: parsed.scoreColumns,
      note:
        "Consensus z-score rescoring applied: each score column was z-scored across poses, " +
        "z values summed, and poses re-ranked by ascending consensus score (lower = better).",
    };
  } else if (format === "csv" && parsed.scoreColumns.length < 2) {
    consensus = {
      applied: false,
      columns: parsed.scoreColumns,
      note: "Only one score column detected — add a second score column to enable consensus z-score rescoring.",
    };
  }

  const affinities = poses.map((p) => p.affinity);
  const best = Math.min(...affinities);
  const worst = Math.max(...affinities);
  const meanAff = affinities.reduce((a, b) => a + b, 0) / affinities.length;

  const chartData: DockingChartDatum[] = poses.map((p) => ({
    mode: format === "csv" && p.ligand ? `${p.ligand} #${p.mode}` : String(p.mode ?? p.rank),
    affinity: p.affinity,
    best: p.rank === 1,
  }));

  return {
    format,
    poses,
    scoreColumns: parsed.scoreColumns,
    chartData,
    stats: { best, worst, mean: meanAff, n: poses.length },
    consensus,
  };
}

function formatLabel(format: DockingFormat): string {
  switch (format) {
    case "vina":
      return "AutoDock Vina";
    case "dlg":
      return "AutoDock4 DLG cluster table";
    case "csv":
      return "generic CSV";
  }
}

/**
 * Re-parse CSV to keep every score column's raw values aligned with the
 * ranked pose order (rank order matches the sort in parseDockingCsv).
 */
function buildColumnValues(
  content: string,
  scoreColumns: string[]
): Array<Map<number, Record<string, number>>> {
  const maps: Array<Map<number, Record<string, number>>> = [];
  const lines = content
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);
  if (lines.length < 2) return maps;

  const delimiter = detectDelimiter(lines[0] ?? "");
  const header = (lines[0] ?? "").split(delimiter).map((h) => h.trim().replace(/^"|"$/g, ""));
  const rows = lines
    .slice(1)
    .map((l) => l.split(delimiter).map((c) => c.trim().replace(/^"|"$/g, "")))
    .filter((cells) => cells.length === header.length);

  // Locate score column indices.
  const colIdx = scoreColumns.map((name) => header.indexOf(name));

  // Primary score for ranking = first score column.
  const primaryIdx = colIdx[0] ?? 0;
  const parsedRows = rows
    .map((cells) => ({
      cells,
      primary: Number.parseFloat(cells[primaryIdx] ?? ""),
    }))
    .filter((r) => Number.isFinite(r.primary))
    .sort((a, b) => a.primary - b.primary); // same order as poses

  for (let k = 0; k < scoreColumns.length; k++) {
    const ci = colIdx[k] ?? -1;
    const m = new Map<number, Record<string, number>>();
    parsedRows.forEach((row, i) => {
      const v = Number.parseFloat(row.cells[ci] ?? "");
      if (Number.isFinite(v)) m.set(i, { [scoreColumns[k] ?? ""]: v });
    });
    maps.push(m);
  }
  return maps;
}
