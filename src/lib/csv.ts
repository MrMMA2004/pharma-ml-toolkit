/**
 * Dependency-free CSV parsing + ML dataset resolution for user-uploaded data.
 *
 * Shared by:
 *  - the client (Playground column pickers parse the header client-side),
 *  - /api/ml/train (trains on the resolved columns),
 *  - /api/ml/dataset (previews the user's rows/stats).
 *
 * Pure TypeScript, no Node/browser-specific APIs, so it runs in both runtimes.
 */

import { mean, populationStd } from "./ml-math";

/** Hard cap on accepted CSV text length (characters ≈ 256 KB). */
export const MAX_CSV_TEXT = 262_144;

/** Minimum usable rows required to train/preview. */
export const MIN_CSV_ROWS = 20;

/** Minimum rows per class for classification. */
export const MIN_CLASS_ROWS = 5;

export interface CsvTable {
  /** Header names (deduplicated, blank cells named `col1`, `col2`, …). */
  columns: string[];
  /** Data rows (raw string cells, trimmed, padded to header width). */
  rows: string[][];
}

/* ------------------------------------------------------------------ */
/* Parsing                                                             */
/* ------------------------------------------------------------------ */

/** Split a CSV record string into fields, honoring double-quoted cells. */
function splitLine(line: string, delim: string): string[] {
  const fields: string[] = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i] ?? "";
    if (inQuotes) {
      if (ch === '"') {
        if (line[i + 1] === '"') {
          cur += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        cur += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === delim) {
      fields.push(cur);
      cur = "";
    } else {
      cur += ch;
    }
  }
  fields.push(cur);
  return fields;
}

/** Sniff the delimiter (`,`, `;` or tab) from the first non-empty line. */
function sniffDelimiter(text: string): string {
  const firstLine = text.split(/\r\n|\r|\n/, 1)[0] ?? "";
  const counts: Array<[string, number]> = [
    [",", (firstLine.match(/,/g) ?? []).length],
    [";", (firstLine.match(/;/g) ?? []).length],
    ["\t", (firstLine.match(/\t/g) ?? []).length],
  ];
  counts.sort((a, b) => b[1] - a[1]);
  return counts[0]?.[1] ? counts[0][0] : ",";
}

/** Unique, non-blank column names; unnamed columns become `colN`. */
function normalizeHeader(raw: string[]): string[] {
  const seen = new Set<string>();
  return raw.map((cell, i) => {
    let name = cell.trim();
    if (!name) name = `col${i + 1}`;
    let unique = name;
    let n = 2;
    while (seen.has(unique)) unique = `${name}_${n++}`;
    seen.add(unique);
    return unique;
  });
}

/**
 * Parse CSV text into a table. Handles quoted fields with escaped `""`,
 * CRLF/CR/LF line endings, `,` `;` and tab delimiters (auto-sniffed), a
 * leading BOM, and ragged rows (padded/truncated to the header width).
 * Fully blank lines are skipped. Trailing data lines without a header
 * still work — columns fall back to `col1…colN`.
 */
export function parseCsv(text: string): CsvTable {
  const clean = text.replace(/^\uFEFF/, "").replace(/\r\n?/g, "\n");
  const delim = sniffDelimiter(clean);
  const records = clean
    .split("\n")
    .map((line) => splitLine(line, delim))
    .filter((fields) => !(fields.length === 1 && fields[0]?.trim() === ""));
  if (records.length === 0) return { columns: [], rows: [] };

  /* Width via loop (spread would overflow the call stack on huge files). */
  let width = 0;
  for (const r of records) if (r.length > width) width = r.length;
  const header = normalizeHeader(
    records[0] !== undefined && records[0].some((c) => c.trim() !== "")
      ? records[0]
      : Array.from({ length: width }, (_, i) => `col${i + 1}`)
  );
  const rows = records.slice(1).map((r) => {
    const padded = r.map((c) => c.trim());
    while (padded.length < header.length) padded.push("");
    return padded.slice(0, header.length);
  });
  return { columns: header, rows };
}

/** Parse one cell as a finite number; null for blank/non-numeric. */
function num(cell: string): number | null {
  if (!cell) return null;
  const v = Number(cell);
  return Number.isFinite(v) ? v : null;
}

/** Column counts as numeric when ≥ 90% of its non-empty cells parse as
 *  finite numbers. Junk cells don't disqualify the column — the rows that
 *  contain them get dropped (and counted) during dataset resolution. */
function columnIsNumeric(rows: string[][], colIdx: number): boolean {
  let nonEmpty = 0;
  let parseable = 0;
  for (const row of rows) {
    const cell = row[colIdx] ?? "";
    if (cell === "") continue;
    nonEmpty++;
    if (num(cell) !== null) parseable++;
  }
  return nonEmpty > 0 && parseable / nonEmpty >= 0.9;
}

/** Names of columns that are numeric (candidates for features/target). */
export function numericColumnNames(table: CsvTable): string[] {
  return table.columns.filter((_, i) => columnIsNumeric(table.rows, i));
}

/** Distinct non-empty raw values of a column (order of first appearance). */
function distinctValues(table: CsvTable, colIdx: number): string[] {
  const seen: string[] = [];
  for (const row of table.rows) {
    const v = (row[colIdx] ?? "").trim();
    if (v !== "" && !seen.includes(v)) seen.push(v);
  }
  return seen;
}

/* ------------------------------------------------------------------ */
/* Dataset resolution                                                  */
/* ------------------------------------------------------------------ */

/** Maximum features accepted in multivariate regression. */
export const MAX_CSV_FEATURES = 8;

export interface CsvRegressionData {
  /** Raw feature values (standardization happens in the trainer). */
  x: number[];
  /** Raw target values. */
  y: number[];
  feature: string;
  target: string;
  /** Numeric columns detected in the file (for the UI pickers). */
  columns: string[];
  /** Rows skipped because a used cell was blank/non-numeric. */
  droppedRows: number;
}

/** Resolve a feature column + target column for regression training. */
export function resolveRegression(
  table: CsvTable,
  featureColumn?: string,
  targetColumn?: string
): CsvRegressionData {
  const numeric = numericColumnNames(table);
  if (numeric.length < 2) {
    throw new Error(
      `CSV needs at least 2 fully-numeric columns (found ${numeric.length}${
        numeric.length ? `: ${numeric.join(", ")}` : ""
      }).`
    );
  }
  const feature =
    featureColumn && numeric.includes(featureColumn) ? featureColumn : numeric[0]!;
  const target =
    targetColumn && numeric.includes(targetColumn) && targetColumn !== feature
      ? targetColumn
      : numeric[numeric.length - 1] !== feature
        ? numeric[numeric.length - 1]!
        : numeric[1]!;

  const fi = table.columns.indexOf(feature);
  const ti = table.columns.indexOf(target);
  const x: number[] = [];
  const y: number[] = [];
  let droppedRows = 0;
  for (const row of table.rows) {
    const xv = num(row[fi] ?? "");
    const yv = num(row[ti] ?? "");
    if (xv === null || yv === null) {
      droppedRows++;
      continue;
    }
    x.push(xv);
    y.push(yv);
  }
  if (x.length < MIN_CSV_ROWS) {
    throw new Error(
      `Only ${x.length} usable rows (need ≥ ${MIN_CSV_ROWS}); ${droppedRows} rows were dropped for blank/non-numeric cells in "${feature}"/"${target}".`
    );
  }
  if (populationStd(x) === 0) {
    throw new Error(`Feature column "${feature}" is constant — nothing to fit.`);
  }
  if (populationStd(y) === 0) {
    throw new Error(`Target column "${target}" is constant — nothing to predict.`);
  }
  return { x, y, feature, target, columns: numeric, droppedRows };
}

export interface CsvMultiRegressionData {
  /** Row-major feature matrix (raw values, n × k). */
  X: number[][];
  /** Raw target values. */
  y: number[];
  /** Selected feature column names, in matrix column order. */
  features: string[];
  target: string;
  /** Numeric columns detected in the file (for the UI pickers). */
  columns: string[];
  /** Rows skipped because a used cell was blank/non-numeric. */
  droppedRows: number;
}

/** Resolve multiple feature columns + a target column for multivariate
 *  regression. Auto mode: every numeric column except the target (≤ 8). */
export function resolveMultiRegression(
  table: CsvTable,
  featureColumns?: string[],
  targetColumn?: string
): CsvMultiRegressionData {
  const numeric = numericColumnNames(table);
  if (numeric.length < 2) {
    throw new Error(
      `CSV needs at least 2 fully-numeric columns (found ${numeric.length}${
        numeric.length ? `: ${numeric.join(", ")}` : ""
      }).`
    );
  }

  /* Target: requested, or the last numeric column. */
  const target =
    targetColumn && numeric.includes(targetColumn)
      ? targetColumn
      : numeric[numeric.length - 1]!;

  /* Features: requested picks (validated, deduplicated) or all numeric
   * columns except the target, capped at MAX_CSV_FEATURES. */
  let features: string[];
  if (featureColumns && featureColumns.length > 0) {
    const seen = new Set<string>();
    features = featureColumns.filter(
      (c) => numeric.includes(c) && c !== target && !seen.has(c) && seen.add(c)
    );
    if (features.length === 0) {
      throw new Error(
        `None of the requested feature columns are usable (numeric and distinct from target "${target}").`
      );
    }
  } else {
    features = numeric.filter((c) => c !== target).slice(0, MAX_CSV_FEATURES);
  }
  if (features.length > MAX_CSV_FEATURES) {
    features = features.slice(0, MAX_CSV_FEATURES);
  }
  if (features.length < 1) {
    throw new Error(`No numeric feature column besides "${target}" was found.`);
  }

  const idxs = features.map((f) => table.columns.indexOf(f));
  const ti = table.columns.indexOf(target);
  const X: number[][] = [];
  const y: number[] = [];
  let droppedRows = 0;
  for (const row of table.rows) {
    const feats: number[] = [];
    let ok = true;
    for (const fi of idxs) {
      const v = num(row[fi] ?? "");
      if (v === null) {
        ok = false;
        break;
      }
      feats.push(v);
    }
    const yv = ok ? num(row[ti] ?? "") : null;
    if (!ok || yv === null) {
      droppedRows++;
      continue;
    }
    X.push(feats);
    y.push(yv);
  }
  if (X.length < Math.max(MIN_CSV_ROWS, features.length + 2)) {
    throw new Error(
      `Only ${X.length} usable rows (need ≥ ${Math.max(MIN_CSV_ROWS, features.length + 2)}); ${droppedRows} rows were dropped for blank/non-numeric cells.`
    );
  }
  if (populationStd(y) === 0) {
    throw new Error(`Target column "${target}" is constant — nothing to predict.`);
  }
  for (let j = 0; j < features.length; j++) {
    const col = X.map((r) => r[j] ?? 0);
    if (populationStd(col) === 0) {
      throw new Error(`Feature column "${features[j]}" is constant — drop it or pick others.`);
    }
  }
  return { X, y, features, target, columns: numeric, droppedRows };
}

export interface CsvClassificationData {
  x1: number[];
  x2: number[];
  /** Binary labels mapped to 0 / 1. */
  label: Array<0 | 1>;
  feature1: string;
  feature2: string;
  target: string;
  /** Raw target values mapped to class 0 / class 1 (sorted). */
  labelMap: [string, string];
  columns: string[];
  droppedRows: number;
}

/** Resolve two feature columns + a binary label column for classification. */
export function resolveClassification(
  table: CsvTable,
  featureColumn?: string,
  featureColumn2?: string,
  targetColumn?: string
): CsvClassificationData {
  const numeric = numericColumnNames(table);

  /* Label column: requested, or the last column with exactly 2 distinct values. */
  let target: string | null = null;
  if (targetColumn && table.columns.includes(targetColumn)) {
    target = targetColumn;
  } else {
    for (let i = table.columns.length - 1; i >= 0; i--) {
      if (distinctValues(table, i).length === 2) {
        target = table.columns[i]!;
        break;
      }
    }
  }
  if (!target) {
    throw new Error(
      "No column with exactly 2 distinct values found — classification needs a binary label column (pick one explicitly)."
    );
  }
  const targetIdx = table.columns.indexOf(target);
  const distinct = distinctValues(table, targetIdx).sort();
  if (distinct.length !== 2) {
    throw new Error(
      `Label column "${target}" has ${distinct.length} distinct values — exactly 2 are required.`
    );
  }
  const [v0, v1] = distinct as [string, string];

  /* Features: two distinct numeric columns, excluding the label. */
  const featurePool = numeric.filter((c) => c !== target);
  if (featurePool.length < 2) {
    throw new Error(
      `CSV needs at least 2 numeric feature columns besides "${target}" (found ${featurePool.length}).`
    );
  }
  const feature1 =
    featureColumn && featurePool.includes(featureColumn) ? featureColumn : featurePool[0]!;
  const feature2 =
    featureColumn2 && featurePool.includes(featureColumn2) && featureColumn2 !== feature1
      ? featureColumn2
      : featurePool.find((c) => c !== feature1)!;

  const f1i = table.columns.indexOf(feature1);
  const f2i = table.columns.indexOf(feature2);
  const x1: number[] = [];
  const x2: number[] = [];
  const label: Array<0 | 1> = [];
  let droppedRows = 0;
  for (const row of table.rows) {
    const a = num(row[f1i] ?? "");
    const b = num(row[f2i] ?? "");
    const raw = (row[targetIdx] ?? "").trim();
    if (a === null || b === null || (raw !== v0 && raw !== v1)) {
      droppedRows++;
      continue;
    }
    x1.push(a);
    x2.push(b);
    label.push(raw === v1 ? 1 : 0);
  }
  if (label.length < MIN_CSV_ROWS) {
    throw new Error(
      `Only ${label.length} usable rows (need ≥ ${MIN_CSV_ROWS}); ${droppedRows} rows were dropped for blank/non-numeric cells.`
    );
  }
  const n1 = label.filter((l) => l === 1).length;
  const n0 = label.length - n1;
  if (n0 < MIN_CLASS_ROWS || n1 < MIN_CLASS_ROWS) {
    throw new Error(
      `Class balance too skewed: "${v0}" = ${n0}, "${v1}" = ${n1} (need ≥ ${MIN_CLASS_ROWS} each).`
    );
  }
  if (populationStd(x1) === 0 || populationStd(x2) === 0) {
    throw new Error(
      `Feature columns "${feature1}"/"${feature2}" must vary — at least one is constant.`
    );
  }
  return { x1, x2, label, feature1, feature2, target, labelMap: [v0, v1], columns: numeric, droppedRows };
}

/** z-score standardization ((v − μ) / σ); constant input maps to zeros. */
export function zscore(values: number[]): number[] {
  const mu = mean(values);
  const sigma = populationStd(values);
  if (sigma === 0) return values.map(() => 0);
  return values.map((v) => (v - mu) / sigma);
}

/** FNV-1a string hash → uint32 (deterministic split seeds for uploaded files). */
export function hashCsvSeed(text: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}
