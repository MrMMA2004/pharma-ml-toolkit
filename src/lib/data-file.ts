/**
 * Pure (isomorphic) helpers for previewing the toolkit's shipped data files
 * — CSV datasets, PDB structures and docking plain-text logs. No `fs` and
 * no Node builtins here (plain string ops only): the API route streams
 * bytes and feeds the decoders below, while client components use the
 * path-based kind detection for icons, badges and loading skeletons.
 *
 * Tree-stat integrity rule: data files are NEVER line-counted. The walk in
 * `toolkit-fs.ts` marks them with a `language` only (`lines` stays
 * undefined) so the LOC stat stays source-only.
 */

export type DataFileKind = "csv" | "pdb" | "text";

/** Hard caps — oversized files get an actionable 400, never a stalled read. */
export const CSV_MAX_BYTES = 512 * 1024;
export const TEXT_MAX_BYTES = 8 * 1024 * 1024;

/** Preview windows (rows / lines) returned to the client. */
export const CSV_PREVIEW_ROWS = 50;
export const PDB_PREVIEW_LINES = 200;
export const TEXT_PREVIEW_LINES = 400;

/** Tree nodes in these extensions are "data nodes" (no LOC, data icons). */
const TREE_DATA_EXTENSIONS = new Set([".csv", ".pdb", ".dlg", ".glg", ".gpf", ".dpf"]);

/** Extensions whose bytes are previewed as plain text (.txt included). */
const TEXT_PREVIEW_EXTENSIONS = new Set([".txt", ".dlg", ".glg", ".gpf", ".dpf"]);

/** Extension of the final path segment, lowercased ("" for dotfiles/dirs). */
function extOf(relPath: string): string {
  const slash = relPath.lastIndexOf("/");
  const base = slash === -1 ? relPath : relPath.slice(slash + 1);
  const dot = base.lastIndexOf(".");
  if (dot <= 0) return "";
  return base.slice(dot).toLowerCase();
}

/**
 * Classify a toolkit-relative path into its preview kind.
 * Returns null for real source files (Python, Markdown, TOML, …).
 */
export function dataFileKindFromPath(relPath: string): DataFileKind | null {
  const ext = extOf(relPath);
  if (ext === ".csv") return "csv";
  if (ext === ".pdb") return "pdb";
  if (TEXT_PREVIEW_EXTENSIONS.has(ext)) return "text";
  return null;
}

/**
 * Tree-surface data node? These get data icons + size labels in the file
 * tree and a `language` marker, but are never line-counted.
 */
export function isTreeDataNode(relPath: string): boolean {
  return TREE_DATA_EXTENSIONS.has(extOf(relPath));
}

/** Byte cap for a preview kind (source files are unbounded, as before). */
export function dataFileMaxBytes(kind: DataFileKind): number {
  return kind === "csv" ? CSV_MAX_BYTES : TEXT_MAX_BYTES;
}

/* ------------------------------------------------------------------ */
/* CSV decoding                                                        */
/* ------------------------------------------------------------------ */

/**
 * RFC4180-aware record scanner: handles quoted fields, embedded commas,
 * doubled quotes and CRLF line endings. Returns every record verbatim
 * (quotes stripped, fields split) — the caller slices the preview window.
 */
function scanCsvRecords(text: string): string[][] {
  const records: string[][] = [];
  let field = "";
  let record: string[] = [];
  let inQuotes = false;
  let fieldStarted = false;

  const pushField = () => {
    record.push(field);
    field = "";
    fieldStarted = false;
  };
  const pushRecord = () => {
    if (record.length > 0 || fieldStarted || field.length > 0) {
      pushField();
      records.push(record);
      record = [];
    } else {
      // Completely empty line outside quotes — skipped.
    }
  };

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
      fieldStarted = true;
    } else if (ch === ",") {
      pushField();
    } else if (ch === "\r") {
      // CRLF: the \n that follows closes the record; tolerate lone \r too.
      if (text[i + 1] === "\n") i++;
      pushRecord();
    } else if (ch === "\n") {
      pushRecord();
    } else {
      field += ch;
      fieldStarted = true;
    }
  }
  // Trailing record without a final newline.
  if (record.length > 0 || fieldStarted || field.length > 0) {
    pushField();
    records.push(record);
  }
  return records;
}

export interface ParsedCsv {
  columns: string[];
  rowCount: number;
  previewRows: string[][];
  truncated: boolean;
}

/** Decode CSV text into the preview payload (header + ≤50 data rows). */
export function parseCsvText(text: string): ParsedCsv {
  const records = scanCsvRecords(text);
  if (records.length === 0) {
    return { columns: [], rowCount: 0, previewRows: [], truncated: false };
  }
  const [header, ...dataRows] = records;
  const columns = header.map((c) => (c.length === 0 ? "column" : c));
  // De-duplicate blank/duplicate headers so React keys stay unique.
  const seen = new Map<string, number>();
  const uniqueColumns = columns.map((c) => {
    const n = seen.get(c) ?? 0;
    seen.set(c, n + 1);
    return n === 0 ? c : `${c} (${n + 1})`;
  });
  const previewRows = dataRows.slice(0, CSV_PREVIEW_ROWS);
  return {
    columns: uniqueColumns,
    rowCount: dataRows.length,
    previewRows,
    truncated: dataRows.length > previewRows.length,
  };
}

/**
 * Column is numeric when every previewed value parses as a finite number
 * (blank cells tolerated). Drives right-aligned tabular-nums rendering.
 */
export function isNumericColumn(previewRows: string[][], colIndex: number): boolean {
  let seen = 0;
  for (const row of previewRows) {
    const raw = row[colIndex];
    if (raw === undefined || raw.trim() === "") continue;
    const v = Number(raw);
    if (!Number.isFinite(v)) return false;
    seen++;
  }
  return seen > 0;
}

/* ------------------------------------------------------------------ */
/* PDB decoding                                                        */
/* ------------------------------------------------------------------ */

export interface ParsedPdb {
  atomCount: number;
  residueCount: number;
  totalLineCount: number;
  previewText: string;
}

/** Count lines the honest way (no phantom line after a trailing newline). */
export function countTextLines(text: string): number {
  if (text.length === 0) return 0;
  return text.split("\n").length - (text.endsWith("\n") ? 1 : 0);
}

function firstLines(text: string, maxLines: number): string {
  const lines = text.split("\n");
  const slice = lines.slice(0, maxLines);
  // Drop the phantom trailing element when the cut lands after a newline.
  while (slice.length > 0 && slice[slice.length - 1] === "" && lines.length > maxLines) {
    slice.pop();
  }
  return slice.join("\n");
}

/** Decode PDB text: ATOM/HETATM census, unique residues, 200-line preview. */
export function parsePdbText(text: string): ParsedPdb {
  const lines = text.split("\n");
  let atomCount = 0;
  const residues = new Set<number>();
  for (const line of lines) {
    if (line.startsWith("ATOM") || line.startsWith("HETATM")) {
      atomCount++;
      // Residue sequence number: columns 23–26 (1-based) → chars 22–26.
      const raw = line.slice(22, 26).trim();
      if (raw.length > 0) {
        const num = Number.parseInt(raw, 10);
        if (Number.isFinite(num)) residues.add(num);
      }
    }
  }
  const totalLineCount = countTextLines(text);
  return {
    atomCount,
    residueCount: residues.size,
    totalLineCount,
    previewText: firstLines(text, PDB_PREVIEW_LINES),
  };
}

/* ------------------------------------------------------------------ */
/* Plain-text preview                                                  */
/* ------------------------------------------------------------------ */

export interface ParsedText {
  previewText: string;
  truncated: boolean;
  totalLineCount: number;
}

/** Slice a plain-text file into its ≤400-line preview window. */
export function previewPlainText(text: string): ParsedText {
  const totalLineCount = countTextLines(text);
  const previewText = firstLines(text, TEXT_PREVIEW_LINES);
  return {
    previewText,
    truncated: totalLineCount > TEXT_PREVIEW_LINES,
    totalLineCount,
  };
}
