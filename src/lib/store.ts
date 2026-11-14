/**
 * Client-side app state (single-page tab navigation + toolkit tree cache
 * + selected file in the code explorer). Zustand keeps the selected file
 * and loaded tree alive across tab switches.
 */

import { create } from "zustand";
import { MAX_CSV_TEXT } from "@/lib/csv";
import type {
  TreeNode,
  TreeResponse,
  TreeStats,
  MLTrainResponse,
  MLLearningCurveResponse,
  DatasetPreviewResponse,
  CsvSourceMeta,
  SurgeryResponse,
} from "@/lib/toolkit-types";

export type TabId = "overview" | "explorer" | "reference" | "playground" | "download";

/** Persisted one-compartment PK regimen (survives playground tab switches). */
export interface PkParams {
  route: "iv" | "oral";
  dose: number; // mg
  vd: number; // L
  cl: number; // L/h
  tau: number; // h
  doses: number;
  f: number; // bioavailability (oral)
  ka: number; // 1/h (oral)
}

export const DEFAULT_PK_PARAMS: PkParams = {
  route: "iv",
  dose: 250,
  vd: 45,
  cl: 3.2,
  tau: 8,
  doses: 6,
  f: 0.9,
  ka: 1.2,
};

/* ---- PK/PD Explorer: PD effect layer + NCA observed-data panel ---- */

/** PD effect model driven by the live PK concentration curve. */
export type PdModelId = "linear" | "emax" | "sigmoid" | "inhibitory";

/** Chart mode for the PD panel (mobile users can declutter). */
export type PdView = "conc" | "effect" | "both";

/** PD model parameters (all models' sliders in one persisted object). */
export interface PdParams {
  model: PdModelId;
  view: PdView;
  slope: number; // linear
  e0: number; // baseline effect
  emax: number; // Emax family
  ec50: number; // mg/L
  hill: number; // sigmoid slope factor
  imax: number; // inhibitory
  ic50: number; // mg/L
}

export const DEFAULT_PD_PARAMS: PdParams = {
  model: "emax",
  view: "both",
  slope: 8,
  e0: 10,
  emax: 80,
  ec50: 2,
  hill: 1.5,
  imax: 10,
  ic50: 2,
};

/** One editable NCA observation row (raw input text, parsed client-side). */
export interface NcaRow {
  id: number;
  t: string;
  c: string;
}

/** NCA panel state — observations table + dose inputs + terminal-point count.
 * Persisted (rows included): edits survive sub-tab switches and reloads,
 * capped so the sessionStorage payload stays small. */
export interface NcaState {
  rows: NcaRow[];
  dose: string; // mg (raw text)
  f: string; // bioavailability (raw text)
  terminalN: number; // 2–6 terminal points for the log-linear fit
}

const NCA_ROWS_CAP = 40;

function defaultNcaRows(): NcaRow[] {
  return [
    ["0", "0"],
    ["0.25", "1.28"],
    ["0.5", "2.21"],
    ["1", "3.33"],
    ["2", "4.08"],
    ["4", "3.85"],
    ["6", "3.31"],
    ["8", "2.82"],
    ["12", "2.05"],
    ["24", "0.79"],
    ["36", "0.30"],
    ["48", "0.12"],
  ].map(([t, c], id) => ({ id, t, c }));
}

export const DEFAULT_NCA_STATE: NcaState = {
  rows: defaultNcaRows(),
  dose: "250",
  f: "0.9",
  terminalN: 3,
};

/** Fresh row ids count up from the current maximum (never reused in a session). */
export function nextRowId(rows: NcaRow[]): number {
  return rows.reduce((m, r) => Math.max(m, r.id), -1) + 1;
}

/** Max entries kept in the "recently viewed" strip. */
export const RECENT_LIMIT = 8;

/** sessionStorage key for the recently-viewed strip (survives reloads). */
const RECENT_STORAGE_KEY = "pharmaml.recentFiles";

/** sessionStorage key for playground state (PK regimen + ML setup + sub-tab). */
const PLAYGROUND_STORAGE_KEY = "pharmaml.playground";

/** sessionStorage key for the last ML training result (metrics + chart series). */
const ML_RESULT_STORAGE_KEY = "pharmaml.mlResult";

/** Hard cap on the persisted ML result JSON size (bytes). */
const ML_RESULT_MAX_BYTES = 400_000;

/** sessionStorage key for the last learning-curve run (100x-127 — the
 *  session-report collector walks it together with mlResult). */
const LC_RESULT_STORAGE_KEY = "pharmaml.lcResult";

/** (100x-131) sessionStorage key for the last molecule-surgery run —
 *  the session-report collector walks it so the report gains the
 *  surgery metrics, the SMILES table and the structure-grid figure. */
const SURGERY_RESULT_STORAGE_KEY = "pharmaml.surgeryResult";

/** Hard cap on the persisted surgery result JSON (bytes) — the grid
 *  PNG rides inside as a data URI, so the cap is a whole order larger
 *  than the ML result's; an oversized grid simply does not persist. */
const SURGERY_RESULT_MAX_BYTES = 1_500_000;

/** (R30-a) sessionStorage key for the last ligand-set reconciliation —
 *  the session-report collector walks it so the report gains the
 *  chem-compare bucket counts and name tables.  The reconciler ALSO
 *  keeps its raw inputs under its own key (survives reloads); this
 *  one holds the RESULT only, exactly like surgeryResult. */
const RECONCILE_RESULT_STORAGE_KEY = "pharmaml.reconcileResult";

/** (R30-a) the persisted shape of one ligand-set reconciliation — the
 *  client-side twin of pharmaml.chem.compare_tables' summary plus the
 *  per-bucket name lists the report tables need. */
export interface ReconcileResultState {
  /** Normalisation flags actually used (the report subtitle cites them). */
  stripExtension: boolean;
  lowercase: boolean;
  /** Table labels for the report headers ("A"/"B" when untitled). */
  labelA: string;
  labelB: string;
  counts: { common: number; onlyA: number; onlyB: number };
  /** Bucket name lists (cap: RECONCILE_BUCKET_CAP entries each). */
  common: string[];
  onlyA: string[];
  onlyB: string[];
}

/** Cap on each persisted bucket list and each report-table row count —
 *  reconciliation outputs can run to thousands of ligand names; the
 *  report is a summary, not a data dump (the reconciler UI keeps the
 *  full lists client-side). */
export const RECONCILE_BUCKET_CAP = 50;

/** sessionStorage key for the ML run history (newest first). */
const ML_HISTORY_STORAGE_KEY = "pharmaml.mlHistory";

/** Maximum training runs kept in the history strip. */
export const ML_HISTORY_LIMIT = 5;

/** One completed training run: config snapshot + full result, for restore. */
export interface MlRunRecord {
  /** Epoch-milliseconds id (also the creation time). */
  id: number;
  /** ISO timestamp of when the run finished. */
  at: string;
  task: "regression" | "classification";
  model: string;
  /** Display name of the headline metric, e.g. "R²" or "Accuracy". */
  keyMetric: string;
  /** Value of the headline metric. */
  keyValue: number;
  /** Config snapshot that produced the run. */
  config: MlConfig;
  /** Full training response (restored verbatim). */
  result: MLTrainResponse;
  /** Present for runs trained on an uploaded CSV (display + column restore). */
  csv?: {
    fileName: string;
    featureColumns: string[];
    targetColumn: string;
  };
}

/** The user-uploaded CSV currently loaded in the playground (text + name). */
export interface MlCsvState {
  fileName: string;
  text: string;
}

/** Read the persisted recent-files list (client only, failure-tolerant). */
function loadRecentFiles(): string[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.sessionStorage.getItem(RECENT_STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((p): p is string => typeof p === "string")
      .slice(0, RECENT_LIMIT);
  } catch {
    return []; // private mode / quota / corrupted JSON — start fresh
  }
}

/** Persist the recent-files list (client only, best-effort). */
function saveRecentFiles(recent: string[]): void {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.setItem(RECENT_STORAGE_KEY, JSON.stringify(recent));
  } catch {
    /* quota exceeded or private mode — persistence is optional */
  }
}

/* ---- Playground persistence (PK regimen + ML config + active sub-tab) ----
 * Hydration-safe for the same reason as recentFiles: every component that
 * reads these values renders only inside the Playground tab, which is never
 * the SSR/hydration render (the default top-level tab is Overview). */

interface PlaygroundPersisted {
  playgroundTab: string;
  /** Active inner tab of the PK/PD Explorer ("regimen" | "pd" | "nca"). */
  pkInnerTab: string;
  pkParams: PkParams;
  pdParams: PdParams;
  ncaState: NcaState;
  mlConfig: MlConfig;
  mlCsv: MlCsvState | null;
}

/** Validate a persisted PK regimen field-by-field, falling back to defaults. */
function validatePk(raw: unknown): PkParams {
  const base = DEFAULT_PK_PARAMS;
  if (typeof raw !== "object" || raw === null) return base;
  const p = raw as Partial<PkParams>;
  const num = (v: unknown, def: number, lo: number, hi: number) =>
    typeof v === "number" && Number.isFinite(v) && v >= lo && v <= hi ? v : def;
  return {
    route: p.route === "oral" ? "oral" : "iv",
    dose: num(p.dose, base.dose, 0.1, 10_000),
    vd: num(p.vd, base.vd, 0.1, 1_000),
    cl: num(p.cl, base.cl, 0.01, 100),
    tau: num(p.tau, base.tau, 0.25, 168),
    doses: Math.round(num(p.doses, base.doses, 1, 48)),
    f: num(p.f, base.f, 0.05, 1),
    ka: num(p.ka, base.ka, 0.01, 12),
  };
}

/** Validate a persisted PD parameter object (sliders bound the live values;
 * this only guards corrupted sessionStorage). */
function validatePd(raw: unknown): PdParams {
  const base = DEFAULT_PD_PARAMS;
  if (typeof raw !== "object" || raw === null) return base;
  const p = raw as Partial<PdParams>;
  const num = (v: unknown, def: number, lo: number, hi: number) =>
    typeof v === "number" && Number.isFinite(v) && v >= lo && v <= hi ? v : def;
  return {
    model:
      p.model === "linear" || p.model === "emax" || p.model === "sigmoid" || p.model === "inhibitory"
        ? p.model
        : base.model,
    view: p.view === "conc" || p.view === "effect" || p.view === "both" ? p.view : base.view,
    slope: num(p.slope, base.slope, -100, 100),
    e0: num(p.e0, base.e0, -1000, 1000),
    emax: num(p.emax, base.emax, -1000, 1000),
    ec50: num(p.ec50, base.ec50, 0.01, 1000),
    hill: num(p.hill, base.hill, 0.05, 20),
    imax: num(p.imax, base.imax, -1000, 1000),
    ic50: num(p.ic50, base.ic50, 0.01, 1000),
  };
}

/** Validate a persisted NCA snapshot: rows are raw text, capped + typed. */
function validateNca(raw: unknown): NcaState {
  const base = DEFAULT_NCA_STATE;
  if (typeof raw !== "object" || raw === null) return base;
  const s = raw as Partial<NcaState>;
  const cell = (v: unknown): string | null =>
    typeof v === "string" && v.length <= 16 && /^[\d.\s-]*$/.test(v) ? v : null;
  const rows: NcaRow[] = Array.isArray(s.rows)
    ? s.rows
        .filter(
          (r): r is NcaRow =>
            typeof r === "object" &&
            r !== null &&
            typeof (r as NcaRow).id === "number" &&
            Number.isInteger((r as NcaRow).id) &&
            (r as NcaRow).id >= 0 &&
            (r as NcaRow).id < 100_000 &&
            typeof (r as NcaRow).t === "string" &&
            typeof (r as NcaRow).c === "string",
        )
        .slice(0, NCA_ROWS_CAP)
        .map((r) => ({ id: r.id, t: cell(r.t) ?? "0", c: cell(r.c) ?? "0" }))
    : [];
  const terminalN =
    typeof s.terminalN === "number" && Number.isFinite(s.terminalN) && s.terminalN >= 2 && s.terminalN <= 6
      ? Math.round(s.terminalN)
      : base.terminalN;
  return {
    rows: rows.length >= 2 ? rows : defaultNcaRows(),
    dose: cell(s.dose) ?? base.dose,
    f: cell(s.f) ?? base.f,
    terminalN,
  };
}

/** Validate a persisted ML config field-by-field, falling back to defaults. */
function validateMl(raw: unknown): MlConfig {
  const base = DEFAULT_ML_CONFIG;
  if (typeof raw !== "object" || raw === null) return base;
  const p = raw as Partial<MlConfig>;
  const str = (v: unknown, def: string) =>
    typeof v === "string" && v.length <= 12 ? v : def;
  const str128 = (v: unknown, def: string) =>
    typeof v === "string" && v.length <= 128 ? v : def;
  const strArr = (v: unknown): string[] =>
    Array.isArray(v)
      ? v.filter((c): c is string => typeof c === "string" && c.length > 0 && c.length <= 128).slice(0, 8)
      : [];
  return {
    task: p.task === "classification" ? "classification" : "regression",
    samples:
      typeof p.samples === "number" && Number.isFinite(p.samples) && p.samples >= 100 && p.samples <= 1000
        ? Math.round(p.samples)
        : base.samples,
    noise:
      typeof p.noise === "number" && Number.isFinite(p.noise) && p.noise >= 0 && p.noise <= 2
        ? p.noise
        : base.noise,
    regModel:
      p.regModel === "polynomial" || p.regModel === "ridge" ? p.regModel : "linear",
    alpha: str(p.alpha, base.alpha),
    l2: str(p.l2, base.l2),
    testSizePct:
      typeof p.testSizePct === "number" && Number.isFinite(p.testSizePct) && p.testSizePct >= 5 && p.testSizePct <= 50
        ? p.testSizePct
        : base.testSizePct,
    cv: str(p.cv, base.cv),
    seed: str(p.seed, base.seed),
    source: p.source === "csv" ? "csv" : "synthetic",
    featureColumn: str128(p.featureColumn, ""),
    featureColumn2: str128(p.featureColumn2, ""),
    targetColumn: str128(p.targetColumn, ""),
    featureColumns: strArr(p.featureColumns),
  };
}

/* ---- ML result persistence (last training run: metrics + chart series) ----
 * Restored field-by-field with the same defensive validation as the rest of
 * the persisted state; anything malformed falls back to "no result". */

function finiteArray(raw: unknown, cap: number, validate: (v: unknown) => boolean): unknown[] | null {
  if (!Array.isArray(raw) || raw.length === 0 || raw.length > cap) return null;
  return raw.every(validate) ? raw : null;
}

const isFiniteNumber = (v: unknown): v is number =>
  typeof v === "number" && Number.isFinite(v);

const isPoint = (v: unknown): boolean => {
  if (typeof v !== "object" || v === null) return false;
  const p = v as Record<string, unknown>;
  return isFiniteNumber(p.x) && isFiniteNumber(p.y);
};

const isCurvePoint = (v: unknown): boolean => {
  if (typeof v !== "object" || v === null) return false;
  const p = v as Record<string, unknown>;
  return isFiniteNumber(p.x) && isFiniteNumber(p.yhat);
};

const isBoundaryPoint = (v: unknown): boolean => {
  if (typeof v !== "object" || v === null) return false;
  const p = v as Record<string, unknown>;
  return isFiniteNumber(p.x) && isFiniteNumber(p.y) && isFiniteNumber(p.pred);
};

/** Validate a persisted ML train response; null when unusable. */
function validateMlResult(raw: unknown): MLTrainResponse | null {
  if (typeof raw !== "object" || raw === null) return null;
  const r = raw as Record<string, unknown>;
  if (r.task !== "regression" && r.task !== "classification") return null;
  if (typeof r.model !== "string" || r.model.length > 40) return null;

  const metricsRaw = r.metrics;
  if (typeof metricsRaw !== "object" || metricsRaw === null) return null;
  const metrics: Record<string, number> = {};
  for (const [k, v] of Object.entries(metricsRaw as Record<string, unknown>)) {
    if (isFiniteNumber(v) && k.length <= 24) metrics[k] = v;
  }
  if (Object.keys(metrics).length === 0) return null;

  const coefsRaw = r.coefficients;
  if (typeof coefsRaw !== "object" || coefsRaw === null) return null;
  const c = coefsRaw as Record<string, unknown>;
  const coefsArr = finiteArray(c.coefs, 512, isFiniteNumber);
  if (!isFiniteNumber(c.intercept) || !coefsArr || typeof c.standardized !== "boolean") {
    return null;
  }

  const seriesRaw = r.series;
  if (typeof seriesRaw !== "object" || seriesRaw === null) return null;
  const s = seriesRaw as Record<string, unknown>;
  const test = finiteArray(s.test, 1200, isPoint);
  if (!test) return null;
  const curve = finiteArray(s.curve, 400, isCurvePoint);
  const boundary = finiteArray(s.boundary, 400, isBoundaryPoint);

  const metaRaw = r.datasetMeta;
  if (typeof metaRaw !== "object" || metaRaw === null) return null;
  const m = metaRaw as Record<string, unknown>;
  const features = Array.isArray(m.features)
    ? m.features.filter((f): f is string => typeof f === "string" && f.length <= 24).slice(0, 12)
    : [];
  if (
    !isFiniteNumber(m.nTotal) ||
    !isFiniteNumber(m.nTrain) ||
    !isFiniteNumber(m.nTest) ||
    !isFiniteNumber(m.noise) ||
    !isFiniteNumber(m.seed)
  ) {
    return null;
  }

  // Optional CSV provenance — preserved when well-formed, dropped otherwise.
  let source: CsvSourceMeta | undefined;
  if (typeof m.source === "object" && m.source !== null) {
    const s = m.source as Record<string, unknown>;
    const colList = (v: unknown, cap: number): string[] =>
      Array.isArray(v)
        ? v.filter((c): c is string => typeof c === "string" && c.length <= 128).slice(0, cap)
        : [];
    const columns = colList(s.columns, 64);
    const featureColumns = colList(s.featureColumns, 10);
    const targetColumn = typeof s.targetColumn === "string" ? s.targetColumn.slice(0, 128) : "";
    if (
      s.kind === "csv" &&
      columns.length > 0 &&
      featureColumns.length > 0 &&
      targetColumn !== "" &&
      isFiniteNumber(s.nRows) &&
      isFiniteNumber(s.droppedRows)
    ) {
      const labelMap =
        Array.isArray(s.labelMap) &&
        s.labelMap.length === 2 &&
        s.labelMap.every((v) => typeof v === "string" && v.length <= 64)
          ? (s.labelMap as [string, string])
          : undefined;
      source = {
        kind: "csv",
        fileName: typeof s.fileName === "string" ? s.fileName.slice(0, 128) : undefined,
        columns,
        featureColumns,
        targetColumn,
        nRows: s.nRows,
        droppedRows: s.droppedRows,
        ...(labelMap ? { labelMap } : {}),
      };
    }
  }

  // Optional cross-validation summary — dropped when malformed.
  let cv: MLTrainResponse["cv"];
  if (typeof r.cv === "object" && r.cv !== null) {
    const cvRaw = r.cv as Record<string, unknown>;
    const folds = Array.isArray(cvRaw.folds)
      ? cvRaw.folds.filter(
          (f): f is { fold: number; value: number } =>
            typeof f === "object" &&
            f !== null &&
            isFiniteNumber((f as Record<string, unknown>).fold) &&
            isFiniteNumber((f as Record<string, unknown>).value)
        )
      : [];
    if (
      (cvRaw.k === 3 || cvRaw.k === 5 || cvRaw.k === 10) &&
      typeof cvRaw.metricName === "string" &&
      folds.length === cvRaw.k &&
      isFiniteNumber(cvRaw.mean) &&
      isFiniteNumber(cvRaw.std)
    ) {
      cv = {
        k: cvRaw.k,
        metricName: cvRaw.metricName,
        folds,
        mean: cvRaw.mean,
        std: cvRaw.std,
      };
    }
  }

  return {
    task: r.task,
    model: r.model,
    metrics,
    cv,
    coefficients: {
      intercept: c.intercept as number,
      coefs: coefsArr as number[],
      standardized: c.standardized,
    },
    series: {
      test: test as { x: number; y: number; label?: number }[],
      ...(curve ? { curve: curve as { x: number; yhat: number }[] } : {}),
      ...(boundary ? { boundary: boundary as { x: number; y: number; pred: number }[] } : {}),
    },
    datasetMeta: {
      nTotal: m.nTotal,
      nTrain: m.nTrain,
      nTest: m.nTest,
      noise: m.noise,
      seed: m.seed,
      features,
      ...(source ? { source } : {}),
    },
  };
}

/** Read the persisted ML training result (client only, failure-tolerant). */
function loadMlResult(): MLTrainResponse | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.sessionStorage.getItem(ML_RESULT_STORAGE_KEY);
    if (!raw) return null;
    return validateMlResult(JSON.parse(raw));
  } catch {
    return null; // corrupted JSON / private mode — start fresh
  }
}

/** Validate a persisted run-history payload; malformed entries are skipped.
 * Accepts both the current `{ currentId, runs }` shape and the legacy bare
 * array (current run = newest). */
function validateMlHistoryPayload(raw: unknown): {
  runs: MlRunRecord[];
  currentId: number | null;
} {
  const body = Array.isArray(raw) ? { runs: raw, currentId: undefined } : raw;
  if (typeof body !== "object" || body === null || !Array.isArray((body as { runs?: unknown }).runs)) {
    return { runs: [], currentId: null };
  }
  const runs: MlRunRecord[] = [];
  for (const item of ((body as { runs: unknown[] }).runs).slice(0, ML_HISTORY_LIMIT)) {
    if (typeof item !== "object" || item === null) continue;
    const r = item as Record<string, unknown>;
    const result = validateMlResult(r.result);
    const csvMeta =
      typeof r.csv === "object" && r.csv !== null ? (r.csv as Record<string, unknown>) : null;
    if (
      !result ||
      typeof r.id !== "number" ||
      !Number.isFinite(r.id) ||
      typeof r.at !== "string" ||
      r.at.length > 40 ||
      (r.task !== "regression" && r.task !== "classification") ||
      typeof r.model !== "string" ||
      r.model.length > 40 ||
      typeof r.keyMetric !== "string" ||
      r.keyMetric.length > 16 ||
      typeof r.keyValue !== "number" ||
      !Number.isFinite(r.keyValue)
    ) {
      continue;
    }
    runs.push({
      id: r.id,
      at: r.at,
      task: r.task,
      model: r.model,
      keyMetric: r.keyMetric,
      keyValue: r.keyValue,
      config: validateMl(r.config),
      result,
      ...(csvMeta
        ? {
            csv: {
              fileName:
                typeof csvMeta.fileName === "string"
                  ? csvMeta.fileName.slice(0, 128) || "pasted.csv"
                  : "pasted.csv",
              featureColumns: Array.isArray(csvMeta.featureColumns)
                ? (csvMeta.featureColumns as unknown[])
                    .filter((c): c is string => typeof c === "string" && c.length <= 128)
                    .slice(0, 8)
                : [],
              targetColumn:
                typeof csvMeta.targetColumn === "string"
                  ? csvMeta.targetColumn.slice(0, 128)
                  : "",
            },
          }
        : {}),
    });
  }
  const rawId = (body as { currentId?: unknown }).currentId;
  const currentId =
    typeof rawId === "number" && Number.isFinite(rawId) && runs.some((h) => h.id === rawId)
      ? rawId
      : (runs[0]?.id ?? null);
  return { runs, currentId };
}

/** Read the persisted run history (client only, failure-tolerant). */
function loadMlHistory(): { runs: MlRunRecord[]; currentId: number | null } {
  if (typeof window === "undefined") return { runs: [], currentId: null };
  try {
    const raw = window.sessionStorage.getItem(ML_HISTORY_STORAGE_KEY);
    if (!raw) return { runs: [], currentId: null };
    return validateMlHistoryPayload(JSON.parse(raw));
  } catch {
    return { runs: [], currentId: null }; // corrupted JSON / private mode — start fresh
  }
}

/** Persist the run history + current-run pointer (client only, best-effort). */
function saveMlHistory(history: MlRunRecord[], currentId: number | null): void {
  if (typeof window === "undefined") return;
  try {
    const json = JSON.stringify({ currentId, runs: history });
    if (json.length > 1_200_000) return; // oversized — persistence is optional
    window.sessionStorage.setItem(ML_HISTORY_STORAGE_KEY, json);
  } catch {
    /* quota exceeded or private mode — persistence is optional */
  }
}

/** Persist the ML training result (client only, size-capped, best-effort). */
function saveMlResult(result: MLTrainResponse | null): void {
  if (typeof window === "undefined") return;
  try {
    if (result === null) {
      window.sessionStorage.removeItem(ML_RESULT_STORAGE_KEY);
      return;
    }
    const json = JSON.stringify(result);
    if (json.length > ML_RESULT_MAX_BYTES) return; // oversized — persistence is optional
    window.sessionStorage.setItem(ML_RESULT_STORAGE_KEY, json);
  } catch {
    /* quota exceeded or private mode — persistence is optional */
  }
}

/** Validate a persisted learning-curve payload (100x-127).  The light
 *  structural check mirrors validateMlResult: a wrong shape restores as
 *  "no curve yet" instead of crashing the report collector. */
function validateLcResult(raw: unknown): MLLearningCurveResponse | null {
  if (typeof raw !== "object" || raw === null) return null;
  const r = raw as Record<string, unknown>;
  if (r.task !== "regression" && r.task !== "classification") return null;
  if (typeof r.model !== "string" || r.model.length > 40) return null;
  if (typeof r.metricName !== "string" || r.metricName.length > 24) return null;
  if (typeof r.gap !== "number" || !Number.isFinite(r.gap)) return null;
  if (typeof r.verdict !== "string") return null;
  if (!Array.isArray(r.fractions) || !Array.isArray(r.points)) return null;
  if (r.fractions.length === 0 || r.points.length !== r.fractions.length) return null;
  for (const point of r.points) {
    if (typeof point !== "object" || point === null) return null;
    const p = point as Record<string, unknown>;
    for (const key of ["size", "trainMean", "trainStd", "valMean", "valStd"]) {
      if (typeof p[key] !== "number" || !Number.isFinite(p[key] as number)) return null;
    }
  }
  return raw as MLLearningCurveResponse;
}

/** Read the persisted learning-curve result (client only, failure-tolerant). */
function loadLcResult(): MLLearningCurveResponse | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.sessionStorage.getItem(LC_RESULT_STORAGE_KEY);
    if (!raw) return null;
    return validateLcResult(JSON.parse(raw));
  } catch {
    return null; // corrupted JSON / private mode — start fresh
  }
}

/** Validate a persisted surgery payload (100x-131).  Mode-discriminated
 *  and as light as validateLcResult: a wrong shape restores as "no
 *  surgery run yet" instead of crashing the report collector.  The grid
 *  PNG data URI is checked as a prefixed, length-capped string — full
 *  base64 validation is the bridge's job (it decodes it). */
function validateSurgeryResult(raw: unknown): SurgeryResponse | null {
  if (typeof raw !== "object" || raw === null) return null;
  const r = raw as Record<string, unknown>;
  const pngOk =
    typeof r.gridPng === "string" &&
    r.gridPng.startsWith("data:image/png;base64,") &&
    r.gridPng.length < 2_000_000;
  if (
    typeof r.summary !== "string" ||
    !pngOk ||
    typeof r.elapsedSeconds !== "number" ||
    !Number.isFinite(r.elapsedSeconds) ||
    typeof r.rdkitVersion !== "string" ||
    typeof r.gridPanels !== "number"
  ) {
    return null;
  }
  if (r.mode === "topology") {
    if (typeof r.source !== "string" || r.source.length > 300) return null;
    if (!Array.isArray(r.mutants)) return null;
    if (!r.mutants.every((m) => typeof m === "string" && m.length <= 300)) return null;
    return raw as SurgeryResponse;
  }
  if (r.mode === "breed") {
    if (typeof r.parentA !== "string" || typeof r.parentB !== "string") return null;
    if (r.parentA.length > 300 || r.parentB.length > 300) return null;
    if (typeof r.children !== "number" || typeof r.seed !== "number") return null;
    if (typeof r.viable !== "number") return null;
    if (!Array.isArray(r.litter)) return null;
    if (!r.litter.every((c) => c === null || (typeof c === "string" && c.length <= 300))) {
      return null;
    }
    return raw as SurgeryResponse;
  }
  return null;
}

/** Read the persisted surgery result (client only, failure-tolerant). */
function loadSurgeryResult(): SurgeryResponse | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.sessionStorage.getItem(SURGERY_RESULT_STORAGE_KEY);
    if (!raw) return null;
    return validateSurgeryResult(JSON.parse(raw));
  } catch {
    return null; // corrupted JSON / private mode — start fresh
  }
}

/** (R30-a) Validate a persisted reconciliation payload; a wrong shape
 *  restores as "no reconciliation yet" instead of poisoning the report
 *  collector.  Mirrors validateSurgeryResult's light structural check. */
function validateReconcileResult(raw: unknown): ReconcileResultState | null {
  if (typeof raw !== "object" || raw === null) return null;
  const r = raw as Record<string, unknown>;
  const counts = r.counts as Record<string, unknown> | undefined;
  const bucket = (key: string): string[] | null => {
    if (!Array.isArray(r[key])) return null;
    const list = r[key] as unknown[];
    if (!list.every((name) => typeof name === "string" && name.length <= 300)) return null;
    return (list as string[]).slice(0, RECONCILE_BUCKET_CAP);
  };
  const common = bucket("common");
  const onlyA = bucket("onlyA");
  const onlyB = bucket("onlyB");
  if (
    typeof r.stripExtension !== "boolean" ||
    typeof r.lowercase !== "boolean" ||
    typeof r.labelA !== "string" ||
    r.labelA.length > 60 ||
    typeof r.labelB !== "string" ||
    r.labelB.length > 60 ||
    typeof counts !== "object" ||
    counts === null ||
    typeof counts.common !== "number" ||
    typeof counts.onlyA !== "number" ||
    typeof counts.onlyB !== "number" ||
    !common ||
    !onlyA ||
    !onlyB
  ) {
    return null;
  }
  return {
    stripExtension: r.stripExtension,
    lowercase: r.lowercase,
    labelA: r.labelA,
    labelB: r.labelB,
    counts: { common: counts.common, onlyA: counts.onlyA, onlyB: counts.onlyB },
    common,
    onlyA,
    onlyB,
  };
}

/** (R30-a) Read the persisted reconciliation (client only, failure-tolerant). */
function loadReconcileResult(): ReconcileResultState | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.sessionStorage.getItem(RECONCILE_RESULT_STORAGE_KEY);
    if (!raw) return null;
    return validateReconcileResult(JSON.parse(raw));
  } catch {
    return null; // corrupted JSON / private mode — start fresh
  }
}

/** (R30-a) Persist the reconciliation (client only, size-capped by the
 *  bucket cap, best-effort). */
function saveReconcileResult(result: ReconcileResultState | null): void {
  if (typeof window === "undefined") return;
  try {
    if (result === null) {
      window.sessionStorage.removeItem(RECONCILE_RESULT_STORAGE_KEY);
      return;
    }
    window.sessionStorage.setItem(
      RECONCILE_RESULT_STORAGE_KEY,
      JSON.stringify(result)
    );
  } catch {
    /* quota exceeded or private mode — persistence is optional */
  }
}

/** Persist the surgery result (client only, size-capped, best-effort). */
function saveSurgeryResult(result: SurgeryResponse | null): void {
  if (typeof window === "undefined") return;
  try {
    if (result === null) {
      window.sessionStorage.removeItem(SURGERY_RESULT_STORAGE_KEY);
      return;
    }
    const json = JSON.stringify(result);
    if (json.length > SURGERY_RESULT_MAX_BYTES) return; // oversized — persistence is optional
    window.sessionStorage.setItem(SURGERY_RESULT_STORAGE_KEY, json);
  } catch {
    /* quota exceeded or private mode — persistence is optional */
  }
}

/** Persist the learning-curve result (client only, size-capped, best-effort). */
function saveLcResult(result: MLLearningCurveResponse | null): void {
  if (typeof window === "undefined") return;
  try {
    if (result === null) {
      window.sessionStorage.removeItem(LC_RESULT_STORAGE_KEY);
      return;
    }
    const json = JSON.stringify(result);
    if (json.length > ML_RESULT_MAX_BYTES) return; // oversized — persistence is optional
    window.sessionStorage.setItem(LC_RESULT_STORAGE_KEY, json);
  } catch {
    /* quota exceeded or private mode — persistence is optional */
  }
}

/** Read the persisted playground snapshot (client only, failure-tolerant). */
function loadPlayground(): Partial<PlaygroundPersisted> {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.sessionStorage.getItem(PLAYGROUND_STORAGE_KEY);
    if (!raw) return {};
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== "object" || parsed === null) return {};
    const p = parsed as Partial<PlaygroundPersisted>;
    return {
      playgroundTab: typeof p.playgroundTab === "string" ? p.playgroundTab.slice(0, 32) : undefined,
      pkInnerTab: typeof p.pkInnerTab === "string" ? p.pkInnerTab.slice(0, 16) : undefined,
      pkParams: validatePk(p.pkParams),
      pdParams: validatePd(p.pdParams),
      ncaState: validateNca(p.ncaState),
      mlConfig: validateMl(p.mlConfig),
      mlCsv: validMlCsv(p.mlCsv),
    };
  } catch {
    return {}; // corrupted JSON / private mode — start fresh
  }
}

/** Persist the current playground snapshot (client only, best-effort). */
function savePlayground(state: {
  playgroundTab: string;
  pkInnerTab: string;
  pkParams: PkParams;
  pdParams: PdParams;
  ncaState: NcaState;
  mlConfig: MlConfig;
  mlCsv: MlCsvState | null;
}): void {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.setItem(PLAYGROUND_STORAGE_KEY, JSON.stringify(state));
  } catch {
    /* quota exceeded or private mode — persistence is optional */
  }
}

/** Validate a persisted uploaded-CSV payload; null when unusable. */
function validMlCsv(raw: unknown): MlCsvState | null {
  if (typeof raw !== "object" || raw === null) return null;
  const c = raw as Partial<MlCsvState>;
  if (
    typeof c.text !== "string" ||
    c.text.length === 0 ||
    c.text.length > MAX_CSV_TEXT ||
    typeof c.fileName !== "string" ||
    c.fileName.length > 128
  ) {
    return null;
  }
  return { fileName: c.fileName, text: c.text };
}

/** ML playground experiment configuration (survives tab switches). */
export interface MlConfig {
  task: "regression" | "classification";
  /** Where the training data comes from. */
  source: "synthetic" | "csv";
  samples: number;
  noise: number;
  regModel: "linear" | "polynomial" | "ridge";
  /** Ridge α as raw input text (validated before use). */
  alpha: string;
  /** Logistic L2 strength as raw input text (validated before use). */
  l2: string;
  testSizePct: number;
  cv: string;
  seed: string;
  /** CSV mode: single feature pick for classification ("" = auto). */
  featureColumn: string;
  /** CSV mode: second feature column for classification ("" = auto). */
  featureColumn2: string;
  /** CSV regression mode: feature column picks ([] = auto → all numeric). */
  featureColumns: string[];
  /** CSV mode: target/label column pick ("" = auto). */
  targetColumn: string;
}

export const DEFAULT_ML_CONFIG: MlConfig = {
  task: "regression",
  source: "synthetic",
  samples: 300,
  noise: 0.4,
  regModel: "linear",
  alpha: "1.0",
  l2: "1.0",
  testSizePct: 20,
  cv: "5",
  seed: "42",
  featureColumn: "",
  featureColumn2: "",
  featureColumns: [],
  targetColumn: "",
};

export const TABS: Array<{ id: TabId; label: string }> = [
  { id: "overview", label: "Overview" },
  { id: "explorer", label: "Code Explorer" },
  { id: "reference", label: "API Reference" },
  { id: "playground", label: "Playground" },
  { id: "download", label: "Download" },
];

/* r268 (review M-8): one-line descriptions for the dense icon+short-text
 * tab strip — hover/focus explains what each section holds before the
 * click. */
export const TAB_DESCRIPTIONS: Record<TabId, string> = {
  overview: "Toolkit health at a glance: version, capability map, quick entry points",
  explorer: "Browse the toolkit's source tree with syntax-highlighted files",
  reference: "The generated API reference: modules, classes and functions",
  playground: "Run real toolkit jobs in the browser: PK/PD, NCA, ML, cheminformatics",
  download: "Releases, checksums and the bundled assets",
};

type TreeStatus = "idle" | "loading" | "ready" | "error";

interface ToolkitStore {
  tab: TabId;
  setTab: (tab: TabId) => void;

  /** Command palette open state (Ctrl/Cmd+K or the header button). */
  paletteOpen: boolean;
  setPaletteOpen: (open: boolean) => void;

  /** Currently selected file (relative path) in the code explorer. */
  selectedFile: string;
  setSelectedFile: (path: string) => void;

  /** Definition line to highlight + scroll to (API-reference deep links). */
  targetLine: number | null;
  setTargetLine: (line: number | null) => void;
  /** Open a file and (optionally) jump to a definition line. */
  openFileAtLine: (path: string, line: number | null) => void;

  /** Most-recently-viewed files (newest first, deduplicated). */
  recentFiles: string[];
  /** Clear the recent-files strip. */
  clearRecentFiles: () => void;

  /** Active playground sub-tab (restored when returning to Playground). */
  playgroundTab: string;
  setPlaygroundTab: (tab: string) => void;

  /** (100x-129) Card-finder dialog over the Playground's cards — the
   * web twin of the desktop Ctrl+F finder. Ephemeral UI state (never
   * persisted), opened by the trigger button or the Ctrl/Cmd+Shift+F
   * hotkey from anywhere in the app. */
  cardFinderOpen: boolean;
  setCardFinderOpen: (open: boolean) => void;

  /** Active inner tab of the PK/PD Explorer ("regimen" | "pd" | "nca"). */
  pkInnerTab: string;
  setPkInnerTab: (tab: string) => void;

  /** PK simulator regimen — persisted so sliders survive tab switches. */
  pkParams: PkParams;
  setPkParam: <K extends keyof PkParams>(key: K, value: PkParams[K]) => void;
  resetPkParams: () => void;

  /** PD effect model + parameters — persisted like the PK regimen. */
  pdParams: PdParams;
  setPdParam: <K extends keyof PdParams>(key: K, value: PdParams[K]) => void;
  resetPdParams: () => void;

  /** NCA observed-data panel state (editable rows + dose inputs) — persisted. */
  ncaState: NcaState;
  setNcaState: (patch: Partial<NcaState>) => void;
  resetNcaState: () => void;

  /** ML playground experiment config — persisted so the setup survives tab switches. */
  mlConfig: MlConfig;
  setMlConfig: (patch: Partial<MlConfig>) => void;
  resetMlConfig: () => void;
  /** User-uploaded CSV (file name + full text); null = not loaded. */
  mlCsv: MlCsvState | null;
  setMlCsv: (csv: MlCsvState | null) => void;
  /** Last training result (rendered until the config changes; survives reloads). */
  mlResult: MLTrainResponse | null;
  setMlResult: (result: MLTrainResponse | null) => void;
  /** Last learning-curve run (100x-127 — survives sub-tab switches and
   *  reloads so the session-report collector can walk it). */
  lcResult: MLLearningCurveResponse | null;
  setLcResult: (result: MLLearningCurveResponse | null) => void;
  /** Last molecule-surgery run, topology or breed — the latest run wins
   *  (100x-131 — survives sub-tab switches and reloads so the
   *  session-report collector can walk its grid figure + SMILES table). */
  surgeryResult: SurgeryResponse | null;
  setSurgeryResult: (result: SurgeryResponse | null) => void;

  /** (R30-a) Last ligand-set reconciliation (the Ligand Sets tab's
   *  result) — the session-report collector walks it so the report
   *  gains the chem-compare bucket counts and name tables; null = no
   *  reconciliation yet. */
  reconcileResult: ReconcileResultState | null;
  setReconcileResult: (result: ReconcileResultState | null) => void;
  /** Last dataset-preview payload. */
  mlPreview: DatasetPreviewResponse | null;
  setMlPreview: (preview: DatasetPreviewResponse | null) => void;

  /** Recent training runs (newest first) — persisted so the strip survives reloads. */
  mlHistory: MlRunRecord[];
  pushMlRun: (record: MlRunRecord) => void;
  clearMlHistory: () => void;
  /** Id of the run whose result is currently displayed (drives the badge). */
  mlCurrentRunId: number | null;
  setMlCurrentRun: (id: number | null) => void;

  tree: TreeNode | null;
  stats: TreeStats | null;
  treeStatus: TreeStatus;
  treeError: string | null;
  /** Fetch the toolkit tree once; concurrent calls are collapsed. */
  ensureTree: () => Promise<void>;
}

/** Newest-first, deduplicated, capped recent list (pure helper). */
function pushRecent(recent: string[], path: string): string[] {
  const next = [path, ...recent.filter((p) => p !== path)];
  return next.slice(0, RECENT_LIMIT);
}

/** Session-restored playground snapshot ({} on the server / fresh session). */
const persistedPlayground = loadPlayground();

/** Session-restored ML run history (empty on the server / fresh session). */
const persistedMlHistory = loadMlHistory();

export const useToolkitStore = create<ToolkitStore>((set, get) => {
  /* Snapshot the whole playground slice to sessionStorage. One helper (not a
   * per-setter literal) so a new persisted field can never be silently
   * dropped by a setter that forgets to list it (round 19-b bug class). */
  const persistPlayground = () => {
    const s = get();
    savePlayground({
      playgroundTab: s.playgroundTab,
      pkInnerTab: s.pkInnerTab,
      pkParams: s.pkParams,
      pdParams: s.pdParams,
      ncaState: s.ncaState,
      mlConfig: s.mlConfig,
      mlCsv: s.mlCsv,
    });
  };

  return {
  tab: "overview",
  setTab: (tab) => set({ tab }),

  paletteOpen: false,
  setPaletteOpen: (open) => set({ paletteOpen: open }),

  selectedFile: "README.md",
  setSelectedFile: (path) =>
    set((s) => {
      const recentFiles = pushRecent(s.recentFiles, path);
      saveRecentFiles(recentFiles);
      return { selectedFile: path, targetLine: null, recentFiles };
    }),

  targetLine: null,
  setTargetLine: (line) => set({ targetLine: line }),
  openFileAtLine: (path, line) =>
    set((s) => {
      const recentFiles = pushRecent(s.recentFiles, path);
      saveRecentFiles(recentFiles);
      return { selectedFile: path, targetLine: line, recentFiles };
    }),

  recentFiles: loadRecentFiles(),
  clearRecentFiles: () => {
    saveRecentFiles([]);
    set({ recentFiles: [] });
  },

  playgroundTab: persistedPlayground.playgroundTab ?? "calculators",
  setPlaygroundTab: (tab) => {
    set({ playgroundTab: tab });
    persistPlayground();
  },

  cardFinderOpen: false,
  setCardFinderOpen: (open) => set({ cardFinderOpen: open }),

  pkInnerTab: persistedPlayground.pkInnerTab ?? "regimen",
  setPkInnerTab: (tab) => {
    set({ pkInnerTab: tab });
    persistPlayground();
  },

  pkParams: persistedPlayground.pkParams ?? DEFAULT_PK_PARAMS,
  setPkParam: (key, value) => {
    set({ pkParams: { ...get().pkParams, [key]: value } });
    persistPlayground();
  },
  resetPkParams: () => {
    set({ pkParams: DEFAULT_PK_PARAMS });
    persistPlayground();
  },

  pdParams: persistedPlayground.pdParams ?? DEFAULT_PD_PARAMS,
  setPdParam: (key, value) => {
    set({ pdParams: { ...get().pdParams, [key]: value } });
    persistPlayground();
  },
  resetPdParams: () => {
    set({ pdParams: DEFAULT_PD_PARAMS });
    persistPlayground();
  },

  ncaState: persistedPlayground.ncaState ?? DEFAULT_NCA_STATE,
  setNcaState: (patch) => {
    set({ ncaState: { ...get().ncaState, ...patch } });
    persistPlayground();
  },
  resetNcaState: () => {
    set({ ncaState: DEFAULT_NCA_STATE });
    persistPlayground();
  },

  mlConfig: persistedPlayground.mlConfig ?? DEFAULT_ML_CONFIG,
  setMlConfig: (patch) => {
    set({ mlConfig: { ...get().mlConfig, ...patch } });
    persistPlayground();
  },
  resetMlConfig: () => {
    set({ mlConfig: DEFAULT_ML_CONFIG });
    persistPlayground();
  },

  mlCsv: persistedPlayground.mlCsv ?? null,
  setMlCsv: (csv) => {
    set({ mlCsv: csv });
    persistPlayground();
  },
  mlResult: loadMlResult(),
  setMlResult: (result) => {
    set({ mlResult: result });
    saveMlResult(result);
  },
  lcResult: loadLcResult(),
  setLcResult: (result) => {
    set({ lcResult: result });
    saveLcResult(result);
  },
  surgeryResult: loadSurgeryResult(),
  setSurgeryResult: (result) => {
    set({ surgeryResult: result });
    saveSurgeryResult(result);
  },
  reconcileResult: loadReconcileResult(),
  setReconcileResult: (result) => {
    set({ reconcileResult: result });
    saveReconcileResult(result);
  },
  mlPreview: null,
  setMlPreview: (preview) => set({ mlPreview: preview }),

  mlHistory: persistedMlHistory.runs,
  pushMlRun: (record) => {
    const mlHistory = [
      record,
      ...get().mlHistory.filter((h) => h.id !== record.id),
    ].slice(0, ML_HISTORY_LIMIT);
    set({ mlHistory, mlCurrentRunId: record.id });
    saveMlHistory(mlHistory, record.id);
  },
  clearMlHistory: () => {
    saveMlHistory([], null);
    set({ mlHistory: [], mlCurrentRunId: null });
  },
  mlCurrentRunId: persistedMlHistory.currentId,
  setMlCurrentRun: (id) => {
    set({ mlCurrentRunId: id });
    saveMlHistory(get().mlHistory, id);
  },

  tree: null,
  stats: null,
  treeStatus: "idle",
  treeError: null,
  ensureTree: async () => {
    const { treeStatus, tree } = get();
    if (treeStatus === "loading" || tree) return;
    set({ treeStatus: "loading", treeError: null });
    try {
      const res = await fetch("/api/toolkit/tree");
      if (!res.ok) {
        throw new Error(`Tree API responded with HTTP ${res.status}`);
      }
      const data = (await res.json()) as TreeResponse;
      set({ tree: data.root, stats: data.stats, treeStatus: "ready" });
    } catch (err) {
      set({
        treeStatus: "error",
        treeError:
          err instanceof Error ? err.message : "Unknown error while loading the file tree.",
      });
    }
  },
  };
});
