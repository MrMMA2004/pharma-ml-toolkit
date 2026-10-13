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
  DatasetPreviewResponse,
  CsvSourceMeta,
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
    featureColumn: string;
    featureColumn2: string;
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
  pkParams: PkParams;
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

/** Validate a persisted ML config field-by-field, falling back to defaults. */
function validateMl(raw: unknown): MlConfig {
  const base = DEFAULT_ML_CONFIG;
  if (typeof raw !== "object" || raw === null) return base;
  const p = raw as Partial<MlConfig>;
  const str = (v: unknown, def: string) =>
    typeof v === "string" && v.length <= 12 ? v : def;
  const str128 = (v: unknown, def: string) =>
    typeof v === "string" && v.length <= 128 ? v : def;
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
    const featureColumns = colList(s.featureColumns, 4);
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
      ...(typeof r.csv === "object" && r.csv !== null
        ? {
            csv: {
              fileName:
                typeof (r.csv as Record<string, unknown>).fileName === "string"
                  ? (r.csv as Record<string, unknown>).fileName.slice(0, 128) || "pasted.csv"
                  : "pasted.csv",
              featureColumn:
                typeof (r.csv as Record<string, unknown>).featureColumn === "string"
                  ? (r.csv as Record<string, unknown>).featureColumn.slice(0, 128)
                  : "",
              featureColumn2:
                typeof (r.csv as Record<string, unknown>).featureColumn2 === "string"
                  ? (r.csv as Record<string, unknown>).featureColumn2.slice(0, 128)
                  : "",
              targetColumn:
                typeof (r.csv as Record<string, unknown>).targetColumn === "string"
                  ? (r.csv as Record<string, unknown>).targetColumn.slice(0, 128)
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
      pkParams: validatePk(p.pkParams),
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
  pkParams: PkParams;
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
  /** CSV mode: feature column pick ("" = auto → first numeric). */
  featureColumn: string;
  /** CSV mode: second feature column for classification ("" = auto). */
  featureColumn2: string;
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
  targetColumn: "",
};

export const TABS: Array<{ id: TabId; label: string }> = [
  { id: "overview", label: "Overview" },
  { id: "explorer", label: "Code Explorer" },
  { id: "reference", label: "API Reference" },
  { id: "playground", label: "Playground" },
  { id: "download", label: "Download" },
];

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

  /** PK simulator regimen — persisted so sliders survive tab switches. */
  pkParams: PkParams;
  setPkParam: <K extends keyof PkParams>(key: K, value: PkParams[K]) => void;
  resetPkParams: () => void;

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

export const useToolkitStore = create<ToolkitStore>((set, get) => ({
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
    savePlayground({
      playgroundTab: tab,
      pkParams: get().pkParams,
      mlConfig: get().mlConfig,
      mlCsv: get().mlCsv,
    });
  },

  pkParams: persistedPlayground.pkParams ?? DEFAULT_PK_PARAMS,
  setPkParam: (key, value) => {
    const pkParams = { ...get().pkParams, [key]: value };
    set({ pkParams });
    savePlayground({
      playgroundTab: get().playgroundTab,
      pkParams,
      mlConfig: get().mlConfig,
    });
  },
  resetPkParams: () => {
    set({ pkParams: DEFAULT_PK_PARAMS });
    savePlayground({
      playgroundTab: get().playgroundTab,
      pkParams: DEFAULT_PK_PARAMS,
      mlConfig: get().mlConfig,
      mlCsv: get().mlCsv,
    });
  },

  mlConfig: persistedPlayground.mlConfig ?? DEFAULT_ML_CONFIG,
  setMlConfig: (patch) => {
    const mlConfig = { ...get().mlConfig, ...patch };
    set({ mlConfig });
    savePlayground({
      playgroundTab: get().playgroundTab,
      pkParams: get().pkParams,
      mlConfig,
      mlCsv: get().mlCsv,
    });
  },
  resetMlConfig: () => {
    set({ mlConfig: DEFAULT_ML_CONFIG });
    savePlayground({
      playgroundTab: get().playgroundTab,
      pkParams: get().pkParams,
      mlConfig: DEFAULT_ML_CONFIG,
      mlCsv: get().mlCsv,
    });
  },

  mlCsv: persistedPlayground.mlCsv ?? null,
  setMlCsv: (csv) => {
    set({ mlCsv: csv });
    savePlayground({
      playgroundTab: get().playgroundTab,
      pkParams: get().pkParams,
      mlConfig: get().mlConfig,
      mlCsv: csv,
    });
  },
  mlResult: loadMlResult(),
  setMlResult: (result) => {
    set({ mlResult: result });
    saveMlResult(result);
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
}));
