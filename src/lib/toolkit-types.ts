/**
 * Shared types for the PharmaML Toolkit explorer web app.
 * Used by both client components and API routes.
 */

/* ------------------------------------------------------------------ */
/* Toolkit file tree                                                   */
/* ------------------------------------------------------------------ */

export interface TreeNode {
  /** Base name of the file or directory. */
  name: string;
  /** POSIX-style path relative to the toolkit root ("" for the root itself). */
  path: string;
  type: "dir" | "file";
  /** Size in bytes (files) or aggregate size (dirs). */
  size: number;
  /** Line count for text files. */
  lines?: number;
  /** Grammar id for syntax highlighting (files only). */
  language?: string;
  children?: TreeNode[];
}

export interface TreeStats {
  files: number;
  dirs: number;
  totalLines: number;
  totalBytes: number;
}

export interface TreeResponse {
  root: TreeNode;
  stats: TreeStats;
  generatedAt: string;
}

export interface FileContentResponse {
  path: string;
  content: string;
  lines: number;
  size: number;
  language: string;
}

/* ------------------------------------------------------------------ */
/* API reference (parsed from docs/INDEX.md)                           */
/* ------------------------------------------------------------------ */

export type SymbolKind = "function" | "class" | "module" | "constant";

export interface ReferenceSymbol {
  name: string;
  kind: SymbolKind;
  /** First line of the docstring (may be empty for constants). */
  summary: string;
  /** 1-based definition line in the module's source file (null if unresolved). */
  line?: number | null;
  /** Source file override when the symbol lives in a re-exported sub-module. */
  sourcePath?: string;
}

export interface ReferenceModule {
  /** Dotted module name, e.g. `pharmaml.pharma`. */
  name: string;
  /** One-line blurb from INDEX.md. */
  blurb: string;
  /** Toolkit-relative path of the module's source file (for view-source). */
  sourcePath: string;
  symbols: ReferenceSymbol[];
}

export interface ReferenceResponse {
  modules: ReferenceModule[];
  stats: {
    modules: number;
    symbols: number;
  };
  generatedAt: string;
}

/* ------------------------------------------------------------------ */
/* ML training playground                                              */
/* ------------------------------------------------------------------ */

export type MLTask = "regression" | "classification";

export interface MLTrainRequest {
  task: MLTask;
  samples: number;
  noise: number;
  model: string;
  /** Ridge alpha (regression) — omitted for other models. */
  alpha?: number;
  /** L2 strength (logistic classification). */
  l2?: number;
  testSize: number;
  cv: 0 | 3 | 5 | 10;
  seed: number;
}

export interface CVFold {
  fold: number;
  value: number;
}

export interface CVSummary {
  k: number;
  metricName: string;
  folds: CVFold[];
  mean: number;
  /** Population standard deviation. */
  std: number;
}

export interface MLTrainResponse {
  task: MLTask;
  model: string;
  metrics: Record<string, number>;
  cv?: CVSummary;
  coefficients: {
    intercept: number;
    coefs: number[];
    /** True when coefficients live in standardized feature space. */
    standardized: boolean;
  };
  series: {
    test: Array<{ x: number; y: number; label?: number }>;
    curve?: Array<{ x: number; yhat: number }>;
    boundary?: Array<{ x: number; y: number; pred: number }>;
  };
  datasetMeta: {
    nTotal: number;
    nTrain: number;
    nTest: number;
    noise: number;
    seed: number;
    features: string[];
  };
}

/* ------------------------------------------------------------------ */
/* ML dataset preview                                                   */
/* ------------------------------------------------------------------ */

export interface DatasetPreviewRequest {
  task: MLTask;
  samples: number;
  noise: number;
  seed: number;
}

export interface DatasetColumnStats {
  column: string;
  mean: number;
  std: number;
  min: number;
  max: number;
}

export interface DatasetPreviewResponse {
  task: MLTask;
  columns: string[];
  /** First rows of the dataset (numbers; label columns are 0/1). */
  rows: Array<Record<string, number>>;
  nTotal: number;
  stats: DatasetColumnStats[];
  /** Regression only: Pearson correlation between x and y. */
  correlation?: number;
  /** Classification only: class balance. */
  classBalance?: Array<{ label: number; count: number; pct: number }>;
  noise: number;
  seed: number;
}

/* ------------------------------------------------------------------ */
/* Docking result parser                                               */
/* ------------------------------------------------------------------ */

export type DockingFormat = "vina" | "dlg" | "csv";

export interface DockingPose {
  /** 1-based rank (best first). */
  rank: number;
  /** Mode / pose number (Vina) or cluster rank (DLG) or pose id (CSV). */
  mode?: number;
  /** Ligand name (CSV format). */
  ligand?: string;
  /** Best score / affinity in kcal/mol (more negative = better). */
  affinity: number;
  rmsdLb?: number;
  rmsdUb?: number;
  run?: number;
  meanEnergy?: number;
  clusterSize?: number;
  /** kcal/mol per heavy atom. */
  ligandEfficiency?: number;
  /** Consensus z-score (CSV with ≥2 score columns only). */
  zScore?: number;
  consensusRank?: number;
  /** Extra numeric score columns (CSV). */
  extraScores?: Record<string, number>;
}

export interface DockingParseRequest {
  format: DockingFormat;
  content: string;
  heavyAtoms: number;
}

export interface DockingChartDatum {
  mode: string;
  affinity: number;
  best: boolean;
}

export interface DockingParseResponse {
  format: DockingFormat;
  poses: DockingPose[];
  stats: {
    best: number;
    worst: number;
    mean: number;
    n: number;
  };
  chartData: DockingChartDatum[];
  consensus?: {
    applied: boolean;
    columns: string[];
    note: string;
  };
}
