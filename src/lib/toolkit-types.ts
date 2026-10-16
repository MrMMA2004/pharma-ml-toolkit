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
/* Data-file previews (CSV / PDB / plain text)                          */
/* ------------------------------------------------------------------ */

/**
 * Union of every payload `/api/toolkit/file` can return. Source files keep
 * the classic `FileContentResponse` shape; shipped data files (.csv, .pdb,
 * .dlg, .glg, .gpf, .dpf, .txt) return a `kind`-tagged preview instead.
 * Discriminate with `"kind" in payload`.
 */
export type AnyFileResponse = FileContentResponse | DataFileResponse;

/** Preview of a delimited data table (shipped example datasets). */
export interface CsvPreviewResponse {
  kind: "csv";
  path: string;
  /** Header row (RFC4180-aware, quotes stripped). */
  columns: string[];
  /** Exact number of data rows (header excluded). */
  rowCount: number;
  /** First ≤50 data rows, each aligned with `columns`. */
  previewRows: string[][];
  /** True when `rowCount > previewRows.length`. */
  truncated: boolean;
  totalBytes: number;
}

/** Preview of a PDB structure file. */
export interface PdbPreviewResponse {
  kind: "pdb";
  path: string;
  totalBytes: number;
  /** ATOM + HETATM records in the file. */
  atomCount: number;
  /** Unique residue sequence numbers across ATOM/HETATM records. */
  residueCount: number;
  /** Every line in the file (records + remarks + connectors). */
  totalLineCount: number;
  /** First ≤200 lines, verbatim. */
  previewText: string;
}

/** Preview of a plain-text data file (docking logs, configs, notes). */
export interface TextPreviewResponse {
  kind: "text";
  path: string;
  /** First ≤400 lines, verbatim. */
  previewText: string;
  truncated: boolean;
  totalBytes: number;
  /** Exact line count of the whole file. */
  totalLineCount: number;
}

export type DataFileResponse = CsvPreviewResponse | PdbPreviewResponse | TextPreviewResponse;

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

/** Provenance + column picks for a user-uploaded CSV dataset. */
export interface CsvSourceMeta {
  kind: "csv";
  /** File name as reported by the client ("pasted.csv" for pasted text). */
  fileName?: string;
  /** Numeric columns detected in the uploaded file. */
  columns: string[];
  /** Columns used as model input (1 for regression, 2 for classification). */
  featureColumns: string[];
  /** Column used as the prediction target. */
  targetColumn: string;
  /** Usable rows after dropping malformed ones. */
  nRows: number;
  /** Rows skipped because a used cell was blank/non-numeric. */
  droppedRows: number;
  /** Classification only: raw target values mapped to class 0 / class 1. */
  labelMap?: [string, string];
}

export interface MLTrainRequest {
  task: MLTask;
  samples: number;
  noise: number;
  model: string;
  /** Ridge/lasso/elastic-net strength — omitted for other models. */
  alpha?: number;
  /** Elastic-net L1 mix in [0, 1] (1 = lasso, 0 = ridge). */
  l1Ratio?: number;
  /** L2 strength (logistic classification). */
  l2?: number;
  /** kNN neighbours (odd, 1–25). */
  k?: number;
  /** CART depth cap (1–12). */
  maxDepth?: number;
  /** Forest size (3–200 trees). */
  nEstimators?: number;
  /** Boosting rounds (5–200). */
  rounds?: number;
  /** Boosting shrinkage (0.005–1). */
  learningRate?: number;
  /** MLP hidden-layer width (2–64). */
  hidden?: number;
  /** MLP training epochs (10–2000). */
  epochs?: number;
  /** MLP learning rate (0.001–1). */
  lr?: number;
  /** Linear SVM margin C (0.01–100). */
  C?: number;
  testSize: number;
  cv: 0 | 3 | 5 | 10;
  seed: number;
  /** Raw CSV text — when present, training uses the uploaded data and
   *  samples/noise/seed are ignored. */
  csv?: string;
  /** Client-side file name, echoed back in CsvSourceMeta. */
  csvName?: string;
  /** Regression only: multiple feature column picks (≥2 = multivariate). */
  featureColumns?: string[];
  /** Single feature column pick; omitted/"" means auto-resolve. */
  featureColumn?: string;
  /** Classification only: second feature column. */
  featureColumn2?: string;
  /** Target/label column. */
  targetColumn?: string;
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

/** Normalized impurity-reduction feature importance (trees & ensembles).
 *  Values are non-negative and sum to 1 — the honest parameter summary
 *  for models that have no closed-form coefficients. */
export interface FeatureImportance {
  /** Feature names (raw column names, no standardization suffix). */
  features: string[];
  /** Normalized importances, same order as `features`. */
  importance: number[];
  /** Impurity measure the reductions were computed with. */
  measure: "variance" | "gini";
}

export interface MLTrainResponse {
  task: MLTask;
  model: string;
  metrics: Record<string, number>;
  cv?: CVSummary;
  /** Present ONLY for models with real closed-form coefficients
   *  (linear / polynomial / ridge / logistic) — never fabricated for
   *  tree, instance-based, probabilistic or neural models. */
  coefficients?: {
    intercept: number;
    coefs: number[];
    /** True when coefficients live in standardized feature space. */
    standardized: boolean;
  };
  /** Trees & ensembles: normalized impurity-reduction importances. */
  featureImportance?: FeatureImportance;
  /** kNN / MLP / Naive Bayes / SVM: echoed hyperparameters (the honest
   *  "learned parameters" card content for hyperparameter-driven models). */
  modelParams?: Record<string, number>;
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
    /** Present when the run trained on a user-uploaded CSV. */
    source?: CsvSourceMeta;
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
  /** Raw CSV text — when present, the preview shows the uploaded data. */
  csv?: string;
  csvName?: string;
  /** Regression only: multiple feature column picks (≥2 = multivariate). */
  featureColumns?: string[];
  featureColumn?: string;
  featureColumn2?: string;
  targetColumn?: string;
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
  /** Evenly strided sample of every point for the overview scatter (≤ 300). */
  points: Array<{ x: number; y: number; label?: number }>;
  /** Number of points sampled into `points`. */
  nPoints: number;
  /** Regression only: the noiseless generating line. */
  trend?: Array<{ x: number; y: number }>;
  /** Regression only: Pearson correlation between x and y. */
  correlation?: number;
  /** Classification only: class balance. */
  classBalance?: Array<{ label: number; count: number; pct: number }>;
  noise: number;
  seed: number;
  /** Present when the preview shows a user-uploaded CSV. */
  source?: CsvSourceMeta;
}

/* ------------------------------------------------------------------ */
/* Cell line biology / biomarker discovery                              */
/* ------------------------------------------------------------------ */

/** The 8 screened compounds (mirrors examples/make_synthetic_cellline_data.py). */
export type BiologyCompound =
  | "erlotinib"
  | "gefitinib"
  | "cisplatin"
  | "doxorubicin"
  | "paclitaxel"
  | "5-fluorouracil"
  | "gemcitabine"
  | "sorafenib";

export interface BiomarkersRequest {
  compound: BiologyCompound;
  /** Number of genes in the expression matrix (60–600). */
  genes: number;
  /** Screen-generation seed (expression matrix uses seed + 1). */
  seed: number;
  /** Panel size restriction (6–24 canonical cell lines). */
  nCellLines?: number;
}

/** One row of the ranked biomarker table. */
export interface BiomarkerRow {
  /** 1-based rank by |rho| descending. */
  rank: number;
  gene: string;
  /** Spearman rho vs the compound's LN_IC50 (negative = sensitizing). */
  rho: number;
  /** Two-sided p-value, normal approximation with tie correction. */
  pValue: number;
  /** Benjamini–Hochberg adjusted p-value across all screened genes. */
  fdr: number;
  /** FDR < 0.05. */
  significant: boolean;
}

/** One cell line of a biomarker scatter payload. */
export interface BiomarkerPoint {
  cellLine: string;
  tissue: string;
  /** Expression of the plotted gene (raw log2 value). */
  value: number;
  /** Response of the selected compound (natural-log IC50, µM). */
  lnIc50: number;
  /** Area under the dose–response curve (0–1, tracks LN_IC50). */
  auc: number;
}

/** Scatter payload for one gene (the top hit, or EGFR specifically). */
export interface BiomarkerScatter {
  gene: string;
  rho: number;
  significant: boolean;
  points: BiomarkerPoint[];
  /** Rank-space fitted trend line, mapped back to value space. */
  trend: Array<{ x: number; y: number }>;
}

export interface BiomarkersResponse {
  /** Top 15 genes by |rho|. */
  biomarkers: BiomarkerRow[];
  /** Full row for EGFR when present in the matrix (always is). */
  egfr: BiomarkerRow | null;
  /** Scatter payload of the #1 ranked gene. */
  topHit: BiomarkerScatter;
  /** Scatter payload for EGFR (same points, EGFR expression values). */
  egfrScatter: BiomarkerScatter | null;
  /** Distinct tissues covered by the panel (chart legend order). */
  tissues: string[];
  meta: {
    nGenes: number;
    nCellLines: number;
    compound: BiologyCompound;
    /** "EGFR" when the planted biomarker ranks #1 and passes FDR < 0.05. */
    plantedHit: "EGFR" | null;
    /** Number of genes significant at FDR < 0.05. */
    nSignificant: number;
    /** Strongest |rho| across the whole screen. */
    strongestRho: number;
    seed: number;
    runtimeMs: number;
  };
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
