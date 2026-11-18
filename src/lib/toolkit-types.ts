/**
 * Shared types for the PharmaML Toolkit explorer web app.
 * Used by both client components and API routes.
 */

import type { AdRow, AdSummary } from "@/lib/ml-ad";

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
  /** Toolkit release version from pyproject.toml ([project] version). */
  version: string;
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
  /** HSVR k-means cluster count (2–8). */
  nClusters?: number;
  /** HSVR trust radius factor (0.25–6): R = factor × the cluster's max
   *  member distance from its centroid. */
  radiusFactor?: number;
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

/** One signed feature-target correlation (the web twin of the desktop
 *  interpretability bundle's signed_feature_correlations). */
export interface SignedCorrelation {
  /** Raw column name. */
  name: string;
  /** Pearson r against the target, sign preserved. */
  r: number;
}

/** The interpretability bundle a regression train response carries
 *  (100x-153): signed feature-target correlations plus the compact
 *  top-N correlation matrix, BOTH computed over the run's OWN held-out
 *  split in the same request — inherently swap-proof (a re-loaded
 *  dataset can never re-pair its target with an old run's features,
 *  the desktop round-36 audit's red finding). Undefined correlations
 *  (a constant column, or fewer than two held-out rows) are dropped,
 *  never zeroed; matrix cells that are undefined serialize as null. */
export interface CorrelationBundle {
  /** Always "pearson" (the honest single method this twin ships). */
  method: "pearson";
  /** The target column name. */
  target: string;
  /** Held-out row count the correlations were computed over. */
  nRows: number;
  /** |r| descending, sign preserved; empty when nothing was defined. */
  correlations: SignedCorrelation[];
  /** The top-N (≤ 7) features + the target pairwise matrix, or null
   *  when fewer than two columns were defined. */
  matrix: {
    columns: string[];
    /** Symmetric row-major values; diagonal 1; null where undefined. */
    values: (number | null)[][];
  } | null;
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
  /** Regression only: the interpretability bundle (signed
   *  feature-target correlations + the top-N matrix over the run's
   *  own held-out rows — the desktop bundle's web twin, 100x-153). */
  correlations?: CorrelationBundle;
  /** kNN / MLP / Naive Bayes / SVM: echoed hyperparameters (the honest
   *  "learned parameters" card content for hyperparameter-driven models). */
  modelParams?: Record<string, number>;
  /** PLS only: VIP scores — VIP > 1 marks an influential descriptor
   *  (the classic QSAR readout the NIPALS fit always computed and the
   *  response previously dropped). */
  vip?: {
    features: string[];
    /** VIP score per feature, same order as `features`. */
    values: number[];
    /** The influence threshold (1 — VIP scores mean-square to 1). */
    threshold: number;
  };
  /** GP only: the kernel readout plus the honest held-out uncertainty
   *  stats — coverage of the ±1.96σ band and the median σ. */
  gp?: {
    /** RBF length scale ℓ (standardized-feature units). */
    lengthScale: number;
    /** Signal amplitude σf (raw target units). */
    signalStd: number;
    /** White-noise observation std σn (raw target units). */
    noiseStd: number;
    /** Log marginal likelihood of the train targets under the kernel. */
    logMarginalLikelihood: number;
    /** Train rows in the GP's support set (all of them). */
    nSupport: number;
    /** True when the kernel hyperparameters were learned by the
     *  deterministic LML grid search (request left them unset). */
    kernelTuned: boolean;
    /** Share of test points inside ŷ ± 1.96σ (a calibrated GP ≈ 0.95). */
    coverage95: number;
    /** Median test-set predictive σ (raw target units). */
    medianSigma: number;
  };
  /** GP classification only: the Laplace posterior readout plus the
   *  honest held-out probabilistic scores (ROC AUC / log-loss / Brier
   *  from the 5-erf predictive probabilities). */
  gpCls?: {
    /** RBF length scale ℓ (standardized-feature units). */
    lengthScale: number;
    /** Latent-kernel amplitude σf. */
    signalStd: number;
    /** Newton iterations spent finding the posterior mode. */
    newtonIterations: number;
    /** Laplace log marginal likelihood at the mode. */
    logMarginalLikelihood: number;
    /** True when (ℓ, σf) were learned by the Laplace-lml grid search. */
    kernelTuned: boolean;
    /** Test-set ROC AUC over the predictive probabilities. */
    rocAuc: number;
    /** Test-set log-loss (natural log, clamped at 1e-12). */
    logLoss: number;
    /** Test-set Brier score (mean squared probability error). */
    brier: number;
  };
  /** HSVR only: per-cluster geometry + local-fit diagnostics. */
  hsvr?: {
    nClusters: number;
    radiusFactor: number;
    alpha: number;
    clusters: Array<{
      /** Train rows assigned to the cluster. */
      size: number;
      /** Trust radius R (standardized-space distance units). */
      radius: number;
      /** In-sample R² of the cluster-local ridge fit (constant → 1). */
      localR2: number;
      /** True when the cluster kept too few rows for ridge and fell back
       *  to the cluster mean. */
      constant: boolean;
    }>;
    /** The train-row geometry for the trust-radius scatter. dimA is the
     * first standardized feature; dimB is the target when the model has a
     * single feature (the radius renders as a horizontal band) or the
     * second standardized feature (the radius renders as a circle). */
    scatter?: {
      dimA: string;
      dimB: string;
      /** True when dimB is the (raw) target axis rather than a feature. */
      bIsTarget: boolean;
      /** True when the model has >2 features — the circle is a projection
       *  of the full-space trust sphere onto the first two dims. */
      projected: boolean;
      /** One point per train row: plot coords + cluster assignment. */
      points: Array<{ a: number; b: number; cluster: number }>;
      /** Cluster centroids in plot coords + their trust radii (the
       * b-coordinate of a centroid is its cluster's mean target when
       * bIsTarget, else the second standardized feature). */
      centroids: Array<{ a: number; b: number; radius: number }>;
    };
  };
  /** Voting / stacking ensembles only: the honest meta-model readout —
   *  members, the blending rule, the level-0 OOF scores and the
   *  meta-learner's weights (stacking), and every member's held-out
   *  score next to the ensemble's own — the "did the ensemble actually
   *  help?" answer in one object. */
  ensemble?: {
    kind: "voting" | "stacking";
    /** How members combine: weighted mean (regression) or averaged
     *  probability (classification — sklearn's soft voting). */
    rule: "average" | "soft";
    /** Registry ids of the level-0 members, blend order. */
    members: string[];
    /** Blend weight per member. */
    weights: number[];
    /** Stacking only: the level-1 learner (ridge / logistic). */
    metaModel?: "ridge" | "logistic";
    /** Stacking only: folds that built the OOF training matrix. */
    oofFolds?: number;
    /** Stacking only: cross-validated score per level-0 member on the
     *  OOF predictions the meta-learner was trained against. */
    oof?: { name: string; score: number }[];
    /** Stacking only: the meta-learner's weights on the (standardized)
     *  OOF columns — which base model the blender actually trusts. */
    metaCoefs?: { intercept: number; coefs: number[] };
    /** Held-out (test) score per level-0 member plus the ensemble
     *  itself (last row, name "ensemble") — R² for regression,
     *  accuracy for classification. */
    memberScores: { name: string; score: number }[];
    /** Soft-rule classification ensembles: the honest held-out
     *  probabilistic scores from the blended probabilities (the same
     *  readout the GP classification card shows). */
    rocAuc?: number;
    logLoss?: number;
    brier?: number;
  };
  series: {
    test: Array<{ x: number; y: number; label?: number; err?: number }>;
    /** Training points (single-feature regression): where the model
     *  actually observed data — the GP band chart draws them and the
     *  acquisition overlay derives its best-observed y from them. */
    train?: Array<{ x: number; y: number }>;
    curve?: Array<{ x: number; yhat: number; lo?: number; hi?: number }>;
    boundary?: Array<{ x: number; y: number; pred: number; proba?: number }>;
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
/* ML hyperparameter search (RandomSearch)                              */
/* ------------------------------------------------------------------ */

export interface MLTuneRequest {
  task: MLTask;
  model: string;
  /** RandomSearch trial budget (4–40, clamped to the space size);
   *  ignored by the Hyperband schedule (brackets fix the trial count). */
  maxTrials: number;
  /** Which searcher to run — random (default) or hyperband. */
  searcher?: "random" | "hyperband";
  /** Hyperband only: the halving/promotion factor η (2 / 3 / 4). */
  eta?: 2 | 3 | 4;
  /** CV folds used to score each trial (3 / 5 / 10 — never off). */
  cv: 3 | 5 | 10;
  seed: number;
  testSize: number;
  /** Synthetic-generator knobs (ignored when csv is present). */
  samples: number;
  noise: number;
  /** Raw CSV text — when present, tuning uses the uploaded data. */
  csv?: string;
  csvName?: string;
  /** Regression only: feature/target column picks (same semantics as train). */
  featureColumns?: string[];
  featureColumn?: string;
  featureColumn2?: string;
  targetColumn?: string;
}

/** One tuner trial — a sampled configuration plus its mean CV score.
 *  Hyperband trials additionally carry the bracket/rung they ran in and
 *  the resource budget they were scored at. */
export interface MLTuneTrial {
  /** 1-based trial number, in the order the configs were drawn. */
  trial: number;
  /** The sampled hyperparameter assignment. */
  params: Record<string, number>;
  /** Mean score across the CV folds (R² or accuracy). */
  score: number;
  /** Hyperband only: the bracket the trial ran in (0 = full-budget
   *  bracket, higher = more exploratory). */
  bracket?: number;
  /** Hyperband only: the rung index within the bracket (0 = cheapest). */
  rung?: number;
  /** Hyperband only: the resource value the trial was scored at
   *  (e.g. 200 epochs / 25 estimators). */
  resource?: number;
  /** Hyperband only: true when the config survived promotion to the
   *  next rung. */
  promoted?: boolean;
}

export interface MLTuneResponse {
  task: MLTask;
  model: string;
  /** Which searcher produced the leaderboard. */
  searcher: "random" | "hyperband";
  /** Scored objective name ("R²" or "Accuracy"). */
  objective: string;
  /** CV folds per trial. */
  cvK: number;
  /** Requested trial budget (random mode; the Hyperband schedule fixes
   *  its own trial count). */
  maxTrials: number;
  /** Total configurations in the model's discrete search grid. */
  spaceSize: number;
  /** True when the whole grid was evaluated (budget ≥ space size). */
  exhaustive: boolean;
  /** Leaderboard — best score first, ties by trial number. */
  trials: MLTuneTrial[];
  /** The default-configuration CV score, for honest comparison. */
  baseline: { params: Record<string, number>; score: number };
  /** Hyperband only: the schedule + resource economics. */
  hyperband?: {
    /** The halving/promotion factor η. */
    eta: number;
    /** The resource hyperparameter's name (e.g. "epochs"). */
    resourceKey: string;
    /** The model-specific resource label ("boosting rounds" vs "trees") —
     *  preferred over any per-key generic label by the UI. */
    resourceLabel: string;
    /** The largest resource budget R (top of the ladder). */
    maxResource: number;
    /** Distinct rung resources across all brackets, ascending. */
    ladder: number[];
    /** Per-bracket summary: how many configs started, at which rungs. */
    brackets: Array<{ bracket: number; configs: number; rungs: number[] }>;
    /** Resource economics: what the search spent vs evaluating every
     *  trial at the full budget. */
    budget: { spent: number; naive: number; savedPct: number };
  };
  best: {
    params: Record<string, number>;
    /** Mean CV score of the winning configuration. */
    cvScore: number;
    /** Held-out test metric of the best-config refit on the train split. */
    testMetric: number;
    testMetricName: string;
    /** Improvement over the default configuration's CV score. */
    improvement: number;
  };
  datasetMeta: {
    nTotal: number;
    features: string[];
    /** Present when tuning ran on a user-uploaded CSV. */
    source?: CsvSourceMeta;
  };
}

/* ------------------------------------------------------------------ */
/* ML model comparison (the web mirror of pharmaml.ml.training.       */
/* compare_models — the rar's signature workflow)                      */
/* ------------------------------------------------------------------ */

export interface MLCompareRequest {
  task: MLTask;
  /** 2–6 registry models to train side by side. */
  models: string[];
  /** CV folds scoring every model (3 / 5 / 10 — never off: a leaderboard
   *  without folds is just a test-split lottery). */
  cv: 3 | 5 | 10;
  seed: number;
  testSize: number;
  /** Synthetic-generator knobs (ignored when csv is present). */
  samples: number;
  noise: number;
  /** Raw CSV text — when present, the comparison uses the uploaded data. */
  csv?: string;
  csvName?: string;
  /** Regression only: feature/target column picks (same semantics as train). */
  featureColumns?: string[];
  featureColumn?: string;
  featureColumn2?: string;
  targetColumn?: string;
}

/** One leaderboard row: a model trained at its default hyperparameters. */
export interface MLCompareEntry {
  /** The model's registry key. */
  model: string;
  /** Mean CV score across the shared folds (R² or accuracy). */
  cvMean: number;
  /** Population std of the per-fold scores. */
  cvStd: number;
  /** Held-out test metric of the refit on the train split. */
  testMetric: number;
  /** "R²" or "Accuracy". */
  metricName: string;
  /** Wall-clock fit time of the held-out refit, milliseconds. */
  fitMs: number;
  /** Gap to the leaderboard's best (0 for the winner; negative for the
   *  rest — reported in the same units as the metric). */
  delta: number;
  /** 1-based leaderboard rank (ties share the earlier rank). */
  rank: number;
}

export interface MLCompareResponse {
  task: MLTask;
  /** Scored objective name ("R²" or "Accuracy"). */
  objective: string;
  /** CV folds every model was scored on (the same folds for all). */
  cvK: number;
  /** Leaderboard — best CV mean first. */
  entries: MLCompareEntry[];
  datasetMeta: {
    nTotal: number;
    features: string[];
    /** Present when the comparison ran on a user-uploaded CSV. */
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

/** One row of the MAD variability ranking (the most_variable_genes twin). */
export interface VariableGeneRow {
  /** 1-based rank by MAD descending. */
  rank: number;
  gene: string;
  /** Median absolute deviation of the gene's expression across the panel. */
  mad: number;
}

export interface BiomarkersResponse {
  /** Top 15 genes by |rho|. */
  biomarkers: BiomarkerRow[];
  /** Top 15 genes by MAD — the robust variability pre-filter (100x-124). */
  variableGenes: VariableGeneRow[];
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

/* ------------------------------------------------------------------ */
/* Consensus rescore (docking summary tables)                          */
/* ------------------------------------------------------------------ */

/** One row of the explainable leaderboard: the identity column value plus
 *  one 0-1 `<column>_norm` unit per evidence column and the consensus. */
export interface DockingRescoreRow {
  [key: string]: string | number | null;
}

/** The "why the winner wins" verdict (per-row argmax over the _norm units). */
export interface DockingRescoreVerdict {
  identity: string | null;
  consensus: number;
  driver: string;
  driverValue: number;
}

/** Response of POST /api/docking/rescore — the pharmaml consensus_breakdown
 *  twin (same shared core as consensus_rescore and the CLI). */
export interface DockingRescoreResponse {
  identityColumn: string;
  columns: string[];
  rows: DockingRescoreRow[];
  /** "col (lower=better, weight 1), ..." — byte-identical to the CLI's line. */
  recipe: string;
  nRows: number;
  verdict: DockingRescoreVerdict | null;
  pharmamlVersion: string;
}

/* ------------------------------------------------------------------ */
/* Feature Factory (chem features + top-K selection)                   */
/* ------------------------------------------------------------------ */

/** The twenty descriptor/fingerprint/solvation families of
 * pharmaml.chem.features (the ten fak_ml.ipynb datamol families +
 * xxx.ipynb's shape_3d joined the original nine). */
export type FeatureFamily =
  | "descriptors"
  | "atompair_count"
  | "ecfp"
  | "ecfp3"
  | "avalon_count"
  | "rdkit_fingerprint"
  | "morgan_full"
  | "morgan_scaffold"
  | "maccs"
  | "fcfp"
  | "fcfp_count"
  | "topological"
  | "topological_count"
  | "rdkit_count"
  | "pattern"
  | "layered"
  | "estate"
  | "erg"
  | "shape_3d"
  | "rism";

/** One ranked, selected feature row returned by /api/chem/features. */
export interface FeatureFactoryRow {
  rank: number;
  feature: string;
  family: string;
  importance: number;
}

/** Response of the feature-factory bridge (real RDKit via pharmaml). */
export interface FeatureFactoryResponse {
  nMolecules: number;
  dropped: Array<{ row: number; smiles: string }>;
  totalFeatures: number;
  familiesComputed: Record<string, number>;
  k: number;
  survivors: number;
  method: string;
  selected: FeatureFactoryRow[];
  familyCounts: Record<string, number>;
  elapsedSeconds: number;
  rdkitVersion: string;
}

/* ------------------------------------------------------------------ */
/* Molecule surgery (generative topology moves + scaffold-hop breed)   */
/* ------------------------------------------------------------------ */

/** Response of /api/gen/surgery mode "topology": the one-step
 * ring-size mutant set of one molecule, plus the captioned 2D
 * structure grid as a base64 PNG data URI (the desktop "Ring
 * topology moves" card's twin — pharmaml.generative.ring_topology_moves). */
export interface SurgeryTopologyResponse {
  mode: "topology";
  source: string;
  mutants: string[];
  summary: string;
  gridPng: string;
  gridPanels: number;
  elapsedSeconds: number;
  rdkitVersion: string;
}

/** Response of /api/gen/surgery mode "breed": the seeded
 * scaffold-hop-crossover litter (a failed draw is an honest null),
 * plus the parents-and-children structure grid (the desktop "Breed
 * two molecules" card's twin — pharmaml.generative.scaffold_hop_crossover). */
export interface SurgeryBreedResponse {
  mode: "breed";
  parentA: string;
  parentB: string;
  children: number;
  seed: number;
  litter: Array<string | null>;
  viable: number;
  summary: string;
  gridPng: string;
  gridPanels: number;
  elapsedSeconds: number;
  rdkitVersion: string;
}

/** Either surgery response, discriminated on `mode`. */
export type SurgeryResponse = SurgeryTopologyResponse | SurgeryBreedResponse;

/* ------------------------------------------------------------------ */
/* MCTS molecule search (the Studio Generative method's web twin)     */
/* ------------------------------------------------------------------ */

/** One leaderboard row of an MCTS run: rank 1 = best. The breakdown
 * carries one key per `MctsResponse.breakdownColumns` (empty for
 * single-objective runs — the blended score is the whole story). */
export interface MctsLeaderboardRow {
  rank: number;
  smiles: string;
  score: number;
  breakdown: Record<string, number>;
}

/** The per-cycle progress of the search (one record per iteration):
 * best/mean score of the molecules the tree found so far, plus the
 * unique count — scores live in [0, 1] for every objective, so one
 * shared scale plots them honestly. */
export interface MctsHistory {
  cycle: number[];
  bestScore: number[];
  meanScore: number[];
  nUnique: number[];
}

/** The search's own accounting, surfaced verbatim from the optimizer's
 * metadata — the tree size, rollout count, validity and the policy
 * knobs the run actually used. */
export interface MctsTreeStats {
  nNodes: number;
  nRollouts: number;
  nValid: number;
  nUnique: number;
  nCacheHits: number;
  maxDepth: number;
  corpusSize: number;
  policy: string;
  cVal: number;
  expansionThreshold: number;
  simulationNum: number;
  maxLen: number;
}

/** Response of /api/gen/mcts: the ranked leaderboard of one
 * MctsOptimizer run (ChemTSv2-style MCTS over an on-the-fly SMILES RNN
 * prior — the desktop Studio Generative page's MCTS method), with the
 * per-cycle history, the tree accounting and the top-molecule
 * structure grid as a base64 PNG data URI. */
export interface MctsResponse {
  objective: string;
  objectiveLabel: string;
  iterations: number;
  treePolicy: "uct" | "puct";
  seed: number;
  corpusMode: "built-in" | "seeded" | "custom";
  best: string;
  bestScore: number;
  leaderboard: MctsLeaderboardRow[];
  breakdownColumns: string[];
  history: MctsHistory;
  gridPng: string;
  gridPanels: number;
  treeStats: MctsTreeStats;
  summary: string;
  elapsedSeconds: number;
  rdkitVersion: string;
}

/* ------------------------------------------------------------------ */
/* Response AI — drug-response learning (the Studio page's web twin)  */
/* ------------------------------------------------------------------ */

/** Which analysis the Response AI card runs — the four headline tabs
 * of the desktop Studio's Response AI page (hill / metrics / model /
 * cv). The page's deeper diagnostics (informative regression, the
 * response VAE, multi-omics fusion) stay desktop-side. */
export type DrugResponseAnalysis = "hill" | "metrics" | "model" | "cv";

/** Response of /api/ml/drugresponse analysis "hill": the four-parameter
 * log-logistic fitted by the toolkit's own Levenberg-Marquardt engine
 * (pharmaml.ml.drugresponse.fit_hill_curve) on the seeded 21-point 4PL
 * demo or a pasted series. `top`/`bottom` follow the module's own
 * convention: the low-dose / high-dose plateaus. The truth block and
 * clean AUC ride only on the seeded demo; the curve arrays are the
 * fitted curve on a padded log-dose grid for the chart. */
export interface HillFitResponse {
  analysis: "hill";
  source: "seeded 4PL demo" | "pasted series";
  parameters: {
    hill_coef: number;
    ic50: number;
    top: number;
    bottom: number;
  };
  rSquared: number;
  converged: boolean;
  nPoints: number;
  fittedAuc: number;
  measuredAuc: number;
  doses: number[];
  response: number[];
  curveDose: number[];
  curveResponse: number[];
  truth?: { hill_coef: number; ic50: number; top: number; bottom: number };
  noise?: number;
  cleanAuc?: number;
  summary: string;
  elapsedSeconds: number;
  pharmamlVersion: string;
}

/** Response of /api/ml/drugresponse analysis "metrics": the evaluation
 * layer every response paper reports (pharmaml.ml.drugresponse
 * .response_metrics — Pearson / Spearman / RMSE / MAE, NaN pairs
 * dropped before everything) plus the rank trio
 * (rank_correlations: Pearson / Spearman / Kendall tau-b) on the same
 * pair — the page's response-metrics and rank-correlations tabs in
 * one run. */
export interface ResponseMetricsResponse {
  analysis: "metrics";
  source: "seeded noisy pair" | "pasted pair";
  n: number;
  /** Pairs where either side was NaN — dropped before every metric
   * (response_metrics' own NaN semantics; the rank trio rides the
   * surviving finite pairs because rank_correlations refuses NaN). */
  nDropped: number;
  pearson: number;
  spearman: number;
  rmse: number;
  mae: number;
  rank: {
    n: number;
    pearson: number;
    spearman: number;
    kendall: number;
  };
  yTrue: number[];
  yPred: number[];
  summary: string;
  elapsedSeconds: number;
  pharmamlVersion: string;
}

/** One predicted-response head row of the CaDRReS model run: the
 * model's predicted response for one cell line across every drug of
 * the seeded 12x4 screen (values are ordered by `drugs`). */
export interface ResponseHeadRow {
  cell: string;
  values: number[];
}

/** Response of /api/ml/drugresponse analysis "model": the CaDRReS
 * bilinear factorisation (pharmaml.ml.drugresponse
 * .DrugResponseModel) trained on the seeded 12x4 screen with its 15%
 * NaN mask — full-batch gradient descent over the observed pairs
 * only, the honest in-sample pearson, the loss history, the predicted
 * head for CELL_00/CELL_07, and every observed/predicted pair for the
 * scatter. */
export interface ResponseModelResponse {
  analysis: "model";
  epochs: number;
  factors: number;
  seed: number;
  shape: { cellLines: number; drugs: number };
  drugs: string[];
  trainLosses: number[];
  finalLoss: number;
  nObserved: number;
  pearsonR: number;
  predictedHead: ResponseHeadRow[];
  observed: number[];
  predicted: number[];
  nPairs: number;
  summary: string;
  elapsedSeconds: number;
  pharmamlVersion: string;
}

/** One cell-line-grouped fold of the cross-validation run. */
export interface ResponseCvFold {
  fold: number;
  nTestCells: number;
  nObserved: number;
  pearson: number;
  rmse: number;
}

/** Response of /api/ml/drugresponse analysis "cv":
 * cross_validate_response over the seeded screen — folds split CELL
 * LINES (no cell line straddles a fold), each fold training a fresh
 * model with the Studio page's exact factory, the honest across-fold
 * mean pearson even when the toy screen barely carries signal. */
export interface ResponseCvResponse {
  analysis: "cv";
  nFolds: number;
  folds: ResponseCvFold[];
  meanPearson: number;
  summary: string;
  elapsedSeconds: number;
  pharmamlVersion: string;
}

/** Either Response AI response, discriminated on `analysis`. */
export type DrugResponseResult =
  | HillFitResponse
  | ResponseMetricsResponse
  | ResponseModelResponse
  | ResponseCvResponse;

/* ------------------------------------------------------------------ */
/* Synthetic Accessibility scoring (part3 §24 — Ertl-Schuffenhauer)    */
/* ------------------------------------------------------------------ */

/** One molecule's SA row: score is null exactly when the SMILES is
 * invalid (the error field then carries the reason, the original
 * string is preserved — never a misleading number). */
export interface SAScoreRow {
  smiles: string;
  score: number | null;
  valid: boolean;
  error: string | null;
  canonicalSmiles: string | null;
  method: string;
  direction: string;
}

/** Direction-aware batch statistics over the valid subset only. */
export interface SASummary {
  nTotal?: number;
  nScored?: number;
  nInvalid?: number;
  min?: number;
  max?: number;
  mean?: number;
  median?: number;
  std?: number;
  method: string;
  direction: string;
  limitations: string;
  elapsedSeconds?: number;
}

/** Response of /api/chem/sa-score: every row, the easiest-first
 * ordering, the summary statistics, and — when a filter bound was
 * requested — the kept/rejected split with both sides preserved. */
export interface SAScoreResponse {
  nTotal: number;
  nScored: number;
  nInvalid: number;
  results: SAScoreRow[];
  sorted: Array<{ smiles: string; score: number | null }>;
  summary: SASummary;
  method: string;
  direction: string;
  limitations: string;
  elapsedSeconds: number;
  rdkitVersion: string;
  kept?: SAScoreRow[];
  rejected?: SAScoreRow[];
  nKept?: number;
  nRejected?: number;
}

/* ------------------------------------------------------------------ */
/* Docking ligand prep (SMILES -> PDBQT + grid box + GPF/DPF)          */
/* ------------------------------------------------------------------ */

/** One prepared (or failed) ligand from /api/docking/prep. */
export interface LigandPrepRow {
  smiles: string;
  ok: boolean;
  name: string;
  atoms: number | null;
  torsdof: number | null;
  atomTypes: string[];
  pdbqtText: string | null;
  error: string | null;
}

/** Grid box computed from the first successfully prepared ligand. */
export interface LigandPrepGridBox {
  centerX: number;
  centerY: number;
  centerZ: number;
  sizeX: number;
  sizeY: number;
  sizeZ: number;
  npts: number[];
  spacing: number;
}

/** Response of the docking-prep bridge (real RDKit/meeko via pharmaml). */
export interface LigandPrepResponse {
  ligands: LigandPrepRow[];
  gridBox: LigandPrepGridBox | null;
  gpfText: string | null;
  dpfText: string | null;
  nOk: number;
  nFailed: number;
  elapsedSeconds: number;
  rdkitVersion: string;
}

/* ------------------------------------------------------------------ */
/* Applicability domain (ml ad — the 2026-10 upload's workflow)        */
/* ------------------------------------------------------------------ */

/** Response of POST /api/ml/ad — the reliability verdict per held-out
 *  row of the last trained configuration (kNN distance domain +
 *  bootstrap-ensemble uncertainty, the upload's two questions). */
export interface MLAdResponse {
  task: "regression" | "classification";
  model: string;
  summary: AdSummary;
  rows: AdRow[];
  /** Training-side mean kNN distances (panel (a) histogram input). */
  trainMean: number[];
  nTrain: number;
  featureNames: string[];
  /** Why the bootstrap half was skipped (classification / adModels 0). */
  bootstrapNote: string | null;
}

/* ------------------------------------------------------------------ */
/* Learning curve (ml learning-curve — the desktop card's 100x-125     */
/* web mirror: train-set size vs CV score, "should we assay more?")    */
/* ------------------------------------------------------------------ */

/** One schedule point of POST /api/ml/learning-curve — both curves
 *  reduced across folds at one training-set size. */
export interface MLLearningCurvePoint {
  /** Absolute training-subset row count this point fit on. */
  size: number;
  /** Mean score on the very rows each fit saw (training curve). */
  trainMean: number;
  /** Population std of the training scores across folds. */
  trainStd: number;
  /** Mean score on each fold's held-out rows (CV curve). */
  valMean: number;
  /** Population std of the CV scores across folds. */
  valStd: number;
}

/** Response of POST /api/ml/learning-curve — the web twin of the
 *  desktop ML card's "Learning curve" figure (pharmaml.viz.
 *  plot_learning_curve's data semantics: the library's default
 *  10 % … 100 % 8-step schedule, mean ± std across folds at every
 *  size, and the gap verdict with the same 0.15 threshold). */
export interface MLLearningCurveResponse {
  task: "regression" | "classification";
  model: string;
  /** "R²" (regression) or "Accuracy" (classification). */
  metricName: string;
  /** The fold count the curve was computed with. */
  folds: 3 | 5 | 10;
  /** The size schedule as fractions of each fold's train part. */
  fractions: number[];
  /** One point per fraction, smallest size first. */
  points: MLLearningCurvePoint[];
  /** trainMean − valMean at the largest size (the overfitting gap). */
  gap: number;
  /** The library's verdict line (threshold 0.15). */
  verdict: string;
  datasetMeta: {
    nTotal: number;
    nFeatures: number;
    kind: "synthetic" | "csv";
    fileName?: string;
  };
  /** Regression GP only: the kernel the curve cloned — learned ONCE on
   *  the full data by the LML grid search when the request left all
   *  three slots unset (the sklearn clone semantics), then fixed for
   *  every refit. */
  gpKernel?: {
    lengthScale: number;
    signalStd: number;
    noiseStd: number;
  };
  /** Classification GP only: (ℓ, σf) learned once on the full data by
   *  the Laplace-lml grid search, then cloned by every refit. */
  gpClsKernel?: {
    lengthScale: number;
    signalStd: number;
  };
}

/* ------------------------------------------------------------------ */
/* SHAP attribution (ml shap — the desktop GUI's fourth button row,   */
/* compact web read: mean-|SHAP| over the run's own held-out rows,     */
/* computed by the toolkit's Python tree twin via the bridge route)    */
/* ------------------------------------------------------------------ */

/** One ranked feature of POST /api/ml/shap's compact read. */
export interface ShapRankRow {
  rank: number;
  feature: string;
  /** Mean |SHAP| across the explained held-out rows (magnitude only —
   *  the signed story is the beeswarm the desktop twin draws). */
  meanAbs: number;
}

/** Response of POST /api/ml/shap — the web twin of the desktop GUI's
 *  SHAP button row (round 37's pharmaml.ml.explain surface) in the
 *  compact form: the ranking needs no canvas, so the card renders in
 *  pure TypeScript while only the computation crosses into Python
 *  (build_model -> fit -> shap_summary over the run's re-derived rows
 *  and split — the frozen-train-request discipline). */
export interface MLShapResponse {
  task: "regression" | "classification";
  /** The web playground's model id (the run's own). */
  model: string;
  /** The Python twin's estimator class name (provenance for the caption). */
  pythonTwin: string;
  /** Mean-|SHAP| ranking over the run's held-out rows, descending. */
  top: ShapRankRow[];
  /** The explainer's expected model output when exposed (null otherwise). */
  baseValue: number | null;
  /** Held-out rows explained. */
  nSamples: number;
  nFeatures: number;
  /** Training rows the twin was fitted on. */
  nTrain: number;
  /** Feature names in frame order. */
  featureNames: string[];
  shapVersion: string;
  elapsedSeconds: number;
  /** Provenance note — the card renders it verbatim under the bars. */
  splitNote: string;
}

/* ------------------------------------------------------------------ */
/* Molecular Dynamics (round 249 — the pharmaml.md web twins)          */
/* ------------------------------------------------------------------ */

/** Mean / population std / coefficient of variation of one parsed
 * metric — null exactly when the values carry no spread or no points
 * (the honest-null discipline: never a made-up number). */
export interface MdXvgStats {
  mean: number | null;
  std: number | null;
  cv: number | null;
}

/** One parsed GROMACS export metric from /api/md/xvg: the time axis is
 * already normalised server-side (long ps axes → ns), energies arrive
 * in kcal/mol, H-bond counts carry the optional 3-point smoothing. */
export interface MdXvgSeries {
  time: number[];
  values: number[];
  /** The export file the metric was parsed from. */
  source: string;
  /** Human title (e.g. "Ligand RMSD"). */
  label: string;
  stats: MdXvgStats;
}

/** One entity's parsed metrics keyed by metric key (e.g.
 * "ligand_rmsd") — the study reader's LigandResult.to_dict(). */
export interface MdXvgEntity {
  name: string;
  series: Record<string, MdXvgSeries>;
}

/** Response of POST /api/md/xvg — the parsed ligand study, the
 * cross-metric Pearson matrix (null when fewer than two metrics),
 * the summary CSV and the parser's own warnings, verbatim. */
export interface MdXvgResponse {
  ligand: MdXvgEntity;
  correlations: { labels: string[]; matrix: number[][] } | null;
  summaryCsv: string;
  warnings: string[];
  elapsedSeconds: number;
  method: string;
}

export interface MdHbondResponse {
  nFrames: number;
  seed: number;
  noise: number;
  time: number[];
  /** Per-frame H-bond count from hbond_timeline (0, 1 or 2). */
  counts: number[];
  /** Per-bond donor-acceptor distance series (2 bonds). */
  distances: number[][];
  cutoffs: {
    distance: number;
    angle: number;
  };
  stats: {
    mean: number;
    max: number;
    bondedFraction: number;
    bond0Fraction: number;
    bond1LeadsBond0: boolean;
  };
  elapsedSeconds: number;
  method: string;
}

/** Response of POST /api/md/saltbridge — the designed two-bridge ionic
 * demo (getcontacts' criterion) analysed by the real toolkit:
 * synthetic_salt_bridge_trajectory builds two phase-shifted
 * cation-anion pairs (Lys NZ···Glu CD, Arg NH1···Asp CG) oscillating
 * across the 4.0 Å cutoff, salt_bridge_timeline counts per frame, and
 * the per-bridge distance series ride along for the chart. */
export interface MdSaltBridgeResponse {
  nFrames: number;
  seed: number;
  noise: number;
  time: number[];
  /** Per-frame salt-bridge count from salt_bridge_timeline (0, 1 or 2). */
  counts: number[];
  /** Per-bridge cation-anion distance series (2 bridges). */
  distances: number[][];
  cutoffs: {
    distance: number;
  };
  stats: {
    mean: number;
    max: number;
    bondedFraction: number;
    bridge1Fraction: number;
    bridge2Fraction: number;
    bridge2LeadsBridge1: boolean;
  };
  elapsedSeconds: number;
  method: string;
}

/** Response of POST /api/md/cv — the COLVAR playground: the pasted
 * plumed.dat snippet parsed and computed over the deterministic
 * alpha-helix demo by the real toolkit's parse_plumed_cvs. Every series
 * rides on one finite-only time axis; `colvar` is the exact text
 * PLUMED's PRINT would have written (`#! FIELDS` header, six decimals). */
export interface MdCvResponse {
  nFrames: number;
  seed: number;
  noise: number;
  nAtoms: number;
  time: number[];
  /** CV labels in declaration order (unique, identifier-safe). */
  names: string[];
  /** One per-CV series per label, aligned with `names`. */
  series: number[][];
  /** Honest per-CV stats over the frames, aligned with `names`. */
  stats: Array<{
    label: string;
    mean: number;
    min: number;
    max: number;
    final: number;
  }>;
  /** The COLVAR text (#! FIELDS header + one row per frame). */
  colvar: string;
  elapsedSeconds: number;
  method: string;
}

/** Response of POST /api/md/trajectory — the deterministic synthetic
 * alpha-helix (10 residues × N/CA/C, residues 5–7 a flexible loop)
 * analysed by the real toolkit: RMSD after Kabsch superposition, Rg,
 * native-contact Q, per-atom/per-residue RMSF, the contact-frequency
 * map, the PCA projection of the frames and the ANM fluctuations of
 * the first frame. */
export interface MdTrajectoryResponse {
  nFrames: number;
  nAtoms: number;
  nResidues: number;
  seed: number;
  noise: number;
  contactCutoff: number;
  time: number[];
  rmsd: number[];
  rg: number[];
  q: number[];
  /** Per-atom RMSF (length nAtoms). */
  rmsf: number[];
  /** Per-residue RMSF (length nResidues). */
  rmsfResidues: number[];
  /** nAtoms × nAtoms contact frequency over the frames (0..1). */
  contactMap: number[][];
  pca: {
    /** One [PC1, PC2] pair per frame. */
    projection: number[][];
    explainedVarianceRatio: number[];
  };
  /** ANM fluctuations from the first frame (length nAtoms). */
  anmFluctuations: number[];
  stats: {
    rmsdMean: number;
    rmsdFinal: number;
    rgMean: number;
    qMean: number;
    contactPairs: number;
  };
  elapsedSeconds: number;
  method: string;
}
