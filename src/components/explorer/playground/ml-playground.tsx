"use client";

import { useMemo, useRef, useState } from "react";
import {
  Brain,
  Check,
  ChevronDown,
  ClipboardPaste,
  Crosshair,
  Download,
  Eraser,
  FileSpreadsheet,
  FileUp,
  History,
  Layers,
  Loader2,
  Network,
  Play,
  RotateCcw,
  Sigma,
  SlidersHorizontal,
  Activity,
  Table2,
  TrendingDown,
  TrendingUp,
  TriangleAlert,
  Trophy,
  X,
  ZoomIn,
} from "lucide-react";
import {
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { CHART, ParamSlider } from "@/components/explorer/playground/calc-ui";
import { ProbabilityCalibration } from "@/components/explorer/playground/calibration-widget";
import {
  MAX_CSV_FEATURES,
  MAX_CSV_TEXT,
  MIN_CSV_ROWS,
  numericColumnNames,
  parseCsv,
} from "@/lib/csv";
import { fmtBytes, fmtNum, fmtTick } from "@/lib/format";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import { DEFAULT_ML_CONFIG, useToolkitStore, type MlRunRecord } from "@/lib/store";
import type {
  DatasetPreviewResponse,
  MLTrainResponse,
  MLTrainRequest,
  MLTuneRequest,
  MLTuneResponse,
} from "@/lib/toolkit-types";

type Task = "regression" | "classification";

/* ------------------------------------------------------------------ */
/* Model registry (mirrors the server-side /api/ml/train registry)      */
/* ------------------------------------------------------------------ */

/** How the "Learned parameters" card summarizes this model family. */
type ModelKind = "coefficients" | "importance" | "params";

type ModelGroup =
  | "linear"
  | "trees"
  | "instance"
  | "kernel"
  | "probabilistic"
  | "neural"
  | "hierarchical";

const GROUP_LABELS: Record<ModelGroup, string> = {
  linear: "Linear family",
  trees: "Trees & ensembles",
  instance: "Instance-based",
  kernel: "Kernel machines",
  probabilistic: "Probabilistic",
  neural: "Neural",
  hierarchical: "Hierarchical",
};

/** One contextual hyperparameter control. */
interface HyperSpec {
  key:
    | "alpha"
    | "l2"
    | "l1Ratio"
    | "k"
    | "maxDepth"
    | "nEstimators"
    | "rounds"
    | "learningRate"
    | "hidden"
    | "epochs"
    | "lr"
    | "C"
    | "epsilon"
    | "gamma"
    | "nClusters"
    | "radiusFactor";
  label: string;
  control: "input" | "slider";
  unit?: string;
  min?: number;
  max?: number;
  step?: number;
}

interface ModelSpec {
  id: string;
  label: string;
  /** Honest one-line description shown under the picker. */
  hint: string;
  group: ModelGroup;
  kind: ModelKind;
  hp: HyperSpec[];
}

const ALPHA_HP: HyperSpec = { key: "alpha", label: "α strength", control: "input" };

const REG_MODELS: ModelSpec[] = [
  {
    id: "linear",
    label: "Linear regression",
    hint: "Closed-form least squares via normal equations",
    group: "linear",
    kind: "coefficients",
    hp: [],
  },
  {
    id: "polynomial",
    label: "Polynomial (degree 2)",
    hint: "Degree-2 features, closed-form solve",
    group: "linear",
    kind: "coefficients",
    hp: [],
  },
  {
    id: "ridge",
    label: "Ridge (α regularized)",
    hint: "L2 penalty α on the normal-equation diagonal",
    group: "linear",
    kind: "coefficients",
    hp: [ALPHA_HP],
  },
  {
    id: "lasso",
    label: "Lasso (L1)",
    hint: "Coordinate descent with soft-thresholding",
    group: "linear",
    kind: "coefficients",
    hp: [ALPHA_HP],
  },
  {
    id: "elastic_net",
    label: "Elastic net",
    hint: "L1 + L2 blend via α and the l1 mix",
    group: "linear",
    kind: "coefficients",
    hp: [
      ALPHA_HP,
      { key: "l1Ratio", label: "l1 mix", control: "slider", unit: "", min: 0, max: 1, step: 0.05 },
    ],
  },
  {
    id: "huber",
    label: "Huber (robust)",
    hint: "IRLS — quadratic core, linear tails; outliers down-weighted",
    group: "linear",
    kind: "coefficients",
    hp: [
      {
        key: "epsilon",
        label: "ε switch point",
        control: "slider",
        unit: "y units",
        min: 1.05,
        max: 4,
        step: 0.05,
      },
    ],
  },
  {
    id: "decision_tree",
    label: "Decision tree (CART)",
    hint: "Variance-reduction splits, depth-capped",
    group: "trees",
    kind: "importance",
    hp: [{ key: "maxDepth", label: "max depth", control: "slider", unit: "levels", min: 1, max: 12, step: 1 }],
  },
  {
    id: "random_forest",
    label: "Random forest",
    hint: "Seeded bagging of depth-capped CART trees",
    group: "trees",
    kind: "importance",
    hp: [
      { key: "nEstimators", label: "trees", control: "slider", unit: "n", min: 5, max: 200, step: 5 },
      { key: "maxDepth", label: "max depth", control: "slider", unit: "levels", min: 1, max: 12, step: 1 },
    ],
  },
  {
    id: "extra_trees",
    label: "Extra trees (ET)",
    hint: "Extremely randomized trees — random thresholds, full sample (no bagging)",
    group: "trees",
    kind: "importance",
    hp: [
      { key: "nEstimators", label: "trees", control: "slider", unit: "n", min: 5, max: 200, step: 5 },
      { key: "maxDepth", label: "max depth", control: "slider", unit: "levels", min: 1, max: 12, step: 1 },
    ],
  },
  {
    id: "gradient_boosting",
    label: "Gradient boosting",
    hint: "Depth-3 stumps fit the residuals, learning-rate scaled",
    group: "trees",
    kind: "importance",
    hp: [
      { key: "rounds", label: "rounds", control: "slider", unit: "n", min: 5, max: 200, step: 5 },
      { key: "learningRate", label: "learning rate", control: "slider", unit: "η", min: 0.02, max: 0.5, step: 0.02 },
    ],
  },
  {
    id: "ada_boost",
    label: "AdaBoost (R2)",
    hint: "Drucker's AdaBoost.R2 — weighted median of depth-2 stumps",
    group: "trees",
    kind: "importance",
    hp: [
      { key: "nEstimators", label: "rounds", control: "slider", unit: "n", min: 5, max: 200, step: 5 },
      { key: "learningRate", label: "learning rate", control: "slider", unit: "η", min: 0.02, max: 0.5, step: 0.02 },
    ],
  },
  {
    id: "svm_rbf",
    label: "SVR · RBF kernel",
    hint: "Least-squares SVR: the (K+λI)⁻¹ dual in one closed form",
    group: "kernel",
    kind: "params",
    hp: [
      { key: "gamma", label: "kernel γ", control: "slider", unit: "γ", min: 0.05, max: 2, step: 0.05 },
      ALPHA_HP,
    ],
  },
  {
    id: "knn",
    label: "k-nearest neighbours",
    hint: "Averages the k nearest train points",
    group: "instance",
    kind: "params",
    hp: [{ key: "k", label: "neighbours", control: "slider", unit: "k", min: 1, max: 25, step: 2 }],
  },
  {
    id: "mlp",
    label: "MLP (1 hidden layer)",
    hint: "tanh hidden layer — demo-scale network, not production deep learning",
    group: "neural",
    kind: "params",
    hp: [
      { key: "hidden", label: "hidden units", control: "slider", unit: "n", min: 2, max: 64, step: 2 },
      { key: "epochs", label: "epochs", control: "slider", unit: "n", min: 20, max: 2000, step: 20 },
      { key: "lr", label: "learning rate", control: "slider", unit: "η", min: 0.005, max: 0.5, step: 0.005 },
    ],
  },
  {
    id: "hsvr",
    label: "Hierarchical SVR (HSVR)",
    hint: "Seeded k-means clusters + per-cluster ridge, blended by trust radius",
    group: "hierarchical",
    kind: "params",
    hp: [
      { key: "nClusters", label: "clusters", control: "slider", unit: "k", min: 2, max: 8, step: 1 },
      { key: "radiusFactor", label: "trust radius ×", control: "slider", unit: "×", min: 0.5, max: 4, step: 0.25 },
      ALPHA_HP,
    ],
  },
];

const CLS_MODELS: ModelSpec[] = [
  {
    id: "logistic",
    label: "Logistic regression",
    hint: "Batch GD, 800 epochs, L2 (bias exempt)",
    group: "linear",
    kind: "coefficients",
    hp: [{ key: "l2", label: "L2 strength λ", control: "input" }],
  },
  {
    id: "svm_linear",
    label: "Linear SVM",
    hint: "Squared-hinge margin via gradient descent",
    group: "linear",
    kind: "params",
    hp: [{ key: "C", label: "margin C", control: "slider", unit: "C", min: 0.05, max: 10, step: 0.05 }],
  },
  {
    id: "svm_rbf",
    label: "SVM · RBF kernel",
    hint: "LS-SVM: closed-form Gaussian kernel machine, λ = 1/(2C)",
    group: "kernel",
    kind: "params",
    hp: [
      { key: "C", label: "margin C", control: "slider", unit: "C", min: 0.05, max: 10, step: 0.05 },
      { key: "gamma", label: "kernel γ", control: "slider", unit: "γ", min: 0.05, max: 2, step: 0.05 },
    ],
  },
  {
    id: "decision_tree",
    label: "Decision tree (gini)",
    hint: "Gini-impurity splits, depth-capped",
    group: "trees",
    kind: "importance",
    hp: [{ key: "maxDepth", label: "max depth", control: "slider", unit: "levels", min: 1, max: 12, step: 1 }],
  },
  {
    id: "random_forest",
    label: "Random forest",
    hint: "Seeded bagging of gini CART trees",
    group: "trees",
    kind: "importance",
    hp: [
      { key: "nEstimators", label: "trees", control: "slider", unit: "n", min: 5, max: 200, step: 5 },
      { key: "maxDepth", label: "max depth", control: "slider", unit: "levels", min: 1, max: 12, step: 1 },
    ],
  },
  {
    id: "extra_trees",
    label: "Extra trees (ET)",
    hint: "Extremely randomized trees — random thresholds, full sample (no bagging)",
    group: "trees",
    kind: "importance",
    hp: [
      { key: "nEstimators", label: "trees", control: "slider", unit: "n", min: 5, max: 200, step: 5 },
      { key: "maxDepth", label: "max depth", control: "slider", unit: "levels", min: 1, max: 12, step: 1 },
    ],
  },
  {
    id: "ada_boost",
    label: "AdaBoost (SAMME)",
    hint: "α-weighted vote of gini stumps — misclassified rows boosted",
    group: "trees",
    kind: "importance",
    hp: [
      { key: "nEstimators", label: "rounds", control: "slider", unit: "n", min: 5, max: 200, step: 5 },
      { key: "learningRate", label: "learning rate", control: "slider", unit: "η", min: 0.02, max: 0.5, step: 0.02 },
    ],
  },
  {
    id: "knn",
    label: "k-nearest neighbours",
    hint: "Majority vote of the k nearest train points",
    group: "instance",
    kind: "params",
    hp: [{ key: "k", label: "neighbours", control: "slider", unit: "k", min: 1, max: 25, step: 2 }],
  },
  {
    id: "gaussian_nb",
    label: "Gaussian Naive Bayes",
    hint: "Per-class Gaussian log-posterior",
    group: "probabilistic",
    kind: "params",
    hp: [],
  },
  {
    id: "mlp",
    label: "MLP (1 hidden layer)",
    hint: "Sigmoid output, cross-entropy — demo-scale network, not production deep learning",
    group: "neural",
    kind: "params",
    hp: [
      { key: "hidden", label: "hidden units", control: "slider", unit: "n", min: 2, max: 64, step: 2 },
      { key: "epochs", label: "epochs", control: "slider", unit: "n", min: 20, max: 2000, step: 20 },
      { key: "lr", label: "learning rate", control: "slider", unit: "η", min: 0.005, max: 0.5, step: 0.005 },
    ],
  },
];

const MODEL_GROUPS: ModelGroup[] = [
  "linear",
  "trees",
  "instance",
  "kernel",
  "probabilistic",
  "neural",
  "hierarchical",
];

/** Slider hyperparameter defaults (server-side defaults mirrored). */
const DEFAULT_HP: Record<string, number> = {
  l1Ratio: 0.5,
  k: 5,
  maxDepth: 4,
  nEstimators: 50,
  rounds: 50,
  learningRate: 0.1,
  hidden: 16,
  epochs: 200,
  lr: 0.05,
  C: 1,
  epsilon: 1.35,
  gamma: 0.5,
  nClusters: 3,
  radiusFactor: 1.5,
};

/** Models with a RandomSearch tuning space on the server (the honest
 *  mirror of REG_TUNE_SPACES / CLS_TUNE_SPACES in lib/ml-models.ts). */
const TUNEABLE_REG = new Set([
  "ridge",
  "lasso",
  "elastic_net",
  "huber",
  "knn",
  "decision_tree",
  "random_forest",
  "extra_trees",
  "gradient_boosting",
  "ada_boost",
  "svm_rbf",
  "mlp",
  "hsvr",
]);
const TUNEABLE_CLS = new Set([
  "logistic",
  "knn",
  "decision_tree",
  "random_forest",
  "extra_trees",
  "ada_boost",
  "svm_linear",
  "svm_rbf",
  "mlp",
]);

/** Honest footer note per model — how it was actually trained. */
const MODEL_NOTES: Record<string, string> = {
  linear: "Solved in closed form via normal equations (Gaussian elimination with partial pivoting).",
  polynomial: "Degree-2 polynomial features, solved in closed form via normal equations.",
  ridge: "Ridge adds α to the normal-equation diagonal, skipping the intercept.",
  lasso: "Cyclic coordinate descent with soft-thresholding; α drives L1 sparsity (exact zeros).",
  elastic_net: "Coordinate descent blending the L1 (l1 mix) and L2 (1 − l1 mix) penalties.",
  logistic: "Batch gradient descent: lr = 0.1, 800 epochs, L2-penalized weights (bias exempt).",
  svm_linear: "Batch gradient descent on the squared-hinge objective; C trades margin width against violations (liblinear-style regularized bias).",
  knn: "No closed-form coefficients — predictions come from the stored train points; k is echoed instead.",
  decision_tree: "CART splits chosen by impurity reduction; the bars are each feature's share of the total reduction.",
  random_forest: "Seeded bootstrap aggregation of depth-capped CART trees; importances are per-tree means, renormalized.",
  extra_trees:
    "Extremely randomized trees: every tree sees the full standardized sample (no bootstrap resamples — sklearn's bootstrap=False), and every split draws one random threshold per feature, keeping the best-gain candidate. The extra threshold randomness carves finer leaves than bagging alone; importances are per-tree means, renormalized.",
  gradient_boosting: "Each round fits a depth-3 tree to the current residuals, shrunk by the learning rate.",
  huber:
    "Iteratively reweighted least squares: residuals within ε keep weight 1, the tails fade as ε/|r| — the final sweep's robust core / linear tail split is echoed in the parameters.",
  ada_boost:
    "Weighted bootstrap rounds of shallow stumps; the ensemble predicts with the α-weighted median (regression) or α-weighted vote (classification). Importances are α-weighted per-stump gains.",
  svm_rbf:
    "Gaussian-kernel machine solved in one closed form: (K + λI)⁻¹ (regression) or the bias-augmented LS-SVM system (classification, λ = 1/(2C)).",
  gaussian_nb: "Class-conditional Gaussians (μ, σ per feature) feed the log-posterior; no tunable hyperparameters.",
  mlp: "One tanh hidden layer, full-batch gradient descent, seeded init — a demo-scale network, not production deep learning.",
  hsvr:
    "Seeded k-means splits the standardized train rows; each cluster keeps its own ridge fit, blended with the global ridge by a linear trust radius (full local weight at the centroid, zero at radius × the cluster's reach; farther queries fall back to the global model). Clusters too small for ridge predict their mean.",
};

/** Pretty labels for echoed modelParams keys. */
const HP_LABELS: Record<string, string> = {
  alpha: "α",
  l1Ratio: "l1 mix",
  k: "k (neighbours)",
  varSmoothing: "var smoothing",
  hidden: "hidden units",
  epochs: "epochs",
  lr: "learning rate",
  C: "C (margin)",
  maxDepth: "max depth",
  nEstimators: "trees",
  rounds: "rounds",
  learningRate: "learning rate",
  epsilon: "ε (switch point)",
  gamma: "kernel γ",
  lambda: "λ (ridge)",
  nRobust: "robust core",
  nOutlier: "linear tail",
  nSupport: "support rows",
  nClusters: "clusters",
  radiusFactor: "trust radius ×",
};

/** Models whose grid contains a resource hyperparameter — the slots
 *  successive halving can promote rung by rung (mirrors the server's
 *  TUNE_RESOURCES registry). */
const HYPERBAND_MODELS = new Set(["mlp", "random_forest", "extra_trees", "gradient_boosting", "ada_boost"]);

/** The resource slot per Hyperband-capable model (mirrors TUNE_RESOURCES). */
const RESOURCE_SPECS: Record<string, { key: string; label: string }> = {
  mlp: { key: "epochs", label: "epochs" },
  random_forest: { key: "nEstimators", label: "estimators" },
  extra_trees: { key: "nEstimators", label: "estimators" },
  gradient_boosting: { key: "rounds", label: "boosting rounds" },
  ada_boost: { key: "nEstimators", label: "boosting rounds" },
};

function specOf(task: Task, model: string): ModelSpec | null {
  const list = task === "regression" ? REG_MODELS : CLS_MODELS;
  return list.find((m) => m.id === model) ?? null;
}

/** Compact slider value pill format (5 → "5", 0.1 → "0.1"). */
function fmtHp(v: number): string {
  return String(Math.round(v * 100) / 100);
}

/** Type-safe hyperparameter assignment onto the request body. */
function assignHp(body: MLTrainRequest, key: string, value: number): void {
  switch (key) {
    case "alpha":
      body.alpha = value;
      break;
    case "l1Ratio":
      body.l1Ratio = value;
      break;
    case "l2":
      body.l2 = value;
      break;
    case "k":
      body.k = value;
      break;
    case "maxDepth":
      body.maxDepth = value;
      break;
    case "nEstimators":
      body.nEstimators = value;
      break;
    case "rounds":
      body.rounds = value;
      break;
    case "learningRate":
      body.learningRate = value;
      break;
    case "hidden":
      body.hidden = value;
      break;
    case "epochs":
      body.epochs = value;
      break;
    case "lr":
      body.lr = value;
      break;
    case "C":
      body.C = value;
      break;
    case "nClusters":
      body.nClusters = value;
      break;
    case "radiusFactor":
      body.radiusFactor = value;
      break;
  }
}

const REG_METRICS: Array<{ key: string; label: string }> = [
  { key: "r2", label: "R²" },
  { key: "rmse", label: "RMSE" },
  { key: "mae", label: "MAE" },
];

const CLS_METRICS: Array<{ key: string; label: string }> = [
  { key: "accuracy", label: "Accuracy" },
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "f1", label: "F1" },
];

function MetricCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-lg border bg-card px-3 py-2.5 shadow-sm">
      <p className="text-xs text-muted-foreground">{label}</p>
      <p className="font-mono text-xl font-semibold tabular-nums">{fmtNum(value, 4)}</p>
    </div>
  );
}

/** Factory for tiny boundary dot renderers (60×60 grid → 3600 points). */
function TinyDot(color: string) {
  return function renderTinyDot(props: unknown) {
    const { cx, cy } = props as { cx?: number; cy?: number };
    if (cx === undefined || cy === undefined) return <g />;
    return <circle cx={cx} cy={cy} r={1.5} fill={color} fillOpacity={0.3} />;
  };
}

/** Factory for the larger moon-class dots. */
function MoonDot(color: string) {
  return function renderMoonDot(props: unknown) {
    const { cx, cy } = props as { cx?: number; cy?: number };
    if (cx === undefined || cy === undefined) return <g />;
    return <circle cx={cx} cy={cy} r={3.2} fill={color} fillOpacity={0.9} stroke={color} strokeWidth={0.6} />;
  };
}

/** Qualitative label for a Pearson correlation coefficient. */
function corrStrength(r: number): string {
  const a = Math.abs(r);
  if (a >= 0.9) return "very strong";
  if (a >= 0.7) return "strong";
  if (a >= 0.5) return "moderate";
  return "weak";
}

/** Compact "time ago" label for run-history rows (client-side only). */
function timeAgo(iso: string): string {
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return "";
  const s = Math.max(0, Math.floor((Date.now() - t) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

/* ------------------------------------------------------------------ */
/* HSVR trust-radius scatter                                           */
/* ------------------------------------------------------------------ */

/** The scatter block of a train response (kept as a type alias so the
 *  component's contract stays tied to the API payload). */
type HsvrScatterData = NonNullable<NonNullable<MLTrainResponse["hsvr"]>["scatter"]>;

/** Cluster palette (cycles past five clusters — stone-safe, no blue). */
const CLUSTER_COLORS = [CHART.emerald, CHART.rose, CHART.amber, CHART.teal, CHART.stone];

/** Cluster colour for index i (cycled). */
function clusterColor(i: number): string {
  return CLUSTER_COLORS[i % CLUSTER_COLORS.length]!;
}

/**
 * The HSVR trust-radius geometry plot — every train row as a dot coloured
 * by its k-means cluster, centroid diamonds, and the dashed trust regions
 * (a vertical band per cluster for single-feature models where dimB is
 * the target; an axis-scaled ellipse otherwise). Hand-rolled SVG so the
 * trust geometry renders exactly at data scale. With `showBlend` (single
 * feature only), each cluster's blend-weight tent w(z) = 1 − |z − c| / R
 * rides on a secondary right axis (0→1) — the linear fade the prediction
 * actually blends through.
 */
function HsvrTrustScatter({
  scatter,
  showBlend,
}: {
  scatter: HsvrScatterData;
  showBlend: boolean;
}) {
  const W = 560;
  const H = 300;
  const overlay = showBlend && scatter.bIsTarget;
  const pad = { l: 46, r: overlay ? 34 : 14, top: 14, b: 34 };
  const pw = W - pad.l - pad.r;
  const ph = H - pad.top - pad.b;

  /* Extents from the points (5% breathing room on both sides). */
  const as = scatter.points.map((p) => p.a);
  const bs = scatter.points.map((p) => p.b);
  let aMin = Math.min(...as, ...scatter.centroids.map((c) => c.a));
  let aMax = Math.max(...as, ...scatter.centroids.map((c) => c.a));
  let bMin = Math.min(...bs, ...scatter.centroids.map((c) => c.b));
  let bMax = Math.max(...bs, ...scatter.centroids.map((c) => c.b));
  if (!Number.isFinite(aMin) || !Number.isFinite(aMax)) {
    aMin = 0;
    aMax = 1;
  }
  if (!Number.isFinite(bMin) || !Number.isFinite(bMax)) {
    bMin = 0;
    bMax = 1;
  }
  if (aMax - aMin < 1e-9) {
    aMin -= 0.5;
    aMax += 0.5;
  }
  if (bMax - bMin < 1e-9) {
    bMin -= 0.5;
    bMax += 0.5;
  }
  const aPad = (aMax - aMin) * 0.05;
  const bPad = (bMax - bMin) * 0.05;
  aMin -= aPad;
  aMax += aPad;
  bMin -= bPad;
  bMax += bPad;

  const sx = pw / (aMax - aMin);
  const sy = ph / (bMax - bMin);
  const px = (a: number) => pad.l + (a - aMin) * sx;
  const py = (b: number) => pad.top + ph - (b - bMin) * sy;
  /* Blend-weight secondary axis (overlay only): w=1 at the frame top,
   * w=0 at the bottom — the tents are explanations, not target-scale
   * series, so the right axis carries their own 0→1 scale. */
  const pyw = (w: number) => pad.top + ph - w * ph;

  /* Four gridlines + tick labels per axis. */
  const aTicks = Array.from({ length: 5 }, (_, i) => aMin + ((aMax - aMin) * i) / 4);
  const bTicks = Array.from({ length: 5 }, (_, i) => bMin + ((bMax - bMin) * i) / 4);
  const nClusters = scatter.centroids.length;

  return (
    <div className="w-full">
      <div className="w-full" style={{ aspectRatio: `${W} / ${H}` }}>
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="h-full w-full overflow-visible"
          role="img"
          aria-label={`HSVR trust-radius scatter: ${scatter.points.length} train rows in ${scatter.dimA} × ${scatter.dimB} coordinates, coloured by k-means cluster, with dashed trust regions${overlay ? " and per-cluster blend-weight tents on a secondary 0–1 axis" : ""}`}
        >
          {/* Grid + ticks */}
          {aTicks.map((v, i) => (
            <g key={`ga-${i}`}>
              <line
                x1={px(v)}
                y1={pad.top}
                x2={px(v)}
                y2={pad.top + ph}
                stroke={CHART.gridStroke}
                strokeOpacity={CHART.gridStrokeOpacity}
              />
              <text
                x={px(v)}
                y={pad.top + ph + 14}
                textAnchor="middle"
                className="fill-muted-foreground"
                style={{ fontSize: 10, fontFamily: "var(--font-mono, monospace)" }}
              >
                {fmtNum(v, 2)}
              </text>
            </g>
          ))}
          {bTicks.map((v, i) => (
            <g key={`gb-${i}`}>
              <line
                x1={pad.l}
                y1={py(v)}
                x2={pad.l + pw}
                y2={py(v)}
                stroke={CHART.gridStroke}
                strokeOpacity={CHART.gridStrokeOpacity}
              />
              <text
                x={pad.l - 6}
                y={py(v) + 3}
                textAnchor="end"
                className="fill-muted-foreground"
                style={{ fontSize: 10, fontFamily: "var(--font-mono, monospace)" }}
              >
                {fmtNum(v, 2)}
              </text>
            </g>
          ))}
          {/* Frame */}
          <rect
            x={pad.l}
            y={pad.top}
            width={pw}
            height={ph}
            fill="none"
            stroke={CHART.gridStroke}
            strokeOpacity={0.4}
          />
          {/* Trust regions (clipped to the frame). */}
          <clipPath id="hsvr-frame">
            <rect x={pad.l} y={pad.top} width={pw} height={ph} />
          </clipPath>
          <g clipPath="url(#hsvr-frame)">
            {scatter.centroids.map((c, i) => {
              const color = clusterColor(i);
              return scatter.bIsTarget ? (
                /* Single feature: the radius is a 1-D distance → band. */
                <g key={`trust-${i}`}>
                  <rect
                    x={px(c.a - c.radius)}
                    y={pad.top}
                    width={Math.max(0, px(c.a + c.radius) - px(c.a - c.radius))}
                    height={ph}
                    fill={color}
                    fillOpacity={0.08}
                  />
                  <line
                    x1={px(c.a - c.radius)}
                    y1={pad.top}
                    x2={px(c.a - c.radius)}
                    y2={pad.top + ph}
                    stroke={color}
                    strokeWidth={1}
                    strokeDasharray="4 3"
                    strokeOpacity={0.7}
                  />
                  <line
                    x1={px(c.a + c.radius)}
                    y1={pad.top}
                    x2={px(c.a + c.radius)}
                    y2={pad.top + ph}
                    stroke={color}
                    strokeWidth={1}
                    strokeDasharray="4 3"
                    strokeOpacity={0.7}
                  />
                </g>
              ) : (
                /* Multi-feature: the Euclidean trust sphere under the
                 *  axis scaling renders as an ellipse. */
                <ellipse
                  key={`trust-${i}`}
                  cx={px(c.a)}
                  cy={py(c.b)}
                  rx={Math.max(0, c.radius * sx)}
                  ry={Math.max(0, c.radius * sy)}
                  fill={color}
                  fillOpacity={0.06}
                  stroke={color}
                  strokeWidth={1}
                  strokeDasharray="4 3"
                  strokeOpacity={0.7}
                />
              );
            })}
          </g>
          {/* Blend-weight tents (single feature: the tent along the
           * feature axis IS the weight function w(z)). Clipped like the
           * trust regions; drawn under the train rows. */}
          {overlay ? (
            <g clipPath="url(#hsvr-frame)">
              {scatter.centroids.map((c, i) => (
                <polyline
                  key={`blend-${i}`}
                  points={`${px(c.a - c.radius)},${pyw(0)} ${px(c.a)},${pyw(1)} ${px(
                    c.a + c.radius
                  )},${pyw(0)}`}
                  fill="none"
                  stroke={clusterColor(i)}
                  strokeWidth={1.75}
                  strokeLinejoin="round"
                  strokeOpacity={0.85}
                >
                  <title>{`cluster #${i + 1} · w(z) = 1 − |z − centroid| / R — 1 at the centroid, 0 at R = ${fmtNum(
                    c.radius,
                    3
                  )}`}</title>
                </polyline>
              ))}
            </g>
          ) : null}
          {/* Train rows, coloured by cluster. */}
          {scatter.points.map((p, i) => {
            /* Exact local weight for the hover title: 1-D for
             * single-feature models, 2-D when the plot IS the feature
             * space; projected (k > 2) models keep the honest
             * coordinates-only title. */
            const cen = scatter.centroids[p.cluster];
            let w: number | null = null;
            if (cen) {
              if (scatter.bIsTarget) {
                w = Math.max(0, 1 - Math.abs(p.a - cen.a) / cen.radius);
              } else if (!scatter.projected) {
                const d = Math.sqrt((p.a - cen.a) ** 2 + (p.b - cen.b) ** 2);
                w = Math.max(0, 1 - d / cen.radius);
              }
            }
            return (
              <circle
                key={`pt-${i}`}
                cx={px(p.a)}
                cy={py(p.b)}
                r={2.8}
                fill={clusterColor(p.cluster)}
                fillOpacity={0.85}
                stroke={clusterColor(p.cluster)}
                strokeWidth={0.5}
              >
                <title>{`cluster #${p.cluster + 1} · ${scatter.dimA} ${fmtNum(
                  p.a,
                  3
                )} · ${scatter.dimB} ${fmtNum(p.b, 3)}${
                  w !== null ? ` · local weight w = ${fmtNum(w, 2)}` : ""
                }`}</title>
              </circle>
            );
          })}
          {/* Centroid diamonds. */}
          {scatter.centroids.map((c, i) => (
            <g key={`cen-${i}`}>
              <rect
                x={px(c.a) - 4.5}
                y={py(c.b) - 4.5}
                width={9}
                height={9}
                transform={`rotate(45 ${px(c.a)} ${py(c.b)})`}
                fill="var(--background, #fff)"
                stroke={clusterColor(i)}
                strokeWidth={1.6}
              >
                <title>{`centroid #${i + 1} · trust radius ${fmtNum(c.radius, 3)}`}</title>
              </rect>
            </g>
          ))}
          {/* Axis labels */}
          <text
            x={pad.l + pw / 2}
            y={H - 6}
            textAnchor="middle"
            className="fill-muted-foreground"
            style={{ fontSize: 10.5 }}
          >
            {scatter.dimA}
          </text>
          <text
            x={pad.l - 34}
            y={pad.top + 4}
            style={{ fontSize: 10.5 }}
            className="fill-muted-foreground"
          >
            {scatter.dimB.length > 22 ? `${scatter.dimB.slice(0, 21)}…` : scatter.dimB}
          </text>
          {/* Secondary blend-weight axis (overlay only): ticks at
           * w = 0 / 0.5 / 1 on the right edge + the mini "w" title. */}
          {overlay ? (
            <g>
              {[0, 0.5, 1].map((w) => (
                <g key={`wt-${w}`}>
                  <line
                    x1={pad.l + pw}
                    y1={pyw(w)}
                    x2={pad.l + pw + 4}
                    y2={pyw(w)}
                    stroke={CHART.gridStroke}
                    strokeOpacity={0.7}
                  />
                  <text
                    x={pad.l + pw + 6}
                    y={pyw(w) + 3}
                    textAnchor="start"
                    className="fill-muted-foreground"
                    style={{
                      fontSize: 9,
                      fontFamily: "var(--font-mono, monospace)",
                    }}
                  >
                    {w === 0.5 ? "0.5" : `${w}`}
                  </text>
                </g>
              ))}
              <text
                x={pad.l + pw + 6}
                y={pad.top - 3}
                textAnchor="start"
                className="fill-muted-foreground"
                style={{ fontSize: 9.5, fontStyle: "italic" }}
              >
                w
              </text>
            </g>
          ) : null}
        </svg>
      </div>
      {/* Legend */}
      <div className="mt-1.5 flex flex-wrap items-center gap-x-3.5 gap-y-1 px-1">
        {Array.from({ length: nClusters }, (_, i) => (
          <span
            key={`lg-${i}`}
            className="flex items-center gap-1.5 text-[11px] text-muted-foreground"
          >
            <span
              className="size-2 rounded-full"
              style={{ backgroundColor: clusterColor(i) }}
              aria-hidden
            />
            cluster {i + 1}
          </span>
        ))}
        <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
          <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden>
            <rect
              x="1.5"
              y="1.5"
              width="7"
              height="7"
              transform="rotate(45 5 5)"
              fill="var(--background, #fff)"
              stroke={CHART.stone}
              strokeWidth="1.4"
            />
          </svg>
          centroid
        </span>
        <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
          <svg width="18" height="8" viewBox="0 0 18 8" aria-hidden>
            <line
              x1="1"
              y1="4"
              x2="17"
              y2="4"
              stroke={CHART.stone}
              strokeWidth="1"
              strokeDasharray="4 3"
            />
          </svg>
          trust radius
        </span>
        {overlay ? (
          <span className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
            <svg width="18" height="10" viewBox="0 0 18 10" aria-hidden>
              <polyline
                points="1,9 9,1 17,9"
                fill="none"
                stroke={CHART.stone}
                strokeWidth="1.5"
                strokeLinejoin="round"
              />
            </svg>
            blend weight w(z)
          </span>
        ) : null}
      </div>
    </div>
  );
}

export function MlPlayground() {
  /* Experiment setup + results live in the store, so the whole playground
   * (config, trained model, dataset preview) survives tab switches. */
  const mlConfig = useToolkitStore((s) => s.mlConfig);
  const setMlConfig = useToolkitStore((s) => s.setMlConfig);
  const resetMlConfig = useToolkitStore((s) => s.resetMlConfig);
  const result = useToolkitStore((s) => s.mlResult);
  const setResult = useToolkitStore((s) => s.setMlResult);
  const preview = useToolkitStore((s) => s.mlPreview);
  const setPreview = useToolkitStore((s) => s.setMlPreview);
  const mlHistory = useToolkitStore((s) => s.mlHistory);
  const pushMlRun = useToolkitStore((s) => s.pushMlRun);
  const clearMlHistory = useToolkitStore((s) => s.clearMlHistory);
  const mlCurrentRunId = useToolkitStore((s) => s.mlCurrentRunId);
  const setMlCurrentRun = useToolkitStore((s) => s.setMlCurrentRun);
  const mlCsv = useToolkitStore((s) => s.mlCsv);
  const setMlCsv = useToolkitStore((s) => s.setMlCsv);

  const {
    task,
    source,
    samples,
    noise,
    regModel,
    alpha,
    l2,
    testSizePct,
    cv,
    seed,
    featureColumn,
    featureColumn2,
    featureColumns,
    targetColumn,
  } = mlConfig;

  /* Model selection + slider hyperparameters are component-local (the
   * persisted store only carries the legacy trio + α/λ strings). They are
   * re-derived from the persisted training result — which echoes the model
   * name and its hyperparameters — so a reload or run-restore puts the
   * pickers back into the trained configuration. */
  const [modelSel, setModelSel] = useState<string>(() => {
    if (result && result.task === mlConfig.task && specOf(mlConfig.task, result.model)) {
      return result.model;
    }
    return mlConfig.task === "regression"
      ? specOf("regression", mlConfig.regModel)
        ? mlConfig.regModel
        : "linear"
      : "logistic";
  });
  const [hp, setHp] = useState<Record<string, number>>(() => ({
    ...DEFAULT_HP,
    ...(result?.modelParams ?? {}),
  }));

  const modelSpec = specOf(task, modelSel) ?? REG_MODELS[0]!;
  const specs = task === "regression" ? REG_MODELS : CLS_MODELS;

  /** Select a model, reset the slider hyperparameters, and drop any
   *  leaderboard that belonged to the previous model. */
  function selectModel(id: string) {
    setModelSel(id);
    setHp({ ...DEFAULT_HP });
    setTuneStatus("idle");
    setTuneResult(null);
    setTuneError(null);
  }

  /* Multivariate mode: regression result with ≥2 standardized features
   * (uploaded CSV only) → the chart switches to predicted-vs-actual. */
  const predActual =
    result !== null &&
    result.task === "regression" &&
    result.datasetMeta.features.length > 1;

  /* Transient request state — not worth persisting across tab switches. */
  const [status, setStatus] = useState<"idle" | "running" | "done" | "error">(
    "idle"
  );
  const [error, setError] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [csvLoading, setCsvLoading] = useState(false);
  const [previewOpen, setPreviewOpen] = useState(true);

  /* X-range zoom on the preview scatter — null = full range. */
  const [xZoom, setXZoom] = useState<[number, number] | null>(null);

  /* Tuner state — transient, mirroring ml tune on the server. maxTrials
   * rides in state so the budget is visible before searching; the searcher
   * toggle picks RandomSearch vs Hyperband (successive halving). */
  const [tuneStatus, setTuneStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [tuneResult, setTuneResult] = useState<MLTuneResponse | null>(null);
  const [tuneError, setTuneError] = useState<string | null>(null);
  const [maxTrials, setMaxTrials] = useState<string>("12");
  const [searcher, setSearcher] = useState<"random" | "hyperband">("random");
  const [etaSel, setEtaSel] = useState<string>("3");
  /* Trust-radius card: whether the HSVR blend-weight tents ride on the
   * scatter (single-feature models only — the fade depends on the whole
   * feature vector otherwise). */
  const [showBlend, setShowBlend] = useState(true);
  const tuning = tuneStatus === "running";
  const tuneable = (task === "regression" ? TUNEABLE_REG : TUNEABLE_CLS).has(modelSel);
  const hyperbandable = HYPERBAND_MODELS.has(modelSel);
  const resourceSpec = RESOURCE_SPECS[modelSel];

  /* Uploaded-CSV state: drag feedback, paste area, hidden file input. */
  const [dragOver, setDragOver] = useState(false);
  const [pasteOpen, setPasteOpen] = useState(false);
  const [pasteText, setPasteText] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);

  /* The loaded CSV parsed once per file (memoized) + client-side sanity. */
  const csvTable = useMemo(() => {
    if (!mlCsv) return null;
    try {
      const table = parseCsv(mlCsv.text);
      return table.columns.length > 0 && table.rows.length > 0 ? table : null;
    } catch {
      return null;
    }
  }, [mlCsv]);
  const csvNumericCols = useMemo(
    () => (csvTable ? numericColumnNames(csvTable) : []),
    [csvTable]
  );
  const csvUsable =
    csvTable !== null &&
    csvTable.rows.length >= MIN_CSV_ROWS &&
    csvNumericCols.length >= 2;

  /* Effective target for chip exclusion: explicit pick, or the auto rule
   * (last numeric column) the server applies when none is picked. */
  const effectiveTarget =
    targetColumn || csvNumericCols[csvNumericCols.length - 1] || "";

  /** Classification preview: the sampled points split by class for the chart. */
  const previewClasses = useMemo(() => {
    if (!preview || preview.task !== "classification") return null;
    return {
      cls0: preview.points.filter((p) => p.label === 0),
      cls1: preview.points.filter((p) => p.label === 1),
    };
  }, [preview]);

  /** Windowed view of the preview dataset (a true brush): the x-window
   *  filters the series data AND re-derives both axis domains, so zooming
   *  rescales the plot instead of merely clipping it. */
  const previewView = useMemo(() => {
    if (!preview || preview.points.length === 0) return null;
    const xs = preview.points.map((p) => p.x);
    const fullX: [number, number] = [Math.min(...xs), Math.max(...xs)];
    const yPad = preview.task === "regression" ? 0.4 : 0.3;
    const zoomed = xZoom !== null && (xZoom[0] > fullX[0] || xZoom[1] < fullX[1]);
    const win: [number, number] = xZoom ?? fullX;

    const inside =
      zoomed ? preview.points.filter((p) => p.x >= win[0] && p.x <= win[1]) : preview.points;
    const ySource = inside.length > 0 ? inside : preview.points;
    const ys = ySource.map((p) => p.y);
    // Pad the window slightly so edge dots aren't cut by the plot boundary.
    const xPad = zoomed ? Math.max((win[1] - win[0]) * 0.04, 0.05) : yPad;

    const inWin = (p: { x: number }) => !zoomed || (p.x >= win[0] && p.x <= win[1]);
    return {
      fullX,
      win,
      zoomed,
      xDomain: [win[0] - xPad, win[1] + xPad] as [number, number],
      yDomain: [Math.min(...ys) - yPad, Math.max(...ys) + yPad] as [number, number],
      points: inside,
      cls0: previewClasses ? previewClasses.cls0.filter(inWin) : [],
      cls1: previewClasses ? previewClasses.cls1.filter(inWin) : [],
      trend: (preview.trend ?? []).filter(inWin),
    };
  }, [preview, previewClasses, xZoom]);

  /** Export the full deterministic dataset (all rows) as a CSV file. */
  async function downloadCsv() {
    if (!seedValid) {
      setError("Seed must be a non-negative integer (0 – 2147483647).");
      return;
    }
    setCsvLoading(true);
    setError(null);
    const seedNum = Number.parseInt(seed, 10) || 0;
    const fileName = `pharmaml_${task}_n${samples}_seed${seedNum}.csv`;
    try {
      const qs = new URLSearchParams({
        task,
        samples: String(samples),
        noise: String(noise),
        seed: String(seedNum),
      });
      const res = await fetch(`/api/ml/dataset/csv?${qs.toString()}`);
      if (!res.ok) {
        let message = `HTTP ${res.status}`;
        try {
          const data: unknown = await res.json();
          if (typeof data === "object" && data !== null && "error" in data) {
            message = String((data as { error: unknown }).error);
          }
        } catch {
          /* body wasn't JSON — keep the status message */
        }
        throw new Error(message);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = fileName;
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      URL.revokeObjectURL(url);
      toast.success("Dataset exported", {
        description: `${fileName} · ${samples} rows · ${fmtBytes(blob.size)}`,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "CSV export failed.");
    } finally {
      setCsvLoading(false);
    }
  }

  const noiseMax = task === "regression" ? 2 : 1.5;
  const metricSpecs = task === "regression" ? REG_METRICS : CLS_METRICS;

  function switchTask(next: Task) {
    setMlConfig({ task: next });
    if (next === "classification" && noise > 1.5) setMlConfig({ noise: 0.3 });
    /* Keep the pick when the model exists for both tasks (decision_tree,
     * random_forest, knn, mlp); otherwise fall back to the task default. */
    if (!specOf(next, modelSel)) {
      setModelSel(next === "regression" ? (specOf("regression", regModel) ? regModel : "linear") : "logistic");
      setHp({ ...DEFAULT_HP });
    }
    setStatus("idle");
    setResult(null);
    setError(null);
    setPreview(null);
    setXZoom(null);
    setTuneStatus("idle");
    setTuneResult(null);
    setTuneError(null);
  }

  /** Restore the default experiment setup and drop any stored results. */
  function resetSetup() {
    resetMlConfig();
    setModelSel("linear");
    setHp({ ...DEFAULT_HP });
    setStatus("idle");
    setResult(null);
    setPreview(null);
    setError(null);
    setXZoom(null);
    setTuneStatus("idle");
    setTuneResult(null);
    setTuneError(null);
  }

  /* ------------------------- CSV data loading ------------------------- */

  /** Load CSV text into the store, reset column picks, drop stale results.
   *  Returns false (with a toast) when the text is unusable. */
  function loadCsv(fileName: string, text: string): boolean {
    if (text.trim().length < 4 || text.length > MAX_CSV_TEXT) {
      toast.error("CSV not loaded", {
        description: "Expected a non-empty file of at most 256 KB.",
      });
      return false;
    }
    const table = parseCsv(text);
    if (table.columns.length === 0 || table.rows.length === 0) {
      toast.error("Could not parse the CSV", {
        description: "Expected a header row followed by data rows.",
      });
      return false;
    }
    setMlCsv({ fileName, text });
    setMlConfig({ featureColumn: "", featureColumn2: "", targetColumn: "" });
    setStatus("idle");
    setResult(null);
    setPreview(null);
    setError(null);
    setXZoom(null);
    toast.success("CSV loaded", {
      description: `${fileName} — ${table.rows.length} rows × ${table.columns.length} columns · ${numericColumnNames(table).length} numeric`,
    });
    return true;
  }

  /** Read a picked/dropped File as text and load it. */
  function handleFile(file: File) {
    if (file.size > MAX_CSV_TEXT) {
      toast.error("File too large", {
        description: `CSV uploads are capped at 256 KB — this file is ${fmtBytes(file.size)}.`,
      });
      return;
    }
    const reader = new FileReader();
    reader.onload = () => loadCsv(file.name, String(reader.result ?? ""));
    reader.onerror = () =>
      toast.error("Could not read the file", {
        description: "Try re-saving it as a plain UTF-8 CSV.",
      });
    reader.readAsText(file);
  }

  /** Load the pasted textarea contents as "pasted.csv". */
  function loadPasted() {
    const text = pasteText.trim();
    if (text.length >= 4 && loadCsv("pasted.csv", text)) {
      setPasteOpen(false);
      setPasteText("");
    }
  }

  /** Drop the loaded CSV and return to the empty upload state. */
  function removeCsv() {
    setMlCsv(null);
    setMlConfig({ featureColumn: "", featureColumn2: "", targetColumn: "" });
    setStatus("idle");
    setResult(null);
    setPreview(null);
    setError(null);
    setXZoom(null);
  }

  /** Switch the data source; stale results/previews are dropped. */
  function switchSource(next: "synthetic" | "csv") {
    if (source === next) return;
    setMlConfig({ source: next });
    setStatus("idle");
    setResult(null);
    setPreview(null);
    setError(null);
    setXZoom(null);
  }

  /** Toggle a feature chip for multivariate regression (capped at 8). */
  function toggleFeature(col: string) {
    const next = featureColumns.includes(col)
      ? featureColumns.filter((c) => c !== col)
      : [...featureColumns, col].slice(0, MAX_CSV_FEATURES);
    setMlConfig({ featureColumns: next });
  }

  const seedValid = /^\d{1,10}$/.test(seed.trim());
  const alphaValid = Number.isFinite(Number.parseFloat(alpha)) && Number.parseFloat(alpha) >= 0;
  const l2Valid = Number.isFinite(Number.parseFloat(l2)) && Number.parseFloat(l2) >= 0;

  async function run() {
    if (source === "synthetic" && !seedValid) {
      setError("Seed must be a non-negative integer (0 – 2147483647).");
      return;
    }
    if (source === "csv" && (!mlCsv || !csvUsable)) {
      setError(
        `Load a CSV with at least ${MIN_CSV_ROWS} rows and 2 numeric columns before training.`
      );
      return;
    }
    const needsAlpha = modelSpec.hp.some((h) => h.key === "alpha");
    if (task === "regression" && needsAlpha && !alphaValid) {
      setError("Alpha must be a number ≥ 0.");
      return;
    }
    const needsL2 = modelSpec.hp.some((h) => h.key === "l2");
    if (task === "classification" && needsL2 && !l2Valid) {
      setError("L2 strength must be a number ≥ 0.");
      return;
    }
    setStatus("running");
    setError(null);
    const body: MLTrainRequest = {
      task,
      samples,
      noise,
      model: modelSel,
      testSize: testSizePct / 100,
      cv: (Number.parseInt(cv, 10) || 0) as 0 | 3 | 5 | 10,
      seed: Number.parseInt(seed, 10) || 0,
    };
    /* Contextual hyperparameters: α/λ come from the legacy input fields,
     * everything else from the slider state. */
    for (const h of modelSpec.hp) {
      if (h.control === "input") {
        if (h.key === "alpha") body.alpha = Number.parseFloat(alpha);
        else if (h.key === "l2") body.l2 = Number.parseFloat(l2);
      } else {
        assignHp(body, h.key, hp[h.key] ?? DEFAULT_HP[h.key] ?? 0);
      }
    }
    if (source === "csv" && mlCsv) {
      body.csv = mlCsv.text;
      body.csvName = mlCsv.fileName;
      if (task === "regression") {
        if (featureColumns.length > 0) body.featureColumns = featureColumns;
        if (targetColumn) body.targetColumn = targetColumn;
      } else {
        if (featureColumn) body.featureColumn = featureColumn;
        if (featureColumn2) body.featureColumn2 = featureColumn2;
        if (targetColumn) body.targetColumn = targetColumn;
      }
    }

    try {
      const res = await fetch("/api/ml/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data: unknown = await res.json();
      if (!res.ok) {
        const message =
          typeof data === "object" && data !== null && "error" in data
            ? `${String((data as { error: unknown }).error)}${
                "detail" in data && Array.isArray((data as { detail: unknown }).detail)
                  ? ` ${((data as { detail: unknown[] }).detail as string[]).join("; ")}`
                  : ""
              }`
            : `HTTP ${res.status}`;
        throw new Error(message);
      }
      const payload = data as MLTrainResponse;
      setResult(payload);
      setStatus("done");
      /* Record the completed run in the history strip (restorable). */
      const key = payload.task === "regression" ? "r2" : "accuracy";
      const keyValue = payload.metrics[key];
      if (typeof keyValue === "number" && Number.isFinite(keyValue)) {
        pushMlRun({
          id: Date.now(),
          at: new Date().toISOString(),
          task: payload.task,
          model: payload.model,
          keyMetric: payload.task === "regression" ? "R²" : "Accuracy",
          keyValue,
          config: { ...mlConfig },
          result: payload,
          ...(source === "csv" && mlCsv
            ? {
                csv: {
                  fileName: mlCsv.fileName,
                  featureColumns:
                    task === "regression"
                      ? featureColumns
                      : [featureColumn, featureColumn2].filter(Boolean),
                  targetColumn,
                },
              }
            : {}),
        });
      }
    } catch (err) {
      setStatus("error");
      const message = err instanceof Error ? err.message : "Training request failed.";
      setError(message);
    }
  }

  /** Run the hyperparameter search over the selected model's discrete
   *  grid — the web mirror of `ml tune --searcher random|hyperband`. Each
   *  trial is scored with the same k-fold CV the server uses, so the
   *  leaderboard is directly comparable to the Train metrics. */
  async function runTune() {
    if (source === "synthetic" && !seedValid) {
      setTuneError("Seed must be a non-negative integer (0 – 2147483647).");
      return;
    }
    if (source === "csv" && (!mlCsv || !csvUsable)) {
      setTuneError(
        `Load a CSV with at least ${MIN_CSV_ROWS} rows and 2 numeric columns before tuning.`
      );
      return;
    }
    if (!tuneable) {
      setTuneError(
        `${modelSpec.label} has no tunable hyperparameters — pick a model with a search space.`
      );
      return;
    }
    if (searcher === "hyperband" && !hyperbandable) {
      setTuneError(
        `Hyperband needs a resource hyperparameter — ${modelSpec.label} has none. Pick the neural net (epochs), random forest (estimators) or gradient boosting (rounds), or switch to RandomSearch.`
      );
      return;
    }
    setTuneStatus("running");
    setTuneError(null);
    const body: MLTuneRequest = {
      task,
      samples,
      noise,
      model: modelSel,
      maxTrials: Number.parseInt(maxTrials, 10) || 12,
      searcher,
      ...(searcher === "hyperband"
        ? { eta: (Number.parseInt(etaSel, 10) as 2 | 3 | 4) || 3 }
        : {}),
      /* The tuner always cross-validates; CV=Off falls back to 5-fold. */
      cv: (Number.parseInt(cv, 10) >= 3 ? Number.parseInt(cv, 10) : 5) as 3 | 5 | 10,
      seed: Number.parseInt(seed, 10) || 0,
      testSize: testSizePct / 100,
    };
    if (source === "csv" && mlCsv) {
      body.csv = mlCsv.text;
      body.csvName = mlCsv.fileName;
      if (task === "regression") {
        if (featureColumns.length > 0) body.featureColumns = featureColumns;
        if (targetColumn) body.targetColumn = targetColumn;
      } else {
        if (featureColumn) body.featureColumn = featureColumn;
        if (featureColumn2) body.featureColumn2 = featureColumn2;
        if (targetColumn) body.targetColumn = targetColumn;
      }
    }
    try {
      const res = await fetch("/api/ml/tune", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data: unknown = await res.json();
      if (!res.ok) {
        const message =
          typeof data === "object" && data !== null && "error" in data
            ? `${String((data as { error: unknown }).error)}${
                "detail" in data && Array.isArray((data as { detail: unknown }).detail)
                  ? ` ${((data as { detail: unknown[] }).detail as string[]).join("; ")}`
                  : ""
              }`
            : `HTTP ${res.status}`;
        throw new Error(message);
      }
      const payload = data as MLTuneResponse;
      setTuneResult(payload);
      setTuneStatus("done");
      toast.success(
        payload.searcher === "hyperband" ? "Hyperband search complete" : "Search complete",
        {
          description:
            payload.searcher === "hyperband" && payload.hyperband
              ? `${payload.trials.length} trials · η ${payload.hyperband.eta} · ${
                  payload.hyperband.ladder.length
                } rungs · best ${payload.objective} ${fmtNum(payload.best.cvScore, 4)} · ${
                  Math.round(payload.hyperband.budget.savedPct * 100)
                }% budget saved`
              : `${payload.trials.length} trials · best ${payload.objective} ${fmtNum(
                  payload.best.cvScore,
                  4
                )} (${payload.best.improvement >= 0 ? "+" : ""}${fmtNum(
                  payload.best.improvement,
                  4
                )} vs defaults)`,
        }
      );
    } catch (err) {
      setTuneStatus("error");
      setTuneResult(null);
      setTuneError(err instanceof Error ? err.message : "Tuning request failed.");
    }
  }

  /** Copy the winning configuration into the setup panel (sliders + α/λ
   *  input fields), ready for a fresh Train-model run. */
  function applyBestParams() {
    if (!tuneResult) return;
    const p = tuneResult.best.params;
    setHp((prev) => ({ ...prev, ...p }));
    if (typeof p.alpha === "number") setMlConfig({ alpha: String(p.alpha) });
    if (typeof p.l2 === "number") setMlConfig({ l2: String(p.l2) });
    toast.success("Best parameters applied", {
      description: Object.entries(p)
        .map(([k, v]) => `${HP_LABELS[k] ?? k} ${fmtHp(v)}`)
        .join(" · "),
    });
  }

  /** Restore a previous run: config snapshot + full result + charts. The
   *  model pickers are re-derived from the result (model name + echoed
   *  hyperparameters), so the setup panel reflects the restored run. */
  function restoreRun(rec: MlRunRecord) {
    setMlConfig(rec.config);
    setResult(rec.result);
    if (specOf(rec.result.task, rec.result.model)) {
      setModelSel(rec.result.model);
      setHp({ ...DEFAULT_HP, ...(rec.result.modelParams ?? {}) });
    }
    setMlCurrentRun(rec.id);
    setStatus("done");
    setError(null);
    setXZoom(null);
    toast.success("Run restored", {
      description: `${rec.model} · ${rec.csv ? rec.csv.fileName : `seed ${rec.config.seed}`} · ${
        rec.keyMetric
      } ${fmtNum(rec.keyValue, 4)}`,
    });
  }  /** Fetch the first rows + summary stats of the (deterministic) dataset. */
  async function previewDataset() {
    if (source === "synthetic" && !seedValid) {
      setError("Seed must be a non-negative integer (0 – 2147483647).");
      return;
    }
    if (source === "csv" && (!mlCsv || !csvUsable)) {
      setError(
        `Load a CSV with at least ${MIN_CSV_ROWS} rows and 2 numeric columns before previewing.`
      );
      return;
    }
    setPreviewLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/ml/dataset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          task,
          samples,
          noise,
          seed: Number.parseInt(seed, 10) || 0,
          ...(source === "csv" && mlCsv
            ? {
                csv: mlCsv.text,
                csvName: mlCsv.fileName,
                ...(task === "regression"
                  ? {
                      ...(featureColumns.length > 0 ? { featureColumns } : {}),
                      ...(targetColumn ? { targetColumn } : {}),
                    }
                  : {
                      ...(featureColumn ? { featureColumn } : {}),
                      ...(featureColumn2 ? { featureColumn2 } : {}),
                      ...(targetColumn ? { targetColumn } : {}),
                    }),
              }
            : {}),
        }),
      });
      const data: unknown = await res.json();
      if (!res.ok) {
        const message =
          typeof data === "object" && data !== null && "error" in data
            ? `${String((data as { error: unknown }).error)}${
                "detail" in data && Array.isArray((data as { detail: unknown }).detail)
                  ? ` ${((data as { detail: unknown[] }).detail as string[]).join("; ")}`
                  : ""
              }`
            : `HTTP ${res.status}`;
        throw new Error(message);
      }
      setPreview(data as DatasetPreviewResponse);
      setPreviewOpen(true);
      setXZoom(null); // fresh dataset — reset the x-range zoom
    } catch (err) {
      setError(err instanceof Error ? err.message : "Preview request failed.");
    } finally {
      setPreviewLoading(false);
    }
  }

  const running = status === "running";

  /* Chart series (memoized splits so recharts gets stable references). */
  const clsSeries = useMemo(() => {
    if (!result || result.task !== "classification") return null;
    const boundary = result.series.boundary ?? [];
    return {
      grid0: boundary.filter((p) => p.pred === 0),
      grid1: boundary.filter((p) => p.pred === 1),
      moon0: result.series.test.filter((p) => p.label === 0),
      moon1: result.series.test.filter((p) => p.label === 1),
    };
  }, [result]);

  const coefficientsText = useMemo(() => {
    if (!result || !result.coefficients) return null;
    const { intercept, coefs } = result.coefficients;
    if (result.task === "regression") {
      /* Multivariate: one weight per named feature (uploaded CSV mode). */
      const names = result.datasetMeta.source?.featureColumns;
      if (names && names.length > 1) {
        const terms = coefs
          .map((c, i) => `${fmtNum(c, 4)}·x̂${names[i] ? `[${names[i]}]` : String(i + 1)}`)
          .join(" + ");
        return `ŷ = ${fmtNum(intercept, 4)} + ${terms}   (x̂ = standardized features)`;
      }
      const terms = coefs
        .map((c, i) => `${fmtNum(c, 4)}·x̂${coefs.length > 1 ? `^${i + 1}` : ""}`)
        .join(" + ");
      return `ŷ = ${fmtNum(intercept, 4)} + ${terms}   (x̂ = standardized x)`;
    }
    return `σ(b₀ + w₁x̂₁ + w₂x̂₂)   b₀=${fmtNum(intercept, 4)}, w=[${coefs
      .map((c) => fmtNum(c, 4))
      .join(", ")}]`;
  }, [result]);

  /** Which honest parameter summary the "Learned parameters" card shows,
   *  keyed off the model name (which survives result persistence). */
  const resultSpec = useMemo(
    () => (result ? specOf(result.task, result.model) : null),
    [result]
  );
  const paramsPanel = useMemo(() => {
    if (!result || !resultSpec) return null;
    if (resultSpec.kind === "coefficients") {
      return result.coefficients ? { kind: "coefficients" as const } : null;
    }
    if (resultSpec.kind === "importance") {
      return { kind: "importance" as const, data: result.featureImportance ?? null };
    }
    return { kind: "params" as const, data: result.modelParams ?? null };
  }, [result, resultSpec]);

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(300px,360px)_minmax(0,1fr)]">
      {/* ------------------------------ Controls ------------------------------ */}
      <Card className="h-fit overflow-hidden py-0 shadow-sm xl:sticky xl:top-24">
        <CardHeader className="border-b bg-muted/30 py-3">
          <div className="flex items-center gap-2.5">
            <span className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <Brain className="size-4.5" aria-hidden />
            </span>
            <div>
              <CardTitle className="text-base">Experiment setup</CardTitle>
              <CardDescription className="text-xs">
                Seeded generator or your own CSV — setup & results survive reloads
              </CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-5 pb-4">
          {/* Task */}
          <div className="space-y-1.5">
            <Label className="text-xs font-medium">Task</Label>
            <div className="grid grid-cols-2 gap-1 rounded-lg border bg-muted/40 p-1" role="tablist" aria-label="ML task">
              {(["regression", "classification"] as const).map((t) => (
                <button
                  key={t}
                  type="button"
                  role="tab"
                  aria-selected={task === t}
                  onClick={() => switchTask(t)}
                  className={`h-9 rounded-md text-sm font-medium capitalize transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring ${
                    task === t ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {t}
                </button>
              ))}
            </div>
          </div>

          {/* Data source: seeded generator or the user's own CSV */}
          <div className="space-y-1.5">
            <Label className="text-xs font-medium">Data source</Label>
            <div className="grid grid-cols-2 gap-1 rounded-lg border bg-muted/40 p-1" role="tablist" aria-label="Data source">
              {(["synthetic", "csv"] as const).map((s) => (
                <button
                  key={s}
                  type="button"
                  role="tab"
                  aria-selected={source === s}
                  onClick={() => switchSource(s)}
                  className={`h-9 rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring ${
                    source === s ? "bg-background shadow-sm text-foreground" : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {s === "synthetic" ? "Synthetic" : "My CSV"}
                </button>
              ))}
            </div>
          </div>

          {/* Uploaded CSV dataset (drop / browse / paste + column picks) */}
          {source === "csv" ? (
            <div className="space-y-2.5">
              <Label className="text-xs font-medium">Your dataset</Label>
              {mlCsv ? (
                <div className="space-y-2.5">
                  <div className="rounded-lg border bg-muted/20 p-2.5">
                    <div className="flex items-center gap-2.5">
                      <span className="flex size-8 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
                        <FileSpreadsheet className="size-4" aria-hidden />
                      </span>
                      <div className="min-w-0 flex-1">
                        <p className="truncate font-mono text-xs font-medium" title={mlCsv.fileName}>
                          {mlCsv.fileName}
                        </p>
                        <p className="text-[11px] leading-snug text-muted-foreground">
                          {csvTable
                            ? `${csvTable.rows.length} rows × ${csvTable.columns.length} cols · ${csvNumericCols.length} numeric`
                            : "Unreadable — re-upload the file"}
                        </p>
                      </div>
                      <button
                        type="button"
                        onClick={removeCsv}
                        aria-label="Remove the loaded CSV"
                        title="Remove CSV"
                        className="flex size-7 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                      >
                        <X className="size-3.5" aria-hidden />
                      </button>
                    </div>
                    {csvTable && !csvUsable ? (
                      <p className="mt-2 rounded-md border border-destructive/25 bg-destructive/5 px-2 py-1.5 text-[11px] leading-snug text-destructive">
                        Needs ≥ {MIN_CSV_ROWS} rows and ≥ 2 numeric columns to train.
                      </p>
                    ) : null}
                  </div>

                  {csvTable ? (
                    task === "regression" ? (
                      <div className="space-y-2">
                        {/* Feature chips — multi-select, ≥2 picks = multivariate */}
                        <div className="space-y-1.5">
                          <div className="flex items-center justify-between">
                            <Label className="text-[11px] font-medium">Features (x)</Label>
                            <span className="font-mono text-[10px] tabular-nums text-muted-foreground">
                              {featureColumns.length > 0
                                ? `${featureColumns.length} picked · multi`
                                : "auto · all numeric"}
                            </span>
                          </div>
                          <div
                            role="group"
                            aria-label="Feature columns"
                            className="flex flex-wrap gap-1.5"
                          >
                            {csvNumericCols
                              .filter((c) => c !== effectiveTarget)
                              .map((col) => {
                                const active = featureColumns.includes(col);
                                return (
                                  <button
                                    key={col}
                                    type="button"
                                    onClick={() => toggleFeature(col)}
                                    aria-pressed={active}
                                    className={cn(
                                      "group inline-flex h-7 items-center gap-1 rounded-full border px-2.5 font-mono text-[11px] transition-all duration-150 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                                      active
                                        ? "border-primary/40 bg-primary/10 text-primary shadow-sm"
                                        : "border-border bg-muted/30 text-muted-foreground hover:border-primary/30 hover:bg-primary/5 hover:text-foreground"
                                    )}
                                    title={
                                      featureColumns.length >= MAX_CSV_FEATURES && !active
                                        ? `Maximum ${MAX_CSV_FEATURES} features`
                                        : `Toggle feature “${col}”`
                                    }
                                  >
                                    <span
                                      className={cn(
                                        "flex size-3.5 items-center justify-center rounded-full transition-colors",
                                        active
                                          ? "bg-primary text-primary-foreground"
                                          : "border border-current opacity-40 group-hover:opacity-70"
                                      )}
                                      aria-hidden
                                    >
                                      {active ? <Check className="size-2.5" /> : null}
                                    </span>
                                    {col}
                                  </button>
                                );
                              })}
                            {csvNumericCols.filter((c) => c !== effectiveTarget).length === 0 ? (
                              <p className="text-[11px] leading-snug text-muted-foreground">
                                No numeric column left besides the target — pick a different target.
                              </p>
                            ) : null}
                          </div>
                          <p className="text-[10px] leading-snug text-muted-foreground">
                            Pick 1 for a single-feature fit · pick ≥ 2 for multivariate
                            (≤ {MAX_CSV_FEATURES}) · none = all numeric columns
                          </p>
                        </div>
                        {/* Target */}
                        <div className="space-y-1">
                          <Label className="text-[11px] font-medium">Target (y)</Label>
                          <Select
                            value={targetColumn || "auto"}
                            onValueChange={(v) =>
                              setMlConfig({ targetColumn: v === "auto" ? "" : v })
                            }
                          >
                            <SelectTrigger className="h-9 text-xs" aria-label="Target column">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="auto">Auto · last numeric</SelectItem>
                              {csvNumericCols.map((c) => (
                                <SelectItem key={c} value={c} className="font-mono text-xs">
                                  {c}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <div className="space-y-1">
                          <Label className="text-[11px] font-medium">Feature x₁</Label>
                          <Select
                            value={featureColumn || "auto"}
                            onValueChange={(v) =>
                              setMlConfig({ featureColumn: v === "auto" ? "" : v })
                            }
                          >
                            <SelectTrigger className="h-9 text-xs" aria-label="First feature column">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="auto">Auto · first numeric</SelectItem>
                              {csvNumericCols.map((c) => (
                                <SelectItem key={c} value={c} className="font-mono text-xs">
                                  {c}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                        <div className="space-y-1">
                          <Label className="text-[11px] font-medium">Feature x₂</Label>
                          <Select
                            value={featureColumn2 || "auto"}
                            onValueChange={(v) =>
                              setMlConfig({ featureColumn2: v === "auto" ? "" : v })
                            }
                          >
                            <SelectTrigger className="h-9 text-xs" aria-label="Second feature column">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="auto">Auto · second numeric</SelectItem>
                              {csvNumericCols.map((c) => (
                                <SelectItem key={c} value={c} className="font-mono text-xs">
                                  {c}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                        <div className="space-y-1">
                          <Label className="text-[11px] font-medium">Label (2 classes)</Label>
                          <Select
                            value={targetColumn || "auto"}
                            onValueChange={(v) =>
                              setMlConfig({ targetColumn: v === "auto" ? "" : v })
                            }
                          >
                            <SelectTrigger className="h-9 text-xs" aria-label="Label column">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="auto">Auto · 2-value column</SelectItem>
                              {csvTable.columns.map((c) => (
                                <SelectItem key={c} value={c} className="font-mono text-xs">
                                  {c}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                    )
                  ) : null}
                </div>
              ) : (
                <div className="space-y-2">
                  <div
                    role="button"
                    tabIndex={0}
                    aria-label="Upload a CSV file"
                    onClick={() => fileInputRef.current?.click()}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        fileInputRef.current?.click();
                      }
                    }}
                    onDragOver={(e) => {
                      e.preventDefault();
                      setDragOver(true);
                    }}
                    onDragLeave={() => setDragOver(false)}
                    onDrop={(e) => {
                      e.preventDefault();
                      setDragOver(false);
                      const file = e.dataTransfer.files?.[0];
                      if (file) void handleFile(file);
                    }}
                    className={cn(
                      "flex cursor-pointer flex-col items-center justify-center gap-1.5 rounded-lg border border-dashed px-4 py-6 text-center outline-none transition-all duration-200 focus-visible:ring-1 focus-visible:ring-ring",
                      dragOver
                        ? "scale-[1.01] border-primary bg-primary/10 shadow-sm"
                        : "border-border bg-muted/20 hover:border-primary/40 hover:bg-primary/5"
                    )}
                  >
                    <FileUp className="size-5 text-primary" aria-hidden />
                    <p className="text-xs font-medium">Drop a CSV here or click to browse</p>
                    <p className="text-[11px] leading-snug text-muted-foreground">
                      .csv · .tsv · .txt · ≤ 256 KB · parsed in your browser
                    </p>
                  </div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv,.tsv,.txt,text/csv,text/plain"
                    className="sr-only"
                    aria-label="Choose a CSV file"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) void handleFile(file);
                      e.target.value = "";
                    }}
                  />
                  <button
                    type="button"
                    onClick={() => setPasteOpen((v) => !v)}
                    aria-expanded={pasteOpen}
                    className="w-full rounded-sm text-center text-[11px] font-medium text-primary underline-offset-2 transition-colors hover:underline focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                  >
                    {pasteOpen ? "Hide paste area" : "…or paste CSV text instead"}
                  </button>
                  {pasteOpen ? (
                    <div className="space-y-1.5">
                      <textarea
                        value={pasteText}
                        onChange={(e) => setPasteText(e.target.value)}
                        rows={5}
                        spellCheck={false}
                        placeholder={"dose,clearance\n250,3.2\n180,2.4\n…"}
                        aria-label="Paste CSV text"
                        className="thin-scrollbar w-full resize-y rounded-md border bg-background p-2 font-mono text-[11px] leading-relaxed outline-none focus-visible:ring-1 focus-visible:ring-ring"
                      />
                      <Button
                        size="sm"
                        className="h-8 w-full gap-1.5 text-xs"
                        onClick={loadPasted}
                        disabled={pasteText.trim().length < 4}
                      >
                        <ClipboardPaste className="size-3.5" aria-hidden />
                        Load pasted CSV
                      </Button>
                    </div>
                  ) : null}
                </div>
              )}
            </div>
          ) : null}

          {/* Samples + noise — synthetic mode only (CSV rows come from the file) */}
          {source === "synthetic" ? (
            <>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="ml-samples" className="text-xs font-medium">
                    Samples
                  </Label>
                  <span className="font-mono text-xs tabular-nums text-muted-foreground">{samples}</span>
                </div>
                <Slider
                  id="ml-samples"
                  min={100}
                  max={1000}
                  step={50}
                  value={[samples]}
                  onValueChange={(v) => setMlConfig({ samples: v[0] ?? samples })}
                />
              </div>

              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="ml-noise" className="text-xs font-medium">
                    Noise {task === "classification" ? "(moons)" : "σ"}
                  </Label>
                  <span className="font-mono text-xs tabular-nums text-muted-foreground">
                    {noise.toFixed(2)}
                  </span>
                </div>
                <Slider
                  id="ml-noise"
                  min={0}
                  max={noiseMax}
                  step={0.05}
                  value={[Math.min(noise, noiseMax)]}
                  onValueChange={(v) => setMlConfig({ noise: v[0] ?? noise })}
                />
              </div>
            </>
          ) : null}

          {/* Model — grouped picker (Linear family / Trees & ensembles /
              Instance-based / Probabilistic / Neural) with honest hints. */}
          <div className="space-y-1.5">
            <Label className="text-xs font-medium">
              Model{" "}
              <span className="font-normal text-muted-foreground">
                ({specs.length}{" "}
                {task === "regression" ? "regression" : "classification"} models)
              </span>
            </Label>
            <Select value={modelSel} onValueChange={selectModel}>
              <SelectTrigger className="h-10" aria-label="Machine learning model">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {MODEL_GROUPS.map((group) => {
                  const groupModels = specs.filter((m) => m.group === group);
                  if (groupModels.length === 0) return null;
                  return (
                    <SelectGroup key={group}>
                      <SelectLabel className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                        {GROUP_LABELS[group]}
                      </SelectLabel>
                      {groupModels.map((m) => (
                        <SelectItem key={m.id} value={m.id} className="text-xs">
                          {m.label}
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  );
                })}
              </SelectContent>
            </Select>
            <p className="text-[11px] leading-snug text-muted-foreground" aria-live="polite">
              {modelSpec.hint}
            </p>
          </div>

          {/* Contextual hyperparameters for the selected model. */}
          {modelSpec.hp.length > 0 ? (
            <div className="space-y-3">
              <Label className="text-xs font-medium">
                {modelSpec.label} parameters
              </Label>
              {modelSpec.hp.map((h) =>
                h.control === "input" ? (
                  <div key={h.key} className="space-y-1.5">
                    <Label htmlFor={`ml-${h.key}`} className="text-xs font-medium">
                      {h.key === "alpha" ? `${modelSpec.label} α` : h.label}
                    </Label>
                    <Input
                      id={`ml-${h.key}`}
                      inputMode="decimal"
                      value={h.key === "alpha" ? alpha : l2}
                      onChange={(e) =>
                        setMlConfig(h.key === "alpha" ? { alpha: e.target.value } : { l2: e.target.value })
                      }
                      aria-invalid={(h.key === "alpha" ? alphaValid : l2Valid) ? undefined : true}
                      className={`h-10 font-mono text-sm tabular-nums ${
                        (h.key === "alpha" ? alphaValid : l2Valid) ? "" : "border-destructive/60"
                      }`}
                    />
                    {(h.key === "alpha" ? !alphaValid : !l2Valid) ? (
                      <p className="text-[11px] text-destructive">Must be a number ≥ 0.</p>
                    ) : null}
                  </div>
                ) : (
                  <ParamSlider
                    key={h.key}
                    id={`ml-${h.key}`}
                    label={h.label}
                    unit={h.unit ?? ""}
                    min={h.min ?? 0}
                    max={h.max ?? 1}
                    step={h.step ?? 1}
                    value={hp[h.key] ?? DEFAULT_HP[h.key] ?? 0}
                    onChange={(v) => setHp((prev) => ({ ...prev, [h.key]: v }))}
                    format={fmtHp}
                  />
                )
              )}
            </div>
          ) : null}

          {/* Test size */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="ml-test" className="text-xs font-medium">
                Test split
              </Label>
              <span className="font-mono text-xs tabular-nums text-muted-foreground">
                {testSizePct}%
              </span>
            </div>
            <Slider
              id="ml-test"
              min={10}
              max={50}
              step={5}
              value={[testSizePct]}
              onValueChange={(v) => setMlConfig({ testSizePct: v[0] ?? testSizePct })}
            />
          </div>

          {/* CV + seed */}
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label className="text-xs font-medium">k-fold CV</Label>
              <Select value={cv} onValueChange={(v) => setMlConfig({ cv: v })}>
                <SelectTrigger className="h-10" aria-label="Cross-validation folds">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="0">Off</SelectItem>
                  <SelectItem value="3">3-fold</SelectItem>
                  <SelectItem value="5">5-fold</SelectItem>
                  <SelectItem value="10">10-fold</SelectItem>
                </SelectContent>
              </Select>
            </div>
            {source === "synthetic" ? (
              <div className="space-y-1.5">
                <Label htmlFor="ml-seed" className="text-xs font-medium">
                  Seed
                </Label>
                <Input
                  id="ml-seed"
                  inputMode="numeric"
                  value={seed}
                  onChange={(e) => setMlConfig({ seed: e.target.value })}
                  aria-invalid={seedValid ? undefined : true}
                  className={`h-10 font-mono text-sm tabular-nums ${seedValid ? "" : "border-destructive/60"}`}
                />
                {!seedValid ? (
                  <p className="text-[11px] text-destructive">Non-negative integer.</p>
                ) : null}
              </div>
            ) : (
              <div className="space-y-1.5">
                <Label className="text-xs font-medium">Rows</Label>
                <div
                  className="flex h-10 items-center rounded-md border bg-muted/30 px-3 font-mono text-sm tabular-nums text-muted-foreground"
                  title={mlCsv?.fileName ?? "No CSV loaded"}
                >
                  {csvTable ? `${csvTable.rows.length} rows` : "No CSV loaded"}
                </div>
              </div>
            )}
          </div>

          {/* Hyperparameter search — RandomSearch or Hyperband (successive
              halving) over the selected model's discrete grid (the ml tune
              mirror). Only for tunable models. */}
          {tuneable ? (
            <div className="space-y-1.5 rounded-lg border bg-muted/20 p-3">
              <div className="flex items-center justify-between gap-2">
                <Label className="text-xs font-medium">Hyperparameter search</Label>
                <div
                  role="group"
                  aria-label="Searcher"
                  className="inline-flex shrink-0 rounded-md border bg-background p-0.5"
                >
                  {(["random", "hyperband"] as const).map((mode) => (
                    <button
                      key={mode}
                      type="button"
                      aria-pressed={searcher === mode}
                      onClick={() => {
                        setSearcher(mode);
                        setTuneError(null);
                      }}
                      className={cn(
                        "flex h-7 items-center gap-1 rounded-[5px] px-2.5 text-[11px] font-medium transition-colors",
                        searcher === mode
                          ? "bg-primary text-primary-foreground shadow-sm"
                          : "text-muted-foreground hover:text-foreground"
                      )}
                      title={
                        mode === "random"
                          ? "Seeded sampling without replacement over the grid"
                          : "Successive halving over the resource hyperparameter"
                      }
                    >
                      {mode === "random" ? (
                        <SlidersHorizontal className="size-3" aria-hidden />
                      ) : (
                        <Layers className="size-3" aria-hidden />
                      )}
                      {mode === "random" ? "Random" : "Hyperband"}
                    </button>
                  ))}
                </div>
              </div>
              <div className="flex gap-2">
                {searcher === "random" ? (
                  <Select value={maxTrials} onValueChange={setMaxTrials}>
                    <SelectTrigger
                      id="ml-max-trials"
                      className="h-10 w-[104px] shrink-0"
                      aria-label="RandomSearch trial budget"
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {[8, 12, 16, 20, 30, 40].map((n) => (
                        <SelectItem key={n} value={String(n)}>
                          {n} trials
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : (
                  <Select value={etaSel} onValueChange={setEtaSel}>
                    <SelectTrigger
                      id="ml-eta"
                      className="h-10 w-[104px] shrink-0"
                      aria-label="Hyperband halving factor η"
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {[2, 3, 4].map((n) => (
                        <SelectItem key={n} value={String(n)}>
                          η = {n}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
                <Button
                  variant="secondary"
                  className="h-10 flex-1 gap-2"
                  onClick={() => void runTune()}
                  disabled={running || tuning}
                >
                  {tuning ? (
                    <>
                      <Loader2 className="size-4 animate-spin" aria-hidden />
                      Searching…
                    </>
                  ) : (
                    <>
                      {searcher === "random" ? (
                        <SlidersHorizontal className="size-4" aria-hidden />
                      ) : (
                        <Layers className="size-4" aria-hidden />
                      )}
                      Tune {modelSpec.label.split(" (")[0]}
                    </>
                  )}
                </Button>
              </div>
              <p className="text-[11px] leading-snug text-muted-foreground">
                {searcher === "random"
                  ? cv === "0"
                    ? "CV is off — trials score with 5-fold CV anyway."
                    : `Every trial scores ${cv}-fold CV ${
                        task === "regression" ? "R²" : "accuracy"
                      }; same seed → same leaderboard.`
                  : hyperbandable && resourceSpec
                    ? `Successive halving over ${resourceSpec.label}: brackets start many configs at a low budget and promote only the top 1/η per rung. The schedule fixes the trial count; the budget knob is η.`
                    : `Hyperband needs a resource hyperparameter — the neural net (epochs), random forest (estimators) or gradient boosting (rounds) qualify. Pick one or switch to Random.`}
              </p>
              {tuneError ? (
                <p className="text-[11px] leading-snug text-destructive">{tuneError}</p>
              ) : null}
            </div>
          ) : (
            <p className="rounded-md border border-dashed px-3 py-2 text-[11px] leading-snug text-muted-foreground">
              {modelSpec.label} has no tunable hyperparameters — the hyperparameter search needs a
              model with a search space.
            </p>
          )}

          <div className="flex gap-2">
            <Button
              size="lg"
              className="h-11 flex-1 gap-2"
              onClick={() => void run()}
              disabled={running}
            >
              {running ? (
                <>
                  <Loader2 className="size-4 animate-spin" aria-hidden />
                  Training…
                </>
              ) : (
                <>
                  <Play className="size-4" aria-hidden />
                  Train model
                </>
              )}
            </Button>
            <Button
              size="lg"
              variant="outline"
              className="h-11 shrink-0 gap-2"
              onClick={() => void previewDataset()}
              disabled={previewLoading}
              aria-label="Preview the dataset before training"
            >
              {previewLoading ? (
                <Loader2 className="size-4 animate-spin" aria-hidden />
              ) : (
                <Table2 className="size-4" aria-hidden />
              )}
              <span className="hidden sm:inline">Preview data</span>
              <span className="sm:hidden">Data</span>
            </Button>
            <Button
              size="lg"
              variant="outline"
              className="h-11 shrink-0 gap-2"
              onClick={resetSetup}
              disabled={running || previewLoading}
              aria-label="Reset the experiment setup to defaults"
              title="Reset setup to defaults"
            >
              <RotateCcw className="size-4" aria-hidden />
              <span className="hidden xl:inline">Reset</span>
            </Button>
          </div>

          {error ? (
            <Alert variant="destructive">
              <TriangleAlert className="size-4" aria-hidden />
              <AlertTitle>Request failed</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          ) : null}
        </CardContent>
      </Card>

      {/* ------------------------------ Results ------------------------------ */}
      <div className="min-w-0 space-y-4">
        {preview ? (
          <Card className="overflow-hidden py-0 shadow-sm">
            <CardHeader className="border-b bg-muted/30 py-3">
              <div className="flex items-center justify-between gap-2">
                <div className="min-w-0">
                  <CardTitle className="text-sm">Dataset preview</CardTitle>
                  <CardDescription className="text-xs">
                    {preview.source
                      ? `First ${preview.rows.length} of ${preview.source.nRows} usable rows · ${
                          preview.source.fileName ?? "pasted.csv"
                        } · ${preview.source.droppedRows} malformed row${
                          preview.source.droppedRows === 1 ? "" : "s"
                        } dropped — raw column values`
                      : `First ${preview.rows.length} of ${preview.nTotal} rows · seed ${preview.seed} ·
                    noise σ ${preview.noise.toFixed(2)} — same generator the trainer uses`}
                  </CardDescription>
                </div>
                <div className="flex items-center gap-1.5">
                {!preview.source ? (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-8 shrink-0 gap-1.5 px-2.5 text-xs"
                    onClick={() => void downloadCsv()}
                    disabled={csvLoading}
                    aria-label="Download the full dataset as CSV"
                    title="Download the full dataset as CSV"
                  >
                    {csvLoading ? (
                      <Loader2 className="size-3.5 animate-spin" aria-hidden />
                    ) : (
                      <Download className="size-3.5" aria-hidden />
                    )}
                    <span className="hidden sm:inline">CSV</span>
                  </Button>
                ) : null}
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 w-8 shrink-0 p-0"
                  onClick={() => setPreviewOpen((v) => !v)}
                  aria-expanded={previewOpen}
                  aria-label={previewOpen ? "Collapse dataset preview" : "Expand dataset preview"}
                >
                  <ChevronDown
                    className={cn(
                      "size-4 transition-transform duration-200",
                      previewOpen ? "" : "-rotate-90"
                    )}
                    aria-hidden
                  />
                </Button>
                </div>
              </div>
            </CardHeader>
            {previewOpen ? (
              <CardContent className="space-y-3 p-3 sm:p-4">
                <div className="grid gap-3 lg:grid-cols-2">
                  <div className="thin-scrollbar max-h-72 overflow-auto rounded-md border">
                    <Table className="data-table">
                      <TableHeader>
                        <TableRow className="bg-muted/40 hover:bg-muted/40">
                          {preview.columns.map((c) => (
                            <TableHead key={c} className="h-8 font-mono text-[11px]">
                              {c}
                            </TableHead>
                          ))}
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {preview.rows.map((row, idx) => (
                          <TableRow key={String(row[preview.columns[0] ?? "i"] ?? idx)}>
                            {preview.columns.map((c) => (
                              <TableCell key={c} className="py-1.5 font-mono text-xs tabular-nums">
                                {c === (preview.source?.targetColumn ?? "label") ? (
                                  <span className="inline-flex items-center gap-1.5">
                                    <span
                                      className={cn(
                                        "size-1.5 rounded-full",
                                        row[c] === 1 ? "bg-chart-1" : "bg-chart-2"
                                      )}
                                      aria-hidden
                                    />
                                    {row[c]}
                                  </span>
                                ) : (
                                  fmtNum(row[c] ?? 0, 4)
                                )}
                              </TableCell>
                            ))}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                  <div className="flex flex-col rounded-md border bg-muted/10 p-2">
                    <p className="px-1 pb-1 font-mono text-[11px] leading-snug text-muted-foreground">
                      all {preview.nTotal} points · {preview.nPoints} evenly-spaced shown
                      {preview.task === "regression" ? " · true line dashed" : ""}
                      {preview.source && preview.source.featureColumns.length > 1
                        ? ` · x = ${preview.source.featureColumns[0]} (of ${preview.source.featureColumns.length} features)`
                        : ""}
                    </p>
                    <div className="h-56 min-h-0 lg:h-auto lg:flex-1">
                      <ResponsiveContainer width="100%" height="100%">
                        {preview.task === "regression" ? (
                          <ComposedChart margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                            <CartesianGrid
                              stroke={CHART.gridStroke}
                              strokeOpacity={CHART.gridStrokeOpacity}
                            />
                            <XAxis
                              type="number"
                              dataKey="x"
                              domain={previewView?.xDomain ?? ["dataMin - 0.4", "dataMax + 0.4"]}
                              tick={CHART.axisTick}
                              tickFormatter={fmtTick}
                              tickLine={false}
                              axisLine={{ stroke: CHART.gridStroke, strokeOpacity: 0.4 }}
                              label={{
                                value:
                                  preview.source?.featureColumns[0] ??
                                  preview.columns[1] ??
                                  "x",
                                position: "insideBottomRight",
                                offset: -2,
                                fill: "#78716c",
                                fontSize: 11,
                              }}
                            />
                            <YAxis
                              type="number"
                              dataKey="y"
                              width={44}
                              domain={previewView?.yDomain ?? ["dataMin - 0.4", "dataMax + 0.4"]}
                              tick={CHART.axisTick}
                              tickFormatter={fmtTick}
                              tickLine={false}
                              axisLine={false}
                            />
                            <Tooltip
                              cursor={{ strokeDasharray: "3 3" }}
                              contentStyle={CHART.tooltipStyle}
                              formatter={(value) => fmtNum(Number(value), 4)}
                            />
                            <Legend wrapperStyle={{ fontSize: 12, color: "#78716c" }} />
                            <Scatter
                              name="sampled points"
                              data={previewView?.points ?? preview.points}
                              fill={CHART.stone}
                              fillOpacity={0.7}
                              isAnimationActive={false}
                            />
                            <Line
                              name="true line (noiseless)"
                              data={previewView?.trend ?? preview.trend ?? []}
                              dataKey="y"
                              stroke={CHART.emerald}
                              strokeWidth={2}
                              strokeDasharray="6 3"
                              dot={false}
                              activeDot={false}
                              isAnimationActive={false}
                            />
                          </ComposedChart>
                        ) : (
                          <ScatterChart margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                            <CartesianGrid
                              stroke={CHART.gridStroke}
                              strokeOpacity={CHART.gridStrokeOpacity}
                            />
                            <XAxis
                              type="number"
                              dataKey="x"
                              domain={previewView?.xDomain ?? ["dataMin - 0.3", "dataMax + 0.3"]}
                              tick={CHART.axisTick}
                              tickFormatter={fmtTick}
                              tickLine={false}
                              axisLine={{ stroke: CHART.gridStroke, strokeOpacity: 0.4 }}
                            />
                            <YAxis
                              type="number"
                              dataKey="y"
                              width={44}
                              domain={previewView?.yDomain ?? ["dataMin - 0.3", "dataMax + 0.3"]}
                              tick={CHART.axisTick}
                              tickFormatter={fmtTick}
                              tickLine={false}
                              axisLine={false}
                            />
                            <Tooltip
                              cursor={{ strokeDasharray: "3 3" }}
                              contentStyle={CHART.tooltipStyle}
                              formatter={(value) => fmtNum(Number(value), 3)}
                            />
                            <Legend wrapperStyle={{ fontSize: 12, color: "#78716c" }} />
                            {previewView ? (
                              <Scatter
                                name={
                                  preview.source?.labelMap
                                    ? `class 0 · ${preview.source.labelMap[0]}`
                                    : "class 0 (outer moon)"
                                }
                                data={previewView.cls0}
                                fill={CHART.emerald}
                                fillOpacity={0.75}
                                isAnimationActive={false}
                              />
                            ) : null}
                            {previewView ? (
                              <Scatter
                                name={
                                  preview.source?.labelMap
                                    ? `class 1 · ${preview.source.labelMap[1]}`
                                    : "class 1 (inner moon)"
                                }
                                data={previewView.cls1}
                                fill={CHART.rose}
                                fillOpacity={0.75}
                                isAnimationActive={false}
                              />
                            ) : null}
                          </ScatterChart>
                        )}
                      </ResponsiveContainer>
                    </div>
                    {previewView ? (
                      <div className="mt-1.5 rounded-md border border-primary/15 bg-primary/5 px-2.5 pb-1.5 pt-2">
                        <div className="flex items-center justify-between gap-2 pb-1">
                          <p className="flex items-center gap-1.5 font-mono text-[11px] font-medium text-primary">
                            <ZoomIn className="size-3 shrink-0" aria-hidden />
                            x-window zoom
                          </p>
                          <div className="flex min-w-0 items-center gap-2">
                            <span
                              className="truncate font-mono text-[10px] tabular-nums text-muted-foreground"
                              aria-live="polite"
                            >
                              [{fmtNum(previewView.win[0], 2)} … {fmtNum(previewView.win[1], 2)}]
                              <span className="sr-only">
                                ; drag the handles to zoom the chart
                              </span>
                            </span>
                            {previewView.zoomed ? (
                              <button
                                type="button"
                                onClick={() => setXZoom(null)}
                                aria-label="Reset the chart zoom window"
                                title="Reset zoom"
                                className="flex size-5 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                              >
                                <RotateCcw className="size-3" aria-hidden />
                              </button>
                            ) : null}
                          </div>
                        </div>
                        <Slider
                          aria-label="Zoom the preview chart x-axis window"
                          min={previewView.fullX[0]}
                          max={previewView.fullX[1]}
                          step={(previewView.fullX[1] - previewView.fullX[0]) / 200}
                          value={[previewView.win[0], previewView.win[1]]}
                          onValueChange={(v) =>
                            setXZoom([
                              v[0] ?? previewView.fullX[0],
                              v[1] ?? previewView.fullX[1],
                            ])
                          }
                        />
                      </div>
                    ) : null}
                  </div>
                </div>
                <div className="grid gap-2 sm:grid-cols-2">
                  {preview.stats.map((s) => (
                    <div key={s.column} className="rounded-md border bg-muted/20 px-3 py-2">
                      <p className="font-mono text-xs font-medium">{s.column}</p>
                      <p className="mt-0.5 font-mono text-[11px] leading-snug tabular-nums text-muted-foreground">
                        mean {fmtNum(s.mean, 4)} · σ {fmtNum(s.std, 4)} · min {fmtNum(s.min, 4)} ·
                        max {fmtNum(s.max, 4)}
                      </p>
                    </div>
                  ))}
                  {typeof preview.correlation === "number" ? (
                    <div className="rounded-md border border-primary/25 bg-primary/5 px-3 py-2">
                      <p className="font-mono text-xs font-medium text-primary">Pearson r(x, y)</p>
                      <p className="mt-0.5 font-mono text-[11px] tabular-nums text-primary/90">
                        {fmtNum(preview.correlation, 4)} — {corrStrength(preview.correlation)} linear
                        signal
                      </p>
                    </div>
                  ) : null}
                  {preview.classBalance ? (
                    <div className="rounded-md border border-primary/25 bg-primary/5 px-3 py-2">
                      <p className="font-mono text-xs font-medium text-primary">Class balance</p>
                      <p className="mt-0.5 font-mono text-[11px] tabular-nums text-primary/90">
                        {preview.classBalance
                          .map(
                            (c) =>
                              `class ${c.label}: ${c.count} (${c.pct.toFixed(1)}%)`
                          )
                          .join(" · ")}
                      </p>
                    </div>
                  ) : null}
                </div>
              </CardContent>
            ) : null}
          </Card>
        ) : null}

        {!result && status !== "running" ? (
          <Card className="py-0 shadow-sm">
            <CardContent className="flex flex-col items-center justify-center gap-2 py-16 text-center">
              <span className="ping-ring flex size-12 items-center justify-center rounded-full bg-primary/10 text-primary">
                <Brain className="size-6" aria-hidden />
              </span>
              <p className="text-sm font-medium">No results yet</p>
              <p className="max-w-sm text-xs text-muted-foreground">
                Pick a data source — the seeded server-side generator or your own CSV — pick one of
                the 15 regression or 10 classification models, then press{" "}
                <strong className="font-semibold text-foreground">Train model</strong>. Every
                run is deterministic: same seed, same data, same numbers.
              </p>
            </CardContent>
          </Card>
        ) : null}

        {status === "running" ? (
          <Card className="py-0 shadow-sm">
            <CardContent className="flex flex-col items-center justify-center gap-3 py-16 text-center">
              <Loader2 className="size-8 animate-spin text-primary" aria-hidden />
              <p className="text-sm font-medium">Training {modelSpec.label}…</p>
              <p className="text-xs text-muted-foreground">
                {source === "csv" && mlCsv
                  ? `Training on ${mlCsv.fileName} (${csvTable?.rows.length ?? 0} rows), fitting ${modelSpec.label}, evaluating the held-out split${
                      cv !== "0" ? ` and running ${cv}-fold CV` : ""
                    }.`
                  : `Generating ${samples} samples, fitting ${modelSpec.label}, evaluating the held-out split${
                      cv !== "0" ? ` and running ${cv}-fold CV` : ""
                    }.`}
              </p>
            </CardContent>
          </Card>
        ) : null}

        {result && status !== "running" ? (
          <>
            {/* Metrics */}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {metricSpecs.map((m) => (
                <MetricCard key={m.key} label={m.label} value={result.metrics[m.key] ?? Number.NaN} />
              ))}
              {result.cv ? (
                <div className="col-span-2 rounded-lg border border-primary/25 bg-primary/5 px-3 py-2.5 shadow-sm sm:col-span-4">
                  <p className="text-xs text-muted-foreground">
                    {result.cv.k}-fold CV {result.cv.metricName}
                  </p>
                  <p className="font-mono text-xl font-semibold tabular-nums text-primary">
                    {fmtNum(result.cv.mean, 4)}{" "}
                    <span className="text-sm font-normal">± {fmtNum(result.cv.std, 3)}</span>
                  </p>
                </div>
              ) : null}
            </div>

            {/* Chart */}
            <Card className="overflow-hidden py-0 shadow-sm">
              <CardHeader className="border-b bg-muted/30 py-3">
                <CardTitle className="text-sm">
                  {result.task === "regression"
                    ? predActual
                      ? "Predicted vs actual (test set)"
                      : "Test points & fitted curve"
                    : result.datasetMeta.source
                      ? "Test points & decision boundary"
                      : "Two moons & decision boundary"}
                </CardTitle>
                <CardDescription className="text-xs">
                  {result.datasetMeta.source
                    ? `${result.datasetMeta.nTotal} rows from ${
                        result.datasetMeta.source.fileName ?? "pasted.csv"
                      } · ${result.datasetMeta.nTrain} train · ${result.datasetMeta.nTest} test · ${
                        result.datasetMeta.source.droppedRows
                      } dropped${
                        predActual
                          ? ` · ${result.datasetMeta.features.length} features`
                          : ""
                      }`
                    : `${result.datasetMeta.nTotal} samples · ${result.datasetMeta.nTrain} train · ${
                        result.datasetMeta.nTest
                      } test · seed ${result.datasetMeta.seed}`}
                </CardDescription>
              </CardHeader>
              <CardContent className="p-3 sm:p-4">
                <div
                  className="h-[340px] w-full"
                  role="img"
                  aria-label={
                    result.task === "regression"
                      ? predActual
                        ? "Scatter plot of model predictions versus actual values on the test set, with the perfect-prediction reference line"
                        : "Scatter plot of test points with the fitted regression curve"
                      : result.datasetMeta.source
                        ? "Scatter plot of the uploaded-data test classes with the model decision boundary"
                        : "Scatter plot of the two moon classes with model decision boundary"
                  }
                >
                  <ResponsiveContainer width="100%" height="100%">
                    {result.task === "regression" ? (
                      <ScatterChart margin={{ top: 8, right: 14, bottom: 4, left: 0 }}>
                        <CartesianGrid stroke={CHART.gridStroke} strokeOpacity={CHART.gridStrokeOpacity} />
                        <XAxis
                          type="number"
                          dataKey="x"
                          domain={["auto", "auto"]}
                          tick={CHART.axisTick}
                          tickFormatter={fmtTick}
                          tickLine={false}
                          axisLine={{ stroke: CHART.gridStroke, strokeOpacity: 0.4 }}
                          label={{
                            value: predActual
                              ? "ŷ (predicted)"
                              : result.datasetMeta.source?.featureColumns[0] ?? "x",
                            position: "insideBottomRight",
                            offset: -2,
                            fill: "#78716c",
                            fontSize: 11,
                          }}
                        />
                        <YAxis
                          type="number"
                          dataKey="y"
                          width={44}
                          tick={CHART.axisTick}
                          tickFormatter={fmtTick}
                          tickLine={false}
                          axisLine={false}
                          label={
                            predActual
                              ? {
                                  value: result.datasetMeta.source?.targetColumn ?? "y (actual)",
                                  angle: -90,
                                  position: "insideBottomLeft",
                                  offset: 18,
                                  fill: "#78716c",
                                  fontSize: 11,
                                }
                              : undefined
                          }
                        />
                        <Tooltip
                          cursor={{ strokeDasharray: "3 3" }}
                          contentStyle={CHART.tooltipStyle}
                          formatter={(value) => fmtNum(Number(value), 4)}
                        />
                        <Legend wrapperStyle={{ fontSize: 12, color: "#78716c" }} />
                        <Scatter
                          name={predActual ? "perfect prediction (ŷ = y)" : "fitted curve"}
                          data={result.series.curve ?? []}
                          line={{ stroke: CHART.emerald, strokeWidth: 2.5 }}
                          shape="square"
                          fill={CHART.emerald}
                          fillOpacity={0.5}
                          isAnimationActive={false}
                        />
                        <Scatter
                          name={predActual ? "test predictions" : "test points"}
                          data={result.series.test}
                          fill={CHART.stone}
                          fillOpacity={0.8}
                          isAnimationActive={false}
                        />
                      </ScatterChart>
                    ) : (
                      <ScatterChart margin={{ top: 8, right: 14, bottom: 4, left: 0 }}>
                        <CartesianGrid stroke={CHART.gridStroke} strokeOpacity={CHART.gridStrokeOpacity} />
                        <XAxis
                          type="number"
                          dataKey="x"
                          domain={["auto", "auto"]}
                          tick={CHART.axisTick}
                          tickFormatter={fmtTick}
                          tickLine={false}
                          axisLine={{ stroke: CHART.gridStroke, strokeOpacity: 0.4 }}
                        />
                        <YAxis
                          type="number"
                          dataKey="y"
                          width={44}
                          tick={CHART.axisTick}
                          tickFormatter={fmtTick}
                          tickLine={false}
                          axisLine={false}
                        />
                        <Tooltip
                          cursor={{ strokeDasharray: "3 3" }}
                          contentStyle={CHART.tooltipStyle}
                          formatter={(value) => fmtNum(Number(value), 3)}
                        />
                        <Legend wrapperStyle={{ fontSize: 12, color: "#78716c" }} />
                        {clsSeries ? (
                          <Scatter
                            name="boundary: class 0"
                            data={clsSeries.grid0}
                            fill={CHART.teal}
                            shape={TinyDot(CHART.teal)}
                            isAnimationActive={false}
                          />
                        ) : null}
                        {clsSeries ? (
                          <Scatter
                            name="boundary: class 1"
                            data={clsSeries.grid1}
                            fill={CHART.amber}
                            shape={TinyDot(CHART.amber)}
                            isAnimationActive={false}
                          />
                        ) : null}
                        {clsSeries ? (
                          <Scatter
                            name={
                              result.datasetMeta.source?.labelMap
                                ? `class 0 · ${result.datasetMeta.source.labelMap[0]} (test)`
                                : "moon 0 (test)"
                            }
                            data={clsSeries.moon0}
                            fill={CHART.emerald}
                            shape={MoonDot(CHART.emerald)}
                            isAnimationActive={false}
                          />
                        ) : null}
                        {clsSeries ? (
                          <Scatter
                            name={
                              result.datasetMeta.source?.labelMap
                                ? `class 1 · ${result.datasetMeta.source.labelMap[1]} (test)`
                                : "moon 1 (test)"
                            }
                            data={clsSeries.moon1}
                            fill={CHART.rose}
                            shape={MoonDot(CHART.rose)}
                            isAnimationActive={false}
                          />
                        ) : null}
                      </ScatterChart>
                    )}
                  </ResponsiveContainer>
                </div>
                <p className="mt-1 text-center text-[11px] text-muted-foreground">
                  {result.task === "regression"
                    ? predActual
                      ? "Stone dots = (ŷ, y) pairs on the held-out test rows; emerald line = perfect prediction."
                      : "Stone dots = held-out test points; emerald line = model prediction (100-point sweep)."
                    : result.datasetMeta.source
                      ? "Emerald/rose = test-set classes; faint teal/amber dots = 60×60 grid classified by the model."
                      : "Emerald/rose = test-set moons; faint teal/amber dots = 60×60 grid classified by the model."}
                </p>
              </CardContent>
            </Card>

            {/* CV folds + coefficients */}
            <div className="grid gap-4 lg:grid-cols-2">
              {result.cv ? (
                <Card className="overflow-hidden py-0 shadow-sm">
                  <CardHeader className="border-b bg-muted/30 py-3">
                    <CardTitle className="text-sm">
                      {result.cv.k}-fold CV — {result.cv.metricName} per fold
                    </CardTitle>
                  </CardHeader>
                  <div className="thin-scrollbar max-h-72 overflow-y-auto">
                    <Table className="data-table">
                      <TableHeader className="sticky top-0 bg-muted/60 backdrop-blur">
                        <TableRow className="hover:bg-transparent">
                          <TableHead className="h-9">Fold</TableHead>
                          <TableHead className="h-9 text-right">{result.cv.metricName}</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {result.cv.folds.map((f) => (
                          <TableRow key={f.fold}>
                            <TableCell className="py-2 font-mono text-xs">fold {f.fold}</TableCell>
                            <TableCell className="py-2 text-right font-mono text-xs tabular-nums">
                              {fmtNum(f.value, 4)}
                            </TableCell>
                          </TableRow>
                        ))}
                        <TableRow className="bg-muted/30 hover:bg-muted/30">
                          <TableCell className="py-2 text-xs font-medium">mean ± std</TableCell>
                          <TableCell className="py-2 text-right font-mono text-xs font-semibold tabular-nums text-primary">
                            {fmtNum(result.cv?.mean ?? 0, 4)} ± {fmtNum(result.cv?.std ?? 0, 3)}
                          </TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </div>
                </Card>
              ) : null}

              <Card className="overflow-hidden py-0 shadow-sm">
                <CardHeader className="border-b bg-muted/30 py-3">
                  <CardTitle className="text-sm flex items-center gap-1.5">
                    {predActual ? <Sigma className="size-3.5 text-primary" aria-hidden /> : null}
                    Learned parameters
                  </CardTitle>
                  <CardDescription className="text-xs">
                    {paramsPanel?.kind === "importance"
                      ? `Feature importance (${paramsPanel.data?.measure === "gini" ? "gini" : "variance"} reduction, normalized to Σ = 1)`
                      : paramsPanel?.kind === "params"
                        ? "Echoed hyperparameters — no closed-form coefficients"
                        : "Standardized feature space (train-set μ, σ)"}
                  </CardDescription>
                </CardHeader>
                <CardContent className="pb-4">
                  {/* Linear family: real coefficients. */}
                  {paramsPanel?.kind === "coefficients" && coefficientsText ? (
                    <pre className="thin-scrollbar overflow-x-auto rounded-md border bg-muted/30 p-3 font-mono text-xs leading-relaxed">
                      {coefficientsText}
                    </pre>
                  ) : null}

                  {/* Trees & ensembles: normalized impurity-reduction importance. */}
                  {paramsPanel?.kind === "importance" ? (
                    paramsPanel.data ? (
                      <ul className="space-y-2" aria-label="Feature importances">
                        {paramsPanel.data.features.map((f, i) => {
                          const v = paramsPanel.data!.importance[i] ?? 0;
                          return (
                            <li key={f} className="space-y-1">
                              <div className="flex items-baseline justify-between gap-2">
                                <span className="truncate font-mono text-xs font-medium" title={f}>
                                  {f}
                                </span>
                                <span className="font-mono text-xs tabular-nums text-muted-foreground">
                                  {fmtNum(v, 4)}
                                </span>
                              </div>
                              <div
                                className="h-2 overflow-hidden rounded-full bg-muted"
                                role="presentation"
                              >
                                <div
                                  className="h-full rounded-full bg-primary/70"
                                  style={{ width: `${Math.max(v, 0) * 100}%` }}
                                />
                              </div>
                            </li>
                          );
                        })}
                      </ul>
                    ) : (
                      <p className="rounded-md border border-dashed px-3 py-2 text-[11px] leading-snug text-muted-foreground">
                        Importances aren’t kept in the persisted result — retrain to view them.
                      </p>
                    )
                  ) : null}

                  {/* Instance-based / neural / NB / SVM / HSVR: hyperparameter echo. */}
                  {paramsPanel?.kind === "params" ? (
                    paramsPanel.data ? (
                      <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1.5" aria-label="Model hyperparameters">
                        {Object.entries(paramsPanel.data).map(([key, value]) => (
                          <div key={key} className="col-span-2 grid grid-cols-subgrid items-baseline">
                            <dt className="truncate font-mono text-xs text-muted-foreground" title={key}>
                              {HP_LABELS[key] ?? key}
                            </dt>
                            <dd className="text-right font-mono text-xs font-semibold tabular-nums">
                              {fmtNum(value, 4)}
                            </dd>
                          </div>
                        ))}
                      </dl>
                    ) : (
                      <p className="rounded-md border border-dashed px-3 py-2 text-[11px] leading-snug text-muted-foreground">
                        This model reports no hyperparameters — {MODEL_NOTES[result.model] ?? "see the model hint."}
                      </p>
                    )
                  ) : null}

                  {/* HSVR: per-cluster geometry + local-fit diagnostics. */}
                  {result.hsvr ? (
                    <div className="mt-3 space-y-1.5">
                      <p className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                        <Network className="size-3" aria-hidden />
                        k-means clusters (train rows)
                      </p>
                      <div className="overflow-hidden rounded-lg border">
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead className="h-8">Cluster</TableHead>
                              <TableHead className="h-8 text-right">Rows</TableHead>
                              <TableHead className="h-8 text-right">Trust radius</TableHead>
                              <TableHead className="h-8 text-right">Local R²</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {result.hsvr.clusters.map((c, i) => (
                              <TableRow key={i}>
                                <TableCell className="py-1.5 font-mono text-xs">
                                  #{i + 1}
                                  {c.constant ? (
                                    <span className="ml-1.5 rounded-sm border bg-muted/50 px-1 text-[9px] uppercase text-muted-foreground">
                                      mean
                                    </span>
                                  ) : null}
                                </TableCell>
                                <TableCell className="py-1.5 text-right font-mono text-xs tabular-nums">
                                  {c.size}
                                </TableCell>
                                <TableCell className="py-1.5 text-right font-mono text-xs tabular-nums">
                                  {fmtNum(c.radius, 3)}
                                </TableCell>
                                <TableCell className="py-1.5 text-right font-mono text-xs tabular-nums">
                                  {fmtNum(c.localR2, 3)}
                                </TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </div>
                      <p className="text-[11px] leading-snug text-muted-foreground">
                        Local R² is in-sample on the cluster&rsquo;s own rows; &ldquo;mean&rdquo;
                        clusters kept too few rows for a ridge fit and predict their mean.
                        Queries beyond a cluster&rsquo;s radius fall back to the global ridge.
                      </p>
                    </div>
                  ) : null}

                  <p className="mt-2 text-[11px] leading-snug text-muted-foreground">
                    {MODEL_NOTES[result.model] ??
                      (result.task === "regression"
                        ? predActual
                          ? "Multivariate least squares: k features standardized on train statistics, solved in closed form via normal equations; ridge adds α to the diagonal, skipping the intercept."
                          : "Solved in closed form via normal equations (Gaussian elimination with partial pivoting); ridge adds α to the diagonal, skipping the intercept."
                        : "Batch gradient descent: lr = 0.1, 800 epochs, L2-penalized weights (bias exempt).")}
                  </p>
                </CardContent>
              </Card>
            </div>
          </>
        ) : null}

        {/* --------------------- HSVR trust-radius geometry --------------------- */}
        {result?.hsvr ? (
          result.hsvr.scatter ? (
            <Card className="overflow-hidden py-0 shadow-sm">
              <CardHeader className="border-b bg-muted/30 py-3">
                <div className="flex min-w-0 items-center gap-2.5">
                  <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                    <Crosshair className="size-4.5" aria-hidden />
                  </span>
                  <div className="min-w-0">
                    <CardTitle className="text-base">Trust-radius geometry</CardTitle>
                    <CardDescription className="text-xs">
                      {result.hsvr.scatter.points.length} train rows ·{" "}
                      {result.hsvr.nClusters} k-means clusters · radius factor ×
                      {result.hsvr.radiusFactor} — hover any row or centroid
                    </CardDescription>
                  </div>
                  {result.hsvr.scatter.bIsTarget ? (
                    <button
                      type="button"
                      aria-pressed={showBlend}
                      onClick={() => {
                        setShowBlend((v) => !v);
                      }}
                      className={cn(
                        "ml-auto inline-flex h-7 shrink-0 items-center gap-1 rounded-md border px-2.5 text-[11px] font-medium transition-colors",
                        showBlend
                          ? "border-primary/40 bg-primary/10 text-primary shadow-sm"
                          : "border-border bg-muted/30 text-muted-foreground hover:border-primary/30 hover:text-foreground"
                      )}
                      title="Overlay each cluster's blend weight w(z) — the linear fade from full local weight at the centroid to pure global model at the trust radius (secondary 0–1 axis)"
                    >
                      <Activity className="size-3" aria-hidden />
                      Blend weights
                    </button>
                  ) : null}
                </div>
              </CardHeader>
              <CardContent className="p-3 sm:p-4">
                <HsvrTrustScatter
                  scatter={result.hsvr.scatter}
                  showBlend={showBlend}
                />
                <p className="mt-2 text-[11px] leading-snug text-muted-foreground">
                  Each dot is a train row in standardized feature space, coloured by its
                  k-means cluster. Dashed{" "}
                  {result.hsvr.scatter.bIsTarget
                    ? "bands mark the trust radius — a 1-D distance on the feature axis"
                    : `ellipses mark the trust radius — the cluster's reach${
                        result.hsvr.scatter.projected
                          ? " (projected from all features onto these two)"
                          : ""
                      }`}{" "}
                  — queries inside it blend the local ridge with the global model, fading
                  linearly to pure global at the edge; beyond it the prediction is the
                  global ridge alone.
                  {result.hsvr.scatter.bIsTarget
                    ? " The tent curves are each cluster's blend weight w(z) = 1 − |z − centroid| / R (right axis, 0–1): full local weight at the centroid, the dashed band's edge is exactly where the local voice falls silent — hover any dot for its own w."
                    : " With two or more features the fade depends on the whole feature vector, so the weight curve is drawn for single-feature models only."}
                </p>
              </CardContent>
            </Card>
          ) : (
            <p className="rounded-md border border-dashed px-3 py-2 text-[11px] leading-snug text-muted-foreground">
              The trust-radius geometry isn&rsquo;t kept in this persisted run — retrain the
              hierarchical model to view it.
            </p>
          )
        ) : null}

        {/* ------------------------------ Tuner results ------------------------------ */}
        {tuning ? (
          <Card className="py-0 shadow-sm">
            <CardContent className="flex flex-col items-center justify-center gap-3 py-16 text-center">
              <Loader2 className="size-8 animate-spin text-primary" aria-hidden />
              <p className="text-sm font-medium">
                Searching {modelSpec.label} hyperparameters…
              </p>
              <p className="max-w-md text-xs text-muted-foreground">
                {searcher === "hyperband"
                  ? `Hyperband successive halving over ${
                      resourceSpec?.label ?? "the resource"
                    } (η = ${etaSel}) — brackets trade breadth for depth, every rung scored with ${
                      Number.parseInt(cv, 10) >= 3 ? cv : "5"
                    }-fold CV ${task === "regression" ? "R²" : "accuracy"} on the ${
                      source === "csv" && mlCsv ? mlCsv.fileName : "seeded generator"
                    } data. The leaderboard lands here when the brackets finish.`
                  : `${maxTrials} RandomSearch trials, each scored with ${
                      Number.parseInt(cv, 10) >= 3 ? cv : "5"
                    }-fold CV ${task === "regression" ? "R²" : "accuracy"} on the ${
                      source === "csv" && mlCsv ? mlCsv.fileName : "seeded generator"
                    } data. The leaderboard lands here when the search finishes.`}
              </p>
            </CardContent>
          </Card>
        ) : null}

        {tuneResult && tuneStatus === "done" ? (
          <Card className="overflow-hidden py-0 shadow-sm">
            <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-2 border-b bg-muted/30 py-3">
              <div className="flex min-w-0 items-center gap-2.5">
                <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  {tuneResult.searcher === "hyperband" ? (
                    <Layers className="size-4.5" aria-hidden />
                  ) : (
                    <Trophy className="size-4.5" aria-hidden />
                  )}
                </span>
                <div className="min-w-0">
                  <CardTitle className="text-base">Hyperparameter search</CardTitle>
                  <CardDescription className="text-xs">
                    {tuneResult.searcher === "hyperband" && tuneResult.hyperband
                      ? `Hyperband · ${tuneResult.model} · ${tuneResult.trials.length} trials · η ${tuneResult.hyperband.eta} · resource: ${
                          tuneResult.hyperband.resourceLabel ??
                          HP_LABELS[tuneResult.hyperband.resourceKey] ??
                          tuneResult.hyperband.resourceKey
                        } · ${tuneResult.cvK}-fold CV ${tuneResult.objective}`
                      : `RandomSearch · ${tuneResult.model} · ${tuneResult.trials.length} ${
                          tuneResult.exhaustive
                            ? "trials (full grid)"
                            : `of ${tuneResult.spaceSize} configs`
                        } · ${tuneResult.cvK}-fold CV ${tuneResult.objective}`}
                  </CardDescription>
                </div>
              </div>
              <Button
                size="sm"
                className="h-8 shrink-0 gap-1.5 text-xs"
                onClick={applyBestParams}
                aria-label="Apply the best hyperparameters to the setup panel"
              >
                <Check className="size-3.5" aria-hidden />
                Apply best
              </Button>
            </CardHeader>
            <CardContent className="space-y-3 pb-4">
              {/* Best-vs-baseline summary strip. */}
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                <div className="rounded-lg border bg-card px-3 py-2.5 shadow-sm">
                  <p className="text-xs text-muted-foreground">Best CV {tuneResult.objective}</p>
                  <p className="font-mono text-xl font-semibold tabular-nums">
                    {fmtNum(tuneResult.best.cvScore, 4)}
                  </p>
                </div>
                <div className="rounded-lg border bg-card px-3 py-2.5 shadow-sm">
                  <p className="text-xs text-muted-foreground">
                    Held-out {tuneResult.best.testMetricName}
                  </p>
                  <p className="font-mono text-xl font-semibold tabular-nums">
                    {fmtNum(tuneResult.best.testMetric, 4)}
                  </p>
                </div>
                <div className="rounded-lg border bg-card px-3 py-2.5 shadow-sm">
                  <p className="text-xs text-muted-foreground">
                    {tuneResult.searcher === "hyperband" ? "Defaults CV (full budget)" : "Defaults CV"}
                  </p>
                  <p className="font-mono text-xl font-semibold tabular-nums">
                    {fmtNum(tuneResult.baseline.score, 4)}
                  </p>
                </div>
                <div className="rounded-lg border bg-card px-3 py-2.5 shadow-sm">
                  <p className="text-xs text-muted-foreground">Gain vs defaults</p>
                  <p
                    className={`font-mono text-xl font-semibold tabular-nums ${
                      tuneResult.best.improvement >= 0
                        ? "text-emerald-600 dark:text-emerald-400"
                        : "text-destructive"
                    }`}
                  >
                    {tuneResult.best.improvement >= 0 ? "+" : ""}
                    {fmtNum(tuneResult.best.improvement, 4)}
                  </p>
                </div>
              </div>

              {/* Hyperband: the resource ladder + budget economics strip. */}
              {tuneResult.searcher === "hyperband" && tuneResult.hyperband ? (
                <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 rounded-lg border bg-muted/20 px-3 py-2">
                  <p className="flex items-center gap-1.5 text-[11px] font-medium text-muted-foreground">
                    <Layers className="size-3" aria-hidden />
                    Resource ladder
                  </p>
                  <ol className="flex flex-wrap items-center gap-1" aria-label="Resource ladder">
                    {tuneResult.hyperband.ladder.map((r, i) => (
                      <li
                        key={r}
                        className="flex items-center gap-1"
                        aria-label={`rung ${i + 1}: ${r}`}
                      >
                        {i > 0 ? (
                          <TrendingUp className="size-3 text-muted-foreground/60" aria-hidden />
                        ) : null}
                        <span
                          className={cn(
                            "rounded-md border px-1.5 py-0.5 font-mono text-[10px] font-semibold tabular-nums",
                            i === tuneResult.hyperband!.ladder.length - 1
                              ? "border-primary/40 bg-primary/10 text-primary"
                              : "bg-background text-muted-foreground"
                          )}
                        >
                          {r}
                        </span>
                      </li>
                    ))}
                  </ol>
                  <p className="font-mono text-[11px] text-muted-foreground">
                    budget{" "}
                    <span className="font-semibold tabular-nums text-foreground">
                      {tuneResult.hyperband.budget.spent.toLocaleString()}
                    </span>{" "}
                    of a naive{" "}
                    <span className="font-semibold tabular-nums">
                      {tuneResult.hyperband.budget.naive.toLocaleString()}
                    </span>{" "}
                    {tuneResult.hyperband.resourceLabel ??
                      HP_LABELS[tuneResult.hyperband.resourceKey] ??
                      tuneResult.hyperband.resourceKey}
                    <span className="ml-1.5 rounded-sm bg-emerald-500/10 px-1.5 py-0.5 font-semibold text-emerald-600 dark:text-emerald-400">
                      {Math.round(tuneResult.hyperband.budget.savedPct * 100)}% saved
                    </span>
                  </p>
                </div>
              ) : null}

              {/* Leaderboard — best first, winner highlighted. Hyperband rows
                  additionally carry their bracket/rung + resource budget. */}
              <div className="max-h-80 overflow-y-auto thin-scrollbar rounded-lg border">
                <Table>
                  <TableHeader className="sticky top-0 z-10 bg-muted/95 backdrop-blur">
                    <TableRow>
                      <TableHead className="h-9 w-12 text-right">#</TableHead>
                      <TableHead className="h-9">Trial</TableHead>
                      <TableHead className="h-9">Configuration</TableHead>
                      {tuneResult.searcher === "hyperband" ? (
                        <TableHead className="h-9 text-right">Resource</TableHead>
                      ) : null}
                      <TableHead className="h-9 text-right">CV {tuneResult.objective}</TableHead>
                      <TableHead className="h-9 text-right">Δ default</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {tuneResult.trials.map((t, rank) => {
                      const winner = rank === 0;
                      const delta = t.score - tuneResult.baseline.score;
                      const isHb = tuneResult.searcher === "hyperband";
                      const resKey = tuneResult.hyperband?.resourceKey;
                      return (
                        <TableRow
                          key={t.trial}
                          className={winner ? "bg-primary/5" : undefined}
                        >
                          <TableCell className="py-1.5 text-right font-mono text-xs tabular-nums text-muted-foreground">
                            {rank + 1}
                          </TableCell>
                          <TableCell className="py-1.5 font-mono text-xs">
                            {t.trial}
                            {isHb && t.bracket !== undefined && t.rung !== undefined ? (
                              <span
                                className="ml-1.5 rounded-sm border bg-muted/50 px-1 text-[9px] uppercase text-muted-foreground"
                                title={`bracket ${t.bracket} (0 = full budget), rung ${t.rung} (0 = cheapest)`}
                              >
                                b{t.bracket}·r{t.rung}
                              </span>
                            ) : null}
                            {winner ? (
                              <Trophy
                                className="ml-1 inline size-3 text-primary"
                                aria-hidden
                              />
                            ) : null}
                          </TableCell>
                          <TableCell className="py-1.5 font-mono text-[11px] text-muted-foreground">
                            {Object.entries(t.params)
                              .filter(([k]) => !(isHb && k === resKey))
                              .map(([k, v]) => `${HP_LABELS[k] ?? k} ${fmtHp(v)}`)
                              .join(" · ")}
                          </TableCell>
                          {isHb ? (
                            <TableCell className="py-1.5 text-right font-mono text-xs tabular-nums">
                              {t.resource}
                              {t.promoted ? (
                                <TrendingUp
                                  className="ml-1 inline size-3 text-emerald-600 dark:text-emerald-400"
                                  aria-hidden
                                  aria-label="promoted to the next rung"
                                />
                              ) : null}
                            </TableCell>
                          ) : null}
                          <TableCell className="py-1.5 text-right font-mono text-xs font-semibold tabular-nums">
                            {fmtNum(t.score, 4)}
                          </TableCell>
                          <TableCell
                            className={`py-1.5 text-right font-mono text-xs tabular-nums ${
                              delta >= 0
                                ? "text-emerald-600 dark:text-emerald-400"
                                : "text-destructive"
                            }`}
                          >
                            {delta >= 0 ? "+" : ""}
                            {fmtNum(delta, 4)}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>

              <p className="text-[11px] leading-snug text-muted-foreground">
                {tuneResult.searcher === "hyperband" && tuneResult.hyperband
                  ? `Every trial is scored on the same ${tuneResult.cvK}-fold folds; the leaderboard mixes resource budgets by design (b = bracket, r = rung, ↑ = promoted). The winner is refit on the train split for the held-out ${tuneResult.best.testMetricName}; defaults are scored at the full ${tuneResult.hyperband.maxResource} ${tuneResult.hyperband.resourceLabel ?? HP_LABELS[tuneResult.hyperband.resourceKey] ?? tuneResult.hyperband.resourceKey} for a fair comparison. Same seed → same brackets.`
                  : `Same data and folds for every trial — only the parameters vary, so the ranking is
                a fair comparison. The winner is refit on the train split for the held-out ${
                    tuneResult.best.testMetricName
                  } above. ${
                    tuneResult.exhaustive
                      ? "The budget covered the whole grid, so this is a full grid search."
                      : `The ${tuneResult.spaceSize}-configuration grid was sampled without replacement; a larger budget explores more of it.`
                  } Deterministic: same seed → same leaderboard.`}
              </p>
            </CardContent>
          </Card>
        ) : null}

        {/* ------------------------------ Run history ------------------------------ */}
        {mlHistory.length > 0 ? (
          <Card className="overflow-hidden py-0 shadow-sm">
            <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-2 border-b bg-muted/30 py-3">
              <div className="flex min-w-0 items-center gap-2.5">
                <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  <History className="size-4.5" aria-hidden />
                </span>
                <div className="min-w-0">
                  <CardTitle className="text-base">Run history</CardTitle>
                  <CardDescription className="text-xs">
                    Last {mlHistory.length} training{" "}
                    {mlHistory.length === 1 ? "run" : "runs"} · survives reloads · Restore
                    reloads the config and results
                  </CardDescription>
                </div>
              </div>
              <Button
                variant="ghost"
                size="sm"
                className="h-8 shrink-0 gap-1.5 text-xs text-muted-foreground hover:text-foreground"
                onClick={clearMlHistory}
                aria-label="Clear the run history"
              >
                <Eraser className="size-3.5" aria-hidden />
                Clear
              </Button>
            </CardHeader>
            <CardContent className="py-0">
              <ul className="divide-y">
                {mlHistory.map((rec, i) => {
                  /* Newest-first list: the run *before* this one is i + 1. */
                  const prev = mlHistory[i + 1];
                  const delta = prev ? rec.keyValue - prev.keyValue : null;
                  const isCurrent = result !== null && rec.id === mlCurrentRunId;
                  const up = delta !== null && delta >= 0;
                  return (
                    <li
                      key={rec.id}
                      className={cn(
                        "flex items-center gap-3 px-3 py-2.5 transition-colors hover:bg-primary/5",
                        isCurrent && "bg-primary/5"
                      )}
                    >
                      <span
                        className={cn(
                          "flex size-8 shrink-0 items-center justify-center rounded-md border font-mono text-[10px] font-semibold uppercase",
                          isCurrent
                            ? "border-primary/30 bg-primary/10 text-primary"
                            : "bg-muted/40 text-muted-foreground"
                        )}
                        title={rec.task}
                      >
                        {rec.task === "regression" ? "reg" : "cls"}
                      </span>
                      <div className="min-w-0 flex-1">
                        <p className="truncate font-mono text-xs font-medium">
                          {rec.model}
                          <span className="font-normal text-muted-foreground">
                            {" "}
                            {rec.csv
                              ? ` · ${rec.csv.fileName} · ${
                                  rec.csv.featureColumns.length > 0
                                    ? `${rec.csv.featureColumns.length} feat${
                                        rec.csv.featureColumns.length > 1 ? "s" : ""
                                      }`
                                    : "auto feats"
                                } · target: ${rec.csv.targetColumn || "auto"}`
                              : ` · seed ${rec.config.seed} · n${rec.config.samples} · noise ${
                                  rec.config.noise
                                }`}
                          </span>
                        </p>
                        <p className="mt-0.5 flex items-center gap-1.5 text-[11px] text-muted-foreground">
                          {timeAgo(rec.at)}
                          {isCurrent ? (
                            <span className="rounded-sm border border-primary/30 bg-primary/10 px-1 font-mono text-[9px] font-semibold uppercase text-primary">
                              current
                            </span>
                          ) : null}
                        </p>
                      </div>
                      <div className="shrink-0 text-right">
                        <p className="font-mono text-sm font-semibold tabular-nums">
                          {fmtNum(rec.keyValue, 4)}
                          <span className="ml-1 text-[10px] font-normal text-muted-foreground">
                            {rec.keyMetric}
                          </span>
                        </p>
                        {delta !== null ? (
                          <p
                            className={cn(
                              "flex items-center justify-end gap-0.5 font-mono text-[10px] tabular-nums",
                              up ? "text-emerald-600 dark:text-emerald-400" : "text-rose-600 dark:text-rose-400"
                            )}
                            title="Change vs the previous run"
                          >
                            {up ? (
                              <TrendingUp className="size-3" aria-hidden />
                            ) : (
                              <TrendingDown className="size-3" aria-hidden />
                            )}
                            {up ? "+" : "−"}
                            {fmtNum(Math.abs(delta), 3)}
                          </p>
                        ) : (
                          <p className="font-mono text-[10px] text-muted-foreground/60">—</p>
                        )}
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-8 shrink-0 px-2.5 text-xs"
                        onClick={() => restoreRun(rec)}
                        aria-label={`Restore run ${rec.model}${
                          rec.csv ? ` from ${rec.csv.fileName}` : ` seed ${rec.config.seed}`
                        }`}
                        disabled={isCurrent}
                      >
                        <RotateCcw className="size-3.5" aria-hidden />
                        <span className="hidden sm:inline">
                          {isCurrent ? "Current" : "Restore"}
                        </span>
                      </Button>
                    </li>
                  );
                })}
              </ul>
            </CardContent>
          </Card>
        ) : null}

        {/* --------------------- Probability calibration --------------------- */}
        <ProbabilityCalibration />
      </div>
    </div>
  );
}
