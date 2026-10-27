"use client";

import { useEffect, useMemo, useRef, useState } from "react";
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
  Area,
  Bar,
  CartesianGrid,
  ComposedChart,
  ErrorBar,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  expectedImprovement,
  upperConfidenceBound,
} from "@/lib/bayes-acquisition";
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
import { fmtBytes, fmtNum, fmtTick, fmtDelta } from "@/lib/format";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import { DEFAULT_ML_CONFIG, useToolkitStore, type MlRunRecord } from "@/lib/store";
import type {
  CorrelationBundle,
  DatasetPreviewResponse,
  MLAdResponse,
  MLCompareEntry,
  MLCompareRequest,
  MLCompareResponse,
  MLLearningCurveResponse,
  MLTrainResponse,
  MLTrainRequest,
  MLTuneRequest,
  MLTuneResponse,
  MLTuneTrial,
} from "@/lib/toolkit-types";
import { buildRungPoints, type RungPoint } from "@/lib/hyperband-rungs";

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
  | "hierarchical"
  | "ensemble";

const GROUP_LABELS: Record<ModelGroup, string> = {
  linear: "Linear family",
  trees: "Trees & ensembles",
  instance: "Instance-based",
  kernel: "Kernel machines",
  probabilistic: "Probabilistic",
  neural: "Neural",
  hierarchical: "Hierarchical",
  ensemble: "Meta-ensembles",
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
    | "radiusFactor"
    | "nComponents"
    | "lengthScale"
    | "signalStd"
    | "noiseStd";
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
    id: "pls",
    label: "PLS regression",
    hint: "NIPALS latent directions maximize X–y covariance — the QSAR staple for collinear descriptors",
    group: "linear",
    kind: "coefficients",
    hp: [
      { key: "nComponents", label: "components", control: "slider", unit: "n", min: 1, max: 8, step: 1 },
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
    id: "gaussian_process",
    label: "Gaussian process (GP)",
    hint: "RBF + white kernel, σ learned by marginal likelihood — every prediction carries its own band",
    group: "kernel",
    kind: "params",
    hp: [
      { key: "lengthScale", label: "length scale", control: "slider", unit: "ℓ", min: 0.25, max: 4, step: 0.25 },
      { key: "signalStd", label: "signal σf", control: "slider", unit: "σf", min: 0.25, max: 3, step: 0.25 },
      { key: "noiseStd", label: "noise σn", control: "slider", unit: "σn", min: 0.01, max: 1, step: 0.01 },
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
  {
    id: "voting",
    label: "Voting ensemble",
    hint: "Weighted average of RF + RBF-SVM + kNN — the sklearn VotingRegressor mirror",
    group: "ensemble",
    kind: "params",
    hp: [
      { key: "nEstimators", label: "RF · trees", control: "slider", unit: "n", min: 5, max: 200, step: 5 },
      { key: "maxDepth", label: "RF · max depth", control: "slider", unit: "levels", min: 1, max: 12, step: 1 },
      { key: "gamma", label: "SVM · kernel γ", control: "slider", unit: "γ", min: 0.05, max: 2, step: 0.05 },
      { key: "alpha", label: "SVM · ridge α", control: "input" },
      { key: "k", label: "kNN · neighbours", control: "slider", unit: "k", min: 1, max: 25, step: 2 },
    ],
  },
  {
    id: "stacking",
    label: "Stacking ensemble",
    hint: "RF + RBF-SVM + kNN level-0, 5-fold OOF features, ridge meta-learner level-1",
    group: "ensemble",
    kind: "params",
    hp: [
      { key: "nEstimators", label: "RF · trees", control: "slider", unit: "n", min: 5, max: 200, step: 5 },
      { key: "maxDepth", label: "RF · max depth", control: "slider", unit: "levels", min: 1, max: 12, step: 1 },
      { key: "gamma", label: "SVM · kernel γ", control: "slider", unit: "γ", min: 0.05, max: 2, step: 0.05 },
      { key: "alpha", label: "SVM · ridge α", control: "input" },
      { key: "k", label: "kNN · neighbours", control: "slider", unit: "k", min: 1, max: 25, step: 2 },
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
    id: "gaussian_process",
    label: "GP classification",
    hint: "Laplace posterior: Newton mode, calibrated 5-erf probabilities",
    group: "kernel",
    kind: "params",
    hp: [
      { key: "lengthScale", label: "length scale", control: "slider", unit: "ℓ", min: 0.25, max: 4, step: 0.25 },
      { key: "signalStd", label: "signal σf", control: "slider", unit: "σf", min: 0.25, max: 3, step: 0.25 },
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
  {
    id: "voting",
    label: "Voting ensemble (soft)",
    hint: "Averaged class-1 probabilities of RF + RBF-SVM + kNN — the sklearn soft VotingClassifier mirror",
    group: "ensemble",
    kind: "params",
    hp: [
      { key: "nEstimators", label: "RF · trees", control: "slider", unit: "n", min: 5, max: 200, step: 5 },
      { key: "maxDepth", label: "RF · max depth", control: "slider", unit: "levels", min: 1, max: 12, step: 1 },
      { key: "C", label: "SVM · margin C", control: "slider", unit: "C", min: 0.05, max: 10, step: 0.05 },
      { key: "gamma", label: "SVM · kernel γ", control: "slider", unit: "γ", min: 0.05, max: 2, step: 0.05 },
      { key: "k", label: "kNN · neighbours", control: "slider", unit: "k", min: 1, max: 25, step: 2 },
    ],
  },
  {
    id: "stacking",
    label: "Stacking ensemble",
    hint: "RF + RBF-SVM + kNN level-0, 5-fold OOF probabilities, logistic meta-learner level-1",
    group: "ensemble",
    kind: "params",
    hp: [
      { key: "nEstimators", label: "RF · trees", control: "slider", unit: "n", min: 5, max: 200, step: 5 },
      { key: "maxDepth", label: "RF · max depth", control: "slider", unit: "levels", min: 1, max: 12, step: 1 },
      { key: "C", label: "SVM · margin C", control: "slider", unit: "C", min: 0.05, max: 10, step: 0.05 },
      { key: "gamma", label: "SVM · kernel γ", control: "slider", unit: "γ", min: 0.05, max: 2, step: 0.05 },
      { key: "k", label: "kNN · neighbours", control: "slider", unit: "k", min: 1, max: 25, step: 2 },
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
  "ensemble",
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
  nComponents: 2,
  lengthScale: 1,
  signalStd: 1,
  noiseStd: 0.1,
};

/** Models with a RandomSearch tuning space on the server (the honest
 *  mirror of REG_TUNE_SPACES / CLS_TUNE_SPACES in lib/ml-models.ts). */
const TUNEABLE_REG = new Set([
  "ridge",
  "lasso",
  "elastic_net",
  "huber",
  "pls",
  "knn",
  "decision_tree",
  "random_forest",
  "extra_trees",
  "gradient_boosting",
  "ada_boost",
  "svm_rbf",
  "gaussian_process",
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
  "gaussian_process",
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
  pls:
    "Partial least squares (NIPALS deflation): each latent direction w ∝ Zᵀy maximizes descriptor–target covariance — not descriptor variance alone — so wide, collinear blocks compress into a few predictive components. Linear once the latent space is chosen; the component count actually extracted (rank-capped) is echoed in the parameters.",
  ada_boost:
    "Weighted bootstrap rounds of shallow stumps; the ensemble predicts with the α-weighted median (regression) or α-weighted vote (classification). Importances are α-weighted per-stump gains.",
  svm_rbf:
    "Gaussian-kernel machine solved in one closed form: (K + λI)⁻¹ (regression) or the bias-augmented LS-SVM system (classification, λ = 1/(2C)).",
  gaussian_process:
    "Kriging with an RBF + white kernel: with Learn kernel on, a deterministic marginal-likelihood grid search picks ℓ / σf / σn (the web mirror of sklearn's LML-tuned kernel) and the σ you see is the prediction interval — latent posterior σ plus the learned σn. Untick it to pin the kernel at the sliders' values.",
  gaussian_process_cls:
    "Laplace GP classification (the library's gaussian_process_classifier): a latent σf²·RBF(ℓ) GP squashed through the logistic likelihood; Newton iteration finds the posterior mode and the Williams–Barber 5-erf integral turns the latent mean/variance into a calibrated probability. With Learn kernel on, a Laplace-lml grid search picks (ℓ, σf); boundary-dot opacity tracks |p − 0.5| — the classifier's own uncertainty surface.",
  gaussian_nb: "Class-conditional Gaussians (μ, σ per feature) feed the log-posterior; no tunable hyperparameters.",
  mlp: "One tanh hidden layer, full-batch gradient descent, seeded init — a demo-scale network, not production deep learning.",
  hsvr:
    "Seeded k-means splits the standardized train rows; each cluster keeps its own ridge fit, blended with the global ridge by a linear trust radius (full local weight at the centroid, zero at radius × the cluster's reach; farther queries fall back to the global model). Clusters too small for ridge predict their mean.",
  voting:
    "VotingRegressor's rule: the equal-weight mean of the three level-0 predictions (random forest · RBF kernel ridge · kNN), each member reading its own hyperparameter slots from the sliders above. The member table scores every base model on the same held-out rows as the blend — the honest “did the ensemble help” comparison.",
  voting_cls:
    "VotingClassifier with voting='soft' (the library's classification default): the class-1 probabilities of random forest · LS-SVM · kNN are averaged — RF by leaf fractions, SVM by the logistic read of the decision value, kNN by neighbour fractions — and the boundary fades by |p − 0.5|. ROC AUC / log-loss / Brier come from the blended probabilities on the held-out rows.",
  stacking:
    "sklearn's StackingRegressor contract, mirrored: 5-fold out-of-fold predictions of the RF · SVM · kNN trio build the meta-learner's training matrix (cv = 5, the library's build_stacking pin), a ridge blend is fit on it, the members are refit on the full train rows, and prediction stacks fresh member outputs through the meta. The OOF table shows each member's cross-validated R² on exactly the matrix the blender trained on; the meta weights show which member it actually trusts.",
  stacking_cls:
    "sklearn's StackingClassifier contract, mirrored: 5-fold out-of-fold class-1 probabilities of the RF · LS-SVM · kNN trio train a multivariate logistic meta-learner (batch GD, L2 bias-exempt — the row-based sibling of the 2-feature logistic above), the members are refit on the full train rows, and prediction stacks their fresh probabilities through the meta. The OOF table shows each member's cross-validated accuracy on exactly the matrix the blender trained on; the meta weights show which member it actually trusts.",
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
  nComponents: "components (used)",
  lengthScale: "length scale ℓ",
  signalStd: "signal σf",
  noiseStd: "noise σn",
  newtonIterations: "Newton iterations",
  nMembers: "level-0 members",
  oofFolds: "OOF folds",
};

/** Task-aware footer-note lookup — gaussian_process, voting and
 *  stacking all live on BOTH sides of the playground (regression =
 *  averaging / ridge meta; classification = soft vote / logistic
 *  meta), so the id alone is ambiguous and each classification entry
 *  reads its own note. */
function modelNote(task: Task, model: string): string | undefined {
  if (task === "classification") {
    if (model === "gaussian_process") return MODEL_NOTES.gaussian_process_cls;
    if (model === "voting") return MODEL_NOTES.voting_cls;
    if (model === "stacking") return MODEL_NOTES.stacking_cls;
  }
  return MODEL_NOTES[model];
}

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

/** Histogram bins for the AD panel (a): the training kNN-distance density. */
function adHistogram(ad: MLAdResponse): Array<{ x: number; count: number }> {
  const values = ad.trainMean;
  if (values.length === 0) return [];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  if (!(hi > lo)) return [{ x: lo, count: values.length }];
  const bins = 24;
  const width = (hi - lo) / bins;
  const counts = new Array<number>(bins).fill(0);
  for (const v of values) {
    const idx = Math.min(bins - 1, Math.floor((v - lo) / width));
    counts[idx] += 1;
  }
  return counts.map((count, i) => ({ x: lo + (i + 0.5) * width, count }));
}

/** Scatter rows for the AD panel (b): distance vs the ensemble prediction. */
function adScatter(ad: MLAdResponse, inside: boolean): Array<{ knn: number; predicted: number }> {
  return ad.rows
    .filter((r) => r.insideAd === inside && r.predicted !== null)
    .map((r) => ({ knn: r.knnDistance, predicted: r.predicted as number }));
}

/** A tiny yes/no/— pill for the reliability table's three verdict columns. */
function VerdictPill({ ok, nullable, strong }: { ok: boolean | null; nullable?: boolean; strong?: boolean }) {
  if (ok === null) {
    return <span className="font-mono text-xs text-muted-foreground">—</span>;
  }
  const tone = ok
    ? strong
      ? "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400"
      : "bg-emerald-500/10 text-emerald-700 dark:text-emerald-400"
    : strong
      ? "bg-rose-500/15 text-rose-700 dark:text-rose-400"
      : "bg-rose-500/10 text-rose-700 dark:text-rose-400";
  return (
    <span
      className={`inline-flex min-w-14 justify-center rounded-full px-1.5 py-0.5 text-[10px] font-semibold ${tone} ${
        nullable ? "" : ""
      }`}
    >
      {ok ? "yes" : "no"}
    </span>
  );
}

/** The Hyperband-capable roster as human labels — generated from
 *  RESOURCE_SPECS so it can never drift behind the registry (three
 *  hard-coded copies shipped stale for two rounds after ada_boost and
 *  extra_trees gained resource slots). */
function hyperbandRoster(): string {
  return Object.entries(RESOURCE_SPECS)
    .map(([id, spec]) => {
      const label =
        specOf("regression", id)?.label ?? specOf("classification", id)?.label ?? id;
      return `${label} (${spec.label})`;
    })
    .join(", ");
}

/** Default model-comparison selection: the model being explored plus two
 *  strong companions per task — deduped, capped at three. */
function defaultCompareSel(task: Task, modelSel: string): string[] {
  const companions =
    task === "regression" ? ["huber", "random_forest"] : ["svm_rbf", "random_forest"];
  return [modelSel, ...companions.filter((m) => m !== modelSel)].slice(0, 3);
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

/** GP classification: boundary dot whose OPACITY carries the calibrated
 *  probability — confident cells (|p − 0.5| → 1) render strong, cells near
 *  the decision boundary fade out. The classifier's own uncertainty
 *  surface, drawn by the posterior itself. */
function GpClsDot(color: string) {
  return function renderGpClsDot(props: unknown) {
    const { cx, cy, payload } = props as { cx?: number; cy?: number; payload?: { proba?: number } };
    if (cx === undefined || cy === undefined) return <g />;
    const proba = typeof payload?.proba === "number" ? payload.proba : 0.5;
    const confidence = Math.abs(proba - 0.5) * 2; // 0 at the boundary, 1 at the poles
    const opacity = 0.1 + 0.6 * confidence;
    return <circle cx={cx} cy={cy} r={1.5} fill={color} fillOpacity={opacity} />;
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

/** One curve row prepared for the GP band chart (stacked-area recipe:
 *  the invisible bandBase + the filled bandSpan stack to lo→hi). */
interface GpBandPoint {
  x: number;
  yhat: number;
  bandBase: number;
  bandSpan: number;
  /** Acquisition value (EI or UCB) at this x — present only when the
   *  overlay is switched on; rides on the secondary right axis. */
  acq?: number;
}

/* -- Bayesian-optimisation acquisition math lives in
 *  @/lib/bayes-acquisition (pure, self-contained, probed against
 *  scipy.stats.norm by scratch/verify_acquisition_parity_114) -- */

/** Custom tooltip for the GP band chart — x, ŷ and the band edges in one
 *  compact card (the default ComposedChart tooltip would list the stacked
 *  bandBase/bandSpan internals). */
function GpBandTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload?: GpBandPoint }>;
}) {
  if (!active || !payload?.length) return null;
  const p = payload[0]?.payload;
  if (!p) return null;
  return (
    <div className="rounded-md border bg-background/95 px-2.5 py-1.5 text-xs shadow-sm">
      <p className="font-mono tabular-nums">x = {fmtNum(p.x, 3)}</p>
      <p className="font-mono tabular-nums">ŷ = {fmtNum(p.yhat, 4)}</p>
      <p className="font-mono tabular-nums text-muted-foreground">
        ±1.96σ band = [{fmtNum(p.bandBase, 3)}, {fmtNum(p.bandBase + p.bandSpan, 3)}]
      </p>
      {p.acq !== undefined ? (
        <p className="font-mono tabular-nums" style={{ color: CHART.amber }}>
          acquisition = {fmtNum(p.acq, 4)}
        </p>
      ) : null}
    </div>
  );
}

/* -- Learning-curve rows (100x-125): the same stacked-area band recipe
 *  the GP chart uses — invisible trainBase/valBase + filled spans
 *  stack to the mean ± std envelope of each curve. -- */

/** One learning-curve chart row (band internals precomputed). */
interface LcRow {
  size: number;
  trainMean: number;
  valMean: number;
  trainBase: number;
  trainSpan: number;
  valBase: number;
  valSpan: number;
}

/** Collapse the response's points into chart rows (bands = ±1 std). */
function lcRows(result: MLLearningCurveResponse): LcRow[] {
  return result.points.map((p) => ({
    size: p.size,
    trainMean: p.trainMean,
    valMean: p.valMean,
    trainBase: p.trainMean - p.trainStd,
    trainSpan: 2 * p.trainStd,
    valBase: p.valMean - p.valStd,
    valSpan: 2 * p.valStd,
  }));
}

/** Custom tooltip for the learning curve: the size plus both scores
 *  with their fold spread in one compact card (the default tooltip
 *  would list the stacked band internals). */
function LcTooltip({
  active,
  payload,
  metricName,
}: {
  active?: boolean;
  payload?: Array<{ payload?: LcRow }>;
  metricName: string;
}) {
  if (!active || !payload?.length) return null;
  const p = payload[0]?.payload;
  if (!p) return null;
  return (
    <div className="rounded-md border bg-background/95 px-2.5 py-1.5 text-xs shadow-sm">
      <p className="font-mono tabular-nums">train size = {p.size}</p>
      <p className="font-mono tabular-nums" style={{ color: CHART.teal }}>
        training {metricName} = {fmtNum(p.trainMean, 4)}
      </p>
      <p className="font-mono tabular-nums" style={{ color: CHART.amber }}>
        CV {metricName} = {fmtNum(p.valMean, 4)}
      </p>
    </div>
  );
}

/** Qualitative label for a Pearson correlation coefficient. */
function corrStrength(r: number): string {
  const a = Math.abs(r);
  if (a >= 0.9) return "very strong";
  if (a >= 0.7) return "strong";
  if (a >= 0.5) return "moderate";
  return "weak";
}

/** Diverging cell colour for the correlation heatmap (100x-153):
 * rose → white → teal over r ∈ [-1, 1] — the CHART house palette's
 * twins of the desktop plot's red→white→teal. */
function corrCellColor(r: number): string {
  const teal = [20, 184, 166]; // CHART.teal #14b8a6
  const rose = [244, 63, 94]; // CHART.rose #f43f5e
  const white = [255, 255, 255];
  const t = Math.max(-1, Math.min(1, r));
  const end = t >= 0 ? teal : rose;
  const alpha = Math.abs(t);
  const rgb = white.map((w, i) => Math.round(w + (end[i] ?? w) * alpha));
  return `rgb(${rgb[0] ?? 255}, ${rgb[1] ?? 255}, ${rgb[2] ?? 255})`;
}

/** The interpretability card (100x-153) — the desktop bundle's web
 *  twin: signed feature-target correlation bars plus the compact
 *  top-N matrix, over the run's OWN held-out rows (computed in the
 *  same request that trained the model, so the pairing is inherently
 *  swap-proof). Undefined correlations are dropped, never zeroed. */
function CorrelationCard({ bundle }: { bundle: CorrelationBundle }) {
  const bars = bundle.correlations.slice(0, 10);
  const matrix = bundle.matrix;
  return (
    <Card className="overflow-hidden py-0 shadow-sm">
      <CardHeader className="border-b bg-muted/30 py-3">
        <CardTitle className="text-sm flex items-center gap-1.5">
          <Activity className="size-3.5 text-primary" aria-hidden />
          Feature–target correlations
        </CardTitle>
        <CardDescription className="text-xs">
          Pearson r over the run's own {bundle.nRows} held-out rows — teal pulls {bundle.target} up,
          rose pulls it down; undefined correlations are dropped, never zeroed
        </CardDescription>
      </CardHeader>
      <CardContent className="pb-4">
        {bars.length > 0 ? (
          <ul className="space-y-2" aria-label="Signed feature-target correlations">
            {bars.map((c) => (
              <li key={c.name} className="space-y-1">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="truncate font-mono text-xs font-medium" title={c.name}>
                    {c.name}
                  </span>
                  <span className="shrink-0 font-mono text-xs tabular-nums text-muted-foreground">
                    {c.r >= 0 ? "+" : ""}
                    {fmtNum(c.r, 4)}
                    <span className="ml-1 text-muted-foreground/70">({corrStrength(c.r)})</span>
                  </span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-muted" role="presentation">
                  <div
                    className={`h-full rounded-full ${c.r >= 0 ? "bg-primary/70" : "bg-destructive/70"}`}
                    style={{ width: `${Math.min(Math.abs(c.r), 1) * 100}%` }}
                  />
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <p className="rounded-md border border-dashed px-3 py-2 text-[11px] leading-snug text-muted-foreground">
            No correlation was defined on the held-out rows (fewer than two points, or every
            feature column was constant there).
          </p>
        )}
        {matrix ? (
          <div className="mt-4 space-y-1.5">
            <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
              Correlation matrix — top {Math.max(matrix.columns.length - 1, 0)} |r| + {bundle.target}
            </p>
            <div className="thin-scrollbar overflow-x-auto">
              <table className="border-separate border-spacing-px font-mono text-[10px]">
                <thead>
                  <tr>
                    <th className="bg-card" aria-label="row label" />
                    {matrix.columns.map((col) => (
                      <th
                        key={col}
                        className="max-w-24 truncate bg-card px-1 py-0.5 text-center font-medium text-muted-foreground"
                        title={col}
                      >
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {matrix.values.map((row, i) => (
                    <tr key={matrix.columns[i] ?? i}>
                      <th
                        className="sticky left-0 z-[1] max-w-28 truncate bg-card px-1 py-0.5 text-left font-medium text-muted-foreground"
                        title={matrix.columns[i] ?? ""}
                      >
                        {matrix.columns[i] ?? ""}
                      </th>
                      {row.map((v, j) => {
                        const background = v === null ? "var(--muted)" : corrCellColor(v);
                        const darkCell = v !== null && Math.abs(v) >= 0.55;
                        return (
                          <td
                            key={`${i}-${j}`}
                            className="px-1.5 py-1 text-center tabular-nums"
                            style={{ background }}
                            title={
                              v === null
                                ? "undefined (a constant column on the held-out rows)"
                                : `r = ${fmtNum(v, 4)} (${corrStrength(v)})`
                            }
                          >
                            {v === null ? (
                              <span className="text-muted-foreground">—</span>
                            ) : darkCell ? (
                              <span className="text-white">{fmtNum(v, 2)}</span>
                            ) : (
                              fmtNum(v, 2)
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="flex items-center gap-2 pt-1">
              <span className="font-mono text-[10px] text-muted-foreground">−1</span>
              <div
                className="h-1.5 flex-1 rounded-full"
                style={{ background: "linear-gradient(90deg, #f43f5e, #ffffff, #14b8a6)" }}
                aria-hidden
              />
              <span className="font-mono text-[10px] text-muted-foreground">+1</span>
            </div>
            <p className="text-[11px] leading-snug text-muted-foreground">
              Computed in the same request that trained the model, on the held-out split only — a
              re-loaded dataset can never re-pair its target with an old run's features (the
              desktop round-36 audit's swap hazard, designed out here from the start).
            </p>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
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

/** CV mean ± std bars — the leaderboard's VISUAL form: the exact numbers
 *  the table tabulates, drawn as zero-axis bars (a negative CV R² reads
 *  honestly leftwards) with a ± std whisker per fold spread. Hand-rolled
 *  SVG in the house style of HsvrTrustScatter; deterministic like every
 *  other card (same seed → same bars). */
function CompareCvBars({
  entries,
  selectedModel,
}: {
  entries: MLCompareEntry[];
  selectedModel: string;
}) {
  const W = 560;
  const ROW = 26;
  const AXIS = 22;
  const LABEL_W = 122;
  const PAD_R = 52;
  const H = entries.length * ROW + AXIS;

  /* Domain: every mean and every whisker end must fit; a hair of
   * headroom so the value labels clear the frame. */
  const ends = entries.flatMap((e) => [e.cvMean - e.cvStd, e.cvMean + e.cvStd]);
  const lo = Math.min(0, ...ends) - 0.015;
  let hi = Math.max(0.02, ...ends) + 0.015;
  if (hi - lo < 0.05) hi = lo + 0.05;
  const x = (v: number) => LABEL_W + ((v - lo) / (hi - lo)) * (W - LABEL_W - PAD_R);
  const zeroX = x(0);

  const span = hi - lo;
  const step = span <= 0.25 ? 0.05 : span <= 0.5 ? 0.1 : span <= 1.05 ? 0.25 : 0.5;
  const ticks: number[] = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) {
    ticks.push(Math.round(t * 100) / 100);
  }

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-auto w-full"
      role="img"
      aria-label={`Comparison bars: ${entries.length} families ranked by CV ${entries[0]?.metricName ?? "score"} mean, whiskers show ± std across the shared folds`}
    >
      {/* Gridlines + axis ticks */}
      {ticks.map((t) => (
        <g key={`t-${t}`}>
          <line
            x1={x(t)}
            y1={4}
            x2={x(t)}
            y2={entries.length * ROW + 4}
            stroke={t === 0 ? "currentColor" : "currentColor"}
            strokeOpacity={t === 0 ? 0.35 : 0.12}
            strokeDasharray={t === 0 ? undefined : "2 3"}
          />
          <text
            x={x(t)}
            y={H - 6}
            textAnchor="middle"
            className="fill-muted-foreground font-mono text-[9px] tabular-nums"
          >
            {t.toFixed(2)}
          </text>
        </g>
      ))}
      {entries.map((e, i) => {
        const label = specOf("regression", e.model)?.label ?? specOf("classification", e.model)?.label ?? e.model;
        const short = label.length > 17 ? `${label.slice(0, 16)}…` : label;
        const y = i * ROW + 7;
        const midY = y + 6;
        const barX = Math.min(zeroX, x(e.cvMean));
        const barW = Math.max(1.5, Math.abs(x(e.cvMean) - zeroX));
        const whiskA = x(e.cvMean - e.cvStd);
        const whiskB = x(e.cvMean + e.cvStd);
        const isSelected = e.model === selectedModel;
        const isWinner = e.rank === 1;
        return (
          <g key={e.model}>
            <text
              x={LABEL_W - 10}
              y={midY + 3.5}
              textAnchor="end"
              className={`text-[10px] ${isSelected ? "fill-foreground font-semibold" : "fill-muted-foreground"}`}
            >
              {isWinner ? "★ " : ""}
              {short}
            </text>
            {/* The bar: from the zero axis; negative means extend left. */}
            <rect
              x={barX}
              y={y}
              width={barW}
              height={12}
              rx={2}
              className={cn(
                "transition-[fill] duration-300",
                isWinner
                  ? "fill-primary"
                  : isSelected
                    ? "fill-primary/55"
                    : "fill-muted-foreground/30",
              )}
            />
            {/* ± std whisker with end caps — the fold spread the ± column
             *  tabulates, drawn over the bar's mean end. */}
            <line
              x1={whiskA}
              y1={midY}
              x2={whiskB}
              y2={midY}
              stroke="currentColor"
              strokeOpacity={0.55}
              strokeWidth={1.5}
            />
            <line
              x1={whiskA}
              y1={midY - 4}
              x2={whiskA}
              y2={midY + 4}
              stroke="currentColor"
              strokeOpacity={0.55}
              strokeWidth={1.5}
            />
            <line
              x1={whiskB}
              y1={midY - 4}
              x2={whiskB}
              y2={midY + 4}
              stroke="currentColor"
              strokeOpacity={0.55}
              strokeWidth={1.5}
            />
            <text
              x={e.cvMean >= 0 ? Math.max(whiskB, x(e.cvMean)) + 5 : Math.min(whiskA, x(e.cvMean)) - 5}
              y={midY + 3.5}
              textAnchor={e.cvMean >= 0 ? "start" : "end"}
              className="fill-foreground/80 font-mono text-[9px] tabular-nums"
            >
              {fmtNum(e.cvMean, 3)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/* RandomSearch per-trial trajectory                                   */
/* ------------------------------------------------------------------ */

/** One row of the search-trajectory chart: the trial in draw order, its
 *  CV score, the best score seen up to that trial (the step line), the
 *  delta versus the default configuration, and the human-readable
 *  parameter assignment for the tooltip. */
interface TrajectoryPoint {
  trial: number;
  score: number;
  bestSoFar: number;
  delta: number;
  isWinner: boolean;
  config: string;
}

/** Custom tooltip for the search trajectory — trial, score, best-so-far,
 *  Δ versus defaults and the configuration in one compact card (the
 *  default ComposedChart tooltip would list the raw series keys). */
function TrajectoryTooltip({
  active,
  payload,
  objective,
}: {
  active?: boolean;
  payload?: Array<{ payload?: TrajectoryPoint }>;
  objective: string;
}) {
  if (!active || !payload?.length) return null;
  const p = payload[0]?.payload;
  if (!p) return null;
  return (
    <div className="rounded-md border bg-background/95 px-2.5 py-1.5 text-xs shadow-sm">
      <p className="font-mono tabular-nums">
        trial {p.trial}
        {p.isWinner ? " · best" : ""}
      </p>
      <p className="font-mono tabular-nums">
        {objective} = {fmtNum(p.score, 4)}
      </p>
      <p className="font-mono tabular-nums text-muted-foreground">
        best so far = {fmtNum(p.bestSoFar, 4)}
      </p>
      <p
        className={`font-mono tabular-nums ${
          p.delta >= 0
            ? "text-emerald-600 dark:text-emerald-400"
            : "text-destructive"
        }`}
      >
        Δ defaults {p.delta >= 0 ? "+" : ""}
        {fmtNum(p.delta, 4)}
      </p>
      <p className="max-w-56 truncate font-mono text-[10px] text-muted-foreground">
        {p.config}
      </p>
    </div>
  );
}

/**
 * The RandomSearch per-trial scatter — the trajectory view the leaderboard
 * table cannot show: every drawn configuration as a dot (x = trial number
 * in draw order, y = its mean CV score), the running-best step line in
 * emerald (how fast the sampling converges on good configurations), the
 * overall winner emphasised, and the default-configuration score as a
 * dashed amber reference line (the honest bar every trial is judged
 * against). Deterministic like every chart in the house: no animation.
 */

/** Module-scope pure builder for the trajectory rows — the running best
 *  is a fold over the draw-ordered trials.  Kept OUT of the component
 *  body so the React compiler never sees an accumulator reassigned
 *  during render (react-hooks/immutability). */
function buildTrajectoryPoints(
  trials: MLTuneTrial[],
  baselineScore: number
): TrajectoryPoint[] {
  const ordered = [...trials].sort((a, b) => a.trial - b.trial);
  const winner = ordered.reduce(
    (best, t) => (t.score > best.score ? t : best),
    ordered[0]!
  );
  let running = -Infinity;
  return ordered.map((t) => {
    running = Math.max(running, t.score);
    return {
      trial: t.trial,
      score: t.score,
      bestSoFar: running,
      delta: t.score - baselineScore,
      isWinner: t.trial === winner.trial,
      config:
        Object.entries(t.params)
          .map(([k, v]) => `${HP_LABELS[k] ?? k} ${fmtHp(v)}`)
          .join(" · ") || "defaults",
    };
  });
}

function RandomSearchTrajectory({
  trials,
  baselineScore,
  objective,
}: {
  trials: MLTuneTrial[];
  baselineScore: number;
  objective: string;
}) {
  const points = buildTrajectoryPoints(trials, baselineScore);
  const winnerPoint = points.find((p) => p.isWinner);

  /* Y extents across both series plus the baseline, padded 8% so the
   * reference line and the step line never kiss the frame. */
  const scores = points.map((p) => p.score);
  let yMin = Math.min(...scores, baselineScore);
  let yMax = Math.max(...scores, baselineScore);
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) {
    yMin = 0;
    yMax = 1;
  }
  if (yMax - yMin < 1e-9) {
    yMin -= 0.05;
    yMax += 0.05;
  }
  const yPad = (yMax - yMin) * 0.08;
  yMin -= yPad;
  yMax += yPad;
  const maxTrial = points.length ? points[points.length - 1]!.trial : 1;

  return (
    <div className="w-full">
      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={points} margin={{ top: 8, right: 14, bottom: 4, left: 0 }}>
            <CartesianGrid stroke={CHART.gridStroke} strokeOpacity={CHART.gridStrokeOpacity} />
            <XAxis
              type="number"
              dataKey="trial"
              domain={[1, Math.max(1, maxTrial)]}
              allowDuplicatedCategory={false}
              tick={CHART.axisTick}
              tickLine={false}
              axisLine={{ stroke: CHART.gridStroke, strokeOpacity: 0.4 }}
              label={{
                value: "trial (draw order)",
                position: "insideBottomRight",
                offset: -2,
                fill: "#78716c",
                fontSize: 11,
              }}
            />
            <YAxis
              type="number"
              dataKey="score"
              width={52}
              domain={[yMin, yMax]}
              tick={CHART.axisTick}
              tickFormatter={(v: number) => fmtNum(v, 2)}
              tickLine={false}
              axisLine={false}
              label={{
                value: `CV ${objective}`,
                angle: -90,
                position: "insideLeft",
                offset: 16,
                fill: "#78716c",
                fontSize: 11,
              }}
            />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              content={<TrajectoryTooltip objective={objective} />}
            />
            <ReferenceLine
              y={baselineScore}
              stroke={CHART.amber}
              strokeDasharray="6 3"
              strokeWidth={1.5}
              label={{
                value: "defaults",
                position: "insideTopLeft",
                fill: CHART.amber,
                fontSize: 10,
              }}
            />
            <Scatter
              name="trial score"
              dataKey="score"
              fill={CHART.stone}
              fillOpacity={0.7}
              isAnimationActive={false}
            />
            {winnerPoint ? (
              <Scatter
                name="winner"
                data={[winnerPoint]}
                dataKey="score"
                fill={CHART.emerald}
                fillOpacity={0.95}
                isAnimationActive={false}
              />
            ) : null}
            <Line
              name="best so far"
              dataKey="bestSoFar"
              type="stepAfter"
              stroke={CHART.emeraldDark}
              strokeWidth={2}
              dot={false}
              activeDot={false}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-1.5 flex flex-wrap items-center justify-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span
            className="size-2 rounded-full"
            style={{ backgroundColor: CHART.stone, opacity: 0.7 }}
            aria-hidden
          />
          trial score
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="size-2 rounded-full"
            style={{ backgroundColor: CHART.emerald }}
            aria-hidden
          />
          winner
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block h-0.5 w-4"
            style={{ backgroundColor: CHART.emeraldDark }}
            aria-hidden
          />
          best so far
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block h-0.5 w-4"
            style={{
              backgroundImage: `repeating-linear-gradient(to right, ${CHART.amber} 0 4px, transparent 4px 7px)`,
            }}
            aria-hidden
          />
          defaults CV
        </span>
      </div>
    </div>
  );
}

/* -- Hyperband per-trial rung scatter (100x-114) ------------------------ */

/** Rung colour progression: rung 0 — the cheapest, most crowded
 *  evaluations — is stone; deeper rungs warm through amber and teal to
 *  emerald, the survivors' colour. Cycled past six rungs (η ≥ 3 rarely
 *  produces more than five). */
const RUNG_COLORS = [
  "#a8a29e",
  CHART.amber,
  CHART.teal,
  CHART.emerald,
  CHART.emeraldDark,
  CHART.rose,
] as const;

function rungColor(rung: number): string {
  return RUNG_COLORS[Math.max(0, rung) % RUNG_COLORS.length]!;
}

/** Per-point shape: the winner is a diamond with a halo, promoted dots
 *  are solid with a ring, culled dots are faint — the halving story in
 *  one glance (dots march right as survivors are re-scored at higher
 *  budgets, and the cloud thins by η each rung). The bracket's own best
 *  (100x-116) wears an emerald ring so each facet panel's local winner
 *  is findable without reading the caption. */
function rungDot(props: unknown) {
  const { cx, cy, payload } = props as {
    cx?: number;
    cy?: number;
    payload?: RungPoint;
  };
  if (cx === undefined || cy === undefined || !payload) return <g />;
  const color = rungColor(payload.rung);
  if (payload.isWinner) {
    return (
      <g>
        <circle cx={cx} cy={cy} r={7} fill={color} fillOpacity={0.22} />
        <path
          d={`M ${cx} ${cy - 5} L ${cx + 4.5} ${cy} L ${cx} ${cy + 5} L ${cx - 4.5} ${cy} Z`}
          fill={color}
          stroke={CHART.emeraldDark}
          strokeWidth={1.2}
        />
      </g>
    );
  }
  if (payload.isBracketBest) {
    return (
      <circle
        cx={cx}
        cy={cy}
        r={4.6}
        fill={color}
        fillOpacity={0.9}
        stroke={CHART.emeraldDark}
        strokeWidth={1.4}
      />
    );
  }
  if (payload.promoted) {
    return (
      <circle
        cx={cx}
        cy={cy}
        r={4.2}
        fill={color}
        fillOpacity={0.9}
        stroke={color}
        strokeWidth={1.4}
      />
    );
  }
  return <circle cx={cx} cy={cy} r={3.2} fill={color} fillOpacity={0.38} />;
}

function RungScatterTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload?: RungPoint }>;
}) {
  if (!active || !payload?.length) return null;
  const p = payload[0]?.payload;
  if (!p) return null;
  return (
    <div className="rounded-md border bg-background/95 px-2.5 py-1.5 text-xs shadow-sm">
      <p className="font-mono tabular-nums">trial {p.trial}</p>
      <p className="font-mono tabular-nums">
        budget = {p.x} · CV score = {fmtNum(p.y, 4)}
      </p>
      <p className="font-mono tabular-nums text-muted-foreground">
        bracket {p.bracket} (0 = full budget) · rung {p.rung}
      </p>
      <p className="text-muted-foreground">
        {p.isWinner
          ? "winner"
          : p.isBracketBest
            ? "bracket best — this bracket's own winner"
            : p.promoted
              ? "promoted — re-scored at the next rung"
              : "culled at this rung"}
      </p>
    </div>
  );
}

/** Every Hyperband evaluation as budget-vs-score dots, coloured by rung.
 *  The X axis is the resource the config was actually scored at, so a
 *  promoted config appears once per rung it survived — the classic
 *  successive-halving picture the leaderboard table cannot show.
 *
 *  Two views (100x-115): "All brackets" overlays every bracket in one
 *  field — the budget ladder as a whole; "Per bracket" is the small-
 *  multiples edition, one mini panel per bracket on SHARED axes, where
 *  the breadth-vs-depth trade the η schedule makes becomes readable at
 *  a glance (bracket 0: many configs at full budget; the last: few
 *  configs, mostly cheap rungs). */
function HyperbandRungScatter({
  trials,
  maxResource,
  resourceLabel,
  baselineScore,
  objective,
}: {
  trials: MLTuneTrial[];
  maxResource: number;
  resourceLabel: string;
  baselineScore: number;
  objective: string;
}) {
  const [facet, setFacet] = useState(false);

  /* The rows (filter → per-bracket best fold → map) live in the shipped
   * pure module src/lib/hyperband-rungs.ts since 100x-117 — probed by
   * scratch/probe_rung_fold_117.ts (bun) and guarded by
   * tests/test_web_parity.py — following the bayes-acquisition.ts
   * precedent. Only `points` is consumed below. */
  const points: RungPoint[] = buildRungPoints(trials);

  if (points.length === 0) return null;

  const yValues = points.map((p) => p.y);
  let yMin = Math.min(...yValues, baselineScore);
  let yMax = Math.max(...yValues, baselineScore);
  if (yMax - yMin < 1e-9) {
    yMin -= 0.05;
    yMax += 0.05;
  }
  const yPad = (yMax - yMin) * 0.08;
  yMin -= yPad;
  yMax += yPad;

  const xValues = points.map((p) => p.x);
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues, maxResource);
  const xPad = Math.max(1, (xMax - xMin) * 0.06);
  const xLo = Math.max(0, xMin - xPad);
  const xHi = xMax + xPad;

  const rungsUsed = [...new Set(points.map((p) => p.rung))].sort((a, b) => a - b);
  const bracketsUsed = [...new Set(points.map((p) => p.bracket))].sort((a, b) => a - b);
  const promotedCount = points.filter((p) => p.promoted).length;
  const winnerPoint = points.find((p) => p.isWinner);

  const overlayLabel = `Hyperband rung scatter: ${points.length} evaluations across ${
    bracketsUsed.length
  } brackets, ${promotedCount} promoted to a higher rung; best CV ${objective} ${fmtNum(
    points[0]!.y,
    4,
  )} at ${resourceLabel} ${points[0]!.x}; dashed line marks the defaults CV ${fmtNum(
    baselineScore,
    4,
  )}`;

  return (
    <div className="w-full">
      {bracketsUsed.length > 1 ? (
        <div
          className="mb-2 flex rounded-lg border bg-muted/40 p-0.5"
          role="group"
          aria-label="Rung scatter view"
        >
          {(
            [
              { key: false, label: "All brackets" },
              { key: true, label: "Per bracket" },
            ] as const
          ).map(({ key, label }) => (
            <button
              key={label}
              type="button"
              aria-pressed={facet === key}
              onClick={() => setFacet(key)}
              className={`h-7 rounded-md px-2.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring ${
                facet === key
                  ? "bg-background shadow-sm text-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      ) : null}

      {facet ? (
        <div
          className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3"
          role="img"
          aria-label={`Hyperband bracket facets: ${points.length} evaluations split one panel per bracket (${bracketsUsed.length} brackets), shared axes — budget on X, CV ${objective} on Y; best CV ${fmtNum(
            winnerPoint?.y ?? points[0]!.y,
            4,
          )} at ${resourceLabel} ${winnerPoint?.x ?? points[0]!.x}; each panel's caption names that bracket's own best CV; dashed lines mark the full budget and the defaults CV ${fmtNum(
            baselineScore,
            4,
          )}`}
        >
          {bracketsUsed.map((bracket) => {
            const facetPoints = points.filter((p) => p.bracket === bracket);
            const facetPromoted = facetPoints.filter((p) => p.promoted).length;
            const hasWinner = facetPoints.some((p) => p.isWinner);
            const facetBest = facetPoints.reduce(
              (b, p) => (p.y > b.y ? p : b),
              facetPoints[0]!
            );
            return (
              <figure key={bracket} className="m-0">
                <figcaption className="mb-1 flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
                  <span className="font-medium text-foreground">bracket {bracket}</span>
                  <span>
                    {new Set(facetPoints.map((p) => p.trial)).size} configs ·{" "}
                    {facetPromoted} promoted
                  </span>
                  <span
                    className={`inline-flex items-center gap-1 rounded-full px-1.5 py-px font-medium tabular-nums ${
                      hasWinner
                        ? "text-emerald-700 dark:text-emerald-400"
                        : "text-foreground/75"
                    }`}
                    style={{
                      backgroundColor: hasWinner
                        ? `${CHART.emerald}26`
                        : "rgba(120,113,108,0.16)",
                    }}
                  >
                    {hasWinner ? (
                      <span
                        className="inline-block size-1.5 rotate-45"
                        style={{ backgroundColor: CHART.emerald }}
                        aria-hidden
                      />
                    ) : null}
                    best CV {fmtNum(facetBest.y, 4)}
                  </span>
                  <span className="ml-auto tabular-nums">
                    {bracket === 0 ? "full budget" : `rungs up to ${Math.max(...facetPoints.map((p) => p.rung))}`}
                  </span>
                </figcaption>
                <div className="h-28 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 4, right: 6, bottom: 0, left: 0 }}>
                      <CartesianGrid
                        stroke={CHART.gridStroke}
                        strokeOpacity={CHART.gridStrokeOpacity}
                      />
                      <XAxis
                        type="number"
                        dataKey="x"
                        domain={[xLo, xHi]}
                        allowDecimals={false}
                        tick={{ fill: "#78716c", fontSize: 9 }}
                        tickLine={false}
                        axisLine={{ stroke: CHART.gridStroke, strokeOpacity: 0.4 }}
                      />
                      <YAxis
                        type="number"
                        dataKey="y"
                        width={38}
                        domain={[yMin, yMax]}
                        tick={{ fill: "#78716c", fontSize: 9 }}
                        tickFormatter={(v: number) => fmtNum(v, 2)}
                        tickLine={false}
                        axisLine={false}
                      />
                      <Tooltip
                        cursor={{ strokeDasharray: "3 3" }}
                        content={<RungScatterTooltip />}
                        isAnimationActive={false}
                      />
                      <ReferenceLine
                        x={maxResource}
                        stroke={CHART.stone}
                        strokeDasharray="6 3"
                        strokeWidth={1.2}
                      />
                      <ReferenceLine
                        y={baselineScore}
                        stroke={CHART.amber}
                        strokeDasharray="6 3"
                        strokeWidth={1.2}
                      />
                      <Scatter
                        name="evaluation"
                        data={facetPoints}
                        shape={rungDot}
                        isAnimationActive={false}
                      />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </figure>
            );
          })}
        </div>
      ) : (
        <div className="h-56 w-full" role="img" aria-label={overlayLabel}>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 8, right: 14, bottom: 4, left: 0 }}>
              <CartesianGrid stroke={CHART.gridStroke} strokeOpacity={CHART.gridStrokeOpacity} />
              <XAxis
                type="number"
                dataKey="x"
                domain={[xLo, xHi]}
                allowDecimals={false}
                tick={CHART.axisTick}
                tickLine={false}
                axisLine={{ stroke: CHART.gridStroke, strokeOpacity: 0.4 }}
                label={{
                  value: `${resourceLabel} (scoring budget)`,
                  position: "insideBottomRight",
                  offset: -2,
                  fill: "#78716c",
                  fontSize: 11,
                }}
              />
              <YAxis
                type="number"
                dataKey="y"
                width={52}
                domain={[yMin, yMax]}
                tick={CHART.axisTick}
                tickFormatter={(v: number) => fmtNum(v, 2)}
                tickLine={false}
                axisLine={false}
                label={{
                  value: `CV ${objective}`,
                  angle: -90,
                  position: "insideLeft",
                  offset: 16,
                  fill: "#78716c",
                  fontSize: 11,
                }}
              />
              <Tooltip
                cursor={{ strokeDasharray: "3 3" }}
                content={<RungScatterTooltip />}
                isAnimationActive={false}
              />
              <ReferenceLine
                x={maxResource}
                stroke={CHART.stone}
                strokeDasharray="6 3"
                strokeWidth={1.5}
                label={{
                  value: "full budget",
                  position: "insideTopRight",
                  fill: CHART.stone,
                  fontSize: 10,
                }}
              />
              <ReferenceLine
                y={baselineScore}
                stroke={CHART.amber}
                strokeDasharray="6 3"
                strokeWidth={1.5}
                label={{
                  value: "defaults",
                  position: "insideTopLeft",
                  fill: CHART.amber,
                  fontSize: 10,
                }}
              />
              <Scatter
                name="evaluation"
                data={points}
                shape={rungDot}
                isAnimationActive={false}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
      <div className="mt-1.5 flex flex-wrap items-center justify-center gap-x-3.5 gap-y-1 text-[11px] text-muted-foreground">
        {rungsUsed.map((r) => (
          <span key={r} className="flex items-center gap-1.5">
            <span
              className="size-2 rounded-full"
              style={{ backgroundColor: rungColor(r) }}
              aria-hidden
            />
            rung {r}
          </span>
        ))}
        <span className="flex items-center gap-1.5">
          <span
            className="size-2.5 rounded-full ring-1 ring-stone-500"
            style={{ backgroundColor: CHART.stone }}
            aria-hidden
          />
          promoted (re-scored)
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block size-2.5 rounded-full"
            style={{
              backgroundColor: CHART.emerald,
              boxShadow: `0 0 0 1.5px ${CHART.emeraldDark}`,
            }}
            aria-hidden
          />
          bracket best
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block size-2 rotate-45"
            style={{ backgroundColor: CHART.emerald }}
            aria-hidden
          />
          winner
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block h-0.5 w-4"
            style={{
              backgroundImage: `repeating-linear-gradient(to right, ${CHART.amber} 0 4px, transparent 4px 7px)`,
            }}
            aria-hidden
          />
          defaults CV
        </span>
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
  /** GP only: when on, the kernel hyperparameters are LEARNED server-side
   *  by the deterministic LML grid search (the sliders stay disabled at
   *  the learned values); unticking pins the kernel at the sliders. */
  const [gpLearn, setGpLearn] = useState(true);

  /* Tuner state — transient, mirroring ml tune on the server. maxTrials
   * rides in state so the budget is visible before searching; the searcher
   * toggle picks RandomSearch vs Hyperband (successive halving). */
  const [tuneStatus, setTuneStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [tuneResult, setTuneResult] = useState<MLTuneResponse | null>(null);
  const [tuneError, setTuneError] = useState<string | null>(null);
  /* Model comparison (the web mirror of pharmaml's compare_models):
   *  2–6 registry keys trained side by side on the SAME folds. */
  const [compareSel, setCompareSel] = useState<string[]>(() =>
    defaultCompareSel(task, modelSel)
  );
  const [compareResult, setCompareResult] = useState<MLCompareResponse | null>(null);
  const [compareStatus, setCompareStatus] = useState<"idle" | "running">("idle");
  const [compareError, setCompareError] = useState<string | null>(null);
  /* Applicability-domain state (the 2026-10 upload's workflow, web
   * edition): the analysis re-runs the LAST train request's split, so
   * the verdict names the very held-out rows the parity chart drew. */
  const [adStatus, setAdStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const [adResult, setAdResult] = useState<MLAdResponse | null>(null);
  const [adError, setAdError] = useState<string | null>(null);
  const [adK, setAdK] = useState("5");
  const [adPct, setAdPct] = useState("95");
  const [adModels, setAdModels] = useState("20");
  /* Learning curve (100x-125): re-runs the LAST trained configuration
   * at growing train sizes — works for synthetic AND CSV runs (the
   * curve judges the parked request, not just a loaded file).  The
   * result lives in the store (100x-127) so the session-report
   * collector can walk it; it still survives sub-tab switches. */
  const [lcStatus, setLcStatus] = useState<"idle" | "running" | "done" | "error">("idle");
  const lcResult = useToolkitStore((s) => s.lcResult);
  const setLcResult = useToolkitStore((s) => s.setLcResult);
  const [lcError, setLcError] = useState<string | null>(null);
  const [lcFolds, setLcFolds] = useState("5");
  const lastTrainBodyRef = useRef<MLTrainRequest | null>(null);
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
    let nextModel = modelSel;
    if (!specOf(next, modelSel)) {
      nextModel =
        next === "regression" ? (specOf("regression", regModel) ? regModel : "linear") : "logistic";
      setModelSel(nextModel);
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
    setCompareSel(defaultCompareSel(next, nextModel));
    setCompareResult(null);
    setCompareError(null);
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
    setCompareSel(defaultCompareSel("regression", "linear"));
    setCompareResult(null);
    setCompareError(null);
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
     * everything else from the slider state. GP exception: with Learn
     * kernel on, the three kernel slots are NOT sent — the request leaves
     * them unset and the server's LML grid search picks them. */
    const gpAuto = modelSel === "gaussian_process" && gpLearn;
    for (const h of modelSpec.hp) {
      if (h.control === "input") {
        if (h.key === "alpha") body.alpha = Number.parseFloat(alpha);
        else if (h.key === "l2") body.l2 = Number.parseFloat(l2);
      } else if (!(gpAuto && (h.key === "lengthScale" || h.key === "signalStd" || h.key === "noiseStd"))) {
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

    /* Park the exact train request for the applicability-domain card —
     * the AD route re-runs this split (content-hashed seed), so its
     * verdict names the very rows this run's parity chart drew. */
    lastTrainBodyRef.current = body;

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
      /* Learned GP kernel: park the sliders at the values the LML grid
       *  search picked so the panel shows what actually ran (they stay
       *  disabled while Learn kernel is on — the next auto run re-learns).
       *  Classification parks only (ℓ, σf) — no noise slot exists there. */
      if (payload.gp?.kernelTuned) {
        setHp((prev) => ({
          ...prev,
          lengthScale: payload.gp!.lengthScale,
          signalStd: payload.gp!.signalStd,
          noiseStd: payload.gp!.noiseStd,
        }));
      } else if (payload.gpCls?.kernelTuned) {
        setHp((prev) => ({
          ...prev,
          lengthScale: payload.gpCls!.lengthScale,
          signalStd: payload.gpCls!.signalStd,
        }));
      }
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

  /** The uploaded Applicability_domain_analysis.py workflow, web
   *  edition: re-run the last train request's split, then judge every
   *  held-out row by its mean kNN distance (inside the training
   *  descriptor space?) and — for regression models — the bootstrap
   *  ensemble's per-row spread (how sure is the model?), folding both
   *  into the upload's reliable/not-reliable verdict. */
  async function runAd() {
    const last = lastTrainBodyRef.current;
    if (!last || !last.csv) {
      setAdError("Train on a loaded CSV first — the analysis judges that run's held-out rows.");
      setAdStatus("error");
      return;
    }
    const kNum = Number.parseInt(adK, 10);
    const pctNum = Number.parseFloat(adPct);
    const modelsNum = Number.parseInt(adModels, 10);
    if (!Number.isInteger(kNum) || kNum < 1 || kNum > 20) {
      setAdError("k (neighbours) must be an integer between 1 and 20.");
      setAdStatus("error");
      return;
    }
    if (!Number.isFinite(pctNum) || pctNum < 50 || pctNum > 99.9) {
      setAdError("Threshold percentile must be between 50 and 99.9.");
      setAdStatus("error");
      return;
    }
    if (!Number.isInteger(modelsNum) || modelsNum < 0 || modelsNum > 60) {
      setAdError("Bootstrap models must be an integer between 0 (quick pass) and 60.");
      setAdStatus("error");
      return;
    }
    setAdStatus("running");
    setAdError(null);
    const body: Record<string, unknown> = {
      task: last.task,
      csv: last.csv,
      csvName: last.csvName,
      model: last.model,
      testSize: last.testSize,
      featureColumns: last.featureColumns,
      featureColumn: last.featureColumn,
      featureColumn2: last.featureColumn2,
      targetColumn: last.targetColumn,
      adK: kNum,
      adPercentile: pctNum,
      adModels: modelsNum,
    };
    /* The hyperparameters the trained run used — the bootstrap members
     * clone the same configuration. */
    for (const key of [
      "alpha", "l1Ratio", "l2", "k", "maxDepth", "nEstimators", "rounds",
      "learningRate", "hidden", "epochs", "lr", "C", "epsilon", "gamma",
      "nClusters", "radiusFactor", "nComponents", "lengthScale", "signalStd",
      "noiseStd",
    ] as const) {
      const value = (last as unknown as Record<string, unknown>)[key];
      if (value !== undefined) body[key] = value;
    }
    try {
      const res = await fetch("/api/ml/ad", {
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
      setAdResult(data as MLAdResponse);
      setAdStatus("done");
    } catch (err) {
      setAdStatus("error");
      setAdError(err instanceof Error ? err.message : "The analysis request failed.");
    }
  }

  /** Plot the learning curve of the last trained configuration —
   *  the web mirror of the desktop ML card's "Learning curve" button
   *  (pharmaml.viz.plot_learning_curve): the parked train request is
   *  re-sent, the server refits it at the library's default 8-step
   *  size schedule with k-fold CV at every size, and both curves land
   *  here with mean ± std bands and the gap verdict. */
  async function runLearningCurve() {
    const last = lastTrainBodyRef.current;
    if (!last) {
      setLcError("Train a model first — the curve re-runs that configuration at growing train sizes.");
      setLcStatus("error");
      return;
    }
    setLcStatus("running");
    setLcError(null);
    const body: Record<string, unknown> = {
      ...last,
      folds: Number.parseInt(lcFolds, 10) as 3 | 5 | 10,
    };
    try {
      const res = await fetch("/api/ml/learning-curve", {
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
      setLcResult(data as MLLearningCurveResponse);
      setLcStatus("done");
    } catch (err) {
      setLcError(err instanceof Error ? err.message : "The learning-curve request failed.");
      setLcStatus("error");
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
        `Hyperband needs a resource hyperparameter — ${modelSpec.label} has none. Pick one of ${hyperbandRoster()}, or switch to RandomSearch.`
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
                )} (${fmtDelta(payload.best.improvement)} vs defaults)`,
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

  /** Keep the comparison selection anchored on the model being explored:
   *  picking a model that isn't selected inserts it at the head (what you
   *  see is what you compare), capped at the server's 6-model maximum. */
  useEffect(() => {
    setCompareSel((sel) => {
      if (sel.includes(modelSel)) return sel;
      return [modelSel, ...sel].slice(0, 6);
    });
  }, [modelSel]);

  /** Toggle one model in the comparison selection (2–6 picked to run). */
  function toggleCompareModel(id: string) {
    setCompareSel((sel) => {
      if (sel.includes(id)) return sel.filter((m) => m !== id);
      return sel.length >= 6 ? sel : [...sel, id];
    });
  }

  /** Train the selected 2–6 models side by side at their default
   *  hyperparameters — the web mirror of `ml compare` /
   *  `train_model(..., compare_models)`. Same folds for every model, so
   *  the ranking is a fair comparison; same seed → same leaderboard. */
  async function runCompare() {
    if (source === "synthetic" && !seedValid) {
      setCompareError("Seed must be a non-negative integer (0 – 2147483647).");
      return;
    }
    if (source === "csv" && (!mlCsv || !csvUsable)) {
      setCompareError(
        `Load a CSV with at least ${MIN_CSV_ROWS} rows and 2 numeric columns before comparing.`
      );
      return;
    }
    if (compareSel.length < 2) {
      setCompareError("Pick at least 2 models to compare (6 at most).");
      return;
    }
    setCompareStatus("running");
    setCompareError(null);
    const body: MLCompareRequest = {
      task,
      models: compareSel,
      /* The comparison always cross-validates; CV=Off falls back to 5-fold. */
      cv: (Number.parseInt(cv, 10) >= 3 ? Number.parseInt(cv, 10) : 5) as 3 | 5 | 10,
      seed: Number.parseInt(seed, 10) || 0,
      testSize: testSizePct / 100,
      samples,
      noise,
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
      const res = await fetch("/api/ml/compare", {
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
      const payload = data as MLCompareResponse;
      setCompareResult(payload);
      const winner = payload.entries[0];
      const winnerLabel =
        specOf(payload.task, winner?.model ?? "")?.label ?? winner?.model ?? "?";
      toast.success("Comparison complete", {
        description: `${payload.entries.length} models · ${payload.cvK}-fold ${payload.objective} · #1 ${winnerLabel} ${fmtNum(winner?.cvMean ?? Number.NaN, 4)}`,
      });
    } catch (err) {
      setCompareResult(null);
      setCompareError(err instanceof Error ? err.message : "Comparison request failed.");
    } finally {
      setCompareStatus("idle");
    }
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

  /* Boundary confidence fade: GP classification and the soft-rule
   *  ensembles both carry calibrated per-cell probabilities (|p − 0.5|
   *  → dot opacity) — the legend and the dot shape follow it. */
  const probaFade =
    !!result &&
    result.task === "classification" &&
    (!!result.gpCls || result.ensemble?.rule === "soft");

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

  /** GP band chart data: the curve points with their ±1.96σ band, shaped
   *  for the stacked-area recipe (bandBase + bandSpan). Only the
   *  single-feature x↔y chart renders the band; multivariate predicted-
   *  vs-actual draws per-point whiskers instead. */
  const bandCurve = useMemo(() => {
    const curve = result?.series.curve;
    if (!result || !curve || curve.length === 0 || predActual) return null;
    if (curve[0]?.lo === undefined || curve[0]?.hi === undefined) return null;
    return curve.map((p) => ({
      x: p.x,
      yhat: p.yhat,
      bandBase: p.lo ?? p.yhat,
      bandSpan: Math.max((p.hi ?? p.yhat) - (p.lo ?? p.yhat), 0),
    }));
  }, [result, predActual]);

  /* -- Bayesian-optimisation overlay on the GP band (since 100x-113) -- */

  /** Acquisition overlay mode: where should the NEXT experiment go? */
  const [acqMode, setAcqMode] = useState<"off" | "ei" | "ucb">("off");
  /** UCB exploration parameter κ (bigger → wider trust in σ). */
  const [ucbKappa, setUcbKappa] = useState(1);

  /** Best observed y on the TRAINING split (EI's f*); falls back to the
   *  test points for older payloads that predate series.train. */
  const bestObserved = useMemo(() => {
    const pts = result?.series.train?.length
      ? result.series.train
      : (result?.series.test ?? []);
    return pts.length ? Math.max(...pts.map((p) => p.y)) : null;
  }, [result]);

  /** The band curve with per-point acquisition values attached (and the
   *  argmax marker). Null when the overlay is off or the band is absent.
   *  σ at each x is recovered from the ±1.96σ band: span / 3.92. */
  const acqCurve = useMemo(() => {
    if (!bandCurve || acqMode === "off" || bestObserved === null) return null;
    let bestX: number | null = null;
    let bestVal = -Infinity;
    const rows = bandCurve.map((p) => {
      const sigma = p.bandSpan / 3.92;
      const value =
        acqMode === "ei"
          ? expectedImprovement(p.yhat, sigma, bestObserved)
          : upperConfidenceBound(p.yhat, sigma, ucbKappa);
      if (value > bestVal) {
        bestVal = value;
        bestX = p.x;
      }
      return { ...p, acq: value };
    });
    return bestX === null ? null : { rows, bestX, bestVal };
  }, [bandCurve, acqMode, ucbKappa, bestObserved]);

  /** GP on the predicted-vs-actual chart: ŷ whiskers per test point. */
  const hasTestErr = useMemo(
    () => (result?.series.test.some((p) => p.err !== undefined) ?? false) && predActual,
    [result, predActual]
  );

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
              {/* GP: the kernel learn/auto toggle — LML grid search on the
               *  server vs pinned slider values (classification searches
               *  (ℓ, σf) by the Laplace lml — no noise slot exists). */}
              {modelSpec.id === "gaussian_process" ? (
                <label className="flex cursor-pointer items-start gap-2.5 rounded-lg border border-primary/20 bg-primary/5 px-3 py-2.5">
                  <input
                    type="checkbox"
                    checked={gpLearn}
                    onChange={(e) => setGpLearn(e.target.checked)}
                    className="mt-0.5 size-4 accent-primary"
                    aria-label="Learn the GP kernel by marginal likelihood"
                  />
                  <span className="space-y-0.5">
                    <span className="block text-xs font-medium">
                      Learn kernel {task === "classification" ? "(Laplace lml)" : "(marginal likelihood)"}
                    </span>
                    <span className="block text-[11px] leading-snug text-muted-foreground">
                      {task === "classification"
                        ? gpLearn
                          ? "The server grid-searches ℓ / σf by the Laplace log marginal likelihood and pins the sliders at the winners — untick to set them yourself."
                          : "Kernel pinned at the sliders — sweep them and watch the reported Laplace lml respond."
                        : gpLearn
                          ? "The server grid-searches ℓ / σf / σn by log marginal likelihood and pins the sliders at the winners — untick to set them yourself."
                          : "Kernel pinned at the sliders — sweep them and watch the reported log marginal likelihood respond."}
                    </span>
                  </span>
                </label>
              ) : null}
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
                    disabled={
                      modelSpec.id === "gaussian_process" &&
                      gpLearn &&
                      (h.key === "lengthScale" || h.key === "signalStd" || h.key === "noiseStd")
                    }
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
                    : `Hyperband needs a resource hyperparameter — ${hyperbandRoster()} qualify. Pick one or switch to Random.`}
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

          {/* ----------------- Model comparison (compare_models) ----------------- */}
          <div className="space-y-2.5 border-t pt-4">
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <p className="flex items-center gap-1.5 text-xs font-semibold">
                  <Trophy className="size-3.5 text-primary" aria-hidden />
                  Compare models
                </p>
                <p className="mt-0.5 text-[11px] leading-snug text-muted-foreground">
                  2–6 families, default hyperparameters, the same folds — the honest
                  leaderboard (the web mirror of{" "}
                  <code className="font-mono text-[10px]">compare_models</code>).
                </p>
              </div>
              <span
                className="shrink-0 rounded-full border bg-muted/50 px-2 py-0.5 text-[10px] font-medium tabular-nums text-muted-foreground"
                aria-live="polite"
              >
                {compareSel.length}/6
              </span>
            </div>
            <div
              className="flex flex-wrap gap-1.5"
              role="group"
              aria-label="Models to compare"
            >
              {specs.map((m) => {
                const picked = compareSel.includes(m.id);
                return (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => toggleCompareModel(m.id)}
                    aria-pressed={picked}
                    aria-label={`${picked ? "Remove" : "Add"} ${m.label} ${
                      picked ? "from" : "to"
                    } the comparison`}
                    className={cn(
                      "rounded-full border px-2.5 py-1 text-[11px] font-medium transition-colors",
                      picked
                        ? "border-primary/40 bg-primary/10 text-primary"
                        : "border-border bg-muted/40 text-muted-foreground hover:border-primary/30 hover:text-foreground"
                    )}
                  >
                    {m.label.split(" (")[0]}
                  </button>
                );
              })}
            </div>
            <Button
              size="sm"
              variant="outline"
              className="h-9 w-full gap-2"
              onClick={() => void runCompare()}
              disabled={compareStatus === "running" || compareSel.length < 2}
            >
              {compareStatus === "running" ? (
                <>
                  <Loader2 className="size-3.5 animate-spin" aria-hidden />
                  Comparing…
                </>
              ) : (
                <>
                  <Trophy className="size-3.5" aria-hidden />
                  Compare {Math.max(compareSel.length, 2)} model
                  {compareSel.length === 2 ? "" : "s"}
                </>
              )}
            </Button>
            {compareError ? (
              <p className="text-[11px] leading-snug text-destructive">{compareError}</p>
            ) : null}
          </div>
        </CardContent>
      </Card>

      {/* ------------------------------ Results ------------------------------ */}
      <div className="min-w-0 space-y-4">
        {compareResult ? (
          <Card className="overflow-hidden py-0 shadow-sm">
            <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-2 border-b bg-muted/30 py-3">
              <div className="min-w-0">
                <CardTitle className="flex items-center gap-1.5 text-sm">
                  <Trophy className="size-4 text-primary" aria-hidden />
                  Model comparison
                </CardTitle>
                <CardDescription className="text-xs">
                  {compareResult.entries.length} families · default hyperparameters ·{" "}
                  {compareResult.cvK}-fold CV {compareResult.objective} ·{" "}
                  {compareResult.datasetMeta.source
                    ? `${compareResult.datasetMeta.nTotal} rows from ${
                        compareResult.datasetMeta.source.fileName ?? "pasted.csv"
                      }`
                    : `seed ${Number.parseInt(seed, 10) || 0} · ${samples} samples`}
                </CardDescription>
              </div>
              <Button
                variant="ghost"
                size="sm"
                className="h-8 shrink-0 gap-1.5 px-2.5 text-xs"
                onClick={() => setCompareResult(null)}
                aria-label="Dismiss the comparison leaderboard"
              >
                <X className="size-3.5" aria-hidden />
                <span className="hidden sm:inline">Dismiss</span>
              </Button>
            </CardHeader>
            <CardContent className="p-0">
              {/* The visual form of the leaderboard — bars before the
               *  table: the eye gets the ranking, the table the digits. */}
              <div className="px-4 pb-2 pt-3">
                <CompareCvBars
                  entries={compareResult.entries}
                  selectedModel={modelSel}
                />
                <p className="mt-1.5 text-[11px] leading-snug text-muted-foreground">
                  Bars: CV {compareResult.objective} mean from the zero axis — a negative
                  bar reads honestly leftwards. Whisker: ± std across the{" "}
                  {compareResult.cvK} shared folds (the same folds every family
                  scored, so bar edges are comparable). ★ marks the winner.
                </p>
              </div>
              <div className="thin-scrollbar overflow-x-auto">
                <Table className="data-table">
                  <TableHeader>
                    <TableRow className="bg-muted/40 hover:bg-muted/40">
                      <TableHead className="w-10 text-center">#</TableHead>
                      <TableHead>Model</TableHead>
                      <TableHead className="text-right">
                        CV {compareResult.objective}
                      </TableHead>
                      <TableHead className="text-right">Held-out</TableHead>
                      <TableHead className="text-right">Δ vs #1</TableHead>
                      <TableHead className="text-right">Fit</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {compareResult.entries.map((entry) => {
                      const spec = specOf(compareResult.task, entry.model);
                      return (
                        <TableRow
                          key={entry.model}
                          className={cn(
                            "font-mono text-xs tabular-nums",
                            entry.rank === 1 && "bg-primary/5"
                          )}
                        >
                          <TableCell className="text-center text-muted-foreground">
                            {entry.rank}
                          </TableCell>
                          <TableCell className="font-sans">
                            <span className="font-medium">{spec?.label ?? entry.model}</span>
                            {entry.model === modelSel ? (
                              <span className="ml-1.5 rounded-full border border-primary/30 bg-primary/10 px-1.5 py-0.5 text-[9px] font-semibold text-primary">
                                selected
                              </span>
                            ) : null}
                          </TableCell>
                          <TableCell className="text-right">
                            {fmtNum(entry.cvMean, 4)}
                            <span className="text-muted-foreground">
                              {" "}
                              ± {fmtNum(entry.cvStd, 3)}
                            </span>
                          </TableCell>
                          <TableCell className="text-right">
                            {fmtNum(entry.testMetric, 4)}
                          </TableCell>
                          <TableCell className="text-right text-muted-foreground">
                            {/* fmtDelta: fixed decimals, never exponential —
                             *  a near-tie reads as -0.0000 ("essentially
                             *  tied"), not -7.3e-6. */}
                            {entry.rank === 1 ? "—" : fmtDelta(entry.delta)}
                          </TableCell>
                          <TableCell className="text-right text-muted-foreground">
                            {entry.fitMs < 10
                              ? `${fmtNum(entry.fitMs, 1)} ms`
                              : `${Math.round(entry.fitMs)} ms`}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>
              <p className="border-t px-3 py-2 text-[11px] leading-snug text-muted-foreground">
                Same data and folds for every family — only the model varies, so the
                ranking is a fair comparison. Every row is the default-configuration
                run (tune a family first, then compare, to see its tuned form).
                Deterministic: same seed → same leaderboard.
              </p>
            </CardContent>
          </Card>
        ) : null}
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
                the {REG_MODELS.length} regression or {CLS_MODELS.length} classification models,
                then press{" "}
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
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div className="min-w-0">
                    <CardTitle className="text-sm">
                      {result.task === "regression"
                        ? bandCurve
                          ? acqCurve
                            ? "GP mean, 95% band & acquisition"
                            : "GP mean & 95% predictive band"
                          : predActual
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
                  </div>
                  {bandCurve ? (
                    /* Bayesian-optimisation overlay switch — only the GP
                     * band chart carries σ, so only it can answer "where
                     * should the next experiment go?". */
                    <div className="flex flex-wrap items-center gap-2">
                      <div
                        className="flex rounded-lg border bg-muted/40 p-0.5"
                        role="group"
                        aria-label="Acquisition overlay"
                      >
                        {(["off", "ei", "ucb"] as const).map((m) => (
                          <button
                            key={m}
                            type="button"
                            aria-pressed={acqMode === m}
                            onClick={() => setAcqMode(m)}
                            className={`h-7 rounded-md px-2.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring ${
                              acqMode === m
                                ? "bg-background shadow-sm text-foreground"
                                : "text-muted-foreground hover:text-foreground"
                            }`}
                          >
                            {m === "off" ? "No acquisition" : m === "ei" ? "EI" : "UCB"}
                          </button>
                        ))}
                      </div>
                      {acqMode === "ucb" ? (
                        <label className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                          <span className="font-mono">κ</span>
                          <input
                            type="range"
                            min={0.25}
                            max={3}
                            step={0.05}
                            value={ucbKappa}
                            onChange={(e) => setUcbKappa(Number(e.target.value))}
                            className="h-1.5 w-24 accent-primary"
                            aria-label="UCB exploration parameter kappa"
                          />
                          <span className="w-8 font-mono tabular-nums">{ucbKappa.toFixed(2)}</span>
                        </label>
                      ) : null}
                    </div>
                  ) : null}
                </div>
                {acqCurve ? (
                  <p className="mt-1.5 text-[11px] leading-snug text-muted-foreground">
                    {acqMode === "ei" ? (
                      <>
                        Expected Improvement over the best observed training y (f* = {fmtNum(bestObserved ?? 0, 3)}) —
                        the amber curve on the right axis peaks where the GP expects the biggest gain{" "}
                        <span className="font-medium text-foreground">
                          next experiment: x ≈ {fmtNum(acqCurve.bestX, 3)}
                        </span>
                      </>
                    ) : (
                      <>
                        Upper Confidence Bound μ + κ·σ (κ = {ucbKappa.toFixed(2)}) — optimism under uncertainty; the amber
                        curve trades exploitation (high μ) against exploration (wide σ){" "}
                        <span className="font-medium text-foreground">
                          next experiment: x ≈ {fmtNum(acqCurve.bestX, 3)}
                        </span>
                      </>
                    )}
                  </p>
                ) : null}
              </CardHeader>
              <CardContent className="p-3 sm:p-4">
                <div
                  className="h-[340px] w-full"
                  role="img"
                  aria-label={
                    result.task === "regression"
                      ? bandCurve
                        ? acqCurve
                          ? "Scatter plot of training and test points with the Gaussian process posterior mean, its shaded 95 percent predictive band, and the expected improvement or upper confidence bound acquisition curve on a secondary axis, with the next candidate experiment marked"
                          : "Scatter plot of test points with the Gaussian process posterior mean and its shaded 95 percent predictive band"
                        : predActual
                          ? "Scatter plot of model predictions versus actual values on the test set, with the perfect-prediction reference line"
                          : "Scatter plot of test points with the fitted regression curve"
                      : result.datasetMeta.source
                        ? "Scatter plot of the uploaded-data test classes with the model decision boundary"
                        : "Scatter plot of the two moon classes with model decision boundary"
                  }
                >
                  <ResponsiveContainer width="100%" height="100%">
                    {result.task === "regression" ? (
                      bandCurve ? (
                        /* GP: filled ±1.96σ band (stacked areas — the
                         * invisible bandBase + the tinted bandSpan) under
                         * the posterior-mean line, train + test points on
                         * top, and (when switched on) the acquisition
                         * curve on its own right axis with the argmax
                         * marked as the next candidate experiment. */
                        <ComposedChart
                          data={acqCurve?.rows ?? bandCurve}
                          margin={{ top: 8, right: acqCurve ? 46 : 14, bottom: 4, left: 0 }}
                        >
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
                              value: result.datasetMeta.source?.featureColumns[0] ?? "x",
                              position: "insideBottomRight",
                              offset: -2,
                              fill: "#78716c",
                              fontSize: 11,
                            }}
                          />
                          <YAxis
                            type="number"
                            width={44}
                            tick={CHART.axisTick}
                            tickFormatter={fmtTick}
                            tickLine={false}
                            axisLine={false}
                          />
                          {acqCurve ? (
                            /* The acquisition rides on its own right axis
                             * (same x grid, independent 0→max domain) so
                             * EI/UCB units never distort the y geometry. */
                            <YAxis
                              yAxisId="acq"
                              type="number"
                              orientation="right"
                              width={38}
                              domain={[0, "auto"]}
                              tick={CHART.axisTick}
                              tickFormatter={(v: number) => fmtNum(v, 2)}
                              tickLine={false}
                              axisLine={false}
                            />
                          ) : null}
                          <Tooltip cursor={{ strokeDasharray: "3 3" }} content={<GpBandTooltip />} />
                          <Legend wrapperStyle={{ fontSize: 12, color: "#78716c" }} />
                          <Area
                            dataKey="bandBase"
                            stackId="gpBand"
                            stroke="none"
                            fill="transparent"
                            legendType="none"
                            isAnimationActive={false}
                          />
                          <Area
                            dataKey="bandSpan"
                            stackId="gpBand"
                            stroke="none"
                            fill={CHART.emerald}
                            fillOpacity={0.16}
                            name="95% band (ŷ ± 1.96σ)"
                            isAnimationActive={false}
                          />
                          <Line
                            dataKey="yhat"
                            stroke={CHART.emerald}
                            strokeWidth={2.5}
                            dot={false}
                            name="GP mean (ŷ)"
                            isAnimationActive={false}
                          />
                          {result.series.train?.length ? (
                            /* Where the GP actually observed data — the
                             * band pinches around them and widens away. */
                            <Scatter
                              data={result.series.train}
                              fill={CHART.stone}
                              fillOpacity={0.35}
                              name="train points"
                              isAnimationActive={false}
                            />
                          ) : null}
                          <Scatter
                            data={result.series.test}
                            fill={CHART.stone}
                            fillOpacity={0.8}
                            name="test points"
                            isAnimationActive={false}
                          />
                          {acqCurve ? (
                            <Line
                              yAxisId="acq"
                              dataKey="acq"
                              stroke={CHART.amber}
                              strokeWidth={1.8}
                              strokeDasharray="5 3"
                              dot={false}
                              name={acqMode === "ei" ? "EI (right axis)" : "UCB (right axis)"}
                              isAnimationActive={false}
                            />
                          ) : null}
                          {acqCurve ? (
                            /* The next candidate: a dashed vertical at the
                             * acquisition argmax + an amber diamond at the
                             * GP mean's y there. */
                            <ReferenceLine
                              yAxisId="acq"
                              x={acqCurve.bestX}
                              stroke={CHART.amber}
                              strokeDasharray="4 3"
                              strokeOpacity={0.7}
                              label={{
                                value: "next x",
                                position: "insideTopRight",
                                fill: CHART.amber,
                                fontSize: 10,
                              }}
                            />
                          ) : null}
                        </ComposedChart>
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
                          >
                            {/* GP multivariate: ŷ ± 1.96σ whiskers on the
                             * prediction axis of the parity chart. */}
                            {hasTestErr ? (
                              <ErrorBar
                                dataKey="err"
                                direction="x"
                                width={4}
                                strokeWidth={1.2}
                                stroke={CHART.emerald}
                                strokeOpacity={0.65}
                              />
                            ) : null}
                          </Scatter>
                        </ScatterChart>
                      )
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
                            name={probaFade ? "boundary: class 0 (opacity = |p − 0.5|)" : "boundary: class 0"}
                            data={clsSeries.grid0}
                            fill={CHART.teal}
                            shape={probaFade ? GpClsDot(CHART.teal) : TinyDot(CHART.teal)}
                            isAnimationActive={false}
                          />
                        ) : null}
                        {clsSeries ? (
                          <Scatter
                            name={probaFade ? "boundary: class 1 (opacity = |p − 0.5|)" : "boundary: class 1"}
                            data={clsSeries.grid1}
                            fill={CHART.amber}
                            shape={probaFade ? GpClsDot(CHART.amber) : TinyDot(CHART.amber)}
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
                        This model reports no hyperparameters — {modelNote(result.task, result.model) ?? "see the model hint."}
                      </p>
                    )
                  ) : null}

                  {/* PLS: VIP scores — the classic QSAR descriptor-influence
                   *  readout (VIP > 1 marks an influential descriptor; the
                   *  dashed line is the threshold). */}
                  {result.vip ? (
                    <div className="mt-3 space-y-1.5">
                      <p className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                        <Sigma className="size-3" aria-hidden />
                        VIP scores (threshold 1)
                      </p>
                      <ul className="space-y-2" aria-label="PLS VIP scores">
                        {result.vip.features.map((f, i) => {
                          const v = result.vip!.values[i] ?? 0;
                          const scale = Math.max(...result.vip!.values, result.vip!.threshold, 1e-9);
                          const width = Math.max((v / scale) * 100, 0.5);
                          const thresholdLeft = Math.min((result.vip!.threshold / scale) * 100, 99.5);
                          const influential = v >= result.vip!.threshold;
                          return (
                            <li key={f} className="space-y-1">
                              <div className="flex items-baseline justify-between gap-2">
                                <span className="truncate font-mono text-xs font-medium" title={f}>
                                  {f}
                                </span>
                                <span
                                  className={`font-mono text-xs tabular-nums ${
                                    influential ? "font-semibold text-primary" : "text-muted-foreground"
                                  }`}
                                >
                                  {fmtNum(v, 4)}
                                  {influential ? " · influential" : ""}
                                </span>
                              </div>
                              <div
                                className="relative h-2 overflow-hidden rounded-full bg-muted"
                                role="presentation"
                              >
                                <div
                                  className={`h-full rounded-full ${
                                    influential ? "bg-primary/75" : "bg-muted-foreground/35"
                                  }`}
                                  style={{ width: `${width}%` }}
                                />
                                <div
                                  className="absolute inset-y-0 border-l border-dashed border-primary/60"
                                  style={{ left: `${thresholdLeft}%` }}
                                  aria-hidden
                                />
                              </div>
                            </li>
                          );
                        })}
                      </ul>
                      <p className="text-[11px] leading-snug text-muted-foreground">
                        VIP &gt; 1 marks an influential descriptor (scores mean-square to 1 — the dashed
                        line); the classic PLS readout of which columns actually drive the model.
                      </p>
                    </div>
                  ) : null}

                  {/* GP: kernel readout + the honest held-out uncertainty stats. */}
                  {result.gp ? (
                    <div className="mt-3 space-y-1.5">
                      <p className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                        <Sigma className="size-3" aria-hidden />
                        GP kernel &amp; predictive uncertainty
                      </p>
                      <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1" aria-label="GP kernel readout">
                        {(
                          [
                            [result.gp.kernelTuned ? "kernel (learned by LML)" : "kernel (pinned)", null],
                            ["log marginal likelihood", result.gp.logMarginalLikelihood],
                            ["support rows", result.gp.nSupport],
                            ["95% band coverage (test)", result.gp.coverage95],
                            ["median σ (test)", result.gp.medianSigma],
                          ] as Array<[string, number | null]>
                        ).map(([label, value]) =>
                          value === null ? (
                            <div key={label} className="col-span-2">
                              <dt className="text-[11px] font-medium text-muted-foreground">
                                {label} — ℓ {fmtNum(result.gp!.lengthScale, 3)} · σf{" "}
                                {fmtNum(result.gp!.signalStd, 3)} · σn {fmtNum(result.gp!.noiseStd, 3)}
                              </dt>
                            </div>
                          ) : (
                            <div key={label} className="col-span-2 grid grid-cols-subgrid items-baseline">
                              <dt className="truncate font-mono text-xs text-muted-foreground" title={label}>
                                {label}
                              </dt>
                              <dd
                                className={`text-right font-mono text-xs font-semibold tabular-nums ${
                                  label.startsWith("95%")
                                    ? result.gp!.coverage95 >= 0.85
                                      ? "text-primary"
                                      : result.gp!.coverage95 >= 0.7
                                        ? "text-amber-600"
                                        : "text-destructive"
                                    : ""
                                }`}
                              >
                                {label.startsWith("95%")
                                  ? `${fmtNum(result.gp!.coverage95 * 100, 1)}%`
                                  : fmtNum(value, 4)}
                              </dd>
                            </div>
                          )
                        )}
                      </dl>
                      <p className="text-[11px] leading-snug text-muted-foreground">
                        {result.gp.kernelTuned
                          ? "Kernel picked by the deterministic marginal-likelihood grid search (ℓ / σf / σn), refit on every train row; the LML says how well the kernel explains the training targets."
                          : "Kernel pinned at the request's values — sweep the sliders and watch the LML respond."}{" "}
                        A calibrated GP straddles ≈95% of held-out truth with its ±1.96σ band; σ grows away
                        from the training data (the extrapolation warning no point estimator gives).
                      </p>
                    </div>
                  ) : null}

                  {/* GP classification: the Laplace posterior + honest
                   *  probabilistic scores (AUC / log-loss / Brier). */}
                  {result.gpCls ? (
                    <div className="mt-3 space-y-1.5">
                      <p className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                        <Sigma className="size-3" aria-hidden />
                        GP classifier · Laplace posterior
                      </p>
                      <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1" aria-label="GP classifier readout">
                        {(
                          [
                            [result.gpCls.kernelTuned ? "kernel (learned by Laplace lml)" : "kernel (pinned)", null],
                            ["Laplace log marginal likelihood", result.gpCls.logMarginalLikelihood],
                            ["Newton iterations to mode", result.gpCls.newtonIterations],
                            ["ROC AUC (test)", result.gpCls.rocAuc],
                            ["log-loss (test)", result.gpCls.logLoss],
                            ["Brier score (test)", result.gpCls.brier],
                          ] as Array<[string, number | null]>
                        ).map(([label, value]) =>
                          value === null ? (
                            <div key={label} className="col-span-2">
                              <dt className="text-[11px] font-medium text-muted-foreground">
                                {label} — ℓ {fmtNum(result.gpCls!.lengthScale, 3)} · σf{" "}
                                {fmtNum(result.gpCls!.signalStd, 3)}
                              </dt>
                            </div>
                          ) : (
                            <div key={label} className="col-span-2 grid grid-cols-subgrid items-baseline">
                              <dt className="truncate font-mono text-xs text-muted-foreground" title={label}>
                                {label}
                              </dt>
                              <dd
                                className={`text-right font-mono text-xs font-semibold tabular-nums ${
                                  label.startsWith("ROC")
                                    ? result.gpCls!.rocAuc >= 0.9
                                      ? "text-primary"
                                      : result.gpCls!.rocAuc >= 0.75
                                        ? "text-amber-600"
                                        : "text-destructive"
                                    : ""
                                }`}
                              >
                                {label.startsWith("ROC")
                                  ? fmtNum(result.gpCls!.rocAuc, 4)
                                  : label.startsWith("Newton")
                                    ? String(result.gpCls!.newtonIterations)
                                    : fmtNum(value, 4)}
                              </dd>
                            </div>
                          )
                        )}
                      </dl>
                      <p className="text-[11px] leading-snug text-muted-foreground">
                        {result.gpCls.kernelTuned
                          ? "Kernel picked by the deterministic Laplace-lml grid search (ℓ / σf), posterior refit on every train row — the lml scores how well the latent GP explains the labels."
                          : "Kernel pinned at the sliders — sweep ℓ / σf and watch the Laplace lml respond."}{" "}
                        ROC AUC ranks the held-out probabilities (1 = perfect ordering); log-loss and Brier
                        grade their calibration — boundary-dot opacity draws the same |p − 0.5| uncertainty
                        surface, fading exactly where the classifier hesitates.
                      </p>
                    </div>
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

                  {/* Meta-ensembles: the member blend + the honest
                   *  "did the ensemble actually help?" table (every
                   *  level-0 member scored on the same held-out rows as
                   *  the blend), plus the OOF/meta-learner readouts. */}
                  {result.ensemble ? (
                    <div className="mt-3 space-y-2.5">
                      <p className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                        <Layers className="size-3" aria-hidden />
                        {result.ensemble.kind === "voting" ? "Voting ensemble" : "Stacking ensemble"} ·{" "}
                        {result.ensemble.rule === "soft" ? "soft vote (mean probability)" : "mean of predictions"}
                      </p>
                      <div className="overflow-hidden rounded-lg border">
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead className="h-8">
                                {result.task === "classification" ? "Member (held-out accuracy)" : "Member (held-out R²)"}
                              </TableHead>
                              <TableHead className="h-8 text-right">Score</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {result.ensemble.memberScores.map((m) => {
                              const isBlend = m.name === "ensemble";
                              const scores = result.ensemble!.memberScores.map((s) => s.score);
                              const best = Math.max(...scores);
                              const scale = Math.max(best, 1e-9);
                              const width = Math.max((Math.max(m.score, 0) / scale) * 100, 0.5);
                              return (
                                <TableRow key={m.name} className={isBlend ? "bg-primary/5 hover:bg-primary/5" : undefined}>
                                  <TableCell className="py-1.5">
                                    <div className="flex items-center gap-2">
                                      <span className={`font-mono text-xs ${isBlend ? "font-semibold text-primary" : ""}`}>
                                        {isBlend
                                          ? (specOf(result.task, result.model)?.label ?? result.model)
                                          : (specOf(result.task, m.name)?.label ?? m.name)}
                                      </span>
                                      {isBlend ? (
                                        <span className="rounded-sm border bg-primary/10 px-1 text-[9px] font-medium uppercase tracking-wide text-primary">
                                          this model
                                        </span>
                                      ) : null}
                                      {m.score >= best - 1e-12 ? (
                                        <span className="rounded-sm border bg-muted/60 px-1 text-[9px] uppercase text-muted-foreground" title="Best score in the table">
                                          best
                                        </span>
                                      ) : null}
                                      <span className="hidden h-1.5 flex-1 overflow-hidden rounded-full bg-muted sm:block" role="presentation">
                                        <span
                                          className={`h-full rounded-full ${isBlend ? "bg-primary" : "bg-primary/50"}`}
                                          style={{ width: `${width}%` }}
                                        />
                                      </span>
                                    </div>
                                  </TableCell>
                                  <TableCell
                                    className={`py-1.5 text-right font-mono text-xs tabular-nums ${
                                      isBlend ? "font-semibold text-primary" : ""
                                    }`}
                                  >
                                    {fmtNum(m.score, 4)}
                                  </TableCell>
                                </TableRow>
                              );
                            })}
                          </TableBody>
                        </Table>
                      </div>

                      {/* Stacking: the level-0 OOF scores (the exact matrix
                       *  the meta-learner trained on) + the blender's
                       *  weights — which member it actually trusts. */}
                      {result.ensemble.oof && result.ensemble.oof.length > 0 ? (
                        <div className="space-y-1.5 rounded-lg border bg-muted/20 p-2.5">
                          <p className="text-[11px] font-medium text-muted-foreground">
                            {result.ensemble.oofFolds}-fold out-of-fold ·{" "}
                            {result.task === "classification" ? "accuracy" : "R²"} per member
                            {result.ensemble.metaModel ? (
                              <>
                                {" "}· meta-learner: <span className="font-mono">{result.ensemble.metaModel === "logistic" ? "logistic regression" : "ridge"}</span>
                              </>
                            ) : null}
                          </p>
                          <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1" aria-label="Out-of-fold member scores">
                            {result.ensemble.oof.map((o) => (
                              <div key={o.name} className="col-span-2 grid grid-cols-subgrid items-baseline">
                                <dt className="truncate font-mono text-xs text-muted-foreground">
                                  {specOf(result.task, o.name)?.label ?? o.name}
                                </dt>
                                <dd className="text-right font-mono text-xs tabular-nums">
                                  {fmtNum(o.score, 4)}
                                </dd>
                              </div>
                            ))}
                            {result.ensemble.metaCoefs ? (
                              <div className="col-span-2 mt-1 border-t pt-1.5" />
                            ) : null}
                            {result.ensemble.metaCoefs
                              ? result.ensemble.metaCoefs.coefs.map((coef, i) => (
                                  <div key={result.ensemble!.members[i] ?? i} className="col-span-2 grid grid-cols-subgrid items-baseline">
                                    <dt className="truncate font-mono text-xs text-muted-foreground">
                                      meta weight · {specOf(result.task, result.ensemble!.members[i] ?? "")?.label ?? result.ensemble!.members[i]}
                                    </dt>
                                    <dd
                                      className={`text-right font-mono text-xs font-semibold tabular-nums ${
                                        coef >= 0 ? "" : "text-destructive"
                                      }`}
                                    >
                                      {fmtNum(coef, 4)}
                                    </dd>
                                  </div>
                                ))
                              : null}
                            {result.ensemble.metaCoefs ? (
                              <div className="col-span-2 grid grid-cols-subgrid items-baseline">
                                <dt className="truncate font-mono text-xs text-muted-foreground">meta weight · intercept</dt>
                                <dd className="text-right font-mono text-xs tabular-nums">
                                  {fmtNum(result.ensemble.metaCoefs.intercept, 4)}
                                </dd>
                              </div>
                            ) : null}
                          </dl>
                          <p className="text-[11px] leading-snug text-muted-foreground">
                            OOF scores are cross-validated on the training rows — the honest quality of each base model
                            exactly as the blender saw it; meta weights live in standardized OOF space, so their sign and
                            relative size (not raw magnitude) say whom the blend leans on.
                          </p>
                        </div>
                      ) : null}

                      {/* Soft-rule classification ensembles: the honest
                       *  probabilistic scores from the blended
                       *  probabilities (the gpCls card's readout). */}
                      {result.ensemble.rocAuc !== undefined ? (
                        <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1" aria-label="Ensemble probabilistic scores">
                          {(
                            [
                              ["ROC AUC (test)", result.ensemble.rocAuc],
                              ["log-loss (test)", result.ensemble.logLoss],
                              ["Brier score (test)", result.ensemble.brier],
                            ] as Array<[string, number | undefined]>
                          ).map(([label, value]) => (
                            <div key={label} className="col-span-2 grid grid-cols-subgrid items-baseline">
                              <dt className="truncate font-mono text-xs text-muted-foreground" title={label}>
                                {label}
                              </dt>
                              <dd
                                className={`text-right font-mono text-xs font-semibold tabular-nums ${
                                  label.startsWith("ROC")
                                    ? (result.ensemble!.rocAuc ?? 0) >= 0.9
                                      ? "text-primary"
                                      : (result.ensemble!.rocAuc ?? 0) >= 0.75
                                        ? "text-amber-600"
                                        : "text-destructive"
                                    : ""
                                }`}
                              >
                                {value === undefined ? "—" : fmtNum(value, 4)}
                              </dd>
                            </div>
                          ))}
                        </dl>
                      ) : null}
                    </div>
                  ) : null}

                  <p className="mt-2 text-[11px] leading-snug text-muted-foreground">
                    {modelNote(result.task, result.model) ??
                      (result.task === "regression"
                        ? predActual
                          ? "Multivariate least squares: k features standardized on train statistics, solved in closed form via normal equations; ridge adds α to the diagonal, skipping the intercept."
                          : "Solved in closed form via normal equations (Gaussian elimination with partial pivoting); ridge adds α to the diagonal, skipping the intercept."
                        : "Batch gradient descent: lr = 0.1, 800 epochs, L2-penalized weights (bias exempt).")}
                  </p>
                </CardContent>
              </Card>

              {/* Feature–target correlations (100x-153): the desktop
               *  interpretability bundle's web twin — signed bars over
               *  the run's OWN held-out rows + the top-N matrix. */}
              {result.correlations ? <CorrelationCard bundle={result.correlations} /> : null}
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

        {/* --------------------- Learning curve (100x-125) --------------------- */}
        {result && status !== "running" ? (
          <Card className="overflow-hidden py-0 shadow-sm">
            <CardHeader className="border-b bg-muted/30 py-3">
              <CardTitle className="flex items-center gap-1.5 text-sm">
                <TrendingUp className="size-3.5 text-primary" aria-hidden />
                Learning curve — is more data still buying accuracy?
              </CardTitle>
              <CardDescription className="text-xs">
                The last trained configuration refit at growing train sizes (the web twin of
                the desktop card&rsquo;s plot_learning_curve figure, 10 % … 100 % in 8 steps,
                k-fold CV at every size). Converging plateau → more rows will not help; CV
                curve still climbing at the right edge → assay more compounds; a persistent
                gap → overfitting, not a data problem.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 pb-4">
              {/* Knobs + run */}
              <div className="flex flex-wrap items-end gap-3">
                <div className="w-28">
                  <Label className="text-[11px] text-muted-foreground">CV folds</Label>
                  <Select value={lcFolds} onValueChange={setLcFolds}>
                    <SelectTrigger
                      className="mt-1 h-8 text-xs"
                      aria-label="cross-validation folds for the learning curve"
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="3">3-fold</SelectItem>
                      <SelectItem value="5">5-fold</SelectItem>
                      <SelectItem value="10">10-fold</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <Button
                  size="sm"
                  onClick={runLearningCurve}
                  disabled={lcStatus === "running"}
                  className="h-8"
                >
                  {lcStatus === "running" ? (
                    <>
                      <Loader2 className="size-3.5 animate-spin" aria-hidden /> Computing…
                    </>
                  ) : (
                    <>
                      <Play className="size-3.5" aria-hidden /> Plot learning curve
                    </>
                  )}
                </Button>
                <p className="text-[11px] text-muted-foreground">
                  re-runs {modelSpec.label} on the last trained {result.datasetMeta.nTotal} rows
                  {result.datasetMeta.source ? ` (${result.datasetMeta.source.fileName})` : ""} —
                  the curve replaces nothing, the parity chart stays.
                </p>
              </div>

              {lcError ? (
                <Alert variant="destructive">
                  <TriangleAlert className="size-4" aria-hidden />
                  <AlertTitle>Learning curve failed</AlertTitle>
                  <AlertDescription>{lcError}</AlertDescription>
                </Alert>
              ) : null}

              {lcResult ? (
                <>
                  {/* Verdict strip */}
                  <div className="grid gap-2 sm:grid-cols-3">
                    <div className="rounded-lg border bg-card px-3 py-2.5 shadow-sm">
                      <p className="text-xs text-muted-foreground">Train/CV gap (full size)</p>
                      <p
                        className={`font-mono text-xl font-semibold tabular-nums ${
                          lcResult.gap > 0.15
                            ? "text-rose-600 dark:text-rose-400"
                            : "text-emerald-600 dark:text-emerald-400"
                        }`}
                      >
                        {fmtNum(lcResult.gap, 3)}
                      </p>
                    </div>
                    <div className="rounded-lg border bg-card px-3 py-2.5 shadow-sm sm:col-span-2">
                      <p className="text-xs text-muted-foreground">Verdict</p>
                      <p className="text-sm font-medium leading-snug">{lcResult.verdict}</p>
                    </div>
                  </div>

                  {lcResult.gpKernel ? (
                    <p className="rounded-md border border-dashed bg-muted/20 px-3 py-2 text-[11px] leading-snug text-muted-foreground">
                      GP kernel cloned from the run&rsquo;s learned values (ℓ ={" "}
                      {fmtNum(lcResult.gpKernel.lengthScale, 3)}, σf ={" "}
                      {fmtNum(lcResult.gpKernel.signalStd, 3)}, σn ={" "}
                      {fmtNum(lcResult.gpKernel.noiseStd, 3)}) — resolved once on the full data,
                      then fixed for every refit.
                    </p>
                  ) : null}
                  {lcResult.gpClsKernel ? (
                    <p className="rounded-md border border-dashed bg-muted/20 px-3 py-2 text-[11px] leading-snug text-muted-foreground">
                      GP kernel cloned from the run&rsquo;s learned values (ℓ ={" "}
                      {fmtNum(lcResult.gpClsKernel.lengthScale, 3)}, σf ={" "}
                      {fmtNum(lcResult.gpClsKernel.signalStd, 3)}) — resolved once on the full
                      data, then fixed for every refit.
                    </p>
                  ) : null}

                  {/* Both curves with ±1σ fold-spread bands */}
                  <div>
                    <p className="mb-1.5 text-xs font-medium text-muted-foreground">
                      {lcResult.metricName} vs training samples — {lcResult.folds}-fold CV, mean ±
                      1 std across folds
                    </p>
                    <div className="h-56">
                      <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart
                          data={lcRows(lcResult)}
                          margin={{ top: 6, right: 12, left: -14, bottom: 0 }}
                        >
                          <CartesianGrid
                            strokeDasharray="3 3"
                            stroke={CHART.gridStroke}
                            strokeOpacity={CHART.gridStrokeOpacity}
                          />
                          <XAxis
                            dataKey="size"
                            type="number"
                            domain={["dataMin", "dataMax"]}
                            tick={CHART.axisTick}
                            tickLine={false}
                            axisLine={false}
                            label={{
                              value: "training samples",
                              position: "insideBottomRight",
                              offset: -2,
                              fontSize: 10,
                              fill: "#78716c",
                            }}
                          />
                          <YAxis
                            tick={CHART.axisTick}
                            tickLine={false}
                            axisLine={false}
                            tickFormatter={(v: number) => fmtNum(v, 2)}
                          />
                          <Tooltip
                            cursor={{ strokeDasharray: "3 3" }}
                            content={
                              <LcTooltip metricName={lcResult.metricName} />
                            }
                          />
                          <Legend wrapperStyle={{ fontSize: 12, color: "#78716c" }} />
                          {/* Training band + line */}
                          <Area
                            dataKey="trainBase"
                            stackId="lcTrain"
                            stroke="none"
                            fill="transparent"
                            legendType="none"
                            isAnimationActive={false}
                          />
                          <Area
                            dataKey="trainSpan"
                            stackId="lcTrain"
                            stroke="none"
                            fill={CHART.teal}
                            fillOpacity={0.15}
                            name="training ± 1 std"
                            legendType="none"
                            isAnimationActive={false}
                          />
                          <Line
                            dataKey="trainMean"
                            stroke={CHART.teal}
                            strokeWidth={2.2}
                            dot={{ r: 2.5, fill: CHART.teal, strokeWidth: 0 }}
                            name={`training ${lcResult.metricName} (${fmtNum(
                              lcResult.points[lcResult.points.length - 1]?.trainMean ?? 0,
                              3
                            )})`}
                            isAnimationActive={false}
                          />
                          {/* CV band + line */}
                          <Area
                            dataKey="valBase"
                            stackId="lcVal"
                            stroke="none"
                            fill="transparent"
                            legendType="none"
                            isAnimationActive={false}
                          />
                          <Area
                            dataKey="valSpan"
                            stackId="lcVal"
                            stroke="none"
                            fill={CHART.amber}
                            fillOpacity={0.15}
                            name="CV ± 1 std"
                            legendType="none"
                            isAnimationActive={false}
                          />
                          <Line
                            dataKey="valMean"
                            stroke={CHART.amber}
                            strokeWidth={2.2}
                            dot={{ r: 2.5, fill: CHART.amber, strokeWidth: 0 }}
                            name={`CV ${lcResult.metricName} (${fmtNum(
                              lcResult.points[lcResult.points.length - 1]?.valMean ?? 0,
                              3
                            )})`}
                            isAnimationActive={false}
                          />
                        </ComposedChart>
                      </ResponsiveContainer>
                    </div>
                    <p className="mt-2 text-[11px] leading-snug text-muted-foreground">
                      Read it like the desktop figure: both curves flattening together means the
                      next 500 assays buy almost nothing; the amber CV curve still rising at the
                      right edge means they are worth running; a teal band sitting far above the
                      amber one means the model memorises — regularise or simplify instead of
                      collecting more rows.
                    </p>
                  </div>
                </>
              ) : null}
            </CardContent>
          </Card>
        ) : null}

        {/* ------------------------- Applicability domain ------------------------- */}
        {result && status !== "running" ? (
          <Card className="overflow-hidden py-0 shadow-sm">
            <CardHeader className="border-b bg-muted/30 py-3">
              <CardTitle className="flex items-center gap-1.5 text-sm">
                <Crosshair className="size-3.5 text-primary" aria-hidden />
                Applicability domain &amp; prediction reliability
              </CardTitle>
              <CardDescription className="text-xs">
                The uploaded Applicability_domain_analysis.py workflow on this run&rsquo;s
                held-out rows: mean kNN distance vs the training space&rsquo;s p
                {adResult ? adResult.summary.percentile : 95} threshold (inside the domain?),
                bootstrap-ensemble spread vs its own training-side threshold (how sure?),
                one reliable verdict per row.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 pb-4">
              {/* Knobs + run */}
              <div className="flex flex-wrap items-end gap-3">
                <div className="w-24">
                  <Label className="text-[11px] text-muted-foreground">k neighbours</Label>
                  <Input
                    inputMode="numeric"
                    value={adK}
                    onChange={(e) => setAdK(e.target.value)}
                    className="mt-1 h-8 font-mono text-xs"
                    aria-label="k nearest neighbours of the distance domain"
                  />
                </div>
                <div className="w-28">
                  <Label className="text-[11px] text-muted-foreground">percentile</Label>
                  <Input
                    inputMode="decimal"
                    value={adPct}
                    onChange={(e) => setAdPct(e.target.value)}
                    className="mt-1 h-8 font-mono text-xs"
                    aria-label="percentile both thresholds are cut at"
                  />
                </div>
                <div className="w-32">
                  <Label className="text-[11px] text-muted-foreground">bootstrap models</Label>
                  <Input
                    inputMode="numeric"
                    value={adModels}
                    onChange={(e) => setAdModels(e.target.value)}
                    className="mt-1 h-8 font-mono text-xs"
                    aria-label="bootstrap ensemble size (0 = kNN-only quick pass)"
                  />
                </div>
                <Button
                  size="sm"
                  onClick={runAd}
                  disabled={adStatus === "running" || !lastTrainBodyRef.current?.csv}
                  className="h-8"
                >
                  {adStatus === "running" ? (
                    <>
                      <Loader2 className="size-3.5 animate-spin" aria-hidden /> Analysing…
                    </>
                  ) : (
                    <>
                      <Play className="size-3.5" aria-hidden /> Analyse reliability
                    </>
                  )}
                </Button>
                {!lastTrainBodyRef.current?.csv ? (
                  <p className="text-[11px] text-muted-foreground">
                    Train on a loaded CSV first — the analysis judges that run&rsquo;s held-out
                    rows.
                  </p>
                ) : null}
              </div>

              {adError ? (
                <Alert variant="destructive">
                  <TriangleAlert className="size-4" aria-hidden />
                  <AlertTitle>Analysis failed</AlertTitle>
                  <AlertDescription>{adError}</AlertDescription>
                </Alert>
              ) : null}

              {adResult ? (
                <>
                  {/* Verdict strip */}
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    <div className="rounded-lg border bg-card px-3 py-2.5 shadow-sm">
                      <p className="text-xs text-muted-foreground">Distance threshold</p>
                      <p className="font-mono text-xl font-semibold tabular-nums">
                        {fmtNum(adResult.summary.distanceThreshold, 3)}
                      </p>
                    </div>
                    <div className="rounded-lg border bg-card px-3 py-2.5 shadow-sm">
                      <p className="text-xs text-muted-foreground">Inside the domain</p>
                      <p className="font-mono text-xl font-semibold tabular-nums">
                        {adResult.summary.insideCount}/{adResult.summary.nQuery}
                      </p>
                    </div>
                    <div className="rounded-lg border bg-card px-3 py-2.5 shadow-sm">
                      <p className="text-xs text-muted-foreground">
                        {adResult.summary.uncertaintyThreshold === null
                          ? "Uncertainty"
                          : "σ threshold"}
                      </p>
                      <p className="font-mono text-xl font-semibold tabular-nums">
                        {adResult.summary.uncertaintyThreshold === null
                          ? "—"
                          : fmtNum(adResult.summary.uncertaintyThreshold, 3)}
                      </p>
                    </div>
                    <div className="rounded-lg border bg-card px-3 py-2.5 shadow-sm">
                      <p className="text-xs text-muted-foreground">Reliable verdicts</p>
                      <p
                        className={`font-mono text-xl font-semibold tabular-nums ${
                          adResult.summary.reliableCount === null
                            ? ""
                            : adResult.summary.reliableCount === adResult.summary.nQuery
                              ? "text-emerald-600 dark:text-emerald-400"
                              : "text-amber-600 dark:text-amber-400"
                        }`}
                      >
                        {adResult.summary.reliableCount === null
                          ? "—"
                          : `${adResult.summary.reliableCount}/${adResult.summary.nQuery}`}
                      </p>
                    </div>
                  </div>

                  {adResult.bootstrapNote ? (
                    <p className="rounded-md border border-dashed bg-muted/20 px-3 py-2 text-[11px] leading-snug text-muted-foreground">
                      {adResult.bootstrapNote}
                    </p>
                  ) : null}

                  {/* The upload's combined figure: (a) distance histogram + (b) verdict scatter */}
                  <div className="grid gap-4 lg:grid-cols-2">
                    <div>
                      <p className="mb-1.5 text-xs font-medium text-muted-foreground">
                        (a) Training distance density — {adResult.nTrain} rows, k ={" "}
                        {adResult.summary.k}
                      </p>
                      <div className="h-44">
                        <ResponsiveContainer width="100%" height="100%">
                          <ComposedChart
                            data={adHistogram(adResult)}
                            margin={{ top: 4, right: 8, left: -18, bottom: 0 }}
                          >
                            <CartesianGrid
                              strokeDasharray="3 3"
                              stroke={CHART.gridStroke}
                              strokeOpacity={CHART.gridStrokeOpacity}
                            />
                            <XAxis
                              dataKey="x"
                              type="number"
                              domain={["dataMin", "dataMax"]}
                              tickFormatter={(v: number) => fmtTick(v)}
                              fontSize={10}
                              stroke={CHART.axisTick.fill}
                            />
                            <YAxis fontSize={10} stroke={CHART.axisTick.fill} />
                            <Tooltip
                              contentStyle={{ fontSize: 11 }}
                              formatter={(v: number) => [v, "rows"]}
                              labelFormatter={(l: number) => `distance ≈ ${fmtNum(l, 3)}`}
                            />
                            <Bar dataKey="count" fill={CHART.teal} fillOpacity={0.75} />
                            <ReferenceLine
                              x={adResult.summary.distanceThreshold}
                              stroke={CHART.rose}
                              strokeDasharray="5 4"
                              ifOverflow="extendDomain"
                            />
                          </ComposedChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                    <div>
                      <p className="mb-1.5 text-xs font-medium text-muted-foreground">
                        (b) Distance vs prediction — {adResult.summary.insideCount}/
                        {adResult.summary.nQuery} inside
                      </p>
                      <div className="h-44">
                        <ResponsiveContainer width="100%" height="100%">
                          <ScatterChart
                            margin={{ top: 4, right: 8, left: -18, bottom: 0 }}
                          >
                            <CartesianGrid
                              strokeDasharray="3 3"
                              stroke={CHART.gridStroke}
                              strokeOpacity={CHART.gridStrokeOpacity}
                            />
                            <XAxis
                              dataKey="knn"
                              type="number"
                              domain={["dataMin", "dataMax"]}
                              tickFormatter={(v: number) => fmtTick(v)}
                              fontSize={10}
                              stroke={CHART.axisTick.fill}
                            />
                            <YAxis
                              dataKey="predicted"
                              type="number"
                              domain={["auto", "auto"]}
                              tickFormatter={(v: number) => fmtTick(v)}
                              fontSize={10}
                              stroke={CHART.axisTick.fill}
                            />
                            <Tooltip
                              contentStyle={{ fontSize: 11 }}
                              formatter={(v: number, n: string) => [fmtNum(v, 3), n]}
                              labelFormatter={() => ""}
                            />
                            <ReferenceLine
                              x={adResult.summary.distanceThreshold}
                              stroke={CHART.amber}
                              strokeDasharray="5 4"
                              ifOverflow="extendDomain"
                            />
                            <Scatter
                              name="inside domain"
                              data={adScatter(adResult, true)}
                              fill={CHART.emerald}
                              fillOpacity={0.8}
                            />
                            <Scatter
                              name="outside domain"
                              data={adScatter(adResult, false)}
                              fill={CHART.rose}
                              shape="diamond"
                              fillOpacity={0.9}
                            />
                          </ScatterChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>

                  {/* The reliability table (the upload's results CSV) */}
                  <div className="thin-scrollbar max-h-72 overflow-y-auto">
                    <Table className="data-table">
                      <TableHeader className="sticky top-0 bg-muted/60 backdrop-blur">
                        <TableRow className="hover:bg-transparent">
                          <TableHead className="h-9">Row</TableHead>
                          <TableHead className="h-9 text-right">Predicted</TableHead>
                          <TableHead className="h-9 text-right">± sd</TableHead>
                          <TableHead className="h-9 text-right">kNN distance</TableHead>
                          <TableHead className="h-9 text-center">Inside AD</TableHead>
                          <TableHead className="h-9 text-center">Low σ</TableHead>
                          <TableHead className="h-9 text-center">Reliable</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {adResult.rows.map((row, i) => (
                          <TableRow key={`${row.name}-${i}`}>
                            <TableCell className="py-1.5 font-mono text-xs">
                              {row.name}
                            </TableCell>
                            <TableCell className="py-1.5 text-right font-mono text-xs tabular-nums">
                              {row.predicted === null ? "—" : fmtNum(row.predicted, 3)}
                            </TableCell>
                            <TableCell className="py-1.5 text-right font-mono text-xs tabular-nums">
                              {row.predictionSd === null ? "—" : fmtNum(row.predictionSd, 3)}
                            </TableCell>
                            <TableCell className="py-1.5 text-right font-mono text-xs tabular-nums">
                              {fmtNum(row.knnDistance, 3)}
                            </TableCell>
                            <TableCell className="py-1.5 text-center">
                              <VerdictPill ok={row.insideAd} />
                            </TableCell>
                            <TableCell className="py-1.5 text-center">
                              <VerdictPill ok={row.lowUncertainty} nullable />
                            </TableCell>
                            <TableCell className="py-1.5 text-center">
                              <VerdictPill ok={row.reliable} nullable strong />
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                  <p className="text-[11px] leading-snug text-muted-foreground">
                    {adResult.summary.nModels === null
                      ? "kNN-only verdict — no bootstrap fits ran."
                      : `${adResult.summary.nModels} bootstrap members in ${adResult.summary.fitSeconds.toFixed(1)} s (each predicts on the query AND training rows — half the upload's 2 × n fits).`}{" "}
                    Feature space: {adResult.featureNames.length} descriptor(s) ·{" "}
                    {adResult.nTrain} training rows.
                  </p>
                </>
              ) : (
                <p className="rounded-md border border-dashed px-3 py-2 text-[11px] leading-snug text-muted-foreground">
                  The upload&rsquo;s two questions answered per held-out row: is the compound
                  inside the training descriptor space, and how much does the model&rsquo;s
                  answer wobble under bootstrap resampling? Train on a CSV, then run the
                  analysis.
                </p>
              )}
            </CardContent>
          </Card>
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
                    {fmtDelta(tuneResult.best.improvement)}
                  </p>
                </div>
              </div>

              {/* RandomSearch: the per-trial trajectory scatter — every drawn
                  configuration as a dot, the running-best step line, and the
                  defaults CV as the dashed reference bar. */}
              {tuneResult.searcher === "random" ? (
                <div className="space-y-1.5">
                  <p className="flex items-center gap-1.5 text-[11px] font-medium text-muted-foreground">
                    <TrendingUp className="size-3" aria-hidden />
                    Search trajectory — every trial, draw order
                  </p>
                  <RandomSearchTrajectory
                    trials={tuneResult.trials}
                    baselineScore={tuneResult.baseline.score}
                    objective={tuneResult.objective}
                  />
                </div>
              ) : null}

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

              {/* Hyperband: the per-trial rung scatter — every evaluation
                  as a budget-vs-score dot, coloured by rung; promoted
                  dots march right as survivors are re-scored, culled dots
                  stay faint (the halving picture the table can't show). */}
              {tuneResult.searcher === "hyperband" && tuneResult.hyperband ? (
                <div className="space-y-1.5">
                  <p className="flex items-center gap-1.5 text-[11px] font-medium text-muted-foreground">
                    <Layers className="size-3" aria-hidden />
                    Rung scatter — every evaluation, budget vs CV {tuneResult.objective}
                  </p>
                  <HyperbandRungScatter
                    trials={tuneResult.trials}
                    maxResource={tuneResult.hyperband.maxResource}
                    resourceLabel={
                      tuneResult.hyperband.resourceLabel ??
                      HP_LABELS[tuneResult.hyperband.resourceKey] ??
                      tuneResult.hyperband.resourceKey
                    }
                    baselineScore={tuneResult.baseline.score}
                    objective={tuneResult.objective}
                  />
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
