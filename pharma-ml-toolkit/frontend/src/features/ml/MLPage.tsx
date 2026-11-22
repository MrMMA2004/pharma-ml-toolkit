import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDownToLine,
  Activity,
  Braces,
  Brain,
  ClipboardCheck,
  Database,
  Dna,
  Filter,
  FlaskConical,
  GitCompare,
  Info,
  Layers,
  Network,
  Pill,
  Radar,
  Scale,
  SlidersHorizontal,
  Target,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useJob } from "@/hooks/useJobTracking";
import { jobsApi } from "@/services/jobs/client";
import { api, ApiError } from "@/services/api/client";
import {
  datasetsApi,
  type DatasetView,
} from "@/services/datasets/client";
import {
  artifactsApi,
  parseArtifactView,
  type ArtifactView,
} from "@/services/artifacts/client";
import {
  AD_MAX_FEATURES,
  ADMET_MAX_SMILES,
  DTA_MAX_PAIRS,
  EXPLAIN_MAX_ROWS,
  mlApi,
  parseADAnalyzeResult,
  parseSemiSupervisedResult,
  parseSemiSupervisedEvaluation,
  parseTrainResult,
  parseWilliamsResult,
  type ADAnalyzeResult,
  type ADFrameRow,
  type AdmetEndpointsResponse,
  type AdmetScoreResponse,
  type AdmetScoreRow,
  type DtaFeaturesResponse,
  type DtaPairCard,
  type DtaRegistryResponse,
  type DrugResponseAnalyzeResponse,
  type DrugResponseRegistryResponse,
  type ExplainResult,
  type EvaluationRow,
  type ModelCatalogEntry,
  type MultiobjectiveOptimizeResponse,
  type MultiobjectiveRegistryResponse,
  type MultiobjectiveTrialRow,
  type PredictResponse,
  type PharmacogenomicsAnalyzeResponse,
  type PharmacogenomicsRegistryResponse,
  type RepurposingRegistryResponse,
  type RepurposingScreenResponse,
  type SemiSupervisedEvaluation,
  type SemiSupervisedView,
  type TrainedModelView,
  type WilliamsFrameRow,
  type WilliamsResultPayload,
  type WorkflowRunRecordView,
  type WorkflowRunsResponse,
  type WorkflowStepsResponse,
  type WorkflowTemplateView,
  type WorkflowTemplatesResponse,
  type WorkflowValidateResponse,
  WORKFLOW_JOB_MAX_NODES,
  WORKFLOW_MAX_SEED,
  WORKFLOW_MAX_SYNTHETIC,
  WORKFLOW_MIN_SYNTHETIC,
  WORKFLOW_SYNC_MAX_NODES,
  parseWorkflowRunRecord,
} from "@/services/ml/client";
import type { components } from "@/types/api.d";
import {
  PlotlyFigure,
  type PlotTrace,
} from "@/components/charts/PlotlyFigure";

/**
 * The Machine Learning workspace — part11 §11.2's complete sequential
 * workflow:
 *
 * Select Dataset → Target & Features → Preprocess → Configure
 * Split/CV → Algorithm → Hyperparameters → Train → Evaluate →
 * Compare → Applicability Domain → ADMET Profile → Drug–Target
 * Interaction → Drug Repurposing → Cancer Cell-Line Drug Response →
 * Pharmacogenomics → Multi-objective Optimization → Workflow
 * Builder → Semi-supervised Learning → Explainability → Export
 *
 * Every control below executes the REAL backend through the typed
 * service layer (no fabricated data, no disabled pretend buttons):
 * the catalog/train/predict lanes ride routes/ml.py, the AD stage
 * rides routes/applicability_domain.py with x_test assembled through
 * the bounded column bridge, the semi-supervised lane rides
 * routes/semi_supervised.py behind the backend's own
 * classification-only gate, the explainability stage rides the
 * /ml/explain SHAP lane (the real shap_summary engine over a bounded
 * sample — the r289 queue row, closed), the drug-repurposing lane
 * rides routes/repurposing.py over the seeded SYNTHETIC screen
 * (every predicted value a labelled model estimate), the cancer
 * cell-line drug-response lane rides routes/drugresponse.py over
 * the seeded SYNTHETIC screen (the bilinear CaDRReS-style fit, the
 * sensitivity ranking, the biomarker readout, and the
 * cross-cell-line folds — every predicted value a labelled model
 * estimate, one response scale only), the multi-objective
 * optimization lane rides routes/multiobjective.py over the seeded
 * SYNTHETIC frame (the Pareto front with its aligned crowding, the
 * exact 2-D hypervolume indicator, the per-objective trade-off
 * corners — every objective value a labelled statistic), and the
 * export registers a real table artifact downloaded through the
 * token-header lane with digest verification.
 */

type VisualizationTableRequest = components["schemas"]["VisualizationTableRequest"];

/** The backend's own table-export row cap (visualization.py's `_MAX_ROWS`). */
const TABLE_EXPORT_MAX_ROWS = 5000;

/** The repurposing screen's parameter block (the route's own defaults). */
type RepurposingParams = {
  n_cells: number;
  n_drugs: number;
  n_features: number;
  n_factors: number;
  lr: number;
  max_epochs: number;
  noise: number;
  observed: number;
  seed: number;
  top_k: number;
  profiles: number;
};

const REPURPOSING_DEFAULTS: RepurposingParams = {
  n_cells: 12,
  n_drugs: 8,
  n_features: 6,
  n_factors: 3,
  lr: 0.05,
  max_epochs: 200,
  noise: 0.3,
  observed: 0.8,
  seed: 42,
  top_k: 5,
  profiles: 3,
};

/** The stage's numeric controls — labels state the route's own bounds. */
const RP_FIELDS: readonly {
  key: keyof RepurposingParams;
  label: string;
  step: number;
}[] = [
  { key: "n_cells", label: "cells (4–40)", step: 1 },
  { key: "n_drugs", label: "drugs (3–30)", step: 1 },
  { key: "n_features", label: "features (2–20)", step: 1 },
  { key: "n_factors", label: "factors (1–10)", step: 1 },
  { key: "lr", label: "lr (0.001–0.5)", step: 0.001 },
  { key: "max_epochs", label: "epochs (1–1000)", step: 10 },
  { key: "noise", label: "noise (0–2)", step: 0.05 },
  { key: "observed", label: "observed (0.3–1)", step: 0.05 },
  { key: "seed", label: "seed", step: 1 },
  { key: "top_k", label: "top-k (1–10)", step: 1 },
  { key: "profiles", label: "profiles (1–8)", step: 1 },
];

/** The drug-response analyze's parameter block (the route's own defaults). */
type DrugResponseParams = {
  n_cells: number;
  n_drugs: number;
  n_features: number;
  n_factors: number;
  lr: number;
  epochs: number;
  l2: number;
  noise: number;
  observed: number;
  seed: number;
  top_k: number;
  profiles: number;
  n_folds: number;
};

const DRUG_RESPONSE_DEFAULTS: DrugResponseParams = {
  n_cells: 12,
  n_drugs: 4,
  n_features: 8,
  n_factors: 3,
  lr: 0.1,
  epochs: 200,
  l2: 0.001,
  noise: 0.05,
  observed: 0.85,
  seed: 42,
  top_k: 5,
  profiles: 3,
  n_folds: 3,
};

/** The stage's numeric controls — labels state the route's own bounds. */
const DR_FIELDS: readonly {
  key: keyof DrugResponseParams;
  label: string;
  step: number;
}[] = [
  { key: "n_cells", label: "cells (4–40)", step: 1 },
  { key: "n_drugs", label: "drugs (2–30)", step: 1 },
  { key: "n_features", label: "features (2–20)", step: 1 },
  { key: "n_factors", label: "factors (1–10)", step: 1 },
  { key: "lr", label: "lr (0.001–0.5)", step: 0.001 },
  { key: "epochs", label: "epochs (1–1000)", step: 10 },
  { key: "l2", label: "l2 (0–1)", step: 0.001 },
  { key: "noise", label: "noise (0–0.5)", step: 0.01 },
  { key: "observed", label: "observed (0.5–1)", step: 0.05 },
  { key: "seed", label: "seed", step: 1 },
  { key: "top_k", label: "top-k (1–10)", step: 1 },
  { key: "profiles", label: "profiles (1–8)", step: 1 },
  { key: "n_folds", label: "folds (2–5)", step: 1 },
];

/** The pharmacogenomics analyze's parameter block (the route's own defaults). */
type PharmacogenomicsParams = {
  seed: number;
  top_k: number;
};

const PHARMACOGENOMICS_DEFAULTS: PharmacogenomicsParams = {
  seed: 42,
  top_k: 5,
};

/** The stage's numeric controls — labels state the route's own bounds. */
const PG_FIELDS: readonly {
  key: keyof PharmacogenomicsParams;
  label: string;
  step: number;
}[] = [
  { key: "seed", label: "seed (0–2147483647)", step: 1 },
  { key: "top_k", label: "top-k (1–20)", step: 1 },
];

/** The multiobjective optimize's parameter block (the route's own
 * defaults — the two default dimensions ride as editable integral
 * randint bounds). */
type MultiobjectiveParams = {
  model: string;
  task: "classification" | "regression";
  objectives: string[];
  sampler: "random" | "tpe";
  weight_mode: "spread" | "random";
  n_trials: number;
  cv: number;
  seed: number;
  n_weight_vectors: number;
  shortlist_keep: number;
  n_estimators_low: number;
  n_estimators_high: number;
  max_depth_low: number;
  max_depth_high: number;
};

const MULTIOBJECTIVE_DEFAULTS: MultiobjectiveParams = {
  model: "random_forest",
  task: "classification",
  objectives: ["score", "fit_seconds"],
  sampler: "tpe",
  weight_mode: "spread",
  n_trials: 20,
  cv: 3,
  seed: 42,
  n_weight_vectors: 8,
  shortlist_keep: 5,
  n_estimators_low: 10,
  n_estimators_high: 80,
  max_depth_low: 2,
  max_depth_high: 10,
};

/** The stage's numeric controls — labels state the route's own
 * bounds; the dimension rows state the low < high rule. */
type MultiobjectiveNumberKey =
  | "n_trials"
  | "cv"
  | "seed"
  | "n_weight_vectors"
  | "shortlist_keep"
  | "n_estimators_low"
  | "n_estimators_high"
  | "max_depth_low"
  | "max_depth_high";

const MO_FIELDS: readonly {
  key: MultiobjectiveNumberKey;
  label: string;
  step: number;
}[] = [
  { key: "n_trials", label: "trials (1–60)", step: 1 },
  { key: "cv", label: "cv (2–5)", step: 1 },
  { key: "seed", label: "seed (0–2147483647)", step: 1 },
  { key: "n_weight_vectors", label: "weights (2–32)", step: 1 },
  { key: "shortlist_keep", label: "keep (1–10)", step: 1 },
  { key: "n_estimators_low", label: "n_estimators lo", step: 1 },
  { key: "n_estimators_high", label: "n_estimators hi", step: 1 },
  { key: "max_depth_low", label: "max_depth lo", step: 1 },
  { key: "max_depth_high", label: "max_depth hi", step: 1 },
];

/** The tunable-objective checkboxes (the backend's own ledger keys). */
const MO_OBJECTIVES: readonly string[] = [
  "score",
  "fit_seconds",
  "score_seconds",
];

/** The multiobjective lane's trial table — the front and the
 * crowding-pruned shortlist share the same honest shape: the
 * objective-named value columns, the scalar, and the crowding cell
 * (boundary members carry the +inf extremes and read as
 * "boundary", an interior null reads as "null" — never a zero
 * spread). */
function MoTrialTable({
  rows,
  objectiveNames,
}: {
  rows: readonly MultiobjectiveTrialRow[];
  objectiveNames: readonly string[];
}) {
  return (
    <table className="mt-1 w-full text-xs">
      <thead>
        <tr className="text-[var(--color-muted-foreground)]">
          <th className="py-1 text-left font-normal">#</th>
          <th className="text-left font-normal">params</th>
          {objectiveNames.map((name) => (
            <th key={name} className="text-right font-normal">
              {name}
            </th>
          ))}
          <th className="text-right font-normal">scalar</th>
          <th className="text-right font-normal">crowding</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr
            key={row.number}
            className="border-t border-[var(--color-border)]"
          >
            <td className="py-1 font-mono">{row.number}</td>
            <td className="font-mono">
              {Object.entries(row.params)
                .map(([name, value]) => `${name}=${value}`)
                .join(", ")}
            </td>
            {row.values.map((value, i) => (
              <td key={i} className="text-right font-mono">
                {value.toFixed(3)}
              </td>
            ))}
            <td className="text-right font-mono">
              {row.scalar?.toFixed(3) ?? "null"}
            </td>
            <td className="text-right font-mono">
              {row.crowding_is_boundary
                ? "boundary"
                : (row.crowding?.toFixed(3) ?? "null")}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    const detail =
      typeof error.detail === "object" && error.detail !== null
        ? JSON.stringify(error.detail)
        : String(error.detail);
    return `API ${error.status}: ${detail}`;
  }
  return error instanceof Error ? error.message : String(error);
}

/** A dismissible error banner (§8.3: clear, actionable errors). */
function ErrorBanner({ message }: { message: string }) {
  return (
    <div
      role="alert"
      className="rounded-md border border-[var(--color-destructive)] bg-[var(--color-destructive)]/10 px-3 py-2 text-sm text-[var(--color-foreground)]"
    >
      <span className="font-medium">Request failed:</span>{" "}
      <span className="text-[var(--color-muted-foreground)]">{message}</span>
    </div>
  );
}

/** The §13 live job card: WS-driven state, progress, cancel, result relay. */
function JobRunCard({
  jobId,
  resultLabel,
  onResult,
}: {
  jobId: string;
  resultLabel: string;
  onResult: (result: unknown) => void;
}) {
  const { snapshot, cancel } = useJob(jobId);
  const relayed = useRef(false);
  const [relayError, setRelayError] = useState<string | null>(null);
  const view = snapshot?.view ?? null;

  useEffect(() => {
    if (
      view !== null &&
      view.state === "COMPLETED" &&
      view.result_ready &&
      !relayed.current
    ) {
      relayed.current = true;
      jobsApi
        .result(jobId)
        .then((response) => onResult(response.result))
        .catch((error) => setRelayError(describeError(error)));
    }
  }, [view, jobId, onResult]);

  if (view === null) {
    return <p className="text-sm text-[var(--color-muted-foreground)]">Tracking job…</p>;
  }
  const terminal = view.state === "COMPLETED" || view.state === "CANCELLED" || view.state === "FAILED";
  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-background)] p-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] font-semibold tracking-wide text-[var(--color-muted-foreground)]">
          {view.state}
        </span>
        <span className="text-xs text-[var(--color-muted-foreground)]">
          {view.operation} · {resultLabel}
        </span>
        {!terminal ? (
          <Button size="sm" variant="outline" onClick={() => void cancel()}>
            Cancel
          </Button>
        ) : null}
      </div>
      <div className="mt-3 h-2 overflow-hidden rounded-full bg-[var(--color-muted)]">
        <div
          className="h-full rounded-full bg-[var(--color-primary)] transition-[width]"
          style={{ width: `${Math.min(100, Math.max(0, view.progress))}%` }}
        />
      </div>
      <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
        {view.progress}% · {view.stage || "queued"}
      </p>
      {view.error !== null ? (
        <p className="mt-2 text-xs text-[var(--color-destructive)]">{view.error}</p>
      ) : null}
      {relayError !== null ? <ErrorBanner message={relayError} /> : null}
    </div>
  );
}

/** A numbered workflow stage header (§11.2's fixed order). */
function StageHeader({
  index,
  title,
  hint,
  icon,
  children,
}: {
  index: number;
  title: string;
  hint: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="mt-8">
      <div className="flex items-center gap-2">
        <span className="flex h-6 w-6 items-center justify-center rounded-md bg-[var(--color-muted)] text-xs font-semibold text-[var(--color-muted-foreground)]">
          {index}
        </span>
        <h2 className="text-lg font-semibold text-[var(--color-foreground)]">{title}</h2>
        <span className="text-xs text-[var(--color-muted-foreground)]">{hint}</span>
      </div>
      <div className="mt-3">{children}</div>
      {icon ? null : null}
    </section>
  );
}

/** One table cell's honest rendering (§17.3: type-aware, bounded). */
function formatValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "—";
  }
  return typeof value === "number"
    ? Number.isInteger(value)
      ? String(value)
      : value.toFixed(4)
    : String(value);
}

/**
 * Validate the hyperparameter overrides BEFORE any request fires
 * (§8.1): empty → null (the registry's own defaults); a JSON object
 * → the overrides; anything else → the thrown reason the inline
 * error renders verbatim.
 */
function validateModelParams(text: string): Record<string, unknown> | null {
  const trimmed = text.trim();
  if (trimmed === "") {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch (error) {
    throw new Error(
      `invalid JSON — ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (
    typeof parsed !== "object" ||
    parsed === null ||
    Array.isArray(parsed)
  ) {
    throw new Error(
      "model_params must be a JSON object of estimator parameter overrides, or empty for the registry's own defaults",
    );
  }
  return parsed as Record<string, unknown>;
}

/** The physchem panel's own display order (11 properties, the
 * backend's own key set — never renamed here). */
const ADMET_PHYSCHEM_ORDER: readonly string[] = [
  "molecular_weight",
  "logP",
  "hydrogen_bond_acceptors",
  "hydrogen_bond_donors",
  "Lipinski",
  "QED",
  "stereo_centers",
  "tpsa",
  "PAINS_alert",
  "BRENK_alert",
  "NIH_alert",
] as const;

const ADMET_PHYSCHEM_LABELS: Readonly<Record<string, string>> = {
  molecular_weight: "MW (Da)",
  logP: "Crippen LogP",
  hydrogen_bond_acceptors: "H-bond acceptors",
  hydrogen_bond_donors: "H-bond donors",
  Lipinski: "Lipinski (of 4)",
  QED: "QED",
  stereo_centers: "stereo centres",
  tpsa: "TPSA (A²)",
  PAINS_alert: "PAINS alerts",
  BRENK_alert: "BRENK alerts",
  NIH_alert: "NIH alerts",
};

/** One molecule's ADMET scorecard row — the honest per-row verdict. */
function AdmetMoleculeCard({ row }: { row: AdmetScoreRow }) {
  if (!row.valid) {
    return (
      <div className="mt-3 rounded-lg border border-[var(--color-border)] p-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-xs text-[var(--color-foreground)]">
            {row.smiles}
          </span>
          <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
            invalid SMILES — reported per-row
          </span>
        </div>
        <p className="mt-1 text-xs text-[var(--color-muted-foreground)]">
          {row.error}
        </p>
      </div>
    );
  }
  const alertTotal =
    (row.physchem["PAINS_alert"] ?? 0) +
    (row.physchem["BRENK_alert"] ?? 0) +
    (row.physchem["NIH_alert"] ?? 0);
  return (
    <div className="mt-3 rounded-lg border border-[var(--color-border)] p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-xs text-[var(--color-foreground)]">
          {row.smiles}
        </span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          valid
        </span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          Lipinski {row.physchem["Lipinski"] ?? 0}/4
        </span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          {alertTotal} structural alerts (PAINS/BRENK/NIH)
        </span>
      </div>
      <div className="mt-2 grid gap-x-6 gap-y-1 md:grid-cols-2">
        <table className="w-full border-collapse text-xs">
          <tbody>
            {ADMET_PHYSCHEM_ORDER.slice(0, 6).map((key) => (
              <tr key={key}>
                <td className="py-0.5 pr-3 text-[var(--color-muted-foreground)]">
                  {ADMET_PHYSCHEM_LABELS[key] ?? key}
                </td>
                <td className="py-0.5 text-right font-mono text-[var(--color-foreground)]">
                  {(row.physchem[key] ?? 0).toFixed(
                    key === "molecular_weight" || key === "tpsa" ? 1 : 2,
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <table className="w-full border-collapse text-xs">
          <tbody>
            {ADMET_PHYSCHEM_ORDER.slice(6).map((key) => (
              <tr key={key}>
                <td className="py-0.5 pr-3 text-[var(--color-muted-foreground)]">
                  {ADMET_PHYSCHEM_LABELS[key] ?? key}
                </td>
                <td className="py-0.5 text-right font-mono text-[var(--color-foreground)]">
                  {(row.physchem[key] ?? 0).toFixed(
                    key === "tpsa" ? 1 : 2,
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-[11px] font-semibold uppercase tracking-wide text-[var(--color-muted-foreground)]">
        Radial summary (0–100, DrugBank percentiles — the offline stand-ins)
      </p>
      <div className="mt-1 space-y-1">
        {Object.entries(row.radial).map(([label, score]) => (
          <div key={label} className="flex items-center gap-2">
            <span className="w-44 shrink-0 text-xs text-[var(--color-muted-foreground)]">
              {label}
            </span>
            <span className="h-2 flex-1 overflow-hidden rounded-full bg-[var(--color-muted)]">
              <span
                className="block h-full rounded-full bg-[var(--color-primary)]"
                style={{ width: `${Math.max(0, Math.min(100, score)).toFixed(1)}%` }}
              />
            </span>
            <span className="w-12 shrink-0 text-right font-mono text-xs text-[var(--color-foreground)]">
              {score.toFixed(1)}
            </span>
          </div>
        ))}
      </div>
      <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
        low-confidence endpoints (vendor&apos;s own &lt;0.6 quality rule):{" "}
        {row.low_confidence_endpoints.length > 0
          ? row.low_confidence_endpoints.join(", ")
          : "none"}
      </p>
    </div>
  );
}

/** One drug–target pair card — the honest per-row verdict (the
 * ligand validity + the protein composition + the affinity presence). */
function DtaPairCardView({ row }: { row: DtaPairCard }) {
  const topAac = Object.entries(row.protein_row.aac)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);
  return (
    <div className="mt-3 rounded-lg border border-[var(--color-border)] p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-xs text-[var(--color-foreground)]">
          {row.smiles}
        </span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          {row.valid ? "kept" : "dropped (invalid SMILES)"}
        </span>
        {row.affinity !== null ? (
          <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
            affinity {row.affinity}
          </span>
        ) : null}
      </div>
      <div className="mt-2 grid gap-x-6 gap-y-1 text-xs md:grid-cols-2">
        <div>
          <p className="text-[var(--color-muted-foreground)]">
            protein ({row.protein_row.prot_length} residues):{" "}
            <span className="font-mono">
              {row.protein.slice(0, 24)}
              {row.protein.length > 24 ? "…" : ""}
            </span>
          </p>
          <p className="mt-1 text-[var(--color-muted-foreground)]">
            standard{" "}
            {(row.protein_row.prot_standard_fraction * 100).toFixed(1)}% ·
            non-standard{" "}
            {(row.protein_row.prot_nonstandard_fraction * 100).toFixed(1)}%
          </p>
        </div>
        <div>
          <p className="text-[var(--color-muted-foreground)]">
            top AAC components:
          </p>
          <div className="mt-1 space-y-1">
            {topAac.map(([symbol, fraction]) => (
              <div key={symbol} className="flex items-center gap-2">
                <span className="w-8 shrink-0 font-mono text-[var(--color-foreground)]">
                  {symbol}
                </span>
                <span className="h-2 flex-1 overflow-hidden rounded-full bg-[var(--color-muted)]">
                  <span
                    className="block h-full rounded-full bg-[var(--color-primary)]"
                    style={{
                      width: `${Math.max(0, Math.min(100, fraction * 100)).toFixed(1)}%`,
                    }}
                  />
                </span>
                <span className="w-12 shrink-0 text-right font-mono text-[var(--color-foreground)]">
                  {(fraction * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export function MLPage() {
  // stage 1: the dataset registry + selection
  const [datasets, setDatasets] = useState<readonly DatasetView[]>([]);
  const [registryError, setRegistryError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selected, setSelected] = useState<DatasetView | null>(null);
  const [selectionError, setSelectionError] = useState<string | null>(null);

  // stage 3: preprocess
  const [preprocess, setPreprocess] = useState(true);

  // stage 4: split/cv configuration
  const [cv, setCv] = useState(5);
  const [tune, setTune] = useState(true);
  const [nJobs, setNJobs] = useState(-1);
  const [randomState, setRandomState] = useState(42);

  // stage 5: the algorithm catalog
  const [catalog, setCatalog] = useState<readonly ModelCatalogEntry[]>([]);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [catalogBusy, setCatalogBusy] = useState(false);

  // stage 6: hyperparameter overrides (JSON, validated on submit)
  const [modelParamsText, setModelParamsText] = useState("");
  const [modelParamsError, setModelParamsError] = useState<string | null>(null);

  // stage 7: train
  const [trainJobId, setTrainJobId] = useState<string | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [trainBusy, setTrainBusy] = useState(false);
  const [trainedModel, setTrainedModel] = useState<TrainedModelView | null>(null);

  // stage 8: evaluate (predict, dataset mode, sync)
  const [trainedModels, setTrainedModels] = useState<readonly TrainedModelView[]>([]);
  const [trainedListError, setTrainedListError] = useState<string | null>(null);
  const [selectedTrainedId, setSelectedTrainedId] = useState<string | null>(null);
  const [selectedTrained, setSelectedTrained] = useState<TrainedModelView | null>(null);
  const [predictProba, setPredictProba] = useState(true);
  const [predictBusy, setPredictBusy] = useState(false);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [predictResult, setPredictResult] = useState<PredictResponse | null>(null);

  // stage 10: applicability domain
  const [adXTest, setAdXTest] = useState<readonly (readonly number[])[] | null>(null);
  const [adAssembly, setAdAssembly] = useState<{
    rows: number;
    datasetRows: number;
    skipped: number;
    truncated: boolean;
  } | null>(null);
  const [adAssemblyError, setAdAssemblyError] = useState<string | null>(null);
  const [adBusy, setAdBusy] = useState(false);
  const [adK, setAdK] = useState(5);
  const [adPercentile, setAdPercentile] = useState(95);
  const [adNModels, setAdNModels] = useState(30);
  const [adUncertainty, setAdUncertainty] = useState(true);
  const [adFigure, setAdFigure] = useState(false);
  const [adJobId, setAdJobId] = useState<string | null>(null);
  const [adError, setAdError] = useState<string | null>(null);
  const [adResult, setAdResult] = useState<ADAnalyzeResult | null>(null);
  const [adArtifactStatus, setAdArtifactStatus] = useState<string | null>(null);

  // stage 10b: the leverage/Williams regression lane (B-4, part12 Phase 2)
  const [williamsResidualLimit, setWilliamsResidualLimit] = useState(3);
  const [williamsFigure, setWilliamsFigure] = useState(true);
  const [williamsBusy, setWilliamsBusy] = useState(false);
  const [williamsJobId, setWilliamsJobId] = useState<string | null>(null);
  const [williamsError, setWilliamsError] = useState<string | null>(null);
  const [williamsResult, setWilliamsResult] = useState<WilliamsResultPayload | null>(null);
  const [williamsArtifactStatus, setWilliamsArtifactStatus] = useState<string | null>(null);

  // stage 10c: the ADMET profile (part13 Phase 5 — the vendored
  // admet_ai registry + the per-molecule physchem panel, dataset-free)
  const [admetSmilesText, setAdmetSmilesText] = useState(
    "CC(=O)Oc1ccccc1C(=O)O\nCN1C=NC2=C1C(=O)N(C)C(=O)N2C",
  );
  const [admetError, setAdmetError] = useState<string | null>(null);
  const [admetBusy, setAdmetBusy] = useState(false);
  const [admetResult, setAdmetResult] = useState<AdmetScoreResponse | null>(null);
  const [admetRegistry, setAdmetRegistry] = useState<AdmetEndpointsResponse | null>(null);

  // stage 11b: the drug–target interaction lane (part13 Phase 5 —
  // DeepDTA's alphabets + the paired ECFP/AAC featurization, dataset-free)
  const [dtaPairsText, setDtaPairsText] = useState(
    "CCO,MKTAYIAKQRQISFVKSHFSRQ,5.0\nCCN,GAVLILKKGGSSHHTRPFSL,5.5\nCCC,ACDEFGHIKLMNPQRSTVWY,6.0",
  );
  const [dtaDipeptides, setDtaDipeptides] = useState(false);
  const [dtaError, setDtaError] = useState<string | null>(null);
  const [dtaBusy, setDtaBusy] = useState(false);
  const [dtaResult, setDtaResult] = useState<DtaFeaturesResponse | null>(null);
  const [dtaRegistry, setDtaRegistry] = useState<DtaRegistryResponse | null>(null);

  // stage 12b: the drug repurposing lane (part13 Phase 5 — the
  // CaDRReS recommender + DeepResponse's ranking over the SYNTHETIC
  // seeded screen, dataset-free)
  const [rpParams, setRpParams] =
    useState<RepurposingParams>(REPURPOSING_DEFAULTS);
  const [rpError, setRpError] = useState<string | null>(null);
  const [rpBusy, setRpBusy] = useState(false);
  const [rpResult, setRpResult] = useState<RepurposingScreenResponse | null>(null);
  const [rpRegistry, setRpRegistry] = useState<RepurposingRegistryResponse | null>(null);

  // stage 13b: the cancer cell-line drug-response lane (part13 Phase 5
  // — the bilinear CaDRReS-style model + the sensitivity ranking +
  // the biomarker readout + the cross-cell-line folds over the
  // SYNTHETIC seeded screen, dataset-free)
  const [drParams, setDrParams] =
    useState<DrugResponseParams>(DRUG_RESPONSE_DEFAULTS);
  const [drError, setDrError] = useState<string | null>(null);
  const [drBusy, setDrBusy] = useState(false);
  const [drResult, setDrResult] = useState<DrugResponseAnalyzeResponse | null>(null);
  const [drRegistry, setDrRegistry] = useState<DrugResponseRegistryResponse | null>(null);
  // stage 14b: the pharmacogenomics lane (part13 Phase 5 — the
  // classic GDSC ANOVA over ml/pharmacogenomics.py)
  const [pgParams, setPgParams] = useState<PharmacogenomicsParams>(PHARMACOGENOMICS_DEFAULTS);
  const [pgResult, setPgResult] = useState<PharmacogenomicsAnalyzeResponse | null>(null);
  const [pgError, setPgError] = useState<string | null>(null);
  const [pgBusy, setPgBusy] = useState(false);
  const [pgRegistry, setPgRegistry] = useState<PharmacogenomicsRegistryResponse | null>(null);

  // stage 15b: the multi-objective optimization lane (part13 Phase
  // 5's last remainder item — the Pareto HPO study over
  // ml/multiobjective.py, dataset-free)
  const [moParams, setMoParams] = useState<MultiobjectiveParams>(
    MULTIOBJECTIVE_DEFAULTS,
  );
  const [moResult, setMoResult] = useState<MultiobjectiveOptimizeResponse | null>(null);
  const [moError, setMoError] = useState<string | null>(null);
  const [moBusy, setMoBusy] = useState(false);
  const [moRegistry, setMoRegistry] = useState<MultiobjectiveRegistryResponse | null>(null);

  // stage 17: the Workflow Builder lane (part13 Phase 6, §8 — the
  // construct/save/reproduce surface over the r322 in-tree engine:
  // templates, the editable spec, validation, sync + job runs, the
  // bounded runs ledger)
  const [wfSteps, setWfSteps] = useState<WorkflowStepsResponse | null>(null);
  const [wfTemplates, setWfTemplates] =
    useState<WorkflowTemplatesResponse | null>(null);
  const [wfTemplateId, setWfTemplateId] = useState<string>("");
  const [wfSpecText, setWfSpecText] = useState<string>("");
  const [wfMode, setWfMode] = useState<"sync" | "job">("sync");
  const [wfSeed, setWfSeed] = useState(42);
  const [wfNSynthetic, setWfNSynthetic] = useState(30);
  const [wfValidate, setWfValidate] =
    useState<WorkflowValidateResponse | null>(null);
  const [wfRun, setWfRun] = useState<WorkflowRunRecordView | null>(null);
  const [wfRuns, setWfRuns] = useState<WorkflowRunsResponse | null>(null);
  const [wfJobId, setWfJobId] = useState<string | null>(null);
  const [wfError, setWfError] = useState<string | null>(null);
  const [wfBusy, setWfBusy] = useState(false);
  const [wfLedgerBusy, setWfLedgerBusy] = useState(false);

  // stage 11: semi-supervised learning (classification-only)
  const [ssMethod, setSsMethod] = useState<
    "self_training" | "label_spreading" | "label_propagation"
  >("self_training");
  const [ssBaseEstimator, setSsBaseEstimator] = useState<string>("");
  const [ssMaskFractionText, setSsMaskFractionText] = useState("");
  const [ssThreshold, setSsThreshold] = useState(0.75);
  const [ssKBest, setSsKBest] = useState(10);
  const [ssMaxIter, setSsMaxIter] = useState(10);
  const [ssKernel, setSsKernel] = useState<"knn" | "rbf">("knn");
  const [ssNNeighbors, setSsNNeighbors] = useState(7);
  const [ssGamma, setSsGamma] = useState(20);
  const [ssRandomState, setSsRandomState] = useState(42);
  const [ssJobId, setSsJobId] = useState<string | null>(null);
  const [ssError, setSsError] = useState<string | null>(null);
  const [ssResult, setSsResult] = useState<SemiSupervisedView | null>(null);

  // stage 11b: the held-out evaluation lane (r303 — the /evaluate GUI
  // consumer over the REAL evaluate_semi_supervised engine)
  const [ssEvalBusy, setSsEvalBusy] = useState(false);
  const [ssEvalJobId, setSsEvalJobId] = useState<string | null>(null);
  const [ssEvalError, setSsEvalError] = useState<string | null>(null);
  const [ssEvalResult, setSsEvalResult] = useState<SemiSupervisedEvaluation | null>(null);

  // stage 12: explainability (the real SHAP lane)
  const [exMaxRows, setExMaxRows] = useState(200);
  const [exFigure, setExFigure] = useState(true);
  const [exBusy, setExBusy] = useState(false);
  const [exError, setExError] = useState<string | null>(null);
  const [exResult, setExResult] = useState<ExplainResult | null>(null);
  const [exArtifactStatuses, setExArtifactStatuses] = useState<
    Record<string, string>
  >({});

  // stage 13: export + the verified download
  const [exportFormat, setExportFormat] = useState<"csv" | "tsv" | "xlsx">("csv");
  const [exportArtifact, setExportArtifact] = useState<ArtifactView | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportBusy, setExportBusy] = useState(false);
  const [digestStatus, setDigestStatus] = useState<string | null>(null);

  // the dataset's own task, narrowed to the registry's two literals
  const datasetTask = selected?.task;
  const selectedTask: "classification" | "regression" | null =
    datasetTask === "classification" || datasetTask === "regression"
      ? datasetTask
      : null;

  const refreshRegistry = useCallback(async () => {
    try {
      const response = await datasetsApi.list();
      setDatasets(response.datasets);
      setRegistryError(null);
    } catch (error) {
      setRegistryError(describeError(error));
    }
  }, []);

  const refreshTrained = useCallback(async () => {
    try {
      const response = await mlApi.trained();
      setTrainedModels(response.models);
      setTrainedListError(null);
    } catch (error) {
      setTrainedListError(describeError(error));
    }
  }, []);

  useEffect(() => {
    void refreshRegistry();
    void refreshTrained();
  }, [refreshRegistry, refreshTrained]);

  // the ADMET registry loads once on mount (the vocabulary line —
  // decorative context, not a gate: the score lane surfaces its own
  // errors and the card renders regardless)
  useEffect(() => {
    let cancelled = false;
    mlApi
      .admetEndpoints()
      .then((response) => {
        if (!cancelled) {
          setAdmetRegistry(response);
        }
      })
      .catch(() => {
        // the registry line stays absent — never a fabricated summary
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // the DTA registry loads once on mount (the vocabulary line —
  // decorative context, not a gate: the featurization lane surfaces
  // its own errors and the cards render regardless)
  useEffect(() => {
    let cancelled = false;
    mlApi
      .dtaRegistry()
      .then((response) => {
        if (!cancelled) {
          setDtaRegistry(response);
        }
      })
      .catch(() => {
        // the registry line stays absent — never a fabricated summary
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // the repurposing registry loads once on mount (the vocabulary
  // line — the same decorative-context contract as the DTA lane)
  useEffect(() => {
    let cancelled = false;
    mlApi
      .repurposingRegistry()
      .then((response) => {
        if (!cancelled) {
          setRpRegistry(response);
        }
      })
      .catch(() => {
        // the registry line stays absent — never a fabricated summary
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // the drug-response registry loads once on mount (the vocabulary
  // line — the same decorative-context contract as the DTA lane)
  useEffect(() => {
    let cancelled = false;
    mlApi
      .drugResponseRegistry()
      .then((response) => {
        if (!cancelled) {
          setDrRegistry(response);
        }
      })
      .catch(() => {
        // the registry line stays absent — never a fabricated summary
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // the pharmacogenomics registry loads once on mount (the
  // vocabulary line — the same decorative-context contract)
  useEffect(() => {
    let cancelled = false;
    mlApi
      .pharmacogenomicsRegistry()
      .then((response) => {
        if (!cancelled) {
          setPgRegistry(response);
        }
      })
      .catch(() => {
        // the registry line stays absent — never a fabricated summary
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // the multiobjective registry loads once on mount (the vocabulary
  // line — the same decorative-context contract)
  useEffect(() => {
    let cancelled = false;
    mlApi
      .multiobjectiveRegistry()
      .then((response) => {
        if (!cancelled) {
          setMoRegistry(response);
        }
      })
      .catch(() => {
        // the registry line stays absent — never a fabricated summary
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // the workflow engine's own vocabulary loads once on mount (the
  // §8 registry line — the same decorative-context contract; the
  // builder's controls render regardless, the lanes surface their own
  // errors)
  useEffect(() => {
    let cancelled = false;
    mlApi
      .workflowSteps()
      .then((response) => {
        if (!cancelled) {
          setWfSteps(response);
        }
      })
      .catch(() => {
        // the registry line stays absent — never a fabricated summary
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // the workflow templates load once on mount (the first entry
  // seeds the builder's editable spec — §8's construct canvas starts
  // from a REAL saved template, never a fabricated chain)
  useEffect(() => {
    let cancelled = false;
    mlApi
      .workflowTemplates()
      .then((response) => {
        if (!cancelled) {
          setWfTemplates(response);
          const first = response.templates[0];
          if (first !== undefined) {
            setWfTemplateId(first.template_id);
            setWfSpecText(JSON.stringify(first.spec, null, 2));
          }
        }
      })
      .catch(() => {
        // the select renders empty without the templates — never a
        // fabricated chain
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // the catalog stays synced with the selected dataset's own task
  useEffect(() => {
    if (selectedTask === null) {
      setCatalog([]);
      return;
    }
    let cancelled = false;
    setCatalogBusy(true);
    // only_available=false: the stage surfaces the unavailable entries
    // too, disabled with the registry's own reason (no pretend rows)
    mlApi
      .catalog(selectedTask, false)
      .then((response) => {
        if (!cancelled) {
          setCatalog(response.models);
          setCatalogError(null);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setCatalogError(describeError(error));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setCatalogBusy(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedTask]);

  const selectDataset = useCallback(async (datasetId: string) => {
    setSelectedId(datasetId);
    setSelectionError(null);
    // dataset-bound stage state resets with the selection
    setTrainedModel(null);
    setPredictResult(null);
    setSelectedTrained(null);
    setSelectedTrainedId(null);
    setAdXTest(null);
    setAdAssembly(null);
    setAdResult(null);
    setAdJobId(null);
    setAdArtifactStatus(null);
    setSsResult(null);
    setSsJobId(null);
    setExportArtifact(null);
    setDigestStatus(null);
    setSelectedModel("");
    try {
      const view = await datasetsApi.view(datasetId);
      setSelected(view);
    } catch (error) {
      setSelected(null);
      setSelectionError(describeError(error));
    }
  }, []);

  const selectTrainedModel = useCallback(async (modelId: string) => {
    setSelectedTrainedId(modelId);
    setPredictResult(null);
    setAdXTest(null);
    setAdAssembly(null);
    setAdResult(null);
    setAdJobId(null);
    setAdArtifactStatus(null);
    setExportArtifact(null);
    setDigestStatus(null);
    try {
      setSelectedTrained(await mlApi.trainedView(modelId));
    } catch (error) {
      setSelectedTrained(null);
      setTrainedListError(describeError(error));
    }
  }, []);

  // stage 7: train (job mode — the backend's default lane)
  const handleTrain = useCallback(async () => {
    if (selectedId === null || selectedModel === "") {
      return;
    }
    // the JSON overrides validate BEFORE any request fires — an
    // invalid body never leaves the page (no doomed 422)
    let modelParams: Record<string, unknown> | null = null;
    try {
      modelParams = validateModelParams(modelParamsText);
      setModelParamsError(null);
    } catch (error) {
      setModelParamsError(
        error instanceof Error ? error.message : String(error),
      );
      return;
    }
    setTrainBusy(true);
    setTrainError(null);
    setTrainJobId(null);
    setTrainedModel(null);
    try {
      const response = await mlApi.train({
        datasetId: selectedId,
        model: selectedModel,
        task: selectedTask,
        tune,
        cv,
        nJobs,
        preprocess,
        randomState,
        modelParams,
        mode: "job",
      });
      if (response.mode === "job") {
        setTrainJobId(response.job.job_id);
      }
    } catch (error) {
      setTrainError(describeError(error));
    } finally {
      setTrainBusy(false);
    }
  }, [
    selectedId,
    selectedModel,
    selectedTask,
    tune,
    cv,
    nJobs,
    preprocess,
    randomState,
    modelParamsText,
  ]);

  // the train job's result IS `{model: TrainedModelView}` — the
  // authoritative REST relay, parsed at runtime
  const handleTrainResult = useCallback(
    (result: unknown) => {
      try {
        const view = parseTrainResult(result, "train result");
        setTrainedModel(view);
        setSelectedTrained(view);
        setSelectedTrainedId(view.model_id);
        void refreshTrained();
      } catch (error) {
        setTrainError(describeError(error));
      }
    },
    [refreshTrained],
  );

  // stage 8: predict (dataset mode, sync — the route's own lane)
  const handlePredict = useCallback(async () => {
    if (selectedId === null || selectedTrainedId === null) {
      return;
    }
    setPredictBusy(true);
    setPredictError(null);
    setPredictResult(null);
    try {
      setPredictResult(
        await mlApi.predict({
          modelId: selectedTrainedId,
          datasetId: selectedId,
          predictProba,
        }),
      );
    } catch (error) {
      // a model without predict_proba answers the backend's own 422
      // detail — surfaced here verbatim, no invented probabilities
      setPredictError(describeError(error));
    } finally {
      setPredictBusy(false);
    }
  }, [selectedId, selectedTrainedId, predictProba]);

  // stage 10: assemble x_test through the bounded column bridge
  const handleAssembleXTest = useCallback(async () => {
    if (selectedTrained === null) {
      return;
    }
    setAdAssemblyError(null);
    setAdResult(null);
    setAdJobId(null);
    setAdArtifactStatus(null);
    try {
      const datasetId = selectedTrained.dataset_id;
      const features = selectedTrained.features;
      if (features.length === 0) {
        throw new Error("the trained model's view carries no feature list");
      }
      if (features.length > AD_MAX_FEATURES) {
        throw new Error(
          `the model trained on ${features.length} features; the AD engine's own cap is ${AD_MAX_FEATURES}`,
        );
      }
      const columns = await Promise.all(
        features.map((name) => datasetsApi.column(datasetId, name, 500)),
      );
      const returned = Math.min(...columns.map((column) => column.returned));
      const datasetRows = columns[0]?.n_rows ?? 0;
      const truncated = columns.some((column) => column.truncated);
      const rows: number[][] = [];
      let skipped = 0;
      for (let i = 0; i < returned; i++) {
        const row: number[] = [];
        let numeric = true;
        for (const column of columns) {
          const value = column.values[i];
          if (typeof value === "number" && Number.isFinite(value)) {
            row.push(value);
          } else {
            numeric = false;
            break;
          }
        }
        if (numeric) {
          rows.push(row);
        } else {
          skipped += 1;
        }
      }
      if (rows.length === 0) {
        throw new Error(
          "no fully-numeric rows assembled from the feature columns",
        );
      }
      setAdXTest(rows);
      setAdAssembly({
        rows: rows.length,
        datasetRows,
        skipped,
        truncated,
      });
    } catch (error) {
      setAdXTest(null);
      setAdAssembly(null);
      setAdAssemblyError(describeError(error));
    }
  }, [selectedTrained]);

  // stage 10: the AD analyze (job mode — the route's default lane)
  const handleAnalyze = useCallback(async () => {
    if (selectedTrained === null || adXTest === null) {
      return;
    }
    setAdBusy(true);
    setAdError(null);
    setAdResult(null);
    setAdJobId(null);
    setAdArtifactStatus(null);
    try {
      const response = await mlApi.analyze({
        modelId: selectedTrained.model_id,
        datasetId: selectedTrained.dataset_id,
        xTest: adXTest,
        k: adK,
        percentile: adPercentile,
        nModels: adNModels,
        uncertainty: adUncertainty,
        figure: adFigure,
        mode: "job",
      });
      if (response.mode === "job") {
        setAdJobId(response.job.job_id);
      }
    } catch (error) {
      setAdError(describeError(error));
    } finally {
      setAdBusy(false);
    }
  }, [
    selectedTrained,
    adXTest,
    adK,
    adPercentile,
    adNModels,
    adUncertainty,
    adFigure,
  ]);

  const handleAnalyzeResult = useCallback((result: unknown) => {
    try {
      setAdResult(parseADAnalyzeResult(result, "ad analyze result"));
    } catch (error) {
      setAdError(describeError(error));
    }
  }, []);

  // stage 10b: the leverage/Williams lane — the regression companion,
  // over the SAME assembled x_test (training-only when not assembled)
  const isRegressionModel =
    selectedTrained !== null && selectedTrained.task === "regression";

  const handleWilliams = useCallback(async () => {
    if (selectedTrained === null) {
      return;
    }
    setWilliamsBusy(true);
    setWilliamsError(null);
    setWilliamsResult(null);
    setWilliamsJobId(null);
    setWilliamsArtifactStatus(null);
    try {
      const response = await mlApi.williams({
        modelId: selectedTrained.model_id,
        datasetId: selectedTrained.dataset_id,
        xTest: adXTest,
        residualLimit: williamsResidualLimit,
        figure: williamsFigure,
        mode: "job",
      });
      if (response.mode === "job") {
        setWilliamsJobId(response.job.job_id);
      }
    } catch (error) {
      setWilliamsError(describeError(error));
    } finally {
      setWilliamsBusy(false);
    }
  }, [selectedTrained, adXTest, williamsResidualLimit, williamsFigure]);

  const handleWilliamsResult = useCallback((result: unknown) => {
    try {
      setWilliamsResult(parseWilliamsResult(result, "williams result"));
    } catch (error) {
      setWilliamsError(describeError(error));
    }
  }, []);

  // stage 11: the semi-supervised train (classification-only gate)
  const isClassification = selected?.task === "classification";

  const handleSemiTrain = useCallback(async () => {
    if (selectedId === null || !isClassification) {
      return;
    }
    setSsError(null);
    setSsResult(null);
    setSsJobId(null);
    const trimmed = ssMaskFractionText.trim();
    let maskFraction: number | null = null;
    if (trimmed !== "") {
      maskFraction = Number(trimmed);
      if (
        !Number.isFinite(maskFraction) ||
        maskFraction < 0.05 ||
        maskFraction > 0.9
      ) {
        setSsError(
          "mask_fraction must be a fraction in [0.05, 0.9], or empty when the target column already carries the unlabeled marker",
        );
        return;
      }
    }
    try {
      const response = await mlApi.semiSupervisedTrain({
        datasetId: selectedId,
        method: ssMethod,
        baseEstimator: ssBaseEstimator === "" ? null : ssBaseEstimator,
        maskFraction,
        threshold: ssThreshold,
        kBest: ssKBest,
        maxIter: ssMaxIter,
        kernel: ssKernel,
        nNeighbors: ssNNeighbors,
        gamma: ssGamma,
        randomState: ssRandomState,
        mode: "job",
      });
      if (response.mode === "job") {
        setSsJobId(response.job.job_id);
      }
    } catch (error) {
      setSsError(describeError(error));
    }
  }, [
    selectedId,
    isClassification,
    ssMaskFractionText,
    ssMethod,
    ssBaseEstimator,
    ssThreshold,
    ssKBest,
    ssMaxIter,
    ssKernel,
    ssNNeighbors,
    ssGamma,
    ssRandomState,
  ]);

  const handleSemiResult = useCallback(
    (result: unknown) => {
      try {
        setSsResult(parseSemiSupervisedResult(result, "semi-supervised result"));
        void refreshTrained();
      } catch (error) {
        setSsError(describeError(error));
      }
    },
    [refreshTrained],
  );

  // stage 10c: the ADMET scorecard (sync light compute — one button,
  // client-side validation first, the honest per-row verdicts)
  const handleAdmetScore = useCallback(async () => {
    const smiles = admetSmilesText
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0);
    if (smiles.length === 0) {
      setAdmetResult(null);
      setAdmetError("the ADMET panel needs at least one SMILES line");
      return;
    }
    if (smiles.length > ADMET_MAX_SMILES) {
      setAdmetResult(null);
      setAdmetError(
        `the route caps a batch at ${ADMET_MAX_SMILES} SMILES; got ${smiles.length}`,
      );
      return;
    }
    setAdmetBusy(true);
    setAdmetError(null);
    try {
      const response = await mlApi.admetScore(smiles);
      setAdmetResult(response);
    } catch (error) {
      setAdmetResult(null);
      setAdmetError(describeError(error));
    } finally {
      setAdmetBusy(false);
    }
  }, [admetSmilesText]);

  // stage 12: the DTA paired featurization (sync light compute — one
  // button, client-side validation first, the honest per-row verdicts;
  // the backend's own 422s surface verbatim)
  const handleDtaFeatures = useCallback(async () => {
    const lines = dtaPairsText
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => line.length > 0);
    if (lines.length === 0) {
      setDtaResult(null);
      setDtaError("the DTA lane needs at least one SMILES,PROTEIN line");
      return;
    }
    if (lines.length > DTA_MAX_PAIRS) {
      setDtaResult(null);
      setDtaError(
        `the route caps a batch at ${DTA_MAX_PAIRS} pairs; got ${lines.length}`,
      );
      return;
    }
    const pairs: {
      smiles: string;
      protein: string;
      affinity?: number;
    }[] = [];
    for (const [i, line] of lines.entries()) {
      const fields = line.split(",").map((field) => field.trim());
      const [smilesField, proteinField, affinityField] = fields;
      if (
        smilesField === undefined ||
        proteinField === undefined ||
        smilesField.length === 0 ||
        proteinField.length === 0
      ) {
        setDtaResult(null);
        setDtaError(
          `line ${i + 1} needs at least SMILES,PROTEIN (affinity is the optional third field)`,
        );
        return;
      }
      const pair: {
        smiles: string;
        protein: string;
        affinity?: number;
      } = { smiles: smilesField, protein: proteinField };
      if (affinityField !== undefined && affinityField.length > 0) {
        const affinity = Number(affinityField);
        if (!Number.isFinite(affinity)) {
          setDtaResult(null);
          setDtaError(`line ${i + 1}: the affinity must be a finite number`);
          return;
        }
        pair.affinity = affinity;
      }
      pairs.push(pair);
    }
    setDtaBusy(true);
    setDtaError(null);
    try {
      const response = await mlApi.dtaFeatures(pairs, dtaDipeptides);
      setDtaResult(response);
    } catch (error) {
      setDtaResult(null);
      setDtaError(describeError(error));
    } finally {
      setDtaBusy(false);
    }
  }, [dtaPairsText, dtaDipeptides]);

  // stage 12b: the drug repurposing screen (sync light compute — one
  // button, client-side validation before any request; the backend's
  // own 422s surface verbatim; every value a labelled model estimate)
  const updateRpParam = useCallback(
    (key: keyof RepurposingParams, value: number) => {
      setRpParams((prev) => ({ ...prev, [key]: value }));
    },
    [],
  );

  const handleRepurposingScreen = useCallback(async () => {
    const p = rpParams;
    const fail = (message: string) => {
      setRpResult(null);
      setRpError(message);
    };
    if (Object.values(p).some((value) => !Number.isFinite(value))) {
      fail("every screen parameter must be a finite number");
      return;
    }
    if (p.n_cells < 4) {
      fail("the screen needs at least 4 cell lines (the backend's own bound)");
      return;
    }
    if (p.n_drugs < 3) {
      fail("the screen needs at least 3 drugs (the backend's own bound)");
      return;
    }
    if (p.n_features < 2) {
      fail("the screen needs at least 2 features (the backend's own bound)");
      return;
    }
    if (p.n_cells > 40 || p.n_drugs > 30 || p.n_features > 20) {
      fail("the route caps the screen at 40 cells × 30 drugs × 20 features (the sync-light budget)");
      return;
    }
    if (p.n_factors < 1 || p.n_factors > 10) {
      fail("the pharmacogenomic space is 1..10 factors");
      return;
    }
    if (p.lr < 0.001 || p.lr > 0.5) {
      fail("the learning rate must lie in 0.001..0.5");
      return;
    }
    if (p.max_epochs < 1 || p.max_epochs > 1000) {
      fail("the epoch ceiling is 1..1000 (the sync-light budget)");
      return;
    }
    if (p.observed < 0.3 || p.observed > 1.0) {
      fail("the observed fraction must lie in 0.3..1.0 (the backend's own bound)");
      return;
    }
    if (p.noise < 0 || p.noise > 2) {
      fail("the noise scale must lie in 0..2");
      return;
    }
    if (p.seed < 0) {
      fail("the seed must be non-negative");
      return;
    }
    if (p.top_k < 1 || p.top_k > 10) {
      fail("top-k per profile is 1..10");
      return;
    }
    if (p.profiles < 1 || p.profiles > 8) {
      fail("the answer carries 1..8 profiles");
      return;
    }
    setRpBusy(true);
    setRpError(null);
    try {
      const response = await mlApi.repurposingScreen({
        n_cells: p.n_cells,
        n_drugs: p.n_drugs,
        n_features: p.n_features,
        n_factors: p.n_factors,
        lr: p.lr,
        max_epochs: p.max_epochs,
        noise: p.noise,
        observed: p.observed,
        seed: p.seed,
        top_k: p.top_k,
        profiles: p.profiles,
      });
      setRpResult(response);
    } catch (error) {
      setRpResult(null);
      setRpError(describeError(error));
    } finally {
      setRpBusy(false);
    }
  }, [rpParams]);

  // stage 13b: the cancer cell-line drug-response analyze (sync light
  // compute — one button, client-side validation before any request;
  // the backend's own 422s surface verbatim; every value a labelled
  // model estimate, one response scale only)
  const updateDrParam = useCallback(
    (key: keyof DrugResponseParams, value: number) => {
      setDrParams((prev) => ({ ...prev, [key]: value }));
    },
    [],
  );

  const handleDrugResponseAnalyze = useCallback(async () => {
    const p = drParams;
    const fail = (message: string) => {
      setDrResult(null);
      setDrError(message);
    };
    if (Object.values(p).some((value) => !Number.isFinite(value))) {
      fail("every analyze parameter must be a finite number");
      return;
    }
    if (p.n_cells < 4) {
      fail("the screen needs at least 4 cell lines (the backend's own bound)");
      return;
    }
    if (p.n_drugs < 2) {
      fail("the screen needs at least 2 drugs (the backend's own bound)");
      return;
    }
    if (p.n_features < 2) {
      fail("the screen needs at least 2 features (the backend's own bound)");
      return;
    }
    if (p.n_cells > 40 || p.n_drugs > 30 || p.n_features > 20) {
      fail("the route caps the screen at 40 cells × 30 drugs × 20 features (the sync-light budget)");
      return;
    }
    if (p.n_factors < 1 || p.n_factors > 10) {
      fail("the latent space is 1..10 factors");
      return;
    }
    if (p.lr < 0.001 || p.lr > 0.5) {
      fail("the learning rate must lie in 0.001..0.5");
      return;
    }
    if (p.epochs < 1 || p.epochs > 1000) {
      fail("the epoch ceiling is 1..1000 (the sync-light budget)");
      return;
    }
    if (p.l2 < 0 || p.l2 > 1) {
      fail("the L2 shrinkage must lie in 0..1");
      return;
    }
    if (p.noise < 0 || p.noise > 0.5) {
      fail("the noise scale must lie in 0..0.5");
      return;
    }
    if (p.observed < 0.5 || p.observed > 1.0) {
      fail("the observed fraction must lie in 0.5..1.0 (the backend's own bound)");
      return;
    }
    if (p.seed < 0) {
      fail("the seed must be non-negative");
      return;
    }
    if (p.top_k < 1 || p.top_k > 10) {
      fail("top-k per profile is 1..10");
      return;
    }
    if (p.profiles < 1 || p.profiles > 8) {
      fail("the answer carries 1..8 profiles");
      return;
    }
    if (p.n_folds < 2 || p.n_folds > 5) {
      fail("the cross-validation carries 2..5 folds");
      return;
    }
    if (p.n_folds > p.n_cells) {
      fail("the folds split cell lines — n_folds may not exceed the cell count (the backend's own 422)");
      return;
    }
    setDrBusy(true);
    setDrError(null);
    try {
      const response = await mlApi.drugResponseAnalyze({
        n_cells: p.n_cells,
        n_drugs: p.n_drugs,
        n_features: p.n_features,
        n_factors: p.n_factors,
        lr: p.lr,
        epochs: p.epochs,
        l2: p.l2,
        noise: p.noise,
        observed: p.observed,
        seed: p.seed,
        top_k: p.top_k,
        profiles: p.profiles,
        n_folds: p.n_folds,
      });
      setDrResult(response);
    } catch (error) {
      setDrResult(null);
      setDrError(describeError(error));
    } finally {
      setDrBusy(false);
    }
  }, [drParams]);

  // stage 14b: the pharmacogenomics analyze (sync light compute —
  // one button, client-side validation before any request; the
  // backend's own 422s surface verbatim; every p-value a labelled
  // statistic over the synthetic screen, one response scale only)
  const updatePgParam = useCallback(
    (key: keyof PharmacogenomicsParams, value: number) => {
      setPgParams((prev) => ({ ...prev, [key]: value }));
    },
    [],
  );

  const handlePharmacogenomicsAnalyze = useCallback(async () => {
    const p = pgParams;
    const fail = (message: string) => {
      setPgResult(null);
      setPgError(message);
    };
    if (Object.values(p).some((value) => !Number.isFinite(value))) {
      fail("every analyze parameter must be a finite number");
      return;
    }
    if (p.seed < 0 || p.seed > 2147483647) {
      fail("the seed must lie in 0..2147483647 (the route's own bound)");
      return;
    }
    if (p.top_k < 1 || p.top_k > 20) {
      fail("the answer carries 1..20 associations (the route's own bound)");
      return;
    }
    setPgBusy(true);
    setPgError(null);
    try {
      const response = await mlApi.pharmacogenomicsAnalyze({
        seed: p.seed,
        top_k: p.top_k,
      });
      setPgResult(response);
    } catch (error) {
      setPgResult(null);
      setPgError(describeError(error));
    } finally {
      setPgBusy(false);
    }
  }, [pgParams]);

  // stage 15b: the multi-objective optimize (sync light compute —
  // one button, client-side validation before any request; the
  // backend's own 422s surface verbatim; every objective value a
  // labelled statistic over the synthetic frame)
  const updateMoParam = useCallback(
    (key: MultiobjectiveNumberKey, value: number) => {
      setMoParams((prev) => ({ ...prev, [key]: value }));
    },
    [],
  );

  const toggleMoObjective = useCallback((label: string, checked: boolean) => {
    setMoParams((prev) => ({
      ...prev,
      objectives: checked
        ? [...prev.objectives, label]
        : prev.objectives.filter((name) => name !== label),
    }));
  }, []);

  const handleMultiobjectiveOptimize = useCallback(async () => {
    const p = moParams;
    const fail = (message: string) => {
      setMoResult(null);
      setMoError(message);
    };
    if (p.model.trim().length === 0 || p.model.length > 64) {
      fail("the model name must be 1..64 characters (the route's own bound)");
      return;
    }
    if (p.objectives.length < 2 || p.objectives.length > 3) {
      fail("a Pareto study needs 2..3 distinct objectives (the route's own bound)");
      return;
    }
    if (!Number.isInteger(p.n_trials) || p.n_trials < 1 || p.n_trials > 60) {
      fail("n_trials must be an integer in [1, 60] (the route's own bound)");
      return;
    }
    if (!Number.isInteger(p.cv) || p.cv < 2 || p.cv > 5) {
      fail("cv must be an integer in [2, 5] (the route's own bound)");
      return;
    }
    if (!Number.isInteger(p.seed) || p.seed < 0 || p.seed > 2147483647) {
      fail("the seed must lie in 0..2147483647 (the route's own bound)");
      return;
    }
    if (
      !Number.isInteger(p.n_weight_vectors) ||
      p.n_weight_vectors < 2 ||
      p.n_weight_vectors > 32
    ) {
      fail("n_weight_vectors must be an integer in [2, 32] (the route's own bound)");
      return;
    }
    if (
      !Number.isInteger(p.shortlist_keep) ||
      p.shortlist_keep < 1 ||
      p.shortlist_keep > 10
    ) {
      fail("shortlist_keep must be an integer in [1, 10] (the route's own bound)");
      return;
    }
    if (
      !Number.isInteger(p.n_estimators_low) ||
      !Number.isInteger(p.n_estimators_high) ||
      p.n_estimators_low >= p.n_estimators_high
    ) {
      fail("n_estimators needs integral bounds with low < high (the route's own rule)");
      return;
    }
    if (
      !Number.isInteger(p.max_depth_low) ||
      !Number.isInteger(p.max_depth_high) ||
      p.max_depth_low >= p.max_depth_high
    ) {
      fail("max_depth needs integral bounds with low < high (the route's own rule)");
      return;
    }
    setMoBusy(true);
    setMoError(null);
    try {
      const response = await mlApi.multiobjectiveOptimize({
        model: p.model.trim(),
        task: p.task,
        objectives: p.objectives,
        dimensions: [
          {
            name: "n_estimators",
            kind: "randint",
            low: p.n_estimators_low,
            high: p.n_estimators_high,
          },
          {
            name: "max_depth",
            kind: "randint",
            low: p.max_depth_low,
            high: p.max_depth_high,
          },
        ],
        n_trials: p.n_trials,
        cv: p.cv,
        sampler: p.sampler,
        seed: p.seed,
        weight_mode: p.weight_mode,
        n_weight_vectors: p.n_weight_vectors,
        shortlist_keep: p.shortlist_keep,
      });
      setMoResult(response);
    } catch (error) {
      setMoResult(null);
      setMoError(describeError(error));
    } finally {
      setMoBusy(false);
    }
  }, [moParams]);

  // stage 17: the Workflow Builder handlers (§8's construct/save/
  // reproduce surface — every gate fires client-side BEFORE any
  // request; the backend's own errors ride the banner verbatim after)
  const handleWfTemplateChange = useCallback(
    (templateId: string) => {
      setWfTemplateId(templateId);
      const template = wfTemplates?.templates.find(
        (t) => t.template_id === templateId,
      );
      if (template !== undefined) {
        setWfSpecText(JSON.stringify(template.spec, null, 2));
        setWfValidate(null);
        setWfRun(null);
        setWfJobId(null);
        setWfError(null);
      }
    },
    [wfTemplates],
  );

  /** Parse the editable spec — the thrown reason rides the banner
   * verbatim, never a silent fallback. */
  const parseWfSpec = useCallback((): Record<string, unknown> => {
    const trimmed = wfSpecText.trim();
    if (trimmed === "") {
      throw new Error("the spec is empty — load a template or paste a spec");
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(trimmed);
    } catch (error) {
      throw new Error(
        `invalid spec JSON — ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    if (
      typeof parsed !== "object" ||
      parsed === null ||
      Array.isArray(parsed)
    ) {
      throw new Error("the spec must be a JSON object (the route's own shape)");
    }
    return parsed as Record<string, unknown>;
  }, [wfSpecText]);

  const handleWorkflowValidate = useCallback(async () => {
    let spec: Record<string, unknown>;
    try {
      spec = parseWfSpec();
    } catch (error) {
      setWfValidate(null);
      setWfError(error instanceof Error ? error.message : String(error));
      return;
    }
    setWfBusy(true);
    setWfError(null);
    try {
      const response = await mlApi.workflowValidate(spec);
      setWfValidate(response);
    } catch (error) {
      setWfValidate(null);
      setWfError(describeError(error));
    } finally {
      setWfBusy(false);
    }
  }, [parseWfSpec]);

  const handleWorkflowRun = useCallback(async () => {
    let spec: Record<string, unknown>;
    try {
      spec = parseWfSpec();
    } catch (error) {
      setWfRun(null);
      setWfJobId(null);
      setWfError(error instanceof Error ? error.message : String(error));
      return;
    }
    if (
      !Number.isInteger(wfSeed) ||
      wfSeed < 0 ||
      wfSeed > WORKFLOW_MAX_SEED
    ) {
      setWfRun(null);
      setWfJobId(null);
      setWfError(
        `the seed must lie in 0..${WORKFLOW_MAX_SEED} (the route's own bound)`,
      );
      return;
    }
    if (
      !Number.isInteger(wfNSynthetic) ||
      wfNSynthetic < WORKFLOW_MIN_SYNTHETIC ||
      wfNSynthetic > WORKFLOW_MAX_SYNTHETIC
    ) {
      setWfRun(null);
      setWfJobId(null);
      setWfError(
        `n_synthetic must be an integer in [${WORKFLOW_MIN_SYNTHETIC}, ${WORKFLOW_MAX_SYNTHETIC}] (the route's own dial)`,
      );
      return;
    }
    const nodes = Array.isArray(spec["nodes"]) ? spec["nodes"].length : 0;
    const nodeCap =
      wfMode === "sync" ? WORKFLOW_SYNC_MAX_NODES : WORKFLOW_JOB_MAX_NODES;
    if (nodes > nodeCap) {
      setWfRun(null);
      setWfJobId(null);
      setWfError(
        `a ${wfMode}-mode run is bounded to ${nodeCap} nodes (the route's own bound — the spec carries ${nodes})`,
      );
      return;
    }
    setWfBusy(true);
    setWfError(null);
    setWfRun(null);
    setWfValidate(null);
    setWfJobId(null);
    try {
      const response = await mlApi.workflowRun({
        spec,
        seed: wfSeed,
        n_synthetic: wfNSynthetic,
        mode: wfMode,
      });
      if (response.mode === "sync") {
        setWfRun(response.run);
      } else {
        setWfJobId(response.job_id);
      }
    } catch (error) {
      setWfError(describeError(error));
    } finally {
      setWfBusy(false);
    }
  }, [parseWfSpec, wfMode, wfSeed, wfNSynthetic]);

  /** The job-mode COMPLETED relay — the authoritative REST result
   * parsed through the REAL run-record parser, never the WS payload
   * alone (§13). */
  const handleWfRunResult = useCallback((result: unknown) => {
    try {
      setWfRun(parseWorkflowRunRecord(result, "workflow job result"));
    } catch (error) {
      setWfError(describeError(error));
    }
  }, []);

  const handleWorkflowRunsLedger = useCallback(async () => {
    setWfLedgerBusy(true);
    try {
      const response = await mlApi.workflowRuns();
      setWfRuns(response);
    } catch (error) {
      setWfRuns(null);
      setWfError(describeError(error));
    } finally {
      setWfLedgerBusy(false);
    }
  }, []);

  // stage 17 derived: the selected template + the step-title lookup
  // (the chain renders the registry's own titles, never invented ones)
  const wfTemplate: WorkflowTemplateView | null =
    wfTemplates?.templates.find((t) => t.template_id === wfTemplateId) ??
    null;
  const wfStepTitles = new Map<string, string>(
    wfSteps?.steps.map((s) => [s.step_id, s.title] as [string, string]) ??
      [],
  );

  // stage 11b: the held-out evaluation (3 seeded repeats, test_size
  // 0.25 — the route's own defaults surfaced, one button, job lane)
  const handleSemiEvaluate = useCallback(async () => {
    if (selectedId === null || !isClassification) {
      return;
    }
    setSsEvalBusy(true);
    setSsEvalError(null);
    setSsEvalResult(null);
    setSsEvalJobId(null);
    try {
      const response = await mlApi.semiSupervisedEvaluate({
        datasetId: selectedId,
        method: ssMethod,
        testSize: 0.25,
        nRepeats: 3,
        randomState: 42,
        mode: "job",
      });
      if (response.mode === "job") {
        setSsEvalJobId(response.job.job_id);
      }
    } catch (error) {
      setSsEvalError(describeError(error));
    } finally {
      setSsEvalBusy(false);
    }
  }, [selectedId, isClassification, ssMethod]);

  const handleSemiEvaluateResult = useCallback(
    (result: unknown) => {
      try {
        const record = result as Record<string, unknown>;
        setSsEvalResult(
          parseSemiSupervisedEvaluation(
            record["evaluation"] ?? record,
            "semi evaluate result",
          ),
        );
      } catch (error) {
        setSsEvalError(describeError(error));
      }
    },
    [],
  );

  // stage 12: the SHAP explainability lane (sync by default; job mode
  // relays the same payload through the JobRunCard)
  const handleExplain = useCallback(async () => {
    if (selectedTrained === null) {
      return;
    }
    if (
      !Number.isInteger(exMaxRows) ||
      exMaxRows < 1 ||
      exMaxRows > EXPLAIN_MAX_ROWS
    ) {
      setExError(
        `max_rows must be an integer in [1, ${EXPLAIN_MAX_ROWS}] (the route's own bound)`,
      );
      return;
    }
    setExError(null);
    setExResult(null);
    setExArtifactStatuses({});
    setExBusy(true);
    try {
      const response = await mlApi.explain({
        modelId: selectedTrained.model_id,
        datasetId: null, // the model's own training dataset
        maxRows: exMaxRows,
        figure: exFigure,
        mode: "sync",
      });
      if (response.mode === "sync") {
        setExResult(response.explain);
      }
    } catch (error) {
      setExError(describeError(error));
    } finally {
      setExBusy(false);
    }
  }, [selectedTrained, exMaxRows, exFigure]);

  // the shared digest-verified download lane (the artifacts client's
  // own raw lane — reused by the AD figure and the Export stage)
  const verifiedDownload = useCallback(
    async (artifact: ArtifactView): Promise<string> => {
      const download = await artifactsApi.download(artifact.artifact_id);
      let status: string;
      if (download.digest_verified === true) {
        status = `digest verified: ${download.actual_sha256?.slice(0, 16)}…`;
      } else if (download.digest_verified === null) {
        status =
          "downloaded; local digest unavailable (insecure context) — the route's X-Artifact-SHA256 was received";
      } else {
        status =
          "DIGEST MISMATCH — the downloaded bytes differ from the route's promise";
      }
      const url = URL.createObjectURL(download.blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = download.filename;
      anchor.click();
      URL.revokeObjectURL(url);
      return status;
    },
    [],
  );

  const handleAdArtifactDownload = useCallback(async () => {
    if (adResult?.artifact === null || adResult === null) {
      return;
    }
    setAdArtifactStatus(null);
    try {
      setAdArtifactStatus(await verifiedDownload(adResult.artifact));

    } catch (error) {
      setAdArtifactStatus(describeError(error));
    }
  }, [adResult, verifiedDownload]);

  const handleWilliamsArtifactDownload = useCallback(async () => {
    if (williamsResult === null || williamsResult.artifact === null) {
      return;
    }
    setWilliamsArtifactStatus(null);
    try {
      setWilliamsArtifactStatus(await verifiedDownload(williamsResult.artifact));
    } catch (error) {
      setWilliamsArtifactStatus(describeError(error));
    }
  }, [williamsResult, verifiedDownload]);

  const handleExplainArtifactDownload = useCallback(
    async (artifactId: string) => {
      const artifact = exResult?.artifacts.find(
        (a) => a.artifact_id === artifactId,
      );
      if (artifact === undefined || exResult === null) {
        return;
      }
      setExArtifactStatuses((prev) => ({ ...prev, [artifactId]: "…" }));
      try {
        const status = await verifiedDownload(artifact);
        setExArtifactStatuses((prev) => ({ ...prev, [artifactId]: status }));
      } catch (error) {
        setExArtifactStatuses((prev) => ({
          ...prev,
          [artifactId]: describeError(error),
        }));
      }
    },
    [exResult, verifiedDownload],
  );

  // stage 13: the export over the LAST predictions block
  const handleExport = useCallback(async () => {
    if (predictResult === null) {
      return;
    }
    setExportBusy(true);
    setExportError(null);
    setExportArtifact(null);
    setDigestStatus(null);
    try {
      const built = buildExportRows(predictResult);
      if (built.rows.length > TABLE_EXPORT_MAX_ROWS) {
        throw new Error(
          `the prediction set carries ${built.rows.length} rows; the table export route's own cap is ${TABLE_EXPORT_MAX_ROWS}`,
        );
      }
      if (built.rows.length === 0) {
        throw new Error("the prediction block carries no rows to export");
      }
      const request: VisualizationTableRequest = {
        rows: built.rows,
        format: exportFormat,
        filename: "ml-predictions",
      };
      const raw = await api.post<Record<string, unknown>>(
        "/api/v1/visualization/table",
        request,
      );
      const artifact = raw["artifact"];
      setExportArtifact(
        typeof artifact === "object" && artifact !== null
          ? parseArtifactView(artifact, "export artifact")
          : null,
      );
    } catch (error) {
      setExportError(describeError(error));
    } finally {
      setExportBusy(false);
    }
  }, [predictResult, exportFormat]);

  const handleDownload = useCallback(async () => {
    if (exportArtifact === null) {
      return;
    }
    setDigestStatus(null);
    try {
      setDigestStatus(await verifiedDownload(exportArtifact));
    } catch (error) {
      setDigestStatus(describeError(error));
    }
  }, [exportArtifact, verifiedDownload]);

  const compareModels = [...trainedModels].sort((a, b) => {
    if (a.cv_mean === null && b.cv_mean === null) {
      return 0;
    }
    if (a.cv_mean === null) {
      return 1;
    }
    if (b.cv_mean === null) {
      return -1;
    }
    return b.cv_mean - a.cv_mean;
  });

  const classificationCatalog = catalog.filter(
    (entry) =>
      (entry.task === "classification" || entry.task === "both") &&
      entry.available,
  );

  return (
    <div className="mx-auto max-w-7xl px-4 py-10">
      <div className="max-w-3xl">
        <h1 className="text-2xl font-bold tracking-tight text-[var(--color-foreground)]">
          Machine Learning
        </h1>
        <p className="mt-2 text-sm leading-relaxed text-[var(--color-muted-foreground)]">
          The complete sequential workflow — select dataset, target and
          features, preprocess, configure split/CV, algorithm,
          hyperparameters, train, evaluate, compare, applicability
          domain, semi-supervised learning, explainability, export —
          every step runs the real Python backend through the local API.
        </p>
        <p className="mt-2 text-xs leading-relaxed text-[var(--color-muted-foreground)]">
          State, honestly: the train and semi-supervised lanes run as
          managed background jobs with the authoritative REST result
          relay; the predict lane runs sync over the registered
          dataset&apos;s own feature matrix (its evaluation block now
          plots through the interactive Plotly lanes — the parity
          scatter with per-click sample metadata, the confusion
          heatmap, the AD query-distance view, the mean-|SHAP| bar);
          the applicability-domain stage assembles x_test through
          the bounded column bridge; the explainability stage runs
          the real shap_summary engine sync over a bounded sample
          (job mode available) and refuses non-tree estimators
          honestly. No feature on this page is a claim that was not
          executed.
        </p>
      </div>

      <StageHeader
        index={1}
        title="Select Dataset"
        hint="the session registry, one dataset selected"
        icon={<Database aria-hidden className="h-4 w-4" />}
      >
        {registryError !== null ? <ErrorBanner message={registryError} /> : null}
        <div className="overflow-hidden rounded-lg border border-[var(--color-border)]">
          <div className="max-h-48 overflow-auto">
            <table className="w-full border-collapse text-xs">
              <thead className="sticky top-0 bg-[var(--color-muted)]">
                <tr>
                  {["Name", "Rows", "Columns", "Target", "Task", "Created"].map((header) => (
                    <th key={header} scope="col" className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]">
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {datasets.map((entry) => (
                  <tr
                    key={entry.dataset_id}
                    onClick={() => void selectDataset(entry.dataset_id)}
                    className={`cursor-pointer border-t border-[var(--color-border)] ${
                      entry.dataset_id === selectedId
                        ? "bg-[var(--color-accent)]"
                        : "bg-[var(--color-card)] hover:bg-[var(--color-accent)]/50"
                    }`}
                  >
                    <td className="whitespace-nowrap px-3 py-1.5 font-medium text-[var(--color-foreground)]">{entry.name}</td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">{entry.shape[0]}</td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">{entry.shape[1]}</td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">{entry.target ?? "—"}</td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">{entry.task ?? "—"}</td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">{entry.created_at}</td>
                  </tr>
                ))}
                {datasets.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="px-3 py-4 text-center text-[var(--color-muted-foreground)]">
                      No datasets registered yet — import one in the Datasets workspace.
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </div>
        {selectionError !== null ? <div className="mt-3"><ErrorBanner message={selectionError} /></div> : null}
        {selected !== null ? (
          <Card className="mt-4">
            <CardHeader>
              <CardTitle className="text-base">{selected.name}</CardTitle>
              <CardDescription>
                {selected.shape[0]} rows × {selected.shape[1]} columns ·{" "}
                {selected.n_features} numeric features · target{" "}
                {selected.target ?? "none"} · task {selected.task ?? "inferred"}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-xs text-[var(--color-muted-foreground)]">
                The registered dataset&apos;s own view — the training,
                prediction and applicability-domain lanes below all
                reference this dataset_id; the authoritative frame stays
                server-side (§17.1&apos;s control-plane rule).
              </p>
            </CardContent>
          </Card>
        ) : null}
      </StageHeader>

      <StageHeader
        index={2}
        title="Target & Features"
        hint="read-only, from the registered dataset's own loader policy"
        icon={<Target aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            {selected === null ? (
              <p className="text-sm text-[var(--color-muted-foreground)]">
                Select a dataset above to see its target and feature
                columns.
              </p>
            ) : (
              <div>
                <div className="flex flex-wrap gap-2">
                  <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                    target: {selected.target ?? "none (auto-detected)"}
                  </span>
                  <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                    task: {selected.task ?? "inferred"}
                  </span>
                  <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                    {selected.n_features} numeric features
                  </span>
                </div>
                <FeatureChips title="Feature columns (the training matrix)" columns={selected.feature_names} />
                <p className="mt-3 text-xs text-[var(--color-muted-foreground)]">
                  The target and feature columns come from the
                  registered dataset&apos;s own loader policy
                  (target auto-detection, task inference, non-numeric
                  column dropping) — this frame does not re-derive or
                  override them; the matrix itself never crosses the
                  JSON boundary.
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={3}
        title="Preprocess"
        hint="the training pipeline's own imputation and scaling"
        icon={<Filter aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            <label className="flex items-center gap-2 text-sm text-[var(--color-foreground)]">
              <input
                type="checkbox"
                checked={preprocess}
                onChange={(event) => setPreprocess(event.target.checked)}
                className="h-4 w-4"
              />
              preprocess (imputation + standardization inside the
              fitted pipeline)
            </label>
            <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
              On, the train engine wraps the estimator in its own
              preprocessing pipeline (missing-value imputation and
              scaling fitted on the training folds only); off, the raw
              numeric feature matrix goes straight to the estimator —
              the backend&apos;s own train_model policy either way.
            </p>
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={4}
        title="Configure Split / CV"
        hint="cv folds, hyperparameter search, workers, seed"
        icon={<SlidersHorizontal aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            <div className="flex flex-wrap items-end gap-3">
              <label className="text-xs text-[var(--color-muted-foreground)]">
                cv folds
                <input
                  type="number"
                  min={2}
                  max={10}
                  value={cv}
                  onChange={(event) => setCv(Number(event.target.value))}
                  className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="flex items-center gap-2 text-xs text-[var(--color-muted-foreground)]">
                <input
                  type="checkbox"
                  checked={tune}
                  onChange={(event) => setTune(event.target.checked)}
                  className="h-4 w-4"
                />
                tune (hyperparameter search)
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                n_jobs
                <input
                  type="number"
                  min={-1}
                  max={64}
                  value={nJobs}
                  onChange={(event) => setNJobs(Number(event.target.value))}
                  className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                random state
                <input
                  type="number"
                  value={randomState}
                  onChange={(event) => setRandomState(Number(event.target.value))}
                  className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
            </div>
            <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
              The cv folds (2-10), the hyperparameter search toggle, the
              worker count (-1 = all cores, the backend passes it
              through verbatim) and the seed feed the TrainRequest body
              verbatim — the split itself is the backend&apos;s own
              cross-validation structure over the registered frame.
            </p>
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={5}
        title="Select Algorithm"
        hint="the registry catalog, task-synced"
        icon={<Layers aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            {selectedTask === null ? (
              <p className="text-sm text-[var(--color-muted-foreground)]">
                Select a dataset whose task is one of the registry&apos;s
                two (classification / regression) — the catalog filter
                syncs with the dataset&apos;s own task.
              </p>
            ) : (
              <>
                <p className="text-xs text-[var(--color-muted-foreground)]">
                  task filter: {selectedTask}
                  {catalogBusy ? " — fetching the catalog…" : ""}
                </p>
                {catalogError !== null ? <div className="mt-3"><ErrorBanner message={catalogError} /></div> : null}
                <div className="mt-3 overflow-hidden rounded-lg border border-[var(--color-border)]">
                  <div className="max-h-64 overflow-auto">
                    <table className="w-full border-collapse text-xs">
                      <thead className="sticky top-0 bg-[var(--color-muted)]">
                        <tr>
                          {["Name", "Task", "Available", "Notes"].map((header) => (
                            <th key={header} scope="col" className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]">
                              {header}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {catalog.map((entry) => (
                          <tr
                            key={entry.name}
                            onClick={() => {
                              if (entry.available) {
                                setSelectedModel(entry.name);
                              }
                            }}
                            className={`border-t border-[var(--color-border)] ${
                              entry.available
                                ? `cursor-pointer ${
                                    entry.name === selectedModel
                                      ? "bg-[var(--color-accent)]"
                                      : "bg-[var(--color-card)] hover:bg-[var(--color-accent)]/50"
                                  }`
                                : "bg-[var(--color-card)] opacity-60"
                            }`}
                          >
                            <td className="whitespace-nowrap px-3 py-1.5 font-medium text-[var(--color-foreground)]">{entry.name}</td>
                            <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">{entry.task}</td>
                            <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                              {entry.available ? "yes" : "unavailable"}
                              {entry.requires !== null ? ` (requires ${entry.requires})` : ""}
                            </td>
                            <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">{entry.notes}</td>
                          </tr>
                        ))}
                        {catalog.length === 0 && catalogBusy === false ? (
                          <tr>
                            <td colSpan={4} className="px-3 py-4 text-center text-[var(--color-muted-foreground)]">
                              The registry catalog returned no entries for this task.
                            </td>
                          </tr>
                        ) : null}
                      </tbody>
                    </table>
                  </div>
                </div>
                <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                  One source of truth: the backend&apos;s own
                  list_models projection (name, task, available,
                  requires, notes). Unavailable entries stay visible,
                  disabled — their optional dependency is not built
                  into this offline bundle, and no request fires for
                  them.
                </p>
              </>
            )}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={6}
        title="Configure Hyperparameters"
        hint="estimator parameter overrides as JSON"
        icon={<Braces aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            <label className="block text-sm">
              <span className="mb-1 block font-medium text-[var(--color-foreground)]">
                model_params (JSON object of estimator parameter overrides)
              </span>
              <textarea
                value={modelParamsText}
                onChange={(event) => setModelParamsText(event.target.value)}
                placeholder='{"max_depth": 6} or empty for the registry defaults'
                rows={4}
                spellCheck={false}
                className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 font-mono text-xs"
              />
            </label>
            {modelParamsError !== null ? (
              <p role="alert" className="mt-2 text-xs text-[var(--color-destructive)]">
                {modelParamsError}
              </p>
            ) : null}
            <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
              Validated on submit: the text must parse as a JSON object
              of parameter overrides, or stay empty for the registry&apos;s
              own defaults — invalid JSON never leaves the page.
              With tune on, the registry&apos;s own hyperparameter
              search runs over its default grid; these overrides pin
              the estimator&apos;s fixed parameters either way.
            </p>
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={7}
        title="Train"
        hint="the REAL train_model engine as a managed job"
        icon={<Brain aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-xs text-[var(--color-muted-foreground)]">
                dataset {selectedId ?? "—"} · model {selectedModel || "—"} ·
                task {selectedTask ?? "inferred"} · cv {cv} · tune{" "}
                {tune ? "on" : "off"} · n_jobs {nJobs} · seed {randomState}
              </span>
              <Button
                size="sm"
                disabled={
                  selectedId === null ||
                  selectedModel === "" ||
                  selectedTask === null ||
                  trainBusy
                }
                onClick={() => void handleTrain()}
              >
                <Brain aria-hidden className="h-4 w-4" />
                {trainBusy ? "Submitting…" : "Train the model"}
              </Button>
            </div>
            {trainError !== null ? <div className="mt-3"><ErrorBanner message={trainError} /></div> : null}
            {trainJobId !== null ? (
              <div className="mt-3">
                <JobRunCard jobId={trainJobId} resultLabel="the trained model view" onResult={handleTrainResult} />
              </div>
            ) : null}
            {trainedModel !== null ? (
              <TrainedModelCard model={trainedModel} />
            ) : (
              <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                The train job registers the fitted model in the
                session&apos;s model store and relays its view — name,
                cv scores, best_params, fit_seconds — through the
                authoritative REST result.
              </p>
            )}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={8}
        title="Evaluate — Inspect Predictions"
        hint="the model's predictions over the selected dataset"
        icon={<ClipboardCheck aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            <div className="flex flex-wrap items-end gap-3">
              <label className="text-sm">
                <span className="mb-1 block font-medium text-[var(--color-foreground)]">Trained model</span>
                <select
                  value={selectedTrainedId ?? ""}
                  onChange={(event) => {
                    const modelId = event.target.value;
                    if (modelId !== "") {
                      void selectTrainedModel(modelId);
                    }
                  }}
                  className="w-56 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
                >
                  <option value="">select…</option>
                  {trainedModels.map((model) => (
                    <option key={model.model_id} value={model.model_id}>
                      {model.name} · {model.model_id}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex items-center gap-2 text-xs text-[var(--color-muted-foreground)]">
                <input
                  type="checkbox"
                  checked={predictProba}
                  onChange={(event) => setPredictProba(event.target.checked)}
                  className="h-4 w-4"
                />
                predict_proba
              </label>
              <Button
                size="sm"
                disabled={
                  selectedId === null ||
                  selectedTrainedId === null ||
                  predictBusy
                }
                onClick={() => void handlePredict()}
              >
                <ClipboardCheck aria-hidden className="h-4 w-4" />
                {predictBusy ? "Predicting…" : "Inspect predictions"}
              </Button>
            </div>
            {predictError !== null ? <div className="mt-3"><ErrorBanner message={predictError} /></div> : null}
            {predictResult !== null ? (
              <div className="mt-4">
                <p className="text-xs text-[var(--color-muted-foreground)]">
                  model {predictResult.model_id} over dataset{" "}
                  {selectedId ?? "—"} · {predictResult.n_rows} rows predicted
                </p>
                <div className="mt-2 overflow-hidden rounded-lg border border-[var(--color-border)]">
                  <div className="max-h-72 overflow-auto">
                    <table className="w-full border-collapse text-xs">
                      <thead className="sticky top-0 bg-[var(--color-muted)]">
                        <tr>
                          <th scope="col" className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]">row</th>
                          <th scope="col" className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]">prediction</th>
                        </tr>
                      </thead>
                      <tbody>
                        {predictResult.predictions.slice(0, 20).map((prediction, i) => (
                          <tr key={i} className="border-t border-[var(--color-border)] bg-[var(--color-card)]">
                            <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">{i}</td>
                            <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">{formatValue(prediction)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
                <p className="border-t border-transparent px-0 py-2 text-[11px] text-[var(--color-muted-foreground)]">
                  20 of {predictResult.n_rows} rows shown — the backend
                  keeps the authoritative prediction set; the Export
                  stage registers the full rows as the table artifact.
                </p>
                <EvaluateCharts
                  result={predictResult}
                  modelId={selectedTrainedId ?? predictResult.model_id}
                  datasetId={selectedId}
                />
                {predictResult.probabilities !== null ? (
                  <div className="mt-3">
                    <p className="text-xs font-medium text-[var(--color-foreground)]">
                      Probabilities (predict_proba over the classes)
                    </p>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {predictResult.probabilities.classes.map((cls, i) => (
                        <span key={i} className="rounded bg-[var(--color-muted)] px-1.5 py-0.5 text-[10px] text-[var(--color-muted-foreground)]">
                          class {String(cls)}
                        </span>
                      ))}
                    </div>
                    <div className="mt-2 overflow-hidden rounded-lg border border-[var(--color-border)]">
                      <div className="max-h-48 overflow-auto">
                        <table className="w-full border-collapse text-[11px]">
                          <thead className="sticky top-0 bg-[var(--color-muted)]">
                            <tr>
                              <th scope="col" className="whitespace-nowrap px-2 py-1 text-left font-semibold">row</th>
                              {predictResult.probabilities.classes.slice(0, 6).map((cls, i) => (
                                <th key={i} scope="col" className="whitespace-nowrap px-2 py-1 text-left font-semibold">
                                  p({String(cls)})
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {predictResult.probabilities.rows.slice(0, 20).map((row, i) => (
                              <tr key={i} className="border-t border-[var(--color-border)]">
                                <td className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">{i}</td>
                                {row.slice(0, 6).map((value, j) => (
                                  <td key={j} className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">
                                    {typeof value === "number" ? value.toFixed(4) : formatValue(value)}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                    <p className="mt-1 text-[11px] text-[var(--color-muted-foreground)]">
                      20 of {predictResult.probabilities.rows.length} probability
                      rows × first{" "}
                      {Math.min(6, predictResult.probabilities.classes.length)}{" "}
                      of {predictResult.probabilities.classes.length} classes
                      shown — the authoritative matrices stay in the
                      prediction payload; Export carries them.
                    </p>
                  </div>
                ) : (
                  <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                    No probabilities block in this answer — either the
                    toggle was off, or the estimator does not expose
                    predict_proba (the backend refuses that combination
                    with its own 422 detail).
                  </p>
                )}
              </div>
            ) : (
              <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                The predict lane runs sync in dataset mode over the
                selected dataset&apos;s feature matrix; a model without
                predict_proba answers the backend&apos;s honest refusal
                verbatim in the error banner above.
              </p>
            )}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={9}
        title="Compare"
        hint="the session's trained models by CV mean"
        icon={<GitCompare aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            <div className="flex items-center gap-3">
              <Button size="sm" variant="outline" onClick={() => void refreshTrained()}>
                <GitCompare aria-hidden className="h-4 w-4" />
                Refresh the trained registry
              </Button>
              <span className="text-xs text-[var(--color-muted-foreground)]">
                {trainedModels.length} models trained this session
              </span>
            </div>
            {trainedListError !== null ? <div className="mt-3"><ErrorBanner message={trainedListError} /></div> : null}
            <div className="mt-3 overflow-hidden rounded-lg border border-[var(--color-border)]">
              <div className="max-h-64 overflow-auto">
                <table className="w-full border-collapse text-xs">
                  <thead className="sticky top-0 bg-[var(--color-muted)]">
                    <tr>
                      {["Name", "model_id", "Task", "CV mean", "CV std", "CV scores", "fit s", "train rows", "Created"].map((header) => (
                        <th key={header} scope="col" className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]">
                          {header}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {compareModels.map((model) => (
                      <tr
                        key={model.model_id}
                        onClick={() => void selectTrainedModel(model.model_id)}
                        className={`cursor-pointer border-t border-[var(--color-border)] ${
                          model.model_id === selectedTrainedId
                            ? "bg-[var(--color-accent)]"
                            : "bg-[var(--color-card)] hover:bg-[var(--color-accent)]/50"
                        }`}
                      >
                        <td className="whitespace-nowrap px-3 py-1.5 font-medium text-[var(--color-foreground)]">{model.name}</td>
                        <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">{model.model_id}</td>
                        <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">{model.task}</td>
                        <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                          {model.cv_mean === null ? "—" : model.cv_mean.toFixed(4)}
                        </td>
                        <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                          {model.cv_std === null ? "—" : model.cv_std.toFixed(4)}
                        </td>
                        <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                          {model.cv_scores === null
                            ? "—"
                            : model.cv_scores.map((score) => score.toFixed(3)).join(", ")}
                        </td>
                        <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">{model.fit_seconds.toFixed(2)}</td>
                        <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">{model.n_train_rows}</td>
                        <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">{model.created_at}</td>
                      </tr>
                    ))}
                    {compareModels.length === 0 ? (
                      <tr>
                        <td colSpan={9} className="px-3 py-4 text-center text-[var(--color-muted-foreground)]">
                          No trained models in this session yet — train one in stage 7.
                        </td>
                      </tr>
                    ) : null}
                  </tbody>
                </table>
              </div>
            </div>
            <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
              Sorted by CV mean over the session registry&apos;s own CV
              columns — models without CV (tune off, or the
              semi-supervised registry entries) show — and sort last.
              Selecting a row arms the Evaluate and Applicability Domain
              stages with that model&apos;s view.
            </p>
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={10}
        title="Applicability Domain"
        hint="the bounded column bridge + the bootstrap ensemble"
        icon={<Radar aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            {selectedTrained === null ? (
              <p className="text-sm text-[var(--color-muted-foreground)]">
                Select a trained model in Evaluate or Compare — the AD
                analysis runs over that model&apos;s own training dataset
                and its feature list.
              </p>
            ) : (
              <>
                <div className="flex flex-wrap items-center gap-3">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => void handleAssembleXTest()}
                  >
                    Assemble x_test
                  </Button>
                  <span className="text-xs text-[var(--color-muted-foreground)]">
                    model {selectedTrained.name} · {selectedTrained.features.length}{" "}
                    features · dataset {selectedTrained.dataset_id}
                  </span>
                  {adAssembly !== null ? (
                    <span className="text-xs text-[var(--color-muted-foreground)]">
                      {adAssembly.rows} of {adAssembly.datasetRows} rows
                      assembled from the first ≤500 rows per feature
                      column through the bounded column bridge
                      {adAssembly.truncated ? " (the bridge clipped the dataset's full height)" : ""}
                      {adAssembly.skipped > 0 ? ` · ${adAssembly.skipped} rows skipped for non-numeric values` : ""}
                    </span>
                  ) : null}
                </div>
                {adAssemblyError !== null ? <div className="mt-3"><ErrorBanner message={adAssemblyError} /></div> : null}

                <div className="mt-4 flex flex-wrap items-end gap-3">
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    k
                    <input
                      type="number"
                      min={1}
                      max={50}
                      value={adK}
                      onChange={(event) => setAdK(Number(event.target.value))}
                      className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    />
                  </label>
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    percentile
                    <input
                      type="number"
                      min={50}
                      max={99.9}
                      step={0.5}
                      value={adPercentile}
                      onChange={(event) => setAdPercentile(Number(event.target.value))}
                      className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    />
                  </label>
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    n_models
                    <input
                      type="number"
                      min={1}
                      max={200}
                      value={adNModels}
                      onChange={(event) => setAdNModels(Number(event.target.value))}
                      className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    />
                  </label>
                  <label className="flex items-center gap-2 text-xs text-[var(--color-muted-foreground)]">
                    <input
                      type="checkbox"
                      checked={adUncertainty}
                      onChange={(event) => setAdUncertainty(event.target.checked)}
                      className="h-4 w-4"
                    />
                    bootstrap uncertainty
                  </label>
                  <label className="flex items-center gap-2 text-xs text-[var(--color-muted-foreground)]">
                    <input
                      type="checkbox"
                      checked={adFigure}
                      onChange={(event) => setAdFigure(event.target.checked)}
                      className="h-4 w-4"
                    />
                    figure artifact
                  </label>
                  <Button
                    size="sm"
                    disabled={adXTest === null || adBusy}
                    onClick={() => void handleAnalyze()}
                  >
                    <Radar aria-hidden className="h-4 w-4" />
                    {adBusy ? "Submitting…" : "Run the AD analysis"}
                  </Button>
                </div>
                <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                  The analyze lane runs as a managed job: the kNN
                  distance domain over standardized features, and — with
                  bootstrap uncertainty on — the n_models ensemble whose
                  per-member fits drive the live progress. The figure
                  toggle registers the real plot_applicability_domain
                  artifact. The engine&apos;s own feature cap is{" "}
                  {AD_MAX_FEATURES}; larger models are refused here
                  honestly instead of failing server-side.
                </p>
                {adError !== null ? <div className="mt-3"><ErrorBanner message={adError} /></div> : null}
                {adJobId !== null ? (
                  <div className="mt-3">
                    <JobRunCard jobId={adJobId} resultLabel="the applicability-domain analysis" onResult={handleAnalyzeResult} />
                  </div>
                ) : null}
                {adResult !== null ? (
                  <ADResultCard
                    result={adResult}
                    artifactStatus={adArtifactStatus}
                    onDownload={() => void handleAdArtifactDownload()}
                  />
                ) : null}

                {/* B-4 (part12 Phase 2): the leverage/Williams regression lane */}
                <div className="mt-4 rounded-lg border border-[var(--color-border)] p-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm font-semibold text-[var(--color-foreground)]">
                      Leverage / Williams lane
                    </span>
                    <span className="rounded-md bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
                      regression models only
                    </span>
                  </div>
                  {!isRegressionModel ? (
                    <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                      The backend refuses this lane honestly for a
                      classification model — a classifier&apos;s labels have
                      no residual scale to standardize (the route 422s).
                      Train or select a regression model to arm it.
                    </p>
                  ) : (
                    <>
                      <div className="mt-3 flex flex-wrap items-end gap-3">
                        <label className="text-xs text-[var(--color-muted-foreground)]">
                          residual_limit (± band)
                          <input
                            type="number"
                            min={0.5}
                            max={10}
                            step={0.5}
                            value={williamsResidualLimit}
                            onChange={(event) => setWilliamsResidualLimit(Number(event.target.value))}
                            className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                          />
                        </label>
                        <label className="flex items-center gap-2 text-xs text-[var(--color-muted-foreground)]">
                          <input
                            type="checkbox"
                            checked={williamsFigure}
                            onChange={(event) => setWilliamsFigure(event.target.checked)}
                            className="h-4 w-4"
                          />
                          figure artifact
                        </label>
                        <Button
                          size="sm"
                          variant="outline"
                          disabled={williamsBusy}
                          onClick={() => void handleWilliams()}
                        >
                          {williamsBusy ? "Submitting…" : "Run the Williams analysis"}
                        </Button>
                      </div>
                      <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                        The hat-matrix leverage of every training row
                        against the warning cut-off h* = 3(p+1)/n, with the
                        standardized-residual ±{williamsResidualLimit} band —
                        over{" "}
                        {adXTest !== null
                          ? "the SAME assembled x_test (the query rows' leverage-only extrapolation verdicts)"
                          : "the training set only (assemble x_test above to add the query extrapolation view)"}
                        . The figure toggle registers the real plot_williams
                        artifact.
                      </p>
                      {williamsError !== null ? <div className="mt-3"><ErrorBanner message={williamsError} /></div> : null}
                      {williamsJobId !== null ? (
                        <div className="mt-3">
                          <JobRunCard jobId={williamsJobId} resultLabel="the leverage/Williams analysis" onResult={handleWilliamsResult} />
                        </div>
                      ) : null}
                      {williamsResult !== null ? (
                        <WilliamsResultCard
                          result={williamsResult}
                          artifactStatus={williamsArtifactStatus}
                          onDownload={() => void handleWilliamsArtifactDownload()}
                        />
                      ) : null}
                    </>
                  )}
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={11}
        title="ADMET Profile"
        hint="the vendored admet_ai registry + the physchem panel — sync, per molecule"
        icon={<FlaskConical aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            <p className="text-xs leading-relaxed text-[var(--color-muted-foreground)]">
              The part13 §11.1 lane over the REAL vendored backend
              (ml/admet.py — swansonk14/admet_ai&apos;s own registry): the
              11-property physicochemical panel with the three
              structural-alert counters, the five-axis radial summary on
              the DrugBank percentile grid, and the low-confidence
              warnings. Predicted values are model estimates, never
              experimental facts; the 41 learned ensembles are not
              shipped — the radial axes use the honest offline
              stand-ins.
              {admetRegistry !== null ? (
                <>
                  {" "}Registry: {admetRegistry.endpoints.length}{" "}
                  endpoints across {admetRegistry.categories.length}{" "}
                  categories ({admetRegistry.categories.join(" · ")});
                  {" "}
                  {admetRegistry.low_confidence_endpoints.length}{" "}
                  endpoints below the vendor&apos;s 0.6 quality
                  threshold.
                </>
              ) : null}
            </p>
            <div className="mt-4 grid gap-3 md:grid-cols-[1fr_auto]">
              <label className="text-xs text-[var(--color-muted-foreground)]">
                SMILES (one per line, up to {ADMET_MAX_SMILES})
                <textarea
                  value={admetSmilesText}
                  onChange={(event) => setAdmetSmilesText(event.target.value)}
                  rows={3}
                  spellCheck={false}
                  aria-label="ADMET SMILES batch"
                  className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-2 font-mono text-xs"
                />
              </label>
              <div className="flex items-end">
                <Button
                  size="sm"
                  onClick={() => void handleAdmetScore()}
                  disabled={admetBusy}
                >
                  {admetBusy ? "Scoring…" : "Score ADMET"}
                </Button>
              </div>
            </div>
            {admetError !== null ? (
              <div className="mt-3">
                <ErrorBanner message={admetError} />
              </div>
            ) : null}
            {admetResult !== null ? (
              <>
                <p className="mt-4 text-xs text-[var(--color-foreground)]">
                  {admetResult.n_valid} valid · {admetResult.n_invalid}{" "}
                  invalid (invalid rows report themselves) · provenance{" "}
                  {admetResult.provenance.backend} —{" "}
                  {admetResult.provenance.shipped_ensembles} shipped
                  ensembles (the honest absence, stated)
                </p>
                {admetResult.molecules.map((row) => (
                  <AdmetMoleculeCard key={row.smiles} row={row} />
                ))}
                <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                  {admetResult.provenance.disclaimer}
                </p>
              </>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={12}
        title="Drug–Target Interaction"
        hint="DeepDTA's alphabets + the paired ECFP/AAC featurization — sync, per pair"
        icon={<Dna aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            <p className="text-xs leading-relaxed text-[var(--color-muted-foreground)]">
              The part13 Phase-5 lane over the REAL in-tree backend
              (ml/dta.py — DeepDTA&apos;s own alphabets and rm2 metric,
              the toolkit&apos;s 2048-bit ECFP pairing): the per-pair
              protein composition cards, the kept/dropped row accounting,
              and the ordinary feature matrix any in-tree regressor can
              train on. The DeepDTA CNN and the DeepPurpose/DGL-LifeSci/
              TorchDrug checkpoints are NOT shipped — this route never
              invents affinity predictions.
              {dtaRegistry !== null ? (
                <>
                  {" "}Alphabets: {dtaRegistry.alphabets.protein.n_symbols}{" "}
                  protein symbols ·{" "}
                  {dtaRegistry.alphabets.smiles_isomeric.n_symbols}{" "}
                  isomeric /{" "}
                  {dtaRegistry.alphabets.smiles_canonical.n_symbols}{" "}
                  canonical SMILES symbols; the{" "}
                  {dtaRegistry.ligand_features.n_bits}-bit ECFP +{" "}
                  {dtaRegistry.protein_features.n_base} AAC columns →{" "}
                  {dtaRegistry.paired_width.base} features (with the
                  dipeptide lane: {dtaRegistry.paired_width.with_dipeptides}).
                </>
              ) : null}
            </p>
            <div className="mt-4 grid gap-3 md:grid-cols-[1fr_auto]">
              <label className="text-xs text-[var(--color-muted-foreground)]">
                Pairs — SMILES,PROTEIN[,AFFINITY] (one per line, up to{" "}
                {DTA_MAX_PAIRS})
                <textarea
                  value={dtaPairsText}
                  onChange={(event) => setDtaPairsText(event.target.value)}
                  rows={4}
                  spellCheck={false}
                  aria-label="DTA pairs batch"
                  className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-2 font-mono text-xs"
                />
              </label>
              <div className="flex flex-col items-end gap-2">
                <label className="flex items-center gap-2 text-xs text-[var(--color-muted-foreground)]">
                  <input
                    type="checkbox"
                    checked={dtaDipeptides}
                    onChange={(event) => setDtaDipeptides(event.target.checked)}
                  />
                  dipeptide panel (+401 columns)
                </label>
                <Button
                  size="sm"
                  onClick={() => void handleDtaFeatures()}
                  disabled={dtaBusy}
                >
                  {dtaBusy ? "Featurizing…" : "Featurize pairs"}
                </Button>
              </div>
            </div>
            {dtaError !== null ? (
              <div className="mt-3">
                <ErrorBanner message={dtaError} />
              </div>
            ) : null}
            {dtaResult !== null ? (
              <>
                <p className="mt-4 text-xs text-[var(--color-foreground)]">
                  {dtaResult.n_pairs} pairs → {dtaResult.n_kept} kept ·{" "}
                  {dtaResult.n_dropped} dropped (invalid rows report
                  themselves) · the matrix: {dtaResult.n_features} features ={" "}
                  {dtaResult.n_ligand_features} ECFP +{" "}
                  {dtaResult.n_protein_features} protein columns
                  {dtaResult.affinity !== null
                    ? ` · affinity ${dtaResult.affinity.n} kept values, ${dtaResult.affinity.min}–${dtaResult.affinity.max} (mean ${dtaResult.affinity.mean.toFixed(2)})`
                    : " · no affinity column supplied"}
                </p>
                {dtaResult.rows.map((row) => (
                  <DtaPairCardView
                    key={`${row.index}-${row.smiles}`}
                    row={row}
                  />
                ))}
                {dtaResult.dropped.map((drop) => (
                  <p
                    key={`dropped-${drop.index}`}
                    className="mt-1 text-xs text-[var(--color-muted-foreground)]"
                  >
                    row {drop.index + 1} ({drop.smiles}): {drop.reason}
                  </p>
                ))}
                <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                  provenance {dtaResult.provenance.backend} —{" "}
                  {dtaResult.provenance.shipped_checkpoints} shipped
                  checkpoints (the honest absence, stated) ·{" "}
                  {dtaResult.provenance.disclaimer}
                </p>
              </>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={13}
        title="Drug Repurposing"
        hint="the CaDRReS recommender over the seeded SYNTHETIC screen — ranked candidates, sync"
        icon={<Pill aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            <p className="text-xs leading-relaxed text-[var(--color-muted-foreground)]">
              The part13 Phase-5 lane over the REAL in-tree backend
              (ml/response_model.py — CaDRReS&apos;s matrix-factorisation
              recommender with the paper&apos;s update order and
              stop-before-diverging guard, DeepResponse&apos;s metric contract
              and RepurposingInferenceEngine ranking): the seeded screen,
              the fit, the predicted IC50 matrix, and the ranked
              candidates per profile. The SELFormer/CNN/FiLM hybrid is NOT
              shipped; the screen is SYNTHETIC — never real pharmacology,
              never a clinical recommendation.
              {rpRegistry !== null ? (
                <>
                  {" "}Strategies:{" "}
                  {rpRegistry.split_strategies.within_table.join(" · ")};
                  the metric contract: {rpRegistry.metrics.keys.length} keys
                  at threshold {rpRegistry.metrics.threshold}; the candidate
                  schema {rpRegistry.repurposing.schema.join(" / ")} —{" "}
                  {rpRegistry.repurposing.ordering}.
                </>
              ) : null}
            </p>
            <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
              {RP_FIELDS.map(({ key, label, step }) => (
                <label
                  key={key}
                  className="text-xs text-[var(--color-muted-foreground)]"
                >
                  {label}
                  <input
                    type="number"
                    step={step}
                    value={rpParams[key]}
                    onChange={(event) =>
                      updateRpParam(key, Number(event.target.value))
                    }
                    aria-label={`repurposing ${key}`}
                    className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-1 font-mono text-xs"
                  />
                </label>
              ))}
            </div>
            <div className="mt-3 flex justify-end">
              <Button
                size="sm"
                onClick={() => void handleRepurposingScreen()}
                disabled={rpBusy}
              >
                {rpBusy ? "Fitting…" : "Run the screen"}
              </Button>
            </div>
            {rpError !== null ? (
              <div className="mt-3">
                <ErrorBanner message={rpError} />
              </div>
            ) : null}
            {rpResult !== null ? (
              <>
                <p className="mt-4 text-xs text-[var(--color-foreground)]">
                  {rpResult.screen.n_cells}×{rpResult.screen.n_drugs} screen
                  (SYNTHETIC) · {rpResult.screen.n_observed}/
                  {rpResult.screen.n_entries} observed entries (
                  {(rpResult.screen.observed_fraction * 100).toFixed(0)}%) ·
                  the fit: {rpResult.training.epochs_run}/
                  {rpResult.training.max_epochs} epochs (observed RMSE{" "}
                  {rpResult.training.observed_rmse?.toFixed(3) ?? "n/a"}
                  {rpResult.training.stopped_early
                    ? ", stopped before diverging"
                    : ""}
                  ) · rmse {rpResult.metrics.rmse?.toFixed(3) ?? "null"} ·
                  accuracy {rpResult.metrics.accuracy?.toFixed(3) ?? "null"}
                </p>
                {Object.entries(rpResult.candidates.per_profile).map(
                  ([cellLine, rows]) => (
                    <div key={cellLine} className="mt-3">
                      <p className="text-xs font-medium text-[var(--color-foreground)]">
                        {cellLine} — top {rows.length} of{" "}
                        {rpResult.screen.n_drugs} (lowest predicted IC50 =
                        rank 1)
                      </p>
                      <table className="mt-1 w-full text-xs">
                        <thead>
                          <tr className="text-[var(--color-muted-foreground)]">
                            <th className="py-1 text-left font-normal">rank</th>
                            <th className="text-left font-normal">drug</th>
                            <th className="text-right font-normal">
                              predicted IC50 (model estimate)
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {rows.map((row) => (
                            <tr
                              key={`${cellLine}-${row.drug}`}
                              className="border-t border-[var(--color-border)]"
                            >
                              <td className="py-1">{row.rank}</td>
                              <td>{row.drug}</td>
                              <td className="text-right font-mono">
                                {row.predicted_ic50.toFixed(2)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ),
                )}
                <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                  provenance {rpResult.provenance.backend} —{" "}
                  {rpResult.provenance.shipped_checkpoints} shipped
                  checkpoints (the honest absence, stated) · the served
                  screen: {rpResult.provenance.screen} ·{" "}
                  {rpResult.provenance.disclaimer}
                </p>
              </>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={14}
        title="Cancer Cell-Line Drug Response"
        hint="the bilinear CaDRReS-style model over the seeded SYNTHETIC screen — sensitivity ranking + biomarkers, sync"
        icon={<Activity aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            <p className="text-xs leading-relaxed text-[var(--color-muted-foreground)]">
              The part13 Phase-5 lane over the REAL in-tree backend
              (ml/drugresponse.py — the CaDRReS-style bilinear
              factorisation with its sigmoid link, the per-cell
              sensitivity ranking, the composed W w_d biomarker readout,
              and the cross-cell-line validation folds): the seeded
              screen, the fit, the predicted response-score matrix, the
              metric contract on the observed entries, and the ranked
              drugs per cell profile. No pretrained checkpoints ship;
              the screen is SYNTHETIC — one response scale only, never
              real pharmacology, never a clinical recommendation.
              {drRegistry !== null ? (
                <>
                  {" "}The metric contract: {drRegistry.metrics.keys.length}{" "}
                  keys; the ranking schema{" "}
                  {drRegistry.sensitivity_ranking.schema.join(" / ")} —{" "}
                  {drRegistry.sensitivity_ranking.ordering};{" "}
                  {drRegistry.cross_validation.rule}.
                </>
              ) : null}
            </p>
            <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
              {DR_FIELDS.map(({ key, label, step }) => (
                <label
                  key={key}
                  className="text-xs text-[var(--color-muted-foreground)]"
                >
                  {label}
                  <input
                    type="number"
                    step={step}
                    value={drParams[key]}
                    onChange={(event) =>
                      updateDrParam(key, Number(event.target.value))
                    }
                    aria-label={`drugresponse ${key}`}
                    className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-1 font-mono text-xs"
                  />
                </label>
              ))}
            </div>
            <div className="mt-3 flex justify-end">
              <Button
                size="sm"
                onClick={() => void handleDrugResponseAnalyze()}
                disabled={drBusy}
              >
                {drBusy ? "Fitting…" : "Run the analysis"}
              </Button>
            </div>
            {drError !== null ? (
              <div className="mt-3">
                <ErrorBanner message={drError} />
              </div>
            ) : null}
            {drResult !== null ? (
              <>
                <p className="mt-4 text-xs text-[var(--color-foreground)]">
                  {drResult.screen.n_cells}×{drResult.screen.n_drugs} screen
                  (SYNTHETIC) · {drResult.screen.n_observed}/
                  {drResult.screen.n_entries} observed entries (
                  {(drResult.screen.observed_fraction * 100).toFixed(0)}%) ·
                  the fit: {drResult.training.epochs_run}/
                  {drResult.training.max_epochs} epochs (final loss{" "}
                  {drResult.training.final_loss?.toFixed(4) ?? "n/a"}, pearson
                  r {drResult.training.pearson_r?.toFixed(3) ?? "null"}) · rmse{" "}
                  {drResult.metrics.rmse?.toFixed(3) ?? "null"} ·
                  cross-cell-line CV: {drResult.cross_validation.n_folds}{" "}
                  folds, mean rmse{" "}
                  {drResult.cross_validation.summary.rmse?.toFixed(3) ??
                    "null"}
                </p>
                {Object.entries(drResult.ranking.per_cell).map(
                  ([cellLine, rows]) => (
                    <div key={cellLine} className="mt-3">
                      <p className="text-xs font-medium text-[var(--color-foreground)]">
                        {cellLine} — top {rows.length} of{" "}
                        {drResult.screen.n_drugs} (highest predicted score =
                        rank 1)
                      </p>
                      <table className="mt-1 w-full text-xs">
                        <thead>
                          <tr className="text-[var(--color-muted-foreground)]">
                            <th className="py-1 text-left font-normal">rank</th>
                            <th className="text-left font-normal">drug</th>
                            <th className="text-right font-normal">
                              predicted score (model estimate)
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {rows.map((row) => (
                            <tr
                              key={`${cellLine}-${row.drug}`}
                              className="border-t border-[var(--color-border)]"
                            >
                              <td className="py-1">{row.rank}</td>
                              <td>{row.drug}</td>
                              <td className="text-right font-mono">
                                {row.predicted_score.toFixed(3)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ),
                )}
                {Object.entries(drResult.biomarkers.per_drug).map(
                  ([drug, rows]) => (
                    <div key={drug} className="mt-3">
                      <p className="text-xs font-medium text-[var(--color-foreground)]">
                        {drug} — top {rows.length} biomarker weights (the
                        composed W w_d readout)
                      </p>
                      <table className="mt-1 w-full text-xs">
                        <thead>
                          <tr className="text-[var(--color-muted-foreground)]">
                            <th className="py-1 text-left font-normal">
                              feature
                            </th>
                            <th className="text-right font-normal">
                              weight (logit contribution)
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {rows.map((row) => (
                            <tr
                              key={`${drug}-${row.feature}`}
                              className="border-t border-[var(--color-border)]"
                            >
                              <td className="py-1">{row.feature}</td>
                              <td className="text-right font-mono">
                                {row.weight.toFixed(3)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ),
                )}
                <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                  provenance {drResult.provenance.backend} —{" "}
                  {drResult.provenance.shipped_checkpoints} shipped
                  checkpoints (the honest absence, stated) · the served
                  screen: {drResult.provenance.screen} ·{" "}
                  {drResult.provenance.disclaimer}
                </p>
              </>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={15}
        title="Pharmacogenomics"
        hint="the classic GDSC ANOVA over the seeded SYNTHETIC screen — top associations + volcano + the guarded baseline, sync"
        icon={<Dna aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            <p className="text-xs leading-relaxed text-[var(--color-muted-foreground)]">
              The part13 Phase-5 lane over the REAL in-tree backend
              (ml/pharmacogenomics.py — the classic GDSC ANOVA with
              its design matrix and feasibility gates, the
              Storey-Tibshirani q-values, the volcano verdicts, the
              guarded metric family and the additive mean-effects
              ruler): the seeded screen, the full association table,
              the top drug × feature rows, and the ruler&apos;s
              guarded + normalized scores. No real
              GDSC/CCLE/CTRP/DepMap table ships; the screen is
              SYNTHETIC — one response scale only (the natural-log µM
              scale), never real pharmacology, never a clinical
              recommendation.
              {pgRegistry !== null ? (
                <>
                  {" "}The feasibility gates: a drug needs ≥ 6 non-NA
                  IC50s, a feature ≥ 3 positives AND ≥ 3 negatives;{" "}
                  {pgRegistry.multiple_testing.percent_convention}.
                </>
              ) : null}
            </p>
            <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
              {PG_FIELDS.map(({ key, label, step }) => (
                <label
                  key={key}
                  className="text-xs text-[var(--color-muted-foreground)]"
                >
                  {label}
                  <input
                    type="number"
                    step={step}
                    value={pgParams[key]}
                    onChange={(event) =>
                      updatePgParam(key, Number(event.target.value))
                    }
                    aria-label={`pharmacogenomics ${key}`}
                    className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-1 font-mono text-xs"
                  />
                </label>
              ))}
            </div>
            <div className="mt-3 flex justify-end">
              <Button
                size="sm"
                onClick={() => void handlePharmacogenomicsAnalyze()}
                disabled={pgBusy}
              >
                {pgBusy ? "Analyzing…" : "Run the ANOVA"}
              </Button>
            </div>
            {pgError !== null ? (
              <div className="mt-3">
                <ErrorBanner message={pgError} />
              </div>
            ) : null}
            {pgResult !== null ? (
              <>
                <p className="mt-4 text-xs text-[var(--color-foreground)]">
                  {pgResult.screen.n_cells}×{pgResult.screen.n_drugs} screen
                  (SYNTHETIC) · {pgResult.screen.n_observed}/
                  {pgResult.screen.n_entries} observed entries (
                  {(pgResult.screen.observed_fraction * 100).toFixed(0)}%) ·{" "}
                  {pgResult.associations.n_rows} associations over{" "}
                  {pgResult.associations.n_drugs_tested} drugs ×{" "}
                  {pgResult.associations.n_features_tested} features ·
                  volcano: {pgResult.volcano.n_sensitive} sensitive /{" "}
                  {pgResult.volcano.n_resistant} resistant /{" "}
                  {pgResult.volcano.n_not_significant} not significant
                </p>
                <div className="mt-3">
                  <p className="text-xs font-medium text-[var(--color-foreground)]">
                    Top {pgResult.associations.top.length} associations
                    (statistic over the synthetic screen, sorted by p)
                  </p>
                  <table className="mt-1 w-full text-xs">
                    <thead>
                      <tr className="text-[var(--color-muted-foreground)]">
                        <th className="py-1 text-left font-normal">drug</th>
                        <th className="text-left font-normal">feature</th>
                        <th className="text-right font-normal">n+ / n−</th>
                        <th className="text-right font-normal">
                          Δ mean (ln µM)
                        </th>
                        <th className="text-right font-normal">signed d</th>
                        <th className="text-right font-normal">p</th>
                        <th className="text-right font-normal">q</th>
                        <th className="text-right font-normal">FDR %</th>
                        <th className="text-right font-normal">verdict</th>
                      </tr>
                    </thead>
                    <tbody>
                      {pgResult.associations.top.map((row) => (
                        <tr
                          key={`${row.drug}-${row.feature}`}
                          className="border-t border-[var(--color-border)]"
                        >
                          <td className="py-1">{row.drug}</td>
                          <td>{row.feature}</td>
                          <td className="text-right font-mono">
                            {row.n_pos} / {row.n_neg}
                          </td>
                          <td className="text-right font-mono">
                            {row.delta_mean?.toFixed(3) ?? "null"}
                          </td>
                          <td className="text-right font-mono">
                            {row.volcano_x?.toFixed(3) ?? "null"}
                          </td>
                          <td className="text-right font-mono">
                            {row.pvalue?.toFixed(5) ?? "null"}
                          </td>
                          <td className="text-right font-mono">
                            {row.qvalue?.toFixed(5) ?? "null"}
                          </td>
                          <td className="text-right font-mono">
                            {row.fdr_percent?.toFixed(2) ?? "null"}
                          </td>
                          <td className="text-right">{row.verdict ?? "n/a"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="mt-3 text-xs text-[var(--color-foreground)]">
                  the mean-effects ruler (guarded pearson{" "}
                  {pgResult.evaluation.guarded.pearson?.toFixed(3) ?? "null"}{" "}
                  · spearman{" "}
                  {pgResult.evaluation.guarded.spearman?.toFixed(3) ?? "null"}{" "}
                  · kendall{" "}
                  {pgResult.evaluation.guarded.kendall?.toFixed(3) ?? "null"});
                  normalized beyond the naive drug-mean ruler (pearson{" "}
                  {
                    pgResult.evaluation.normalized_beyond_drug_mean.pearson?.toFixed(
                      3,
                    ) ?? "null"
                  }
                  )
                </p>
                <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                  provenance {pgResult.provenance.backend} —{" "}
                  {pgResult.provenance.shipped_tables} shipped tables,{" "}
                  {pgResult.provenance.shipped_checkpoints} shipped
                  checkpoints (the honest absence, stated) · the served
                  screen: {pgResult.provenance.screen} ·{" "}
                  {pgResult.provenance.disclaimer}
                </p>
              </>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={16}
        title="Multi-objective Optimization"
        hint="the Pareto HPO study over the seeded SYNTHETIC frame — front + crowding + the 2-D indicator, sync"
        icon={<Scale aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            <p className="text-xs leading-relaxed text-[var(--color-muted-foreground)]">
              The part13 §12 lane over the REAL in-tree backend
              (ml/multiobjective.py — the optuna minimize convention
              and the Pareto order, the exact 2-D hypervolume
              staircase, NSGA-II&apos;s crowding, the MOEA/D Tchebycheff
              decomposition driving both samplers): the seeded 80×6
              SYNTHETIC frame, the cross-validated study over the
              sklearn registry model, the non-dominated front with its
              aligned crowding, the per-objective trade-off corners
              and the crowding-pruned shortlist. No real dataset ships
              with this route; every objective value is a
              cross-validated statistic — front members are model
              estimates for triage, never experimental facts.
              {moRegistry !== null ? (
                <>
                  {" "}Every objective is minimized — &apos;score&apos; rides
                  the negated mean CV score; the tunable ledger:{" "}
                  {Object.keys(moRegistry.tunable_objectives.ledger).join(
                    " · ",
                  )}
                  .
                </>
              ) : null}
            </p>
            <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
              <label className="text-xs text-[var(--color-muted-foreground)]">
                model (any sklearn registry name)
                <input
                  type="text"
                  value={moParams.model}
                  onChange={(event) =>
                    setMoParams((prev) => ({
                      ...prev,
                      model: event.target.value,
                    }))
                  }
                  aria-label="multiobjective model"
                  className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-1 font-mono text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                task
                <select
                  value={moParams.task}
                  onChange={(event) =>
                    setMoParams((prev) => ({
                      ...prev,
                      task: event.target.value as MultiobjectiveParams["task"],
                    }))
                  }
                  aria-label="multiobjective task"
                  className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-1 text-xs"
                >
                  <option value="classification">classification</option>
                  <option value="regression">regression</option>
                </select>
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                sampler
                <select
                  value={moParams.sampler}
                  onChange={(event) =>
                    setMoParams((prev) => ({
                      ...prev,
                      sampler: event.target.value as MultiobjectiveParams["sampler"],
                    }))
                  }
                  aria-label="multiobjective sampler"
                  className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-1 text-xs"
                >
                  <option value="tpe">tpe</option>
                  <option value="random">random</option>
                </select>
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                weights
                <select
                  value={moParams.weight_mode}
                  onChange={(event) =>
                    setMoParams((prev) => ({
                      ...prev,
                      weight_mode: event.target.value as MultiobjectiveParams["weight_mode"],
                    }))
                  }
                  aria-label="multiobjective weight_mode"
                  className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-1 text-xs"
                >
                  <option value="spread">spread</option>
                  <option value="random">random</option>
                </select>
              </label>
            </div>
            <div className="mt-2 flex flex-wrap gap-3">
              {MO_OBJECTIVES.map((label) => (
                <label
                  key={label}
                  className="flex items-center gap-1 text-xs text-[var(--color-muted-foreground)]"
                >
                  <input
                    type="checkbox"
                    checked={moParams.objectives.includes(label)}
                    onChange={(event) =>
                      toggleMoObjective(label, event.target.checked)
                    }
                    aria-label={`multiobjective objective ${label}`}
                    className="h-4 w-4"
                  />
                  {label}
                </label>
              ))}
            </div>
            <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
              {MO_FIELDS.map(({ key, label, step }) => (
                <label
                  key={key}
                  className="text-xs text-[var(--color-muted-foreground)]"
                >
                  {label}
                  <input
                    type="number"
                    step={step}
                    value={moParams[key]}
                    onChange={(event) =>
                      updateMoParam(key, Number(event.target.value))
                    }
                    aria-label={`multiobjective ${key}`}
                    className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-1 font-mono text-xs"
                  />
                </label>
              ))}
            </div>
            <div className="mt-3 flex justify-end">
              <Button
                size="sm"
                onClick={() => void handleMultiobjectiveOptimize()}
                disabled={moBusy}
              >
                {moBusy ? "Optimizing…" : "Run the study"}
              </Button>
            </div>
            {moError !== null ? (
              <div className="mt-3">
                <ErrorBanner message={moError} />
              </div>
            ) : null}
            {moResult !== null ? (
              <>
                <p className="mt-4 text-xs text-[var(--color-foreground)]">
                  {moResult.study.n_trials} trials over{" "}
                  {moResult.dataset.n_samples}×
                  {moResult.dataset.n_features} (SYNTHETIC) · front{" "}
                  {moResult.study.n_front} ·{" "}
                  {moResult.study.objective_names.join(" vs ")} · sampler{" "}
                  {moResult.study.sampler} ({moResult.study.weight_mode}{" "}
                  weights) · hypervolume{" "}
                  {moResult.indicator.hypervolume?.toFixed(4) ?? "null"}
                  {moResult.indicator.ref !== null
                    ? ` under ref [${moResult.indicator.ref
                        .map((v) => v.toFixed(3))
                        .join(", ")}]`
                    : ""}{" "}
                  · curve {moResult.indicator.hypervolume_curve.length}{" "}
                  points
                </p>
                {moResult.front.length > 0 ? (
                  <div className="mt-3">
                    <p className="text-xs font-medium text-[var(--color-foreground)]">
                      The non-dominated front ({moResult.front.length}{" "}
                      members — every objective value a labelled
                      statistic over the synthetic frame)
                    </p>
                    <MoTrialTable
                      rows={moResult.front}
                      objectiveNames={moResult.study.objective_names}
                    />
                  </div>
                ) : null}
                <div className="mt-3">
                  <p className="text-xs font-medium text-[var(--color-foreground)]">
                    Trade-off corners (the front member minimizing each
                    objective)
                  </p>
                  {Object.entries(moResult.best_for).map(([name, row]) => (
                    <p
                      key={name}
                      className="mt-1 text-xs text-[var(--color-foreground)]"
                    >
                      best {name}: trial #{row.number} ({" "}
                      {Object.entries(row.params)
                        .map(([k, v]) => `${k}=${v}`)
                        .join(", ")}
                      ) —{" "}
                      {row.values.map((v) => v.toFixed(3)).join(" / ")}
                    </p>
                  ))}
                </div>
                {moResult.shortlist.trials.length > 0 ? (
                  <div className="mt-3">
                    <p className="text-xs font-medium text-[var(--color-foreground)]">
                      Shortlist — keep {moResult.shortlist.keep} by
                      full-front crowding ({" "}
                      {moResult.shortlist.trials.length} rows)
                    </p>
                    <MoTrialTable
                      rows={moResult.shortlist.trials}
                      objectiveNames={moResult.study.objective_names}
                    />
                  </div>
                ) : null}
                <p className="mt-3 text-xs text-[var(--color-muted-foreground)]">
                  provenance {moResult.provenance.backend} —{" "}
                  {moResult.provenance.shipped_checkpoints} shipped
                  checkpoints (the honest absence, stated) · the
                  served dataset: {moResult.dataset.label} ·{" "}
                  {moResult.provenance.disclaimer}
                </p>
              </>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={17}
        title="Workflow Builder"
        hint="§8's construct/save/reproduce surface over the in-tree engine — templates, the editable spec, validation, sync + job runs, the ledger"
        icon={<Braces aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            <p className="text-xs leading-relaxed text-[var(--color-muted-foreground)]">
              The part13 §8 lane over the REAL in-tree engine
              (pharmaml.workflow — the lightweight layer the §8
              assessment itself ordered: no external engine adopted,
              every step delegating to a REAL backend that already
              existed, zero new dependencies; job-mode runs ride the
              EXISTING JobManager for progress, cooperative
              cancellation between nodes, and the SQLite history).
              {wfSteps !== null ? (
                <>
                  {" "}The registry: {wfSteps.steps.length} steps ·{" "}
                  {wfSteps.engine.port_kinds.length} port kinds (
                  {wfSteps.engine.port_kinds.join(" · ")}) — two ports
                  compatible only when their kinds are EQUAL, never an
                  implicit conversion.
                </>
              ) : null}
              {" "}The default dataset is the route&apos;s own seeded
              SYNTHETIC stand-in — every metric a statistic over it,
              never an experimental fact.
            </p>
            {wfTemplate !== null ? (
              <div className="mt-3">
                <p className="text-xs font-medium text-[var(--color-foreground)]">
                  {wfTemplate.name} — {wfTemplate.node_count} nodes (
                  {wfTemplate.nodes.map((n) => n.node_id).join(" → ")})
                </p>
                <p className="mt-1 text-xs text-[var(--color-muted-foreground)]">
                  {wfTemplate.nodes
                    .map(
                      (n) =>
                        `${n.node_id}: ${
                          wfStepTitles.get(n.step_id) ?? n.step_id
                        }`,
                    )
                    .join(" → ")}
                  {typeof wfTemplate.metadata["leakage_note"] === "string" ? (
                    <>
                      {" "}· the honest leakage reorder: the split runs
                      BEFORE feature selection (the toolkit&apos;s own
                      rule, stated in the template&apos;s metadata —
                      §8&apos;s illustrative order lists selection
                      first, the leakage contract wins)
                    </>
                  ) : null}
                </p>
              </div>
            ) : null}
            <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
              <label className="text-xs text-[var(--color-muted-foreground)]">
                template (the saved chains)
                <select
                  value={wfTemplateId}
                  onChange={(event) =>
                    handleWfTemplateChange(event.target.value)
                  }
                  aria-label="workflow template"
                  className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-1 text-xs"
                >
                  {(wfTemplates?.templates ?? []).map((t) => (
                    <option key={t.template_id} value={t.template_id}>
                      {t.template_id} ({t.node_count} nodes)
                    </option>
                  ))}
                </select>
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                mode
                <select
                  value={wfMode}
                  onChange={(event) =>
                    setWfMode(event.target.value as "sync" | "job")
                  }
                  aria-label="workflow run mode"
                  className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-1 text-xs"
                >
                  <option value="sync">sync (bounded)</option>
                  <option value="job">job (the JobManager)</option>
                </select>
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                seed (0–{WORKFLOW_MAX_SEED})
                <input
                  type="number"
                  value={wfSeed}
                  onChange={(event) => setWfSeed(Number(event.target.value))}
                  aria-label="workflow seed"
                  className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-1 font-mono text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                records ({WORKFLOW_MIN_SYNTHETIC}–{WORKFLOW_MAX_SYNTHETIC},{" "}
                SYNTHETIC stand-in)
                <input
                  type="number"
                  value={wfNSynthetic}
                  onChange={(event) =>
                    setWfNSynthetic(Number(event.target.value))
                  }
                  aria-label="workflow synthetic records"
                  className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-1 font-mono text-xs"
                />
              </label>
            </div>
            <label className="mt-3 block text-xs text-[var(--color-muted-foreground)]">
              the workflow spec (the construct canvas — edit the
              nodes, wiring, and per-node params as JSON; the run
              executes the spec as edited)
              <textarea
                value={wfSpecText}
                onChange={(event) => setWfSpecText(event.target.value)}
                aria-label="workflow spec"
                rows={8}
                spellCheck={false}
                className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-2 font-mono text-xs"
              />
            </label>
            <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
              the run bounds (the route&apos;s own): sync ≤{" "}
              {WORKFLOW_SYNC_MAX_NODES} nodes · job ≤{" "}
              {WORKFLOW_JOB_MAX_NODES} nodes — the gates fire
              client-side before any request; the records ride the
              route&apos;s own seeded SYNTHETIC stand-in (never a
              server-side file read)
            </p>
            <div className="mt-3 flex flex-wrap justify-end gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => void handleWorkflowValidate()}
                disabled={wfBusy}
              >
                {wfBusy ? "Working…" : "Validate the spec"}
              </Button>
              <Button
                size="sm"
                onClick={() => void handleWorkflowRun()}
                disabled={wfBusy}
              >
                {wfBusy ? "Running…" : "Run the workflow"}
              </Button>
            </div>
            {wfError !== null ? (
              <div className="mt-3">
                <ErrorBanner message={wfError} />
              </div>
            ) : null}
            {wfValidate !== null ? (
              <div className="mt-3">
                <p className="text-xs font-medium text-[var(--color-foreground)]">
                  {wfValidate.valid
                    ? `valid — ${wfValidate.node_count} nodes · digest ${wfValidate.spec_digest.slice(0, 16)}…`
                    : `invalid — ${wfValidate.errors.length} finding(s)`}
                </p>
                {wfValidate.topological_order !== null ? (
                  <p className="mt-1 text-xs text-[var(--color-muted-foreground)]">
                    the executor&apos;s order:{" "}
                    {wfValidate.topological_order.join(" → ")} · the
                    canonical digest is the spec&apos;s identity — the
                    reproduce anchor that rides every run&apos;s
                    reproducibility block
                  </p>
                ) : null}
                {wfValidate.errors.length > 0 ? (
                  <ul className="mt-1 list-disc pl-4 text-xs text-[var(--color-destructive)]">
                    {wfValidate.errors.map((e) => (
                      <li key={e} className="font-mono">
                        {e}
                      </li>
                    ))}
                  </ul>
                ) : null}
              </div>
            ) : null}
            {wfJobId !== null ? (
              <div className="mt-3">
                <JobRunCard
                  jobId={wfJobId}
                  resultLabel="the workflow run record"
                  onResult={handleWfRunResult}
                />
              </div>
            ) : null}
            {wfRun !== null ? (
              <>
                <p className="mt-4 text-xs text-[var(--color-foreground)]">
                  run {wfRun.run_id} · {wfRun.workflow_name} ·{" "}
                  {wfRun.state} · seed {wfRun.seed} ·{" "}
                  {wfRun.node_counts["COMPLETED"] ?? 0} completed /{" "}
                  {wfRun.node_counts["FAILED"] ?? 0} failed /{" "}
                  {wfRun.node_counts["SKIPPED"] ?? 0} skipped /{" "}
                  {wfRun.node_counts["CANCELLED"] ?? 0} cancelled
                </p>
                <div className="mt-2 max-h-72 overflow-y-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-[var(--color-muted-foreground)]">
                        <th className="py-1 text-left font-normal">node</th>
                        <th className="text-left font-normal">step</th>
                        <th className="text-left font-normal">state</th>
                        <th className="text-left font-normal">
                          error / reason
                        </th>
                        <th className="text-left font-normal">
                          artifacts (kind · digest_kind · lineage)
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {wfRun.nodes.map((n) => (
                        <tr
                          key={n.node_id}
                          className="border-t border-[var(--color-border)] align-top"
                        >
                          <td className="py-1 font-mono">{n.node_id}</td>
                          <td className="font-mono">
                            {wfStepTitles.get(n.step_id) ?? n.step_id}
                          </td>
                          <td className="font-mono">{n.state}</td>
                          <td
                            className="font-mono text-[var(--color-destructive)]"
                          >
                            {n.error ?? n.skipped_reason ?? "—"}
                          </td>
                          <td className="font-mono">
                            {n.artifacts
                              .map(
                                (a) =>
                                  `${a.port} (${a.kind}, ${a.digest_kind}, ${a.lineage.length} upstream)`,
                              )
                              .join("; ") || "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="mt-3 text-xs text-[var(--color-muted-foreground)]">
                  reproducibility — toolkit{" "}
                  {wfRun.reproducibility.toolkit_version} · python{" "}
                  {wfRun.reproducibility.python} (
                  {wfRun.reproducibility.implementation}) ·{" "}
                  {wfRun.reproducibility.platform} · spec digest{" "}
                  {wfRun.reproducibility.spec_digest.slice(0, 16)}… ·{" "}
                  {wfRun.reproducibility.determinism_note}
                </p>
              </>
            ) : null}
            <div className="mt-3">
              <Button
                size="sm"
                variant="outline"
                onClick={() => void handleWorkflowRunsLedger()}
                disabled={wfLedgerBusy}
              >
                {wfLedgerBusy ? "Reading…" : "Refresh the runs ledger"}
              </Button>
            </div>
            {wfRuns !== null ? (
              <div className="mt-3">
                <p className="text-xs font-medium text-[var(--color-foreground)]">
                  The execution history ({wfRuns.runs.length} runs —{" "}
                  {wfRuns.note})
                </p>
                {wfRuns.runs.length > 0 ? (
                  <table className="mt-1 w-full text-xs">
                    <thead>
                      <tr className="text-[var(--color-muted-foreground)]">
                        <th className="py-1 text-left font-normal">run</th>
                        <th className="text-left font-normal">workflow</th>
                        <th className="text-left font-normal">state</th>
                        <th className="text-left font-normal">seed</th>
                        <th className="text-left font-normal">nodes</th>
                      </tr>
                    </thead>
                    <tbody>
                      {wfRuns.runs.map((row) => (
                        <tr
                          key={row.run_id}
                          className="border-t border-[var(--color-border)]"
                        >
                          <td className="py-1 font-mono">{row.run_id}</td>
                          <td className="font-mono">
                            {row.workflow_name}
                          </td>
                          <td className="font-mono">{row.state}</td>
                          <td className="font-mono">{row.seed}</td>
                          <td className="font-mono">
                            {Object.entries(row.node_counts)
                              .map(([state, count]) => `${count} ${state.toLowerCase()}`)
                              .join(" / ") || "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ) : null}
              </div>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={18}
        title="Semi-supervised Learning"
        hint="classification datasets only — the backend's own gate"
        icon={<Network aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            {selected === null ? (
              <p className="text-sm text-[var(--color-muted-foreground)]">
                Select a dataset first — the semi-supervised lane runs
                over the registered frame.
              </p>
            ) : !isClassification ? (
              <p className="text-sm text-[var(--color-muted-foreground)]">
                The backend limits semi-supervised training to
                classification datasets and refuses others with its own
                422 (&quot;semi-supervised training is a classification
                surface&quot;); this dataset&apos;s task is{" "}
                {selected.task ?? "inferred"} — the card stays disabled
                rather than firing a request the backend will refuse.
              </p>
            ) : (
              <>
                <div className="flex flex-wrap items-end gap-3">
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    method
                    <select
                      value={ssMethod}
                      onChange={(event) => setSsMethod(event.target.value as typeof ssMethod)}
                      className="mt-1 w-44 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    >
                      <option value="self_training">self-training</option>
                      <option value="label_spreading">label spreading</option>
                      <option value="label_propagation">label propagation</option>
                    </select>
                  </label>
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    base estimator
                    <select
                      value={ssBaseEstimator}
                      onChange={(event) => setSsBaseEstimator(event.target.value)}
                      className="mt-1 w-44 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    >
                      <option value="">backend default (LogisticRegression)</option>
                      {classificationCatalog.map((entry) => (
                        <option key={entry.name} value={entry.name}>{entry.name}</option>
                      ))}
                    </select>
                  </label>
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    mask_fraction
                    <input
                      type="number"
                      min={0.05}
                      max={0.9}
                      step={0.05}
                      value={ssMaskFractionText}
                      placeholder="marker-native"
                      onChange={(event) => setSsMaskFractionText(event.target.value)}
                      className="mt-1 w-28 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    />
                  </label>
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    threshold
                    <input
                      type="number"
                      min={0.01}
                      max={0.99}
                      step={0.05}
                      value={ssThreshold}
                      onChange={(event) => setSsThreshold(Number(event.target.value))}
                      className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    />
                  </label>
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    k_best
                    <input
                      type="number"
                      min={1}
                      max={100}
                      value={ssKBest}
                      onChange={(event) => setSsKBest(Number(event.target.value))}
                      className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    />
                  </label>
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    max_iter
                    <input
                      type="number"
                      min={1}
                      max={100}
                      value={ssMaxIter}
                      onChange={(event) => setSsMaxIter(Number(event.target.value))}
                      className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    />
                  </label>
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    kernel
                    <select
                      value={ssKernel}
                      onChange={(event) => setSsKernel(event.target.value as typeof ssKernel)}
                      className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    >
                      <option value="knn">knn</option>
                      <option value="rbf">rbf</option>
                    </select>
                  </label>
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    n_neighbors
                    <input
                      type="number"
                      min={1}
                      max={50}
                      value={ssNNeighbors}
                      onChange={(event) => setSsNNeighbors(Number(event.target.value))}
                      className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    />
                  </label>
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    gamma
                    <input
                      type="number"
                      min={0.1}
                      step={1}
                      value={ssGamma}
                      onChange={(event) => setSsGamma(Number(event.target.value))}
                      className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    />
                  </label>
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    random state
                    <input
                      type="number"
                      value={ssRandomState}
                      onChange={(event) => setSsRandomState(Number(event.target.value))}
                      className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    />
                  </label>
                  <Button size="sm" onClick={() => void handleSemiTrain()}>
                    <Network aria-hidden className="h-4 w-4" />
                    Train semi-supervised
                  </Button>
                </div>
                <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                  The unlabeled pool arrives the way the backend
                  understands it: leave mask_fraction empty when the
                  target column already carries the unlabeled marker
                  (-1), or carve a seeded fraction of labels out. The
                  fitted model registers in the session&apos;s model
                  store (task classification), so its predictions run
                  through the single /api/v1/ml/predict path — select
                  it in Evaluate.
                </p>
                {ssError !== null ? <div className="mt-3"><ErrorBanner message={ssError} /></div> : null}
                {ssJobId !== null ? (
                  <div className="mt-3">
                    <JobRunCard jobId={ssJobId} resultLabel="the semi-supervised fit" onResult={handleSemiResult} />
                  </div>
                ) : null}
                {ssResult !== null ? (
                  <div className="mt-4">
                    <div className="flex flex-wrap gap-2">
                      <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                        method: {ssResult.method}
                      </span>
                      <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                        n_labeled: {ssResult.n_labeled}
                      </span>
                      <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                        n_unlabeled: {ssResult.n_unlabeled}
                      </span>
                      <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                        n_pseudo_labeled: {ssResult.n_pseudo_labeled}
                      </span>
                      <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                        fit_seconds: {ssResult.fit_seconds.toFixed(2)}
                      </span>
                      <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                        target: {ssResult.target}
                      </span>
                    </div>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {Object.entries(ssResult.params).map(([key, value]) => (
                        <span key={key} className="rounded bg-[var(--color-muted)] px-1.5 py-0.5 text-[10px] text-[var(--color-muted-foreground)]">
                          {key}: {formatValue(value)}
                        </span>
                      ))}
                    </div>
                    <div className="mt-2 flex flex-wrap gap-1">
                      {ssResult.classes.map((cls, i) => (
                        <span key={i} className="rounded bg-[var(--color-muted)] px-1.5 py-0.5 text-[10px] text-[var(--color-muted-foreground)]">
                          class {formatValue(cls)}
                        </span>
                      ))}
                    </div>
                    {ssResult.confidence_summary !== null ? (
                      <div className="mt-2 flex flex-wrap gap-1">
                        <span className="rounded bg-[var(--color-muted)] px-1.5 py-0.5 text-[10px] text-[var(--color-muted-foreground)]">
                          confidence min: {ssResult.confidence_summary.min.toFixed(4)}
                        </span>
                        <span className="rounded bg-[var(--color-muted)] px-1.5 py-0.5 text-[10px] text-[var(--color-muted-foreground)]">
                          confidence mean: {ssResult.confidence_summary.mean.toFixed(4)}
                        </span>
                        <span className="rounded bg-[var(--color-muted)] px-1.5 py-0.5 text-[10px] text-[var(--color-muted-foreground)]">
                          confidence max: {ssResult.confidence_summary.max.toFixed(4)}
                        </span>
                      </div>
                    ) : null}
                    <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                      labels preview (first {ssResult.labels_preview.length} of{" "}
                      {ssResult.n_labels}):{" "}
                      {ssResult.labels_preview.map((label) => formatValue(label)).join(", ")}
                      {ssResult.model_id !== null
                        ? ` — registered as ${ssResult.model_id}; predictions run through /api/v1/ml/predict`
                        : " — this method fits without a scikit-learn predict surface; no model registered"}
                    </p>
                  </div>
                ) : null}

                {/* stage 11b (r303): the held-out evaluation card — the
                    tested-but-non-GUI /evaluate route's first product
                    consumer */}
                <div className="mt-4 rounded-lg border border-[var(--color-border)] p-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm font-semibold text-[var(--color-foreground)]">
                      Held-out evaluation
                    </span>
                    <span className="rounded-md bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
                      the real evaluate_semi_supervised engine
                    </span>
                  </div>
                  <div className="mt-3 flex flex-wrap items-center gap-3">
                    <span className="text-xs text-[var(--color-muted-foreground)]">
                      method {ssMethod} · 3 seeded repeats · test_size 0.25
                      · random_state 42
                    </span>
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={ssEvalBusy}
                      onClick={() => void handleSemiEvaluate()}
                    >
                      {ssEvalBusy ? "Submitting…" : "Evaluate vs the labelled-only baseline"}
                    </Button>
                  </div>
                  <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                    The semi-supervised method&apos;s held-out accuracy/F1
                    against the labelled-only baseline over seeded
                    repeats of a test split — the honest evidence the
                    unlabeled pool helped or did not.
                  </p>
                  {ssEvalError !== null ? <div className="mt-3"><ErrorBanner message={ssEvalError} /></div> : null}
                  {ssEvalJobId !== null ? (
                    <div className="mt-3">
                      <JobRunCard jobId={ssEvalJobId} resultLabel="the held-out semi-supervised evaluation" onResult={handleSemiEvaluateResult} />
                    </div>
                  ) : null}
                  {ssEvalResult !== null ? (
                    <div className="mt-3">
                      <div className="flex flex-wrap gap-2">
                        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                          held-out accuracy:{" "}
                          {ssEvalResult.heldout_accuracy === null ? "—" : ssEvalResult.heldout_accuracy.toFixed(4)}
                        </span>
                        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                          held-out F1 (macro):{" "}
                          {ssEvalResult.heldout_f1_macro === null ? "—" : ssEvalResult.heldout_f1_macro.toFixed(4)}
                        </span>
                        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                          baseline accuracy:{" "}
                          {ssEvalResult.baseline_accuracy === null ? "—" : ssEvalResult.baseline_accuracy.toFixed(4)}
                        </span>
                        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                          baseline F1 (macro):{" "}
                          {ssEvalResult.baseline_f1_macro === null ? "—" : ssEvalResult.baseline_f1_macro.toFixed(4)}
                        </span>
                        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                          delta vs baseline:{" "}
                          {ssEvalResult.delta_vs_baseline === null
                            ? "—"
                            : `${ssEvalResult.delta_vs_baseline >= 0 ? "+" : ""}${ssEvalResult.delta_vs_baseline.toFixed(4)}`}
                        </span>
                        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                          repeats: {ssEvalResult.n_repeats}
                        </span>
                      </div>
                      {ssEvalResult.delta_vs_baseline !== null && ssEvalResult.delta_vs_baseline < 0 ? (
                        <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                          A negative delta is a result, not a failure — the
                          unlabeled pool did not help this method on this
                          split; the seeded repeats make the number
                          reproducible, not a promise.
                        </p>
                      ) : null}
                    </div>
                  ) : null}
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={19}
        title="Explainability & Uncertainty"
        hint="the real SHAP lane + the AD bootstrap's uncertainty surface"
        icon={<Info aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            {selectedTrained === null ? (
              <p className="text-sm text-[var(--color-muted-foreground)]">
                Select a trained model in Evaluate or Compare — the SHAP
                explanation runs over that model&apos;s own training
                dataset&apos;s feature matrix (a bounded, explicitly
                reported sample).
              </p>
            ) : (
              <>
                <div className="flex flex-wrap items-end gap-3">
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    max rows
                    <input
                      type="number"
                      min={1}
                      max={EXPLAIN_MAX_ROWS}
                      value={exMaxRows}
                      onChange={(event) => setExMaxRows(Number(event.target.value))}
                      className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    />
                  </label>
                  <label className="flex items-center gap-2 text-xs text-[var(--color-muted-foreground)]">
                    <input
                      type="checkbox"
                      checked={exFigure}
                      onChange={(event) => setExFigure(event.target.checked)}
                      className="h-4 w-4"
                    />
                    figure artifacts
                  </label>
                  <Button
                    size="sm"
                    disabled={exBusy}
                    onClick={() => void handleExplain()}
                  >
                    <Info aria-hidden className="h-4 w-4" />
                    {exBusy ? "Explaining…" : "Run the SHAP explanation"}
                  </Button>
                  <span className="text-xs text-[var(--color-muted-foreground)]">
                    model {selectedTrained.name} · dataset{" "}
                    {selectedTrained.dataset_id}
                  </span>
                </div>
                <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                  The explain lane rides POST /api/v1/ml/explain — the
                  real pharmaml.ml.explain.shap_summary engine (the
                  tree-family SHAP lane) over the first max_rows rows of
                  the model&apos;s dataset: the per-feature mean |SHAP|
                  ranking, the bounded beeswarm preview, and the
                  optional importance + beeswarm figure artifacts
                  (digest-verified downloads below). Non-tree
                  estimators are refused honestly with the engine&apos;s
                  own message naming the permutation-importance
                  sibling. The uncertainty surface that also runs today
                  is the Applicability Domain stage&apos;s bootstrap
                  ensemble (stage 10): per-row prediction_sd, the
                  uncertainty_threshold cut and the reliable verdict.
                </p>
                {exError !== null ? <div className="mt-3"><ErrorBanner message={exError} /></div> : null}
                {exResult !== null ? (
                  <ExplainResultCard
                    result={exResult}
                    artifactStatuses={exArtifactStatuses}
                    onDownload={(artifactId) => void handleExplainArtifactDownload(artifactId)}
                  />
                ) : null}
              </>
            )}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={20}
        title="Export"
        hint="a registered table artifact + a digest-verified download"
        icon={<ArrowDownToLine aria-hidden className="h-4 w-4" />}
      >
        <Card>
          <CardContent>
            <div className="flex flex-wrap items-end gap-3">
              <label className="text-xs text-[var(--color-muted-foreground)]">
                format
                <select
                  value={exportFormat}
                  onChange={(event) => setExportFormat(event.target.value as typeof exportFormat)}
                  className="mt-1 w-28 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                >
                  <option value="csv">CSV</option>
                  <option value="tsv">TSV</option>
                  <option value="xlsx">XLSX</option>
                </select>
              </label>
              <Button size="sm" disabled={predictResult === null || exportBusy} onClick={() => void handleExport()}>
                {exportBusy ? "Exporting…" : "Export the predictions"}
              </Button>
              {exportArtifact !== null ? (
                <Button size="sm" variant="outline" onClick={() => void handleDownload()}>
                  <ArrowDownToLine aria-hidden className="h-4 w-4" />
                  Download {exportArtifact.filename}
                </Button>
              ) : null}
            </div>
            <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
              Exports the LAST predictions block from Evaluate — the
              prediction column plus one probability column per class —
              as a real registered table artifact (POST
              /api/v1/visualization/table), then downloads it through
              the token-header lane with the served digest verified.
            </p>
            {exportError !== null ? <div className="mt-3"><ErrorBanner message={exportError} /></div> : null}
            {exportArtifact !== null ? (
              <p className="mt-3 text-xs text-[var(--color-muted-foreground)]">
                artifact {exportArtifact.artifact_id} ·{" "}
                {exportArtifact.size_bytes} bytes · sha256{" "}
                {exportArtifact.sha256.slice(0, 16)}… — the download
                rides the token-header lane and verifies the served
                digest.
              </p>
            ) : null}
            {digestStatus !== null ? (
              <p className="mt-2 text-xs font-medium text-[var(--color-foreground)]">{digestStatus}</p>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>
    </div>
  );
}

/** Feature chips — the training matrix's own column list. */
function FeatureChips({
  title,
  columns,
}: {
  title: string;
  columns: readonly string[];
}) {
  return (
    <div className="mt-3">
      <p className="text-xs font-semibold text-[var(--color-foreground)]">
        {title} ({columns.length})
      </p>
      <div className="mt-2 flex flex-wrap gap-1">
        {columns.length === 0 ? (
          <span className="text-[11px] text-[var(--color-muted-foreground)]">none</span>
        ) : (
          columns.slice(0, 24).map((name) => (
            <span
              key={name}
              className="rounded bg-[var(--color-muted)] px-1.5 py-0.5 text-[10px] text-[var(--color-muted-foreground)]"
            >
              {name}
            </span>
          ))
        )}
        {columns.length > 24 ? (
          <span className="text-[10px] text-[var(--color-muted-foreground)]">
            +{columns.length - 24} more
          </span>
        ) : null}
      </div>
    </div>
  );
}

/** The trained model card — the train job's relayed result view. */
function TrainedModelCard({ model }: { model: TrainedModelView }) {
  return (
    <div className="mt-4 rounded-lg border border-[var(--color-border)] p-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-medium text-[var(--color-foreground)]">{model.name}</span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          {model.task}
        </span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          {model.model_id}
        </span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          {model.n_train_rows} train rows
        </span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          fit {model.fit_seconds.toFixed(2)} s
        </span>
      </div>
      {model.cv_scores !== null && model.cv_scores.length > 0 ? (
        <div className="mt-3 overflow-hidden rounded-lg border border-[var(--color-border)]">
          <table className="w-full border-collapse text-[11px]">
            <thead className="bg-[var(--color-muted)]">
              <tr>
                <th scope="col" className="whitespace-nowrap px-2 py-1 text-left font-semibold text-[var(--color-foreground)]">fold</th>
                {model.cv_scores?.map((_, i) => (
                  <th key={i} scope="col" className="whitespace-nowrap px-2 py-1 text-left font-semibold text-[var(--color-foreground)]">{i}</th>
                ))}
                <th scope="col" className="whitespace-nowrap px-2 py-1 text-left font-semibold text-[var(--color-foreground)]">mean ± std</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-t border-[var(--color-border)] bg-[var(--color-card)]">
                <td className="whitespace-nowrap px-2 py-1 font-medium text-[var(--color-foreground)]">score</td>
                {model.cv_scores?.map((score, i) => (
                  <td key={i} className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">{score.toFixed(4)}</td>
                ))}
                <td className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">
                  {model.cv_mean === null ? "—" : model.cv_mean.toFixed(4)} ±{" "}
                  {model.cv_std === null ? "—" : model.cv_std.toFixed(4)}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      ) : (
        <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
          No CV scores in this view — the registry entry carries no
          cross-validation summary (they show as — in Compare).
        </p>
      )}
      <div className="mt-3 flex flex-wrap gap-1">
        {Object.entries(model.best_params).map(([key, value]) => (
          <span key={key} className="rounded bg-[var(--color-muted)] px-1.5 py-0.5 text-[10px] text-[var(--color-muted-foreground)]">
            {key}: {formatValue(value)}
          </span>
        ))}
      </div>
      <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
        The authoritative REST result of the train job, parsed at
        runtime — the fitted estimator itself stays server-side in the
        session model store.
      </p>
    </div>
  );
}

/**
 * part11 §22.2/§22.5 — the Evaluate stage's interactive Plotly lanes.
 *
 * Regression answers get the parity scatter (actual vs predicted over
 * the evaluation block's aligned sample, the identity line for
 * reference); classification answers get the confusion-matrix heatmap.
 * Every point carries the row's own metadata (`customdata`); a click
 * opens the sample inspector below the chart — the real row's name,
 * SMILES when the dataset carried structures, actual, predicted, and
 * residual, never a plot position or array index.
 */
function EvaluateCharts({
  result,
  modelId,
  datasetId,
}: {
  result: PredictResponse;
  modelId: string;
  datasetId: string | null;
}) {
  const evaluation = result.evaluation;
  const [selected, setSelected] = useState<EvaluationRow | null>(null);
  const numericRows = useMemo(
    () =>
      evaluation === null
        ? []
        : evaluation.rows.filter(
            (row) =>
              typeof row.actual === "number" && typeof row.predicted === "number",
          ),
    [evaluation],
  );

  if (evaluation === null) {
    return (
      <p className="mt-3 text-xs text-[var(--color-muted-foreground)]">
        No evaluation block in this answer — inline-row predictions carry
        no aligned actuals to chart (the dataset-mode lane plots them).
      </p>
    );
  }

  const isRegression =
    evaluation.task === "regression" && numericRows.length >= 2;
  const hasClasses =
    evaluation.task === "classification" && evaluation.rows.length > 0;

  return (
    <div className="mt-4">
      <p className="mb-2 text-xs font-medium text-[var(--color-foreground)]">
        Interactive chart (Plotly) — {isRegression
          ? "parity: actual vs predicted"
          : "confusion matrix: actual vs predicted class"}
        {evaluation.n_rows_available !== evaluation.n_rows_evaluated
          ? ` · ${evaluation.n_rows_evaluated} of ${evaluation.n_rows_available} rows plotted (the route's own sample bound)`
          : ` · ${evaluation.n_rows_evaluated} rows plotted`}
      </p>
      {isRegression ? (
        <ParityChart rows={numericRows} onRowSelect={setSelected} />
      ) : hasClasses ? (
        <ConfusionChart rows={evaluation.rows} />
      ) : (
        <p className="rounded-md border border-dashed border-[var(--color-border)] px-3 py-4 text-xs text-[var(--color-muted-foreground)]">
          No plottable numeric pairs in the evaluation sample — the parity
          lane needs numeric actual/predicted values.
        </p>
      )}
      <EvalRowInspector
        row={selected}
        target={evaluation.target}
        task={evaluation.task}
        modelId={modelId}
        datasetId={datasetId}
        onClear={() => setSelected(null)}
      />
    </div>
  );
}

/** The regression parity scatter with the identity reference line. */
function ParityChart({
  rows,
  onRowSelect,
}: {
  rows: readonly {
    readonly name: unknown;
    readonly actual: unknown;
    readonly predicted: unknown;
    readonly residual: number | null;
    readonly smiles: string | null;
  }[];
  onRowSelect: (row: EvaluationRow) => void;
}) {
  const traces = useMemo<readonly PlotTrace[]>(() => {
    const xs = rows.map((row) => row.actual as number);
    const ys = rows.map((row) => row.predicted as number);
    const lo = Math.min(...xs, ...ys);
    const hi = Math.max(...xs, ...ys);
    const pad = (hi - lo) * 0.05 || 0.5;
    const x0 = lo - pad;
    const x1 = hi + pad;
    return [
      {
        type: "scatter",
        mode: "markers",
        name: "samples",
        x: xs,
        y: ys,
        customdata: rows.map((row) => [
          row.name,
          row.smiles ?? "—",
          row.actual,
          row.predicted,
          row.residual ?? "—",
        ]),
        hovertemplate:
          "row %{customdata[0]}<br>actual %{customdata[2]:.4g}" +
          "<br>predicted %{customdata[3]:.4g}" +
          "<br>residual %{customdata[4]:.4g}" +
          "<br>%{customdata[1]}<extra></extra>",
        marker: { color: "var(--chart-1)", size: 7, opacity: 0.85 },
      },
      {
        type: "scatter",
        mode: "lines",
        name: "y = x",
        x: [x0, x1],
        y: [x0, x1],
        showlegend: false,
        hoverinfo: "skip",
        line: { color: "var(--chart-5)", width: 1, dash: "dot" },
      },
    ];
  }, [rows]);
  const layout = useMemo(
    () => ({
      xaxis: { title: { text: "actual" }, scaleanchor: "y", scaleratio: 1 },
      yaxis: { title: { text: "predicted" } },
      margin: { l: 56, r: 12, t: 8, b: 40 },
    }),
    [],
  );
  return (
    <PlotlyFigure
      data={traces}
      layout={layout}
      height={300}
      ariaLabel="the actual-versus-predicted parity scatter over the evaluation sample"
      emptyHint="No numeric actual/predicted pairs to plot."
      onPointSelect={(point) => {
        const index = point.pointIndex ?? null;
        if (index === null || index >= rows.length) return;
        onRowSelect(rows[index] as EvaluationRow);
      }}
    />
  );
}

/** The classification confusion-matrix heatmap over the sample rows. */
function ConfusionChart({ rows }: { rows: readonly EvaluationRow[] }) {
  const { traces, layout, classes } = useMemo(() => {
    const classSet = new Set<string>();
    for (const row of rows) {
      classSet.add(String(row.actual));
      classSet.add(String(row.predicted));
    }
    const classes = Array.from(classSet).sort();
    const index = new Map(classes.map((cls, i) => [cls, i]));
    const z = classes.map(() => classes.map(() => 0));
    for (const row of rows) {
      const i = index.get(String(row.actual));
      const j = index.get(String(row.predicted));
      if (i === undefined || j === undefined) continue;
      z[i]![j]! += 1;
    }
    const annotations = [];
    for (let i = 0; i < classes.length; i++) {
      for (let j = 0; j < classes.length; j++) {
        annotations.push({
          x: classes[j],
          y: classes[i],
          text: String(z[i]![j]!),
          showarrow: false,
          font: { size: 11 },
        });
      }
    }
    const traces: readonly PlotTrace[] = [
      {
        type: "heatmap",
        z,
        x: classes,
        y: classes,
        hovertemplate:
          "actual %{y}<br>predicted %{x}<br>count %{z}<extra></extra>",
        colorscale: [
          [0, "rgba(47, 125, 95, 0.06)"],
          [1, "rgba(47, 125, 95, 0.92)"],
        ],
        showscale: false,
      },
    ];
    return {
      traces,
      layout: {
        xaxis: { title: { text: "predicted class" }, constrain: "domain" },
        yaxis: { title: { text: "actual class" }, autorange: "reversed" },
        annotations,
        margin: { l: 64, r: 12, t: 8, b: 48 },
      },
      classes,
    };
  }, [rows]);
  return (
    <PlotlyFigure
      data={traces}
      layout={layout}
      height={300}
      ariaLabel={`the confusion matrix heatmap over ${rows.length} evaluation rows and ${classes.length} classes`}
      emptyHint="No class pairs to chart."
    />
  );
}

/** §22.5's sample inspector — the clicked row's own metadata. */
function EvalRowInspector({
  row,
  target,
  task,
  modelId,
  datasetId,
  onClear,
}: {
  row: EvaluationRow | null;
  target: string;
  task: string;
  modelId: string;
  datasetId: string | null;
  onClear: () => void;
}) {
  if (row === null) {
    return (
      <p className="mt-2 text-[11px] text-[var(--color-muted-foreground)]">
        Click a point in the parity chart to inspect that sample&apos;s own
        metadata (row identity, SMILES when the dataset carried them,
        actual, predicted, residual).
      </p>
    );
  }
  const fields: readonly [string, string][] = [
    ["row", formatValue(row.name)],
    ["SMILES", row.smiles ?? "—"],
    [`actual ${target}`, formatValue(row.actual)],
    ["predicted", formatValue(row.predicted)],
    ["residual", row.residual === null ? "—" : row.residual.toFixed(4)],
    ["task", task],
    ["model", modelId],
    ["dataset", datasetId ?? "—"],
  ];
  return (
    <div className="mt-3 rounded-lg border border-[var(--color-border)] bg-[var(--color-muted)] p-3">
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium text-[var(--color-foreground)]">
          Selected sample (the point&apos;s own metadata)
        </p>
        <Button size="sm" variant="outline" onClick={onClear}>
          Clear
        </Button>
      </div>
      <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 sm:grid-cols-4">
        {fields.map(([label, value]) => (
          <div key={label} className="min-w-0">
            <dt className="text-[10px] uppercase tracking-wide text-[var(--color-muted-foreground)]">
              {label}
            </dt>
            <dd className="truncate text-[11px] font-medium text-[var(--color-foreground)]">
              {value}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/**
 * §24.2.B — the query-distance scatter: every query compound by its
 * kNN distance (x) against the predicted value (y), in-domain and
 * out-of-domain as distinct categories, with the analysis&apos;s own
 * distance threshold drawn as the reference line. §24.2.C rides the
 * same frame when the bootstrap uncertainty ran: the prediction_sd
 * values with their threshold.
 */
function ADQueryScatter({
  analysis,
}: {
  analysis: ADAnalyzeResult["analysis"];
}) {
  const frame = analysis.frame;
  const hasUncertainty = frame.some(
    (row) => row.prediction_sd !== null && row.knn_distance !== null,
  );
  const domainTraces = useMemo<readonly PlotTrace[]>(() => {
    const inside = frame.filter((row) => row.inside_ad === true);
    const outside = frame.filter((row) => row.inside_ad !== true);
    const point = (row: ADFrameRow): readonly unknown[] => [
      row.name,
      row.inside_ad === true ? "in-domain" : "out-of-domain",
      row.low_uncertainty === null
        ? "—"
        : row.low_uncertainty === true
          ? "low uncertainty"
          : "high uncertainty",
      row.reliable === null ? "—" : row.reliable === true ? "reliable" : "not reliable",
    ];
    return [
      {
        type: "scatter",
        mode: "markers",
        name: "in-domain",
        x: inside.map((row) => row.knn_distance),
        y: inside.map((row) => row.predicted),
        customdata: inside.map(point),
        hovertemplate:
          "%{customdata[0]}<br>kNN distance %{x:.4g}" +
          "<br>predicted %{y:.4g}<br>%{customdata[1]} · %{customdata[2]}" +
          "<br>%{customdata[3]}<extra></extra>",
        marker: { color: "var(--chart-1)", size: 7, opacity: 0.85 },
      },
      {
        type: "scatter",
        mode: "markers",
        name: "out-of-domain",
        x: outside.map((row) => row.knn_distance),
        y: outside.map((row) => row.predicted),
        customdata: outside.map(point),
        hovertemplate:
          "%{customdata[0]}<br>kNN distance %{x:.4g}" +
          "<br>predicted %{y:.4g}<br>%{customdata[1]} · %{customdata[2]}" +
          "<br>%{customdata[3]}<extra></extra>",
        marker: { color: "var(--chart-3)", size: 7, opacity: 0.85 },
      },
    ];
  }, [frame]);
  const domainLayout = useMemo(
    () => ({
      xaxis: { title: { text: "kNN distance to the training domain" } },
      yaxis: { title: { text: "predicted value" } },
      shapes: [
        {
          type: "line",
          x0: analysis.distance_threshold,
          x1: analysis.distance_threshold,
          y0: 0,
          y1: 1,
          yref: "paper",
          line: { color: "var(--chart-3)", width: 1, dash: "dash" },
        },
      ],
      annotations: [
        {
          x: analysis.distance_threshold,
          y: 1,
          yref: "paper",
          text: "distance threshold",
          showarrow: false,
          yanchor: "bottom",
          font: { size: 10 },
        },
      ],
      margin: { l: 56, r: 12, t: 8, b: 40 },
    }),
    [analysis.distance_threshold],
  );

  const sdTraces = useMemo<readonly PlotTrace[]>(() => {
    const rows = frame.filter(
      (row) => row.prediction_sd !== null && row.knn_distance !== null,
    );
    return [
      {
        type: "scatter",
        mode: "markers",
        name: "prediction sd",
        x: rows.map((row) => row.knn_distance),
        y: rows.map((row) => row.prediction_sd),
        customdata: rows.map((row) => [row.name]),
        hovertemplate:
          "%{customdata[0]}<br>kNN distance %{x:.4g}" +
          "<br>prediction sd %{y:.4g}<extra></extra>",
        marker: { color: "var(--chart-2)", size: 7, opacity: 0.85 },
      },
    ];
  }, [frame]);
  const sdLayout = useMemo(
    () => ({
      xaxis: { title: { text: "kNN distance to the training domain" } },
      yaxis: { title: { text: "prediction sd (bootstrap)" } },
      shapes:
        analysis.uncertainty_threshold === null
          ? []
          : [
              {
                type: "line",
                x0: 0,
                x1: 1,
                xref: "paper",
                y0: analysis.uncertainty_threshold,
                y1: analysis.uncertainty_threshold,
                line: { color: "var(--chart-3)", width: 1, dash: "dash" },
              },
            ],
      annotations:
        analysis.uncertainty_threshold === null
          ? []
          : [
              {
                x: 1,
                xref: "paper",
                y: analysis.uncertainty_threshold,
                text: "uncertainty threshold",
                showarrow: false,
                xanchor: "right",
                yanchor: "bottom",
                font: { size: 10 },
              },
            ],
      margin: { l: 56, r: 12, t: 8, b: 40 },
    }),
    [analysis.uncertainty_threshold],
  );

  if (frame.length === 0) {
    return null;
  }
  return (
    <div className="mt-3">
      <p className="mb-2 text-xs font-medium text-[var(--color-foreground)]">
        Query-distance view (Plotly) — the §24.2.B scatter
        {hasUncertainty ? " with the §24.2.C uncertainty panel" : ""}
      </p>
      <PlotlyFigure
        data={domainTraces}
        layout={domainLayout}
        height={260}
        ariaLabel="the query compounds by kNN distance versus predicted value, in-domain and out-of-domain with the distance threshold"
      />
      {hasUncertainty ? (
        <div className="mt-3">
          <p className="mb-2 text-xs font-medium text-[var(--color-foreground)]">
            Prediction uncertainty (bootstrap sd) with the threshold
          </p>
          <PlotlyFigure
            data={sdTraces}
            layout={sdLayout}
            height={220}
            ariaLabel="the bootstrap prediction standard deviation by kNN distance with the uncertainty threshold"
          />
        </div>
      ) : null}
    </div>
  );
}

/**
 * §22.2's feature-importance form — the mean |SHAP| ranking as the
 * real horizontal Plotly bar chart over the top-ranked features (the
 * table below stays the full inventory; the chart is the visual over
 * the head of the same ranking, never a different number).
 */
function ShapRankingBar({
  ranking,
}: {
  ranking: readonly { readonly feature: string; readonly mean_abs_shap: number }[];
}) {
  const rows = ranking.slice(0, 12);
  const traces = useMemo<readonly PlotTrace[]>(
    () => [
      {
        type: "bar",
        orientation: "h",
        x: rows.map((row) => row.mean_abs_shap),
        y: rows.map((row) => row.feature),
        customdata: rows.map((row) => [row.feature]),
        hovertemplate:
          "%{customdata[0]}<br>mean |SHAP| %{x:.4g}<extra></extra>",
        marker: { color: "var(--chart-1)", opacity: 0.9 },
        showlegend: false,
      },
    ],
    [rows],
  );
  const layout = useMemo(
    () => ({
      xaxis: { title: { text: "mean |SHAP|" } },
      yaxis: { autorange: "reversed", tickfont: { size: 10 } },
      margin: { l: 120, r: 12, t: 8, b: 40 },
    }),
    [],
  );
  if (rows.length === 0) {
    return null;
  }
  const height = Math.max(180, rows.length * 22 + 60);
  return (
    <div className="mt-3">
      <p className="mb-2 text-xs font-medium text-[var(--color-foreground)]">
        Mean |SHAP| ranking (Plotly) — top {rows.length} of{" "}
        {ranking.length} features
      </p>
      <PlotlyFigure
        data={traces}
        layout={layout}
        height={height}
        ariaLabel={`the mean absolute SHAP value bar chart over the top ${rows.length} of ${ranking.length} ranked features`}
      />
    </div>
  );
}

/** The AD analysis result card — every key as it arrived. */
function ADResultCard({
  result,
  artifactStatus,
  onDownload,
}: {
  result: ADAnalyzeResult;
  artifactStatus: string | null;
  onDownload: () => void;
}) {
  const analysis = result.analysis;
  const ratio =
    analysis.n_test > 0 ? analysis.inside_ad / analysis.n_test : null;
  return (
    <div className="mt-4 rounded-lg border border-[var(--color-border)] p-4">
      <div className="flex flex-wrap gap-2">
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          inside_ad: {analysis.inside_ad}/{analysis.n_test}
          {ratio !== null ? ` (${(ratio * 100).toFixed(1)}%)` : ""}
        </span>
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          distance_threshold: {analysis.distance_threshold.toFixed(4)}
        </span>
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          k: {analysis.k} · percentile: {analysis.percentile}
        </span>
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          n_train: {analysis.n_train}
        </span>
      </div>
      <div className="mt-2 flex flex-wrap gap-2">
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          n_models: {analysis.n_models ?? "skipped"}
        </span>
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          uncertainty_threshold:{" "}
          {analysis.uncertainty_threshold === null ? "—" : analysis.uncertainty_threshold.toFixed(4)}
        </span>
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          low_uncertainty: {analysis.low_uncertainty ?? "—"}
        </span>
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          reliable: {analysis.reliable ?? "—"}
        </span>
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          uncertainty_kind: {analysis.uncertainty_kind ?? "skipped"}
        </span>
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          fit_seconds: {analysis.fit_seconds.toFixed(2)}
        </span>
      </div>
      <ADQueryScatter analysis={analysis} />
      {analysis.frame.length > 0 ? (
        <div className="mt-3 overflow-hidden rounded-lg border border-[var(--color-border)]">
          <div className="max-h-64 overflow-auto">
            <table className="w-full border-collapse text-[11px]">
              <thead className="sticky top-0 bg-[var(--color-muted)]">
                <tr>
                  {["name", "predicted", "prediction_sd", "knn_distance", "inside_ad", "low_uncertainty", "reliable"].map((header) => (
                    <th key={header} scope="col" className="whitespace-nowrap px-2 py-1 text-left font-semibold text-[var(--color-foreground)]">
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {analysis.frame.slice(0, 20).map((row) => (
                  <tr key={row.name} className="border-t border-[var(--color-border)] bg-[var(--color-card)]">
                    <td className="whitespace-nowrap px-2 py-1 font-medium text-[var(--color-foreground)]">{row.name}</td>
                    <td className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">{formatValue(row.predicted)}</td>
                    <td className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">{formatValue(row.prediction_sd)}</td>
                    <td className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">{formatValue(row.knn_distance)}</td>
                    <td className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">{formatValue(row.inside_ad)}</td>
                    <td className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">{formatValue(row.low_uncertainty)}</td>
                    <td className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">{formatValue(row.reliable)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="border-t border-[var(--color-border)] bg-[var(--color-card)] px-3 py-2 text-[11px] text-[var(--color-muted-foreground)]">
            20 of {analysis.n_test} query rows shown — the analysis
            dict&apos;s counts above summarize the full query set.
          </p>
        </div>
      ) : null}
      {result.artifact !== null ? (
        <div className="mt-3">
          <p className="text-xs text-[var(--color-muted-foreground)]">
            figure artifact registered: {result.artifact.filename} (
            {result.artifact.size_bytes} bytes · sha256{" "}
            {result.artifact.sha256.slice(0, 16)}…) — the real
            plot_applicability_domain rendering.
          </p>
          <Button size="sm" variant="outline" className="mt-2" onClick={onDownload}>
            <ArrowDownToLine aria-hidden className="h-4 w-4" />
            Download the figure
          </Button>
        </div>
      ) : null}
      {artifactStatus !== null ? (
        <p className="mt-2 text-xs font-medium text-[var(--color-foreground)]">{artifactStatus}</p>
      ) : null}
    </div>
  );
}

/**
 * The Williams result card (B-4, part12 Phase 2): the analysis chips
 * (h*, residual band, rmse, the outlier counts), the live Plotly
 * Williams scatter + query extrapolation view, the per-row training
 * table and the query verdict table, and the digest-verified
 * plot_williams figure download.
 */
function WilliamsResultCard({
  result,
  artifactStatus,
  onDownload,
}: {
  result: WilliamsResultPayload;
  artifactStatus: string | null;
  onDownload: () => void;
}) {
  const analysis = result.analysis;
  return (
    <div className="mt-4 rounded-lg border border-[var(--color-border)] p-4">
      <div className="flex flex-wrap gap-2">
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          h* = {analysis.leverage_threshold.toFixed(4)}
        </span>
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          residual band ±{analysis.residual_limit}
        </span>
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          rmse: {analysis.rmse.toFixed(4)}
        </span>
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          n_train: {analysis.n_train} · p: {analysis.n_features}
        </span>
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          inside: {analysis.train_inside}/{analysis.n_train}
        </span>
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          leverage outliers: {analysis.n_train_outlier_leverage}
        </span>
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          residual outliers: {analysis.n_train_outlier_residual}
        </span>
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          queries:{" "}
          {analysis.n_test === null
            ? "none (training-only)"
            : `${analysis.test_inside ?? 0}/${analysis.n_test} inside`}
        </span>
      </div>
      <WilliamsScatter analysis={analysis} />
      {analysis.frame.length > 0 ? (
        <div className="mt-3 overflow-hidden rounded-lg border border-[var(--color-border)]">
          <div className="max-h-64 overflow-auto">
            <table className="w-full border-collapse text-[11px]">
              <thead className="sticky top-0 bg-[var(--color-muted)]">
                <tr>
                  {["name", "leverage", "std_residual", "inside"].map((header) => (
                    <th key={header} scope="col" className="whitespace-nowrap px-2 py-1 text-left font-semibold text-[var(--color-foreground)]">
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {analysis.frame.slice(0, 20).map((row) => (
                  <tr key={row.name} className="border-t border-[var(--color-border)] bg-[var(--color-card)]">
                    <td className="whitespace-nowrap px-2 py-1 font-medium text-[var(--color-foreground)]">{row.name}</td>
                    <td className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">{row.leverage.toFixed(4)}</td>
                    <td className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">{row.std_residual.toFixed(4)}</td>
                    <td className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">{formatValue(row.inside)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="border-t border-[var(--color-border)] bg-[var(--color-card)] px-3 py-2 text-[11px] text-[var(--color-muted-foreground)]">
            20 of {analysis.n_train} training rows shown — the table
            scrolls; the trace identity Σh = p+1 underlies the threshold.
          </p>
        </div>
      ) : null}
      {analysis.test_frame.length > 0 ? (
        <div className="mt-3 overflow-hidden rounded-lg border border-[var(--color-border)]">
          <div className="max-h-64 overflow-auto">
            <table className="w-full border-collapse text-[11px]">
              <thead className="sticky top-0 bg-[var(--color-muted)]">
                <tr>
                  {["query row", "leverage", "inside (leverage-only)", "predicted"].map((header) => (
                    <th key={header} scope="col" className="whitespace-nowrap px-2 py-1 text-left font-semibold text-[var(--color-foreground)]">
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {analysis.test_frame.slice(0, 20).map((row, i) => (
                  <tr key={`query-${i}`} className="border-t border-[var(--color-border)] bg-[var(--color-card)]">
                    <td className="whitespace-nowrap px-2 py-1 font-medium text-[var(--color-foreground)]">x_test[{i}]</td>
                    <td className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">{row.leverage.toFixed(4)}</td>
                    <td className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">{formatValue(row.inside)}</td>
                    <td className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">{row.predicted.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="border-t border-[var(--color-border)] bg-[var(--color-card)] px-3 py-2 text-[11px] text-[var(--color-muted-foreground)]">
            The query rows carry no ground truth — the verdict is
            leverage-only (no residual axis); the predicted column is the
            model&apos;s own extrapolation-axis value.
          </p>
        </div>
      ) : null}
      {result.artifact !== null ? (
        <div className="mt-3">
          <p className="text-xs text-[var(--color-muted-foreground)]">
            figure artifact registered: {result.artifact.filename} (
            {result.artifact.size_bytes} bytes · sha256{" "}
            {result.artifact.sha256.slice(0, 16)}…) — the real
            plot_williams rendering.
          </p>
          <Button size="sm" variant="outline" className="mt-2" onClick={onDownload}>
            <ArrowDownToLine aria-hidden className="h-4 w-4" />
            Download the figure
          </Button>
        </div>
      ) : null}
      {artifactStatus !== null ? (
        <p className="mt-2 text-xs font-medium text-[var(--color-foreground)]">{artifactStatus}</p>
      ) : null}
    </div>
  );
}

/**
 * The Williams result's two live Plotly lanes: (a) the Williams
 * scatter — training leverage h(i) against standardized residual r(i),
 * the h* vertical line and the ±band; (b) the query extrapolation —
 * the query rows' leverage against the model's own predictions
 * (omitted honestly when no query rows were passed).
 */
function WilliamsScatter({
  analysis,
}: {
  analysis: WilliamsResultPayload["analysis"];
}) {
  const frame = analysis.frame;
  const band = analysis.residual_limit;
  const hStar = analysis.leverage_threshold;
  const scatterTraces = useMemo<readonly PlotTrace[]>(() => {
    const inside = frame.filter((row) => row.inside === true);
    const outside = frame.filter((row) => row.inside !== true);
    const point = (row: WilliamsFrameRow): readonly unknown[] => [
      row.name,
      row.inside === true ? "inside domain" : "outside domain",
    ];
    return [
      {
        type: "scatter",
        mode: "markers",
        name: "inside domain",
        x: inside.map((row) => row.leverage),
        y: inside.map((row) => row.std_residual),
        customdata: inside.map(point),
        hovertemplate:
          "%{customdata[0]}<br>leverage h(i) %{x:.4g}" +
          "<br>std residual %{y:.4g}<br>%{customdata[1]}<extra></extra>",
        marker: { color: "var(--chart-1)", size: 7, opacity: 0.85 },
      },
      {
        type: "scatter",
        mode: "markers",
        name: "outside domain",
        x: outside.map((row) => row.leverage),
        y: outside.map((row) => row.std_residual),
        customdata: outside.map(point),
        hovertemplate:
          "%{customdata[0]}<br>leverage h(i) %{x:.4g}" +
          "<br>std residual %{y:.4g}<br>%{customdata[1]}<extra></extra>",
        marker: { color: "var(--chart-3)", size: 8, opacity: 0.9, symbol: "diamond" },
      },
    ];
  }, [frame]);
  const scatterLayout = useMemo(
    () => ({
      xaxis: { title: { text: "leverage h(i)" } },
      yaxis: { title: { text: "standardized residual" } },
      shapes: [
        {
          type: "line",
          x0: hStar,
          x1: hStar,
          y0: 0,
          y1: 1,
          yref: "paper",
          line: { color: "var(--chart-3)", width: 1, dash: "dash" },
        },
        {
          type: "line",
          x0: 0,
          x1: 1,
          xref: "paper",
          y0: band,
          y1: band,
          line: { color: "var(--chart-4)", width: 1, dash: "dot" },
        },
        {
          type: "line",
          x0: 0,
          x1: 1,
          xref: "paper",
          y0: -band,
          y1: -band,
          line: { color: "var(--chart-4)", width: 1, dash: "dot" },
        },
      ],
      annotations: [
        {
          x: hStar,
          y: 1,
          yref: "paper",
          text: `h* = ${hStar.toFixed(3)}`,
          showarrow: false,
          yanchor: "bottom",
          font: { size: 10 },
        },
      ],
      margin: { l: 56, r: 12, t: 8, b: 40 },
    }),
    [hStar, band],
  );
  const queryTraces = useMemo<readonly PlotTrace[]>(() => {
    const rows = analysis.test_frame;
    if (rows.length === 0) {
      return [];
    }
    const inside = rows.filter((row) => row.inside === true);
    const outside = rows.filter((row) => row.inside !== true);
    return [
      {
        type: "scatter",
        mode: "markers",
        name: "inside domain",
        x: inside.map((row) => row.leverage),
        y: inside.map((row) => row.predicted),
        hovertemplate:
          "query row<br>leverage h(i) %{x:.4g}" +
          "<br>predicted %{y:.4g}<extra></extra>",
        marker: { color: "var(--chart-1)", size: 8, opacity: 0.85 },
      },
      {
        type: "scatter",
        mode: "markers",
        name: "outside domain",
        x: outside.map((row) => row.leverage),
        y: outside.map((row) => row.predicted),
        hovertemplate:
          "query row<br>leverage h(i) %{x:.4g}" +
          "<br>predicted %{y:.4g}<extra></extra>",
        marker: { color: "var(--chart-3)", size: 10, opacity: 0.9, symbol: "diamond" },
      },
    ];
  }, [analysis.test_frame]);
  const queryLayout = useMemo(
    () => ({
      xaxis: { title: { text: "leverage h(i) — query compounds" } },
      yaxis: { title: { text: "predicted value" } },
      shapes: [
        {
          type: "line",
          x0: hStar,
          x1: hStar,
          y0: 0,
          y1: 1,
          yref: "paper",
          line: { color: "var(--chart-3)", width: 1, dash: "dash" },
        },
      ],
      annotations: [
        {
          x: hStar,
          y: 1,
          yref: "paper",
          text: `h* = ${hStar.toFixed(3)}`,
          showarrow: false,
          yanchor: "bottom",
          font: { size: 10 },
        },
      ],
      margin: { l: 56, r: 12, t: 8, b: 40 },
    }),
    [hStar],
  );
  return (
    <div className="mt-3 grid gap-3 lg:grid-cols-2">
      <div>
        <p className="mb-1 text-xs font-medium text-[var(--color-foreground)]">
          (a) Williams plot — h(i) vs standardized residual
        </p>
        <PlotlyFigure
          data={scatterTraces}
          layout={scatterLayout}
          height={300}
          ariaLabel="the Williams plot: training leverage against standardized residual with the warning-leverage line and residual band"
          emptyHint="No training rows to chart."
        />
      </div>
      <div>
        <p className="mb-1 text-xs font-medium text-[var(--color-foreground)]">
          {queryTraces.length > 0
            ? "(b) query extrapolation — leverage vs predicted"
            : "(b) no query rows — the training-only lane"}
        </p>
        {queryTraces.length > 0 ? (
          <PlotlyFigure
            data={queryTraces}
            layout={queryLayout}
            height={300}
            ariaLabel="the query extrapolation view: query compounds' leverage against the model's own predictions"
            emptyHint="No query rows to chart."
          />
        ) : (
          <p className="rounded-md border border-dashed border-[var(--color-border)] p-3 text-xs text-[var(--color-muted-foreground)]">
            Assemble x_test in the kNN/bootstrap lane above and re-run —
            the query rows&apos; leverage-only verdicts land here.
          </p>
        )}
      </div>
    </div>
  );
}

/**
 * The stage-12 result card: the SHAP summary's chips (model, task,
 * engine, the honest sample bound, base value), the full mean-|SHAP|
 * ranking table (bounded viewport with the full count stated), the
 * compact per-feature beeswarm preview, and the digest-verified
 * figure-artifact downloads.
 */
function ExplainResultCard({
  result,
  artifactStatuses,
  onDownload,
}: {
  result: ExplainResult;
  artifactStatuses: Record<string, string>;
  onDownload: (artifactId: string) => void;
}) {
  return (
    <div className="mt-4 rounded-lg border border-[var(--color-border)] p-4">
      <div className="flex flex-wrap gap-2">
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          engine: {result.engine}
        </span>
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          model: {result.model_name}
        </span>
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          task: {result.task}
        </span>
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          explained {result.n_samples} of {result.n_rows_available} rows
        </span>
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          features: {result.n_features}
        </span>
        <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
          base value: {result.base_value === null ? "—" : result.base_value.toFixed(4)}
        </span>
      </div>

      <ShapRankingBar ranking={result.ranking} />

      <div className="mt-3 overflow-hidden rounded-lg border border-[var(--color-border)]">
        <div className="max-h-64 overflow-auto">
          <table className="w-full border-collapse text-[11px]">
            <thead className="sticky top-0 bg-[var(--color-muted)]">
              <tr>
                <th scope="col" className="whitespace-nowrap px-2 py-1 text-left font-semibold text-[var(--color-foreground)]">
                  feature
                </th>
                <th scope="col" className="whitespace-nowrap px-2 py-1 text-left font-semibold text-[var(--color-foreground)]">
                  mean |SHAP|
                </th>
              </tr>
            </thead>
            <tbody>
              {result.ranking.map((row) => (
                <tr
                  key={row.feature}
                  className="border-t border-[var(--color-border)] bg-[var(--color-card)]"
                >
                  <td className="whitespace-nowrap px-2 py-1 font-medium text-[var(--color-foreground)]">
                    {row.feature}
                  </td>
                  <td className="whitespace-nowrap px-2 py-1 text-[var(--color-muted-foreground)]">
                    {row.mean_abs_shap.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="border-t border-[var(--color-border)] bg-[var(--color-card)] px-3 py-2 text-[11px] text-[var(--color-muted-foreground)]">
          All {result.ranking.length} features ranked (descending mean
          |SHAP|) — the preview matrices below carry only the top{" "}
          {result.preview.features.length} column
          {result.preview.features.length === 1 ? "" : "s"} (the
          route&apos;s own {result.preview.max_features}-column bound).
        </p>
      </div>

      {result.preview.features.length > 0 ? (
        <ShapStrip result={result} />
      ) : null}

      {result.artifacts.length > 0 ? (
        <div className="mt-3 space-y-2">
          {result.artifacts.map((artifact) => (
            <div key={artifact.artifact_id} className="flex flex-wrap items-center gap-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => onDownload(artifact.artifact_id)}
              >
                <ArrowDownToLine aria-hidden className="h-4 w-4" />
                Download {artifact.filename}
              </Button>
              <span className="text-[11px] text-[var(--color-muted-foreground)]">
                {artifact.size_bytes} bytes · sha256{" "}
                {artifact.sha256.slice(0, 16)}…
              </span>
              {artifactStatuses[artifact.artifact_id] !== undefined ? (
                <span className="text-[11px] font-medium text-[var(--color-foreground)]">
                  {artifactStatuses[artifact.artifact_id]}
                </span>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

/**
 * The compact beeswarm preview: one strip per top feature (≤6), one
 * dot per explained sample — the x position is the sample&apos;s SHAP
 * value (normalized within the strip), the colour the feature&apos;s own
 * value on that sample (the house amber→teal ramp: teal = high,
 * amber = low). A preview only — the beeswarm artifact carries the
 * true figure.
 */
function ShapStrip({ result }: { result: ExplainResult }) {
  const rows = result.preview.features.slice(0, 6);
  const rowHeight = 26;
  const labelWidth = 96;
  const plotWidth = 100;
  return (
    <div className="mt-3">
      <svg
        role="img"
        aria-label="SHAP beeswarm preview: one dot per explained sample per top feature"
        viewBox={`0 0 100 ${rows.length * rowHeight}`}
        className="w-full"
        style={{ maxHeight: rows.length * rowHeight }}
      >
        {rows.map((feature, featureIndex) => {
          const columnIndex = result.preview.features.indexOf(feature);
          // noUncheckedIndexedAccess: every indexed read is guarded
          const column: number[] = [];
          for (const row of result.preview.values) {
            const value = row[columnIndex];
            if (value !== undefined) {
              column.push(value);
            }
          }
          const colorColumn: number[] = [];
          for (const row of result.preview.feature_values) {
            const value = row[columnIndex];
            if (value !== undefined) {
              colorColumn.push(value);
            }
          }
          let lo = Number.POSITIVE_INFINITY;
          let hi = Number.NEGATIVE_INFINITY;
          for (const value of column) {
            if (Number.isFinite(value)) {
              lo = Math.min(lo, value);
              hi = Math.max(hi, value);
            }
          }
          const span = hi - lo;
          const finiteColors = colorColumn.filter(Number.isFinite);
          const colorLo = finiteColors.length > 0
            ? Math.min(...finiteColors)
            : 0;
          const colorHi = finiteColors.length > 0
            ? Math.max(...finiteColors)
            : 0;
          const colorSpan = colorHi - colorLo;
          const y = featureIndex * rowHeight + rowHeight / 2;
          return (
            <g key={feature}>
              <text
                x={0}
                y={y + 3}
                fontSize={7}
                fill="var(--color-muted-foreground)"
              >
                {feature.length > 14 ? `${feature.slice(0, 13)}…` : feature}
              </text>
              <line
                x1={labelWidth}
                x2={96}
                y1={y}
                y2={y}
                stroke="var(--color-border)"
                strokeWidth={0.4}
              />
              {column.map((value, sampleIndex) => {
                if (!Number.isFinite(value)) {
                  return null;
                }
                const t = span > 0 ? (value - lo) / span : 0.5;
                const x = labelWidth + t * (plotWidth - 8) + 2;
                const colorValue = colorColumn[sampleIndex];
                const colorT =
                  colorSpan > 0 &&
                  colorValue !== undefined &&
                  Number.isFinite(colorValue)
                    ? (colorValue - colorLo) / colorSpan
                    : 0.5;
                // the house ramp: amber (chart-2) for low feature
                // values → teal (chart-1) for high ones
                const fill = `color-mix(in oklch, var(--color-chart-2) ${Math.round((1 - colorT) * 100)}%, var(--color-chart-1))`;
                return (
                  <circle
                    key={sampleIndex}
                    cx={x}
                    cy={y}
                    r={2.2}
                    fill={fill}
                    fillOpacity={0.85}
                  />
                );
              })}
            </g>
          );
        })}
      </svg>
      <p className="mt-1 text-[11px] text-[var(--color-muted-foreground)]">
        beeswarm preview: the top {rows.length} features, one dot per
        explained sample (x = the SHAP value normalized per strip;
        colour = the feature&apos;s own value, amber = low → teal =
        high). The beeswarm artifact carries the true figure.
      </p>
    </div>
  );
}

/**
 * Build the Export stage's rows from the LAST predictions block: one
 * row per prediction, the prediction column plus one probability
 * column per class (duplicated stringified class labels get a
 * disambiguating suffix so no column swallows another).
 */
function buildExportRows(
  result: PredictResponse,
): { rows: Record<string, unknown>[]; columns: string[] } {
  const columns: string[] = ["prediction"];
  const probabilities = result.probabilities;
  if (probabilities !== null) {
    const used = new Set<string>(columns);
    for (const cls of probabilities.classes) {
      let label = `probability_${String(cls)}`;
      while (used.has(label)) {
        label += "_";
      }
      used.add(label);
      columns.push(label);
    }
  }
  const probLabels = columns.slice(1);
  const rows: Record<string, unknown>[] = [];
  for (const [i, prediction] of result.predictions.entries()) {
    const row: Record<string, unknown> = { prediction };
    if (probabilities !== null) {
      const probRow = probabilities.rows[i];
      if (probRow !== undefined) {
        for (const [j, label] of probLabels.entries()) {
          row[label] = probRow[j] ?? null;
        }
      }
    }
    rows.push(row);
  }
  return { rows, columns };
}
