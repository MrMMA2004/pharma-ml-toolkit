import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowDownToLine } from "lucide-react";
import { Button } from "@/components/ui/button";
import { DemoDataSection } from "@/components/demo/DemoDataSection";
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
import { mlApi, type PredictResponse } from "@/services/ml/client";
import { getEventStream } from "@/services/websocket/client";
import { LossCurveChart } from "@/components/training/LossCurveChart";
import {
  EvaluationCharts,
  buildPredictionExportRows,
  buildEvaluationExportRows,
} from "@/components/evaluation/EvaluationCharts";
import {
  PlotlyFigure,
  type PlotTrace,
} from "@/components/charts/PlotlyFigure";
import {
  DL_ACTIVATIONS,
  DL_HYPERPARAM_KEYS,
  dlApi,
  parseDLTuneResult,
  parseTrainingMetricsPayload,
  parseGnnInlineGraphs,
  parseGnnTrainResult,
  parseMpnnTrainResult,
  parseCnnTrainResult,
  parseAutoencoderTrainResult,
  type AutoencoderEncodeResponse,
  type AutoencoderTrainResult,
  type CnnPredictResponse,
  type CnnTrainResult,
  type DLEfficiencyResult,
  type DLModelView,
  type DLNetworkName,
  type DLTrainOptionsTune,
  type DLTuneReport,
  type GnnConv,
  type GnnExplainGraphRow,
  type GnnExplainResult,
  type GnnInlineSource,
  type GnnPool,
  type GnnPredictResponse,
  type GnnTrainResult,
  type MpnnPredictResponse,
  type MpnnTrainResult,
  type NetworkCatalogEntry,
  type NetworkCatalogGnn,
  type TrainingMetricsPoint,
} from "@/services/dl/client";
import type { components } from "@/types/api.d";

/**
 * The Deep Learning workspace — part11 §11.3's complete sequential
 * workflow:
 *
 * Select Dataset → Validate → Preprocess → Select Architecture →
 * Configure Model → Configure Training → Train → Monitor → Evaluate
 * → Visualize → Save Checkpoint / Export
 *
 * Every control below executes the REAL backend through the typed
 * service layer (no fabricated data, no disabled pretend buttons):
 * the architecture catalog rides routes/dl.py's live signature
 * defaults; the train lane runs the torch network's own
 * `fit(X, y, on_epoch=…)` as a managed job whose per-epoch
 * `training.metrics` envelopes drive the Monitor stage's live
 * loss/validation curves; predictions ride the SINGLE session
 * prediction path (`POST /api/v1/ml/predict` — the model registers in
 * the same session model store, never a second path); the checkpoint
 * stage persists the fitted estimator as the standard joblib bundle
 * with the full provenance metadata sidecar (`POST
 * /api/v1/artifacts/model`); the curve visualisation registers a REAL
 * figure artifact through the §17 four-format export stack; the
 * predictions export rides the rows-based table lane with the
 * backend's own 5000-row cap mirrored client-side. The two honest
 * exceptions are stated where they live: the DL train route carries
 * no preprocess toggle (the torch estimator standardises per-feature
 * inside its own fit — that note sits in the Preprocess stage), and
 * the trained view carries no CV scores (routes/dl.py: "no CV in
 * this path — honest, not faked" — the Compare lane that owns
 * cv_mean lives in the Machine Learning workspace).
 */

type VisualizationTableRequest = components["schemas"]["VisualizationTableRequest"];

/** The backend's own table-export row cap (visualization.py's `_MAX_ROWS`). */
const TABLE_EXPORT_MAX_ROWS = 5000;

function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    const detail = error.detail;
    if (
      typeof detail === "object" &&
      detail !== null &&
      "detail" in detail &&
      typeof (detail as Record<string, unknown>)["detail"] === "string"
    ) {
      return (detail as Record<string, string>)["detail"] ?? `HTTP ${error.status}`;
    }
    if (typeof detail === "string") {
      return detail;
    }
    return `HTTP ${error.status}`;
  }
  return error instanceof Error ? error.message : String(error);
}

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

/** A numbered workflow stage header (§11.3's fixed order). */
function StageHeader({
  index,
  title,
  hint,
  children,
}: {
  index: number;
  title: string;
  hint: string;
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
    </section>
  );
}

/** One table cell's honest rendering (§17.3: type-aware, bounded). */
function formatValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "—";
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      return "—";
    }
    return Number.isInteger(value) ? String(value) : value.toFixed(4);
  }
  if (Array.isArray(value)) {
    return `[${value.map((v) => formatValue(v)).join(", ")}]`;
  }
  return String(value);
}

function formatLoss(value: number | null): string {
  if (value === null || !Number.isFinite(value)) {
    return "—";
  }
  return value.toFixed(4);
}

/**
 * Validate the hidden_layers overrides BEFORE any request fires
 * (§8.1): empty → null (the signature's own default); a JSON list of
 * positive ints → the list; anything else → the thrown reason the
 * inline error renders verbatim.
 */
function validateHiddenLayers(text: string): number[] | null {
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
  if (!Array.isArray(parsed) || parsed.length === 0) {
    throw new Error(
      "hidden_layers must be a non-empty JSON list of positive integers, e.g. [64, 32]",
    );
  }
  for (const item of parsed) {
    if (typeof item !== "number" || !Number.isInteger(item) || item <= 0) {
      throw new Error(
        "hidden_layers must be a non-empty JSON list of positive integers, e.g. [64, 32]",
      );
    }
  }
  return parsed as number[];
}

/** The DL-1 search-block sample — the same paste-lane pattern. */
const TUNE_SEARCH_SAMPLE = `{
  "learning_rate": {"low": 0.001, "high": 0.3, "log": true},
  "dropout": {"low": 0.05, "high": 0.5},
  "hidden_layers": {"choices": [[32], [64, 32], [128, 64]]}
}`;

/**
 * DL-1 — parse + pre-flight the search block: the client mirrors
 * the route's own contract (each key over the constructor vocabulary,
 * each entry choices or low/high) so a doomed search is refused
 * here, honestly, never fired.
 */
function parseTuneSearch(text: string): Record<string, Record<string, unknown>> {
  const trimmed = text.trim();
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch (error) {
    throw new Error(
      `invalid JSON — ${error instanceof Error ? error.message : String(error)}`,
    );
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error(
      "the search block must be a JSON object of {key: {choices: [...]} or {low, high}}",
    );
  }
  const accepted = new Set(DL_HYPERPARAM_KEYS);
  const clean: Record<string, Record<string, unknown>> = {};
  for (const [key, spec] of Object.entries(parsed)) {
    if (!accepted.has(key)) {
      throw new Error(
        `unknown search key ${JSON.stringify(key)}; the route's own ` +
          `vocabulary is ${DL_HYPERPARAM_KEYS.join(", ")}`,
      );
    }
    if (typeof spec !== "object" || spec === null || Array.isArray(spec)) {
      throw new Error(
        `the search entry for ${JSON.stringify(key)} must be an object with ` +
          "either choices: [...] or low/high bounds",
      );
    }
    const entry = spec as Record<string, unknown>;
    if ("choices" in entry) {
      if (!Array.isArray(entry["choices"]) || entry["choices"].length === 0) {
        throw new Error(
          `the search entry for ${JSON.stringify(key)} carries an empty ` +
            "or non-list choices block",
        );
      }
    } else if ("low" in entry && "high" in entry) {
      const low = entry["low"];
      const high = entry["high"];
      if (
        typeof low !== "number" ||
        !Number.isFinite(low) ||
        typeof high !== "number" ||
        !Number.isFinite(high)
      ) {
        throw new Error(
          `the search bounds for ${JSON.stringify(key)} must be numbers`,
        );
      }
      if (!(low < high)) {
        throw new Error(
          `the search bounds for ${JSON.stringify(key)} must satisfy low < high`,
        );
      }
    } else {
      throw new Error(
        `the search entry for ${JSON.stringify(key)} carries neither ` +
          "choices nor low/high bounds",
      );
    }
    clean[key] = entry;
  }
  if (Object.keys(clean).length === 0) {
    throw new Error("the search block is empty — the tune lane needs at least one key");
  }
  return clean;
}

/** The model-metadata card — the train job's relayed result view. */
function DLModelCard({ model }: { model: DLModelView }) {
  return (
    <div className="mt-4 rounded-lg border border-[var(--color-border)] p-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-medium text-[var(--color-foreground)]">{model.name}</span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          {model.task}
        </span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          engine {model.engine}
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
      <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
        dataset {model.dataset_id} · target {model.target} ·{" "}
        {model.features.length} features · trained at {model.created_at}
      </p>
      <div className="mt-3 flex flex-wrap gap-1">
        {Object.entries(model.best_params).map(([key, value]) => (
          <span
            key={key}
            className="rounded bg-[var(--color-muted)] px-1.5 py-0.5 text-[10px] text-[var(--color-muted-foreground)]"
          >
            {key}: {formatValue(value)}
          </span>
        ))}
      </div>
      <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
        The authoritative REST result of the train job, parsed at
        runtime — the fitted torch estimator itself stays server-side
        in the session model store; no CV scores ride this view (the DL
        train path carries none — honest, not faked).
      </p>
    </div>
  );
}

/** The gnn-train result card — the graph model's own honest view. */
function GnnModelCard({ result }: { result: GnnTrainResult }) {
  const model = result.model;
  return (
    <div className="mt-4 rounded-lg border border-[var(--color-border)] p-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-medium text-[var(--color-foreground)]">{model.name}</span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          {model.task}
        </span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          engine {model.engine}
        </span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          graph model
        </span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          {model.model_id}
        </span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          {result.n_graphs} graphs
        </span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          fit {model.fit_seconds.toFixed(2)} s
        </span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          conv {result.conv} · pool {result.pool}
        </span>
      </div>
      <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
        classes {result.classes.map((c) => formatValue(c)).join(" / ")} · no
        tabular dataset behind a graph model — the node features are
        per-graph matrices, never columns (honest, not faked)
      </p>
      <div className="mt-3 flex flex-wrap gap-1">
        {Object.entries(model.best_params).map(([key, value]) => (
          <span
            key={key}
            className="rounded bg-[var(--color-muted)] px-1.5 py-0.5 text-[10px] text-[var(--color-muted-foreground)]"
          >
            {key}: {formatValue(value)}
          </span>
        ))}
      </div>
      <p className="mt-2 text-xs text-[var(--color-foreground)]">
        train accuracy {result.train_accuracy.toFixed(4)} · loss history
        (tail) {result.loss_history_tail.map((v) => v.toFixed(3)).join(" → ")}
      </p>
      <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
        The authoritative REST result of the gnn-train job, parsed at
        runtime — the fitted GraphClassifier stays server-side in the
        session model store as a GRAPH model. The loss history is the
        trainer&apos;s own `loss_history_` tail (the last five entries),
        surfaced AFTER the fit — the torch GNN trainer exposes no live
        per-epoch callback, so this lane never fabricates an epoch
        stream (the honest stage-level telemetry the job card shows).
      </p>
    </div>
  );
}

/** DL-4 — the D-MPNN lane's result card: the held-out carve's own
 *  verdicts (the roadmap's own validation step) + the honest
 *  rejected-rows accounting. */
function MpnnModelCard({ result }: { result: MpnnTrainResult }) {
  const model = result.model;
  return (
    <div className="mt-4 rounded-lg border border-[var(--color-border)] p-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-medium text-[var(--color-foreground)]">{model.name}</span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          {model.task}
        </span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          engine {model.engine}
        </span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          molecular model
        </span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          {model.model_id}
        </span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          {result.n_rows} molecules
        </span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          {result.n_parameters} parameters
        </span>
        <span className="rounded-full bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
          fit {model.fit_seconds.toFixed(2)} s
        </span>
      </div>
      <p className="mt-2 text-xs text-[var(--color-foreground)]">
        held-out RMSE {result.val_rmse.toFixed(3)} over {result.val_n} rows ·{" "}
        {result.val_correlation !== null
          ? `correlation ${result.val_correlation.toFixed(3)}`
          : "correlation n/a (the carve too small or degenerate — honestly null)"}
        {" "}· train loss (tail){" "}
        {result.train_history_tail.map((v) => v.toFixed(3)).join(" → ")}
      </p>
      {result.n_rejected > 0 ? (
        <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
          {result.n_rejected} rejected row(s):{" "}
          {result.rejected
            .map((row) => `#${row.index} — ${row.error}`)
            .join("; ")}
        </p>
      ) : null}
      <div className="mt-3 flex flex-wrap gap-1">
        {Object.entries(model.best_params).map(([key, value]) => (
          <span
            key={key}
            className="rounded bg-[var(--color-muted)] px-1.5 py-0.5 text-[10px] text-[var(--color-muted-foreground)]"
          >
            {key}: {formatValue(value)}
          </span>
        ))}
      </div>
      <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
        {result.held_out_note}
      </p>
    </div>
  );
}

/** DL-2 — one explained graph: the fidelity line + the edge table
 *  (top rows highlighted) + the sized-edge scatter over the
 *  backend's own circular layout. */
function GnnExplainRow({ row }: { row: GnnExplainGraphRow }) {
  const topSet = new Set(row.top_edges);
  const nodes = row.layout.nodes;
  // the edge segments (null-separated, one trace) + the sized-edge
  // midpoints (the score-scaled markers) + the node markers
  const segX: (number | null)[] = [];
  const segY: (number | null)[] = [];
  const midX: number[] = [];
  const midY: number[] = [];
  const midSize: number[] = [];
  const midColor: string[] = [];
  for (let i = 0; i < row.edge_index.length; i++) {
    const endpoints = row.edge_index[i];
    if (endpoints === undefined || endpoints.length < 2) {
      continue;
    }
    const [src, dst] = [endpoints[0], endpoints[1]];
    const srcPos = nodes[src ?? 0];
    const dstPos = nodes[dst ?? 0];
    if (srcPos === undefined || dstPos === undefined) {
      continue;
    }
    segX.push(srcPos[0] ?? 0, dstPos[0] ?? 0, null);
    segY.push(srcPos[1] ?? 0, dstPos[1] ?? 0, null);
    const score = row.scores[i] ?? 0;
    midX.push(((srcPos[0] ?? 0) + (dstPos[0] ?? 0)) / 2);
    midY.push(((srcPos[1] ?? 0) + (dstPos[1] ?? 0)) / 2);
    midSize.push(4 + 36 * score);
    midColor.push(topSet.has(i) ? "#d97706" : "#a8a29e");
  }
  const traces: PlotTrace[] = [
    {
      type: "scatter",
      mode: "lines",
      name: "edges",
      x: segX,
      y: segY,
      line: { color: "#d6d3d1", width: 1 },
      showlegend: false,
    },
    {
      type: "scatter",
      mode: "markers",
      name: "nodes",
      x: nodes.map((pos) => pos[0] ?? 0),
      y: nodes.map((pos) => pos[1] ?? 0),
      marker: { color: "#57534e", size: 8, symbol: "circle" },
      showlegend: false,
    },
    {
      type: "scatter",
      mode: "markers",
      name: "edge importance",
      x: midX,
      y: midY,
      marker: {
        color: midColor,
        size: midSize,
        sizemode: "area",
        sizeref: 2.5,
      },
      text: row.scores.map(
        (score, i) =>
          `edge ${i} (${row.edge_index[i]?.[0]}→${row.edge_index[i]?.[1]}): ${score.toFixed(3)}${topSet.has(i) ? " (top)" : ""}`,
      ),
      hovertemplate: "%{text}<extra></extra>",
      showlegend: false,
    },
  ];
  return (
    <div className="mt-3 rounded-lg border border-[var(--color-border)] p-3">
      <p className="text-xs text-[var(--color-foreground)]">
        graph {row.index} · {row.n_nodes} nodes · {row.n_edges} edges ·
        method {row.method} · target class {row.target_class} ·
        fidelity+ {formatValue(row.fidelity_plus)} · fidelity−{" "}
        {formatValue(row.fidelity_minus)} · sparsity{" "}
        {formatValue(row.sparsity)}
      </p>
      <div className="mt-2 grid gap-3 lg:grid-cols-2">
        <div className="max-h-64 overflow-y-auto rounded-md border border-[var(--color-border)]">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-[var(--color-muted)] text-left text-[var(--color-muted-foreground)]">
                <th className="px-2 py-1.5 font-medium">edge</th>
                <th className="px-2 py-1.5 font-medium">importance</th>
                <th className="px-2 py-1.5 font-medium">top</th>
              </tr>
            </thead>
            <tbody>
              {row.scores.map((score, i) => {
                const isTop = topSet.has(i);
                return (
                  <tr
                    key={i}
                    data-top={isTop ? "true" : undefined}
                    className={`border-t border-[var(--color-border)] ${
                      isTop
                        ? "bg-[var(--color-accent)]/40 font-medium text-[var(--color-foreground)]"
                        : ""
                    }`}
                  >
                    <td className="px-2 py-1.5 font-mono">
                      {row.edge_index[i]?.[0]}→{row.edge_index[i]?.[1]}
                    </td>
                    <td className="px-2 py-1.5">{formatValue(score)}</td>
                    <td className="px-2 py-1.5">{isTop ? "top" : ""}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="rounded-md border border-[var(--color-border)] p-2">
          <PlotlyFigure
            data={traces}
            layout={{
              showlegend: false,
              margin: { t: 8, r: 8, b: 8, l: 8 },
              xaxis: { zeroline: false, showgrid: false },
              yaxis: { zeroline: false, showgrid: false, scaleanchor: "x" },
            }}
            ariaLabel={`graph ${row.index} — the sized-edge scatter over the backend's own circular layout (the marker size rides the edge importance)`}
          />
        </div>
      </div>
    </div>
  );
}

export function DLPage() {
  // stage 1: the dataset registry + selection
  const [datasets, setDatasets] = useState<readonly DatasetView[]>([]);
  const [registryError, setRegistryError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selected, setSelected] = useState<DatasetView | null>(null);
  const [selectionError, setSelectionError] = useState<string | null>(null);

  // stage 4: the architecture catalog (the live signature defaults)
  const [networks, setNetworks] = useState<readonly NetworkCatalogEntry[]>([]);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [selectedNetwork, setSelectedNetwork] = useState<DLNetworkName | "">("");

  // stage 5: the model (architecture) configuration
  const [hiddenLayersText, setHiddenLayersText] = useState("");
  const [hiddenLayersError, setHiddenLayersError] = useState<string | null>(null);
  const [dropout, setDropout] = useState(0.1);
  const [activation, setActivation] = useState<string>("relu");
  const [batchnorm, setBatchnorm] = useState(false);

  // stage 6: the training configuration
  const [epochs, setEpochs] = useState(40);
  const [batchSize, setBatchSize] = useState(32);
  const [learningRate, setLearningRate] = useState(0.001);
  const [weightDecay, setWeightDecay] = useState(0.0001);
  const [patience, setPatience] = useState(10);
  const [valSplit, setValSplit] = useState(0.15);
  const [randomState, setRandomState] = useState(42);
  const [device, setDevice] = useState<string>("auto");

  // stage 7: train (job mode — the route's default lane)
  const [trainJobId, setTrainJobId] = useState<string | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [trainBusy, setTrainBusy] = useState(false);
  const [trainedModel, setTrainedModel] = useState<DLModelView | null>(null);

  // stage 7b (DL-1): the hyperparameter search lane — tune instead
  // of one fixed configuration (the same route, `tune: true`, the
  // in-house TPE/Random engine under it)
  const [tuneEnabled, setTuneEnabled] = useState(false);
  const [tuneSearchText, setTuneSearchText] = useState(TUNE_SEARCH_SAMPLE);
  const [tuneNTrials, setTuneNTrials] = useState(12);
  const [tuneSampler, setTuneSampler] = useState<"tpe" | "random">("tpe");
  const [tuneSeed, setTuneSeed] = useState(0);
  const [tuneResult, setTuneResult] = useState<DLTuneReport | null>(null);

  // stage 8: monitor (the live per-epoch telemetry)
  const [metricsPoints, setMetricsPoints] = useState<readonly TrainingMetricsPoint[]>([]);
  const [monitorError, setMonitorError] = useState<string | null>(null);

  // stage 9: evaluate (predict, dataset mode, sync — the single path)
  const [predictProba, setPredictProba] = useState(true);
  const [predictBusy, setPredictBusy] = useState(false);
  const [predictError, setPredictError] = useState<string | null>(null);
  const [predictResult, setPredictResult] = useState<PredictResponse | null>(null);

  // stage 10: visualize (the loss-curve figure artifact)
  const [figureFormat, setFigureFormat] = useState<"PNG" | "SVG" | "TIFF" | "JPEG">("PNG");
  const [figureArtifact, setFigureArtifact] = useState<ArtifactView | null>(null);
  const [figureError, setFigureError] = useState<string | null>(null);
  const [figureBusy, setFigureBusy] = useState(false);
  const [figureDigestStatus, setFigureDigestStatus] = useState<string | null>(null);

  // stage 11: save checkpoint / export
  const [checkpointArtifact, setCheckpointArtifact] = useState<ArtifactView | null>(null);
  const [checkpointError, setCheckpointError] = useState<string | null>(null);
  const [checkpointBusy, setCheckpointBusy] = useState(false);
  const [checkpointDigestStatus, setCheckpointDigestStatus] = useState<string | null>(null);
  const [exportFormat, setExportFormat] = useState<"csv" | "tsv" | "xlsx">("csv");
  const [exportSource, setExportSource] = useState<"predictions" | "evaluation">("predictions");
  const [exportArtifact, setExportArtifact] = useState<ArtifactView | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportBusy, setExportBusy] = useState(false);
  const [digestStatus, setDigestStatus] = useState<string | null>(null);

  // stage 11c: the efficiency lab (DL-3 — the roadmap's row 15, the
  // weight engines' own lane over the trained tabular model)
  const [efficiencyMode, setEfficiencyMode] = useState<
    "quantize_int8" | "quantize_nf4" | "prune"
  >("quantize_int8");
  const [efficiencyLevelText, setEfficiencyLevelText] = useState("0.5");
  const [efficiencyScope, setEfficiencyScope] = useState<"local" | "global">("local");
  const [efficiencyResult, setEfficiencyResult] = useState<DLEfficiencyResult | null>(null);
  const [efficiencyError, setEfficiencyError] = useState<string | null>(null);
  const [efficiencyBusy, setEfficiencyBusy] = useState(false);
  const [efficiencyDigestStatus, setEfficiencyDigestStatus] = useState<string | null>(null);

  // the GNN/MPNN lane (part13 Phase 5 — the graph models' own lane,
  // parallel to the tabular §11.3 chain above)
  const [gnnSource, setGnnSource] = useState<"synthetic" | "inline">("synthetic");
  const [gnnSyntheticN, setGnnSyntheticN] = useState(18);
  const [gnnSeed, setGnnSeed] = useState(11);
  const [gnnGraphsText, setGnnGraphsText] = useState("");
  const [gnnGraphsError, setGnnGraphsError] = useState<string | null>(null);
  const [gnnVocab, setGnnVocab] = useState<NetworkCatalogGnn | null>(null);
  const [gnnConv, setGnnConv] = useState<GnnConv>("gcn");
  const [gnnPool, setGnnPool] = useState<GnnPool>("mean");
  const [gnnHidden, setGnnHidden] = useState(16);
  const [gnnLayers, setGnnLayers] = useState(2);
  const [gnnEpochs, setGnnEpochs] = useState(120);
  const [gnnLr, setGnnLr] = useState(0.08);
  const [gnnWeightDecay, setGnnWeightDecay] = useState(0);
  const [gnnHeads, setGnnHeads] = useState(1);
  const [gnnRatio, setGnnRatio] = useState(0.5);
  const [gnnTrainJobId, setGnnTrainJobId] = useState<string | null>(null);
  const [gnnTrainError, setGnnTrainError] = useState<string | null>(null);
  const [gnnTrainBusy, setGnnTrainBusy] = useState(false);
  const [gnnTrainResult, setGnnTrainResult] = useState<GnnTrainResult | null>(null);
  const [gnnPredictText, setGnnPredictText] = useState("");
  const [gnnPredictError, setGnnPredictError] = useState<string | null>(null);
  const [gnnPredictProba, setGnnPredictProba] = useState(true);
  const [gnnPredictBusy, setGnnPredictBusy] = useState(false);
  const [gnnPredictResult, setGnnPredictResult] = useState<GnnPredictResponse | null>(null);

  // DL-2: the GNN lane's edge-attribution card — the explainability
  // state (the same inline-graphs paste lane the predict card rides)
  const [gnnExplainText, setGnnExplainText] = useState("");
  const [gnnExplainMethod, setGnnExplainMethod] = useState<
    "gnnexplainer" | "pgexplainer" | "attention" | "subgraphx"
  >("gnnexplainer");
  const [gnnExplainTopK, setGnnExplainTopK] = useState(8);
  const [gnnExplainSeed, setGnnExplainSeed] = useState(0);
  const [gnnExplainItersText, setGnnExplainItersText] = useState("");
  const [gnnExplainTargetText, setGnnExplainTargetText] = useState("");
  const [gnnExplainLabelsText, setGnnExplainLabelsText] = useState("");
  const [gnnExplainBusy, setGnnExplainBusy] = useState(false);
  const [gnnExplainError, setGnnExplainError] = useState<string | null>(null);
  const [gnnExplainResult, setGnnExplainResult] = useState<GnnExplainResult | null>(null);

  // the sequence-CNN lane (part18 §5.2 — the convolutional lane over
  // 3-D sequences or the backend's motif fixture, its OWN predict path)
  const [cnnSource, setCnnSource] = useState<"synthetic" | "inline">("synthetic");
  const [cnnSyntheticN, setCnnSyntheticN] = useState(40);
  const [cnnSeed, setCnnSeed] = useState(0);
  const [cnnTask, setCnnTask] = useState<"classification" | "regression">("classification");
  const [cnnChannels, setCnnChannels] = useState(8);
  const [cnnKernel, setCnnKernel] = useState(3);
  const [cnnLayers, setCnnLayers] = useState(2);
  const [cnnEpochs, setCnnEpochs] = useState(25);
  const [cnnLr, setCnnLr] = useState(3e-3);
  const [cnnInlineText, setCnnInlineText] = useState("");
  const [cnnInlineError, setCnnInlineError] = useState<string | null>(null);
  const [cnnTrainJobId, setCnnTrainJobId] = useState<string | null>(null);
  const [cnnTrainError, setCnnTrainError] = useState<string | null>(null);
  const [cnnTrainBusy, setCnnTrainBusy] = useState(false);
  const [cnnTrainResult, setCnnTrainResult] = useState<CnnTrainResult | null>(null);
  const [cnnPredictText, setCnnPredictText] = useState("");
  const [cnnPredictError, setCnnPredictError] = useState<string | null>(null);
  const [cnnPredictProba, setCnnPredictProba] = useState(true);
  const [cnnPredictBusy, setCnnPredictBusy] = useState(false);
  const [cnnPredictResult, setCnnPredictResult] = useState<CnnPredictResponse | null>(null);

  // the tabular-autoencoder lane (part18 §5.5 — the representation /
  // reconstruction / anomaly lane over the selected dataset)
  const [aeLatentDim, setAeLatentDim] = useState(2);
  const [aeHiddenText, setAeHiddenText] = useState("[16]");
  const [aeEpochs, setAeEpochs] = useState(40);
  const [aeTrainJobId, setAeTrainJobId] = useState<string | null>(null);
  const [aeTrainError, setAeTrainError] = useState<string | null>(null);
  const [aeTrainBusy, setAeTrainBusy] = useState(false);
  const [aeTrainResult, setAeTrainResult] = useState<AutoencoderTrainResult | null>(null);
  const [aeEncodeBusy, setAeEncodeBusy] = useState(false);
  const [aeEncodeError, setAeEncodeError] = useState<string | null>(null);
  const [aeEncodeResult, setAeEncodeResult] = useState<AutoencoderEncodeResponse | null>(null);

  // the D-MPNN lane (DL-4 — the roadmap's row 16: the molecular
  // property predictor over a registered dataset's own SMILES column;
  // its own dataset pick so the tabular chain's selection stands alone)
  const [mpnnDatasetId, setMpnnDatasetId] = useState<string | null>(null);
  const [mpnnView, setMpnnView] = useState<DatasetView | null>(null);
  const [mpnnViewError, setMpnnViewError] = useState<string | null>(null);
  const [mpnnSmilesColumn, setMpnnSmilesColumn] = useState<string>("");
  const [mpnnHidden, setMpnnHidden] = useState(32);
  const [mpnnDepth, setMpnnDepth] = useState(3);
  const [mpnnEpochs, setMpnnEpochs] = useState(200);
  const [mpnnLr, setMpnnLr] = useState(0.02);
  const [mpnnSeed, setMpnnSeed] = useState(0);
  const [mpnnTrainBusy, setMpnnTrainBusy] = useState(false);
  const [mpnnTrainJobId, setMpnnTrainJobId] = useState<string | null>(null);
  const [mpnnTrainError, setMpnnTrainError] = useState<string | null>(null);
  const [mpnnTrainResult, setMpnnTrainResult] = useState<MpnnTrainResult | null>(null);
  const [mpnnPredictText, setMpnnPredictText] = useState("");
  const [mpnnPredictBusy, setMpnnPredictBusy] = useState(false);
  const [mpnnPredictError, setMpnnPredictError] = useState<string | null>(null);
  const [mpnnPredictResult, setMpnnPredictResult] = useState<MpnnPredictResponse | null>(null);

  // the tabular Transformer's architecture controls (part18 §5.4 — the
  // family rides the SAME tabular train lane, its own keys compose here)
  const [ttrDModel, setTtrDModel] = useState(32);
  const [ttrHeads, setTtrHeads] = useState(4);
  const [ttrBlocks, setTtrBlocks] = useState(2);
  const [ttrFeedforward, setTtrFeedforward] = useState(64);
  // the tabular Transformer rides the same lane with its OWN architecture
  // keys — the Configure Model stage swaps its controls on this flag
  const isTransformerNetwork = selectedNetwork.startsWith("tabular_transformer");

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

  useEffect(() => {
    void refreshRegistry();
  }, [refreshRegistry]);

  // stage 4: the live network catalog (one fetch — the signature
  // defaults are the backend's own, never re-declared here; since
  // part13 Phase 5 the same response carries the live GNN vocabulary)
  useEffect(() => {
    let cancelled = false;
    dlApi
      .networks()
      .then((response) => {
        if (!cancelled) {
          setNetworks(response.networks);
          setGnnVocab(response.gnn);
          setCatalogError(null);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setCatalogError(describeError(error));
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const selectDataset = useCallback(async (datasetId: string) => {
    setSelectedId(datasetId);
    setSelectionError(null);
    // dataset-bound stage state resets with the selection
    setTrainedModel(null);
    setTrainJobId(null);
    setTrainError(null);
    setMetricsPoints([]);
    setMonitorError(null);
    setPredictResult(null);
    setPredictError(null);
    setFigureArtifact(null);
    setFigureError(null);
    setFigureDigestStatus(null);
    setCheckpointArtifact(null);
    setCheckpointError(null);
    setCheckpointDigestStatus(null);
    setExportArtifact(null);
    setDigestStatus(null);
    setSelectedNetwork("");
    try {
      const view = await datasetsApi.view(datasetId);
      setSelected(view);
    } catch (error) {
      setSelected(null);
      setSelectionError(describeError(error));
    }
  }, []);

  // stage 7: the assembled hyperparameters (the page's controls →
  // the backend's own vocabulary, validated before any request)
  const buildHyperparams = useCallback(():
    Record<string, unknown> | null => {
    let hiddenLayers: number[] | null = null;
    try {
      hiddenLayers = validateHiddenLayers(hiddenLayersText);
      setHiddenLayersError(null);
    } catch (error) {
      setHiddenLayersError(
        error instanceof Error ? error.message : String(error),
      );
      throw error;
    }
    if (hiddenLayers === null && hiddenLayersText.trim() !== "") {
      return null;
    }
    const params: Record<string, unknown> = {
      epochs,
      batch_size: batchSize,
      learning_rate: learningRate,
      weight_decay: weightDecay,
      patience,
      val_split: valSplit,
      random_state: randomState,
      device,
    };
    // part18 §5.4 — the tabular Transformer family rides the same lane
    // with its OWN architecture keys (d_model/heads/n_blocks/
    // dim_feedforward); the MLP-only keys never ride its requests (the
    // backend validates per-network and 422s an unknown key honestly)
    if (selectedNetwork.startsWith("tabular_transformer")) {
      params["d_model"] = ttrDModel;
      params["heads"] = ttrHeads;
      params["n_blocks"] = ttrBlocks;
      params["dim_feedforward"] = ttrFeedforward;
      return params;
    }
    if (hiddenLayers !== null) {
      params["hidden_layers"] = hiddenLayers;
    }
    // the architecture overrides ride the same dict the backend
    // validates (dropout/activation/batchnorm ARE the Configure
    // Model stage's controls)
    params["dropout"] = dropout;
    params["activation"] = activation;
    params["batchnorm"] = batchnorm;
    return params;
  }, [
    hiddenLayersText,
    epochs,
    batchSize,
    learningRate,
    weightDecay,
    patience,
    valSplit,
    randomState,
    device,
    dropout,
    activation,
    batchnorm,
    selectedNetwork,
    ttrDModel,
    ttrHeads,
    ttrBlocks,
    ttrFeedforward,
  ]);

  // stage 7: train (job mode — the backend's default lane); with the
  // DL-1 search lane armed, the SAME route rides `tune: true` with
  // the parsed search block (the client pre-flight above mirrors the
  // route's own contract so a doomed search is refused here)
  const handleTrain = useCallback(async () => {
    if (selectedId === null || selectedNetwork === "") {
      return;
    }
    let hyperparams: Record<string, unknown> | null = null;
    try {
      hyperparams = buildHyperparams();
      if (hyperparams === null) {
        return;
      }
      setHiddenLayersError(null);
    } catch (error) {
      setHiddenLayersError(
        error instanceof Error ? error.message : String(error),
      );
      return;
    }
    let tune: DLTrainOptionsTune | null = null;
    if (tuneEnabled) {
      try {
        const search = parseTuneSearch(tuneSearchText);
        tune = {
          search,
          nTrials: tuneNTrials,
          sampler: tuneSampler,
          seed: tuneSeed,
        };
      } catch (error) {
        setTrainError(
          error instanceof Error ? error.message : String(error),
        );
        return;
      }
    }
    setTrainBusy(true);
    setTrainError(null);
    setTrainJobId(null);
    setTrainedModel(null);
    setTuneResult(null);
    setMetricsPoints([]);
    setMonitorError(null);
    try {
      const response = await dlApi.train({
        datasetId: selectedId,
        network: selectedNetwork,
        hyperparams,
        tune,
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
  }, [selectedId, selectedNetwork, buildHyperparams, tuneEnabled, tuneSearchText, tuneNTrials, tuneSampler, tuneSeed]);

  // the train job's result IS `{model: DLModelView}` on the
  // fixed-config lane and `{model, tune}` on the DL-1 search lane —
  // the authoritative REST relay, parsed at runtime
  const handleTrainResult = useCallback((result: unknown) => {
    try {
      const parsed = parseDLTuneResult(result, "dl train result");
      setTrainedModel(parsed.model);
      setTuneResult(parsed.tune);
    } catch (error) {
      setTrainError(describeError(error));
    }
  }, []);

  // stage 8: the Monitor — the REAL per-epoch telemetry envelopes
  // (training.metrics / training.progress) for THIS train job, on the
  // live WS bus; a malformed envelope counts, it is never faked
  useEffect(() => {
    if (trainJobId === null) {
      return;
    }
    const stream = getEventStream();
    const unsubscribe = stream.subscribe((envelope) => {
      if (envelope.job_id !== trainJobId) {
        return;
      }
      if (envelope.event_type === "training.metrics") {
        try {
          const point = parseTrainingMetricsPayload(
            envelope.payload,
            "training.metrics",
          );
          setMetricsPoints((prev) => [...prev, point]);
        } catch (error) {
          setMonitorError(describeError(error));
        }
      }
    });
    return unsubscribe;
  }, [trainJobId]);

  // stage 9: predict (dataset mode, sync — the SINGLE session
  // prediction path; a model without predict_proba answers the
  // backend's own 422, surfaced verbatim, no invented probabilities)
  const handlePredict = useCallback(async () => {
    if (trainedModel === null) {
      return;
    }
    setPredictBusy(true);
    setPredictError(null);
    setPredictResult(null);
    try {
      setPredictResult(
        await mlApi.predict({
          modelId: trainedModel.model_id,
          datasetId: trainedModel.dataset_id,
          predictProba,
        }),
      );
    } catch (error) {
      setPredictError(describeError(error));
    } finally {
      setPredictBusy(false);
    }
  }, [trainedModel, predictProba]);

  // the shared digest-verified download lane (the artifacts client's
  // own raw lane — reused by the figure, the checkpoint and the export)
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

  // stage 10: the loss-curve figure artifact (the REAL §17 export
  // stack — the telemetry the Monitor actually accumulated)
  const handleFigure = useCallback(async () => {
    setFigureBusy(true);
    setFigureError(null);
    setFigureArtifact(null);
    setFigureDigestStatus(null);
    try {
      const trainLoss = metricsPoints.map((p) => p.train_loss);
      const valLoss = metricsPoints.map((p) => p.val_loss);
      setFigureArtifact(
        await dlApi.lossCurveFigure({
          trainLoss,
          valLoss,
          format: figureFormat,
        }),
      );
    } catch (error) {
      setFigureError(describeError(error));
    } finally {
      setFigureBusy(false);
    }
  }, [metricsPoints, figureFormat]);

  const handleFigureDownload = useCallback(async () => {
    if (figureArtifact === null) {
      return;
    }
    setFigureDigestStatus(null);
    try {
      setFigureDigestStatus(await verifiedDownload(figureArtifact));
    } catch (error) {
      setFigureDigestStatus(describeError(error));
    }
  }, [figureArtifact, verifiedDownload]);

  // stage 11a: the checkpoint (the joblib bundle + metadata sidecar)
  const handleSaveCheckpoint = useCallback(async () => {
    if (trainedModel === null) {
      return;
    }
    setCheckpointBusy(true);
    setCheckpointError(null);
    setCheckpointArtifact(null);
    setCheckpointDigestStatus(null);
    try {
      // the metrics the checkpoint's sidecar carries: the best
      // validation loss the Monitor actually saw (never invented)
      const bestVals = metricsPoints
        .map((p) => p.best_val)
        .filter((v): v is number => v !== null && Number.isFinite(v));
      const metrics: Record<string, unknown> =
        bestVals.length > 0 ? { best_val_loss: Math.min(...bestVals) } : {};
      setCheckpointArtifact(
        await dlApi.saveCheckpoint(trainedModel.model_id, metrics),
      );
    } catch (error) {
      setCheckpointError(describeError(error));
    } finally {
      setCheckpointBusy(false);
    }
  }, [trainedModel, metricsPoints]);

  const handleCheckpointDownload = useCallback(async () => {
    if (checkpointArtifact === null) {
      return;
    }
    setCheckpointDigestStatus(null);
    try {
      setCheckpointDigestStatus(await verifiedDownload(checkpointArtifact));
    } catch (error) {
      setCheckpointDigestStatus(describeError(error));
    }
  }, [checkpointArtifact, verifiedDownload]);

  // stage 11b: the predictions export over the LAST predictions block
  const handleExport = useCallback(async () => {
    if (predictResult === null) {
      return;
    }
    setExportBusy(true);
    setExportError(null);
    setExportArtifact(null);
    setDigestStatus(null);
    try {
      const built = exportSource === "evaluation"
        ? buildEvaluationExportRows(predictResult)
        : buildPredictionExportRows(predictResult);
      if (built.rows.length > TABLE_EXPORT_MAX_ROWS) {
        throw new Error(
          `the export set carries ${built.rows.length} rows; the table export route's own cap is ${TABLE_EXPORT_MAX_ROWS}`,
        );
      }
      if (built.rows.length === 0) {
        throw new Error(
          exportSource === "evaluation"
            ? "the evaluation block carries no rows to export (dataset-mode answers only)"
            : "the prediction block carries no rows to export",
        );
      }
      const request: VisualizationTableRequest = {
        rows: built.rows,
        format: exportFormat,
        filename: exportSource === "evaluation" ? "dl-evaluation-sample" : "dl-predictions",
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
  }, [predictResult, exportFormat, exportSource]);

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

  // stage 11c: the efficiency lab (DL-3 — the roadmap's row 15).
  // The honest pre-flight mirrors the route's own guards: the prune
  // lane's level must PARSE and land in [0, 1) (never a silent
  // default), the quantize lanes never carry one; only then does the
  // request ride.  The report renders per-layer with the mode's own
  // fields, and the artifact downloads through the same
  // digest-verified lane every other artifact rides.
  const handleEfficiency = useCallback(async () => {
    if (trainedModel === null) {
      return;
    }
    let level: number | null = null;
    if (efficiencyMode === "prune") {
      const parsed = Number(efficiencyLevelText);
      if (!Number.isFinite(parsed) || parsed < 0 || parsed >= 1) {
        setEfficiencyError(
          `the prune lane's level must land in [0, 1) — "${efficiencyLevelText}" does not parse there`,
        );
        return;
      }
      level = parsed;
    }
    setEfficiencyBusy(true);
    setEfficiencyError(null);
    setEfficiencyResult(null);
    setEfficiencyDigestStatus(null);
    try {
      const result = await dlApi.efficiency({
        modelId: trainedModel.model_id,
        mode: efficiencyMode,
        level,
        scope: efficiencyScope,
        artifact: true,
      });
      setEfficiencyResult(result);
    } catch (error) {
      setEfficiencyError(describeError(error));
    } finally {
      setEfficiencyBusy(false);
    }
  }, [trainedModel, efficiencyMode, efficiencyLevelText, efficiencyScope]);

  const handleEfficiencyDownload = useCallback(async () => {
    const artifact = efficiencyResult?.artifact ?? null;
    if (artifact === null) {
      return;
    }
    setEfficiencyDigestStatus(null);
    try {
      setEfficiencyDigestStatus(await verifiedDownload(artifact));
    } catch (error) {
      setEfficiencyDigestStatus(describeError(error));
    }
  }, [efficiencyResult, verifiedDownload]);

  // the GNN lane's train (job mode — the graph trainer's own lane):
  // the inline block parses + validates client-side BEFORE any
  // request; the synthetic fixture rides the backend's own generator
  const handleGnnTrain = useCallback(async () => {
    setGnnTrainBusy(true);
    setGnnTrainError(null);
    setGnnTrainJobId(null);
    setGnnTrainResult(null);
    try {
      let inline: GnnInlineSource | null = null;
      if (gnnSource === "inline") {
        inline = parseGnnInlineGraphs(gnnGraphsText);
        if (inline === null) {
          throw new Error(
            "the inline source needs a graphs block — {\"graphs\": " +
              '[{"x": [[…]], "edge_index": [[…], […]}], "labels": […]}',
          );
        }
        setGnnGraphsError(null);
      }
      const response = await dlApi.gnnTrain({
        syntheticN: gnnSource === "synthetic" ? gnnSyntheticN : undefined,
        seed: gnnSeed,
        inline,
        conv: gnnConv,
        pool: gnnPool,
        hidden: gnnHidden,
        layers: gnnLayers,
        lr: gnnLr,
        epochs: gnnEpochs,
        weightDecay: gnnWeightDecay,
        heads: gnnHeads,
        ratio: gnnRatio,
        mode: "job",
      });
      if (response.mode === "job") {
        setGnnTrainJobId(response.job.job_id);
      }
    } catch (error) {
      setGnnTrainError(describeError(error));
    } finally {
      setGnnTrainBusy(false);
    }
  }, [
    gnnSource,
    gnnSyntheticN,
    gnnSeed,
    gnnGraphsText,
    gnnConv,
    gnnPool,
    gnnHidden,
    gnnLayers,
    gnnLr,
    gnnEpochs,
    gnnWeightDecay,
    gnnHeads,
    gnnRatio,
  ]);

  // the gnn-train job's COMPLETED result is the full train body —
  // the authoritative REST relay, parsed at runtime
  const handleGnnTrainResult = useCallback((result: unknown) => {
    try {
      setGnnTrainResult(parseGnnTrainResult(result, "dl gnn-train result"));
    } catch (error) {
      setGnnTrainError(describeError(error));
    }
  }, []);

  // the GNN lane's predict — the graph models' OWN prediction path
  // (never the tabular /ml/predict, which refuses graph models)
  const handleGnnPredict = useCallback(async () => {
    if (gnnTrainResult === null) {
      return;
    }
    setGnnPredictBusy(true);
    setGnnPredictError(null);
    setGnnPredictResult(null);
    try {
      const inline = parseGnnInlineGraphs(gnnPredictText, "predict");
      if (inline === null) {
        throw new Error(
          "the predict lane needs an inline graphs block — the same " +
            '{"graphs": […]} JSON the train lane accepts',
        );
      }
      setGnnPredictResult(
        await dlApi.gnnPredict({
          modelId: gnnTrainResult.model.model_id,
          graphs: inline.graphs,
          probabilities: gnnPredictProba,
        }),
      );
    } catch (error) {
      setGnnPredictError(describeError(error));
    } finally {
      setGnnPredictBusy(false);
    }
  }, [gnnTrainResult, gnnPredictText, gnnPredictProba]);

  // DL-4: the D-MPNN lane's own dataset pick — the molecular lane
  // reads a registered dataset's own SMILES column (its own selection
  // so the tabular chain's selected dataset stands untouched)
  const handleMpnnDatasetPick = useCallback(async (datasetId: string) => {
    setMpnnDatasetId(datasetId);
    setMpnnView(null);
    setMpnnViewError(null);
    setMpnnSmilesColumn("");
    try {
      const view = await datasetsApi.view(datasetId);
      setMpnnView(view);
    } catch (error) {
      setMpnnViewError(describeError(error));
    }
  }, []);

  // DL-4: the D-MPNN train — the REAL numpy chemprop-style engine
  // over the dataset's own SMILES column (job mode with HONEST
  // stage-level telemetry; the held-out carve rides the result)
  const handleMpnnTrain = useCallback(async () => {
    if (mpnnDatasetId === null || mpnnSmilesColumn === "") {
      return;
    }
    setMpnnTrainBusy(true);
    setMpnnTrainError(null);
    setMpnnTrainJobId(null);
    setMpnnTrainResult(null);
    try {
      const response = await dlApi.mpnnTrain({
        datasetId: mpnnDatasetId,
        smilesColumn: mpnnSmilesColumn,
        hiddenSize: mpnnHidden,
        depth: mpnnDepth,
        epochs: mpnnEpochs,
        lr: mpnnLr,
        seed: mpnnSeed,
        mode: "job",
      });
      if (response.mode === "job") {
        setMpnnTrainJobId(response.job.job_id);
      }
    } catch (error) {
      setMpnnTrainError(describeError(error));
    } finally {
      setMpnnTrainBusy(false);
    }
  }, [mpnnDatasetId, mpnnSmilesColumn, mpnnHidden, mpnnDepth, mpnnEpochs, mpnnLr, mpnnSeed]);

  // the mpnn-train job's COMPLETED result is the full train body —
  // the authoritative REST relay, parsed at runtime
  const handleMpnnTrainResult = useCallback((result: unknown) => {
    try {
      setMpnnTrainResult(parseMpnnTrainResult(result, "dl mpnn-train result"));
    } catch (error) {
      setMpnnTrainError(describeError(error));
    }
  }, []);

  // DL-4: the molecular models' OWN prediction path over raw SMILES
  // (one per line; the honest pre-flight: the block needs at least
  // one non-empty line before any request rides)
  const handleMpnnPredict = useCallback(async () => {
    if (mpnnTrainResult === null) {
      return;
    }
    setMpnnPredictBusy(true);
    setMpnnPredictError(null);
    setMpnnPredictResult(null);
    try {
      const smiles = mpnnPredictText
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => line.length > 0);
      if (smiles.length === 0) {
        throw new Error(
          "the predict lane needs at least one SMILES line — one per row",
        );
      }
      setMpnnPredictResult(
        await dlApi.mpnnPredict({
          modelId: mpnnTrainResult.model.model_id,
          smiles,
        }),
      );
    } catch (error) {
      setMpnnPredictError(describeError(error));
    } finally {
      setMpnnPredictBusy(false);
    }
  }, [mpnnTrainResult, mpnnPredictText]);

  // DL-2: the edge-attribution lane — the same inline-graphs paste
  // the predict card rides, through the gnnxai engine; the
  // pre-flight mirrors the route's own contract (labels ride the
  // pgexplainer lane only, one label per graph, the iters knob on
  // the optimizer explainers only) so a doomed request is refused
  // here, honestly, never fired
  const handleGnnExplain = useCallback(async () => {
    if (gnnTrainResult === null) {
      return;
    }
    setGnnExplainBusy(true);
    setGnnExplainError(null);
    setGnnExplainResult(null);
    try {
      const parsed = parseGnnInlineGraphs(gnnExplainText, "predict");
      if (parsed === null || parsed.graphs.length === 0) {
        throw new Error(
          'the explain lane needs a graphs block — {"graphs": [...]} ' +
            "(the same shape the predict card rides, no labels)",
        );
      }
      let labels: number[] | null = null;
      const labelsTrimmed = gnnExplainLabelsText.trim();
      if (labelsTrimmed !== "") {
        if (gnnExplainMethod !== "pgexplainer") {
          throw new Error(
            "labels ride the pgexplainer lane only (its scorer trains " +
              "over the request's own labeled graphs)",
          );
        }
        let parsedLabels: unknown;
        try {
          parsedLabels = JSON.parse(labelsTrimmed);
        } catch (error) {
          throw new Error(
            `the labels block is not valid JSON — ${
              error instanceof Error ? error.message : String(error)
            }`,
          );
        }
        if (
          !Array.isArray(parsedLabels) ||
          parsedLabels.some(
            (label) =>
              typeof label !== "number" || !Number.isInteger(label),
          )
        ) {
          throw new Error(
            "the labels block must be a JSON list of integers",
          );
        }
        labels = parsedLabels as number[];
        if (labels.length !== parsed.graphs.length) {
          throw new Error(
            `labels carries ${labels.length} entries for ` +
              `${parsed.graphs.length} graphs — exactly one per graph`,
          );
        }
      }
      const itersText = gnnExplainItersText.trim();
      let iters: number | null = null;
      if (itersText !== "") {
        if (
          gnnExplainMethod !== "gnnexplainer" &&
          gnnExplainMethod !== "pgexplainer"
        ) {
          throw new Error(
            "the iters knob rides the optimizer explainers only " +
              "(gnnexplainer/pgexplainer); subgraphx/attention carry " +
              "their own sampling scales",
          );
        }
        iters = Number(itersText);
        if (!Number.isInteger(iters) || (iters ?? 0) < 10 || (iters ?? 0) > 2000) {
          throw new Error(
            "iters must be an integer in [10, 2000] (the route's own bound)",
          );
        }
      }
      const targetText = gnnExplainTargetText.trim();
      let target: number | null = null;
      if (targetText !== "") {
        target = Number(targetText);
        if (!Number.isInteger(target) || target < 0) {
          throw new Error(
            "the target class must be a non-negative integer (the model's own class index)",
          );
        }
      }
      const response = await dlApi.gnnExplain({
        modelId: gnnTrainResult.model.model_id,
        graphs: parsed.graphs,
        method: gnnExplainMethod,
        topK: gnnExplainTopK,
        seed: gnnExplainSeed,
        target,
        labels,
        iters,
      });
      setGnnExplainResult(response);
    } catch (error) {
      setGnnExplainError(describeError(error));
    } finally {
      setGnnExplainBusy(false);
    }
  }, [
    gnnTrainResult,
    gnnExplainText,
    gnnExplainMethod,
    gnnExplainTopK,
    gnnExplainSeed,
    gnnExplainItersText,
    gnnExplainTargetText,
    gnnExplainLabelsText,
  ]);

  // the sequence-CNN lane's train (job mode — the estimator's own
  // on_epoch drives the per-epoch training.metrics stream)
  const handleCnnTrain = useCallback(async () => {
    setCnnTrainBusy(true);
    setCnnTrainError(null);
    setCnnTrainJobId(null);
    setCnnTrainResult(null);
    setCnnInlineError(null);
    try {
      let inline: { sequences: (number[][])[]; labels: number[] } | undefined;
      if (cnnSource === "inline") {
        inline = parseCnnInlineSequences(cnnInlineText, "train");
        if (inline === undefined) {
          throw new Error(
            "the inline lane needs a sequences block — " +
              '{"sequences": [[[…steps…], …], …], "labels": [...]}',
          );
        }
      }
      const response = await dlApi.cnnTrain({
        syntheticN: cnnSource === "synthetic" ? cnnSyntheticN : undefined,
        seed: cnnSeed,
        inline,
        task: cnnTask,
        channels: cnnChannels,
        kernel: cnnKernel,
        nLayers: cnnLayers,
        epochs: cnnEpochs,
        batchSize: 32,
        learningRate: cnnLr,
        weightDecay: 1e-4,
        patience: 15,
        valSplit: 0.15,
        randomState: cnnSeed,
      });
      if (response.mode === "job") {
        setCnnTrainJobId(response.job.job_id);
      } else {
        setCnnTrainResult(response.result);
      }
    } catch (error) {
      setCnnTrainError(describeError(error));
      if (error instanceof Error && error.message.includes("inline")) {
        setCnnInlineError(error.message);
      }
    } finally {
      setCnnTrainBusy(false);
    }
  }, [
    cnnSource, cnnSyntheticN, cnnSeed, cnnTask, cnnChannels, cnnKernel,
    cnnLayers, cnnEpochs, cnnLr, cnnInlineText,
  ]);

  // the cnn-train job's COMPLETED result relays into the sequence
  // model card (the same one-DTO parser as the sync lane)
  const handleCnnTrainResult = useCallback((result: unknown) => {
    try {
      setCnnTrainResult(parseCnnTrainResult(result, "dl cnn-train result"));
    } catch (error) {
      setCnnTrainError(describeError(error));
    }
  }, []);

  // the CNN lane's predict — the sequence models' OWN prediction path
  // (never the tabular /ml/predict, which refuses sequence models)
  const handleCnnPredict = useCallback(async () => {
    if (cnnTrainResult === null) {
      return;
    }
    setCnnPredictBusy(true);
    setCnnPredictError(null);
    setCnnPredictResult(null);
    try {
      const sequences = parseCnnInlineSequences(cnnPredictText, "predict");
      if (sequences === undefined) {
        throw new Error(
          "the predict lane needs a sequences block — " +
            '{"sequences": [[[…steps…], …], …]}',
        );
      }
      setCnnPredictResult(
        await dlApi.cnnPredict({
          modelId: cnnTrainResult.model.model_id,
          sequences: sequences.sequences,
          probabilities: cnnPredictProba,
        }),
      );
    } catch (error) {
      setCnnPredictError(describeError(error));
    } finally {
      setCnnPredictBusy(false);
    }
  }, [cnnTrainResult, cnnPredictText, cnnPredictProba]);

  // the autoencoder lane's train (job mode over the SELECTED dataset)
  const handleAeTrain = useCallback(async () => {
    if (selectedId === null) {
      return;
    }
    let hidden: number[];
    try {
      hidden = parseAeHidden(aeHiddenText);
    } catch (error) {
      setAeTrainError(
        error instanceof Error ? error.message : String(error),
      );
      return;
    }
    setAeTrainBusy(true);
    setAeTrainError(null);
    setAeTrainJobId(null);
    setAeTrainResult(null);
    setAeEncodeResult(null);
    try {
      const response = await dlApi.autoencoderTrain({
        datasetId: selectedId,
        hiddenLayers: hidden,
        latentDim: aeLatentDim,
        epochs: aeEpochs,
        batchSize: 32,
        learningRate: 1e-3,
        weightDecay: 1e-4,
        patience: 15,
        valSplit: 0.15,
        randomState: 42,
      });
      if (response.mode === "job") {
        setAeTrainJobId(response.job.job_id);
      } else {
        setAeTrainResult(response.result);
      }
    } catch (error) {
      setAeTrainError(describeError(error));
    } finally {
      setAeTrainBusy(false);
    }
  }, [selectedId, aeHiddenText, aeLatentDim, aeEpochs]);

  // the autoencoder-train job's COMPLETED result relay
  const handleAeTrainResult = useCallback((result: unknown) => {
    try {
      setAeTrainResult(
        parseAutoencoderTrainResult(result, "dl autoencoder-train result"),
      );
    } catch (error) {
      setAeTrainError(describeError(error));
    }
  }, []);

  // the autoencoder's encode lane over the SAME selected dataset
  const handleAeEncode = useCallback(async () => {
    if (aeTrainResult === null || selectedId === null) {
      return;
    }
    setAeEncodeBusy(true);
    setAeEncodeError(null);
    try {
      setAeEncodeResult(
        await dlApi.autoencoderEncode({
          modelId: aeTrainResult.model.model_id,
          datasetId: selectedId,
          topK: 10,
        }),
      );
    } catch (error) {
      setAeEncodeError(describeError(error));
    } finally {
      setAeEncodeBusy(false);
    }
  }, [aeTrainResult, selectedId]);

  // the task gate: a network of the other task is disabled with the
  // backend's own reason (the honest 422 becomes a visible row)
  const networkRows = useMemo(
    () =>
      networks.map((entry) => ({
        entry,
        usable: selectedTask === null || entry.task === selectedTask,
      })),
    [networks, selectedTask],
  );

  const monitorStats = useMemo(() => {
    if (metricsPoints.length === 0) {
      return null;
    }
    const last = metricsPoints[metricsPoints.length - 1];
    if (last === undefined) {
      return null;
    }
    const bestVals = metricsPoints
      .map((p) => p.best_val)
      .filter((v): v is number => v !== null && Number.isFinite(v));
    const lastStale =
      metricsPoints.reduce((acc, p) => acc + (p.epoch === last.epoch ? p.stale : 0), 0);
    return {
      epochsSeen: metricsPoints.length,
      lastEpoch: last.epoch,
      totalEpochs: last.epochs,
      bestVal: bestVals.length > 0 ? Math.min(...bestVals) : null,
      stale: lastStale,
    };
  }, [metricsPoints]);

  const usableNetworks = networkRows.filter((row) => row.usable);
  const selectedEntry =
    networks.find((entry) => entry.network === selectedNetwork) ?? null;

  return (
    <div className="mx-auto max-w-7xl px-4 py-10">
      <div className="max-w-3xl">
        <h1 className="text-2xl font-bold tracking-tight text-[var(--color-foreground)]">
          Deep Learning
        </h1>
        <p className="mt-2 text-sm leading-relaxed text-[var(--color-muted-foreground)]">
          The complete sequential workflow — select dataset, validate,
          preprocess, select the torch architecture, configure the model
          and the training, train, monitor the live loss and validation
          curves, evaluate, inspect predictions, visualize, save the
          checkpoint, export — every step runs the real Python backend
          through the local API.
        </p>
        <p className="mt-2 text-xs leading-relaxed text-[var(--color-muted-foreground)]">
          State, honestly: the train lane runs as a managed background
          job whose REAL per-epoch `training.metrics` telemetry drives
          the Monitor stage; predictions ride the SINGLE session
          prediction path `/api/v1/ml/predict` (a DL-trained model
          registers in the same model store — never a second path); the
          checkpoint is the estimator&apos;s own joblib bundle with the
          full provenance metadata sidecar. The DL train route carries
          no preprocess toggle and no CV scores — both stated plainly
          in their own stages below. No feature on this page is a claim
          that was not executed.
        </p>
      </div>

      <div className="mt-6">
        {/* part16 §8 (Phase 5): the workspace's demo-data entry point */}
        <DemoDataSection
          workflow="deep-learning"
          onDatasetLoaded={() => void refreshRegistry()}
        />
      </div>

      <StageHeader
        index={1}
        title="Select Dataset"
        hint="the session registry, one dataset selected"
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
      </StageHeader>

      <StageHeader
        index={2}
        title="Validate"
        hint="read-only, from the registered dataset's own loader policy"
      >
        <Card>
          <CardContent>
            {selected === null ? (
              <p className="text-sm text-[var(--color-muted-foreground)]">
                Select a dataset above to see its validation view.
              </p>
            ) : (
              <div>
                <div className="flex flex-wrap gap-2">
                  <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                    {selected.shape[0]} rows × {selected.shape[1]} columns
                  </span>
                  <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                    target: {selected.target ?? "none (auto-detected)"}
                  </span>
                  <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                    task: {selected.task ?? "inferred"}
                  </span>
                  <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                    {selected.n_features} numeric features
                  </span>
                  <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                    {selected.dropped_columns.length} dropped
                  </span>
                  <span className="rounded-md bg-[var(--color-muted)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                    {selected.categorical_columns.length} categorical
                  </span>
                </div>
                {selectedTask === null ? (
                  <p className="mt-3 text-xs text-[var(--color-destructive)]">
                    The dataset&apos;s task is not one of the two the torch
                    families serve (classification / regression) — the DL
                    train route would refuse it with its own 422; no
                    network below is selectable until a suited dataset is.
                  </p>
                ) : null}
                <p className="mt-3 text-xs text-[var(--color-muted-foreground)]">
                  The validation view comes from the registered
                  dataset&apos;s own loader policy — target auto-detection,
                  task inference, non-numeric column dropping — this frame
                  does not re-derive or override any of it; the training
                  matrix itself never crosses the JSON boundary (§17.1&apos;s
                  control-plane rule).
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={3}
        title="Preprocess"
        hint="the torch estimator's own standardisation, fitted on the training rows"
      >
        <Card>
          <CardContent>
            {selected === null ? (
              <p className="text-sm text-[var(--color-muted-foreground)]">
                Select a dataset above — its numeric feature matrix is
                what the network trains on.
              </p>
            ) : (
              <p className="text-sm leading-relaxed text-[var(--color-muted-foreground)]">
                There is no preprocess toggle on this lane — that is the
                backend&apos;s own design, stated honestly: the DL train
                route (`POST /api/v1/dl/train`) passes the dataset&apos;s
                numeric feature matrix straight to the torch estimator,
                whose own `fit` standardises per-feature on the
                training rows only (and regression targets too), so the
                validation split never leaks into the scaling — the
                estimator&apos;s internal policy, not a page-side
                pretence. Non-numeric columns were already dropped by
                the loader (stage 2&apos;s count); the matrix the network
                sees is the {selected.n_features}-column numeric frame.
              </p>
            )}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={4}
        title="Select Architecture"
        hint="the live torch catalog, task-matched"
      >
        {catalogError !== null ? <ErrorBanner message={catalogError} /> : null}
        <div className="overflow-hidden rounded-lg border border-[var(--color-border)]">
          <div className="max-h-72 overflow-auto">
            <table className="w-full border-collapse text-xs">
              <thead className="sticky top-0 bg-[var(--color-muted)]">
                <tr>
                  {["Network", "Class", "Task", "Engine", "Signature defaults", ""].map((header) => (
                    <th key={header} scope="col" className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]">
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {networkRows.map(({ entry, usable }) => (
                  <tr
                    key={entry.network}
                    className={`border-t border-[var(--color-border)] bg-[var(--color-card)] ${
                      usable
                        ? "cursor-pointer hover:bg-[var(--color-accent)]/50"
                        : "opacity-60"
                    } ${entry.network === selectedNetwork ? "bg-[var(--color-accent)]" : ""}`}
                    onClick={usable ? () => setSelectedNetwork(entry.network) : undefined}
                  >
                    <td className="whitespace-nowrap px-3 py-1.5 font-medium text-[var(--color-foreground)]">{entry.network}</td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">{entry.class}</td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">{entry.task}</td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">{entry.engine}</td>
                    <td className="px-3 py-1.5 text-[var(--color-muted-foreground)]">
                      <span className="text-[10px]">
                        {Object.entries(entry.defaults)
                          .slice(0, 6)
                          .map(([k, v]) => `${k}=${formatValue(v)}`)
                          .join(" · ")}
                      </span>
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-[11px] text-[var(--color-muted-foreground)]">
                      {usable ? (
                        selectedNetwork === entry.network ? "selected" : ""
                      ) : (
                        <span title={`network ${entry.network} is a ${entry.task} family but the dataset is ${selected?.task}`}>
                          task mismatch — the backend&apos;s 422
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
                {networks.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="px-3 py-4 text-center text-[var(--color-muted-foreground)]">
                      {catalogError !== null
                        ? "the catalog request failed above"
                        : "the torch catalog carried no entries"}
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </div>
        <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
          The four tabular torch families from the backend&apos;s live
          catalog — the defaults column is read off the class
          signatures server-side (one source of truth, never
          re-declared here). A network whose task differs from the
          selected dataset&apos;s stays disabled with the backend&apos;s own
          refusal reason.
        </p>
        {selectedEntry !== null ? (
          <Card className="mt-3">
            <CardHeader>
              <CardTitle className="text-base">{selectedEntry.class}</CardTitle>
              <CardDescription>
                {selectedEntry.module} · engine {selectedEntry.engine} ·{" "}
                {selectedEntry.task}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-1">
                {Object.entries(selectedEntry.defaults).map(([key, value]) => (
                  <span key={key} className="rounded bg-[var(--color-muted)] px-1.5 py-0.5 text-[10px] text-[var(--color-muted-foreground)]">
                    {key}: {formatValue(value)}
                  </span>
                ))}
              </div>
              <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                The full signature defaults — Configure Model and
                Configure Training below start from these; only the
                overrides the page sends ride the request (the unknown
                key is the backend&apos;s honest 422, the vocabulary is{" "}
                {DL_HYPERPARAM_KEYS.join(", ")}).
              </p>
            </CardContent>
          </Card>
        ) : null}
      </StageHeader>

      <StageHeader
        index={5}
        title="Configure Model"
        hint="the architecture overrides (hidden layers, dropout, activation, batchnorm)"
      >
        <Card>
          <CardContent>
            <div className="flex flex-wrap items-end gap-3">
              {isTransformerNetwork ? (
                <>
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    d_model (divisible by heads)
                    <input
                      type="number"
                      min={8}
                      max={256}
                      step={8}
                      value={ttrDModel}
                      onChange={(event) => setTtrDModel(Number(event.target.value))}
                      className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                      aria-label="transformer d_model"
                    />
                  </label>
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    heads
                    <input
                      type="number"
                      min={1}
                      max={16}
                      value={ttrHeads}
                      onChange={(event) => setTtrHeads(Number(event.target.value))}
                      className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                      aria-label="transformer heads"
                    />
                  </label>
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    blocks
                    <input
                      type="number"
                      min={1}
                      max={6}
                      value={ttrBlocks}
                      onChange={(event) => setTtrBlocks(Number(event.target.value))}
                      className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                      aria-label="transformer blocks"
                    />
                  </label>
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    feedforward
                    <input
                      type="number"
                      min={8}
                      max={512}
                      step={8}
                      value={ttrFeedforward}
                      onChange={(event) => setTtrFeedforward(Number(event.target.value))}
                      className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                      aria-label="transformer feedforward"
                    />
                  </label>
                </>
              ) : (
                <>
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    hidden layers (JSON list)
                    <input
                      type="text"
                      value={hiddenLayersText}
                      onChange={(event) => setHiddenLayersText(event.target.value)}
                      placeholder="[64, 32] — empty = the signature default"
                      className="mt-1 w-64 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    />
                  </label>
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    activation
                    <select
                      value={activation}
                      onChange={(event) => setActivation(event.target.value)}
                      className="mt-1 w-28 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    >
                      {DL_ACTIVATIONS.map((token) => (
                        <option key={token} value={token}>{token}</option>
                      ))}
                    </select>
                  </label>
                  <label className="flex items-center gap-2 text-xs text-[var(--color-muted-foreground)]">
                    <input
                      type="checkbox"
                      checked={batchnorm}
                      onChange={(event) => setBatchnorm(event.target.checked)}
                      className="h-4 w-4"
                    />
                    batch normalisation
                  </label>
                </>
              )}
              <label className="text-xs text-[var(--color-muted-foreground)]">
                dropout
                <input
                  type="number"
                  min={0}
                  max={0.9}
                  step={0.05}
                  value={dropout}
                  onChange={(event) => setDropout(Number(event.target.value))}
                  className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
            </div>
            {hiddenLayersError !== null ? (
              <p className="mt-2 text-xs text-[var(--color-destructive)]">{hiddenLayersError}</p>
            ) : null}
            <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
              {isTransformerNetwork
                ? "The tabular Transformer (part18 §5.4) rides the same tabular train lane with its OWN architecture keys — d_model/heads/n_blocks/dim_feedforward compose the request; the MLP-only keys never ride it (the backend validates per-network and 422s an unknown key honestly). From scratch, every time: no pretrained molecular transformer is claimed anywhere."
                : "The overrides validate BEFORE any request fires — an invalid body never leaves the page. The activation tokens are the backend's own accepted vocabulary (`relu`, `tanh`, `gelu`, `sigmoid`); the estimator validates `dropout` in [0, 1) with its own actionable error — surfaced verbatim, never pre-faked."}
            </p>
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={6}
        title="Configure Training"
        hint="epochs, batch size, optimiser, early stopping, the validation split, seed, device"
      >
        <Card>
          <CardContent>
            <div className="flex flex-wrap items-end gap-3">
              <label className="text-xs text-[var(--color-muted-foreground)]">
                epochs
                <input
                  type="number"
                  min={1}
                  max={2000}
                  step={10}
                  value={epochs}
                  onChange={(event) => setEpochs(Number(event.target.value))}
                  className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                batch size
                <input
                  type="number"
                  min={1}
                  step={8}
                  value={batchSize}
                  onChange={(event) => setBatchSize(Number(event.target.value))}
                  className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                learning rate
                <input
                  type="number"
                  min={0.00001}
                  max={1}
                  step={0.0005}
                  value={learningRate}
                  onChange={(event) => setLearningRate(Number(event.target.value))}
                  className="mt-1 w-28 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                weight decay
                <input
                  type="number"
                  min={0}
                  max={0.1}
                  step={0.0001}
                  value={weightDecay}
                  onChange={(event) => setWeightDecay(Number(event.target.value))}
                  className="mt-1 w-28 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                patience (early stopping)
                <input
                  type="number"
                  min={1}
                  max={200}
                  value={patience}
                  onChange={(event) => setPatience(Number(event.target.value))}
                  className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                validation split
                <input
                  type="number"
                  min={0.05}
                  max={0.5}
                  step={0.05}
                  value={valSplit}
                  onChange={(event) => setValSplit(Number(event.target.value))}
                  className="mt-1 w-28 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                random seed
                <input
                  type="number"
                  min={0}
                  value={randomState}
                  onChange={(event) => setRandomState(Number(event.target.value))}
                  className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                device
                <select
                  value={device}
                  onChange={(event) => setDevice(event.target.value)}
                  className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                >
                  <option value="auto">auto</option>
                  <option value="cpu">cpu</option>
                  <option value="cuda">cuda</option>
                </select>
              </label>
            </div>
            <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
              The minibatch Adam loop&apos;s own controls — the early stop
              watches the held-out validation split (the `patience`
              stale-epoch counter the Monitor&apos;s `stale` column
              carries); the seed fixes both torch and NumPy RNGs so the
              same config reproduces (CPU).
            </p>
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={7}
        title="Train"
        hint="job mode — the torch network's own fit, tracked live"
      >
        <Card>
          <CardContent>
            {/* DL-1: the hyperparameter search lane — the same route
                with tune: true (the in-house TPE/Random engine under
                it); the fixed-config controls above stay the base the
                candidates ride on top of */}
            <div className="rounded-lg border border-[var(--color-border)] p-3">
              <div className="flex flex-wrap items-center gap-2">
                <label className="flex items-center gap-2 text-xs text-[var(--color-muted-foreground)]">
                  <input
                    type="checkbox"
                    aria-label="arm the hyperparameter search"
                    checked={tuneEnabled}
                    onChange={(event) => setTuneEnabled(event.target.checked)}
                    className="h-4 w-4"
                  />
                  search the hyperparameters instead of one fixed configuration
                </label>
                <span className="rounded-md bg-[var(--color-muted)] px-2 py-0.5 text-[11px] text-[var(--color-muted-foreground)]">
                  DL-1 · the in-house TPE/Random engine
                </span>
              </div>
              {tuneEnabled ? (
                <>
                  <div className="mt-3 grid gap-3 md:grid-cols-2">
                    <label className="text-xs text-[var(--color-muted-foreground)]">
                      the search block (one JSON object; each key over the
                      route&apos;s own constructor vocabulary — choices, or
                      low/high with optional log)
                      <textarea
                        aria-label="the tune search block"
                        value={tuneSearchText}
                        onChange={(event) => setTuneSearchText(event.target.value)}
                        rows={7}
                        spellCheck={false}
                        className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-2 font-mono text-xs"
                      />
                    </label>
                    <div className="flex flex-col justify-center gap-3 text-xs text-[var(--color-muted-foreground)]">
                      <label className="text-xs">
                        n_trials
                        <input
                          type="number"
                          min={1}
                          max={50}
                          value={tuneNTrials}
                          onChange={(event) => setTuneNTrials(Number(event.target.value))}
                          className="mt-1 ml-2 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                        />
                      </label>
                      <label className="text-xs">
                        sampler
                        <select
                          aria-label="the tune sampler"
                          value={tuneSampler}
                          onChange={(event) => setTuneSampler(event.target.value as "tpe" | "random")}
                          className="mt-1 ml-2 block w-28 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                        >
                          <option value="tpe">tpe</option>
                          <option value="random">random</option>
                        </select>
                      </label>
                      <label className="text-xs">
                        seed
                        <input
                          type="number"
                          min={0}
                          value={tuneSeed}
                          onChange={(event) => setTuneSeed(Number(event.target.value))}
                          className="mt-1 ml-2 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                        />
                      </label>
                    </div>
                  </div>
                  <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                    `POST /api/v1/dl/train` with `tune: true` — each candidate
                    is built and fit on a seeded train side and scored on the
                    held-out val side (accuracy / R²); cancellation lands
                    between trials; the winner refits on the full data and
                    registers like any trained model. The controls above stay
                    the base every candidate rides on top of (keep epochs low
                    for a fast search).
                  </p>
                </>
              ) : null}
              {tuneResult !== null ? (
                <div className="mt-3">
                  <p className="text-xs text-[var(--color-foreground)]">
                    <span className="font-semibold">
                      best of {tuneResult.n_trials} trial(s)
                    </span>{" "}
                    — score {formatValue(tuneResult.best_score)} · refit{" "}
                    {formatValue(tuneResult.refit_score)} · sampler{" "}
                    {tuneResult.sampler} (seed {tuneResult.seed}) · best{" "}
                    {JSON.stringify(tuneResult.best_params)}
                  </p>
                  <div className="mt-2 max-h-64 overflow-y-auto rounded-md border border-[var(--color-border)]">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="bg-[var(--color-muted)] text-left text-[var(--color-muted-foreground)]">
                          <th className="px-2 py-1.5 font-medium">#</th>
                          <th className="px-2 py-1.5 font-medium">params</th>
                          <th className="px-2 py-1.5 font-medium">score</th>
                          <th className="px-2 py-1.5 font-medium">objective</th>
                        </tr>
                      </thead>
                      <tbody>
                        {tuneResult.trials.map((trial) => {
                          // the study's own convention: the best is
                          // the smallest objective value (the negated
                          // score) — the argmin row IS the reported
                          // best configuration
                          const bestValue = Math.min(
                            ...tuneResult.trials.map((t) => t.value),
                          );
                          const isBest = trial.value === bestValue;
                          return (
                            <tr
                              key={trial.number}
                              data-best={isBest ? "true" : undefined}
                              className={`border-t border-[var(--color-border)] ${
                                isBest
                                  ? "bg-[var(--color-accent)]/40 font-medium"
                                  : ""
                              }`}
                            >
                              <td className="px-2 py-1.5">{trial.number}</td>
                              <td className="px-2 py-1.5 font-mono">
                                {JSON.stringify(trial.params)}
                              </td>
                              <td className="px-2 py-1.5">
                                {formatValue(-trial.value)}
                              </td>
                              <td className="px-2 py-1.5 text-[var(--color-muted-foreground)]">
                                {formatValue(trial.value)}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                    the objective column rides the study&apos;s own
                    smaller-is-better convention (the negated score); the
                    score column is the honest bigger-is-better verdict
                    (accuracy / R² on the held-out carve).
                  </p>
                </div>
              ) : null}
            </div>
            <div className="mt-4 flex flex-wrap items-center gap-3">
              <Button
                size="sm"
                disabled={
                  selectedId === null ||
                  selectedNetwork === "" ||
                  usableNetworks.every((row) => row.entry.network !== selectedNetwork) ||
                  trainBusy
                }
                onClick={() => void handleTrain()}
              >
                {trainBusy ? "Submitting…" : "Train the network"}
              </Button>
              {selectedEntry !== null ? (
                <span className="text-xs text-[var(--color-muted-foreground)]">
                  {selectedEntry.network} on {selected?.name ?? "—"} · job
                  mode
                </span>
              ) : null}
            </div>
            {trainError !== null ? <div className="mt-3"><ErrorBanner message={trainError} /></div> : null}
            {trainJobId !== null ? (
              <div className="mt-3">
                <JobRunCard jobId={trainJobId} resultLabel="the trained DL model view" onResult={handleTrainResult} />
              </div>
            ) : null}
            {trainedModel !== null ? <DLModelCard model={trainedModel} /> : null}
            <p className="mt-3 text-xs text-[var(--color-muted-foreground)]">
              The train lane rides `POST /api/v1/dl/train` in job mode —
              the torch network&apos;s own `fit(X, y, on_epoch=…)` with the
              per-epoch telemetry hook; the COMPLETED result (the
              registered model view) is relayed by the authoritative
              REST lane above, never the WS event payload alone.
            </p>
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={8}
        title="Monitor"
        hint="the live loss and validation curves — the REAL per-epoch telemetry"
      >
        <Card>
          <CardContent>
            {trainJobId === null ? (
              <p className="text-sm text-[var(--color-muted-foreground)]">
                Train above first — the Monitor accumulates the
                `training.metrics` envelopes the running job actually
                publishes (one per epoch, with the train/val losses, the
                best-so-far and the stale counter).
              </p>
            ) : (
              <div>
                <LossCurveChart points={metricsPoints} />
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <span className="flex items-center gap-1 text-[11px] text-[var(--color-muted-foreground)]">
                    <span className="inline-block h-0.5 w-5 bg-[var(--color-primary)]" aria-hidden /> train loss
                  </span>
                  <span className="flex items-center gap-1 text-[11px] text-[var(--color-muted-foreground)]">
                    <span className="inline-block h-0.5 w-5 border-t-2 border-dashed border-[var(--color-destructive)]" aria-hidden /> validation loss
                  </span>
                </div>
                {monitorStats !== null ? (
                  <p className="mt-3 text-xs text-[var(--color-muted-foreground)]">
                    {monitorStats.epochsSeen} epochs seen ·{" "}
                    epoch {monitorStats.lastEpoch}/{monitorStats.totalEpochs} ·
                    best validation loss{" "}
                    {formatLoss(monitorStats.bestVal)} ·{" "}
                    {monitorStats.stale} stale (the early-stop counter)
                  </p>
                ) : null}
                {metricsPoints.length > 0 ? (
                  <div className="mt-3 overflow-hidden rounded-lg border border-[var(--color-border)]">
                    <div className="max-h-64 overflow-auto">
                      <table className="w-full border-collapse text-[11px]">
                        <thead className="sticky top-0 bg-[var(--color-muted)]">
                          <tr>
                            {["epoch", "train loss", "validation loss", "best val", "stale"].map((header) => (
                              <th key={header} scope="col" className="whitespace-nowrap px-3 py-1.5 text-left font-semibold text-[var(--color-foreground)]">
                                {header}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {metricsPoints
                            .slice()
                            .reverse()
                            .map((point) => (
                              <tr key={`${point.epoch}-${point.epochs}`} className="border-t border-[var(--color-border)] bg-[var(--color-card)]">
                                <td className="whitespace-nowrap px-3 py-1 text-[var(--color-muted-foreground)]">{point.epoch}/{point.epochs}</td>
                                <td className="whitespace-nowrap px-3 py-1 text-[var(--color-muted-foreground)]">{formatLoss(point.train_loss)}</td>
                                <td className="whitespace-nowrap px-3 py-1 text-[var(--color-muted-foreground)]">{formatLoss(point.val_loss)}</td>
                                <td className="whitespace-nowrap px-3 py-1 text-[var(--color-muted-foreground)]">{formatLoss(point.best_val)}</td>
                                <td className="whitespace-nowrap px-3 py-1 text-[var(--color-muted-foreground)]">{point.stale}</td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ) : (
                  <p className="mt-3 text-xs text-[var(--color-muted-foreground)]">
                    No telemetry envelopes yet — the per-epoch metrics
                    arrive on the WS bus as the epochs complete.
                  </p>
                )}
                {monitorError !== null ? <div className="mt-3"><ErrorBanner message={monitorError} /></div> : null}
              </div>
            )}
            <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
              The curve and the table are the SAME points — the
              `training.metrics` envelopes the backend&apos;s per-epoch
              hook published for this job, parsed at runtime; nothing is
              interpolated or invented (a missing loss renders as —).
            </p>
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={9}
        title="Evaluate — Inspect Predictions"
        hint="the SINGLE session prediction path, dataset mode, sync"
      >
        <Card>
          <CardContent>
            <div className="flex flex-wrap items-center gap-3">
              <label className="flex items-center gap-2 text-xs text-[var(--color-muted-foreground)]">
                <input
                  type="checkbox"
                  checked={predictProba}
                  onChange={(event) => setPredictProba(event.target.checked)}
                  className="h-4 w-4"
                />
                probabilities (classifiers only)
              </label>
              <Button
                size="sm"
                disabled={trainedModel === null || predictBusy}
                onClick={() => void handlePredict()}
              >
                {predictBusy ? "Predicting…" : "Predict on the dataset"}
              </Button>
            </div>
            {predictError !== null ? <div className="mt-3"><ErrorBanner message={predictError} /></div> : null}
            {predictResult !== null ? (
              <div className="mt-3">
                <p className="text-xs text-[var(--color-muted-foreground)]">
                  model {predictResult.model_id} · {predictResult.n_rows}{" "}
                  rows · the model&apos;s own training dataset
                </p>
                <div className="mt-2 overflow-hidden rounded-lg border border-[var(--color-border)]">
                  <div className="max-h-64 overflow-auto">
                    <table className="w-full border-collapse text-[11px]">
                      <thead className="sticky top-0 bg-[var(--color-muted)]">
                        <tr>
                          <th scope="col" className="whitespace-nowrap px-3 py-1.5 text-left font-semibold text-[var(--color-foreground)]">row</th>
                          <th scope="col" className="whitespace-nowrap px-3 py-1.5 text-left font-semibold text-[var(--color-foreground)]">prediction</th>
                          {predictResult.probabilities !== null
                            ? predictResult.probabilities.classes.map((cls, i) => (
                                <th key={`${String(cls)}-${i}`} scope="col" className="whitespace-nowrap px-3 py-1.5 text-left font-semibold text-[var(--color-foreground)]">
                                  p({formatValue(cls)})
                                </th>
                              ))
                            : null}
                        </tr>
                      </thead>
                      <tbody>
                        {predictResult.predictions.map((prediction, i) => (
                          <tr key={i} className="border-t border-[var(--color-border)] bg-[var(--color-card)]">
                            <td className="whitespace-nowrap px-3 py-1 text-[var(--color-muted-foreground)]">{i}</td>
                            <td className="whitespace-nowrap px-3 py-1 font-medium text-[var(--color-foreground)]">{formatValue(prediction)}</td>
                            {predictResult.probabilities !== null
                              ? (predictResult.probabilities.rows[i] ?? []).map((p, j) => (
                                  <td key={j} className="whitespace-nowrap px-3 py-1 text-[var(--color-muted-foreground)]">
                                    {typeof p === "number" ? p.toFixed(4) : "—"}
                                  </td>
                                ))
                              : null}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            ) : (
              <p className="mt-3 text-sm text-[var(--color-muted-foreground)]">
                Train above first — predictions run through `POST
                /api/v1/ml/predict` (the single session prediction path
                every trainer shares), dataset mode, sync; a model
                without `predict_proba` answers the backend&apos;s own
                422 — surfaced verbatim, no invented probabilities.
              </p>
            )}
            {predictResult !== null ? (
              <EvaluationCharts
                result={predictResult}
                modelId={trainedModel?.model_id ?? ""}
                datasetId={trainedModel?.dataset_id ?? null}
              />
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={10}
        title="Visualize"
        hint="the loss-curve figure artifact through the real export stack"
      >
        <Card>
          <CardContent>
            <div className="flex flex-wrap items-end gap-3">
              <label className="text-xs text-[var(--color-muted-foreground)]">
                format
                <select
                  value={figureFormat}
                  onChange={(event) =>
                    setFigureFormat(event.target.value as typeof figureFormat)
                  }
                  className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                >
                  <option value="PNG">PNG</option>
                  <option value="SVG">SVG</option>
                  <option value="TIFF">TIFF</option>
                  <option value="JPEG">JPEG</option>
                </select>
              </label>
              <Button
                size="sm"
                disabled={metricsPoints.length < 2 || figureBusy}
                onClick={() => void handleFigure()}
              >
                {figureBusy ? "Registering…" : "Register the loss-curve figure"}
              </Button>
              {figureArtifact !== null ? (
                <Button size="sm" variant="outline" onClick={() => void handleFigureDownload()}>
                  <ArrowDownToLine aria-hidden className="h-4 w-4" />
                  Download {figureArtifact.filename}
                </Button>
              ) : null}
            </div>
            <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
              Registers the Monitor stage&apos;s actual train/validation
              series as a REAL figure artifact — `POST
              /api/v1/artifacts/figure` with the per-epoch points,
              built through the §17 four-format export stack
              (lossless TIFF codecs included), then downloaded through
              the token-header lane with the served digest verified.
            </p>
            {figureError !== null ? <div className="mt-3"><ErrorBanner message={figureError} /></div> : null}
            {figureArtifact !== null ? (
              <p className="mt-3 text-xs text-[var(--color-muted-foreground)]">
                artifact {figureArtifact.artifact_id} ·{" "}
                {figureArtifact.size_bytes} bytes · sha256{" "}
                {figureArtifact.sha256.slice(0, 16)}… — the download rides
                the token-header lane and verifies the served digest.
              </p>
            ) : null}
            {figureDigestStatus !== null ? (
              <p className="mt-2 text-xs font-medium text-[var(--color-foreground)]">{figureDigestStatus}</p>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={11}
        title="Save Checkpoint / Export"
        hint="the joblib bundle with the metadata sidecar + the digest-verified table export"
      >
        <Card>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <p className="text-xs font-semibold text-[var(--color-foreground)]">
                  Checkpoint (model persistence)
                </p>
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <Button
                    size="sm"
                    disabled={trainedModel === null || checkpointBusy}
                    onClick={() => void handleSaveCheckpoint()}
                  >
                    {checkpointBusy ? "Saving…" : "Save the checkpoint"}
                  </Button>
                  {checkpointArtifact !== null ? (
                    <Button size="sm" variant="outline" onClick={() => void handleCheckpointDownload()}>
                      <ArrowDownToLine aria-hidden className="h-4 w-4" />
                      Download {checkpointArtifact.filename}
                    </Button>
                  ) : null}
                </div>
                <p className="mt-2 text-xs leading-relaxed text-[var(--color-muted-foreground)]">
                  `POST /api/v1/artifacts/model` — the fitted estimator
                  saved by the backend&apos;s own persistence engine as the
                  standard joblib bundle with the full provenance
                  metadata sidecar (the network, the hyperparameters,
                  the feature order, the versions, the best validation
                  loss the Monitor actually saw): a model loaded two
                  years later still knows what it is.
                </p>
                {checkpointError !== null ? <div className="mt-2"><ErrorBanner message={checkpointError} /></div> : null}
                {checkpointArtifact !== null ? (
                  <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                    artifact {checkpointArtifact.artifact_id} ·{" "}
                    {checkpointArtifact.size_bytes} bytes · kind{" "}
                    {checkpointArtifact.kind} · sha256{" "}
                    {checkpointArtifact.sha256.slice(0, 16)}…
                  </p>
                ) : null}
                {checkpointDigestStatus !== null ? (
                  <p className="mt-2 text-xs font-medium text-[var(--color-foreground)]">{checkpointDigestStatus}</p>
                ) : null}
              </div>
              <div>
                <p className="text-xs font-semibold text-[var(--color-foreground)]">
                  Export (the predictions table)
                </p>
                <div className="mt-2 flex flex-wrap items-end gap-2">
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    source
                    <select
                      aria-label="export source"
                      value={exportSource}
                      onChange={(event) =>
                        setExportSource(event.target.value as typeof exportSource)
                      }
                      className="mt-1 w-44 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    >
                      <option value="predictions">predictions table</option>
                      <option value="evaluation">evaluation sample</option>
                    </select>
                  </label>
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    format
                    <select
                      value={exportFormat}
                      onChange={(event) =>
                        setExportFormat(event.target.value as typeof exportFormat)
                      }
                      className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    >
                      <option value="csv">CSV</option>
                      <option value="tsv">TSV</option>
                      <option value="xlsx">XLSX</option>
                    </select>
                  </label>
                  <Button
                    size="sm"
                    disabled={predictResult === null || exportBusy}
                    onClick={() => void handleExport()}
                  >
                    {exportBusy ? "Exporting…" : "Export the predictions"}
                  </Button>
                  {exportArtifact !== null ? (
                    <Button size="sm" variant="outline" onClick={() => void handleDownload()}>
                      <ArrowDownToLine aria-hidden className="h-4 w-4" />
                      Download {exportArtifact.filename}
                    </Button>
                  ) : null}
                </div>
                <p className="mt-2 text-xs leading-relaxed text-[var(--color-muted-foreground)]">
                  Exports the LAST predictions block from Evaluate — the
                  prediction column plus one probability column per
                  class — as a real registered table artifact (POST
                  /api/v1/visualization/table with the backend&apos;s own{" "}
                  {TABLE_EXPORT_MAX_ROWS}-row cap mirrored client-side),
                  then downloads it through the token-header lane with
                  the served digest verified.
                </p>
                {exportError !== null ? <div className="mt-2"><ErrorBanner message={exportError} /></div> : null}
                {exportArtifact !== null ? (
                  <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                    artifact {exportArtifact.artifact_id} ·{" "}
                    {exportArtifact.size_bytes} bytes · sha256{" "}
                    {exportArtifact.sha256.slice(0, 16)}…
                  </p>
                ) : null}
                {digestStatus !== null ? (
                  <p className="mt-2 text-xs font-medium text-[var(--color-foreground)]">{digestStatus}</p>
                ) : null}
              </div>
              <div className="md:col-span-2">
                <p className="text-xs font-semibold text-[var(--color-foreground)]">
                  Efficiency lab (DL-3 — quantization / pruning reports)
                </p>
                <div className="mt-2 flex flex-wrap items-end gap-2">
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    mode
                    <select
                      aria-label="the efficiency mode"
                      value={efficiencyMode}
                      onChange={(event) =>
                        setEfficiencyMode(
                          event.target.value as
                            | "quantize_int8"
                            | "quantize_nf4"
                            | "prune",
                        )
                      }
                      className="mt-1 w-36 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    >
                      <option value="quantize_int8">quantize_int8</option>
                      <option value="quantize_nf4">quantize_nf4</option>
                      <option value="prune">prune</option>
                    </select>
                  </label>
                  {efficiencyMode === "prune" ? (
                    <>
                      <label className="text-xs text-[var(--color-muted-foreground)]">
                        level (sparsity)
                        <input
                          type="number"
                          min={0}
                          max={0.99}
                          step={0.05}
                          value={efficiencyLevelText}
                          onChange={(event) =>
                            setEfficiencyLevelText(event.target.value)
                          }
                          className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                        />
                      </label>
                      <label className="text-xs text-[var(--color-muted-foreground)]">
                        scope
                        <select
                          aria-label="the pruning scope"
                          value={efficiencyScope}
                          onChange={(event) =>
                            setEfficiencyScope(
                              event.target.value as "local" | "global",
                            )
                          }
                          className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                        >
                          <option value="local">local</option>
                          <option value="global">global</option>
                        </select>
                      </label>
                    </>
                  ) : null}
                  <Button
                    size="sm"
                    disabled={trainedModel === null || efficiencyBusy}
                    onClick={() => void handleEfficiency()}
                  >
                    {efficiencyBusy ? "Measuring…" : "Run the efficiency pass"}
                  </Button>
                  {efficiencyResult?.artifact != null ? (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => void handleEfficiencyDownload()}
                    >
                      <ArrowDownToLine aria-hidden className="h-4 w-4" />
                      Download {efficiencyResult.artifact.filename}
                    </Button>
                  ) : null}
                </div>
                <p className="mt-2 text-xs leading-relaxed text-[var(--color-muted-foreground)]">
                  `POST /api/v1/dl/efficiency` — the trained network&apos;s
                  own weight matrices through the REAL engines: int8 absmax
                  or the NF4 block pack (the engine&apos;s own report:
                  bits/weight, RMSE, relative error, the honest memory line)
                  or magnitude pruning (local/global masks, the metric delta
                  scored on the model&apos;s own training rows through a
                  pruned COPY — the persisted estimator is never mutated).
                  The report registers as a downloadable table artifact.
                </p>
                {trainedModel === null ? (
                  <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                    Train a network in stage 6 first — the efficiency lab
                    reads the trained model&apos;s own weight matrices.
                  </p>
                ) : null}
                {efficiencyError !== null ? (
                  <div className="mt-2">
                    <ErrorBanner message={efficiencyError} />
                  </div>
                ) : null}
                {efficiencyResult !== null ? (
                  <div className="mt-3">
                    <p className="text-xs font-medium text-[var(--color-foreground)]">
                      {efficiencyResult.efficiencyMode === "prune" ? (
                        <>
                          target sparsity{" "}
                          {efficiencyResult.targetSparsity?.toFixed(2)} ·{" "}
                          achieved{" "}
                          {(efficiencyResult.achievedSparsity ?? 0).toFixed(3)}{" "}
                          ({efficiencyResult.nZeroed} of{" "}
                          {efficiencyResult.nWeights} weights zeroed) · scope{" "}
                          {efficiencyResult.scope}
                          {efficiencyResult.metric?.before != null &&
                          efficiencyResult.metric?.after != null ? (
                            <>
                              {" "}· score{" "}
                              {efficiencyResult.metric.before.toFixed(3)} →{" "}
                              {efficiencyResult.metric.after.toFixed(3)} (Δ{" "}
                              {(efficiencyResult.metric.after - efficiencyResult.metric.before).toFixed(3)})
                            </>
                          ) : null}
                        </>
                      ) : (
                        <>
                          method {efficiencyResult.method} · mean{" "}
                          {efficiencyResult.bitsPerWeight?.toFixed(2)}{" "}
                          bits/weight · memory{" "}
                          {(100 * (efficiencyResult.memorySavings ?? 0)).toFixed(1)}%
                          saved ({efficiencyResult.originalBits} →{" "}
                          {efficiencyResult.quantizedBits} bits over{" "}
                          {efficiencyResult.nWeights} weights)
                        </>
                      )}
                    </p>
                    <div className="mt-2 max-h-72 overflow-y-auto rounded-md border border-[var(--color-border)]">
                      <table className="w-full text-xs">
                        <thead className="sticky top-0 bg-[var(--color-muted)]">
                          <tr className="text-left text-[var(--color-muted-foreground)]">
                            <th scope="col" className="px-3 py-2">layer</th>
                            <th scope="col" className="px-3 py-2">shape</th>
                            <th scope="col" className="px-3 py-2">weights</th>
                            {efficiencyResult.efficiencyMode === "prune" ? (
                              <>
                                <th scope="col" className="px-3 py-2">zeroed</th>
                                <th scope="col" className="px-3 py-2">kept</th>
                                <th scope="col" className="px-3 py-2">sparsity</th>
                              </>
                            ) : (
                              <>
                                <th scope="col" className="px-3 py-2">bits/w</th>
                                <th scope="col" className="px-3 py-2">rmse</th>
                                <th scope="col" className="px-3 py-2">rel err</th>
                              </>
                            )}
                          </tr>
                        </thead>
                        <tbody>
                          {efficiencyResult.layers.map((row) => (
                            <tr
                              key={row.layer}
                              className="border-t border-[var(--color-border)] text-[var(--color-foreground)]"
                            >
                              <td className="px-3 py-1.5 font-mono">{row.layer}</td>
                              <td className="px-3 py-1.5">{row.shape.join("×")}</td>
                              <td className="px-3 py-1.5">{row.nWeights}</td>
                              {row.kind === "prune" ? (
                                <>
                                  <td className="px-3 py-1.5">{row.nZeroed}</td>
                                  <td className="px-3 py-1.5">{row.nKept}</td>
                                  <td className="px-3 py-1.5">{row.sparsity.toFixed(3)}</td>
                                </>
                              ) : (
                                <>
                                  <td className="px-3 py-1.5">{row.bitsPerWeight.toFixed(2)}</td>
                                  <td className="px-3 py-1.5">{row.rmse.toPrecision(3)}</td>
                                  <td className="px-3 py-1.5">{row.relativeError.toPrecision(3)}</td>
                                </>
                              )}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <p className="mt-2 text-xs leading-relaxed text-[var(--color-muted-foreground)]">
                      {efficiencyResult.note}
                    </p>
                    {efficiencyResult.artifact !== null ? (
                      <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                        artifact {efficiencyResult.artifact.artifact_id} ·{" "}
                        {efficiencyResult.artifact.size_bytes} bytes · sha256{" "}
                        {efficiencyResult.artifact.sha256.slice(0, 16)}…
                      </p>
                    ) : null}
                    {efficiencyDigestStatus !== null ? (
                      <p className="mt-1 text-xs font-medium text-[var(--color-foreground)]">
                        {efficiencyDigestStatus}
                      </p>
                    ) : null}
                  </div>
                ) : null}
              </div>
            </div>
          </CardContent>
        </Card>
      </StageHeader>

      {/*
       * The GNN/MPNN lane — part13 Phase 5's graph card, deliberately
       * OUTSIDE the §11.3 StageHeader chain (a different lineage: the
       * graph models' own lane over /api/v1/dl/gnn-train +
       * /api/v1/dl/gnn-predict, not the tabular workflow's stages).
       */}
      <section className="mt-12" aria-labelledby="gnn-lane-title">
        <div className="flex flex-wrap items-baseline gap-2 border-t border-[var(--color-border)] pt-6">
          <h2
            id="gnn-lane-title"
            className="text-lg font-semibold text-[var(--color-foreground)]"
          >
            GNN / MPNN — the graph lane
          </h2>
          <span className="text-xs text-[var(--color-muted-foreground)]">
            part13 Phase 5 · the REAL GraphClassifier (conv stack → pool →
            linear head) over inline graphs or the synthetic motif fixture
          </span>
        </div>
        <Card className="mt-3">
          <CardHeader>
            <CardTitle className="text-base">Train a graph model</CardTitle>
            <CardDescription>
              the source (synthetic motifs or inline graphs) + the
              architecture (conv · pool, the live vocabulary below) + the
              trainer’s own controls — job mode, stage-level telemetry
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-end gap-3">
              <label className="text-xs text-[var(--color-muted-foreground)]">
                source
                <select
                  value={gnnSource}
                  onChange={(event) =>
                    setGnnSource(event.target.value as typeof gnnSource)
                  }
                  className="mt-1 w-32 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                  aria-label="gnn source"
                >
                  <option value="synthetic">synthetic motifs</option>
                  <option value="inline">inline graphs</option>
                </select>
              </label>
              {gnnSource === "synthetic" ? (
                <label className="text-xs text-[var(--color-muted-foreground)]">
                  motif count (10–500)
                  <input
                    type="number"
                    min={10}
                    max={500}
                    value={gnnSyntheticN}
                    onChange={(event) => setGnnSyntheticN(Number(event.target.value))}
                    className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                  />
                </label>
              ) : null}
              <label className="text-xs text-[var(--color-muted-foreground)]">
                seed
                <input
                  type="number"
                  min={0}
                  value={gnnSeed}
                  onChange={(event) => setGnnSeed(Number(event.target.value))}
                  className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                conv
                <select
                  value={gnnConv}
                  onChange={(event) => setGnnConv(event.target.value as GnnConv)}
                  className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                  aria-label="gnn conv"
                >
                  {(gnnVocab?.convs ?? ["gcn", "sage", "gin", "gat", "edge"]).map(
                    (token) => (
                      <option key={token} value={token}>{token}</option>
                    ),
                  )}
                </select>
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                pool
                <select
                  value={gnnPool}
                  onChange={(event) => setGnnPool(event.target.value as GnnPool)}
                  className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                  aria-label="gnn pool"
                >
                  {(gnnVocab?.pools ?? ["mean", "sum", "max", "topk", "sag"]).map(
                    (token) => (
                      <option key={token} value={token}>{token}</option>
                    ),
                  )}
                </select>
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                hidden
                <input
                  type="number"
                  min={1}
                  max={128}
                  value={gnnHidden}
                  onChange={(event) => setGnnHidden(Number(event.target.value))}
                  className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                layers
                <input
                  type="number"
                  min={1}
                  max={6}
                  value={gnnLayers}
                  onChange={(event) => setGnnLayers(Number(event.target.value))}
                  className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                epochs
                <input
                  type="number"
                  min={1}
                  max={2000}
                  step={20}
                  value={gnnEpochs}
                  onChange={(event) => setGnnEpochs(Number(event.target.value))}
                  className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                lr
                <input
                  type="number"
                  min={0.001}
                  max={1}
                  step={0.01}
                  value={gnnLr}
                  onChange={(event) => setGnnLr(Number(event.target.value))}
                  className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                weight decay
                <input
                  type="number"
                  min={0}
                  max={1}
                  step={0.0001}
                  value={gnnWeightDecay}
                  onChange={(event) => setGnnWeightDecay(Number(event.target.value))}
                  className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                heads (gat)
                <input
                  type="number"
                  min={1}
                  max={8}
                  value={gnnHeads}
                  onChange={(event) => setGnnHeads(Number(event.target.value))}
                  className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                ratio (topk/sag)
                <input
                  type="number"
                  min={0.05}
                  max={0.95}
                  step={0.05}
                  value={gnnRatio}
                  onChange={(event) => setGnnRatio(Number(event.target.value))}
                  className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
            </div>
            {gnnSource === "inline" ? (
              <div className="mt-3">
                <label className="text-xs text-[var(--color-muted-foreground)]">
                  inline graphs + labels (JSON)
                  <textarea
                    value={gnnGraphsText}
                    onChange={(event) => setGnnGraphsText(event.target.value)}
                    rows={6}
                    placeholder='{"graphs": [{"x": [[1, 0], [0, 1]], "edge_index": [[0, 1], [1, 0]]}, …], "labels": [0, 1, …]}'
                    className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 font-mono text-xs"
                  />
                </label>
                {gnnGraphsError !== null ? (
                  <p className="mt-1 text-xs text-[var(--color-destructive)]">{gnnGraphsError}</p>
                ) : null}
                <p className="mt-1 text-xs text-[var(--color-muted-foreground)]">
                  Every row validates BEFORE any request fires — node
                  features as number arrays, one width per graph, 2×E
                  endpoints within the graph’s node count, one integer label
                  per graph; a malformed block never leaves the page.
                </p>
              </div>
            ) : null}
            <div className="mt-3 flex flex-wrap items-center gap-3">
              <Button
                size="sm"
                disabled={gnnTrainBusy || (gnnSource === "inline" && gnnGraphsText.trim() === "")}
                onClick={() => void handleGnnTrain()}
              >
                {gnnTrainBusy ? "Submitting…" : "Train the graph model"}
              </Button>
              <span className="text-xs text-[var(--color-muted-foreground)]">
                {gnnSource === "synthetic"
                  ? `synthetic motif fixture · ${gnnSyntheticN} graphs · seed ${gnnSeed}`
                  : "inline graphs · job mode"}
              </span>
            </div>
            {gnnTrainError !== null ? (
              <div className="mt-3"><ErrorBanner message={gnnTrainError} /></div>
            ) : null}
            {gnnTrainJobId !== null ? (
              <div className="mt-3">
                <JobRunCard
                  jobId={gnnTrainJobId}
                  resultLabel="the trained GNN model view"
                  onResult={handleGnnTrainResult}
                />
              </div>
            ) : null}
            {gnnTrainResult !== null ? <GnnModelCard result={gnnTrainResult} /> : null}
            <p className="mt-3 text-xs leading-relaxed text-[var(--color-muted-foreground)]">
              The lane rides `POST /api/v1/dl/gnn-train` in job mode — the
              REAL `GraphClassifier` fit with HONEST stage-level telemetry
              (assembling → training): the torch GNN trainer exposes its
              `loss_history_` only AFTER the fit, so this card never
              fabricates a per-epoch stream (unlike the tabular lane’s
              real `on_epoch` callback, there is none to subscribe to).
              The conv/pool vocabularies above arrive live from the same
              catalog call (`gnn.convs` / `gnn.pools` — the backend’s own
              CONV_NAMES / POOL_NAMES, never re-declared here).
            </p>
          </CardContent>
        </Card>

        <Card className="mt-4">
          <CardHeader>
            <CardTitle className="text-base">Predict on graphs</CardTitle>
            <CardDescription>
              the graph models’ OWN prediction path — `/api/v1/dl/gnn-predict`,
              never the tabular `/api/v1/ml/predict` (which refuses graph
              models up front)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-center gap-3">
              <label className="flex items-center gap-2 text-xs text-[var(--color-muted-foreground)]">
                <input
                  type="checkbox"
                  checked={gnnPredictProba}
                  onChange={(event) => setGnnPredictProba(event.target.checked)}
                  className="h-4 w-4"
                />
                class probabilities
              </label>
              <Button
                size="sm"
                disabled={gnnTrainResult === null || gnnPredictBusy || gnnPredictText.trim() === ""}
                onClick={() => void handleGnnPredict()}
              >
                {gnnPredictBusy ? "Predicting…" : "Predict on the graphs"}
              </Button>
            </div>
            <label className="mt-3 block text-xs text-[var(--color-muted-foreground)]">
              graphs to predict (JSON — same shape as the train block, no labels)
              <textarea
                value={gnnPredictText}
                onChange={(event) => setGnnPredictText(event.target.value)}
                rows={5}
                placeholder='{"graphs": [{"x": [[…]], "edge_index": [[…], […]]}, …]}'
                className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 font-mono text-xs"
              />
            </label>
            {gnnPredictError !== null ? (
              <div className="mt-3"><ErrorBanner message={gnnPredictError} /></div>
            ) : null}
            {gnnPredictResult !== null ? (
              <div className="mt-3">
                <p className="text-xs text-[var(--color-muted-foreground)]">
                  model {gnnPredictResult.model_id} · {gnnPredictResult.n_graphs} graphs ·
                  classes {gnnPredictResult.classes.map((c) => formatValue(c)).join(" / ")}
                </p>
                <div className="mt-2 overflow-hidden rounded-lg border border-[var(--color-border)]">
                  <div className="max-h-64 overflow-auto">
                    <table className="w-full border-collapse text-[11px]">
                      <thead className="sticky top-0 bg-[var(--color-muted)]">
                        <tr>
                          <th scope="col" className="whitespace-nowrap px-3 py-1.5 text-left font-semibold text-[var(--color-foreground)]">graph</th>
                          <th scope="col" className="whitespace-nowrap px-3 py-1.5 text-left font-semibold text-[var(--color-foreground)]">prediction</th>
                          {gnnPredictResult.probabilities !== null
                            ? gnnPredictResult.classes.map((cls, i) => (
                                <th key={`${String(cls)}-${i}`} scope="col" className="whitespace-nowrap px-3 py-1.5 text-left font-semibold text-[var(--color-foreground)]">
                                  p({formatValue(cls)})
                                </th>
                              ))
                            : null}
                        </tr>
                      </thead>
                      <tbody>
                        {gnnPredictResult.predictions.map((prediction, i) => (
                          <tr key={i} className="border-t border-[var(--color-border)] bg-[var(--color-card)]">
                            <td className="whitespace-nowrap px-3 py-1 text-[var(--color-muted-foreground)]">{i}</td>
                            <td className="whitespace-nowrap px-3 py-1 font-medium text-[var(--color-foreground)]">{formatValue(prediction)}</td>
                            {gnnPredictResult.probabilities !== null
                              ? (gnnPredictResult.probabilities[i] ?? []).map((p, j) => (
                                  <td key={j} className="whitespace-nowrap px-3 py-1 text-[var(--color-muted-foreground)]">
                                    {typeof p === "number" ? p.toFixed(4) : "—"}
                                  </td>
                                ))
                              : null}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            ) : (
              <p className="mt-3 text-sm text-[var(--color-muted-foreground)]">
                Train above first — a mismatched node-feature width surfaces
                the backend’s own 422 verbatim (“the GNN was trained on W-wide
                node features; graph i has V”), never a silent failure.
              </p>
            )}
          </CardContent>
        </Card>

        {/* DL-2: the edge-attribution card — the gnnxai engine's
            per-edge importance + the sized-edge scatter over the
            backend's own circular layout */}
        <Card className="mt-4">
          <CardHeader>
            <CardTitle className="text-base">Edge attribution (GNN explainability)</CardTitle>
            <CardDescription>
              DL-2 — `/api/v1/dl/gnn-explain`, the REAL gnnxai engine: the
              per-edge importance + the fidelity verdicts over the trained
              graph model (the role SHAP plays for the tabular models)
            </CardDescription>
          </CardHeader>
          <CardContent>
            {gnnTrainResult === null ? (
              <p className="text-sm text-[var(--color-muted-foreground)]">
                Train a graph model above first — the explainability lane
                runs over the session-trained GNN.
              </p>
            ) : (
              <>
                <div className="flex flex-wrap items-end gap-3">
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    method
                    <select
                      aria-label="the explain method"
                      value={gnnExplainMethod}
                      onChange={(event) =>
                        setGnnExplainMethod(
                          event.target.value as
                            | "gnnexplainer"
                            | "pgexplainer"
                            | "attention"
                            | "subgraphx",
                        )
                      }
                      className="mt-1 block w-32 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    >
                      <option value="gnnexplainer">gnnexplainer</option>
                      <option value="pgexplainer">pgexplainer</option>
                      <option value="attention">attention</option>
                      <option value="subgraphx">subgraphx</option>
                    </select>
                  </label>
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    top_k
                    <input
                      type="number"
                      min={1}
                      max={100}
                      value={gnnExplainTopK}
                      onChange={(event) => setGnnExplainTopK(Number(event.target.value))}
                      className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    />
                  </label>
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    seed
                    <input
                      type="number"
                      min={0}
                      value={gnnExplainSeed}
                      onChange={(event) => setGnnExplainSeed(Number(event.target.value))}
                      className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    />
                  </label>
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    iters (optimizer methods)
                    <input
                      type="number"
                      min={10}
                      max={2000}
                      value={gnnExplainItersText}
                      onChange={(event) => setGnnExplainItersText(event.target.value)}
                      placeholder="engine default"
                      className="mt-1 w-28 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    />
                  </label>
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    target class
                    <input
                      type="number"
                      min={0}
                      value={gnnExplainTargetText}
                      onChange={(event) => setGnnExplainTargetText(event.target.value)}
                      placeholder="argmax"
                      className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    />
                  </label>
                  <label className="text-xs text-[var(--color-muted-foreground)]">
                    labels (pgexplainer only — JSON list)
                    <input
                      type="text"
                      value={gnnExplainLabelsText}
                      onChange={(event) => setGnnExplainLabelsText(event.target.value)}
                      placeholder="[0, 1]"
                      className="mt-1 w-28 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 font-mono text-xs"
                    />
                  </label>
                </div>
                <label className="mt-3 block text-xs text-[var(--color-muted-foreground)]">
                  graphs to explain (JSON — the same shape as the predict block)
                  <textarea
                    aria-label="the graphs to explain"
                    value={gnnExplainText}
                    onChange={(event) => setGnnExplainText(event.target.value)}
                    rows={5}
                    placeholder='{"graphs": [{"x": [[…]], "edge_index": [[…], […]]}, …]}'
                    className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-2 font-mono text-xs"
                  />
                </label>
                <div className="mt-3 flex flex-wrap items-center gap-3">
                  <Button
                    size="sm"
                    disabled={gnnExplainBusy || gnnExplainText.trim() === ""}
                    onClick={() => void handleGnnExplain()}
                  >
                    {gnnExplainBusy ? "Explaining…" : "Explain the edges"}
                  </Button>
                  <span className="text-xs text-[var(--color-muted-foreground)]">
                    model {gnnTrainResult.model.model_id} · sync · method{" "}
                    {gnnExplainMethod} · masked forward passes
                  </span>
                </div>
                {gnnExplainError !== null ? (
                  <div className="mt-3"><ErrorBanner message={gnnExplainError} /></div>
                ) : null}
                {gnnExplainResult !== null ? (
                  <div className="mt-3">
                    <p className="text-xs text-[var(--color-muted-foreground)]">
                      {gnnExplainResult.graphs.length} graph(s) explained ·
                      top_k {gnnExplainResult.top_k} · seed{" "}
                      {gnnExplainResult.seed}
                    </p>
                    {gnnExplainResult.graphs.map((row) => (
                      <GnnExplainRow key={row.index} row={row} />
                    ))}
                  </div>
                ) : null}
                <p className="mt-3 text-xs leading-relaxed text-[var(--color-muted-foreground)]">
                  The lane rides `POST /api/v1/dl/gnn-explain` — the
                  gnnxai engine’s per-edge scores aligned with each graph’s
                  own edge_index columns, the fidelity_plus/minus drops, and
                  the deterministic circular layout (the sized-edge
                  scatter’s canvas). The `pgexplainer` lane trains its
                  parametric scorer over the request’s own labeled graphs
                  (one label per graph); the `attention` lane needs a GAT
                  model (the engine’s own honest refusal otherwise).
                </p>
              </>
            )}
          </CardContent>
        </Card>

      </section>

      {/*
       * The D-MPNN lane — DL-4 (the roadmap's row 16): the molecular
       * property predictor over a registered dataset's own SMILES
       * column — "give a SMILES column, get a property model" — with
       * the molecular models' OWN predict path over raw SMILES.
       */}
      <section className="mt-12" aria-labelledby="mpnn-lane-title">
        <div className="flex flex-wrap items-baseline gap-2 border-t border-[var(--color-border)] pt-6">
          <h2
            id="mpnn-lane-title"
            className="text-lg font-semibold text-[var(--color-foreground)]"
          >
            D-MPNN — the molecular lane
          </h2>
          <span className="text-xs text-[var(--color-muted-foreground)]">
            DL-4 · the REAL numpy chemprop-style directed message passing —
            SMILES in, property out
          </span>
        </div>

        <Card className="mt-4">
          <CardHeader>
            <CardTitle className="text-base">Train over a SMILES column</CardTitle>
            <CardDescription>
              `/api/v1/dl/mpnn-train` — the registered dataset&apos;s own
              SMILES column re-read server-side, the honest rejected-rows
              accounting, the seeded held-out carve, the shipped model as the
              full-corpus refit
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-end gap-3">
              <label className="text-xs text-[var(--color-muted-foreground)]">
                dataset
                <select
                  aria-label="the molecular lane's dataset"
                  value={mpnnDatasetId ?? ""}
                  onChange={(event) => {
                    const value = event.target.value;
                    if (value !== "") {
                      void handleMpnnDatasetPick(value);
                    } else {
                      setMpnnDatasetId(null);
                      setMpnnView(null);
                      setMpnnSmilesColumn("");
                    }
                  }}
                  className="mt-1 block w-48 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                >
                  <option value="">— pick a dataset —</option>
                  {datasets.map((row) => (
                    <option key={row.dataset_id} value={row.dataset_id}>
                      {row.name} · {row.shape[0]} rows
                    </option>
                  ))}
                </select>
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                SMILES column
                <select
                  aria-label="the SMILES column"
                  value={mpnnSmilesColumn}
                  onChange={(event) => setMpnnSmilesColumn(event.target.value)}
                  className="mt-1 block w-36 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                >
                  <option value="">— pick —</option>
                  {(mpnnView?.columns ?? [])
                    .filter((column) => column !== mpnnView?.target)
                    .map((column) => (
                      <option key={column} value={column}>
                        {column}
                      </option>
                    ))}
                </select>
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                hidden
                <input
                  type="number"
                  min={1}
                  max={512}
                  value={mpnnHidden}
                  onChange={(event) => setMpnnHidden(Number(event.target.value))}
                  className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                depth
                <input
                  type="number"
                  min={1}
                  max={8}
                  value={mpnnDepth}
                  onChange={(event) => setMpnnDepth(Number(event.target.value))}
                  className="mt-1 w-16 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                epochs
                <input
                  type="number"
                  min={1}
                  max={2000}
                  value={mpnnEpochs}
                  onChange={(event) => setMpnnEpochs(Number(event.target.value))}
                  className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                lr
                <input
                  type="number"
                  min={0.0001}
                  max={1}
                  step={0.001}
                  value={mpnnLr}
                  onChange={(event) => setMpnnLr(Number(event.target.value))}
                  className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                seed
                <input
                  type="number"
                  min={0}
                  value={mpnnSeed}
                  onChange={(event) => setMpnnSeed(Number(event.target.value))}
                  className="mt-1 w-16 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <Button
                size="sm"
                disabled={
                  mpnnDatasetId === null ||
                  mpnnSmilesColumn === "" ||
                  mpnnTrainBusy ||
                  (mpnnView !== null && mpnnView.task !== "regression")
                }
                onClick={() => void handleMpnnTrain()}
              >
                {mpnnTrainBusy ? "Submitting…" : "Train the molecular model"}
              </Button>
            </div>
            {mpnnView !== null && mpnnView.task !== "regression" ? (
              <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                this dataset&apos;s task is {mpnnView.task} — the D-MPNN is a
                regressor; the lane needs a regression target (the route&apos;s
                own refusal)
              </p>
            ) : null}
            {mpnnViewError !== null ? (
              <div className="mt-2">
                <ErrorBanner message={mpnnViewError} />
              </div>
            ) : null}
            {mpnnTrainError !== null ? (
              <div className="mt-2">
                <ErrorBanner message={mpnnTrainError} />
              </div>
            ) : null}
            {mpnnTrainJobId !== null ? (
              <div className="mt-3">
                <JobRunCard
                  jobId={mpnnTrainJobId}
                  resultLabel="the trained D-MPNN model view"
                  onResult={handleMpnnTrainResult}
                />
              </div>
            ) : null}
            {mpnnTrainResult !== null ? <MpnnModelCard result={mpnnTrainResult} /> : null}
            <p className="mt-3 text-xs leading-relaxed text-[var(--color-muted-foreground)]">
              The lane rides `POST /api/v1/dl/mpnn-train` in job mode — the
              REAL `DMPNN` fit (chemprop&apos;s directed message passing over
              the house graphs) with stage-level progress only (assembling
              → the carve → the refit) — the per-iteration history lands
              AFTER the run, never a fabricated epoch stream: the held-out
              carve&apos;s RMSE and correlation ride the result, the shipped
              model is the FULL-corpus refit, and every unparseable row
              rides its own rejection accounting (never dropped silently).
            </p>
          </CardContent>
        </Card>

        <Card className="mt-4">
          <CardHeader>
            <CardTitle className="text-base">Predict on molecules</CardTitle>
            <CardDescription>
              the molecular models&apos; OWN prediction path —
              `/api/v1/dl/mpnn-predict` over raw SMILES (never the tabular
              `/api/v1/ml/predict`)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <label className="block text-xs text-[var(--color-muted-foreground)]">
              SMILES (one per line)
              <textarea
                aria-label="the SMILES to predict"
                rows={4}
                value={mpnnPredictText}
                onChange={(event) => setMpnnPredictText(event.target.value)}
                placeholder={"CCO\nCCN\nc1ccccc1"}
                className="mt-1 block w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 font-mono text-xs"
              />
            </label>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <Button
                size="sm"
                disabled={mpnnTrainResult === null || mpnnPredictBusy}
                onClick={() => void handleMpnnPredict()}
              >
                {mpnnPredictBusy ? "Predicting…" : "Predict the properties"}
              </Button>
              {mpnnTrainResult === null ? (
                <span className="text-xs text-[var(--color-muted-foreground)]">
                  train a molecular model above first
                </span>
              ) : null}
            </div>
            {mpnnPredictError !== null ? (
              <div className="mt-2">
                <ErrorBanner message={mpnnPredictError} />
              </div>
            ) : null}
            {mpnnPredictResult !== null ? (
              <div className="mt-3">
                <p className="text-xs font-medium text-[var(--color-foreground)]">
                  {mpnnPredictResult.n_valid} prediction(s) ·{" "}
                  {mpnnPredictResult.n_rejected} rejected
                </p>
                <div className="mt-2 max-h-64 overflow-y-auto rounded-md border border-[var(--color-border)]">
                  <table className="w-full text-xs">
                    <thead className="sticky top-0 bg-[var(--color-muted)]">
                      <tr className="text-left text-[var(--color-muted-foreground)]">
                        <th scope="col" className="px-3 py-2">row</th>
                        <th scope="col" className="px-3 py-2">SMILES</th>
                        <th scope="col" className="px-3 py-2">prediction</th>
                      </tr>
                    </thead>
                    <tbody>
                      {mpnnPredictResult.predictions.map((row) => (
                        <tr
                          key={row.index}
                          className="border-t border-[var(--color-border)] text-[var(--color-foreground)]"
                        >
                          <td className="px-3 py-1.5">{row.index}</td>
                          <td className="px-3 py-1.5 font-mono">{row.smiles}</td>
                          <td className="px-3 py-1.5">{row.prediction.toFixed(3)}</td>
                        </tr>
                      ))}
                      {mpnnPredictResult.rejected.map((row) => (
                        <tr
                          key={`rejected-${row.index}`}
                          className="border-t border-[var(--color-border)] text-[var(--color-muted-foreground)]"
                        >
                          <td className="px-3 py-1.5">{row.index}</td>
                          <td className="px-3 py-1.5 font-mono">
                            {row.smiles ?? "—"}
                          </td>
                          <td className="px-3 py-1.5">rejected: {row.error}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : null}
          </CardContent>
        </Card>

      </section>

      {/*
       * The sequence-CNN lane — part18 §5.2's convolutional card: the
       * 3-D-only temporal conv over inline sequences or the backend's
       * deterministic motif fixture, with the sequence models' OWN
       * predict path (never the tabular /ml/predict).
       */}
      <section className="mt-12" aria-labelledby="cnn-lane-title">
        <div className="flex flex-wrap items-baseline gap-2 border-t border-[var(--color-border)] pt-6">
          <h2
            id="cnn-lane-title"
            className="text-lg font-semibold text-[var(--color-foreground)]"
          >
            Sequence CNN — the convolutional lane
          </h2>
          <span className="text-xs text-[var(--color-muted-foreground)]">
            part18 §5.2 · the REAL 1-D temporal conv (kernel → batch-norm →
            ReLU, avg+max pool) over 3-D sequences or the motif fixture
          </span>
        </div>
        <Card className="mt-3">
          <CardHeader>
            <CardTitle className="text-base">Train a sequence model</CardTitle>
            <CardDescription>
              the source (the motif fixture or inline sequences) + the
              architecture (channels · kernel · layers) + the trainer’s
              own controls — job mode, the estimator’s real per-epoch
              telemetry
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-end gap-3">
              <label className="text-xs text-[var(--color-muted-foreground)]">
                source
                <select
                  value={cnnSource}
                  onChange={(event) =>
                    setCnnSource(event.target.value as typeof cnnSource)
                  }
                  className="mt-1 w-32 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                  aria-label="cnn source"
                >
                  <option value="synthetic">synthetic motifs</option>
                  <option value="inline">inline sequences</option>
                </select>
              </label>
              {cnnSource === "synthetic" ? (
                <label className="text-xs text-[var(--color-muted-foreground)]">
                  motif count (10–500)
                  <input
                    type="number"
                    min={10}
                    max={500}
                    value={cnnSyntheticN}
                    onChange={(event) => setCnnSyntheticN(Number(event.target.value))}
                    className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                  />
                </label>
              ) : (
                <label className="text-xs text-[var(--color-muted-foreground)]">
                  task
                  <select
                    value={cnnTask}
                    onChange={(event) =>
                      setCnnTask(event.target.value as typeof cnnTask)
                    }
                    className="mt-1 w-32 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                    aria-label="cnn task"
                  >
                    <option value="classification">classification</option>
                    <option value="regression">regression</option>
                  </select>
                </label>
              )}
              <label className="text-xs text-[var(--color-muted-foreground)]">
                seed
                <input
                  type="number"
                  min={0}
                  value={cnnSeed}
                  onChange={(event) => setCnnSeed(Number(event.target.value))}
                  className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                channels
                <input
                  type="number"
                  min={1}
                  max={128}
                  value={cnnChannels}
                  onChange={(event) => setCnnChannels(Number(event.target.value))}
                  className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                kernel (odd)
                <input
                  type="number"
                  min={1}
                  max={31}
                  value={cnnKernel}
                  onChange={(event) => setCnnKernel(Number(event.target.value))}
                  className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                layers
                <input
                  type="number"
                  min={1}
                  max={6}
                  value={cnnLayers}
                  onChange={(event) => setCnnLayers(Number(event.target.value))}
                  className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                epochs
                <input
                  type="number"
                  min={1}
                  max={2000}
                  step={5}
                  value={cnnEpochs}
                  onChange={(event) => setCnnEpochs(Number(event.target.value))}
                  className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                lr
                <input
                  type="number"
                  min={0.0001}
                  max={1}
                  step={0.001}
                  value={cnnLr}
                  onChange={(event) => setCnnLr(Number(event.target.value))}
                  className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
            </div>
            {cnnSource === "inline" ? (
              <div className="mt-3">
                <label className="text-xs text-[var(--color-muted-foreground)]">
                  inline sequences + labels (JSON)
                  <textarea
                    value={cnnInlineText}
                    onChange={(event) => setCnnInlineText(event.target.value)}
                    rows={6}
                    placeholder='{"sequences": [[[0.1, 0.2], [0.3, 0.4], …], …], "labels": [0, 1, …]}'
                    className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 font-mono text-xs"
                  />
                </label>
                {cnnInlineError !== null ? (
                  <p className="mt-1 text-xs text-[var(--color-destructive)]">{cnnInlineError}</p>
                ) : null}
                <p className="mt-1 text-xs text-[var(--color-muted-foreground)]">
                  Every row validates BEFORE any request fires — rectangular
                  step lists (one width per sequence), one numeric label
                  per sequence; a malformed block never leaves the page.
                  A plain tabular matrix is refused by the backend&apos;s
                  own 422 with the pointer to the DeepMLP / Tabular
                  Transformer families — never a fake image reshape.
                </p>
              </div>
            ) : null}
            <div className="mt-3 flex flex-wrap items-center gap-3">
              <Button
                size="sm"
                disabled={cnnTrainBusy || (cnnSource === "inline" && cnnInlineText.trim() === "")}
                onClick={() => void handleCnnTrain()}
              >
                {cnnTrainBusy ? "Submitting…" : "Train the sequence model"}
              </Button>
              <span className="text-xs text-[var(--color-muted-foreground)]">
                {cnnSource === "synthetic"
                  ? `synthetic motif fixture · ${cnnSyntheticN} sequences · seed ${cnnSeed}`
                  : "inline sequences · job mode"}
              </span>
            </div>
            {cnnTrainError !== null ? (
              <div className="mt-3"><ErrorBanner message={cnnTrainError} /></div>
            ) : null}
            {cnnTrainJobId !== null ? (
              <div className="mt-3">
                <JobRunCard
                  jobId={cnnTrainJobId}
                  resultLabel="the trained sequence model view"
                  onResult={handleCnnTrainResult}
                />
              </div>
            ) : null}
            {cnnTrainResult !== null ? <CnnModelCard result={cnnTrainResult} /> : null}
            <p className="mt-3 text-xs leading-relaxed text-[var(--color-muted-foreground)]">
              The lane rides `POST /api/v1/dl/cnn-train` in job mode — the
              REAL `SequenceCNN` fit with the estimator&apos;s own
              per-epoch `training.metrics` telemetry (the same
              `on_epoch` contract as the tabular lane). The convolution
              is scientifically meaningful ONLY over real sequence
              representations: a tabular matrix is refused up front
              (part18 §5.2&apos;s no-fake-images rule, on the wire).
            </p>
          </CardContent>
        </Card>

        <Card className="mt-4">
          <CardHeader>
            <CardTitle className="text-base">Predict on sequences</CardTitle>
            <CardDescription>
              the sequence models&apos; OWN prediction path —
              `/api/v1/dl/cnn-predict`, never the tabular
              `/api/v1/ml/predict` (which refuses sequence models up front)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-center gap-3">
              <label className="flex items-center gap-2 text-xs text-[var(--color-muted-foreground)]">
                <input
                  type="checkbox"
                  checked={cnnPredictProba}
                  onChange={(event) => setCnnPredictProba(event.target.checked)}
                  className="h-4 w-4"
                />
                class probabilities
              </label>
              <Button
                size="sm"
                disabled={cnnTrainResult === null || cnnPredictBusy || cnnPredictText.trim() === ""}
                onClick={() => void handleCnnPredict()}
              >
                {cnnPredictBusy ? "Predicting…" : "Predict on the sequences"}
              </Button>
            </div>
            <label className="mt-3 block text-xs text-[var(--color-muted-foreground)]">
              sequences to predict (JSON — same shape as the train block, no labels)
              <textarea
                value={cnnPredictText}
                onChange={(event) => setCnnPredictText(event.target.value)}
                rows={5}
                placeholder='{"sequences": [[[…steps…], …], …]}'
                className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 font-mono text-xs"
              />
            </label>
            {cnnPredictError !== null ? (
              <div className="mt-3"><ErrorBanner message={cnnPredictError} /></div>
            ) : null}
            {cnnPredictResult !== null ? (
              <div className="mt-3">
                <p className="text-xs text-[var(--color-muted-foreground)]">
                  model {cnnPredictResult.modelId} · {cnnPredictResult.nSequences} sequences
                  {cnnPredictResult.classes !== null
                    ? ` · classes ${cnnPredictResult.classes.map((c) => formatValue(c)).join(" / ")}`
                    : ""}
                </p>
                <div className="mt-2 overflow-hidden rounded-lg border border-[var(--color-border)]">
                  <div className="max-h-64 overflow-auto">
                    <table className="w-full border-collapse text-[11px]">
                      <thead className="sticky top-0 bg-[var(--color-muted)]">
                        <tr>
                          <th scope="col" className="whitespace-nowrap px-3 py-1.5 text-left font-semibold text-[var(--color-foreground)]">sequence</th>
                          <th scope="col" className="whitespace-nowrap px-3 py-1.5 text-left font-semibold text-[var(--color-foreground)]">prediction</th>
                          {cnnPredictResult.probabilities !== null && cnnPredictResult.classes !== null
                            ? cnnPredictResult.classes.map((cls, i) => (
                                <th key={`${String(cls)}-${i}`} scope="col" className="whitespace-nowrap px-3 py-1.5 text-left font-semibold text-[var(--color-foreground)]">
                                  p({formatValue(cls)})
                                </th>
                              ))
                            : null}
                        </tr>
                      </thead>
                      <tbody>
                        {cnnPredictResult.predictions.map((prediction, i) => (
                          <tr key={i} className="border-t border-[var(--color-border)] bg-[var(--color-card)]">
                            <td className="whitespace-nowrap px-3 py-1 text-[var(--color-muted-foreground)]">{i}</td>
                            <td className="whitespace-nowrap px-3 py-1 font-medium text-[var(--color-foreground)]">{formatValue(prediction)}</td>
                            {cnnPredictResult.probabilities !== null
                              ? (cnnPredictResult.probabilities[i] ?? []).map((p, j) => (
                                  <td key={j} className="whitespace-nowrap px-3 py-1 text-[var(--color-muted-foreground)]">
                                    {typeof p === "number" ? p.toFixed(4) : "—"}
                                  </td>
                                ))
                              : null}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            ) : (
              <p className="mt-3 text-sm text-[var(--color-muted-foreground)]">
                Train above first — a mismatched channel width surfaces the
                backend&apos;s own 422 verbatim, never a silent failure.
              </p>
            )}
          </CardContent>
        </Card>
      </section>

      {/*
       * The tabular-autoencoder lane — part18 §5.5's representation
       * card: latent coordinates, reconstruction errors and the honest
       * outlier ranking over the SELECTED dataset (unsupervised — its
       * own encode lane, never the supervised predict path).
       */}
      <section className="mt-12" aria-labelledby="ae-lane-title">
        <div className="flex flex-wrap items-baseline gap-2 border-t border-[var(--color-border)] pt-6">
          <h2
            id="ae-lane-title"
            className="text-lg font-semibold text-[var(--color-foreground)]"
          >
            Tabular autoencoder — the representation lane
          </h2>
          <span className="text-xs text-[var(--color-muted-foreground)]">
            part18 §5.5 · the REAL encoder/decoder pair — latent
            representation, reconstruction, dimensionality reduction,
            anomaly ranking by reconstruction error
          </span>
        </div>
        <Card className="mt-3">
          <CardHeader>
            <CardTitle className="text-base">Train an autoencoder</CardTitle>
            <CardDescription>
              the SELECTED dataset&apos;s feature matrix · the bottleneck
              width (2–3 gives the scatter-plotable projection) · job
              mode with the real per-epoch reconstruction telemetry
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-end gap-3">
              <label className="text-xs text-[var(--color-muted-foreground)]">
                latent dim
                <input
                  type="number"
                  min={1}
                  max={64}
                  value={aeLatentDim}
                  onChange={(event) => setAeLatentDim(Number(event.target.value))}
                  className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                  aria-label="autoencoder latent dim"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                hidden (JSON list)
                <input
                  type="text"
                  value={aeHiddenText}
                  onChange={(event) => setAeHiddenText(event.target.value)}
                  placeholder="[16] — the encoder stack, the decoder mirrors"
                  className="mt-1 w-48 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
              <label className="text-xs text-[var(--color-muted-foreground)]">
                epochs
                <input
                  type="number"
                  min={1}
                  max={2000}
                  step={10}
                  value={aeEpochs}
                  onChange={(event) => setAeEpochs(Number(event.target.value))}
                  className="mt-1 w-20 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-xs"
                />
              </label>
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-3">
              <Button
                size="sm"
                disabled={aeTrainBusy || selectedId === null}
                onClick={() => void handleAeTrain()}
              >
                {aeTrainBusy ? "Submitting…" : "Train the autoencoder"}
              </Button>
              <span className="text-xs text-[var(--color-muted-foreground)]">
                {selectedId !== null
                  ? `dataset ${selectedId} · ${selected?.n_features ?? "?"} features`
                  : "select a dataset above"}
              </span>
            </div>
            {aeTrainError !== null ? (
              <div className="mt-3"><ErrorBanner message={aeTrainError} /></div>
            ) : null}
            {aeTrainJobId !== null ? (
              <div className="mt-3">
                <JobRunCard
                  jobId={aeTrainJobId}
                  resultLabel="the trained autoencoder view"
                  onResult={handleAeTrainResult}
                />
              </div>
            ) : null}
            {aeTrainResult !== null ? (
              <div className="mt-3">
                <p className="text-xs text-[var(--color-muted-foreground)]">
                  model {aeTrainResult.model.model_id} · task{" "}
                  {aeTrainResult.model.task} · latent dim{" "}
                  {aeTrainResult.latentDim} · {aeTrainResult.nIter} epochs ·
                  mean reconstruction error{" "}
                  {typeof aeTrainResult.meanReconstructionError === "number"
                    ? aeTrainResult.meanReconstructionError.toFixed(4)
                    : "—"}
                </p>
              </div>
            ) : null}
            {aeTrainResult !== null ? (
              <div className="mt-3">
                <Button
                  size="sm"
                  variant="outline"
                  disabled={aeEncodeBusy || selectedId === null}
                  onClick={() => void handleAeEncode()}
                >
                  {aeEncodeBusy ? "Encoding…" : "Encode the dataset"}
                </Button>
                <span className="ml-2 text-xs text-[var(--color-muted-foreground)]">
                  latent coordinates + per-row reconstruction errors + the
                  honest outlier ranking
                </span>
              </div>
            ) : null}
            {aeEncodeError !== null ? (
              <div className="mt-3"><ErrorBanner message={aeEncodeError} /></div>
            ) : null}
            {aeEncodeResult !== null ? (
              <div className="mt-3">
                {aeEncodeResult.latentDim === 2 ? (
                  <PlotlyFigure
                    data={buildLatentTraces(aeEncodeResult)}
                    ariaLabel="the autoencoder's 2-D latent projection"
                  />
                ) : null}
                <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                  {aeEncodeResult.nRows} rows · mean reconstruction error{" "}
                  {typeof aeEncodeResult.meanReconstructionError === "number"
                    ? aeEncodeResult.meanReconstructionError.toFixed(4)
                    : "—"}
                  {" · top-"}{aeEncodeResult.topOutliers.length}{" outliers: "}
                  {aeEncodeResult.topOutliers
                    .map((o) =>
                      `#${o.index} (${typeof o.error === "number" ? o.error.toFixed(3) : "—"})`,
                    )
                    .join(", ")}
                </p>
                <p className="mt-1 text-xs text-[var(--color-muted-foreground)]">
                  {aeEncodeResult.note} — the reconstruction error is a
                  heuristic ranking, never ground truth (the dedicated
                  anomaly lane owns Isolation Forest).
                </p>
              </div>
            ) : null}
            <p className="mt-3 text-xs leading-relaxed text-[var(--color-muted-foreground)]">
              The lane rides `POST /api/v1/dl/autoencoder-train` +
              `POST /api/v1/dl/autoencoder-encode` — unsupervised, the
              model registers with task `unsupervised` and its own encode
              URL; the supervised `/api/v1/ml/predict` refuses it up
              front (one path per model class, never two).
            </p>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}

/** The sequence-CNN lane's inline JSON gate (validated client-side). */
function parseCnnInlineSequences(
  text: string,
  mode: "train" | "predict",
): { sequences: number[][][]; labels: number[] } | undefined {
  const trimmed = text.trim();
  if (trimmed === "") {
    return undefined;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    throw new Error(
      "the inline sequences block is not valid JSON — expected " +
        '{"sequences": [[[…steps…], …], …], "labels": […]}',
    );
  }
  if (typeof parsed !== "object" || parsed === null) {
    throw new Error("the inline sequences block must be an object");
  }
  const r = parsed as Record<string, unknown>;
  const rows = r["sequences"];
  if (!Array.isArray(rows) || rows.length === 0) {
    throw new Error('the block needs a non-empty "sequences" array');
  }
  const sequences: number[][][] = [];
  for (const row of rows) {
    if (!Array.isArray(row) || row.length === 0) {
      throw new Error("each sequence must be a non-empty array of steps");
    }
    const steps: number[][] = [];
    for (const step of row) {
      if (!Array.isArray(step) || step.some((v) => typeof v !== "number")) {
        throw new Error("each step must be an array of numbers (the channels)");
      }
      steps.push(step as number[]);
    }
    sequences.push(steps);
  }
  const widths = new Set(sequences.map((s) => s.length));
  const channelWidths = new Set(
    sequences.flatMap((s) => s.map((step) => step.length)),
  );
  if (widths.size > 1) {
    throw new Error(
      `all sequences must share one seq_len — got lengths ${[...widths].slice(0, 5)}`,
    );
  }
  if (channelWidths.size > 1) {
    throw new Error(
      `all steps must share one channel width — got ${[...channelWidths].slice(0, 5)}`,
    );
  }
  if (mode === "train") {
    const labels = r["labels"];
    if (!Array.isArray(labels)) {
      throw new Error('the inline training block needs its "labels" array');
    }
    if (labels.length !== sequences.length) {
      throw new Error(
        `sequences has ${sequences.length} rows; labels has ${labels.length}`,
      );
    }
    return {
      sequences,
      labels: labels.map((v) => {
        if (typeof v !== "number" || !Number.isFinite(v)) {
          throw new Error("labels must be numeric (one per sequence)");
        }
        return v;
      }),
    };
  }
  return { sequences, labels: [] };
}

/** The autoencoder lane's hidden-stack gate. */
function parseAeHidden(text: string): number[] {
  const trimmed = text.trim();
  if (trimmed === "") {
    return [];
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    throw new Error("the hidden layers block is not valid JSON — expected [16]");
  }
  if (!Array.isArray(parsed)) {
    throw new Error("the hidden layers must be a JSON list of widths");
  }
  return parsed.map((v) => {
    if (typeof v !== "number" || !Number.isInteger(v) || v < 1) {
      throw new Error("hidden layer widths must be positive integers");
    }
    return v;
  });
}

/** The latent scatter's traces (the 2-D projection only). */
function buildLatentTraces(
  result: AutoencoderEncodeResponse,
): PlotTrace[] {
  const xs = result.latent.map((row) => row[0] ?? 0);
  const ys = result.latent.map((row) => row[1] ?? 0);
  const top = new Set(result.topOutliers.map((o) => o.index));
  const normalXs: number[] = [];
  const normalYs: number[] = [];
  const outlierXs: number[] = [];
  const outlierYs: number[] = [];
  for (const [i, x] of xs.entries()) {
    if (top.has(i)) {
      outlierXs.push(x);
      outlierYs.push(ys[i] ?? 0);
    } else {
      normalXs.push(x);
      normalYs.push(ys[i] ?? 0);
    }
  }
  const traces: PlotTrace[] = [
    {
      name: "rows",
      x: normalXs,
      y: normalYs,
      mode: "markers",
      type: "scatter",
      marker: { size: 5, opacity: 0.7 },
    },
  ];
  if (outlierXs.length > 0) {
    traces.push({
      name: "top reconstruction errors",
      x: outlierXs,
      y: outlierYs,
      mode: "markers",
      type: "scatter",
      marker: { size: 8, symbol: "x", opacity: 0.95 },
    });
  }
  return traces;
}

/** The sequence model's own honest card (the train result view). */
function CnnModelCard({ result }: { result: CnnTrainResult }) {
  return (
    <div className="mt-3 rounded-lg border border-[var(--color-border)] p-3">
      <p className="text-xs text-[var(--color-muted-foreground)]">
        model {result.model.model_id} · task {result.model.task} ·{" "}
        {result.model.n_train_rows} sequences · {result.seqLen} steps ×{" "}
        {result.nChannels} channels · {result.nIter} epochs run
        {result.model.sequenceModel ? " · sequence model (own predict path)" : ""}
      </p>
      <p className="mt-1 text-xs text-[var(--color-muted-foreground)]">
        {typeof result.trainAccuracy === "number"
          ? `train accuracy ${result.trainAccuracy.toFixed(3)}`
          : ""}
        {typeof result.trainMae === "number"
          ? `train MAE ${result.trainMae.toFixed(3)}`
          : ""}
        {result.classes !== null
          ? ` · classes ${result.classes.map((c) => formatValue(c)).join(" / ")}`
          : ""}
      </p>
    </div>
  );
}
