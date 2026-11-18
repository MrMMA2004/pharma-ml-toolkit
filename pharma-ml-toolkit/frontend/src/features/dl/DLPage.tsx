import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowDownToLine } from "lucide-react";
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
import { mlApi, type PredictResponse } from "@/services/ml/client";
import { getEventStream } from "@/services/websocket/client";
import {
  DL_ACTIVATIONS,
  DL_HYPERPARAM_KEYS,
  dlApi,
  parseDLTrainResult,
  parseTrainingMetricsPayload,
  type DLModelView,
  type DLNetworkName,
  type NetworkCatalogEntry,
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

/**
 * The live loss/validation curve — an inline SVG drawn from the
 * REAL per-epoch telemetry points (no chart fabrication: two
 * polylines, the points the backend actually published, axes
 * labelled honestly, missing points skipped).
 */
function LossCurveChart({ points }: { points: readonly TrainingMetricsPoint[] }) {
  if (points.length < 2) {
    return (
      <p className="text-sm text-[var(--color-muted-foreground)]">
        Waiting for at least two epochs of telemetry — the backend
        publishes one `training.metrics` envelope per epoch.
      </p>
    );
  }
  const width = 640;
  const height = 240;
  const pad = { left: 56, right: 16, top: 12, bottom: 28 };
  const epochs = points.map((p) => p.epoch);
  const minEpoch = Math.min(...epochs);
  const maxEpoch = Math.max(...epochs);
  const losses = points
    .flatMap((p) => [p.train_loss, p.val_loss])
    .filter((v): v is number => v !== null && Number.isFinite(v));
  const minLoss = Math.min(...losses);
  const maxLoss = Math.max(...losses);
  const spanX = Math.max(1, maxEpoch - minEpoch);
  const spanY = Math.max(1e-9, maxLoss - minLoss);
  const x = (epoch: number) =>
    pad.left + ((epoch - minEpoch) / spanX) * (width - pad.left - pad.right);
  const y = (loss: number) =>
    height - pad.bottom - ((loss - minLoss) / spanY) * (height - pad.top - pad.bottom);

  const polyline = (key: "train_loss" | "val_loss"): string =>
    points
      .filter((p) => {
        const value = p[key];
        return value !== null && Number.isFinite(value);
      })
      .map((p) => `${x(p.epoch).toFixed(1)},${y(p[key] as number).toFixed(1)}`)
      .join(" ");

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className="w-full max-w-2xl"
      role="img"
      aria-label="the live train and validation loss per epoch"
    >
      {/* the axes */}
      <line
        x1={pad.left}
        y1={height - pad.bottom}
        x2={width - pad.right}
        y2={height - pad.bottom}
        stroke="var(--color-border)"
        strokeWidth="1"
      />
      <line
        x1={pad.left}
        y1={pad.top}
        x2={pad.left}
        y2={height - pad.bottom}
        stroke="var(--color-border)"
        strokeWidth="1"
      />
      {/* the y ticks (3, honest to the actual range) */}
      {[0, 0.5, 1].map((fraction) => {
        const value = minLoss + fraction * spanY;
        return (
          <g key={fraction}>
            <line
              x1={pad.left - 4}
              y1={y(value)}
              x2={width - pad.right}
              y2={y(value)}
              stroke="var(--color-border)"
              strokeDasharray="2 4"
              strokeWidth="0.5"
            />
            <text
              x={pad.left - 8}
              y={y(value) + 3}
              textAnchor="end"
              fontSize="10"
              fill="var(--color-muted-foreground)"
            >
              {value.toFixed(3)}
            </text>
          </g>
        );
      })}
      {/* the x labels (first, mid, last epoch) */}
      {[minEpoch, Math.round((minEpoch + maxEpoch) / 2), maxEpoch].map((epoch) => (
        <text
          key={epoch}
          x={x(epoch)}
          y={height - pad.bottom + 16}
          textAnchor="middle"
          fontSize="10"
          fill="var(--color-muted-foreground)"
        >
          {epoch}
        </text>
      ))}
      <text
        x={(pad.left + width - pad.right) / 2}
        y={height - 4}
        textAnchor="middle"
        fontSize="10"
        fill="var(--color-muted-foreground)"
      >
        epoch
      </text>
      {/* the two REAL series */}
      <polyline
        points={polyline("train_loss")}
        fill="none"
        stroke="var(--color-primary)"
        strokeWidth="2"
        aria-label="train loss"
      />
      <polyline
        points={polyline("val_loss")}
        fill="none"
        stroke="var(--color-destructive)"
        strokeWidth="2"
        strokeDasharray="5 3"
        aria-label="validation loss"
      />
    </svg>
  );
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
  const [exportArtifact, setExportArtifact] = useState<ArtifactView | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportBusy, setExportBusy] = useState(false);
  const [digestStatus, setDigestStatus] = useState<string | null>(null);

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
  // defaults are the backend's own, never re-declared here)
  useEffect(() => {
    let cancelled = false;
    dlApi
      .networks()
      .then((response) => {
        if (!cancelled) {
          setNetworks(response.networks);
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
  ]);

  // stage 7: train (job mode — the backend's default lane)
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
    setTrainBusy(true);
    setTrainError(null);
    setTrainJobId(null);
    setTrainedModel(null);
    setMetricsPoints([]);
    setMonitorError(null);
    try {
      const response = await dlApi.train({
        datasetId: selectedId,
        network: selectedNetwork,
        hyperparams,
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
  }, [selectedId, selectedNetwork, buildHyperparams]);

  // the train job's result IS `{model: DLModelView}` — the
  // authoritative REST relay, parsed at runtime
  const handleTrainResult = useCallback((result: unknown) => {
    try {
      setTrainedModel(parseDLTrainResult(result, "dl train result"));
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
        filename: "dl-predictions",
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
            </div>
            {hiddenLayersError !== null ? (
              <p className="mt-2 text-xs text-[var(--color-destructive)]">{hiddenLayersError}</p>
            ) : null}
            <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
              The overrides validate BEFORE any request fires — an
              invalid body never leaves the page. The activation
              tokens are the backend&apos;s own accepted vocabulary
              (`relu`, `tanh`, `gelu`, `sigmoid`); the estimator
              validates `dropout` in [0, 1) with its own actionable
              error — surfaced verbatim, never pre-faked.
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
            <div className="flex flex-wrap items-center gap-3">
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
            </div>
          </CardContent>
        </Card>
      </StageHeader>
    </div>
  );
}

/** The predictions export rows (the prediction + one column per class). */
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
