import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDownToLine,
  Braces,
  Brain,
  ClipboardCheck,
  Database,
  Filter,
  GitCompare,
  Info,
  Layers,
  Network,
  Radar,
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
  EXPLAIN_MAX_ROWS,
  mlApi,
  parseADAnalyzeResult,
  parseSemiSupervisedResult,
  parseTrainResult,
  type ADAnalyzeResult,
  type ADFrameRow,
  type ExplainResult,
  type EvaluationRow,
  type ModelCatalogEntry,
  type PredictResponse,
  type SemiSupervisedView,
  type TrainedModelView,
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
 * Compare → Applicability Domain → Semi-supervised Learning →
 * Explainability → Export
 *
 * Every control below executes the REAL backend through the typed
 * service layer (no fabricated data, no disabled pretend buttons):
 * the catalog/train/predict lanes ride routes/ml.py, the AD stage
 * rides routes/applicability_domain.py with x_test assembled through
 * the bounded column bridge, the semi-supervised lane rides
 * routes/semi_supervised.py behind the backend's own
 * classification-only gate, the explainability stage rides the
 * /ml/explain SHAP lane (the real shap_summary engine over a bounded
 * sample — the r289 queue row, closed), and the export registers a
 * real table artifact downloaded through the token-header lane with
 * digest verification.
 */

type VisualizationTableRequest = components["schemas"]["VisualizationTableRequest"];

/** The backend's own table-export row cap (visualization.py's `_MAX_ROWS`). */
const TABLE_EXPORT_MAX_ROWS = 5000;

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
              </>
            )}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={11}
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
              </>
            )}
          </CardContent>
        </Card>
      </StageHeader>

      <StageHeader
        index={12}
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
        index={13}
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
