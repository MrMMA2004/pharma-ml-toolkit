import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDownToLine,
  Beaker,
  Dna,
  Filter,
  FlaskConical,
  LineChart,
  ListOrdered,
  Sparkles,
  Target,
  Wand2,
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
import { ApiError } from "@/services/api/client";
import {
  datasetsApi,
  type DatasetView,
} from "@/services/datasets/client";
import {
  artifactsApi,
  type ArtifactView,
} from "@/services/artifacts/client";
import {
  chemistryApi,
  parseDescriptorFrame,
  type DescriptorFrame,
} from "@/services/chemistry/client";
import { getEventStream } from "@/services/websocket/client";
import {
  OBJECTIVE_KINDS,
  generativeApi,
  parseIterationUpdatePayload,
  parseOptimizeResult,
  parsePriorTrainResult,
  parseTrainingMetricsPayload,
  type ObjectiveCatalogEntry,
  type OptimizationView,
  type PriorView,
  type SampleResponse,
  type TrainingMetricsPayload,
  type IterationUpdatePayload,
} from "@/services/generative/client";

/**
 * The Generative AI workspace — part11 §11.4's complete sequential
 * workflow:
 *
 * Select Generation Method → Define Starting Molecules → Configure
 * Constraints/Objectives → Generate → Validate → Calculate Properties
 * → Filter → Rank → Visualize → Export
 *
 * Every control below executes the REAL backend through the typed
 * service layer (no fabricated data, no disabled pretend buttons).
 * The two REAL generation engines are the backend's own: the
 * `SmilesRnn` prior + sample lane (the prior's per-epoch loss curve
 * is REAL telemetry — one `training.metrics` envelope when training
 * ends — and the sample lane is sync by design: the route declares no
 * job mode, stated honestly, no pretend tracking) and the
 * `GeneticOptimizer` evolution over the nine-objective scoring
 * catalog (the optimizer's own progress callback drives one
 * `iteration.update` envelope per generation; the native cancel path
 * lands the job in CANCELLED, never FAILED). The Validate and
 * Calculate Properties stages ride the REAL descriptor engine
 * (`POST /api/v1/chemistry/descriptors`, `rdkit_all`): the frame's own
 * index IS the validity verdict — the backend drops every SMILES it
 * cannot parse, so the generated set minus the frame's index is the
 * invalid lane, never a client-side guess. The Filter stage selects
 * kept/filtered lanes over the REAL computed property values; the
 * Rank stage orders by the objective score (the optimize lane) or a
 * selected property (the sample lane); the Visualize stage registers
 * REAL figure artifacts (the prior's loss curve, the ranked score
 * profile) through the §17 four-format export stack; the Export stage
 * rides the rows-based table lane with the backend's own 5000-row cap
 * mirrored client-side. The honest exceptions are stated where they
 * live: the similarity/substructure/novelty objectives need their
 * own params (a missing param is the route's own 422, surfaced
 * verbatim), and the multi objective nests the catalog's own kinds.
 */

/** The backend's own table-export row cap (visualization.py's `_MAX_ROWS`). */
const TABLE_EXPORT_MAX_ROWS = 5000;

/** The chemistry engine's own request cap (chemistry.py's max_length). */
const CHEMISTRY_MAX_SMILES = 500;

type GenerationMethod = "prior-sample" | "optimize";

/** One working-set molecule — the unified row both engines produce. */
interface WorkingRow {
  readonly smiles: string;
  /** The optimize lane's objective score (null on the sample lane). */
  readonly score: number | null;
  readonly breakdown: Readonly<Record<string, unknown>>;
}

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

/** A numbered workflow stage header (§11.4's fixed order). */
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
  icon?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="mt-8">
      <div className="flex items-center gap-2">
        <span className="flex h-6 w-6 items-center justify-center rounded-md bg-[var(--color-muted)] text-xs font-semibold text-[var(--color-muted-foreground)]">
          {index}
        </span>
        <h2 className="text-lg font-semibold text-[var(--color-foreground)]">{title}</h2>
        {icon ?? null}
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

/** The property sheet's numeric column picker payload. */
interface PropertyColumn {
  readonly name: string;
  readonly values: readonly (number | null)[];
}

export function GenerativePage() {
  // stage 1 — the generation method + the live objective catalog
  const [method, setMethod] = useState<GenerationMethod>("prior-sample");
  const [objectives, setObjectives] = useState<readonly ObjectiveCatalogEntry[]>([]);
  const [objectivesError, setObjectivesError] = useState<string | null>(null);
  const [objectivesBusy, setObjectivesBusy] = useState(true);

  // stage 2 — the starting molecules
  const [datasets, setDatasets] = useState<readonly DatasetView[]>([]);
  const [registryError, setRegistryError] = useState<string | null>(null);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<DatasetView | null>(null);
  const [selectionError, setSelectionError] = useState<string | null>(null);
  const [selectedColumn, setSelectedColumn] = useState<string>("");
  const [inlineSmiles, setInlineSmiles] = useState<string>(
    "CCO\nCCN\nCC(=O)O\nc1ccccc1C\nCC(C)Cc1ccc(O)cc1",
  );

  // stage 3 — the configuration (prior lane / optimize lane)
  const [priorName, setPriorName] = useState("smiles-prior");
  const [embeddingDim, setEmbeddingDim] = useState(48);
  const [hiddenDim, setHiddenDim] = useState(96);
  const [nLayers, setNLayers] = useState(1);
  const [epochs, setEpochs] = useState(8);
  const [batchSize, setBatchSize] = useState(32);
  const [lr, setLr] = useState(0.005);
  const [gradClip, setGradClip] = useState(5);
  const [priorSeed, setPriorSeed] = useState(42);

  const [objectiveKind, setObjectiveKind] = useState("qed");
  const [objectiveParams, setObjectiveParams] = useState<Record<string, string>>({});
  const [populationSize, setPopulationSize] = useState(64);
  const [generations, setGenerations] = useState(20);
  const [mutationRate, setMutationRate] = useState(0.08);
  const [crossoverRate, setCrossoverRate] = useState(0.7);
  const [elitism, setElitism] = useState(4);
  const [tournamentK, setTournamentK] = useState(3);
  const [maxSmilesLength, setMaxSmilesLength] = useState(120);
  const [gaSeed, setGaSeed] = useState(42);
  const [topN, setTopN] = useState(10);

  // the prior registry (a trained prior can be selected directly)
  const [priors, setPriors] = useState<readonly PriorView[]>([]);
  const [selectedPriorId, setSelectedPriorId] = useState<string>("");
  const [selectedPrior, setSelectedPrior] = useState<PriorView | null>(null);

  // stage 4 — generate
  const [priorJobId, setPriorJobId] = useState<string | null>(null);
  const [trainedPrior, setTrainedPrior] = useState<PriorView | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [trainBusy, setTrainBusy] = useState(false);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetricsPayload | null>(null);
  const [sampleN, setSampleN] = useState(20);
  const [temperature, setTemperature] = useState(1);
  const [topK, setTopK] = useState<string>("");
  const [sampleSeed, setSampleSeed] = useState<string>("");
  const [maxLength, setMaxLength] = useState(120);
  const [sampleResult, setSampleResult] = useState<SampleResponse | null>(null);
  const [sampleError, setSampleError] = useState<string | null>(null);
  const [sampleBusy, setSampleBusy] = useState(false);

  const [optimizeJobId, setOptimizeJobId] = useState<string | null>(null);
  const [optimization, setOptimization] = useState<OptimizationView | null>(null);
  const [optimizeError, setOptimizeError] = useState<string | null>(null);
  const [optimizeBusy, setOptimizeBusy] = useState(false);
  const [iterations, setIterations] = useState<readonly IterationUpdatePayload[]>([]);

  // the unified working set (both engines land here)
  const [workingSet, setWorkingSet] = useState<readonly WorkingRow[]>([]);
  const [workingSetLabel, setWorkingSetLabel] = useState<string>("");

  // stage 5/6 — validate + properties (the REAL descriptor engine)
  const [descriptorJobId, setDescriptorJobId] = useState<string | null>(null);
  const [descriptorFrame, setDescriptorFrame] = useState<DescriptorFrame | null>(null);
  const [descriptorError, setDescriptorError] = useState<string | null>(null);
  const [descriptorBusy, setDescriptorBusy] = useState(false);
  const [descriptorProgress, setDescriptorProgress] = useState<string | null>(null);

  // stage 7 — filter
  const [filterColumn, setFilterColumn] = useState<string>("");
  const [filterMin, setFilterMin] = useState<string>("");
  const [filterMax, setFilterMax] = useState<string>("");

  // stage 8 — rank
  const [rankKey, setRankKey] = useState<string>("");
  const [rankDesc, setRankDesc] = useState(true);

  // stage 9 — visualize
  const [figureFormat, setFigureFormat] = useState<"PNG" | "SVG" | "TIFF" | "JPEG">("PNG");
  const [lossFigure, setLossFigure] = useState<ArtifactView | null>(null);
  const [rankFigure, setRankFigure] = useState<ArtifactView | null>(null);
  const [figureError, setFigureError] = useState<string | null>(null);
  const [figureBusy, setFigureBusy] = useState(false);
  const [lossDigest, setLossDigest] = useState<string | null>(null);
  const [rankDigest, setRankDigest] = useState<string | null>(null);

  // stage 10 — export
  const [exportFormat, setExportFormat] = useState<"csv" | "tsv" | "xlsx" | "md" | "json" | "html">("csv");
  const [exportArtifact, setExportArtifact] = useState<ArtifactView | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportBusy, setExportBusy] = useState(false);
  const [exportDigest, setExportDigest] = useState<string | null>(null);

  // stage 1: the live objective catalog (the backend's own vocabulary)
  useEffect(() => {
    let cancelled = false;
    setObjectivesBusy(true);
    generativeApi
      .objectives()
      .then((response) => {
        if (!cancelled) {
          setObjectives(response.objectives);
          setObjectivesError(null);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setObjectivesError(describeError(error));
        }
      })
      .finally(() => {
        if (!cancelled) {
          setObjectivesBusy(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const refreshRegistries = useCallback(async () => {
    try {
      const response = await datasetsApi.list();
      setDatasets(response.datasets);
      setRegistryError(null);
    } catch (error) {
      setRegistryError(describeError(error));
    }
    try {
      const response = await generativeApi.priors();
      setPriors(response.priors);
    } catch {
      // the prior registry is best-effort here: the lane's own
      // errors surface at selection/training time
    }
  }, []);

  useEffect(() => {
    void refreshRegistries();
  }, [refreshRegistries]);

  const selectDataset = useCallback(async (datasetId: string) => {
    setSelectedDatasetId(datasetId);
    setSelectionError(null);
    setSelectedColumn("");
    try {
      const view = await datasetsApi.view(datasetId);
      setSelectedDataset(view);
    } catch (error) {
      setSelectedDataset(null);
      setSelectionError(describeError(error));
    }
  }, []);

  const selectPrior = useCallback((priorId: string) => {
    setSelectedPriorId(priorId);
    if (priorId === "") {
      setSelectedPrior(null);
      setTrainedPrior(null);
      return;
    }
    const found = priors.find((p) => p.prior_id === priorId);
    setSelectedPrior(found ?? null);
    setTrainedPrior(found ?? null);
  }, [priors]);

  // stage 2: the inline corpus (the route's own ≥ 5 rule)
  const inlineCorpus = useMemo(
    () =>
      inlineSmiles
        .split(/[\n,;]+/)
        .map((s) => s.trim())
        .filter((s) => s.length > 0),
    [inlineSmiles],
  );

  const corpusSource = useMemo<"inline" | "dataset" | "none">(() => {
    if (inlineCorpus.length >= 5) {
      return "inline";
    }
    if (selectedDatasetId !== null && selectedColumn !== "") {
      return "dataset";
    }
    return "none";
  }, [inlineCorpus.length, selectedDatasetId, selectedColumn]);

  // stage 3: the objective spec the optimize lane submits (the
  // catalog's own params; a missing required param is the route's 422)
  const objectiveSpec = useMemo<Record<string, unknown>>(() => {
    const spec: Record<string, unknown> = { kind: objectiveKind };
    for (const [key, raw] of Object.entries(objectiveParams)) {
      const trimmed = raw.trim();
      if (trimmed === "") {
        continue;
      }
      const numeric = Number(trimmed);
      spec[key] = Number.isFinite(numeric) && trimmed !== "" ? numeric : trimmed;
    }
    return spec;
  }, [objectiveKind, objectiveParams]);

  const objectiveEntry = useMemo(
    () => objectives.find((o) => o.kind === objectiveKind) ?? null,
    [objectives, objectiveKind],
  );

  // stage 4: the prior train lane (job mode — the REAL per-epoch loss
  // curve rides the training.metrics envelope + the REST relay)
  const handleTrainPrior = useCallback(async () => {
    setTrainBusy(true);
    setTrainError(null);
    setTrainedPrior(null);
    setTrainingMetrics(null);
    setSelectedPriorId("");
    setSelectedPrior(null);
    setSampleResult(null);
    setWorkingSet([]);
    setDescriptorFrame(null);
    setDescriptorJobId(null);
    try {
      const response = await generativeApi.trainPrior({
        smiles: corpusSource === "inline" ? inlineCorpus : null,
        datasetId: corpusSource === "dataset" ? selectedDatasetId : null,
        column: corpusSource === "dataset" ? selectedColumn : null,
        name: priorName,
        embeddingDim,
        hiddenDim,
        nLayers,
        epochs,
        batchSize,
        lr,
        gradClip,
        seed: priorSeed,
      });
      if (response.mode === "sync") {
        setTrainedPrior(response.prior);
        setPriorJobId(null);
      } else {
        setPriorJobId(response.job.job_id);
      }
    } catch (error) {
      setTrainError(describeError(error));
    } finally {
      setTrainBusy(false);
    }
  }, [
    corpusSource, inlineCorpus, selectedDatasetId, selectedColumn,
    priorName, embeddingDim, hiddenDim, nLayers, epochs, batchSize,
    lr, gradClip, priorSeed,
  ]);

  const handlePriorTrainResult = useCallback((result: unknown) => {
    try {
      const prior = parsePriorTrainResult(result, "generative prior train result");
      setTrainedPrior(prior);
      setSelectedPriorId(prior.prior_id);
      setSelectedPrior(prior);
    } catch (error) {
      setTrainError(describeError(error));
    }
  }, []);

  // stage 4: the sample lane (sync by design — the route's own shape)
  const handleSample = useCallback(async () => {
    const prior = trainedPrior ?? selectedPrior;
    if (prior === null) {
      return;
    }
    setSampleBusy(true);
    setSampleError(null);
    setSampleResult(null);
    setWorkingSet([]);
    setDescriptorFrame(null);
    setDescriptorJobId(null);
    setDescriptorProgress(null);
    setFilterColumn("");
    setRankKey("");
    setLossFigure(null);
    setRankFigure(null);
    setExportArtifact(null);
    try {
      const response = await generativeApi.sample({
        priorId: prior.prior_id,
        n: sampleN,
        temperature,
        topK: topK.trim() === "" ? null : Number(topK),
        seed: sampleSeed.trim() === "" ? null : Number(sampleSeed),
        maxLength,
      });
      setSampleResult(response);
      setWorkingSet(
        response.samples.map((smiles) => ({
          smiles,
          score: null,
          breakdown: {},
        })),
      );
      setWorkingSetLabel(
        `${response.n_sampled} sampled SMILES off ${prior.name} (sync lane)`,
      );
    } catch (error) {
      setSampleError(describeError(error));
    } finally {
      setSampleBusy(false);
    }
  }, [trainedPrior, selectedPrior, sampleN, temperature, topK, sampleSeed, maxLength]);

  // stage 4: the optimize lane (job mode — the evolution's own
  // iteration.update telemetry + the REST relay)
  const handleOptimize = useCallback(async () => {
    setOptimizeBusy(true);
    setOptimizeError(null);
    setOptimization(null);
    setIterations([]);
    setWorkingSet([]);
    setDescriptorFrame(null);
    setDescriptorJobId(null);
    setDescriptorProgress(null);
    setFilterColumn("");
    setRankKey("");
    setLossFigure(null);
    setRankFigure(null);
    setExportArtifact(null);
    try {
      const response = await generativeApi.optimize({
        objective: objectiveSpec,
        seedSmiles:
          corpusSource === "inline" && inlineCorpus.length > 0
            ? inlineCorpus.slice(0, 200)
            : null,
        datasetId: corpusSource === "dataset" ? selectedDatasetId : null,
        column: corpusSource === "dataset" ? selectedColumn : null,
        populationSize,
        generations,
        mutationRate,
        crossoverRate,
        elitism,
        tournamentK,
        maxSmilesLength,
        seed: gaSeed,
        topN,
      });
      if (response.mode === "sync") {
        setOptimization(response.optimization);
        setOptimizeJobId(null);
        setWorkingSet(
          response.optimization.top.map((row) => ({
            smiles: row.smiles,
            score: row.score,
            breakdown: row.breakdown,
          })),
        );
        setWorkingSetLabel(
          `the evolution's top ${response.optimization.top.length} molecules`,
        );
      } else {
        setOptimizeJobId(response.job.job_id);
      }
    } catch (error) {
      setOptimizeError(describeError(error));
    } finally {
      setOptimizeBusy(false);
    }
  }, [
    objectiveSpec, corpusSource, inlineCorpus, selectedDatasetId,
    selectedColumn, populationSize, generations, mutationRate,
    crossoverRate, elitism, tournamentK, maxSmilesLength, gaSeed, topN,
  ]);

  const handleOptimizeResult = useCallback((result: unknown) => {
    try {
      const optimization = parseOptimizeResult(result, "generative optimize result");
      setOptimization(optimization);
      setWorkingSet(
        optimization.top.map((row) => ({
          smiles: row.smiles,
          score: row.score,
          breakdown: row.breakdown,
        })),
      );
      setWorkingSetLabel(
        `the evolution's top ${optimization.top.length} molecules`,
      );
    } catch (error) {
      setOptimizeError(describeError(error));
    }
  }, []);

  // stage 4's live telemetry: the prior lane's training.metrics loss
  // curve + the optimize lane's iteration.update generations
  useEffect(() => {
    if (priorJobId === null && optimizeJobId === null) {
      return;
    }
    const stream = getEventStream();
    const unsubscribe = stream.subscribe((envelope) => {
      if (priorJobId !== null && envelope.job_id === priorJobId) {
        if (envelope.event_type === "training.metrics") {
          try {
            setTrainingMetrics(
              parseTrainingMetricsPayload(envelope.payload, "training.metrics"),
            );
          } catch (error) {
            setTrainError(describeError(error));
          }
        }
      }
      if (optimizeJobId !== null && envelope.job_id === optimizeJobId) {
        if (envelope.event_type === "iteration.update") {
          try {
            const payload = parseIterationUpdatePayload(
              envelope.payload,
              "iteration.update",
            );
            setIterations((prev) => [...prev, payload]);
          } catch (error) {
            setOptimizeError(describeError(error));
          }
        }
      }
    });
    return unsubscribe;
  }, [priorJobId, optimizeJobId]);

  // stage 5 + 6: the REAL descriptor engine over the working set (the
  // frame's own index IS the validity verdict — the backend drops
  // every SMILES it cannot parse)
  // the descriptor request's own contract: every SMILES non-empty
  // (chemistry.py's validator) — the RNN can sample zero-length
  // sequences, so the request list drops the blanks (they stay in the
  // validity verdict's invalid lane, stated below, never hidden)
  const descriptorRequest = useMemo(() => {
    const request: { row: WorkingRow; position: number }[] = [];
    workingSet.forEach((row, position) => {
      if (row.smiles.trim() !== "") {
        request.push({ row, position });
      }
    });
    return request;
  }, [workingSet]);

  const blankCount = workingSet.length - descriptorRequest.length;

  const handleValidate = useCallback(async () => {
    if (workingSet.length === 0) {
      return;
    }
    if (descriptorRequest.length === 0) {
      setDescriptorError(
        "every sampled SMILES is blank — the descriptor engine's request contract needs at least one non-empty string",
      );
      return;
    }
    if (descriptorRequest.length > CHEMISTRY_MAX_SMILES) {
      setDescriptorError(
        `the working set carries ${descriptorRequest.length} non-empty SMILES; the chemistry engine's own cap is ${CHEMISTRY_MAX_SMILES}`,
      );
      return;
    }
    setDescriptorBusy(true);
    setDescriptorError(null);
    setDescriptorFrame(null);
    setDescriptorJobId(null);
    setDescriptorProgress(null);
    setFilterColumn("");
    setRankKey("");
    try {
      const response = await chemistryApi.descriptors(
        descriptorRequest.map((entry) => entry.row.smiles),
        { descriptorSet: "rdkit_all" },
      );
      if (response.mode === "sync") {
        setDescriptorFrame(response.result);
      } else {
        setDescriptorJobId(response.job.job_id);
      }
    } catch (error) {
      setDescriptorError(describeError(error));
    } finally {
      setDescriptorBusy(false);
    }
  }, [workingSet, descriptorRequest]);

  const handleDescriptorResult = useCallback((result: unknown) => {
    try {
      const frame = parseDescriptorFrame(result, "descriptor job result");
      setDescriptorFrame(frame);
    } catch (error) {
      setDescriptorError(describeError(error));
    }
  }, []);

  // the descriptor lane's own progress telemetry (descriptor.progress)
  useEffect(() => {
    if (descriptorJobId === null) {
      return;
    }
    const stream = getEventStream();
    const unsubscribe = stream.subscribe((envelope) => {
      if (envelope.job_id !== descriptorJobId) {
        return;
      }
      if (envelope.event_type === "descriptor.progress") {
        const progress = envelope.payload["progress"];
        const stage = envelope.payload["stage"];
        setDescriptorProgress(
          `${typeof progress === "number" ? progress : "—"}% · ${
            typeof stage === "string" ? stage : ""
          }`,
        );
      }
    });
    return unsubscribe;
  }, [descriptorJobId]);

  // the validity verdict: the frame's index (the REQUEST-list
  // positions the backend kept) vs the working set — a blank SMILES
  // is invalid by definition (the backend's own compute_descriptors
  // counts blank/whitespace-only SMILES as invalid the same way)
  const validity = useMemo(() => {
    if (descriptorFrame === null || workingSet.length === 0) {
      return null;
    }
    const kept = new Set(
      descriptorFrame.index.map((v) => Number.parseInt(String(v), 10)),
    );
    const valid: string[] = [];
    const invalid: string[] = [];
    descriptorRequest.forEach((entry, requestPosition) => {
      if (kept.has(requestPosition)) {
        valid.push(entry.row.smiles);
      } else {
        invalid.push(entry.row.smiles);
      }
    });
    workingSet.forEach((row) => {
      if (row.smiles.trim() === "") {
        invalid.push(row.smiles);
      }
    });
    return { valid, invalid };
  }, [descriptorFrame, workingSet, descriptorRequest]);

  // the properties table: the frame's rows aligned to the working set
  // (through the request-list positions the frame's own index pins)
  const propertyRows = useMemo(() => {
    if (descriptorFrame === null || workingSet.length === 0) {
      return [];
    }
    const kept = new Map<number, readonly unknown[]>();
    descriptorFrame.index.forEach((rawIndex, rowNumber) => {
      const requestPosition = Number.parseInt(String(rawIndex), 10);
      const row = descriptorFrame.rows[rowNumber];
      if (Number.isFinite(requestPosition) && Array.isArray(row)) {
        kept.set(requestPosition, row);
      }
    });
    return workingSet.map((row) => {
      const requestPosition = descriptorRequest.findIndex(
        (entry) => entry.row === row,
      );
      return {
        smiles: row.smiles,
        values: requestPosition >= 0 ? (kept.get(requestPosition) ?? null) : null,
      };
    });
  }, [descriptorFrame, workingSet, descriptorRequest]);

  // the numeric property columns (for Filter + Rank)
  const propertyColumns = useMemo<readonly PropertyColumn[]>(() => {
    if (descriptorFrame === null) {
      return [];
    }
    const columns: PropertyColumn[] = [];
    descriptorFrame.columns.forEach((name, columnIndex) => {
      if (name === "SMILES") {
        return;
      }
      const values = propertyRows.map((row) => {
        const cell = row.values === null ? null : row.values[columnIndex];
        return typeof cell === "number" && Number.isFinite(cell) ? cell : null;
      });
      const numericCount = values.filter((v) => v !== null).length;
      if (numericCount >= 2) {
        columns.push({ name, values });
      }
    });
    return columns;
  }, [descriptorFrame, propertyRows]);

  useEffect(() => {
    if (propertyColumns.length > 0 && filterColumn === "") {
      setFilterColumn(propertyColumns[0]?.name ?? "");
    }
  }, [propertyColumns, filterColumn]);

  // stage 7: the filter (kept/filtered lanes over the REAL values)
  const filterMinValue = useMemo(() => {
    const raw = filterMin.trim();
    if (raw === "") {
      return null;
    }
    const value = Number(raw);
    return Number.isFinite(value) ? value : null;
  }, [filterMin]);

  const filterMaxValue = useMemo(() => {
    const raw = filterMax.trim();
    if (raw === "") {
      return null;
    }
    const value = Number(raw);
    return Number.isFinite(value) ? value : null;
  }, [filterMax]);

  const filterColumnIndex = useMemo(
    () => descriptorFrame?.columns.indexOf(filterColumn) ?? -1,
    [descriptorFrame, filterColumn],
  );

  const filteredRows = useMemo(() => {
    return propertyRows.map((row) => {
      const cell =
        filterColumnIndex >= 0 && row.values !== null
          ? row.values[filterColumnIndex]
          : null;
      const value = typeof cell === "number" && Number.isFinite(cell) ? cell : null;
      let kept = true;
      if (value !== null && filterMinValue !== null && value < filterMinValue) {
        kept = false;
      }
      if (value !== null && filterMaxValue !== null && value > filterMaxValue) {
        kept = false;
      }
      return { ...row, value, kept };
    });
  }, [propertyRows, filterColumnIndex, filterMinValue, filterMaxValue]);

  const keptCount = useMemo(
    () => filteredRows.filter((row) => row.kept).length,
    [filteredRows],
  );

  // stage 8: the rank (the objective score on the optimize lane, a
  // real property on the sample lane — the key names itself)
  const rankKeys = useMemo(() => {
    const keys = propertyColumns.map((c) => c.name);
    if (optimization !== null) {
      keys.unshift("objective score");
    }
    return keys;
  }, [propertyColumns, optimization]);

  useEffect(() => {
    if (rankKeys.length > 0 && (rankKey === "" || !rankKeys.includes(rankKey))) {
      setRankKey(rankKeys[0] ?? "");
    }
  }, [rankKeys, rankKey]);

  const rankedRows = useMemo(() => {
    const rows = filteredRows.map((row, i) => {
      const working = workingSet[i] ?? null;
      const scoreValue =
        rankKey === "objective score"
          ? working === null
            ? null
            : working.score
          : row.value;
      return {
        smiles: row.smiles,
        kept: row.kept,
        value: row.value,
        score: scoreValue,
      };
    });
    const withScore = rows.filter(
      (row) => typeof row.score === "number" && Number.isFinite(row.score),
    );
    const withoutScore = rows.filter(
      (row) => !(typeof row.score === "number" && Number.isFinite(row.score)),
    );
    withScore.sort((a, b) => {
      const av = a.score as number;
      const bv = b.score as number;
      return rankDesc ? bv - av : av - bv;
    });
    return [...withScore, ...withoutScore];
  }, [filteredRows, rankKey, rankDesc, workingSet, optimization]);

  // stage 9: the REAL figure artifacts
  const priorLosses = useMemo(
    () =>
      trainingMetrics !== null
        ? trainingMetrics.losses
        : (trainedPrior?.losses ?? selectedPrior?.losses ?? []),
    [trainingMetrics, trainedPrior, selectedPrior],
  );

  const handleVisualize = useCallback(async () => {
    setFigureBusy(true);
    setFigureError(null);
    setLossFigure(null);
    setRankFigure(null);
    setLossDigest(null);
    setRankDigest(null);
    try {
      if (priorLosses.length >= 2) {
        const figure = await generativeApi.priorLossFigure(
          priorLosses,
          figureFormat,
        );
        setLossFigure(figure);
      }
      const scoreRows = rankedRows.map((row) => ({
        score: row.score,
        kept: row.kept,
      }));
      if (scoreRows.length >= 2) {
        const figure = await generativeApi.rankFigure(
          scoreRows,
          figureFormat,
          rankKey,
        );
        setRankFigure(figure);
      }
      if (priorLosses.length < 2 && scoreRows.length < 2) {
        throw new Error(
          "neither figure lane has its minimum: a prior loss curve needs two epochs, a score profile needs two scored molecules",
        );
      }
    } catch (error) {
      setFigureError(describeError(error));
    } finally {
      setFigureBusy(false);
    }
  }, [priorLosses, rankedRows, figureFormat, rankKey]);

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
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
      return status;
    },
    [],
  );

  // stage 10: the export (the ranked table through the rows lane)
  const exportRows = useMemo(() => {
    return rankedRows.map((row, position) => {
      const record: Record<string, unknown> = {
        rank: position + 1,
        smiles: row.smiles,
        kept: row.kept,
      };
      if (typeof row.score === "number" && Number.isFinite(row.score)) {
        record[rankKey === "objective score" ? "score" : rankKey] = row.score;
      }
      if (filterColumn !== "") {
        record[filterColumn] = row.value;
      }
      return record;
    });
  }, [rankedRows, rankKey, filterColumn]);

  const handleExport = useCallback(async () => {
    setExportBusy(true);
    setExportError(null);
    setExportArtifact(null);
    setExportDigest(null);
    try {
      if (exportRows.length === 0) {
        throw new Error("the ranked table carries no rows to export");
      }
      if (exportRows.length > TABLE_EXPORT_MAX_ROWS) {
        throw new Error(
          `the ranked table carries ${exportRows.length} rows; the table export route's own cap is ${TABLE_EXPORT_MAX_ROWS}`,
        );
      }
      const artifact = await generativeApi.exportTable({
        rows: exportRows,
        format: exportFormat,
        filename: "generative-ranked-molecules",
      });
      setExportArtifact(artifact);
    } catch (error) {
      setExportError(describeError(error));
    } finally {
      setExportBusy(false);
    }
  }, [exportRows, exportFormat]);

  const handleExportDownload = useCallback(async () => {
    if (exportArtifact === null) {
      return;
    }
    setExportDigest(null);
    try {
      setExportDigest(await verifiedDownload(exportArtifact));
    } catch (error) {
      setExportError(describeError(error));
    }
  }, [exportArtifact, verifiedDownload]);

  const handleLossDownload = useCallback(async () => {
    if (lossFigure === null) {
      return;
    }
    setLossDigest(null);
    try {
      setLossDigest(await verifiedDownload(lossFigure));
    } catch (error) {
      setFigureError(describeError(error));
    }
  }, [lossFigure, verifiedDownload]);

  const handleRankDownload = useCallback(async () => {
    if (rankFigure === null) {
      return;
    }
    setRankDigest(null);
    try {
      setRankDigest(await verifiedDownload(rankFigure));
    } catch (error) {
      setFigureError(describeError(error));
    }
  }, [rankFigure, verifiedDownload]);

  const inlineCount = inlineCorpus.length;
  const lastIteration =
    iterations.length > 0
      ? (iterations[iterations.length - 1] ?? null)
      : null;
  const lastIterationStage = lastIteration === null ? "" : lastIteration.stage;
  const lastIterationProgress =
    lastIteration === null ? 0 : lastIteration.progress;

  return (
    <div className="mx-auto max-w-7xl px-4 pb-16 pt-6">
      <h1 className="text-xl font-semibold text-[var(--color-foreground)]">
        Generative AI
      </h1>
      <p className="mt-1 text-sm text-[var(--color-muted-foreground)]">
        The §11.4 workflow on the REAL backend: the RNN prior + sample
        lane and the genetic-optimizer evolution over the nine-objective
        scoring catalog, validated and profiled by the descriptor engine,
        ranked, visualized, and exported as artifacts.
      </p>

      <StageHeader
        index={1}
        title="Select Generation Method"
        hint="the backend's two REAL engines + the live objective catalog"
        icon={<Sparkles aria-hidden className="h-4 w-4" />}
      >
        <div className="grid gap-3 md:grid-cols-2">
          {(
            [
              {
                id: "prior-sample" as GenerationMethod,
                title: "RNN prior + sample",
                description:
                  "SmilesRnn.train_prior over a SMILES corpus (a managed job with the REAL per-epoch loss curve), then sampled molecules at your temperature / top-k / seed / max length.",
              },
              {
                id: "optimize" as GenerationMethod,
                title: "Genetic optimization",
                description:
                  "GeneticOptimizer over the scoring catalog's own objective — the evolution's per-generation iteration.update telemetry, native cancellation, and its ranked top molecules.",
              },
            ] as const
          ).map((entry) => (
            <Card
              key={entry.id}
              className={`cursor-pointer transition-colors ${
                method === entry.id
                  ? "border-[var(--color-primary)]"
                  : "hover:border-[var(--color-primary)]/50"
              }`}
              onClick={() => setMethod(entry.id)}
            >
              <CardHeader>
                <CardTitle className="text-base">{entry.title}</CardTitle>
                <CardDescription>{entry.description}</CardDescription>
              </CardHeader>
              <CardContent>
                <p className="text-[11px] text-[var(--color-muted-foreground)]">
                  {method === entry.id ? "selected — configure below" : "click to select"}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
        {objectivesError !== null ? (
          <div className="mt-3">
            <ErrorBanner message={objectivesError} />
          </div>
        ) : null}
        <div className="mt-4 overflow-hidden rounded-lg border border-[var(--color-border)]">
          <div className="max-h-64 overflow-auto">
            <table className="w-full border-collapse text-xs">
              <thead className="sticky top-0 bg-[var(--color-muted)]">
                <tr>
                  {["Kind", "Class", "Params", "Description"].map((header) => (
                    <th
                      key={header}
                      scope="col"
                      className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]"
                    >
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {objectives.map((entry) => (
                  <tr
                    key={entry.kind}
                    className={`cursor-pointer border-t border-[var(--color-border)] bg-[var(--color-card)] hover:bg-[var(--color-accent)]/50 ${
                      method === "optimize" && entry.kind === objectiveKind
                        ? "bg-[var(--color-accent)]"
                        : ""
                    }`}
                    onClick={() => {
                      setMethod("optimize");
                      setObjectiveKind(entry.kind);
                      setObjectiveParams({});
                    }}
                  >
                    <td className="whitespace-nowrap px-3 py-1.5 font-medium text-[var(--color-foreground)]">
                      {entry.kind}
                    </td>
                    <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                      {entry.class}
                    </td>
                    <td className="px-3 py-1.5 text-[10px] text-[var(--color-muted-foreground)]">
                      {Object.entries(entry.params)
                        .map(([k, v]) => `${k}=${formatValue(v)}`)
                        .join(" · ")}
                    </td>
                    <td className="px-3 py-1.5 text-[var(--color-muted-foreground)]">
                      {entry.description}
                    </td>
                  </tr>
                ))}
                {objectives.length === 0 ? (
                  <tr>
                    <td
                      colSpan={4}
                      className="px-3 py-4 text-center text-[var(--color-muted-foreground)]"
                    >
                      {objectivesBusy
                        ? "loading the objective catalog…"
                        : "the objective catalog carried no entries"}
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </div>
        <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
          The nine scoring objectives from the backend&apos;s live catalog
          — selecting a row switches to the optimization lane and seeds
          the Configure stage&apos;s spec editor with that objective&apos;s own
          parameter schema.
        </p>
      </StageHeader>

      <StageHeader
        index={2}
        title="Define Starting Molecules"
        hint="an inline corpus (≥ 5, the route's own rule) or a registered dataset's SMILES column"
        icon={<Dna aria-hidden className="h-4 w-4" />}
      >
        {registryError !== null ? <ErrorBanner message={registryError} /> : null}
        <div className="grid gap-4 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Inline corpus</CardTitle>
              <CardDescription>
                One SMILES per line (or comma-separated). The prior lane
                needs at least 5 — the route&apos;s own rule, its 422 surfaces
                verbatim below if not met.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <textarea
                value={inlineSmiles}
                onChange={(event) => setInlineSmiles(event.target.value)}
                rows={6}
                spellCheck={false}
                className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 font-mono text-xs"
              />
              <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                {inlineCount} parsed SMILES ·{" "}
                {corpusSource === "inline" ? "the active source" : "below the route's ≥ 5 rule"}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Dataset corpus</CardTitle>
              <CardDescription>
                A registered dataset&apos;s own SMILES column — resolved
                server-side (the authoritative frame never crosses the
                JSON boundary, §17.1&apos;s control-plane rule).
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="max-h-40 overflow-auto">
                <table className="w-full border-collapse text-xs">
                  <thead className="sticky top-0 bg-[var(--color-muted)]">
                    <tr>
                      {["Name", "Rows", "Target", "Task"].map((header) => (
                        <th
                          key={header}
                          scope="col"
                          className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]"
                        >
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
                          entry.dataset_id === selectedDatasetId
                            ? "bg-[var(--color-accent)]"
                            : "bg-[var(--color-card)] hover:bg-[var(--color-accent)]/50"
                        }`}
                      >
                        <td className="whitespace-nowrap px-3 py-1.5 font-medium text-[var(--color-foreground)]">
                          {entry.name}
                        </td>
                        <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                          {entry.shape[0]}
                        </td>
                        <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                          {entry.target ?? "—"}
                        </td>
                        <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                          {entry.task ?? "—"}
                        </td>
                      </tr>
                    ))}
                    {datasets.length === 0 ? (
                      <tr>
                        <td
                          colSpan={4}
                          className="px-3 py-4 text-center text-[var(--color-muted-foreground)]"
                        >
                          No datasets registered yet — import one in the Datasets workspace.
                        </td>
                      </tr>
                    ) : null}
                  </tbody>
                </table>
              </div>
              {selectedDataset !== null ? (
                <select
                  value={selectedColumn}
                  onChange={(event) => setSelectedColumn(event.target.value)}
                  className="mt-3 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
                >
                  <option value="">select the corpus column…</option>
                  {selectedDataset.columns.map((column) => (
                    <option key={column} value={column}>
                      {column}
                    </option>
                  ))}
                </select>
              ) : null}
              {selectionError !== null ? (
                <div className="mt-2">
                  <ErrorBanner message={selectionError} />
                </div>
              ) : null}
            </CardContent>
          </Card>
        </div>
        <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
          Active corpus source:{" "}
          {corpusSource === "inline"
            ? `inline (${inlineCount} SMILES)`
            : corpusSource === "dataset"
              ? `dataset ${selectedDataset?.name ?? ""} · column ${selectedColumn}`
              : "none — the Generate lane needs one"}
        </p>
      </StageHeader>

      <StageHeader
        index={3}
        title="Configure Constraints/Objectives"
        hint="the engine's own parameters, mirrored from the route's bounds"
        icon={<Target aria-hidden className="h-4 w-4" />}
      >
        {method === "prior-sample" ? (
          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Prior architecture + training</CardTitle>
                <CardDescription>
                  The `PriorTrainRequest` bounds, verbatim from the route.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <label className="flex flex-col gap-1">
                    name
                    <input
                      value={priorName}
                      onChange={(event) => setPriorName(event.target.value)}
                      className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    embedding dim (8–256)
                    <input
                      type="number"
                      min={8}
                      max={256}
                      value={embeddingDim}
                      onChange={(event) => setEmbeddingDim(Number(event.target.value))}
                      className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    hidden dim (8–512)
                    <input
                      type="number"
                      min={8}
                      max={512}
                      value={hiddenDim}
                      onChange={(event) => setHiddenDim(Number(event.target.value))}
                      className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    layers (1–8)
                    <input
                      type="number"
                      min={1}
                      max={8}
                      value={nLayers}
                      onChange={(event) => setNLayers(Number(event.target.value))}
                      className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    epochs (1–200)
                    <input
                      type="number"
                      min={1}
                      max={200}
                      value={epochs}
                      onChange={(event) => setEpochs(Number(event.target.value))}
                      className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    batch size (4–512)
                    <input
                      type="number"
                      min={4}
                      max={512}
                      value={batchSize}
                      onChange={(event) => setBatchSize(Number(event.target.value))}
                      className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    learning rate (0–1)
                    <input
                      type="number"
                      min={0.0001}
                      max={1}
                      step={0.001}
                      value={lr}
                      onChange={(event) => setLr(Number(event.target.value))}
                      className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    grad clip (0–100)
                    <input
                      type="number"
                      min={0.1}
                      max={100}
                      step={0.5}
                      value={gradClip}
                      onChange={(event) => setGradClip(Number(event.target.value))}
                      className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    seed
                    <input
                      type="number"
                      min={0}
                      value={priorSeed}
                      onChange={(event) => setPriorSeed(Number(event.target.value))}
                      className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                    />
                  </label>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-base">The session prior registry</CardTitle>
                <CardDescription>
                  A trained prior can be selected here directly — the
                  sample lane rides any prior the session registered.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <select
                  value={selectedPriorId}
                  onChange={(event) => selectPrior(event.target.value)}
                  className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
                >
                  <option value="">train a new prior below…</option>
                  {priors.map((prior) => (
                    <option key={prior.prior_id} value={prior.prior_id}>
                      {prior.name} · {prior.n_train_smiles} SMILES ·{" "}
                      {prior.n_epochs} epochs · final loss{" "}
                      {formatValue(prior.final_loss)}
                    </option>
                  ))}
                </select>
                {selectedPrior !== null ? (
                  <div className="mt-3">
                    <p className="text-xs text-[var(--color-muted-foreground)]">
                      {selectedPrior.prior_id} · kind {selectedPrior.kind} ·{" "}
                      corpus {formatValue(selectedPrior.corpus_source["source"])} ·
                      created {selectedPrior.created_at}
                    </p>
                    <p className="mt-1 text-xs text-[var(--color-muted-foreground)]">
                      losses: {selectedPrior.losses.map((v) => formatValue(v)).join(", ")}
                    </p>
                  </div>
                ) : null}
              </CardContent>
            </Card>
          </div>
        ) : (
          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Objective spec</CardTitle>
                <CardDescription>
                  The catalog&apos;s own parameter schema — a missing
                  required param is the route&apos;s honest 422, surfaced
                  verbatim at Generate.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <select
                  value={objectiveKind}
                  onChange={(event) => {
                    setObjectiveKind(event.target.value);
                    setObjectiveParams({});
                  }}
                  className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
                >
                  {OBJECTIVE_KINDS.map((kind) => (
                    <option key={kind} value={kind}>
                      {kind}
                    </option>
                  ))}
                </select>
                <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                  {objectiveEntry?.description ?? ""}
                </p>
                <div className="mt-3 grid grid-cols-2 gap-3">
                  {Object.entries(objectiveEntry?.params ?? {}).map(
                    ([param, example]) => (
                      <label key={param} className="flex flex-col gap-1 text-xs">
                        {param}
                        {param in (objectiveEntry?.params ?? {})
                          ? ` (e.g. ${formatValue(example)})`
                          : ""}
                        <input
                          value={objectiveParams[param] ?? ""}
                          onChange={(event) =>
                            setObjectiveParams((prev) => ({
                              ...prev,
                              [param]: event.target.value,
                            }))
                          }
                          placeholder={String(example ?? "")}
                          className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                        />
                      </label>
                    ),
                  )}
                </div>
                <p className="mt-3 font-mono text-[11px] text-[var(--color-muted-foreground)]">
                  spec: {JSON.stringify(objectiveSpec)}
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Evolution parameters</CardTitle>
                <CardDescription>
                  The `OptimizeRequest` bounds, verbatim from the route.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-3 text-xs">
                  <label className="flex flex-col gap-1">
                    population (4–512)
                    <input
                      type="number"
                      min={4}
                      max={512}
                      value={populationSize}
                      onChange={(event) =>
                        setPopulationSize(Number(event.target.value))
                      }
                      className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    generations (1–500)
                    <input
                      type="number"
                      min={1}
                      max={500}
                      value={generations}
                      onChange={(event) => setGenerations(Number(event.target.value))}
                      className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    mutation rate (0–1)
                    <input
                      type="number"
                      min={0}
                      max={1}
                      step={0.01}
                      value={mutationRate}
                      onChange={(event) => setMutationRate(Number(event.target.value))}
                      className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    crossover rate (0–1)
                    <input
                      type="number"
                      min={0}
                      max={1}
                      step={0.01}
                      value={crossoverRate}
                      onChange={(event) => setCrossoverRate(Number(event.target.value))}
                      className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    elitism (0–64)
                    <input
                      type="number"
                      min={0}
                      max={64}
                      value={elitism}
                      onChange={(event) => setElitism(Number(event.target.value))}
                      className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    tournament k (2–16)
                    <input
                      type="number"
                      min={2}
                      max={16}
                      value={tournamentK}
                      onChange={(event) => setTournamentK(Number(event.target.value))}
                      className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    max SMILES length (20–256)
                    <input
                      type="number"
                      min={20}
                      max={256}
                      value={maxSmilesLength}
                      onChange={(event) => setMaxSmilesLength(Number(event.target.value))}
                      className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    seed
                    <input
                      type="number"
                      min={0}
                      value={gaSeed}
                      onChange={(event) => setGaSeed(Number(event.target.value))}
                      className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                    />
                  </label>
                  <label className="flex flex-col gap-1">
                    top n (1–50)
                    <input
                      type="number"
                      min={1}
                      max={50}
                      value={topN}
                      onChange={(event) => setTopN(Number(event.target.value))}
                      className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                    />
                  </label>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </StageHeader>

      <StageHeader
        index={4}
        title="Generate"
        hint="the REAL engine — job mode with live telemetry"
        icon={<Wand2 aria-hidden className="h-4 w-4" />}
      >
        {method === "prior-sample" ? (
          <>
            {trainError !== null ? <ErrorBanner message={trainError} /> : null}
            <div className="flex flex-wrap items-center gap-3">
              <Button
                onClick={() => void handleTrainPrior()}
                disabled={
                  trainBusy ||
                  corpusSource === "none" ||
                  priorJobId !== null
                }
              >
                {trainBusy ? "training…" : "Train prior"}
              </Button>
              <span className="text-xs text-[var(--color-muted-foreground)]">
                POST /api/v1/generative/rnn/train (job mode) — the REAL
                SmilesRnn.train_prior over the corpus
              </span>
            </div>
            {priorJobId !== null ? (
              <div className="mt-3">
                <JobRunCard
                  jobId={priorJobId}
                  resultLabel="the prior view relay"
                  onResult={handlePriorTrainResult}
                />
                {trainingMetrics !== null ? (
                  <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                    training.metrics: {trainingMetrics.stage} ·{" "}
                    {trainingMetrics.epochs} epochs · final loss{" "}
                    {formatValue(trainingMetrics.final_loss)} —{" "}
                    {trainingMetrics.losses
                      .map((v) => formatValue(v))
                      .join(", ")}
                  </p>
                ) : (
                  <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                    the prior lane publishes its loss curve as ONE
                    training.metrics envelope when training ends — the
                    live per-epoch values arrive with the authoritative
                    REST relay below
                  </p>
                )}
              </div>
            ) : null}
            {trainedPrior !== null ? (
              <Card className="mt-3">
                <CardHeader>
                  <CardTitle className="text-base">{trainedPrior.name}</CardTitle>
                  <CardDescription>
                    {trainedPrior.prior_id} · {trainedPrior.n_train_smiles}{" "}
                    training SMILES · {trainedPrior.n_epochs} epochs ·
                    final loss {formatValue(trainedPrior.final_loss)} ·
                    corpus {formatValue(trainedPrior.corpus_source["source"])}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 gap-3 text-xs md:grid-cols-3">
                    <label className="flex flex-col gap-1">
                      n (1–500)
                      <input
                        type="number"
                        min={1}
                        max={500}
                        value={sampleN}
                        onChange={(event) => setSampleN(Number(event.target.value))}
                        className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                      />
                    </label>
                    <label className="flex flex-col gap-1">
                      temperature (0.05–3)
                      <input
                        type="number"
                        min={0.05}
                        max={3}
                        step={0.05}
                        value={temperature}
                        onChange={(event) => setTemperature(Number(event.target.value))}
                        className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                      />
                    </label>
                    <label className="flex flex-col gap-1">
                      top k (1–256, blank = off)
                      <input
                        value={topK}
                        onChange={(event) => setTopK(event.target.value)}
                        placeholder="off"
                        className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                      />
                    </label>
                    <label className="flex flex-col gap-1">
                      seed (blank = random)
                      <input
                        value={sampleSeed}
                        onChange={(event) => setSampleSeed(event.target.value)}
                        placeholder="random"
                        className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                      />
                    </label>
                    <label className="flex flex-col gap-1">
                      max length (10–256)
                      <input
                        type="number"
                        min={10}
                        max={256}
                        value={maxLength}
                        onChange={(event) => setMaxLength(Number(event.target.value))}
                        className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                      />
                    </label>
                  </div>
                  <div className="mt-3 flex flex-wrap items-center gap-3">
                    <Button onClick={() => void handleSample()} disabled={sampleBusy}>
                      {sampleBusy ? "sampling…" : "Sample molecules"}
                    </Button>
                    <span className="text-xs text-[var(--color-muted-foreground)]">
                      POST /api/v1/generative/rnn/sample — this lane is
                      sync by design: the route declares no job mode
                      (honest, no pretend tracking)
                    </span>
                  </div>
                  {sampleError !== null ? (
                    <div className="mt-2">
                      <ErrorBanner message={sampleError} />
                    </div>
                  ) : null}
                  {sampleResult !== null ? (
                    <div className="mt-3">
                      <p className="text-xs text-[var(--color-muted-foreground)]">
                        {sampleResult.n_sampled} sampled ·{" "}
                        {sampleResult.unique} unique · temperature{" "}
                        {sampleResult.params.temperature} · top_k{" "}
                        {formatValue(sampleResult.params.top_k)}
                      </p>
                      <div className="mt-2 max-h-40 overflow-auto rounded-md border border-[var(--color-border)]">
                        <ul className="divide-y divide-[var(--color-border)]">
                          {sampleResult.samples.map((smiles, i) => (
                            <li
                              key={`${i}-${smiles}`}
                              className="px-3 py-1 font-mono text-xs text-[var(--color-foreground)]"
                            >
                              {smiles}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  ) : null}
                </CardContent>
              </Card>
            ) : (
              <p className="mt-3 text-sm text-[var(--color-muted-foreground)]">
                Train a prior above (or select one from the session
                registry in the Configure stage) — the sample lane opens
                with the registered prior view.
              </p>
            )}
          </>
        ) : (
          <>
            {optimizeError !== null ? <ErrorBanner message={optimizeError} /> : null}
            <div className="flex flex-wrap items-center gap-3">
              <Button
                onClick={() => void handleOptimize()}
                disabled={optimizeBusy || optimizeJobId !== null}
              >
                {optimizeBusy ? "submitting…" : "Run evolution"}
              </Button>
              <span className="text-xs text-[var(--color-muted-foreground)]">
                POST /api/v1/generative/optimize (job mode) — the REAL
                GeneticOptimizer over the objective spec; cancelling is
                native (the job lands in CANCELLED, never FAILED)
              </span>
            </div>
            {optimizeJobId !== null ? (
              <div className="mt-3">
                <JobRunCard
                  jobId={optimizeJobId}
                  resultLabel="the optimization view relay"
                  onResult={handleOptimizeResult}
                />
                {lastIteration !== null ? (
                  <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                    {iterations.length} iteration.update envelopes ·
                    last: {lastIterationProgress}% · {lastIterationStage}
                  </p>
                ) : (
                  <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                    the optimizer&apos;s own progress callback drives one
                    iteration.update envelope per generation
                  </p>
                )}
              </div>
            ) : null}
            {optimization !== null ? (
              <Card className="mt-3">
                <CardHeader>
                  <CardTitle className="text-base">The evolution&apos;s verdict</CardTitle>
                  <CardDescription>
                    best {formatValue(optimization.best)} · score{" "}
                    {formatValue(optimization.best_score)} ·{" "}
                    {optimization.generations} generations ·{" "}
                    {optimization.top.length} ranked molecules
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="max-h-48 overflow-auto rounded-md border border-[var(--color-border)]">
                    <table className="w-full border-collapse text-xs">
                      <thead className="sticky top-0 bg-[var(--color-muted)]">
                        <tr>
                          {["SMILES", "Score", "Breakdown"].map((header) => (
                            <th
                              key={header}
                              scope="col"
                              className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]"
                            >
                              {header}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {optimization.top.map((row) => (
                          <tr
                            key={row.smiles}
                            className="border-t border-[var(--color-border)] bg-[var(--color-card)]"
                          >
                            <td className="px-3 py-1.5 font-mono text-[11px] text-[var(--color-foreground)]">
                              {row.smiles}
                            </td>
                            <td className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]">
                              {formatValue(row.score)}
                            </td>
                            <td className="px-3 py-1.5 text-[10px] text-[var(--color-muted-foreground)]">
                              {Object.entries(row.breakdown)
                                .map(([k, v]) => `${k}=${formatValue(v)}`)
                                .join(" · ")}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardContent>
              </Card>
            ) : null}
          </>
        )}
        {workingSet.length > 0 ? (
          <p className="mt-3 text-sm text-[var(--color-foreground)]">
            Working set: {workingSetLabel} — continue to Validate below.
          </p>
        ) : null}
      </StageHeader>

      <StageHeader
        index={5}
        title="Validate"
        hint="the descriptor engine's own verdict — the frame's index IS the validity answer"
        icon={<FlaskConical aria-hidden className="h-4 w-4" />}
      >
        {descriptorError !== null ? <ErrorBanner message={descriptorError} /> : null}
        {blankCount > 0 ? (
          <p className="mb-2 text-xs text-[var(--color-muted-foreground)]">
            {blankCount} blank sampled SMILES — the descriptor request’s
            own contract (chemistry.py’s non-empty-string validator)
            drops them before the request fires; they count in the
            invalid lane below, the backend’s own blank-as-invalid
            verdict, never hidden.
          </p>
        ) : null}
        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={() => void handleValidate()} disabled={descriptorBusy || workingSet.length === 0}>
            {descriptorBusy ? "validating…" : "Validate generated molecules"}
          </Button>
          <span className="text-xs text-[var(--color-muted-foreground)]">
            POST /api/v1/chemistry/descriptors (rdkit_all) over the working
            set — the backend drops every SMILES it cannot parse, so the
            frame&apos;s own index is the validity verdict, never a
            client-side guess
          </span>
        </div>
        {descriptorJobId !== null ? (
          <div className="mt-3">
            <JobRunCard
              jobId={descriptorJobId}
              resultLabel="the descriptor frame relay"
              onResult={handleDescriptorResult}
            />
            {descriptorProgress !== null ? (
              <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                descriptor.progress: {descriptorProgress}
              </p>
            ) : null}
          </div>
        ) : null}
        {validity !== null ? (
          <div className="mt-3 grid gap-3 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">
                  Valid — {validity.valid.length}
                </CardTitle>
                <CardDescription>
                  the molecules the descriptor engine actually parsed
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="max-h-40 overflow-auto">
                  <ul className="font-mono text-xs text-[var(--color-foreground)]">
                    {validity.valid.map((smiles) => (
                      <li key={smiles}>{smiles}</li>
                    ))}
                  </ul>
                </div>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-base">
                  Invalid — {validity.invalid.length}
                </CardTitle>
                <CardDescription>
                  dropped by the backend&apos;s own parse — stated, never hidden
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="max-h-40 overflow-auto">
                  <ul className="font-mono text-xs text-[var(--color-foreground)]">
                    {validity.invalid.map((smiles) => (
                      <li key={smiles}>{smiles}</li>
                    ))}
                  </ul>
                </div>
              </CardContent>
            </Card>
          </div>
        ) : (
          <p className="mt-3 text-sm text-[var(--color-muted-foreground)]">
            The verdict arrives with the descriptor frame — run the lane
            above (needs a working set from Generate).
          </p>
        )}
      </StageHeader>

      <StageHeader
        index={6}
        title="Calculate Properties"
        hint="the REAL rdkit_all descriptor frame over the valid molecules"
        icon={<Beaker aria-hidden className="h-4 w-4" />}
      >
        {descriptorFrame !== null ? (
          <>
            <p className="text-xs text-[var(--color-muted-foreground)]">
              {descriptorFrame.shape[0]} valid molecules ×{" "}
              {descriptorFrame.shape[1]} columns · the full rdkit_all set
              (MolWt, LogP, TPSA, …) — the values below are the backend&apos;s
              own computation, mirrored row-for-row
            </p>
            <div className="mt-2 max-h-72 overflow-auto rounded-lg border border-[var(--color-border)]">
              <table className="w-full border-collapse text-xs">
                <thead className="sticky top-0 bg-[var(--color-muted)]">
                  <tr>
                    {["SMILES", ...descriptorFrame.columns.filter((c) => c !== "SMILES")].map(
                      (header) => (
                        <th
                          key={header}
                          scope="col"
                          className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]"
                        >
                          {header}
                        </th>
                      ),
                    )}
                  </tr>
                </thead>
                <tbody>
                  {propertyRows.map((row) => (
                    <tr
                      key={row.smiles}
                      className="border-t border-[var(--color-border)] bg-[var(--color-card)]"
                    >
                      <td className="whitespace-nowrap px-3 py-1.5 font-mono text-[11px] text-[var(--color-foreground)]">
                        {row.smiles}
                      </td>
                      {(row.values ?? []).map((cell, i) => (
                        <td
                          key={i}
                          className="whitespace-nowrap px-3 py-1.5 text-[var(--color-muted-foreground)]"
                        >
                          {formatValue(cell)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
              The display shows every valid molecule (bounded by the
              chemistry engine&apos;s own 500-SMILES request cap, enforced
              before the request fires) — a display limit never silently
              truncates the working set the backend analyzed.
            </p>
          </>
        ) : (
          <p className="text-sm text-[var(--color-muted-foreground)]">
            The properties arrive with the Validate lane&apos;s descriptor
            frame — this stage displays the same REAL frame, not a
            second computation.
          </p>
        )}
      </StageHeader>

      <StageHeader
        index={7}
        title="Filter"
        hint="kept / filtered lanes over the REAL computed values"
        icon={<Filter aria-hidden className="h-4 w-4" />}
      >
        {propertyColumns.length > 0 ? (
          <>
            <div className="flex flex-wrap items-end gap-3">
              <label className="flex flex-col gap-1 text-xs">
                property
                <select
                  value={filterColumn}
                  onChange={(event) => setFilterColumn(event.target.value)}
                  className="w-56 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
                >
                  {propertyColumns.map((column) => (
                    <option key={column.name} value={column.name}>
                      {column.name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-1 text-xs">
                min (blank = none)
                <input
                  value={filterMin}
                  onChange={(event) => setFilterMin(event.target.value)}
                  placeholder="no lower bound"
                  className="w-40 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                />
              </label>
              <label className="flex flex-col gap-1 text-xs">
                max (blank = none)
                <input
                  value={filterMax}
                  onChange={(event) => setFilterMax(event.target.value)}
                  placeholder="no upper bound"
                  className="w-40 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                />
              </label>
            </div>
            <p className="mt-3 text-sm text-[var(--color-foreground)]">
              {keptCount} kept · {filteredRows.length - keptCount} filtered ·
              on {filterColumn} between{" "}
              {filterMinValue === null ? "−∞" : filterMinValue} and{" "}
              {filterMaxValue === null ? "+∞" : filterMaxValue} — the
              thresholds apply to the backend&apos;s own computed values;
              molecules whose property is missing stay kept (stated,
              never silently dropped)
            </p>
          </>
        ) : (
          <p className="text-sm text-[var(--color-muted-foreground)]">
            The Filter stage needs the descriptor frame&apos;s numeric
            columns — run Validate first.
          </p>
        )}
      </StageHeader>

      <StageHeader
        index={8}
        title="Rank"
        hint="the objective score or a selected property — descending by default"
        icon={<ListOrdered aria-hidden className="h-4 w-4" />}
      >
        {rankKeys.length > 0 ? (
          <>
            <div className="flex flex-wrap items-end gap-3">
              <label className="flex flex-col gap-1 text-xs">
                ranking key
                <select
                  value={rankKey}
                  onChange={(event) => setRankKey(event.target.value)}
                  className="w-56 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
                >
                  {rankKeys.map((key) => (
                    <option key={key} value={key}>
                      {key}
                    </option>
                  ))}
                </select>
              </label>
              <label className="flex flex-col gap-1 text-xs">
                direction
                <select
                  value={rankDesc ? "desc" : "asc"}
                  onChange={(event) =>
                    setRankDesc(event.target.value === "desc")
                  }
                  className="w-32 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
                >
                  <option value="desc">descending</option>
                  <option value="asc">ascending</option>
                </select>
              </label>
            </div>
            <div className="mt-3 max-h-64 overflow-auto rounded-lg border border-[var(--color-border)]">
              <table className="w-full border-collapse text-xs">
                <thead className="sticky top-0 bg-[var(--color-muted)]">
                  <tr>
                    {["Rank", "SMILES", "Kept", rankKey].map((header) => (
                      <th
                        key={header}
                        scope="col"
                        className="whitespace-nowrap px-3 py-2 text-left font-semibold text-[var(--color-foreground)]"
                      >
                        {header}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rankedRows.map((row, position) => (
                    <tr
                      key={row.smiles}
                      className={`border-t border-[var(--color-border)] ${
                        row.kept
                          ? "bg-[var(--color-card)]"
                          : "bg-[var(--color-card)] opacity-60"
                      }`}
                    >
                      <td className="px-3 py-1.5 text-[var(--color-muted-foreground)]">
                        {position + 1}
                      </td>
                      <td className="px-3 py-1.5 font-mono text-[11px] text-[var(--color-foreground)]">
                        {row.smiles}
                      </td>
                      <td className="px-3 py-1.5 text-[var(--color-muted-foreground)]">
                        {row.kept ? "kept" : "filtered"}
                      </td>
                      <td className="px-3 py-1.5 text-[var(--color-muted-foreground)]">
                        {formatValue(row.score)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
              The ranked table orders the REAL values — the objective
              score the evolution assigned (the optimize lane) or the
              property the descriptor engine computed (the sample lane);
              unscored rows sink to the bottom, stated, never invented.
            </p>
          </>
        ) : (
          <p className="text-sm text-[var(--color-muted-foreground)]">
            The Rank stage needs the descriptor frame (or the
            optimization&apos;s own scores) — run Validate or Generate
            first.
          </p>
        )}
      </StageHeader>

      <StageHeader
        index={9}
        title="Visualize"
        hint="REAL figure artifacts — the prior loss curve + the ranked score profile"
        icon={<LineChart aria-hidden className="h-4 w-4" />}
      >
        {figureError !== null ? <ErrorBanner message={figureError} /> : null}
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1 text-xs">
            format
            <select
              value={figureFormat}
              onChange={(event) =>
                setFigureFormat(
                  event.target.value as "PNG" | "SVG" | "TIFF" | "JPEG",
                )
              }
              className="w-32 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
            >
              {(["PNG", "SVG", "TIFF", "JPEG"] as const).map((format) => (
                <option key={format} value={format}>
                  {format}
                </option>
              ))}
            </select>
          </label>
          <Button onClick={() => void handleVisualize()} disabled={figureBusy}>
            {figureBusy ? "rendering…" : "Register figures"}
          </Button>
          <span className="text-xs text-[var(--color-muted-foreground)]">
            POST /api/v1/artifacts/figure — the §17 four-format export
            stack; the prior lane needs a trained prior&apos;s loss curve
            (≥ 2 epochs), the profile lane needs ≥ 2 scored molecules
          </span>
        </div>
        <div className="mt-3 grid gap-3 md:grid-cols-2">
          {lossFigure !== null ? (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Prior loss curve</CardTitle>
                <CardDescription>
                  {lossFigure.filename} · {lossFigure.size_bytes} B ·{" "}
                  {lossFigure.sha256.slice(0, 16)}…
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-3">
                  <Button size="sm" variant="outline" onClick={() => void handleLossDownload()}>
                    <ArrowDownToLine aria-hidden className="mr-1 h-3 w-3" />
                    Download
                  </Button>
                  {lossDigest !== null ? (
                    <span className="text-[11px] text-[var(--color-muted-foreground)]">
                      {lossDigest}
                    </span>
                  ) : null}
                </div>
              </CardContent>
            </Card>
          ) : null}
          {rankFigure !== null ? (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">
                  Ranked score profile
                </CardTitle>
                <CardDescription>
                  {rankFigure.filename} · {rankFigure.size_bytes} B ·{" "}
                  {rankFigure.sha256.slice(0, 16)}…
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex items-center gap-3">
                  <Button size="sm" variant="outline" onClick={() => void handleRankDownload()}>
                    <ArrowDownToLine aria-hidden className="mr-1 h-3 w-3" />
                    Download
                  </Button>
                  {rankDigest !== null ? (
                    <span className="text-[11px] text-[var(--color-muted-foreground)]">
                      {rankDigest}
                    </span>
                  ) : null}
                </div>
              </CardContent>
            </Card>
          ) : null}
        </div>
        {lossFigure === null && rankFigure === null ? (
          <p className="mt-2 text-sm text-[var(--color-muted-foreground)]">
            The figure artifacts register through the export stack — the
            prior lane needs a trained prior (its loss curve), both lanes
            need the ranked table.
          </p>
        ) : null}
      </StageHeader>

      <StageHeader
        index={10}
        title="Export"
        hint="the ranked table through the rows-based table lane, digest-verified"
        icon={<ArrowDownToLine aria-hidden className="h-4 w-4" />}
      >
        {exportError !== null ? <ErrorBanner message={exportError} /> : null}
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1 text-xs">
            format
            <select
              value={exportFormat}
              onChange={(event) =>
                setExportFormat(
                  event.target.value as
                    | "csv"
                    | "tsv"
                    | "xlsx"
                    | "md"
                    | "json"
                    | "html",
                )
              }
              className="w-32 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
            >
              {(["csv", "tsv", "xlsx", "md", "json", "html"] as const).map(
                (format) => (
                  <option key={format} value={format}>
                    {format}
                  </option>
                ),
              )}
            </select>
          </label>
          <Button onClick={() => void handleExport()} disabled={exportBusy}>
            {exportBusy ? "exporting…" : "Export ranked table"}
          </Button>
          <span className="text-xs text-[var(--color-muted-foreground)]">
            POST /api/v1/visualization/table — {exportRows.length} rows ·
            the backend&apos;s own 5000-row cap mirrored client-side
          </span>
        </div>
        {exportArtifact !== null ? (
          <Card className="mt-3">
            <CardHeader>
              <CardTitle className="text-base">
                {exportArtifact.filename}
              </CardTitle>
              <CardDescription>
                {exportArtifact.size_bytes} B ·{" "}
                {exportArtifact.sha256.slice(0, 16)}… · artifact{" "}
                {exportArtifact.artifact_id}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-3">
                <Button size="sm" variant="outline" onClick={() => void handleExportDownload()}>
                  <ArrowDownToLine aria-hidden className="mr-1 h-3 w-3" />
                  Download
                </Button>
                {exportDigest !== null ? (
                  <span className="text-[11px] text-[var(--color-muted-foreground)]">
                    {exportDigest}
                  </span>
                ) : null}
              </div>
            </CardContent>
          </Card>
        ) : (
          <p className="mt-2 text-sm text-[var(--color-muted-foreground)]">
            The export rides the LAST ranked table — run Rank first; the
            artifact downloads through the token-header lane with digest
            verification.
          </p>
        )}
      </StageHeader>
    </div>
  );
}
