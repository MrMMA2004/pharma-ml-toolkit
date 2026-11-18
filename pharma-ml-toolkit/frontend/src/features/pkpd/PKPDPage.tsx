import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ArrowDownToLine,
  Beaker,
  LineChart,
  Play,
  Sigma,
  Stethoscope,
  Upload,
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
  artifactsApi,
  type ArtifactView,
} from "@/services/artifacts/client";
import { getEventStream } from "@/services/websocket/client";
import {
  assertNcaSeriesBounds,
  assertPopulationBounds,
  assertTimesBounds,
  parseJobProgressPayload,
  parsePopulationBody,
  pkpdApi,
  type BatemanResponse,
  type IvBolusResponse,
  type JobProgressPayload,
  type NcaResponse,
  type PbpkResponse,
  type PdResponse,
  type PkpdCatalogResponse,
  type PopulationBodyView,
  type SteadyStateResponse,
  type SuperpositionResponse,
} from "@/services/pkpd/client";

/**
 * The PK/PD workspace — part11 §11.6's complete sequential workflow:
 *
 * Import Data → Inspect → Select Model → Configure Parameters →
 * Fit/Simulate → Diagnostics → Exposure/Response Visualization →
 * Export
 *
 * Every control below executes the REAL backend through the typed
 * service layer (no fabricated data, no disabled pretend buttons).
 * The Import lane parses a pasted concentration-time series (the
 * route's own `validate_concentration_series` remains the authority —
 * a bad series is its honest 422, surfaced verbatim); the Inspect
 * stage runs the REAL non-compartmental analysis (`nca_summary` +
 * both trapezoid variants + the `fit_terminal_phase` regression) and,
 * over a 0…τ profile, the interval metrics; the Select Model stage is
 * the backend's own `GET /pkpd/catalog` inventory (four groups, each
 * row naming the backend functions it wraps — never a client-side
 * guess); the parameter stage edits the models' own parameters (the
 * bateman/superposition ka·ke pair, the MinimalPBPK physiology set,
 * the sigmoid Emax scalars, the population's theta/omega/seed);
 * Fit/Simulate executes the selected model — the population lane
 * defaults to job mode (the COMPLETED job's REST result is
 * authoritative; the WS `job.progress` envelopes drive the per-batch
 * progress bar only, and cancelling is native — the job lands
 * CANCELLED, never FAILED); Diagnostics renders the terminal-phase
 * regression quality (r², the span verdict — `span_low` is the
 * backend's own warning, rendered verbatim) and the steady-state
 * interval metrics; the Exposure/Response Visualization stage draws
 * the inline curve ONLY from points that actually arrived and
 * registers a REAL figure artifact through the §17 four-format
 * export stack; the Export stage rides the rows-based table lane
 * with the backend's own 5000-row cap mirrored client-side.
 */

/** The backend's own table-export row cap (visualization.py's `_MAX_ROWS`). */
const TABLE_EXPORT_MAX_ROWS = 5000;

/** The six `nca_summary` metric rows, in the backend's own order. */
const NCA_SUMMARY_KEYS: readonly (keyof NcaResponse["summary"])[] = [
  "AUC",
  "AUMC",
  "MRT",
  "Cmax",
  "Tmax",
  "t_half_terminal",
];

/** The steady-state interval metric rows, in the backend's own order. */
const STEADY_STATE_KEYS: readonly (keyof SteadyStateResponse["metrics"])[] = [
  "AUC_tau",
  "Cavg",
  "Cmax_ss",
  "Cmin_ss",
  "Ctrough",
  "fluctuation",
  "swing",
  "CL_ss",
  "R_accum",
];

type ModelKey =
  | "iv-bolus"
  | "bateman"
  | "superposition"
  | "pbpk"
  | "pd"
  | "population";

type ExportSource = "curve" | "pbpk-rows" | "pd-table" | "nca-metrics";

type ExportFormat = "csv" | "tsv" | "xlsx" | "md" | "json" | "html";

/**
 * A realistic single-dose IV profile (eleven points, a distribution
 * phase into a log-linear terminal tail) — the paste lane's own
 * editable sample, in the `time conc` format the parser reads.
 */
const SAMPLE_PROFILE = [
  "time conc",
  "0 8.0",
  "0.25 6.9",
  "0.5 5.8",
  "1 4.4",
  "2 3.1",
  "4 1.9",
  "6 1.2",
  "8 0.75",
  "12 0.30",
  "16 0.12",
  "24 0.02",
].join("\n");

/**
 * A realistic steady-state 0…τ profile (one dosing interval at τ = 12:
 * the trough, the rise to Cmax, the decay back to trough) — the
 * interval-metrics lane's own editable sample.
 */
const SAMPLE_SS_PROFILE = [
  "time conc",
  "0 0.80",
  "1 2.90",
  "2 4.10",
  "3 4.60",
  "4 4.45",
  "6 3.40",
  "8 2.20",
  "10 1.30",
  "12 0.80",
].join("\n");

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
  const terminal =
    view.state === "COMPLETED" ||
    view.state === "CANCELLED" ||
    view.state === "FAILED";
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

/** A numbered workflow stage header (§11.6's fixed order). */
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
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

/** The union of key names across a rows list (column discovery). */
function unionRowKeys(rows: readonly Record<string, unknown>[]): string[] {
  const keys: string[] = [];
  const seen = new Set<string>();
  for (const row of rows) {
    for (const key of Object.keys(row)) {
      if (!seen.has(key)) {
        seen.add(key);
        keys.push(key);
      }
    }
  }
  return keys;
}

/** One pasted series row (`time conc`). */
interface SeriesPoint {
  readonly time: number;
  readonly conc: number;
}

/**
 * Parse a pasted concentration-time series — one `time conc` pair per
 * line (whitespace or comma separated; a `time conc` header line is
 * skipped). Every malformed row is this parser's own labeled error,
 * never a silent drop; the backend's own validator remains the
 * scientific authority (negatives, ordering, duplicates).
 */
function parseSeries(text: string, label: string): SeriesPoint[] {
  const rows: SeriesPoint[] = [];
  const lines = text
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line !== "");
  lines.forEach((line, i) => {
    const cells = line.split(/[\s,]+/).filter((cell) => cell !== "");
    if (cells[0] === "time") {
      return; // the header row
    }
    if (cells.length !== 2) {
      throw new Error(
        `${label} row ${i + 1}: expected exactly 2 cells (time conc), got ${cells.length}`,
      );
    }
    const time = Number(cells[0]);
    const conc = Number(cells[1]);
    if (!Number.isFinite(time) || !Number.isFinite(conc)) {
      throw new Error(
        `${label} row ${i + 1}: "${cells[0]} ${cells[1]}" is not a pair of finite numbers`,
      );
    }
    rows.push({ time, conc });
  });
  if (rows.length < 2) {
    throw new Error(
      `${label}: ${rows.length} points parsed, at least 2 required`,
    );
  }
  return rows;
}

/**
 * The inline curve — an SVG polyline drawn ONLY from the points that
 * actually arrived (no interpolation the data does not own, no
 * fabricated axes: the domain/range labels state the real min/max).
 */
function CurveFigure({
  points,
  xLabel,
  yLabel,
  height = 220,
}: {
  points: readonly { x: number; y: number }[];
  xLabel: string;
  yLabel: string;
  height?: number;
}) {
  if (points.length < 2) {
    return (
      <p className="text-xs text-[var(--color-muted-foreground)]">
        the curve draws only from points that actually arrived — two
        finite points is the honest minimum
      </p>
    );
  }
  const width = 640;
  const pad = 34;
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const toX = (x: number) => pad + ((x - minX) / spanX) * (width - pad * 2);
  const toY = (y: number) => height - pad - ((y - minY) / spanY) * (height - pad * 2);
  const path = points
    .map((p, i) => `${i === 0 ? "M" : "L"}${toX(p.x).toFixed(2)},${toY(p.y).toFixed(2)}`)
    .join(" ");
  return (
    <figure className="mt-2">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)]"
        role="img"
        aria-label={`${yLabel} versus ${xLabel} curve over ${points.length} real points`}
      >
        <path
          d={path}
          fill="none"
          stroke="var(--color-primary)"
          strokeWidth="2"
          strokeLinejoin="round"
        />
        {points.map((p, i) => (
          <circle key={i} cx={toX(p.x)} cy={toY(p.y)} r="2.5" fill="var(--color-primary)" />
        ))}
      </svg>
      <figcaption className="mt-1 text-[11px] text-[var(--color-muted-foreground)]">
        {points.length} real points · {xLabel} {formatValue(minX)}–{formatValue(maxX)} ·{" "}
        {yLabel} {formatValue(minY)}–{formatValue(maxY)}
      </figcaption>
    </figure>
  );
}

/** A small labeled scalar input row (the parameter editors' own cell). */
function ParamInput({
  label,
  value,
  onChange,
  step = "any",
  min,
  max,
  hint,
}: {
  label: string;
  value: string;
  onChange: (next: string) => void;
  step?: string;
  min?: number;
  max?: number;
  hint?: string;
}) {
  return (
    <label className="flex flex-col gap-1">
      {label}
      <input
        type="number"
        step={step}
        {...(min === undefined ? {} : { min })}
        {...(max === undefined ? {} : { max })}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
      />
      {hint ? (
        <span className="text-[10px] text-[var(--color-muted-foreground)]">{hint}</span>
      ) : null}
    </label>
  );
}

/**
 * The PK/PD workspace page — one component per established pattern,
 * the eight §11.6 stages in order.
 */
export function PKPDPage() {
  // stage 1: the imported series
  const [seriesText, setSeriesText] = useState("");
  const [series, setSeries] = useState<SeriesPoint[] | null>(null);
  const [seriesError, setSeriesError] = useState<string | null>(null);

  // stage 2: NCA + steady state
  const [ncaBusy, setNcaBusy] = useState(false);
  const [ncaError, setNcaError] = useState<string | null>(null);
  const [ncaResult, setNcaResult] = useState<NcaResponse | null>(null);
  const [ssTau, setSsTau] = useState("12");
  const [ssDose, setSsDose] = useState("");
  const [ssLambdaZ, setSsLambdaZ] = useState("");
  const [ssBusy, setSsBusy] = useState(false);
  const [ssError, setSsError] = useState<string | null>(null);
  const [ssResult, setSsResult] = useState<SteadyStateResponse | null>(null);

  // stage 3: the backend's own model catalog
  const [catalog, setCatalog] = useState<PkpdCatalogResponse | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [modelKey, setModelKey] = useState<ModelKey>("bateman");

  // stage 4: the model parameters (the route's own defaults)
  const [ivC0, setIvC0] = useState("10");
  const [ivKe, setIvKe] = useState("0.35");
  const [gridStart, setGridStart] = useState("0");
  const [gridStop, setGridStop] = useState("24");
  const [gridStep, setGridStep] = useState("0.5");
  const [batDose, setBatDose] = useState("5");
  const [batKa, setBatKa] = useState("1.5");
  const [batKe, setBatKe] = useState("0.25");
  const [supTau, setSupTau] = useState("6");
  const [supDoses, setSupDoses] = useState("8");
  const [pbRoute, setPbRoute] = useState<"iv" | "oral">("iv");
  const [pbDose, setPbDose] = useState("100");
  const [pbKa, setPbKa] = useState("1");
  const [pbF, setPbF] = useState("0.85");
  const [pbBw, setPbBw] = useState("70");
  const [pbClink, setPbClink] = useState("50");
  const [pbFu, setPbFu] = useState("0.10");
  const [pbBp, setPbBp] = useState("1.20");
  const [pbKpLung, setPbKpLung] = useState("2");
  const [pbKpLiver, setPbKpLiver] = useState("5");
  const [pbKpKidney, setPbKpKidney] = useState("3");
  const [pbKpRest, setPbKpRest] = useState("1.5");
  const [pdGrid, setPdGrid] = useState("0.01 0.1 0.3 1 3 10 30 100");
  const [pdE0, setPdE0] = useState("0");
  const [pdEmax, setPdEmax] = useState("100");
  const [pdEc50, setPdEc50] = useState("10");
  const [pdHill, setPdHill] = useState("1");
  const [popThetaDose, setPopThetaDose] = useState("5");
  const [popThetaKa, setPopThetaKa] = useState("1.5");
  const [popThetaKe, setPopThetaKe] = useState("0.25");
  const [popOmega, setPopOmega] = useState("0.09");
  const [popSubjects, setPopSubjects] = useState("120");
  const [popSeed, setPopSeed] = useState("42");
  const [popSigmaAdd, setPopSigmaAdd] = useState("0");
  const [popSigmaProp, setPopSigmaProp] = useState("0");
  const [popProfiles, setPopProfiles] = useState("3");
  const [popMode, setPopMode] = useState<"job" | "sync">("job");

  // stage 5: the model results (one lane at a time, honestly typed)
  const [simBusy, setSimBusy] = useState(false);
  const [simError, setSimError] = useState<string | null>(null);
  const [ivResult, setIvResult] = useState<IvBolusResponse | null>(null);
  const [batResult, setBatResult] = useState<BatemanResponse | null>(null);
  const [supResult, setSupResult] = useState<SuperpositionResponse | null>(null);
  const [pbResult, setPbResult] = useState<PbpkResponse | null>(null);
  const [pdResult, setPdResult] = useState<PdResponse | null>(null);
  const [popJobId, setPopJobId] = useState<string | null>(null);
  const [popBody, setPopBody] = useState<PopulationBodyView | null>(null);
  const [popProgress, setPopProgress] = useState<readonly JobProgressPayload[]>([]);
  const [popError, setPopError] = useState<string | null>(null);

  // stage 7: the figure artifacts
  const [figureError, setFigureError] = useState<string | null>(null);
  const [exposureArtifact, setExposureArtifact] = useState<ArtifactView | null>(null);
  const [responseArtifact, setResponseArtifact] = useState<ArtifactView | null>(null);
  const [exposureDigest, setExposureDigest] = useState<string | null>(null);
  const [responseDigest, setResponseDigest] = useState<string | null>(null);

  // stage 8: the export lane
  const [exportSource, setExportSource] = useState<ExportSource>("curve");
  const [exportFormat, setExportFormat] = useState<ExportFormat>("csv");
  const [exportArtifact, setExportArtifact] = useState<ArtifactView | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportDigest, setExportDigest] = useState<string | null>(null);

  // stage 1: the paste lane
  const handleImport = useCallback(() => {
    setSeriesError(null);
    setNcaResult(null);
    setSsResult(null);
    try {
      const parsed = parseSeries(seriesText, "the imported series");
      setSeries(parsed);
    } catch (error) {
      setSeries(null);
      setSeriesError(describeError(error));
    }
  }, [seriesText]);

  const loadSample = useCallback(() => {
    setSeriesText(SAMPLE_PROFILE);
    setSeries(null);
    setSeriesError(null);
    setNcaResult(null);
    setSsResult(null);
  }, []);

  const loadSsSample = useCallback(() => {
    setSeriesText(SAMPLE_SS_PROFILE);
    setSeries(null);
    setSeriesError(null);
    setNcaResult(null);
    setSsResult(null);
  }, []);

  // stage 2: NCA over the imported series
  const handleNca = useCallback(async () => {
    if (series === null) {
      setNcaError("import a series first (stage one)");
      return;
    }
    setNcaBusy(true);
    setNcaError(null);
    setNcaResult(null);
    try {
      const times = series.map((p) => p.time);
      const conc = series.map((p) => p.conc);
      assertNcaSeriesBounds("nca", times, conc);
      const result = await pkpdApi.nca({ times, conc });
      setNcaResult(result);
    } catch (error) {
      setNcaError(describeError(error));
    } finally {
      setNcaBusy(false);
    }
  }, [series]);

  const handleSteadyState = useCallback(async () => {
    if (series === null) {
      setSsError("import a steady-state 0…τ profile first (stage one)");
      return;
    }
    setSsBusy(true);
    setSsError(null);
    setSsResult(null);
    try {
      const times = series.map((p) => p.time);
      const conc = series.map((p) => p.conc);
      assertNcaSeriesBounds("steady-state", times, conc);
      const tau = Number(ssTau);
      if (!Number.isFinite(tau) || tau <= 0) {
        throw new Error("tau must be a finite positive dosing interval");
      }
      const dose = ssDose.trim() === "" ? null : Number(ssDose);
      if (dose !== null && !Number.isFinite(dose)) {
        throw new Error("dose is not a finite number");
      }
      const lambdaZ = ssLambdaZ.trim() === "" ? null : Number(ssLambdaZ);
      if (lambdaZ !== null && !Number.isFinite(lambdaZ)) {
        throw new Error("lambda_z is not a finite number");
      }
      const result = await pkpdApi.steadyState({
        times,
        conc,
        tau,
        dose,
        lambdaZ: lambdaZ,
      });
      setSsResult(result);
    } catch (error) {
      setSsError(describeError(error));
    } finally {
      setSsBusy(false);
    }
  }, [series, ssTau, ssDose, ssLambdaZ]);

  // stage 3: the backend's own catalog (fetched once, honestly)
  useEffect(() => {
    let cancelled = false;
    pkpdApi
      .catalog()
      .then((response) => {
        if (!cancelled) {
          setCatalog(response);
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

  // the time grid the simulation lanes share (honest linear grid)
  const grid = useMemo(() => {
    const start = Number(gridStart);
    const stop = Number(gridStop);
    const step = Number(gridStep);
    if (![start, stop, step].every((v) => Number.isFinite(v))) {
      return null;
    }
    if (stop <= start || step <= 0) {
      return null;
    }
    const points: number[] = [];
    for (let t = start; t <= stop + 1e-9; t += step) {
      points.push(Number(t.toFixed(6)));
      if (points.length > 10000) {
        break; // the route's own cap — the pre-flight names it below
      }
    }
    return points;
  }, [gridStart, gridStop, gridStep]);

  // stage 5: the selected model's own lane
  const clearSimResults = useCallback(() => {
    setIvResult(null);
    setBatResult(null);
    setSupResult(null);
    setPbResult(null);
    setPdResult(null);
    setPopJobId(null);
    setPopBody(null);
    setPopProgress([]);
    setPopError(null);
    setSimError(null);
    setExposureArtifact(null);
    setResponseArtifact(null);
    setExposureDigest(null);
    setResponseDigest(null);
  }, []);

  const handleSimulate = useCallback(async () => {
    clearSimResults();
    if (grid === null) {
      setSimError(
        "the time grid is not usable — start < stop and step > 0, all finite",
      );
      return;
    }
    if (modelKey === "population") {
      // the population lane's own submit (job or sync)
      setSimBusy(true);
      try {
        const theta = [
          Number(popThetaDose),
          Number(popThetaKa),
          Number(popThetaKe),
        ];
        if (theta.some((v) => !Number.isFinite(v))) {
          throw new Error("theta (dose, ka, ke) must be three finite numbers");
        }
        const omega = Number(popOmega);
        if (!Number.isFinite(omega)) {
          throw new Error("omega_diag must be a finite number");
        }
        const subjects = Number(popSubjects);
        const seed = Number(popSeed);
        const profiles = Number(popProfiles);
        if (
          !Number.isFinite(subjects) || !Number.isFinite(seed) ||
          !Number.isFinite(profiles)
        ) {
          throw new Error("n_subjects/seed/n_profiles must be finite");
        }
        assertPopulationBounds("population", theta, [omega], grid);
        const response = await pkpdApi.population({
          theta,
          omegaDiag: [omega],
          times: grid,
          nSubjects: subjects,
          seed,
          sigmaAdd: Number(popSigmaAdd),
          sigmaProp: Number(popSigmaProp),
          nProfiles: profiles,
          mode: popMode,
        });
        if (response.mode === "sync") {
          setPopBody(response);
        } else {
          setPopJobId(response.job.job_id);
        }
      } catch (error) {
        setPopError(describeError(error));
      } finally {
        setSimBusy(false);
      }
      return;
    }
    setSimBusy(true);
    try {
      assertTimesBounds(modelKey, grid);
      if (modelKey === "iv-bolus") {
        const c0 = Number(ivC0);
        const ke = Number(ivKe);
        if (!Number.isFinite(c0) || !Number.isFinite(ke)) {
          throw new Error("c0 and ke must be finite numbers");
        }
        setIvResult(await pkpdApi.ivBolus({ c0, ke, times: grid }));
      } else if (modelKey === "bateman") {
        const dose = Number(batDose);
        const ka = Number(batKa);
        const ke = Number(batKe);
        if (!Number.isFinite(dose) || !Number.isFinite(ka) || !Number.isFinite(ke)) {
          throw new Error("dose_per_volume, ka and ke must be finite numbers");
        }
        setBatResult(
          await pkpdApi.bateman({ dosePerVolume: dose, ka, ke, times: grid }),
        );
      } else if (modelKey === "superposition") {
        const dose = Number(batDose);
        const ka = Number(batKa);
        const ke = Number(batKe);
        const tau = Number(supTau);
        const doses = Number(supDoses);
        if (
          !Number.isFinite(dose) || !Number.isFinite(ka) || !Number.isFinite(ke) ||
          !Number.isFinite(tau) || !Number.isFinite(doses)
        ) {
          throw new Error("the superposition scalars must be finite numbers");
        }
        setSupResult(
          await pkpdApi.superposition({
            dosePerVolume: dose,
            ka,
            ke,
            times: grid,
            tau,
            nDoses: doses,
          }),
        );
      } else if (modelKey === "pbpk") {
        const dose = Number(pbDose);
        if (!Number.isFinite(dose)) {
          throw new Error("dose_mg must be a finite number");
        }
        setPbResult(
          await pkpdApi.pbpk({
            times: grid,
            doseMg: dose,
            route: pbRoute,
            ka: Number(pbKa),
            f: Number(pbF),
            bw: Number(pbBw),
            clint: Number(pbClink),
            fu: Number(pbFu),
            bp: Number(pbBp),
            kpLung: Number(pbKpLung),
            kpLiver: Number(pbKpLiver),
            kpKidney: Number(pbKpKidney),
            kpRest: Number(pbKpRest),
          }),
        );
      } else if (modelKey === "pd") {
        const concentrations = pdGrid
          .split(/[\s,]+/)
          .filter((cell) => cell !== "")
          .map(Number);
        if (concentrations.some((v) => !Number.isFinite(v))) {
          throw new Error("the PD concentration grid must be finite numbers");
        }
        if (concentrations.length < 1 || concentrations.length > 2000) {
          throw new Error(
            `the PD grid carries ${concentrations.length} points; the route's bounds are 1…2000`,
          );
        }
        setPdResult(
          await pkpdApi.pd({
            concentrations,
            e0: Number(pdE0),
            emax: Number(pdEmax),
            ec50: Number(pdEc50),
            hill: Number(pdHill),
          }),
        );
      }
    } catch (error) {
      setSimError(describeError(error));
    } finally {
      setSimBusy(false);
    }
  }, [
    clearSimResults,
    grid,
    modelKey,
    ivC0,
    ivKe,
    batDose,
    batKa,
    batKe,
    supTau,
    supDoses,
    pbRoute,
    pbDose,
    pbKa,
    pbF,
    pbBw,
    pbClink,
    pbFu,
    pbBp,
    pbKpLung,
    pbKpLiver,
    pbKpKidney,
    pbKpRest,
    pdGrid,
    pdE0,
    pdEmax,
    pdEc50,
    pdHill,
    popThetaDose,
    popThetaKa,
    popThetaKe,
    popOmega,
    popSubjects,
    popSeed,
    popSigmaAdd,
    popSigmaProp,
    popProfiles,
    popMode,
  ]);

  // the population job's authoritative REST relay (§13)
  const handlePopulationResult = useCallback((result: unknown) => {
    try {
      setPopBody(parsePopulationBody(result, "population job result"));
    } catch (error) {
      setPopError(describeError(error));
    }
  }, []);

  // stage 5's live telemetry: the population job's per-batch
  // job.progress envelopes (progress-only — the body comes from the
  // REST relay)
  useEffect(() => {
    if (popJobId === null) {
      return;
    }
    const stream = getEventStream();
    const unsubscribe = stream.subscribe((envelope) => {
      if (envelope.job_id !== popJobId) {
        return;
      }
      if (envelope.event_type === "job.progress") {
        try {
          const payload = parseJobProgressPayload(
            envelope.payload,
            "job.progress",
          );
          setPopProgress((prev) => [...prev, payload]);
        } catch (error) {
          setPopError(describeError(error));
        }
      }
    });
    return unsubscribe;
  }, [popJobId]);

  // the digest-verified download lane (§15: the token-header raw lane)
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

  // stage 7: the exposure figure — the curve the backend computed
  const activeCurve = useMemo((): { x: number; y: number }[] | null => {
    if (ivResult !== null) {
      return ivResult.concentration.map((c, i) => ({
        x: grid?.[i] ?? i,
        y: c,
      }));
    }
    if (batResult !== null) {
      return batResult.concentration.map((c, i) => ({
        x: grid?.[i] ?? i,
        y: c,
      }));
    }
    if (supResult !== null) {
      return supResult.concentration.map((c, i) => ({
        x: grid?.[i] ?? i,
        y: c,
      }));
    }
    if (pbResult !== null) {
      return pbResult.rows.map((row) => ({ x: row.time, y: row["C_plasma"] }));
    }
    if (popBody !== null) {
      return popBody.summary.mean.map((c, i) => ({
        x: grid?.[i] ?? i,
        y: c,
      }));
    }
    return null;
  }, [ivResult, batResult, supResult, pbResult, popBody, grid]);

  const handleExposureFigure = useCallback(async () => {
    setFigureError(null);
    setExposureArtifact(null);
    setExposureDigest(null);
    try {
      if (activeCurve === null || activeCurve.length < 2) {
        throw new Error("run a simulation first — the figure draws only from real points");
      }
      const artifact = await pkpdApi.exposureFigure({
        times: activeCurve.map((p) => p.x),
        conc: activeCurve.map((p) => p.y),
        title: "PK/PD — concentration-time exposure profile",
        filename: "pkpd-exposure-profile",
        format: "PNG",
      });
      setExposureArtifact(artifact);
      setExposureDigest(await verifiedDownload(artifact));
    } catch (error) {
      setFigureError(describeError(error));
    }
  }, [activeCurve, verifiedDownload]);

  const handleResponseFigure = useCallback(async () => {
    setFigureError(null);
    setResponseArtifact(null);
    setResponseDigest(null);
    try {
      if (pdResult === null) {
        throw new Error("run the sigmoid Emax model first — the response curve is its own verdict");
      }
      const artifact = await pkpdApi.responseFigure({
        concentrations: pdResult.concentration,
        effect: pdResult.effect,
        title: "PK/PD — sigmoid Emax response curve",
        filename: "pkpd-response-curve",
        format: "PNG",
      });
      setResponseArtifact(artifact);
      setResponseDigest(await verifiedDownload(artifact));
    } catch (error) {
      setFigureError(describeError(error));
    }
  }, [pdResult, verifiedDownload]);

  // stage 8: the rows the export carries (per source, honestly)
  const exportRows = useMemo((): Record<string, unknown>[] => {
    if (exportSource === "curve") {
      if (activeCurve === null) {
        return [];
      }
      return activeCurve.map((p) => ({ time: p.x, concentration: p.y }));
    }
    if (exportSource === "pbpk-rows") {
      return pbResult === null
        ? []
        : pbResult.rows.map((row) => ({ ...row }));
    }
    if (exportSource === "pd-table") {
      return pdResult === null
        ? []
        : pdResult.concentration.map((c, i) => ({
            concentration: c,
            effect: pdResult.effect[i] ?? null,
          }));
    }
    // the NCA metrics + the terminal-phase diagnostics, flattened
    if (ncaResult === null) {
      return [];
    }
    const rows: Record<string, unknown>[] = NCA_SUMMARY_KEYS.map((key) => ({
      metric: key,
      value: ncaResult.summary[key],
    }));
    rows.push(
      { metric: "auc_trapezoid_linear", value: ncaResult.auc_trapezoid_linear },
      { metric: "auc_trapezoid_log", value: ncaResult.auc_trapezoid_log },
    );
    for (const [key, value] of Object.entries(ncaResult.terminal)) {
      rows.push({ metric: `terminal_${key}`, value });
    }
    return rows;
  }, [exportSource, activeCurve, pbResult, pdResult, ncaResult]);

  const handleExport = useCallback(async () => {
    setExportError(null);
    setExportArtifact(null);
    setExportDigest(null);
    try {
      if (exportRows.length === 0) {
        throw new Error("the export carries no rows — run the lane first");
      }
      if (exportRows.length > TABLE_EXPORT_MAX_ROWS) {
        throw new Error(
          `the export carries ${exportRows.length} rows; the backend's own cap is ${TABLE_EXPORT_MAX_ROWS} (narrow the time grid)`,
        );
      }
      const artifact = await pkpdApi.exportTable({
        rows: exportRows,
        format: exportFormat,
        filename: `pkpd-${exportSource}`,
      });
      setExportArtifact(artifact);
      setExportDigest(await verifiedDownload(artifact));
    } catch (error) {
      setExportError(describeError(error));
    }
  }, [exportRows, exportFormat, exportSource, verifiedDownload]);

  // -- the render ------------------------------------------------------

  const seriesView = useMemo(
    () =>
      series === null
        ? null
        : series.map((p) => ({ time: p.time, conc: p.conc })),
    [series],
  );

  return (
    <div className="mx-auto max-w-5xl px-4 pb-16">
      <header className="pt-8">
        <h1 className="text-2xl font-semibold text-[var(--color-foreground)]">
          PK/PD
        </h1>
        <p className="mt-1 text-sm text-[var(--color-muted-foreground)]">
          Import Data → Inspect → Select Model → Configure Parameters →
          Fit/Simulate → Diagnostics → Exposure/Response Visualization →
          Export — every lane the REAL pharmacometrics backend.
        </p>
      </header>

      {/* stage 1: Import Data */}
      <StageHeader
        index={1}
        title="Import Data"
        hint="the pasted concentration-time series (time conc per line)"
        icon={<Upload className="h-4 w-4 text-[var(--color-muted-foreground)]" />}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Concentration-time series</CardTitle>
            <CardDescription>
              POST /api/v1/pkpd/nca and /steady-state read this series; the
              backend&apos;s own validator stays the authority — a bad series
              is its honest 422, surfaced verbatim.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <textarea
              value={seriesText}
              onChange={(event) => setSeriesText(event.target.value)}
              rows={7}
              spellCheck={false}
              placeholder={"time conc\n0 8.0\n0.25 6.9\n…"}
              className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-2 font-mono text-xs"
            />
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <Button onClick={handleImport}>Import series</Button>
              <Button variant="outline" onClick={loadSample}>
                Load single-dose sample
              </Button>
              <Button variant="outline" onClick={loadSsSample}>
                Load steady-state 0…τ sample
              </Button>
            </div>
            {seriesError ? <div className="mt-3"><ErrorBanner message={seriesError} /></div> : null}
            {seriesView !== null ? (
              <div className="mt-3">
                <p className="text-xs text-[var(--color-muted-foreground)]">
                  {seriesView.length} points imported · t {formatValue(seriesView[0]?.time)}–
                  {formatValue(seriesView[seriesView.length - 1]?.time)} · C{" "}
                  {formatValue(Math.min(...seriesView.map((p) => p.conc)))}–
                  {formatValue(Math.max(...seriesView.map((p) => p.conc)))}
                </p>
                <div className="mt-2 max-h-48 overflow-auto rounded-md border border-[var(--color-border)]">
                  <table className="w-full border-collapse text-[11px]">
                    <thead className="sticky top-0 bg-[var(--color-muted)]">
                      <tr>
                        <th className="px-2 py-1 text-left font-medium">time</th>
                        <th className="px-2 py-1 text-left font-medium">conc</th>
                      </tr>
                    </thead>
                    <tbody>
                      {seriesView.map((p, i) => (
                        <tr key={i} className="odd:bg-[var(--color-muted)]/40">
                          <td className="px-2 py-0.5">{formatValue(p.time)}</td>
                          <td className="px-2 py-0.5">{formatValue(p.conc)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      {/* stage 2: Inspect */}
      <StageHeader
        index={2}
        title="Inspect"
        hint="the REAL non-compartmental analysis over the imported series"
        icon={<Sigma className="h-4 w-4 text-[var(--color-muted-foreground)]" />}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Non-compartmental analysis</CardTitle>
            <CardDescription>
              POST /api/v1/pkpd/nca — nca_summary + both auc_trapezoid
              variants + the fit_terminal_phase regression (the backend&apos;s
              own terminal-phase selection, honest r² and span verdict).
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-center gap-2">
              <Button onClick={() => void handleNca()} disabled={ncaBusy || series === null}>
                {ncaBusy ? "running…" : "Run NCA"}
              </Button>
              <span className="text-xs text-[var(--color-muted-foreground)]">
                {series === null
                  ? "import a series in stage one first"
                  : `${series.length} points · the series the backend sees`}
              </span>
            </div>
            {ncaError ? <div className="mt-3"><ErrorBanner message={ncaError} /></div> : null}
            {ncaResult !== null ? (
              <div className="mt-3 grid gap-3 md:grid-cols-2">
                <div className="rounded-md border border-[var(--color-border)] p-3">
                  <p className="text-xs font-semibold text-[var(--color-foreground)]">
                    nca_summary
                  </p>
                  <table className="mt-2 w-full text-[11px]">
                    <tbody>
                      {NCA_SUMMARY_KEYS.map((key) => (
                        <tr key={key}>
                          <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">{key}</td>
                          <td className="py-0.5 text-right">{formatValue(ncaResult.summary[key])}</td>
                        </tr>
                      ))}
                      <tr>
                        <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">
                          auc_trapezoid_linear
                        </td>
                        <td className="py-0.5 text-right">
                          {formatValue(ncaResult.auc_trapezoid_linear)}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">
                          auc_trapezoid_log
                        </td>
                        <td className="py-0.5 text-right">
                          {ncaResult.auc_trapezoid_log === null
                            ? "null (a non-positive conc point)"
                            : formatValue(ncaResult.auc_trapezoid_log)}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <div className="rounded-md border border-[var(--color-border)] p-3">
                  <p className="text-xs font-semibold text-[var(--color-foreground)]">
                    terminal phase ({ncaResult.terminal.method})
                  </p>
                  <table className="mt-2 w-full text-[11px]">
                    <tbody>
                      <tr>
                        <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">points</td>
                        <td className="py-0.5 text-right">
                          [{ncaResult.terminal.start_index}…{ncaResult.terminal.end_index}] ·{" "}
                          {ncaResult.terminal.n_points}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">lambda_z</td>
                        <td className="py-0.5 text-right">{formatValue(ncaResult.terminal.lambda_z)}</td>
                      </tr>
                      <tr>
                        <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">half_life</td>
                        <td className="py-0.5 text-right">{formatValue(ncaResult.terminal.half_life)}</td>
                      </tr>
                      <tr>
                        <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">r2 / r2_adj</td>
                        <td className="py-0.5 text-right">
                          {formatValue(ncaResult.terminal.r2)} / {formatValue(ncaResult.terminal.r2_adj)}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">span_half_lives</td>
                        <td className="py-0.5 text-right">
                          {formatValue(ncaResult.terminal.span_half_lives)}
                          {ncaResult.terminal.span_low ? " — span_low (the backend's own warning)" : ""}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">clast_pred</td>
                        <td className="py-0.5 text-right">{formatValue(ncaResult.terminal.clast_pred)}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            ) : null}
          </CardContent>
        </Card>
        <Card className="mt-4">
          <CardHeader>
            <CardTitle className="text-base">Steady-state interval metrics</CardTitle>
            <CardDescription>
              POST /api/v1/pkpd/steady-state — one 0…τ interval&apos;s NCA
              (legitimate zeros are data here: troughs and not-yet-absorbed
              segments).
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-end gap-3">
              <ParamInput label="tau (h)" value={ssTau} onChange={setSsTau} min={0.01} />
              <ParamInput label="dose (optional)" value={ssDose} onChange={setSsDose} hint="null when blank" />
              <ParamInput label="lambda_z (optional)" value={ssLambdaZ} onChange={setSsLambdaZ} hint="null when blank" />
              <Button onClick={() => void handleSteadyState()} disabled={ssBusy || series === null}>
                {ssBusy ? "running…" : "Run steady-state"}
              </Button>
            </div>
            {ssError ? <div className="mt-3"><ErrorBanner message={ssError} /></div> : null}
            {ssResult !== null ? (
              <div className="mt-3 rounded-md border border-[var(--color-border)] p-3">
                <p className="text-xs font-semibold text-[var(--color-foreground)]">
                  interval metrics (τ = {formatValue(ssResult.tau)})
                </p>
                <table className="mt-2 w-full text-[11px]">
                  <tbody>
                    {STEADY_STATE_KEYS.map((key) => (
                      <tr key={key}>
                        <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">{key}</td>
                        <td className="py-0.5 text-right">{formatValue(ssResult.metrics[key])}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      {/* stage 3: Select Model */}
      <StageHeader
        index={3}
        title="Select Model"
        hint="the backend's own /pkpd/catalog inventory — never a client-side guess"
        icon={<Beaker className="h-4 w-4 text-[var(--color-muted-foreground)]" />}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Model catalog</CardTitle>
            <CardDescription>
              GET /api/v1/pkpd/catalog — four groups, each row naming the
              backend functions it wraps. The PK/PD models this toolkit
              actually supports, stated by the backend itself.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {catalogError ? <ErrorBanner message={catalogError} /> : null}
            {catalog === null && catalogError === null ? (
              <p className="text-sm text-[var(--color-muted-foreground)]">
                fetching the catalog…
              </p>
            ) : null}
            {catalog !== null ? (
              <div className="grid gap-2">
                {catalog.groups.map((group) => (
                  <div
                    key={group.group}
                    className="rounded-md border border-[var(--color-border)] p-3"
                  >
                    <p className="text-xs font-semibold text-[var(--color-foreground)]">
                      {group.group} — {group.backend}
                    </p>
                    <p className="mt-1 text-[11px] text-[var(--color-muted-foreground)]">
                      routes: {group.routes.join(" ")}
                    </p>
                  </div>
                ))}
              </div>
            ) : null}
            <div className="mt-4 flex flex-wrap gap-2">
              {(
                [
                  ["iv-bolus", "IV bolus (1-compartment)"],
                  ["bateman", "Oral 1-compartment (Bateman)"],
                  ["superposition", "Repeat-dose superposition"],
                  ["pbpk", "Minimal PBPK (5-compartment)"],
                  ["pd", "Pharmacodynamic (sigmoid Emax)"],
                  ["population", "Population simulation (IIV on ka)"],
                ] as readonly [ModelKey, string][]
              ).map(([key, label]) => (
                <label
                  key={key}
                  className="flex cursor-pointer items-center gap-2 rounded-md border border-[var(--color-border)] px-3 py-1.5 text-xs"
                >
                  <input
                    type="radio"
                    name="pkpd-model"
                    checked={modelKey === key}
                    onChange={() => {
                      setModelKey(key);
                      clearSimResults();
                    }}
                  />
                  {label}
                </label>
              ))}
            </div>
          </CardContent>
        </Card>
      </StageHeader>

      {/* stage 4: Configure Parameters */}
      <StageHeader
        index={4}
        title="Configure Parameters"
        hint="the selected model's own parameters (the route's own defaults)"
        icon={<Stethoscope className="h-4 w-4 text-[var(--color-muted-foreground)]" />}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">
              {modelKey === "iv-bolus"
                ? "IV bolus parameters"
                : modelKey === "bateman"
                  ? "Bateman parameters"
                  : modelKey === "superposition"
                    ? "Superposition parameters"
                    : modelKey === "pbpk"
                      ? "Minimal PBPK parameters"
                      : modelKey === "pd"
                        ? "Sigmoid Emax parameters"
                        : "Population simulation parameters"}
            </CardTitle>
            <CardDescription>
              The value bounds are the backend&apos;s own honest 422s — a
              scientific verdict, surfaced verbatim, never a client-side
              guess.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 sm:grid-cols-3">
              <ParamInput label="grid start (h)" value={gridStart} onChange={setGridStart} />
              <ParamInput label="grid stop (h)" value={gridStop} onChange={setGridStop} />
              <ParamInput label="grid step (h)" value={gridStep} onChange={setGridStep} />
            </div>
            {grid === null ? (
              <p className="mt-2 text-xs text-[var(--color-destructive)]">
                the time grid is not usable — start &lt; stop and step &gt; 0
              </p>
            ) : (
              <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                {grid.length} grid points (the shared simulation grid)
              </p>
            )}
            {modelKey === "iv-bolus" ? (
              <div className="mt-3 grid gap-3 sm:grid-cols-3">
                <ParamInput label="c0 (initial conc)" value={ivC0} onChange={setIvC0} min={0} />
                <ParamInput label="ke (1/h)" value={ivKe} onChange={setIvKe} min={0.001} max={10} />
              </div>
            ) : null}
            {modelKey === "bateman" || modelKey === "superposition" || modelKey === "population" ? (
              <div className="mt-3 grid gap-3 sm:grid-cols-3">
                {modelKey === "population" ? (
                  <ParamInput label="theta[0] dose_per_volume" value={popThetaDose} onChange={setPopThetaDose} min={0} />
                ) : (
                  <ParamInput label="dose per volume" value={batDose} onChange={setBatDose} min={0} />
                )}
                {modelKey === "population" ? (
                  <ParamInput label="theta[1] ka (1/h)" value={popThetaKa} onChange={setPopThetaKa} min={0.001} max={100} />
                ) : (
                  <ParamInput label="ka (1/h)" value={batKa} onChange={setBatKa} min={0.001} max={100} />
                )}
                {modelKey === "population" ? (
                  <ParamInput label="theta[2] ke (1/h)" value={popThetaKe} onChange={setPopThetaKe} min={0.001} max={100} />
                ) : (
                  <ParamInput label="ke (1/h)" value={batKe} onChange={setBatKe} min={0.001} max={100} />
                )}
              </div>
            ) : null}
            {modelKey === "superposition" ? (
              <div className="mt-3 grid gap-3 sm:grid-cols-3">
                <ParamInput label="tau (dosing interval)" value={supTau} onChange={setSupTau} min={0.01} />
                <ParamInput label="n_doses" value={supDoses} onChange={setSupDoses} min={1} max={1000} step="1" />
              </div>
            ) : null}
            {modelKey === "pbpk" ? (
              <div className="mt-3 grid gap-3 sm:grid-cols-3">
                <ParamInput label="dose_mg" value={pbDose} onChange={setPbDose} min={0.01} />
                <label className="flex flex-col gap-1">
                  route
                  <select
                    value={pbRoute}
                    onChange={(event) =>
                      setPbRoute(event.target.value === "oral" ? "oral" : "iv")
                    }
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                  >
                    <option value="iv">iv</option>
                    <option value="oral">oral</option>
                  </select>
                </label>
                <ParamInput label="ka (oral, 1/h)" value={pbKa} onChange={setPbKa} min={0.001} />
                <ParamInput label="f (bioavailability)" value={pbF} onChange={setPbF} min={0.01} max={1} />
                <ParamInput label="bw (kg)" value={pbBw} onChange={setPbBw} min={0.1} />
                <ParamInput label="CLint (L/h)" value={pbClink} onChange={setPbClink} min={0.01} />
                <ParamInput label="fu (unbound fraction)" value={pbFu} onChange={setPbFu} min={0.01} max={1} />
                <ParamInput label="bp (blood:plasma)" value={pbBp} onChange={setPbBp} min={0.01} />
                <ParamInput label="Kp lung" value={pbKpLung} onChange={setPbKpLung} min={0.01} />
                <ParamInput label="Kp liver" value={pbKpLiver} onChange={setPbKpLiver} min={0.01} />
                <ParamInput label="Kp kidney" value={pbKpKidney} onChange={setPbKpKidney} min={0.01} />
                <ParamInput label="Kp rest" value={pbKpRest} onChange={setPbKpRest} min={0.01} />
              </div>
            ) : null}
            {modelKey === "pd" ? (
              <div className="mt-3">
                <label className="flex flex-col gap-1">
                  concentration grid (whitespace separated)
                  <input
                    value={pdGrid}
                    onChange={(event) => setPdGrid(event.target.value)}
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 font-mono text-xs"
                  />
                </label>
                <div className="mt-3 grid gap-3 sm:grid-cols-4">
                  <ParamInput label="e0 (baseline)" value={pdE0} onChange={setPdE0} />
                  <ParamInput label="emax" value={pdEmax} onChange={setPdEmax} />
                  <ParamInput label="ec50" value={pdEc50} onChange={setPdEc50} min={0.01} />
                  <ParamInput label="hill" value={pdHill} onChange={setPdHill} min={0.01} max={20} />
                </div>
              </div>
            ) : null}
            {modelKey === "population" ? (
              <div className="mt-3">
                <div className="grid gap-3 sm:grid-cols-4">
                  <ParamInput label="omega_diag[0] (IIV var)" value={popOmega} onChange={setPopOmega} hint=">= 0 — the route's own 422 otherwise" />
                  <ParamInput label="n_subjects" value={popSubjects} onChange={setPopSubjects} min={10} max={20000} step="1" />
                  <ParamInput label="seed" value={popSeed} onChange={setPopSeed} min={0} step="1" />
                  <ParamInput label="n_profiles" value={popProfiles} onChange={setPopProfiles} min={0} max={10} step="1" />
                  <ParamInput label="sigma_add" value={popSigmaAdd} onChange={setPopSigmaAdd} min={0} />
                  <ParamInput label="sigma_prop" value={popSigmaProp} onChange={setPopSigmaProp} min={0} />
                </div>
                <div className="mt-3 flex items-center gap-2">
                  <span className="text-xs text-[var(--color-muted-foreground)]">mode:</span>
                  {(["job", "sync"] as const).map((mode) => (
                    <label key={mode} className="flex cursor-pointer items-center gap-1 text-xs">
                      <input
                        type="radio"
                        name="pop-mode"
                        checked={popMode === mode}
                        onChange={() => setPopMode(mode)}
                      />
                      {mode}
                    </label>
                  ))}
                  <span className="text-xs text-[var(--color-muted-foreground)]">
                    job is the default — the COMPLETED job&apos;s REST result
                    is authoritative
                  </span>
                </div>
              </div>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      {/* stage 5: Fit/Simulate */}
      <StageHeader
        index={5}
        title="Fit/Simulate"
        hint="the selected model executed on the REAL backend"
        icon={<Play className="h-4 w-4 text-[var(--color-muted-foreground)]" />}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Run the model</CardTitle>
            <CardDescription>
              {modelKey === "population"
                ? "POST /api/v1/pkpd/population — the seeded simulate_population with IIV on ka; job mode batches 50 subjects per progress envelope, cancelling is native (CANCELLED, never FAILED)."
                : modelKey === "pd"
                  ? "POST /api/v1/pkpd/pd — the sigmoid Emax effect_table (the response curve is the engine's own verdict)."
                  : `POST /api/v1/pkpd/${modelKey} — the analytical model curve over the configured grid.`}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-center gap-2">
              <Button onClick={() => void handleSimulate()} disabled={simBusy}>
                {simBusy ? "running…" : "Simulate"}
              </Button>
              <span className="text-xs text-[var(--color-muted-foreground)]">
                {modelKey === "population"
                  ? `${popSubjects} subjects · ${popMode} mode`
                  : `${grid?.length ?? 0} grid points`}
              </span>
            </div>
            {simError ? <div className="mt-3"><ErrorBanner message={simError} /></div> : null}
            {popError ? <div className="mt-3"><ErrorBanner message={popError} /></div> : null}

            {popJobId !== null ? (
              <div className="mt-3">
                <JobRunCard
                  jobId={popJobId}
                  resultLabel="the population body relay"
                  onResult={handlePopulationResult}
                />
                {popProgress.length > 0 ? (
                  <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                    {popProgress.length} job.progress envelopes · last:{" "}
                    {popProgress[popProgress.length - 1]?.progress}% ·{" "}
                    {popProgress[popProgress.length - 1]?.stage}
                  </p>
                ) : (
                  <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                    the backend&apos;s own progress callback drives one
                    job.progress envelope per 50-subject batch
                  </p>
                )}
              </div>
            ) : null}

            {ivResult !== null ? (
              <div className="mt-3">
                <p className="text-xs text-[var(--color-muted-foreground)]">
                  c0 {formatValue(ivResult.c0)} · ke {formatValue(ivResult.ke)} ·{" "}
                  {ivResult.n_points} points · POST /api/v1/pkpd/iv-bolus
                </p>
                <CurveFigure
                  points={ivResult.concentration.map((c, i) => ({ x: grid?.[i] ?? i, y: c }))}
                  xLabel="time"
                  yLabel="concentration"
                />
              </div>
            ) : null}
            {batResult !== null ? (
              <div className="mt-3">
                <p className="text-xs text-[var(--color-muted-foreground)]">
                  dose/V {formatValue(batResult.dose_per_volume)} · ka{" "}
                  {formatValue(batResult.ka)} · ke {formatValue(batResult.ke)} ·{" "}
                  {batResult.n_points} points · POST /api/v1/pkpd/bateman
                </p>
                <CurveFigure
                  points={batResult.concentration.map((c, i) => ({ x: grid?.[i] ?? i, y: c }))}
                  xLabel="time"
                  yLabel="concentration"
                />
              </div>
            ) : null}
            {supResult !== null ? (
              <div className="mt-3">
                <p className="text-xs text-[var(--color-muted-foreground)]">
                  τ {formatValue(supResult.tau)} · {supResult.n_doses} doses ·{" "}
                  {supResult.n_points} points · POST /api/v1/pkpd/superposition
                </p>
                <CurveFigure
                  points={supResult.concentration.map((c, i) => ({ x: grid?.[i] ?? i, y: c }))}
                  xLabel="time"
                  yLabel="concentration"
                />
              </div>
            ) : null}
            {pbResult !== null ? (
              <div className="mt-3">
                <p className="text-xs text-[var(--color-muted-foreground)]">
                  {pbResult.route} · {formatValue(pbResult.dose_mg)} mg ·{" "}
                  {pbResult.compartments.join(" ")} + C_plasma ·{" "}
                  {pbResult.n_points} rows · POST /api/v1/pkpd/pbpk
                </p>
                <CurveFigure
                  points={pbResult.rows.map((row) => ({ x: row.time, y: row["C_plasma"] }))}
                  xLabel="time"
                  yLabel="C_plasma"
                />
                <div className="mt-2 max-h-64 overflow-auto rounded-md border border-[var(--color-border)]">
                  <table className="w-full border-collapse text-[11px]">
                    <thead className="sticky top-0 bg-[var(--color-muted)]">
                      <tr>
                        <th className="px-2 py-1 text-left font-medium">time</th>
                        {pbResult.compartments.map((c) => (
                          <th key={c} className="px-2 py-1 text-left font-medium">{c}</th>
                        ))}
                        <th className="px-2 py-1 text-left font-medium">C_plasma</th>
                      </tr>
                    </thead>
                    <tbody>
                      {pbResult.rows.map((row, i) => (
                        <tr key={i} className="odd:bg-[var(--color-muted)]/40">
                          <td className="px-2 py-0.5">{formatValue(row.time)}</td>
                          {pbResult.compartments.map((c) => (
                            <td key={c} className="px-2 py-0.5">{formatValue(row[c])}</td>
                          ))}
                          <td className="px-2 py-0.5">{formatValue(row["C_plasma"])}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : null}
            {pdResult !== null ? (
              <div className="mt-3">
                <p className="text-xs text-[var(--color-muted-foreground)]">
                  e0 {formatValue(pdResult.e0)} · emax {formatValue(pdResult.emax)} ·
                  ec50 {formatValue(pdResult.ec50)} · hill {formatValue(pdResult.hill)} ·{" "}
                  {pdResult.n_points} points · POST /api/v1/pkpd/pd
                </p>
                <CurveFigure
                  points={pdResult.concentration.map((c, i) => ({
                    x: c,
                    y: pdResult.effect[i] ?? 0,
                  }))}
                  xLabel="concentration"
                  yLabel="effect"
                />
              </div>
            ) : null}
            {popBody !== null ? (
              <div className="mt-3">
                <p className="text-xs text-[var(--color-muted-foreground)]">
                  {popBody.n_subjects} subjects · {popBody.n_times} grid points ·
                  seed {popBody.seed} · mean/p5/p95 +{" "}
                  {popBody.subject_profiles.length} subject profiles
                </p>
                <CurveFigure
                  points={popBody.summary.mean.map((c, i) => ({ x: grid?.[i] ?? i, y: c }))}
                  xLabel="time"
                  yLabel="mean concentration"
                />
                <div className="mt-2 grid gap-2 md:grid-cols-2">
                  <div className="rounded-md border border-[var(--color-border)] p-3">
                    <p className="text-xs font-semibold text-[var(--color-foreground)]">
                      the p5/p95 envelope (the first/last third of the grid)
                    </p>
                    <table className="mt-2 w-full text-[11px]">
                      <thead>
                        <tr>
                          <th className="text-left">time</th>
                          <th className="text-right">p5</th>
                          <th className="text-right">mean</th>
                          <th className="text-right">p95</th>
                        </tr>
                      </thead>
                      <tbody>
                        {popBody.summary.mean.map((mean, i) =>
                          i % Math.max(1, Math.floor(popBody.n_times / 12)) === 0 ? (
                            <tr key={i}>
                              <td>{formatValue(grid?.[i] ?? i)}</td>
                              <td className="text-right">
                                {formatValue(popBody.summary.p5[i])}
                              </td>
                              <td className="text-right">{formatValue(mean)}</td>
                              <td className="text-right">
                                {formatValue(popBody.summary.p95[i])}
                              </td>
                            </tr>
                          ) : null,
                        )}
                      </tbody>
                    </table>
                  </div>
                  <div className="rounded-md border border-[var(--color-border)] p-3">
                    <p className="text-xs font-semibold text-[var(--color-foreground)]">
                      subject profiles ({popBody.subject_profiles.length} shown)
                    </p>
                    <CurveFigure
                      points={
                        popBody.subject_profiles[0]?.map((c, i) => ({
                          x: grid?.[i] ?? i,
                          y: c,
                        })) ?? []
                      }
                      xLabel="time"
                      yLabel="subject 1 concentration"
                      height={160}
                    />
                  </div>
                </div>
              </div>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      {/* stage 6: Diagnostics */}
      <StageHeader
        index={6}
        title="Diagnostics"
        hint="the terminal-phase regression quality + the interval metrics, honestly stated"
        icon={<Activity className="h-4 w-4 text-[var(--color-muted-foreground)]" />}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Model diagnostics</CardTitle>
            <CardDescription>
              The backend&apos;s own verdicts rendered verbatim: the
              terminal-phase r² and the span_half_lives warning
              (span_low means the terminal slope covers under one
              half-life — a sampling-design limitation, not an error we
              hide), the steady-state fluctuation/swing, and the
              population envelope width.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {ncaResult === null && ssResult === null && popBody === null ? (
              <p className="text-sm text-[var(--color-muted-foreground)]">
                run the NCA (stage two), the steady-state lane or the
                population simulation — the diagnostics are their own
                honest outputs, nothing is invented here.
              </p>
            ) : null}
            {ncaResult !== null ? (
              <div className="rounded-md border border-[var(--color-border)] p-3">
                <p className="text-xs font-semibold text-[var(--color-foreground)]">
                  terminal-phase regression (nca lane)
                </p>
                <ul className="mt-2 list-disc space-y-1 pl-4 text-xs text-[var(--color-muted-foreground)]">
                  <li>
                    r² {formatValue(ncaResult.terminal.r2)} (adjusted{" "}
                    {formatValue(ncaResult.terminal.r2_adj)}) over{" "}
                    {ncaResult.terminal.n_points} tail points — the
                    backend&apos;s own fit, method {ncaResult.terminal.method}
                  </li>
                  <li>
                    span {formatValue(ncaResult.terminal.span_half_lives)}{" "}
                    half-lives{" "}
                    {ncaResult.terminal.span_low
                      ? "— SPAN_LOW: the tail covers under one half-life (the backend's own warning, rendered verbatim)"
                      : "(span adequate)"}
                  </li>
                  <li>
                    clast_pred {formatValue(ncaResult.terminal.clast_pred)} ·
                    lambda_z {formatValue(ncaResult.terminal.lambda_z)} ·
                    half_life {formatValue(ncaResult.terminal.half_life)}
                  </li>
                </ul>
              </div>
            ) : null}
            {ssResult !== null ? (
              <div className="mt-3 rounded-md border border-[var(--color-border)] p-3">
                <p className="text-xs font-semibold text-[var(--color-foreground)]">
                  steady-state interval (τ = {formatValue(ssResult.tau)})
                </p>
                <ul className="mt-2 list-disc space-y-1 pl-4 text-xs text-[var(--color-muted-foreground)]">
                  <li>
                    fluctuation {formatValue(ssResult.metrics.fluctuation)} ·
                    swing {formatValue(ssResult.metrics.swing)} — the
                    backend&apos;s own interval verdicts
                  </li>
                  <li>
                    Cmax_ss {formatValue(ssResult.metrics.Cmax_ss)} · Cmin_ss{" "}
                    {formatValue(ssResult.metrics.Cmin_ss)} · Ctrough{" "}
                    {formatValue(ssResult.metrics.Ctrough)}
                  </li>
                  <li>
                    CL_ss {formatValue(ssResult.metrics.CL_ss)} · R_accum{" "}
                    {formatValue(ssResult.metrics.R_accum)} (null — not
                    computable without dose/lambda_z — when the inputs
                    were not supplied)
                  </li>
                </ul>
              </div>
            ) : null}
            {popBody !== null ? (
              <div className="mt-3 rounded-md border border-[var(--color-border)] p-3">
                <p className="text-xs font-semibold text-[var(--color-foreground)]">
                  population envelope ({popBody.n_subjects} subjects)
                </p>
                <ul className="mt-2 list-disc space-y-1 pl-4 text-xs text-[var(--color-muted-foreground)]">
                  {popBody.summary.mean.map((mean, i) =>
                    i === Math.floor(popBody.n_times / 2) ? (
                      <li key={i}>
                        the grid midpoint (t = {formatValue(grid?.[i] ?? i)}):
                        p5 {formatValue(popBody.summary.p5[i])} · mean{" "}
                        {formatValue(mean)} · p95{" "}
                        {formatValue(popBody.summary.p95[i])} — the real
                        simulated spread at one point, stated
                      </li>
                    ) : null,
                  )}
                  <li>
                    seed {popBody.seed} — the same seed reproduces the
                    same population (the backend&apos;s own seeded
                    simulation, never re-randomized client-side)
                  </li>
                </ul>
              </div>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      {/* stage 7: Exposure/Response Visualization */}
      <StageHeader
        index={7}
        title="Exposure/Response Visualization"
        hint="REAL figure artifacts through the §17 four-format export stack"
        icon={<LineChart className="h-4 w-4 text-[var(--color-muted-foreground)]" />}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Figure artifacts</CardTitle>
            <CardDescription>
              POST /api/v1/artifacts/figure — the exposure profile (the
              curve the backend computed, only from points that arrived)
              and the sigmoid Emax response curve; each registers a
              downloadable, digest-verified artifact.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {figureError ? <ErrorBanner message={figureError} /> : null}
            <div className="flex flex-wrap items-center gap-2">
              <Button
                onClick={() => void handleExposureFigure()}
                disabled={activeCurve === null}
              >
                Register exposure figure
              </Button>
              <Button
                variant="outline"
                onClick={() => void handleResponseFigure()}
                disabled={pdResult === null}
              >
                Register response figure
              </Button>
              <span className="text-xs text-[var(--color-muted-foreground)]">
                {activeCurve === null
                  ? "run a simulation first — the figure needs real points"
                  : `${activeCurve.length} real curve points`}
                {pdResult === null ? " · run the PD model for the response curve" : ""}
              </span>
            </div>
            {exposureArtifact !== null ? (
              <p className="mt-3 text-xs text-[var(--color-muted-foreground)]">
                exposure figure · {exposureArtifact.filename} ·{" "}
                {exposureArtifact.size_bytes} B · sha256{" "}
                {exposureArtifact.sha256.slice(0, 16)}… ·{" "}
                {exposureDigest ?? "downloading…"}
              </p>
            ) : null}
            {responseArtifact !== null ? (
              <p className="mt-3 text-xs text-[var(--color-muted-foreground)]">
                response figure · {responseArtifact.filename} ·{" "}
                {responseArtifact.size_bytes} B · sha256{" "}
                {responseArtifact.sha256.slice(0, 16)}… ·{" "}
                {responseDigest ?? "downloading…"}
              </p>
            ) : null}
            {activeCurve !== null ? (
              <div className="mt-3">
                <p className="text-xs text-[var(--color-muted-foreground)]">
                  the live inline curve (the figure&apos;s own data):
                </p>
                <CurveFigure points={activeCurve} xLabel="time" yLabel="concentration" />
              </div>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      {/* stage 8: Export */}
      <StageHeader
        index={8}
        title="Export"
        hint="the rows-based table lane, the backend's own 5000-row cap mirrored"
        icon={<ArrowDownToLine className="h-4 w-4 text-[var(--color-muted-foreground)]" />}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Table export</CardTitle>
            <CardDescription>
              POST /api/v1/visualization/table — the curve rows, the PBPK
              frame, the PD table or the NCA metrics as a REAL table
              artifact; the download is digest-verified.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-center gap-2">
              {(
                [
                  ["curve", "curve rows (time, concentration)"],
                  ["pbpk-rows", "PBPK frame rows"],
                  ["pd-table", "PD table (concentration, effect)"],
                  ["nca-metrics", "NCA metrics + terminal phase"],
                ] as readonly [ExportSource, string][]
              ).map(([source, label]) => (
                <label
                  key={source}
                  className="flex cursor-pointer items-center gap-2 rounded-md border border-[var(--color-border)] px-3 py-1.5 text-xs"
                >
                  <input
                    type="radio"
                    name="pkpd-export-source"
                    checked={exportSource === source}
                    onChange={() => {
                      setExportSource(source);
                      setExportArtifact(null);
                      setExportDigest(null);
                      setExportError(null);
                    }}
                  />
                  {label}
                </label>
              ))}
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <label className="flex flex-col gap-1">
                format
                <select
                  value={exportFormat}
                  onChange={(event) => setExportFormat(event.target.value as ExportFormat)}
                  className="rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1.5 text-sm"
                >
                  {(["csv", "tsv", "xlsx", "md", "json", "html"] as const).map((f) => (
                    <option key={f} value={f}>{f}</option>
                  ))}
                </select>
              </label>
              <Button onClick={() => void handleExport()}>
                Export table
              </Button>
              <span className="text-xs text-[var(--color-muted-foreground)]">
                {exportRows.length} rows ready · cap {TABLE_EXPORT_MAX_ROWS}
              </span>
            </div>
            {exportError ? <div className="mt-3"><ErrorBanner message={exportError} /></div> : null}
            {exportArtifact !== null ? (
              <p className="mt-3 text-xs text-[var(--color-muted-foreground)]">
                {exportArtifact.filename} · {exportArtifact.size_bytes} B · sha256{" "}
                {exportArtifact.sha256.slice(0, 16)}… ·{" "}
                {exportDigest ?? "downloading…"}
              </p>
            ) : null}
            {exportRows.length > 0 ? (
              <div className="mt-3 max-h-64 overflow-auto rounded-md border border-[var(--color-border)]">
                <table className="w-full border-collapse text-[11px]">
                  <thead className="sticky top-0 bg-[var(--color-muted)]">
                    <tr>
                      {unionRowKeys(exportRows).map((key) => (
                        <th key={key} className="px-2 py-1 text-left font-medium">
                          {key}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {exportRows.slice(0, 50).map((row, i) => (
                      <tr key={i} className="odd:bg-[var(--color-muted)]/40">
                        {unionRowKeys(exportRows).map((key) => (
                          <td key={key} className="px-2 py-0.5">
                            {formatValue(row[key])}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                {exportRows.length > 50 ? (
                  <p className="px-2 py-1 text-[10px] text-[var(--color-muted-foreground)]">
                    first 50 of {exportRows.length} rows shown (the export
                    carries them all)
                  </p>
                ) : null}
              </div>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>
    </div>
  );
}
