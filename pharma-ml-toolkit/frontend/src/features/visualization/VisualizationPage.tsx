import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDownToLine,
  ClipboardCheck,
  ClipboardList,
  Columns2,
  Database,
  Eye,
  Play,
  SlidersHorizontal,
  Trash2,
} from "lucide-react";
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
import { datasetsApi, type DatasetView } from "@/services/datasets/client";
import { ApiError } from "@/services/api/client";
import {
  artifactsApi,
  type ArtifactView,
} from "@/services/artifacts/client";
import { getEventStream } from "@/services/websocket/client";
import {
  assertDoseAccumulationPair,
  assertMdSeriesBounds,
  assertReportBounds,
  assertSeriesPairBounds,
  assertStatBounds,
  vizApi,
  parseJobProgressPayload,
  parseReportBody,
  type ConcentrationProfileResponse,
  type DoseAccumulationResponse,
  type JobProgressPayload,
  type ReportBodyView,
  type StatResponse,
  type VizCatalogResponse,
  type VizMdSeriesResponse,
  type VizTableResponse,
} from "@/services/visualization/client";

/**
 * The Visualization workspace — part11 §11.8's complete sequential
 * workflow:
 *
 * Select Dataset/Analysis Result → Select Visualization → Configure
 * Variables → Generate → Inspect → Filter/Compare → Export
 *
 * Every control below executes the REAL backend through the typed
 * service layer (no fabricated data, no disabled pretend buttons).
 * The Select stage reads the session's artifact registry — the
 * outputs OTHER workspaces registered (figures, tables, reports) —
 * plus the paste lanes for raw series, and VIZ-1's registered-dataset
 * column bridge (stage 3): the stat family's values ride the same
 * bounded `datasetsApi.column()` lane every other workspace rides,
 * one code path with two entry points (the paste textarea and the
 * bridge both fill `valuesText`); the Select Visualization
 * stage is the backend's own `GET /visualization/catalog` inventory
 * (five families, each row naming the backend functions it wraps —
 * never a client-side guess); Configure edits the family's own
 * variables; Generate posts the lane — the report lane defaults to
 * job mode (the COMPLETED job's REST result is authoritative; the
 * WS `file.progress` envelopes drive the per-section progress bar
 * only, and cancelling is native — the job lands CANCELLED, never
 * FAILED); Inspect renders the stats dict the statplots family
 * actually computed (type-aware, bounded); Filter/Compare filters
 * the registry by kind and compares the last two generated
 * artifacts side-by-side (honest — only over what actually
 * arrived); Export is the digest-verified download lane plus the
 * rows-based table export with the backend's own 5000-row cap
 * mirrored client-side.
 */

/** The backend's own table-export row cap (visualization.py's `_MAX_ROWS`). */
const TABLE_EXPORT_MAX_ROWS = 5000;

type FamilyKey =
  | "stat"
  | "concentration-profile"
  | "dose-accumulation"
  | "md-series"
  | "table"
  | "report";

type StatKind = "box" | "histogram" | "ecdf" | "regression";

type ExportFormat = "csv" | "tsv" | "xlsx" | "md" | "json" | "html";

type FigureFormat = "PNG" | "SVG" | "TIFF" | "JPEG";

/**
 * A realistic concentration-time series (the paste lane's own
 * editable sample, in the `time conc` format the parser reads).
 */
const SAMPLE_PROFILE = [
  "time conc",
  "0 0.00",
  "0.5 2.10",
  "1 3.40",
  "2 3.90",
  "4 3.10",
  "6 2.30",
  "8 1.60",
  "12 0.80",
  "16 0.38",
  "24 0.09",
].join("\n");

/**
 * A realistic values sample (forty log-normal-ish draws) — the stat
 * lane's own editable sample.
 */
const SAMPLE_VALUES = [
  2.4, 2.9, 3.3, 3.7, 4.1, 4.4, 4.9, 5.2, 5.6, 6.0,
  6.3, 6.8, 7.1, 7.5, 7.9, 8.4, 8.8, 9.3, 9.8, 10.4,
  11.0, 11.6, 12.3, 13.0, 13.8, 14.7, 15.6, 16.6, 17.7, 19.0,
  20.4, 22.0, 23.8, 25.8, 28.1, 30.8, 33.9, 37.7, 42.3, 48.6,
].join("\n");

/** A small demo `.xvg` text — the md-series lane's own editable sample. */
const SAMPLE_XVG = [
  "# GROMACS — backbone RMSD",
  "@ s0 legend \"RMSD (nm)\"",
  "0.0     0.0000",
  "100.0   0.0821",
  "200.0   0.1543",
  "300.0   0.2077",
  "400.0   0.2411",
  "500.0   0.2638",
  "600.0   0.2812",
  "700.0   0.2976",
  "800.0   0.3104",
  "900.0   0.3189",
  "1000.0  0.3250",
].join("\n");

/**
 * VIZ-1's honest cell converter — the same guards the PK/PD bridge
 * rides: a `null` cell (json_safe's NaN → null) is a row-labeled
 * error, never `Number(null) === 0` fabricating a value the frame
 * never measured; a non-numeric string cell carries the same
 * labeled refusal; the backend's own 422 lanes (an unknown dataset,
 * an unknown column, a non-numeric column) surface verbatim through
 * `describeError` instead.
 */
function numericCell(
  value: string | number | null | undefined,
  row: number,
  column: string,
): number {
  if (value === null || value === undefined) {
    throw new Error(
      `row ${row + 1} of column "${column}": a null value (NaN/Inf in the frame) — never silently zero`,
    );
  }
  const numeric = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(numeric)) {
    throw new Error(
      `row ${row + 1} of column "${column}": "${String(value)}" is not a finite number`,
    );
  }
  return numeric;
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

/** A numbered workflow stage header (§11.8's fixed order). */
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

/** The stats dict's rows in the backend's own key order. */
function statsEntries(
  stats: Readonly<Record<string, unknown>>,
): readonly (readonly [string, unknown])[] {
  return Object.entries(stats);
}

/** One pasted pair-series row (`time conc`). */
interface PairPoint {
  readonly time: number;
  readonly conc: number;
}

/**
 * Parse a pasted pair series — one `time conc` pair per line
 * (whitespace or comma separated; a header line is skipped). Every
 * malformed row is this parser's own labeled error, never a silent
 * drop; the backend's own validator remains the authority.
 */
function parsePairSeries(text: string, label: string): PairPoint[] {
  const rows: PairPoint[] = [];
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
 * Parse a pasted number list (one per line, or comma/space
 * separated) — the stat lane's own paste format.
 */
function parseNumberList(text: string, label: string): number[] {
  const cells = text
    .split(/[\s,]+/)
    .map((cell) => cell.trim())
    .filter((cell) => cell !== "");
  const values: number[] = [];
  cells.forEach((cell, i) => {
    const v = Number(cell);
    if (!Number.isFinite(v)) {
      throw new Error(`${label} entry ${i + 1}: "${cell}" is not a finite number`);
    }
    values.push(v);
  });
  return values;
}

/** One registry row + digest-verified download. */
function ArtifactRow({
  artifact,
  onDownload,
  digest,
  selected,
  onSelect,
  onView,
  onDelete,
}: {
  artifact: ArtifactView;
  onDownload: (artifact: ArtifactView) => void;
  digest: string | null;
  selected?: boolean;
  onSelect?: (artifact: ArtifactView) => void;
  onView?: (artifact: ArtifactView) => void;
  onDelete?: (artifact: ArtifactView) => void;
}) {
  return (
    <div
      className={`flex flex-wrap items-center justify-between gap-2 rounded-md border p-3 ${
        selected === true
          ? "border-[var(--color-primary)]"
          : "border-[var(--color-border)]"
      }`}
    >
      <div className="min-w-0">
        <p className="truncate text-sm font-medium text-[var(--color-foreground)]">
          {artifact.filename}
        </p>
        <p className="text-[11px] text-[var(--color-muted-foreground)]">
          {artifact.kind} · {artifact.content_type} ·{" "}
          {(artifact.size_bytes / 1024).toFixed(1)} KiB · sha256{" "}
          {artifact.sha256.slice(0, 16)}… ·{" "}
          {artifact.created_at.slice(0, 19).replace("T", " ")}
        </p>
      </div>
      <div className="flex items-center gap-2">
        {onSelect !== undefined ? (
          <Button size="sm" variant="outline" onClick={() => onSelect(artifact)}>
            Select
          </Button>
        ) : null}
        <Button size="sm" variant="outline" onClick={() => onDownload(artifact)}>
          Download
        </Button>
        {onView !== undefined ? (
          <Button
            size="sm"
            variant="outline"
            onClick={() => onView(artifact)}
            aria-label={`View artifact ${artifact.filename}`}
          >
            <Eye aria-hidden className="h-4 w-4" />
            View
          </Button>
        ) : null}
        {onDelete !== undefined ? (
          <Button
            size="sm"
            variant="outline"
            onClick={() => onDelete(artifact)}
            aria-label={`Delete artifact ${artifact.filename}`}
          >
            <Trash2 aria-hidden className="h-4 w-4" />
            Delete
          </Button>
        ) : null}
        {digest !== null ? (
          <span className="text-[11px] text-[var(--color-muted-foreground)]">
            {digest}
          </span>
        ) : null}
      </div>
    </div>
  );
}

export function VisualizationPage() {
  // stage 1: the source lanes
  const [registry, setRegistry] = useState<readonly ArtifactView[] | null>(null);
  const [registryError, setRegistryError] = useState<string | null>(null);
  const [selectedArtifact, setSelectedArtifact] = useState<ArtifactView | null>(null);
  const [selectedDigest, setSelectedDigest] = useState<string | null>(null);
  // part17 §3.2 (the G1 lane) — the view/delete consumers the
  // consumption hub always owed: the one-row View fetch (the full
  // ArtifactView with its meta) and the two-step Delete over the
  // EXISTING typed artifactsApi.delete (no duplicate client)
  const [viewedArtifact, setViewedArtifact] = useState<ArtifactView | null>(null);
  const [viewBusy, setViewBusy] = useState(false);
  const [viewError, setViewError] = useState<string | null>(null);
  const [deletePending, setDeletePending] = useState<ArtifactView | null>(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [deleteNotice, setDeleteNotice] = useState<string | null>(null);
  const [valuesText, setValuesText] = useState(SAMPLE_VALUES);
  const [xText, setXText] = useState("");
  const [yText, setYText] = useState("");
  const [pairText, setPairText] = useState(SAMPLE_PROFILE);
  const [xvgText, setXvgText] = useState(SAMPLE_XVG);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loadedShape, setLoadedShape] = useState<string | null>(null);

  // stage 3 (VIZ-1): the registered-dataset column bridge — the
  // dataset registry, fetched once on mount through the SAME
  // datasetsApi.list() lane the other bridge cards ride
  const [bridgeDatasets, setBridgeDatasets] = useState<readonly DatasetView[]>([]);
  const [bridgeRegistryError, setBridgeRegistryError] = useState<string | null>(null);
  const [bridgeDataset, setBridgeDataset] = useState<DatasetView | null>(null);
  const [bridgeColumn, setBridgeColumn] = useState("");
  const [bridgeBusy, setBridgeBusy] = useState(false);
  const [bridgeError, setBridgeError] = useState<string | null>(null);
  const [bridgeNote, setBridgeNote] = useState<string | null>(null);

  // stage 2: the backend's own catalog + the family selection
  const [catalog, setCatalog] = useState<VizCatalogResponse | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [family, setFamily] = useState<FamilyKey>("stat");

  // stage 3: the per-family configuration
  const [statKind, setStatKind] = useState<StatKind>("histogram");
  const [statBins, setStatBins] = useState("auto");
  const [statWhis, setStatWhis] = useState("1.5");
  const [statBoot, setStatBoot] = useState("200");
  const [statSeed, setStatSeed] = useState("0");
  const [statColor, setStatColor] = useState("");
  const [statFormat, setStatFormat] = useState<"PNG" | "SVG">("PNG");
  const [cpAuc, setCpAuc] = useState("");
  const [cpTitle, setCpTitle] = useState("Concentration-time profile");
  const [cpFormat, setCpFormat] = useState<FigureFormat>("PNG");
  const [cpDpi, setCpDpi] = useState("150");
  const [daDose, setDaDose] = useState("500");
  const [daTau, setDaTau] = useState("12");
  const [daHalf, setDaHalf] = useState("6");
  const [daKe, setDaKe] = useState("");
  const [daVd, setDaVd] = useState("50");
  const [daF, setDaF] = useState("1");
  const [daDoses, setDaDoses] = useState("6");
  const [daTitle, setDaTitle] = useState("Repeated dosing — approach to steady state");
  const [daFormat, setDaFormat] = useState<FigureFormat>("PNG");
  const [daDpi, setDaDpi] = useState("150");
  const [msTitle, setMsTitle] = useState("MD series — RMSD");
  const [msXLabel, setMsXLabel] = useState("Time (ps)");
  const [msYLabel, setMsYLabel] = useState("RMSD (nm)");
  const [msFormat, setMsFormat] = useState<FigureFormat>("PNG");
  const [tableFormat, setTableFormat] = useState<ExportFormat>("csv");
  const [tableFilename, setTableFilename] = useState("visualization-table");
  const [rpTitle, setRpTitle] = useState("PharmaML results");
  const [rpSubtitle, setRpSubtitle] = useState("");
  const [rpTheme, setRpTheme] = useState("auto");
  const [rpMetrics, setRpMetrics] = useState(true);
  const [rpTables, setRpTables] = useState(true);
  const [rpCp, setRpCp] = useState(true);
  const [rpDa, setRpDa] = useState(true);
  const [rpStat, setRpStat] = useState(false);
  const [rpMode, setRpMode] = useState<"job" | "sync">("job");

  // stage 4: the generated lanes
  const [busy, setBusy] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [statResult, setStatResult] = useState<StatResponse | null>(null);
  const [cpResult, setCpResult] = useState<ConcentrationProfileResponse | null>(null);
  const [daResult, setDaResult] = useState<DoseAccumulationResponse | null>(null);
  const [msResult, setMsResult] = useState<VizMdSeriesResponse | null>(null);
  const [tableResult, setTableResult] = useState<VizTableResponse | null>(null);
  const [reportJobId, setReportJobId] = useState<string | null>(null);
  const [reportBody, setReportBody] = useState<ReportBodyView | null>(null);
  const [reportProgress, setReportProgress] = useState<
    readonly JobProgressPayload[]
  >([]);

  // stage 6: the filter + compare state
  const [kindFilter, setKindFilter] = useState("");
  const [generated, setGenerated] = useState<readonly ArtifactView[]>([]);
  const [digests, setDigests] = useState<Record<string, string | null>>({});

  // stage 1: the registry fetch (the cross-workspace outputs)
  const refreshRegistry = useCallback(() => {
    setRegistryError(null);
    artifactsApi
      .list()
      .then((items) => setRegistry(items))
      .catch((error) => setRegistryError(describeError(error)));
  }, []);
  useEffect(() => {
    refreshRegistry();
  }, [refreshRegistry]);

  // stage 2: the catalog fetch (once, honestly)
  useEffect(() => {
    let cancelled = false;
    vizApi
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

  // stage 3 (VIZ-1): the dataset registry fetch, once on mount — the
  // bridge lists what the Datasets workspace actually registered
  useEffect(() => {
    let cancelled = false;
    datasetsApi
      .list()
      .then((response) => {
        if (!cancelled) {
          setBridgeDatasets(response.datasets);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setBridgeRegistryError(describeError(error));
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // stage 3 (VIZ-1): selecting a dataset fetches its view for the
  // column inventory (the same bounded lane every bridge rides)
  const selectBridgeDataset = useCallback(async (datasetId: string) => {
    setBridgeNote(null);
    setBridgeError(null);
    setBridgeColumn("");
    if (datasetId === "") {
      setBridgeDataset(null);
      return;
    }
    try {
      const view = await datasetsApi.view(datasetId);
      setBridgeDataset(view);
    } catch (error) {
      setBridgeDataset(null);
      setBridgeError(describeError(error));
    }
  }, []);

  // stage 3 (VIZ-1): the column pull — ONE column through the same
  // bounded column() lane every other workspace rides, converted to
  // the SAME newline-joined values text the paste lane parses (one
  // code path, two entry points — the bridge fills the values
  // textarea so the stat request rides parseNumberList exactly as
  // before). A truncated pull names its 500-row bound (§17.3 — the
  // bounded view never masquerades as the whole column); a null cell
  // is a row-labeled error, never a silent zero; every backend
  // 422/404 surfaces verbatim through describeError.
  const handleBridgeColumn = useCallback(async () => {
    if (bridgeDataset === null || bridgeColumn === "") {
      return;
    }
    setBridgeBusy(true);
    setBridgeError(null);
    setBridgeNote(null);
    try {
      const columnPayload = await datasetsApi.column(
        bridgeDataset.dataset_id,
        bridgeColumn,
      );
      const values = columnPayload.values.map((value, index) =>
        numericCell(value, index, bridgeColumn),
      );
      if (values.length < 4) {
        throw new Error(
          `the column carries ${values.length} finite values, at least 4 required by the stat lane`,
        );
      }
      setValuesText(values.map((value) => String(value)).join("\n"));
      setLoadError(null);
      setLoadedShape(null);
      if (columnPayload.truncated) {
        setBridgeNote(
          `the column bridge is bounded at 500 rows (the chemistry engine's own request cap) — the values lane carries the first ${columnPayload.returned} of the dataset's ${columnPayload.n_rows} rows`,
        );
      }
    } catch (error) {
      setBridgeError(describeError(error));
    } finally {
      setBridgeBusy(false);
    }
  }, [bridgeDataset, bridgeColumn]);

  // stage 1: the paste lanes' own validation
  const handleLoad = useCallback(() => {
    setLoadError(null);
    setLoadedShape(null);
    try {
      const values = parseNumberList(valuesText, "the values lane");
      const pair = parsePairSeries(pairText, "the time/conc lane");
      setLoadedShape(
        `${values.length} values · ${pair.length} time/conc points · ${
          xvgText.length > 0 ? "the .xvg text is pasted" : "no .xvg text"
        }${xText.trim() !== "" && yText.trim() !== "" ? " · regression x/y pasted" : ""}`,
      );
    } catch (error) {
      setLoadError(describeError(error));
    }
  }, [valuesText, pairText, xvgText, xText, yText]);

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

  const rememberGenerated = useCallback((artifact: ArtifactView) => {
    setGenerated((prev) => [artifact, ...prev].slice(0, 8));
  }, []);

  const handleDownload = useCallback(
    async (artifact: ArtifactView) => {
      try {
        const digest = await verifiedDownload(artifact);
        setDigests((prev) => ({ ...prev, [artifact.artifact_id]: digest }));
      } catch (error) {
        setRunError(describeError(error));
      }
    },
    [verifiedDownload],
  );

  const handleSelect = useCallback(
    async (artifact: ArtifactView) => {
      setSelectedArtifact(artifact);
      setSelectedDigest(null);
      try {
        const digest = await verifiedDownload(artifact);
        setSelectedDigest(digest);
      } catch (error) {
        setRunError(describeError(error));
      }
    },
    [verifiedDownload],
  );

  // part17 §3.2 (G1) — the View lane: the one-row fetch through the
  // EXISTING typed artifactsApi.view (GET /api/v1/artifacts/{id}),
  // rendered as the full registry record below the list. A 404 is
  // shown honestly (the row can be evicted server-side between the
  // list and the click).
  const handleArtifactView = useCallback(async (artifact: ArtifactView) => {
    setViewBusy(true);
    setViewError(null);
    try {
      const view = await artifactsApi.view(artifact.artifact_id);
      setViewedArtifact(view);
    } catch (error) {
      setViewedArtifact(null);
      setViewError(describeError(error));
    } finally {
      setViewBusy(false);
    }
  }, []);

  // part17 §3.2 (G1) — the Delete lane, the r356 dataset-deletion
  // discipline applied to the artifact registry: the row action only
  // ARMS the confirmation; only the confirm button fires the typed
  // artifactsApi.delete (no second client, no inline fetch); the
  // honest 404 already-gone lane reconciles; and every stage-local
  // derivative that referenced the evicted row is cleared (the
  // selection, the compare memory, the digest cache, the viewed
  // record — never a dangling reference to an evicted id).
  const requestArtifactDelete = useCallback((artifact: ArtifactView) => {
    setDeletePending(artifact);
    setDeleteError(null);
    setDeleteNotice(null);
  }, []);

  const cancelArtifactDelete = useCallback(() => {
    setDeletePending(null);
    setDeleteError(null);
  }, []);

  const confirmArtifactDelete = useCallback(
    async (artifact: ArtifactView) => {
      setDeleteBusy(true);
      setDeleteError(null);
      setDeleteNotice(null);
      try {
        await artifactsApi.delete(artifact.artifact_id);
        setDeletePending(null);
        if (selectedArtifact?.artifact_id === artifact.artifact_id) {
          setSelectedArtifact(null);
          setSelectedDigest(null);
        }
        if (viewedArtifact?.artifact_id === artifact.artifact_id) {
          setViewedArtifact(null);
        }
        setGenerated((prev) =>
          prev.filter((row) => row.artifact_id !== artifact.artifact_id),
        );
        setDigests((prev) => {
          const next = { ...prev };
          delete next[artifact.artifact_id];
          return next;
        });
        await refreshRegistry();
        setDeleteNotice(
          `Deleted "${artifact.filename}" from the artifact registry.`,
        );
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) {
          // the honest already-gone lane: the row was evicted
          // server-side (or by another surface) — reconcile and say so
          setDeletePending(null);
          if (selectedArtifact?.artifact_id === artifact.artifact_id) {
            setSelectedArtifact(null);
            setSelectedDigest(null);
          }
          await refreshRegistry();
          setDeleteNotice(
            `"${artifact.filename}" was already gone (404) — registry refreshed.`,
          );
          return;
        }
        setDeleteError(describeError(error));
      } finally {
        setDeleteBusy(false);
      }
    },
    [refreshRegistry, selectedArtifact, viewedArtifact],
  );

  // stage 4: the report job's authoritative REST relay (§13)
  const handleReportResult = useCallback((result: unknown) => {
    try {
      setReportBody(parseReportBody(result, "report job result"));
    } catch (error) {
      setRunError(describeError(error));
    }
  }, []);

  // the report job's live telemetry: the per-section file.progress
  // envelopes (progress-only — the body comes from the REST relay)
  useEffect(() => {
    if (reportJobId === null) {
      return;
    }
    const stream = getEventStream();
    const unsubscribe = stream.subscribe((envelope) => {
      if (envelope.job_id !== reportJobId) {
        return;
      }
      if (envelope.event_type === "file.progress") {
        try {
          const payload = parseJobProgressPayload(
            envelope.payload,
            "file.progress",
          );
          setReportProgress((prev) => [...prev, payload]);
        } catch (error) {
          setRunError(describeError(error));
        }
      }
    });
    return unsubscribe;
  }, [reportJobId]);

  // stage 4: the generate lane
  const handleGenerate = useCallback(async () => {
    setBusy(true);
    setRunError(null);
    setStatResult(null);
    setCpResult(null);
    setDaResult(null);
    setMsResult(null);
    setTableResult(null);
    setReportJobId(null);
    setReportBody(null);
    setReportProgress([]);
    try {
      if (family === "stat") {
        const values = parseNumberList(valuesText, "the values lane");
        const x = xText.trim() === "" ? null : parseNumberList(xText, "the x lane");
        const y = yText.trim() === "" ? null : parseNumberList(yText, "the y lane");
        assertStatBounds("stat", statKind, values, x, y);
        const result = await vizApi.stat({
          kind: statKind,
          values: statKind === "regression" ? null : values,
          x,
          y,
          bins: statBins === "auto" ? "auto" : Number(statBins),
          whis: Number(statWhis),
          nBoot: Number(statBoot),
          seed: Number(statSeed),
          color: statColor.trim() === "" ? null : statColor,
          withArtifact: true,
          format: statFormat,
        });
        setStatResult(result);
        if (result.artifact !== null) {
          rememberGenerated(result.artifact);
        }
      } else if (family === "concentration-profile") {
        const pair = parsePairSeries(pairText, "the time/conc lane");
        assertSeriesPairBounds(
          "concentration-profile",
          pair.map((p) => p.time),
          pair.map((p) => p.conc),
        );
        const auc = cpAuc.trim() === "" ? null : Number(cpAuc);
        if (auc !== null && !Number.isFinite(auc)) {
          throw new Error("auc is not a finite number");
        }
        const dpi = Number(cpDpi);
        if (!Number.isFinite(dpi)) {
          throw new Error("dpi is not a finite number");
        }
        const result = await vizApi.concentrationProfile({
          times: pair.map((p) => p.time),
          conc: pair.map((p) => p.conc),
          auc,
          title: cpTitle,
          format: cpFormat,
          dpi,
        });
        setCpResult(result);
        rememberGenerated(result.artifact);
      } else if (family === "dose-accumulation") {
        const dose = Number(daDose);
        const tau = Number(daTau);
        const vd = Number(daVd);
        const f = Number(daF);
        const nDoses = Number(daDoses);
        const dpi = Number(daDpi);
        if (![dose, tau, vd, f, nDoses, dpi].every((v) => Number.isFinite(v))) {
          throw new Error(
            "the dosing parameters must all be finite numbers",
          );
        }
        const tHalf = daHalf.trim() === "" ? null : Number(daHalf);
        const ke = daKe.trim() === "" ? null : Number(daKe);
        if (tHalf !== null && !Number.isFinite(tHalf)) {
          throw new Error("t_half is not a finite number");
        }
        if (ke !== null && !Number.isFinite(ke)) {
          throw new Error("ke is not a finite number");
        }
        assertDoseAccumulationPair("dose-accumulation", tHalf, ke);
        const result = await vizApi.doseAccumulation({
          dose,
          tau,
          tHalf,
          ke,
          vd,
          f,
          nDoses,
          title: daTitle,
          format: daFormat,
          dpi,
        });
        setDaResult(result);
        rememberGenerated(result.artifact);
      } else if (family === "md-series") {
        assertMdSeriesBounds("md-series", { rmsd: xvgText });
        const result = await vizApi.mdSeries({
          series: { rmsd: xvgText },
          title: msTitle,
          xLabel: msXLabel,
          yLabel: msYLabel,
          format: msFormat,
        });
        setMsResult(result);
        rememberGenerated(result.artifact);
      } else if (family === "table") {
        const pair = parsePairSeries(pairText, "the time/conc lane");
        const rows = pair.map((p, i) => ({ index: i, time: p.time, conc: p.conc }));
        if (rows.length > TABLE_EXPORT_MAX_ROWS) {
          throw new Error(
            `the export carries ${rows.length} rows; the backend's own cap is ${TABLE_EXPORT_MAX_ROWS}`,
          );
        }
        const result = await vizApi.exportTable({
          rows,
          format: tableFormat,
          filename: tableFilename,
        });
        setTableResult(result);
        rememberGenerated(result.artifact);
      } else {
        // the report lane (job default / sync)
        const theme = rpTheme;
        if (catalog !== null) {
          assertReportBounds("report", theme, catalog.report_themes, null);
        }
        const pair = parsePairSeries(pairText, "the time/conc lane");
        const values = parseNumberList(valuesText, "the values lane");
        const dose = Number(daDose);
        const tau = Number(daTau);
        const vd = Number(daVd);
        const f = Number(daF);
        const nDoses = Number(daDoses);
        const tHalf = daHalf.trim() === "" ? null : Number(daHalf);
        const ke = daKe.trim() === "" ? null : Number(daKe);
        assertDoseAccumulationPair("report dose accumulation", tHalf, ke);
        const metrics: Record<string, number | string> =
          rpMetrics && statResult !== null
            ? Object.fromEntries(
                statsEntries(statResult.stats)
                  .filter(([, v]) => typeof v === "number")
                  .map(([k, v]) => [k, v as number]),
              )
            : {};
        const tables = rpTables
          ? [
              {
                heading: "the pasted series",
                rows: pair.map((p, i) => ({
                  index: i,
                  time: p.time,
                  conc: p.conc,
                })),
              },
              {
                heading: "the pasted values",
                rows: values.map((v, i) => ({ index: i, value: v })),
              },
            ]
          : null;
        const concentrationProfile = rpCp
          ? {
              times: pair.map((p) => p.time),
              conc: pair.map((p) => p.conc),
              title: cpTitle,
              caption: "the pasted concentration-time profile",
            }
          : null;
        const doseAccumulation = rpDa
          ? {
              dose,
              tau,
              t_half: tHalf,
              ke,
              vd,
              f,
              n_doses: nDoses,
              title: daTitle,
              caption: "the dosing accumulation panel",
            }
          : null;
        const stat = rpStat
          ? {
              kind: statKind,
              values,
              caption: `the ${statKind} panel`,
            }
          : null;
        const response = await vizApi.report({
          title: rpTitle,
          subtitle: rpSubtitle.trim() === "" ? null : rpSubtitle,
          theme,
          metrics: Object.keys(metrics).length > 0 ? metrics : null,
          tables,
          concentrationProfile,
          doseAccumulation,
          stat,
          mode: rpMode,
        });
        if (response.mode === "sync") {
          setReportBody(response);
          rememberGenerated(response.artifact);
        } else {
          setReportJobId(response.job.job_id);
        }
      }
    } catch (error) {
      setRunError(describeError(error));
    } finally {
      setBusy(false);
    }
  }, [
    catalog,
    cpAuc,
    cpFormat,
    cpTitle,
    daDoses,
    daDose,
    daF,
    daFormat,
    daHalf,
    daKe,
    daTau,
    daTitle,
    daVd,
    family,
    msFormat,
    msTitle,
    msXLabel,
    msYLabel,
    pairText,
    rememberGenerated,
    reportJobId,
    rpMetrics,
    rpMode,
    rpStat,
    rpSubtitle,
    rpTables,
    rpTheme,
    rpTitle,
    statBoot,
    statColor,
    statFormat,
    statKind,
    statSeed,
    statWhis,
    statResult,
    tableFilename,
    tableFormat,
    valuesText,
    xText,
    xvgText,
    yText,
  ]);

  // the registry filtered by kind (the Filter stage)
  const filteredRegistry = useMemo(() => {
    if (registry === null) {
      return null;
    }
    if (kindFilter === "") {
      return registry;
    }
    return registry.filter((artifact) => artifact.kind === kindFilter);
  }, [registry, kindFilter]);

  const registryKinds = useMemo(() => {
    if (registry === null) {
      return [];
    }
    const kinds = new Set(registry.map((artifact) => artifact.kind));
    return [...kinds];
  }, [registry]);

  // the compare pair: the last two generated artifacts (honest —
  // only over what actually arrived)
  const comparePair = useMemo(
    () => generated.slice(0, 2),
    [generated],
  );

  // -- the render ------------------------------------------------------

  return (
    <div className="mx-auto max-w-5xl px-4 pb-16">
      <header className="pt-8">
        <h1 className="text-2xl font-semibold text-[var(--color-foreground)]">
          Visualization
        </h1>
        <p className="mt-1 text-sm text-[var(--color-muted-foreground)]">
          Select Dataset/Analysis Result → Select Visualization → Configure
          Variables → Generate → Inspect → Filter/Compare → Export — the REAL
          plot families over the session&apos;s own artifact registry (the
          outputs other workspaces created).
        </p>
      </header>

      <div className="mt-6">
        {/* part16 §8 (Phase 5): the workspace's demo-data entry point —
            a series load fills the time/conc paste lane verbatim */}
        <DemoDataSection
          workflow="visualization"
          onSeriesLoaded={(payload) => {
            setPairText(
              payload.times
                .map((time, index) => `${time} ${payload.conc[index]}`)
                .join("\n"),
            );
          }}
        />
      </div>

      {/* stage 1: Select Dataset/Analysis Result */}
      <StageHeader
        index={1}
        title="Select Dataset/Analysis Result"
        hint="the session's artifact registry (the other workspaces' outputs) + the paste lanes + a registered dataset's own column (the stage-3 bridge, VIZ-1)"
        icon={<Database className="h-4 w-4 text-[var(--color-muted-foreground)]" />}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">The source</CardTitle>
            <CardDescription>
              GET /api/v1/artifacts — the session&apos;s registry, newest
              first: every figure, table and report the other workspaces
              registered. The paste lanes below carry the raw series the
              stat / pharma-figure / md-series families read.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {registryError ? <ErrorBanner message={registryError} /> : null}
            {filteredRegistry !== null ? (
              <div className="max-h-64 space-y-2 overflow-auto">
                {filteredRegistry.length === 0 ? (
                  <p className="text-xs text-[var(--color-muted-foreground)]">
                    the registry is empty — generate a figure in any workspace
                    and refresh
                  </p>
                ) : (
                  filteredRegistry.map((artifact) => (
                    <ArtifactRow
                      key={artifact.artifact_id}
                      artifact={artifact}
                      onDownload={handleDownload}
                      digest={digests[artifact.artifact_id] ?? null}
                      selected={selectedArtifact?.artifact_id === artifact.artifact_id}
                      onSelect={handleSelect}
                      onView={handleArtifactView}
                      onDelete={requestArtifactDelete}
                    />
                  ))
                )}
              </div>
            ) : (
              <p className="text-xs text-[var(--color-muted-foreground)]">
                loading the registry…
              </p>
            )}
            {deleteNotice !== null ? (
              <p
                role="status"
                className="mt-3 text-sm text-[var(--color-muted-foreground)]"
              >
                {deleteNotice}
              </p>
            ) : null}
            {viewError !== null ? (
              <div className="mt-3">
                <ErrorBanner message={viewError} />
              </div>
            ) : null}
            {viewBusy ? (
              <p className="mt-3 text-xs text-[var(--color-muted-foreground)]">
                fetching the full registry record…
              </p>
            ) : null}
            {viewedArtifact !== null ? (
              <div className="mt-4 rounded-md border border-[var(--color-border)] p-3">
                <p className="text-xs font-semibold text-[var(--color-foreground)]">
                  The registry record — {viewedArtifact.filename}
                </p>
                <table className="mt-2 w-full text-[11px]">
                  <tbody>
                    <tr>
                      <td className="py-0.5 pr-2 align-top text-[var(--color-muted-foreground)]">artifact_id</td>
                      <td className="py-0.5 text-right font-mono">{viewedArtifact.artifact_id}</td>
                    </tr>
                    <tr>
                      <td className="py-0.5 pr-2 align-top text-[var(--color-muted-foreground)]">kind</td>
                      <td className="py-0.5 text-right">{viewedArtifact.kind}</td>
                    </tr>
                    <tr>
                      <td className="py-0.5 pr-2 align-top text-[var(--color-muted-foreground)]">content_type</td>
                      <td className="py-0.5 text-right">{viewedArtifact.content_type}</td>
                    </tr>
                    <tr>
                      <td className="py-0.5 pr-2 align-top text-[var(--color-muted-foreground)]">size</td>
                      <td className="py-0.5 text-right">
                        {(viewedArtifact.size_bytes / 1024).toFixed(1)} KiB ({viewedArtifact.size_bytes.toLocaleString()} B)
                      </td>
                    </tr>
                    <tr>
                      <td className="py-0.5 pr-2 align-top text-[var(--color-muted-foreground)]">sha256</td>
                      <td className="py-0.5 text-right font-mono break-all">{viewedArtifact.sha256}</td>
                    </tr>
                    <tr>
                      <td className="py-0.5 pr-2 align-top text-[var(--color-muted-foreground)]">created</td>
                      <td className="py-0.5 text-right">
                        {viewedArtifact.created_at.slice(0, 19).replace("T", " ")}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-0.5 pr-2 align-top text-[var(--color-muted-foreground)]">meta</td>
                      <td className="py-0.5 text-right font-mono break-all">
                        {JSON.stringify(viewedArtifact.meta)}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-0.5 pr-2 align-top text-[var(--color-muted-foreground)]">download route</td>
                      <td className="py-0.5 text-right font-mono break-all">{viewedArtifact.download_url}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            ) : null}
            {deletePending !== null ? (
              <Card className="mt-4 border-[var(--color-destructive)]/40">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Trash2 aria-hidden className="h-4 w-4" />
                    Delete artifact “{deletePending.filename}”?
                  </CardTitle>
                  <CardDescription>
                    {deletePending.kind} · {deletePending.content_type} ·{" "}
                    {(deletePending.size_bytes / 1024).toFixed(1)} KiB · id{" "}
                    {deletePending.artifact_id}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-[var(--color-muted-foreground)]">
                    This evicts the session registry row together with its
                    persisted copy (the SQLite row and the vault payload —
                    artifact_id {deletePending.artifact_id}). Datasets, trained
                    models, priors and the persisted job history stay
                    untouched — this action never deletes another record.
                    A workspace that already captured the figure stays
                    unaffected.
                  </p>
                  {deleteError !== null ? (
                    <div className="mt-3">
                      <ErrorBanner message={deleteError} />
                    </div>
                  ) : null}
                  <div className="mt-4 flex gap-2">
                    <Button variant="outline" onClick={cancelArtifactDelete} disabled={deleteBusy}>
                      Cancel
                    </Button>
                    <Button
                      variant="destructive"
                      onClick={() => void confirmArtifactDelete(deletePending)}
                      disabled={deleteBusy}
                    >
                      <Trash2 aria-hidden className="h-4 w-4" />
                      {deleteBusy ? "Deleting…" : "Delete artifact"}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ) : null}
            <div className="mt-3">
              <p className="text-xs font-semibold text-[var(--color-foreground)]">
                The paste lanes
              </p>
              <textarea
                value={valuesText}
                onChange={(event) => setValuesText(event.target.value)}
                rows={5}
                spellCheck={false}
                placeholder={"one value per line (or comma/space separated)"}
                className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-2 font-mono text-xs"
              />
              <div className="mt-2 grid gap-3 md:grid-cols-2">
                <textarea
                  value={xText}
                  onChange={(event) => setXText(event.target.value)}
                  rows={4}
                  spellCheck={false}
                  placeholder={"regression x — one per line (optional)"}
                  className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-2 font-mono text-xs"
                />
                <textarea
                  value={yText}
                  onChange={(event) => setYText(event.target.value)}
                  rows={4}
                  spellCheck={false}
                  placeholder={"regression y — one per line (optional)"}
                  className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-2 font-mono text-xs"
                />
              </div>
              <textarea
                value={pairText}
                onChange={(event) => setPairText(event.target.value)}
                rows={6}
                spellCheck={false}
                placeholder={"time conc (one pair per line)"}
                className="mt-2 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-2 font-mono text-xs"
              />
              <textarea
                value={xvgText}
                onChange={(event) => setXvgText(event.target.value)}
                rows={5}
                spellCheck={false}
                placeholder={"# GROMACS .xvg text"}
                className="mt-2 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-2 font-mono text-xs"
              />
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <Button onClick={handleLoad}>Load &amp; validate</Button>
              <Button
                variant="outline"
                onClick={() => {
                  setValuesText(SAMPLE_VALUES);
                  setPairText(SAMPLE_PROFILE);
                  setXvgText(SAMPLE_XVG);
                  setXText("");
                  setYText("");
                }}
              >
                Load samples
              </Button>
            </div>
            {loadError ? <div className="mt-3"><ErrorBanner message={loadError} /></div> : null}
            {loadedShape !== null ? (
              <p className="mt-3 text-xs text-[var(--color-muted-foreground)]">
                loaded: {loadedShape}
              </p>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      {/* stage 2: Select Visualization */}
      <StageHeader
        index={2}
        title="Select Visualization"
        hint="the backend's own GET /api/v1/visualization/catalog inventory"
        icon={<ClipboardList className="h-4 w-4 text-[var(--color-muted-foreground)]" />}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">The plot families</CardTitle>
            <CardDescription>
              Five families, each row naming the backend module it wraps —
              never a client-side guess. The selection below only picks
              which of THOSE routes the Generate stage calls.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {catalogError ? <ErrorBanner message={catalogError} /> : null}
            {catalog !== null ? (
              <div className="overflow-auto rounded-md border border-[var(--color-border)]">
                <table className="w-full border-collapse text-[11px]">
                  <thead className="bg-[var(--color-muted)]">
                    <tr>
                      <th className="px-2 py-1 text-left font-medium">family</th>
                      <th className="px-2 py-1 text-left font-medium">kinds / formats</th>
                      <th className="px-2 py-1 text-left font-medium">backend</th>
                    </tr>
                  </thead>
                  <tbody>
                    {catalog.families.map((row) => (
                      <tr key={row.family} className="odd:bg-[var(--color-muted)]/40">
                        <td className="px-2 py-0.5">{row.family}</td>
                        <td className="px-2 py-0.5">
                          {(row.kinds.length > 0 ? row.kinds : row.formats).join(", ")}
                        </td>
                        <td className="px-2 py-0.5 text-[var(--color-muted-foreground)]">
                          {row.backend}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="border-t border-[var(--color-border)] px-2 py-1 text-[11px] text-[var(--color-muted-foreground)]">
                  palette: {catalog.palette.join(", ")} · export formats:{" "}
                  {catalog.export_formats.join(", ")} · report themes:{" "}
                  {catalog.report_themes.join(", ")}
                </p>
              </div>
            ) : (
              <p className="text-xs text-[var(--color-muted-foreground)]">
                loading the catalog…
              </p>
            )}
            <div className="mt-3 flex flex-wrap gap-2">
              {(
                [
                  ["stat", "Statplots (box / histogram / ecdf / regression)"],
                  ["concentration-profile", "Concentration-time profile"],
                  ["dose-accumulation", "Dose accumulation"],
                  ["md-series", "MD series (`.xvg`)"],
                  ["table", "Table export"],
                  ["report", "HTML report (managed job)"],
                ] as const
              ).map(([key, label]) => (
                <Button
                  key={key}
                  size="sm"
                  variant={family === key ? "default" : "outline"}
                  onClick={() => setFamily(key)}
                >
                  {label}
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>
      </StageHeader>

      {/* stage 3: Configure Variables */}
      <StageHeader
        index={3}
        title="Configure Variables"
        hint="the selected family's own parameters"
        icon={<SlidersHorizontal className="h-4 w-4 text-[var(--color-muted-foreground)]" />}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">The configuration</CardTitle>
            <CardDescription>
              The scalar ranges stay the backend&apos;s own honest 422s —
              they are scientific verdicts, not transport errors; the
              pre-flight mirrors only the list bounds the page can see
              coming.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {family === "stat" ? (
              <div className="flex flex-wrap items-center gap-2">
                {(["box", "histogram", "ecdf", "regression"] as const).map((kind) => (
                  <Button
                    key={kind}
                    size="sm"
                    variant={statKind === kind ? "default" : "outline"}
                    onClick={() => setStatKind(kind)}
                  >
                    {kind}
                  </Button>
                ))}
                <div className="mt-3 grid w-full gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {[
                    ["bins (auto or int)", statBins, setStatBins],
                    ["whis (0…10)", statWhis, setStatWhis],
                    ["n_boot (0…2000)", statBoot, setStatBoot],
                    ["seed", statSeed, setStatSeed],
                    ["color (optional)", statColor, setStatColor],
                  ].map(([label, value, setter]) => (
                    <label key={String(label)} className="text-xs">
                      <span className="text-[var(--color-muted-foreground)]">{String(label)}</span>
                      <input
                        value={String(value)}
                        onChange={(event) =>
                          (setter as (v: string) => void)(event.target.value)
                        }
                        className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1 font-mono text-xs"
                      />
                    </label>
                  ))}
                  <div className="flex items-end gap-1">
                    {(["PNG", "SVG"] as const).map((format) => (
                      <Button
                        key={format}
                        size="sm"
                        variant={statFormat === format ? "default" : "outline"}
                        onClick={() => setStatFormat(format)}
                      >
                        {format}
                      </Button>
                    ))}
                  </div>
                </div>
              </div>
            ) : family === "concentration-profile" ? (
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {[
                  ["title", cpTitle, setCpTitle],
                  ["auc (optional)", cpAuc, setCpAuc],
                  ["dpi (72…300)", cpDpi, setCpDpi],
                ].map(([label, value, setter]) => (
                  <label key={String(label)} className="text-xs">
                    <span className="text-[var(--color-muted-foreground)]">{String(label)}</span>
                    <input
                      value={String(value)}
                      onChange={(event) =>
                        (setter as (v: string) => void)(event.target.value)
                      }
                      className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1 font-mono text-xs"
                    />
                  </label>
                ))}
                <div className="flex items-end gap-1">
                  {(["PNG", "SVG", "TIFF", "JPEG"] as const).map((format) => (
                    <Button
                      key={format}
                      size="sm"
                      variant={cpFormat === format ? "default" : "outline"}
                      onClick={() => setCpFormat(format)}
                    >
                      {format}
                    </Button>
                  ))}
                </div>
              </div>
            ) : family === "dose-accumulation" ? (
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {[
                  ["dose (mg)", daDose, setDaDose],
                  ["τ — the dosing interval (h)", daTau, setDaTau],
                  ["t_half (h) — exactly one of t_half/ke", daHalf, setDaHalf],
                  ["ke (1/h) — exactly one of t_half/ke", daKe, setDaKe],
                  ["Vd (L)", daVd, setDaVd],
                  ["F (bioavailability)", daF, setDaF],
                  ["n_doses (1…24)", daDoses, setDaDoses],
                  ["title", daTitle, setDaTitle],
                  ["dpi (72…300)", daDpi, setDaDpi],
                ].map(([label, value, setter]) => (
                  <label key={String(label)} className="text-xs">
                    <span className="text-[var(--color-muted-foreground)]">{String(label)}</span>
                    <input
                      value={String(value)}
                      onChange={(event) =>
                        (setter as (v: string) => void)(event.target.value)
                      }
                      className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1 font-mono text-xs"
                    />
                  </label>
                ))}
                <div className="flex items-end gap-1">
                  {(["PNG", "SVG", "TIFF", "JPEG"] as const).map((format) => (
                    <Button
                      key={format}
                      size="sm"
                      variant={daFormat === format ? "default" : "outline"}
                      onClick={() => setDaFormat(format)}
                    >
                      {format}
                    </Button>
                  ))}
                </div>
              </div>
            ) : family === "md-series" ? (
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {[
                  ["title", msTitle, setMsTitle],
                  ["x label", msXLabel, setMsXLabel],
                  ["y label", msYLabel, setMsYLabel],
                ].map(([label, value, setter]) => (
                  <label key={String(label)} className="text-xs">
                    <span className="text-[var(--color-muted-foreground)]">{String(label)}</span>
                    <input
                      value={String(value)}
                      onChange={(event) =>
                        (setter as (v: string) => void)(event.target.value)
                      }
                      className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1 font-mono text-xs"
                    />
                  </label>
                ))}
                <div className="flex items-end gap-1">
                  {(["PNG", "SVG", "TIFF", "JPEG"] as const).map((format) => (
                    <Button
                      key={format}
                      size="sm"
                      variant={msFormat === format ? "default" : "outline"}
                      onClick={() => setMsFormat(format)}
                    >
                      {format}
                    </Button>
                  ))}
                </div>
              </div>
            ) : family === "table" ? (
              <div className="grid gap-3 sm:grid-cols-2">
                <label className="text-xs">
                  <span className="text-[var(--color-muted-foreground)]">filename</span>
                  <input
                    value={tableFilename}
                    onChange={(event) => setTableFilename(event.target.value)}
                    className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1 font-mono text-xs"
                  />
                </label>
                <div className="flex items-end gap-1">
                  {(["csv", "tsv", "xlsx", "md", "json", "html"] as const).map(
                    (format) => (
                      <Button
                        key={format}
                        size="sm"
                        variant={tableFormat === format ? "default" : "outline"}
                        onClick={() => setTableFormat(format)}
                      >
                        {format}
                      </Button>
                    ),
                  )}
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {[
                    ["title", rpTitle, setRpTitle],
                    ["subtitle (optional)", rpSubtitle, setRpSubtitle],
                  ].map(([label, value, setter]) => (
                    <label key={String(label)} className="text-xs">
                      <span className="text-[var(--color-muted-foreground)]">{String(label)}</span>
                      <input
                        value={String(value)}
                        onChange={(event) =>
                          (setter as (v: string) => void)(event.target.value)
                        }
                        className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1 font-mono text-xs"
                      />
                    </label>
                  ))}
                  <label className="text-xs">
                    <span className="text-[var(--color-muted-foreground)]">
                      theme ({catalog?.report_themes.join(" / ") ?? "auto"})
                    </span>
                    <input
                      value={rpTheme}
                      onChange={(event) => setRpTheme(event.target.value)}
                      className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1 font-mono text-xs"
                    />
                  </label>
                </div>
                <div className="flex flex-wrap gap-2">
                  {(
                    [
                      ["metrics (from the last stat run)", rpMetrics, setRpMetrics],
                      ["tables (the pasted series + values)", rpTables, setRpTables],
                      ["concentration-profile figure", rpCp, setRpCp],
                      ["dose-accumulation figure", rpDa, setRpDa],
                      ["stat figure", rpStat, setRpStat],
                    ] as const
                  ).map(([label, value, setter]) => (
                    <Button
                      key={String(label)}
                      size="sm"
                      variant={value ? "default" : "outline"}
                      onClick={() => (setter as (v: boolean) => void)(!value)}
                    >
                      {String(label)}
                    </Button>
                  ))}
                </div>
                <div className="flex gap-2">
                  {(["job", "sync"] as const).map((mode) => (
                    <Button
                      key={mode}
                      size="sm"
                      variant={rpMode === mode ? "default" : "outline"}
                      onClick={() => setRpMode(mode)}
                    >
                      {mode === "job" ? "Managed job (default)" : "Synchronous"}
                    </Button>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
        {family === "stat" ? (
          <Card className="mt-4">
            <CardHeader>
              <CardTitle className="text-base">Registered dataset column bridge</CardTitle>
              <CardDescription>
                A registered dataset&apos;s own numeric column — the same
                bounded column() bridge every other workspace rides
                (VIZ-1); the pull fills the values textarea verbatim so
                the stat request rides the paste lane&apos;s own
                parser (one code path, two entry points). A non-numeric
                column is the backend&apos;s own 422, surfaced
                verbatim; a null cell is a row-labeled error, never a
                silent zero.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-3 sm:grid-cols-2">
                <select
                  aria-label="the registered dataset"
                  value={bridgeDataset?.dataset_id ?? ""}
                  onChange={(event) => void selectBridgeDataset(event.target.value)}
                  className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm"
                >
                  <option value="">select a registered dataset…</option>
                  {bridgeDatasets.map((entry) => (
                    <option key={entry.dataset_id} value={entry.dataset_id}>
                      {entry.name} · {entry.shape[0]} rows · {entry.columns.length} columns
                    </option>
                  ))}
                </select>
                {bridgeDataset !== null ? (
                  <select
                    aria-label="the dataset column"
                    value={bridgeColumn}
                    onChange={(event) => setBridgeColumn(event.target.value)}
                    className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-3 py-2 text-sm text-[var(--color-foreground)]"
                  >
                    <option value="">select the column…</option>
                    {bridgeDataset.columns.map((column) => (
                      <option key={column} value={column}>
                        {column}
                      </option>
                    ))}
                  </select>
                ) : (
                  <span />
                )}
              </div>
              {bridgeDatasets.length === 0 && bridgeRegistryError === null ? (
                <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                  No datasets registered yet — import one in the Datasets
                  workspace to plot its columns.
                </p>
              ) : null}
              {bridgeRegistryError !== null ? (
                <div className="mt-2">
                  <ErrorBanner message={bridgeRegistryError} />
                </div>
              ) : null}
              <div className="mt-3 flex flex-wrap items-center gap-2">
                <Button
                  disabled={
                    bridgeBusy ||
                    bridgeDataset === null ||
                    bridgeColumn === ""
                  }
                  onClick={() => void handleBridgeColumn()}
                >
                  {bridgeBusy ? "Pulling column…" : "Pull column"}
                </Button>
                <span className="text-xs text-[var(--color-muted-foreground)]">
                  the column rides the bounded bridge (the 500-row cap);
                  the pulled values fill the values textarea below stage 1
                </span>
              </div>
              {bridgeNote !== null ? (
                <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
                  {bridgeNote}
                </p>
              ) : null}
              {bridgeError !== null ? (
                <div className="mt-2">
                  <ErrorBanner message={bridgeError} />
                </div>
              ) : null}
            </CardContent>
          </Card>
        ) : null}
      </StageHeader>

      {/* stage 4: Generate */}
      <StageHeader
        index={4}
        title="Generate"
        hint="post the selected family over the REAL backend"
        icon={<Play className="h-4 w-4 text-[var(--color-muted-foreground)]" />}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Run the lane</CardTitle>
            <CardDescription>
              The report lane defaults to a managed job — the COMPLETED
              job&apos;s REST result is authoritative; the WS
              `file.progress` envelopes drive the per-section progress bar
              only, and cancelling between sections lands the job
              CANCELLED, never FAILED.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-center gap-2">
              <Button onClick={() => void handleGenerate()} disabled={busy}>
                {busy ? "generating…" : "Generate"}
              </Button>
              <span className="text-xs text-[var(--color-muted-foreground)]">
                {family === "stat"
                  ? "POST /api/v1/visualization/stat"
                  : family === "concentration-profile"
                    ? "POST /api/v1/visualization/concentration-profile"
                    : family === "dose-accumulation"
                      ? "POST /api/v1/visualization/dose-accumulation"
                      : family === "md-series"
                        ? "POST /api/v1/visualization/md-series"
                        : family === "table"
                          ? "POST /api/v1/visualization/table"
                          : "POST /api/v1/visualization/report"}
              </span>
            </div>
            {runError ? <div className="mt-3"><ErrorBanner message={runError} /></div> : null}
            {reportJobId !== null ? (
              <div className="mt-3">
                <JobRunCard
                  jobId={reportJobId}
                  resultLabel="the self-contained HTML report"
                  onResult={handleReportResult}
                />
                {reportProgress.length > 0 ? (
                  <p className="mt-2 text-[11px] text-[var(--color-muted-foreground)]">
                    {reportProgress.length} `file.progress` envelopes · latest:{" "}
                    {reportProgress[reportProgress.length - 1]?.stage ?? ""} (
                    {reportProgress[reportProgress.length - 1]?.progress ?? 0}%)
                  </p>
                ) : null}
              </div>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      {/* stage 5: Inspect */}
      <StageHeader
        index={5}
        title="Inspect"
        hint="the stats dict the backend actually computed + the artifact metadata"
        icon={<ClipboardCheck className="h-4 w-4 text-[var(--color-muted-foreground)]" />}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">The result</CardTitle>
            <CardDescription>
              Every number below is the backend&apos;s own verdict — the
              statplots stats dict (type-aware, bounded), the series
              stats, the report&apos;s own stages and figure count.
              Nothing is client-side imputed.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {statResult !== null ? (
              <div className="rounded-md border border-[var(--color-border)] p-3">
                <p className="text-xs font-semibold text-[var(--color-foreground)]">
                  {statResult.kind} stats — {statResult.n_points} points
                </p>
                <div className="mt-2 max-h-64 overflow-auto">
                  <table className="w-full text-[11px]">
                    <tbody>
                      {statsEntries(statResult.stats).map(([key, value]) => (
                        <tr key={key}>
                          <td className="py-0.5 pr-2 align-top text-[var(--color-muted-foreground)]">
                            {key}
                          </td>
                          <td className="py-0.5 text-right">{formatValue(value)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : null}
            {cpResult !== null ? (
              <div className="mt-3 rounded-md border border-[var(--color-border)] p-3">
                <p className="text-xs font-semibold text-[var(--color-foreground)]">
                  concentration profile — {cpResult.n_points} points
                </p>
                <ArtifactRow
                  artifact={cpResult.artifact}
                  onDownload={handleDownload}
                  digest={digests[cpResult.artifact.artifact_id] ?? null}
                />
              </div>
            ) : null}
            {daResult !== null ? (
              <div className="mt-3 rounded-md border border-[var(--color-border)] p-3">
                <p className="text-xs font-semibold text-[var(--color-foreground)]">
                  dose accumulation — τ {formatValue(daResult.tau)} ·{" "}
                  {daResult.n_doses} doses
                </p>
                <ArtifactRow
                  artifact={daResult.artifact}
                  onDownload={handleDownload}
                  digest={digests[daResult.artifact.artifact_id] ?? null}
                />
              </div>
            ) : null}
            {msResult !== null ? (
              <div className="mt-3 rounded-md border border-[var(--color-border)] p-3">
                <p className="text-xs font-semibold text-[var(--color-foreground)]">
                  md-series — {msResult.n_series} series
                </p>
                <table className="mt-2 w-full text-[11px]">
                  <thead>
                    <tr>
                      <th className="py-0.5 pr-2 text-left font-medium">series</th>
                      <th className="py-0.5 pr-2 text-left font-medium">points</th>
                      <th className="py-0.5 pr-2 text-left font-medium">mean</th>
                      <th className="py-0.5 text-left font-medium">cv</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(msResult.series).map(([name, row]) => (
                      <tr key={name}>
                        <td className="py-0.5 pr-2">{name}</td>
                        <td className="py-0.5 pr-2">{row.n_points}</td>
                        <td className="py-0.5 pr-2">{formatValue(row.stats.mean)}</td>
                        <td className="py-0.5">{formatValue(row.stats.cv)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="mt-2">
                  <ArtifactRow
                    artifact={msResult.artifact}
                    onDownload={handleDownload}
                    digest={digests[msResult.artifact.artifact_id] ?? null}
                  />
                </div>
              </div>
            ) : null}
            {tableResult !== null ? (
              <div className="mt-3 rounded-md border border-[var(--color-border)] p-3">
                <p className="text-xs font-semibold text-[var(--color-foreground)]">
                  table — {tableResult.rows} rows
                </p>
                <ArtifactRow
                  artifact={tableResult.artifact}
                  onDownload={handleDownload}
                  digest={digests[tableResult.artifact.artifact_id] ?? null}
                />
              </div>
            ) : null}
            {reportBody !== null ? (
              <div className="mt-3 rounded-md border border-[var(--color-border)] p-3">
                <p className="text-xs font-semibold text-[var(--color-foreground)]">
                  report “{reportBody.title}” · {reportBody.figures} figures
                </p>
                <p className="mt-1 text-[11px] text-[var(--color-muted-foreground)]">
                  stages: {reportBody.stages.join(" · ")}
                </p>
                <div className="mt-2">
                  <ArtifactRow
                    artifact={reportBody.artifact}
                    onDownload={handleDownload}
                    digest={digests[reportBody.artifact.artifact_id] ?? null}
                  />
                </div>
              </div>
            ) : null}
            {statResult === null && cpResult === null && daResult === null &&
            msResult === null && tableResult === null && reportBody === null ? (
              <p className="text-xs text-[var(--color-muted-foreground)]">
                generate a figure first (stage four)
              </p>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      {/* stage 6: Filter/Compare */}
      <StageHeader
        index={6}
        title="Filter/Compare"
        hint="filter the registry by kind; compare the last two generated artifacts"
        icon={<Columns2 className="h-4 w-4 text-[var(--color-muted-foreground)]" />}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Filter &amp; compare</CardTitle>
            <CardDescription>
              The registry filter narrows the session&apos;s artifacts by
              kind; the compare pane is honest — only the artifacts that
              actually arrived, side by side, with the backend&apos;s own
              sha256 promises.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-center gap-2">
              <Button size="sm" variant={kindFilter === "" ? "default" : "outline"} onClick={() => setKindFilter("")}>
                all ({registry?.length ?? 0})
              </Button>
              {registryKinds.map((kind) => (
                <Button
                  key={kind}
                  size="sm"
                  variant={kindFilter === kind ? "default" : "outline"}
                  onClick={() => setKindFilter(kind)}
                >
                  {kind} (
                  {registry?.filter((artifact) => artifact.kind === kind).length ?? 0}
                  )
                </Button>
              ))}
              <Button size="sm" variant="outline" onClick={refreshRegistry}>
                Refresh
              </Button>
            </div>
            {comparePair.length === 2 ? (
              <div className="mt-3 grid gap-3 md:grid-cols-2">
                {comparePair.map((artifact, i) => (
                  <div
                    key={artifact.artifact_id}
                    className="rounded-md border border-[var(--color-border)] p-3"
                  >
                    <p className="text-xs font-semibold text-[var(--color-foreground)]">
                      {i === 0 ? "latest" : "previous"} — {artifact.kind}
                    </p>
                    <table className="mt-2 w-full text-[11px]">
                      <tbody>
                        <tr>
                          <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">filename</td>
                          <td className="py-0.5 text-right">{artifact.filename}</td>
                        </tr>
                        <tr>
                          <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">size</td>
                          <td className="py-0.5 text-right">
                            {(artifact.size_bytes / 1024).toFixed(1)} KiB
                          </td>
                        </tr>
                        <tr>
                          <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">sha256</td>
                          <td className="py-0.5 text-right">
                            {artifact.sha256.slice(0, 16)}…
                          </td>
                        </tr>
                        <tr>
                          <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">created</td>
                          <td className="py-0.5 text-right">
                            {artifact.created_at.slice(0, 19).replace("T", " ")}
                          </td>
                        </tr>
                      </tbody>
                    </table>
                    <div className="mt-2">
                      <Button size="sm" variant="outline" onClick={() => void handleDownload(artifact)}>
                        Download (digest-verified)
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="mt-3 text-xs text-[var(--color-muted-foreground)]">
                {comparePair.length === 1
                  ? "generate one more artifact to compare (two are needed)"
                  : "generate artifacts first — the compare pane needs two"}
              </p>
            )}
          </CardContent>
        </Card>
      </StageHeader>

      {/* stage 7: Export */}
      <StageHeader
        index={7}
        title="Export"
        hint="the digest-verified download lane + the rows-based table export"
        icon={<ArrowDownToLine className="h-4 w-4 text-[var(--color-muted-foreground)]" />}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Export</CardTitle>
            <CardDescription>
              The selected registry artifact downloads through the
              token-header raw lane with the local digest re-computed
              when the context is secure; the table lane carries the
              pasted series as one REAL table artifact (the backend&apos;s
              own 5000-row cap mirrored client-side).
            </CardDescription>
          </CardHeader>
          <CardContent>
            {selectedArtifact !== null ? (
              <ArtifactRow
                artifact={selectedArtifact}
                onDownload={handleDownload}
                digest={selectedDigest}
              />
            ) : (
              <p className="text-xs text-[var(--color-muted-foreground)]">
                select a registry artifact in stage one to export it
              </p>
            )}
            {tableResult !== null ? (
              <div className="mt-3">
                <p className="text-xs text-[var(--color-muted-foreground)]">
                  the last table export: {tableResult.rows} rows ·{" "}
                  {tableResult.artifact.filename}
                </p>
              </div>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>
    </div>
  );
}
