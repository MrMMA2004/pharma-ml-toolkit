import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDownToLine,
  Atom,
  ClipboardCheck,
  ClipboardList,
  LineChart,
  Play,
  ShieldCheck,
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
import { ApiError } from "@/services/api/client";
import {
  artifactsApi,
  type ArtifactView,
} from "@/services/artifacts/client";
import { getEventStream } from "@/services/websocket/client";
import {
  PlotlyFigure,
  type PlotTrace,
} from "@/components/charts/PlotlyFigure";
import {
  assertCoordsBounds,
  assertEnmBounds,
  assertPcaBounds,
  assertSyntheticBounds,
  assertXvgBounds,
  mdApi,
  parseJobProgressPayload,
  parseStudyBody,
  type EnmResponse,
  type JobProgressPayload,
  type MdCatalogResponse,
  type MdSeriesFigureResponse,
  type MetricBundleView,
  type PcaResponse,
  type StudyBodyView,
  type XvgResponse,
} from "@/services/md/client";
import {
  Molecule3D,
  type Molecule3DAtom,
} from "@/components/molecular3d/Molecule3D";
import {
  inferDisplayBonds,
  UNIFORM_COORDINATE_ELEMENT,
} from "@/services/molecular3d/pdbqt";

/** The Phase Eighteen display-heuristic honesty note (required). */
const MD_3D_BOND_NOTE =
  "The inline stack carries positions only — element identities never arrive, so every atom renders honestly uniform; bonds are drawn by a distance heuristic (covalent radii × 1.25), display only and never scientific output; each frame renders only the coordinates that actually arrived.";

/**
 * The Molecular Dynamics Analysis workspace — part11 §11.7's complete
 * sequential workflow:
 *
 * Load Topology/Trajectory → Validate → Select Analysis → Execute →
 * Inspect Results → Visualize → Export
 *
 * Every control below executes the REAL backend through the typed
 * service layer (no fabricated data, no disabled pretend buttons).
 * This is trajectory ANALYSIS — RMSD/RMSF/Rg, native contacts, PCA
 * projections, `.xvg` series, ANM fluctuations — never simulation:
 * no MD engine is bundled or launched (the distinction the spec
 * demands stays visible in the page header). The Load lane builds
 * the seeded deterministic demo trajectory or parses inline
 * `frames x atoms x 3` coordinates and pasted GROMACS `.xvg` text
 * (the route's own `parse_xvg`/validators remain the authority — a
 * malformed stack or series is its honest 422, surfaced verbatim);
 * the Select Analysis stage is the backend's own `GET /md/catalog`
 * inventory (five groups, each row naming the backend functions it
 * wraps — never a client-side guess); Execute runs the selected
 * analysis — the full study lane defaults to job mode (the COMPLETED
 * job's REST result is authoritative; the WS `job.progress`
 * envelopes drive the per-stage progress bar only, and cancelling is
 * native — the job lands CANCELLED, never FAILED); Inspect renders
 * the metric bundle the backend actually computed; the Visualize
 * stage draws the inline curves ONLY from points that actually
 * arrived and registers a REAL figure artifact through the §17
 * four-format export stack; the Export stage rides the rows-based
 * table lane with the backend's own 5000-row cap mirrored
 * client-side.
 */

/** The backend's own table-export row cap (visualization.py's `_MAX_ROWS`). */
const TABLE_EXPORT_MAX_ROWS = 5000;

/** The per-frame preview's own bound (a preview, never the payload). */
const PREVIEW_ROWS = 200;

/** The RMSD summary rows, in the bundle's own key order. */
const METRIC_BUNDLE_KEYS: readonly (keyof Pick<
  MetricBundleView,
  | "n_frames"
  | "n_atoms"
  | "rmsd"
  | "rmsf"
  | "radius_of_gyration"
  | "native_contact_fraction"
>)[] = [
  "n_frames",
  "n_atoms",
  "rmsd",
  "rmsf",
  "radius_of_gyration",
  "native_contact_fraction",
];

type AnalysisKey =
  | "trajectory-metrics"
  | "xvg-stats"
  | "enm"
  | "pca"
  | "study";

type ExportFormat = "csv" | "tsv" | "xlsx" | "md" | "json" | "html";

/**
 * A small demo `.xvg` (the GROMACS plain-text contract the parser
 * honours: `#` comments, `@` directives, two numeric columns) — the
 * xvg lane's own editable sample.
 */
const SAMPLE_XVG = [
  "# GROMACS 2026 — Backbone RMSD after least-squares fit",
  "@ s0 legend \"RMSD (nm)\"",
  "@ xaxis label \"Time (ps)\"",
  "@ yaxis label \"RMSD (nm)\"",
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
 * A small inline coordinate stack (three frames x four atoms, the
 * `x y z` per line, blank line between frames) — the inline lane's
 * own editable sample.
 */
// (r288 fix, caught live by the Phase Eighteen 3D lane's E2E): the
// previous sample put four atoms' 12 numbers on ONE line — malformed
// for the page's own one-triple-per-line parser, so the default
// sample could never validate; the exact grammar below is
// one `x y z` triple per line, blank lines between frames.
const SAMPLE_COORDS = [
  [
    "0.0   0.0   0.0",
    "3.1   0.0   0.0",
    "0.0   3.2   0.0",
    "1.0   1.0   1.0",
  ].join("\n"),
  [
    "0.05  0.02  0.01",
    "3.15  0.03  0.02",
    "0.04  3.18  0.03",
    "1.04  1.03  1.02",
  ].join("\n"),
  [
    "0.11  0.05  0.03",
    "3.22  0.07  0.05",
    "0.09  3.26  0.08",
    "1.09  1.07  1.05",
  ].join("\n"),
].join("\n\n");

/** The ENM lane's own editable sample (ten atoms). */
const SAMPLE_ENM_COORDS = [
  "12.045 1.209 1.542",
  "13.784 1.301 1.871",
  "14.512 2.870 2.109",
  "13.290 3.998 3.417",
  "11.876 4.201 3.088",
  "10.534 3.310 2.470",
  "10.992 2.008 1.922",
  "12.456 0.988 0.876",
  "13.911 1.890 0.655",
  "14.378 3.410 0.901",
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

/** A numbered workflow stage header (§11.7's fixed order). */
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

/** One parsed coordinate triple. */
type Atom = readonly [number, number, number];

/**
 * Parse a pasted `frames x atoms x 3` coordinate stack — one `x y z`
 * triple per line, a blank line between frames. Every malformed row
 * is this parser's own labeled error, never a silent drop; the
 * backend's own shape validator remains the scientific authority.
 */
function parseCoordsStack(
  text: string,
  label: string,
): readonly (readonly Atom[])[] {
  const frames: (readonly Atom[])[] = [];
  let current: Atom[] = [];
  let frameIndex = 0;
  let atomIndex = 0;
  const flush = () => {
    if (current.length > 0) {
      frames.push(current);
      current = [];
      frameIndex += 1;
      atomIndex = 0;
    }
  };
  const lines = text.split("\n").map((line) => line.trim());
  lines.forEach((line, i) => {
    if (line === "") {
      flush();
      return;
    }
    const cells = line.split(/[\s,]+/).filter((cell) => cell !== "");
    if (cells[0]?.startsWith("#")) {
      return; // a comment row
    }
    if (cells.length !== 3) {
      throw new Error(
        `${label} line ${i + 1} (frame ${frameIndex} atom ${atomIndex}): expected exactly 3 cells (x y z), got ${cells.length}`,
      );
    }
    const x = Number(cells[0]);
    const y = Number(cells[1]);
    const z = Number(cells[2]);
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      throw new Error(
        `${label} line ${i + 1}: "${line}" is not a triple of finite numbers`,
      );
    }
    current.push([x, y, z]);
    atomIndex += 1;
  });
  flush();
  if (frames.length < 1) {
    throw new Error(`${label}: no frames parsed, at least 1 required`);
  }
  return frames;
}

/** Parse a pasted `atoms x 3` single-frame coordinate set. */
function parseAtomList(text: string, label: string): readonly Atom[] {
  const stack = parseCoordsStack(text, label);
  if (stack.length !== 1) {
    throw new Error(
      `${label}: ${stack.length} frames parsed, the ENM lane takes exactly one (no blank lines)`,
    );
  }
  return stack[0] ?? [];
}

/**
 * The inline curve — a real Plotly line chart drawn ONLY from the
 * points that actually arrived (no interpolation the data does not
 * own, no fabricated axes: the domain/range labels state the real
 * min/max; §22.2's curve form, §22.4's per-point hover without any
 * mouse-following crosshair).
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
  const traces = useMemo<readonly PlotTrace[]>(
    () => [
      {
        type: "scatter",
        mode: "lines+markers",
        name: yLabel,
        x: points.map((p) => p.x),
        y: points.map((p) => p.y),
        customdata: points.map((p) => [p.x, p.y]),
        hovertemplate: `${xLabel}: %{customdata[0]:.4g}<br>${yLabel}: %{customdata[1]:.4g}<extra></extra>`,
        line: { color: "var(--chart-1)", width: 2 },
        marker: { size: 4 },
        showlegend: false,
      },
    ],
    [points, xLabel, yLabel],
  );
  if (points.length < 2) {
    return (
      <p className="text-xs text-[var(--color-muted-foreground)]">
        the curve draws only from points that actually arrived — two
        finite points is the honest minimum
      </p>
    );
  }
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  return (
    <figure>
      <PlotlyFigure
        data={traces}
        layout={{
          xaxis: { title: { text: xLabel } },
          yaxis: { title: { text: yLabel } },
          margin: { l: 56, r: 12, t: 8, b: 40 },
        }}
        height={height}
        ariaLabel={`${yLabel} versus ${xLabel}`}
      />
      <figcaption className="mt-1 text-[11px] text-[var(--color-muted-foreground)]">
        {points.length} real points · {xLabel}: {formatValue(Math.min(...xs))}…{formatValue(Math.max(...xs))} ·{" "}
        {yLabel}: {formatValue(Math.min(...ys))}…{formatValue(Math.max(...ys))}
      </figcaption>
    </figure>
  );
}

/** Two overlaid Plotly lines (the raw series + its moving average). */
function OverlayCurveFigure({
  primary,
  secondary,
  xLabel,
  yLabel,
  height = 220,
}: {
  primary: readonly { x: number; y: number }[];
  secondary: readonly { x: number; y: number }[] | null;
  xLabel: string;
  yLabel: string;
  height?: number;
}) {
  const traces = useMemo<readonly PlotTrace[]>(
    () => [
      {
        type: "scatter",
        mode: "lines",
        name: "raw series",
        x: primary.map((p) => p.x),
        y: primary.map((p) => p.y),
        customdata: primary.map((p) => [p.x, p.y]),
        hovertemplate: `${xLabel}: %{customdata[0]:.4g}<br>${yLabel}: %{customdata[1]:.4g}<extra>raw</extra>`,
        line: { color: "var(--chart-1)", width: 2 },
        showlegend: secondary !== null,
      },
      ...(secondary !== null
        ? [
            {
              type: "scatter",
              mode: "lines",
              name: "moving average",
              x: secondary.map((p) => p.x),
              y: secondary.map((p) => p.y),
              customdata: secondary.map((p) => [p.x, p.y]),
              hovertemplate: `${xLabel}: %{customdata[0]:.4g}<br>${yLabel}: %{customdata[1]:.4g}<extra>moving average</extra>`,
              line: { color: "var(--chart-3)", width: 1.5, dash: "dash" },
            },
          ]
        : []),
    ],
    [primary, secondary, xLabel, yLabel],
  );
  if (primary.length < 2) {
    return (
      <p className="text-xs text-[var(--color-muted-foreground)]">
        the curve draws only from points that actually arrived — two
        finite points is the honest minimum
      </p>
    );
  }
  return (
    <figure>
      <PlotlyFigure
        data={traces}
        layout={{
          xaxis: { title: { text: xLabel } },
          yaxis: { title: { text: yLabel } },
          margin: { l: 56, r: 12, t: 8, b: 40 },
        }}
        height={height}
        ariaLabel={`${yLabel} versus ${xLabel} with the moving-average overlay`}
      />
      <figcaption className="mt-1 text-[11px] text-[var(--color-muted-foreground)]">
        solid = the raw series · dashed = the backend&apos;s moving average
        {secondary === null ? " (not requested)" : ""}
      </figcaption>
    </figure>
  );
}

/** One registered figure artifact's row + digest-verified download. */
function ArtifactRow({
  artifact,
  onDownload,
  digest,
}: {
  artifact: ArtifactView;
  onDownload: (artifact: ArtifactView) => void;
  digest: string | null;
}) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-[var(--color-border)] p-3">
      <div className="min-w-0">
        <p className="truncate text-sm font-medium text-[var(--color-foreground)]">
          {artifact.filename}
        </p>
        <p className="text-[11px] text-[var(--color-muted-foreground)]">
          {artifact.kind} · {artifact.content_type} ·{" "}
          {(artifact.size_bytes / 1024).toFixed(1)} KiB · sha256{" "}
          {artifact.sha256.slice(0, 16)}…
        </p>
      </div>
      <div className="flex items-center gap-2">
        <Button size="sm" variant="outline" onClick={() => onDownload(artifact)}>
          Download
        </Button>
        {digest !== null ? (
          <span className="text-[11px] text-[var(--color-muted-foreground)]">
            {digest}
          </span>
        ) : null}
      </div>
    </div>
  );
}

export function MDPage() {
  // stage 1: the trajectory source lanes
  const [sourceKey, setSourceKey] = useState<"synthetic" | "inline">(
    "synthetic",
  );
  const [synFrames, setSynFrames] = useState("80");
  const [synSeed, setSynSeed] = useState("7");
  const [synResidues, setSynResidues] = useState("12");
  const [synNoise, setSynNoise] = useState("0.25");
  const [synDrift, setSynDrift] = useState("0.05");
  const [synFlexible, setSynFlexible] = useState("");
  const [coordsText, setCoordsText] = useState(SAMPLE_COORDS);
  const [coordsMasses, setCoordsMasses] = useState("");
  const [xvgText, setXvgText] = useState(SAMPLE_XVG);
  const [xvgWindow, setXvgWindow] = useState("3");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loadedShape, setLoadedShape] = useState<string | null>(null);

  // stage 2: the validated stack (the page's working trajectory)
  const [coordsStack, setCoordsStack] = useState<
    readonly (readonly Atom[])[] | null
  >(null);
  const [coordsMassesValue, setCoordsMassesValue] = useState<
    readonly number[] | null
  >(null);

  // stage 2 — the Phase Eighteen 3D playback lane (Three.js): the
  // validated stack rendered frame by frame; the inline stack
  // carries positions only, so every atom renders honestly uniform
  // (never a guessed element) and the frames render only the
  // coordinates that actually arrived
  const trajectory3d = useMemo(() => {
    if (coordsStack === null || coordsStack.length === 0) {
      return null;
    }
    const firstFrame = coordsStack[0]!;
    const atoms: Molecule3DAtom[] = firstFrame.map((position) => ({
      element: UNIFORM_COORDINATE_ELEMENT,
      x: position[0],
      y: position[1],
      z: position[2],
      label: null,
    }));
    const displayAtoms = firstFrame.map((position, index) => ({
      index,
      atomType: UNIFORM_COORDINATE_ELEMENT,
      element: UNIFORM_COORDINATE_ELEMENT,
      x: position[0],
      y: position[1],
      z: position[2],
      serial: index + 1,
      name: null,
      residue: null,
    }));
    const bonds = inferDisplayBonds(displayAtoms);
    const frames: readonly (readonly [number, number, number])[][] =
      coordsStack.map((frame) =>
        frame.map((position) => [position[0], position[1], position[2]] as const),
      );
    return { atoms, bonds, frames };
  }, [coordsStack]);

  // stage 3: the backend's own catalog
  const [catalog, setCatalog] = useState<MdCatalogResponse | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [analysisKey, setAnalysisKey] = useState<AnalysisKey>(
    "trajectory-metrics",
  );

  // stage 4: the executed lanes
  const [busy, setBusy] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [bundle, setBundle] = useState<MetricBundleView | null>(null);
  const [xvgResult, setXvgResult] = useState<XvgResponse | null>(null);
  const [enmResult, setEnmResult] = useState<EnmResponse | null>(null);
  const [pcaResult, setPcaResult] = useState<PcaResponse | null>(null);
  const [enmCoordsText, setEnmCoordsText] = useState(SAMPLE_ENM_COORDS);
  const [enmCutoff, setEnmCutoff] = useState("13");
  const [enmGamma, setEnmGamma] = useState("1");
  const [pcaComponents, setPcaComponents] = useState("2");
  const [studyMode, setStudyMode] = useState<"job" | "sync">("job");
  const [studyTitle, setStudyTitle] = useState("MD study");
  const [studyJobId, setStudyJobId] = useState<string | null>(null);
  const [studyBody, setStudyBody] = useState<StudyBodyView | null>(null);
  const [studyProgress, setStudyProgress] = useState<
    readonly JobProgressPayload[]
  >([]);

  // stage 6: the md-series figure lane
  const [figureBusy, setFigureBusy] = useState(false);
  const [figureError, setFigureError] = useState<string | null>(null);
  const [mdSeries, setMdSeries] = useState<MdSeriesFigureResponse | null>(null);
  const [figureDigests, setFigureDigests] = useState<
    Record<string, string | null>
  >({});

  // stage 7: the export lane
  const [exportFormat, setExportFormat] = useState<ExportFormat>("csv");
  const [exportBusy, setExportBusy] = useState(false);
  const [exportError, setExportError] = useState<string | null>(null);
  const [exportArtifact, setExportArtifact] = useState<ArtifactView | null>(null);
  const [exportDigest, setExportDigest] = useState<string | null>(null);

  // stage 3: the catalog fetch (once, honestly)
  useEffect(() => {
    let cancelled = false;
    mdApi
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

  // stage 1→2: the load + validate lane
  const handleLoad = useCallback(() => {
    setLoadError(null);
    setLoadedShape(null);
    setCoordsStack(null);
    setCoordsMassesValue(null);
    try {
      if (sourceKey === "inline") {
        const stack = parseCoordsStack(coordsText, "the inline stack");
        const masses = coordsMasses.trim() === ""
          ? null
          : coordsMasses
              .split(/[\s,]+/)
              .filter((cell) => cell !== "")
              .map((cell, i) => {
                const v = Number(cell);
                if (!Number.isFinite(v)) {
                  throw new Error(
                    `the masses row ${i + 1}: "${cell}" is not a finite number`,
                  );
                }
                return v;
              });
        assertCoordsBounds("the inline stack", stack, masses);
        setCoordsStack(stack);
        setCoordsMassesValue(masses);
        const nAtoms = stack[0]?.length ?? 0;
        setLoadedShape(
          `${stack.length} frames × ${nAtoms} atoms × 3 coordinates${
            masses === null ? "" : ` · ${masses.length} masses`
          }`,
        );
      } else {
        const nFrames = Number(synFrames);
        const seed = Number(synSeed);
        const nResidues = Number(synResidues);
        const noise = Number(synNoise);
        const drift = Number(synDrift);
        if (![nFrames, seed, nResidues, noise, drift].every((v) => Number.isFinite(v))) {
          throw new Error(
            "the synthetic parameters must all be finite numbers",
          );
        }
        const flexible = synFlexible.trim() === ""
          ? null
          : synFlexible
              .split(/[\s,]+/)
              .filter((cell) => cell !== "")
              .map((cell, i) => {
                const v = Number(cell);
                if (!Number.isFinite(v)) {
                  throw new Error(
                    `flexible_residues entry ${i + 1}: "${cell}" is not a finite number`,
                  );
                }
                return v;
              });
        assertSyntheticBounds(
          "the synthetic trajectory",
          nFrames,
          nResidues,
          flexible,
        );
        setLoadedShape(
          `the seeded demo trajectory — ${nFrames} frames over ${nResidues} residues (flexible: ${
            flexible === null ? "none" : flexible.join(", ")
          })`,
        );
      }
    } catch (error) {
      setLoadError(describeError(error));
    }
  }, [
    sourceKey,
    coordsText,
    coordsMasses,
    synFrames,
    synSeed,
    synResidues,
    synNoise,
    synDrift,
    synFlexible,
  ]);

  const clearRunResults = useCallback(() => {
    setBundle(null);
    setXvgResult(null);
    setEnmResult(null);
    setPcaResult(null);
    setStudyJobId(null);
    setStudyBody(null);
    setStudyProgress([]);
    setRunError(null);
    setMdSeries(null);
    setFigureDigests({});
  }, []);

  // stage 4: the executed lane's result relay for the study job
  const handleStudyResult = useCallback((result: unknown) => {
    try {
      setStudyBody(parseStudyBody(result, "study job result"));
    } catch (error) {
      setRunError(describeError(error));
    }
  }, []);

  // the study job's live telemetry: the per-stage job.progress
  // envelopes (progress-only — the body comes from the REST relay)
  useEffect(() => {
    if (studyJobId === null) {
      return;
    }
    const stream = getEventStream();
    const unsubscribe = stream.subscribe((envelope) => {
      if (envelope.job_id !== studyJobId) {
        return;
      }
      if (envelope.event_type === "job.progress") {
        try {
          const payload = parseJobProgressPayload(
            envelope.payload,
            "job.progress",
          );
          setStudyProgress((prev) => [...prev, payload]);
        } catch (error) {
          setRunError(describeError(error));
        }
      }
    });
    return unsubscribe;
  }, [studyJobId]);

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

  // stage 4: the execute lane
  const handleExecute = useCallback(async () => {
    clearRunResults();
    setBusy(true);
    try {
      if (analysisKey === "trajectory-metrics") {
        if (sourceKey === "inline") {
          if (coordsStack === null) {
            throw new Error(
              "load and validate the inline stack first (stage two)",
            );
          }
          assertCoordsBounds("analyze", coordsStack, coordsMassesValue);
          setBundle(
            await mdApi.analyze({
              coords: coordsStack,
              masses: coordsMassesValue,
            }),
          );
        } else {
          const nFrames = Number(synFrames);
          const seed = Number(synSeed);
          const nResidues = Number(synResidues);
          const noise = Number(synNoise);
          const drift = Number(synDrift);
          const flexible =
            synFlexible.trim() === ""
              ? null
              : synFlexible
                  .split(/[\s,]+/)
                  .filter((cell) => cell !== "")
                  .map(Number);
          assertSyntheticBounds("synthetic", nFrames, nResidues, flexible);
          setBundle(
            await mdApi.synthetic({
              nFrames,
              seed,
              nResidues,
              noise,
              driftRadians: drift,
              flexibleResidues: flexible,
            }),
          );
        }
      } else if (analysisKey === "xvg-stats") {
        assertXvgBounds("xvg", xvgText, Number(xvgWindow));
        setXvgResult(
          await mdApi.xvg({
            text: xvgText,
            movingAverageWindow: Number(xvgWindow),
          }),
        );
      } else if (analysisKey === "enm") {
        const coords = parseAtomList(enmCoordsText, "the ENM coordinate set");
        const cutoff = Number(enmCutoff);
        const gamma = Number(enmGamma);
        if (!Number.isFinite(cutoff) || !Number.isFinite(gamma)) {
          throw new Error("cutoff and gamma must be finite numbers");
        }
        assertEnmBounds("enm", coords);
        setEnmResult(await mdApi.enm({ coords, cutoff, gamma }));
      } else if (analysisKey === "pca") {
        if (coordsStack === null) {
          throw new Error(
            "load and validate the inline stack first (stage two — the PCA lane takes the inline stack)",
          );
        }
        const components = Number(pcaComponents);
        if (!Number.isInteger(components)) {
          throw new Error("n_components must be an integer");
        }
        assertPcaBounds("pca", coordsStack, components);
        setPcaResult(
          await mdApi.pca({
            coords: coordsStack,
            nComponents: components,
          }),
        );
      } else {
        // the study lane (job default / sync)
        const nFrames = Number(synFrames);
        const seed = Number(synSeed);
        const nResidues = Number(synResidues);
        const noise = Number(synNoise);
        const drift = Number(synDrift);
        if (
          ![nFrames, seed, nResidues, noise, drift].every((v) =>
            Number.isFinite(v),
          )
        ) {
          throw new Error("the study parameters must all be finite numbers");
        }
        if (!Number.isInteger(nFrames) || nFrames < 2) {
          throw new Error("the study lane needs n_frames >= 2");
        }
        if (!Number.isInteger(nResidues) || nResidues < 1 || nResidues > 60) {
          throw new Error(
            "the study lane's n_residues must be an integer in 1…60 (the route's own cap)",
          );
        }
        const response = await mdApi.study({
          nFrames,
          seed,
          nResidues,
          noise,
          driftRadians: drift,
          title: studyTitle,
          mode: studyMode,
        });
        if (response.mode === "sync") {
          setStudyBody(response);
        } else {
          setStudyJobId(response.job.job_id);
        }
      }
    } catch (error) {
      setRunError(describeError(error));
    } finally {
      setBusy(false);
    }
  }, [
    analysisKey,
    clearRunResults,
    coordsMassesValue,
    coordsStack,
    enmCoordsText,
    enmCutoff,
    enmGamma,
    pcaComponents,
    sourceKey,
    studyMode,
    studyTitle,
    synDrift,
    synFlexible,
    synFrames,
    synNoise,
    synResidues,
    synSeed,
    xvgText,
    xvgWindow,
  ]);

  // stage 6: the md-series figure lane (the pasted .xvg text)
  const handleFigure = useCallback(async () => {
    setFigureBusy(true);
    setFigureError(null);
    setMdSeries(null);
    setFigureDigests({});
    try {
      assertXvgBounds("md-series", xvgText, 3);
      const response = await mdApi.mdSeries({
        series: { rmsd: xvgText },
        title: "MD series — RMSD",
        xLabel: "Time (ps)",
        yLabel: "RMSD (nm)",
        format: "PNG",
      });
      setMdSeries(response);
    } catch (error) {
      setFigureError(describeError(error));
    } finally {
      setFigureBusy(false);
    }
  }, [xvgText]);

  const handleFigureDownload = useCallback(
    async (artifact: ArtifactView) => {
      try {
        const digest = await verifiedDownload(artifact);
        setFigureDigests((prev) => ({
          ...prev,
          [artifact.artifact_id]: digest,
        }));
      } catch (error) {
        setFigureError(describeError(error));
      }
    },
    [verifiedDownload],
  );

  const handleStudyArtifactDownload = useCallback(
    async (artifact: ArtifactView) => {
      try {
        const digest = await verifiedDownload(artifact);
        setFigureDigests((prev) => ({
          ...prev,
          [artifact.artifact_id]: digest,
        }));
      } catch (error) {
        setRunError(describeError(error));
      }
    },
    [verifiedDownload],
  );

  // stage 7: the rows the export carries
  const exportRows = useMemo((): readonly Record<string, unknown>[] => {
    if (bundle !== null) {
      return bundle.rmsd.map((rmsd, i) => ({
        frame: i,
        rmsd,
        radius_of_gyration: bundle.radius_of_gyration[i] ?? null,
        native_contact_fraction: bundle.native_contact_fraction[i] ?? null,
      }));
    }
    if (xvgResult !== null) {
      return xvgResult.values.map((value, i) => ({
        point: i,
        time: xvgResult.time[i] ?? null,
        value,
        moving_average: xvgResult.moving_average?.[i] ?? null,
      }));
    }
    if (enmResult !== null) {
      return enmResult.fluctuations.map((value, i) => ({
        atom: i,
        fluctuation: value,
      }));
    }
    if (pcaResult !== null) {
      return pcaResult.projections.map((row, i) => {
        const record: Record<string, unknown> = { frame: i };
        row.forEach((v, c) => {
          record[`pc${c + 1}`] = v;
        });
        return record;
      });
    }
    return [];
  }, [bundle, xvgResult, enmResult, pcaResult]);

  const handleExport = useCallback(async () => {
    setExportBusy(true);
    setExportError(null);
    setExportArtifact(null);
    setExportDigest(null);
    try {
      if (exportRows.length === 0) {
        throw new Error(
          "run an analysis first — the export carries the rows the backend actually computed",
        );
      }
      if (exportRows.length > TABLE_EXPORT_MAX_ROWS) {
        throw new Error(
          `the export carries ${exportRows.length} rows; the backend's own cap is ${TABLE_EXPORT_MAX_ROWS}`,
        );
      }
      const artifact = await mdApi.exportTable({
        rows: exportRows,
        format: exportFormat,
        filename: "md-analysis",
      });
      setExportArtifact(artifact);
      setExportDigest(await verifiedDownload(artifact));
    } catch (error) {
      setExportError(describeError(error));
    } finally {
      setExportBusy(false);
    }
  }, [exportRows, exportFormat, verifiedDownload]);

  // -- the render ------------------------------------------------------

  // the per-frame preview (bounded — a preview, never the payload)
  const previewRows = useMemo(
    () => exportRows.slice(0, PREVIEW_ROWS),
    [exportRows],
  );

  const rmsdCurve = useMemo(
    () =>
      bundle === null
        ? null
        : bundle.rmsd.map((v, i) => ({ x: i, y: v })),
    [bundle],
  );
  const rgCurve = useMemo(
    () =>
      bundle === null
        ? null
        : bundle.radius_of_gyration.map((v, i) => ({ x: i, y: v })),
    [bundle],
  );
  const qCurve = useMemo(
    () =>
      bundle === null
        ? null
        : bundle.native_contact_fraction.map((v, i) => ({ x: i, y: v })),
    [bundle],
  );
  const xvgCurve = useMemo(
    () =>
      xvgResult === null
        ? null
        : xvgResult.values.map((v, i) => ({
            x: xvgResult.time[i] ?? i,
            y: v,
          })),
    [xvgResult],
  );
  const xvgSmoothedCurve = useMemo(
    () =>
      xvgResult === null || xvgResult.moving_average === null
        ? null
        : xvgResult.moving_average.map((v, i) => ({
            x: xvgResult.time[i] ?? i,
            y: v,
          })),
    [xvgResult],
  );

  return (
    <div className="mx-auto max-w-5xl px-4 pb-16">
      <header className="pt-8">
        <h1 className="text-2xl font-semibold text-[var(--color-foreground)]">
          Molecular Dynamics Analysis
        </h1>
        <p className="mt-1 text-sm text-[var(--color-muted-foreground)]">
          Load Topology/Trajectory → Validate → Select Analysis → Execute →
          Inspect Results → Visualize → Export — trajectory ANALYSIS over the
          REAL MD backend: RMSD/RMSF/Rg, native contacts, PCA, `.xvg` series
          and ANM fluctuations. This workspace analyzes trajectories; it does
          not run simulations — no MD engine is bundled or launched.
        </p>
      </header>

      <div className="mt-6">
        {/* part16 §8 (Phase 5): the workspace's demo-data entry point
            (the lysozyme receptor structure + the ligand files the
            analysis lanes inspect; content copies to the paste lanes) */}
        <DemoDataSection workflow="molecular-dynamics" />
      </div>

      {/* stage 1: Load Topology/Trajectory */}
      <StageHeader
        index={1}
        title="Load Topology/Trajectory"
        hint="the seeded demo trajectory, an inline coordinate stack, or pasted GROMACS .xvg text"
        icon={<Atom className="h-4 w-4 text-[var(--color-muted-foreground)]" />}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Trajectory source</CardTitle>
            <CardDescription>
              POST /api/v1/md/synthetic reads the seeded demo trajectory;
              POST /api/v1/md/analyze reads the inline `frames x atoms x 3`
              stack (one x/y/z triple per line, a blank line between frames);
              POST /api/v1/md/xvg reads the pasted `.xvg` text — the
              backend&apos;s own parsers stay the authority.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {(["synthetic", "inline"] as const).map((key) => (
                <Button
                  key={key}
                  size="sm"
                  variant={sourceKey === key ? "default" : "outline"}
                  onClick={() => setSourceKey(key)}
                >
                  {key === "synthetic"
                    ? "Seeded demo trajectory"
                    : "Inline coordinate stack"}
                </Button>
              ))}
            </div>
            {sourceKey === "synthetic" ? (
              <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {[
                  ["frames (1…2000)", synFrames, setSynFrames],
                  ["seed (0…2³¹−1)", synSeed, setSynSeed],
                  ["residues (1…100)", synResidues, setSynResidues],
                  ["noise (0…10)", synNoise, setSynNoise],
                  ["drift radians (0…3)", synDrift, setSynDrift],
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
                    flexible residues (optional, comma/space separated)
                  </span>
                  <input
                    value={synFlexible}
                    onChange={(event) => setSynFlexible(event.target.value)}
                    placeholder="e.g. 4, 5, 6"
                    className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1 font-mono text-xs"
                  />
                </label>
              </div>
            ) : (
              <div className="mt-3">
                <textarea
                  value={coordsText}
                  onChange={(event) => setCoordsText(event.target.value)}
                  rows={7}
                  spellCheck={false}
                  placeholder={"x y z\nx y z\n\n(blank line = next frame)"}
                  className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-2 font-mono text-xs"
                />
                <label className="mt-2 block text-xs">
                  <span className="text-[var(--color-muted-foreground)]">
                    masses (optional, one per atom, space/comma separated)
                  </span>
                  <input
                    value={coordsMasses}
                    onChange={(event) => setCoordsMasses(event.target.value)}
                    placeholder="12.0 12.0 14.0 14.0 …"
                    className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1 font-mono text-xs"
                  />
                </label>
              </div>
            )}
            <div className="mt-3">
              <p className="text-xs font-semibold text-[var(--color-foreground)]">
                The `.xvg` series lane (stages four and six both read it)
              </p>
              <textarea
                value={xvgText}
                onChange={(event) => setXvgText(event.target.value)}
                rows={7}
                spellCheck={false}
                placeholder={"# GROMACS\n@ s0 legend \"RMSD\"\n0.0 0.05\n…"}
                className="mt-1 w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-2 font-mono text-xs"
              />
              <label className="mt-2 block text-xs">
                <span className="text-[var(--color-muted-foreground)]">
                  moving-average window (1…101)
                </span>
                <input
                  value={xvgWindow}
                  onChange={(event) => setXvgWindow(event.target.value)}
                  className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1 font-mono text-xs"
                />
              </label>
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <Button onClick={handleLoad}>Load &amp; validate</Button>
              <Button
                variant="outline"
                onClick={() => {
                  setCoordsText(SAMPLE_COORDS);
                  setXvgText(SAMPLE_XVG);
                  setEnmCoordsText(SAMPLE_ENM_COORDS);
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

      {/* stage 2: Validate */}
      <StageHeader
        index={2}
        title="Validate"
        hint="the pre-flight mirrors the route's own bounds; the backend stays the authority"
        icon={<ShieldCheck className="h-4 w-4 text-[var(--color-muted-foreground)]" />}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Shape &amp; bounds</CardTitle>
            <CardDescription>
              The client-side pre-flight mirrors the route&apos;s own list
              bounds (frames 1…2000, atoms ≤ 300, elements ≤ 240000, the
              `.xvg` text ≤ 2,000,000 chars); the scalar value ranges stay
              the backend&apos;s own honest 422s — they are scientific
              verdicts, not transport errors.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {coordsStack !== null ? (
              <p className="text-xs text-[var(--color-muted-foreground)]">
                the inline stack validates: {coordsStack.length} frames ×{" "}
                {coordsStack[0]?.length ?? 0} atoms × 3 — the analyze lane
                is ready
              </p>
            ) : (
              <p className="text-xs text-[var(--color-muted-foreground)]">
                {loadedShape === null
                  ? "load a trajectory source first (stage one)"
                  : `the ${loadedShape}`}
              </p>
            )}
          </CardContent>
        </Card>

        {/* Phase Eighteen — the genuine 3D trajectory lane (Three.js):
            the validated stack played frame by frame; this is an
            ANALYSIS view over the coordinates that actually arrived
            (the workspace never runs simulations — no MD engine is
            bundled or launched). */}
        {trajectory3d !== null ? (
          <Molecule3D
            title="Trajectory playback (3D)"
            description="The validated inline stack rendered frame by frame — play or step through the frames that actually arrived; click an atom to inspect its position in the current frame."
            atoms={trajectory3d.atoms}
            bonds={trajectory3d.bonds}
            frames={trajectory3d.frames}
            bondNote={MD_3D_BOND_NOTE}
            height={320}
          />
        ) : (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Trajectory playback (3D)</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-[var(--color-muted-foreground)]">
              <p>
                The 3D playback renders the validated stack — load a
                trajectory source in stage one first (the seeded demo, an
                inline coordinate stack, or pasted `.xvg` text). The view
                draws only frames that actually arrived.
              </p>
            </CardContent>
          </Card>
        )}
      </StageHeader>

      {/* stage 3: Select Analysis */}
      <StageHeader
        index={3}
        title="Select Analysis"
        hint="the backend's own GET /api/v1/md/catalog inventory (no client-side guess)"
        icon={<ClipboardList className="h-4 w-4 text-[var(--color-muted-foreground)]" />}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Analysis catalog</CardTitle>
            <CardDescription>
              GET /api/v1/md/catalog — five groups, each row naming the
              backend functions it wraps; the selection below only picks
              which of THOSE routes the Execute stage calls.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {catalogError ? <ErrorBanner message={catalogError} /> : null}
            {catalog !== null ? (
              <div className="overflow-auto rounded-md border border-[var(--color-border)]">
                <table className="w-full border-collapse text-[11px]">
                  <thead className="bg-[var(--color-muted)]">
                    <tr>
                      <th className="px-2 py-1 text-left font-medium">group</th>
                      <th className="px-2 py-1 text-left font-medium">routes</th>
                      <th className="px-2 py-1 text-left font-medium">backend</th>
                    </tr>
                  </thead>
                  <tbody>
                    {catalog.groups.map((group) => (
                      <tr key={group.group} className="odd:bg-[var(--color-muted)]/40">
                        <td className="px-2 py-0.5">{group.group}</td>
                        <td className="px-2 py-0.5">{group.routes.join(", ")}</td>
                        <td className="px-2 py-0.5 text-[var(--color-muted-foreground)]">
                          {group.backend}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-xs text-[var(--color-muted-foreground)]">
                loading the catalog…
              </p>
            )}
            <div className="mt-3 flex flex-wrap gap-2">
              {(
                [
                  ["trajectory-metrics", "Trajectory metrics (RMSD/RMSF/Rg/Q/PCA)"],
                  ["xvg-stats", "`.xvg` series stats + moving average"],
                  ["enm", "Elastic-network ANM fluctuations"],
                  ["pca", "Standalone trajectory PCA"],
                  ["study", "Full study job (metrics + 2 figure artifacts)"],
                ] as const
              ).map(([key, label]) => (
                <Button
                  key={key}
                  size="sm"
                  variant={analysisKey === key ? "default" : "outline"}
                  onClick={() => setAnalysisKey(key)}
                >
                  {label}
                </Button>
              ))}
            </div>
            {analysisKey === "enm" ? (
              <div className="mt-3">
                <textarea
                  value={enmCoordsText}
                  onChange={(event) => setEnmCoordsText(event.target.value)}
                  rows={6}
                  spellCheck={false}
                  placeholder={"one x y z triple per line (a single frame)"}
                  className="w-full rounded-md border border-[var(--color-border)] bg-[var(--color-background)] p-2 font-mono text-xs"
                />
                <div className="mt-2 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {[
                    ["cutoff Å (0…100)", enmCutoff, setEnmCutoff],
                    ["gamma (0…1e6)", enmGamma, setEnmGamma],
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
                </div>
              </div>
            ) : null}
            {analysisKey === "pca" ? (
              <label className="mt-3 block text-xs">
                <span className="text-[var(--color-muted-foreground)]">
                  components (1…10) — over the inline stack
                </span>
                <input
                  value={pcaComponents}
                  onChange={(event) => setPcaComponents(event.target.value)}
                  className="mt-1 w-24 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1 font-mono text-xs"
                />
              </label>
            ) : null}
            {analysisKey === "study" ? (
              <div className="mt-3 flex flex-wrap items-end gap-3">
                <label className="text-xs">
                  <span className="text-[var(--color-muted-foreground)]">title</span>
                  <input
                    value={studyTitle}
                    onChange={(event) => setStudyTitle(event.target.value)}
                    className="mt-1 w-48 rounded-md border border-[var(--color-border)] bg-[var(--color-background)] px-2 py-1 font-mono text-xs"
                  />
                </label>
                <div className="flex gap-2">
                  {(["job", "sync"] as const).map((mode) => (
                    <Button
                      key={mode}
                      size="sm"
                      variant={studyMode === mode ? "default" : "outline"}
                      onClick={() => setStudyMode(mode)}
                    >
                      {mode === "job"
                        ? "Managed job (default)"
                        : "Synchronous"}
                    </Button>
                  ))}
                </div>
              </div>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      {/* stage 4: Execute */}
      <StageHeader
        index={4}
        title="Execute"
        hint="run the selected analysis over the REAL backend"
        icon={<Play className="h-4 w-4 text-[var(--color-muted-foreground)]" />}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Run the analysis</CardTitle>
            <CardDescription>
              The study lane defaults to a managed job — the COMPLETED job&apos;s
              REST result is authoritative; the WS `job.progress` envelopes
              drive the progress bar only, and cancelling between stages
              lands the job CANCELLED, never FAILED.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-center gap-2">
              <Button onClick={() => void handleExecute()} disabled={busy}>
                {busy ? "running…" : "Execute analysis"}
              </Button>
              <span className="text-xs text-[var(--color-muted-foreground)]">
                {analysisKey === "trajectory-metrics"
                  ? sourceKey === "inline"
                    ? "POST /api/v1/md/analyze over the validated inline stack"
                    : "POST /api/v1/md/synthetic over the seeded demo trajectory"
                  : analysisKey === "xvg-stats"
                    ? "POST /api/v1/md/xvg over the pasted .xvg text"
                    : analysisKey === "enm"
                      ? "POST /api/v1/md/enm over the pasted coordinate set"
                      : analysisKey === "pca"
                        ? "POST /api/v1/md/pca over the validated inline stack"
                        : "POST /api/v1/md/study — the full managed bundle"}
              </span>
            </div>
            {runError ? <div className="mt-3"><ErrorBanner message={runError} /></div> : null}
            {studyJobId !== null ? (
              <div className="mt-3">
                <JobRunCard
                  jobId={studyJobId}
                  resultLabel="the full study bundle"
                  onResult={handleStudyResult}
                />
                {studyProgress.length > 0 ? (
                  <p className="mt-2 text-[11px] text-[var(--color-muted-foreground)]">
                    {studyProgress.length} `job.progress` envelopes · latest:{" "}
                    {studyProgress[studyProgress.length - 1]?.stage ?? ""} (
                    {studyProgress[studyProgress.length - 1]?.progress ?? 0}%)
                  </p>
                ) : null}
              </div>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      {/* stage 5: Inspect Results */}
      <StageHeader
        index={5}
        title="Inspect Results"
        hint="the metric bundle the backend actually computed"
        icon={<ClipboardCheck className="h-4 w-4 text-[var(--color-muted-foreground)]" />}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Analysis results</CardTitle>
            <CardDescription>
              Every number below is the backend&apos;s own verdict — RMSD /
              RMSF / Rg / native-contact series, the contact-map occupancy
              summary, the PCA explained variance; the study lane adds its
              own summary block. Nothing is client-side imputed.
            </CardDescription>
          </CardHeader>
          <CardContent>
            {bundle !== null ? (
              <div className="grid gap-3 md:grid-cols-2">
                <div className="rounded-md border border-[var(--color-border)] p-3">
                  <p className="text-xs font-semibold text-[var(--color-foreground)]">
                    metric bundle summary ({bundle.source} lane)
                  </p>
                  <table className="mt-2 w-full text-[11px]">
                    <tbody>
                      {METRIC_BUNDLE_KEYS.map((key) => {
                        const value = bundle[key];
                        const text =
                          typeof value === "number"
                            ? formatValue(value)
                            : `${(value as readonly number[]).length} entries · max ${formatValue(
                                Math.max(...(value as readonly number[])),
                              )}`;
                        return (
                          <tr key={key}>
                            <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">{key}</td>
                            <td className="py-0.5 text-right">{text}</td>
                          </tr>
                        );
                      })}
                      <tr>
                        <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">
                          rmsd first / last
                        </td>
                        <td className="py-0.5 text-right">
                          {formatValue(bundle.rmsd[0])} /{" "}
                          {formatValue(bundle.rmsd[bundle.rmsd.length - 1])}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">
                          radius_of_gyration first / last
                        </td>
                        <td className="py-0.5 text-right">
                          {formatValue(bundle.radius_of_gyration[0])} /{" "}
                          {formatValue(
                            bundle.radius_of_gyration[
                              bundle.radius_of_gyration.length - 1
                            ],
                          )}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">
                          native_contact_fraction last
                        </td>
                        <td className="py-0.5 text-right">
                          {formatValue(
                            bundle.native_contact_fraction[
                              bundle.native_contact_fraction.length - 1
                            ],
                          )}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <div className="rounded-md border border-[var(--color-border)] p-3">
                  <p className="text-xs font-semibold text-[var(--color-foreground)]">
                    contact map &amp; PCA
                  </p>
                  <table className="mt-2 w-full text-[11px]">
                    <tbody>
                      <tr>
                        <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">
                          contact_map.n_pairs
                        </td>
                        <td className="py-0.5 text-right">
                          {formatValue(bundle.contact_map.n_pairs)}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">
                          contact_map.strong_pairs
                        </td>
                        <td className="py-0.5 text-right">
                          {formatValue(bundle.contact_map.strong_pairs)}
                        </td>
                      </tr>
                      <tr>
                        <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">
                          contact_map.strong_pair_fraction
                        </td>
                        <td className="py-0.5 text-right">
                          {formatValue(bundle.contact_map.strong_pair_fraction)}
                        </td>
                      </tr>
                      {bundle.pca.explained_variance_ratio.map((v, i) => (
                        <tr key={`pc${i}`}>
                          <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">
                            pca.explained_variance_ratio[{i}]
                          </td>
                          <td className="py-0.5 text-right">{formatValue(v)}</td>
                        </tr>
                      ))}
                      <tr>
                        <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">
                          pca.method
                        </td>
                        <td className="py-0.5 text-right">{bundle.pca.method}</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            ) : null}
            {xvgResult !== null ? (
              <div className="rounded-md border border-[var(--color-border)] p-3">
                <p className="text-xs font-semibold text-[var(--color-foreground)]">
                  `.xvg` series — {xvgResult.label || xvgResult.source} ·{" "}
                  {xvgResult.n_points} points
                </p>
                <table className="mt-2 w-full text-[11px]">
                  <tbody>
                    <tr>
                      <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">mean</td>
                      <td className="py-0.5 text-right">{formatValue(xvgResult.stats.mean)}</td>
                    </tr>
                    <tr>
                      <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">std</td>
                      <td className="py-0.5 text-right">{formatValue(xvgResult.stats.std)}</td>
                    </tr>
                    <tr>
                      <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">cv</td>
                      <td className="py-0.5 text-right">{formatValue(xvgResult.stats.cv)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            ) : null}
            {enmResult !== null ? (
              <div className="mt-3 rounded-md border border-[var(--color-border)] p-3">
                <p className="text-xs font-semibold text-[var(--color-foreground)]">
                  ANM fluctuations — {enmResult.n_atoms} atoms · cutoff{" "}
                  {formatValue(enmResult.cutoff)} Å · gamma{" "}
                  {formatValue(enmResult.gamma)}
                </p>
                <div className="mt-2 max-h-48 overflow-auto rounded-md border border-[var(--color-border)]">
                  <table className="w-full border-collapse text-[11px]">
                    <thead className="sticky top-0 bg-[var(--color-muted)]">
                      <tr>
                        <th className="px-2 py-1 text-left font-medium">atom</th>
                        <th className="px-2 py-1 text-left font-medium">fluctuation</th>
                      </tr>
                    </thead>
                    <tbody>
                      {enmResult.fluctuations.map((v, i) => (
                        <tr key={i} className="odd:bg-[var(--color-muted)]/40">
                          <td className="px-2 py-0.5">{i}</td>
                          <td className="px-2 py-0.5">{formatValue(v)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : null}
            {pcaResult !== null ? (
              <div className="mt-3 rounded-md border border-[var(--color-border)] p-3">
                <p className="text-xs font-semibold text-[var(--color-foreground)]">
                  trajectory PCA — {pcaResult.method} · {pcaResult.n_frames}{" "}
                  frames × {pcaResult.n_components} components
                </p>
                <div className="mt-2 max-h-48 overflow-auto rounded-md border border-[var(--color-border)]">
                  <table className="w-full border-collapse text-[11px]">
                    <thead className="sticky top-0 bg-[var(--color-muted)]">
                      <tr>
                        <th className="px-2 py-1 text-left font-medium">frame</th>
                        {pcaResult.explained_variance_ratio.map((_, c) => (
                          <th key={c} className="px-2 py-1 text-left font-medium">
                            pc{c + 1}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {pcaResult.projections.slice(0, PREVIEW_ROWS).map((row, i) => (
                        <tr key={i} className="odd:bg-[var(--color-muted)]/40">
                          <td className="px-2 py-0.5">{i}</td>
                          {row.map((v, c) => (
                            <td key={c} className="px-2 py-0.5">{formatValue(v)}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="mt-1 text-[11px] text-[var(--color-muted-foreground)]">
                  {pcaResult.projections.length} rows total · the first{" "}
                  {PREVIEW_ROWS} shown (the export carries every row)
                </p>
              </div>
            ) : null}
            {studyBody !== null ? (
              <div className="mt-3 rounded-md border border-[var(--color-border)] p-3">
                <p className="text-xs font-semibold text-[var(--color-foreground)]">
                  study “{studyBody.title}” · seed {studyBody.seed}
                </p>
                <table className="mt-2 w-full text-[11px]">
                  <tbody>
                    <tr>
                      <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">n_frames</td>
                      <td className="py-0.5 text-right">{formatValue(studyBody.metrics.n_frames)}</td>
                    </tr>
                    <tr>
                      <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">n_atoms</td>
                      <td className="py-0.5 text-right">{formatValue(studyBody.metrics.n_atoms)}</td>
                    </tr>
                    <tr>
                      <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">rmsd first / last</td>
                      <td className="py-0.5 text-right">
                        {formatValue(studyBody.metrics.rmsd_first)} /{" "}
                        {formatValue(studyBody.metrics.rmsd_last)}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">rmsf_max</td>
                      <td className="py-0.5 text-right">{formatValue(studyBody.metrics.rmsf_max)}</td>
                    </tr>
                    <tr>
                      <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">rg first / last</td>
                      <td className="py-0.5 text-right">
                        {formatValue(studyBody.metrics.rg_first)} /{" "}
                        {formatValue(studyBody.metrics.rg_last)}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">
                        native_contact_fraction_last
                      </td>
                      <td className="py-0.5 text-right">
                        {formatValue(studyBody.metrics.native_contact_fraction_last)}
                      </td>
                    </tr>
                    {studyBody.metrics.pca_explained_variance_ratio.map((v, i) => (
                      <tr key={`spc${i}`}>
                        <td className="py-0.5 pr-2 text-[var(--color-muted-foreground)]">
                          pca_explained_variance_ratio[{i}]
                        </td>
                        <td className="py-0.5 text-right">{formatValue(v)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
            {bundle === null && xvgResult === null && enmResult === null &&
            pcaResult === null && studyBody === null ? (
              <p className="text-xs text-[var(--color-muted-foreground)]">
                run an analysis first (stage four)
              </p>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      {/* stage 6: Visualize */}
      <StageHeader
        index={6}
        title="Visualize"
        hint="inline curves from the points that arrived + REAL figure artifacts"
        icon={<LineChart className="h-4 w-4 text-[var(--color-muted-foreground)]" />}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Curves &amp; figures</CardTitle>
            <CardDescription>
              The inline curves draw ONLY from the points the backend
              returned; the md-series lane registers a REAL figure artifact
              through `POST /api/v1/visualization/md-series` (the §17
              four-format export stack), and the study lane&apos;s own
              artifacts arrive with the COMPLETED job — every download is
              digest-verified.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-2">
              {rmsdCurve !== null ? (
                <div className="rounded-md border border-[var(--color-border)] p-3">
                  <p className="text-xs font-semibold text-[var(--color-foreground)]">
                    RMSD per frame (Å)
                  </p>
                  <CurveFigure points={rmsdCurve} xLabel="frame" yLabel="rmsd" />
                </div>
              ) : null}
              {rgCurve !== null ? (
                <div className="rounded-md border border-[var(--color-border)] p-3">
                  <p className="text-xs font-semibold text-[var(--color-foreground)]">
                    Radius of gyration per frame (Å)
                  </p>
                  <CurveFigure points={rgCurve} xLabel="frame" yLabel="rg" />
                </div>
              ) : null}
              {qCurve !== null ? (
                <div className="rounded-md border border-[var(--color-border)] p-3">
                  <p className="text-xs font-semibold text-[var(--color-foreground)]">
                    Native contact fraction per frame
                  </p>
                  <CurveFigure points={qCurve} xLabel="frame" yLabel="q" />
                </div>
              ) : null}
              {xvgCurve !== null ? (
                <div className="rounded-md border border-[var(--color-border)] p-3">
                  <p className="text-xs font-semibold text-[var(--color-foreground)]">
                    `.xvg` series + moving average
                  </p>
                  <OverlayCurveFigure
                    primary={xvgCurve}
                    secondary={xvgSmoothedCurve}
                    xLabel="time"
                    yLabel="value"
                  />
                </div>
              ) : null}
            </div>
            <div className="mt-4 flex flex-wrap items-center gap-2">
              <Button onClick={() => void handleFigure()} disabled={figureBusy}>
                {figureBusy ? "rendering…" : "Register md-series figure artifact"}
              </Button>
              <span className="text-xs text-[var(--color-muted-foreground)]">
                POST /api/v1/visualization/md-series over the pasted `.xvg`
                text (PNG, dpi 150)
              </span>
            </div>
            {figureError ? <div className="mt-3"><ErrorBanner message={figureError} /></div> : null}
            {mdSeries !== null ? (
              <div className="mt-3">
                <ArtifactRow
                  artifact={mdSeries.artifact}
                  onDownload={handleFigureDownload}
                  digest={figureDigests[mdSeries.artifact.artifact_id] ?? null}
                />
                <p className="mt-1 text-[11px] text-[var(--color-muted-foreground)]">
                  {mdSeries.n_series} series ·{" "}
                  {Object.entries(mdSeries.series)
                    .map(
                      ([name, row]) =>
                        `${name}: ${row.n_points} pts (mean ${formatValue(row.stats.mean)})`,
                    )
                    .join(" · ")}
                </p>
              </div>
            ) : null}
            {studyBody !== null && studyBody.artifacts.length > 0 ? (
              <div className="mt-3 space-y-2">
                <p className="text-xs font-semibold text-[var(--color-foreground)]">
                  the study job&apos;s figure artifacts
                </p>
                {studyBody.artifacts.map((artifact) => (
                  <ArtifactRow
                    key={artifact.artifact_id}
                    artifact={artifact}
                    onDownload={handleStudyArtifactDownload}
                    digest={figureDigests[artifact.artifact_id] ?? null}
                  />
                ))}
              </div>
            ) : null}
          </CardContent>
        </Card>
      </StageHeader>

      {/* stage 7: Export */}
      <StageHeader
        index={7}
        title="Export"
        hint="the rows the backend computed, through the rows-based table lane"
        icon={<ArrowDownToLine className="h-4 w-4 text-[var(--color-muted-foreground)]" />}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Table export</CardTitle>
            <CardDescription>
              POST /api/v1/visualization/table — the analysis rows (per-frame
              metrics, `.xvg` points, ENM fluctuations or PCA projections)
              as one downloadable, digest-verified artifact; the backend&apos;s
              own 5000-row cap is mirrored client-side.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-center gap-2">
              <Button onClick={() => void handleExport()} disabled={exportBusy}>
                {exportBusy ? "exporting…" : "Export analysis rows"}
              </Button>
              <div className="flex gap-1">
                {(["csv", "tsv", "xlsx", "md", "json", "html"] as const).map(
                  (format) => (
                    <Button
                      key={format}
                      size="sm"
                      variant={exportFormat === format ? "default" : "outline"}
                      onClick={() => setExportFormat(format)}
                    >
                      {format}
                    </Button>
                  ),
                )}
              </div>
            </div>
            {exportError ? <div className="mt-3"><ErrorBanner message={exportError} /></div> : null}
            {exportArtifact !== null ? (
              <div className="mt-3">
                <ArtifactRow
                  artifact={exportArtifact}
                  onDownload={handleStudyArtifactDownload}
                  digest={exportDigest}
                />
              </div>
            ) : null}
            {previewRows.length > 0 ? (
              <div className="mt-3">
                <p className="text-xs text-[var(--color-muted-foreground)]">
                  {exportRows.length} rows carry (the first {PREVIEW_ROWS}{" "}
                  shown; the export lane carries every row)
                </p>
                <div className="mt-2 max-h-64 overflow-auto rounded-md border border-[var(--color-border)]">
                  <table className="w-full border-collapse text-[11px]">
                    <thead className="sticky top-0 bg-[var(--color-muted)]">
                      <tr>
                        {unionRowKeys(previewRows).map((key) => (
                          <th key={key} className="px-2 py-1 text-left font-medium">
                            {key}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {previewRows.map((row, i) => (
                        <tr key={i} className="odd:bg-[var(--color-muted)]/40">
                          {unionRowKeys(previewRows).map((key) => (
                            <td key={key} className="px-2 py-0.5">
                              {formatValue(row[key])}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <p className="mt-3 text-xs text-[var(--color-muted-foreground)]">
                run an analysis first — the export carries the rows the
                backend actually computed
              </p>
            )}
          </CardContent>
        </Card>
      </StageHeader>
    </div>
  );
}
