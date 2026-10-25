/**
 * Session report collector (100x-127) — the web twin of the desktop
 * playground's `File ▸ Save session report (HTML)…`.
 *
 * The desktop report walks its live session (every figure on screen +
 * every filled table) into pharmaml.viz.HTMLReport.  The web twin's
 * "session" is the persisted playground state: the last ML training
 * result (metrics + chart series), the last learning-curve run, the
 * PK/PD regimen (always live — the regimen card always renders), the
 * NCA observed-data rows and the ML run history.  This module is the
 * PURE collector: it turns that state into the section payload the
 * `/api/report/session` route forwards to the Python bridge, which
 * composes real matplotlib figures and the SAME self-contained themed
 * HTMLReport the desktop saves (theme toggle included).
 *
 * Trace contract (mirrors what the web charts draw):
 *  - `line`   → solid polyline (model curve / PK profile)
 *  - `band`   → mean line + ±1 std fill (learning curve, GP band)
 *  - `scatter` with `classes` → class-coloured dots (classification
 *    test points / boundary grid), `probas` fades opacity by
 *    |p − 0.5| exactly the way the web boundary dots do
 */

import { concentrationAt, derivePk, pdEffect } from "@/lib/pk-pd";
import type {
  MLLearningCurveResponse,
  MLTrainResponse,
  SurgeryResponse,
} from "@/lib/toolkit-types";
import type {
  MlRunRecord,
  NcaState,
  PdParams,
  PkParams,
} from "@/lib/store";

/* ------------------------------------------------------------------ */
/* Payload types — the bridge contract                                 */
/* ------------------------------------------------------------------ */

export type ReportTheme = "auto" | "light" | "dark";

/** One drawable series inside a figure section. */
export interface ReportFigureTrace {
  name: string;
  style: "line" | "band" | "scatter";
  /**
   * `[x, y]` pairs; `band` traces carry `[x, y, lo, hi]` (mean ± band
   * edges).  Numbers stay raw — the bridge's matplotlib does the
   * formatting, exactly like the desktop figures.
   */
  points: Array<[number, number] | [number, number, number, number]>;
  /** Per-point class (0/1) for classification scatters — color split. */
  classes?: number[];
  /** Per-point class-1 probability — opacity = |p − 0.5| × 2. */
  probas?: number[];
}

export interface ReportFigureSection {
  kind: "figure";
  title: string;
  xLabel: string;
  yLabel: string;
  traces: ReportFigureTrace[];
}

export interface ReportTableSection {
  kind: "table";
  title: string;
  columns: string[];
  rows: Array<Array<string | number>>;
}

/** (100x-131) A pre-rasterised PNG section — the surgery structure
 * grids arrive from the server bridge as data URIs; the report bridge
 * decodes them onto a full-bleed matplotlib figure and embeds it through
 * the same `add_figure` contract every other figure rides. */
export interface ReportImageSection {
  kind: "image";
  title: string;
  dataUri: string;
}

export interface ReportMetricsSection {
  kind: "metrics";
  title: string;
  /** Metric cards — string values (the bridge escapes them). */
  items: Array<[string, string]>;
}

export type ReportSection =
  | ReportFigureSection
  | ReportTableSection
  | ReportMetricsSection
  | ReportImageSection;

/** POST /api/report/session request body. */
export interface SessionReportRequest {
  theme: ReportTheme;
  subtitle: string;
  sections: ReportSection[];
}

/** POST /api/report/session response (the bridge's census). */
export interface SessionReportResponse {
  html: string;
  figures: number;
  tables: number;
  bytes: number;
  theme: string;
}

/** The persisted playground state the collector walks. */
export interface SessionSnapshot {
  mlResult: MLTrainResponse | null;
  lcResult: MLLearningCurveResponse | null;
  mlHistory: MlRunRecord[];
  pkParams: PkParams;
  pdParams: PdParams;
  ncaState: NcaState;
  /** (100x-131) the last molecule-surgery run (topology or breed — the
   *  latest run wins, exactly like mlResult); null = no surgery yet. */
  surgery: SurgeryResponse | null;
}

/** (100x-131) Per-group section control — the report card's toggles.
 *  Every group defaults to shipped; the collector silently skips a
 *  group the snapshot has no content for either way, so the toggle set
 *  the UI renders comes from `sessionGroupsAvailable`. */
export interface SessionReportOptions {
  ml?: boolean;
  learningCurve?: boolean;
  pkpd?: boolean;
  nca?: boolean;
  history?: boolean;
  surgery?: boolean;
}

/** One report group's id (the toggle + reorder axis). */
export type SessionGroupId = keyof SessionReportOptions;

/** (100x-132) The canonical walking order — models before curves,
 *  regimen before data tables, surgery last.  The report card's chips
 *  can drag into any permutation; `sanitizeSessionOrder` guards what
 *  the collector will actually accept. */
export const SESSION_GROUP_ORDER: SessionGroupId[] = [
  "ml",
  "learningCurve",
  "pkpd",
  "nca",
  "history",
  "surgery",
];

/** (100x-132) Normalize a walking order: unknown ids dropped,
 *  duplicates dropped (first occurrence wins), missing groups appended
 *  in canonical order at the END — an order stored before a new group
 *  existed still ships that group, never silently drops it. */
export function sanitizeSessionOrder(
  order: SessionGroupId[] | undefined,
): SessionGroupId[] {
  if (!order || order.length === 0) return [...SESSION_GROUP_ORDER];
  const seen = new Set<SessionGroupId>();
  const known: SessionGroupId[] = [];
  for (const id of order) {
    if (SESSION_GROUP_ORDER.includes(id) && !seen.has(id)) {
      seen.add(id);
      known.push(id);
    }
  }
  for (const id of SESSION_GROUP_ORDER) {
    if (!seen.has(id)) {
      seen.add(id);
      known.push(id);
    }
  }
  return known;
}

/** Which groups would actually ship something for this snapshot — the
 * honest source for the UI: a group with no content renders no toggle
 * (nothing to skip), so a fresh session shows only "PK / PD" and the
 * buttons appear as the session fills. */
export function sessionGroupsAvailable(
  snapshot: SessionSnapshot,
): Record<keyof SessionReportOptions, boolean> {
  const rows = snapshot.ncaState.rows.filter(
    (r) => r.t.trim() !== "" && r.c.trim() !== "",
  );
  return {
    ml: snapshot.mlResult !== null,
    learningCurve: snapshot.lcResult !== null,
    pkpd: true, // the regimen card is always live in the web twin
    nca: rows.length > 0,
    history: snapshot.mlHistory.length > 0,
    surgery: snapshot.surgery !== null,
  };
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

/** Compact number formatting for table cells (≤ 4 significant digits,
 *  trailing zeros trimmed — the fmtNum convention of the web cards). */
export function fmtCell(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 1000 || abs < 0.001) return value.toExponential(3);
  const fixed = abs >= 100 ? 1 : abs >= 1 ? 3 : 4;
  return String(Number(value.toFixed(fixed)));
}

/** The PK concentration series the regimen card draws (same sampling:
 *  3 half-lives or 2 intervals of tail, 420 samples + the endpoints). */
function pkSeries(params: PkParams): Array<[number, number]> {
  const derived = derivePk(params);
  const tail = Math.max(3 * derived.halfLife, 2 * params.tau);
  const total = (params.doses - 1) * params.tau + tail;
  const samples = 420;
  const step = total / samples;
  const points: Array<[number, number]> = [];
  for (let i = 0; i <= samples; i += 1) {
    const t = i * step;
    points.push([Number(t.toFixed(2)), concentrationAt(t, params, derived.ke)]);
  }
  return points;
}

/* ------------------------------------------------------------------ */
/* The collector                                                       */
/* ------------------------------------------------------------------ */

/** (100x-132) Each group's own sections, in the group's internal order.
 *  The group builders are pure: they read the snapshot and return only
 *  what that group ships — a group with no content returns [].  The
 *  collector assembles them under the (sanitized) walking order, so
 *  "the report reads the way the reader wants" is a reorder of WHOLE
 *  groups, never a re-cut of a group's own figures. */

function mlSections(snapshot: SessionSnapshot): ReportSection[] {
  const ml = snapshot.mlResult;
  if (!ml) return [];
  const sections: ReportSection[] = [];
  const metrics: Array<[string, string]> = [
    ["model", ml.model],
    ["task", ml.task],
  ];
  for (const [key, value] of Object.entries(ml.metrics)) {
    metrics.push([key, fmtCell(value)]);
  }
  metrics.push(["rows", `${ml.datasetMeta.nTotal}`]);
  metrics.push(["features", String(ml.datasetMeta.features.length)]);
  if (ml.datasetMeta.source) {
    metrics.push(["source", ml.datasetMeta.source.fileName ?? "csv"]);
  }
  sections.push({ kind: "metrics", title: "ML training", items: metrics });

  if (ml.task === "regression") {
    const traces: ReportFigureTrace[] = [
      {
        name: "model prediction",
        style: "line",
        points: (ml.series.curve ?? []).map((p) => [p.x, p.yhat] as [number, number]),
      },
      {
        name: "test observations",
        style: "scatter",
        points: ml.series.test.map((p) => [p.x, p.y] as [number, number]),
      },
    ];
    if (ml.series.train && ml.series.train.length > 0) {
      traces.push({
        name: "train observations",
        style: "scatter",
        points: ml.series.train.map((p) => [p.x, p.y] as [number, number]),
      });
    }
    const bandTrace = bandFromCurve(ml.series.curve ?? []);
    if (bandTrace) traces.splice(1, 0, bandTrace);
    sections.push({
      kind: "figure",
      title: "Predicted vs observed",
      xLabel: "observed",
      yLabel: "predicted",
      traces,
    });
  } else {
    const traces: ReportFigureTrace[] = [];
    if (ml.series.boundary && ml.series.boundary.length > 0) {
      traces.push({
        name: "decision regions",
        style: "scatter",
        points: ml.series.boundary.map(
          (p) => [p.x, p.y] as [number, number],
        ),
        classes: ml.series.boundary.map((p) => Math.round(p.pred)),
        probas: ml.series.boundary.map((p) => p.proba ?? Math.round(p.pred)),
      });
    }
    traces.push({
      name: "test observations",
      style: "scatter",
      points: ml.series.test.map((p) => [p.x, p.y] as [number, number]),
      classes: ml.series.test.map((p) => Math.round(p.label ?? 0)),
    });
    sections.push({
      kind: "figure",
      title: "Decision map",
      xLabel: "feature 1",
      yLabel: "feature 2",
      traces,
    });
  }
  return sections;
}

function learningCurveSections(snapshot: SessionSnapshot): ReportSection[] {
  const lc = snapshot.lcResult;
  if (!lc) return [];
  return [
    {
      kind: "metrics",
      title: "Learning curve",
      items: [
        ["model", lc.model],
        ["metric", lc.metricName],
        ["folds", `${lc.folds}`],
        ["gap", fmtCell(lc.gap)],
        ["dataset", lc.datasetMeta.kind === "csv"
          ? (lc.datasetMeta.fileName ?? "csv")
          : "synthetic"],
      ],
    },
    {
      kind: "figure",
      title: `Learning curve — ${lc.metricName} vs training samples`,
      xLabel: "training samples",
      yLabel: lc.metricName,
      traces: [
        {
          name: `training ${lc.metricName}`,
          style: "band",
          points: lc.points.map(
            (p) => [p.size, p.trainMean, p.trainMean - p.trainStd, p.trainMean + p.trainStd] as [number, number, number, number],
          ),
        },
        {
          name: `CV ${lc.metricName}`,
          style: "band",
          points: lc.points.map(
            (p) => [p.size, p.valMean, p.valMean - p.valStd, p.valMean + p.valStd] as [number, number, number, number],
          ),
        },
      ],
    },
  ];
}

function pkpdSections(snapshot: SessionSnapshot): ReportSection[] {
  // the regimen card is always live in the web twin — no guard
  const conc = pkSeries(snapshot.pkParams);
  const effect = conc.map(
    ([t, c]) =>
      [t, pdEffect(c, snapshot.pdParams.model, snapshot.pdParams)] as [number, number],
  );
  return [
    {
      kind: "figure",
      title: "PK regimen — concentration vs time",
      xLabel: "time (h)",
      yLabel: "concentration (mg/L)",
      traces: [
        {
          name: `${snapshot.pkParams.route === "iv" ? "IV bolus" : "oral"} regimen`,
          style: "line",
          points: conc,
        },
      ],
    },
    {
      kind: "figure",
      title: `PD effect — ${snapshot.pdParams.model} model`,
      xLabel: "time (h)",
      yLabel: "effect",
      traces: [
        { name: "effect", style: "line", points: effect },
      ],
    },
  ];
}

function ncaSections(snapshot: SessionSnapshot): ReportSection[] {
  const rows = snapshot.ncaState.rows.filter(
    (r) => r.t.trim() !== "" && r.c.trim() !== "",
  );
  if (rows.length === 0) return [];
  return [
    {
      kind: "table",
      title: "NCA observed data",
      columns: ["#", "time (h)", "concentration (mg/L)"],
      rows: rows.map((r, i) => [i + 1, r.t, r.c]),
    },
  ];
}

function historySections(snapshot: SessionSnapshot): ReportSection[] {
  if (snapshot.mlHistory.length === 0) return [];
  return [
    {
      kind: "table",
      title: "ML run history",
      columns: ["when", "task", "model", "metric"],
      rows: snapshot.mlHistory.map((run) => [
        new Date(run.at).toLocaleString(),
        run.task,
        run.model,
        `${run.keyMetric} ${fmtCell(run.keyValue)}`,
      ]),
    },
  ];
}

function surgerySections(snapshot: SessionSnapshot): ReportSection[] {
  const surgery = snapshot.surgery;
  if (!surgery) return [];
  if (surgery.mode === "topology") {
    return [
      {
        kind: "metrics",
        title: "Ring topology moves",
        items: [
          ["source", surgery.source],
          ["distinct mutants", `${surgery.mutants.length}`],
          ["compute time", `${fmtCell(surgery.elapsedSeconds)} s`],
          ["RDKit", surgery.rdkitVersion],
        ],
      },
      {
        kind: "table",
        title: "Ring topology mutants",
        columns: ["#", "mutant SMILES"],
        rows: surgery.mutants.map((mutant, i) => [i + 1, mutant]),
      },
      {
        kind: "image",
        title: `Ring topology mutants — ${surgery.mutants.length} structure panel(s)`,
        dataUri: surgery.gridPng,
      },
    ];
  }
  return [
    {
      kind: "metrics",
      title: "Scaffold-hop crossover",
      items: [
        ["parent A", surgery.parentA],
        ["parent B", surgery.parentB],
        ["viable children", `${surgery.viable}/${surgery.children}`],
        ["seed", `${surgery.seed}`],
        ["compute time", `${fmtCell(surgery.elapsedSeconds)} s`],
        ["RDKit", surgery.rdkitVersion],
      ],
    },
    {
      kind: "table",
      title: "Crossover litter",
      columns: ["draw", "child SMILES"],
      rows: surgery.litter.map((child, i) => [
        i + 1,
        child ?? "(no viable crossover)",
      ]),
    },
    {
      kind: "image",
      title: `Parents & children — ${surgery.gridPanels} structure panel(s)`,
      dataUri: surgery.gridPng,
    },
  ];
}

/** Build the report payload from the session snapshot.  Pure: no DOM,
 *  no store subscription — the UI component reads the store and hands
 *  the pieces in, so the collector is trivially unit-testable.
 *
 *  Sections walk in the (100x-132) WALKING ORDER: the group builders
 *  each contribute their own sections (a group with no content
 *  contributes nothing), and the collector assembles them under
 *  ``order`` — the report card's drag-and-drop chips.  The canonical
 *  order (models → curve → regimen → data tables → surgery) is what
 *  an absent/invalid ``order`` falls back to; `sanitizeSessionOrder`
 *  documents the exact rules.
 *
 * (100x-131) `options` turns whole groups off — the report card's
 *  per-section toggles: skip the PK figures when only the ML run
 *  matters, ship only the surgery grid, whatever the session needs.
 *  A group with no content is skipped regardless, so the census the
 *  UI shows always matches what would ship.
 */
export function collectSessionReport(
  snapshot: SessionSnapshot,
  options: SessionReportOptions = {},
  order?: SessionGroupId[],
): SessionReportRequest {
  const enabled = {
    ml: options.ml !== false,
    learningCurve: options.learningCurve !== false,
    pkpd: options.pkpd !== false,
    nca: options.nca !== false,
    history: options.history !== false,
    surgery: options.surgery !== false,
  };
  const builders: Record<
    SessionGroupId,
    (snapshot: SessionSnapshot) => ReportSection[]
  > = {
    ml: mlSections,
    learningCurve: learningCurveSections,
    pkpd: pkpdSections,
    nca: ncaSections,
    history: historySections,
    surgery: surgerySections,
  };
  const sections: ReportSection[] = [];
  for (const id of sanitizeSessionOrder(order)) {
    if (enabled[id]) sections.push(...builders[id](snapshot));
  }

  return {
    theme: "auto",
    subtitle: "web playground session",
    sections,
  };
}

/** Turn a curve with optional lo/hi into a band trace (GP-style). */
function bandFromCurve(
  curve: Array<{ x: number; yhat: number; lo?: number; hi?: number }>,
): ReportFigureTrace | null {
  const banded = curve.filter((p) => p.lo !== undefined && p.hi !== undefined);
  if (banded.length < 2) return null;
  return {
    name: "± band",
    style: "band",
    points: banded.map(
      (p) => [p.x, p.yhat, p.lo as number, p.hi as number] as [number, number, number, number],
    ),
  };
}

/** The live census for the UI button ("what will ship"): figure and
 *  table counts over the same walk the collector performs — image
 *  sections count separately (the surgery grids) so the button line
 *  can say "2 figures · 1 structure grid" honestly.  (100x-132) the
 *  walking order never changes the counts, but it rides the same
 *  signature so the UI always pins the census to the report it would
 *  actually ship. */
export function sessionReportCensus(
  snapshot: SessionSnapshot,
  options: SessionReportOptions = {},
  order?: SessionGroupId[],
): {
  figures: number;
  tables: number;
  metrics: number;
  images: number;
} {
  const { sections } = collectSessionReport(snapshot, options, order);
  let figures = 0;
  let tables = 0;
  let metrics = 0;
  let images = 0;
  for (const section of sections) {
    if (section.kind === "figure") figures += 1;
    else if (section.kind === "table") tables += 1;
    else if (section.kind === "image") images += 1;
    else metrics += 1;
  }
  return { figures, tables, metrics, images };
}

/* ------------------------------------------------------------------ */
/* The preview (100x-133)                                              */
/* ------------------------------------------------------------------ */

/** Rows a table section shows in the preview dialog before the "+N more
 *  rows" confession — the download ships every row, the preview stays
 *  light.  Exported so the probe can pin the contract. */
export const PREVIEW_TABLE_ROWS = 6;

/** Dots a scatter trace draws in the preview sketch (stride-sampled, so
 *  the shape survives the cap) — the downloaded report renders every
 *  point through matplotlib. */
export const PREVIEW_SCATTER_POINTS = 120;

/** (100x-133) The x/y bounds a figure section's preview sketch maps its
 *  traces into — band edges (lo/hi) participate, a degenerate axis (all
 *  values equal, or no finite points) pads itself by ±1 so a flat line
 *  still draws inside the viewBox instead of collapsing to a hairline.
 *  Pure: the probe pins the arithmetic. */
export function figureSketchBounds(
  traces: ReportFigureTrace[],
): { xMin: number; xMax: number; yMin: number; yMax: number } {
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  for (const trace of traces) {
    for (const point of trace.points) {
      const [x, y, lo, hi] = point as [
        number,
        number,
        number | undefined,
        number | undefined,
      ];
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      if (x < xMin) xMin = x;
      if (x > xMax) xMax = x;
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
      if (lo !== undefined && Number.isFinite(lo) && lo < yMin) yMin = lo;
      if (hi !== undefined && Number.isFinite(hi) && hi > yMax) yMax = hi;
    }
  }
  if (xMin === Infinity) return { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };
  if (xMin === xMax) {
    xMin -= 1;
    xMax += 1;
  }
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  return { xMin, xMax, yMin, yMax };
}

/* ------------------------------------------------------------------ */
/* The preview's Contents twin (100x-137)                              */
/* ------------------------------------------------------------------ */

/** (100x-137) One line of the preview's Contents strip — the twin of
 *  the downloaded file's header Contents block: the section's title
 *  (the SAME string the file's line prints) plus its overall index in
 *  the payload, so a click can scroll the dialog's list to it. */
export interface ReportContentsEntry {
  /** The section's overall index in the payload (0-based) — the number
   *  the preview's own numbered entries carry, so the strip's numbers
   *  and the list's numbers are the SAME coordinates. */
  index: number;
  /** The title the file's Contents line prints (the section title). */
  title: string;
  /** The section's kind — table / figure / image (metrics sections are
   *  title-page material and never listed, the file twin's rule). */
  kind: "table" | "figure" | "image";
}

/** (100x-137) Derive the Contents lines the preview dialog shows —
 *  exactly what the downloaded file's header Contents block lists:
 *  every table / figure / image section's title in walking order
 *  (images ride the add_figure contract in the file, so they list like
 *  figures here).  Metrics sections are NOT listed (they are
 *  title-page material, the file twin's discipline), so a metrics-only
 *  report yields [] — the noise rule: the strip hides itself.  Pure:
 *  the probe pins the derivation. */
export function sessionReportContents(
  sections: ReportSection[],
): ReportContentsEntry[] {
  const entries: ReportContentsEntry[] = [];
  for (let index = 0; index < sections.length; index += 1) {
    const section = sections[index];
    if (section.kind === "metrics") continue;
    entries.push({ index, title: section.title, kind: section.kind });
  }
  return entries;
}

/** (100x-138) The KIND CENSUS of a Contents list — the counts of
 *  exactly what the strip lists (figures / tables / images).  Here
 *  metrics are excluded by construction: the census derives from the
 *  Contents entries themselves (the strip's own noise rule — the
 *  footer's census counts metric cards too, this one never does). */
export interface SessionContentsCensus {
  /** Figure sections listed (image sections ride add_figure in the
   *  file, so they are counted separately below, never as figures). */
  figures: number;
  /** Table sections listed. */
  tables: number;
  /** Image sections listed — the structure grids. */
  images: number;
}

/** (100x-138) Derive the Contents strip's kind census — pure, so the
 *  probe can pin it: over the strip's own entries (metrics never
 *  listed), figures/tables/images are counted by their kind. */
export function sessionContentsCensus(
  entries: ReportContentsEntry[],
): SessionContentsCensus {
  let figures = 0;
  let tables = 0;
  let images = 0;
  for (const entry of entries) {
    if (entry.kind === "figure") figures += 1;
    else if (entry.kind === "table") tables += 1;
    else images += 1;
  }
  return { figures, tables, images };
}

/** (100x-138) The census's readable line — "2 figures · 1 table ·
 *  1 structure grid", the footer's singular/plural discipline (images
 *  read as "structure grid(s)", the established wording; a census
 *  with no images drops the leg entirely).  Pure — the probe pins the wording. */
export function contentsCensusText(census: SessionContentsCensus): string {
  const bits: string[] = [
    `${census.figures} figure${census.figures === 1 ? "" : "s"}`,
    `${census.tables} table${census.tables === 1 ? "" : "s"}`,
  ];
  if (census.images > 0) {
    bits.push(`${census.images} structure grid${census.images === 1 ? "" : "s"}`);
  }
  return bits.join(" · ");
}

/** (100x-134) The hover-INSPECT text a preview sketch's trace carries —
 *  the honest numbers behind the approximation: the trace's name, its
 *  point count, its x/y ranges (fmtCell-compacted), and the kind's own
 *  confession — a band reports its CENTER mean (the polygon's edges are
 *  the ± spread, the mean line is what reads), a classified scatter
 *  reports its class split (how many dots ride each color).  Pure: the
 *  probe pins the wording.  Rendered as a native SVG <title> inside the
 *  trace's group, so every browser shows it on hover with zero JS. */
export function traceInspectText(trace: ReportFigureTrace): string {
  const total = trace.points.length;
  if (total === 0) return `${trace.name} — no points`;
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  let sum = 0;
  let finite = 0;
  for (const point of trace.points) {
    const [x, y] = point as [number, number, number?, number?];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if (x < xMin) xMin = x;
    if (x > xMax) xMax = x;
    if (y < yMin) yMin = y;
    if (y > yMax) yMax = y;
    sum += y;
    finite += 1;
  }
  if (finite === 0) return `${trace.name} — no finite points`;
  const span = (lo: number, hi: number) =>
    lo === hi ? fmtCell(lo) : `${fmtCell(lo)}…${fmtCell(hi)}`;
  if (trace.style === "band") {
    return (
      `${trace.name} — ${total} pts · x ${span(xMin, xMax)} · ` +
      `center mean ${fmtCell(sum / finite)}`
    );
  }
  if (trace.style === "scatter" && trace.classes) {
    let ones = 0;
    for (const klass of trace.classes) if (klass === 1) ones += 1;
    const zeros = trace.classes.length - ones;
    return (
      `${trace.name} — ${total} pts · x ${span(xMin, xMax)} · ` +
      `y ${span(yMin, yMax)} · ${ones} class 1 / ${zeros} class 0`
    );
  }
  return (
    `${trace.name} — ${total} pts · x ${span(xMin, xMax)} · ` +
    `y ${span(yMin, yMax)}`
  );
}
