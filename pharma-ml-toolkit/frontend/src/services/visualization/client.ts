/**
 * The Visualization workspace client — part11 §11.8's workflow
 * service (the cross-workspace output registry + the REAL plot
 * families) over the REAL backend.
 *
 * The backend contract (mirrored field-for-field from the route
 * source `routes/visualization.py`; the request bodies come from
 * the generated OpenAPI types — the same one-DTO-source rule the
 * Python drift guards in `tests/test_visualization_workspace.py`
 * pin for this client):
 *
 * * `GET /api/v1/visualization/catalog` — the plot families this
 *   route serves (statplots, pharma figures, MD series, tables,
 *   reports) + `PALETTE` + the export formats + the report themes.
 * * `POST /api/v1/visualization/stat` — `StatRequest` (the
 *   statplots family: box / histogram / ecdf / regression — the
 *   REAL stats dict plus the rendered figure artifact).
 * * `POST /api/v1/visualization/concentration-profile` —
 *   `ConcentrationProfileRequest` (the REAL
 *   `plot_concentration_profile` exposure figure).
 * * `POST /api/v1/visualization/dose-accumulation` —
 *   `DoseAccumulationRequest` (the REAL `plot_dose_accumulation`
 *   approach-to-steady-state figure; exactly one of `t_half` /
 *   `ke` is the route's own verdict).
 * * `POST /api/v1/visualization/md-series` — `MdSeriesRequest`
 *   (the REAL `plot_md_series` over parsed `.xvg` text — one series
 *   or several).
 * * `POST /api/v1/visualization/table` —
 *   `VisualizationTableRequest` (the REAL `export_table`:
 *   csv/tsv/xlsx/md/json/html).
 * * `POST /api/v1/visualization/report` — `ReportRequest` (the
 *   REAL `HTMLReport` as a managed job: metrics, tables and figure
 *   sections assembled into one self-contained HTML artifact, with
 *   honest `file.progress` events per section; the theme must be
 *   the catalog's own).
 *
 * §11.8's "must be able to access relevant outputs created in other
 * workspaces" rides the shared registry client
 * (`artifactsApi.list`) — this client never duplicates it.
 *
 * Pre-flight bounds (the route's own pydantic bounds, enforced
 * before the request fires): the stat series `4…5000` points
 * (regression: both vectors, equal lengths), the pharma figures'
 * series `2…5000` points, the md-series map `1…4` series, the
 * table rows `1…5000`, the report `tables` list `1…5`. The scalar
 * ranges (whis, dpi, dose, tau, …) stay the backend's own honest
 * 422s — they are scientific verdicts, not transport errors.
 */

import { api } from "@/services/api/client";
import { parseArtifactView, type ArtifactView } from "@/services/artifacts/client";
import { type JobView, parseJobView } from "@/services/jobs/types";
import type { components } from "@/types/api.d";

type StatRequest = components["schemas"]["StatRequest"];
type ConcentrationProfileRequest =
  components["schemas"]["ConcentrationProfileRequest"];
type DoseAccumulationRequest =
  components["schemas"]["DoseAccumulationRequest"];
type MdSeriesRequest = components["schemas"]["MdSeriesRequest"];
type VisualizationTableRequest =
  components["schemas"]["VisualizationTableRequest"];
type ReportRequest = components["schemas"]["ReportRequest"];

// -- the response views (hand-written, runtime-narrowed) --------------

/** One `/catalog` family row — the route's own keys. */
export interface VizCatalogFamily {
  readonly family: string;
  /** The stat/figure families carry kinds; the table family carries formats. */
  readonly kinds: readonly string[];
  readonly formats: readonly string[];
  readonly backend: string;
}

/** The `GET /api/v1/visualization/catalog` answer. */
export interface VizCatalogResponse {
  readonly families: readonly VizCatalogFamily[];
  readonly palette: readonly string[];
  readonly export_formats: readonly string[];
  readonly report_themes: readonly string[];
}

/**
 * The stat lane's REAL stats dict — the statplots family's own
 * verdict (`box_stats` / `hist_stats` / `ecdf_stats` /
 * `regression_stats`), rendered type-aware and bounded by the page.
 */
export interface StatResponse {
  readonly kind: "box" | "histogram" | "ecdf" | "regression";
  readonly n_points: number;
  readonly stats: Readonly<Record<string, unknown>>;
  readonly artifact: ArtifactView | null;
}

/** The `POST /concentration-profile` answer. */
export interface ConcentrationProfileResponse {
  readonly artifact: ArtifactView;
  readonly n_points: number;
}

/** The `POST /dose-accumulation` answer. */
export interface DoseAccumulationResponse {
  readonly artifact: ArtifactView;
  readonly tau: number;
  readonly n_doses: number;
}

/** One named `.xvg` series' stats row — the route's own `series` block. */
export interface VizSeriesStatView {
  readonly n_points: number;
  readonly label: string;
  readonly stats: { readonly mean: number | null; readonly std: number | null; readonly cv: number | null };
}

/** The `POST /md-series` answer. */
export interface VizMdSeriesResponse {
  readonly n_series: number;
  readonly series: Readonly<Record<string, VizSeriesStatView>>;
  readonly plot: Readonly<Record<string, unknown>> | null;
  readonly artifact: ArtifactView;
}

/** The `POST /table` answer. */
export interface VizTableResponse {
  readonly artifact: ArtifactView;
  readonly rows: number;
}

/** The report body — `_report_body`'s own return dict. */
export interface ReportBodyView {
  readonly title: string;
  readonly stages: readonly string[];
  readonly figures: number;
  readonly artifact: ArtifactView;
}

/** The sync-mode report answer. */
export interface ReportSyncResponse extends ReportBodyView {
  readonly mode: "sync";
}

/** The job-mode report answer. */
export interface ReportJobResponse {
  readonly mode: "job";
  readonly job: JobView;
}

export type ReportResponse = ReportSyncResponse | ReportJobResponse;

/**
 * One `file.progress` envelope payload — the report lane's REAL
 * per-section telemetry (`JobManager.report_progress`'s own
 * `{progress, stage}` dict; progress-only, never the result).
 */
export interface JobProgressPayload {
  readonly progress: number;
  readonly stage: string;
}

// -- the option payloads ----------------------------------------------

export interface StatOptions {
  readonly kind: "box" | "histogram" | "ecdf" | "regression";
  readonly values: readonly number[] | null;
  readonly x: readonly number[] | null;
  readonly y: readonly number[] | null;
  readonly bins: string | number;
  readonly whis: number;
  readonly nBoot: number;
  readonly seed: number;
  readonly color: string | null;
  readonly withArtifact: boolean;
  readonly format: "PNG" | "SVG";
}

export interface ConcentrationProfileOptions {
  readonly times: readonly number[];
  readonly conc: readonly number[];
  readonly auc: number | null;
  readonly title: string;
  readonly format: "PNG" | "SVG" | "TIFF" | "JPEG";
  readonly dpi: number;
}

export interface DoseAccumulationOptions {
  readonly dose: number;
  readonly tau: number;
  readonly tHalf: number | null;
  readonly ke: number | null;
  readonly vd: number;
  readonly f: number;
  readonly nDoses: number;
  readonly title: string;
  readonly format: "PNG" | "SVG" | "TIFF" | "JPEG";
  readonly dpi: number;
}

export interface VizMdSeriesOptions {
  readonly series: Readonly<Record<string, string>>;
  readonly title: string | null;
  readonly xLabel: string | null;
  readonly yLabel: string | null;
  readonly format: "PNG" | "SVG" | "TIFF" | "JPEG";
}

export interface TableExportOptions {
  /** The rows the export carries (the backend's own 5000-row cap). */
  readonly rows: Readonly<Record<string, unknown>[]>;
  /** The backend's own export enum (visualization.py's own literals). */
  readonly format: "csv" | "tsv" | "xlsx" | "md" | "json" | "html";
  readonly filename: string;
}

export interface ReportOptions {
  readonly title: string;
  readonly subtitle: string | null;
  readonly theme: string;
  readonly metrics: Readonly<Record<string, number | string>> | null;
  readonly tables: readonly Record<string, unknown>[] | null;
  readonly concentrationProfile: Readonly<Record<string, unknown>> | null;
  readonly doseAccumulation: Readonly<Record<string, unknown>> | null;
  readonly stat: Readonly<Record<string, unknown>> | null;
  readonly mode: "job" | "sync";
}

// -- the api object ----------------------------------------------------

export const vizApi = {
  /** `GET /api/v1/visualization/catalog` — the plot families. */
  async catalog(): Promise<VizCatalogResponse> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/visualization/catalog",
    );
    return parseVizCatalog(raw, "visualization catalog");
  },

  /**
   * `POST /api/v1/visualization/stat` — the statplots family's REAL
   * stats dict + the rendered figure artifact. Regression needs both
   * `x` and `y` (equal lengths); the others need `values` — the
   * pre-flight mirrors the route's own 4…5000 window.
   */
  async stat(options: StatOptions): Promise<StatResponse> {
    const request: StatRequest = {
      kind: options.kind,
      ...(options.values === null ? {} : { values: [...options.values] }),
      ...(options.x === null ? {} : { x: [...options.x] }),
      ...(options.y === null ? {} : { y: [...options.y] }),
      bins: options.bins,
      whis: options.whis,
      n_boot: options.nBoot,
      seed: options.seed,
      ...(options.color === null ? {} : { color: options.color }),
      with_artifact: options.withArtifact,
      format: options.format,
    };
    const raw = (await api.post(
      "/api/v1/visualization/stat",
      request,
    )) as Record<string, unknown>;
    return parseStatResponse(raw, "stat");
  },

  /**
   * `POST /api/v1/visualization/concentration-profile` — the REAL
   * exposure figure (times + conc, equal lengths; the honest
   * minimum is two finite points).
   */
  async concentrationProfile(
    options: ConcentrationProfileOptions,
  ): Promise<ConcentrationProfileResponse> {
    const request: ConcentrationProfileRequest = {
      times: [...options.times],
      conc: [...options.conc],
      ...(options.auc === null ? {} : { auc: options.auc }),
      title: options.title,
      format: options.format,
      dpi: options.dpi,
    };
    const raw = (await api.post(
      "/api/v1/visualization/concentration-profile",
      request,
    )) as Record<string, unknown>;
    return parseConcentrationProfileResponse(raw, "concentration-profile");
  },

  /**
   * `POST /api/v1/visualization/dose-accumulation` — the REAL
   * approach-to-steady-state figure (exactly ONE of `t_half` / `ke`
   * — the route's own verdict, mirrored in the pre-flight).
   */
  async doseAccumulation(
    options: DoseAccumulationOptions,
  ): Promise<DoseAccumulationResponse> {
    const request: DoseAccumulationRequest = {
      dose: options.dose,
      tau: options.tau,
      ...(options.tHalf === null ? {} : { t_half: options.tHalf }),
      ...(options.ke === null ? {} : { ke: options.ke }),
      vd: options.vd,
      f: options.f,
      n_doses: options.nDoses,
      title: options.title,
      format: options.format,
      dpi: options.dpi,
    };
    const raw = (await api.post(
      "/api/v1/visualization/dose-accumulation",
      request,
    )) as Record<string, unknown>;
    return parseDoseAccumulationResponse(raw, "dose-accumulation");
  },

  /**
   * `POST /api/v1/visualization/md-series` — the REAL `plot_md_series`
   * figure over parsed `.xvg` text (one named series or up to four),
   * registered as a downloadable artifact through the §17 four-format
   * export stack.
   */
  async mdSeries(
    options: VizMdSeriesOptions,
  ): Promise<VizMdSeriesResponse> {
    if (Object.keys(options.series).length === 0) {
      throw new Error("the md-series figure needs at least one named series");
    }
    const request: MdSeriesRequest = {
      series: { ...options.series },
      ...(options.title === null ? {} : { title: options.title }),
      ...(options.xLabel === null ? {} : { x_label: options.xLabel }),
      ...(options.yLabel === null ? {} : { y_label: options.yLabel }),
      format: options.format,
    };
    const raw = (await api.post(
      "/api/v1/visualization/md-series",
      request,
    )) as Record<string, unknown>;
    return parseVizMdSeriesResponse(raw, "md-series");
  },

  /**
   * `POST /api/v1/visualization/table` — the REAL `export_table`
   * (csv/tsv/xlsx/md/json/html). The backend's own 5000-row cap is
   * mirrored here — the error names both numbers, never a silent
   * truncation.
   */
  async exportTable(options: TableExportOptions): Promise<VizTableResponse> {
    if (options.rows.length === 0) {
      throw new Error("the export carries no rows");
    }
    if (options.rows.length > 5000) {
      throw new Error(
        `the export carries ${options.rows.length} rows; the backend's own cap is 5000`,
      );
    }
    const request: VisualizationTableRequest = {
      rows: [...options.rows],
      format: options.format,
      filename: options.filename,
    };
    const raw = (await api.post(
      "/api/v1/visualization/table",
      request,
    )) as Record<string, unknown>;
    return parseVizTableResponse(raw, "table");
  },

  /**
   * `POST /api/v1/visualization/report` — the REAL self-contained
   * HTML report as a managed job (metrics + tables + figure
   * sections, honest `file.progress` events per section). The
   * COMPLETED job's REST result payload carries the same body the
   * sync lane returns — the page relays it through the jobs client
   * (§13), never the WS event payload alone.
   */
  async report(options: ReportOptions): Promise<ReportResponse> {
    const request: ReportRequest = {
      title: options.title,
      ...(options.subtitle === null ? {} : { subtitle: options.subtitle }),
      theme: options.theme,
      ...(options.metrics === null
        ? {}
        : { metrics: { ...options.metrics } }),
      ...(options.tables === null ? {} : { tables: [...options.tables] }),
      ...(options.concentrationProfile === null
        ? {}
        : { concentration_profile: { ...options.concentrationProfile } }),
      ...(options.doseAccumulation === null
        ? {}
        : { dose_accumulation: { ...options.doseAccumulation } }),
      ...(options.stat === null ? {} : { stat: { ...options.stat } }),
      mode: options.mode,
    };
    const raw = (await api.post(
      "/api/v1/visualization/report",
      request,
    )) as Record<string, unknown>;
    if (raw["mode"] === "sync") {
      return { mode: "sync", ...parseReportBody(raw, "report(sync)") };
    }
    return { mode: "job", job: parseJobView(raw["job"], "report(job)") };
  },
};

// -- validation helpers (§8.1) ------------------------------------------

function expectObject(raw: unknown, what: string): Record<string, unknown> {
  if (typeof raw !== "object" || raw === null) {
    throw new Error(`${what}: expected an object`);
  }
  return raw as Record<string, unknown>;
}

function asNumber(raw: unknown): number | null {
  return typeof raw === "number" && Number.isFinite(raw) ? raw : null;
}

function asNumberOrNull(raw: unknown): number | null {
  return raw === null ? null : asNumber(raw);
}

function asString(raw: unknown): string {
  return typeof raw === "string" ? raw : "";
}

// -- the response parsers ------------------------------------------------

/** Narrow the `/catalog` answer (families + palette + formats + themes). */
export function parseVizCatalog(
  raw: unknown,
  what: string,
): VizCatalogResponse {
  const r = expectObject(raw, what);
  const families = r["families"];
  if (!Array.isArray(families)) {
    throw new Error(`${what}: families is not an array`);
  }
  const parsedFamilies = families.map((row, i) => {
    const f = expectObject(row, `${what}[${i}]`);
    const family = f["family"];
    if (typeof family !== "string") {
      throw new Error(`${what}[${i}]: family is not a string`);
    }
    const kinds = f["kinds"];
    const formats = f["formats"];
    return {
      family,
      kinds: Array.isArray(kinds) ? kinds.map((k) => String(k)) : [],
      formats: Array.isArray(formats) ? formats.map((v) => String(v)) : [],
      backend: asString(f["backend"]),
    };
  });
  const palette = r["palette"];
  const exportFormats = r["export_formats"];
  const themes = r["report_themes"];
  return {
    families: parsedFamilies,
    palette: Array.isArray(palette) ? palette.map((v) => String(v)) : [],
    export_formats: Array.isArray(exportFormats)
      ? exportFormats.map((v) => String(v))
      : [],
    report_themes: Array.isArray(themes) ? themes.map((v) => String(v)) : [],
  };
}

/** Narrow the `POST /stat` answer (the stats dict stays type-aware). */
export function parseStatResponse(
  raw: unknown,
  what: string,
): StatResponse {
  const r = expectObject(raw, what);
  const kind = r["kind"];
  if (
    kind !== "box" && kind !== "histogram" &&
    kind !== "ecdf" && kind !== "regression"
  ) {
    throw new Error(`${what}: kind is not a statplots kind`);
  }
  const nPoints = asNumber(r["n_points"]);
  if (nPoints === null) {
    throw new Error(`${what}: n_points is not a finite number`);
  }
  const stats = expectObject(r["stats"], `${what} stats`);
  const artifact = r["artifact"];
  return {
    kind,
    n_points: nPoints,
    stats,
    artifact: artifact === null
      ? null
      : parseArtifactView(artifact, `${what} artifact`),
  };
}

/** Narrow the `POST /concentration-profile` answer. */
export function parseConcentrationProfileResponse(
  raw: unknown,
  what: string,
): ConcentrationProfileResponse {
  const r = expectObject(raw, what);
  const nPoints = asNumber(r["n_points"]);
  if (nPoints === null) {
    throw new Error(`${what}: n_points is not a finite number`);
  }
  const artifact = r["artifact"];
  if (typeof artifact !== "object" || artifact === null) {
    throw new Error(`${what}: the figure lane answered without an artifact`);
  }
  return {
    artifact: parseArtifactView(artifact, `${what} artifact`),
    n_points: nPoints,
  };
}

/** Narrow the `POST /dose-accumulation` answer. */
export function parseDoseAccumulationResponse(
  raw: unknown,
  what: string,
): DoseAccumulationResponse {
  const r = expectObject(raw, what);
  const tau = asNumber(r["tau"]);
  const nDoses = asNumber(r["n_doses"]);
  if (tau === null || nDoses === null) {
    throw new Error(`${what}: tau/n_doses is not a finite number`);
  }
  const artifact = r["artifact"];
  if (typeof artifact !== "object" || artifact === null) {
    throw new Error(`${what}: the figure lane answered without an artifact`);
  }
  return {
    artifact: parseArtifactView(artifact, `${what} artifact`),
    tau,
    n_doses: nDoses,
  };
}

/** Narrow the `POST /md-series` answer. */
export function parseVizMdSeriesResponse(
  raw: unknown,
  what: string,
): VizMdSeriesResponse {
  const r = expectObject(raw, what);
  const nSeries = asNumber(r["n_series"]);
  if (nSeries === null) {
    throw new Error(`${what}: n_series is not a finite number`);
  }
  const seriesRaw = expectObject(r["series"], `${what} series`);
  const series: Record<string, VizSeriesStatView> = {};
  for (const [name, row] of Object.entries(seriesRaw)) {
    const s = expectObject(row, `${what} series[${name}]`);
    const nPoints = asNumber(s["n_points"]);
    if (nPoints === null) {
      throw new Error(`${what} series[${name}]: n_points is not a finite number`);
    }
    const stats = expectObject(s["stats"], `${what} series[${name}] stats`);
    series[name] = {
      n_points: nPoints,
      label: asString(s["label"]),
      stats: {
        mean: asNumberOrNull(stats["mean"]),
        std: asNumberOrNull(stats["std"]),
        cv: asNumberOrNull(stats["cv"]),
      },
    };
  }
  if (Object.keys(series).length !== nSeries) {
    throw new Error(
      `${what}: the series block carries ${Object.keys(series).length} rows, n_series is ${nSeries}`,
    );
  }
  const plot = r["plot"];
  const artifact = r["artifact"];
  if (typeof artifact !== "object" || artifact === null) {
    throw new Error(`${what}: the figure lane answered without an artifact`);
  }
  return {
    n_series: nSeries,
    series,
    plot: typeof plot === "object" && plot !== null
      ? (plot as Readonly<Record<string, unknown>>)
      : null,
    artifact: parseArtifactView(artifact, `${what} artifact`),
  };
}

/** Narrow the `POST /table` answer. */
export function parseVizTableResponse(
  raw: unknown,
  what: string,
): VizTableResponse {
  const r = expectObject(raw, what);
  const rows = asNumber(r["rows"]);
  if (rows === null) {
    throw new Error(`${what}: rows is not a finite number`);
  }
  const artifact = r["artifact"];
  if (typeof artifact !== "object" || artifact === null) {
    throw new Error(`${what}: the table lane answered without an artifact`);
  }
  return {
    artifact: parseArtifactView(artifact, `${what} artifact`),
    rows,
  };
}

/** Narrow the report body (the sync answer / the job REST result). */
export function parseReportBody(
  raw: unknown,
  what: string,
): ReportBodyView {
  const r = expectObject(raw, what);
  const title = r["title"];
  if (typeof title !== "string") {
    throw new Error(`${what}: title is not a string`);
  }
  const stages = r["stages"];
  if (!Array.isArray(stages)) {
    throw new Error(`${what}: stages is not an array`);
  }
  const figures = asNumber(r["figures"]);
  if (figures === null) {
    throw new Error(`${what}: figures is not a finite number`);
  }
  const artifact = r["artifact"];
  if (typeof artifact !== "object" || artifact === null) {
    throw new Error(`${what}: the report lane answered without an artifact`);
  }
  return {
    title,
    stages: stages.map((stage) => String(stage)),
    figures,
    artifact: parseArtifactView(artifact, `${what} artifact`),
  };
}

/**
 * Narrow one `file.progress` envelope payload — the report lane's
 * per-section telemetry (progress-only; the authoritative result
 * rides the REST jobs client, §13).
 */
export function parseJobProgressPayload(
  raw: unknown,
  what: string,
): JobProgressPayload {
  const r = expectObject(raw, what);
  const progress = r["progress"];
  if (typeof progress !== "number") {
    throw new Error(`${what}: progress is not a number`);
  }
  const stage = r["stage"];
  if (typeof stage !== "string") {
    throw new Error(`${what}: stage is not a string`);
  }
  return { progress, stage };
}

// -- the pre-flight bound helpers (the page calls these before the POST)

/** The route's own stat-series point window (`StatRequest`'s bounds). */
export const STAT_MIN_POINTS = 4;
export const STAT_MAX_POINTS = 5000;
/** The route's own md-series series cap (`_MAX_SERIES`). */
export const MD_SERIES_MAX = 4;
/** The route's own table row cap (`_MAX_ROWS`). */
export const TABLE_MAX_ROWS = 5000;
/** The route's own report tables cap (`ReportRequest`'s bound). */
export const REPORT_MAX_TABLES = 5;

/**
 * The stat lane's pre-flight (the route's own list bounds): box /
 * histogram / ecdf need `values` with `4…5000` points; regression
 * needs BOTH `x` and `y`, equal lengths, `4…5000` points each.
 */
export function assertStatBounds(
  what: string,
  kind: "box" | "histogram" | "ecdf" | "regression",
  values: readonly number[] | null,
  x: readonly number[] | null,
  y: readonly number[] | null,
): void {
  if (kind === "regression") {
    if (x === null || y === null) {
      throw new Error(`${what}: regression needs both x=[…] and y=[…]`);
    }
    if (x.length !== y.length) {
      throw new Error(
        `${what}: x carries ${x.length} points, y carries ${y.length} (equal lengths required)`,
      );
    }
    if (x.length < STAT_MIN_POINTS) {
      throw new Error(
        `${what}: ${x.length} points, at least ${STAT_MIN_POINTS} required`,
      );
    }
    if (x.length > STAT_MAX_POINTS) {
      throw new Error(
        `${what}: ${x.length} points, over the ${STAT_MAX_POINTS} cap`,
      );
    }
    return;
  }
  if (values === null) {
    throw new Error(`${what}: ${kind} needs values=[…]`);
  }
  if (values.length < STAT_MIN_POINTS) {
    throw new Error(
      `${what}: ${values.length} points, at least ${STAT_MIN_POINTS} required`,
    );
  }
  if (values.length > STAT_MAX_POINTS) {
    throw new Error(
      `${what}: ${values.length} points, over the ${STAT_MAX_POINTS} cap`,
    );
  }
}

/**
 * The pharma figures' pre-flight (the route's own list bounds):
 * `times` and `conc` equal lengths, `2…5000` points.
 */
export function assertSeriesPairBounds(
  what: string,
  times: readonly number[],
  conc: readonly number[],
): void {
  if (times.length !== conc.length) {
    throw new Error(
      `${what}: the times (${times.length}) and conc (${conc.length}) series must be the same length`,
    );
  }
  if (times.length < 2) {
    throw new Error(`${what}: ${times.length} points, at least 2 required`);
  }
  if (times.length > STAT_MAX_POINTS) {
    throw new Error(
      `${what}: ${times.length} points, over the ${STAT_MAX_POINTS} cap`,
    );
  }
}

/**
 * The md-series lane's pre-flight (the route's own bounds): a
 * non-empty series map with at most four entries.
 */
export function assertMdSeriesBounds(
  what: string,
  series: Readonly<Record<string, string>>,
): void {
  const names = Object.keys(series);
  if (names.length === 0) {
    throw new Error(`${what}: the series map is empty`);
  }
  if (names.length > MD_SERIES_MAX) {
    throw new Error(
      `${what}: ${names.length} series, over the ${MD_SERIES_MAX} cap`,
    );
  }
  for (const [name, text] of Object.entries(series)) {
    if (text.length === 0) {
      throw new Error(`${what}: series "${name}" is empty`);
    }
  }
}

/**
 * The dose-accumulation lane's pre-flight (the route's own verdict):
 * exactly ONE of `t_half` / `ke`.
 */
export function assertDoseAccumulationPair(
  what: string,
  tHalf: number | null,
  ke: number | null,
): void {
  if ((tHalf === null) === (ke === null)) {
    throw new Error(`${what}: give exactly one of t_half / ke`);
  }
}

/**
 * The report lane's pre-flight (the route's own bounds): the theme
 * must be the catalog's own, the tables list (when given) `1…5`.
 */
export function assertReportBounds(
  what: string,
  theme: string,
  reportThemes: readonly string[],
  tables: readonly Record<string, unknown>[] | null,
): void {
  if (!reportThemes.includes(theme)) {
    throw new Error(
      `${what}: unknown theme ${theme}; see GET /api/v1/visualization/catalog`,
    );
  }
  if (tables !== null) {
    if (tables.length < 1) {
      throw new Error(`${what}: the tables list must be non-empty when given`);
    }
    if (tables.length > REPORT_MAX_TABLES) {
      throw new Error(
        `${what}: ${tables.length} tables, over the ${REPORT_MAX_TABLES} cap`,
      );
    }
  }
}
