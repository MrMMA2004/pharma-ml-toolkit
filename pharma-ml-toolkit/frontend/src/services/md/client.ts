/**
 * The Molecular Dynamics Analysis client — part11 §11.7's workflow
 * service (trajectory ANALYSIS, never simulation) over the REAL
 * backend.
 *
 * The backend contract (mirrored field-for-field from the route
 * source `routes/md.py`; the request bodies come from the generated
 * OpenAPI types — the same one-DTO-source rule the Python drift
 * guards in `tests/test_md_workspace.py` pin for this client):
 *
 * * `GET /api/v1/md/catalog` — the route inventory (five groups:
 *   trajectory / xvg / enm / dimred / study, each naming the backend
 *   functions the group wraps).
 * * `POST /api/v1/md/synthetic` — `SyntheticRequest` (the seeded
 *   deterministic demo trajectory — alpha-helix + flexible loop —
 *   followed by the REAL metric bundle: `rmsd_trajectory`, `rmsf`,
 *   `radius_of_gyration_trajectory`, `native_contact_fraction`, the
 *   `contact_map` occupancy summary and `trajectory_pca`).
 * * `POST /api/v1/md/analyze` — `MDAnalyzeRequest` (the same metric
 *   bundle over inline `frames x atoms x 3` coordinates; a
 *   non-3D or oversized array is the validator's own honest 422).
 * * `POST /api/v1/md/xvg` — `XvgRequest` (the REAL `parse_xvg` over
 *   pasted GROMACS `.xvg` text with `metric_stats` +
 *   `moving_average`).
 * * `POST /api/v1/md/enm` — `EnmRequest` (the REAL
 *   `anm_fluctuations` ANM over one `atoms x 3` coordinate set).
 * * `POST /api/v1/md/pca` — `PcaRequest` (standalone
 *   `trajectory_pca` with its projections).
 * * `POST /api/v1/md/study` — `StudyRequest` (the full bundle as a
 *   managed job: the synthetic trajectory, the metric suite, and
 *   two REAL figures rendered by `viz.md_plots.plot_md_series`,
 *   registered as downloadable artifacts; every stage reports
 *   honest `job.progress` events and cancelling between stages
 *   lands the job CANCELLED, never FAILED).
 *
 * The figure lane rides `POST /api/v1/visualization/md-series` (the
 * REAL `plot_md_series` over `.xvg` text — one series or several —
 * registered as a downloadable artifact, drawn only from points
 * that actually arrived); the export rides the rows-based
 * `POST /api/v1/visualization/table` — both posted from the generated
 * types, parsed here with `parseArtifactView`.
 *
 * Pre-flight bounds (the route's own pydantic bounds, enforced
 * before the request fires so the page never meets a 422 it did
 * not anticipate): frames `1…2000`, residues `1…100`, atoms `<= 300`
 * and elements `<= 240000` for the inline lane, ENM atoms `3…2000`,
 * PCA frames `2…2000` with `1…10` components, and the `.xvg` text
 * `<= 2_000_000` chars with the moving-average window `1…101`. The
 * scalar value ranges stay the backend's own honest 422s — they are
 * scientific verdicts, not transport errors.
 */

import { api } from "@/services/api/client";
import { parseArtifactView, type ArtifactView } from "@/services/artifacts/client";
import { type JobView, parseJobView } from "@/services/jobs/types";
import type { components } from "@/types/api.d";

type SyntheticRequest = components["schemas"]["SyntheticRequest"];
type MDAnalyzeRequest = components["schemas"]["MDAnalyzeRequest"];
type XvgRequest = components["schemas"]["XvgRequest"];
type EnmRequest = components["schemas"]["EnmRequest"];
type PcaRequest = components["schemas"]["PcaRequest"];
type StudyRequest = components["schemas"]["StudyRequest"];
type MdSeriesRequest = components["schemas"]["MdSeriesRequest"];
type VisualizationTableRequest =
  components["schemas"]["VisualizationTableRequest"];

// -- the response views (hand-written, runtime-narrowed) --------------

/** One `/catalog` group row — the route's own keys. */
export interface MdCatalogGroup {
  readonly group: string;
  readonly routes: readonly string[];
  readonly backend: string;
}

/** The `GET /api/v1/md/catalog` answer. */
export interface MdCatalogResponse {
  readonly groups: readonly MdCatalogGroup[];
}

/** The `contact_map` occupancy summary — the route's own block. */
export interface ContactMapSummaryView {
  readonly n_pairs: number;
  readonly strong_pairs: number;
  readonly strong_pair_fraction: number;
}

/** The trajectory PCA block — the route's own keys. */
export interface TrajectoryPcaView {
  readonly method: string;
  readonly explained_variance_ratio: readonly number[];
  readonly projections: readonly (readonly number[])[];
}

/**
 * The REAL metric bundle both trajectory lanes return
 * (`_metric_bundle`'s own keys; `source` distinguishes the seeded
 * demo path from the inline coordinate path).
 */
export interface MetricBundleView {
  readonly source: "synthetic" | "inline";
  readonly n_frames: number;
  readonly n_atoms: number;
  readonly rmsd: readonly number[];
  readonly rmsf: readonly number[];
  readonly radius_of_gyration: readonly number[];
  readonly native_contact_fraction: readonly number[];
  readonly contact_map: ContactMapSummaryView;
  readonly pca: TrajectoryPcaView;
}

/** `metric_stats()` — mean / population std / cv (all null-aware). */
export interface XvgStatsView {
  readonly mean: number | null;
  readonly std: number | null;
  readonly cv: number | null;
}

/** The `POST /api/v1/md/xvg` answer. */
export interface XvgResponse {
  readonly source: string;
  readonly label: string;
  readonly n_points: number;
  readonly time: readonly number[];
  readonly values: readonly number[];
  readonly stats: XvgStatsView;
  readonly moving_average: readonly number[] | null;
}

/** The `POST /api/v1/md/enm` answer. */
export interface EnmResponse {
  readonly n_atoms: number;
  readonly cutoff: number;
  readonly gamma: number;
  readonly fluctuations: readonly number[];
}

/** The `POST /api/v1/md/pca` answer. */
export interface PcaResponse {
  readonly method: string;
  readonly n_frames: number;
  readonly n_components: number;
  readonly explained_variance_ratio: readonly number[];
  readonly projections: readonly (readonly number[])[];
}

/** The study body's metric block — `_study_body`'s own keys. */
export interface StudyMetricsView {
  readonly n_frames: number;
  readonly n_atoms: number;
  readonly rmsd_first: number;
  readonly rmsd_last: number;
  readonly rmsf_max: number;
  readonly rg_first: number;
  readonly rg_last: number;
  readonly native_contact_fraction_last: number;
  readonly pca_explained_variance_ratio: readonly number[];
}

/**
 * The study body — the shape BOTH lanes return (the sync answer and
 * the COMPLETED job's REST `result` payload; the page relays the
 * authoritative REST result through the jobs client, never the WS
 * event payload alone).
 */
export interface StudyBodyView {
  readonly title: string;
  readonly seed: number;
  readonly metrics: StudyMetricsView;
  readonly artifacts: readonly ArtifactView[];
}

/** The sync-mode study answer. */
export interface StudySyncResponse extends StudyBodyView {
  readonly mode: "sync";
}

/** The job-mode study answer. */
export interface StudyJobResponse {
  readonly mode: "job";
  readonly job: JobView;
}

export type StudyResponse = StudySyncResponse | StudyJobResponse;

/**
 * One `job.progress` envelope payload — the study lane's REAL
 * per-stage telemetry (`JobManager.report_progress`'s own
 * `{progress, stage}` dict; progress-only, never the result).
 */
export interface JobProgressPayload {
  readonly progress: number;
  readonly stage: string;
}

// -- the option payloads ----------------------------------------------

export interface SyntheticOptions {
  readonly nFrames: number;
  readonly seed: number;
  readonly nResidues: number;
  readonly noise: number;
  readonly driftRadians: number;
  readonly flexibleResidues: readonly number[] | null;
}

export interface AnalyzeOptions {
  /** The `frames x atoms x 3` coordinate stack (the route's own cap). */
  readonly coords: readonly (readonly (readonly number[])[])[];
  readonly masses: readonly number[] | null;
}

export interface XvgOptions {
  readonly text: string;
  readonly movingAverageWindow: number;
}

export interface EnmOptions {
  /** One `atoms x 3` coordinate set (the route's own bounds). */
  readonly coords: readonly (readonly number[])[];
  readonly cutoff: number;
  readonly gamma: number;
}

export interface PcaOptions {
  readonly coords: readonly (readonly (readonly number[])[])[];
  readonly nComponents: number;
}

export interface StudyOptions {
  readonly nFrames: number;
  readonly seed: number;
  readonly nResidues: number;
  readonly noise: number;
  readonly driftRadians: number;
  readonly title: string;
  readonly mode: "job" | "sync";
}

/** The md-series figure lane's named `.xvg` series + labels. */
export interface MdSeriesFigureOptions {
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

// -- the md-series figure answer view ----------------------------------

/** One named series' stats row — the route's own `series` block. */
export interface MdSeriesStatView {
  readonly n_points: number;
  readonly label: string;
  readonly stats: XvgStatsView;
}

/** The `POST /api/v1/visualization/md-series` answer. */
export interface MdSeriesFigureResponse {
  readonly n_series: number;
  readonly series: Readonly<Record<string, MdSeriesStatView>>;
  readonly plot: Readonly<Record<string, unknown>> | null;
  readonly artifact: ArtifactView;
}

// -- the api object ----------------------------------------------------

export const mdApi = {
  /** `GET /api/v1/md/catalog` — the route inventory. */
  async catalog(): Promise<MdCatalogResponse> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/md/catalog",
    );
    return parseMdCatalog(raw, "md catalog");
  },

  /**
   * `POST /api/v1/md/synthetic` — the seeded deterministic demo
   * trajectory followed by the REAL metric suite. The backend's own
   * pydantic bounds (frames, residues, the flexible-residue window)
   * stay its honest 422s; the pre-flight mirrors only the list
   * bounds the page can see coming.
   */
  async synthetic(options: SyntheticOptions): Promise<MetricBundleView> {
    const request: SyntheticRequest = {
      n_frames: options.nFrames,
      seed: options.seed,
      n_residues: options.nResidues,
      noise: options.noise,
      drift_radians: options.driftRadians,
      flexible_residues:
        options.flexibleResidues === null
          ? null
          : [...options.flexibleResidues],
    };
    const raw = (await api.post(
      "/api/v1/md/synthetic",
      request,
    )) as Record<string, unknown>;
    return parseMetricBundle(raw, "synthetic");
  },

  /**
   * `POST /api/v1/md/analyze` — the same REAL metric bundle over
   * inline `frames x atoms x 3` coordinates (the route's own shape
   * verdict — a non-3D stack, an oversized atom count or a mismatched
   * masses vector is its honest 422, surfaced verbatim).
   */
  async analyze(options: AnalyzeOptions): Promise<MetricBundleView> {
    const request: MDAnalyzeRequest = {
      coords: options.coords.map((frame) =>
        frame.map((atom) => [...atom]),
      ),
      masses: options.masses === null ? null : [...options.masses],
    };
    const raw = (await api.post(
      "/api/v1/md/analyze",
      request,
    )) as Record<string, unknown>;
    return parseMetricBundle(raw, "analyze");
  },

  /**
   * `POST /api/v1/md/xvg` — the REAL `parse_xvg` + `metric_stats` +
   * `moving_average` over pasted GROMACS `.xvg` text. A malformed
   * body is the parser's own honest 422, surfaced verbatim.
   */
  async xvg(options: XvgOptions): Promise<XvgResponse> {
    const request: XvgRequest = {
      text: options.text,
      moving_average_window: options.movingAverageWindow,
    };
    const raw = (await api.post(
      "/api/v1/md/xvg",
      request,
    )) as Record<string, unknown>;
    return parseXvgResponse(raw, "xvg");
  },

  /**
   * `POST /api/v1/md/enm` — the REAL ANM `anm_fluctuations` over one
   * `atoms x 3` coordinate set (the cutoff/gamma ranges stay the
   * backend's own honest 422s).
   */
  async enm(options: EnmOptions): Promise<EnmResponse> {
    const request: EnmRequest = {
      coords: options.coords.map((atom) => [...atom]),
      cutoff: options.cutoff,
      gamma: options.gamma,
    };
    const raw = (await api.post(
      "/api/v1/md/enm",
      request,
    )) as Record<string, unknown>;
    return parseEnmResponse(raw, "enm");
  },

  /**
   * `POST /api/v1/md/pca` — the standalone REAL `trajectory_pca`
   * (the projections are the engine's own verdict, never a
   * client-side guess).
   */
  async pca(options: PcaOptions): Promise<PcaResponse> {
    const request: PcaRequest = {
      coords: options.coords.map((frame) =>
        frame.map((atom) => [...atom]),
      ),
      n_components: options.nComponents,
    };
    const raw = (await api.post(
      "/api/v1/md/pca",
      request,
    )) as Record<string, unknown>;
    return parsePcaResponse(raw, "pca");
  },

  /**
   * `POST /api/v1/md/study` — the full bundle as a managed job: the
   * synthetic trajectory, the REAL metric suite and two REAL figure
   * artifacts. The default job lane publishes honest `job.progress`
   * envelopes the tracker renders; the COMPLETED job's REST result
   * payload carries the same body the sync lane returns — the page
   * relays it through the jobs client (§13), never the
   * WS-event-payload-alone. Cancelling between stages lands CANCELLED,
   * never FAILED.
   */
  async study(options: StudyOptions): Promise<StudyResponse> {
    const request: StudyRequest = {
      n_frames: options.nFrames,
      seed: options.seed,
      n_residues: options.nResidues,
      noise: options.noise,
      drift_radians: options.driftRadians,
      title: options.title,
      mode: options.mode,
    };
    const raw = (await api.post(
      "/api/v1/md/study",
      request,
    )) as Record<string, unknown>;
    if (raw["mode"] === "sync") {
      return { mode: "sync", ...parseStudyBody(raw, "study(sync)") };
    }
    return { mode: "job", job: parseJobView(raw["job"], "study(job)") };
  },

  /**
   * `POST /api/v1/visualization/md-series` — the REAL `plot_md_series`
   * figure over parsed `.xvg` text (one named series or several),
   * registered as a downloadable artifact through the §17 four-format
   * export stack. The parsed series stats ride the answer's own
   * `series` block; the honest minimum is one non-empty series.
   */
  async mdSeries(
    options: MdSeriesFigureOptions,
  ): Promise<MdSeriesFigureResponse> {
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
    return parseMdSeriesFigureResponse(raw, "md-series");
  },

  /**
   * `POST /api/v1/visualization/table` — the workspace's rows as a
   * REAL table artifact (the §17 export stack). The backend's own
   * 5000-row cap is mirrored here — the error names both numbers,
   * never a silent truncation.
   */
  async exportTable(options: TableExportOptions): Promise<ArtifactView> {
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
    const artifact = raw["artifact"];
    if (typeof artifact !== "object" || artifact === null) {
      throw new Error("the table lane answered without an artifact");
    }
    return parseArtifactView(artifact, "table export");
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

/** A required finite-number array with a length window (labeled errors). */
function numberArray(
  raw: unknown,
  what: string,
  minLength: number,
  maxLength: number,
): readonly number[] {
  if (!Array.isArray(raw)) {
    throw new Error(`${what}: expected an array`);
  }
  if (raw.length < minLength) {
    throw new Error(`${what}: ${raw.length} points, at least ${minLength} required`);
  }
  if (raw.length > maxLength) {
    throw new Error(`${what}: ${raw.length} points, over the ${maxLength} cap`);
  }
  const out: number[] = [];
  for (let i = 0; i < raw.length; i += 1) {
    const v = asNumber(raw[i]);
    if (v === null) {
      throw new Error(`${what}[${i}]: not a finite number`);
    }
    out.push(v);
  }
  return out;
}

// -- the response parsers ------------------------------------------------

/** Narrow the `/catalog` answer. */
export function parseMdCatalog(
  raw: unknown,
  what: string,
): MdCatalogResponse {
  const r = expectObject(raw, what);
  const groups = r["groups"];
  if (!Array.isArray(groups)) {
    throw new Error(`${what}: groups is not an array`);
  }
  return {
    groups: groups.map((row, i) => {
      const g = expectObject(row, `${what}[${i}]`);
      const group = g["group"];
      if (typeof group !== "string") {
        throw new Error(`${what}[${i}]: group is not a string`);
      }
      const routes = g["routes"];
      if (!Array.isArray(routes)) {
        throw new Error(`${what}[${i}]: routes is not an array`);
      }
      return {
        group,
        routes: routes.map((route) => String(route)),
        backend: asString(g["backend"]),
      };
    }),
  };
}

/** Narrow the `metric_stats()` block (all three null-aware). */
export function parseXvgStats(raw: unknown, what: string): XvgStatsView {
  const r = expectObject(raw, what);
  return {
    mean: asNumberOrNull(r["mean"]),
    std: asNumberOrNull(r["std"]),
    cv: asNumberOrNull(r["cv"]),
  };
}

/** Narrow the `contact_map` occupancy summary block. */
function parseContactMapSummary(
  raw: unknown,
  what: string,
): ContactMapSummaryView {
  const r = expectObject(raw, what);
  const nPairs = asNumber(r["n_pairs"]);
  const strong = asNumber(r["strong_pairs"]);
  const fraction = asNumber(r["strong_pair_fraction"]);
  if (nPairs === null || strong === null || fraction === null) {
    throw new Error(
      `${what}: a contact-map summary field is not a finite number`,
    );
  }
  return {
    n_pairs: nPairs,
    strong_pairs: strong,
    strong_pair_fraction: fraction,
  };
}

/** Narrow the trajectory PCA block. */
function parseTrajectoryPca(raw: unknown, what: string): TrajectoryPcaView {
  const r = expectObject(raw, what);
  const method = r["method"];
  if (typeof method !== "string") {
    throw new Error(`${what}: method is not a string`);
  }
  const variance = numberArray(
    r["explained_variance_ratio"], `${what} explained_variance_ratio`, 1, 10,
  );
  const projectionsRaw = r["projections"];
  if (!Array.isArray(projectionsRaw)) {
    throw new Error(`${what}: projections is not an array`);
  }
  const projections = projectionsRaw.map((row, i) =>
    numberArray(row, `${what} projections[${i}]`, 1, 10),
  );
  return { method, explained_variance_ratio: variance, projections };
}

/**
 * Narrow the REAL metric bundle (both trajectory lanes' answer:
 * `source` distinguishes the seeded demo path from the inline
 * coordinate path).
 */
export function parseMetricBundle(
  raw: unknown,
  what: string,
): MetricBundleView {
  const r = expectObject(raw, what);
  const source = r["source"];
  if (source !== "synthetic" && source !== "inline") {
    throw new Error(`${what}: source is not "synthetic" or "inline"`);
  }
  const nFrames = asNumber(r["n_frames"]);
  const nAtoms = asNumber(r["n_atoms"]);
  if (nFrames === null || nAtoms === null) {
    throw new Error(`${what}: n_frames/n_atoms is not a finite number`);
  }
  const rmsd = numberArray(r["rmsd"], `${what} rmsd`, 1, 2000);
  const rmsf = numberArray(r["rmsf"], `${what} rmsf`, 1, 2000);
  const rg = numberArray(
    r["radius_of_gyration"], `${what} radius_of_gyration`, 1, 2000,
  );
  const q = numberArray(
    r["native_contact_fraction"], `${what} native_contact_fraction`, 1, 2000,
  );
  if (rmsd.length !== nFrames || rg.length !== nFrames || q.length !== nFrames) {
    throw new Error(
      `${what}: a per-frame series carries ${rmsd.length} points, the trajectory has ${nFrames} frames`,
    );
  }
  if (rmsf.length !== nAtoms) {
    throw new Error(
      `${what}: rmsf carries ${rmsf.length} entries, the trajectory has ${nAtoms} atoms`,
    );
  }
  return {
    source,
    n_frames: nFrames,
    n_atoms: nAtoms,
    rmsd,
    rmsf,
    radius_of_gyration: rg,
    native_contact_fraction: q,
    contact_map: parseContactMapSummary(r["contact_map"], `${what} contact_map`),
    pca: parseTrajectoryPca(r["pca"], `${what} pca`),
  };
}

/** Narrow the `POST /xvg` answer. */
export function parseXvgResponse(
  raw: unknown,
  what: string,
): XvgResponse {
  const r = expectObject(raw, what);
  const nPoints = asNumber(r["n_points"]);
  if (nPoints === null) {
    throw new Error(`${what}: n_points is not a finite number`);
  }
  const time = numberArray(r["time"], `${what} time`, 1, 10000);
  const values = numberArray(r["values"], `${what} values`, 1, 10000);
  if (time.length !== nPoints || values.length !== nPoints) {
    throw new Error(
      `${what}: the series carries ${time.length} points, the route counted ${nPoints}`,
    );
  }
  const moving = r["moving_average"];
  const movingAverage = moving === null
    ? null
    : numberArray(moving, `${what} moving_average`, 1, 10000);
  return {
    source: asString(r["source"]),
    label: asString(r["label"]),
    n_points: nPoints,
    time,
    values,
    stats: parseXvgStats(r["stats"], `${what} stats`),
    moving_average: movingAverage,
  };
}

/** Narrow the `POST /enm` answer. */
export function parseEnmResponse(
  raw: unknown,
  what: string,
): EnmResponse {
  const r = expectObject(raw, what);
  const nAtoms = asNumber(r["n_atoms"]);
  const cutoff = asNumber(r["cutoff"]);
  const gamma = asNumber(r["gamma"]);
  if (nAtoms === null || cutoff === null || gamma === null) {
    throw new Error(`${what}: n_atoms/cutoff/gamma is not a finite number`);
  }
  const fluctuations = numberArray(
    r["fluctuations"], `${what} fluctuations`, 1, 2000,
  );
  if (fluctuations.length !== nAtoms) {
    throw new Error(
      `${what}: fluctuations carries ${fluctuations.length} entries, the coordinate set has ${nAtoms} atoms`,
    );
  }
  return { n_atoms: nAtoms, cutoff, gamma, fluctuations };
}

/** Narrow the `POST /pca` answer. */
export function parsePcaResponse(
  raw: unknown,
  what: string,
): PcaResponse {
  const r = expectObject(raw, what);
  const method = r["method"];
  if (typeof method !== "string") {
    throw new Error(`${what}: method is not a string`);
  }
  const nFrames = asNumber(r["n_frames"]);
  const nComponents = asNumber(r["n_components"]);
  if (nFrames === null || nComponents === null) {
    throw new Error(`${what}: n_frames/n_components is not a finite number`);
  }
  const variance = numberArray(
    r["explained_variance_ratio"], `${what} explained_variance_ratio`, 1, 10,
  );
  if (variance.length !== nComponents) {
    throw new Error(
      `${what}: explained_variance_ratio carries ${variance.length} entries, n_components is ${nComponents}`,
    );
  }
  const projectionsRaw = r["projections"];
  if (!Array.isArray(projectionsRaw)) {
    throw new Error(`${what}: projections is not an array`);
  }
  const projections = projectionsRaw.map((row, i) =>
    numberArray(row, `${what} projections[${i}]`, 1, 10),
  );
  if (projections.length !== nFrames) {
    throw new Error(
      `${what}: projections carries ${projections.length} rows, the trajectory has ${nFrames} frames`,
    );
  }
  return { method, n_frames: nFrames, n_components: nComponents,
    explained_variance_ratio: variance, projections };
}

/** Narrow the study body (the sync answer / the job REST result). */
export function parseStudyBody(
  raw: unknown,
  what: string,
): StudyBodyView {
  const r = expectObject(raw, what);
  const title = r["title"];
  if (typeof title !== "string") {
    throw new Error(`${what}: title is not a string`);
  }
  const seed = asNumber(r["seed"]);
  if (seed === null) {
    throw new Error(`${what}: seed is not a finite number`);
  }
  const m = expectObject(r["metrics"], `${what} metrics`);
  const nFrames = asNumber(m["n_frames"]);
  const nAtoms = asNumber(m["n_atoms"]);
  const rmsdFirst = asNumber(m["rmsd_first"]);
  const rmsdLast = asNumber(m["rmsd_last"]);
  const rmsfMax = asNumber(m["rmsf_max"]);
  const rgFirst = asNumber(m["rg_first"]);
  const rgLast = asNumber(m["rg_last"]);
  const qLast = asNumber(m["native_contact_fraction_last"]);
  if (
    nFrames === null || nAtoms === null || rmsdFirst === null ||
    rmsdLast === null || rmsfMax === null || rgFirst === null ||
    rgLast === null || qLast === null
  ) {
    throw new Error(`${what}: a study metric is not a finite number`);
  }
  const variance = numberArray(
    m["pca_explained_variance_ratio"],
    `${what} metrics.pca_explained_variance_ratio`,
    1, 10,
  );
  const artifactsRaw = r["artifacts"];
  if (!Array.isArray(artifactsRaw)) {
    throw new Error(`${what}: artifacts is not an array`);
  }
  const artifacts = artifactsRaw.map((item, i) =>
    parseArtifactView(item, `${what} artifacts[${i}]`),
  );
  return {
    title,
    seed,
    metrics: {
      n_frames: nFrames,
      n_atoms: nAtoms,
      rmsd_first: rmsdFirst,
      rmsd_last: rmsdLast,
      rmsf_max: rmsfMax,
      rg_first: rgFirst,
      rg_last: rgLast,
      native_contact_fraction_last: qLast,
      pca_explained_variance_ratio: variance,
    },
    artifacts,
  };
}

/**
 * Narrow one `job.progress` envelope payload — the study lane's
 * per-stage telemetry (progress-only; the authoritative result rides
 * the REST jobs client, §13).
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

/** Narrow the `POST /visualization/md-series` answer. */
export function parseMdSeriesFigureResponse(
  raw: unknown,
  what: string,
): MdSeriesFigureResponse {
  const r = expectObject(raw, what);
  const nSeries = asNumber(r["n_series"]);
  if (nSeries === null) {
    throw new Error(`${what}: n_series is not a finite number`);
  }
  const seriesRaw = expectObject(r["series"], `${what} series`);
  const series: Record<string, MdSeriesStatView> = {};
  for (const [name, row] of Object.entries(seriesRaw)) {
    const s = expectObject(row, `${what} series[${name}]`);
    const nPoints = asNumber(s["n_points"]);
    if (nPoints === null) {
      throw new Error(`${what} series[${name}]: n_points is not a finite number`);
    }
    series[name] = {
      n_points: nPoints,
      label: asString(s["label"]),
      stats: parseXvgStats(s["stats"], `${what} series[${name}] stats`),
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

// -- the pre-flight bound helpers (the page calls these before the POST)

/** The route's own trajectory frames/residues caps (`routes/md.py`). */
export const MD_MAX_FRAMES = 2000;
/** The route's own inline atom cap (`_MAX_ATOMS`). */
export const MD_MAX_ATOMS = 300;
/** The route's own inline element cap (`_MAX_COORD_ELEMENTS`). */
export const MD_MAX_COORD_ELEMENTS = 240_000;
/** The route's own xvg text cap (`XvgRequest.max_length`). */
export const MD_MAX_XVG_CHARS = 2_000_000;

/**
 * The synthetic lane's pre-flight (the route's own list bounds):
 * frames `1…2000`, residues `1…100`, the flexible-residue list
 * non-empty and `<= 50` entries when given.
 */
export function assertSyntheticBounds(
  what: string,
  nFrames: number,
  nResidues: number,
  flexibleResidues: readonly number[] | null,
): void {
  if (!Number.isInteger(nFrames) || nFrames < 1 || nFrames > MD_MAX_FRAMES) {
    throw new Error(
      `${what}: n_frames must be an integer in 1…${MD_MAX_FRAMES}`,
    );
  }
  if (!Number.isInteger(nResidues) || nResidues < 1 || nResidues > 100) {
    throw new Error(`${what}: n_residues must be an integer in 1…100`);
  }
  if (flexibleResidues !== null) {
    if (flexibleResidues.length === 0) {
      throw new Error(
        `${what}: flexible_residues must be non-empty when given (the route's own verdict)`,
      );
    }
    if (flexibleResidues.length > 50) {
      throw new Error(
        `${what}: flexible_residues carries ${flexibleResidues.length} entries, over the 50 cap`,
      );
    }
  }
}

/**
 * The inline coordinate stack's pre-flight (the route's own shape +
 * size bounds): a `frames x atoms x 3` stack with atoms `<= 300` and
 * `<= 240000` total elements, masses (when given) exactly one per
 * atom.
 */
export function assertCoordsBounds(
  what: string,
  coords: readonly (readonly (readonly number[])[])[],
  masses: readonly number[] | null,
): void {
  if (coords.length < 1) {
    throw new Error(`${what}: the coordinate stack carries no frames`);
  }
  if (coords.length > MD_MAX_FRAMES) {
    throw new Error(
      `${what}: ${coords.length} frames, over the ${MD_MAX_FRAMES} cap`,
    );
  }
  let nAtoms = -1;
  for (let f = 0; f < coords.length; f += 1) {
    const frame = coords[f];
    if (!Array.isArray(frame) || frame.length === 0) {
      throw new Error(`${what}: frame ${f} carries no atoms`);
    }
    if (nAtoms === -1) {
      nAtoms = frame.length;
    } else if (frame.length !== nAtoms) {
      throw new Error(
        `${what}: frame ${f} carries ${frame.length} atoms, frame 0 carries ${nAtoms} (a ragged stack)`,
      );
    }
    for (let a = 0; a < frame.length; a += 1) {
      const atom = frame[a];
      if (!Array.isArray(atom) || atom.length !== 3) {
        throw new Error(
          `${what}: frame ${f} atom ${a} is not an x/y/z triple`,
        );
      }
    }
  }
  if (nAtoms > MD_MAX_ATOMS) {
    throw new Error(
      `${what}: ${nAtoms} atoms per frame, over the ${MD_MAX_ATOMS} cap (use the study lane for larger systems)`,
    );
  }
  if (coords.length * nAtoms > MD_MAX_COORD_ELEMENTS) {
    throw new Error(
      `${what}: ${coords.length * nAtoms} coordinate elements, over the ${MD_MAX_COORD_ELEMENTS} cap`,
    );
  }
  if (masses !== null && masses.length !== nAtoms) {
    throw new Error(
      `${what}: masses carries ${masses.length} entries, the stack has ${nAtoms} atoms`,
    );
  }
}

/**
 * The ENM lane's pre-flight (the route's own bounds): an
 * `atoms x 3` coordinate set with atoms `3…2000`.
 */
export function assertEnmBounds(
  what: string,
  coords: readonly (readonly number[])[],
): void {
  if (coords.length < 3) {
    throw new Error(
      `${what}: ${coords.length} atoms, at least 3 required`,
    );
  }
  if (coords.length > 2000) {
    throw new Error(`${what}: ${coords.length} atoms, over the 2000 cap`);
  }
  for (let a = 0; a < coords.length; a += 1) {
    const atom = coords[a];
    if (!Array.isArray(atom) || atom.length !== 3) {
      throw new Error(`${what}: atom ${a} is not an x/y/z triple`);
    }
  }
}

/**
 * The PCA lane's pre-flight (the route's own bounds): a
 * `frames x atoms x 3` stack with frames `2…2000` and components
 * `1…10`.
 */
export function assertPcaBounds(
  what: string,
  coords: readonly (readonly (readonly number[])[])[],
  nComponents: number,
): void {
  if (coords.length < 2) {
    throw new Error(
      `${what}: ${coords.length} frames, at least 2 required`,
    );
  }
  if (coords.length > MD_MAX_FRAMES) {
    throw new Error(
      `${what}: ${coords.length} frames, over the ${MD_MAX_FRAMES} cap`,
    );
  }
  if (!Number.isInteger(nComponents) || nComponents < 1 || nComponents > 10) {
    throw new Error(`${what}: n_components must be an integer in 1…10`);
  }
}

/**
 * The `.xvg` lane's pre-flight (the route's own bounds): non-empty
 * text under the 2,000,000-char cap, the moving-average window an
 * integer in `1…101`.
 */
export function assertXvgBounds(
  what: string,
  text: string,
  movingAverageWindow: number,
): void {
  if (text.length === 0) {
    throw new Error(`${what}: the .xvg text is empty`);
  }
  if (text.length > MD_MAX_XVG_CHARS) {
    throw new Error(
      `${what}: ${text.length} characters, over the ${MD_MAX_XVG_CHARS} cap`,
    );
  }
  if (
    !Number.isInteger(movingAverageWindow) ||
    movingAverageWindow < 1 || movingAverageWindow > 101
  ) {
    throw new Error(
      `${what}: the moving-average window must be an integer in 1…101`,
    );
  }
}
