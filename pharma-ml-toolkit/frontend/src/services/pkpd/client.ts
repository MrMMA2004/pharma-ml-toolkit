/**
 * The PK/PD client — part11 §11.6's workflow service (the
 * non-compartmental analysis, compartment models, minimal PBPK,
 * sigmoid Emax pharmacodynamics and population simulation surface
 * over the REAL backend).
 *
 * The backend contract (mirrored field-for-field from the route source
 * `routes/pkpd.py`; the request bodies come from the generated OpenAPI
 * types — the same one-DTO-source rule the Python drift guards in
 * `tests/test_pkpd_workspace.py` pin for this client):
 *
 * * `GET /api/v1/pkpd/catalog` — the route inventory (four groups:
 *   nca / models / pd / population, each naming the backend functions
 *   the group wraps).
 * * `POST /api/v1/pkpd/nca` — `NcaRequest` (the REAL `nca_summary` +
 *   both `auc_trapezoid` variants + the `fit_terminal_phase`
 *   regression; an invalid series is the validator's own honest 422).
 * * `POST /api/v1/pkpd/steady-state` — `SteadyStateRequest` (the
 *   interval NCA `steady_state_metrics` over one 0…τ profile).
 * * `POST /api/v1/pkpd/iv-bolus` — `IvBolusRequest` (the analytical
 *   one-compartment IV curve).
 * * `POST /api/v1/pkpd/bateman` — `BatemanRequest` (the oral
 *   one-compartment absorption/elimination curve).
 * * `POST /api/v1/pkpd/superposition` — `SuperpositionRequest` (the
 *   REAL `superposition` accumulation over the causal bateman unit
 *   response).
 * * `POST /api/v1/pkpd/pbpk` — `PbpkRequest` (the `MinimalPBPK`
 *   five-compartment perfusion-limited model; the frame's own columns
 *   `time, venous, lung, liver, kidney, rest, C_plasma`).
 * * `POST /api/v1/pkpd/pd` — `PdRequest` (the sigmoid Emax
 *   `effect_table` over a concentration grid).
 * * `POST /api/v1/pkpd/population` — `PopulationRequest` →
 *   `{mode: "job", job: JobView}` (the default lane; the route batches
 *   the subjects 50 per batch, each batch a REAL seeded
 *   `simulate_population` call, and publishes honest `job.progress`
 *   envelopes per batch — narrowed here by `parseJobProgressPayload`;
 *   the COMPLETED job's REST result payload is the same body the sync
 *   lane returns, and the page relays the authoritative REST result
 *   through the jobs client (§13), never the WS event payload alone) /
 *   `{mode: "sync", …}` (the population summary curves — mean / p5 /
 *   p95 — plus the first subject profiles). A negative omega is the
 *   route's own honest 422.
 *
 * The figure lanes (the concentration-time exposure profile and the
 * sigmoid Emax response curve — §11.6 stage seven's Exposure/Response
 * Visualization, drawn only from points that actually arrived) ride
 * `POST /api/v1/artifacts/figure`; the export rides the rows-based
 * `POST /api/v1/visualization/table` — both posted from the
 * generated types, parsed here with `parseArtifactView`.
 *
 * Pre-flight bounds (the route's own pydantic list bounds, enforced
 * before the request fires so the page never meets a 422 it did not
 * anticipate): the series lengths (`>= 2` for NCA/steady-state and
 * the population grid, `>= 1` for the model lanes, all `<= 10000`;
 * the PD grid `<= 2000`; the population grid `<= 2000`) and the
 * population vectors' exact lengths (`theta` 3, `omega_diag` 1). The
 * scalar value ranges stay the backend's own honest 422s — they are
 * scientific verdicts, not transport errors.
 */

import { api } from "@/services/api/client";
import { parseArtifactView, type ArtifactView } from "@/services/artifacts/client";
import { type JobView, parseJobView } from "@/services/jobs/types";
import type { components } from "@/types/api.d";

type NcaRequest = components["schemas"]["NcaRequest"];
type SteadyStateRequest = components["schemas"]["SteadyStateRequest"];
type IvBolusRequest = components["schemas"]["IvBolusRequest"];
type BatemanRequest = components["schemas"]["BatemanRequest"];
type SuperpositionRequest = components["schemas"]["SuperpositionRequest"];
type PbpkRequest = components["schemas"]["PbpkRequest"];
type PdRequest = components["schemas"]["PdRequest"];
type PopulationRequest = components["schemas"]["PopulationRequest"];
type FigureRequest = components["schemas"]["FigureRequest"];
type VisualizationTableRequest =
  components["schemas"]["VisualizationTableRequest"];

// -- the response views (hand-written, runtime-narrowed) --------------

/** One `/catalog` group row — the route's own keys. */
export interface PkpdCatalogGroup {
  readonly group: string;
  readonly routes: readonly string[];
  readonly backend: string;
}

/** The `GET /api/v1/pkpd/catalog` answer. */
export interface PkpdCatalogResponse {
  readonly groups: readonly PkpdCatalogGroup[];
}

/** `nca_summary().as_dict()` — the six non-compartmental metrics. */
export interface NcaSummaryView {
  readonly AUC: number;
  readonly AUMC: number;
  readonly MRT: number;
  readonly Cmax: number;
  readonly Tmax: number;
  readonly t_half_terminal: number;
}

/** `fit_terminal_phase` — the terminal-phase log-linear regression. */
export interface TerminalPhaseView {
  readonly start_index: number;
  readonly end_index: number;
  readonly n_points: number;
  readonly lambda_z: number;
  readonly r2: number;
  readonly r2_adj: number;
  readonly half_life: number;
  readonly clast_pred: number;
  readonly span_half_lives: number;
  readonly span_low: boolean;
  readonly method: string;
}

/** The `POST /api/v1/pkpd/nca` answer. */
export interface NcaResponse {
  readonly summary: NcaSummaryView;
  readonly auc_trapezoid_linear: number;
  readonly auc_trapezoid_log: number | null;
  readonly terminal: TerminalPhaseView;
  readonly n_points: number;
}

/** `steady_state_metrics().as_dict()` — the interval metrics. */
export interface SteadyStateMetricsView {
  readonly AUC_tau: number;
  readonly Cavg: number;
  readonly Cmax_ss: number;
  readonly Cmin_ss: number;
  readonly Ctrough: number;
  readonly fluctuation: number;
  readonly swing: number;
  readonly CL_ss: number | null;
  readonly R_accum: number | null;
}

/** The `POST /api/v1/pkpd/steady-state` answer. */
export interface SteadyStateResponse {
  readonly tau: number;
  readonly metrics: SteadyStateMetricsView;
  readonly n_points: number;
}

/** The `POST /api/v1/pkpd/iv-bolus` answer. */
export interface IvBolusResponse {
  readonly c0: number;
  readonly ke: number;
  readonly n_points: number;
  readonly concentration: readonly number[];
}

/** The `POST /api/v1/pkpd/bateman` answer. */
export interface BatemanResponse {
  readonly dose_per_volume: number;
  readonly ka: number;
  readonly ke: number;
  readonly n_points: number;
  readonly concentration: readonly number[];
}

/** The `POST /api/v1/pkpd/superposition` answer. */
export interface SuperpositionResponse {
  readonly tau: number;
  readonly n_doses: number;
  readonly n_points: number;
  readonly concentration: readonly number[];
}

/**
 * One PBPK frame row — `time` plus one column per compartment and
 * the derived plasma concentration (`MinimalPBPK.COMPARTMENTS` is the
 * route's own tuple; `C_plasma` is the frame's own inserted column,
 * always present).
 */
export interface PbpkRow {
  readonly time: number;
  readonly C_plasma: number;
  readonly [compartment: string]: number;
}

/** The `POST /api/v1/pkpd/pbpk` answer. */
export interface PbpkResponse {
  readonly route: "iv" | "oral";
  readonly dose_mg: number;
  readonly compartments: readonly string[];
  readonly n_points: number;
  readonly rows: readonly PbpkRow[];
}

/** The `POST /api/v1/pkpd/pd` answer (the sigmoid Emax table). */
export interface PdResponse {
  readonly e0: number;
  readonly emax: number;
  readonly ec50: number;
  readonly hill: number;
  readonly n_points: number;
  readonly concentration: readonly number[];
  readonly effect: readonly number[];
}

/** The population summary curves (mean / p5 / p95 per time point). */
export interface PopulationSummaryView {
  readonly mean: readonly number[];
  readonly p5: readonly number[];
  readonly p95: readonly number[];
}

/**
 * The population body — the shape BOTH lanes return (the sync answer
 * and the COMPLETED job's REST `result` payload; the page relays the
 * authoritative REST result through the jobs client, never the WS
 * event payload alone).
 */
export interface PopulationBodyView {
  readonly n_subjects: number;
  readonly n_times: number;
  readonly seed: number;
  readonly summary: PopulationSummaryView;
  readonly subject_profiles: readonly (readonly number[])[];
}

/** The sync-mode population answer. */
export interface PopulationSyncResponse extends PopulationBodyView {
  readonly mode: "sync";
}

/** The job-mode population answer. */
export interface PopulationJobResponse {
  readonly mode: "job";
  readonly job: JobView;
}

export type PopulationResponse =
  | PopulationSyncResponse
  | PopulationJobResponse;

/**
 * One `job.progress` envelope payload — the population lane's REAL
 * per-batch telemetry (`JobManager.report_progress`'s own
 * `{progress, stage}` dict; progress-only, never the result).
 */
export interface JobProgressPayload {
  readonly progress: number;
  readonly stage: string;
}

// -- the option payloads ----------------------------------------------

export interface NcaOptions {
  readonly times: readonly number[];
  readonly conc: readonly number[];
}

export interface SteadyStateOptions extends NcaOptions {
  readonly tau: number;
  readonly dose: number | null;
  readonly lambdaZ: number | null;
}

export interface IvBolusOptions {
  readonly c0: number;
  readonly ke: number;
  readonly times: readonly number[];
}

export interface BatemanOptions {
  readonly dosePerVolume: number;
  readonly ka: number;
  readonly ke: number;
  readonly times: readonly number[];
}

export interface SuperpositionOptions extends BatemanOptions {
  readonly tau: number;
  readonly nDoses: number;
}

export interface PbpkOptions {
  readonly times: readonly number[];
  readonly doseMg: number;
  readonly route: "iv" | "oral";
  readonly ka: number;
  readonly f: number;
  readonly bw: number;
  readonly clint: number;
  readonly fu: number;
  readonly bp: number;
  readonly kpLung: number;
  readonly kpLiver: number;
  readonly kpKidney: number;
  readonly kpRest: number;
}

export interface PdOptions {
  readonly concentrations: readonly number[];
  readonly e0: number;
  readonly emax: number;
  readonly ec50: number;
  readonly hill: number;
}

export interface PopulationOptions {
  readonly theta: readonly number[];
  readonly omegaDiag: readonly number[];
  readonly times: readonly number[];
  readonly nSubjects: number;
  readonly seed: number;
  readonly sigmaAdd: number;
  readonly sigmaProp: number;
  readonly nProfiles: number;
  readonly mode: "job" | "sync";
}

/** The concentration-time exposure figure's curve + format. */
export interface ExposureFigureOptions {
  readonly times: readonly number[];
  readonly conc: readonly number[];
  readonly title: string;
  readonly filename: string;
  readonly format: "PNG" | "SVG" | "TIFF" | "JPEG";
}

/** The sigmoid Emax response figure's table + format. */
export interface ResponseFigureOptions {
  readonly concentrations: readonly number[];
  readonly effect: readonly number[];
  readonly title: string;
  readonly filename: string;
  readonly format: "PNG" | "SVG" | "TIFF" | "JPEG";
}

export interface TableExportOptions {
  /** The rows the export carries (the backend's own 5000-row cap). */
  readonly rows: Readonly<Record<string, unknown>[]>;
  /** The backend's own export enum (visualization.py's own literals). */
  readonly format: "csv" | "tsv" | "xlsx" | "md" | "json" | "html";
  readonly filename: string;
}

// -- the api object ----------------------------------------------------

export const pkpdApi = {
  /** `GET /api/v1/pkpd/catalog` — the route inventory. */
  async catalog(): Promise<PkpdCatalogResponse> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/pkpd/catalog",
    );
    return parsePkpdCatalog(raw, "pkpd catalog");
  },

  /**
   * `POST /api/v1/pkpd/nca` — the REAL non-compartmental analysis:
   * `nca_summary` + both `auc_trapezoid` variants + the terminal-phase
   * fit. An invalid series (negative concentrations, non-finite
   * values, mismatched lengths) is the validator's own honest 422,
   * surfaced verbatim.
   */
  async nca(options: NcaOptions): Promise<NcaResponse> {
    const request: NcaRequest = {
      times: [...options.times],
      conc: [...options.conc],
    };
    const raw = (await api.post(
      "/api/v1/pkpd/nca",
      request,
    )) as Record<string, unknown>;
    return parseNcaResponse(raw, "nca");
  },

  /**
   * `POST /api/v1/pkpd/steady-state` — the interval NCA over one
   * steady-state 0…τ profile (legitimate zeros — troughs and
   * not-yet-absorbed segments — are data here; only the backend's own
   * DataValidationError 422 states the rest).
   */
  async steadyState(
    options: SteadyStateOptions,
  ): Promise<SteadyStateResponse> {
    const request: SteadyStateRequest = {
      times: [...options.times],
      conc: [...options.conc],
      tau: options.tau,
      dose: options.dose,
      lambda_z: options.lambdaZ,
    };
    const raw = (await api.post(
      "/api/v1/pkpd/steady-state",
      request,
    )) as Record<string, unknown>;
    return parseSteadyStateResponse(raw, "steady-state");
  },

  /** `POST /api/v1/pkpd/iv-bolus` — the analytical IV curve. */
  async ivBolus(options: IvBolusOptions): Promise<IvBolusResponse> {
    const request: IvBolusRequest = {
      c0: options.c0,
      ke: options.ke,
      times: [...options.times],
    };
    const raw = (await api.post(
      "/api/v1/pkpd/iv-bolus",
      request,
    )) as Record<string, unknown>;
    return parseIvBolusResponse(raw, "iv-bolus");
  },

  /** `POST /api/v1/pkpd/bateman` — the oral one-compartment curve. */
  async bateman(options: BatemanOptions): Promise<BatemanResponse> {
    const request: BatemanRequest = {
      dose_per_volume: options.dosePerVolume,
      ka: options.ka,
      ke: options.ke,
      times: [...options.times],
    };
    const raw = (await api.post(
      "/api/v1/pkpd/bateman",
      request,
    )) as Record<string, unknown>;
    return parseBatemanResponse(raw, "bateman");
  },

  /**
   * `POST /api/v1/pkpd/superposition` — the REAL dose accumulation
   * over the causal bateman unit response (the clamped time-since-dose
   * is the route's own honest causal continuation).
   */
  async superposition(
    options: SuperpositionOptions,
  ): Promise<SuperpositionResponse> {
    const request: SuperpositionRequest = {
      dose_per_volume: options.dosePerVolume,
      ka: options.ka,
      ke: options.ke,
      times: [...options.times],
      tau: options.tau,
      n_doses: options.nDoses,
    };
    const raw = (await api.post(
      "/api/v1/pkpd/superposition",
      request,
    )) as Record<string, unknown>;
    return parseSuperpositionResponse(raw, "superposition");
  },

  /**
   * `POST /api/v1/pkpd/pbpk` — the `MinimalPBPK` five-compartment
   * perfusion-limited model (the frame's own columns, verbatim).
   */
  async pbpk(options: PbpkOptions): Promise<PbpkResponse> {
    const request: PbpkRequest = {
      times: [...options.times],
      dose_mg: options.doseMg,
      route: options.route,
      ka: options.ka,
      f: options.f,
      bw: options.bw,
      clint: options.clint,
      fu: options.fu,
      bp: options.bp,
      kp_lung: options.kpLung,
      kp_liver: options.kpLiver,
      kp_kidney: options.kpKidney,
      kp_rest: options.kpRest,
    };
    const raw = (await api.post(
      "/api/v1/pkpd/pbpk",
      request,
    )) as Record<string, unknown>;
    return parsePbpkResponse(raw, "pbpk");
  },

  /**
   * `POST /api/v1/pkpd/pd` — the sigmoid Emax `effect_table` over the
   * concentration grid (the response curve is the engine's own
   * verdict, never a client-side guess).
   */
  async pd(options: PdOptions): Promise<PdResponse> {
    const request: PdRequest = {
      concentrations: [...options.concentrations],
      e0: options.e0,
      emax: options.emax,
      ec50: options.ec50,
      hill: options.hill,
    };
    const raw = (await api.post(
      "/api/v1/pkpd/pd",
      request,
    )) as Record<string, unknown>;
    return parsePdResponse(raw, "pd");
  },

  /**
   * `POST /api/v1/pkpd/population` — the REAL seeded population
   * simulation with IIV on `ka`. The default job lane batches the
   * subjects (50 per batch, each batch its own seeded call) and
   * publishes honest `job.progress` envelopes the tracker renders;
   * the COMPLETED job's REST result payload carries the same body the
   * sync lane returns — the page relays it through the jobs client
   * (§13), never the WS event payload alone. A negative omega is the route's
   * own honest 422, surfaced verbatim.
   */
  async population(
    options: PopulationOptions,
  ): Promise<PopulationResponse> {
    const request: PopulationRequest = {
      theta: [...options.theta],
      omega_diag: [...options.omegaDiag],
      times: [...options.times],
      n_subjects: options.nSubjects,
      seed: options.seed,
      sigma_add: options.sigmaAdd,
      sigma_prop: options.sigmaProp,
      n_profiles: options.nProfiles,
      mode: options.mode,
    };
    const raw = (await api.post(
      "/api/v1/pkpd/population",
      request,
    )) as Record<string, unknown>;
    if (raw["mode"] === "sync") {
      return { mode: "sync", ...parsePopulationBody(raw, "population(sync)") };
    }
    return { mode: "job", job: parseJobView(raw["job"], "population(job)") };
  },

  /**
   * `POST /api/v1/artifacts/figure` — the concentration-time exposure
   * profile as one REAL scatter figure (x = the time grid, y = the
   * concentrations the backend actually computed — drawn only from
   * points that arrived; the honest minimum is two finite points).
   */
  async exposureFigure(
    options: ExposureFigureOptions,
  ): Promise<ArtifactView> {
    const x: number[] = [];
    const y: number[] = [];
    const n = Math.min(options.times.length, options.conc.length);
    for (let i = 0; i < n; i += 1) {
      const t = options.times[i] as number | undefined;
      const c = options.conc[i] as number | undefined;
      if (t !== undefined && c !== undefined && Number.isFinite(t) && Number.isFinite(c)) {
        x.push(t);
        y.push(c);
      }
    }
    if (x.length < 2) {
      throw new Error(
        "an exposure figure needs at least two finite time/concentration points",
      );
    }
    const request: FigureRequest = {
      kind: "scatter",
      x,
      y,
      title: options.title,
      format: options.format,
      dpi: 150,
      filename: options.filename,
    };
    const raw = (await api.post(
      "/api/v1/artifacts/figure",
      request,
    )) as Record<string, unknown>;
    const artifact = raw["artifact"];
    if (typeof artifact !== "object" || artifact === null) {
      throw new Error("the figure lane answered without an artifact");
    }
    return parseArtifactView(artifact, "exposure figure");
  },

  /**
   * `POST /api/v1/artifacts/figure` — the sigmoid Emax response curve
   * as one REAL scatter figure (x = the concentration grid, y = the
   * effect the engine computed — the EC50 midpoint visible as the
   * curve's own bend, never an annotation the client invented).
   */
  async responseFigure(
    options: ResponseFigureOptions,
  ): Promise<ArtifactView> {
    const x: number[] = [];
    const y: number[] = [];
    const n = Math.min(options.concentrations.length, options.effect.length);
    for (let i = 0; i < n; i += 1) {
      const c = options.concentrations[i] as number | undefined;
      const e = options.effect[i] as number | undefined;
      if (c !== undefined && e !== undefined && Number.isFinite(c) && Number.isFinite(e)) {
        x.push(c);
        y.push(e);
      }
    }
    if (x.length < 2) {
      throw new Error(
        "a response figure needs at least two finite concentration/effect points",
      );
    }
    const request: FigureRequest = {
      kind: "scatter",
      x,
      y,
      title: options.title,
      format: options.format,
      dpi: 150,
      filename: options.filename,
    };
    const raw = (await api.post(
      "/api/v1/artifacts/figure",
      request,
    )) as Record<string, unknown>;
    const artifact = raw["artifact"];
    if (typeof artifact !== "object" || artifact === null) {
      throw new Error("the figure lane answered without an artifact");
    }
    return parseArtifactView(artifact, "response figure");
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

/** The series pre-flight (equal lengths, `>= 2` points, `<= 10000`). */
function assertSeriesBounds(
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
  if (times.length > 10000) {
    throw new Error(
      `${what}: ${times.length} points, over the 10000 cap`,
    );
  }
}

// -- the response parsers ------------------------------------------------

/** Narrow the `/catalog` answer. */
export function parsePkpdCatalog(
  raw: unknown,
  what: string,
): PkpdCatalogResponse {
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

/** Narrow the `nca_summary().as_dict()` block. */
function parseNcaSummary(raw: unknown, what: string): NcaSummaryView {
  const r = expectObject(raw, what);
  const AUC = asNumber(r["AUC"]);
  const AUMC = asNumber(r["AUMC"]);
  const MRT = asNumber(r["MRT"]);
  const Cmax = asNumber(r["Cmax"]);
  const Tmax = asNumber(r["Tmax"]);
  const halfTerminal = asNumber(r["t_half_terminal"]);
  if (
    AUC === null || AUMC === null || MRT === null ||
    Cmax === null || Tmax === null || halfTerminal === null
  ) {
    throw new Error(`${what}: a summary metric is not a finite number`);
  }
  return {
    AUC,
    AUMC,
    MRT,
    Cmax,
    Tmax,
    t_half_terminal: halfTerminal,
  };
}

/** Narrow the `fit_terminal_phase` block. */
function parseTerminalPhase(raw: unknown, what: string): TerminalPhaseView {
  const r = expectObject(raw, what);
  const spanLow = r["span_low"];
  if (typeof spanLow !== "boolean") {
    throw new Error(`${what}: span_low is not a boolean`);
  }
  const start = asNumber(r["start_index"]);
  const end = asNumber(r["end_index"]);
  const points = asNumber(r["n_points"]);
  const lambdaZ = asNumber(r["lambda_z"]);
  const r2 = asNumber(r["r2"]);
  const r2Adj = asNumber(r["r2_adj"]);
  const halfLife = asNumber(r["half_life"]);
  const clastPred = asNumber(r["clast_pred"]);
  const span = asNumber(r["span_half_lives"]);
  if (
    start === null || end === null || points === null || lambdaZ === null ||
    r2 === null || r2Adj === null || halfLife === null ||
    clastPred === null || span === null
  ) {
    throw new Error(`${what}: a terminal-phase field is not a finite number`);
  }
  return {
    start_index: start,
    end_index: end,
    n_points: points,
    lambda_z: lambdaZ,
    r2,
    r2_adj: r2Adj,
    half_life: halfLife,
    clast_pred: clastPred,
    span_half_lives: span,
    span_low: spanLow,
    method: asString(r["method"]),
  };
}

/** Narrow the `POST /nca` answer. */
export function parseNcaResponse(
  raw: unknown,
  what: string,
): NcaResponse {
  const r = expectObject(raw, what);
  const linear = asNumber(r["auc_trapezoid_linear"]);
  if (linear === null) {
    throw new Error(`${what}: auc_trapezoid_linear is not a finite number`);
  }
  const n = asNumber(r["n_points"]);
  if (n === null) {
    throw new Error(`${what}: n_points is not a finite number`);
  }
  return {
    summary: parseNcaSummary(r["summary"], `${what} summary`),
    auc_trapezoid_linear: linear,
    auc_trapezoid_log: asNumberOrNull(r["auc_trapezoid_log"]),
    terminal: parseTerminalPhase(r["terminal"], `${what} terminal`),
    n_points: n,
  };
}

/** Narrow the `steady_state_metrics().as_dict()` block. */
function parseSteadyStateMetrics(
  raw: unknown,
  what: string,
): SteadyStateMetricsView {
  const r = expectObject(raw, what);
  const AUC_tau = asNumber(r["AUC_tau"]);
  const Cavg = asNumber(r["Cavg"]);
  const Cmax = asNumber(r["Cmax_ss"]);
  const Cmin = asNumber(r["Cmin_ss"]);
  const Ctrough = asNumber(r["Ctrough"]);
  const fluctuation = asNumber(r["fluctuation"]);
  const swing = asNumber(r["swing"]);
  if (
    AUC_tau === null || Cavg === null || Cmax === null || Cmin === null ||
    Ctrough === null || fluctuation === null || swing === null
  ) {
    throw new Error(`${what}: an interval metric is not a finite number`);
  }
  return {
    AUC_tau,
    Cavg,
    Cmax_ss: Cmax,
    Cmin_ss: Cmin,
    Ctrough,
    fluctuation,
    swing,
    CL_ss: asNumberOrNull(r["CL_ss"]),
    R_accum: asNumberOrNull(r["R_accum"]),
  };
}

/** Narrow the `POST /steady-state` answer. */
export function parseSteadyStateResponse(
  raw: unknown,
  what: string,
): SteadyStateResponse {
  const r = expectObject(raw, what);
  const tau = asNumber(r["tau"]);
  const n = asNumber(r["n_points"]);
  if (tau === null || n === null) {
    throw new Error(`${what}: tau/n_points is not a finite number`);
  }
  return {
    tau,
    metrics: parseSteadyStateMetrics(r["metrics"], `${what} metrics`),
    n_points: n,
  };
}

/** Narrow the `POST /iv-bolus` answer. */
export function parseIvBolusResponse(
  raw: unknown,
  what: string,
): IvBolusResponse {
  const r = expectObject(raw, what);
  const c0 = asNumber(r["c0"]);
  const ke = asNumber(r["ke"]);
  const n = asNumber(r["n_points"]);
  if (c0 === null || ke === null || n === null) {
    throw new Error(`${what}: c0/ke/n_points is not a finite number`);
  }
  return {
    c0,
    ke,
    n_points: n,
    concentration: numberArray(r["concentration"], `${what} concentration`, 1, 10000),
  };
}

/** Narrow the `POST /bateman` answer. */
export function parseBatemanResponse(
  raw: unknown,
  what: string,
): BatemanResponse {
  const r = expectObject(raw, what);
  const dpv = asNumber(r["dose_per_volume"]);
  const ka = asNumber(r["ka"]);
  const ke = asNumber(r["ke"]);
  const n = asNumber(r["n_points"]);
  if (dpv === null || ka === null || ke === null || n === null) {
    throw new Error(`${what}: dose_per_volume/ka/ke/n_points is not finite`);
  }
  return {
    dose_per_volume: dpv,
    ka,
    ke,
    n_points: n,
    concentration: numberArray(r["concentration"], `${what} concentration`, 1, 10000),
  };
}

/** Narrow the `POST /superposition` answer. */
export function parseSuperpositionResponse(
  raw: unknown,
  what: string,
): SuperpositionResponse {
  const r = expectObject(raw, what);
  const tau = asNumber(r["tau"]);
  const nd = asNumber(r["n_doses"]);
  const n = asNumber(r["n_points"]);
  if (tau === null || nd === null || n === null) {
    throw new Error(`${what}: tau/n_doses/n_points is not a finite number`);
  }
  return {
    tau,
    n_doses: nd,
    n_points: n,
    concentration: numberArray(r["concentration"], `${what} concentration`, 1, 10000),
  };
}

/** Narrow the `POST /pbpk` answer. */
export function parsePbpkResponse(
  raw: unknown,
  what: string,
): PbpkResponse {
  const r = expectObject(raw, what);
  const route = r["route"];
  if (route !== "iv" && route !== "oral") {
    throw new Error(`${what}: route is not "iv" or "oral"`);
  }
  const dose = asNumber(r["dose_mg"]);
  const n = asNumber(r["n_points"]);
  if (dose === null || n === null) {
    throw new Error(`${what}: dose_mg/n_points is not a finite number`);
  }
  const compartmentsRaw = r["compartments"];
  if (!Array.isArray(compartmentsRaw) || compartmentsRaw.length < 1) {
    throw new Error(`${what}: compartments is not a non-empty array`);
  }
  const compartments = compartmentsRaw.map((c) => String(c));
  const rowsRaw = r["rows"];
  if (!Array.isArray(rowsRaw)) {
    throw new Error(`${what}: rows is not an array`);
  }
  const rows: PbpkRow[] = rowsRaw.map((row, i) => {
    const rr = expectObject(row, `${what} rows[${i}]`);
    const time = asNumber(rr["time"]);
    if (time === null) {
      throw new Error(`${what} rows[${i}]: time is not a finite number`);
    }
    const out: Record<string, number> = { time };
    for (const compartment of compartments) {
      const v = asNumber(rr[compartment]);
      if (v === null) {
        throw new Error(
          `${what} rows[${i}]: compartment ${compartment} is not a finite number`,
        );
      }
      out[compartment] = v;
    }
    const plasma = asNumber(rr["C_plasma"]);
    if (plasma === null) {
      throw new Error(
        `${what} rows[${i}]: C_plasma is not a finite number (the frame's own derived column)`,
      );
    }
    out["C_plasma"] = plasma;
    return out as PbpkRow;
  });
  return { route, dose_mg: dose, compartments, n_points: n, rows };
}

/** Narrow the `POST /pd` answer. */
export function parsePdResponse(raw: unknown, what: string): PdResponse {
  const r = expectObject(raw, what);
  const e0 = asNumber(r["e0"]);
  const emax = asNumber(r["emax"]);
  const ec50 = asNumber(r["ec50"]);
  const hill = asNumber(r["hill"]);
  const n = asNumber(r["n_points"]);
  if (e0 === null || emax === null || ec50 === null || hill === null || n === null) {
    throw new Error(`${what}: a model scalar is not a finite number`);
  }
  const concentration = numberArray(
    r["concentration"], `${what} concentration`, 1, 2000,
  );
  const effect = numberArray(r["effect"], `${what} effect`, 1, 2000);
  if (concentration.length !== effect.length) {
    throw new Error(
      `${what}: concentration (${concentration.length}) and effect (${effect.length}) lengths differ`,
    );
  }
  return { e0, emax, ec50, hill, n_points: n, concentration, effect };
}

/** Narrow the population body (the sync answer / the job REST result). */
export function parsePopulationBody(
  raw: unknown,
  what: string,
): PopulationBodyView {
  const r = expectObject(raw, what);
  const nSubjects = asNumber(r["n_subjects"]);
  const nTimes = asNumber(r["n_times"]);
  const seed = asNumber(r["seed"]);
  if (nSubjects === null || nTimes === null || seed === null) {
    throw new Error(`${what}: n_subjects/n_times/seed is not a finite number`);
  }
  const summaryRaw = expectObject(r["summary"], `${what} summary`);
  const mean = numberArray(summaryRaw["mean"], `${what} summary.mean`, 1, 2000);
  const p5 = numberArray(summaryRaw["p5"], `${what} summary.p5`, 1, 2000);
  const p95 = numberArray(summaryRaw["p95"], `${what} summary.p95`, 1, 2000);
  if (mean.length !== p5.length || mean.length !== p95.length) {
    throw new Error(`${what}: the summary curve lengths differ`);
  }
  if (mean.length !== nTimes) {
    throw new Error(
      `${what}: summary carries ${mean.length} points, the grid is ${nTimes}`,
    );
  }
  const profilesRaw = r["subject_profiles"];
  if (!Array.isArray(profilesRaw)) {
    throw new Error(`${what}: subject_profiles is not an array`);
  }
  const subjectProfiles = profilesRaw.map((profile, i) =>
    numberArray(profile, `${what} subject_profiles[${i}]`, 1, 2000),
  );
  return {
    n_subjects: nSubjects,
    n_times: nTimes,
    seed,
    summary: { mean, p5, p95 },
    subject_profiles: subjectProfiles,
  };
}

/**
 * Narrow one `job.progress` envelope payload — the population lane's
 * per-batch telemetry (progress-only; the authoritative result rides
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

// -- the pre-flight bound helpers (the page calls these before the POST)

/**
 * The NCA/steady-state pre-flight (the route's own list bounds):
 * equal lengths, `>= 2` points, `<= 10000`.
 */
export function assertNcaSeriesBounds(
  what: string,
  times: readonly number[],
  conc: readonly number[],
): void {
  assertSeriesBounds(what, times, conc);
}

/**
 * The model lanes' pre-flight (the route's own list bounds): `>= 1`
 * time point, `<= 10000`.
 */
export function assertTimesBounds(
  what: string,
  times: readonly number[],
): void {
  if (times.length < 1) {
    throw new Error(`${what}: the time grid carries no points`);
  }
  if (times.length > 10000) {
    throw new Error(`${what}: ${times.length} points, over the 10000 cap`);
  }
}

/**
 * The population pre-flight (the route's own exact vector shapes):
 * `theta` exactly 3, `omega_diag` exactly 1, the grid `>= 2` and
 * `<= 2000` points.
 */
export function assertPopulationBounds(
  what: string,
  theta: readonly number[],
  omegaDiag: readonly number[],
  times: readonly number[],
): void {
  if (theta.length !== 3) {
    throw new Error(
      `${what}: theta carries ${theta.length} values, exactly 3 required`,
    );
  }
  if (omegaDiag.length !== 1) {
    throw new Error(
      `${what}: omega_diag carries ${omegaDiag.length} values, exactly 1 required`,
    );
  }
  if (times.length < 2) {
    throw new Error(`${what}: ${times.length} grid points, at least 2 required`);
  }
  if (times.length > 2000) {
    throw new Error(`${what}: ${times.length} grid points, over the 2000 cap`);
  }
}
