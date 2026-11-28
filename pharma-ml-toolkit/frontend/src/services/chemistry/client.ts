/**
 * The chemistry client — part11 §11.1's Descriptor/Fingerprint
 * Generation stage (the Chemistry Lab featurization surface), plus
 * DS-2's Clean-stage standardization verb (roadmap row 6) and DS-3's
 * frames-register verb (roadmap row 7).
 *
 * The backend contract (mirrored field-for-field; the Python drift
 * guards in `tests/test_frontend_services.py`,
 * `tests/test_frontend_standardize_action.py` and
 * `tests/test_frontend_frames_register.py` pin the route list AND
 * the request-bodies-from-generated-types rule against
 * `routes/chemistry.py`):
 *
 * * `POST /api/v1/chemistry/descriptors` — `{smiles, descriptor_set,
 *   mode}` → `{mode: "job", job: JobView}` or `{mode: "sync",
 *   result: DescriptorFrame}`.
 * * `POST /api/v1/chemistry/fingerprints` — `{smiles, kind, radius,
 *   n_bits, as_counts, include_chirality, mode}` → job/sync answers;
 *   the sync result is the `FingerprintMatrix` payload
 *   (`include_chirality` is morgan-only — B-5, part12 Phase 2: the
 *   other kinds carry no such native parameter and 422).
 * * `POST /api/v1/chemistry/standardize` — `{dataset_id, column, ops,
 *   neutralize, mode}` (DS-2) → job/sync answers carrying the
 *   standardization report (counts, the honest rejected list, the
 *   derived dataset's view with its provenance block).
 * * `POST /api/v1/chemistry/frames/register` — `{dataset_id, column,
 *   featurization, …, mode}` (DS-3) → job/sync answers carrying the
 *   registration report (counts with `features_appended`, the honest
 *   rejected list, the derived dataset's view with its
 *   `derived-descriptors|fingerprints` provenance block).
 *
 * The smiles source for a registered dataset is the datasets
 * client's bounded `column()` bridge (`CHEMISTRY_MAX_SMILES` = the
 * engine's own request cap — §17.3: explicit, never silent).
 *
 * Request bodies come from the generated OpenAPI types
 * (`src/types/api.d.ts`, §12.2's one DTO source — no hand duplicates).
 */

import { api } from "@/services/api/client";
import type { components } from "@/types/api.d";
import { type JobView, parseJobView } from "@/services/jobs/types";

type DescriptorsRequest = components["schemas"]["DescriptorsRequest"];
type FingerprintsRequest = components["schemas"]["FingerprintsRequest"];
type StandardizeRequestDto = components["schemas"]["StandardizeRequest"];
type FramesRegisterRequestDto = components["schemas"]["FramesRegisterRequest"];

/** The descriptor engine's own request cap (chemistry.py's max_length). */
export const CHEMISTRY_MAX_SMILES = 500;

/**
 * DS-2's pass vocabulary — the engine's own real callables
 * (`chem/standardize.py`), one carrier end to the other (the drift
 * guard pins route Literal == contract enum == this list).
 */
export const STANDARDIZE_OP_LITERALS = [
  "canonical_tautomer",
  "reinvent",
] as const;
export type StandardizeOp = (typeof STANDARDIZE_OP_LITERALS)[number];

/** `_frame_to_payload` — the descriptor DataFrame's JSON-safe head. */
export interface DescriptorFrame {
  readonly shape: readonly [number, number];
  readonly columns: readonly string[];
  readonly rows: readonly (readonly unknown[])[];
  readonly index: readonly string[];
}

/** `_fp_to_payload` — the fingerprint bit/count matrix. */
export interface FingerprintMatrix {
  readonly shape: readonly [number, number];
  readonly dtype: string;
  readonly invalid_indices: readonly number[];
  readonly matrix: readonly (readonly number[])[];
}

/** The sync-mode descriptor answer. */
export interface DescriptorsSyncResponse {
  readonly mode: "sync";
  readonly result: DescriptorFrame;
}

/** One honestly-rejected row (the no-silent-drop accounting). */
export interface StandardizeRejectedRow {
  readonly row: number;
  readonly input: string;
  readonly reason: string;
}

/** The derived dataset's identity, as seen in the report. */
export interface StandardizeDerivedDataset {
  readonly dataset_id: string;
  readonly name: string;
  readonly shape: readonly [number, number];
  readonly columns: readonly string[];
}

/** `POST /api/v1/chemistry/standardize`'s report (DS-2). */
export interface StandardizeReport {
  readonly source_dataset_id: string;
  readonly column: string;
  readonly ops: readonly StandardizeOp[];
  readonly counts: {
    readonly n_rows: number;
    readonly changed: number;
    readonly unchanged: number;
    readonly rejected: number;
  };
  readonly rejected: readonly StandardizeRejectedRow[];
  readonly provenance: Readonly<Record<string, unknown>>;
  readonly dataset: StandardizeDerivedDataset;
}

/** The featurization branch of DS-3's register verb. */
export type FramesFeaturization = "descriptors" | "fingerprints";

/** `POST /api/v1/chemistry/frames/register`'s report (DS-3). */
export interface FramesRegisterReport {
  readonly source_dataset_id: string;
  readonly column: string;
  readonly featurization: FramesFeaturization;
  readonly counts: {
    readonly n_rows: number;
    readonly kept: number;
    readonly rejected: number;
    readonly features_appended: number;
  };
  readonly rejected: readonly StandardizeRejectedRow[];
  readonly provenance: Readonly<Record<string, unknown>>;
  readonly dataset: StandardizeDerivedDataset;
}

/** The job-mode answer (the default for both routes). */
export interface ChemistryJobResponse {
  readonly mode: "job";
  readonly job: JobView;
}

export type DescriptorsResponse = DescriptorsSyncResponse | ChemistryJobResponse;

export interface DescriptorsOptions {
  readonly descriptorSet: "rdkit_all" | "morgan" | "mordred";
  readonly mode?: "job" | "sync";
}

export interface FingerprintOptions {
  readonly kind: "morgan" | "rdkit" | "atom_pair" | "avalon";
  readonly radius: number;
  readonly nBits: number;
  readonly asCounts: boolean;
  /**
   * Morgan only (B-5, part12 Phase 2): pass `includeChirality` through
   * so stereochemistry-annotated environments fold into distinct bits.
   * Default OFF keeps the historical vectors byte-identical — enabling
   * it CHANGES FEATURE SEMANTICS (a chirality-sensitive pair becomes
   * distinguishable); a data-compat decision the caller owns.
   */
  readonly includeChirality: boolean;
  readonly mode?: "job" | "sync";
}

export interface StandardizeOptions {
  /** The SMILES column; `null` = the backend's honest auto-detection. */
  readonly column: string | null;
  /** Ordered passes over the engine's real callables. */
  readonly ops: readonly StandardizeOp[];
  /** The REINVENT chain's own step-4 toggle (honoured when included). */
  readonly neutralize: boolean;
  readonly mode?: "job" | "sync";
}

export interface FramesRegisterOptions {
  /** The SMILES column; `null` = the backend's honest auto-detection. */
  readonly column: string | null;
  /** Which featurization engine registers the derived frame. */
  readonly featurization: FramesFeaturization;
  /** The descriptors branch's set (`rdkit_all` | `morgan` | `mordred`). */
  readonly descriptorSet: "rdkit_all" | "morgan" | "mordred";
  /** The fingerprints branch's own knobs (mirrors FingerprintOptions). */
  readonly kind: "morgan" | "rdkit" | "atom_pair" | "avalon";
  readonly radius: number;
  readonly nBits: number;
  readonly asCounts: boolean;
  /**
   * Morgan only (B-5): the same semantics as the display lane — the
   * backend honours it for morgan and 422s the other kinds.
   */
  readonly includeChirality: boolean;
  /**
   * The target column carried into the derived dataset; `null`
   * inherits the source record's own target.
   */
  readonly target: string | null;
  readonly mode?: "job" | "sync";
}

export const chemistryApi = {
  /** `POST /api/v1/chemistry/descriptors` — the REAL compute engine. */
  async descriptors(
    smiles: readonly string[],
    options: DescriptorsOptions,
  ): Promise<DescriptorsResponse> {
    const request: DescriptorsRequest = {
      smiles: [...smiles],
      descriptor_set: options.descriptorSet,
      mode: options.mode ?? "job",
    };
    const raw = (await api.post("/api/v1/chemistry/descriptors", request)) as Record<
      string,
      unknown
    >;
    if (raw["mode"] === "sync") {
      return {
        mode: "sync",
        result: parseDescriptorFrame(raw["result"], "descriptors"),
      };
    }
    return { mode: "job", job: parseJobView(raw["job"], "descriptors(job)") };
  },

  /** `POST /api/v1/chemistry/fingerprints` — the REAL featurizer. */
  async fingerprints(
    smiles: readonly string[],
    options: FingerprintOptions,
  ): Promise<ChemistryJobResponse | { mode: "sync"; result: FingerprintMatrix }> {
    const request: FingerprintsRequest = {
      smiles: [...smiles],
      kind: options.kind,
      radius: options.radius,
      n_bits: options.nBits,
      as_counts: options.asCounts,
      include_chirality: options.includeChirality,
      mode: options.mode ?? "job",
    };
    const raw = (await api.post(
      "/api/v1/chemistry/fingerprints",
      request,
    )) as Record<string, unknown>;
    if (raw["mode"] === "sync") {
      return {
        mode: "sync",
        result: parseFingerprintMatrix(raw["result"], "fingerprints"),
      };
    }
    return { mode: "job", job: parseJobView(raw["job"], "fingerprints(job)") };
  },

  /**
   * `POST /api/v1/chemistry/standardize` — DS-2, the Clean stage's
   * verb: the REAL REINVENT-style engine over a dataset SMILES column,
   * registering the standardized frame as a NEW derived dataset (the
   * source record is never mutated; rejected rows ride the report).
   */
  async standardize(
    datasetId: string,
    options: StandardizeOptions,
  ): Promise<ChemistryJobResponse | { mode: "sync"; report: StandardizeReport }> {
    const request: StandardizeRequestDto = {
      dataset_id: datasetId,
      column: options.column,
      ops: [...options.ops],
      neutralize: options.neutralize,
      mode: options.mode ?? "job",
    };
    const raw = (await api.post(
      "/api/v1/chemistry/standardize",
      request,
    )) as Record<string, unknown>;
    if (raw["mode"] === "sync") {
      return {
        mode: "sync",
        report: parseStandardizeReport(raw, "standardize(sync)"),
      };
    }
    return { mode: "job", job: parseJobView(raw["job"], "standardize(job)") };
  },

  /**
   * `POST /api/v1/chemistry/frames/register` — DS-3, the
   * Descriptor/Fingerprint stage's missing verb: the REAL engines
   * over one dataset SMILES column (re-read server-side,
   * row-aligned), registering the featurized frame as a NEW derived
   * dataset (source columns + features + target; the source record
   * is never mutated; rejected rows ride the report).
   */
  async registerFrames(
    datasetId: string,
    options: FramesRegisterOptions,
  ): Promise<ChemistryJobResponse | { mode: "sync"; report: FramesRegisterReport }> {
    const request: FramesRegisterRequestDto = {
      dataset_id: datasetId,
      column: options.column,
      featurization: options.featurization,
      descriptor_set: options.descriptorSet,
      kind: options.kind,
      radius: options.radius,
      n_bits: options.nBits,
      as_counts: options.asCounts,
      include_chirality: options.includeChirality,
      target: options.target,
      mode: options.mode ?? "job",
    };
    const raw = (await api.post(
      "/api/v1/chemistry/frames/register",
      request,
    )) as Record<string, unknown>;
    if (raw["mode"] === "sync") {
      return {
        mode: "sync",
        report: parseFramesRegisterReport(raw, "frames/register(sync)"),
      };
    }
    return { mode: "job", job: parseJobView(raw["job"], "frames/register(job)") };
  },
};

// -- validation helpers (§8.1) ------------------------------------------

function expectObject(raw: unknown, what: string): Record<string, unknown> {
  if (typeof raw !== "object" || raw === null) {
    throw new Error(`${what}: expected an object`);
  }
  return raw as Record<string, unknown>;
}

/** Narrow an untrusted payload to the descriptor frame view. */
export function parseDescriptorFrame(
  raw: unknown,
  what: string,
): DescriptorFrame {
  const r = expectObject(raw, what);
  const shape = r["shape"];
  const columns = r["columns"];
  const rows = r["rows"];
  return {
    shape: Array.isArray(shape)
      ? ([shape[0] ?? 0, shape[1] ?? 0] as [number, number])
      : [0, 0],
    columns: Array.isArray(columns) ? (columns as string[]) : [],
    rows: Array.isArray(rows)
      ? (rows as unknown[]).map((row) =>
          Array.isArray(row) ? (row as readonly unknown[]) : [],
        )
      : [],
    index: Array.isArray(r["index"]) ? (r["index"] as string[]) : [],
  };
}

/** Narrow an untrusted payload to the fingerprint matrix view. */
export function parseFingerprintMatrix(
  raw: unknown,
  what: string,
): FingerprintMatrix {
  const r = expectObject(raw, what);
  const shape = r["shape"];
  const matrix = r["matrix"];
  const invalid = r["invalid_indices"];
  return {
    shape: Array.isArray(shape)
      ? ([shape[0] ?? 0, shape[1] ?? 0] as [number, number])
      : [0, 0],
    dtype: typeof r["dtype"] === "string" ? r["dtype"] : "",
    invalid_indices: Array.isArray(invalid) ? (invalid as number[]) : [],
    matrix: Array.isArray(matrix)
      ? (matrix as unknown[]).map((row) =>
          Array.isArray(row) ? (row as readonly number[]) : [],
        )
      : [],
  };
}

/**
 * Narrow an untrusted standardize answer to the report view (the sync
 * payload carries the report at the top level; a job's fetched result
 * carries the same fields minus `mode`).
 */
export function parseStandardizeReport(
  raw: unknown,
  what: string,
): StandardizeReport {
  const r = expectObject(raw, what);
  const counts = expectObject(r["counts"], `${what}: counts`);
  const dataset = expectObject(r["dataset"], `${what}: dataset`);
  const shape = dataset["shape"];
  const ops = Array.isArray(r["ops"])
    ? (r["ops"] as unknown[]).filter(
        (op): op is StandardizeOp =>
          typeof op === "string" &&
          (STANDARDIZE_OP_LITERALS as readonly string[]).includes(op),
      )
    : [];
  const rejected = Array.isArray(r["rejected"])
    ? (r["rejected"] as unknown[]).map((entry) => {
        const row = expectObject(entry, `${what}: rejected row`);
        return {
          row: typeof row["row"] === "number" ? row["row"] : -1,
          input: typeof row["input"] === "string" ? row["input"] : "",
          reason: typeof row["reason"] === "string" ? row["reason"] : "",
        };
      })
    : [];
  return {
    source_dataset_id:
      typeof r["source_dataset_id"] === "string" ? r["source_dataset_id"] : "",
    column: typeof r["column"] === "string" ? r["column"] : "",
    ops,
    counts: {
      n_rows: typeof counts["n_rows"] === "number" ? counts["n_rows"] : 0,
      changed: typeof counts["changed"] === "number" ? counts["changed"] : 0,
      unchanged:
        typeof counts["unchanged"] === "number" ? counts["unchanged"] : 0,
      rejected:
        typeof counts["rejected"] === "number" ? counts["rejected"] : 0,
    },
    rejected,
    provenance:
      typeof r["provenance"] === "object" && r["provenance"] !== null
        ? (r["provenance"] as Readonly<Record<string, unknown>>)
        : {},
    dataset: {
      dataset_id:
        typeof dataset["dataset_id"] === "string"
          ? dataset["dataset_id"]
          : "",
      name: typeof dataset["name"] === "string" ? dataset["name"] : "",
      shape: Array.isArray(shape)
        ? ([shape[0] ?? 0, shape[1] ?? 0] as [number, number])
        : [0, 0],
      columns: Array.isArray(dataset["columns"])
        ? (dataset["columns"] as string[])
        : [],
    },
  };
}

/**
 * Narrow an untrusted frames-register answer to the report view (the
 * sync payload carries the report at the top level; a job's fetched
 * result carries the same fields minus `mode`).
 */
export function parseFramesRegisterReport(
  raw: unknown,
  what: string,
): FramesRegisterReport {
  const r = expectObject(raw, what);
  const counts = expectObject(r["counts"], `${what}: counts`);
  const dataset = expectObject(r["dataset"], `${what}: dataset`);
  const shape = dataset["shape"];
  const featurization = r["featurization"];
  const rejected = Array.isArray(r["rejected"])
    ? (r["rejected"] as unknown[]).map((entry) => {
        const row = expectObject(entry, `${what}: rejected row`);
        return {
          row: typeof row["row"] === "number" ? row["row"] : -1,
          input: typeof row["input"] === "string" ? row["input"] : "",
          reason: typeof row["reason"] === "string" ? row["reason"] : "",
        };
      })
    : [];
  return {
    source_dataset_id:
      typeof r["source_dataset_id"] === "string" ? r["source_dataset_id"] : "",
    column: typeof r["column"] === "string" ? r["column"] : "",
    featurization:
      featurization === "fingerprints" ? "fingerprints" : "descriptors",
    counts: {
      n_rows: typeof counts["n_rows"] === "number" ? counts["n_rows"] : 0,
      kept: typeof counts["kept"] === "number" ? counts["kept"] : 0,
      rejected:
        typeof counts["rejected"] === "number" ? counts["rejected"] : 0,
      features_appended:
        typeof counts["features_appended"] === "number"
          ? counts["features_appended"]
          : 0,
    },
    rejected,
    provenance:
      typeof r["provenance"] === "object" && r["provenance"] !== null
        ? (r["provenance"] as Readonly<Record<string, unknown>>)
        : {},
    dataset: {
      dataset_id:
        typeof dataset["dataset_id"] === "string"
          ? dataset["dataset_id"]
          : "",
      name: typeof dataset["name"] === "string" ? dataset["name"] : "",
      shape: Array.isArray(shape)
        ? ([shape[0] ?? 0, shape[1] ?? 0] as [number, number])
        : [0, 0],
      columns: Array.isArray(dataset["columns"])
        ? (dataset["columns"] as string[])
        : [],
    },
  };
}
