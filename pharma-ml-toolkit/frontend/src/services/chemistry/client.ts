/**
 * The chemistry client — part11 §11.1's Descriptor/Fingerprint
 * Generation stage (the Chemistry Lab featurization surface).
 *
 * The backend contract (mirrored field-for-field; the Python drift
 * guards in `tests/test_frontend_services.py` pin the route list AND
 * the request-bodies-from-generated-types rule against
 * `routes/chemistry.py`):
 *
 * * `POST /api/v1/chemistry/descriptors` — `{smiles, descriptor_set,
 *   mode}` → `{mode: "job", job: JobView}` or `{mode: "sync",
 *   result: DescriptorFrame}`.
 * * `POST /api/v1/chemistry/fingerprints` — `{smiles, kind, radius,
 *   n_bits, as_counts, mode}` → job/sync answers; the sync result is
 *   the `FingerprintMatrix` payload.
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

/** The descriptor engine's own request cap (chemistry.py's max_length). */
export const CHEMISTRY_MAX_SMILES = 500;

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
