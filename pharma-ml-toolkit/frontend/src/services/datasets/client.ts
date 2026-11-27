/**
 * The dataset client — part11 §11.1's Datasets workflow service
 * (Phase Three services, part two: the Dataset Client block of §5.2).
 *
 * The backend contract (mirrored field-for-field; the Python drift
 * guards in `tests/test_frontend_services.py` pin the route list AND
 * the multipart form-field names against `routes/datasets.py`):
 *
 * * `POST /api/v1/datasets` — multipart upload: the fields `file`
 *   (the CSV/TSV/TXT/XLSX table), `target`, `task`, `mode` — answered
 *   by `{mode: "job", job: JobView}` (default) or `{mode: "sync",
 *   dataset: DatasetView}`. One request = one file (§17.2: upload
 *   once, then reference the stable `dataset_id`).
 * * `GET /api/v1/datasets` / `GET /api/v1/datasets/{id}` /
 *   `GET /api/v1/datasets/{id}/preview?limit=N` (paginated preview,
 *   §17.3) / `DELETE /api/v1/datasets/{id}`.
 * * `POST /api/v1/datasets/{id}/profile` and `POST …/{id}/split` —
 *   `ProfileRequest` / `SplitRequest` bodies straight from the
 *   generated OpenAPI types (`src/types/api.d.ts`, §12.2's one DTO
 *   source — no hand-duplicated request schema).
 *
 * The frames themselves NEVER cross the JSON boundary in full: the
 * backend serves bounded previews and the workflows reference
 * `dataset_id` (§17.1's control-plane rule).
 */

import { api } from "@/services/api/client";
import type { components } from "@/types/api.d";
import { type JobView, parseJobView } from "@/services/jobs/types";

type ProfileRequest = components["schemas"]["ProfileRequest"];
type SplitRequest = components["schemas"]["SplitRequest"];

/** `frame_payload()` — the backend's bounded, JSON-safe head view. */
export interface FramePreview {
  readonly columns: readonly string[];
  readonly dtypes: Readonly<Record<string, string>>;
  readonly n_rows: number;
  readonly returned: number;
  readonly rows: readonly (readonly unknown[])[];
}

/** `DatasetRecord.view()` — the registry's dataset summary. */
export interface DatasetView {
  readonly dataset_id: string;
  readonly name: string;
  readonly source_filename: string;
  readonly size_bytes: number;
  readonly shape: readonly [number, number];
  readonly columns: readonly string[];
  readonly target: string | null;
  readonly task: string | null;
  readonly n_features: number;
  readonly feature_names: readonly string[];
  readonly classes: unknown;
  readonly class_counts: unknown;
  readonly dropped_columns: readonly string[];
  readonly categorical_columns: readonly string[];
  readonly summary: Readonly<Record<string, unknown>>;
  readonly created_at: string;
  readonly preview?: FramePreview;
}

export interface DatasetListResponse {
  readonly datasets: readonly DatasetView[];
}

/** The sync-mode upload answer. */
export interface DatasetSyncUploadResponse {
  readonly mode: "sync";
  readonly dataset: DatasetView;
}

/** The job-mode upload answer (the default). */
export interface DatasetJobUploadResponse {
  readonly mode: "job";
  readonly job: JobView;
}

export type DatasetUploadResponse =
  | DatasetSyncUploadResponse
  | DatasetJobUploadResponse;

/** The split-side preview (`_side_view`). */
export interface SplitSideView {
  readonly rows: number;
  readonly columns: readonly string[];
  readonly target_balance?: Readonly<Record<string, number>>;
  readonly preview: FramePreview;
}

/** The split strategy vocabulary the backend's DS-1 lane serves. */
export type SplitStrategy =
  | "random"
  | "stratified"
  | "scaffold"
  | "group"
  | "kennard_stone"
  | "temporal";

/** One side's cohort census for the group-carrying strategies (DS-1). */
export interface SplitGroupCensusSide {
  readonly n_groups: number;
  readonly top_groups: ReadonlyArray<readonly [string, number]>;
}

/** The DS-1 group census — per-side cohort counts + the leakage verdict. */
export interface SplitGroupCensus {
  readonly sides: Readonly<Record<string, SplitGroupCensusSide>>;
  readonly shared_groups: readonly string[];
}

/** `POST /api/v1/datasets/{id}/split` — the split report. */
export interface DatasetSplitResponse {
  readonly target: string | null;
  readonly stratified: boolean;
  readonly random_state: number;
  readonly strategy: SplitStrategy | null;
  readonly smiles_column: string | null;
  readonly group_column: string | null;
  readonly group_census: SplitGroupCensus | null;
  readonly sides: Readonly<Record<string, SplitSideView>>;
  readonly n_rows: number;
}

/** `POST /api/v1/datasets/{id}/profile` — the profile wrapper. */
export interface DatasetProfileResponse {
  readonly profile: Readonly<Record<string, unknown>>;
}

/**
 * `GET /api/v1/datasets/{id}/column/{column}` — one column's values,
 * bounded by the chemistry engine's own request cap (500). `truncated`
 * states plainly whether the bound clipped the dataset's real rows.
 */
export interface ColumnValues {
  readonly column: string;
  readonly dtype: string;
  readonly n_rows: number;
  readonly returned: number;
  readonly truncated: boolean;
  readonly values: readonly (string | number | null)[];
}

export interface DatasetUploadOptions {
  /** The target column (null = the loader's auto-detection). */
  readonly target?: string | null;
  /** The task override (null = inferred). */
  readonly task?: string | null;
  /** "job" (default) or "sync" — the backend's two execution lanes. */
  readonly mode?: "job" | "sync";
}

export const datasetsApi = {
  /**
   * `POST /api/v1/datasets` — multipart streaming upload through the
   * REAL reader. The file is appended untouched; the browser sets
   * the multipart boundary itself (the REST client's postForm lane
   * deliberately sends no JSON content-type).
   */
  async upload(
    file: File,
    options?: DatasetUploadOptions,
  ): Promise<DatasetUploadResponse> {
    const form = new FormData();
    form.append("file", file, file.name);
    const target = options?.target ?? null;
    const task = options?.task ?? null;
    form.append("target", target === null ? "" : target);
    form.append("task", task === null ? "" : task);
    form.append("mode", options?.mode ?? "job");
    const raw = (await api.postForm("/api/v1/datasets", form)) as Record<
      string,
      unknown
    >;
    if (raw["mode"] === "sync") {
      return {
        mode: "sync",
        dataset: parseDatasetView(raw["dataset"], "upload(sync)"),
      };
    }
    return {
      mode: "job",
      job: parseJobView(raw["job"], "upload(job)"),
    };
  },

  /** `GET /api/v1/datasets` — the registry (oldest-first). */
  async list(): Promise<DatasetListResponse> {
    const raw = await api.get<Record<string, unknown>>("/api/v1/datasets");
    const items = raw["datasets"];
    if (!Array.isArray(items)) {
      return { datasets: [] };
    }
    return {
      datasets: items.map((item, i) =>
        parseDatasetView(item, `datasets[${i}]`),
      ),
    };
  },

  /** `GET /api/v1/datasets/{id}` — the summary + a 10-row preview. */
  async view(datasetId: string): Promise<DatasetView> {
    const raw = await api.get<Record<string, unknown>>(
      `/api/v1/datasets/${encodeURIComponent(datasetId)}`,
    );
    return parseDatasetView(raw, `dataset ${datasetId}`);
  },

  /** `GET /api/v1/datasets/{id}/preview?limit=N` — §17.3 pagination. */
  async preview(
    datasetId: string,
    limit: number = 10,
  ): Promise<FramePreview> {
    const bounded = Math.min(Math.max(1, Math.floor(limit)), 500);
    const raw = await api.get<Record<string, unknown>>(
      `/api/v1/datasets/${encodeURIComponent(datasetId)}/preview?limit=${bounded}`,
    );
    return parseFramePreview(raw, `preview ${datasetId}`);
  },

  /**
   * `GET /api/v1/datasets/{id}/column/{column}?limit=N` — the §11.1
   * chemistry-stage bridge: one column's values bounded by the
   * chemistry engine's own request cap (`CHEMISTRY_MAX_SMILES`), with
   * the dataset's real row count and an explicit `truncated` flag
   * (§17.3: a bounded view never masquerades as the whole column).
   */
  async column(
    datasetId: string,
    column: string,
    limit: number = 500,
  ): Promise<ColumnValues> {
    const bounded = Math.min(Math.max(1, Math.floor(limit)), 500);
    const raw = await api.get<Record<string, unknown>>(
      `/api/v1/datasets/${encodeURIComponent(datasetId)}/column/${encodeURIComponent(column)}?limit=${bounded}`,
    );
    return parseColumnValues(raw, `column ${column}`);
  },

  /** `DELETE /api/v1/datasets/{id}` — evict from the session store. */
  async remove(datasetId: string): Promise<void> {
    await api.delete(`/api/v1/datasets/${encodeURIComponent(datasetId)}`);
  },

  /** `POST /api/v1/datasets/{id}/profile` — the REAL profile_frame. */
  async profile(
    datasetId: string,
    request: ProfileRequest = { mode: "sync" },
  ): Promise<DatasetProfileResponse> {
    const raw = await api.post<Record<string, unknown>>(
      `/api/v1/datasets/${encodeURIComponent(datasetId)}/profile`,
      request,
    );
    return { profile: (raw["profile"] ?? {}) as Record<string, unknown> };
  },

  /** `POST /api/v1/datasets/{id}/split` — stratified splits. */
  async split(
    datasetId: string,
    request: SplitRequest,
  ): Promise<DatasetSplitResponse> {
    const raw = await api.post<Record<string, unknown>>(
      `/api/v1/datasets/${encodeURIComponent(datasetId)}/split`,
      request,
    );
    return parseSplitResponse(raw);
  },
};

// -- validation helpers (§8.1) ------------------------------------------

function expectObject(raw: unknown, what: string): Record<string, unknown> {
  if (typeof raw !== "object" || raw === null) {
    throw new Error(`${what}: expected an object`);
  }
  return raw as Record<string, unknown>;
}

/** Narrow an untrusted payload to the registry's dataset view. */
export function parseDatasetView(raw: unknown, what: string): DatasetView {
  const r = expectObject(raw, what);
  const id = r["dataset_id"];
  const name = r["name"];
  if (typeof id !== "string" || typeof name !== "string") {
    throw new Error(`${what}: dataset_id/name are not strings`);
  }
  const shape = r["shape"];
  const columns = r["columns"];
  const preview = r["preview"];
  return {
    dataset_id: id,
    name,
    source_filename: (r["source_filename"] as string) ?? "",
    size_bytes: typeof r["size_bytes"] === "number" ? r["size_bytes"] : 0,
    shape: Array.isArray(shape)
      ? ([shape[0] ?? 0, shape[1] ?? 0] as [number, number])
      : [0, 0],
    columns: Array.isArray(columns) ? (columns as string[]) : [],
    target: (r["target"] as string | null) ?? null,
    task: (r["task"] as string | null) ?? null,
    n_features: typeof r["n_features"] === "number" ? r["n_features"] : 0,
    feature_names: Array.isArray(r["feature_names"])
      ? (r["feature_names"] as string[])
      : [],
    classes: r["classes"] ?? null,
    class_counts: r["class_counts"] ?? null,
    dropped_columns: Array.isArray(r["dropped_columns"])
      ? (r["dropped_columns"] as string[])
      : [],
    categorical_columns: Array.isArray(r["categorical_columns"])
      ? (r["categorical_columns"] as string[])
      : [],
    summary: (r["summary"] as Record<string, unknown>) ?? {},
    created_at: (r["created_at"] as string) ?? "",
    preview:
      preview === undefined
        ? undefined
        : parseFramePreview(preview, `${what} preview`),
  };
}

function parseFramePreview(raw: unknown, what: string): FramePreview {
  const r = expectObject(raw, what);
  const columns = r["columns"];
  const rows = r["rows"];
  return {
    columns: Array.isArray(columns) ? (columns as string[]) : [],
    dtypes: (r["dtypes"] as Record<string, string>) ?? {},
    n_rows: typeof r["n_rows"] === "number" ? r["n_rows"] : 0,
    returned: typeof r["returned"] === "number" ? r["returned"] : 0,
    rows: Array.isArray(rows)
      ? (rows as unknown[]).map((row) =>
          Array.isArray(row) ? (row as readonly unknown[]) : [],
        )
      : [],
  };
}

/** Narrow an untrusted payload to the column-bridge view (§8.1). */
function parseColumnValues(raw: unknown, what: string): ColumnValues {
  const r = expectObject(raw, what);
  const values = r["values"];
  return {
    column: typeof r["column"] === "string" ? r["column"] : "",
    dtype: typeof r["dtype"] === "string" ? r["dtype"] : "",
    n_rows: typeof r["n_rows"] === "number" ? r["n_rows"] : 0,
    returned: typeof r["returned"] === "number" ? r["returned"] : 0,
    truncated: r["truncated"] === true,
    values: Array.isArray(values)
      ? (values as unknown[]).map((v) =>
          typeof v === "string" || typeof v === "number" ? v : null,
        )
      : [],
  };
}

function parseSplitResponse(raw: unknown): DatasetSplitResponse {
  const r = expectObject(raw, "split");
  const sidesRaw = r["sides"];
  const sides: Record<string, SplitSideView> = {};
  if (typeof sidesRaw === "object" && sidesRaw !== null) {
    for (const [name, sideRaw] of Object.entries(
      sidesRaw as Record<string, unknown>,
    )) {
      const side = expectObject(sideRaw, `split side ${name}`);
      sides[name] = {
        rows: typeof side["rows"] === "number" ? side["rows"] : 0,
        columns: Array.isArray(side["columns"])
          ? (side["columns"] as string[])
          : [],
        target_balance:
          typeof side["target_balance"] === "object" &&
          side["target_balance"] !== null
            ? (side["target_balance"] as Record<string, number>)
            : undefined,
        preview: parseFramePreview(
          side["preview"],
          `split side ${name} preview`,
        ),
      };
    }
  }
  return {
    target: (r["target"] as string | null) ?? null,
    stratified: r["stratified"] === true,
    random_state: typeof r["random_state"] === "number" ? r["random_state"] : 0,
    strategy: (STRATEGY_LITERALS as readonly string[]).includes(r["strategy"] as string)
      ? (r["strategy"] as SplitStrategy)
      : null,
    smiles_column: typeof r["smiles_column"] === "string" ? r["smiles_column"] : null,
    group_column: typeof r["group_column"] === "string" ? r["group_column"] : null,
    group_census: parseGroupCensus(r["group_census"]),
    sides,
    n_rows: typeof r["n_rows"] === "number" ? r["n_rows"] : 0,
  };
}

const STRATEGY_LITERALS: readonly string[] = [
  "random",
  "stratified",
  "scaffold",
  "group",
  "kennard_stone",
  "temporal",
];

/** Narrow the DS-1 census payload (absent on the legacy lane). */
function parseGroupCensus(raw: unknown): SplitGroupCensus | null {
  if (typeof raw !== "object" || raw === null) {
    return null;
  }
  const r = raw as Record<string, unknown>;
  const sides: Record<string, SplitGroupCensusSide> = {};
  for (const [name, sideRaw] of Object.entries(r)) {
    if (name === "shared_groups") {
      continue;
    }
    if (typeof sideRaw !== "object" || sideRaw === null) {
      continue;
    }
    const side = sideRaw as Record<string, unknown>;
    const top = Array.isArray(side["top_groups"])
      ? (side["top_groups"] as unknown[]).map((entry) =>
          Array.isArray(entry) &&
          typeof entry[0] === "string" &&
          typeof entry[1] === "number"
            ? ([entry[0], entry[1]] as const)
            : (["", 0] as const),
        )
      : [];
    sides[name] = {
      n_groups: typeof side["n_groups"] === "number" ? side["n_groups"] : 0,
      top_groups: top,
    };
  }
  return {
    sides,
    shared_groups: Array.isArray(r["shared_groups"])
      ? (r["shared_groups"] as unknown[]).filter(
          (v): v is string => typeof v === "string",
        )
      : [],
  };
}
