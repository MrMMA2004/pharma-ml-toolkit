/**
 * The sample-data client — part16 §8 (Phase 5): the demo-data lane's
 * service layer.
 *
 * The backend contract (mirrored field-for-field from
 * `src/pharmaml/api/routes/samples.py`; the Python drift guards in
 * `tests/test_frontend_services.py` pin the route surface against
 * the backend module):
 *
 * * `GET /api/v1/samples?workflow=<id>` — the curated catalog over
 *   the REAL shipped `examples/data` files (every entry carries
 *   `available`, `size_bytes`, `sha256`; a preview-before-load
 *   never registers anything);
 * * `GET /api/v1/samples/{sample_id}/preview?limit=N` — the RAW
 *   file head (tabular/library/series: the frame head; text_file:
 *   the first lines);
 * * `POST /api/v1/samples/{sample_id}/load` — the one-click load,
 *   four payload kinds: `tabular` (a FRESH dataset record through
 *   the real import path — never an overwrite), `series` (the
 *   times/conc arrays the PK/PD forms take), `library` (the
 *   name+SMILES columns the chemistry/generative/docking lanes
 *   take), `text_file` (the bounded file content).
 */

import { api } from "@/services/api/client";

/** One catalog entry — metadata over a REAL shipped file. */
export interface SampleView {
  readonly sample_id: string;
  readonly title: string;
  readonly filename: string;
  readonly kind: "tabular" | "series" | "library" | "text_file";
  readonly file_type: string;
  readonly series_columns: { time: string; value: string } | null;
  readonly library_columns: { name: string; smiles: string } | null;
  readonly description: string;
  readonly scientific_context: string;
  readonly workflows: readonly string[];
  readonly target: string | null;
  readonly task: string | null;
  readonly size_bytes: number | null;
  readonly sha256: string | null;
  readonly available: boolean;
  readonly cleanup: string;
}

export interface SampleCatalog {
  readonly count: number;
  readonly workflow: string | null;
  readonly samples: readonly SampleView[];
}

/** The raw preview — tabular-like kinds carry the frame head. */
export interface SamplePreview {
  readonly sample_id: string;
  readonly kind: "tabular" | "series" | "library" | "text_file";
  readonly columns: readonly string[];
  readonly dtypes?: Readonly<Record<string, string>>;
  readonly n_rows: number;
  readonly returned: number;
  readonly truncated: boolean;
  readonly rows?: readonly (readonly unknown[])[];
  readonly lines?: readonly string[];
  readonly size_bytes?: number;
}

/** `POST /{id}/load` — the tabular payload kind. */
export interface TabularLoadResponse {
  readonly mode: "sync";
  readonly kind: "tabular";
  readonly sample_id: string;
  readonly dataset: Record<string, unknown>;
}

/** `POST /{id}/load` — the series payload kind (the PK/PD arrays). */
export interface SeriesLoadResponse {
  readonly mode: "sync";
  readonly kind: "series";
  readonly sample_id: string;
  readonly columns: { time: string; value: string };
  readonly times: readonly number[];
  readonly conc: readonly number[];
  readonly n_points: number;
}

/** `POST /{id}/load` — the library payload kind (name+SMILES). */
export interface LibraryLoadResponse {
  readonly mode: "sync";
  readonly kind: "library";
  readonly sample_id: string;
  readonly columns: { name: string; smiles: string };
  readonly names: readonly string[];
  readonly smiles: readonly string[];
  readonly n_molecules: number;
}

/** `POST /{id}/load` — the text-file payload kind (raw content). */
export interface TextLoadResponse {
  readonly mode: "sync";
  readonly kind: "text_file";
  readonly sample_id: string;
  readonly filename: string;
  readonly file_type: string;
  readonly size_bytes: number;
  readonly sha256: string;
  readonly n_lines: number;
  readonly content: string;
  readonly truncated: boolean;
}

export type SampleLoadResponse =
  | TabularLoadResponse
  | SeriesLoadResponse
  | LibraryLoadResponse
  | TextLoadResponse;

export const samplesApi = {
  /** The catalog — `workflow` filters to one workspace tag. */
  async list(workflow?: string): Promise<SampleCatalog> {
    const suffix = workflow ? `?workflow=${encodeURIComponent(workflow)}` : "";
    return api.get<SampleCatalog>(`/api/v1/samples${suffix}`);
  },

  /** The raw preview head (never registers anything). */
  async preview(sampleId: string, limit = 8): Promise<SamplePreview> {
    return api.get<SamplePreview>(
      `/api/v1/samples/${encodeURIComponent(sampleId)}/preview?limit=${limit}`,
    );
  },

  /** The one-click load — the payload kind decides the consumer. */
  async load(sampleId: string): Promise<SampleLoadResponse> {
    return api.post<SampleLoadResponse>(
      `/api/v1/samples/${encodeURIComponent(sampleId)}/load`,
      {},
    );
  },
};
