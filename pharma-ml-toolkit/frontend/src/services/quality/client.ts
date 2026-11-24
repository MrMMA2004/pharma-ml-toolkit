/**
 * The data-quality REST client — part16 §15.4 (the Phase-12 lane).
 *
 * One POST operation, exactly the backend's
 * `POST /api/v1/datasets/{id}/quality` surface: the read-only §15.4
 * scan over one registered dataset. The request mirrors
 * `QualityRequest` (the generated `api.d.ts` is the single DTO
 * source); the response is narrowed through the parse guards — an
 * unexpected backend shape surfaces as `ApiSchemaError`, never a
 * silent `any`. No mutation operation exists on this client by
 * design: the report is advisory, §15.4's "never silently alter
 * user-provided scientific data" is the contract.
 */

import type { components } from "@/types/api";
import { api } from "@/services/api/client";
import { parseDataQualityReport, type DataQualityReport } from "./types";

type QualityRequest = components["schemas"]["QualityRequest"];

export type {
  DataQualityReport,
  DatasetQualityResponse,
  QualityCheckEntry,
  QualityCode,
  QualityFinding,
  QualitySeverity,
  QualityCategory,
} from "./types";

/** The scan knobs (mirrors the backend's QualityRequest). */
export interface QualityScanParams {
  /** Null = the honest auto-detection; "" = the explicit disable. */
  readonly smilesColumn?: string;
  /** Null = the record's own target. */
  readonly target?: string;
}

export const qualityApi = {
  /** `POST /api/v1/datasets/{datasetId}/quality` — the §15.4 scan. */
  async run(
    datasetId: string,
    params?: QualityScanParams,
  ): Promise<DataQualityReport> {
    const request: QualityRequest = {
      smiles_column: params?.smilesColumn ?? null,
      target: params?.target ?? null,
      mode: "sync",
    };
    const raw = await api.post<Record<string, unknown>>(
      `/api/v1/datasets/${encodeURIComponent(datasetId)}/quality`,
      request,
    );
    const quality = raw["quality"];
    if (quality === undefined) {
      throw new Error("quality response carries no report");
    }
    return parseDataQualityReport(quality);
  },
};
