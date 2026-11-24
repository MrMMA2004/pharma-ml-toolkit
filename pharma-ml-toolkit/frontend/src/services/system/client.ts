/**
 * part16 §8 — the system-resource service client.
 *
 * One operation, exactly the backend's §15 read-only surface:
 * `GET /api/v1/system/resources`. The panel integrates with the
 * EXISTING resource service and the JobManager queue depth through
 * this single typed call — §8's own rule ("integrate with the
 * backend job manager and existing resource services rather than
 * introducing a separate, inconsistent monitoring architecture")
 * means no parallel monitoring architecture is created here.
 */

import { api } from "@/services/api/client";
import { parseResourceSnapshot, type ResourceSnapshot } from "./types";
import {
  parseCacheCleanupAnswer,
  parseDiagnosticsSnapshot,
  type CacheCleanupAnswer,
  type DiagnosticsSnapshot,
} from "./diagnostics";

export type {
  ResourceSnapshot,
  SystemBlock,
  ProcessBlock,
  JobsBlock,
  GpuBlock,
  ResourceMemoryBlock,
  ResourceSwapBlock,
  ResourceDiskBlock,
} from "./types";
export type {
  CacheCleanupAnswer,
  CrashBlock,
  CrashRow,
  DependencyRow,
  DiagnosticsDatabaseBlock,
  DiagnosticsDiskBlock,
  DiagnosticsScope,
  DiagnosticsSnapshot,
  DiagnosticsStorageBlock,
  LogHandlerRow,
  LogsBlock,
  RecoveryBlock,
  ScientificToolRow,
} from "./diagnostics";

export const systemApi = {
  /** `GET /api/v1/system/resources` — the whole §15 snapshot. */
  async resources(): Promise<ResourceSnapshot> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/system/resources",
    );
    return parseResourceSnapshot(raw, "system resources");
  },

  /**
   * `GET /api/v1/system/diagnostics` — the §15.5 read-only
   * diagnostics answer (dependencies, scientific tools, storage,
   * crash, logs, recovery, scope).
   */
  async diagnostics(): Promise<DiagnosticsSnapshot> {
    const raw = await api.get<Record<string, unknown>>(
      "/api/v1/system/diagnostics",
    );
    return parseDiagnosticsSnapshot(raw, "system diagnostics");
  },

  /**
   * `POST /api/v1/system/cache/cleanup` — §15.5's safe cleanup: the
   * enumerated derived cache only, user data always preserved.
   */
  async cleanupCache(): Promise<CacheCleanupAnswer> {
    const raw = await api.post<Record<string, unknown>>(
      "/api/v1/system/cache/cleanup",
    );
    return parseCacheCleanupAnswer(raw, "cache cleanup");
  },
};
