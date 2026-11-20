/**
 * The studio database client — the `/api/v1/database` surface (part12
 * Phase 3: the persistence routes had NO GUI consumer while the
 * Introduction page claimed "backup and restore built in"; this client
 * is the wiring that makes the claim true).
 *
 * The backend contract (mirrored field-for-field from the route source
 * `routes/database.py` and the persistence layer behind it):
 *
 * * `GET /api/v1/database` — the durable-status dict: `persistent`,
 *   `data_dir`, `database`, `schema_version`, `journal_mode`,
 *   `integrity`, `tables` (row counts for datasets,
 *   trained_models, generative_priors, jobs, artifacts,
 *   user_preferences) and the optional `startup_recovery` report; a
 *   non-durable instance answers `{persistent: false, reason}`.
 * * `GET /api/v1/database/jobs?limit=N` — `{jobs: JobHistoryRow[]}`
 *   (the SQLite-upserted job history, newest-first).
 * * `POST /api/v1/database/backup` `{filename?}` — registers a
 *   `database`-kind artifact in the session registry and answers
 *   `{artifact: ArtifactView}` (the bundle is the sqlite3 online
 *   backup API's snapshot plus the vault payloads).
 * * `POST /api/v1/database/restore` `{artifact_id}` — restores the
 *   bundle, rebuilds the live registry from it and answers
 *   `{restore, recovered}` (404 unknown artifact, 422 non-database
 *   kind or a rejected bundle).
 *
 * §8.1: every payload is narrowed at runtime — the label names the
 * lane in a thrown error, so a drifted contract fails loudly.
 */

import { api } from "@/services/api/client";
import { parseArtifactView } from "@/services/artifacts/client";
import type { ArtifactView } from "@/services/artifacts/client";

/** `GET /database` for a durable instance — `persistence.status()`. */
export interface DatabaseStatus {
  readonly persistent: true;
  readonly data_dir: string;
  readonly database: string;
  readonly schema_version: number;
  readonly journal_mode: string;
  readonly integrity: string;
  readonly tables: Readonly<Record<string, number>>;
  readonly startup_recovery?: Readonly<Record<string, unknown>>;
}

/** `GET /database` for a non-durable instance — the honest fallback. */
export interface EphemeralStatus {
  readonly persistent: false;
  readonly reason: string;
}

export type DatabaseStatusResponse = DatabaseStatus | EphemeralStatus;

/** One row of the persisted `jobs` table (the history view). */
export interface JobHistoryRow {
  readonly job_id: string;
  readonly operation: string;
  readonly workspace: string;
  readonly state: string;
  readonly created_at: string;
  readonly started_at: string | null;
  readonly completed_at: string | null;
  readonly progress: number;
  readonly error: string | null;
}

/** `POST /database/backup` — the registered bundle artifact. */
export interface BackupResult {
  readonly artifact: ArtifactView;
}

/** `POST /database/restore` — the restore verdict plus the registry
 *  recovery report (the live stores are rebuilt from the backup). */
export interface RestoreResult {
  readonly restore: Readonly<Record<string, unknown>>;
  readonly recovered: Readonly<Record<string, unknown>>;
}

function _field(raw: unknown, key: string, label: string): unknown {
  if (typeof raw !== "object" || raw === null) {
    throw new Error(`${label} is not an object`);
  }
  const value = (raw as Record<string, unknown>)[key];
  if (value === undefined) {
    throw new Error(`${label} is missing the ${key} field`);
  }
  return value;
}

function _string(raw: unknown, key: string, label: string): string {
  const value = _field(raw, key, label);
  if (typeof value !== "string") {
    throw new Error(`${label}.${key} is not a string`);
  }
  return value;
}

function _optString(raw: unknown, key: string, label: string): string | null {
  const value = (raw as Record<string, unknown>)[key];
  if (value === undefined || value === null) return null;
  if (typeof value !== "string") {
    throw new Error(`${label}.${key} is not a string`);
  }
  return value;
}

function _number(raw: unknown, key: string, label: string): number {
  const value = _field(raw, key, label);
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label}.${key} is not a finite number`);
  }
  return value;
}

function _record(
  raw: unknown,
  key: string,
  label: string,
): Readonly<Record<string, unknown>> {
  const value = _field(raw, key, label);
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${label}.${key} is not an object`);
  }
  return value as Readonly<Record<string, unknown>>;
}

/** Narrow `GET /database` — either branch validates its own fields. */
export function parseDatabaseStatus(raw: unknown): DatabaseStatusResponse {
  const persistent = _field(raw, "persistent", "the database status");
  if (typeof persistent !== "boolean") {
    throw new Error("the database status's persistent field is not a boolean");
  }
  if (!persistent) {
    return {
      persistent: false,
      reason: _string(raw, "reason", "the ephemeral status"),
    };
  }
  const tablesRaw = _record(raw, "tables", "the database status");
  const tables: Record<string, number> = {};
  for (const key of Object.keys(tablesRaw)) {
    const value = tablesRaw[key];
    if (typeof value === "number" && Number.isFinite(value)) {
      tables[key] = value;
    }
  }
  return {
    persistent: true,
    data_dir: _string(raw, "data_dir", "the database status"),
    database: _string(raw, "database", "the database status"),
    schema_version: _number(raw, "schema_version", "the database status"),
    journal_mode: _string(raw, "journal_mode", "the database status"),
    integrity: _string(raw, "integrity", "the database status"),
    tables,
  };
}

/** Narrow one persisted job row. */
export function parseJobHistoryRow(raw: unknown, label: string): JobHistoryRow {
  return {
    job_id: _string(raw, "job_id", label),
    operation: _string(raw, "operation", label),
    workspace: _string(raw, "workspace", label),
    state: _string(raw, "state", label),
    created_at: _string(raw, "created_at", label),
    started_at: _optString(raw, "started_at", label),
    completed_at: _optString(raw, "completed_at", label),
    progress: _number(raw, "progress", label),
    error: _optString(raw, "error", label),
  };
}

export const databaseApi = {
  /** `GET /api/v1/database` — durable status or the honest fallback. */
  async status(): Promise<DatabaseStatusResponse> {
    const raw = await api.get<unknown>("/api/v1/database");
    return parseDatabaseStatus(raw);
  },

  /** `GET /api/v1/database/jobs?limit=` — persisted history, newest-first. */
  async jobHistory(limit = 8): Promise<readonly JobHistoryRow[]> {
    const raw = await api.get<unknown>(
      `/api/v1/database/jobs?limit=${encodeURIComponent(String(limit))}`,
    );
    const jobs = _field(raw, "jobs", "the job history") as unknown[];
    return jobs.map((row, index) =>
      parseJobHistoryRow(row, `the job history row ${index}`),
    );
  },

  /** `POST /api/v1/database/backup` — registers + answers the bundle artifact. */
  async backup(filename?: string): Promise<BackupResult> {
    const raw = await api.post<unknown>("/api/v1/database/backup", {
      ...(filename ? { filename } : {}),
    });
    return {
      artifact: parseArtifactView(
        _field(raw, "artifact", "the backup answer"),
        "the backup answer's artifact",
      ),
    };
  },

  /** `POST /api/v1/database/restore` — restore + registry rebuild. */
  async restore(artifactId: string): Promise<RestoreResult> {
    const raw = await api.post<unknown>("/api/v1/database/restore", {
      artifact_id: artifactId,
    });
    return {
      restore: _record(raw, "restore", "the restore answer"),
      recovered: _record(raw, "recovered", "the restore answer"),
    };
  },
};
