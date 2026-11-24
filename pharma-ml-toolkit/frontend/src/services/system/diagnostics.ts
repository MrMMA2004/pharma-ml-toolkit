/**
 * part16 §15.5 — the diagnostics/recovery snapshot DTOs.
 *
 * Field-for-field mirrors of the backend's
 * `GET /api/v1/system/diagnostics` answer (the dependency
 * availability probes, the scientific-tool functional probes, the
 * storage health, the crash diagnostics, the log configuration
 * report, and the recoverable job state — the two views honestly
 * split). The Python drift guards pin the vocabulary, so a backend
 * rename turns this suite red before the frontend drifts. Absent
 * values ride through as the honest null — never fabricated.
 */

/** One dependency probe row (the requirements.txt vocabulary). */
export interface DependencyRow {
  readonly name: string;
  readonly role: string;
  readonly tier: "required" | "optional";
  readonly importable: boolean;
  readonly version: string | null;
  readonly error: string | null;
}

/** One scientific-tool probe (functional_probe null = absent tool). */
export interface ScientificToolRow {
  readonly name: string;
  readonly version: string | null;
  readonly importable: boolean;
  readonly functional_probe: boolean | null;
  readonly probe: string;
}

/** The disk block (present on both the persistent/ephemeral lanes). */
export interface DiagnosticsDiskBlock {
  readonly path: string;
  readonly total: number;
  readonly used: number;
  readonly free: number;
}

/** The SQLite database sub-block (null on the ephemeral session). */
export interface DiagnosticsDatabaseBlock {
  readonly file: string;
  readonly size_bytes: number | null;
  readonly schema_version: number | null;
  readonly journal_mode: string | null;
}

/** The §15.5 storage health block. */
export interface DiagnosticsStorageBlock {
  readonly persistent: boolean;
  readonly data_dir: string | null;
  readonly reason: string | null;
  readonly disk: DiagnosticsDiskBlock;
  readonly database: DiagnosticsDatabaseBlock | null;
  readonly tables: Readonly<Record<string, number>> | null;
  readonly vault_bytes: number | null;
  readonly integrity: string | null;
}

/** One recent FAILED job row (bounded at ten, newest first). */
export interface CrashRow {
  readonly job_id: string | null;
  readonly operation: string | null;
  readonly workspace: string | null;
  readonly error: string | null;
  readonly completed_at: string | null;
}

/** The §15.5 crash diagnostics block. */
export interface CrashBlock {
  readonly source: string;
  readonly recent_failed: readonly CrashRow[];
}

/** One configured log handler (console / file / rotating-file). */
export interface LogHandlerRow {
  readonly kind: string;
  readonly path: string | null;
}

/** The §15.5 log configuration report (read-only). */
export interface LogsBlock {
  readonly level: string;
  readonly handlers: readonly LogHandlerRow[];
  readonly log_file: string | null;
  readonly note: string;
}

/** The recoverable job state — the two views, honestly split. */
export interface RecoveryBlock {
  readonly persisted_by_state: Readonly<Record<string, number>>;
  readonly in_session_by_state: Readonly<Record<string, number>>;
  readonly source: string;
  readonly note: string;
}

/** The machine-readable scope contract. */
export interface DiagnosticsScope {
  readonly implemented: readonly string[];
  readonly not_implemented: readonly string[];
  readonly note: string;
}

/** `GET /api/v1/system/diagnostics` — the whole §15.5 answer. */
export interface DiagnosticsSnapshot {
  readonly backend: {
    readonly name: string;
    readonly version: string;
    readonly python: string;
  };
  readonly dependencies: readonly DependencyRow[];
  readonly scientific_tools: readonly ScientificToolRow[];
  readonly storage: DiagnosticsStorageBlock;
  readonly crash: CrashBlock;
  readonly logs: LogsBlock;
  readonly recovery: RecoveryBlock;
  readonly scope: DiagnosticsScope;
}

/** `POST /api/v1/system/cache/cleanup` — the safe-cleanup answer. */
export interface CacheCleanupAnswer {
  readonly cleared: { readonly samples_stats_entries: number };
  readonly preserved: readonly string[];
  readonly note: string;
}

/** Thrown when a response cannot be narrowed to the wire DTO. */
export class ApiSchemaError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ApiSchemaError";
  }
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function stringOrNull(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" ? value : null;
}

function stringArray(value: unknown, what: string): readonly string[] {
  if (!Array.isArray(value)) {
    throw new ApiSchemaError(`${what}: expected a string array`);
  }
  return value.map((entry): string => {
    if (typeof entry !== "string") {
      throw new ApiSchemaError(`${what}: a non-string entry`);
    }
    return entry;
  });
}

function recordOfNumbers(
  value: unknown,
  what: string,
): Readonly<Record<string, number>> {
  if (!isObject(value)) {
    throw new ApiSchemaError(`${what}: expected a string->number record`);
  }
  const out: Record<string, number> = {};
  for (const [key, entry] of Object.entries(value)) {
    if (typeof entry !== "number") {
      throw new ApiSchemaError(`${what}: ${key} is not a number`);
    }
    out[key] = entry;
  }
  return out;
}

function parseDependencies(
  raw: unknown,
  what: string,
): readonly DependencyRow[] {
  if (!Array.isArray(raw)) {
    throw new ApiSchemaError(`${what}: dependencies must be an array`);
  }
  return raw.map((entry): DependencyRow => {
    if (!isObject(entry)) {
      throw new ApiSchemaError(`${what}: a dependency row is not an object`);
    }
    const tier = entry["tier"];
    if (tier !== "required" && tier !== "optional") {
      throw new ApiSchemaError(
        `${what}: dependency tier must be required|optional`,
      );
    }
    if (
      typeof entry["name"] !== "string" ||
      typeof entry["role"] !== "string" ||
      typeof entry["importable"] !== "boolean"
    ) {
      throw new ApiSchemaError(
        `${what}: dependency name/role/importable malformed`,
      );
    }
    return {
      name: entry["name"],
      role: entry["role"],
      tier,
      importable: entry["importable"],
      version: stringOrNull(entry["version"]),
      error: stringOrNull(entry["error"]),
    };
  });
}

function parseScientificTools(
  raw: unknown,
  what: string,
): readonly ScientificToolRow[] {
  if (!Array.isArray(raw)) {
    throw new ApiSchemaError(`${what}: scientific_tools must be an array`);
  }
  return raw.map((entry): ScientificToolRow => {
    if (!isObject(entry)) {
      throw new ApiSchemaError(`${what}: a tool row is not an object`);
    }
    const probe = entry["functional_probe"];
    if (
      typeof probe !== "boolean" &&
      probe !== null &&
      probe !== undefined
    ) {
      throw new ApiSchemaError(`${what}: functional_probe is malformed`);
    }
    if (
      typeof entry["name"] !== "string" ||
      typeof entry["importable"] !== "boolean" ||
      typeof entry["probe"] !== "string"
    ) {
      throw new ApiSchemaError(`${what}: tool name/importable/probe malformed`);
    }
    return {
      name: entry["name"],
      version: stringOrNull(entry["version"]),
      importable: entry["importable"],
      functional_probe:
        probe === null || probe === undefined ? null : (probe as boolean),
      probe: entry["probe"],
    };
  });
}

function parseDiskBlock(raw: unknown, what: string): DiagnosticsDiskBlock {
  if (!isObject(raw)) {
    throw new ApiSchemaError(`${what}: disk block missing`);
  }
  for (const field of ["total", "used", "free"] as const) {
    if (typeof raw[field] !== "number") {
      throw new ApiSchemaError(`${what}: disk.${field} is not a number`);
    }
  }
  if (typeof raw["path"] !== "string") {
    throw new ApiSchemaError(`${what}: disk.path is not a string`);
  }
  return {
    path: raw["path"],
    total: raw["total"] as number,
    used: raw["used"] as number,
    free: raw["free"] as number,
  };
}

function parseStorageBlock(
  raw: unknown,
  what: string,
): DiagnosticsStorageBlock {
  if (!isObject(raw)) {
    throw new ApiSchemaError(`${what}: storage block missing`);
  }
  if (typeof raw["persistent"] !== "boolean") {
    throw new ApiSchemaError(`${what}: storage.persistent is not a boolean`);
  }
  const databaseRaw = raw["database"];
  let database: DiagnosticsDatabaseBlock | null = null;
  if (isObject(databaseRaw)) {
    database = {
      file: databaseRaw["file"] as string,
      size_bytes: numberOrNull(databaseRaw["size_bytes"]),
      schema_version: numberOrNull(databaseRaw["schema_version"]),
      journal_mode: stringOrNull(databaseRaw["journal_mode"]),
    };
  } else if (databaseRaw !== null && databaseRaw !== undefined) {
    throw new ApiSchemaError(`${what}: storage.database is malformed`);
  }
  const tablesRaw = raw["tables"];
  const tables = isObject(tablesRaw)
    ? recordOfNumbers(tablesRaw, `${what}: tables`)
    : null;
  return {
    persistent: raw["persistent"] as boolean,
    data_dir: stringOrNull(raw["data_dir"]),
    reason: stringOrNull(raw["reason"]),
    disk: parseDiskBlock(raw["disk"], `${what}: storage`),
    database,
    tables,
    vault_bytes: numberOrNull(raw["vault_bytes"]),
    integrity: stringOrNull(raw["integrity"]),
  };
}

function parseCrashBlock(raw: unknown, what: string): CrashBlock {
  if (!isObject(raw)) {
    throw new ApiSchemaError(`${what}: crash block missing`);
  }
  if (typeof raw["source"] !== "string") {
    throw new ApiSchemaError(`${what}: crash.source is not a string`);
  }
  const rowsRaw = raw["recent_failed"];
  if (!Array.isArray(rowsRaw)) {
    throw new ApiSchemaError(`${what}: crash.recent_failed must be an array`);
  }
  const recent_failed = rowsRaw.map((entry): CrashRow => {
    if (!isObject(entry)) {
      throw new ApiSchemaError(`${what}: a crash row is not an object`);
    }
    return {
      job_id: stringOrNull(entry["job_id"]),
      operation: stringOrNull(entry["operation"]),
      workspace: stringOrNull(entry["workspace"]),
      error: stringOrNull(entry["error"]),
      completed_at: stringOrNull(entry["completed_at"]),
    };
  });
  return { source: raw["source"], recent_failed };
}

function parseLogsBlock(raw: unknown, what: string): LogsBlock {
  if (!isObject(raw)) {
    throw new ApiSchemaError(`${what}: logs block missing`);
  }
  if (
    typeof raw["level"] !== "string" ||
    typeof raw["note"] !== "string"
  ) {
    throw new ApiSchemaError(`${what}: logs.level/note malformed`);
  }
  const handlersRaw = raw["handlers"];
  if (!Array.isArray(handlersRaw)) {
    throw new ApiSchemaError(`${what}: logs.handlers must be an array`);
  }
  const handlers = handlersRaw.map((entry): LogHandlerRow => {
    if (!isObject(entry) || typeof entry["kind"] !== "string") {
      throw new ApiSchemaError(`${what}: a log handler row is malformed`);
    }
    return {
      kind: entry["kind"],
      path: stringOrNull(entry["path"]),
    };
  });
  return {
    level: raw["level"],
    handlers,
    log_file: stringOrNull(raw["log_file"]),
    note: raw["note"],
  };
}

function parseRecoveryBlock(
  raw: unknown,
  what: string,
): RecoveryBlock {
  if (!isObject(raw)) {
    throw new ApiSchemaError(`${what}: recovery block missing`);
  }
  if (typeof raw["source"] !== "string" || typeof raw["note"] !== "string") {
    throw new ApiSchemaError(`${what}: recovery.source/note malformed`);
  }
  return {
    persisted_by_state: recordOfNumbers(
      raw["persisted_by_state"],
      `${what}: recovery.persisted_by_state`,
    ),
    in_session_by_state: recordOfNumbers(
      raw["in_session_by_state"],
      `${what}: recovery.in_session_by_state`,
    ),
    source: raw["source"],
    note: raw["note"],
  };
}

/**
 * Validate an untrusted payload as a DiagnosticsSnapshot (§8.1:
 * validate API responses where appropriate). The strictness is on
 * the required skeleton; the honest nulls ride through as null.
 */
export function parseDiagnosticsSnapshot(
  raw: unknown,
  what: string,
): DiagnosticsSnapshot {
  if (!isObject(raw)) {
    throw new ApiSchemaError(`${what}: expected an object`);
  }
  const r = raw;

  const backendRaw = r["backend"];
  if (!isObject(backendRaw)) {
    throw new ApiSchemaError(`${what}: backend block missing`);
  }
  if (
    typeof backendRaw["name"] !== "string" ||
    typeof backendRaw["version"] !== "string" ||
    typeof backendRaw["python"] !== "string"
  ) {
    throw new ApiSchemaError(`${what}: backend name/version/python malformed`);
  }

  const scopeRaw = r["scope"];
  if (!isObject(scopeRaw)) {
    throw new ApiSchemaError(`${what}: scope block missing`);
  }
  if (typeof scopeRaw["note"] !== "string") {
    throw new ApiSchemaError(`${what}: scope.note is not a string`);
  }

  return {
    backend: {
      name: backendRaw["name"],
      version: backendRaw["version"],
      python: backendRaw["python"],
    },
    dependencies: parseDependencies(r["dependencies"], what),
    scientific_tools: parseScientificTools(r["scientific_tools"], what),
    storage: parseStorageBlock(r["storage"], what),
    crash: parseCrashBlock(r["crash"], what),
    logs: parseLogsBlock(r["logs"], what),
    recovery: parseRecoveryBlock(r["recovery"], what),
    scope: {
      implemented: stringArray(scopeRaw["implemented"], `${what}: scope`),
      not_implemented: stringArray(
        scopeRaw["not_implemented"],
        `${what}: scope`,
      ),
      note: scopeRaw["note"],
    },
  };
}

/** Validate an untrusted payload as a CacheCleanupAnswer. */
export function parseCacheCleanupAnswer(
  raw: unknown,
  what: string,
): CacheCleanupAnswer {
  if (!isObject(raw)) {
    throw new ApiSchemaError(`${what}: expected an object`);
  }
  const clearedRaw = raw["cleared"];
  if (!isObject(clearedRaw)) {
    throw new ApiSchemaError(`${what}: cleared block missing`);
  }
  if (typeof clearedRaw["samples_stats_entries"] !== "number") {
    throw new ApiSchemaError(
      `${what}: cleared.samples_stats_entries is not a number`,
    );
  }
  if (typeof raw["note"] !== "string") {
    throw new ApiSchemaError(`${what}: note is not a string`);
  }
  return {
    cleared: {
      samples_stats_entries: clearedRaw["samples_stats_entries"],
    },
    preserved: stringArray(raw["preserved"], `${what}: preserved`),
    note: raw["note"],
  };
}
