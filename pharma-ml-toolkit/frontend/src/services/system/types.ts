/**
 * part16 §8 — the system-resource snapshot DTOs.
 *
 * Field-for-field mirrors of the backend's `/api/v1/system/resources`
 * answer (part13 §15's read-only lane: the host over psutil or the
 * honest stdlib subset, this backend process's own footprint, and the
 * JobManager's queue depth by state). The Python drift guards pin the
 * vocabulary, so a backend rename turns this suite red before the
 * frontend drifts. Absent numbers are preserved as null — the panel
 * renders them as the honest dash and NEVER fabricates a value.
 */

/** The memory block (`psutil.virtual_memory()` or absent under fallback). */
export interface ResourceMemoryBlock {
  readonly total: number;
  readonly available: number;
  readonly used: number;
  readonly percent: number;
}

/** The swap block (`psutil.swap_memory()` or absent under fallback). */
export interface ResourceSwapBlock {
  readonly total: number;
  readonly used: number;
  readonly free: number;
  readonly percent: number;
}

/** The disk block (`shutil.disk_usage` — present in both sources). */
export interface ResourceDiskBlock {
  readonly path: string;
  readonly total: number;
  readonly used: number;
  readonly free: number;
}

/** The host-level system block. Nulls = the stdlib-fallback's absences. */
export interface SystemBlock {
  readonly cpu_count: number | null;
  readonly cpu_percent: number | null;
  readonly loadavg: readonly number[] | null;
  readonly memory: ResourceMemoryBlock | null;
  readonly swap: ResourceSwapBlock | null;
  readonly disk: ResourceDiskBlock;
  /** The fallback source's own honest explanations, verbatim. */
  readonly notes: readonly string[];
}

/** This backend process's own footprint. */
export interface ProcessBlock {
  readonly pid: number;
  readonly cpu_percent: number | null;
  readonly memory_rss_bytes: number | null;
  readonly num_threads: number | null;
}

/** The JobManager queue depth (read-only, by state). */
export interface JobsBlock {
  readonly active: number;
  readonly by_state: Readonly<Record<string, number>>;
}

/** The optional-GPU contract: absent stays absent, stated honestly. */
export interface GpuBlock {
  readonly available: boolean;
  readonly note: string;
}

/** `GET /api/v1/system/resources` — the whole §15 snapshot. */
export interface ResourceSnapshot {
  readonly source: "psutil" | "stdlib-fallback";
  readonly backend: { readonly name: string; readonly version: string };
  readonly system: SystemBlock;
  readonly process: ProcessBlock;
  readonly jobs: JobsBlock;
  readonly gpu: GpuBlock;
}

/** Thrown when a response cannot be narrowed to the wire DTO (§8.1). */
export class ApiSchemaError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ApiSchemaError";
  }
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" ? value : null;
}

function parseNumberBlock(
  raw: unknown,
  what: string,
  fields: readonly string[],
): Record<string, number> {
  if (!isObject(raw)) {
    throw new ApiSchemaError(`${what}: expected an object`);
  }
  const block: Record<string, number> = {};
  for (const field of fields) {
    if (typeof raw[field] !== "number") {
      throw new ApiSchemaError(`${what}: field ${field} is not a number`);
    }
    block[field] = raw[field] as number;
  }
  return block;
}

function parseLoadavg(raw: unknown, what: string): readonly number[] | null {
  if (raw === null || raw === undefined) {
    return null;
  }
  if (!Array.isArray(raw) || raw.length !== 3) {
    throw new ApiSchemaError(`${what}: loadavg must be the 1/5/15-minute trio`);
  }
  if (!raw.every((x) => typeof x === "number")) {
    throw new ApiSchemaError(`${what}: loadavg entries must be numbers`);
  }
  return raw as readonly number[];
}

/**
 * Validate an untrusted payload as a ResourceSnapshot (§8.1: validate
 * API responses where appropriate). The strictness is on the required
 * skeleton; the honest nulls ride through as null.
 */
export function parseResourceSnapshot(
  raw: unknown,
  what: string,
): ResourceSnapshot {
  if (!isObject(raw)) {
    throw new ApiSchemaError(`${what}: expected an object`);
  }
  const r = raw;

  const source = r["source"];
  if (source !== "psutil" && source !== "stdlib-fallback") {
    throw new ApiSchemaError(`${what}: unknown source ${JSON.stringify(source)}`);
  }

  const backendRaw = r["backend"];
  if (!isObject(backendRaw)) {
    throw new ApiSchemaError(`${what}: backend block missing`);
  }
  if (typeof backendRaw["name"] !== "string" || typeof backendRaw["version"] !== "string") {
    throw new ApiSchemaError(`${what}: backend name/version must be strings`);
  }

  const systemRaw = r["system"];
  if (!isObject(systemRaw)) {
    throw new ApiSchemaError(`${what}: system block missing`);
  }
  const diskRaw = systemRaw["disk"];
  if (!isObject(diskRaw)) {
    throw new ApiSchemaError(`${what}: system.disk block missing`);
  }
  for (const field of ["path", "total", "used", "free"] as const) {
    if (field === "path" ? typeof diskRaw[field] !== "string" : typeof diskRaw[field] !== "number") {
      throw new ApiSchemaError(`${what}: system.disk.${field} is malformed`);
    }
  }

  const memoryRaw = systemRaw["memory"];
  const memory = isObject(memoryRaw)
    ? parseNumberBlock(memoryRaw, `${what}: memory`, [
        "total",
        "available",
        "used",
        "percent",
      ])
    : null;
  const swapRaw = systemRaw["swap"];
  const swap = isObject(swapRaw)
    ? parseNumberBlock(swapRaw, `${what}: swap`, [
        "total",
        "used",
        "free",
        "percent",
      ])
    : null;

  const notesRaw = systemRaw["notes"];
  const notes = Array.isArray(notesRaw)
    ? notesRaw.filter((note): note is string => typeof note === "string")
    : [];

  const processRaw = r["process"];
  if (!isObject(processRaw)) {
    throw new ApiSchemaError(`${what}: process block missing`);
  }
  if (typeof processRaw["pid"] !== "number") {
    throw new ApiSchemaError(`${what}: process.pid is not a number`);
  }

  const jobsRaw = r["jobs"];
  if (!isObject(jobsRaw)) {
    throw new ApiSchemaError(`${what}: jobs block missing`);
  }
  if (typeof jobsRaw["active"] !== "number") {
    throw new ApiSchemaError(`${what}: jobs.active is not a number`);
  }
  const byStateRaw = jobsRaw["by_state"];
  if (!isObject(byStateRaw)) {
    throw new ApiSchemaError(`${what}: jobs.by_state is not an object`);
  }
  const by_state: Record<string, number> = {};
  for (const [state, count] of Object.entries(byStateRaw)) {
    if (typeof count !== "number") {
      throw new ApiSchemaError(`${what}: jobs.by_state.${state} is not a number`);
    }
    by_state[state] = count;
  }

  const gpuRaw = r["gpu"];
  if (!isObject(gpuRaw)) {
    throw new ApiSchemaError(`${what}: gpu block missing`);
  }
  if (typeof gpuRaw["available"] !== "boolean") {
    throw new ApiSchemaError(`${what}: gpu.available is not a boolean`);
  }
  if (typeof gpuRaw["note"] !== "string") {
    throw new ApiSchemaError(`${what}: gpu.note is not a string`);
  }

  return {
    source,
    backend: {
      name: backendRaw["name"] as string,
      version: backendRaw["version"] as string,
    },
    system: {
      cpu_count: numberOrNull(systemRaw["cpu_count"]),
      cpu_percent: numberOrNull(systemRaw["cpu_percent"]),
      loadavg: parseLoadavg(systemRaw["loadavg"], what),
      memory:
        memory === null
          ? null
          : {
              total: memory["total"] as number,
              available: memory["available"] as number,
              used: memory["used"] as number,
              percent: memory["percent"] as number,
            },
      swap:
        swap === null
          ? null
          : {
              total: swap["total"] as number,
              used: swap["used"] as number,
              free: swap["free"] as number,
              percent: swap["percent"] as number,
            },
      disk: {
        path: diskRaw["path"] as string,
        total: diskRaw["total"] as number,
        used: diskRaw["used"] as number,
        free: diskRaw["free"] as number,
      },
      notes,
    },
    process: {
      pid: processRaw["pid"] as number,
      cpu_percent: numberOrNull(processRaw["cpu_percent"]),
      memory_rss_bytes: numberOrNull(processRaw["memory_rss_bytes"]),
      num_threads: numberOrNull(processRaw["num_threads"]),
    },
    jobs: { active: jobsRaw["active"] as number, by_state },
    gpu: {
      available: gpuRaw["available"] as boolean,
      note: gpuRaw["note"] as string,
    },
  };
}
