/**
 * part16 §8 — the system-resource client's own contract.
 *
 * The `/api/v1/system/resources` snapshot (part13 §15's read-only
 * lane, consumed — never duplicated) narrowed through the house
 * parse pattern: the full psutil shape field-for-field, the honest
 * stdlib-fallback shape (the absent numbers preserved as null), and
 * the ApiSchemaError verdict for malformed payloads. Nothing is ever
 * fabricated — what the backend did not send stays null and renders
 * as the honest dash.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

const getMock = vi.hoisted(() => vi.fn());
vi.mock("@/services/api/client", () => ({
  api: { get: (path: string) => getMock(path) },
}));

import { systemApi } from "@/services/system/client";
import {
  ApiSchemaError,
  parseResourceSnapshot,
} from "@/services/system/types";

const DISK = {
  path: "/app/pharma-ml-toolkit",
  total: 50_000_000_000,
  used: 28_000_000_000,
  free: 22_000_000_000,
};

const BY_STATE = {
  QUEUED: 1,
  RUNNING: 1,
  CANCELLATION_REQUESTED: 0,
  CANCELLED: 0,
  COMPLETED: 0,
  FAILED: 0,
};

function psutilPayload(): Record<string, unknown> {
  return {
    source: "psutil",
    backend: { name: "pharmaml", version: "1.2.240" },
    system: {
      cpu_count: 8,
      cpu_percent: 23.5,
      loadavg: [1.1, 0.8, 0.6],
      memory: {
        total: 16_000_000_000,
        available: 6_200_000_000,
        used: 9_800_000_000,
        percent: 61.2,
      },
      swap: {
        total: 4_000_000_000,
        used: 500_000_000,
        free: 3_500_000_000,
        percent: 12.5,
      },
      disk: DISK,
    },
    process: {
      pid: 12345,
      cpu_percent: 4.2,
      memory_rss_bytes: 250_000_000,
      num_threads: 18,
    },
    jobs: { active: 2, by_state: BY_STATE, note: "in-session ledger" },
    gpu: { available: false, note: "GPU monitoring is deliberately absent" },
    scope: { implemented: [], not_implemented: [], note: "" },
  };
}

function fallbackPayload(): Record<string, unknown> {
  return {
    source: "stdlib-fallback",
    backend: { name: "pharmaml", version: "1.2.240" },
    system: {
      cpu_count: 8,
      cpu_percent: null,
      loadavg: [0.9, 0.7, 0.5],
      memory: null,
      swap: null,
      disk: DISK,
      notes: [
        "memory, swap and cpu_percent are absent without psutil — nulled, never fabricated",
      ],
    },
    process: {
      pid: 12345,
      cpu_percent: null,
      memory_rss_bytes: 260_000_000,
      num_threads: null,
    },
    jobs: { active: 0, by_state: BY_STATE, note: "in-session ledger" },
    gpu: { available: false, note: "GPU monitoring is deliberately absent" },
    scope: { implemented: [], not_implemented: [], note: "" },
  };
}

beforeEach(() => {
  getMock.mockReset();
});

describe("systemApi.resources (§8)", () => {
  it("fetches the pinned path and narrows the psutil snapshot field-for-field", async () => {
    getMock.mockResolvedValue(psutilPayload());

    const snap = await systemApi.resources();

    expect(getMock).toHaveBeenCalledWith("/api/v1/system/resources");
    expect(snap.source).toBe("psutil");
    expect(snap.backend.name).toBe("pharmaml");
    expect(snap.backend.version).toBe("1.2.240");
    expect(snap.system.cpu_count).toBe(8);
    expect(snap.system.cpu_percent).toBe(23.5);
    expect(snap.system.loadavg).toEqual([1.1, 0.8, 0.6]);
    expect(snap.system.memory?.total).toBe(16_000_000_000);
    expect(snap.system.memory?.available).toBe(6_200_000_000);
    expect(snap.system.memory?.percent).toBe(61.2);
    expect(snap.system.swap?.percent).toBe(12.5);
    expect(snap.system.disk.free).toBe(22_000_000_000);
    expect(snap.process.pid).toBe(12345);
    expect(snap.process.cpu_percent).toBe(4.2);
    expect(snap.process.memory_rss_bytes).toBe(250_000_000);
    expect(snap.process.num_threads).toBe(18);
    expect(snap.jobs.active).toBe(2);
    expect(snap.jobs.by_state["RUNNING"]).toBe(1);
    expect(snap.gpu.available).toBe(false);
  });

  it("preserves the stdlib-fallback nulls and its notes verbatim", async () => {
    getMock.mockResolvedValue(fallbackPayload());

    const snap = await systemApi.resources();

    expect(snap.source).toBe("stdlib-fallback");
    expect(snap.system.cpu_percent).toBeNull();
    expect(snap.system.memory).toBeNull();
    expect(snap.system.swap).toBeNull();
    expect(snap.process.cpu_percent).toBeNull();
    expect(snap.process.num_threads).toBeNull();
    expect(snap.system.notes?.[0]).toContain("nulled, never fabricated");
    // the honest subset is still real: the disk and the rusage RSS ride on
    expect(snap.system.disk.total).toBe(50_000_000_000);
    expect(snap.process.memory_rss_bytes).toBe(260_000_000);
  });

  it("throws ApiSchemaError on malformed payloads — never a silent any", async () => {
    const cases: Array<[string, unknown]> = [
      ["not an object", 42],
      [
        "system missing",
        {
          source: "psutil",
          backend: {},
          jobs: { active: 0, by_state: {} },
          process: {},
          gpu: {},
        },
      ],
      ["unknown source", { ...psutilPayload(), source: "fabricated" }],
      [
        "jobs.by_state not an object",
        {
          ...psutilPayload(),
          jobs: { active: 0, by_state: "many" },
        },
      ],
      [
        "system.disk missing",
        {
          ...psutilPayload(),
          system: {
            cpu_count: 8,
            cpu_percent: 1.0,
            loadavg: [1, 1, 1],
            memory: null,
            swap: null,
          },
        },
      ],
    ];
    for (const [, payload] of cases) {
      getMock.mockResolvedValue(payload);
      await expect(systemApi.resources()).rejects.toBeInstanceOf(ApiSchemaError);
      getMock.mockReset();
    }
  });

  it("parseResourceSnapshot keeps the GPU optional-absence contract", () => {
    const snap = parseResourceSnapshot(psutilPayload(), "test");
    expect(snap.gpu.available).toBe(false);
    expect(snap.gpu.note).toContain("deliberately absent");
    expect(snap.jobs.by_state).toStrictEqual(BY_STATE);
  });
});
