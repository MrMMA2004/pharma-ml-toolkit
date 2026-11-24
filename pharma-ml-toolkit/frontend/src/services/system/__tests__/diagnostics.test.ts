/**
 * part16 §15.5 — the diagnostics client's own contract.
 *
 * The `/api/v1/system/diagnostics` snapshot (the §15.5 read-only
 * lane) and the `POST /api/v1/system/cache/cleanup` answer narrowed
 * through the house parse pattern: the full durable shape
 * field-for-field (the dependency table with the required/optional
 * tiers, the scientific-tool functional probes, the persistent
 * storage block), the ephemeral-session shape (the honest nulls),
 * the cleanup guarantee shape, and the ApiSchemaError verdict for
 * malformed payloads. Nothing is fabricated — what the backend did
 * not send stays null.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

const getMock = vi.hoisted(() => vi.fn());
const postMock = vi.hoisted(() => vi.fn());
vi.mock("@/services/api/client", () => ({
  api: {
    get: (path: string) => getMock(path),
    post: (path: string) => postMock(path),
  },
}));

import { systemApi } from "@/services/system/client";
import {
  parseCacheCleanupAnswer,
  parseDiagnosticsSnapshot,
} from "@/services/system/diagnostics";

const DISK = {
  path: "/app/pharma-ml-toolkit",
  total: 50_000_000_000,
  used: 28_000_000_000,
  free: 22_000_000_000,
};

function durablePayload(): Record<string, unknown> {
  return {
    backend: { name: "pharmaml", version: "1.2.240", python: "3.12.16" },
    dependencies: [
      {
        name: "numpy",
        role: "the numeric core (the 1.26.4 pin)",
        tier: "required",
        importable: true,
        version: "1.26.4",
        error: null,
      },
      {
        name: "mordred",
        role: "the descriptor engine",
        tier: "optional",
        importable: false,
        version: null,
        error: "ModuleNotFoundError",
      },
    ],
    scientific_tools: [
      {
        name: "rdkit",
        version: "2026.9.1",
        importable: true,
        functional_probe: true,
        probe: "Chem.MolFromSmiles('CCO') is not None",
      },
      {
        name: "torch",
        version: null,
        importable: false,
        functional_probe: null,
        probe: "torch.tensor([1,2]) + torch.tensor([1,1]) == [2,3]",
      },
    ],
    storage: {
      persistent: true,
      data_dir: "/home/z/.local/share/pharmaml-studio",
      reason: null,
      disk: DISK,
      database: {
        file: "studio.db",
        size_bytes: 94_208,
        schema_version: 3,
        journal_mode: "wal",
      },
      tables: { datasets: 2, jobs: 1, artifacts: 0 },
      vault_bytes: 4_096,
      integrity: "ok",
    },
    crash: {
      source: "sqlite job history (the FAILED rows survive restarts)",
      recent_failed: [
        {
          job_id: "job-abc123",
          operation: "ml.train",
          workspace: "machine-learning",
          error: "RuntimeError: the seeded failure",
          completed_at: "2026-11-24T08:00:00Z",
        },
      ],
    },
    logs: {
      level: "WARNING",
      handlers: [{ kind: "console" }],
      log_file: null,
      note: "read-only configuration report",
    },
    recovery: {
      persisted_by_state: { FAILED: 1 },
      in_session_by_state: {
        QUEUED: 0,
        RUNNING: 0,
        COMPLETED: 0,
        FAILED: 0,
      },
      source: "sqlite job history (survives the process)",
      note: "the two views never pretend to be one",
    },
    scope: {
      implemented: ["dependency availability checks"],
      not_implemented: ["automatic job resume after a crash"],
      note: "Do not delete user data during diagnostic or cleanup operations.",
    },
  };
}

function ephemeralPayload(): Record<string, unknown> {
  const payload = durablePayload();
  const recovery = payload["recovery"] as Record<string, unknown>;
  return {
    ...payload,
    storage: {
      persistent: false,
      data_dir: null,
      reason: "ephemeral session — no data_dir was configured",
      disk: DISK,
      database: null,
      tables: null,
      vault_bytes: null,
      integrity: null,
    },
    crash: {
      source: "in-session JobManager ledger (ephemeral session)",
      recent_failed: [],
    },
    recovery: {
      ...recovery,
      persisted_by_state: {},
      source: "none — ephemeral session keeps no history",
    },
  };
}

beforeEach(() => {
  getMock.mockReset();
  postMock.mockReset();
});

describe("systemApi.diagnostics (§15.5)", () => {
  it("fetches the pinned path and narrows the durable snapshot field-for-field", async () => {
    getMock.mockResolvedValue(durablePayload());

    const snap = await systemApi.diagnostics();

    expect(getMock).toHaveBeenCalledWith("/api/v1/system/diagnostics");
    expect(snap.backend.name).toBe("pharmaml");
    expect(snap.backend.version).toBe("1.2.240");
    expect(snap.backend.python).toBe("3.12.16");

    expect(snap.dependencies).toHaveLength(2);
    expect(snap.dependencies[0]?.importable).toBe(true);
    expect(snap.dependencies[0]?.version).toBe("1.26.4");
    expect(snap.dependencies[1]?.importable).toBe(false);
    expect(snap.dependencies[1]?.version).toBeNull();
    expect(snap.dependencies[1]?.error).toBe("ModuleNotFoundError");

    expect(snap.scientific_tools[0]?.functional_probe).toBe(true);
    expect(snap.scientific_tools[1]?.functional_probe).toBeNull();

    expect(snap.storage.persistent).toBe(true);
    expect(snap.storage.data_dir).toBe(
      "/home/z/.local/share/pharmaml-studio",
    );
    expect(snap.storage.database?.size_bytes).toBe(94_208);
    expect(snap.storage.database?.journal_mode).toBe("wal");
    expect(snap.storage.tables?.["datasets"]).toBe(2);
    expect(snap.storage.vault_bytes).toBe(4_096);
    expect(snap.storage.integrity).toBe("ok");
    expect(snap.storage.disk.free).toBe(22_000_000_000);

    expect(snap.crash.recent_failed[0]?.job_id).toBe("job-abc123");
    expect(snap.crash.recent_failed[0]?.error).toContain("RuntimeError");

    expect(snap.logs.level).toBe("WARNING");
    expect(snap.logs.handlers[0]?.kind).toBe("console");
    expect(snap.logs.log_file).toBeNull();

    expect(snap.recovery.persisted_by_state["FAILED"]).toBe(1);
    expect(snap.recovery.in_session_by_state["QUEUED"]).toBe(0);

    expect(snap.scope.not_implemented[0]).toContain("resume after a crash");
    expect(snap.scope.note).toContain("Do not delete user data");
  });

  it("preserves the ephemeral-session nulls verbatim", async () => {
    getMock.mockResolvedValue(ephemeralPayload());

    const snap = await systemApi.diagnostics();

    expect(snap.storage.persistent).toBe(false);
    expect(snap.storage.database).toBeNull();
    expect(snap.storage.tables).toBeNull();
    expect(snap.storage.vault_bytes).toBeNull();
    expect(snap.storage.integrity).toBeNull();
    expect(snap.storage.reason).toContain("ephemeral session");
    expect(snap.crash.recent_failed).toHaveLength(0);
    expect(snap.recovery.persisted_by_state).toEqual({});
  });
});

describe("systemApi.cleanupCache (§15.5)", () => {
  it("posts the pinned path and narrows the guarantee answer", async () => {
    postMock.mockResolvedValue({
      cleared: { samples_stats_entries: 17 },
      preserved: ["datasets", "trained models", "the job history"],
      note: "Do not delete user data during diagnostic or cleanup operations.",
    });

    const answer = await systemApi.cleanupCache();

    expect(postMock).toHaveBeenCalledWith("/api/v1/system/cache/cleanup");
    expect(answer.cleared.samples_stats_entries).toBe(17);
    expect(answer.preserved).toHaveLength(3);
    expect(answer.note).toContain("Do not delete user data");
  });
});

describe("parseDiagnosticsSnapshot strictness", () => {
  it("rejects a payload missing the backend identity", () => {
    const payload = durablePayload();
    delete payload["backend"];
    expect(() => parseDiagnosticsSnapshot(payload, "test")).toThrow(
      /backend block missing/,
    );
  });

  it("rejects a dependency row with an unknown tier", () => {
    const payload = durablePayload();
    (payload["dependencies"] as Record<string, unknown>[])[0]!.tier = "core";
    expect(() => parseDiagnosticsSnapshot(payload, "test")).toThrow(
      /tier/,
    );
  });

  it("rejects a malformed dependency row", () => {
    const payload = durablePayload();
    (payload["dependencies"] as unknown[])[0] = "numpy";
    expect(() => parseDiagnosticsSnapshot(payload, "test")).toThrow(
      /dependency row is not an object/,
    );
  });

  it("rejects a crash block that is not an object", () => {
    const payload = durablePayload();
    payload["crash"] = null;
    expect(() => parseDiagnosticsSnapshot(payload, "test")).toThrow(
      /crash block missing/,
    );
  });

  it("rejects a non-string log handler kind", () => {
    const payload = durablePayload();
    (payload["logs"] as Record<string, unknown>)["handlers"] = [42];
    expect(() => parseDiagnosticsSnapshot(payload, "test")).toThrow(
      /log handler row is malformed/,
    );
  });

  it("rejects a malformed storage disk block", () => {
    const payload = durablePayload();
    (payload["storage"] as Record<string, unknown>)["disk"] = {
      path: 12,
    };
    expect(() => parseDiagnosticsSnapshot(payload, "test")).toThrow(
      /disk/,
    );
  });
});

describe("parseCacheCleanupAnswer strictness", () => {
  it("rejects a payload missing the cleared count", () => {
    expect(() =>
      parseCacheCleanupAnswer({ preserved: [], note: "x" }, "test"),
    ).toThrow(/cleared block missing/);
  });

  it("rejects a non-numeric cleared count", () => {
    expect(() =>
      parseCacheCleanupAnswer(
        { cleared: { samples_stats_entries: "many" }, note: "x" },
        "test",
      ),
    ).toThrow(/samples_stats_entries/);
  });

  it("rejects a payload missing the note", () => {
    expect(() =>
      parseCacheCleanupAnswer(
        { cleared: { samples_stats_entries: 1 }, preserved: [] },
        "test",
      ),
    ).toThrow(/note/);
  });
});
